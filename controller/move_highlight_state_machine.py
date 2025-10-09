"""State machine for managing move highlighting sequences."""

import numpy as np


class MoveHighlightStateMachine:
    """Manages the multi-phase highlighting sequence for showing moves."""

    def __init__(self, renderer, game, highlight_duration):
        """Initialize the state machine.

        Args:
            renderer: ZertzRenderer instance
            game: ZertzGame instance
            highlight_duration: Duration in seconds for each highlight phase
        """
        self.renderer = renderer
        self.game = game
        self.highlight_duration = highlight_duration

        # State tracking
        self.phase = None  # Current phase: 'placement_highlights', 'selected_placement', etc.
        self.pending_action = None  # (ax, ay, player) tuple
        self.pending_action_dict = None  # Action dict before game state changes
        self.pending_result = None  # Result from take_action

    def is_active(self):
        """Check if the state machine is currently active."""
        return self.phase is not None

    def start(self, ax, ay, player):
        """Start the highlighting sequence for an action.

        Args:
            ax: Action type ("PUT", "CAP", or "PASS")
            ay: Action tuple (or None for PASS)
            player: Player making the move
        """
        self.pending_action = (ax, ay, player)

        if ax == "PUT":
            # Queue placement highlights and start the sequence
            self._queue_placement_highlights()
            self.phase = 'placement_highlights'
        elif ax == "CAP":
            # Queue capture highlights and start the sequence
            self._queue_capture_highlights()
            self.phase = 'capture_highlights'
        elif ax == "PASS":
            # PASS has no visuals, but execute the action to switch players
            result = self.game.take_action(ax, ay)
            self.pending_result = result
            self.phase = None  # Done immediately (no highlight phases)

    def update(self, task):
        """Update the state machine. Called each frame.

        Args:
            task: Panda3D task object

        Returns:
            True if the state machine should continue (waiting for highlights or processing)
            False if the state machine is done and game should proceed
        """
        if not self.is_active():
            return False

        # Check if highlights are still active
        if self.renderer.is_highlight_active():
            return True  # Still waiting for highlights to finish

        # Highlights finished, advance to next phase
        if self.phase == 'placement_highlights':
            self._on_placement_highlights_done()
        elif self.phase == 'selected_placement':
            self._on_selected_placement_done(task)
        elif self.phase == 'removal_highlights':
            self._on_removal_highlights_done()
        elif self.phase == 'selected_removal':
            return self._on_selected_removal_done(task)
        elif self.phase == 'capture_highlights':
            self._on_capture_highlights_done()
        elif self.phase == 'selected_capture':
            return self._on_selected_capture_done(task)

        return self.is_active()

    def _queue_placement_highlights(self):
        """Queue highlights for all valid placement positions."""
        placement, capture = self.game.get_valid_actions()

        # Find destinations with ANY valid placement (any marble type, any removal option)
        valid_dests = np.any(placement, axis=(0, 2))  # Shape: (width²,)
        dest_indices = np.argwhere(valid_dests).flatten()

        placement_rings = []
        width = self.game.board.width
        for dst_idx in dest_indices:
            dst_y = dst_idx // width
            dst_x = dst_idx % width
            pos_str = self.game.board.index_to_str((dst_y, dst_x))
            if pos_str:
                placement_rings.append(pos_str)

        if placement_rings:
            self.renderer.queue_highlight(placement_rings, self.highlight_duration)

    def _queue_removal_highlights(self, marble_idx, dst, placement_array, board):
        """Queue highlights for all valid removal positions for this action."""
        width = board.width

        # Get valid removal indices for this (marble_idx, dst) combination
        removal_mask = placement_array[marble_idx, dst, :]
        removable_indices = np.argwhere(removal_mask).flatten()

        removable_rings = []
        for rem_idx in removable_indices:
            if rem_idx != width ** 2 and rem_idx != dst:  # Exclude "no removal" and destination
                rem_y = rem_idx // width
                rem_x = rem_idx % width
                rem_str = board.index_to_str((rem_y, rem_x))
                if rem_str:
                    removable_rings.append(rem_str)

        if removable_rings:
            self.renderer.queue_highlight(
                removable_rings,
                self.highlight_duration,
                color=self.renderer.REMOVABLE_HIGHLIGHT_COLOR,
                emission=self.renderer.REMOVABLE_HIGHLIGHT_EMISSION
            )

    def _queue_capture_highlights(self):
        """Queue highlights for all valid capture moves, grouped by source marble."""
        placement, capture = self.game.get_valid_actions()

        # Find all valid captures: (direction, src_y, src_x)
        capture_positions = np.argwhere(capture)

        # Group captures by source position
        captures_by_source = {}  # {src_str: [dst_str1, dst_str2, ...]}
        for direction, src_y, src_x in capture_positions:
            # Convert to action format to get source and destination
            try:
                _, action_dict = self.game.action_to_str("CAP", (direction, src_y, src_x))
                src_str = action_dict['src']
                dst_str = action_dict['dst']

                if src_str not in captures_by_source:
                    captures_by_source[src_str] = set()
                if dst_str:
                    captures_by_source[src_str].add(dst_str)
            except (IndexError, KeyError):
                continue

        # Queue a separate highlight for each source marble and its destinations
        for src_str, destinations in captures_by_source.items():
            # Highlight the source marble and all its possible destinations
            highlight_rings = [src_str] + list(destinations)
            self.renderer.queue_highlight(
                highlight_rings,
                self.highlight_duration,
                color=self.renderer.CAPTURE_HIGHLIGHT_COLOR,
                emission=self.renderer.CAPTURE_HIGHLIGHT_EMISSION
            )

    def _on_placement_highlights_done(self):
        """Handle completion of placement highlights phase."""
        ax, ay, player = self.pending_action

        try:
            _, action_dict = self.game.action_to_str(ax, ay)
        except IndexError as e:
            print(f"Error converting action to string: {e}")
            print(f"Action type: {ax}, Action: {ay}")
            raise

        # Queue highlight for selected placement ring only
        selected_ring = action_dict['dst']
        self.renderer.queue_highlight([selected_ring], self.highlight_duration)
        self.phase = 'selected_placement'

    def _on_selected_placement_done(self, task):
        """Handle completion of selected placement highlight phase."""
        ax, ay, player = self.pending_action

        try:
            _, action_dict = self.game.action_to_str(ax, ay)
        except IndexError as e:
            print(f"Error converting action to string: {e}")
            print(f"Action type: {ax}, Action: {ay}")
            raise

        # Save action_dict before game state changes
        self.pending_action_dict = action_dict

        # Get placement array BEFORE executing action (for removal highlights)
        placement_before, _ = self.game.get_valid_actions()

        # Queue removal highlights BEFORE executing action (so board is in original state)
        if ax == "PUT":
            marble_idx, dst, rem = ay  # Unpack action tuple
            self._queue_removal_highlights(marble_idx, dst, placement_before, self.game.board)

        # Animate marble placement
        if ax == "PUT":
            self.renderer.show_marble_placement(player, action_dict, task.delay_time)

        # Execute game logic
        result = self.game.take_action(ax, ay)
        self.pending_result = result

        # Move to removal highlights phase
        if ax == "PUT":
            self.phase = 'removal_highlights'
        else:
            # Capture actions don't have removal phase - animate immediately
            self.renderer.show_action(player, action_dict, task.delay_time)
            self.phase = None  # Done

    def _on_removal_highlights_done(self):
        """Handle completion of removal highlights phase."""
        action_dict = self.pending_action_dict

        # Queue highlight for selected removal ring only
        selected_removal = action_dict['remove']
        if selected_removal:  # Only if a ring is being removed
            self.renderer.queue_highlight(
                [selected_removal],
                self.highlight_duration,
                color=self.renderer.REMOVABLE_HIGHLIGHT_COLOR,
                emission=self.renderer.REMOVABLE_HIGHLIGHT_EMISSION
            )
        self.phase = 'selected_removal'

    def _on_selected_removal_done(self, task):
        """Handle completion of selected removal highlight phase."""
        action_dict = self.pending_action_dict
        result = self.pending_result

        # Now animate ring removal only (marble was already placed)
        self.renderer.show_ring_removal(action_dict, task.delay_time)

        # Cleanup
        self.phase = None
        self.pending_action = None
        self.pending_action_dict = None
        self.pending_result = None

        return result

    def _on_capture_highlights_done(self):
        """Handle completion of capture highlights phase."""
        ax, ay, player = self.pending_action

        try:
            _, action_dict = self.game.action_to_str(ax, ay)
        except IndexError as e:
            print(f"Error converting action to string: {e}")
            print(f"Action type: {ax}, Action: {ay}")
            raise

        # Queue highlight for selected capture (src and dst) only in cornflower blue
        src_ring = action_dict['src']
        dst_ring = action_dict['dst']
        selected_rings = [src_ring, dst_ring]

        self.renderer.queue_highlight(
            selected_rings,
            self.highlight_duration,
            color=self.renderer.SELECTED_CAPTURE_HIGHLIGHT_COLOR,
            emission=self.renderer.SELECTED_CAPTURE_HIGHLIGHT_EMISSION
        )
        self.phase = 'selected_capture'

    def _on_selected_capture_done(self, task):
        """Handle completion of selected capture highlight phase."""
        ax, ay, player = self.pending_action

        try:
            _, action_dict = self.game.action_to_str(ax, ay)
        except IndexError as e:
            print(f"Error converting action to string: {e}")
            print(f"Action type: {ax}, Action: {ay}")
            raise

        # Save action_dict before game state changes
        self.pending_action_dict = action_dict

        # Animate capture action
        self.renderer.show_action(player, action_dict, task.delay_time)

        # Execute game logic
        result = self.game.take_action(ax, ay)
        self.pending_result = result

        # Cleanup
        self.phase = None
        self.pending_action = None
        self.pending_action_dict = None
        self.pending_result = None

        return result

    def get_pending_player(self):
        """Get the player from the pending action."""
        if self.pending_action:
            return self.pending_action[2]
        return None

    def get_pending_result(self):
        """Get the pending result after action execution."""
        return self.pending_result