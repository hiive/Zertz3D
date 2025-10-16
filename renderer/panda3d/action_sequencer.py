"""Manager class for action visualization sequencing."""

from shared.render_data import RenderData
from shared.materials_modifiers import (
    PLACEMENT_HIGHLIGHT_MATERIAL_MOD,
    REMOVABLE_HIGHLIGHT_MATERIAL_MOD,
    CAPTURE_HIGHLIGHT_MATERIAL_MOD,
    SELECTED_CAPTURE_MATERIAL_MOD,
    # ISOLATION_HIGHLIGHT_MATERIAL_MOD,
)


class ActionVisualizationSequencer:
    """Manages the multiphase highlighting sequence for showing moves."""

    # Phase constants
    PHASE_PLACEMENT_HIGHLIGHTS = "placement_highlights"
    PHASE_SELECTED_PLACEMENT = "selected_placement"
    PHASE_REMOVAL_HIGHLIGHTS = "removal_highlights"
    PHASE_SELECTED_REMOVAL = "selected_removal"
    PHASE_CAPTURE_HIGHLIGHTS = "capture_highlights"
    PHASE_SELECTED_CAPTURE = "selected_capture"
    PHASE_ANIMATING = "animating"  # Waiting for final move animations to complete

    # Highlight and animation durations (seconds)
    ISOLATION_HIGHLIGHT_DURATION = 0.5  # Yellow flash for isolated region captures

    def __init__(self, renderer):
        """Initialize the state machine.

        Args:
            renderer: PandaRenderer instance
        """
        self.renderer = renderer
        self.highlight_durations = {
            self.PHASE_PLACEMENT_HIGHLIGHTS: 0.5,
            self.PHASE_SELECTED_PLACEMENT: 0.5,
            self.PHASE_REMOVAL_HIGHLIGHTS: 0.5,
            self.PHASE_SELECTED_REMOVAL: 0.5,
            self.PHASE_CAPTURE_HIGHLIGHTS: 0.5,
            self.PHASE_SELECTED_CAPTURE: 0.5,
        }

        # State tracking
        self.phase = (
            None  # Current phase: 'placement_highlights', 'selected_placement', etc.
        )
        self.pending_player = None  # Player making the move
        self.pending_action_dict = None  # Action dict
        self.placement_positions = (
            None  # List of position strings for placement highlights
        )
        self.capture_moves = None  # List of capture move dicts
        self.removal_positions = None  # List of removable position strings
        self.task_delay_time = 0  # Animation duration from controller's task
        self.start_delay = renderer.start_delay

    def is_active(self):
        """Check if the state machine is currently active."""
        return self.phase is not None

    def start(self, player, render_data, task_delay_time):
        """Start the highlighting sequence for an action.

        Args:
            player: Player making the move
            render_data: RenderData value object with action_dict and highlight data
            task_delay_time: Animation duration from controller's task
            start_delay: Animation delay from controller's task
        """
        self.pending_player = player
        self.pending_action_dict = render_data.action_dict
        self.task_delay_time = task_delay_time
        self.placement_positions = render_data.placement_positions
        self.capture_moves = render_data.capture_moves
        self.removal_positions = render_data.removal_positions

        action_type = render_data.action_dict["action"]
        if action_type == "PUT":
            # Queue placement highlights and start the sequence
            self._queue_placement_highlights(render_data.placement_positions)
            self.phase = self.PHASE_PLACEMENT_HIGHLIGHTS
        elif action_type == "CAP":
            # Queue capture highlights and start the sequence
            self._queue_capture_highlights(render_data.capture_moves)
            self.phase = self.PHASE_CAPTURE_HIGHLIGHTS
        elif action_type == "PASS":
            # PASS has no visuals, action already executed by controller
            self.phase = None  # Done immediately (no highlight phases)
        self.start_delay = 0

    def update(self, task):
        """Update the state machine. Called each frame.

        Args:
            task: Panda3D task object

        Returns:
            True if the state machine should continue (waiting for animations or processing)
            False if the state machine is done and game should proceed
        """
        if not self.is_active():
            return False

        # Check if animations are still active
        if self.renderer.is_animation_active():
            return True  # Still waiting for animations to finish

        # Animations finished, advance to next phase
        if self.phase == self.PHASE_PLACEMENT_HIGHLIGHTS:
            self._on_placement_highlights_done()
        elif self.phase == self.PHASE_SELECTED_PLACEMENT:
            self._on_selected_placement_done(task)
        elif self.phase == self.PHASE_REMOVAL_HIGHLIGHTS:
            self._on_removal_highlights_done()
        elif self.phase == self.PHASE_SELECTED_REMOVAL:
            self._on_selected_removal_done(task)
        elif self.phase == self.PHASE_CAPTURE_HIGHLIGHTS:
            self._on_capture_highlights_done()
        elif self.phase == self.PHASE_SELECTED_CAPTURE:
            self._on_selected_capture_done(task)
        elif self.phase == self.PHASE_ANIMATING:
            # Waiting for final move animations to complete
            # When we reach here, animations are done, so phase can be set to None
            self.phase = None

        return self.is_active()

    def _queue_placement_highlights(self, placement_positions):
        """Queue highlights for all valid placement positions.

        Args:
            placement_positions: List of position strings (pre-converted by controller)
        """
        if placement_positions:
            self.renderer.queue_highlight(
                rings=placement_positions,
                material_mod=PLACEMENT_HIGHLIGHT_MATERIAL_MOD,
                duration=self.highlight_durations[self.PHASE_PLACEMENT_HIGHLIGHTS],
                defer=self.start_delay,
            )

    def _queue_removal_highlights(self, removal_positions, defer=0):
        """Queue highlights for all valid removal positions for this action.

        Args:
            removal_positions: List of position strings (pre-converted by controller)
            defer: Delay before starting the highlight (seconds)
        """
        if removal_positions:
            self.renderer.queue_highlight(
                rings=removal_positions,
                duration=self.highlight_durations[self.PHASE_REMOVAL_HIGHLIGHTS],
                material_mod=REMOVABLE_HIGHLIGHT_MATERIAL_MOD,
                defer=defer,
            )

    def _queue_capture_highlights(self, capture_moves):
        """Queue highlights for all valid capture moves, grouped by source marble.

        If only one capture is available, skip highlighting (will auto-advance to selected_capture phase).

        Args:
            capture_moves: List of capture move dicts (pre-converted by controller)
        """
        # Skip highlighting if only one capture available
        if capture_moves and len(capture_moves) == 1:
            return

        # Group captures by source position
        captures_by_source = {}  # {src_str: [dst_str1, dst_str2, ...]}
        for action_dict in capture_moves:
            src_str = action_dict["src"]
            dst_str = action_dict["dst"]

            if src_str not in captures_by_source:
                captures_by_source[src_str] = set()
            if dst_str:
                captures_by_source[src_str].add(dst_str)

        # Queue highlights sequentially - each group displays one after another
        capture_duration = self.highlight_durations[self.PHASE_CAPTURE_HIGHLIGHTS]
        defer_time = 0
        for src_str, destinations in captures_by_source.items():
            # Highlight the source marble and all its possible destinations
            highlight_rings = [src_str] + list(destinations)
            self.renderer.queue_highlight(
                highlight_rings,
                capture_duration,
                material_mod=CAPTURE_HIGHLIGHT_MATERIAL_MOD,
                defer=defer_time,
            )
            # Next group starts when this one ends
            defer_time += capture_duration

    def _on_placement_highlights_done(self):
        """Handle completion of placement highlights phase."""
        action_dict = self.pending_action_dict

        # Queue highlight for selected placement ring only
        selected_ring = action_dict["dst"]
        self.renderer.queue_highlight(
            [selected_ring], self.highlight_durations[self.PHASE_SELECTED_PLACEMENT]
        )
        self.phase = self.PHASE_SELECTED_PLACEMENT

    def _on_selected_placement_done(self, task):
        """Handle completion of selected placement highlight phase."""
        player = self.pending_player
        action_dict = self.pending_action_dict

        # Animate marble placement (action already executed by controller)
        self.renderer.show_marble_placement(player, action_dict, self.task_delay_time)

        # Queue removal highlights AFTER marble placement animation completes
        # task_delay_time already contains the animation duration from controller
        if self.removal_positions:
            self._queue_removal_highlights(
                self.removal_positions, defer=self.task_delay_time
            )

        # Move to removal highlights phase
        self.phase = self.PHASE_REMOVAL_HIGHLIGHTS

    def _on_removal_highlights_done(self):
        """Handle completion of removal highlights phase."""
        action_dict = self.pending_action_dict

        # Queue highlight for selected removal ring only
        selected_removal = action_dict["remove"]
        if selected_removal:  # Only if a ring is being removed
            self.renderer.queue_highlight(
                [selected_removal],
                self.highlight_durations[self.PHASE_SELECTED_REMOVAL],
                material_mod=REMOVABLE_HIGHLIGHT_MATERIAL_MOD,
            )
        self.phase = self.PHASE_SELECTED_REMOVAL

    def _on_selected_removal_done(self, task):
        """Handle completion of selected removal highlight phase."""
        action_dict = self.pending_action_dict

        # Now animate ring removal only (marble was already placed)
        self.renderer.show_ring_removal(action_dict, self.task_delay_time)

        # Wait for final move animations to complete
        self.phase = self.PHASE_ANIMATING

    def _on_capture_highlights_done(self):
        """Handle completion of capture highlights phase."""
        action_dict = self.pending_action_dict

        # Queue highlight for selected capture (src and dst) only in cornflower blue
        src_ring = action_dict["src"]
        dst_ring = action_dict["dst"]
        selected_rings = [src_ring, dst_ring]

        self.renderer.queue_highlight(
            selected_rings,
            self.highlight_durations[self.PHASE_SELECTED_CAPTURE],
            material_mod=SELECTED_CAPTURE_MATERIAL_MOD,
        )
        self.phase = self.PHASE_SELECTED_CAPTURE

    def _on_selected_capture_done(self, task):
        """Handle completion of selected capture highlight phase."""
        player = self.pending_player
        action_dict = self.pending_action_dict

        # Animate capture action (action already executed by controller)
        # Create minimal RenderData with just the action_dict
        render_data = RenderData(action_dict)
        self.renderer.show_action(
            player,
            render_data,
            self.task_delay_time,
            self.renderer.current_action_result,
        )

        # Wait for final move animations to complete
        self.phase = self.PHASE_ANIMATING
