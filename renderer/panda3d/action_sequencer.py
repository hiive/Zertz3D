"""Manager class for action visualization sequencing."""

from shared.render_data import RenderData
from shared.materials_modifiers import (
    DARK_GREEN_MATERIAL_MOD,
    DARK_RED_MATERIAL_MOD,
    DARK_BLUE_MATERIAL_MOD,
    CORNFLOWER_BLUE_MATERIAL_MOD,
    WHITE_MATERIAL_MOD
)
from renderer.panda3d.material_modifier import MaterialModifier


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
    DEFAULT_HIGHLIGHT_DURATION = 1.0  # Yellow flash for isolated region captures

    # Heat-map configuration
    MIN_VISIBILITY_THRESHOLD = 0.1  # Minimum normalized score to show highlight (default: 0.1)

    def __init__(self, renderer, highlight_choices: str | None = None, min_threshold: float = 0.1):
        """Initialize the state machine.

        Args:
            renderer: PandaRenderer instance
            highlight_choices: Highlight mode ('uniform' or 'heatmap')
            min_threshold: Minimum normalized score threshold for heat-map visibility (default: 0.1)
        """
        self.renderer = renderer
        self.highlight_choices = highlight_choices
        self.min_threshold = min_threshold
        self.highlight_durations = {
            self.PHASE_PLACEMENT_HIGHLIGHTS: self.DEFAULT_HIGHLIGHT_DURATION,
            self.PHASE_SELECTED_PLACEMENT: self.DEFAULT_HIGHLIGHT_DURATION,
            self.PHASE_REMOVAL_HIGHLIGHTS: self.DEFAULT_HIGHLIGHT_DURATION,
            self.PHASE_SELECTED_REMOVAL: self.DEFAULT_HIGHLIGHT_DURATION,
            self.PHASE_CAPTURE_HIGHLIGHTS: self.DEFAULT_HIGHLIGHT_DURATION,
            self.PHASE_SELECTED_CAPTURE: self.DEFAULT_HIGHLIGHT_DURATION,
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

    def _interpolate_material(self, score: float, max_mod: MaterialModifier, min_mod: MaterialModifier) -> MaterialModifier:
        """Interpolate between MIN and MAX material modifiers based on normalized score.

        In uniform mode, all scores are treated as maximum (returns max_mod).
        In heatmap mode, interpolates between min_mod and max_mod based on score.

        Args:
            score: Normalized score in range [0.0, 1.0]
            max_mod: Material modifier for maximum score (1.0)
            min_mod: Material modifier for minimum visible score (min_threshold)

        Returns:
            MaterialModifier interpolated for the given score
        """
        if self.highlight_choices == 'uniform':
            # Uniform mode: all actions get maximum highlight
            return max_mod

        # Heat-map mode: interpolate based on score
        # Scores below threshold are not shown (caller responsibility)
        # Map score range [min_threshold, 1.0] to blend_ratio [0.0, 1.0]
        if score >= 1.0:
            return max_mod
        elif score <= self.min_threshold:
            return min_mod
        else:
            # Linear interpolation from min_mod (at min_threshold) to max_mod (at 1.0)
            blend_ratio = (score - self.min_threshold) / (1.0 - self.min_threshold)
            return MaterialModifier.blend_mods(min_mod, max_mod, blend_ratio)

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
            placement_positions: List of dicts with position and score
                Format: [{'pos': 'A1', 'score': 0.8}, {'pos': 'B2', 'score': 1.0}, ...]
                OR legacy format: ['A1', 'B2', ...] (treated as uniform)
        """
        if not placement_positions:
            return

        # Handle both enriched (list-of-dicts) and legacy (list-of-strings) formats

        # Enriched format with scores - highlight each position with its score
        for pos_dict in placement_positions:
            pos_str = pos_dict['pos']
            score = pos_dict.get('score', 1.0)

            # Skip positions below visibility threshold in heatmap mode
            if self.highlight_choices == 'heatmap' and score < self.min_threshold:
                continue

            # Interpolate material based on score
            material_mod = self._interpolate_material(
                score,
                DARK_GREEN_MATERIAL_MOD, # MAGENTA_MATERIAL_MOD
                WHITE_MATERIAL_MOD,  # DARK_GREEN_MATERIAL_MOD,
            )
            self.renderer.queue_highlight(
                rings=[pos_str],
                material_mod=material_mod,
                duration=self.highlight_durations[self.PHASE_PLACEMENT_HIGHLIGHTS],
                defer=self.start_delay,
            )

    def _queue_removal_highlights(self, removal_positions, defer=0):
        """Queue highlights for all valid removal positions for this action.

        Args:
            removal_positions: List of dicts with position and score
                Format: [{'pos': 'A1', 'score': 0.7}, {'pos': 'B2', 'score': 0.5}, ...]
                OR legacy format: ['A1', 'B2', ...] (treated as uniform)
            defer: Delay before starting the highlight (seconds)
        """
        if not removal_positions:
            return

        # Handle both enriched (list-of-dicts) and legacy (list-of-strings) formats

        # Enriched format with scores - highlight each position with its score
        for pos_dict in removal_positions:
            pos_str = pos_dict['pos']
            score = pos_dict.get('score', 1.0)

            # Skip positions below visibility threshold in heatmap mode
            if self.highlight_choices == 'heatmap' and score < self.min_threshold:
                continue

            # Interpolate material based on score
            material_mod = self._interpolate_material(
                score,
                DARK_RED_MATERIAL_MOD, # CYAN_MATERIAL_MOD
                WHITE_MATERIAL_MOD,  # DARK_RED_MATERIAL_MOD,
            )
            self.renderer.queue_highlight(
                rings=[pos_str],
                duration=self.highlight_durations[self.PHASE_REMOVAL_HIGHLIGHTS],
                material_mod=material_mod,
                defer=defer,
            )


    def _queue_capture_highlights(self, capture_moves):
        """Queue highlights for all valid capture moves, grouped by source marble.

        If only one capture is available, skip highlighting (will auto-advance to selected_capture phase).

        Args:
            capture_moves: List of capture move dicts with scores
                Format: [{'action': 'CAP', 'src': 'C4', 'dst': 'E6', ..., 'score': 0.9}, ...]
                OR legacy format: [{'action': 'CAP', 'src': 'C4', 'dst': 'E6', ...}, ...] (no scores)
        """
        # Skip highlighting if only one capture available
        if capture_moves and len(capture_moves) == 1:
            return

        # Group captures by source position and collect best score for each source
        captures_by_source = {}  # {src_str: (destinations_set, best_score)}
        for action_dict in capture_moves:
            src_str = action_dict["src"]
            dst_str = action_dict["dst"]
            score = action_dict.get("score", 1.0)  # Default to 1.0 if no score

            if src_str not in captures_by_source:
                captures_by_source[src_str] = (set(), 0.0)

            destinations, best_score = captures_by_source[src_str]
            if dst_str:
                destinations.add(dst_str)
            best_score = max(best_score, score)
            captures_by_source[src_str] = (destinations, best_score)

        # Queue highlights sequentially - each group displays one after another
        capture_duration = self.highlight_durations[self.PHASE_CAPTURE_HIGHLIGHTS]
        defer_time = 0
        for src_str, (destinations, score) in captures_by_source.items():
            # Skip this source if below visibility threshold in heatmap mode
            if self.highlight_choices == 'heatmap' and score < self.min_threshold:
                continue

            # Highlight the source marble and all its possible destinations
            highlight_rings = [src_str] + list(destinations)

            # Interpolate material based on score
            material_mod = self._interpolate_material(
                score,
                DARK_BLUE_MATERIAL_MOD, # OLIVE_MATERIAL_MOD
                WHITE_MATERIAL_MOD,  # DARK_BLUE_MATERIAL_MOD,
            )
            self.renderer.queue_highlight(
                highlight_rings,
                capture_duration,
                material_mod=material_mod,
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
                material_mod=DARK_RED_MATERIAL_MOD,
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
            material_mod=CORNFLOWER_BLUE_MATERIAL_MOD,
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
            self.task_delay_time
        )

        # Wait for final move animations to complete
        self.phase = self.PHASE_ANIMATING
