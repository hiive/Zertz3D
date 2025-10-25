"""Manager class for action visualization sequencing."""

from shared.render_data import RenderData
from shared.materials_modifiers import (
    DARK_GREEN_MATERIAL_MOD,
    DARK_RED_MATERIAL_MOD,
    DARK_BLUE_MATERIAL_MOD,
    CORNFLOWER_BLUE_MATERIAL_MOD,
)
from renderer.panda3d.material_modifier import MaterialModifier


class ActionVisualizationSequencer:
    """Manages the multiphase highlighting sequence for showing moves."""

    # Highlight and animation durations (seconds)
    DEFAULT_HIGHLIGHT_DURATION = 1.0

    # Heat-map configuration
    MIN_VISIBILITY_THRESHOLD = 0.1  # Minimum normalized score to show highlight (default: 0.1)

    def __init__(self, renderer, highlight_choices: str | None = None, min_threshold: float = 0.1):
        """Initialize the sequencer.

        Args:
            renderer: PandaRenderer instance
            highlight_choices: Highlight mode ('uniform' or 'heatmap')
            min_threshold: Minimum normalized score threshold for heat-map visibility (default: 0.1)
        """
        self.renderer = renderer
        self.highlight_choices = highlight_choices
        self.min_threshold = min_threshold
        self.highlight_duration = self.DEFAULT_HIGHLIGHT_DURATION

        # State tracking
        self.active = False  # Whether sequencer is currently active
        self.start_delay = renderer.start_delay

    def is_active(self):
        """Check if the sequencer is currently active."""
        return self.active

    def _queue_position_highlights(
        self,
        positions,
        selected_position,
        material_mod,
        animation_delay,
        defer=0
    ):
        """Common logic for queuing highlights on positions (placements or removals).

        All positions stay highlighted through both phases to provide visual context,
        fading after the animation completes. Selected position stays visible slightly longer.

        Args:
            positions: List of dicts with position and score
            selected_position: The position that will be selected (stays visible longer)
            material_mod: Base material modifier for the highlight color
            animation_delay: Time for the marble/ring animation to complete
            defer: Delay before starting highlights (seconds)
        """
        if not positions:
            return

        for pos_dict in positions:
            pos_str = pos_dict['pos']
            score = pos_dict.get('score', 1.0)

            # Skip positions below visibility threshold in heatmap mode
            if self.highlight_choices == 'heatmap' and score < self.min_threshold:
                continue

            # Use color with alpha based on score
            interpolated_material = self._interpolate_material(score, material_mod)

            # Check if this is the selected position
            is_selected = (pos_str == selected_position)

            # Highlights last through: 2 phases (each 1 second) + animation
            # Selected position gets extra time to fade after others
            base_duration = 2 * self.highlight_duration + animation_delay
            duration = base_duration + (0.5 if is_selected else 0.0)

            self.renderer.queue_highlight(
                rings=[pos_str],
                material_mod=interpolated_material,
                duration=duration,
                defer=defer,
                entity_type="ring",  # Force highlight on rings only, not marbles
            )

    def _interpolate_material(self, score: float, base_mod: MaterialModifier) -> MaterialModifier:
        """Create material modifier with alpha based on normalized score.

        In uniform mode, all scores use full alpha (1.0).
        In heatmap mode, alpha scales from 0.2 (minimum) to 1.0 (maximum) using: 0.2 + 0.8 * score
        This ensures all valid options are visible while maintaining score differentiation.

        Args:
            score: Normalized score in range [0.0, 1.0]
            base_mod: Base material modifier (color will be used, alpha will be set by score)

        Returns:
            MaterialModifier with alpha set according to score
        """
        if self.highlight_choices == 'uniform':
            # Uniform mode: all actions get maximum alpha
            return base_mod

        # Heat-map mode: scale score to alpha range [0.2, 1.0]
        # This ensures all options are visible (min 20% opacity) while showing score differences
        alpha = 0.2 + 0.8 * score

        # Create new material with same colors but modified alpha
        return MaterialModifier(
            highlight_color=(
                base_mod.highlight_color[0],
                base_mod.highlight_color[1],
                base_mod.highlight_color[2],
                alpha
            ),
            emission_color=(
                base_mod.emission_color[0],
                base_mod.emission_color[1],
                base_mod.emission_color[2],
                alpha
            )
        )

    def start(self, player, render_data, task_delay_time):
        """Start the highlighting sequence for an action.

        Queues all highlights and animations upfront with appropriate delays.

        Args:
            player: Player making the move
            render_data: RenderData value object with action_dict and highlight data
            task_delay_time: Animation duration from controller's task
        """
        action_dict = render_data.action_dict
        action_type = action_dict["action"]

        if action_type == "PASS":
            # PASS has no visuals, don't activate sequencer
            return

        # Activate sequencer
        self.active = True

        if action_type == "PUT":
            self._start_put_sequence(player, render_data, task_delay_time)
        elif action_type == "CAP":
            self._start_cap_sequence(player, render_data, task_delay_time)

        # Reset start_delay for next action
        self.start_delay = 0

    def _start_put_sequence(self, player, render_data, task_delay_time):
        """Queue all animations and highlights for a PUT action.

        Args:
            player: Player making the move
            render_data: RenderData value object
            task_delay_time: Animation duration
        """
        action_dict = render_data.action_dict
        selected_ring = action_dict["dst"]

        # Queue placement highlights
        if render_data.placement_positions:
            self._queue_position_highlights(
                render_data.placement_positions,
                selected_ring,
                DARK_GREEN_MATERIAL_MOD,
                task_delay_time,
                defer=self.start_delay
            )

        # Queue marble animation to start after 2 highlight phases (2 seconds)
        marble_delay = self.start_delay + 2 * self.highlight_duration
        self.renderer.show_marble_placement(
            player,
            action_dict,
            task_delay_time,
            delay=marble_delay
        )

        # Queue removal highlights and ring removal if needed
        if render_data.removal_positions:
            selected_removal = action_dict.get("remove")
            removal_defer = marble_delay + task_delay_time

            self._queue_position_highlights(
                render_data.removal_positions,
                selected_removal,
                DARK_RED_MATERIAL_MOD,
                task_delay_time,
                defer=removal_defer
            )

            # Queue ring removal to start after 2 removal highlight phases
            ring_removal_delay = removal_defer + 2 * self.highlight_duration
            self.renderer.show_ring_removal(
                action_dict,
                task_delay_time,
                delay=ring_removal_delay
            )

    def _start_cap_sequence(self, player, render_data, task_delay_time):
        """Queue all animations and highlights for a CAP action.

        Args:
            player: Player making the move
            render_data: RenderData value object
            task_delay_time: Animation duration
        """
        action_dict = render_data.action_dict

        # Queue capture highlights (grouped by source, shown sequentially)
        total_capture_highlight_time = self._queue_capture_highlights(render_data.capture_moves)

        # Queue selected capture highlight after capture highlights complete
        src_ring = action_dict["src"]
        dst_ring = action_dict["dst"]
        selected_rings = [src_ring, dst_ring]

        self.renderer.queue_highlight(
            selected_rings,
            self.highlight_duration,
            material_mod=CORNFLOWER_BLUE_MATERIAL_MOD,
            defer=total_capture_highlight_time,
        )

        # Queue capture animation to start after selected highlight
        animation_delay = total_capture_highlight_time + self.highlight_duration
        minimal_render_data = RenderData(action_dict)

        # Use show_action with delay by temporarily modifying start_delay
        saved_start_delay = self.renderer.start_delay
        self.renderer.start_delay = animation_delay
        self.renderer.show_action(player, minimal_render_data, task_delay_time)
        self.renderer.start_delay = saved_start_delay

    def update(self, task):
        """Update the sequencer. Called each frame.

        Returns:
            bool: True if sequencer should continue, False if done
        """
        if not self.active:
            return False

        # Wait for all animations/highlights to complete
        if self.renderer.is_animation_active():
            return True

        # All done
        self.active = False
        return False

    def _queue_capture_highlights(self, capture_moves):
        """Queue highlights for all valid capture moves, grouped by source marble.

        Returns total time for all capture highlights (for scheduling subsequent animations).

        Args:
            capture_moves: List of capture move dicts with scores
                Format: [{'action': 'CAP', 'src': 'C4', 'dst': 'E6', ..., 'score': 0.9}, ...]
                OR legacy format: [{'action': 'CAP', 'src': 'C4', 'dst': 'E6', ...}, ...] (no scores)

        Returns:
            float: Total duration of all capture highlights in seconds
        """
        # Skip highlighting if only one capture available
        if not capture_moves or len(capture_moves) == 1:
            return 0.0

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
        defer_time = 0.0
        for src_str, (destinations, score) in captures_by_source.items():
            # Skip this source if below visibility threshold in heatmap mode
            if self.highlight_choices == 'heatmap' and score < self.min_threshold:
                continue

            # Highlight the source marble and all its possible destinations
            highlight_rings = [src_str] + list(destinations)

            # Use blue color with alpha based on score
            material_mod = self._interpolate_material(score, DARK_BLUE_MATERIAL_MOD)
            self.renderer.queue_highlight(
                highlight_rings,
                self.highlight_duration,
                material_mod=material_mod,
                defer=defer_time,
            )
            # Next group starts when this one ends
            defer_time += self.highlight_duration

        return defer_time
