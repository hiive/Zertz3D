"""Manager class for action visualization sequencing."""
from shared.render_data import RenderData
from shared.materials_modifiers import (
    DARK_GREEN_MATERIAL_MOD,
    DARK_RED_MATERIAL_MOD,
    DARK_BLUE_MATERIAL_MOD,
    CORNFLOWER_BLUE_MATERIAL_MOD,
    BRIGHT_YELLOW_MATERIAL_MOD,
)
from renderer.panda3d.material_modifier import MaterialModifier


class ActionVisualizationSequencer:
    """Manages the multiphase highlighting sequence for showing moves."""

    # Highlight and animation durations (seconds)
    DEFAULT_HIGHLIGHT_DURATION = 0.5

    def __init__(self, renderer, highlight_choices: str | None = None):
        """Initialize the sequencer.

        Args:
            renderer: PandaRenderer instance
            highlight_choices: Highlight mode ('uniform' or 'heatmap')
        """
        self.renderer = renderer
        self.highlight_choices = highlight_choices
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

            # Use color with alpha based on score
            interpolated_material = self._interpolate_material(score, material_mod)

            # Check if this is the selected position
            is_selected = (pos_str == selected_position)

            # Highlights last through: 2 phases (each 1 second) + animation
            # Selected position gets extra time to fade after others
            base_duration = self.highlight_duration + animation_delay
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
        In heatmap mode, applies exponential scaling to exaggerate differences:
        - Uses score^3 to make differences more visible
        - Low scores (0.0-0.5) become very dim
        - High scores (0.8-1.0) remain bright
        This prevents all actions from looking the same when visit counts are similar.

        Args:
            score: Normalized score in range [0.0, 1.0]
            base_mod: Base material modifier (color will be used, alpha will be set by score)

        Returns:
            MaterialModifier with alpha set according to score
        """
        if self.highlight_choices == 'uniform':
            # Uniform mode: all actions get maximum alpha
            return base_mod

        # Heat-map mode: use exponential scaling to exaggerate differences
        # score^3 makes low scores much dimmer while keeping high scores bright
        # Examples: 0.5^3=0.125, 0.7^3=0.343, 0.9^3=0.729, 1.0^3=1.0
        # score = Random().uniform(0., 1.)
        brightness = score #** 3

        # Scale RGB values by brightness (alpha stays at 1.0 for full opacity)
        return MaterialModifier(
            highlight_color=(
                base_mod.highlight_color[0] * brightness,
                base_mod.highlight_color[1] * brightness,
                base_mod.highlight_color[2] * brightness,
                1.0  # Full opacity
            ),
            emission_color=(
                base_mod.emission_color[0] * brightness,
                base_mod.emission_color[1] * brightness,
                base_mod.emission_color[2] * brightness,
                1.0  # Full opacity
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

        # Queue marble animation to start after 1 highlight phase (0.5 seconds)
        marble_delay = self.start_delay + self.highlight_duration
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

            # Queue ring removal to start after 1 removal highlight phase
            ring_removal_delay = removal_defer + self.highlight_duration
            self.renderer.show_ring_removal(
                action_dict,
                task_delay_time,
                delay=ring_removal_delay
            )

    def _start_cap_sequence(self, player, render_data, task_delay_time):
        """Queue all animations and highlights for a CAP action.

        For captures, all highlights complete before the marble animation starts.

        Args:
            player: Player making the move
            render_data: RenderData value object
            task_delay_time: Animation duration
        """
        action_dict = render_data.action_dict

        # Queue capture highlights (grouped by source, shown sequentially)
        total_capture_highlight_time = self._queue_capture_highlights(render_data.capture_moves)

        # All highlights start at the same time (after all options highlights complete)
        selected_highlight_defer = self.start_delay + total_capture_highlight_time

        # Calculate timing for capture animation
        animation_delay = total_capture_highlight_time + self.highlight_duration

        # Build timing dict for renderer
        timing = {
            'capturing_marble_defer': animation_delay,
            'captured_marble_defer': animation_delay + 2 * task_delay_time + self.highlight_duration,
            'flash_captured_marble': True,
            'flash_defer': selected_highlight_defer,
            'flash_duration': self.highlight_duration,
        }

        # Create render_data with timing
        render_data_with_timing = RenderData(action_dict, timing=timing)

        # Call renderer's show_action method with timing
        self.renderer.show_action(player, render_data_with_timing, task_delay_time)

        # Queue selected capture highlight AFTER show_action has updated board state
        # This ensures pos_to_marble[dst] is set before highlights are applied
        # Extract positions for selected capture highlights
        src_ring = action_dict["src"]
        dst_ring = action_dict["dst"]
        cap_ring = action_dict["cap"]

        # Get the captured marble's key (it's now in the capture pool)
        # The marble has been configured with a captured key by the renderer
        captured_marble_color = action_dict["capture"]
        captured_marbles = self.renderer.capture_pools[player.n][captured_marble_color]
        if captured_marbles:
            # Get the most recently captured marble (just added by renderer)
            captured_marble = captured_marbles[-1]
            captured_key = self.renderer._captured_key(captured_marble)

            # Queue yellow flash for captured marble (1 phase, half duration of placement)
            self.renderer.queue_highlight(
                [captured_key],
                self.highlight_duration,
                material_mod=BRIGHT_YELLOW_MATERIAL_MOD,
                defer=selected_highlight_defer,
                entity_type="captured_marble",
            )

        # Highlight src ring only (marble moved away in board state) - 1 phase
        self.renderer.queue_highlight(
            [src_ring],
            self.highlight_duration,
            material_mod=CORNFLOWER_BLUE_MATERIAL_MOD,
            defer=selected_highlight_defer,
            entity_type="ring",
        )

        # Highlight everything at dst (marble + ring) - 1 phase
        self.renderer.queue_highlight(
            [dst_ring],
            self.highlight_duration,
            material_mod=CORNFLOWER_BLUE_MATERIAL_MOD,
            defer=selected_highlight_defer,
            entity_type=None,  # Discover all entities at this position
        )

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

        # Queue highlights sequentially - each group completes before the next starts
        defer_time = 0.0
        for src_str, (destinations, score) in captures_by_source.items():
            # Use blue color with alpha based on score
            material_mod = self._interpolate_material(score, DARK_BLUE_MATERIAL_MOD)

            # Highlight everything at the source position (marble + ring)
            self.renderer.queue_highlight(
                [src_str],
                self.highlight_duration,
                material_mod=material_mod,
                defer=defer_time,
                entity_type=None,  # Discover all entities at this position
            )

            # Highlight all destination rings
            if destinations:
                self.renderer.queue_highlight(
                    list(destinations),
                    self.highlight_duration,
                    material_mod=material_mod,
                    defer=defer_time,
                    entity_type="ring",
                )

            # Next group starts when this one completes (defer advances by duration)
            defer_time += self.highlight_duration

        # Return total time for all sequential highlights to complete
        return defer_time
