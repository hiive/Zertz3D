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

        # Thinking phase state
        self.thinking_active = False
        self.thinking_highlights = {}  # {pos_str: (entity, entity_type, original_material)}
        self.skip_placement_highlights = False  # Skip placement highlights if thinking just happened

    def is_active(self):
        """Check if the sequencer is currently active."""
        return self.active

    def on_mcts_event(self, event: dict):
        """Handle MCTS search events for thinking visualization.

        Args:
            event: Event dict from Rust MCTS with keys:
                - 'event': 'SearchStarted' | 'SearchProgress' | 'SearchEnded'
                - Additional data depending on event type
        """
        event_type = event['event']

        if event_type == 'SearchStarted':
            self._on_search_started(event)
        elif event_type == 'SearchProgress':
            self._on_search_progress(event)
        elif event_type == 'SearchEnded':
            self._on_search_ended(event)

    def _on_search_started(self, event):
        """Start thinking phase visualization.

        Minimal setup - clears previous highlights and marks phase as active.
        Highlights will be applied by the first SearchProgress event.

        Args:
            event: SearchStarted event with:
                - 'total_iterations': total number of MCTS iterations
        """
        # Skip if renderer window isn't open yet (first move before rendering starts)
        if not hasattr(self.renderer, 'win') or not self.renderer.win:
            return

        # Clear ALL highlights from previous action execution phase
        # This includes queued and active highlights from placement/removal/capture
        self.renderer.highlighting_manager.clear()

        # Also clear ALL context highlights (including previous thinking highlights)
        self.renderer.highlighting_manager.clear_context_highlights(None)

        # Mark thinking phase as active - that's all we need to do
        # SearchProgress will handle the actual highlighting
        self.thinking_active = True
        self.thinking_highlights = {}

    def _on_search_progress(self, event):
        """Update thinking phase highlights based on current MCTS scores.

        Directly modifies materials on active thinking highlights (no fade animations).

        Args:
            event: SearchProgress event with:
                - 'iteration': current iteration number
                - 'action_stats': list of (action_type, action_data, score) tuples
                    Same format as get_last_action_scores():
                    [('PUT', (marble_type, dst_flat, rem_flat), 0.8), ...]
        """
        if not self.thinking_active:
            return

        # Let Panda3D render a frame to keep the visualization responsive
        # This is critical - without this, the entire app freezes during MCTS search
        # We only need to render graphics - taskMgr is already running (that's how we got here)
        if hasattr(self.renderer, 'graphicsEngine') and self.renderer.graphicsEngine:
            # Render a frame to update the display
            self.renderer.graphicsEngine.renderFrame()
        # Convert action_stats list to dict for enrichment methods
        action_stats = event.get('action_stats', [])
        iteration = event.get('iteration', 0)

        # Apply an overall brightness multiplier based on search progress
        # Early iterations should be very dim, later iterations brighter
        # This shows increasing confidence as the search progresses
        total_iterations = event.get('total_iterations', 50000)
        progress = min(1.0, iteration / min(5000, total_iterations * 0.5))  # Ramp up over first half or 5000 iters
        overall_brightness = progress * progress * progress # 0.05 + (0.95 * progress)  # Scale from 5% to 100% brightness

        action_scores = {(action_type, action_data): score
                        for action_type, action_data, score in action_stats}

        # We need access to the game to use enrichment methods
        # The renderer should have a reference to the game through the board
        # For now, we'll calculate position scores directly from action_scores

        # Extract placement, removal, and capture scores by position
        placement_scores = {}  # {pos_str: max_score}
        removal_scores = {}    # {pos_str: max_score}

        width = event.get('board_width', 7)

        # Process action scores to extract position-level scores
        if hasattr(self.renderer, 'pos_array') and self.renderer.pos_array is not None:
            pos_array = self.renderer.pos_array

            for (action_type, action_data), score in action_scores.items():
                if action_type == 'PUT' and action_data:
                    marble_type, dst_flat, rem_flat = action_data

                    # Convert dst_flat to position string
                    dst_pos = self._flat_to_pos_str(dst_flat, width, pos_array)
                    if dst_pos:
                        placement_scores[dst_pos] = max(placement_scores.get(dst_pos, 0.0), score)

                    # Convert rem_flat to position string (if not "no removal")
                    if rem_flat < width * width:
                        rem_pos = self._flat_to_pos_str(rem_flat, width, pos_array)
                        if rem_pos:
                            removal_scores[rem_pos] = max(removal_scores.get(rem_pos, 0.0), score)

        # Update highlight materials based on scores
        # We need to update the context highlights with new brightness

        # Recalculate which positions are both placement AND removal
        all_placement = set(placement_scores.keys())
        all_removal = set(removal_scores.keys())
        both_positions = all_placement & all_removal
        only_placement = all_placement - both_positions
        only_removal = all_removal - both_positions

        # print(f"[DEBUG Progress] Converted {len(placement_scores)} placement, {len(removal_scores)} removal positions")

        # Update highlights by applying individual materials per position based on scores
        # We need to call set_context_highlights separately for each position to get different brightnesses

        # Clear existing highlights first
        self.renderer.highlighting_manager.clear_context_highlights("thinking_placement")
        self.renderer.highlighting_manager.clear_context_highlights("thinking_removal")
        self.renderer.highlighting_manager.clear_context_highlights("thinking_both")

        # Apply individual placement highlights with per-position scores
        if only_placement:
            for pos_str in only_placement:
                score = placement_scores.get(pos_str, 0.5) * overall_brightness
                material_mod = self._interpolate_material(score, DARK_GREEN_MATERIAL_MOD)
                # Each position gets its own context subkey to avoid clearing others
                self.renderer.highlighting_manager.set_context_highlights(
                    f"thinking_placement_{pos_str}",
                    [pos_str],
                    material_mod,
                )
            # print(f"[DEBUG Progress] Applied {len(only_placement)} green highlights with individual scores")

        # Apply individual removal highlights
        if only_removal:
            for pos_str in only_removal:
                score = removal_scores.get(pos_str, 0.5) * overall_brightness
                material_mod = self._interpolate_material(score, DARK_RED_MATERIAL_MOD)
                self.renderer.highlighting_manager.set_context_highlights(
                    f"thinking_removal_{pos_str}",
                    [pos_str],
                    material_mod,
                )
            # print(f"[DEBUG Progress] Applied {len(only_removal)} red highlights with individual scores")

        # Apply individual "both" highlights with additive color blending
        if both_positions:
            for pos_str in both_positions:
                placement_score = placement_scores.get(pos_str, 0.0) * overall_brightness
                removal_score = removal_scores.get(pos_str, 0.0) * overall_brightness

                # Get the individual color contributions
                green_material = self._interpolate_material(placement_score, DARK_GREEN_MATERIAL_MOD)
                red_material = self._interpolate_material(removal_score, DARK_RED_MATERIAL_MOD)

                # Additively blend the colors (green + red = yellow when both are bright)
                blended_material = MaterialModifier(
                    highlight_color=(
                        green_material.highlight_color[0] + red_material.highlight_color[0],
                        green_material.highlight_color[1] + red_material.highlight_color[1],
                        green_material.highlight_color[2] + red_material.highlight_color[2],
                        1.0
                    ),
                    emission_color=(
                        green_material.emission_color[0] + red_material.emission_color[0],
                        green_material.emission_color[1] + red_material.emission_color[1],
                        green_material.emission_color[2] + red_material.emission_color[2],
                        1.0
                    )
                )

                self.renderer.highlighting_manager.set_context_highlights(
                    f"thinking_both_{pos_str}",
                    [pos_str],
                    blended_material,
                )
            # print(f"[DEBUG Progress] Applied {len(both_positions)} blended highlights with individual scores")

        # Track highlighted positions for cleanup
        self.thinking_highlights = {
            'placement': only_placement,
            'removal': only_removal,
            'both': both_positions,
        }

    def _on_search_ended(self, event):
        """End thinking phase and prepare for action execution.

        Leaves highlights active - they will be selectively cleared when the
        selected action is known in start().

        Args:
            event: SearchEnded event with:
                - 'total_iterations': final iteration count
                - 'total_time_ms': total search time
        """
        if not self.thinking_active:
            return

        # DON'T clear highlights here - we want to leave the selected positions
        # highlighted for visual continuity. The start() method will selectively
        # clear non-selected highlights.

        # Mark thinking phase as inactive
        self.thinking_active = False

        # Skip placement highlights for the next action execution (they were already shown during thinking)
        self.skip_placement_highlights = True

    def _fade_thinking_highlights(self):
        """Fade out all thinking highlights by scheduling delayed clear."""
        # Simply clear the context highlights after a delay
        # The highlighting system doesn't support fading context highlights directly,
        # so we just clear them immediately for now
        # TODO: If we want a fade effect, we'd need to convert context highlights to timed highlights
        self._clear_thinking_highlights()

    def _clear_thinking_highlights(self):
        """Remove all thinking phase highlights immediately."""
        # Clear all thinking context highlights (both old bulk and new individual)
        self.renderer.highlighting_manager.clear_context_highlights("thinking_placement")
        self.renderer.highlighting_manager.clear_context_highlights("thinking_removal")
        self.renderer.highlighting_manager.clear_context_highlights("thinking_both")

        # Also clear all individual position highlights
        if 'placement' in self.thinking_highlights:
            for pos_str in self.thinking_highlights['placement']:
                self.renderer.highlighting_manager.clear_context_highlights(f"thinking_placement_{pos_str}")

        if 'removal' in self.thinking_highlights:
            for pos_str in self.thinking_highlights['removal']:
                self.renderer.highlighting_manager.clear_context_highlights(f"thinking_removal_{pos_str}")

        if 'both' in self.thinking_highlights:
            for pos_str in self.thinking_highlights['both']:
                self.renderer.highlighting_manager.clear_context_highlights(f"thinking_both_{pos_str}")

        self.thinking_highlights.clear()

    def _clear_non_selected_thinking_highlights(self, selected_placement, selected_removal):
        """Remove thinking highlights except for the selected placement/removal positions.

        Args:
            selected_placement: Position string for the selected placement (e.g., 'C4')
            selected_removal: Position string for the selected removal (e.g., 'D5'), or None if no removal
        """
        # Clear bulk highlights (won't affect individual position highlights)
        self.renderer.highlighting_manager.clear_context_highlights("thinking_placement")
        self.renderer.highlighting_manager.clear_context_highlights("thinking_removal")
        self.renderer.highlighting_manager.clear_context_highlights("thinking_both")

        # Clear non-selected placement highlights
        if 'placement' in self.thinking_highlights:
            for pos_str in self.thinking_highlights['placement']:
                if pos_str != selected_placement:
                    self.renderer.highlighting_manager.clear_context_highlights(f"thinking_placement_{pos_str}")

        # Clear non-selected removal highlights
        if 'removal' in self.thinking_highlights:
            for pos_str in self.thinking_highlights['removal']:
                if pos_str != selected_removal:
                    self.renderer.highlighting_manager.clear_context_highlights(f"thinking_removal_{pos_str}")

        # Clear both highlights (unless they match selected positions)
        if 'both' in self.thinking_highlights:
            for pos_str in self.thinking_highlights['both']:
                # Keep if it's the selected placement OR selected removal
                if pos_str != selected_placement and pos_str != selected_removal:
                    self.renderer.highlighting_manager.clear_context_highlights(f"thinking_both_{pos_str}")

        # Don't clear the thinking_highlights dict - we might need it later
        # But do clear the highlight references for cleared positions
        if 'placement' in self.thinking_highlights:
            self.thinking_highlights['placement'] = {selected_placement} & self.thinking_highlights['placement']
        if 'removal' in self.thinking_highlights:
            self.thinking_highlights['removal'] = {selected_removal} & self.thinking_highlights['removal'] if selected_removal else set()
        if 'both' in self.thinking_highlights:
            self.thinking_highlights['both'] = ({selected_placement, selected_removal} & self.thinking_highlights['both']) if selected_removal else ({selected_placement} & self.thinking_highlights['both'])

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

    def _flat_to_pos_str(self, flat_idx: int, width: int, pos_array) -> str | None:
        """Convert flat index to position string.

        Args:
            flat_idx: Flat index (y * width + x)
            width: Board width
            pos_array: Renderer's pos_array (2D array of position strings)

        Returns:
            Position string (e.g., 'B5') or None if invalid
        """
        try:
            y = flat_idx // width
            x = flat_idx % width
            pos_str = str(pos_array[y][x])
            return pos_str if pos_str and pos_str != '' else None
        except:
            return None

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
        selected_removal = action_dict.get("remove")

        # If thinking phase just happened, selectively clear non-selected highlights
        if self.skip_placement_highlights and self.thinking_highlights:
            self._clear_non_selected_thinking_highlights(selected_ring, selected_removal)

        # Queue placement highlights (skip if thinking phase just showed them)
        if render_data.placement_positions and not self.skip_placement_highlights:
            self._queue_position_highlights(
                render_data.placement_positions,
                selected_ring,
                DARK_GREEN_MATERIAL_MOD,
                task_delay_time,
                defer=self.start_delay
            )
            highlight_duration = self.highlight_duration
        elif self.skip_placement_highlights:
            # Thinking phase already showed all placements, just highlight the selected one
            self.renderer.queue_highlight(
                rings=[selected_ring],
                material_mod=DARK_GREEN_MATERIAL_MOD,
                duration=self.highlight_duration + task_delay_time + 0.5,  # Same as selected position duration
                defer=self.start_delay,
                entity_type="ring",
            )
            highlight_duration = self.highlight_duration
        else:
            # No placement positions at all
            highlight_duration = 0

        # Reset the skip flag after checking it
        self.skip_placement_highlights = False

        # Queue marble animation to start after 1 highlight phase (0.5 seconds)
        marble_delay = self.start_delay + highlight_duration
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

            # For selection phase, only highlight the actual ring being removed
            # (render_data contains all possible removals with scores, but we only want the selected one)
            selected_removal_only = [
                pos_dict for pos_dict in render_data.removal_positions
                if pos_dict.get('pos') == selected_removal or pos_dict == selected_removal
            ]

            # If no match found (shouldn't happen), show all with full brightness on selected
            if not selected_removal_only and selected_removal:
                selected_removal_only = [{'pos': selected_removal, 'score': 1.0}]

            self._queue_position_highlights(
                selected_removal_only,
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
