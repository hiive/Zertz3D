"""Human player interaction manager for Zertz 3D.

Handles hover feedback, selection feedback, and visual highlighting for human players.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from shared.constants import MARBLE_TYPES


class HumanPlayerInteractionManager:
    """Manages visual feedback for human player interactions."""

    def __init__(self, renderer, session):
        """Initialize interaction manager.

        Args:
            renderer: The renderer instance (or None for headless mode)
            session: The game session instance
        """
        self.renderer = renderer
        self.session = session

    def clear_hover_feedback(self):
        """Clear all hover highlights from the renderer."""
        if self.renderer and hasattr(self.renderer, "clear_hover_highlights"):
            self.renderer.clear_hover_highlights()

    def update_hover_feedback(self, player) -> None:
        """Update hover feedback based on current player state.

        Args:
            player: The current player
        """
        if not self.renderer or not hasattr(self.renderer, "show_hover_feedback"):
            return
        if not hasattr(player, "get_current_options") or not hasattr(
            player, "get_selection_state"
        ):
            self.clear_hover_feedback()
            return

        options = player.get_current_options()
        if not options:
            self.clear_hover_feedback()
            return

        board = self.session.game.board
        state = player.get_selection_state()
        primary: set[str] = set()
        secondary: set[str] = set()
        supply_colors: set[str] = set()
        captured_targets: set[tuple[int, str]] = set()
        player_id = getattr(player, "n", None)

        def idx_to_label(idx: tuple[int, int] | None) -> str | None:
            if idx is None:
                return None
            y, x = idx
            try:
                # Check if ring is removed before converting
                if board.state[board.RING_LAYER, y, x] == 0:
                    return None
                from hiivelabs_mcts import coordinate_to_algebraic
                label = coordinate_to_algebraic(y, x, board.config)
            except (IndexError, ValueError):
                return None
            return label or None

        placement_mask, capture_mask = (
            player.get_context_masks()
            if hasattr(player, "get_context_masks")
            else (None, None)
        )
        context = options.get("context")

        if context == "placement":
            color = state.get("placement_color")
            color_idx = state.get("placement_color_idx")
            if color_idx is None:
                # Highlight all available supply colors
                counts = options.get("supply_counts", ())
                for idx, marble in enumerate(MARBLE_TYPES):
                    if idx < len(counts) and counts[idx] > 0:
                        supply_colors.add(marble)
            else:
                # Check if a specific marble is selected (has supply_key or captured_key)
                supply_key = state.get("placement_supply_key")
                captured_key = state.get("placement_captured_key")

                if supply_key:
                    # Highlight the specific supply marble, not all marbles of that color
                    primary.add(supply_key)
                elif captured_key:
                    # Highlight the specific captured marble, not all marbles of that color
                    primary.add(captured_key)
                else:
                    # Fallback: highlight all marbles of the selected color
                    if color:
                        supply_colors.add(color)
                    if (
                        player_id is not None
                        and state.get("placement_source") == "captured"
                        and color
                    ):
                        captured_targets.add((player_id, color))
            if player_id is not None and options.get("supply_total", 0) == 0:
                for marble in options.get("supply_colors", set()):
                    captured_targets.add((player_id, marble))
            placement_dst = state.get("placement_dst")
            if placement_dst is not None:
                label = idx_to_label(placement_dst)
                if label:
                    primary.add(label)
            pending_removals = state.get("placement_pending_removals") or set()
            for idx in pending_removals:
                label = idx_to_label(idx)
                if label:
                    secondary.add(label)

            if (
                color_idx is not None
                and placement_mask is not None
                and placement_mask.size > 0
            ):
                width = board.width
                if placement_dst is None:
                    dest_indices = np.where(np.any(placement_mask[color_idx], axis=1))[
                        0
                    ]
                    for flat in dest_indices:
                        y, x = divmod(int(flat), width)
                        label = idx_to_label((y, x))
                        if label:
                            primary.add(label)
                elif not pending_removals:
                    allow_none = state.get("placement_allow_none_removal", False)
                    if allow_none:
                        label = idx_to_label(placement_dst)
                        if label:
                            secondary.add(label)

        elif context == "capture":
            capture_src = state.get("capture_src")
            if capture_src is not None:
                label = idx_to_label(capture_src)
                if label:
                    primary.add(label)
                capture_paths = options.get("capture_paths", {}).get(capture_src, set())
                for idx in capture_paths:
                    label_dst = idx_to_label(idx)
                    if label_dst:
                        secondary.add(label_dst)
            else:
                for idx in options.get("capture_sources", set()):
                    label = idx_to_label(idx)
                    if label:
                        primary.add(label)

        hover_state = state.get("hover")
        if hover_state:
            h_type = hover_state.get("type")
            if h_type in ("ring", "board_marble"):
                label = idx_to_label(hover_state.get("index"))
                if label:
                    if context == "capture" and state.get("capture_src") is not None:
                        secondary.add(label)
                    elif (
                        context == "placement"
                        and state.get("placement_dst") is not None
                    ):
                        secondary.add(label)
                    else:
                        primary.add(label)
            # Note: Supply/captured marble hovering is now handled above by checking
            # for specific marble keys (supply_key/captured_key) and adding them to
            # primary highlights instead of supply_colors/captured_targets, which
            # would highlight ALL marbles of that color instead of just the selected one.

        if not primary and not secondary and not supply_colors and not captured_targets:
            self.clear_hover_feedback()
            return

        self.renderer.show_hover_feedback(
            primary, secondary, supply_colors, captured_targets
        )

    def handle_renderer_selection(self, selection: dict) -> None:
        """Handle selection event from renderer.

        Args:
            selection: Selection data from renderer
        """
        player = self.session.get_current_player()
        payload = dict(selection)
        label = selection.get("label")
        if label:
            try:
                from hiivelabs_mcts import algebraic_to_coordinate
                payload["index"] = algebraic_to_coordinate(label, self.session.game.board.config)
            except ValueError:
                payload["index"] = None

        index = payload.get("index")
        if index is not None:
            payload["index"] = (int(index[0]), int(index[1]))

        player.handle_selection(payload)
        self.update_hover_feedback(player)

    def handle_renderer_hover(self, hover: Optional[dict]) -> None:
        """Handle hover event from renderer.

        Args:
            hover: Hover data from renderer (or None to clear)
        """
        player = self.session.get_current_player()
        if not hasattr(player, "handle_hover"):
            return

        payload = None
        if hover:
            payload = dict(hover)
            label = hover.get("label")
            if label:
                try:
                    from hiivelabs_mcts import algebraic_to_coordinate
                    payload["index"] = algebraic_to_coordinate(label, self.session.game.board.config)
                except ValueError:
                    payload["index"] = None
            index = payload.get("index")
            if index is not None:
                payload["index"] = (int(index[0]), int(index[1]))

        player.handle_hover(payload)
        self.update_hover_feedback(player)