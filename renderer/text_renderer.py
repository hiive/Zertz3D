"""Text-based renderer implementation for status output."""

from __future__ import annotations

import sys
from typing import Any, Callable, TextIO

from shared.interfaces import IRenderer
from shared.render_data import RenderData


class TextRenderer(IRenderer):
    """Minimal renderer that emits textual status updates to a stream."""

    def __init__(self, stream: TextIO | None = None):
        self._stream: TextIO = stream or sys.stdout

    def run(self) -> None:
        """No-op run loop for text renderer."""
        pass

    def reset_board(self) -> None:
        self.report_status("Board reset.")

    def execute_action(
        self,
        player: Any,
        render_data: RenderData,
        action_result: Any,
        task_delay_time: float,
        on_complete: Callable[[Any, Any], None] | None,
    ) -> None:
        """Text renderer does not animate; immediately notify completion."""
        action = render_data.action_dict
        self.report_status(
            f"Executing action for Player {getattr(player, 'n', '?')}: {action}"
        )
        if on_complete:
            on_complete(player, action_result)

    def show_isolated_removal(
        self, player: Any, pos: str, marble: str | None, delay_time: float
    ) -> None:
        marble_label = marble if marble is not None else "none"
        self.report_status(
            f"Isolation capture for Player {getattr(player, 'n', '?')}: {pos} ({marble_label})"
        )

    def report_status(self, message: str) -> None:
        if message is None:
            return
        print(message, file=self._stream)

    def attach_update_loop(
        self, update_fn: Callable[[], bool], interval: float
    ) -> bool:
        return False

    # Context highlighting is a no-op for the text renderer
    def set_context_highlights(
        self, context, positions, color=None, emission=None
    ) -> None:
        pass

    def clear_context_highlights(self, context=None) -> None:
        pass

    def highlight_context(self, context, positions) -> None:
        pass

    def clear_highlight_context(self, context=None) -> None:
        pass

    def set_selection_callback(self, callback) -> None:
        pass

    def show_hover_feedback(
        self, primary=None, secondary=None, supply_colors=None, captured_targets=None
    ) -> None:
        pass

    def clear_hover_highlights(self) -> None:
        pass

    def set_hover_callback(self, callback) -> None:
        pass

    def apply_context_masks(self, board, placement_mask, capture_mask) -> None:
        pass
