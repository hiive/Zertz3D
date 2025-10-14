"""Composite renderer that forwards calls to multiple renderer implementations."""

from __future__ import annotations

from typing import Iterable, List, Any, Callable

from shared.interfaces import IRenderer
from shared.render_data import RenderData


class CompositeRenderer(IRenderer):
    """Composite pattern implementation for chaining renderers."""

    def __init__(self, renderers: Iterable[IRenderer]):
        self._renderers: List[IRenderer] = [renderer for renderer in renderers if renderer is not None]
        if not self._renderers:
            raise ValueError("CompositeRenderer requires at least one renderer instance.")

    def run(self) -> None:
        for renderer in self._renderers:
            renderer.run()

    def reset_board(self) -> None:
        for renderer in self._renderers:
            renderer.reset_board()

    def execute_action(
        self,
        player: Any,
        render_data: RenderData,
        action_result: Any,
        task_delay_time: float,
        on_complete: Callable[[Any, Any], None] | None,
    ) -> None:
        if len(self._renderers) == 1:
            self._renderers[0].execute_action(player, render_data, action_result, task_delay_time, on_complete)
            return

        if on_complete is None:
            for renderer in self._renderers:
                renderer.execute_action(player, render_data, action_result, task_delay_time, None)
            return

        completed = set()

        def make_callback(renderer: IRenderer) -> Callable[[Any, Any], None]:
            def _callback(p: Any, result: Any) -> None:
                completed.add(renderer)
                if len(completed) == len(self._renderers):
                    on_complete(p, result)
            return _callback

        for renderer in self._renderers:
            renderer.execute_action(
                player,
                render_data,
                action_result,
                task_delay_time,
                make_callback(renderer),
            )

    def show_isolated_removal(self, player: Any, pos: str, marble: Any, delay_time: float) -> None:
        for renderer in self._renderers:
            renderer.show_isolated_removal(player, pos, marble, delay_time)

    def set_context_highlights(self, context, positions, color=None, emission=None):
        for renderer in self._renderers:
            if hasattr(renderer, "set_context_highlights"):
                renderer.set_context_highlights(context, positions, color=color, emission=emission)

    def clear_context_highlights(self, context=None):
        for renderer in self._renderers:
            if hasattr(renderer, "clear_context_highlights"):
                renderer.clear_context_highlights(context)

    def highlight_context(self, context, positions):
        for renderer in self._renderers:
            if hasattr(renderer, "highlight_context"):
                renderer.highlight_context(context, positions)

    def clear_highlight_context(self, context=None):
        for renderer in self._renderers:
            if hasattr(renderer, "clear_highlight_context"):
                renderer.clear_highlight_context(context)

    def apply_context_masks(self, board, placement_mask, capture_mask):
        for renderer in self._renderers:
            if hasattr(renderer, "apply_context_masks"):
                renderer.apply_context_masks(board, placement_mask, capture_mask)

    def report_status(self, message: str) -> None:
        for renderer in self._renderers:
            renderer.report_status(message)

    def attach_update_loop(self, update_fn: Callable[[], bool], interval: float) -> bool:
        handled = False
        for renderer in self._renderers:
            if hasattr(renderer, "attach_update_loop"):
                if renderer.attach_update_loop(update_fn, interval):
                    handled = True
        return handled
