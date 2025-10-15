"""Shared protocol definitions."""

from __future__ import annotations

from typing import Protocol, Callable, Any, TYPE_CHECKING, runtime_checkable

from shared.render_data import RenderData

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from controller.zertz_game_controller import ZertzGameController


@runtime_checkable
class IRenderer(Protocol):
    """Protocol describing renderer capabilities required by the controller."""

    def run(self) -> None: ...

    def reset_board(self) -> None: ...

    def execute_action(
        self,
        player: Any,
        render_data: RenderData,
        action_result: Any,
        task_delay_time: float,
        on_complete: Callable[[Any, Any], None] | None,
    ) -> None: ...

    def attach_update_loop(
        self, update_fn: Callable[[], bool], interval: float
    ) -> bool: ...

    def show_isolated_removal(
        self, player: Any, pos: str, marble: str | None, delay_time: float
    ) -> None: ...

    def report_status(self, message: str) -> None: ...

    def set_context_highlights(
        self,
        context: str,
        positions: list[str] | set[str],
        color: Any | None = None,
        emission: Any | None = None,
    ) -> None: ...

    def clear_context_highlights(self, context: str | None = None) -> None: ...

    def highlight_context(
        self, context: str, positions: list[str] | set[str]
    ) -> None: ...

    def clear_highlight_context(self, context: str | None = None) -> None: ...

    def apply_context_masks(
        self, board: Any, placement_mask: Any, capture_mask: Any
    ) -> None: ...

    def set_selection_callback(
        self, callback: Callable[[dict], None] | None
    ) -> None: ...

    def set_hover_callback(
        self, callback: Callable[[dict | None], None] | None
    ) -> None: ...

    def show_hover_feedback(
        self,
        primary: Any | None = None,
        secondary: Any | None = None,
        supply_colors: Any | None = None,
        captured_targets: Any | None = None,
    ) -> None: ...

    def clear_hover_highlights(self) -> None: ...


@runtime_checkable
class IRendererFactory(Protocol):
    """Protocol for factories that produce renderers for a controller."""

    def __call__(self, controller: ZertzGameController) -> IRenderer: ...
