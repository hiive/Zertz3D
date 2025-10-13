"""Shared protocol definitions."""

from __future__ import annotations

from typing import Protocol, Callable, Any, TYPE_CHECKING, runtime_checkable

from shared.render_data import RenderData

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from controller.zertz_game_controller import ZertzGameController


@runtime_checkable
class IRenderer(Protocol):
    """Protocol describing renderer capabilities required by the controller."""

    def run(self) -> None:
        ...

    def reset_board(self) -> None:
        ...

    def execute_action(
        self,
        player: Any,
        render_data: RenderData,
        action_result: Any,
        task_delay_time: float,
        on_complete: Callable[[Any, Any], None] | None,
    ) -> None:
        ...

    def attach_update_loop(self, update_fn: Callable[[], bool], interval: float) -> bool:
        ...

    def show_isolated_removal(self, player: Any, pos: str, marble: str | None, delay_time: float) -> None:
        ...

    def report_status(self, message: str) -> None:
        ...


@runtime_checkable
class IRendererFactory(Protocol):
    """Protocol for factories that produce renderers for a controller."""

    def __call__(self, controller: ZertzGameController) -> IRenderer:
        ...
