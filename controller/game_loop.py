"""Game loop orchestration for Zertz 3D."""

from __future__ import annotations

from dataclasses import dataclass

from shared.interfaces import IRenderer


@dataclass
class _LoopTask:
    """Internal task representation passed to `update_game`."""

    delay_time: float

    def __post_init__(self) -> None:  # pragma: no cover - trivial setters
        self.done = object()
        self.again = object()


class GameLoop:
    """Drives the controller update cycle independent of renderer mode."""

    def __init__(self, controller, renderer: IRenderer | None, move_duration: float) -> None:
        self._controller = controller
        self._renderer = renderer
        self._move_duration = move_duration

    def run(self) -> None:
        """Run the game loop until the controller signals completion."""

        if self._renderer:
            try:
                if self._renderer.attach_update_loop(lambda: self._tick(self._move_duration), self._move_duration):
                    self._renderer.run()
                    return
            except AttributeError:
                pass

        while not self._tick(self._move_duration):
            pass

    def _tick(self, delay: float) -> bool:
        task = _LoopTask(delay_time=delay)
        result = self._controller.update_game(task)
        return result == getattr(task, "done", None)
