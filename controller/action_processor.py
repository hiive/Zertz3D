"""Utilities for applying post-action effects in the controller layer."""

from __future__ import annotations

from typing import Any

from game.action_result import ActionResult
from shared.interfaces import IRenderer


class ActionProcessor:
    """Encapsulate controller-side post-action handling.

    Applies capture bookkeeping to the active player and forwards isolation
    removals to the renderer when present. By centralizing this logic we make
    it easier to unit test without relying on the full controller.
    """

    def __init__(self, renderer: IRenderer | None = None) -> None:
        self._renderer = renderer

    def set_renderer(self, renderer: IRenderer | None) -> None:
        """Update the renderer reference if the controller swaps renderers."""
        self._renderer = renderer

    def process(self, player: Any, result: ActionResult | None, delay_time: float) -> None:
        """Apply the effects of ``result`` for ``player``.

        Args:
            player: Active player executing the action.
            result: ActionResult returned by the game layer. May be ``None``.
            delay_time: Animation duration associated with the action.
        """
        if result is None:
            return

        if result.is_isolation():
            self._handle_isolation(player, result, delay_time)
            return

        if result.has_captures():
            player.add_capture(result.captured_marbles)

    # Internal helpers -------------------------------------------------

    def _handle_isolation(self, player: Any, result: ActionResult, delay_time: float) -> None:
        """Handle isolation captures and renderer notifications."""
        if not isinstance(result.captured_marbles, list):
            return

        for removal in result.captured_marbles:
            marble = removal.get('marble')
            if marble:
                player.add_capture(marble)
            if self._renderer is not None:
                self._renderer.show_isolated_removal(
                    player,
                    removal.get('pos', ''),
                    marble,
                    delay_time,
                )
