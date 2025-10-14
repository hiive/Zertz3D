"""Factory helpers for constructing Zertz game components."""

from __future__ import annotations

from typing import TextIO

from controller.zertz_game_controller import ZertzGameController
from renderer.zertz_renderer import ZertzRenderer
from renderer.text_renderer import TextRenderer
from renderer.composite_renderer import CompositeRenderer
from shared.interfaces import IRenderer
from game.zertz_board import ZertzBoard


class ZertzFactory:
    """Centralised factory for assembling ZertzGameController instances."""

    def __init__(self, text_stream: TextIO | None = None):
        self._text_stream = text_stream

    def create_controller(
        self,
        *,
        rings: int = 37,
        replay_file: str | None = None,
        seed: int | None = None,
        log_to_file: bool = False,
        partial_replay: bool = False,
        headless: bool = False,
        max_games: int | None = None,
        highlight_choices: bool = False,
        show_coords: bool = False,
        log_notation: bool = False,
        blitz: bool = False,
        move_duration: float = 0.666,
    ) -> ZertzGameController:
        """Create a fully-wired ZertzGameController with appropriate renderers."""

        def renderer_factory(controller: ZertzGameController) -> IRenderer:
            renderers: list[IRenderer] = []

            if not headless:
                board_layout = ZertzBoard.generate_standard_board_layout(controller.session.rings)
                renderers.append(
                    ZertzRenderer(
                        rings=controller.session.rings,
                        board_layout=board_layout,
                        show_coords=controller.show_coords,
                        highlight_choices=controller.highlight_choices,
                        update_callback=controller.update_game,
                        move_duration=controller.move_duration,
                    )
                )

            text_renderer = TextRenderer(stream=self._text_stream)
            renderers.append(text_renderer)

            if len(renderers) == 1:
                return renderers[0]
            return CompositeRenderer(renderers)

        return ZertzGameController(
            rings=rings,
            replay_file=replay_file,
            seed=seed,
            log_to_file=log_to_file,
            partial_replay=partial_replay,
            max_games=max_games,
            highlight_choices=highlight_choices,
            show_coords=show_coords,
            log_notation=log_notation,
            blitz=blitz,
            move_duration=move_duration,
            renderer_or_factory=renderer_factory,
        )
