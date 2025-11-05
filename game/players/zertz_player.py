# import sys
#
# sys.path.append('..')
from __future__ import annotations

from typing import Optional

from game.zertz_game import ZertzGame
import numpy as np


class ZertzPlayer:
    """Base player with shared state and lifecycle hooks."""

    def __init__(self, game: ZertzGame, n):
        self.captured = {"b": 0, "g": 0, "w": 0}
        self.game = game
        self.n = n
        self.name = f"Player {n}"

    def get_action(self):
        raise NotImplementedError

    def get_last_action_scores(self):
        """Get normalized scores for all legal actions from last search.

        Returns:
            Dict mapping action tuples to normalized scores [0.0, 1.0]
        """
        raise NotImplementedError

    #
    # Lifecycle hooks
    #
    def on_turn_start(
        self,
        context: str,
        placement_mask: Optional[np.ndarray],
        capture_mask: Optional[np.ndarray],
    ) -> None:
        """Inform the player that a new decision context has begun."""
        return None

    def clear_context(self) -> None:
        """Signal that the current decision context is complete."""
        return None

    def handle_selection(self, selection: dict) -> bool:
        """Handle renderer-driven selection events (default: ignore)."""
        return False

    def handle_hover(self, selection: Optional[dict]) -> bool:
        return False

    def add_capture(self, capture):
        self.captured[capture] += 1


