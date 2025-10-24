#!/usr/bin/env python3
"""Test to verify progressive widening is causing the poor performance."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))
from utils.project_path import find_project_root

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from game.players.random_zertz_player import RandomZertzPlayer

def test_with_progressive_widening(enabled, iterations=1500, games=10, seed=20972):
    """Test MCTS win rate with progressive widening enabled/disabled."""
    np.random.seed(seed)

    wins = 0
    for i in range(games):
        game = ZertzGame(rings=37)

        # Player 1: Random
        player1 = RandomZertzPlayer(game, 1)

        # Player 2: MCTS with Rust backend
        player2 = MCTSZertzPlayer(
            game,
            n=2,
            iterations=iterations,
            backend='rust',
            verbose=False
        )

        # Manually override progressive widening
        player2.rust_mcts.set_progressive_widening(enabled)

        # Play game
        while game.get_game_ended() is None:
            current_player = game.get_current_player()
            if current_player == 0:
                action = player1.get_action()
            else:
                action = player2.get_action()
            game.execute_action(action)

        outcome = game.get_game_ended()
        if outcome == -1:  # Player 2 wins
            wins += 1

    return wins / games

if __name__ == "__main__":
    print("Testing progressive widening impact...")
    print()

    print("WITH progressive widening (default):")
    win_rate_with = test_with_progressive_widening(True, iterations=1500, games=10)
    print(f"  Win rate: {win_rate_with:.1%}")
    print()

    print("WITHOUT progressive widening:")
    win_rate_without = test_with_progressive_widening(False, iterations=1500, games=10)
    print(f"  Win rate: {win_rate_without:.1%}")
    print()

    print(f"Difference: {(win_rate_without - win_rate_with) * 100:+.1f} percentage points")