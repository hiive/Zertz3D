#!/usr/bin/env python3
"""Debug script to trace MCTS logic and identify the bug."""

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

from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard
from learner.mcts.mcts_tree import MCTSTree

def test_mcts_perspective():
    """Test MCTS evaluation perspective with a simple scenario."""

    # Create a game
    game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37, seed=12345)
    mcts = MCTSTree()

    print("=" * 70)
    print("MCTS PERSPECTIVE DEBUG")
    print("=" * 70)
    print()

    # Run a few iterations to see what happens
    print("Initial state:")
    print(f"  Current player: Player {game.get_current_player() + 1} (index={game.get_current_player()})")
    print()

    # Run search with verbose output
    action = mcts.search(
        game,
        iterations=100,
        verbose=True
    )

    print()
    print(f"Selected action: {action}")
    print()

    # Check root node statistics
    print("Root node analysis:")
    print(f"  Total visits: {mcts._last_root_visits}")
    print(f"  Total value: {mcts._last_root_value:.2f}")
    print(f"  Average value: {mcts._last_root_value / mcts._last_root_visits:.4f}")
    print()

    # A positive average value for the root means the current player (Player 0)
    # is expected to win. A negative value means they're expected to lose.
    avg_value = mcts._last_root_value / mcts._last_root_visits
    if avg_value > 0:
        print(f"  Interpretation: Player {game.get_current_player() + 1} is WINNING (avg={avg_value:.4f})")
    elif avg_value < 0:
        print(f"  Interpretation: Player {game.get_current_player() + 1} is LOSING (avg={avg_value:.4f})")
    else:
        print(f"  Interpretation: Position is EVEN (avg={avg_value:.4f})")

if __name__ == "__main__":
    test_mcts_perspective()