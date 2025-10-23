#!/usr/bin/env python3
"""Quick test to run a single game with MCTS players."""

import sys
from pathlib import Path

# Add project root to Python path to support running from any directory
# Find project root by looking for pyproject.toml
def find_project_root(start_path: Path) -> Path:
    """Find project root by searching for pyproject.toml."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (pyproject.toml not found)")

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from learner.mcts.backend import get_backend_info

# Print backend info
backend_info = get_backend_info()
print("=" * 60)
print("BACKEND INFORMATION")
print("=" * 60)
print(f"Python available: {backend_info['python_available']}")
print(f"Rust available:   {backend_info['rust_available']}")
print(f"Current backend:  {backend_info['current']}")
print(f"Default backend:  {backend_info['default']}")
print()

# Set random seed for reproducibility
np.random.seed(42)

# Create game
print("=" * 60)
print("STARTING GAME")
print("=" * 60)
game = ZertzGame(rings=37)

# Create MCTS players with auto backend selection
player1 = MCTSZertzPlayer(
    game,
    n=1,
    iterations=100,  # Small number for quick test
    parallel=False,   # Serial mode for simplicity
    verbose=True,
    backend='auto'
)

player2 = MCTSZertzPlayer(
    game,
    n=2,
    iterations=100,
    parallel=False,
    verbose=True,
    backend='auto'
)

players = [player1, player2]

print(f"Player 1: MCTS (backend={player1.backend.value})")
print(f"Player 2: MCTS (backend={player2.backend.value})")
print()

# Play game
move_count = 0
max_moves = 10  # Just test a few moves

while game.get_game_ended() is None and move_count < max_moves:
    current_player = game.get_cur_player_value()
    player = players[current_player - 1]

    print(f"\n--- Move {move_count + 1} (Player {current_player}) ---")

    action_type, action_data = player.get_action()
    print(f"Action: {action_type} {action_data}")

    if action_type == "PASS":
        game.take_action("PASS", None)
    else:
        game.take_action(action_type, action_data)

    move_count += 1

# Game over
print("\n" + "=" * 60)
print("GAME OVER")
print("=" * 60)

result = game.get_game_ended()
if result == 1:
    print("Winner: Player 1")
elif result == -1:
    print("Winner: Player 2")
elif result == 0:
    print("Result: Tie")

print(f"Total moves: {move_count}")