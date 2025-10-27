#!/usr/bin/env python3
"""Quick test to run a single game with MCTS players."""

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
from learner.mcts.backend import HAS_RUST

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
    num_workers=1,   # Serial mode for simplicity
    verbose=True)

player2 = MCTSZertzPlayer(
    game,
    n=2,
    iterations=100,
    num_workers=1,
    verbose=True)

players = [player1, player2]

print("Player 1: MCTS (Rust backend)")
print("Player 2: MCTS (Rust backend)")
print()

# Play game
move_count = 0
max_moves = 1000  # Just test a few moves

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
else:
    print(f"Result: Unknown: {result}")

print(f"Total moves: {move_count}")