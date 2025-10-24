"""Train MCTS player by accumulating knowledge across thousands of games.

Builds a persistent transposition table that serves as a knowledge base
of explored positions and their values.

Usage:
    # Start fresh training
    python train_mcts.py --games 1000 --iterations 100

    # Resume from checkpoint
    python train_mcts.py --games 1000 --iterations 100 --resume

    # Custom save path
    python train_mcts.py --games 1000 --save-path data/mcts_37rings_10k.pkl
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))
from utils.project_path import find_project_root

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from learner.mcts.transposition_table import TranspositionTable


def train_mcts(
    num_games=1000,
    iterations_per_move=100,
    rings=37,
    save_path="data/mcts_knowledge.npz",
    save_interval=10,
    resume=False
):
    """Run self-play games to build MCTS knowledge base.

    Args:
        num_games: Number of self-play games to run
        iterations_per_move: MCTS iterations per move
        rings: Board size (37, 48, or 61)
        save_path: Path to save transposition table
        save_interval: Save every N games
        resume: Load existing table if available
    """
    # Create or load transposition table
    table = TranspositionTable()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if resume and save_path.exists():
        print(f"Loading existing knowledge base from {save_path}...")
        if table.load(save_path):
            print(f"  Loaded {table.size():,} positions with {table.total_visits():,} total visits")
        else:
            print("  Failed to load, starting fresh")
    else:
        print("Starting fresh knowledge base")

    # Statistics
    total_moves = 0
    total_time = 0
    wins = {1: 0, -1: 0, 0: 0}  # P1 wins, P2 wins, ties
    start_time = time.time()

    print("\nTraining Configuration:")
    print(f"  Games: {num_games}")
    print(f"  Iterations per move: {iterations_per_move}")
    print(f"  Board size: {rings} rings")
    print(f"  Save path: {save_path}")
    print(f"  Save interval: every {save_interval} games")
    print()

    for game_num in range(1, num_games + 1):
        game_start = time.time()

        # Create fresh game
        game = ZertzGame(rings=rings)

        # Create players sharing the PERSISTENT table
        player1 = MCTSZertzPlayer(
            game, n=1,
            iterations=iterations_per_move,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table_each_move=False,  # CRITICAL: don't clear!
            verbose=False
        )
        player2 = MCTSZertzPlayer(
            game, n=2,
            iterations=iterations_per_move,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table_each_move=False,  # CRITICAL: don't clear!
            verbose=False
        )

        # Both players share the same table
        player1.transposition_table = table
        player2.transposition_table = table

        # Play game
        move_count = 0
        while game.get_game_ended() is None:
            current_player = player1 if game.board.get_cur_player() == 0 else player2
            action_type, action_data = current_player.get_action()

            if action_type == "PASS":
                game.take_action("PASS", None)
            else:
                game.take_action(action_type, action_data)

            move_count += 1

        # Record results
        outcome = game.get_game_ended()
        wins[outcome] += 1
        total_moves += move_count
        game_time = time.time() - game_start
        total_time += game_time

        # Print progress
        if game_num % save_interval == 0 or game_num == num_games:
            elapsed = time.time() - start_time
            avg_moves = total_moves / game_num
            avg_time = total_time / game_num
            positions = table.size()
            total_visits = table.total_visits()

            print(f"Game {game_num}/{num_games} ({game_num/num_games*100:.1f}%)")
            print(f"  Outcome: {outcome:+d} (P1={wins[1]}, P2={wins[-1]}, Tie={wins[0]})")
            print(f"  Moves: {move_count} (avg: {avg_moves:.1f})")
            print(f"  Time: {game_time:.2f}s (avg: {avg_time:.2f}s)")
            print(f"  Knowledge: {positions:,} positions, {total_visits:,} visits")
            print(f"  Rate: {game_num/elapsed:.2f} games/min")
            print()

            # Save checkpoint
            print(f"Saving checkpoint to {save_path}...")
            table.save(save_path)
            print("Saved!\n")

    # Final summary
    elapsed = time.time() - start_time
    print("="*60)
    print("Training Complete!")
    print(f"  Total games: {num_games}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Total moves: {total_moves:,} (avg: {total_moves/num_games:.1f}/game)")
    print(f"  Results: P1={wins[1]}, P2={wins[-1]}, Tie={wins[0]}")
    print(f"  Knowledge base: {table.size():,} positions")
    print(f"  Total visits: {table.total_visits():,}")
    print(f"  Saved to: {save_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCTS player via self-play")
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of self-play games (default: 1000)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="MCTS iterations per move (default: 100)")
    parser.add_argument("--rings", type=int, default=37, choices=[37, 48, 61],
                        help="Board size (default: 37)")
    parser.add_argument("--save-path", type=str, default="data/mcts_knowledge.npz",
                        help="Path to save knowledge base (default: data/mcts_knowledge.npz)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save every N games (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint if available")

    args = parser.parse_args()

    train_mcts(
        num_games=args.games,
        iterations_per_move=args.iterations,
        rings=args.rings,
        save_path=args.save_path,
        save_interval=args.save_interval,
        resume=args.resume
    )