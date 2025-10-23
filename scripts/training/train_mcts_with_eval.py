"""Train MCTS player with periodic evaluation tournaments.

Combines self-play or competitive training with periodic strength evaluation.

Usage:
    # MCTS vs MCTS self-play (default)
    python train_mcts_with_eval.py --games 1000 --iterations 100

    # MCTS vs Random
    python train_mcts_with_eval.py --games 1000 --player2 random

    # Random vs Random (baseline)
    python train_mcts_with_eval.py --games 1000 --player1 random --player2 random

    # Resume from checkpoint
    python train_mcts_with_eval.py --games 1000 --resume
"""

import sys
from pathlib import Path

# Add project root to Python path to support running from any directory
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

import argparse
import time
from pathlib import Path
import json

from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from game.zertz_player import RandomZertzPlayer
from learner.mcts.transposition_table import TranspositionTable
from learner.mcts.elo_tracker import EloTracker


def play_game(player1, player2, game):
    """Play a single game between two players.

    Args:
        player1: First player
        player2: Second player
        game: Fresh ZertzGame instance

    Returns:
        Game outcome: 1 (player 1 won), -1 (player 2 won), 0 (tie)
    """
    move_count = 0
    while game.get_game_ended() is None:
        current_player = player1 if game.board.get_cur_player() == 0 else player2
        action_type, action_data = current_player.get_action()

        if action_type == "PASS":
            game.take_action("PASS", None)
        else:
            game.take_action(action_type, action_data)

        move_count += 1

    return game.get_game_ended(), move_count


def run_evaluation_tournament(
    trained_table,
    iterations_per_move,
    rings,
    eval_games=20
):
    """Run evaluation tournament against baseline opponents.

    Args:
        trained_table: Current trained transposition table
        iterations_per_move: MCTS iterations for trained player
        rings: Board size
        eval_games: Games to play against each opponent

    Returns:
        dict with results against each opponent
    """
    results = {}

    # Opponent configurations
    opponents = [
        {
            'id': 'Random',
            'type': 'random'
        },
        {
            'id': f'MCTS_untrained_{iterations_per_move}',
            'type': 'mcts',
            'iterations': iterations_per_move,
            'trained': False
        }
    ]

    trained_id = f'MCTS_trained_{iterations_per_move}'

    for opponent in opponents:
        wins = {1: 0, -1: 0, 0: 0}
        total_moves = 0

        for game_num in range(eval_games):
            game = ZertzGame(rings=rings)

            # Create trained player
            trained_player = MCTSZertzPlayer(
                game, n=1,
                iterations=iterations_per_move,
                use_transposition_table=True,
                use_transposition_lookups=True,
                clear_table_each_move=False,
                verbose=False
            )
            trained_player.transposition_table = trained_table

            # Create opponent
            if opponent['type'] == 'random':
                opponent_player = RandomZertzPlayer(game, n=2)
            elif opponent['type'] == 'mcts':
                opponent_player = MCTSZertzPlayer(
                    game, n=2,
                    iterations=opponent['iterations'],
                    use_transposition_table=False,
                    verbose=False
                )

            # Alternate who plays first
            if game_num % 2 == 0:
                outcome, moves = play_game(trained_player, opponent_player, game)
            else:
                outcome, moves = play_game(opponent_player, trained_player, game)
                outcome = -outcome  # Flip outcome since trained player is player 2

            wins[outcome] += 1
            total_moves += moves

        # Calculate win rate
        win_rate = wins[1] / eval_games

        results[opponent['id']] = {
            'wins': wins[1],
            'losses': wins[-1],
            'ties': wins[0],
            'win_rate': win_rate,
            'avg_moves': total_moves / eval_games
        }

    return results


def train_mcts_with_eval(
    num_games=1000,
    iterations_per_move=100,
    rings=37,
    save_path="data/mcts_knowledge_eval.npz",
    eval_interval=100,
    eval_games=20,
    eval_history_path="data/mcts_eval_history.json",
    elo_path="data/mcts_eval_elo.json",
    resume=False,
    player1_type="mcts",
    player2_type="mcts"
):
    """Run training with periodic evaluation tournaments.

    Args:
        num_games: Number of training games
        iterations_per_move: MCTS iterations per move
        rings: Board size
        save_path: Path to save knowledge base
        eval_interval: Run evaluation every N games
        eval_games: Games per opponent in evaluation
        eval_history_path: Path to save evaluation history
        elo_path: Path to save ELO tracker
        resume: Load existing checkpoint if available
        player1_type: Player 1 type ('random' or 'mcts')
        player2_type: Player 2 type ('random' or 'mcts')
    """
    # Setup paths
    save_path = Path(save_path)
    eval_history_path = Path(eval_history_path)
    elo_path = Path(elo_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    eval_history_path.parent.mkdir(parents=True, exist_ok=True)

    # Create or load transposition table
    table = TranspositionTable()
    elo_tracker = EloTracker()
    eval_history = []

    if resume:
        if save_path.exists():
            print(f"Loading existing knowledge base from {save_path}...")
            if table.load(save_path):
                print(f"  Loaded {table.size():,} positions with {table.total_visits():,} total visits")

        if elo_path.exists():
            print(f"Loading ELO tracker from {elo_path}...")
            elo_tracker.load(elo_path)

        if eval_history_path.exists():
            print(f"Loading evaluation history from {eval_history_path}...")
            with open(eval_history_path, 'r') as f:
                eval_history = json.load(f)
            print(f"  Loaded {len(eval_history)} evaluation checkpoints")

    # Statistics
    total_moves = 0
    total_time = 0
    wins = {1: 0, -1: 0, 0: 0}
    start_time = time.time()

    print("\nTraining Configuration:")
    print(f"  Total games: {num_games}")
    print(f"  Player 1: {player1_type}")
    print(f"  Player 2: {player2_type}")
    print(f"  Iterations per move: {iterations_per_move}")
    print(f"  Board size: {rings} rings")
    print(f"  Evaluation interval: every {eval_interval} games")
    print(f"  Evaluation games: {eval_games} per opponent")
    print(f"  Save path: {save_path}")
    print(f"  Eval history: {eval_history_path}")
    print(f"  ELO path: {elo_path}")
    print()

    for game_num in range(1, num_games + 1):
        game_start = time.time()

        # Create fresh game
        game = ZertzGame(rings=rings)

        # Create player 1
        if player1_type == "random":
            player1 = RandomZertzPlayer(game, n=1)
        else:  # mcts
            player1 = MCTSZertzPlayer(
                game, n=1,
                iterations=iterations_per_move,
                use_transposition_table=True,
                use_transposition_lookups=True,
                clear_table_each_move=False,
                verbose=False
            )
            player1.transposition_table = table

        # Create player 2
        if player2_type == "random":
            player2 = RandomZertzPlayer(game, n=2)
        else:  # mcts
            player2 = MCTSZertzPlayer(
                game, n=2,
                iterations=iterations_per_move,
                use_transposition_table=True,
                use_transposition_lookups=True,
                clear_table_each_move=False,
                verbose=False
            )
            player2.transposition_table = table

        # Play game
        outcome, move_count = play_game(player1, player2, game)

        # Record results
        wins[outcome] += 1
        total_moves += move_count
        game_time = time.time() - game_start
        total_time += game_time

        # Print frequent progress updates (every 10 games)
        if game_num % 10 == 0:
            elapsed = time.time() - start_time
            pct = game_num / num_games * 100
            rate = game_num / elapsed * 60  # games/hour
            print(f"[{game_num}/{num_games}] {pct:.1f}% | "
                  f"Last: {outcome:+d} ({move_count} moves) | "
                  f"Rate: {rate:.1f} games/hr | "
                  f"Knowledge: {table.size():,} pos")

        # Run evaluation tournament at intervals
        if game_num % eval_interval == 0 or game_num == num_games:
            elapsed = time.time() - start_time
            avg_moves = total_moves / game_num
            avg_time = total_time / game_num

            print(f"\n{'='*60}")
            print(f"Training Progress: {game_num}/{num_games} games ({game_num/num_games*100:.1f}%)")
            print(f"{'='*60}")
            print("Training stats:")
            print(f"  Outcomes: P1={wins[1]}, P2={wins[-1]}, Tie={wins[0]}")
            print(f"  Avg moves/game: {avg_moves:.1f}")
            print(f"  Avg time/game: {avg_time:.2f}s")
            print(f"  Knowledge: {table.size():,} positions, {table.total_visits():,} visits")
            print(f"  Training rate: {game_num/elapsed*60:.1f} games/hour")
            print()

            # Run evaluation tournament
            print(f"Running evaluation tournament ({eval_games} games per opponent)...")
            eval_start = time.time()
            eval_results = run_evaluation_tournament(
                table,
                iterations_per_move,
                rings,
                eval_games
            )
            eval_time = time.time() - eval_start

            # Record evaluation checkpoint
            checkpoint = {
                'training_games': game_num,
                'table_positions': table.size(),
                'table_visits': table.total_visits(),
                'timestamp': time.time(),
                'results': eval_results
            }
            eval_history.append(checkpoint)

            # Print evaluation results
            print(f"\nEvaluation Results (completed in {eval_time:.1f}s):")
            for opponent_id, stats in eval_results.items():
                win_rate = stats['win_rate'] * 100
                print(f"  vs {opponent_id}:")
                print(f"    W/L/T: {stats['wins']}/{stats['losses']}/{stats['ties']}")
                print(f"    Win rate: {win_rate:.1f}%")
                print(f"    Avg moves: {stats['avg_moves']:.1f}")
            print()

            # Save checkpoints
            print("Saving checkpoints...")
            table.save(save_path)
            elo_tracker.save(elo_path)
            with open(eval_history_path, 'w') as f:
                json.dump(eval_history, f, indent=2)
            print(f"  Knowledge base: {save_path}")
            print(f"  Evaluation history: {eval_history_path}")
            print(f"  ELO tracker: {elo_path}")
            print()

    # Final summary
    elapsed = time.time() - start_time
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total games: {num_games}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Total moves: {total_moves:,} (avg: {total_moves/num_games:.1f}/game)")
    print(f"Results: P1={wins[1]}, P2={wins[-1]}, Tie={wins[0]}")
    print(f"Final knowledge base: {table.size():,} positions, {table.total_visits():,} visits")
    print()

    # Show evaluation progress
    if len(eval_history) > 1:
        print("Evaluation Progress:")
        print(f"  Checkpoints: {len(eval_history)}")
        first = eval_history[0]
        last = eval_history[-1]

        for opponent_id in first['results'].keys():
            first_wr = first['results'][opponent_id]['win_rate'] * 100
            last_wr = last['results'][opponent_id]['win_rate'] * 100
            improvement = last_wr - first_wr
            print(f"  vs {opponent_id}:")
            print(f"    Initial: {first_wr:.1f}% → Final: {last_wr:.1f}% ({improvement:+.1f}%)")

    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MCTS with periodic evaluation")
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of training games (default: 1000)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="MCTS iterations per move (default: 100)")
    parser.add_argument("--rings", type=int, default=37, choices=[37, 48, 61],
                        help="Board size (default: 37)")
    parser.add_argument("--save-path", type=str, default="data/mcts_knowledge_eval.npz",
                        help="Path to save knowledge base")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Run evaluation every N games (default: 100)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games per opponent in evaluation (default: 20)")
    parser.add_argument("--eval-history", type=str, default="data/mcts_eval_history.json",
                        help="Path to save evaluation history")
    parser.add_argument("--elo-path", type=str, default="data/mcts_eval_elo.json",
                        help="Path to save ELO tracker")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--player1", type=str, default="mcts", choices=["random", "mcts"],
                        help="Player 1 type: random or mcts (default: mcts)")
    parser.add_argument("--player2", type=str, default="mcts", choices=["random", "mcts"],
                        help="Player 2 type: random or mcts (default: mcts)")

    args = parser.parse_args()

    train_mcts_with_eval(
        num_games=args.games,
        iterations_per_move=args.iterations,
        rings=args.rings,
        save_path=args.save_path,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_history_path=args.eval_history,
        elo_path=args.elo_path,
        resume=args.resume,
        player1_type=args.player1,
        player2_type=args.player2
    )