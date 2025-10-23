"""Parallel MCTS training with periodic evaluation tournaments.

Combines parallel training with periodic strength evaluation.
Workers run independently, tables are merged at evaluation checkpoints.

Usage:
    # MCTS vs MCTS self-play (default)
    python train_mcts_parallel_with_eval.py --games 1000 --workers 4

    # MCTS vs Random
    python train_mcts_parallel_with_eval.py --games 1000 --player2 random

    # Random vs Random (baseline)
    python train_mcts_parallel_with_eval.py --games 1000 --player1 random --player2 random

    # Resume from checkpoint
    python train_mcts_parallel_with_eval.py --games 1000 --resume
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
import multiprocessing as mp
from pathlib import Path
import json
import tempfile
import os

from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from game.zertz_player import RandomZertzPlayer
from learner.mcts.transposition_table import TranspositionTable
from learner.mcts.elo_tracker import EloTracker


def worker_process(
    worker_id,
    games_per_batch,
    iterations_per_move,
    rings,
    initial_table_path,
    result_queue,
    player1_type="mcts",
    player2_type="mcts"
):
    """Worker process that runs training games.

    Args:
        worker_id: Unique worker identifier
        games_per_batch: Number of games in this batch
        iterations_per_move: MCTS iterations per move
        rings: Board size
        initial_table_path: Path to initial knowledge base
        result_queue: Queue to send results back
        player1_type: Player 1 type ('random' or 'mcts')
        player2_type: Player 2 type ('random' or 'mcts')
    """
    # Load or create local table
    local_table = TranspositionTable()
    if initial_table_path and Path(initial_table_path).exists():
        local_table.load(initial_table_path)

    stats = {
        'worker_id': worker_id,
        'games_played': 0,
        'total_moves': 0,
        'wins': {1: 0, -1: 0, 0: 0},
    }

    for game_num in range(games_per_batch):
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
            player1.transposition_table = local_table

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
            player2.transposition_table = local_table

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
        stats['games_played'] += 1
        stats['total_moves'] += move_count
        stats['wins'][outcome] += 1

    # Save table to temp file instead of sending through queue
    # This avoids memory explosion from pickling large tables
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.npz', delete=False) as f:
        temp_table_path = f.name

    local_table.save(temp_table_path)

    # Send results back with path to table file
    result_queue.put({
        'stats': stats,
        'table_path': temp_table_path,
    })


def merge_tables(global_table, worker_tables):
    """Merge worker transposition tables into global table.

    Args:
        global_table: TranspositionTable to merge into
        worker_tables: List of table dicts from workers

    Returns:
        Number of positions updated
    """
    positions_updated = 0

    for worker_table in worker_tables:
        for zobrist_hash, chain in worker_table.items():
            if zobrist_hash in global_table.table:
                # Merge chains
                global_chain = global_table.table[zobrist_hash]

                for worker_entry in chain:
                    # Search for matching canonical state
                    found = False
                    for global_entry in global_chain:
                        import numpy as np
                        if np.array_equal(global_entry['canonical_state'],
                                         worker_entry['canonical_state']):
                            # Merge statistics
                            global_entry['visits'] += worker_entry['visits']
                            global_entry['value'] += worker_entry['value']
                            found = True
                            break

                    if not found:
                        # New state in this chain
                        global_chain.append(worker_entry.copy())

                positions_updated += len(chain)
            else:
                # New hash - copy entire chain
                global_table.table[zobrist_hash] = [entry.copy() for entry in chain]
                positions_updated += len(chain)

    return positions_updated


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
                # Trained is player 1
                move_count = 0
                while game.get_game_ended() is None:
                    current_player = trained_player if game.board.get_cur_player() == 0 else opponent_player
                    action_type, action_data = current_player.get_action()
                    if action_type == "PASS":
                        game.take_action("PASS", None)
                    else:
                        game.take_action(action_type, action_data)
                    move_count += 1
                outcome = game.get_game_ended()
            else:
                # Trained is player 2
                move_count = 0
                while game.get_game_ended() is None:
                    current_player = opponent_player if game.board.get_cur_player() == 0 else trained_player
                    action_type, action_data = current_player.get_action()
                    if action_type == "PASS":
                        game.take_action("PASS", None)
                    else:
                        game.take_action(action_type, action_data)
                    move_count += 1
                outcome = -game.get_game_ended()  # Flip since trained is player 2

            wins[outcome] += 1
            total_moves += move_count

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


def train_mcts_parallel_with_eval(
    num_games=1000,
    iterations_per_move=100,
    num_workers=4,
    rings=37,
    save_path="data/mcts_knowledge_parallel_eval.npz",
    eval_interval=200,
    eval_games=20,
    eval_history_path="data/mcts_parallel_eval_history.json",
    elo_path="data/mcts_parallel_eval_elo.json",
    resume=False,
    player1_type="mcts",
    player2_type="mcts"
):
    """Run parallel training with periodic evaluation.

    Args:
        num_games: Total number of training games
        iterations_per_move: MCTS iterations per move
        num_workers: Number of parallel worker processes
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

    # Load or create global table
    global_table = TranspositionTable()
    elo_tracker = EloTracker()
    eval_history = []

    if resume:
        if save_path.exists():
            print(f"Loading existing knowledge base from {save_path}...")
            if global_table.load(save_path):
                print(f"  Loaded {global_table.size():,} positions with {global_table.total_visits():,} total visits")

        if elo_path.exists():
            print(f"Loading ELO tracker from {elo_path}...")
            elo_tracker.load(elo_path)

        if eval_history_path.exists():
            print(f"Loading evaluation history from {eval_history_path}...")
            with open(eval_history_path, 'r') as f:
                eval_history = json.load(f)
            print(f"  Loaded {len(eval_history)} evaluation checkpoints")

    print("\nParallel Training Configuration:")
    print(f"  Workers: {num_workers}")
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

    start_time = time.time()
    games_completed = 0
    batch_num = 0

    # Track cumulative stats for evaluation
    cumulative_wins = {1: 0, -1: 0, 0: 0}
    cumulative_moves = 0
    games_since_eval = 0

    # Training loop with frequent mini-batches
    while games_completed < num_games:
        # Run small batches for frequent updates (num_workers games at a time)
        games_per_worker = 1  # Each worker does 1 game per mini-batch

        batch_num += 1
        batch_start = time.time()

        # Spawn worker processes
        result_queue = mp.Queue()
        processes = []

        for worker_id in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    worker_id,
                    games_per_worker,
                    iterations_per_move,
                    rings,
                    save_path if resume or games_completed > 0 else None,
                    result_queue,
                    player1_type,
                    player2_type
                )
            )
            p.start()
            processes.append(p)

        # Wait for workers
        worker_results = []
        for _ in range(num_workers):
            result = result_queue.get()
            worker_results.append(result)

        for p in processes:
            p.join()

        # Load and merge tables from temp files
        worker_tables = []
        temp_files = []
        for r in worker_results:
            temp_path = r['table_path']
            temp_files.append(temp_path)

            # Load table from file
            temp_table = TranspositionTable()
            temp_table.load(temp_path)
            worker_tables.append(temp_table.table)

        # Merge into global table
        positions_updated = merge_tables(global_table, worker_tables)

        # Clean up temp files
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except Exception:
                pass  # Ignore cleanup errors

        # Aggregate statistics
        batch_stats = {
            'games_played': sum(r['stats']['games_played'] for r in worker_results),
            'total_moves': sum(r['stats']['total_moves'] for r in worker_results),
            'wins': {1: 0, -1: 0, 0: 0}
        }
        for r in worker_results:
            for outcome, count in r['stats']['wins'].items():
                batch_stats['wins'][outcome] += count

        # Update cumulative stats
        for outcome, count in batch_stats['wins'].items():
            cumulative_wins[outcome] += count
        cumulative_moves += batch_stats['total_moves']
        games_completed += batch_stats['games_played']
        games_since_eval += batch_stats['games_played']
        batch_time = time.time() - batch_start

        # Print progress update after each batch (every num_workers games)
        elapsed = time.time() - start_time
        pct = games_completed / num_games * 100
        rate = games_completed / elapsed * 60  # games/min
        avg_moves = cumulative_moves / games_completed if games_completed > 0 else 0

        # Get last game outcome for this batch
        last_outcome = None
        for outcome in [1, -1, 0]:
            if batch_stats['wins'][outcome] > 0:
                last_outcome = outcome
                break

        print(f"[{games_completed}/{num_games}] {pct:.1f}% | "
              f"Last batch: {batch_stats['games_played']} games in {batch_time:.1f}s | "
              f"Rate: {rate:.1f} g/min | "
              f"Knowledge: {global_table.size():,} pos")

        # Run evaluation tournament at intervals
        if games_since_eval >= eval_interval or games_completed >= num_games:
            print(f"\n{'='*60}")
            print(f"Evaluation Checkpoint: {games_completed}/{num_games} games")
            print(f"{'='*60}")

            print(f"\nRunning evaluation tournament ({eval_games} games per opponent)...")
            eval_start = time.time()
            eval_results = run_evaluation_tournament(
                global_table,
                iterations_per_move,
                rings,
                eval_games
            )
            eval_time = time.time() - eval_start

            # Record checkpoint
            checkpoint = {
                'training_games': games_completed,
                'table_positions': global_table.size(),
                'table_visits': global_table.total_visits(),
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
            global_table.save(save_path)
            elo_tracker.save(elo_path)
            with open(eval_history_path, 'w') as f:
                json.dump(eval_history, f, indent=2)
            print("  Checkpoints saved")
            print()

            # Reset counter
            games_since_eval = 0

    # Final summary
    elapsed = time.time() - start_time
    print("="*60)
    print("Parallel Training Complete!")
    print("="*60)
    print(f"Workers: {num_workers}")
    print(f"Total games: {games_completed}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Rate: {games_completed/(elapsed/60):.1f} games/min")
    print(f"Knowledge base: {global_table.size():,} positions, {global_table.total_visits():,} visits")
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
    parser = argparse.ArgumentParser(description="Parallel MCTS training with evaluation")
    parser.add_argument("--games", type=int, default=1000,
                        help="Total number of games (default: 1000)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="MCTS iterations per move (default: 100)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--rings", type=int, default=37, choices=[37, 48, 61],
                        help="Board size (default: 37)")
    parser.add_argument("--save-path", type=str, default="data/mcts_knowledge_parallel_eval.npz",
                        help="Path to save knowledge base")
    parser.add_argument("--eval-interval", type=int, default=200,
                        help="Run evaluation every N games (default: 200)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games per opponent in evaluation (default: 20)")
    parser.add_argument("--eval-history", type=str, default="data/mcts_parallel_eval_history.json",
                        help="Path to save evaluation history")
    parser.add_argument("--elo-path", type=str, default="data/mcts_parallel_eval_elo.json",
                        help="Path to save ELO tracker")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")
    parser.add_argument("--player1", type=str, default="mcts", choices=["random", "mcts"],
                        help="Player 1 type: random or mcts (default: mcts)")
    parser.add_argument("--player2", type=str, default="mcts", choices=["random", "mcts"],
                        help="Player 2 type: random or mcts (default: mcts)")

    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    train_mcts_parallel_with_eval(
        num_games=args.games,
        iterations_per_move=args.iterations,
        num_workers=args.workers,
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