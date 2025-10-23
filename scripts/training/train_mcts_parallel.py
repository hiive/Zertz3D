"""Parallel MCTS training with periodic table merging.

Each worker process maintains its own local transposition table.
Tables are merged periodically to share knowledge across workers.

This avoids lock contention while still sharing knowledge.
"""

import argparse
import sys
import time
import multiprocessing as mp
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

from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from learner.mcts.transposition_table import TranspositionTable
from learner.mcts.elo_tracker import EloTracker


def worker_process(
    worker_id,
    games_per_worker,
    iterations_per_move,
    rings,
    initial_table_path,
    result_queue
):
    """Worker process that runs self-play games.

    Args:
        worker_id: Unique worker identifier
        games_per_worker: Number of games this worker should play
        iterations_per_move: MCTS iterations per move
        rings: Board size
        initial_table_path: Path to initial knowledge base (if any)
        result_queue: Queue to send results back to main process
    """
    # Load or create local transposition table
    local_table = TranspositionTable()
    if initial_table_path and Path(initial_table_path).exists():
        local_table.load(initial_table_path)

    stats = {
        'worker_id': worker_id,
        'games_played': 0,
        'total_moves': 0,
        'wins': {1: 0, -1: 0, 0: 0},
    }

    for game_num in range(games_per_worker):
        # Create fresh game
        game = ZertzGame(rings=rings)

        # Create players sharing local table
        player1 = MCTSZertzPlayer(
            game, n=1,
            iterations=iterations_per_move,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table_each_move=False,
            verbose=False
        )
        player2 = MCTSZertzPlayer(
            game, n=2,
            iterations=iterations_per_move,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table_each_move=False,
            verbose=False
        )

        player1.transposition_table = local_table
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

    # Send results back
    result_queue.put({
        'stats': stats,
        'table': local_table.table,  # Send the raw dict
    })


def merge_tables(global_table, worker_tables):
    """Merge worker transposition tables into global table.

    Args:
        global_table: TranspositionTable to merge into
        worker_tables: List of table dicts from workers

    Returns:
        Number of positions added/updated
    """
    positions_updated = 0

    for worker_table in worker_tables:
        for key, worker_entry in worker_table.items():
            if key in global_table.table:
                # Merge: add visits and values
                global_table.table[key]['visits'] += worker_entry['visits']
                global_table.table[key]['value'] += worker_entry['value']
            else:
                # Add new position
                global_table.table[key] = worker_entry.copy()
            positions_updated += 1

    return positions_updated


def train_mcts_parallel(
    num_games=1000,
    iterations_per_move=100,
    num_workers=4,
    rings=37,
    save_path="data/mcts_knowledge.npz",
    elo_path="data/mcts_elo.json",
    resume=False
):
    """Run parallel self-play training with periodic table merging.

    Args:
        num_games: Total number of games to play across all workers
        iterations_per_move: MCTS iterations per move
        num_workers: Number of parallel worker processes
        rings: Board size (37, 48, or 61)
        save_path: Path to save transposition table
        elo_path: Path to save ELO ratings
        resume: Load existing table/ratings if available
    """
    save_path = Path(save_path)
    elo_path = Path(elo_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    elo_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or create global table and ELO tracker
    global_table = TranspositionTable()
    elo_tracker = EloTracker()

    if resume:
        if save_path.exists():
            print(f"Loading existing knowledge base from {save_path}...")
            if global_table.load(save_path):
                print(f"  Loaded {global_table.size():,} positions with {global_table.total_visits():,} total visits")

        if elo_path.exists():
            print(f"Loading ELO ratings from {elo_path}...")
            elo_tracker.load(elo_path)
            print(f"  Loaded {len(elo_tracker.ratings)} player ratings")

    print("\nParallel Training Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  Total games: {num_games}")
    print(f"  Games per worker: {num_games // num_workers}")
    print(f"  Iterations per move: {iterations_per_move}")
    print(f"  Board size: {rings} rings")
    print(f"  Save path: {save_path}")
    print(f"  ELO path: {elo_path}")
    print()

    start_time = time.time()

    # Spawn worker processes
    result_queue = mp.Queue()
    processes = []

    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                num_games // num_workers,
                iterations_per_move,
                rings,
                save_path if resume else None,
                result_queue
            )
        )
        p.start()
        processes.append(p)
        print(f"Started worker {worker_id}")

    # Wait for workers and collect results
    print(f"\nWaiting for {num_workers} workers to complete...\n")

    worker_results = []
    for _ in range(num_workers):
        result = result_queue.get()
        worker_results.append(result)
        stats = result['stats']
        print(f"Worker {stats['worker_id']} completed:")
        print(f"  Games: {stats['games_played']}")
        print(f"  Moves: {stats['total_moves']} (avg: {stats['total_moves']/stats['games_played']:.1f})")
        print(f"  Results: P1={stats['wins'][1]}, P2={stats['wins'][-1]}, Tie={stats['wins'][0]}")
        print()

    # Join processes
    for p in processes:
        p.join()

    # Merge tables
    print("Merging transposition tables...")
    worker_tables = [r['table'] for r in worker_results]
    positions_updated = merge_tables(global_table, worker_tables)
    print(f"  Merged {positions_updated:,} position updates")
    print(f"  Global table: {global_table.size():,} positions, {global_table.total_visits():,} visits")
    print()

    # Aggregate statistics
    total_stats = {
        'games_played': sum(r['stats']['games_played'] for r in worker_results),
        'total_moves': sum(r['stats']['total_moves'] for r in worker_results),
        'wins': {1: 0, -1: 0, 0: 0}
    }
    for r in worker_results:
        for outcome, count in r['stats']['wins'].items():
            total_stats['wins'][outcome] += count

    # Update ELO ratings (treating this as MCTS_iter{iterations_per_move} player)
    player_id = f"MCTS_iter{iterations_per_move}"
    # For self-play, we can't directly update ELO, but we track the configuration
    # ELO tracking would happen when playing against other configurations

    # Save results
    print("Saving results...")
    global_table.save(save_path)
    elo_tracker.save(elo_path)
    print(f"  Saved to {save_path}")
    print(f"  ELO saved to {elo_path}")
    print()

    # Final summary
    elapsed = time.time() - start_time
    print("="*60)
    print("Parallel Training Complete!")
    print(f"  Workers: {num_workers}")
    print(f"  Total games: {total_stats['games_played']}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Rate: {total_stats['games_played']/(elapsed/60):.1f} games/min")
    print(f"  Total moves: {total_stats['total_moves']:,} (avg: {total_stats['total_moves']/total_stats['games_played']:.1f}/game)")
    print(f"  Results: P1={total_stats['wins'][1]}, P2={total_stats['wins'][-1]}, Tie={total_stats['wins'][0]}")
    print(f"  Knowledge base: {global_table.size():,} positions")
    print(f"  Total visits: {global_table.total_visits():,}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel MCTS training")
    parser.add_argument("--games", type=int, default=1000,
                        help="Total number of games (default: 1000)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="MCTS iterations per move (default: 100)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--rings", type=int, default=37, choices=[37, 48, 61],
                        help="Board size (default: 37)")
    parser.add_argument("--save-path", type=str, default="data/mcts_knowledge.npz",
                        help="Path to save knowledge base")
    parser.add_argument("--elo-path", type=str, default="data/mcts_elo.json",
                        help="Path to save ELO ratings")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint")

    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    train_mcts_parallel(
        num_games=args.games,
        iterations_per_move=args.iterations,
        num_workers=args.workers,
        rings=args.rings,
        save_path=args.save_path,
        elo_path=args.elo_path,
        resume=args.resume
    )