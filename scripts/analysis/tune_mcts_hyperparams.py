#!/usr/bin/env python3
"""
MCTS Hyperparameter Tuning for Zèrtz.

Searches for optimal MCTS hyperparameters (exploration constant, FPU reduction, etc.)
by running games against a random opponent and measuring win rate.

Uses grid search or random search to explore the hyperparameter space.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add project root to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))
from utils.project_path import find_project_root

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from game.zertz_player import RandomZertzPlayer


@dataclass
class MCTSHyperparams:
    """MCTS hyperparameters to tune."""
    exploration_constant: float  # UCB1 exploration (default: 1.41, √2)
    fpu_reduction: float | None  # First Play Urgency reduction (default: None)
    max_simulation_depth: int | None  # Max rollout depth (default: None = full game)
    widening_constant: float | None  # Progressive widening constant (None = disabled, 10.0 = moderate)
    rave_constant: float | None  # RAVE constant (None = disabled, 300-3000 = enabled)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TuningResult:
    """Result from evaluating a hyperparameter configuration."""
    hyperparams: MCTSHyperparams
    win_rate: float
    games_played: int
    wins: int
    losses: int
    ties: int
    mean_time_per_game: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hyperparams": self.hyperparams.to_dict(),
            "win_rate": float(self.win_rate),
            "games_played": int(self.games_played),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "ties": int(self.ties),
            "mean_time_per_game": float(self.mean_time_per_game),
        }


def _play_evaluation_game(
    game_idx: int,
    hyperparams: MCTSHyperparams,
    iterations: int,
    rings: int,
    seed: int,
    verbose: bool = False) -> tuple[int, float, int]:
    """
    Worker function to play a single game for hyperparameter evaluation.

    Args:
        game_idx: Game index (used for alternating player positions and seeding)
        hyperparams: MCTS hyperparameters to evaluate
        iterations: Number of MCTS iterations per move
        rings: Board size (37, 48, or 61)
        seed: Base RNG seed
        verbose: Whether to print diagnostic messages

    Returns:
        (result, elapsed_time, game_idx) tuple where result is from MCTS perspective
    """
    import os
    process_id = os.getpid()

    # Alternate MCTS between player 1 and 2 to reduce first-player bias
    mcts_as_p1 = (game_idx % 2 == 0)

    game_seed = seed + game_idx
    if game_seed is not None:
        random.seed(game_seed)
        np.random.seed(game_seed)

    if verbose and False:
        print(f"  [PID {process_id}] Game {game_idx} STARTING (MCTS as P{'1' if mcts_as_p1 else '2'}, seed={game_seed})")

    start_time = time.perf_counter()

    result, move_count = play_single_game(
        iterations=iterations,
        hyperparams=hyperparams,
        mcts_as_player1=mcts_as_p1,
        rings=rings,
        rng_seed=game_seed)

    elapsed = time.perf_counter() - start_time

    # Convert result to MCTS perspective
    if not mcts_as_p1:
        result = -result

    if verbose and False :
        outcome_str = "WIN" if result > 0 else ("LOSS" if result < 0 else "TIE")
        print(f"  [PID {process_id}] Game {game_idx} FINISHED: {outcome_str} in {elapsed:.4f}s ({move_count} moves, {elapsed/move_count if move_count > 0 else 0:.4f}s/move)")

    return (result, elapsed, game_idx)


def evaluate_hyperparams(
    hyperparams: MCTSHyperparams,
    games: int,
    iterations: int,
    rings: int = 37,
    seed: int | None = None,
    verbose: bool = False,
    num_processes: int = 10) -> TuningResult:
    """
    Evaluate a hyperparameter configuration by playing games vs random.

    Games are run in parallel using multiprocessing for faster evaluation.

    Args:
        hyperparams: MCTS hyperparameters to evaluate
        games: Number of games to play
        iterations: Number of MCTS iterations per move
        rings: Board size (37, 48, or 61)
        seed: Base RNG seed for reproducibility
        verbose: Whether to print progress (currently disabled for multiprocessing)
        num_processes: Number of parallel processes to use (default: 10)

    Returns:
        TuningResult with win rate and statistics
    """
    if seed is None:
        seed = 0

    if verbose:
        print(f"  Starting {games} games across {num_processes} processes...")

    pool_start = time.perf_counter()

    # Create a partial function with fixed parameters
    worker_fn = partial(
        _play_evaluation_game,
        hyperparams=hyperparams,
        iterations=iterations,
        rings=rings,
        seed=seed,
        verbose=verbose)

    # Run games in parallel using multiprocessing
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(worker_fn, range(games))

    pool_elapsed = time.perf_counter() - pool_start
    if verbose:
        print(f"  All games completed in {pool_elapsed:.2f}s (wall time)")

    # Aggregate results
    wins = 0
    losses = 0
    ties = 0
    total_time = 0.0

    for result, elapsed, game_idx in results:
        total_time += elapsed
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            ties += 1

    win_rate = wins / games if games > 0 else 0.0
    mean_time = total_time / games if games > 0 else 0.0

    return TuningResult(
        hyperparams=hyperparams,
        win_rate=win_rate,
        games_played=games,
        wins=wins,
        losses=losses,
        ties=ties,
        mean_time_per_game=mean_time)


def play_single_game(
    iterations: int,
    hyperparams: MCTSHyperparams,
    mcts_as_player1: bool,
    rings: int,
    rng_seed: int | None) -> tuple[int, int]:
    """Play a single game and return the terminal result and move count.

    Returns:
        (result, move_count) where result is 1, -1, or 0, and move_count is the number of moves
    """
    game = ZertzGame(rings=rings)

    if mcts_as_player1:
        player1 = make_mcts_player(
            game, player_n=1, iterations=iterations,
            hyperparams=hyperparams, rng_seed=rng_seed
        )
        player2 = RandomZertzPlayer(game, n=2)
    else:
        player1 = RandomZertzPlayer(game, n=1)
        player2 = make_mcts_player(
            game, player_n=2, iterations=iterations,
            hyperparams=hyperparams, rng_seed=rng_seed
        )

    players = {1: player1, -1: player2}  # Map player value (1 or -1) to player object
    move_count = 0

    while game.get_game_ended() is None:
        player_value = game.get_cur_player_value()  # Returns 1 for P1, -1 for P2
        current = players[player_value]
        action_type, action_data = current.get_action()
        if action_type == "PASS":
            game.take_action("PASS", None)
        else:
            game.take_action(action_type, action_data)
        move_count += 1

    return game.get_game_ended(), move_count


def make_mcts_player(
    game: ZertzGame,
    player_n: int,
    iterations: int,
    hyperparams: MCTSHyperparams,
    rng_seed: int | None) -> MCTSZertzPlayer:
    """Create a configured MCTS player with given hyperparameters.

    """
    return MCTSZertzPlayer(
        game,
        n=player_n,
        iterations=iterations,
        exploration_constant=hyperparams.exploration_constant,
        max_simulation_depth=hyperparams.max_simulation_depth,
        fpu_reduction=hyperparams.fpu_reduction,
        widening_constant=hyperparams.widening_constant,
        rave_constant=hyperparams.rave_constant,
        num_workers=4,
        use_transposition_table=True,
        use_transposition_lookups=True,
        clear_table_each_move=True,
        verbose=False,
        rng_seed=rng_seed)


def grid_search(
    iterations: int,
    games_per_config: int,
    rings: int,
    seed: int,
    verbose: bool = True,
    num_processes: int = 10) -> list[TuningResult]:
    """
    Perform grid search over hyperparameter space.

    Searches:
    - exploration_constant: [0.5, 1.0, 1.41 (√2), 2.0, 3.0]
    - fpu_reduction: [None, 0.1, 0.2, 0.3, 0.5]
    - max_simulation_depth: [None (full game)]
    - progressive_widening: [False] (disabled by default after bug fix)
    """
    exploration_values = [0.2, 0.35, 0.5, 1.0, 1.41, 2.0, 3.0]
    fpu_values = [None, 0.1, 0.2, 0.3, 0.5]
    depth_values = [None]  # Start with full-depth only
    widening_values = [None, 8.0, 12.0, 16.0, 20.0]  # Progressive widening disabled by default

    results = []
    total_configs = len(exploration_values) * len(fpu_values) * len(depth_values) * len(widening_values)
    config_num = 0

    if verbose:
        print(f"\nGrid Search: Testing {total_configs} configurations")
        print(f"Each config: {games_per_config} games, {iterations} iterations/move\n")

    for exploration in exploration_values:
        for fpu in fpu_values:
            for depth in depth_values:
                for widening in widening_values:
                    config_num += 1
                    hyperparams = MCTSHyperparams(
                        exploration_constant=exploration,
                        fpu_reduction=fpu,
                        max_simulation_depth=depth,
                        widening_constant=widening,
                        rave_constant=None)  # Disabled for grid search by default

                    if verbose:
                        print(f"[{config_num}/{total_configs}] Testing: "
                              f"exploration={exploration:.2f}, "
                              f"fpu={fpu if fpu else 'None'}, "
                              f"depth={depth if depth else 'Full'}, "
                              f"widening={widening if widening else 'None'}, ")

                    result = evaluate_hyperparams(
                        hyperparams=hyperparams,
                        games=games_per_config,
                        iterations=iterations,
                        rings=rings,
                        seed=seed + config_num * 1000,
                        verbose=verbose,
                        num_processes=num_processes)

                    results.append(result)

                    if verbose:
                        print(f"  → Win rate: {result.win_rate:.1%} "
                              f"({result.wins}W/{result.losses}L/{result.ties}T), "
                              f"avg time: {result.mean_time_per_game:.3f}s\n")

    return results


def random_search(
    iterations: int,
    games_per_config: int,
    rings: int,
    seed: int,
    num_samples: int = 20,
    verbose: bool = True,
    num_processes: int = 10) -> list[TuningResult]:
    """
    Perform random search over hyperparameter space.

    BALANCED SEARCH - Based on 70% win rate empirical results:
    - exploration_constant: uniform[1.4, 2.4] (top performers: 1.778, 2.278, 1.836 cluster 1.5-2.3)
    - fpu_reduction: 50% None, 50% uniform[0.08, 0.14] (top: 0.120, 0.110, None, 0.092)
    - max_simulation_depth: ALWAYS None (full game depth required)
    - widening_constant: 65% enabled with uniform[8.0, 20.0] (top: 17.9, 16.1, 13.2, 8.2)
    - rave_constant: 30% enabled with uniform[300, 3000] (expected 15-25% boost)
    """
    rng = np.random.RandomState(seed)
    results = []

    if verbose:
        print(f"\nRandom Search: Testing {num_samples} random configurations")
        print(f"Each config: {games_per_config} games, {iterations} iterations/move\n")

    for sample_num in range(num_samples):
        # Sample hyperparameters - BALANCED search based on 70% win rate results
        # Top performers: exploration 1.5-2.3, FPU 0.08-0.13 or None, widening 8-20

        # Exploration: Slightly wider range [1.4, 2.4] to not miss edge cases
        # Core sweet spot is 1.5-2.3 where 70% win rate configs cluster
        exploration = rng.uniform(1.4, 2.4)

        # FPU reduction: 50/50 between None and low values [0.08, 0.14]
        # Top performers: 0.120, 0.110, None, 0.092 (never above 0.14)
        use_fpu = rng.rand() < 0.50
        fpu = rng.uniform(0.08, 0.14) if use_fpu else None

        # Max depth: Always full game - limited depth consistently underperforms
        depth = None

        # Progressive widening: 65% enabled, range [8.0, 20.0]
        # Top performers: 17.9, 16.1, 13.2, 8.2 (8-20 covers all good configs)
        use_widening = rng.rand() < 0.65
        widening = rng.uniform(8.0, 20.0) if use_widening else None

        # RAVE: 30% enabled, range [300, 3000] as recommended
        # Expected 15-25% improvement when enabled
        use_rave = rng.rand() < 0.30
        rave = rng.uniform(300, 3000) if use_rave else None

        hyperparams = MCTSHyperparams(
            exploration_constant=exploration,
            fpu_reduction=fpu,
            max_simulation_depth=depth,
            widening_constant=widening,
            rave_constant=rave)

        if verbose:
            fpu_str = f"{fpu:.2f}" if fpu is not None else "None"
            depth_str = str(depth) if depth is not None else "Full"
            widening_str = f"{widening:.1f}" if widening is not None else "None"
            rave_str = f"{rave:.0f}" if rave is not None else "None"
            print(f"[{sample_num + 1}/{num_samples}] Testing: "
                  f"exploration={exploration:.2f}, "
                  f"fpu={fpu_str}, "
                  f"widening={widening_str}, "
                  f"rave={rave_str}")

        result = evaluate_hyperparams(
            hyperparams=hyperparams,
            games=games_per_config,
            iterations=iterations,
            rings=rings,
            seed=seed + sample_num * 1000,
            verbose=verbose,
            num_processes=num_processes)

        results.append(result)

        if verbose:
            print(f"  → Win rate: {result.win_rate:.1%} "
                  f"({result.wins}W/{result.losses}L/{result.ties}T), "
                  f"avg time: {result.mean_time_per_game:.3f}s\n")

    return results


def print_summary(results: list[TuningResult], top_n: int = 5) -> None:
    """Print summary of top performing configurations."""
    # Sort by win rate
    sorted_results = sorted(results, key=lambda r: r.win_rate, reverse=True)

    print("\n" + "=" * 80)
    print(f"TOP {top_n} CONFIGURATIONS (by win rate)")
    print("=" * 80)

    for i, result in enumerate(sorted_results[:top_n], 1):
        hp = result.hyperparams
        print(f"\n#{i}: Win rate: {result.win_rate:.1%} "
              f"({result.wins}W/{result.losses}L/{result.ties}T)")
        print(f"    Exploration constant: {hp.exploration_constant:.3f}")

        fpu_str = f"{hp.fpu_reduction:.3f}" if hp.fpu_reduction is not None else "None"
        print(f"    FPU reduction: {fpu_str}")

        depth_str = str(hp.max_simulation_depth) if hp.max_simulation_depth is not None else "Full game"
        print(f"    Max depth: {depth_str}")

        pw_str = f"Yes (constant={hp.widening_constant})" if hp.widening_constant is not None else "No"
        print(f"    Progressive widening: {pw_str}")

        rave_str = f"Yes (constant={hp.rave_constant:.0f})" if hp.rave_constant is not None else "No"
        print(f"    RAVE: {rave_str}")

        print(f"    Avg time/game: {result.mean_time_per_game:.3f}s")


def save_results(results: list[TuningResult], output_file: str) -> None:
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "results": [r.to_dict() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["grid", "random"],
        default="random",
        help="Search method: grid or random (default: random)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Games per configuration (default: 100)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1500,
        help="MCTS iterations per move (default: 1500)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of random samples (only for random search, default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tuning/tuning_results.json",
        help="Output file for results (default: data/tuning_results.json)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top configurations to display (default: 10)"
    )
    parser.add_argument(
        "--rings",
        type=int,
        choices=[37, 48, 61],
        default=37,
        help="Board size: 37, 48, or 61 rings (default: 37)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=10,
        help="Number of parallel processes for game evaluation (default: 10)"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Run search
    if args.method == "grid":
        results = grid_search(
            iterations=args.iterations,
            games_per_config=args.games,
            rings=args.rings,
            seed=args.seed,
            verbose=verbose,
            num_processes=args.processes)
    else:  # random
        results = random_search(
            iterations=args.iterations,
            games_per_config=args.games,
            rings=args.rings,
            seed=args.seed,
            num_samples=args.samples,
            verbose=verbose,
            num_processes=args.processes)

    # Print summary
    print_summary(results, top_n=args.top)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()