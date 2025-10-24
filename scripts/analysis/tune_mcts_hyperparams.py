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
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

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
            "win_rate": self.win_rate,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "mean_time_per_game": self.mean_time_per_game,
        }


def evaluate_hyperparams(
    hyperparams: MCTSHyperparams,
    games: int,
    iterations: int,
    backend: str,
    rings: int = 37,
    seed: int | None = None,
    verbose: bool = False,
) -> TuningResult:
    """Evaluate a hyperparameter configuration by playing games vs random."""
    if seed is None:
        seed = 0

    wins = 0
    losses = 0
    ties = 0
    total_time = 0.0

    for game_idx in range(games):
        # Alternate MCTS between player 1 and 2 to reduce first-player bias
        mcts_as_p1 = (game_idx % 2 == 0)

        game_seed = seed + game_idx
        if game_seed is not None:
            random.seed(game_seed)
            np.random.seed(game_seed)

        start_time = time.perf_counter()

        result = play_single_game(
            backend=backend,
            iterations=iterations,
            hyperparams=hyperparams,
            mcts_as_player1=mcts_as_p1,
            rings=rings,
            rng_seed=game_seed,
        )

        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        # Convert result to MCTS perspective
        if not mcts_as_p1:
            result = -result

        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            ties += 1

        if verbose and (game_idx + 1) % 10 == 0:
            print(f"  Progress: {game_idx + 1}/{games} games, "
                  f"current win rate: {wins / (game_idx + 1):.1%}")

    win_rate = wins / games if games > 0 else 0.0
    mean_time = total_time / games if games > 0 else 0.0

    return TuningResult(
        hyperparams=hyperparams,
        win_rate=win_rate,
        games_played=games,
        wins=wins,
        losses=losses,
        ties=ties,
        mean_time_per_game=mean_time,
    )


def play_single_game(
    backend: str,
    iterations: int,
    hyperparams: MCTSHyperparams,
    mcts_as_player1: bool,
    rings: int,
    rng_seed: int | None,
) -> int:
    """Play a single game and return the terminal result (1, -1, or 0)."""
    game = ZertzGame(rings=rings)

    if mcts_as_player1:
        player1 = make_mcts_player(
            game, player_n=1, backend=backend, iterations=iterations,
            hyperparams=hyperparams, rng_seed=rng_seed
        )
        player2 = RandomZertzPlayer(game, n=2)
    else:
        player1 = RandomZertzPlayer(game, n=1)
        player2 = make_mcts_player(
            game, player_n=2, backend=backend, iterations=iterations,
            hyperparams=hyperparams, rng_seed=rng_seed
        )

    players = [player1, player2]

    while game.get_game_ended() is None:
        current = players[game.get_cur_player_value() - 1]
        action_type, action_data = current.get_action()
        if action_type == "PASS":
            game.take_action("PASS", None)
        else:
            game.take_action(action_type, action_data)

    return game.get_game_ended()


def make_mcts_player(
    game: ZertzGame,
    player_n: int,
    backend: str,
    iterations: int,
    hyperparams: MCTSHyperparams,
    rng_seed: int | None,
) -> MCTSZertzPlayer:
    """Create a configured MCTS player with given hyperparameters."""
    return MCTSZertzPlayer(
        game,
        n=player_n,
        iterations=iterations,
        exploration_constant=hyperparams.exploration_constant,
        max_simulation_depth=hyperparams.max_simulation_depth,
        fpu_reduction=hyperparams.fpu_reduction,
        widening_constant=hyperparams.widening_constant,
        backend=backend,
        parallel=False,
        use_transposition_table=True,
        use_transposition_lookups=True,
        clear_table_each_move=True,
        verbose=False,
        rng_seed=rng_seed,
    )


def grid_search(
    iterations: int,
    games_per_config: int,
    backend: str,
    rings: int,
    seed: int,
    verbose: bool = True,
) -> list[TuningResult]:
    """
    Perform grid search over hyperparameter space.

    Searches:
    - exploration_constant: [0.5, 1.0, 1.41 (√2), 2.0, 3.0]
    - fpu_reduction: [None, 0.1, 0.2, 0.3, 0.5]
    - max_simulation_depth: [None (full game)]
    - progressive_widening: [False] (disabled by default after bug fix)
    """
    exploration_values = [0.5, 1.0, 1.41, 2.0, 3.0]
    fpu_values = [None, 0.1, 0.2, 0.3, 0.5]
    depth_values = [None]  # Start with full-depth only
    widening_values = [None]  # Progressive widening disabled by default

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
                    )

                if verbose:
                    print(f"[{config_num}/{total_configs}] Testing: "
                          f"exploration={exploration:.2f}, "
                          f"fpu={fpu if fpu else 'None'}, "
                          f"depth={depth if depth else 'Full'}")

                result = evaluate_hyperparams(
                    hyperparams=hyperparams,
                    games=games_per_config,
                    iterations=iterations,
                    backend=backend,
                    rings=rings,
                    seed=seed + config_num * 1000,
                    verbose=verbose,
                )

                results.append(result)

                if verbose:
                    print(f"  → Win rate: {result.win_rate:.1%} "
                          f"({result.wins}W/{result.losses}L/{result.ties}T), "
                          f"avg time: {result.mean_time_per_game:.3f}s\n")

    return results


def random_search(
    iterations: int,
    games_per_config: int,
    backend: str,
    rings: int,
    seed: int,
    num_samples: int = 20,
    verbose: bool = True,
) -> list[TuningResult]:
    """
    Perform random search over hyperparameter space.

    Samples random configurations from:
    - exploration_constant: uniform[0.7, 2.5] (focused around √2 ≈ 1.41)
    - fpu_reduction: 40% None, 60% uniform[0.05, 0.5]
    - max_simulation_depth: None (full game)
    - widening_constant: 40% None, 60% uniform[4.0, 20.0]
    """
    rng = np.random.RandomState(seed)
    results = []

    if verbose:
        print(f"\nRandom Search: Testing {num_samples} random configurations")
        print(f"Each config: {games_per_config} games, {iterations} iterations/move\n")

    for sample_num in range(num_samples):
        # Sample hyperparameters with refined ranges
        # Exploration: focus on reasonable values around √2 ≈ 1.41
        exploration = rng.uniform(0.7, 2.5)

        # FPU reduction: 40% None, 60% in reasonable range
        use_fpu = rng.rand() < 0.6
        fpu = rng.uniform(0.05, 0.5) if use_fpu else None

        # Max depth: full game (could explore limited depth in future)
        depth = None

        # Progressive widening: 40% None, 60% with refined range
        # Lower values (3-8) = aggressive widening, higher (8-20) = conservative
        use_widening = rng.rand() < 0.6
        widening = rng.uniform(4.0, 20.0) if use_widening else None

        hyperparams = MCTSHyperparams(
            exploration_constant=exploration,
            fpu_reduction=fpu,
            max_simulation_depth=depth,
            widening_constant=widening,
        )

        if verbose:
            fpu_str = f"{fpu:.2f}" if fpu is not None else "None"
            depth_str = str(depth) if depth is not None else "Full"
            print(f"[{sample_num + 1}/{num_samples}] Testing: "
                  f"exploration={exploration:.2f}, "
                  f"fpu={fpu_str}, "
                  f"depth={depth_str}")

        result = evaluate_hyperparams(
            hyperparams=hyperparams,
            games=games_per_config,
            iterations=iterations,
            backend=backend,
            rings=rings,
            seed=seed + sample_num * 1000,
            verbose=verbose,
        )

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
        json.dump(data, f, indent=2)

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
        default=20,
        help="Games per configuration (default: 20)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1500,
        help="MCTS iterations per move (default: 1500)"
    )
    parser.add_argument(
        "--backend",
        choices=["python", "rust"],
        default="rust",
        help="MCTS backend to use (default: rust)"
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
        default="data/tuning_results.json",
        help="Output file for results (default: data/tuning_results.json)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top configurations to display (default: 5)"
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

    args = parser.parse_args()

    verbose = not args.quiet

    # Run search
    if args.method == "grid":
        results = grid_search(
            iterations=args.iterations,
            games_per_config=args.games,
            backend=args.backend,
            rings=args.rings,
            seed=args.seed,
            verbose=verbose,
        )
    else:  # random
        results = random_search(
            iterations=args.iterations,
            games_per_config=args.games,
            backend=args.backend,
            rings=args.rings,
            seed=args.seed,
            num_samples=args.samples,
            verbose=verbose,
        )

    # Print summary
    print_summary(results, top_n=args.top)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()