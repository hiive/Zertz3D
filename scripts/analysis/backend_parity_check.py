#!/usr/bin/env python3
"""
Backend parity harness for MCTS vs Random on Zèrtz.

Runs a 100-game series for both Python and Rust MCTS backends with 100
iterations per move against a random opponent, alternating sides to reduce
first-player bias. Results are printed so you can compare win rates.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

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


def play_series(
    backend: str,
    games: int,
    iterations: int,
    seed: int | None = None,
) -> Counter:
    """Play a series of games and collect win/loss statistics."""
    if seed is None:
        seed = 0

    stats: Counter = Counter()
    import time
    for game_idx in range(games):
        mcts_as_p1 = (game_idx % 2 == 0)

        game_seed = seed + game_idx if seed is not None else None
        if game_seed is not None:
            random.seed(game_seed)
            np.random.seed(game_seed)

        #Record start time
        start_time = time.perf_counter()

        result = play_single_game(
            backend=backend,
            iterations=iterations,
            mcts_as_player1=mcts_as_p1,
            rng_seed=game_seed,
        )

        # Record the end time
        end_time = time.perf_counter()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Print the result
        print(f"Game {game_idx + 1} took {elapsed_time:.6f} seconds.")

        stats["games_played"] += 1
        stats[f"result_{result}"] += 1  # 1, -1, or 0 (tie)

        if mcts_as_p1:
            role = "mcts_as_p1"
            outcome = result
        else:
            role = "mcts_as_p2"
            outcome = -result  # from MCTS perspective

        if outcome > 0:
            stats[f"{role}_wins"] += 1
        elif outcome < 0:
            stats[f"{role}_losses"] += 1
        else:
            stats[f"{role}_ties"] += 1

    stats["mcts_total_wins"] = stats["mcts_as_p1_wins"] + stats["mcts_as_p2_wins"]
    return stats


def play_single_game(
    backend: str,
    iterations: int,
    mcts_as_player1: bool,
    rng_seed: int | None,
) -> int:
    """Play a single game and return the terminal result (1, -1, or 0)."""
    game = ZertzGame(rings=37)

    if mcts_as_player1:
        player1 = make_mcts_player(game, player_n=1, backend=backend, iterations=iterations, rng_seed=rng_seed)
        player2 = RandomZertzPlayer(game, n=2)
    else:
        player1 = RandomZertzPlayer(game, n=1)
        player2 = make_mcts_player(game, player_n=2, backend=backend, iterations=iterations, rng_seed=rng_seed)

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
    rng_seed: int | None,
) -> MCTSZertzPlayer:
    """Create a configured MCTS player."""
    return MCTSZertzPlayer(
        game,
        n=player_n,
        iterations=iterations,
        backend=backend,
        parallel=False,
        use_transposition_table=True,
        use_transposition_lookups=True,
        clear_table_each_move=True,
        verbose=False,
        rng_seed=rng_seed,
    )


def format_stats(name: str, stats: Counter) -> str:
    """Format results into a readable block."""
    total = stats["games_played"]
    wins = stats["mcts_total_wins"]
    losses = stats["mcts_as_p1_losses"] + stats["mcts_as_p2_losses"]
    ties = total - wins - losses

    def pct(value: int) -> float:
        return (value / total * 100.0) if total else 0.0

    lines = [
        f"{name}:",
        f"  Games: {total}",
        f"  MCTS wins:   {wins} ({pct(wins):.1f}%)",
        f"  MCTS losses: {losses} ({pct(losses):.1f}%)",
        f"  Ties:        {ties} ({pct(ties):.1f}%)",
        f"  As P1: {stats['mcts_as_p1_wins']}W/"
        f"{stats['mcts_as_p1_losses']}L/{stats.get('mcts_as_p1_ties', 0)}T",
        f"  As P2: {stats['mcts_as_p2_wins']}W/"
        f"{stats['mcts_as_p2_losses']}L/{stats.get('mcts_as_p2_ties', 0)}T",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="Games per backend (default: 100)")
    parser.add_argument("--iterations", type=int, default=100, help="MCTS iterations per move (default: 100)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    args = parser.parse_args()

    python_stats = play_series("python", args.games, args.iterations, seed=args.seed)
    rust_stats = play_series("rust", args.games, args.iterations, seed=args.seed)

    print(())
    print(format_stats("Python backend", python_stats))
    print()
    print(format_stats("Rust backend", rust_stats))


if __name__ == "__main__":
    main()
