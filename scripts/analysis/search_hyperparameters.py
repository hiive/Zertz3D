#!/usr/bin/env python3
"""
MCTS Hyperparameter Tuning for Zèrtz.

Searches for optimal MCTS hyperparameters (exploration constant, FPU reduction, etc.)
by running games against a random opponent and measuring win rate.

Supports three modes via YAML configuration:
- Grid search: Exhaustively test all combinations of hyperparameter values
- Random search: Sample random configurations from specified distributions
- Repetition: Test specific configurations multiple times for statistical analysis

Usage:
    python tune_mcts_hyperparams_finetune.py <config_file.yaml>

Example:
    python tune_mcts_hyperparams_finetune.py data/tuning/default_grid_search.yaml

See data/tuning/example_config.yaml for YAML format documentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats


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
from game.players import MCTSZertzPlayer, RandomZertzPlayer


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
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "hyperparameters": self.hyperparams.to_dict(),
            "win_rate": float(self.win_rate),
            "games_played": int(self.games_played),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "ties": int(self.ties),
            "mean_time_per_game": float(self.mean_time_per_game),
            "timestamp": self.timestamp,
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
        mean_time_per_game=mean_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))


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




def get_git_version() -> str:
    """Get git version info with dirty flag.

    Returns:
        Git describe string (e.g., "v0.5.2-3-g1a2b3c4-dirty") or "unknown" if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'describe', '--always', '--dirty', '--tags'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_root
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass
    return "unknown"


def hash_source_files() -> tuple[str, str | None]:
    """Hash critical source files that affect MCTS behavior.

    Hashes all Rust files in rust/src/ and key Python files to create a unique
    identifier for the code version, even with uncommitted changes.

    Returns:
        Tuple of (hash, last_modified_uncommitted):
        - hash: SHA256 hash of all source files (first 16 characters)
        - last_modified_uncommitted: ISO timestamp of most recently modified uncommitted file, or None
    """
    hasher = hashlib.sha256()

    # Collect files to hash
    files_to_hash = []

    # Add all Rust source files
    rust_src = project_root / "rust" / "src"
    if rust_src.exists():
        files_to_hash.extend(sorted(rust_src.glob("**/*.rs")))

    # Add key Python files that affect MCTS
    python_files = [
        project_root / "game" / "zertz_game.py",
        project_root / "game" / "zertz_logic.py",
        project_root / "game" / "zertz_board.py",
        project_root / "game" / "players" / "mcts_zertz_player.py",
        project_root / "learner" / "mcts" / "backend.py",
    ]

    for filepath in python_files:
        if filepath.exists():
            files_to_hash.append(filepath)

    # Get list of uncommitted files from git
    uncommitted_files = set()
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project_root
        )
        if result.returncode == 0:
            uncommitted_files = set(result.stdout.strip().split('\n'))
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Hash file contents in sorted order and track modification times
    latest_mtime = None
    for filepath in sorted(files_to_hash):
        try:
            with open(filepath, 'rb') as f:
                hasher.update(f.read())

            # Check if this file is uncommitted and update latest mtime
            relative_path = filepath.relative_to(project_root).as_posix()
            if relative_path in uncommitted_files:
                mtime = filepath.stat().st_mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime

        except (IOError, OSError):
            continue

    # Convert latest_mtime to ISO format timestamp
    last_modified_str = None
    if latest_mtime is not None:
        last_modified_str = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Return hash and last modified time
    return hasher.hexdigest()[:16], last_modified_str


def format_hyperparams(hp: MCTSHyperparams, indent: str = "    ") -> list[str]:
    """Format hyperparameters as list of display strings.

    Args:
        hp: Hyperparameters to format
        indent: Indentation prefix for each line

    Returns:
        List of formatted strings
    """
    lines = []
    lines.append(f"{indent}Exploration constant: {hp.exploration_constant:.3f}")

    fpu_str = f"{hp.fpu_reduction:.3f}" if hp.fpu_reduction is not None else "None"
    lines.append(f"{indent}FPU reduction: {fpu_str}")

    depth_str = str(hp.max_simulation_depth) if hp.max_simulation_depth is not None else "Full game"
    lines.append(f"{indent}Max depth: {depth_str}")

    pw_str = f"Yes (constant={hp.widening_constant})" if hp.widening_constant is not None else "No"
    lines.append(f"{indent}Progressive widening: {pw_str}")

    rave_str = f"Yes (constant={hp.rave_constant:.0f})" if hp.rave_constant is not None else "No"
    lines.append(f"{indent}RAVE: {rave_str}")

    return lines


def format_duration(total_seconds: int) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        total_seconds: Duration in seconds

    Returns:
        Formatted string like "2h 15m 30s"
    """
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_summary(
    results: list[TuningResult],
    top_n: int = 5,
    output_file: str | Path | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    git_version: str | None = None,
    code_hash: str | None = None,
    last_modified_uncommitted: str | None = None,
    skipped_time_seconds: float = 0.0
) -> None:
    """Print summary of top performing configurations and optionally write to file.

    Args:
        results: List of tuning results
        top_n: Number of top configurations to display
        output_file: Optional output file path (for the summary text file)
        start_time: Optional start time string
        end_time: Optional end time string
        git_version: Optional git version string
        code_hash: Optional code hash string
        last_modified_uncommitted: Optional timestamp of most recently modified uncommitted file
        skipped_time_seconds: Time in seconds from skipped runs (for resumed experiments)
    """
    # Sort by win rate
    sorted_results = sorted(results, key=lambda r: r.win_rate, reverse=True)

    # Build summary as list of lines
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"MCTS HYPERPARAMETER TUNING RESULTS")
    summary_lines.append("=" * 80)

    # Add timing metadata
    if start_time:
        summary_lines.append(f"Start time:  {start_time}")
    if end_time:
        summary_lines.append(f"End time:    {end_time}")
    if start_time and end_time:
        # Calculate duration
        try:
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            duration = end_dt - start_dt
            # Format current run duration nicely
            total_seconds = int(duration.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"
            summary_lines.append(f"Duration:    {duration_str}")

            # If we have skipped time from previous runs, show total time invested
            if skipped_time_seconds > 0:
                total_time_seconds = int(duration.total_seconds() + skipped_time_seconds)
                hours, remainder = divmod(total_time_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours > 0:
                    total_time_str = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    total_time_str = f"{minutes}m {seconds}s"
                else:
                    total_time_str = f"{seconds}s"
                summary_lines.append(f"Total time:  {total_time_str} (includes previous runs)")
        except ValueError:
            pass

    # Detect if this is repetition mode by checking for duplicate hyperparameters
    from collections import defaultdict
    config_groups = defaultdict(list)
    for result in results:
        # Create a hashable key from hyperparameters
        key = (
            result.hyperparams.exploration_constant,
            result.hyperparams.fpu_reduction,
            result.hyperparams.max_simulation_depth,
            result.hyperparams.widening_constant,
            result.hyperparams.rave_constant
        )
        config_groups[key].append(result)

    is_repetition_mode = any(len(group) > 1 for group in config_groups.values())

    summary_lines.append("")

    if is_repetition_mode:
        # Repetition mode: show statistics for each configuration
        summary_lines.append("=" * 80)
        summary_lines.append(f"REPETITION MODE RESULTS")
        summary_lines.append("=" * 80)
        summary_lines.append("")

        # Sort configurations by mean win rate
        sorted_configs = sorted(config_groups.items(),
                               key=lambda x: np.mean([r.win_rate for r in x[1]]),
                               reverse=True)

        for config_idx, (key, group_results) in enumerate(sorted_configs[:top_n], 1):
            hp = group_results[0].hyperparams  # All have same hyperparams
            win_rates = [r.win_rate for r in group_results]
            times = [r.mean_time_per_game for r in group_results]

            # Calculate statistics
            n = len(win_rates)
            mean_wr = np.mean(win_rates)
            median_wr = np.median(win_rates)
            std_wr = np.std(win_rates, ddof=1) if n > 1 else 0.0
            min_wr = np.min(win_rates)
            max_wr = np.max(win_rates)

            # Calculate 95% confidence interval for the mean
            if n > 1:
                sem = std_wr / np.sqrt(n)  # Standard error of the mean
                ci_95 = stats.t.interval(0.95, n - 1, loc=mean_wr, scale=sem)
            else:
                ci_95 = None

            mean_time = np.mean(times)

            summary_lines.append(f"Configuration #{config_idx}:")
            summary_lines.append(f"  Hyperparameters:")
            summary_lines.extend(format_hyperparams(hp))

            summary_lines.append(f"")
            summary_lines.append(f"  Statistics over {n} repetitions:")
            summary_lines.append(f"    Mean (SD):        {mean_wr:.1%} ({std_wr:.1%})")
            if n > 1:
                summary_lines.append(f"    95% CI:           [{ci_95[0]:.1%}, {ci_95[1]:.1%}]")
            summary_lines.append(f"    Median:           {median_wr:.1%}")
            summary_lines.append(f"    Range:            [{min_wr:.1%}, {max_wr:.1%}]")
            summary_lines.append(f"    Avg time/game:    {mean_time:.3f}s")
            summary_lines.append("")

    else:
        # Standard mode: show top N individual results
        summary_lines.append("=" * 80)
        summary_lines.append(f"TOP {top_n} CONFIGURATIONS (by win rate)")
        summary_lines.append("=" * 80)

        for i, result in enumerate(sorted_results[:top_n], 1):
            hp = result.hyperparams
            summary_lines.append(f"\n#{i}: Win rate: {result.win_rate:.1%} "
                               f"({result.wins}W/{result.losses}L/{result.ties}T)")
            summary_lines.extend(format_hyperparams(hp))

            summary_lines.append(f"    Avg time/game: {result.mean_time_per_game:.3f}s")

            if result.timestamp:
                summary_lines.append(f"    Timestamp: {result.timestamp}")

    # Add version/code hash info at the end
    if git_version or code_hash or last_modified_uncommitted:
        summary_lines.append("")
        summary_lines.append("=" * 80)
        summary_lines.append("CODE VERSION")
        summary_lines.append("=" * 80)
        if git_version:
            summary_lines.append(f"Git version: {git_version}")
        if code_hash:
            summary_lines.append(f"Code hash:   {code_hash}")
        if last_modified_uncommitted:
            summary_lines.append(f"Last modified (uncommitted): {last_modified_uncommitted}")

    # Print to console
    print("\n" + "\n".join(summary_lines))

    # Write to file if output_file is provided
    if output_file is not None:
        output_path = Path(output_file)
        summary_file = output_path.parent / "tuning_summary.txt"
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_lines))
            f.write("\n")

        print(f"\nSummary saved to: {summary_file}")


def calculate_wall_clock_time(results: list[TuningResult]) -> float:
    """Calculate wall-clock time from timestamps in results.

    Args:
        results: List of tuning results with timestamps

    Returns:
        Wall-clock time in seconds (0 if can't calculate)
    """
    if not results or len(results) < 2:
        return 0.0

    # Filter results with valid timestamps
    timestamped_results = [r for r in results if r.timestamp]
    if len(timestamped_results) < 2:
        return 0.0

    try:
        # Parse first and last timestamps
        first_ts = datetime.strptime(timestamped_results[0].timestamp, "%Y-%m-%d %H:%M:%S")
        last_ts = datetime.strptime(timestamped_results[-1].timestamp, "%Y-%m-%d %H:%M:%S")

        return (last_ts - first_ts).total_seconds()
    except (ValueError, AttributeError):
        return 0.0


def load_existing_results(output_file: str | Path) -> list[TuningResult]:
    """Load existing results from JSON file if it exists."""
    output_path = Path(output_file)

    if not output_path.exists():
        return []

    try:
        with open(output_path, 'r') as f:
            data = json.load(f)

        results = []
        for r in data.get("results", []):
            hp_dict = r["hyperparameters"]
            hyperparams = MCTSHyperparams(
                exploration_constant=hp_dict["exploration_constant"],
                fpu_reduction=hp_dict["fpu_reduction"],
                max_simulation_depth=hp_dict["max_simulation_depth"],
                widening_constant=hp_dict["widening_constant"],
                rave_constant=hp_dict["rave_constant"]
            )

            result = TuningResult(
                hyperparams=hyperparams,
                win_rate=r["win_rate"],
                games_played=r["games_played"],
                wins=r["wins"],
                losses=r["losses"],
                ties=r["ties"],
                mean_time_per_game=r["mean_time_per_game"],
                timestamp=r.get("timestamp", "")  # Backward compatible - default to empty string
            )
            results.append(result)

        return results
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results from {output_path}: {e}")
        return []


def hyperparams_already_tested(
    hyperparams: MCTSHyperparams,
    existing_results: list[TuningResult],
    tolerance: float = 1e-6
) -> bool:
    """Check if a hyperparameter configuration has already been tested.

    Args:
        hyperparams: Hyperparameters to check
        existing_results: List of existing results
        tolerance: Tolerance for floating point comparison

    Returns:
        True if configuration already exists, False otherwise
    """
    for result in existing_results:
        existing_hp = result.hyperparams

        # Compare all hyperparameters with tolerance for floats
        if abs(existing_hp.exploration_constant - hyperparams.exploration_constant) > tolerance:
            continue

        # Handle None values for optional parameters
        if existing_hp.fpu_reduction is None and hyperparams.fpu_reduction is None:
            fpu_match = True
        elif existing_hp.fpu_reduction is not None and hyperparams.fpu_reduction is not None:
            fpu_match = abs(existing_hp.fpu_reduction - hyperparams.fpu_reduction) <= tolerance
        else:
            fpu_match = False

        if not fpu_match:
            continue

        if existing_hp.max_simulation_depth != hyperparams.max_simulation_depth:
            continue

        if existing_hp.widening_constant is None and hyperparams.widening_constant is None:
            widening_match = True
        elif existing_hp.widening_constant is not None and hyperparams.widening_constant is not None:
            widening_match = abs(existing_hp.widening_constant - hyperparams.widening_constant) <= tolerance
        else:
            widening_match = False

        if not widening_match:
            continue

        if existing_hp.rave_constant is None and hyperparams.rave_constant is None:
            rave_match = True
        elif existing_hp.rave_constant is not None and hyperparams.rave_constant is not None:
            rave_match = abs(existing_hp.rave_constant - hyperparams.rave_constant) <= tolerance
        else:
            rave_match = False

        if not rave_match:
            continue

        # All parameters match!
        return True

    return False


def save_results(results: list[TuningResult], output_file: str | Path, announce: bool = True) -> None:
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "results": [r.to_dict() for r in results],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    if announce:
        print(f"\nResults saved to: {output_path}")


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing validated configuration
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if 'mode' not in config:
        raise ValueError("Configuration must specify 'mode' (grid, random, or repetition)")

    if config['mode'] not in ['grid', 'random', 'repetition']:
        raise ValueError(f"Invalid mode: {config['mode']}. Must be 'grid', 'random', or 'repetition'")

    # Set defaults for optional fields
    if 'game' not in config:
        config['game'] = {}

    game_defaults = {
        'rings': 37,
        'iterations': 1500,
        'games_per_config': 100,
        'seed': 42,
        'num_processes': 10,
    }

    for key, default_value in game_defaults.items():
        if key not in config['game']:
            config['game'][key] = default_value

    # Handle output configuration
    if 'output' not in config:
        config['output'] = {}

    if 'directory' not in config['output'] or config['output']['directory'] is None:
        config['output']['directory'] = config_file.parent
    else:
        config['output']['directory'] = Path(config['output']['directory'])

    if 'filename' not in config['output']:
        config['output']['filename'] = 'tuning_results.json'

    # Set defaults for display
    if 'display' not in config:
        config['display'] = {}

    if 'top_n' not in config['display']:
        config['display']['top_n'] = 10

    if 'verbose' not in config['display']:
        config['display']['verbose'] = True

    return config


def generate_hyperparameter_values(
    param_spec: dict[str, Any],
    mode: str,
    rng: np.random.RandomState | None = None
) -> list[Any]:
    """Generate hyperparameter values from specification.

    Args:
        param_spec: Parameter specification from YAML
        mode: Search mode ('grid' or 'random')
        rng: Random number generator (required for random mode)

    Returns:
        List of values for this parameter
    """
    param_type = param_spec.get('type', 'fixed')

    if param_type == 'fixed':
        return [param_spec['value']]

    elif param_type == 'values':
        return param_spec['values']

    elif param_type == 'range':
        # Generate range with start, stop, step
        min_val = param_spec['min']
        max_val = param_spec['max']
        step = param_spec.get('step', 1)

        # Use numpy arange and round to avoid floating point issues
        values = np.arange(min_val, max_val + step / 2, step)
        return [float(v) for v in values]

    elif param_type == 'uniform' and mode == 'random':
        # For random mode: return specification for later sampling
        if rng is None:
            raise ValueError("RNG required for random mode")
        return [('uniform', param_spec['min'], param_spec['max'])]

    elif param_type == 'choice' and mode == 'random':
        # For random mode: return specification for later sampling
        return [('choice', param_spec['values'])]

    elif param_type == 'mixed' and mode == 'random':
        # Mixed: null with probability, or sample from values/range
        return [('mixed', param_spec)]

    else:
        raise ValueError(f"Invalid parameter type: {param_type} for mode {mode}")


def yaml_grid_search(
    config: dict[str, Any],
    config_path: Path
) -> list[TuningResult]:
    """Perform grid search using YAML configuration.

    Args:
        config: Parsed YAML configuration
        config_path: Path to configuration file (for relative paths)

    Returns:
        List of tuning results
    """
    game_config = config['game']
    hyperparam_specs = config['hyperparameters']

    # Generate value lists for each hyperparameter
    param_values = {}
    for param_name, param_spec in hyperparam_specs.items():
        param_values[param_name] = generate_hyperparameter_values(param_spec, 'grid')

    # Ensure all required parameters are present
    required_params = ['exploration_constant', 'fpu_reduction', 'widening_constant',
                      'rave_constant', 'max_simulation_depth']

    for param in required_params:
        if param not in param_values:
            raise ValueError(f"Missing required hyperparameter: {param}")

    # Construct output path
    output_path = config['output']['directory'] / config['output']['filename']

    # Load existing results
    existing_results = load_existing_results(output_path)
    results = list(existing_results)

    verbose = config['display']['verbose']

    if existing_results and verbose:
        print(f"\nLoaded {len(existing_results)} existing results from {output_path}")
        print("Will skip already-tested configurations\n")

    # Calculate total configurations
    total_configs = 1
    for values in param_values.values():
        total_configs *= len(values)

    if verbose:
        print(f"\nYAML Grid Search: Testing {total_configs} configurations")
        print(f"Each config: {game_config['games_per_config']} games, "
              f"{game_config['iterations']} iterations/move\n")

    config_num = 0
    skipped_count = 0

    # Nested loops for grid search
    for exploration in param_values['exploration_constant']:
        for fpu in param_values['fpu_reduction']:
            for depth in param_values['max_simulation_depth']:
                for widening in param_values['widening_constant']:
                    for rave in param_values['rave_constant']:
                        config_num += 1

                        hyperparams = MCTSHyperparams(
                            exploration_constant=exploration,
                            fpu_reduction=fpu,
                            max_simulation_depth=depth,
                            widening_constant=widening,
                            rave_constant=rave
                        )

                        # Check if already tested
                        if hyperparams_already_tested(hyperparams, existing_results):
                            skipped_count += 1
                            # Find existing win rate for display
                            existing_win_rate = None
                            for result in existing_results:
                                if hyperparams_already_tested(hyperparams, [result]):
                                    existing_win_rate = result.win_rate
                                    break

                            if verbose:
                                win_rate_str = f", win_rate={existing_win_rate:.1%}" if existing_win_rate is not None else ""
                                print(
                                    f"[{config_num}/{total_configs}] SKIPPED{win_rate_str}: "
                                    f"exploration={exploration:.2f}, fpu={fpu}, "
                                    f"widening={widening}, rave={rave}",
                                    flush=True
                                )
                            continue

                        if verbose:
                            print(
                                f"[{config_num}/{total_configs}] Testing: "
                                f"exploration={exploration:.2f}, fpu={fpu}, "
                                f"widening={widening}, rave={rave}",
                                flush=True
                            )

                        result = evaluate_hyperparams(
                            hyperparams=hyperparams,
                            games=game_config['games_per_config'],
                            iterations=game_config['iterations'],
                            rings=game_config['rings'],
                            seed=game_config['seed'] + config_num * 1000,
                            verbose=verbose,
                            num_processes=game_config['num_processes']
                        )

                        results.append(result)
                        save_results(results, output_path, announce=False)

                        if verbose:
                            print(
                                f"  → Win rate: {result.win_rate:.1%} "
                                f"({result.wins}W/{result.losses}L/{result.ties}T), "
                                f"avg time: {result.mean_time_per_game:.3f}s\n",
                                flush=True
                            )

    if verbose and skipped_count > 0:
        print(f"\n{'='*80}")
        print(f"Skipped {skipped_count}/{total_configs} already-tested configurations")
        print(f"Tested {total_configs - skipped_count} new configurations")
        print(f"{'='*80}\n")

    # Calculate wall-clock time from existing results for skipped configurations
    skipped_time = calculate_wall_clock_time(existing_results) if existing_results else 0.0
    config['skipped_time_seconds'] = skipped_time

    return results


def yaml_random_search(
    config: dict[str, Any],
    config_path: Path
) -> list[TuningResult]:
    """Perform random search using YAML configuration.

    Args:
        config: Parsed YAML configuration
        config_path: Path to configuration file

    Returns:
        List of tuning results
    """
    game_config = config['game']
    hyperparam_specs = config['hyperparameters']

    # Get number of samples
    num_samples = config.get('random_samples', 20)

    rng = np.random.RandomState(game_config['seed'])

    output_path = config['output']['directory'] / config['output']['filename']
    results = []
    verbose = config['display']['verbose']

    if verbose:
        print(f"\nYAML Random Search: Testing {num_samples} random configurations")
        print(f"Each config: {game_config['games_per_config']} games, "
              f"{game_config['iterations']} iterations/move\n")

    for sample_num in range(num_samples):
        # Sample each hyperparameter
        sampled_params = {}

        for param_name, param_spec in hyperparam_specs.items():
            param_type = param_spec.get('type', 'fixed')

            if param_type == 'fixed':
                sampled_params[param_name] = param_spec['value']

            elif param_type == 'values' or param_type == 'choice':
                values = param_spec['values']
                sampled_params[param_name] = rng.choice(values)

            elif param_type == 'uniform':
                min_val = param_spec['min']
                max_val = param_spec['max']
                sampled_params[param_name] = rng.uniform(min_val, max_val)

            elif param_type == 'mixed':
                null_prob = param_spec.get('null_probability', 0.5)
                if rng.rand() < null_prob:
                    sampled_params[param_name] = None
                else:
                    if 'values' in param_spec:
                        sampled_params[param_name] = float(rng.choice(param_spec['values']))
                    elif 'min' in param_spec and 'max' in param_spec:
                        sampled_params[param_name] = rng.uniform(param_spec['min'], param_spec['max'])
                    else:
                        raise ValueError(f"Mixed parameter {param_name} requires 'values' or 'min'/'max'")

            else:
                raise ValueError(f"Invalid parameter type for random search: {param_type}")

        hyperparams = MCTSHyperparams(
            exploration_constant=sampled_params['exploration_constant'],
            fpu_reduction=sampled_params.get('fpu_reduction'),
            max_simulation_depth=sampled_params.get('max_simulation_depth'),
            widening_constant=sampled_params.get('widening_constant'),
            rave_constant=sampled_params.get('rave_constant')
        )

        if verbose:
            print(
                f"[{sample_num + 1}/{num_samples}] Testing: "
                f"exploration={hyperparams.exploration_constant:.2f}, "
                f"fpu={hyperparams.fpu_reduction}, "
                f"widening={hyperparams.widening_constant}, "
                f"rave={hyperparams.rave_constant}",
                flush=True
            )

        result = evaluate_hyperparams(
            hyperparams=hyperparams,
            games=game_config['games_per_config'],
            iterations=game_config['iterations'],
            rings=game_config['rings'],
            seed=game_config['seed'] + sample_num * 1000,
            verbose=verbose,
            num_processes=game_config['num_processes']
        )

        results.append(result)
        save_results(results, output_path, announce=False)

        if verbose:
            print(
                f"  → Win rate: {result.win_rate:.1%} "
                f"({result.wins}W/{result.losses}L/{result.ties}T), "
                f"avg time: {result.mean_time_per_game:.3f}s\n",
                flush=True
            )

    return results


def repetition_mode(
    config: dict[str, Any],
    config_path: Path
) -> list[TuningResult]:
    """Test specific configurations multiple times for statistical analysis.

    Args:
        config: Parsed YAML configuration
        config_path: Path to configuration file

    Returns:
        List of tuning results (one per repetition)
    """
    game_config = config['game']
    repetition_config = config.get('repetition', {})

    if 'configurations' not in repetition_config:
        raise ValueError("Repetition mode requires 'configurations' list in YAML")

    num_repetitions = repetition_config.get('num_repetitions', 30)
    configurations = repetition_config['configurations']

    output_path = config['output']['directory'] / config['output']['filename']
    verbose = config['display']['verbose']
    overwrite = config.get('overwrite', False)

    # Load existing results unless overwrite is True
    if overwrite:
        results = []
        if verbose:
            print("\n--overwrite flag set: Starting fresh, ignoring existing results\n")
    else:
        existing_results = load_existing_results(output_path)
        results = list(existing_results)
        if existing_results and verbose:
            print(f"\nLoaded {len(existing_results)} existing results from {output_path}")
            print("Will skip already-completed repetitions\n")

    total_runs = len(configurations) * num_repetitions

    if verbose:
        print(f"\nRepetition Mode: Testing {len(configurations)} configurations")
        print(f"Each config repeated {num_repetitions} times ({total_runs} total runs)")
        print(f"Games per run: {game_config['games_per_config']}, "
              f"Iterations: {game_config['iterations']}\n")

    run_num = 0
    skipped_count = 0

    for config_idx, hp_dict in enumerate(configurations, 1):
        hyperparams = MCTSHyperparams(
            exploration_constant=hp_dict['exploration_constant'],
            fpu_reduction=hp_dict.get('fpu_reduction'),
            max_simulation_depth=hp_dict.get('max_simulation_depth'),
            widening_constant=hp_dict.get('widening_constant'),
            rave_constant=hp_dict.get('rave_constant')
        )

        if verbose:
            print(f"\n{'='*80}")
            print(f"Configuration {config_idx}/{len(configurations)}:")
            print(f"  exploration={hyperparams.exploration_constant:.2f}, "
                  f"fpu={hyperparams.fpu_reduction}, "
                  f"widening={hyperparams.widening_constant}, "
                  f"rave={hyperparams.rave_constant}")
            print(f"{'='*80}\n")

        # Count existing repetitions for this configuration
        existing_reps = sum(1 for r in results if hyperparams_already_tested(hyperparams, [r]))
        config_results = [r for r in results if hyperparams_already_tested(hyperparams, [r])]

        if existing_reps > 0 and verbose:
            print(f"Found {existing_reps} existing repetitions, need {num_repetitions - existing_reps} more")

        for rep in range(num_repetitions):
            run_num += 1

            # Skip if we already have this repetition
            if rep < existing_reps:
                skipped_count += 1
                if verbose:
                    print(f"[Run {run_num}/{total_runs}] SKIPPED: Repetition {rep + 1}/{num_repetitions} already exists", flush=True)
                continue

            if verbose:
                print(f"[Run {run_num}/{total_runs}] Repetition {rep + 1}/{num_repetitions}", flush=True)

            result = evaluate_hyperparams(
                hyperparams=hyperparams,
                games=game_config['games_per_config'],
                iterations=game_config['iterations'],
                rings=game_config['rings'],
                seed=game_config['seed'] + run_num * 1000,
                verbose=False,  # Don't show per-game details
                num_processes=game_config['num_processes']
            )

            config_results.append(result)
            results.append(result)

            if verbose:
                print(
                    f"  → Win rate: {result.win_rate:.1%} "
                    f"({result.wins}W/{result.losses}L/{result.ties}T), "
                    f"time: {result.mean_time_per_game:.3f}s"
                )

        # Calculate statistics for this configuration
        win_rates = [r.win_rate for r in config_results]
        mean_win_rate = np.mean(win_rates)
        std_win_rate = np.std(win_rates)

        if verbose:
            print(f"\nStatistics for configuration {config_idx}:")
            print(f"  Mean win rate: {mean_win_rate:.1%} ± {std_win_rate:.1%}")
            print(f"  Min: {min(win_rates):.1%}, Max: {max(win_rates):.1%}")

        # Save intermediate results
        save_results(results, output_path, announce=False)

    if verbose:
        print(f"\n{'='*80}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count}/{total_runs} existing repetitions")
            print(f"Completed {total_runs - skipped_count} new repetitions")
        else:
            print("All repetitions complete")
        print(f"{'='*80}\n")

    # Calculate wall-clock time from existing results
    if not overwrite and 'existing_results' in locals():
        skipped_time = calculate_wall_clock_time(existing_results)
    else:
        skipped_time = 0.0
    config['skipped_time_seconds'] = skipped_time

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "config",
        type=str,
        help="YAML configuration file (required). See data/tuning/example_config.yaml for format."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results instead of skipping them (default: skip existing)"
    )

    args = parser.parse_args()

    # Load and process YAML configuration
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    # Add overwrite flag to config
    config['overwrite'] = args.overwrite

    mode = config['mode']

    # Capture version information before starting
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_version = get_git_version()
    code_hash, last_modified_uncommitted = hash_source_files()

    # Run appropriate search mode
    if mode == 'grid':
        results = yaml_grid_search(config, config_path)
    elif mode == 'random':
        results = yaml_random_search(config, config_path)
    elif mode == 'repetition':
        results = repetition_mode(config, config_path)
    else:
        raise ValueError(f"Invalid mode in config: {mode}")

    # Capture end time
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get output path from config
    output_path = config['output']['directory'] / config['output']['filename']
    top_n = config['display']['top_n']

    # Print summary with metadata
    print_summary(
        results,
        top_n=top_n,
        output_file=output_path,
        start_time=start_time,
        end_time=end_time,
        git_version=git_version,
        code_hash=code_hash,
        last_modified_uncommitted=last_modified_uncommitted,
        skipped_time_seconds=config.get('skipped_time_seconds', 0.0)
    )

    # Results are already saved incrementally during search
    print(f"\nAll results saved to: {output_path}")


if __name__ == "__main__":
    main()
