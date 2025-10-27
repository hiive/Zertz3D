"""Comprehensive MCTS performance benchmarking script.

Tests different MCTS configurations to identify performance characteristics
and optimization opportunities.
"""

import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))
from utils.project_path import find_project_root

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer


@dataclass
class MCTSConfig:
    """Configuration for MCTS testing."""
    name: str
    iterations: int = 1000
    exploration: float = 1.41
    fpu_reduction: Optional[float] = None
    widening_constant: Optional[float] = None
    rave_constant: Optional[float] = None
    use_transposition_table: bool = True
    use_transposition_lookups: bool = True
    parallel: bool = False
    num_workers: int = 16


@dataclass
class BenchmarkResult:
    """Results from benchmarking a configuration."""
    config_name: str
    total_moves: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    iterations_per_sec: float


def benchmark_config(config: MCTSConfig, num_moves: int = 20, seed: int = 42) -> BenchmarkResult:
    """Benchmark a single MCTS configuration.

    Args:
        config: MCTS configuration to test
        num_moves: Number of moves to benchmark
        seed: Random seed for reproducibility

    Returns:
        BenchmarkResult with timing statistics
    """
    # Create fresh game for each config
    game = ZertzGame(rings=37)

    # Create player with configuration
    player = MCTSZertzPlayer(
        game, n=1,
        iterations=config.iterations,
        exploration_constant=config.exploration,
        fpu_reduction=config.fpu_reduction,
        widening_constant=config.widening_constant,
        rave_constant=config.rave_constant,
        use_transposition_table=config.use_transposition_table,
        use_transposition_lookups=config.use_transposition_lookups,
        clear_table_each_move=True,  # Clear for consistent timing
        parallel=config.parallel,
        num_workers=config.num_workers,
        verbose=False,
        rng_seed=seed
    )

    # Collect timing data
    move_times = []
    moves_completed = 0

    # Run benchmark
    while moves_completed < num_moves and game.get_game_ended() is None:
        move_start = time.time()

        action_type, action_data = player.get_action()

        move_time = time.time() - move_start
        move_times.append(move_time)
        moves_completed += 1

        # Execute action
        game.take_action(action_type, action_data)

    # Calculate statistics
    total_time = sum(move_times)
    avg_time = total_time / len(move_times) if move_times else 0
    iterations_per_sec = (config.iterations * moves_completed) / total_time if total_time > 0 else 0

    return BenchmarkResult(
        config_name=config.name,
        total_moves=moves_completed,
        total_time=total_time,
        avg_time=avg_time,
        min_time=min(move_times) if move_times else 0,
        max_time=max(move_times) if move_times else 0,
        iterations_per_sec=iterations_per_sec
    )


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print()
    print("=" * 100)
    print("MCTS Performance Benchmark Results")
    print("=" * 100)
    print()

    # Print header
    header = f"{'Configuration':<30} {'Moves':>6} {'Total(s)':>9} {'Avg(s)':>8} {'Min(s)':>8} {'Max(s)':>8} {'Iter/s':>10}"
    print(header)
    print("-" * 100)

    # Print results sorted by average time (fastest first)
    sorted_results = sorted(results, key=lambda r: r.avg_time)

    for result in sorted_results:
        row = (
            f"{result.config_name:<30} "
            f"{result.total_moves:>6} "
            f"{result.total_time:>9.2f} "
            f"{result.avg_time:>8.3f} "
            f"{result.min_time:>8.3f} "
            f"{result.max_time:>8.3f} "
            f"{result.iterations_per_sec:>10.0f}"
        )
        print(row)

    print("-" * 100)
    print()

    # Print relative performance comparison
    if sorted_results:
        baseline = sorted_results[0]
        print("Relative Performance (compared to fastest):")
        print(f"{'Configuration':<30} {'Speedup':>10} {'Relative Time':>15}")
        print("-" * 60)

        for result in sorted_results:
            speedup = baseline.avg_time / result.avg_time if result.avg_time > 0 else 0
            relative = result.avg_time / baseline.avg_time if baseline.avg_time > 0 else 0
            print(f"{result.config_name:<30} {speedup:>10.2f}x {relative:>14.2f}x")

        print("-" * 60)

    print()
    print("=" * 100)


def main():
    """Run comprehensive MCTS benchmarks."""
    print("MCTS Performance Benchmarking Tool")
    print("Testing various MCTS configurations...")
    print()

    # Define configurations to test
    configs = [
        # Baseline configurations
        MCTSConfig(
            name="Baseline (1k iters)",
            iterations=1000,
            use_transposition_table=False,
        ),
        MCTSConfig(
            name="Baseline + Transposition",
            iterations=1000,
            use_transposition_table=True,
            use_transposition_lookups=True,
        ),

        # FPU variations
        MCTSConfig(
            name="FPU 0.2",
            iterations=1000,
            fpu_reduction=0.2,
            use_transposition_table=True,
        ),
        MCTSConfig(
            name="FPU 0.5",
            iterations=1000,
            fpu_reduction=0.5,
            use_transposition_table=True,
        ),

        # Progressive Widening variations
        MCTSConfig(
            name="Widening 10.0",
            iterations=1000,
            widening_constant=10.0,
            use_transposition_table=True,
        ),
        MCTSConfig(
            name="Widening 20.0",
            iterations=1000,
            widening_constant=20.0,
            use_transposition_table=True,
        ),

        # RAVE variations
        MCTSConfig(
            name="RAVE 1000",
            iterations=1000,
            rave_constant=1000.0,
            use_transposition_table=True,
        ),
        MCTSConfig(
            name="RAVE 3000",
            iterations=1000,
            rave_constant=3000.0,
            use_transposition_table=True,
        ),

        # Combined advanced features
        MCTSConfig(
            name="FPU + Widening",
            iterations=1000,
            fpu_reduction=0.2,
            widening_constant=10.0,
            use_transposition_table=True,
        ),
        MCTSConfig(
            name="FPU + RAVE",
            iterations=1000,
            fpu_reduction=0.2,
            rave_constant=1000.0,
            use_transposition_table=True,
        ),
        MCTSConfig(
            name="Widening + RAVE",
            iterations=1000,
            widening_constant=10.0,
            rave_constant=1000.0,
            use_transposition_table=True,
        ),
        MCTSConfig(
            name="All Features",
            iterations=1000,
            fpu_reduction=0.2,
            widening_constant=10.0,
            rave_constant=1000.0,
            use_transposition_table=True,
        ),

        # Parallel search (fewer iterations for reasonable timing)
        MCTSConfig(
            name="Parallel 4 threads",
            iterations=1000,
            use_transposition_table=True,
            parallel=True,
            num_workers=4,
        ),
        MCTSConfig(
            name="Parallel 8 threads",
            iterations=1000,
            use_transposition_table=True,
            parallel=True,
            num_workers=8,
        ),
        MCTSConfig(
            name="Parallel 16 threads",
            iterations=1000,
            use_transposition_table=True,
            parallel=True,
            num_workers=16,
        ),

        # Higher iteration counts for scaling analysis
        MCTSConfig(
            name="5k iters",
            iterations=5000,
            use_transposition_table=True,
        ),
        MCTSConfig(
            name="10k iters",
            iterations=10000,
            use_transposition_table=True,
        ),
        MCTSConfig(
            name="10k + RAVE",
            iterations=10000,
            rave_constant=1000.0,
            use_transposition_table=True,
        ),
    ]

    # Run benchmarks
    num_moves = 20  # Test each config for 20 moves
    seed = 42  # Fixed seed for reproducibility

    results = []
    total_configs = len(configs)

    for i, config in enumerate(configs, 1):
        print(f"[{i}/{total_configs}] Testing: {config.name}...", end=" ", flush=True)

        try:
            result = benchmark_config(config, num_moves=num_moves, seed=seed)
            results.append(result)
            print(f"✓ ({result.avg_time:.3f}s avg)")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Print results
    print_results(results)

    # Additional analysis
    print()
    print("Key Findings:")
    print("-" * 60)

    # Find fastest config
    if results:
        fastest = min(results, key=lambda r: r.avg_time)
        print(f"• Fastest configuration: {fastest.config_name}")
        print(f"  Average time: {fastest.avg_time:.3f}s")
        print(f"  Throughput: {fastest.iterations_per_sec:.0f} iterations/sec")
        print()

        # Compare baseline vs advanced features
        baseline = next((r for r in results if r.config_name == "Baseline (1k iters)"), None)
        transposition = next((r for r in results if r.config_name == "Baseline + Transposition"), None)
        rave = next((r for r in results if r.config_name == "RAVE 1000"), None)

        if baseline and transposition:
            improvement = (baseline.avg_time - transposition.avg_time) / baseline.avg_time * 100
            print(f"• Transposition table impact: {improvement:+.1f}%")

        if baseline and rave:
            improvement = (baseline.avg_time - rave.avg_time) / baseline.avg_time * 100
            print(f"• RAVE impact: {improvement:+.1f}%")

        # Parallel scaling analysis
        serial_1k = next((r for r in results if r.config_name == "Baseline + Transposition"), None)
        parallel_4 = next((r for r in results if r.config_name == "Parallel 4 threads"), None)
        parallel_8 = next((r for r in results if r.config_name == "Parallel 8 threads"), None)
        parallel_16 = next((r for r in results if r.config_name == "Parallel 16 threads"), None)

        if serial_1k:
            print()
            print("• Parallel scaling:")
            if parallel_4:
                speedup = serial_1k.avg_time / parallel_4.avg_time
                print(f"  4 threads: {speedup:.2f}x speedup")
            if parallel_8:
                speedup = serial_1k.avg_time / parallel_8.avg_time
                print(f"  8 threads: {speedup:.2f}x speedup")
            if parallel_16:
                speedup = serial_1k.avg_time / parallel_16.avg_time
                print(f"  16 threads: {speedup:.2f}x speedup")

    print("-" * 60)
    print()


if __name__ == "__main__":
    main()