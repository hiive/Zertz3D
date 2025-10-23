"""View and visualize MCTS evaluation history.

Displays training progress and win rates over time from evaluation checkpoints.

Usage:
    # View text summary
    python view_eval_history.py data/mcts_eval_history.json

    # Generate plots (requires matplotlib)
    python view_eval_history.py data/mcts_eval_history.json --plot
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
import json
from pathlib import Path


def print_eval_summary(eval_history):
    """Print text summary of evaluation history.

    Args:
        eval_history: List of evaluation checkpoint dicts
    """
    if not eval_history:
        print("No evaluation history found.")
        return

    print("="*70)
    print("MCTS Training Evaluation History")
    print("="*70)
    print()

    # Overall summary
    first = eval_history[0]
    last = eval_history[-1]
    total_games = last['training_games']
    total_positions = last['table_positions']

    print("Training Summary:")
    print(f"  Total training games: {total_games:,}")
    print(f"  Evaluation checkpoints: {len(eval_history)}")
    print(f"  Final knowledge base: {total_positions:,} positions ({last['table_visits']:,} visits)")
    print()

    # Get all opponent IDs
    opponent_ids = list(first['results'].keys())

    print("Win Rate Progress:")
    print()

    for opponent_id in opponent_ids:
        print(f"  vs {opponent_id}:")
        print(f"  {'Games':<8} {'Win%':<8} {'W/L/T':<12} {'Positions':<12} {'Avg Moves'}")
        print(f"  {'-'*60}")

        for checkpoint in eval_history:
            games = checkpoint['training_games']
            positions = checkpoint['table_positions']
            result = checkpoint['results'][opponent_id]

            win_rate = result['win_rate'] * 100
            wlt = f"{result['wins']}/{result['losses']}/{result['ties']}"
            avg_moves = result['avg_moves']

            print(f"  {games:<8} {win_rate:<8.1f} {wlt:<12} {positions:<12,} {avg_moves:.1f}")

        print()

    # Show improvement
    print("Overall Improvement (first → last):")
    for opponent_id in opponent_ids:
        first_wr = first['results'][opponent_id]['win_rate'] * 100
        last_wr = last['results'][opponent_id]['win_rate'] * 100
        improvement = last_wr - first_wr

        first_wins = first['results'][opponent_id]['wins']
        last_wins = last['results'][opponent_id]['wins']

        print(f"  vs {opponent_id}:")
        print(f"    Win rate: {first_wr:.1f}% → {last_wr:.1f}% ({improvement:+.1f}%)")
        print(f"    Wins: {first_wins} → {last_wins} ({last_wins - first_wins:+d})")

    print()
    print("="*70)


def plot_eval_history(eval_history, output_path=None):
    """Generate plots of evaluation history.

    Args:
        eval_history: List of evaluation checkpoint dicts
        output_path: Optional path to save plot image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    if not eval_history:
        print("No evaluation history to plot.")
        return

    # Extract data
    training_games = [c['training_games'] for c in eval_history]
    positions = [c['table_positions'] for c in eval_history]
    opponent_ids = list(eval_history[0]['results'].keys())

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Win rates over training games
    for opponent_id in opponent_ids:
        win_rates = [c['results'][opponent_id]['win_rate'] * 100
                     for c in eval_history]
        ax1.plot(training_games, win_rates, marker='o', label=opponent_id)

    ax1.set_xlabel('Training Games')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('MCTS Training Progress: Win Rate vs Training Games')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (even)')

    # Plot 2: Knowledge base growth
    ax2.plot(training_games, positions, marker='o', color='green')
    ax2.set_xlabel('Training Games')
    ax2.set_ylabel('Positions in Knowledge Base')
    ax2.set_title('Knowledge Base Growth')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="View MCTS evaluation history")
    parser.add_argument("history_file", type=str,
                        help="Path to evaluation history JSON file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots (requires matplotlib)")
    parser.add_argument("--output", type=str,
                        help="Path to save plot image (implies --plot)")

    args = parser.parse_args()

    history_path = Path(args.history_file)

    if not history_path.exists():
        print(f"Error: File not found: {history_path}")
        return

    # Load history
    with open(history_path, 'r') as f:
        eval_history = json.load(f)

    # Print summary
    print_eval_summary(eval_history)

    # Generate plots if requested
    if args.plot or args.output:
        plot_eval_history(eval_history, args.output)


if __name__ == "__main__":
    main()