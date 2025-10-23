"""View and visualize ELO rating history.

Displays ELO progression over time from tournament or training records.

Usage:
    # View text summary
    python view_elo_history.py data/tournament_elo.json

    # Generate plots (requires matplotlib)
    python view_elo_history.py data/tournament_elo.json --plot

    # Save plot to file
    python view_elo_history.py data/tournament_elo.json --output elo_progress.png
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
from collections import defaultdict


def print_elo_summary(elo_data):
    """Print text summary of ELO history.

    Args:
        elo_data: Loaded ELO tracker data dict
    """
    history = elo_data.get('history', [])
    ratings = elo_data.get('ratings', {})

    if not history:
        print("No game history found.")
        return

    print("="*70)
    print("ELO Rating History Summary")
    print("="*70)
    print()

    # Overall summary
    print("Configuration:")
    print(f"  K-factor: {elo_data.get('k_factor', 32)}")
    print(f"  Initial rating: {elo_data.get('initial_rating', 1500)}")
    print(f"  Total games: {len(history)}")
    print()

    # Current leaderboard
    print("Final Ratings:")
    print(f"  {'Rank':<6} {'Player':<30} {'Rating':<10} {'Games'}")
    print(f"  {'-'*60}")

    # Count games per player
    games_per_player = defaultdict(int)
    for game in history:
        games_per_player[game['player_a']] += 1
        games_per_player[game['player_b']] += 1

    # Sort by rating
    leaderboard = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (player_id, rating) in enumerate(leaderboard, 1):
        games = games_per_player[player_id]
        print(f"  {rank:<6} {player_id:<30} {rating:<10.1f} {games}")

    print()

    # Rating changes summary
    print("Rating Changes (first game → last game):")
    print(f"  {'Player':<30} {'Start':<10} {'End':<10} {'Change':<10} {'Games'}")
    print(f"  {'-'*65}")

    # Track first and last rating for each player
    first_rating = {}
    last_rating = {}

    for game in history:
        player_a = game['player_a']
        player_b = game['player_b']

        if player_a not in first_rating:
            first_rating[player_a] = game['rating_a_before']
        if player_b not in first_rating:
            first_rating[player_b] = game['rating_b_before']

        last_rating[player_a] = game['rating_a_after']
        last_rating[player_b] = game['rating_b_after']

    # Sort by final rating
    for player_id, final_rating in sorted(last_rating.items(), key=lambda x: x[1], reverse=True):
        start = first_rating[player_id]
        change = final_rating - start
        games = games_per_player[player_id]
        print(f"  {player_id:<30} {start:<10.1f} {final_rating:<10.1f} {change:+10.1f} {games}")

    print()

    # Head-to-head records
    print("Head-to-Head Records:")
    matchups = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})

    for game in history:
        player_a = game['player_a']
        player_b = game['player_b']
        outcome = game['outcome']

        if outcome == 1.0:
            matchups[(player_a, player_b)]['wins'] += 1
            matchups[(player_b, player_a)]['losses'] += 1
        elif outcome == 0.0:
            matchups[(player_a, player_b)]['losses'] += 1
            matchups[(player_b, player_a)]['wins'] += 1
        else:  # 0.5 = draw
            matchups[(player_a, player_b)]['draws'] += 1
            matchups[(player_b, player_a)]['draws'] += 1

    # Print unique matchups
    printed = set()
    for (player_a, player_b), record in sorted(matchups.items()):
        if (player_b, player_a) in printed:
            continue
        printed.add((player_a, player_b))

        a_wins = record['wins']
        a_losses = record['losses']
        draws = record['draws']
        total = a_wins + a_losses + draws

        if total > 0:
            a_pct = (a_wins + 0.5*draws) / total * 100
            print(f"  {player_a} vs {player_b}:")
            print(f"    {a_wins}-{a_losses}-{draws} ({a_pct:.1f}% for {player_a})")

    print()
    print("="*70)


def plot_elo_history(elo_data, output_path=None):
    """Generate plots of ELO progression.

    Args:
        elo_data: Loaded ELO tracker data dict
        output_path: Optional path to save plot image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    history = elo_data.get('history', [])

    if not history:
        print("No game history to plot.")
        return

    # Reconstruct ELO progression for each player
    player_progression = defaultdict(lambda: {'games': [], 'ratings': []})

    for i, game in enumerate(history, 1):
        player_a = game['player_a']
        player_b = game['player_b']

        # Record game number and rating after this game
        player_progression[player_a]['games'].append(i)
        player_progression[player_a]['ratings'].append(game['rating_a_after'])

        player_progression[player_b]['games'].append(i)
        player_progression[player_b]['ratings'].append(game['rating_b_after'])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: ELO progression over games
    for player_id, data in sorted(player_progression.items()):
        ax1.plot(data['games'], data['ratings'], marker='', label=player_id, linewidth=2, alpha=0.8)

    ax1.set_xlabel('Game Number')
    ax1.set_ylabel('ELO Rating')
    ax1.set_title('ELO Rating Progression')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=elo_data.get('initial_rating', 1500), color='gray', linestyle='--', alpha=0.5, label='Initial rating')

    # Plot 2: Rating distribution (final ratings)
    final_ratings = elo_data.get('ratings', {})
    players = list(final_ratings.keys())
    ratings = [final_ratings[p] for p in players]

    colors = plt.cm.viridis([i/len(players) for i in range(len(players))])
    bars = ax2.barh(players, ratings, color=colors)

    ax2.set_xlabel('ELO Rating')
    ax2.set_ylabel('Player')
    ax2.set_title('Final ELO Ratings')
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.axvline(x=elo_data.get('initial_rating', 1500), color='gray', linestyle='--', alpha=0.5, label='Initial rating')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="View ELO rating history")
    parser.add_argument("elo_file", type=str,
                        help="Path to ELO tracker JSON file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots (requires matplotlib)")
    parser.add_argument("--output", type=str,
                        help="Path to save plot image (implies --plot)")

    args = parser.parse_args()

    elo_path = Path(args.elo_file)

    if not elo_path.exists():
        print(f"Error: File not found: {elo_path}")
        return

    # Load ELO data
    with open(elo_path, 'r') as f:
        elo_data = json.load(f)

    # Print summary
    print_elo_summary(elo_data)

    # Generate plots if requested
    if args.plot or args.output:
        plot_elo_history(elo_data, args.output)


if __name__ == "__main__":
    main()