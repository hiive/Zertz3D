"""Run ELO tournaments between different MCTS configurations.

Compare different:
- Iteration counts
- Knowledge bases
- Search parameters

Usage:
    # Default: Random, MCTS(50), MCTS(100), MCTS(200)
    python tournament.py --games 100

    # Add trained player to defaults
    python tournament.py --games 100 --knowledge-base data/mcts_knowledge.npz

    # Custom: Only Random vs MCTS(100)
    python tournament.py --games 100 --custom-players --add-random --add-mcts 100

    # Custom: Compare multiple MCTS configs
    python tournament.py --games 100 --custom-players --add-mcts 50 100 150 200

    # With ELO progression plot (display interactively)
    python tournament.py --games 100 --plot-elo

    # Save ELO progression plot to file
    python tournament.py --games 100 --plot-output data/tournament_elo.png
"""

import argparse
import time
from pathlib import Path

from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from game.zertz_player import RandomZertzPlayer
from learner.mcts.transposition_table import TranspositionTable
from learner.mcts.elo_tracker import EloTracker


def play_game(player1_config, player2_config, rings=37):
    """Play a single game between two player configurations.

    Args:
        player1_config: dict with player 1 configuration
        player2_config: dict with player 2 configuration
        rings: Board size

    Returns:
        Game outcome: 1 (player 1 won), -1 (player 2 won), 0 (tie)
    """
    game = ZertzGame(rings=rings)

    # Create player 1
    if player1_config['type'] == 'mcts':
        player1 = MCTSZertzPlayer(
            game, n=1,
            iterations=player1_config['iterations'],
            use_transposition_table=player1_config.get('use_table', False),
            clear_table_each_move=True,  # Fair comparison
            verbose=False
        )
        if player1_config.get('table'):
            player1.transposition_table = player1_config['table']
    elif player1_config['type'] == 'random':
        player1 = RandomZertzPlayer(game, n=1)

    # Create player 2
    if player2_config['type'] == 'mcts':
        player2 = MCTSZertzPlayer(
            game, n=2,
            iterations=player2_config['iterations'],
            use_transposition_table=player2_config.get('use_table', False),
            clear_table_each_move=True,  # Fair comparison
            verbose=False
        )
        if player2_config.get('table'):
            player2.transposition_table = player2_config['table']
    elif player2_config['type'] == 'random':
        player2 = RandomZertzPlayer(game, n=2)

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

    return game.get_game_ended()


def plot_elo_progression(elo_tracker, output_path=None):
    """Generate plot of ELO progression during tournament.

    Args:
        elo_tracker: EloTracker instance with game history
        output_path: Optional path to save plot image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    from collections import defaultdict

    history = elo_tracker.history

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
    plt.figure(figsize=(12, 6))

    # Plot ELO progression over games
    for player_id, data in sorted(player_progression.items()):
        plt.plot(data['games'], data['ratings'], marker='', label=player_id, linewidth=2, alpha=0.8)

    plt.xlabel('Game Number')
    plt.ylabel('ELO Rating')
    plt.title('ELO Rating Progression During Tournament')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=elo_tracker.initial_rating, color='gray', linestyle='--', alpha=0.5, label='Initial rating')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nELO progression plot saved to {output_path}")
    else:
        plt.show()


def run_tournament(
    player_configs,
    games_per_matchup=50,
    rings=37,
    elo_path="data/tournament_elo.json",
    plot_elo=False,
    plot_output=None
):
    """Run round-robin tournament between player configurations.

    Args:
        player_configs: List of player configuration dicts
        games_per_matchup: Games to play per matchup
        rings: Board size
        elo_path: Path to save ELO ratings
        plot_elo: Generate ELO progression plot
        plot_output: Optional path to save plot (implies plot_elo=True)

    Returns:
        EloTracker with final ratings
    """
    elo_tracker = EloTracker()
    elo_path = Path(elo_path)
    elo_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing ratings if available
    if elo_path.exists():
        elo_tracker.load(elo_path)

    print("\n" + "="*60)
    print("Tournament Configuration")
    print("="*60)
    print(f"Players: {len(player_configs)}")
    print(f"Games per matchup: {games_per_matchup}")
    print(f"Total games: {len(player_configs) * (len(player_configs) - 1) * games_per_matchup}")
    print(f"Board size: {rings} rings")
    print()

    for i, config in enumerate(player_configs):
        print(f"  Player {i+1}: {config['id']}")

    print("="*60 + "\n")

    # Round-robin tournament
    total_games = 0
    start_time = time.time()

    for i, config1 in enumerate(player_configs):
        for j, config2 in enumerate(player_configs):
            if i >= j:  # Skip self-play and duplicate matchups
                continue

            print(f"\nMatchup: {config1['id']} vs {config2['id']}")
            print(f"  Playing {games_per_matchup} games...")

            matchup_start = time.time()
            results = {1: 0, -1: 0, 0: 0}

            for game_num in range(games_per_matchup):
                # Alternate who plays as player 1 for fairness
                if game_num % 2 == 0:
                    outcome = play_game(config1, config2, rings)
                    # Record from config1's perspective
                    elo_tracker.record_game(config1['id'], config2['id'], outcome)
                    results[outcome] += 1
                else:
                    outcome = play_game(config2, config1, rings)
                    # Record from config2's perspective (flip outcome)
                    elo_tracker.record_game(config2['id'], config1['id'], outcome)
                    results[-outcome] += 1

                total_games += 1

                # Progress update
                if (game_num + 1) % 10 == 0:
                    print(f"    {game_num + 1}/{games_per_matchup} games complete")

            matchup_time = time.time() - matchup_start

            # Matchup summary
            print(f"\n  Results:")
            print(f"    {config1['id']} wins: {results[1]}")
            print(f"    {config2['id']} wins: {results[-1]}")
            print(f"    Ties: {results[0]}")
            print(f"  Time: {matchup_time:.1f}s ({games_per_matchup/matchup_time:.2f} games/s)")

            # Current ratings
            rating1 = elo_tracker.get_rating(config1['id'])
            rating2 = elo_tracker.get_rating(config2['id'])
            print(f"  Ratings: {config1['id']}={rating1:.1f}, {config2['id']}={rating2:.1f}")

            # Save checkpoint
            elo_tracker.save(elo_path)

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("Tournament Complete!")
    print("="*60)
    print(f"Total games: {total_games}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Rate: {total_games/elapsed:.2f} games/s")
    print()

    elo_tracker.print_summary()
    print(f"Results saved to {elo_path}")

    # Generate ELO progression plot if requested
    if plot_elo or plot_output:
        plot_elo_progression(elo_tracker, plot_output)

    return elo_tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ELO tournament")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per matchup (default: 50)")
    parser.add_argument("--rings", type=int, default=37, choices=[37, 48, 61],
                        help="Board size (default: 37)")
    parser.add_argument("--knowledge-base", type=str,
                        help="Path to trained knowledge base")
    parser.add_argument("--elo-path", type=str, default="data/tournament_elo.json",
                        help="Path to save ELO ratings")
    parser.add_argument("--custom-players", action="store_true",
                        help="Use custom player configs instead of defaults")
    parser.add_argument("--add-random", action="store_true",
                        help="Add random player to tournament")
    parser.add_argument("--add-mcts", type=int, nargs='+',
                        help="Add MCTS players with specified iteration counts (e.g., --add-mcts 50 100 200)")
    parser.add_argument("--plot-elo", action="store_true",
                        help="Generate ELO progression plot (requires matplotlib)")
    parser.add_argument("--plot-output", type=str,
                        help="Path to save ELO plot (implies --plot-elo)")

    args = parser.parse_args()

    # Define player configurations to test
    if args.custom_players:
        # Start with empty config and only add what user specifies
        player_configs = []

        if args.add_random:
            player_configs.append({'id': 'Random', 'type': 'random'})

        if args.add_mcts:
            for iterations in args.add_mcts:
                player_configs.append({
                    'id': f'MCTS_iter{iterations}',
                    'type': 'mcts',
                    'iterations': iterations,
                    'use_table': False
                })
    else:
        # Default configuration
        player_configs = [
            {'id': 'Random', 'type': 'random'},
            {'id': 'MCTS_iter50', 'type': 'mcts', 'iterations': 50, 'use_table': False},
            {'id': 'MCTS_iter100', 'type': 'mcts', 'iterations': 100, 'use_table': False},
            {'id': 'MCTS_iter200', 'type': 'mcts', 'iterations': 200, 'use_table': False},
        ]

    # Add trained player if knowledge base provided
    if args.knowledge_base:
        kb_path = Path(args.knowledge_base)
        if kb_path.exists():
            table = TranspositionTable()
            table.load(args.knowledge_base)
            player_configs.append({
                'id': f'MCTS_iter100_trained({table.size()}pos)',
                'type': 'mcts',
                'iterations': 100,
                'use_table': True,
                'table': table
            })
            print(f"Loaded knowledge base: {table.size():,} positions")

    if not player_configs:
        print("Error: No players configured. Use --add-random or --add-mcts with --custom-players")
        exit(1)

    run_tournament(
        player_configs,
        games_per_matchup=args.games,
        rings=args.rings,
        elo_path=args.elo_path,
        plot_elo=args.plot_elo,
        plot_output=args.plot_output
    )