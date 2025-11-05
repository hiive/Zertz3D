"""Debug MCTS player to see what's happening."""

import random

import numpy as np

import sys
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))
from utils.project_path import find_project_root

project_root = find_project_root(Path(__file__).parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from game.formatters.transcript_formatter import TranscriptFormatter
from game.zertz_game import ZertzGame
from game.players import MCTSZertzPlayer, RandomZertzPlayer

# Configuration
NUM_GAMES = 10
MCTS_ITERATIONS = 50000
VERBOSE_FIRST_GAME = True  # Show detailed output for first game only
MAX_MOVES_PER_GAME = 100

# Set seeds for reproducibility
random.seed(43)
np.random.seed(43)

# Statistics tracking
stats = {
    'p1_wins': 0,
    'p2_wins': 0,
    'ties': 0,
    'total_moves': [],
    'mcts_as_p1': {'wins': 0, 'losses': 0, 'ties': 0},
    'mcts_as_p2': {'wins': 0, 'losses': 0, 'ties': 0},
    'win_record': []
}

print(f"Running {NUM_GAMES} games to gather MCTS statistics")
print(f"MCTS iterations per move: {MCTS_ITERATIONS}")
print("=" * 70)

for game_num in range(NUM_GAMES):
    # Alternate which player is MCTS
    mcts_is_player1 = (game_num % 2 == 0)

    # Create game
    game = ZertzGame(rings=37)

    # Create players
    if mcts_is_player1:
        player1 = MCTSZertzPlayer(
            game, n=1,
            iterations=MCTS_ITERATIONS,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table_each_move=True,
            verbose=(VERBOSE_FIRST_GAME and game_num == 0)
        )
        player2 = RandomZertzPlayer(game, n=2)
        game_type = "MCTS vs Random"
    else:
        player1 = RandomZertzPlayer(game, n=1)
        player2 = MCTSZertzPlayer(
            game, n=2,
            iterations=MCTS_ITERATIONS,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table_each_move=True,
            verbose=(VERBOSE_FIRST_GAME and game_num == 0)
        )
        game_type = "Random vs MCTS"

    if game_num == 0 and VERBOSE_FIRST_GAME:
        print(f"\nGame {game_num + 1}/{NUM_GAMES} ({game_type}) - VERBOSE MODE")
        print("=" * 70)
    else:
        print(f"\nGame {game_num + 1}/{NUM_GAMES} ({game_type})...", end=" ", flush=True)

    # Play game
    move_count = 0
    exploration_stats = []  # Track exploration coverage per move

    while game.get_game_ended() is None and move_count < MAX_MOVES_PER_GAME:
        current_player = player1 if game.board.get_cur_player() == 0 else player2

        if VERBOSE_FIRST_GAME and game_num == 0:
            print(f"\n{'-'*60}")
            print(f"Move {move_count + 1} - Player {current_player.n}")
            print(f"{'-'*60}")

        # Get total legal actions before MCTS runs
        placement_mask, capture_mask = game.get_valid_actions()
        total_actions = placement_mask.sum() + capture_mask.sum()

        if total_actions == 0:
            total_actions = 1  # PASS action

        action_type, action_data = current_player.get_action()

        # For MCTS player, track exploration coverage
        if isinstance(current_player, MCTSZertzPlayer):
            explored_actions = getattr(current_player, "_last_root_children", 0)
            exploration_pct = (explored_actions / total_actions * 100) if total_actions > 0 else 0

            exploration_stats.append({
                'move': move_count + 1,
                'player': current_player.n,
                'total_actions': int(total_actions),
                'explored_actions': explored_actions,
                'exploration_pct': exploration_pct,
                'root_visits': getattr(current_player, "_last_root_visits", 0),
                'root_value': getattr(current_player, "_last_root_value", 0.0),
            })

        if VERBOSE_FIRST_GAME and game_num == 0:
            if isinstance(current_player, MCTSZertzPlayer):
                explored = getattr(current_player, "_last_root_children", 0)
                pct = (explored / total_actions * 100) if total_actions > 0 else 0
                visits = getattr(current_player, "_last_root_visits", 0)
                value = getattr(current_player, "_last_root_value", 0.0)
                print(f"Action space: {explored}/{int(total_actions)} explored ({pct:.1f}%)")
                print(f"Root visits: {visits}, value: {value:.3f}")

            if action_type == "PASS":
                action_str = "PASS"
                action_dict = {"action": "PASS"}
            else:
                action_str, action_dict = game.action_to_str(action_type, action_data)

            notation = game.action_to_notation(action_dict)
            transcript = TranscriptFormatter.action_to_transcript(action_dict)
            print(f"Action: {action_str}")
            print(f"Notation: {notation}")
            print(f"Transcript: {transcript}")

        if action_type == "PASS":
            game.take_action("PASS", None)
        else:
            game.take_action(action_type, action_data)

        move_count += 1

    # Record result
    result = game.get_game_ended()
    stats['total_moves'].append(move_count)
    stats['win_record'].append(result)

    if result == 1:
        stats['p1_wins'] += 1
        if mcts_is_player1:
            stats['mcts_as_p1']['wins'] += 1
            outcome = "MCTS (P1) won"
        else:
            stats['mcts_as_p2']['losses'] += 1
            outcome = "Random (P1) won"
    elif result == -1:
        stats['p2_wins'] += 1
        if mcts_is_player1:
            stats['mcts_as_p1']['losses'] += 1
            outcome = "Random (P2) won"
        else:
            stats['mcts_as_p2']['wins'] += 1
            outcome = "MCTS (P2) won"
    else:
        stats['ties'] += 1
        if mcts_is_player1:
            stats['mcts_as_p1']['ties'] += 1
        else:
            stats['mcts_as_p2']['ties'] += 1
        outcome = "Tie"

    if game_num == 0 and VERBOSE_FIRST_GAME:
        print(f"\n{'='*60}")
        print(f"Game ended after {move_count} moves: {outcome}")
        print(f"{'='*60}")
    else:
        print(f"{outcome} ({move_count} moves)")

# Print aggregate statistics
print(f"\n{'='*70}")
print(f"AGGREGATE STATISTICS ({NUM_GAMES} games)")
print(f"{'='*70}")
print("\nOverall Results:")
print(f"  Player 1 wins: {stats['p1_wins']} ({stats['p1_wins']/NUM_GAMES*100:.1f}%)")
print(f"  Player 2 wins: {stats['p2_wins']} ({stats['p2_wins']/NUM_GAMES*100:.1f}%)")
print(f"  Ties: {stats['ties']} ({stats['ties']/NUM_GAMES*100:.1f}%)")
if stats['win_record']:
    overall_mean = stats['p1_wins'] / NUM_GAMES
    win_values = [1 if r == 1 else (0 if r == -1 else 0.5) for r in stats['win_record']]
    print(f"  Win rate mean: {overall_mean*100:.1f}%")

print("\nMCTS as Player 1:")
mcts_p1_games = sum(stats['mcts_as_p1'].values())
if mcts_p1_games > 0:
    print(f"  Wins: {stats['mcts_as_p1']['wins']}/{mcts_p1_games} ({stats['mcts_as_p1']['wins']/mcts_p1_games*100:.1f}%)")
    print(f"  Losses: {stats['mcts_as_p1']['losses']}/{mcts_p1_games} ({stats['mcts_as_p1']['losses']/mcts_p1_games*100:.1f}%)")
    print(f"  Ties: {stats['mcts_as_p1']['ties']}/{mcts_p1_games} ({stats['mcts_as_p1']['ties']/mcts_p1_games*100:.1f}%)")
    if stats['mcts_as_p1']['wins'] + stats['mcts_as_p1']['losses'] > 0:
        p1_rate = stats['mcts_as_p1']['wins'] / max(1, (stats['mcts_as_p1']['wins'] + stats['mcts_as_p1']['losses']))
        print(f"  Win rate mean as P1: {p1_rate*100:.1f}%")

print("\nMCTS as Player 2:")
mcts_p2_games = sum(stats['mcts_as_p2'].values())
if mcts_p2_games > 0:
    print(f"  Wins: {stats['mcts_as_p2']['wins']}/{mcts_p2_games} ({stats['mcts_as_p2']['wins']/mcts_p2_games*100:.1f}%)")
    print(f"  Losses: {stats['mcts_as_p2']['losses']}/{mcts_p2_games} ({stats['mcts_as_p2']['losses']/mcts_p2_games*100:.1f}%)")
    print(f"  Ties: {stats['mcts_as_p2']['ties']}/{mcts_p2_games} ({stats['mcts_as_p2']['ties']/mcts_p2_games*100:.1f}%)")
    if stats['mcts_as_p2']['wins'] + stats['mcts_as_p2']['losses'] > 0:
        p2_rate = stats['mcts_as_p2']['wins'] / max(1, (stats['mcts_as_p2']['wins'] + stats['mcts_as_p2']['losses']))
        print(f"  Win rate mean as P2: {p2_rate*100:.1f}%")

total_mcts_games = mcts_p1_games + mcts_p2_games
total_mcts_wins = stats['mcts_as_p1']['wins'] + stats['mcts_as_p2']['wins']
print("\nOverall MCTS Performance:")
print(f"  Total wins: {total_mcts_wins}/{total_mcts_games} ({total_mcts_wins/total_mcts_games*100:.1f}%)")

avg_moves = sum(stats['total_moves']) / len(stats['total_moves'])
std_moves = np.std(stats['total_moves'])
print("\nGame Length:")
print(f"  Average moves per game: {avg_moves:.1f}, std: {std_moves:.1f}")
print(f"  Min: {min(stats['total_moves'])}, Max: {max(stats['total_moves'])}")
print(f"{'='*70}")
