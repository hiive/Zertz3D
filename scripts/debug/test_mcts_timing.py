"""Quick test to measure MCTS timing per move."""

import time
from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer

# Create game
game = ZertzGame(rings=37)

iterations = 1000
# Create players with 100 iterations
player1 = MCTSZertzPlayer(
    game, n=1,
    iterations=iterations,
    use_transposition_table=True,
    use_transposition_lookups=True,
    clear_table_each_move=False,
    verbose=False
)
player2 = MCTSZertzPlayer(
    game, n=2,
    iterations=iterations,
    use_transposition_table=True,
    use_transposition_lookups=True,
    clear_table_each_move=False,
    verbose=False
)

# Share transposition table
from learner.mcts.transposition_table import TranspositionTable
table = TranspositionTable()
player1.transposition_table = table
player2.transposition_table = table

# Play game and time each move
move_times = []
move_count = 0
game_start = time.time()

print("Playing test game with 100 iterations per move...")
print()

while game.get_game_ended() is None:
    move_start = time.time()

    current_player = player1 if game.board.get_cur_player() == 0 else player2
    action_type, action_data = current_player.get_action()

    move_time = time.time() - move_start
    move_times.append(move_time)
    move_count += 1

    if action_type == "PASS":
        game.take_action("PASS", None)
        print(f"Move {move_count}: PASS ({move_time:.3f}s)")
    else:
        game.take_action(action_type, action_data)
        print(f"Move {move_count}: {action_type} ({move_time:.3f}s)")

game_time = time.time() - game_start

# Print statistics
print()
print("="*60)
print("Timing Results (100 iterations per move)")
print("="*60)
print(f"Total moves: {move_count}")
print(f"Total game time: {game_time:.2f}s")
print(f"Average time per move: {sum(move_times)/len(move_times):.3f}s")
print(f"Min time per move: {min(move_times):.3f}s")
print(f"Max time per move: {max(move_times):.3f}s")
print(f"Game outcome: {game.get_game_ended()}")
print()
print(f"Positions learned: {table.size():,}")
print(f"Total visits: {table.total_visits():,}")
print(f"Hit rate: {table.get_hit_rate():.1%}")
print()
print("Estimated training rates:")
print(f"  Time per game: ~{game_time:.1f}s")
print(f"  Games per hour: ~{3600/game_time:.1f}")
print(f"  1000 games would take: ~{game_time*1000/3600:.1f} hours")
print("="*60)