# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from game.zertz_game import ZertzGame
from game.zertz_player import RandomZertzPlayer
from renderer.zertz_renderer import ZertzRenderer
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    rings = 48
    marbles = {'w': 6, 'g': 8, 'b': 10}
    # the first player to obtain either 3 marbles of each color, or 4 white
    # marbles, or 5 grey marbles, or 6 black marbles wins the game.
    win_con = [{'w': 4}, {'g': 5}, {'b': 6}, {'w': 3, 'g': 3, 'b': 3}]
    t = 5

    renderer = ZertzRenderer()
    board_layout = renderer.pos_array != ''
    # Setup
    game = ZertzGame(rings, marbles, win_con, t, layout=board_layout)
    game.print_state()

    player1 = RandomZertzPlayer(game, 1)
    player2 = RandomZertzPlayer(game, 2)

    # game.print_state()
    game_over = 0
    winner = None
    while game_over == 0:
        p_ix = game.get_cur_player_value()
        player = player1 if p_ix == 1 else player2
        ax, ay = player.get_action()

        action_str, action_dict = game.action_to_str(ax, ay)
        result = game.take_action(ax, ay)
        if result is not None:
            player.add_capture(result)
        print(f'Player {player.n}: {action_dict}')
        game_over = game.get_game_ended()
    winner = 1 if game_over == 1 else 2
    print()
    print(f'Winner: Player {winner}')
    game.print_state()

    print(player1.captured)
    print(player2.captured)

    # valid_actions = game.get_valid_actions()
    # symmetries = game.get_symmetries()
#    print(valid_actions)

    # print(symmetries)
    renderer.run()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

