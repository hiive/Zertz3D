# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from game.zertz_game import ZertzGame
from game.zertz_player import RandomZertzPlayer
from renderer.zertz_renderer import ZertzRenderer
import numpy as np


class ZertzGameController:

    def __init__(self, rings=48):
        self.rings = rings
        self.marbles = {'w': 6, 'g': 8, 'b': 10}
        # the first player to obtain either 3 marbles of each color, or 4 white
        # marbles, or 5 grey marbles, or 6 black marbles wins the game.
        self.win_condition = [{'w': 4}, {'g': 5}, {'b': 6}, {'w': 3, 'g': 3, 'b': 3}]
        self.t = 5

        self.renderer = ZertzRenderer()
        self.board_layout = np.copy(self.renderer.pos_array)

        self.player1 = None
        self.player2 = None
        self.game = None
        self._reset_board()

        self.task = self.renderer.taskMgr.doMethodLater(0.05, self.update_game, 'update_game', sort=49)

    def run(self):
        self.renderer.run()

    def _reset_board(self):
        # Setup
        self.game = ZertzGame(self.rings, self.marbles, self.win_condition, self.t,
                              board_layout=self.board_layout)
        # game.print_state()
        self.renderer.reset_board()

        self.player1 = RandomZertzPlayer(self.game, 1)
        self.player2 = RandomZertzPlayer(self.game, 2)

    def update_game(self, task):
        p_ix = self.game.get_cur_player_value()
        player = self.player1 if p_ix == 1 else self.player2
        ax, ay = player.get_action()

        action_str, action_dict = self.game.action_to_str(ax, ay)
        self.renderer.show_action(player, action_dict)
        result = self.game.take_action(ax, ay)
        if result is not None:
            player.add_capture(result)

        game_over = self.game.get_game_ended()
        if game_over:
            winner = 1 if game_over == 1 else 2
            print()
            print(f'Winner: Player {winner}')
            self.game.print_state()

            print(self.player1.captured)
            print(self.player2.captured)
            self._reset_board()
        return task.again


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    game = ZertzGameController()
    game.run()

    # # game.print_state()
    # game_over = 0
    # winner = None
    # while game_over == 0:
    #     p_ix = game.get_cur_player_value()
    #     player = player1 if p_ix == 1 else player2
    #     ax, ay = player.get_action()
    #
    #     action_str, action_dict = game.action_to_str(ax, ay)
    #     result = game.take_action(ax, ay)
    #     if result is not None:
    #         player.add_capture(result)
    #     print(f'Player {player.n}: {action_dict}')
    #     game_over = game.get_game_ended()
    # winner = 1 if game_over == 1 else 2
    # print()
    # print(f'Winner: Player {winner}')
    # game.print_state()
    #
    # print(player1.captured)
    # print(player2.captured)
    #
    # # valid_actions = game.get_valid_actions()
    # # symmetries = game.get_symmetries()
    # #    print(valid_actions)
    #
    # # print(symmetries)
