# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random
import time
import argparse
import ast

from panda3d.core import loadPrcFileData

from game.zertz_game import ZertzGame
from game.zertz_player import RandomZertzPlayer, ReplayZertzPlayer
from renderer.zertz_renderer import ZertzRenderer
import numpy as np


class ZertzGameController:

    def __init__(self, rings=37, replay_file=None):
        self.rings = rings
        self.marbles = {'w': 6, 'g': 8, 'b': 10}
        # the first player to obtain either 3 marbles of each color, or 4 white
        # marbles, or 5 grey marbles, or 6 black marbles wins the game.
        self.win_condition = [{'w': 4}, {'g': 5}, {'b': 6}, {'w': 3, 'g': 3, 'b': 3}]
        self.t = 5

        self.player1 = None
        self.player2 = None
        self.game = None
        self.replay_mode = replay_file is not None
        self.replay_actions = None

        # Set random seed before any random operations
        # This ensures reproducibility for marble rotations and game moves
        if not self.replay_mode:
            # t = int(time.time())
            # t = 1619318796
            t = 1726887625
            print(f"-- Setting Seed: {t}")
            np.random.seed(t)
            random.seed(t)
            # Test that seed is working
            test_val = np.random.randint(1000)
            print(f"-- Seed test value (should always be 11): {test_val}")
            # Reset seed after test
            np.random.seed(t)
            random.seed(t)

        # Load replay first to detect board size
        if self.replay_mode:
            self.replay_actions = self._load_replay(replay_file)

        # Create renderer with detected board size
        self.renderer = ZertzRenderer(rings=self.rings)
        self.board_layout = np.copy(self.renderer.pos_array)

        self._reset_board()

        move_time = 0.666
        self.task = self.renderer.taskMgr.doMethodLater(move_time, self.update_game, 'update_game', sort=49)

    def _detect_board_size(self, all_actions):
        """Detect board size by finding the maximum ring coordinate used."""
        max_letter = 'A'
        max_number = 1

        for action in all_actions:
            # Check all position fields that might contain ring coordinates
            for key in ['dst', 'src', 'remove', 'cap']:
                if key in action and action[key]:
                    pos = str(action[key])
                    if len(pos) >= 2:
                        letter = pos[0].upper()
                        try:
                            number = int(pos[1:])
                            if letter > max_letter:
                                max_letter = letter
                            if number > max_number:
                                max_number = number
                        except ValueError:
                            continue

        # Calculate board size from max coordinates
        # For hexagonal boards: A-G (width 7) = 37 rings, A-H (width 8) = 48 rings, A-J (width 9) = 61 rings
        max_x = ord(max_letter) - ord('A')
        width = max(max_x, max_number) + 1

        # Map width to standard board sizes
        if width <= 7:
            return 37
        elif width <= 8:
            return 48
        else:
            return 61

    def _load_replay(self, replay_file):
        """Load replay actions from a text file and detect board size."""
        print(f"Loading replay from: {replay_file}")
        player1_actions = []
        player2_actions = []
        all_actions = []

        with open(replay_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse line format: "Player N: {'action': 'PUT', ...}"
                if line.startswith('Player '):
                    parts = line.split(': ', 1)
                    player_num = int(parts[0].split()[1])
                    action_dict = ast.literal_eval(parts[1])

                    all_actions.append(action_dict)
                    if player_num == 1:
                        player1_actions.append(action_dict)
                    elif player_num == 2:
                        player2_actions.append(action_dict)

        # Detect board size from coordinates
        detected_rings = self._detect_board_size(all_actions)
        print(f"Detected board size: {detected_rings} rings")
        self.rings = detected_rings

        print(f"Loaded {len(player1_actions)} actions for Player 1")
        print(f"Loaded {len(player2_actions)} actions for Player 2")
        return player1_actions, player2_actions

    def run(self):
        self.renderer.run()

    def _reset_board(self):
        # Setup
        print("** New game **")

        self.game = ZertzGame(self.rings, self.marbles, self.win_condition, self.t,
                              board_layout=self.board_layout)
        # game.print_state()
        self.renderer.reset_board()

        if self.replay_mode:
            print("-- Replay Mode --")
            player1_actions, player2_actions = self.replay_actions
            self.player1 = ReplayZertzPlayer(self.game, 1, player1_actions)
            self.player2 = ReplayZertzPlayer(self.game, 2, player2_actions)
        else:
            self.player1 = RandomZertzPlayer(self.game, 1)
            self.player2 = RandomZertzPlayer(self.game, 2)

    def update_game(self, task):
        p_ix = self.game.get_cur_player_value()
        player = self.player1 if p_ix == 1 else self.player2

        try:
            ax, ay = player.get_action()
        except ValueError as e:
            print(f"Error getting action: {e}")
            if self.replay_mode:
                print("Replay finished")
                return task.done
            raise

        try:
            _, action_dict = self.game.action_to_str(ax, ay)
        except IndexError as e:
            print(f"Error converting action to string: {e}")
            print(f"Action type: {ax}, Action: {ay}")
            raise

        self.renderer.show_action(player, action_dict, task.delay_time)
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

            if self.replay_mode:
                print("Replay complete")
                return task.done
            else:
                self._reset_board()
        return task.again


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zertz 3D Game')
    parser.add_argument('--replay', type=str, help='Path to replay file (board size auto-detected)')
    parser.add_argument('--rings', type=int, default=37, help='Number of rings on the board (default: 61, ignored if --replay is used)')
    args = parser.parse_args()

    loadPrcFileData("", "gl-version 3 2")
    game = ZertzGameController(rings=args.rings, replay_file=args.replay)
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
