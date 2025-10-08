# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random
import argparse
import ast
import time
import hashlib

from panda3d.core import loadPrcFileData

from game.zertz_game import ZertzGame
from game.zertz_player import RandomZertzPlayer, ReplayZertzPlayer
from renderer.zertz_renderer import ZertzRenderer
import numpy as np


class ZertzGameController:

    def __init__(self, rings=37, replay_file=None, seed=None, log_to_file=False, partial_replay=False):
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
        self.partial_replay = partial_replay
        self.replay_actions = None
        self.current_seed = None
        self.log_to_file = log_to_file
        self.log_file = None
        self.log_filename = None

        # Set random seed before any random operations
        # This ensures reproducibility for game moves (not used in replay mode)
        if not self.replay_mode:
            if seed is None:
                seed = int(time.time())
            self.current_seed = seed
            self._apply_seed(seed)

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
                if not line or line.startswith('#'):
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

    def _apply_seed(self, seed):
        """Apply a seed to both random number generators and print it."""
        print(f"-- Setting Seed: {seed}")
        np.random.seed(seed)
        random.seed(seed)

    def _generate_next_seed(self):
        """Generate the next seed deterministically from the current seed using hash."""
        # Hash the current seed to get a new one
        hash_obj = hashlib.sha256(str(self.current_seed).encode())
        # Take first 8 bytes and convert to integer
        new_seed = int.from_bytes(hash_obj.digest()[:8], byteorder='big')
        # Keep it in a reasonable range (32-bit unsigned int)
        new_seed = new_seed % (2**32)
        return new_seed

    def _open_log_file(self):
        """Open a new log file for the current game."""
        if self.log_to_file and not self.replay_mode:
            self.log_filename = f"zertzlog_{self.current_seed}.txt"
            self.log_file = open(self.log_filename, 'w')
            self.log_file.write(f"# Seed: {self.current_seed}\n")
            self.log_file.write(f"# Rings: {self.rings}\n")
            self.log_file.write(f"#\n")
            print(f"Logging game to: {self.log_filename}")

    def _close_log_file(self):
        """Close the current log file and append final game state."""
        if self.log_file is not None:
            # Append final game state as comments
            self.log_file.write(f"#\n")
            self.log_file.write(f"# Final game state:\n")
            self.log_file.write(f"# ---------------\n")
            self.log_file.write(f"# Board state:\n")
            board_state = self.game.board.state[0] + self.game.board.state[1] + self.game.board.state[2] * 2 + self.game.board.state[3] * 3
            for row in board_state:
                self.log_file.write(f"# {row}\n")
            self.log_file.write(f"# ---------------\n")
            self.log_file.write(f"# Marble supply:\n")
            self.log_file.write(f"# {self.game.board.state[-10:-1, 0, 0]}\n")
            self.log_file.write(f"# ---------------\n")
            self.log_file.close()
            self.log_file = None
            print(f"Game log saved to: {self.log_filename}")

    def _log_action(self, player_num, action_dict):
        """Log an action to the file if logging is enabled."""
        if self.log_file is not None:
            self.log_file.write(f"Player {player_num}: {action_dict}\n")
            self.log_file.flush()

    def run(self):
        self.renderer.run()

    def _reset_board(self):
        # Setup
        print("** New game **")

        # Close previous log file if it exists
        if self.log_file is not None:
            self._close_log_file()

        # Generate new seed for non-replay games (only after the first game)
        if not self.replay_mode and self.current_seed is not None and self.game is not None:
            self.current_seed = self._generate_next_seed()
            self._apply_seed(self.current_seed)

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
            # Open log file for new game
            self._open_log_file()

    def update_game(self, task):
        p_ix = self.game.get_cur_player_value()
        player = self.player1 if p_ix == 1 else self.player2

        try:
            ax, ay = player.get_action()
        except ValueError as e:
            print(f"Error getting action: {e}")
            if self.replay_mode:
                if self.partial_replay:
                    print("Replay finished - continuing with random play")
                    # Switch to random players
                    self.player1 = RandomZertzPlayer(self.game, 1)
                    self.player2 = RandomZertzPlayer(self.game, 2)
                    # Copy captured marbles from replay players
                    self.player1.captured = player.captured if player.n == 1 else self.player1.captured
                    self.player2.captured = player.captured if player.n == 2 else self.player2.captured
                    self.replay_mode = False
                    # Get new action from random player
                    player = self.player1 if p_ix == 1 else self.player2
                    ax, ay = player.get_action()
                else:
                    print("Replay finished")
                    return task.done
            else:
                raise

        try:
            _, action_dict = self.game.action_to_str(ax, ay)
        except IndexError as e:
            print(f"Error converting action to string: {e}")
            print(f"Action type: {ax}, Action: {ay}")
            raise

        # Log the action
        self._log_action(player.n, action_dict)

        self.renderer.show_action(player, action_dict, task.delay_time)
        result = self.game.take_action(ax, ay)

        # Handle result - could be captured marble type (CAP) or isolated removals list (PUT) or None
        if result is not None:
            if isinstance(result, list):
                # Isolated region removals from PUT action
                for removal in result:
                    if removal['marble'] is not None:
                        # Ring with marble - capture it
                        player.add_capture(removal['marble'])
                    # Animate the isolated ring/marble removal
                    self.renderer.show_isolated_removal(player, removal['pos'], removal['marble'], task.delay_time)
            else:
                # Normal capture from CAP action
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
    parser.add_argument('--rings', type=int, default=37, help='Number of rings on the board (default: 37, ignored if --replay is used)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible games (ignored if --replay is used)')
    parser.add_argument('--log', action='store_true', help='Log game actions to zertzlog_<seed>.txt (ignored if --replay is used)')
    parser.add_argument('--partial', action='store_true', help='Continue with random play after replay ends (only with --replay)')
    args = parser.parse_args()

    loadPrcFileData("", "gl-version 3 2")
    game = ZertzGameController(rings=args.rings, replay_file=args.replay, seed=args.seed,
                                log_to_file=args.log, partial_replay=args.partial)
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
