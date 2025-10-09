# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import random
import argparse
import ast
import time
import hashlib

from game.zertz_game import ZertzGame, PLAYER_1_WIN, PLAYER_2_WIN
from game.zertz_player import RandomZertzPlayer, ReplayZertzPlayer
from renderer.zertz_renderer import ZertzRenderer
import numpy as np


class ZertzGameController:

    def __init__(self, rings=37, replay_file=None, seed=None, log_to_file=False, partial_replay=False, headless=False, max_games=None, show_moves=False):
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
        self.headless = headless
        self.max_games = max_games  # None means play indefinitely
        self.games_played = 0

        # Show moves mode
        self.show_moves = show_moves
        self.highlight_duration = 1.5  # Duration to show each highlight phase (seconds)
        self.pending_action = None  # Store action while showing highlights
        self.pending_player = None  # Store player while showing highlights
        self.show_moves_phase = None  # Track phase: 'placement_highlights', 'selected_placement', 'removal_highlights', 'selected_removal', None
        self.pending_result = None  # Store result from take_action for later processing
        self.pending_action_dict = None  # Store action_dict before take_action changes game state

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

        # Create renderer with detected board size (only if not headless)
        self.renderer = None if headless else ZertzRenderer(rings=self.rings)

        self._reset_board()

        # Setup game loop
        if not headless:
            move_time = 0.666
            self.task = self.renderer.taskMgr.doMethodLater(move_time, self.update_game, 'update_game', sort=49)

    def _detect_board_size(self, all_actions):
        """Detect board size by finding the maximum letter coordinate used."""
        from game.zertz_board import ZertzBoard

        max_letter = 'A'

        for action in all_actions:
            # Check all position fields that might contain ring coordinates
            for key in ['dst', 'src', 'remove', 'cap']:
                if key in action and action[key]:
                    pos = str(action[key])
                    if len(pos) >= 2:
                        letter = pos[0].upper()
                        if letter > max_letter:
                            max_letter = letter

        # Detect board size from max letter (number of columns)
        # For hexagonal boards:
        #   A-G (7 letters) = 37 rings
        #   A-H (8 letters) = 48 rings
        #   A-J (9 letters, skipping I) = 61 rings
        if max_letter <= 'G':
            return ZertzBoard.SMALL_BOARD_37
        elif max_letter <= 'H':
            return ZertzBoard.MEDIUM_BOARD_48
        else:
            # J or beyond = 61 ring board
            return ZertzBoard.LARGE_BOARD_61

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
            self.log_file.write("#\n")
            print(f"Logging game to: {self.log_filename}")

    def _close_log_file(self):
        """Close the current log file and append final game state."""
        if self.log_file is not None:
            # Append final game state as comments
            self.log_file.write("#\n")
            self.log_file.write("# Final game state:\n")
            self.log_file.write("# ---------------\n")
            self.log_file.write("# Board state:\n")
            board_state = self.game.board.state[0] + self.game.board.state[1] + self.game.board.state[2] * 2 + self.game.board.state[3] * 3
            for row in board_state:
                self.log_file.write(f"# {row}\n")
            self.log_file.write("# ---------------\n")
            self.log_file.write("# Marble supply:\n")
            self.log_file.write(f"# {self.game.board.state[-10:-1, 0, 0]}\n")
            self.log_file.write("# ---------------\n")
            self.log_file.close()
            self.log_file = None
            print(f"Game log saved to: {self.log_filename}")

    def _log_action(self, player_num, action_dict):
        """Log an action to the file if logging is enabled."""
        if self.log_file is not None:
            self.log_file.write(f"Player {player_num}: {action_dict}\n")
            self.log_file.flush()

    def run(self):
        if self.headless:
            self._run_headless()
        else:
            self.renderer.run()

    def _run_headless(self):
        """Run game loop without renderer."""
        # Create a simple task object for compatibility with update_game
        class SimpleTask:
            def __init__(self):
                self.delay_time = 0
                self.done = False
                self.again = True

        task = SimpleTask()

        # Run game until it's done
        while not task.done:
            result = self.update_game(task)
            if result == task.done:
                task.done = True

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

        self.game = ZertzGame(self.rings, self.marbles, self.win_condition, self.t)
        # game.print_state()
        if self.renderer is not None:
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

    def _display_valid_moves(self, player):
        """Display valid move information for the current player.

        Args:
            player: The current player
        """
        placement, capture = self.game.get_valid_actions()

        print(f"\n--- Valid Moves for Player {player.n} ---")

        # Count valid moves
        num_placement = np.sum(placement)
        num_capture = np.sum(capture)

        print(f"Placement moves: {num_placement}")
        print(f"Capture moves: {num_capture}")

        # If there are captures, they are mandatory
        if num_capture > 0:
            print("CAPTURES ARE MANDATORY")
            # Show some example capture positions
            capture_positions = np.argwhere(capture)
            if len(capture_positions) > 0:
                print(f"Sample capture positions:")
                for i, (direction, y, x) in enumerate(capture_positions[:5]):  # Show up to 5
                    src_str = self.game.board.index_to_str((y, x))
                    print(f"  - Jump from {src_str} (direction {direction})")
                if len(capture_positions) > 5:
                    print(f"  ... and {len(capture_positions) - 5} more")
        else:
            # Show placement information - separate positions from removals
            placement_positions = np.argwhere(placement)
            if len(placement_positions) > 0:
                # Group by (marble, destination) to find unique placement positions
                marble_types = ['w', 'g', 'b']
                placements = {}  # {(marble, dst_str): [list of removal positions]}
                removals_set = set()  # Track all possible removals

                for marble_idx, dst, rem in placement_positions:
                    marble = marble_types[marble_idx]
                    dst_y = dst // self.game.board.width
                    dst_x = dst % self.game.board.width
                    dst_str = self.game.board.index_to_str((dst_y, dst_x))

                    key = (marble, dst_str)
                    if key not in placements:
                        placements[key] = []

                    if rem != self.game.board.width ** 2:
                        rem_y = rem // self.game.board.width
                        rem_x = rem % self.game.board.width
                        rem_str = self.game.board.index_to_str((rem_y, rem_x))
                        placements[key].append(rem_str)
                        removals_set.add(rem_str)

                print(f"\nPlacement positions ({len(placements)} unique):")
                for i, ((marble, dst_str), _) in enumerate(list(placements.items())[:10]):
                    print(f"  - PUT {marble} at {dst_str}")
                if len(placements) > 10:
                    print(f"  ... and {len(placements) - 10} more")

                if removals_set:
                    print(f"\nRings available to remove ({len(removals_set)}):")
                    removals_list = sorted(list(removals_set))
                    # Show first 10
                    print(f"  {', '.join(removals_list[:10])}")
                    if len(removals_list) > 10:
                        print(f"  ... and {len(removals_list) - 10} more")

        # Show current captured marbles
        print(f"\nPlayer {player.n} captured: {player.captured}")
        print("---" + "-" * 30 + "\n")

    def _queue_placement_highlights(self):
        """Queue placement highlights only."""
        placement, capture = self.game.get_valid_actions()

        # Find destinations with ANY valid placement (any marble type, any removal option)
        valid_dests = np.any(placement, axis=(0, 2))  # Shape: (width²,)
        dest_indices = np.argwhere(valid_dests).flatten()

        placement_rings = []
        width = self.game.board.width
        for dst_idx in dest_indices:
            dst_y = dst_idx // width
            dst_x = dst_idx % width
            pos_str = self.game.board.index_to_str((dst_y, dst_x))
            if pos_str:
                placement_rings.append(pos_str)

        if placement_rings:
            self.renderer.queue_highlight(placement_rings, self.highlight_duration)

    def _queue_removal_highlights(self, marble_idx, dst, placement_array, board):
        """Queue removal highlights for a specific placement action.

        Args:
            marble_idx: Index of marble type (0=w, 1=g, 2=b)
            dst: Destination index where marble will be placed
            placement_array: The placement array from BEFORE the action
            board: ZertzBoard instance to use for index_to_str conversion
        """
        width = board.width

        # Get valid removal indices for this (marble_idx, dst) combination
        removal_mask = placement_array[marble_idx, dst, :]
        removable_indices = np.argwhere(removal_mask).flatten()

        removable_rings = []
        for rem_idx in removable_indices:
            if rem_idx != width ** 2 and rem_idx != dst:  # Exclude "no removal" and destination
                rem_y = rem_idx // width
                rem_x = rem_idx % width
                rem_str = board.index_to_str((rem_y, rem_x))
                if rem_str:
                    removable_rings.append(rem_str)

        if removable_rings:
            self.renderer.queue_highlight(
                removable_rings,
                self.highlight_duration,
                color=self.renderer.REMOVABLE_HIGHLIGHT_COLOR,
                emission=self.renderer.REMOVABLE_HIGHLIGHT_EMISSION
            )

    def update_game(self, task):
        # State machine for show_moves mode
        if self.pending_action is not None:
            if self.renderer is not None and self.renderer.is_highlight_active():
                # Still showing highlights, wait
                return task.again

            # Highlights finished, check which phase we're in
            if self.show_moves_phase == 'placement_highlights':
                # All placement highlights done, now show just the selected ring
                ax, ay, player = self.pending_action

                try:
                    _, action_dict = self.game.action_to_str(ax, ay)
                except IndexError as e:
                    print(f"Error converting action to string: {e}")
                    print(f"Action type: {ax}, Action: {ay}")
                    raise

                # Queue highlight for selected placement ring only
                selected_ring = action_dict['dst']
                self.renderer.queue_highlight([selected_ring], self.highlight_duration)
                self.show_moves_phase = 'selected_placement'

            elif self.show_moves_phase == 'selected_placement':
                # Selected placement highlight done, place marble and execute game logic
                ax, ay, player = self.pending_action

                try:
                    _, action_dict = self.game.action_to_str(ax, ay)
                except IndexError as e:
                    print(f"Error converting action to string: {e}")
                    print(f"Action type: {ax}, Action: {ay}")
                    raise

                # Save action_dict before game state changes
                self.pending_action_dict = action_dict

                self._log_action(player.n, action_dict)
                print(f'Player {player.n}: {action_dict}')

                # Get placement array BEFORE executing action (for removal highlights)
                placement_before, _ = self.game.get_valid_actions()

                # Queue removal highlights BEFORE executing action (so board is in original state)
                if ax == "PUT":
                    marble_idx, dst, rem = ay  # Unpack action tuple
                    self._queue_removal_highlights(marble_idx, dst, placement_before, self.game.board)

                # Animate marble placement
                if self.renderer is not None and ax == "PUT":
                    self.renderer.show_marble_placement(player, action_dict, task.delay_time)

                # Execute game logic
                result = self.game.take_action(ax, ay)
                self.pending_result = result

                # Move to removal highlights phase
                if ax == "PUT":
                    self.show_moves_phase = 'removal_highlights'
                else:
                    # Capture actions don't have removal phase - animate immediately
                    if self.renderer is not None:
                        self.renderer.show_action(player, action_dict, task.delay_time)

                    self.pending_action = None
                    self.show_moves_phase = None
                    # Process result immediately
                    if result is not None:
                        if isinstance(result, list):
                            for removal in result:
                                if removal['marble'] is not None:
                                    player.add_capture(removal['marble'])
                                if self.renderer is not None:
                                    self.renderer.show_isolated_removal(player, removal['pos'], removal['marble'], task.delay_time)
                        else:
                            player.add_capture(result)

            elif self.show_moves_phase == 'removal_highlights':
                # All removal highlights done, now show just the selected ring
                # Use the saved action_dict (before game state changed)
                action_dict = self.pending_action_dict

                # Queue highlight for selected removal ring only
                selected_removal = action_dict['remove']
                if selected_removal:  # Only if a ring is being removed
                    self.renderer.queue_highlight(
                        [selected_removal],
                        self.highlight_duration,
                        color=self.renderer.REMOVABLE_HIGHLIGHT_COLOR,
                        emission=self.renderer.REMOVABLE_HIGHLIGHT_EMISSION
                    )
                self.show_moves_phase = 'selected_removal'

            elif self.show_moves_phase == 'selected_removal':
                # Selected removal highlight done, now animate ring removal and process result
                ax, ay, player = self.pending_action
                result = self.pending_result
                action_dict = self.pending_action_dict  # Use saved action_dict

                # Now animate ring removal only (marble was already placed)
                if self.renderer is not None:
                    self.renderer.show_ring_removal(action_dict, task.delay_time)

                # Process result
                if result is not None:
                    if isinstance(result, list):
                        for removal in result:
                            if removal['marble'] is not None:
                                player.add_capture(removal['marble'])
                            if self.renderer is not None:
                                self.renderer.show_isolated_removal(player, removal['pos'], removal['marble'], task.delay_time)
                    else:
                        player.add_capture(result)

                self.pending_action = None
                self.show_moves_phase = None
                self.pending_result = None

        else:
            # Get next action
            p_ix = self.game.get_cur_player_value()
            player = self.player1 if p_ix == 1 else self.player2

            try:
                ax, ay = player.get_action()
            except ValueError as e:
                print(f"Error getting action: {e}")
                if self.replay_mode:
                    if self.partial_replay:
                        print("Replay finished - continuing with random play")
                        self.player1 = RandomZertzPlayer(self.game, 1)
                        self.player2 = RandomZertzPlayer(self.game, 2)
                        self.player1.captured = player.captured if player.n == 1 else self.player1.captured
                        self.player2.captured = player.captured if player.n == 2 else self.player2.captured
                        self.replay_mode = False
                        player = self.player1 if p_ix == 1 else self.player2
                        ax, ay = player.get_action()
                    else:
                        print("Replay finished")
                        return task.done
                else:
                    raise

            # If show_moves is enabled, queue highlights and wait
            if self.show_moves and self.renderer is not None:
                self._display_valid_moves(player)
                # Only highlight for PUT actions, skip CAP for now
                if ax == "PUT":
                    self._queue_placement_highlights()
                    self.pending_action = (ax, ay, player)
                    self.show_moves_phase = 'placement_highlights'
                    return task.again
                # For CAP actions, fall through to execute immediately

            # Execute action immediately if not showing moves or headless
            if self.show_moves:
                self._display_valid_moves(player)

            try:
                _, action_dict = self.game.action_to_str(ax, ay)
            except IndexError as e:
                print(f"Error converting action to string: {e}")
                print(f"Action type: {ax}, Action: {ay}")
                raise

            self._log_action(player.n, action_dict)
            print(f'Player {player.n}: {action_dict}')

            if self.renderer is not None:
                self.renderer.show_action(player, action_dict, task.delay_time)

            result = self.game.take_action(ax, ay)

            # Handle result
            if result is not None:
                if isinstance(result, list):
                    for removal in result:
                        if removal['marble'] is not None:
                            player.add_capture(removal['marble'])
                        if self.renderer is not None:
                            self.renderer.show_isolated_removal(player, removal['pos'], removal['marble'], task.delay_time)
                else:
                    player.add_capture(result)

        game_over = self.game.get_game_ended()
        if game_over is not None:  # None means game continuing.
            if game_over in [PLAYER_1_WIN, PLAYER_2_WIN]:
                winner = game_over
            else:
                winner = None  # Tie

            print()
            if winner:
                print(f'Winner: Player {winner}')
            else:
                print('Game ended in a tie')
            self.game.print_state()

            print(self.player1.captured)
            print(self.player2.captured)

            # Increment games played counter
            self.games_played += 1

            if self.replay_mode:
                print("Replay complete")
                return task.done
            elif self.max_games is not None and self.games_played >= self.max_games:
                # Reached max games limit
                print(f"Completed {self.games_played} game(s)")
                return task.done
            else:
                # Continue with next game
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
    parser.add_argument('--headless', action='store_true', help='Run without 3D renderer')
    parser.add_argument('--games', type=int, help='Number of games to play (default: play indefinitely)')
    parser.add_argument('--show-moves', action='store_true', help='Show valid moves before each turn')
    args = parser.parse_args()

    game = ZertzGameController(rings=args.rings, replay_file=args.replay, seed=args.seed,
                                log_to_file=args.log, partial_replay=args.partial, headless=args.headless,
                                max_games=args.games, show_moves=args.show_moves)
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
