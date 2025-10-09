"""Game controller for Zertz 3D.

Manages game loop, player actions, renderer updates, and game state.
"""

import random
import ast
import time
import hashlib

from game.zertz_game import (ZertzGame, PLAYER_1_WIN, PLAYER_2_WIN,
                             STANDARD_MARBLES, BLITZ_MARBLES,
                             STANDARD_WIN_CONDITIONS, BLITZ_WIN_CONDITIONS)
from game.zertz_player import RandomZertzPlayer, ReplayZertzPlayer
from renderer.zertz_renderer import ZertzRenderer
from controller.move_highlight_state_machine import MoveHighlightStateMachine
import numpy as np


class ZertzGameController:

    def __init__(self, rings=37, replay_file=None, seed=None, log_to_file=False, partial_replay=False, headless=False, max_games=None, show_moves=False, blitz=False):
        self.rings = rings
        self.blitz = blitz

        # Set marbles and win conditions based on variant
        if blitz:
            self.marbles = BLITZ_MARBLES
            self.win_condition = BLITZ_WIN_CONDITIONS
        else:
            self.marbles = STANDARD_MARBLES
            self.win_condition = STANDARD_WIN_CONDITIONS

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
        self.highlight_duration = 0.2  # Duration to show each highlight phase (seconds)
        self.highlight_sm = None  # State machine for highlights (created after game is initialized)

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

        # Validate blitz mode (only works with 37 rings)
        if self.blitz and self.rings != 37:
            raise ValueError("Blitz mode only works with 37 rings")

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
        """Load replay actions from a text file and detect board size and variant."""
        print(f"Loading replay from: {replay_file}")
        player1_actions = []
        player2_actions = []
        all_actions = []
        detected_blitz = False

        with open(replay_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Check for variant in comment headers
                if line.startswith('# Variant:'):
                    variant = line.split(':', 1)[1].strip().lower()
                    if variant == 'blitz':
                        detected_blitz = True

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

        # Handle blitz variant detection
        if detected_blitz:
            print("Detected blitz variant from replay file")
            if self.blitz:
                print("  (--blitz flag also specified)")
            else:
                print("  (automatically enabling blitz mode)")
                self.blitz = True
                self.marbles = BLITZ_MARBLES
                self.win_condition = BLITZ_WIN_CONDITIONS
        elif self.blitz:
            print("Warning: --blitz flag specified but replay file is standard mode")
            print("         Using blitz rules anyway")

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
            variant = "_blitz" if self.blitz else ""
            self.log_filename = f"zertzlog{variant}_{self.current_seed}.txt"
            self.log_file = open(self.log_filename, 'w')
            self.log_file.write(f"# Seed: {self.current_seed}\n")
            self.log_file.write(f"# Rings: {self.rings}\n")
            if self.blitz:
                self.log_file.write("# Variant: Blitz\n")
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
        variant_text = " (BLITZ)" if self.blitz else ""
        print(f"** New game{variant_text} **")

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

        # Create or recreate highlight state machine
        if self.show_moves and self.renderer is not None:
            self.highlight_sm = MoveHighlightStateMachine(self.renderer, self.game, self.highlight_duration)

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
            # Show detailed capture moves
            capture_positions = np.argwhere(capture)
            if len(capture_positions) > 0:
                print("Capture moves available:")
                for i, (direction, y, x) in enumerate(capture_positions[:10]):  # Show up to 10
                    try:
                        _, action_dict = self.game.action_to_str("CAP", (direction, y, x))
                        marble = action_dict['marble']
                        src = action_dict['src']
                        dst = action_dict['dst']
                        captured = action_dict['capture']
                        cap_pos = action_dict['cap']
                        print(f"  - CAP {marble} {src} -> {dst} capturing {captured} at {cap_pos}")
                    except (IndexError, KeyError):
                        # Fallback to basic info if conversion fails
                        src_str = self.game.board.index_to_str((y, x))
                        print(f"  - Jump from {src_str} (direction {direction})")
                if len(capture_positions) > 10:
                    print(f"  ... and {len(capture_positions) - 10} more")
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

    def update_game(self, task):
        # State machine for show_moves mode
        if self.highlight_sm and self.highlight_sm.is_active():
            # Update state machine
            should_continue = self.highlight_sm.update(task)
            if should_continue:
                return task.again

            # State machine finished, get result and player
            result = self.highlight_sm.get_pending_result()
            player = self.highlight_sm.get_pending_player()

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

        # Get next action
        if not (self.highlight_sm and self.highlight_sm.is_active()):
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

            # If show_moves is enabled, start the state machine
            if self.show_moves and self.renderer is not None:
                self._display_valid_moves(player)
                # Log and print action before starting state machine
                try:
                    _, action_dict = self.game.action_to_str(ax, ay)
                except IndexError as e:
                    print(f"Error converting action to string: {e}")
                    print(f"Action type: {ax}, Action: {ay}")
                    raise
                self._log_action(player.n, action_dict)
                print(f'Player {player.n}: {action_dict}')

                # Start state machine for all actions
                self.highlight_sm.start(ax, ay, player)
                # For PASS actions, state machine completes immediately (phase=None)
                # Don't return early - fall through to game_over check
                if self.highlight_sm.is_active():
                    return task.again
                # PASS completed, fall through to game_over check below

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

            # Update frozen region visuals after any action
            if self.renderer is not None:
                self.renderer.update_frozen_regions(self.game.board)

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