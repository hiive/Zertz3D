import numpy as np
import copy

from .zertz_board import ZertzBoard
from .action_result import ActionResult


# For full rules: http://www.gipf.com/zertz/rules/rules.html
# Class interface inspired by https://github.com/suragnair/alpha-zero-general

# Game outcome constants
PLAYER_1_WIN = 1
PLAYER_2_WIN = -1
TIE = 0
# None = game not over

# Game variant configurations
STANDARD_MARBLES = {'w': 6, 'g': 8, 'b': 10}
BLITZ_MARBLES = {'w': 5, 'g': 7, 'b': 9}

STANDARD_WIN_CONDITIONS = [{'w': 3, 'g': 3, 'b': 3}, {'w': 4}, {'g': 5}, {'b': 6}]
BLITZ_WIN_CONDITIONS = [{'w': 2, 'g': 2, 'b': 2}, {'w': 3}, {'g': 4}, {'b': 5}]

class ZertzGame:
    def __init__(self, rings=37, marbles=None, win_con=None, t=1, board_layout=None, clone=None, clone_state=None):
        if clone is not None:
            # Creates an instance of ZertzGame with settings copied from clone and updated to
            # have the same board state as clone_state
            self.initial_rings = clone.initial_rings
            self.t = clone.t
            self.marbles = copy.copy(clone.marbles)
            self.win_con = copy.copy(clone.win_con)
            self.board = ZertzBoard(clone=clone.board)
            self.board.state = np.copy(clone_state)
            self.board_layout = np.copy(clone.board_layout)
            self.move_history = list(clone.move_history)  # Copy history
            self.loop_detection_pairs = clone.loop_detection_pairs
            assert clone.board.state.shape[0] == clone_state.shape[0]
        else:
            # The size of the game board
            #   default: 37 rings (approximately 7x7 hex)
            self.initial_rings = rings
            self.t = t
            self.marbles = marbles
            self.board_layout = board_layout

            self.board = ZertzBoard(self.initial_rings, self.marbles, self.t, board_layout=self.board_layout)

            # The win conditions (amount of each marble needed)
            #   default:
            #     -3 marbles of each color
            #     -4 white marbles
            #     -5 gray marbles
            #     -6 black marbles
            if win_con is None:
                # Use the default win conditions
                self.win_con = [{'w': 3, 'g': 3, 'b': 3},
                                {'w': 4}, {'g': 5}, {'b': 6}]
            else:
                self.win_con = win_con

            # Move history for loop detection
            # Track (action_type, action) tuples; 'PASS' when player has no valid moves
            # Loop detection: if last 2 move-pairs == preceding 2 move-pairs → tie
            # Immobilization: if last 2 moves are both 'PASS' → game ends
            self.move_history = []
            self.loop_detection_pairs = 2  # Number of repeated pairs to trigger tie

    def __deepcopy__(self, memo):
        return ZertzGame(clone=self, clone_state=self.board.state)

    def reset_board(self):
        self.board = ZertzBoard(self.initial_rings, self.marbles, self.t)

    def get_cur_player_value(self):
        # Returns 1 if current player is player 0 and -1 if current player is player 1
        player_value = None
        if self.board.global_state[self.board.CUR_PLAYER] == self.board.PLAYER_1:
            player_value = 1
        elif self.board.global_state[self.board.CUR_PLAYER] == self.board.PLAYER_2:
            player_value = -1
        return player_value

    def _has_valid_moves(self):
        """Check if current player has any valid moves.

        Returns:
            bool: True if player has at least one valid action
        """
        placement, capture = self.get_valid_actions()
        return np.any(placement) or np.any(capture)

    def _has_move_loop(self):
        """Detect if last k move-pairs equal preceding k move-pairs.

        For k=2: Check if moves[-4:] (last 2 pairs) == moves[-8:-4] (preceding 2 pairs)

        Returns:
            bool: True if loop detected
        """
        k = self.loop_detection_pairs
        needed_moves = k * 4  # k pairs = 2k full turns = 4k moves

        if len(self.move_history) < needed_moves:
            return False

        # Get last k pairs (4k moves)
        last_k_pairs = self.move_history[-needed_moves // 2:]
        # Get preceding k pairs
        preceding_k_pairs = self.move_history[-needed_moves:-needed_moves // 2]

        return last_k_pairs == preceding_k_pairs

    def _both_players_immobilized(self):
        """Check if last 2 moves were both passes (both players immobilized).

        Returns:
            bool: True if both players consecutively passed
        """
        if len(self.move_history) < 2:
            return False

        return (self.move_history[-1] == ('PASS', None) and
                self.move_history[-2] == ('PASS', None))

    def get_current_state(self):
        """Returns complete observable game state for ML.

        Returns:
            dict: Complete state with keys:
                - 'spatial': (L, H, W) ndarray - rings, marbles, history, capture flag
                - 'global': (10,) ndarray - supply counts, captured counts, current player
                - 'player': int - 1 for Player 1, -1 for Player 2 (perspective value)
        """
        return {
            'spatial': np.copy(self.board.state),
            'global': np.copy(self.board.global_state),
            'player': self.get_cur_player_value()
        }

    def get_next_state(self, action, action_type, cur_state=None):
        """Apply action and return resulting game state.

        Args:
            action: Index into the action space matrix
            action_type: 'PUT' for placement or 'CAP' for capture
            cur_state: Optional spatial state array (L, H, W) to use instead of current state

        Returns:
            dict: Complete state after action (same format as get_current_state())
                - 'spatial': (L, H, W) ndarray
                - 'global': (10,) ndarray
                - 'player': int (1 or -1)
        """
        if cur_state is None:
            # Use the internal game state to determine the next state
            self.take_action(action_type, action)  # Records state history
            return self.get_current_state()
        else:
            # Return the next state for an arbitrary spatial state
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            return temp_game.get_next_state(action, action_type)

    def get_valid_actions(self, cur_state=None):
        # Returns two filtering matrices that can be used to filter and renormalize the policy
        # probability distributions. Capturing is compulsory so if there is a valid capture action
        # then the matrix of placement actions will all be False. Matrix shape depends on the action type.
        #   - for placement actions, shape is 3 x width^2 x (width^2 + 1)
        #   - for capture actions, shape is 6 x width x width
        #     - capture actions only end the current players turn if there are no more chain captures
        if cur_state is None:
            placement, capture = self.board.get_valid_moves()
        else:
            # Return the valid actions for an arbitrary marble supply, board and player
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            placement, capture = temp_game.get_valid_actions()
        return placement, capture

    def get_capture_action_size(self):
        # Return the number of possible capture actions
        return 6 * self.board.width ** 2

    def get_capture_action_shape(self):
        # Return the shape of the capture actions as a tuple
        return self.board.get_capture_shape()

    def get_placement_action_size(self):
        # Return the number of possible placement actions
        return 3 * self.board.width ** 2 * (self.board.width ** 2 + 1)

    def get_placement_action_shape(self):
        # Return the shape of the placement actions as a tuple
        return self.board.get_placement_shape()

    def _is_game_over(self):
        """Return True if game has ended.

        End conditions:
        1. Win by captured marbles (3 of each, or 4W/5G/6B)
        2. Board fully occupied
        3. Player has no marbles
        4. Both players immobilized (consecutive passes)
        5. Move loop detected (last 2 pairs == preceding 2 pairs)
        """
        marble_types = ['w', 'g', 'b']

        # Check if any player's captured marbles are enough to satisfy a win condition
        for win_con in self.win_con:
            # Build the list of required marble amounts
            required = np.zeros(3)
            for i, marble_type in enumerate(marble_types):
                if marble_type in win_con:
                    required[i] = win_con[marble_type]

            # Check player 1's captured marbles
            player1_captured = self.board.global_state[self.board.P1_CAP_SLICE]
            if np.all(player1_captured >= required):
                return True

            # Check player 2's captured marbles
            player2_captured = self.board.global_state[self.board.P2_CAP_SLICE]
            if np.all(player2_captured >= required):
                return True

        # If board has every ring covered with a marble then the last player who played is winner
        if np.all(np.sum(self.board.state[self.board.BOARD_LAYERS], axis=0) != 1):
            return True

        # Check if current player has no marbles available (pool + captured = 0 for all types)
        # If so, the opponent wins
        pool_marbles = self.board.global_state[self.board.SUPPLY_SLICE]

        # Get current player's captured marbles from global_state
        if self.board.get_cur_player() == self.board.PLAYER_1:
            captured_marbles = self.board.global_state[self.board.P1_CAP_SLICE]
        else:
            captured_marbles = self.board.global_state[self.board.P2_CAP_SLICE]

        # If player has no marbles in pool or captured, they lose
        if np.all(pool_marbles + captured_marbles == 0):
            return True

        # Check for both players immobilized (consecutive passes)
        if self._both_players_immobilized():
            return True

        # Check for move loop (last 2 pairs == preceding 2 pairs)
        if self._has_move_loop():
            return True

        return False

    def get_game_ended(self, cur_state=None):
        """Returns outcome of the game.

        Returns:
            PLAYER_1_WIN (1): Player 1 won
            PLAYER_2_WIN (-1): Player 2 won
            TIE (0): Tie
            None: Game not over
        """
        if cur_state is None:
            if not self._is_game_over():
                return None

            # Check for tie conditions first
            if self._has_move_loop():
                # Move loop detected → tie
                return TIE

            if self._both_players_immobilized():
                # Both players immobilized → determine winner by captured marbles
                p1_captured = self.board.global_state[self.board.P1_CAP_SLICE]
                p2_captured = self.board.global_state[self.board.P2_CAP_SLICE]

                # Check each win condition
                for win_con in self.win_con:
                    required = np.zeros(3)
                    marble_types = ['w', 'g', 'b']
                    for i, marble_type in enumerate(marble_types):
                        if marble_type in win_con:
                            required[i] = win_con[marble_type]

                    if np.all(p1_captured >= required):
                        return PLAYER_1_WIN
                    if np.all(p2_captured >= required):
                        return PLAYER_2_WIN

                # Neither player met win condition → tie
                return TIE

            # Standard win conditions
            # The winner is the player that made the previous action
            if np.sum(self.board.state[self.board.CAPTURE_LAYER]) == 0:
                return PLAYER_2_WIN if self.get_cur_player_value() == 1 else PLAYER_1_WIN
            else:
                # The game is over in the middle of the players turn during a chain capture
                # if they have enough marbles to meet a win condition.
                return PLAYER_1_WIN if self.get_cur_player_value() == 1 else PLAYER_2_WIN
        else:
            # Return if game is ended for an arbitrary game state
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            return temp_game.get_game_ended()

    def get_symmetries(self, cur_state=None):
        # There are many symmetries in Zertz
        # First, there are rotational symmetry in that every board position can be rotated in
        # six different ways
        # Second, there are mirror symmetry with every rotation being able to be flipped
        # Third, there are translational symmetries once the board has gotten small enough that 
        # it can be shifted in one of the six directions and still be able to fit within the
        # original space.
        # Total board symmetries = 6 * 2 * (# of shift symmetries)
        # Total implemented currently = 4
        if cur_state is None:
            symmetries = self.board.get_state_symmetries()
        else:
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            symmetries = temp_game.get_symmetries()
        return symmetries

    def translate_action_symmetry(self, action_type, symmetry, actions):
        translated = np.copy(actions)
        if len(translated.shape) != 3:
            if action_type == 'PUT':
                translated = translated.reshape(self.get_placement_action_shape())
            elif action_type == 'CAP':
                translated = translated.reshape(self.get_capture_action_shape())
        if symmetry == 0:  # mirror
            translated = self.board.mirror_action(action_type, translated)
        elif symmetry == 1:  # rotated
            translated = self.board.rotate_action(action_type, translated)
        elif symmetry == 2:  # mirror/rotated
            translated = self.board.mirror_action(action_type, translated)
            translated = self.board.rotate_action(action_type, translated)
        elif symmetry == 3:  # opponent
            pass
        elif symmetry == 4:  # opponent mirror
            pass
        elif symmetry == 5:  # opponent rotated
            pass
        elif symmetry == 6:  # opponent mirror/rotated
            pass
        return translated

    def str_to_action(self, action_str):
        # Translate an action string [i.e. 'PUT w A1 B2' or 'CAP b C4 g C2'] to a tuple/type
        args = action_str.split()
        action_type = args[0]
        if action_type == 'PUT':
            if len(args) == 4:
                marble_type, put_str, rem_str = args[1:]
            elif len(args) == 3:
                marble_type, put_str = args[1:]
                rem_str = None
            else:
                return '', None
            layer = self.board.MARBLE_TO_LAYER[marble_type] - 1
            y, x = self.board.str_to_index(put_str)
            put = y * self.board.width + x
            if rem_str is not None:
                y, x = self.board.str_to_index(rem_str)
                rem = y * self.board.width + x
            else:
                rem = self.board.width ** 2
            action = (layer, put, rem)
        elif action_type == 'CAP':
            if len(args) == 5:
                _a, src_str, _b, dst_str = args[1:]
            else:
                return '', None
            src = self.board.str_to_index(src_str)
            dst = self.board.str_to_index(dst_str)
            cap = self.board.get_middle_ring(src, dst)
            neighbors = self.board.get_neighbors(src)
            direction = neighbors.index(cap)
            action = (direction, src[0], src[1])
        else:
            action = None
        return action_type, action

    def action_to_str(self, action_type, action):
        # Translate an action tuple and type to a human readable string representation
        action_str = action_type + ' '
        action_dict = {}

        if action_type == 'PASS':
            # Player has no valid moves and must pass
            action_str = 'PASS'
            action_dict = {'action': 'PASS'}
            return action_str, action_dict

        if action_type == 'PUT':
            marble_type, put, rem = action
            marble_type = self.board.LAYER_TO_MARBLE[marble_type + 1]

            put_index = (put // self.board.width, put % self.board.width)
            put_str = self.board.index_to_str(put_index)

            if rem == self.board.width ** 2:
                rem_str = ''
            else:
                rem_index = rem // self.board.width, rem % self.board.width
                rem_str = self.board.index_to_str(rem_index)
            action_str = "{} {} {} {}".format(
                action_type, marble_type, put_str, rem_str).rstrip()
            action_dict = {
                'action': action_type,
                'marble': marble_type,
                'dst': str(put_str),
                'remove': str(rem_str)
            }

        elif action_type == 'CAP':
            direction, y, x = action
            src = (y, x)
            src_marble = self.board.get_marble_type_at(src)
            src_str = self.board.index_to_str(src)

            dy, dx = self.board.DIRECTIONS[direction]
            cap = (y + dy, x + dx)
            cap_marble = self.board.get_marble_type_at(cap)
            cap_str = self.board.index_to_str(cap)
            dst = self.board.get_jump_destination(src, cap)
            dst_str = self.board.index_to_str(dst)

            action_str = "{} {} {} {} {}".format(
                action_type, src_marble, src_str, cap_marble, dst_str)
            action_dict = {
                'action': action_type,
                'marble': src_marble,
                'src': str(src_str),
                'dst': str(dst_str),
                'capture': cap_marble,
                'cap': str(cap_str)
            }
        return action_str, action_dict

    def action_to_notation(self, action_dict, action_result=None):
        """Convert action_dict to official Zèrtz notation.

        Notation format from http://www.gipf.com/zertz/notations/notation.html:
        - Placement: [Color][coord] or [Color][coord],[removed_coord]
          Examples: "Wd4" or "Bd7,b2"
        - Placement with isolation: [Color][coord],[removed_coord] x [Color][pos]...
          Example: "Bd7,b2 x Wa1Wa2"
        - Capture: x [src][captured_color][dst]
          Example: "x e3Wg3"
        - Pass: "-"

        Args:
            action_dict: Dictionary with action details
            action_result: Optional ActionResult object from take_action()

        Returns:
            str: Notation string
        """
        if action_dict['action'] == 'PASS':
            return '-'

        if action_dict['action'] == 'PUT':
            # Convert marble color to uppercase
            marble = action_dict['marble'].upper()
            # Convert destination to lowercase
            dst = action_dict['dst'].lower()

            # Check if a ring was removed
            if action_dict['remove']:
                remove = action_dict['remove'].lower()
                notation = f"{marble}{dst},{remove}"
            else:
                notation = f"{marble}{dst}"

            # Add isolation captures if any
            if action_result and action_result.is_isolation():
                isolated_parts = []
                for removal in action_result.captured_marbles:
                    if removal['marble']:
                        color = removal['marble'].upper()
                        pos = removal['pos'].lower()
                        isolated_parts.append(f"{color}{pos}")
                if isolated_parts:
                    notation += " x " + "".join(isolated_parts)

            return notation

        elif action_dict['action'] == 'CAP':
            # Convert to lowercase and get captured marble as uppercase
            src = action_dict['src'].lower()
            dst = action_dict['dst'].lower()
            captured = action_dict['capture'].upper()
            return f"x {src}{captured}{dst}"

        return ''

    def get_placement_positions(self, placement_array):
        """Convert placement array to list of position strings.

        Args:
            placement_array: Placement array from get_valid_actions() - shape (3, width², width²+1)

        Returns:
            List of position strings for valid placement destinations
        """
        # Find valid destinations by collapsing marble type and removal dimensions
        valid_dests = np.any(placement_array, axis=(0, 2))
        dest_indices = np.argwhere(valid_dests).flatten()

        placement_positions = []
        for dst_idx in dest_indices:
            dst_y = dst_idx // self.board.width
            dst_x = dst_idx % self.board.width
            pos_str = self.board.index_to_str((dst_y, dst_x))
            if pos_str:
                placement_positions.append(pos_str)

        return placement_positions

    def get_capture_dicts(self, capture_array):
        """Convert capture array to list of capture move dicts.

        Args:
            capture_array: Capture array from get_valid_actions() - shape (6, width, width)

        Returns:
            List of dicts with {action, marble, src, dst, capture, cap} for each valid capture
        """
        capture_positions = np.argwhere(capture_array)
        capture_moves = []

        for direction, src_y, src_x in capture_positions:
            try:
                _, action_dict = self.action_to_str("CAP", (direction, src_y, src_x))
                capture_moves.append(action_dict)
            except (IndexError, KeyError):
                continue

        return capture_moves

    def get_removal_positions(self, placement_array, action_type, action):
        """Get list of removal positions for a specific PUT action.

        Args:
            placement_array: Placement array from get_valid_actions()
            action_type: Action type (must be "PUT")
            action: Action tuple (marble_idx, dst, rem)

        Returns:
            List of removable position strings
            Returns empty list if action_type is not "PUT"
        """
        if action_type != "PUT":
            return []

        marble_idx, dst, rem = action
        width = self.board.width

        # Get the removal dimension for this specific (marble, destination) pair
        removal_mask = placement_array[marble_idx, dst, :]
        removable_indices = np.argwhere(removal_mask).flatten()

        removable_positions = []
        for rem_idx in removable_indices:
            # Skip the "no removal" option (width²) and the destination itself
            if rem_idx != width ** 2 and rem_idx != dst:
                rem_y = rem_idx // width
                rem_x = rem_idx % width
                rem_str = self.board.index_to_str((rem_y, rem_x))
                if rem_str:
                    removable_positions.append(rem_str)

        return removable_positions

    def print_state(self):
        # Print the board state and supplies to the console
        #   0 - empty space
        #   1 - ring
        #   2 - white marble
        #   3 - gray marble
        #   4 - black marble
        print("---------------")
        print("Board state:")
        print(self.board.state[self.board.RING_LAYER] +
              self.board.state[self.board.MARBLE_TO_LAYER['w']] +
              self.board.state[self.board.MARBLE_TO_LAYER['g']] * 2 +
              self.board.state[self.board.MARBLE_TO_LAYER['b']] * 3)
        print("---------------")
        print("Marble supply:")
        print(self.board.global_state[self.board.SUPPLY_SLICE])
        print("---------------")

    def take_action(self, action_type, action):
        """Execute an action and record move for loop detection.

        Args:
            action_type: 'PUT', 'CAP', or 'PASS'
            action: Action tuple (or None for PASS)

        Returns:
            ActionResult: Encapsulates captured marbles and newly frozen positions
        """
        # Record the move for loop detection
        self.move_history.append((action_type, action))

        if action_type == 'PASS':
            # Player passes (no valid moves), switch player
            self.board._next_player()
            return ActionResult(captured_marbles=None, newly_frozen_positions=set())
        else:
            # Capture frozen positions BEFORE action (Recommendation 1)
            frozen_before = set(self.board.frozen_positions)

            # Execute the action
            captured = self.board.take_action(action, action_type)

            # Get NEWLY frozen positions after action
            frozen_after = set(self.board.frozen_positions)
            newly_frozen = frozen_after - frozen_before

            # Convert positions to strings
            frozen_position_strs = {
                self.board.index_to_str(pos)
                for pos in newly_frozen
                if self.board.index_to_str(pos)
            }

            return ActionResult(captured_marbles=captured, newly_frozen_positions=frozen_position_strs)
