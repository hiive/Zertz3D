import copy
import numpy as np

from .zertz_board import ZertzBoard
from .action_result import ActionResult
from .formatters import NotationFormatter
from .constants import (
    PLAYER_1_WIN,
    PLAYER_2_WIN,
    TIE,
    BOTH_LOSE,
)
from shared.render_data import RenderData
from shared.constants import MARBLE_TYPES


# For full rules: http://www.gipf.com/zertz/rules/rules.html
# Class interface inspired by https://github.com/suragnair/alpha-zero-general


class ZertzGame:
    def __init__(
        self,
        rings=37,
        marbles=None,
        win_con=None,
        t=1,
        clone=None,
        clone_state=None,
    ):
        if clone is not None:
            # Creates an instance of ZertzGame with settings copied from clone and updated to
            # have the same board state as clone_state
            self.initial_rings = clone.initial_rings
            self.t = clone.t
            self.marbles = copy.copy(clone.marbles)
            self.win_con = copy.copy(clone.win_con)
            self.board = ZertzBoard(clone=clone.board)
            self.board.state = np.copy(clone_state)
            self.move_history = list(clone.move_history)  # Copy history
            self.loop_detection_pairs = clone.loop_detection_pairs
            assert clone.board.state.shape[0] == clone_state.shape[0]
        else:
            # The size of the game board
            #   default: 37 rings (approximately 7x7 hex)
            self.initial_rings = rings
            self.t = t
            self.marbles = marbles

            self.board = ZertzBoard(
                self.initial_rings, self.marbles, self.t
            )

            # The win conditions (amount of each marble needed)
            #   default:
            #     -3 marbles of each color
            #     -4 white marbles
            #     -5 gray marbles
            #     -6 black marbles
            if win_con is None:
                # Use the default win conditions
                self.win_con = [{"w": 3, "g": 3, "b": 3}, {"w": 4}, {"g": 5}, {"b": 6}]
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

    def _normalize_move_for_loop_detection(self, move):
        """Normalize a move for position-based loop detection (ignore marble colors).

        Args:
            move: (action_type, action) tuple from move_history

        Returns:
            Normalized move tuple with marble colors removed
        """
        action_type, action = move

        if action_type == "PASS":
            return ("PASS", None)
        elif action_type == "PUT":
            # PUT action: (marble_type, put_position, remove_position)
            # Normalize by removing marble_type (set to 0)
            _, put_pos, rem_pos = action
            return ("PUT", (None, put_pos, rem_pos))
        elif action_type == "CAP":
            # CAP action: (direction, src_y, src_x)
            # Already position-only, no normalization needed
            return move
        else:
            return move

    def _has_move_loop(self):
        """Detect if last k move-pairs equal preceding k move-pairs.

        For k=2: Check if moves[-4:] (last 2 pairs) == moves[-8:-4] (preceding 2 pairs)
        Comparison is position-based (marble colors are ignored).

        Returns:
            bool: True if loop detected
        """
        k = self.loop_detection_pairs
        needed_moves = k * 4  # k pairs = 2k full turns = 4k moves

        if len(self.move_history) < needed_moves:
            return False

        # Get last k pairs (4k moves) and normalize
        last_k_pairs = [
            self._normalize_move_for_loop_detection(move)
            for move in self.move_history[-needed_moves // 2 :]
        ]
        # Get preceding k pairs and normalize
        preceding_k_pairs = [
            self._normalize_move_for_loop_detection(move)
            for move in self.move_history[-needed_moves : -needed_moves // 2]
        ]

        return last_k_pairs == preceding_k_pairs

    def _both_players_immobilized(self):
        """Check if last 2 moves were both passes (both players immobilized).

        Returns:
            bool: True if both players consecutively passed
        """
        if len(self.move_history) < 2:
            return False

        return self.move_history[-1] == ("PASS", None) and self.move_history[-2] == (
            "PASS",
            None,
        )

    def get_current_state(self):
        """Returns complete observable game state for ML.

        Returns:
            dict: Complete state with keys:
                - 'spatial': (L, H, W) ndarray - rings, marbles, history, capture flag
                - 'global': (10,) ndarray - supply counts, captured counts, current player
                - 'player': int - 1 for Player 1, -1 for Player 2 (perspective value)
        """
        return {
            "spatial": np.copy(self.board.state),
            "global": np.copy(self.board.global_state),
            "player": self.get_cur_player_value(),
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
        return 6 * self.board.config.width**2

    def get_capture_action_shape(self):
        # Return the shape of the capture actions as a tuple
        return self.board.get_capture_shape()

    def get_placement_action_size(self):
        # Return the number of possible placement actions
        return 3 * self.board.config.width**2 * (self.board.config.width**2 + 1)

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
        # Check if any player's captured marbles are enough to satisfy a win condition
        for win_con in self.win_con:
            # Build the list of required marble amounts
            required = np.zeros(3)
            for i, marble_type in enumerate(MARBLE_TYPES):
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
            BOTH_LOSE (-2): Both players lose (collaboration detected)
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
                    for i, marble_type in enumerate(MARBLE_TYPES):
                        if marble_type in win_con:
                            required[i] = win_con[marble_type]

                    if np.all(p1_captured >= required):
                        return PLAYER_1_WIN
                    if np.all(p2_captured >= required):
                        return PLAYER_2_WIN

                # Neither player met win condition → tie
                return TIE

            # Check for full board with no captures (collaboration rule)
            if np.all(np.sum(self.board.state[self.board.BOARD_LAYERS], axis=0) != 1):
                # Board is full - check if either player has captured any marble
                p1_captured = self.board.global_state[self.board.P1_CAP_SLICE]
                p2_captured = self.board.global_state[self.board.P2_CAP_SLICE]

                # If NEITHER player has captured ANY marble, both lose (collaboration detected)
                if np.all(p1_captured == 0) and np.all(p2_captured == 0):
                    return BOTH_LOSE

                # Normal full board win - last player wins
                if self.board.last_acting_player == self.board.PLAYER_1:
                    return PLAYER_1_WIN
                else:
                    return PLAYER_2_WIN

            # Standard win conditions
            # The winner is the player that made the previous action
            # Use last_acting_player which was captured before any player switches
            if self.board.last_acting_player == self.board.PLAYER_1:
                return PLAYER_1_WIN
            else:
                return PLAYER_2_WIN
        else:
            # Return if game is ended for an arbitrary game state
            temp_game = ZertzGame(clone=self, clone_state=cur_state)
            return temp_game.get_game_ended()

    def get_game_end_reason(self):
        """Returns detailed reason for why the game ended.

        Returns:
            str: Human-readable description of why the game ended
            None: Game not over
        """
        if not self._is_game_over():
            return None

        # Check for tie conditions first
        if self._has_move_loop():
            return "Move loop detected (repeated position)"

        if self._both_players_immobilized():
            # Both players immobilized → determine winner by captured marbles
            p1_captured = self.board.global_state[self.board.P1_CAP_SLICE]
            p2_captured = self.board.global_state[self.board.P2_CAP_SLICE]

            # Check each win condition
            for win_con in self.win_con:
                required = np.zeros(3)
                for i, marble_type in enumerate(MARBLE_TYPES):
                    if marble_type in win_con:
                        required[i] = win_con[marble_type]

                if np.all(p1_captured >= required):
                    return self._format_win_condition(p1_captured, "Both players immobilized")
                if np.all(p2_captured >= required):
                    return self._format_win_condition(p2_captured, "Both players immobilized")

            # Neither player met win condition → tie
            return "Both players immobilized with no winner"

        # Check if board is full
        if np.all(np.sum(self.board.state[self.board.BOARD_LAYERS], axis=0) != 1):
            # Board is full - check collaboration rule
            p1_captured = self.board.global_state[self.board.P1_CAP_SLICE]
            p2_captured = self.board.global_state[self.board.P2_CAP_SLICE]

            if np.all(p1_captured == 0) and np.all(p2_captured == 0):
                return "Both players lose: Board filled with no captures (collaboration detected)"

            return "Board completely filled with marbles"

        # Check if current player has no marbles
        pool_marbles = self.board.global_state[self.board.SUPPLY_SLICE]
        if self.board.get_cur_player() == self.board.PLAYER_1:
            captured_marbles = self.board.global_state[self.board.P1_CAP_SLICE]
        else:
            captured_marbles = self.board.global_state[self.board.P2_CAP_SLICE]

        if np.all(pool_marbles + captured_marbles == 0):
            return "Opponent has no marbles left to place"

        # Standard win by captured marbles
        p1_captured = self.board.global_state[self.board.P1_CAP_SLICE]
        p2_captured = self.board.global_state[self.board.P2_CAP_SLICE]

        for win_con in self.win_con:
            required = np.zeros(3)
            for i, marble_type in enumerate(MARBLE_TYPES):
                if marble_type in win_con:
                    required[i] = win_con[marble_type]

            if np.all(p1_captured >= required):
                return self._format_win_condition(p1_captured, "Captured required marbles")
            if np.all(p2_captured >= required):
                return self._format_win_condition(p2_captured, "Captured required marbles")

        return "Game ended"

    def _format_win_condition(self, captured_marbles, prefix=""):
        """Format a win condition message based on captured marbles.

        Args:
            captured_marbles: Array of captured marble counts [w, g, b]
            prefix: Optional prefix for the message

        Returns:
            str: Formatted message like "Captured 4 white" or "Captured 3 of each color"
        """
        w, g, b = int(captured_marbles[0]), int(captured_marbles[1]), int(captured_marbles[2])

        # Check which win condition was met
        for win_con in self.win_con:
            required = np.zeros(3)
            for i, marble_type in enumerate(MARBLE_TYPES):
                if marble_type in win_con:
                    required[i] = win_con[marble_type]

            if np.all(captured_marbles >= required):
                # Format the specific condition
                if len(win_con) == 3:
                    # 3 of each
                    msg = f"{int(required[0])} of each color"
                elif "w" in win_con:
                    msg = f"{int(required[0])} white"
                elif "g" in win_con:
                    msg = f"{int(required[1])} gray"
                elif "b" in win_con:
                    msg = f"{int(required[2])} black"
                else:
                    msg = f"w={w}, g={g}, b={b}"

                if prefix:
                    return f"{prefix}: {msg}"
                return msg

        # Fallback
        return f"Captured w={w}, g={g}, b={b}"


    def str_to_action(self, action_str):
        # Translate an action string [i.e. 'PUT w A1 B2' or 'CAP b C4 g C2'] to a tuple/type
        args = action_str.split()
        action_type = args[0]
        if action_type == "PUT":
            if len(args) == 4:
                marble_type, put_str, rem_str = args[1:]
            elif len(args) == 3:
                marble_type, put_str = args[1:]
                rem_str = None
            else:
                return "", None
            from hiivelabs_mcts import algebraic_to_coordinate
            layer = self.board.MARBLE_TO_LAYER[marble_type] - 1
            y, x = algebraic_to_coordinate(put_str, self.board.config)
            put = y * self.board.config.width + x
            if rem_str is not None:
                y, x = algebraic_to_coordinate(rem_str, self.board.config)
                rem = y * self.board.config.width + x
            else:
                rem = self.board.config.width**2
            action = (layer, put, rem)
        elif action_type == "CAP":
            from hiivelabs_mcts import algebraic_to_coordinate
            if len(args) == 4:
                src_str, _b, dst_str = args[1:]
            else:
                return "", None
            src = algebraic_to_coordinate(src_str, self.board.config)
            dst = algebraic_to_coordinate(dst_str, self.board.config)

            # Convert to internal format: (None, src_flat, dst_flat)
            src_flat = src[0] * self.board.config.width + src[1]
            dst_flat = dst[0] * self.board.config.width + dst[1]
            action = (None, src_flat, dst_flat)
        else:
            action = None
        return action_type, action

    def action_to_str(self, action_type, action):
        # Translate an action tuple and type to a human readable string representation
        action_str = action_type + " "
        action_dict = {}

        if action_type == "PASS":
            # Player has no valid moves and must pass
            action_str = "PASS"
            action_dict = {"action": "PASS"}
            return action_str, action_dict

        if action_type == "PUT":
            from hiivelabs_mcts import coordinate_to_algebraic
            marble_type, put, rem = action
            marble_type = self.board.LAYER_TO_MARBLE[marble_type + 1]
            width = self.board.config.width
            put_y, put_x = divmod(put, width)
            put_str = coordinate_to_algebraic(put_y, put_x, self.board.config)

            if rem == self.board.config.width**2:
                rem_str = ""
            else:
                rem_y, rem_x = divmod(rem, width)
                rem_str = coordinate_to_algebraic(rem_y, rem_x, self.board.config)
            action_str = "{} {} {} {}".format(
                action_type, marble_type, put_str, rem_str
            ).rstrip()
            action_dict = {
                "action": action_type,
                "marble": marble_type,
                "dst": str(put_str),
                "remove": str(rem_str),
            }

        elif action_type == "CAP":
            # New format: (None, src_flat, dst_flat)
            _, src_flat, dst_flat = action
            src_y, src_x = divmod(src_flat, self.board.config.width)
            dst_y, dst_x = divmod(dst_flat, self.board.config.width)

            # Calculate captured marble position (midpoint)
            cap_y = (src_y + dst_y) // 2
            cap_x = (src_x + dst_x) // 2
            cap = (cap_y, cap_x)
            cap_marble = self.board.get_marble_type_at(cap)

            # Use Rust functions for coordinate conversion
            import hiivelabs_mcts
            src_str = hiivelabs_mcts.coordinate_to_algebraic(src_y, src_x, self.board.config)
            cap_str = hiivelabs_mcts.coordinate_to_algebraic(cap_y, cap_x, self.board.config)
            dst_str = hiivelabs_mcts.coordinate_to_algebraic(dst_y, dst_x, self.board.config)

            action_str = "{} {} {} {}".format(
                action_type, src_str, cap_marble, dst_str
            )
            action_dict = {
                "action": action_type,
                "src": str(src_str),
                "dst": str(dst_str),
                "capture": cap_marble,
                "cap": str(cap_str),
            }
        return action_str, action_dict

    def action_to_notation(self, action_dict, action_result=None):
        """Convert action_dict to official Zèrtz notation.

        Delegates to NotationFormatter for conversion.

        Args:
            action_dict: Dictionary with action details
            action_result: Optional ActionResult object from take_action()

        Returns:
            str: Notation string
        """
        return NotationFormatter.action_to_notation(action_dict, action_result)

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

        from hiivelabs_mcts import coordinate_to_algebraic
        placement_positions = []
        for dst_idx in dest_indices:
            dst_y, dst_x = divmod(dst_idx, self.board.config.width)
            if self.board.state[self.board.RING_LAYER, dst_y, dst_x]:
                pos_label = coordinate_to_algebraic(dst_y, dst_x, self.board.config)
                placement_positions.append(pos_label)

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
                # Convert from capture mask indices to action format using helper
                from game.zertz_board import ZertzBoard
                action = ZertzBoard.capture_indices_to_action(
                    direction, src_y, src_x, self.board.config.width, self.board.DIRECTIONS
                )
                _, action_dict = self.action_to_str("CAP", action)
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
        width = self.board.config.width

        # Get the removal dimension for this specific (marble, destination) pair
        removal_mask = placement_array[marble_idx, dst, :]
        removable_indices = np.argwhere(removal_mask).flatten()

        from hiivelabs_mcts import coordinate_to_algebraic
        removable_positions = []
        for rem_idx in removable_indices:
            # Skip the "no removal" option (width²) and the destination itself
            if rem_idx != width**2 and rem_idx != dst:
                rem_y, rem_x = divmod(rem_idx, width)
                if self.board.state[self.board.RING_LAYER, rem_y, rem_x]:
                    rem_label = coordinate_to_algebraic(rem_y, rem_x, self.board.config)
                    removable_positions.append(rem_label)

        return removable_positions

    def _enrich_placements_with_scores(self, placement_positions, placement_array, action_scores):
        """Enrich placement position list with scores from action_scores dict.

        Args:
            placement_positions: List of position strings ['A1', 'B2', ...]
            placement_array: Placement array (3, width², width²+1) for action tuple reconstruction
            action_scores: Dict mapping action tuples to scores {('PUT', (0, 5, 37)): 0.8, ...}

        Returns:
            List of dicts with position and score: [{'pos': 'A1', 'score': 0.8}, ...]
        """
        enriched = []
        width = self.board.config.width

        from hiivelabs_mcts import algebraic_to_coordinate
        for pos_str in placement_positions:
            # Convert position string to y, x
            y, x = algebraic_to_coordinate(pos_str, self.board.config)
            dst_flat = y * width + x

            # Find the best score for any marble type and removal at this destination
            # Valid actions not explored by MCTS get score 0.0
            best_score = 0.0
            for marble_type in range(3):  # w=0, g=1, b=2
                for rem_flat in range(width * width + 1):  # All removals + no-removal
                    if placement_array[marble_type, dst_flat, rem_flat]:
                        action_key = ('PUT', (marble_type, dst_flat, rem_flat))
                        score = action_scores.get(action_key, 0.0)  # Unexplored actions get 0.0
                        best_score = max(best_score, score)

            enriched.append({'pos': pos_str, 'score': best_score})

        return enriched

    def _enrich_captures_with_scores(self, capture_moves, action_scores):
        """Enrich capture move dicts with scores from action_scores dict.

        Args:
            capture_moves: List of capture dicts [{'action': 'CAP', 'src': 'C4', ...}, ...]
            action_scores: Dict mapping action tuples to scores {('CAP', (2, 3, 4)): 0.9, ...}

        Returns:
            List of dicts with score added: [{'action': 'CAP', ..., 'score': 0.9}, ...]
        """
        enriched = []
        width = self.board.config.width

        from hiivelabs_mcts import algebraic_to_coordinate
        for cap_dict in capture_moves:
            # Reconstruct action tuple from dict (new format: (None, src_flat, dst_flat))
            src_y, src_x = algebraic_to_coordinate(cap_dict['src'], self.board.config)
            dst_y, dst_x = algebraic_to_coordinate(cap_dict['dst'], self.board.config)

            src_flat = src_y * width + src_x
            dst_flat = dst_y * width + dst_x

            # Create action key in new format
            action_key = ('CAP', (None, src_flat, dst_flat))
            score = action_scores.get(action_key, 0.0)  # Unexplored actions get 0.0

            # Add score to dict
            enriched_dict = dict(cap_dict)  # Copy
            enriched_dict['score'] = score
            enriched.append(enriched_dict)

        return enriched

    def _enrich_removals_with_scores(self, removal_positions, placement_array, action_type, action, action_scores):
        """Enrich removal position list with scores from action_scores dict.

        Args:
            removal_positions: List of position strings ['A1', 'B2', ...]
            placement_array: Placement array for action tuple reconstruction
            action_type: Action type (should be 'PUT')
            action: Action tuple (marble_idx, dst, rem)
            action_scores: Dict mapping action tuples to scores

        Returns:
            List of dicts with position and score: [{'pos': 'A1', 'score': 0.7}, ...]
        """
        if action_type != 'PUT':
            return []

        enriched = []
        marble_idx, dst_flat, _ = action
        width = self.board.config.width

        from hiivelabs_mcts import algebraic_to_coordinate
        for pos_str in removal_positions:
            # Convert position string to flat index
            y, x = algebraic_to_coordinate(pos_str, self.board.config)
            rem_flat = y * width + x

            # Lookup score for this specific (marble, dst, removal) tuple
            action_key = ('PUT', (marble_idx, dst_flat, rem_flat))
            score = action_scores.get(action_key, 0.0)  # Unexplored actions get 0.0

            enriched.append({'pos': pos_str, 'score': score})

        return enriched

    def get_render_data(self, action_type, action, highlight_choices=None, action_scores=None):
        """Get all rendering data for an action in one call.

        This method encapsulates all data transformations needed by the renderer,
        eliminating the need for the controller to perform transformations or
        check renderer state.

        Args:
            action_type: Action type ('PUT', 'CAP', or 'PASS')
            action: Action tuple (or None for PASS)
            highlight_choices: Highlight mode - None (no highlights), 'uniform', or 'heatmap'
            action_scores: Optional dict mapping action tuples to scores [0.0, 1.0]
                Format: {('PUT', (0, 5, 37)): 0.8, ('CAP', (2, 3, 4)): 1.0, ...}
                Required for 'heatmap' mode, optional for 'uniform' mode

        Returns:
            RenderData: Value object containing action_dict and optional highlight data

        Architecture: Part of Recommendation 1 from architecture_report4.md
        """
        # Get the action dictionary
        _, action_dict = self.action_to_str(action_type, action)

        # If highlights not requested, return minimal data
        if highlight_choices is None:
            return RenderData(action_dict)

        # Get valid actions for highlighting (before the action is executed)
        placement_array, capture_array = self.get_valid_actions()

        # Convert arrays to renderer-friendly formats
        placement_positions = self.get_placement_positions(placement_array)
        capture_moves = self.get_capture_dicts(capture_array)
        removal_positions = self.get_removal_positions(
            placement_array, action_type, action
        )

        # Enrich with scores if provided
        if action_scores:
            placement_positions = self._enrich_placements_with_scores(
                placement_positions, placement_array, action_scores
            )
            capture_moves = self._enrich_captures_with_scores(
                capture_moves, action_scores
            )
            removal_positions = self._enrich_removals_with_scores(
                removal_positions, placement_array, action_type, action, action_scores
            )

        return RenderData(
            action_dict=action_dict,
            placement_positions=placement_positions,
            capture_moves=capture_moves,
            removal_positions=removal_positions,
        )

    def print_state(self, reporter=None):
        """Emit the board state and supplies via provided reporter."""
        if reporter is None:
            reporter = print

        reporter("---------------")
        reporter("Board state:")
        reporter(
            self.board.state[self.board.RING_LAYER]
            + self.board.state[self.board.MARBLE_TO_LAYER["w"]]
            + self.board.state[self.board.MARBLE_TO_LAYER["g"]] * 2
            + self.board.state[self.board.MARBLE_TO_LAYER["b"]] * 3
        )
        reporter("---------------")
        reporter("Marble supply:")
        reporter(self.board.global_state[self.board.SUPPLY_SLICE])
        reporter("---------------")

    def take_action(self, action_type, action):
        """Execute an action and record move for loop detection.

        Args:
            action_type: 'PUT', 'CAP', or 'PASS'
            action: Action tuple (or None for PASS)

        Returns:
            ActionResult: Encapsulates captured marbles
        """
        # Record the move for loop detection
        self.move_history.append((action_type, action))

        if action_type == "PASS":
            # Player passes (no valid moves), switch player
            self.board._next_player()
            return ActionResult(captured_marbles=None)
        else:
            # Execute the action
            captured = self.board.take_action(action, action_type)
            return ActionResult(captured_marbles=captured)
