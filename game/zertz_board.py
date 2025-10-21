from collections import deque
import numpy as np

from game.zertz_position import ZertzPosition, ZertzPositionCollection
from game.utils.canonicalization import TransformFlags, CanonicalizationManager
from game import stateless_logic


class ZertzBoard:
    # The zertz board is a hexagon and looks like this:
    #   A 2D array where each location is a ring
    #   Each ring is adjacent to the rings below, left, above/left, above, right, and down/right
    #
    # 37-ring board
    #  A4 B5 C6 D7
    #  A3 B4 C5 D6 E6
    #  A2 B3 C4 D5 E5 F5
    #  A1 B2 C3 D4 E4 F4 G4
    #     B1 C2 D3 E3 F3 G3
    #        C1 D2 E2 F2 G2
    #           D1 E1 F1 G1
    #
    # 48-ring board
    #  A5 B6 C7 D8
    #  A4 B5 C6 D7 E7
    #  A3 B4 C5 D6 E6 F6
    #  A2 B3 C4 D5 E5 F5 G5
    #  A1 B2 C3 D4 E4 F4 G4 H4
    #     B1 C2 D3 E3 F3 G3 H3
    #        C1 D2 E2 F2 G2 H2
    #           D1 E1 F1 G1 H1
    #
    # 61-ring board
    #
    #  A5 B6 C7 D8 E8
    #  A4 B5 C6 D7 E7 F7
    #  A3 B4 C5 D6 E6 F6 G6
    #  A2 B3 C4 D5 E5 F5 G5 H4
    #  A1 B2 C3 D4 E4 F4 G4 H4 J4
    #     B1 C2 D3 E3 F3 G3 H3 J3
    #        C1 D2 E2 F2 G2 H2 J2
    #           D1 E1 F1 G1 H1 J1
    #              E0 F0 G0 H0 J0

    # A placement action is a tuple in the form:
    #   ((placement, marble_type, ring), (remove, ring))
    #   i.e. (('PUT', 'b', 'E6'), ('REM', 'A2'))
    #     - this puts a marble at E6 and removes the ring A2
    #
    # Capture action is a tuple in the form:
    #   ((capture, marble_type, start_ring), (marble_type, dst_ring), (marble_type, dst_ring), etc.)
    #   - here, each marble after the first is a marble being captured
    #   - the final location of the marble doing the capturing is the final dst_ring
    #   - the marble being captured is the one on the ring between the start_ring and dst_ring
    #   i.e. (('CAP', 'b', 'A3'), ('b', 'C5'))
    #        or (('CAP', 'g', 'D6'), ('w', 'D4'), ('w', 'B2'))
    #     - this action uses the marble at D6 to capture the marbles at D5 and C3 before ending at B2
    ACTION_VERBS = ["PUT", "REM", "CAP"]

    # Board size constants
    SMALL_BOARD_37 = 37
    MEDIUM_BOARD_48 = 48
    LARGE_BOARD_61 = 61

    # For mapping number of rings to board width
    MARBLE_TO_LAYER = {"w": 1, "g": 2, "b": 3}
    LAYER_TO_MARBLE = dict((v, k) for k, v in MARBLE_TO_LAYER.items())
    HEX_NUMBERS = [
        (1, 1),
        (7, 3),
        (19, 5),
        (SMALL_BOARD_37, 7),
        (MEDIUM_BOARD_48, 8),
        (LARGE_BOARD_61, 9),
        (91, 11),
        (127, 13),
    ]
    DIRECTIONS = [(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)]

    # Player constants
    PLAYER_1 = 0
    PLAYER_2 = 1
    NUM_PLAYERS = 2

    # ==================================================================================
    # GLOBAL STATE STRUCTURE (1D vector, length 10)
    # ==================================================================================
    # The global_state vector stores marble counts and player turn information.
    # Unlike the spatial state (3D array), this is a simple 1D array with 10 values.
    #
    # Layout:
    #   [0-2]:  Supply pool - available marbles for placement (white, gray, black)
    #   [3-5]:  Player 1 captured marbles (white, gray, black)
    #   [6-8]:  Player 2 captured marbles (white, gray, black)
    #   [9]:    Current player (0 = Player 1, 1 = Player 2)
    # ==================================================================================

    # Supply indices (0-2)
    SUPPLY_W = 0
    SUPPLY_G = 1
    SUPPLY_B = 2
    # Player 1 captured indices (3-5)
    P1_CAP_W = 3
    P1_CAP_G = 4
    P1_CAP_B = 5
    # Player 2 captured indices (6-8)
    P2_CAP_W = 6
    P2_CAP_G = 7
    P2_CAP_B = 8
    # Current player index (9)
    CUR_PLAYER = 9

    # Global state slices (convenience accessors)
    SUPPLY_SLICE = slice(0, 3)  # All supply marbles (w, g, b)
    P1_CAP_SLICE = slice(3, 6)  # All P1 captured marbles (w, g, b)
    P2_CAP_SLICE = slice(6, 9)  # All P2 captured marbles (w, g, b)

    # ==================================================================================
    # SPATIAL STATE STRUCTURE (3D array, shape: L x H x W)
    # ==================================================================================
    # The spatial state is a 3D numpy array representing the board configuration over time.
    #
    # Dimensions:
    #   L (layers):  Number of layers = 4 * t + 1, where t = time history depth
    #   H (height):  Board width (e.g., 7 for 37-ring board)
    #   W (width):   Board width (same as height)
    #
    # Layer structure (for t=5, default):
    #   Layers 0-3:   Current board state (rings, white, gray, black marbles)
    #   Layers 4-7:   Board state 1 turn ago
    #   Layers 8-11:  Board state 2 turns ago
    #   Layers 12-15: Board state 3 turns ago
    #   Layers 16-19: Board state 4 turns ago
    #   Layer 20:     Capture flag (marks marble that must be used for capture)
    #
    # The first 4 layers (current state) are the most commonly accessed:
    #   Layer 0: Ring presence (1 = ring exists, 0 = removed)
    #   Layer 1: White marbles (1 = white marble at position, 0 = none)
    #   Layer 2: Gray marbles  (1 = gray marble at position, 0 = none)
    #   Layer 3: Black marbles (1 = black marble at position, 0 = none)
    # ==================================================================================

    # Spatial state layer constants
    RING_LAYER = 0
    MARBLE_LAYERS = slice(1, 4)  # White, gray, black marble layers (current state)
    BOARD_LAYERS = slice(0, 4)  # Rings + all marbles (current state)
    LAYERS_PER_TIMESTEP = 4  # Number of layers per timestep (ring + 3 marble types)

    @staticmethod
    def generate_standard_board_layout(rings):
        """
        Generate board layout for standard ring configurations.

        This is the canonical board layout algorithm, ported from the renderer.
        Returns a 2D numpy array of position strings (e.g., "A1", "B2", etc.).
        Empty positions are represented as empty strings.

        Args:
            rings: Number of rings (must be SMALL_BOARD_37, MEDIUM_BOARD_48, or LARGE_BOARD_61)

        Returns:
            2D numpy array of strings with shape (width, width)

        Raises:
            ValueError: If rings is not a supported standard size
        """
        # Map rings to letter sets
        if rings == ZertzBoard.SMALL_BOARD_37:
            letters = "ABCDEFG"
        elif rings == ZertzBoard.MEDIUM_BOARD_48:
            letters = "ABCDEFGH"
        elif rings == ZertzBoard.LARGE_BOARD_61:
            letters = "ABCDEFGHJ"
        else:
            raise ValueError(
                f"Unsupported standard board size: {rings} rings. "
                f"Supported sizes are {ZertzBoard.SMALL_BOARD_37}, {ZertzBoard.MEDIUM_BOARD_48}, and {ZertzBoard.LARGE_BOARD_61}."
            )

        # Algorithm ported from zertz_renderer.py _build_base() method
        r_max = len(letters)
        is_even = r_max % 2 == 0

        # Lambda to calculate how many letters/positions each row uses
        h_max = lambda xx: r_max - abs(letters.index(letters[xx]) - (r_max // 2))

        # Calculate minimum number for positions
        r_min = h_max(0)
        if is_even:
            r_min += 1

        # Build position array
        pos_array = []
        for i in range(r_max):
            hh = h_max(i)  # Number of letters this row uses

            # Select which letters to use for this row
            ll = letters[:hh] if i < hh / 2 else letters[-hh:]

            # Calculate number range for this row
            nn_max = r_max - i
            nn_min = max(r_min - i, 1)

            # Initialize row with empty strings
            row = [""] * r_max

            # Fill in positions for this row
            for k in range(len(ll)):
                ix = min(k + nn_min, nn_max)
                lt = ll[k]
                pa = letters.find(lt)
                pos = f"{lt}{ix}"
                row[pa] = pos

            pos_array.append(row)

        return np.array(pos_array)

    def __init__(self, rings=37, marbles=None, t=1, clone=None, board_layout=None):
        """Initialize a Zèrtz board.

        The board maintains two separate state representations:

        1. SPATIAL STATE (self.state): 3D array (L x H x W)
           - Rings and marble positions
           - Time history of previous board states
           - Capture flag layer

        2. GLOBAL STATE (self.global_state): 1D array (10 elements)
           - Supply pool counts (white, gray, black)
           - Player 1 captured counts (white, gray, black)
           - Player 2 captured counts (white, gray, black)
           - Current player (0 or 1)

        For ML applications, use ZertzGame.get_current_state() which returns
        both spatial and global state in a dictionary format for complete observability.

        Args:
            rings: Number of rings (37, 48, or 61 for standard boards)
            marbles: Dict with 'w', 'g', 'b' keys for marble counts (default: {w:6, g:8, b:10})
            t: Time history depth (default: 1)
            clone: ZertzBoard instance to clone from
            board_layout: Custom 2D array of position strings (for non-standard boards)
        """
        # Initialize attributes that may be used before assignment
        self.letter_layout = None
        self.flattened_letters = None
        self.board_layout = None

        # Position cache manager (built lazily)
        self.positions = ZertzPositionCollection(self)

        # Canonicalization manager for state transformations
        self.canonicalizer = CanonicalizationManager(self)

        if clone is not None:
            self.rings = clone.rings
            self.width = clone.width
            self.t = clone.t
            self.CAPTURE_LAYER = clone.CAPTURE_LAYER
            self.state = np.copy(clone.state)
            self.global_state = np.copy(clone.global_state)
            self.last_acting_player = clone.last_acting_player if hasattr(clone, "last_acting_player") else None
            if hasattr(clone, "letter_layout") and clone.letter_layout is not None:
                self.letter_layout = np.copy(clone.letter_layout)
            if (
                hasattr(clone, "flattened_letters")
                and clone.flattened_letters is not None
            ):
                self.flattened_letters = np.copy(clone.flattened_letters)
            if hasattr(clone, "board_layout") and clone.board_layout is not None:
                self.board_layout = np.copy(clone.board_layout)
        else:
            # Determine width of board from the number of rings
            if board_layout is None:
                # Use generated layout for standard board sizes
                if rings in [
                    self.SMALL_BOARD_37,
                    self.MEDIUM_BOARD_48,
                    self.LARGE_BOARD_61,
                ]:
                    board_layout = self.generate_standard_board_layout(rings)
                    self.letter_layout = board_layout
                    self.flattened_letters = np.reshape(
                        board_layout, (board_layout.size,)
                    )
                    self.board_layout = self.letter_layout != ""
                    self.rings = np.count_nonzero(board_layout)
                    self.width = board_layout.shape[0]
                else:
                    # Fall back to hexagon formula for non-standard sizes
                    self.rings = rings
                    self.width = 0
                    for total, width in self.HEX_NUMBERS:
                        if total == self.rings:
                            self.width = width
                    if self.width == 0:
                        raise ValueError(
                            f"Unsupported board size: {rings} rings. "
                            f"Use generate_standard_board_layout() for custom layouts."
                        )
            else:
                # Custom board layout provided
                self.letter_layout = board_layout
                self.flattened_letters = np.reshape(board_layout, (board_layout.size,))
                self.board_layout = self.letter_layout != ""
                self.rings = np.count_nonzero(board_layout)
                self.width = board_layout.shape[0]

            # Calculate the number of layers for SPATIAL state
            # 4 * t layers for all pieces going back t steps, 1 for capture layer
            # Spatial Layer:
            #   -          0 = rings (binary)
            #   -          1 = white marbles (binary)
            #   -          2 = gray marbles (binary)
            #   -          3 = black marbles (binary)
            #   -        ...   (t - 1) * 4 more layers for ring and marble state on previous time steps
            #   -      t * 4 = capturing marble (binary)
            self.t = t
            spatial_layers = 4 * self.t + 1
            self.CAPTURE_LAYER = self.t * 4

            # Initialize SPATIAL state as 3d array
            self.state = np.zeros(
                (spatial_layers, self.width, self.width), dtype=np.uint8
            )

            # Initialize GLOBAL state as 1d array
            # Global state vector (10 values):
            #   - 0-2: supply (white, gray, black) [0, 10]
            #   - 3-5: player 1 captured (white, gray, black) [0, 10]
            #   - 6-8: player 2 captured (white, gray, black) [0, 10]
            #   - 9: current player (0 or 1)
            self.global_state = np.zeros(10, dtype=np.uint8)

            # Place rings
            if self.board_layout is None:
                middle = self.width // 2
                for i in range(self.width):
                    lb = max(0, i - middle)
                    ub = min(self.width, middle + i + 1)
                    self.state[self.RING_LAYER, lb:ub, i] = 1
            else:
                self.state[self.RING_LAYER, self.board_layout] = 1

            # Set the number of each type of marble available in the supply
            #   default: 6x white, 8x gray, 10x black
            if marbles is None:
                self.global_state[self.SUPPLY_SLICE] = [6, 8, 10]  # white, gray, black
            else:
                self.global_state[self.SUPPLY_W] = marbles["w"]
                self.global_state[self.SUPPLY_G] = marbles["g"]
                self.global_state[self.SUPPLY_B] = marbles["b"]

            # Player captured marbles start at 0 (indices 3-8)
            # Current player starts at 0 (index 9)

            # Track which player made the last action (for win detection)
            self.last_acting_player = None

        # Coordinate maps are built lazily via PositionCollection

    def _get_config(self):
        """Create BoardConfig from current board state.

        This allows delegation to pure stateless functions.
        BoardConfig should be cached if performance becomes an issue.
        """
        return stateless_logic.BoardConfig(
            width=self.width,
            rings=self.rings,
            t=self.t,
            directions=tuple(self.DIRECTIONS),
            ring_layer=self.RING_LAYER,
            marble_layers=self.MARBLE_LAYERS,
            board_layers=self.BOARD_LAYERS,
            capture_layer=self.CAPTURE_LAYER,
            layers_per_timestep=self.LAYERS_PER_TIMESTEP,
            supply_w=self.SUPPLY_W,
            supply_g=self.SUPPLY_G,
            supply_b=self.SUPPLY_B,
            p1_cap_w=self.P1_CAP_W,
            p1_cap_g=self.P1_CAP_G,
            p1_cap_b=self.P1_CAP_B,
            p2_cap_w=self.P2_CAP_W,
            p2_cap_g=self.P2_CAP_G,
            p2_cap_b=self.P2_CAP_B,
            cur_player=self.CUR_PLAYER,
            supply_slice=self.SUPPLY_SLICE,
            p1_cap_slice=self.P1_CAP_SLICE,
            p2_cap_slice=self.P2_CAP_SLICE,
            player_1=self.PLAYER_1,
            player_2=self.PLAYER_2,
            num_players=self.NUM_PLAYERS,
            marble_to_layer=self.MARBLE_TO_LAYER,
            layer_to_marble=self.LAYER_TO_MARBLE,
            board_layout=self.board_layout,
        )

    @staticmethod
    def get_middle_ring(src, dst):
        # Return the (y, x) index of the ring between src and dst
        y1, x1 = src
        y2, x2 = dst
        return (y1 + y2) // 2, (x1 + x2) // 2

    def get_neighbors(self, index):
        # Return a list of (y, x) indices that are adjacent to index on the board.
        # The neighboring index may not be within the board space so it must be checked
        # that it is inbounds (see _is_inbounds).
        y, x = index
        neighbors = [(y + dy, x + dx) for dy, dx in self.DIRECTIONS]
        return neighbors

    def _is_adjacent(self, l1, l2):
        # Return True if l1 and l2 are adjacent to each other on the hexagonal board
        return l2 in self.get_neighbors(l1)

    @staticmethod
    def get_jump_destination(start, cap):
        # Return the landing index after capturing the marble at cap from start.
        # The landing index may not be within the board space so it must be checked
        # that it is inbounds (see _is_inbounds).
        sy, sx = start
        cy, cx = cap
        dy = (cy - sy) * 2
        dx = (cx - sx) * 2
        return sy + dy, sx + dx

    def _is_inbounds(self, index):
        # Return True if the index is in bounds for board's width
        y, x = index
        return 0 <= y < self.width and 0 <= x < self.width

    def _get_regions(self):
        """Return list of connected regions on the board.

        Delegates to stateless_logic.get_regions() for zero code duplication.
        """
        config = self._get_config()
        return stateless_logic.get_regions(self.state, config)

    def get_cur_player(self):
        return int(self.global_state[self.CUR_PLAYER])

    def _next_player(self):
        self.global_state[self.CUR_PLAYER] = (
            self.global_state[self.CUR_PLAYER] + 1
        ) % self.NUM_PLAYERS

    def _get_supply_index(self, marble_type):
        """Get global_state index for marble in supply."""
        marble_to_supply_idx = {
            "w": self.SUPPLY_W,
            "g": self.SUPPLY_G,
            "b": self.SUPPLY_B,
        }
        return marble_to_supply_idx[marble_type]

    def _get_captured_index(self, marble_type, player):
        """Get global_state index for captured marble for given player."""
        if player == self.PLAYER_1:
            marble_to_cap_idx = {
                "w": self.P1_CAP_W,
                "g": self.P1_CAP_G,
                "b": self.P1_CAP_B,
            }
        else:
            marble_to_cap_idx = {
                "w": self.P2_CAP_W,
                "g": self.P2_CAP_G,
                "b": self.P2_CAP_B,
            }
        return marble_to_cap_idx[marble_type]

    def get_marble_type_at(self, index):
        y, x = index
        marble_type = self.LAYER_TO_MARBLE[
            np.argmax(self.state[self.MARBLE_LAYERS, y, x]) + 1
        ]
        return marble_type

    def take_action(self, action, action_type):
        # Input: action is an index into the action space matrix
        #        action_type is 'PUT' or 'CAP'

        # Capture who is acting BEFORE any player switches occur
        # This simplifies win detection logic
        self.last_acting_player = self.get_cur_player()

        # Push back the previous t states and copy the most recent state to the top layers
        layers_per_step = self.LAYERS_PER_TIMESTEP
        self.state[0 : layers_per_step * self.t] = np.concatenate(
            [
                self.state[self.BOARD_LAYERS],
                self.state[0 : layers_per_step * (self.t - 1)],
            ],
            axis=0,
        )

        if action_type == "PUT":
            return self.take_placement_action(action)
        elif action_type == "CAP":
            return self.take_capture_action(action)

    def take_placement_action(self, action):
        # Placement actions have dimension (3 x w^2 x w^2 + 1)
        # Translate the action dimensions into marble_type, put_index, and rem_index
        type_index, put_loc, rem_loc = action
        marble_type = self.LAYER_TO_MARBLE[type_index + 1]
        put_index = (put_loc // self.width, put_loc % self.width)
        if rem_loc == self.width**2:
            rem_index = None
        else:
            rem_index = (rem_loc // self.width, rem_loc % self.width)

        # Place the marble on the board
        y, x = put_index
        if np.sum(self.state[self.BOARD_LAYERS, y, x]) != 1:
            raise ValueError(
                f"Invalid placement: position ({y}, {x}) is not an empty ring"
            )
        put_layer = self.MARBLE_TO_LAYER[marble_type]
        self.state[put_layer][put_index] = 1

        # Remove the marble from the supply or current player's captured marbles
        # Per official rules: captured marbles can only be used when ENTIRE pool is empty
        supply_idx = self._get_supply_index(marble_type)
        if self.global_state[supply_idx] >= 1:
            # Marble available in supply - use it
            self.global_state[supply_idx] -= 1
        elif np.all(self.global_state[self.SUPPLY_SLICE] == 0):
            # Entire supply pool is empty - can use captured marbles
            captured_idx = self._get_captured_index(marble_type, self.get_cur_player())
            if self.global_state[captured_idx] < 1:
                raise ValueError(
                    f"No {marble_type} marbles available in supply or captured by player {self.get_cur_player()}"
                )
            self.global_state[captured_idx] -= 1
        else:
            # This marble type is empty but pool has other marbles - cannot use captured marbles yet
            raise ValueError(
                f"No {marble_type} marbles in supply. Cannot use captured marbles until entire pool is empty."
            )

        # Remove the ring from the board
        captured_marbles = None
        if rem_index is not None:
            self.state[self.RING_LAYER][rem_index] = 0
            # Check for isolated regions and handle captures per official rules
            captured_marbles = self._check_for_isolation_capture()
        self._next_player()
        return captured_marbles

    def _check_for_isolation_capture(self):
        captured_marbles = []
        regions = self._get_regions()
        if len(regions) > 1:
            # Multiple regions exist - check each isolated region
            main_region = max(regions, key=len)

            for region in regions:
                if region == main_region:
                    continue  # Skip the main region

                # Check if ALL rings in this isolated region are occupied
                all_occupied = all(
                    np.sum(self.state[self.MARBLE_LAYERS, y, x]) == 1
                    for y, x in region
                )

                if all_occupied:
                    # Capture all marbles in this fully-occupied isolated region
                    for y, x in region:
                        marble_type = self.get_marble_type_at((y, x))
                        pos_str = self.index_to_str((y, x))

                        # Add to current player's captured marbles
                        captured_idx = self._get_captured_index(
                            marble_type, self.get_cur_player()
                        )
                        self.global_state[captured_idx] += 1

                        # Remove marble from board
                        self.state[self.MARBLE_LAYERS, y, x] = 0

                        # Remove ring from board
                        self.state[self.RING_LAYER, y, x] = 0

                        # Add to captured list for return value
                        captured_marbles.append({"marble": marble_type, "pos": pos_str})

        return captured_marbles if captured_marbles else None

    def take_capture_action(self, action):
        # Capture actions have dimension (6 x w x w)
        # Translate the action dimensions into src_index, marble_type, cap_index and dst_index
        direction, y, x = action
        src_index = (y, x)
        marble_type = self.get_marble_type_at(src_index)
        dy, dx = self.DIRECTIONS[direction]
        cap_index = (y + dy, x + dx)
        dst_index = self.get_jump_destination(src_index, cap_index)
        y, x = cap_index

        # Reset the capture layer
        self.state[self.CAPTURE_LAYER] = 0

        # Remove capturing marble from src_index and place it at dst_index
        marble_layer = self.MARBLE_TO_LAYER[marble_type]
        self.state[marble_layer][src_index] = 0
        self.state[marble_layer][dst_index] = 1

        # Give the captured marble to the current player and remove it from the board
        if np.sum(self.state[self.MARBLE_LAYERS, y, x]) != 1:
            raise ValueError(f"Invalid capture: no marble at position ({y}, {x})")
        captured_type = self.get_marble_type_at(cap_index)
        captured_idx = self._get_captured_index(captured_type, self.get_cur_player())
        self.global_state[captured_idx] += 1
        self.state[self.MARBLE_LAYERS, y, x] = 0

        # Update the capture layer if there is a forced chain capture
        neighbors = self.get_neighbors(dst_index)
        for neighbor in neighbors:
            y, x = neighbor
            # Check each neighbor to see if it has a marble
            if (
                self._is_inbounds(neighbor)
                and np.sum(self.state[self.MARBLE_LAYERS, y, x]) == 1
            ):
                next_dst = self.get_jump_destination(dst_index, neighbor)
                y, x = next_dst
                if (
                    self._is_inbounds(next_dst)
                    and np.sum(self.state[self.BOARD_LAYERS, y, x]) == 1
                ):
                    # Set the captured layer to 1 at dst_index
                    self.state[self.CAPTURE_LAYER][dst_index] = 1
                    break

        # Update current player if there are no forced chain captures
        if np.sum(self.state[self.CAPTURE_LAYER]) == 0:
            self._next_player()
        return captured_type

    def get_valid_moves(self):
        """Return valid placement and capture moves.

        Delegates to stateless_logic.get_valid_actions() for zero code duplication.
        """
        config = self._get_config()
        return stateless_logic.get_valid_actions(self.state, self.global_state, config)

    def get_placement_shape(self):
        # get shape of placement moves as a tuple
        return 3, self.width**2, self.width**2 + 1

    def get_capture_shape(self):
        # get shape of capture moves as a tuple
        return 6, self.width, self.width

    def get_placement_moves(self):
        """Return valid placement moves.

        Delegates to stateless_logic.get_placement_moves() for zero code duplication.
        """
        config = self._get_config()
        return stateless_logic.get_placement_moves(self.state, self.global_state, config)

    def get_capture_moves(self):
        """Return valid capture moves.

        Delegates to stateless_logic.get_capture_moves() for zero code duplication.
        """
        config = self._get_config()
        return stateless_logic.get_capture_moves(self.state, self.global_state, config)

    def _get_open_rings(self):
        """Return indices of all empty rings on the board.

        Delegates to stateless_logic.get_open_rings() for zero code duplication.
        """
        config = self._get_config()
        return stateless_logic.get_open_rings(self.state, config)

    def _is_removable(self, index):
        """Check if ring at index can be removed.

        Delegates to stateless_logic.is_removable() for zero code duplication.
        """
        config = self._get_config()
        return stateless_logic.is_removable(index, self.state, config)

    def _get_removable_rings(self):
        """Return list of removable ring indices.

        Delegates to stateless_logic.get_removable_rings() for zero code duplication.
        """
        config = self._get_config()
        return stateless_logic.get_removable_rings(self.state, config)

    # =========================  GEOMETRIC RING REMOVAL  =========================

    def _yx_to_cartesian(self, y, x):
        """Convert board indices (y, x) to Cartesian coordinates.

        Uses standard pointy-top hexagonal grid conversion:
        - q = x - center
        - r = y - x
        - xc = sqrt(3) * (q + r/2)
        - yc = 1.5 * r

        Returns:
            tuple: (xc, yc) Cartesian coordinates
        """
        c = self.width // 2
        q = x - c
        r = y - x
        sqrt3 = np.sqrt(3)
        xc = sqrt3 * (q + r / 2.0)
        yc = 1.5 * r
        return xc, yc

    def _is_removable_geometric(self, index, ring_radius=None):
        """Check if a ring can be removed using geometric collision detection.

        Uses actual Cartesian coordinates and perpendicular distance calculations
        to verify that the ring can be slid out in some direction without colliding
        with any other rings.

        This method validates that the simple adjacency-based heuristic (_is_removable)
        is geometrically correct.

        Args:
            index: (y, x) position of the ring to check
            ring_radius: Radius of a ring (default √3/2 ≈ 0.866 for unit grid)

        Returns:
            bool: True if the ring can be removed
        """
        if ring_radius is None:
            ring_radius = (
                np.sqrt(3) / 2.0
            )  # Correct radius for touching rings in unit hex grid

        y, x = index

        # Ring must be empty (no marble on it)
        if np.sum(self.state[self.BOARD_LAYERS, y, x]) != 1:
            return False

        # Get Cartesian position of this ring
        ring_x, ring_y = self._yx_to_cartesian(y, x)

        # Get all neighbor positions
        neighbors = self.get_neighbors(index)

        # Check each pair of consecutive empty neighbors - each creates a potential slide direction
        for i in range(len(neighbors)):
            curr = neighbors[i]
            next_neighbor = neighbors[(i + 1) % len(neighbors)]

            curr_empty = (
                not self._is_inbounds(curr) or self.state[self.RING_LAYER][curr] == 0
            )
            next_empty = (
                not self._is_inbounds(next_neighbor)
                or self.state[self.RING_LAYER][next_neighbor] == 0
            )

            if curr_empty and next_empty:
                # Found a gap. Calculate the slide direction (angle bisector of the gap)
                # The gap is between directions i and i+1
                dy1, dx1 = self.DIRECTIONS[i]
                dy2, dx2 = self.DIRECTIONS[(i + 1) % len(self.DIRECTIONS)]

                # Convert direction offsets to actual neighbor positions, then to Cartesian vectors
                neighbor1_pos = (y + dy1, x + dx1)
                neighbor2_pos = (y + dy2, x + dx2)

                # Get Cartesian positions of where these neighbors would be
                n1_x, n1_y = self._yx_to_cartesian(*neighbor1_pos)
                n2_x, n2_y = self._yx_to_cartesian(*neighbor2_pos)

                # Direction vectors from ring to neighbor positions
                dir1_x, dir1_y = n1_x - ring_x, n1_y - ring_y
                dir2_x, dir2_y = n2_x - ring_x, n2_y - ring_y

                # Normalize
                norm1 = np.sqrt(dir1_x**2 + dir1_y**2)
                norm2 = np.sqrt(dir2_x**2 + dir2_y**2)
                if norm1 > 0:
                    dir1_x, dir1_y = dir1_x / norm1, dir1_y / norm1
                if norm2 > 0:
                    dir2_x, dir2_y = dir2_x / norm2, dir2_y / norm2

                # Angle bisector (slide direction)
                slide_dx = dir1_x + dir2_x
                slide_dy = dir1_y + dir2_y
                slide_norm = np.sqrt(slide_dx**2 + slide_dy**2)

                if slide_norm > 0:
                    slide_dx /= slide_norm
                    slide_dy /= slide_norm

                    # Check if we can slide out in this direction without hitting other rings
                    if self._can_slide_ring_out(
                        ring_x, ring_y, slide_dx, slide_dy, index, ring_radius
                    ):
                        return True

        return False

    def _can_slide_ring_out(
        self, ring_x, ring_y, slide_dx, slide_dy, ring_index, ring_radius
    ):
        """Check if a ring can be slid out in a given direction.

        A ring is removable if it can slide at least one hex spacing (√3) in some
        direction without colliding with other rings. This represents sliding the
        ring one full hex-position away, which effectively removes it from play.

        Physics of ring collision:
        - In unit hex grid (size=1.0), adjacent centers are √3 apart
        - For touching rings: ring_radius = √3/2 ≈ 0.866
        - Ring diameter = 2 * ring_radius = √3
        - Minimum slide distance = √3 (one hex spacing)
        - Two rings collide if their centers are < √3 apart
        - For slide path: rings collide if perpendicular distance < diameter (2 * ring_radius)

        Args:
            ring_x, ring_y: Cartesian position of the ring to slide
            slide_dx, slide_dy: Normalized direction vector to slide
            ring_index: (y, x) index of the ring being slid (to exclude from checks)
            ring_radius: Radius of rings (should be √3/2 for unit grid)

        Returns:
            bool: True if ring can slide at least √3 distance without collision
        """
        # Minimum slide distance: one hex spacing
        sqrt3 = np.sqrt(3)
        min_slide_distance = sqrt3

        # For a ring to be slideable, no other ring should be within 2*radius
        # (diameter) of the slide path AND within the first √3 of travel

        for y in range(self.width):
            for x in range(self.width):
                if (y, x) == ring_index:
                    continue  # Don't check against ourselves

                if self._is_inbounds((y, x)) and self.state[self.RING_LAYER, y, x] == 1:
                    # This ring exists, check if it would block the slide
                    other_x, other_y = self._yx_to_cartesian(y, x)

                    # Vector from our ring to the other ring
                    to_other_x = other_x - ring_x
                    to_other_y = other_y - ring_y
                    dist_sq = to_other_x**2 + to_other_y**2

                    # Project onto slide direction
                    projection = to_other_x * slide_dx + to_other_y * slide_dy

                    # Only check rings in front of us AND within one hex spacing
                    # Rings behind us or beyond the minimum slide distance don't matter
                    if projection < 0 or projection > min_slide_distance:
                        continue

                    # Calculate perpendicular distance from the slide path
                    # perp_dist² = |v|² - proj²
                    perp_dist_sq = dist_sq - projection**2

                    # If perpendicular distance < 2*radius, rings would collide during slide
                    # Need at least 2*radius separation (ring diameters)
                    # Use small tolerance for floating point comparison
                    tolerance = 1e-6
                    min_clearance_sq = (2 * ring_radius) ** 2
                    if perp_dist_sq < min_clearance_sq - tolerance:
                        # Would collide during the first √3 of slide
                        return False

        return True

    def _get_rotational_symmetries(self, state=None):
        # Rotate the board 180 degrees
        # Always copy to ensure immutability of self.state or passed state
        state_to_rotate = np.copy(state if state is not None else self.state)
        return np.rot90(np.rot90(state_to_rotate, axes=(1, 2)), axes=(1, 2))

    def _get_mirror_symmetries(self, state=None):
        # Flip the board while maintaining adjacency
        # Always copy to ensure immutability of self.state or passed state
        mirror_state = np.copy(state if state is not None else self.state)
        layers = mirror_state.shape[0]
        for i in range(layers):
            mirror_state[i] = mirror_state[i].T
        return mirror_state

    def get_state_symmetries(self):
        # Return a list of symmetrical states by mirroring and rotating the board
        # noinspection PyListCreation
        symmetries = []
        symmetries.append((0, self._get_mirror_symmetries()))
        symmetries.append((1, self._get_rotational_symmetries()))
        symmetries.append((2, self._get_rotational_symmetries(symmetries[0][1])))
        return symmetries

    def _flat_to_2d(self, flat_index):
        """Convert flat index to 2D board coordinates (y, x)."""
        return flat_index // self.width, flat_index % self.width

    def _2d_to_flat(self, y, x):
        """Convert 2D board coordinates to flat index."""
        return y * self.width + x

    def _mirror_coords(self, y, x):
        """Mirror coordinates by swapping x and y axes."""
        return x, y

    def _rotate_coords(self, y, x):
        """Rotate coordinates 180 degrees around board center."""
        # Universal formula that works for both odd and even widths
        return (self.width - 1) - y, (self.width - 1) - x

    def mirror_action(self, action_type, translated):
        if action_type == "CAP":
            # swap capture direction axes
            temp = np.copy(translated)
            translated[3], translated[1] = temp[1], temp[3]
            translated[4], translated[0] = temp[0], temp[4]

            # transpose location axes
            d = translated.shape[0]
            for i in range(d):
                translated[i] = translated[i].T

        elif action_type == "PUT":
            temp = np.copy(translated)
            _, put, rem = translated.shape
            for p in range(put):
                # Translate the put index
                put_y, put_x = self._flat_to_2d(p)
                new_put_y, new_put_x = self._mirror_coords(put_y, put_x)
                new_p = self._2d_to_flat(new_put_y, new_put_x)
                for r in range(rem - 1):
                    # Translate the rem index
                    rem_y, rem_x = self._flat_to_2d(r)
                    new_rem_y, new_rem_x = self._mirror_coords(rem_y, rem_x)
                    new_r = self._2d_to_flat(new_rem_y, new_rem_x)
                    translated[:, new_p, new_r] = temp[:, p, r]

                # The last rem index is the same
                translated[:, new_p, rem - 1] = translated[:, new_p, rem - 1]

        return translated

    def rotate_action(self, action_type, translated):
        if action_type == "CAP":
            # swap capture direction axes
            temp = np.copy(translated)
            translated[3], translated[0] = temp[0], temp[3]
            translated[4], translated[1] = temp[1], temp[4]
            translated[5], translated[2] = temp[2], temp[5]

            # rotate location axes using universal formula
            temp = np.copy(translated)
            _, y, x = temp.shape
            for i in range(y):
                new_i = (self.width - 1) - i
                for j in range(x):
                    new_j = (self.width - 1) - j
                    translated[:, new_i, new_j] = temp[:, i, j]

        if action_type == "PUT":
            temp = np.copy(translated)
            _, put, rem = translated.shape
            for p in range(put):
                # Translate the put index
                put_y, put_x = self._flat_to_2d(p)
                new_put_y, new_put_x = self._rotate_coords(put_y, put_x)
                new_p = self._2d_to_flat(new_put_y, new_put_x)
                for r in range(rem - 1):
                    # Translate the rem index
                    rem_y, rem_x = self._flat_to_2d(r)
                    new_rem_y, new_rem_x = self._rotate_coords(rem_y, rem_x)
                    new_r = self._2d_to_flat(new_rem_y, new_rem_x)
                    translated[:, new_p, new_r] = temp[:, p, r]

                # The last rem index is the same
                translated[:, new_p, rem - 1] = translated[:, new_p, rem - 1]

        return translated

    def _compute_label(self, index: tuple[int, int]) -> str:
        y, x = index
        if self.letter_layout is not None:
            label = self.letter_layout[y][x]
            if label:
                return label
        letter = chr(x + 65)
        mid = self.width // 2
        offset = max(mid - x, 0)
        number = (self.width - y) - offset
        return f"{letter}{number}"

    def _ensure_positions_built(self) -> None:
        self.positions.ensure()

    def _label_to_yx(self, index_str: str) -> tuple[int, int]:
        self._ensure_positions_built()
        pos = self.positions.get_by_label(index_str)
        if pos is None:
            raise ValueError(f"Coordinate '{index_str}' not found in board layout")
        return pos.yx

    def _yx_to_label(self, index: tuple[int, int]) -> str:
        self._ensure_positions_built()
        pos = self.positions.get_by_yx(index)
        return pos.label if pos else ""

    @property
    def _positions(self):
        return self.positions.by_yx

    @property
    def _positions_by_label(self):
        return self.positions.by_label

    @property
    def _positions_by_axial(self):
        return self.positions.by_axial

    def str_to_index(self, index_str):
        """
        Convert a coordinate string like 'A1' (bottom numbering) to array indices (y, x).
        In this official layout, row numbers count upward from the bottom of the board.
        """
        return self._label_to_yx(index_str)

    def index_to_str(self, index):
        y, x = index
        if not self._is_inbounds(index):
            raise IndexError(f"Position ({y}, {x}) is out of bounds")
        pos = self.position_from_yx(index)
        if self.state[self.RING_LAYER, y, x] == 0:
            return ""
        return pos.label

    def position_from_yx(self, index: tuple[int, int]) -> ZertzPosition:
        self._ensure_positions_built()
        pos = self.positions.get_by_yx(index)
        if pos is None:
            raise ValueError(f"Position {index} is not a valid ring coordinate")
        return pos

    def position_from_label(self, label: str) -> ZertzPosition:
        self._ensure_positions_built()
        pos = self.positions.get_by_label(label)
        if pos is None:
            raise ValueError(f"Coordinate '{label}' not found in board layout")
        return pos

    def position_from_axial(self, axial: tuple[int, int]) -> ZertzPosition:
        self._ensure_positions_built()
        pos = self.positions.get_by_axial(axial)
        if pos is None:
            raise ValueError(f"Axial coordinate {axial} is not on this board")
        return pos

    # =========================  AXIAL COORDINATES  =========================

    def _build_axial_maps(self):
        self._ensure_positions_built()

    @property
    def _yx_to_ax(self):
        return self.positions.yx_to_ax

    @property
    def _ax_to_yx(self):
        return self.positions.ax_to_yx

    @staticmethod
    def _ax_rot60(q, r, k=1):
        """Rotate (q,r) by k * 60° counterclockwise in axial coords.

        Works for both regular and doubled coordinates (for even-width boards).
        """
        k %= 6
        for _ in range(k):
            q, r = -r, q + r  # 60° CCW
        return q, r

    @staticmethod
    def _ax_mirror_q_axis(q, r):
        """Reflect (q,r) across the q-axis (cube: swap y and z)."""
        # In cube coords (x=q, z=r, y=-q-r), mirror over q-axis => (x, z, y)
        return q, -q - r

    def canonicalize_state(self, transforms=TransformFlags.ALL):
        """
        Return (canonical_state, transform_name, inverse_name).

        Delegates to CanonicalizationManager.

        Finds the lexicographically smallest representation among all enabled symmetry
        transformations. Transformations are applied in order: translation, then rotation/mirror.

        Args:
            transforms: TransformFlags specifying which transforms to use (default: ALL)

        Returns:
            tuple: (canonical_state, transform_name, inverse_name)
                - canonical_state: The transformed state with minimum lexicographic key
                - transform_name: Name of transform applied (e.g., "T2,1_MR120", "R60", "T1,-1")
                - inverse_name: Inverse transform to map back to original orientation
        """
        return self.canonicalizer.canonicalize_state(state=None, transforms=transforms)


    # =========================  GEOMETRIC RING REMOVAL  =========================

    def yx_to_cartesian(self, y, x):
        """Convert board indices (y, x) to Cartesian coordinates.

        Uses standard pointy-top hexagonal grid conversion:
        - q = x - center
        - r = y - x
        - xc = sqrt(3) * (q + r/2)
        - yc = 1.5 * r

        Returns:
            tuple: (xc, yc) Cartesian coordinates
        """
        c = self.width // 2
        q = x - c
        r = y - x
        sqrt3 = np.sqrt(3)
        xc = sqrt3 * (q + r / 2.0)
        yc = 1.5 * r
        return xc, yc

    def is_removable_geometric(self, index, ring_radius=None):
        """Check if a ring can be removed using geometric collision detection.

        Uses actual Cartesian coordinates and perpendicular distance calculations
        to verify that the ring can be slid out in some direction without colliding
        with any other rings.

        This method validates that the simple adjacency-based heuristic (_is_removable)
        is geometrically correct.

        Args:
            index: (y, x) position of the ring to check
            ring_radius: Radius of a ring (default √3/2 ≈ 0.866 for unit grid)

        Returns:
            bool: True if the ring can be removed
        """
        if ring_radius is None:
            ring_radius = (
                np.sqrt(3) / 2.0
            )  # Correct radius for touching rings in unit hex grid

        y, x = index

        # Ring must be empty (no marble on it)
        if np.sum(self.state[self.BOARD_LAYERS, y, x]) != 1:
            return False

        # Get Cartesian position of this ring
        ring_x, ring_y = self.yx_to_cartesian(y, x)

        # Get all neighbor positions
        neighbors = self.get_neighbors(index)

        # Check each pair of consecutive empty neighbors - each creates a potential slide direction
        for i in range(len(neighbors)):
            curr = neighbors[i]
            next_neighbor = neighbors[(i + 1) % len(neighbors)]

            curr_empty = (
                not self._is_inbounds(curr) or self.state[self.RING_LAYER][curr] == 0
            )
            next_empty = (
                not self._is_inbounds(next_neighbor)
                or self.state[self.RING_LAYER][next_neighbor] == 0
            )

            if curr_empty and next_empty:
                # Found a gap. Calculate the slide direction (angle bisector of the gap)
                # The gap is between directions i and i+1
                dy1, dx1 = self.DIRECTIONS[i]
                dy2, dx2 = self.DIRECTIONS[(i + 1) % len(self.DIRECTIONS)]

                # Convert direction offsets to actual neighbor positions, then to Cartesian vectors
                neighbor1_pos = (y + dy1, x + dx1)
                neighbor2_pos = (y + dy2, x + dx2)

                # Get Cartesian positions of where these neighbors would be
                n1_x, n1_y = self.yx_to_cartesian(*neighbor1_pos)
                n2_x, n2_y = self.yx_to_cartesian(*neighbor2_pos)

                # Direction vectors from ring to neighbor positions
                dir1_x, dir1_y = n1_x - ring_x, n1_y - ring_y
                dir2_x, dir2_y = n2_x - ring_x, n2_y - ring_y

                # Normalize
                norm1 = np.sqrt(dir1_x**2 + dir1_y**2)
                norm2 = np.sqrt(dir2_x**2 + dir2_y**2)
                if norm1 > 0:
                    dir1_x, dir1_y = dir1_x / norm1, dir1_y / norm1
                if norm2 > 0:
                    dir2_x, dir2_y = dir2_x / norm2, dir2_y / norm2

                # Angle bisector (slide direction)
                slide_dx = dir1_x + dir2_x
                slide_dy = dir1_y + dir2_y
                slide_norm = np.sqrt(slide_dx**2 + slide_dy**2)

                if slide_norm > 0:
                    slide_dx /= slide_norm
                    slide_dy /= slide_norm

                    # Check if we can slide out in this direction without hitting other rings
                    if self._can_slide_ring_out(
                        ring_x, ring_y, slide_dx, slide_dy, index, ring_radius
                    ):
                        return True

        return False

    def _can_slide_ring_out(
        self, ring_x, ring_y, slide_dx, slide_dy, ring_index, ring_radius
    ):
        """Check if a ring can be slid out in a given direction.

        A ring is removable if it can slide at least one hex spacing (√3) in some
        direction without colliding with other rings. This represents sliding the
        ring one full hex-position away, which effectively removes it from play.

        Physics of ring collision:
        - In unit hex grid (size=1.0), adjacent centers are √3 apart
        - For touching rings: ring_radius = √3/2 ≈ 0.866
        - Ring diameter = 2 * ring_radius = √3
        - Minimum slide distance = √3 (one hex spacing)
        - Two rings collide if their centers are < √3 apart
        - For slide path: rings collide if perpendicular distance < diameter (2 * ring_radius)

        Args:
            ring_x, ring_y: Cartesian position of the ring to slide
            slide_dx, slide_dy: Normalized direction vector to slide
            ring_index: (y, x) index of the ring being slid (to exclude from checks)
            ring_radius: Radius of rings (should be √3/2 for unit grid)

        Returns:
            bool: True if ring can slide at least √3 distance without collision
        """
        # Minimum slide distance: one hex spacing
        sqrt3 = np.sqrt(3)
        min_slide_distance = sqrt3

        # For a ring to be slideable, no other ring should be within 2*radius
        # (diameter) of the slide path AND within the first √3 of travel

        for y in range(self.width):
            for x in range(self.width):
                if (y, x) == ring_index:
                    continue  # Don't check against ourselves

                if self._is_inbounds((y, x)) and self.state[self.RING_LAYER, y, x] == 1:
                    # This ring exists, check if it would block the slide
                    other_x, other_y = self.yx_to_cartesian(y, x)

                    # Vector from our ring to the other ring
                    to_other_x = other_x - ring_x
                    to_other_y = other_y - ring_y
                    dist_sq = to_other_x**2 + to_other_y**2

                    # Project onto slide direction
                    projection = to_other_x * slide_dx + to_other_y * slide_dy

                    # Only check rings in front of us AND within one hex spacing
                    # Rings behind us or beyond the minimum slide distance don't matter
                    if projection < 0 or projection > min_slide_distance:
                        continue

                    # Calculate perpendicular distance from the slide path
                    # perp_dist² = |v|² - proj²
                    perp_dist_sq = dist_sq - projection**2

                    # If perpendicular distance < 2*radius, rings would collide during slide
                    # Need at least 2*radius separation (ring diameters)
                    # Use small tolerance for floating point comparison
                    tolerance = 1e-6
                    min_clearance_sq = (2 * ring_radius) ** 2
                    if perp_dist_sq < min_clearance_sq - tolerance:
                        # Would collide during the first √3 of slide
                        return False

        return True
