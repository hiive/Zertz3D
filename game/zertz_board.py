from collections import deque
import numpy as np

from game.zertz_position import ZertzPosition, ZertzPositionCollection


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

        if clone is not None:
            self.rings = clone.rings
            self.width = clone.width
            self.t = clone.t
            self.CAPTURE_LAYER = clone.CAPTURE_LAYER
            self.state = np.copy(clone.state)
            self.global_state = np.copy(clone.global_state)
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

        # Coordinate maps are built lazily via PositionCollection

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
        # Return a list of continuous regions on the board. A region consists of a list of indices.
        # If any index can be reached from any other index then this will return a list of length 1.
        regions = []
        not_visited = set(zip(*np.where(self.state[self.RING_LAYER] == 1)))
        while not_visited:
            # While there are indices that have not been added to a region start a new empty region
            region = []
            queue = deque()
            queue.appendleft(not_visited.pop())
            # Add all indices to the region that can be reached from the starting index
            while queue:
                index = queue.pop()
                region.append(index)
                # Add all neighbors to the queue and mark visited to add them to the same region
                for neighbor in self.get_neighbors(index):
                    if (
                        neighbor in not_visited
                        and self._is_inbounds(neighbor)
                        and self.state[self.RING_LAYER][neighbor] != 0
                    ):
                        not_visited.remove(neighbor)
                        queue.appendleft(neighbor)
            regions.append(region)
        return regions

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
        # Return two matrices that can be used to filter the placement and capture action policy
        # distribution for actions that are valid with the current game state.
        capture = self.get_capture_moves()
        if np.any(capture):
            # no placement move is allowed if there is a valid capture move
            placement = np.zeros((3, self.width**2, self.width**2 + 1), dtype=bool)
        else:
            placement = self.get_placement_moves()
        return placement, capture

    def get_placement_shape(self):
        # get shape of placement moves as a tuple
        return 3, self.width**2, self.width**2 + 1

    def get_capture_shape(self):
        # get shape of capture moves as a tuple
        return 6, self.width, self.width

    def get_placement_moves(self):
        # Return a boolean matrix of size (3 x w^2 x w^2 + 1) with the value True at
        # every index that corresponds to a valid placement action.
        # Marble types correspond to the following indices {'w':0, 'g':1, 'b':2}
        # A ring removal value of w^2 indicates no ring is removed
        moves = np.zeros((3, self.width**2, self.width**2 + 1), dtype=bool)

        # Build list of open and removable rings for marble placement and ring removal
        open_rings = list(self._get_open_rings())
        removable_rings = list(self._get_removable_rings())

        # Get list of marble types that can be placed. If supply is empty then
        # the player must use a captured marble.
        supply_counts = self.global_state[self.SUPPLY_SLICE]
        if np.all(supply_counts == 0):
            # Use current player's captured marbles
            if self.get_cur_player() == self.PLAYER_1:
                marble_counts = self.global_state[self.P1_CAP_SLICE]
            else:
                marble_counts = self.global_state[self.P2_CAP_SLICE]
        else:
            marble_counts = supply_counts

        # Assign 1 to all indices that are valid actions
        for m, marble_count in enumerate(marble_counts):
            if marble_count == 0:
                continue
            for put_index in open_rings:
                put = put_index[0] * self.width + put_index[1]
                for rem_index in removable_rings:
                    rem = rem_index[0] * self.width + rem_index[1]
                    if put != rem:
                        moves[m, put, rem] = True
                # If there are no removable rings then you are not required to remove one
                if not removable_rings or (
                    len(removable_rings) == 1 and removable_rings[0] == put_index
                ):
                    rem = self.width**2
                    moves[m, put, rem] = True
        return moves

    def get_capture_moves(self):
        # Return a boolean matrix of size (6 x w x w) with the value True at
        # every index that corresponds to a valid capture action.
        # The six directions are given by self.DIRECTIONS
        moves = np.zeros((6, self.width, self.width), dtype=bool)

        # Create list of the indices of marbles that can be used to capture
        if np.sum(self.state[self.CAPTURE_LAYER]) == 1:
            occupied_rings = zip(*np.where(self.state[self.CAPTURE_LAYER] == 1))
        else:
            occupied_rings = zip(
                *np.where(np.sum(self.state[self.MARBLE_LAYERS], axis=0) == 1)
            )

        # Update matrix with all possible capture directions from each capturing marble
        for src_index in occupied_rings:
            src_y, src_x = src_index
            neighbors = self.get_neighbors(src_index)
            for direction, neighbor in enumerate(neighbors):
                # Check each neighbor to see if it has a marble and the jump destination is empty
                y, x = neighbor
                if (
                    self._is_inbounds(neighbor)
                    and np.sum(self.state[self.MARBLE_LAYERS, y, x]) == 1
                ):
                    dst_index = self.get_jump_destination(src_index, neighbor)
                    y, x = dst_index
                    if (
                        self._is_inbounds(dst_index)
                        and np.sum(self.state[self.BOARD_LAYERS, y, x]) == 1
                    ):
                        # Set this move as a valid action in the filter matrix
                        moves[direction, src_y, src_x] = True
        return moves

    def _get_open_rings(self):
        # Return a list of indices for all of the open rings in the main (connected) region
        # Frozen isolated regions are excluded per official rules

        # Get all vacant rings
        all_open = list(
            zip(*np.where(np.sum(self.state[self.BOARD_LAYERS], axis=0) == 1))
        )

        # Check if there are multiple regions (some might be frozen/isolated)
        regions = self._get_regions()
        if len(regions) <= 1:
            # Single connected region - all open rings are valid
            return all_open

        # Multiple regions exist - only return rings from the largest (main) region
        main_region = max(regions, key=len)
        open_rings = [ring for ring in all_open if ring in main_region]
        return open_rings

    def _is_removable(self, index):
        # Check if the ring at index is removable. A ring is removable if two of its neighbors
        # in a row are missing and the ring itself is empty.
        y, x = index
        if np.sum(self.state[self.BOARD_LAYERS, y, x]) != 1:
            return False
        neighbors = self.get_neighbors(index)
        # Add the first neighbor index to the end so that if the first and last are both empty then it still passes
        neighbors.append(neighbors[0])
        # Track the number of consecutive empty neighboring rings
        adjacent_empty = 0
        for neighbor in neighbors:
            if (
                self._is_inbounds(neighbor)
                and self.state[self.RING_LAYER][neighbor] == 1
            ):
                # If the neighbor index is in bounds and not removed then reset the empty counter
                adjacent_empty = 0
            else:
                adjacent_empty += 1
                if adjacent_empty >= 2:
                    return True
        return False

    def _get_removable_rings(self):
        # Return a list of indices to rings that can be removed
        removable = [
            index for index in self._get_open_rings() if self._is_removable(index)
        ]
        return removable

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

    def _transform_state_hex(self, state, rot60_k=0, mirror=False, mirror_first=False):
        """Apply rotation and/or mirror to the whole SPATIAL state.

        Args:
            state: Board state to transform
            rot60_k: Number of 60° rotations (0-5 for D6, 0,2,4 for D3)
            mirror: Whether to apply mirror transformation
            mirror_first: If True, mirror then rotate. If False, rotate then mirror.
        """
        self._build_axial_maps()
        out = np.zeros_like(state)
        for (y, x), (q, r) in self._yx_to_ax.items():
            if mirror_first:
                # Mirror first, then rotate (for R{k}M transforms)
                if mirror:
                    q2, r2 = self._ax_mirror_q_axis(q, r)
                else:
                    q2, r2 = q, r
                q2, r2 = self._ax_rot60(q2, r2, rot60_k)
            else:
                # Rotate first, then mirror (for MR{k} transforms)
                q2, r2 = self._ax_rot60(q, r, rot60_k)
                if mirror:
                    q2, r2 = self._ax_mirror_q_axis(q2, r2)
            dst = self._ax_to_yx.get((q2, r2))
            if dst is not None:
                y2, x2 = dst
                out[:, y2, x2] = state[:, y, x]
        return out

    # ======================  D6 / D3 SYMMETRY ENUMERATION  =================

    def _get_all_symmetry_transforms(self):
        """Return (name, fn) pairs for all valid symmetries of this board.

        Returns all 12 (D6) or 6 (D3) transformations:
        - R{k}: Pure rotations
        - MR{k}: Rotate then mirror
        - R{k}M: Mirror then rotate
        """
        if self.rings in (self.SMALL_BOARD_37, self.LARGE_BOARD_61):
            # full hex (D6): 6 rotations + 12 mirror combinations
            rot_steps = [0, 1, 2, 3, 4, 5]  # multiples of 60°
        else:
            # alternating 4/5 sides (48): D3 subset
            rot_steps = [0, 2, 4]  # 0°, 120°, 240°

        ops = []
        # Pure rotations
        for k in rot_steps:
            ops.append(
                (
                    f"R{60 * k}",
                    lambda s, kk=k: self._transform_state_hex(
                        s, rot60_k=kk, mirror=False
                    ),
                )
            )
        # Rotate then mirror (MR{k})
        for k in rot_steps:
            ops.append(
                (
                    f"MR{60 * k}",
                    lambda s, kk=k: self._transform_state_hex(
                        s, rot60_k=kk, mirror=True, mirror_first=False
                    ),
                )
            )
        # Mirror then rotate (R{k}M)
        for k in rot_steps:
            ops.append(
                (
                    f"R{60 * k}M",
                    lambda s, kk=k: self._transform_state_hex(
                        s, rot60_k=kk, mirror=True, mirror_first=True
                    ),
                )
            )
        return ops

    def _canonical_key(self, state):
        """Lexicographic key over valid cells only (rings+marbles now)."""
        masked = (state[self.BOARD_LAYERS] * self.board_layout).astype(np.uint8)
        return masked.tobytes()

    def canonicalize_state(self):
        """
        Return (canonical_state, transform_name, inverse_name).
        - transform_name is the symmetry that was applied to reach canonical.
        - inverse_name is what to apply to map policy logits back to original.
        """
        best_name = "R0"
        best_state = np.copy(self.state)
        best_key = self._canonical_key(best_state)

        for name, fn in self._get_all_symmetry_transforms():
            s2 = fn(self.state)
            key = self._canonical_key(s2)
            if key < best_key:
                best_key, best_state, best_name = key, s2, name

        inv = self._get_inverse_transform(best_name)

        return best_state, best_name, inv

    # ======================  ACTION / MASK TRANSFORMERS  ===================
    def _get_inverse_transform(self, transform_name):
        """
        Get the inverse of a symmetry transform.

        For hex boards with D6 symmetry (37, 61 rings): 18 total (6 rot + 6 MR + 6 RM)
        For boards with D3 symmetry (48 rings): 9 total (3 rot + 3 MR + 3 RM)

        Inverse relationships:
        - R(k)⁻¹ = R(-k mod 360°)
        - MR(k)⁻¹ = R(-k mod 360°)M  (rotate-then-mirror inverts to mirror-then-rotate)
        - R(k)M⁻¹ = MR(-k mod 360°)  (mirror-then-rotate inverts to rotate-then-mirror)

        Args:
            transform_name: String like "R60", "MR120", "R240M", etc.

        Returns:
            String naming the inverse transform
        """
        if transform_name.endswith("M") and not transform_name.startswith("MR"):
            # R{k}M (mirror-then-rotate): inverse is MR{-k} (rotate-then-mirror)
            # Extract angle from name (e.g., "R120M" -> 120)
            angle = int(transform_name[1:-1])  # Strip "R" and "M"
            # Inverse angle in full circle
            inv_angle = (360 - angle) % 360
            return f"MR{inv_angle}"

        elif transform_name.startswith("MR"):
            # MR{k} (rotate-then-mirror): inverse is R{-k}M (mirror-then-rotate)
            # Extract angle from name (e.g., "MR120" -> 120)
            angle = int(transform_name[2:])
            # Inverse angle in full circle
            inv_angle = (360 - angle) % 360
            return f"R{inv_angle}M"

        elif transform_name.startswith("R"):
            # Pure rotation: inverse is opposite rotation
            angle = int(transform_name[1:])
            # Inverse angle in full circle
            inv_angle = (360 - angle) % 360
            return f"R{inv_angle}"

        else:
            raise ValueError(f"Unknown transform name format: {transform_name}")

    def _dir_index_map(self, rot60_k=0, mirror=False):
        """
        Map capture direction indices (0..5) under the same symmetry used for state.
        We transform a direction vector v=(dy,dx) via axial:
           (dq,dr) = (dx, dy-dx)
           rotate/mirror in axial
           back to (dy',dx') with: dx' = dq, dy' = dr + dq
        """
        # Original direction vectors in your order:
        dirs = self.DIRECTIONS

        def xform_vec(dy, dx):
            dq, dr = dx, (dy - dx)
            dq, dr = self._ax_rot60(dq, dr, rot60_k)
            if mirror:
                dq, dr = self._ax_mirror_q_axis(dq, dr)
            dx2 = dq
            dy2 = dr + dq
            return (dy2, dx2)

        idx_map = {}
        for i, v in enumerate(dirs):
            vv = xform_vec(*v)
            j = next(k for k, u in enumerate(dirs) if u == vv)
            idx_map[i] = j
        return idx_map

    def transform_capture_mask(self, cap_mask, rot60_k=0, mirror=False):
        """
        cap_mask shape: (6, H, W). Returns transformed mask matching state xform.
        """
        self._build_axial_maps()
        out = np.zeros_like(cap_mask)
        dmap = self._dir_index_map(rot60_k, mirror)

        for (y, x), (q, r) in self._yx_to_ax.items():
            q2, r2 = self._ax_rot60(q, r, rot60_k)
            if mirror:
                q2, r2 = self._ax_mirror_q_axis(q2, r2)
            dst = self._ax_to_yx.get((q2, r2))
            if dst is None:
                continue
            y2, x2 = dst
            for d in range(6):
                out[dmap[d], y2, x2] = cap_mask[d, y, x]
        return out

    def transform_put_mask(self, put_mask, rot60_k=0, mirror=False):
        """
        put_mask shape: (3, W*W, W*W+1). Applies same symmetry to (put, rem) indices.
        """
        self._build_axial_maps()
        out = np.zeros_like(put_mask)

        # Build flat-index permutation for valid cells
        flat_map = {}  # src_flat -> dst_flat
        for (y, x), (q, r) in self._yx_to_ax.items():
            q2, r2 = self._ax_rot60(q, r, rot60_k)
            if mirror:
                q2, r2 = self._ax_mirror_q_axis(q2, r2)
            dst = self._ax_to_yx.get((q2, r2))
            if dst is None:
                continue
            y2, x2 = dst
            flat_map[y * self.width + x] = y2 * self.width + x2

        M, P, R = put_mask.shape
        last = R - 1  # "no ring removed" slot stays put

        for m in range(M):
            for p in range(P):
                p2 = flat_map.get(p)
                if p2 is None:
                    continue
                # all concrete removals
                for r in range(R - 1):
                    r2 = flat_map.get(r)
                    if r2 is not None:
                        out[m, p2, r2] = put_mask[m, p, r]
                # "no remove" column
                out[m, p2, last] = put_mask[m, p, last]

        return out
