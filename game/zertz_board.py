import numpy as np

from game.zertz_position import ZertzPosition, ZertzPositionCollection
from game.utils.canonicalization import TransformFlags, CanonicalizationManager
from game import zertz_logic


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
            # Use float32 for neural network compatibility (one-hot encoding)
            self.state = np.zeros(
                (spatial_layers, self.width, self.width), dtype=np.float32
            )

            # Initialize GLOBAL state as 1d array
            # Global state vector (10 values):
            #   - 0-2: supply (white, gray, black) [0, 10]
            #   - 3-5: player 1 captured (white, gray, black) [0, 10]
            #   - 6-8: player 2 captured (white, gray, black) [0, 10]
            #   - 9: current player (0 or 1)
            # Use float32 for neural network compatibility
            self.global_state = np.zeros(10, dtype=np.float32)

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
        # Use Rust BoardConfig constructor
        return zertz_logic.BoardConfig.standard_config(rings=self.rings, t=self.t)

    @staticmethod
    def get_middle_ring(src, dst):
        # Return the (y, x) index of the ring between src and dst
        y1, x1 = src
        y2, x2 = dst
        return (y1 + y2) // 2, (x1 + x2) // 2

    def get_neighbors(self, index):
        """Return list of neighboring indices (delegates to zertz_logic).

        The neighboring index may not be within the board space so it must be checked
        that it is inbounds (see _is_inbounds).
        """
        config = self._get_config()
        return zertz_logic.get_neighbors(index, config)

    def _is_adjacent(self, l1, l2):
        # Return True if l1 and l2 are adjacent to each other on the hexagonal board
        return l2 in self.get_neighbors(l1)

    @staticmethod
    def get_jump_destination(start, cap):
        """Return landing index after jump (delegates to zertz_logic).

        The landing index may not be within the board space so it must be checked
        that it is inbounds (see _is_inbounds).
        """
        return zertz_logic.get_jump_destination(start, cap)

    def _is_inbounds(self, index):
        """Check if index is in bounds (delegates to zertz_logic)."""
        config = self._get_config()
        return zertz_logic.is_inbounds(index, config)

    def _get_regions(self):
        """Return list of connected regions on the board.

        Delegates to zertz_logic.get_regions() for zero code duplication.
        """
        config = self._get_config()
        return zertz_logic.get_regions(self.state, config)

    def get_cur_player(self):
        return int(self.global_state[self.CUR_PLAYER])

    def _next_player(self):
        self.global_state[self.CUR_PLAYER] = (
            self.global_state[self.CUR_PLAYER] + 1
        ) % self.NUM_PLAYERS

    def _get_supply_index(self, marble_type):
        """Get global_state index for marble in supply (delegates to zertz_logic)."""
        config = self._get_config()
        return zertz_logic.get_supply_index(marble_type, config)

    def _get_captured_index(self, marble_type, player):
        """Get global_state index for captured marble (delegates to zertz_logic)."""
        config = self._get_config()
        return zertz_logic.get_captured_index(marble_type, player, config)

    def get_marble_type_at(self, index):
        """Get marble type at position (delegates to zertz_logic)."""
        config = self._get_config()
        return zertz_logic.get_marble_type_at(index, self.state, config)

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
            return self._take_placement_action(action)
        elif action_type == "CAP":
            return self._take_capture_action(action)

    def _take_placement_action(self, action):
        """Execute a placement action (delegates to stateless Rust implementation).

        Args:
            action: (type_index, put_loc, rem_loc) tuple
                - type_index: 0=white, 1=gray, 2=black
                - put_loc: flat index for placement position
                - rem_loc: flat index for ring to remove (or width² for no removal)

        Returns:
            List of captured marbles from isolation (or None if no captures)
        """
        # Parse action for validation
        type_index, put_loc, rem_loc = action
        put_y, put_x = divmod(put_loc, self.width)

        # Validate placement position before delegating to Rust
        if np.sum(self.state[self.BOARD_LAYERS, put_y, put_x]) != 1:
            raise ValueError(
                f"Invalid placement: position ({put_y}, {put_x}) is not an empty ring"
            )

        # Validate marble availability before delegating to Rust
        # (Rust would panic instead of raising ValueError)
        marble_type = self.LAYER_TO_MARBLE[type_index + 1]
        supply_idx = self._get_supply_index(marble_type)
        if self.global_state[supply_idx] >= 1:
            # Marble available in supply - OK
            pass
        elif np.all(self.global_state[self.SUPPLY_SLICE] == 0):
            # Entire supply pool is empty - can use captured marbles
            captured_idx = self._get_captured_index(marble_type, self.get_cur_player())
            if self.global_state[captured_idx] < 1:
                raise ValueError(
                    f"No {marble_type} marbles available in supply or captured by player {self.get_cur_player()}"
                )
        else:
            # This marble type is empty but pool has other marbles - cannot use captured marbles yet
            raise ValueError(
                f"No {marble_type} marbles in supply. Cannot use captured marbles until entire pool is empty."
            )

        # Delegate to Rust (which handles marble placement, ring removal,
        # isolation capture, supply/captured pool management, and player switching)
        # Returns list of captured marble positions from isolation
        config = self._get_config()
        captured_positions = zertz_logic.apply_placement_action(
            self.state,
            self.global_state,
            action,  # Pass the full action tuple
            config
        )

        # Convert captured positions to expected format for ActionResult
        if captured_positions:
            captured_marbles = []
            for marble_layer, y, x in captured_positions:
                # Convert marble layer to marble type
                marble_type = self.LAYER_TO_MARBLE[marble_layer]
                # Convert (y, x) to board position string
                # Note: Don't use index_to_str() because the ring has already been removed
                # by the Rust code, so we need to get the label directly
                pos = self.position_from_yx((y, x)).label
                print(f"[DEBUG] Isolation capture: marble_layer={marble_layer}, y={y}, x={x}, pos={pos}, marble_type={marble_type}")
                captured_marbles.append({"marble": marble_type, "pos": pos})
            return captured_marbles
        return None

    def _take_capture_action(self, action):
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

        Delegates to zertz_logic.get_valid_actions() for zero code duplication.
        """
        config = self._get_config()
        return zertz_logic.get_valid_actions(self.state, self.global_state, config)

    #todo check
    def get_placement_shape(self):
        # get shape of placement moves as a tuple
        return 3, self.width**2, self.width**2 + 1

    #todo check
    def get_capture_shape(self):
        # get shape of capture moves as a tuple
        return 6, self.width, self.width

    def get_placement_moves(self):
        """Return valid placement moves.

        Delegates to zertz_logic.get_placement_moves() for zero code duplication.
        """
        config = self._get_config()
        return zertz_logic.get_placement_moves(self.state, self.global_state, config)

    def get_capture_moves(self):
        """Return valid capture moves.

        Delegates to zertz_logic.get_capture_moves() for zero code duplication.
        """
        config = self._get_config()
        return zertz_logic.get_capture_moves(self.state, self.global_state, config)

    #todo only used in test
    def _get_open_rings(self):
        """Return indices of all empty rings on the board.

        Delegates to zertz_logic.get_open_rings() for zero code duplication.
        """
        config = self._get_config()
        return zertz_logic.get_open_rings(self.state, config)

    # todo unused
    def _is_removable(self, index):
        """Check if ring at index can be removed.

        Delegates to zertz_logic.is_removable() for zero code duplication.
        """
        config = self._get_config()
        return zertz_logic.is_ring_removable(index, self.state, config)

    # todo unused
    def _get_removable_rings(self):
        """Return list of removable ring indices.

        Delegates to zertz_logic.get_removable_rings() for zero code duplication.
        """
        config = self._get_config()
        return zertz_logic.get_removable_rings(self.state, config)

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

    def label_to_yx(self, index_str: str) -> tuple[int, int]:
        self._ensure_positions_built()
        pos = self.positions.get_by_label(index_str)
        if pos is None:
            raise ValueError(f"Coordinate '{index_str}' not found in board layout")
        return pos.yx

    def yx_to_label(self, index: tuple[int, int]) -> str:
        self._ensure_positions_built()
        pos = self.positions.get_by_yx(index)
        return pos.label if pos else ""

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
        return self.label_to_yx(index_str)

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

    # =========================  CANONICALIZATION  =========================

    def _build_axial_maps(self):
        """Ensure axial coordinate mappings are built (for canonicalization/tests)."""
        self._ensure_positions_built()

    @property
    def _yx_to_ax(self):
        """Access (y,x) → (q,r) mapping (for canonicalization)."""
        return self.positions.yx_to_ax

    @property
    def _ax_to_yx(self):
        """Access (q,r) → (y,x) mapping (for canonicalization)."""
        return self.positions.ax_to_yx

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
