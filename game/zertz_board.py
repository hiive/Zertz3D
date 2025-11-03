import numpy as np

from game.zertz_position import ZertzPosition, ZertzPositionCollection
from game.utils.canonicalization import TransformFlags, CanonicalizationManager
from hiivelabs_mcts import (
    BoardConfig,
    coordinate_to_algebraic,
    generate_standard_layout_mask,
    get_neighbors,
    get_jump_destination,
    is_inbounds,
    get_regions,
    get_supply_index,
    get_captured_index,
    get_marble_type_at,
    get_valid_actions,
    get_placement_moves,
    get_capture_moves,
    get_open_rings,
    is_ring_removable,
    get_removable_rings,
    apply_placement_action,
)


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

    def __init__(self, rings=37, marbles=None, t=1, clone=None):
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
        """
        # Position cache manager (built lazily)
        self.positions = ZertzPositionCollection(self)

        # Canonicalization manager for state transformations
        self.canonicalizer = CanonicalizationManager(self)


        if clone is not None:
            self.rings = clone.rings
            self.config = clone.config
            self.t = clone.t
            self.CAPTURE_LAYER = clone.CAPTURE_LAYER
            self.state = np.copy(clone.state)
            self.global_state = np.copy(clone.global_state)
            self.last_acting_player = clone.last_acting_player if hasattr(clone, "last_acting_player") else None
        else:
            # Validate board size
            if rings not in [self.SMALL_BOARD_37, self.MEDIUM_BOARD_48, self.LARGE_BOARD_61]:
                raise ValueError(
                    f"Unsupported board size: {rings} rings. "
                    f"Supported sizes are {self.SMALL_BOARD_37}, {self.MEDIUM_BOARD_48}, and {self.LARGE_BOARD_61}."
                )

            # Get boolean layout mask from Rust (no Python duplication)
            self.config = BoardConfig.standard_config(rings, t=t)

            board_layout = generate_standard_layout_mask(rings, self.config.width)
            self.rings = int(board_layout.sum())

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
                (spatial_layers, self.config.width, self.config.width), dtype=np.float32
            )

            # Initialize GLOBAL state as 1d array
            # Global state vector (10 values):
            #   - 0-2: supply (white, gray, black) [0, 10]
            #   - 3-5: player 1 captured (white, gray, black) [0, 10]
            #   - 6-8: player 2 captured (white, gray, black) [0, 10]
            #   - 9: current player (0 or 1)
            # Use float32 for neural network compatibility
            self.global_state = np.zeros(10, dtype=np.float32)

            # Place rings using layout mask from Rust
            self.state[self.RING_LAYER, board_layout] = 1

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

    @staticmethod
    def get_middle_ring(src, dst):
        # Return the (y, x) index of the ring between src and dst
        y1, x1 = src
        y2, x2 = dst
        return (y1 + y2) // 2, (x1 + x2) // 2

    @staticmethod
    def capture_indices_to_action(direction, y, x, width, directions):
        """Convert capture mask indices to action tuple.

        Args:
            direction: Direction index from capture mask (0-5)
            y: Source y coordinate from capture mask
            x: Source x coordinate from capture mask
            width: Board width
            directions: Direction deltas (usually ZertzBoard.DIRECTIONS)

        Returns:
            Tuple (None, src_flat, dst_flat) for internal action format
        """
        src_flat = y * width + x
        dy, dx = directions[direction]
        dst_y = y + 2 * dy
        dst_x = x + 2 * dx
        dst_flat = dst_y * width + dst_x
        return (None, src_flat, dst_flat)

    def get_neighbors(self, index):
        """Return list of neighboring indices (delegates to zertz_logic).

        The neighboring index may not be within the board space so it must be checked
        that it is inbounds (see _is_inbounds).
        """
        y, x = index
        return get_neighbors(y, x, self.config)

    def _is_adjacent(self, l1, l2):
        # Return True if l1 and l2 are adjacent to each other on the hexagonal board
        return l2 in self.get_neighbors(l1)

    def _is_inbounds(self, index):
        """Check if index is in bounds (delegates to zertz_logic)."""
        y, x = index
        return is_inbounds(y, x, self.config.width)

    def _get_regions(self):
        """Return list of connected regions on the board.

        Delegates to get_regions() for zero code duplication.
        """
        return get_regions(self.state, self.config)

    def get_cur_player(self):
        return int(self.global_state[self.CUR_PLAYER])

    def _next_player(self):
        self.global_state[self.CUR_PLAYER] = (
            self.global_state[self.CUR_PLAYER] + 1
        ) % self.NUM_PLAYERS

    def _get_supply_index(self, marble_type):
        """Get global_state index for marble in supply (delegates to Rust)."""
        return get_supply_index(marble_type)

    def _get_captured_index(self, marble_type, player):
        """Get global_state index for captured marble (delegates to Rust)."""
        # Rust signature: get_captured_index(player: usize, marble_type: char)
        return get_captured_index(player, marble_type)

    def get_marble_type_at(self, index):
        """Get marble type at position (delegates to Rust)."""
        y, x = index
        return get_marble_type_at(self.state, y, x)

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
                - put_y, put_x: index for placement position
                - rem_y, rem_x: index for ring to remove (== put_y, put_x for no removal)

        Returns:
            List of captured marbles from isolation (or None if no captures)
        """
        # Parse action for validation
        type_index, put_y, put_x, rem_y, rem_x = action

        # Validate placement position before delegating to Rust
        if np.sum(self.state[self.BOARD_LAYERS, put_y, put_x]) != 1:
            raise ValueError(
                f"Invalid placement: position ({put_y}, {put_x}) is not an empty ring"
            )

        # # Validate marble availability before delegating to Rust
        # # (Rust would panic instead of raising ValueError)
        # marble_type = self.LAYER_TO_MARBLE[type_index + 1]
        # supply_idx = self._get_supply_index(marble_type)
        # if self.global_state[supply_idx] >= 1:
        #     # Marble available in supply - OK
        #     pass
        # elif np.all(self.global_state[self.SUPPLY_SLICE] == 0):
        #     # Entire supply pool is empty - can use captured marbles
        #     captured_idx = self._get_captured_index(marble_type, self.get_cur_player())
        #     if self.global_state[captured_idx] < 1:
        #         raise ValueError(
        #             f"No {marble_type} marbles available in supply or captured by player {self.get_cur_player()}"
        #         )
        # else:
        #     # This marble type is empty but pool has other marbles - cannot use captured marbles yet
        #     raise ValueError(
        #         f"No {marble_type} marbles in supply. Cannot use captured marbles until entire pool is empty."
        #     )

        # Delegate to Rust (which handles marble placement, ring removal,
        # isolation capture, supply/captured pool management, and player switching)
        # Returns list of captured marble positions from isolation
        # Unpack action tuple for Rust function signature
        # type_index, put_loc, rem_loc = action
        # put_y, put_x = divmod(put_loc, self.config.width)
        if (put_y, put_x) == (rem_y, rem_x):
            # No removal - Rust uses None for Option<usize>
            rem_y, rem_x = None, None

        captured_positions = apply_placement_action(
            self.state,
            self.global_state,
            type_index,
            put_y,
            put_x,
            rem_y,
            rem_x,
            self.config
        )

        # Convert captured positions to expected format for ActionResult
        if captured_positions:
            captured_marbles = []
            for marble_layer, y, x in captured_positions:
                # Convert marble layer to marble type
                marble_type = self.LAYER_TO_MARBLE[marble_layer]
                # Convert (y, x) to board position string
                # Use Rust coordinate_to_algebraic() instead of position_from_yx()
                # because position_from_yx() only works for positions in the standard
                # layout collection, and tests may create custom topologies
                pos = coordinate_to_algebraic(y, x, self.config)
                # print(f"[DEBUG] Isolation capture: marble_layer={marble_layer}, y={y}, x={x}, pos={pos}, marble_type={marble_type}")
                captured_marbles.append({"marble": marble_type, "pos": pos})
            return captured_marbles
        return None

    def _take_capture_action(self, action):
        # Capture actions: (None, src_flat, dst_flat)
        # Unflatten both coordinates and calculate captured marble position as midpoint
        _, src_flat, dst_flat = action
        src_y, src_x = divmod(src_flat, self.config.width)
        dst_y, dst_x = divmod(dst_flat, self.config.width)

        src_index = (src_y, src_x)
        dst_index = (dst_y, dst_x)
        marble_type = self.get_marble_type_at(src_index)

        # Calculate captured marble position (midpoint between src and dst)
        cap_y = (src_y + dst_y) // 2
        cap_x = (src_x + dst_x) // 2
        cap_index = (cap_y, cap_x)
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
            ny, nx = neighbor
            # Check each neighbor to see if it has a marble
            if (
                self._is_inbounds(neighbor)
                and np.sum(self.state[self.MARBLE_LAYERS, ny, nx]) == 1
            ):
                dy, dx = dst_index
                next_dst = get_jump_destination(dy, dx, ny, nx)
                ky, kx = next_dst
                if (
                    self._is_inbounds(next_dst)
                    and np.sum(self.state[self.BOARD_LAYERS, ky, kx]) == 1
                ):
                    # Set the captured layer to 1 at dst_index
                    self.state[self.CAPTURE_LAYER][dst_index] = 1
                    break

        # Update current player if there are no forced chain captures
        if np.sum(self.state[self.CAPTURE_LAYER]) == 0:
            self._next_player()
        return captured_type


    def get_valid_actions(self):
        """Return valid placement and capture moves.

        Delegates to get_valid_actions() for zero code duplication.
        """
        return get_valid_actions(self.state, self.global_state, self.config)

    def get_placement_moves(self):
        """Return valid placement moves.

        Delegates to get_placement_moves() for zero code duplication.
        """
        return get_placement_moves(self.state, self.global_state, self.config)

    def get_capture_moves(self):
        """Return valid capture moves.

        Delegates to get_capture_moves() for zero code duplication.
        """
        return get_capture_moves(self.state, self.config)

    #todo only used in test
    def _get_open_rings(self):
        """Return indices of all empty rings on the board.

        Delegates to get_open_rings() for zero code duplication.
        """
        return get_open_rings(self.state, self.config)

    # todo unused
    def _is_removable(self, index):
        """Check if ring at index can be removed.

        Delegates to zertz_logic.is_removable() for zero code duplication.
        """
        return is_ring_removable(index, self.state, self.config)

    # todo unused
    def _get_removable_rings(self):
        """Return list of removable ring indices.

        Delegates to get_removable_rings() for zero code duplication.
        """
        return get_removable_rings(self.state, self.config)

    def _compute_label(self, index: tuple[int, int]) -> str:
        """Generate position label using Rust coordinate conversion."""
        y, x = index
        return coordinate_to_algebraic(y, x, self.config)

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
