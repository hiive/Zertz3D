"""Stateless game logic for Zertz.

All functions are pure: same inputs → same outputs.
No side effects, no mutations, no hidden state.

This enables:
- Efficient neural network batching
- Parallelization across multiple cores/GPUs
- Memory-efficient MCTS (store arrays, not objects)
- Reproducible simulations

Architecture:
    BoardConfig: Immutable configuration (constants, dimensions, rules)
    Pure functions: Take (state, config) → return new state or actions

Usage:
    config = BoardConfig.standard(rings=37)
    placement, capture = get_valid_actions(board_state, global_state, config)
    new_board, new_global = apply_action(board_state, global_state, action, config)
"""

from collections import deque
from typing import Dict, NamedTuple, Tuple
import numpy as np


class BoardConfig(NamedTuple):
    """Immutable board configuration.

    Contains all constants needed for stateless game logic.
    Shared across all nodes in MCTS tree - only created once.
    """
    # Board dimensions
    width: int
    rings: int
    t: int  # Time history depth

    # Hex directions (y, x) offsets
    directions: Tuple[Tuple[int, int], ...]  # ((1,0), (0,-1), (-1,-1), (-1,0), (0,1), (1,1))

    # Layer indices
    ring_layer: int  # 0
    marble_layers: slice  # slice(1, 4)
    board_layers: slice  # slice(0, 4)
    capture_layer: int  # t * 4
    layers_per_timestep: int  # 4

    # Global state indices
    supply_w: int  # 0
    supply_g: int  # 1
    supply_b: int  # 2
    p1_cap_w: int  # 3
    p1_cap_g: int  # 4
    p1_cap_b: int  # 5
    p2_cap_w: int  # 6
    p2_cap_g: int  # 7
    p2_cap_b: int  # 8
    cur_player: int  # 9

    # Slices
    supply_slice: slice  # slice(0, 3)
    p1_cap_slice: slice  # slice(3, 6)
    p2_cap_slice: slice  # slice(6, 9)

    # Player constants
    player_1: int  # 0
    player_2: int  # 1
    num_players: int  # 2

    # Marble type mappings
    marble_to_layer: dict  # {"w": 1, "g": 2, "b": 3}
    layer_to_marble: dict  # {1: "w", 2: "g", 3: "b"}

    # Optional: board layout for custom boards
    board_layout: np.ndarray = None  # Boolean mask of valid positions

    @classmethod
    def standard(cls, rings=37, t=1):
        """Create standard BoardConfig for common board sizes.

        Args:
            rings: 37, 48, or 61
            t: Time history depth

        Returns:
            BoardConfig instance
        """
        # Map rings to width
        width_map = {37: 7, 48: 8, 61: 9}
        if rings not in width_map:
            raise ValueError(f"Unsupported ring count: {rings}. Use 37, 48, or 61.")

        width = width_map[rings]
        marble_to_layer = {"w": 1, "g": 2, "b": 3}
        layer_to_marble = {1: "w", 2: "g", 3: "b"}

        return cls(
            width=width,
            rings=rings,
            t=t,
            directions=((1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)),
            ring_layer=0,
            marble_layers=slice(1, 4),
            board_layers=slice(0, 4),
            capture_layer=t * 4,
            layers_per_timestep=4,
            supply_w=0,
            supply_g=1,
            supply_b=2,
            p1_cap_w=3,
            p1_cap_g=4,
            p1_cap_b=5,
            p2_cap_w=6,
            p2_cap_g=7,
            p2_cap_b=8,
            cur_player=9,
            supply_slice=slice(0, 3),
            p1_cap_slice=slice(3, 6),
            p2_cap_slice=slice(6, 9),
            player_1=0,
            player_2=1,
            num_players=2,
            marble_to_layer=marble_to_layer,
            layer_to_marble=layer_to_marble,
            board_layout=None,
        )


# ============================================================================
# PURE HELPER FUNCTIONS
# ============================================================================

def is_inbounds(index: Tuple[int, int], config: BoardConfig) -> bool:
    """Check if index is within board bounds."""
    y, x = index
    return 0 <= y < config.width and 0 <= x < config.width


def get_neighbors(index: Tuple[int, int], config: BoardConfig) -> list:
    """Get list of neighboring indices (may be out of bounds)."""
    y, x = index
    return [(y + dy, x + dx) for dy, dx in config.directions]


def get_jump_destination(start: Tuple[int, int], cap: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate landing position after capturing marble at cap from start."""
    sy, sx = start
    cy, cx = cap
    dy = (cy - sy) * 2
    dx = (cx - sx) * 2
    return sy + dy, sx + dx


def get_regions(board_state: np.ndarray, config: BoardConfig) -> list:
    """Find all connected regions on the board.

    Pure function version of ZertzBoard._get_regions()

    Returns:
        List of regions, where each region is a list of (y, x) indices
    """
    regions = []
    not_visited = set(zip(*np.where(board_state[config.ring_layer] == 1)))

    while not_visited:
        region = []
        queue = deque()
        queue.appendleft(not_visited.pop())

        while queue:
            index = queue.pop()
            region.append(index)

            for neighbor in get_neighbors(index, config):
                if (neighbor in not_visited and
                    is_inbounds(neighbor, config) and
                    board_state[config.ring_layer][neighbor] != 0):
                    not_visited.remove(neighbor)
                    queue.appendleft(neighbor)

        regions.append(region)

    return regions


def get_open_rings(board_state: np.ndarray, config: BoardConfig, regions: list = None) -> list:
    """Get list of empty ring indices across the entire board.

    Pure function version of ZertzBoard._get_open_rings()

    Args:
        board_state: Board state array
        config: BoardConfig
        regions: Unused (retained for API compatibility)
    """
    # Vacant rings are those with a ring present but no marble on top
    return list(
        zip(*np.where(np.sum(board_state[config.board_layers], axis=0) == 1))
    )


# ============================================================================
# ACTION TRANSFORMATIONS (used for symmetry-aware move handling)
# ============================================================================

_LAYOUT_MASK_CACHE: Dict[Tuple[int, int, bytes | None], np.ndarray] = {}
_AXIAL_CACHE: Dict[Tuple[int, int, bytes | None], Tuple[Dict[Tuple[int, int], Tuple[int, int]], Dict[Tuple[int, int], Tuple[int, int]]]] = {}


def _generate_standard_layout_mask(rings: int) -> np.ndarray:
    """Generate boolean layout mask for standard boards (37/48/61)."""
    width_map = {37: 7, 48: 8, 61: 9}
    if rings not in width_map:
        raise ValueError(f"Unsupported board size for standard layout: {rings}")

    width = width_map[rings]
    mask = np.zeros((width, width), dtype=bool)

    if rings == 37:
        letters = "ABCDEFG"
    elif rings == 48:
        letters = "ABCDEFGH"
    else:  # 61
        letters = "ABCDEFGHJ"

    r_max = len(letters)
    is_even = r_max % 2 == 0

    def h_max(idx: int) -> int:
        return r_max - abs(letters.index(letters[idx]) - (r_max // 2))

    r_min = h_max(0)
    if is_even:
        r_min += 1

    for row_idx in range(r_max):
        hh = h_max(row_idx)
        letters_row = letters[:hh] if row_idx < hh / 2 else letters[-hh:]
        num_max = r_max - row_idx
        num_min = max(r_min - row_idx, 1)

        for k, letter in enumerate(letters_row):
            col = min(k + num_min, num_max)
            col_idx = letters.find(letter)
            mask[row_idx, col_idx] = True

    return mask


def _get_layout_mask(config: BoardConfig) -> np.ndarray:
    """Return boolean mask for valid board cells."""
    layout = config.board_layout
    if layout is not None:
        return layout.astype(bool)

    key = (int(config.rings), int(config.width), None)
    cached = _LAYOUT_MASK_CACHE.get(key)
    if cached is None:
        mask = _generate_standard_layout_mask(int(config.rings))
        _LAYOUT_MASK_CACHE[key] = mask
        return mask
    return cached


def _build_axial_maps(config: BoardConfig) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], Dict[Tuple[int, int], Tuple[int, int]]]:
    """Build mapping between (y, x) and axial coordinates."""
    layout = _get_layout_mask(config)
    layout_bytes = layout.astype(np.uint8).tobytes()
    key = (int(config.rings), int(config.width), layout_bytes)
    cached = _AXIAL_CACHE.get(key)
    if cached is not None:
        return cached

    ys, xs = np.where(layout)
    width = layout.shape[0]
    c = width // 2
    sqrt3 = np.sqrt(3.0)

    records = []
    for y, x in zip(ys, xs):
        q = x - c
        r = y - x
        xc = sqrt3 * (q + r / 2.0)
        yc = 1.5 * r
        records.append((int(y), int(x), q, r, xc, yc))

    xc_center = sum(rec[4] for rec in records) / len(records)
    yc_center = sum(rec[5] for rec in records) / len(records)
    q_center = (sqrt3 / 3.0) * xc_center - (1.0 / 3.0) * yc_center
    r_center = (2.0 / 3.0) * yc_center
    scale = 3 if int(config.rings) == 48 else 1

    yx_to_ax: Dict[Tuple[int, int], Tuple[int, int]] = {}
    ax_to_yx: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for y, x, q, r, _, _ in records:
        q_centered = q - q_center
        r_centered = r - r_center
        q_adj = int(round(scale * q_centered))
        r_adj = int(round(scale * r_centered))
        axial = (q_adj, r_adj)
        yx_to_ax[(y, x)] = axial
        ax_to_yx[axial] = (y, x)

    _AXIAL_CACHE[key] = (yx_to_ax, ax_to_yx)
    return yx_to_ax, ax_to_yx


def _ax_rot60(q: int, r: int, k: int = 1) -> Tuple[int, int]:
    """Rotate axial coordinate by k * 60° counterclockwise."""
    k %= 6
    for _ in range(k):
        q, r = -r, q + r
    return q, r


def _ax_mirror_q_axis(q: int, r: int) -> Tuple[int, int]:
    """Mirror axial coordinate across the q-axis."""
    return q, -q - r


def _transform_coordinate(y: int, x: int, rot60_k: int, mirror: bool, mirror_first: bool, config: BoardConfig) -> Tuple[int, int]:
    """Apply rotation/mirror to a coordinate (capture behaviour, honours mirror order)."""
    yx_to_ax, ax_to_yx = _build_axial_maps(config)
    key = (int(y), int(x))
    if key not in yx_to_ax:
        raise ValueError(f"Coordinate {(y, x)} is not on the board")

    q, r = yx_to_ax[key]
    if mirror_first:
        if mirror:
            q, r = _ax_mirror_q_axis(q, r)
        q, r = _ax_rot60(q, r, rot60_k)
    else:
        q, r = _ax_rot60(q, r, rot60_k)
        if mirror:
            q, r = _ax_mirror_q_axis(q, r)

    dest = ax_to_yx.get((q, r))
    if dest is None:
        raise ValueError("Rotated coordinate is not on the board")
    return dest


def _transform_coordinate_put(y: int, x: int, rot60_k: int, mirror: bool, config: BoardConfig) -> Tuple[int, int]:
    """Apply rotation/mirror to placement coordinate (rotation then optional mirror)."""
    yx_to_ax, ax_to_yx = _build_axial_maps(config)
    key = (int(y), int(x))
    if key not in yx_to_ax:
        raise ValueError(f"Coordinate {(y, x)} is not on the board")

    q, r = yx_to_ax[key]
    q, r = _ax_rot60(q, r, rot60_k)
    if mirror:
        q, r = _ax_mirror_q_axis(q, r)

    dest = ax_to_yx.get((q, r))
    if dest is None:
        raise ValueError("Rotated coordinate is not on the board")
    return dest


def _dir_index_map(rot60_k: int, mirror: bool, mirror_first: bool, config: BoardConfig) -> Dict[int, int]:
    """Map direction indices under rotation/mirror."""
    mapping = {}
    for idx, (dy, dx) in enumerate(config.directions):
        dq = dx
        dr = dy - dx
        if mirror_first:
            if mirror:
                dq, dr = _ax_mirror_q_axis(dq, dr)
            dq, dr = _ax_rot60(dq, dr, rot60_k)
        else:
            dq, dr = _ax_rot60(dq, dr, rot60_k)
            if mirror:
                dq, dr = _ax_mirror_q_axis(dq, dr)
        new_dx = dq
        new_dy = dr + dq
        try:
            new_idx = config.directions.index((new_dy, new_dx))
        except ValueError as exc:
            raise ValueError(f"Transformed direction {(new_dy, new_dx)} not in direction set") from exc
        mapping[idx] = new_idx
    return mapping


def _translate_action(action, dy: int, dx: int, config: BoardConfig):
    """Translate action coordinates by (dy, dx)."""
    layout = _get_layout_mask(config)
    width = config.width
    action_type, payload = action

    if action_type == "PUT":
        marble_idx, put_flat, rem_flat = payload
        py, px = divmod(int(put_flat), width)
        new_py, new_px = py + dy, px + dx
        if not (0 <= new_py < width and 0 <= new_px < width and layout[new_py, new_px]):
            raise ValueError(f"Translation moves placement {(py, px)} outside board")

        if rem_flat == width * width:
            new_rem_flat = rem_flat
        else:
            ry, rx = divmod(int(rem_flat), width)
            new_ry, new_rx = ry + dy, rx + dx
            if not (0 <= new_ry < width and 0 <= new_rx < width and layout[new_ry, new_rx]):
                raise ValueError(f"Translation moves removal {(ry, rx)} outside board")
            new_rem_flat = new_ry * width + new_rx

        new_put_flat = new_py * width + new_px
        return "PUT", (marble_idx, new_put_flat, new_rem_flat)

    if action_type == "CAP":
        direction, y, x = payload
        new_y, new_x = int(y) + dy, int(x) + dx
        if not (0 <= new_y < width and 0 <= new_x < width and layout[new_y, new_x]):
            raise ValueError(f"Translation moves capture {(y, x)} outside board")
        return "CAP", (direction, new_y, new_x)

    if action_type == "PASS":
        return action

    raise ValueError(f"Unknown action type: {action_type}")


def _apply_orientation(action, rot60_k: int, mirror: bool, mirror_first: bool, config: BoardConfig):
    """Apply rotation/mirror orientation components to an action."""
    action_type, payload = action
    width = config.width

    if action_type == "PUT":
        marble_idx, put_flat, rem_flat = payload
        py, px = divmod(int(put_flat), width)
        new_py, new_px = _transform_coordinate_put(py, px, rot60_k, mirror, config)
        if rem_flat == width * width:
            new_rem_flat = rem_flat
        else:
            ry, rx = divmod(int(rem_flat), width)
            new_ry, new_rx = _transform_coordinate_put(ry, rx, rot60_k, mirror, config)
            new_rem_flat = new_ry * width + new_rx
        new_put_flat = new_py * width + new_px
        return "PUT", (marble_idx, new_put_flat, new_rem_flat)

    if action_type == "CAP":
        direction, y, x = payload
        dir_map = _dir_index_map(rot60_k, mirror, mirror_first, config)
        new_dir = dir_map[int(direction)]
        new_y, new_x = _transform_coordinate(int(y), int(x), rot60_k, mirror, mirror_first, config)
        return "CAP", (new_dir, new_y, new_x)

    if action_type == "PASS":
        return action

    raise ValueError(f"Unknown action type: {action_type}")


def _rotate_action(action, angle_degrees: int, config: BoardConfig):
    """Rotate action by specified angle (multiple of 60 degrees)."""
    angle = angle_degrees % 360
    if angle == 0:
        return action
    rot60_k = (angle // 60) % 6
    if rot60_k == 0:
        return action
    return _apply_orientation(action, rot60_k, mirror=False, mirror_first=False, config=config)


def _mirror_action(action, angle_degrees: int, mirror_first: bool, config: BoardConfig):
    """Apply mirror combined with rotation component."""
    angle = angle_degrees % 360
    rot60_k = (angle // 60) % 6
    return _apply_orientation(action, rot60_k, mirror=True, mirror_first=mirror_first, config=config)


def transform_action(action, transform: str, config: BoardConfig):
    """
    Transform an action according to canonicalization transform string.

    Supports combined transforms involving translations (Tdy,dx),
    rotations (Rk where k is multiple of 60), and mirrors (MRk or RkM).
    """
    if not transform or transform == "R0":
        return action

    current = action
    parts = transform.split("_")
    for part in parts:
        if not part or part == "R0":
            continue
        if part.startswith("T"):
            coords = part[1:]
            dy, dx = map(int, coords.split(","))
            current = _translate_action(current, dy, dx, config)
        elif part.startswith("MR"):
            angle = int(part[2:]) if len(part) > 2 else 0
            current = _mirror_action(current, angle, mirror_first=False, config=config)
        elif part.endswith("M") and part.startswith("R"):
            angle = int(part[1:-1]) if len(part) > 1 else 0
            current = _mirror_action(current, angle, mirror_first=True, config=config)
        elif part.startswith("R"):
            angle = int(part[1:]) if len(part) > 1 else 0
            current = _rotate_action(current, angle, config=config)
        else:
            raise ValueError(f"Unknown transform component: {part}")
    return current


def is_removable(index: Tuple[int, int], board_state: np.ndarray, config: BoardConfig) -> bool:
    """Check if ring at index can be removed.

    Pure function version of ZertzBoard._is_removable()

    A ring is removable if:
    1. It's empty (no marble)
    2. Two consecutive neighbors are missing
    """
    y, x = index
    if np.sum(board_state[config.board_layers, y, x]) != 1:
        return False

    neighbors = get_neighbors(index, config)
    # Add first neighbor to end for wrap-around check
    neighbors.append(neighbors[0])

    adjacent_empty = 0
    for neighbor in neighbors:
        if (is_inbounds(neighbor, config) and
            board_state[config.ring_layer][neighbor] == 1):
            adjacent_empty = 0
        else:
            adjacent_empty += 1
            if adjacent_empty >= 2:
                return True

    return False


def get_removable_rings(board_state: np.ndarray, config: BoardConfig) -> list:
    """Get list of removable ring indices.

    Pure function version of ZertzBoard._get_removable_rings()

    Args:
        board_state: Board state array
        config: BoardConfig
    """
    open_rings = get_open_rings(board_state, config)
    return [ring for ring in open_rings if is_removable(ring, board_state, config)]


# ============================================================================
# VALID ACTIONS (Pure function version)
# ============================================================================

def get_placement_moves(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> np.ndarray:
    """Get valid placement moves as boolean array.

    Pure function version of ZertzBoard.get_placement_moves()

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig

    Returns:
        Boolean array of shape (3, width², width² + 1)
    """
    moves = np.zeros((3, config.width**2, config.width**2 + 1), dtype=bool)

    # Get open and removable rings
    open_rings = get_open_rings(board_state, config)
    removable_rings = get_removable_rings(board_state, config)

    # Determine which marbles can be placed
    supply_counts = global_state[config.supply_slice]
    current_player = int(global_state[config.cur_player])

    if np.all(supply_counts == 0):
        # Use current player's captured marbles
        if current_player == config.player_1:
            marble_counts = global_state[config.p1_cap_slice]
        else:
            marble_counts = global_state[config.p2_cap_slice]
    else:
        marble_counts = supply_counts

    # Build moves matrix
    for m, marble_count in enumerate(marble_counts):
        if marble_count == 0:
            continue

        for put_index in open_rings:
            put = put_index[0] * config.width + put_index[1]

            for rem_index in removable_rings:
                rem = rem_index[0] * config.width + rem_index[1]
                if put != rem:
                    moves[m, put, rem] = True

            # If no removable rings, or only one (the destination), allow no removal
            if not removable_rings or (len(removable_rings) == 1 and removable_rings[0] == put_index):
                rem = config.width**2
                moves[m, put, rem] = True

    return moves


def get_capture_moves(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> np.ndarray:
    """Get valid capture moves as boolean array.

    Pure function version of ZertzBoard.get_capture_moves()

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig

    Returns:
        Boolean array of shape (6, width, width)
    """
    moves = np.zeros((6, config.width, config.width), dtype=bool)

    # Find marbles that can capture
    if np.sum(board_state[config.capture_layer]) == 1:
        # Forced chain capture - only specific marble can move
        occupied_rings = zip(*np.where(board_state[config.capture_layer] == 1))
    else:
        # Any marble can capture
        occupied_rings = zip(
            *np.where(np.sum(board_state[config.marble_layers], axis=0) == 1)
        )

    # Check each marble for valid captures
    for src_index in occupied_rings:
        src_y, src_x = src_index
        neighbors = get_neighbors(src_index, config)

        for direction, neighbor in enumerate(neighbors):
            y, x = neighbor
            if (is_inbounds(neighbor, config) and
                np.sum(board_state[config.marble_layers, y, x]) == 1):
                # Neighbor has a marble - check if we can jump over it
                dst_index = get_jump_destination(src_index, neighbor)
                y, x = dst_index
                if (is_inbounds(dst_index, config) and
                    np.sum(board_state[config.board_layers, y, x]) == 1):
                    # Valid capture!
                    moves[direction, src_y, src_x] = True

    return moves


def get_valid_actions(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Get valid actions for current state.

    PURE FUNCTION - Main interface for stateless action generation.

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig

    Returns:
        (placement_mask, capture_mask) tuple
        - placement_mask: (3, width², width² + 1) boolean array
        - capture_mask: (6, width, width) boolean array

    Note: If any captures are available, placement_mask will be all zeros
          (captures are mandatory in Zertz rules)
    """
    capture = get_capture_moves(board_state, global_state, config)

    if np.any(capture):
        # Captures are mandatory
        placement = np.zeros((3, config.width**2, config.width**2 + 1), dtype=bool)
    else:
        placement = get_placement_moves(board_state, global_state, config)

    return placement, capture


# ============================================================================
# ACTION APPLICATION (Pure function version)
# ============================================================================

def get_marble_type_at(index: Tuple[int, int], board_state: np.ndarray, config: BoardConfig) -> str:
    """Get marble type at given position.

    Args:
        index: (y, x) position
        board_state: (L, H, W) spatial state array
        config: BoardConfig

    Returns:
        Marble type ('w', 'g', or 'b')
    """
    y, x = index
    layer_idx = np.argmax(board_state[config.marble_layers, y, x]) + 1
    return config.layer_to_marble[layer_idx]


def get_supply_index(marble_type: str, config: BoardConfig) -> int:
    """Get global_state index for marble in supply."""
    marble_to_idx = {'w': config.supply_w, 'g': config.supply_g, 'b': config.supply_b}
    return marble_to_idx[marble_type]


def get_captured_index(marble_type: str, player: int, config: BoardConfig) -> int:
    """Get global_state index for captured marble for given player."""
    if player == config.player_1:
        marble_to_idx = {'w': config.p1_cap_w, 'g': config.p1_cap_g, 'b': config.p1_cap_b}
    else:
        marble_to_idx = {'w': config.p2_cap_w, 'g': config.p2_cap_g, 'b': config.p2_cap_b}
    return marble_to_idx[marble_type]


def check_for_isolation_capture(board_state: np.ndarray, global_state: np.ndarray, config: BoardConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Check for and apply isolation captures (fully occupied isolated regions).

    Pure function version of ZertzBoard._check_for_isolation_capture()

    Args:
        board_state: (L, H, W) spatial state array (will be copied)
        global_state: (10,) global state array (will be copied)
        config: BoardConfig

    Returns:
        (new_board_state, new_global_state) with isolation captures applied
    """
    # Work on copies to maintain purity
    new_board = np.copy(board_state)
    new_global = np.copy(global_state)

    regions = get_regions(new_board, config)
    if len(regions) <= 1:
        return new_board, new_global

    # Get main region and current player
    main_region = max(regions, key=len)
    current_player = int(new_global[config.cur_player])

    # Check each isolated region
    for region in regions:
        if region == main_region:
            continue

        # Check if all rings in this isolated region are occupied
        all_occupied = all(
            np.sum(new_board[config.marble_layers, y, x]) == 1
            for y, x in region
        )

        if all_occupied:
            # Capture all marbles in this fully-occupied isolated region
            for y, x in region:
                marble_type = get_marble_type_at((y, x), new_board, config)

                # Add to current player's captured marbles
                captured_idx = get_captured_index(marble_type, current_player, config)
                new_global[captured_idx] += 1

                # Remove marble from board
                new_board[config.marble_layers, y, x] = 0

                # Remove ring from board
                new_board[config.ring_layer, y, x] = 0

    return new_board, new_global


def apply_placement_action(
    board_state: np.ndarray,
    global_state: np.ndarray,
    action: Tuple[int, int, int],
    config: BoardConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply placement action to state.

    Pure function version of ZertzBoard.take_placement_action()

    Args:
        board_state: (L, H, W) spatial state array (will be copied)
        global_state: (10,) global state array (will be copied)
        action: (type_index, put_loc, rem_loc) placement action
        config: BoardConfig

    Returns:
        (new_board_state, new_global_state) after applying action
    """
    # Work on copies to maintain purity
    new_board = np.copy(board_state)
    new_global = np.copy(global_state)

    # Decode action
    type_index, put_loc, rem_loc = action
    marble_type = config.layer_to_marble[type_index + 1]
    put_index = (put_loc // config.width, put_loc % config.width)
    rem_index = None if rem_loc == config.width**2 else (rem_loc // config.width, rem_loc % config.width)

    # Place marble on board
    y, x = put_index
    put_layer = config.marble_to_layer[marble_type]
    new_board[put_layer, y, x] = 1

    # Remove marble from supply or current player's captured marbles
    current_player = int(new_global[config.cur_player])
    supply_idx = get_supply_index(marble_type, config)

    if new_global[supply_idx] >= 1:
        # Use from supply
        new_global[supply_idx] -= 1
    elif np.all(new_global[config.supply_slice] == 0):
        # Entire supply empty - use captured marbles
        captured_idx = get_captured_index(marble_type, current_player, config)
        new_global[captured_idx] -= 1

    # Remove ring if specified
    if rem_index is not None:
        new_board[config.ring_layer][rem_index] = 0
        # Check for isolation captures
        new_board, new_global = check_for_isolation_capture(new_board, new_global, config)

    # Switch player
    new_global[config.cur_player] = (new_global[config.cur_player] + 1) % config.num_players

    return new_board, new_global


def apply_capture_action(
    board_state: np.ndarray,
    global_state: np.ndarray,
    action: Tuple[int, int, int],
    config: BoardConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply capture action to state.

    Pure function version of ZertzBoard.take_capture_action()

    Args:
        board_state: (L, H, W) spatial state array (will be copied)
        global_state: (10,) global state array (will be copied)
        action: (direction, y, x) capture action
        config: BoardConfig

    Returns:
        (new_board_state, new_global_state) after applying action
    """
    # Work on copies to maintain purity
    new_board = np.copy(board_state)
    new_global = np.copy(global_state)

    # Decode action
    direction, y, x = action
    src_index = (y, x)
    marble_type = get_marble_type_at(src_index, new_board, config)
    dy, dx = config.directions[direction]
    cap_index = (y + dy, x + dx)
    dst_index = get_jump_destination(src_index, cap_index)

    # Reset capture layer
    new_board[config.capture_layer] = 0

    # Move capturing marble
    marble_layer = config.marble_to_layer[marble_type]
    new_board[marble_layer][src_index] = 0
    new_board[marble_layer][dst_index] = 1

    # Capture marble
    current_player = int(new_global[config.cur_player])
    y_cap, x_cap = cap_index
    captured_type = get_marble_type_at(cap_index, new_board, config)
    captured_idx = get_captured_index(captured_type, current_player, config)
    new_global[captured_idx] += 1
    new_board[config.marble_layers, y_cap, x_cap] = 0

    # Check for forced chain capture
    neighbors = get_neighbors(dst_index, config)
    for neighbor in neighbors:
        y_n, x_n = neighbor
        if (is_inbounds(neighbor, config) and
            np.sum(new_board[config.marble_layers, y_n, x_n]) == 1):
            next_dst = get_jump_destination(dst_index, neighbor)
            y_next, x_next = next_dst
            if (is_inbounds(next_dst, config) and
                np.sum(new_board[config.board_layers, y_next, x_next]) == 1):
                # Set capture layer at dst_index
                new_board[config.capture_layer][dst_index] = 1
                break

    # Switch player only if no chain capture
    if np.sum(new_board[config.capture_layer]) == 0:
        new_global[config.cur_player] = (new_global[config.cur_player] + 1) % config.num_players

    return new_board, new_global


def apply_action(
    board_state: np.ndarray,
    global_state: np.ndarray,
    action: Tuple[int, ...],
    action_type: str,
    config: BoardConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply action to game state.

    PURE FUNCTION - Main interface for stateless action application.

    This matches the behavior of ZertzBoard.take_action() but without mutation.

    Args:
        board_state: (L, H, W) spatial state array (will be copied)
        global_state: (10,) global state array (will be copied)
        action: Action tuple (format depends on action_type)
                - PUT: (type_index, put_loc, rem_loc)
                - CAP: (direction, y, x)
        action_type: 'PUT' or 'CAP'
        config: BoardConfig

    Returns:
        (new_board_state, new_global_state) after applying action

    Note: This function handles time history pushback automatically
    """
    # Copy states to maintain purity
    new_board = np.copy(board_state)
    new_global = np.copy(global_state)

    # Push back time history
    # Copy current board layers to history, shifting older history back
    layers_per_step = config.layers_per_timestep
    new_board[0:layers_per_step * config.t] = np.concatenate([
        new_board[config.board_layers],
        new_board[0:layers_per_step * (config.t - 1)]
    ], axis=0)

    # Apply action
    if action_type == 'PUT':
        return apply_placement_action(new_board, new_global, action, config)
    elif action_type == 'CAP':
        return apply_capture_action(new_board, new_global, action, config)
    else:
        raise ValueError(f"Unknown action type: {action_type}")


# ============================================================================
# GAME TERMINATION CHECKS (Pure function version)
# ============================================================================

def is_game_over(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig,
    win_conditions: list = None
) -> bool:
    """Check if game has ended (stateless version).

    Checks basic terminal conditions without requiring move history:
    1. Win by captured marbles
    2. Board fully occupied
    3. Current player has no marbles

    Note: This does NOT check for consecutive passes or move loops,
    which require move history. Use ZertzGame.get_game_ended() for complete checks.

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig
        win_conditions: List of win condition dicts (default: standard 3-of-each or 4W/5G/6B)

    Returns:
        True if game is over, False otherwise
    """
    if win_conditions is None:
        win_conditions = [{"w": 3, "g": 3, "b": 3}, {"w": 4}, {"g": 5}, {"b": 6}]

    # Check win by captured marbles
    p1_captured = global_state[config.p1_cap_slice]
    p2_captured = global_state[config.p2_cap_slice]

    marble_types = ['w', 'g', 'b']
    for win_con in win_conditions:
        required = np.zeros(3)
        for i, marble_type in enumerate(marble_types):
            if marble_type in win_con:
                required[i] = win_con[marble_type]

        if np.all(p1_captured >= required) or np.all(p2_captured >= required):
            return True

    # Check if board is fully occupied
    if np.all(np.sum(board_state[config.board_layers], axis=0) != 1):
        return True

    # Check if current player has no marbles available
    supply = global_state[config.supply_slice]
    current_player = int(global_state[config.cur_player])

    if current_player == config.player_1:
        captured = global_state[config.p1_cap_slice]
    else:
        captured = global_state[config.p2_cap_slice]

    if np.all(supply + captured == 0):
        return True

    return False


def get_game_outcome(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig,
    win_conditions: list = None
) -> int:
    """Determine game outcome from terminal state (stateless version).

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig
        win_conditions: List of win condition dicts (default: standard 3-of-each or 4W/5G/6B)

    Returns:
        1 if Player 1 wins, -1 if Player 2 wins, 0 for tie/draw, None if not terminal
    """
    if win_conditions is None:
        win_conditions = [{"w": 3, "g": 3, "b": 3}, {"w": 4}, {"g": 5}, {"b": 6}]

    p1_captured = global_state[config.p1_cap_slice]
    p2_captured = global_state[config.p2_cap_slice]

    # Check win by captured marbles
    p1_won = False
    p2_won = False

    marble_types = ['w', 'g', 'b']
    for win_con in win_conditions:
        required = np.zeros(3, dtype=int)
        for i, marble_type in enumerate(marble_types):
            if marble_type in win_con:
                required[i] = win_con[marble_type]

        if np.all(p1_captured >= required):
            p1_won = True
        if np.all(p2_captured >= required):
            p2_won = True

    if p1_won and p2_won:
        # Both win simultaneously (rare) - tie
        return 0
    elif p1_won:
        return 1
    elif p2_won:
        return -1

    # Check if board is fully occupied
    board_full = np.all(np.sum(board_state[config.board_layers], axis=0) != 1)

    # Check if players have no marbles
    supply = global_state[config.supply_slice]
    p1_marbles = supply + p1_captured
    p2_marbles = supply + p2_captured

    p1_has_marbles = np.any(p1_marbles > 0)
    p2_has_marbles = np.any(p2_marbles > 0)

    if board_full:
        # Board full - last player to move wins (current player loses)
        current_player = int(global_state[config.cur_player])
        return -1 if current_player == config.player_1 else 1

    if not p1_has_marbles and not p2_has_marbles:
        # Both out of marbles - compare captures
        p1_score = np.dot(p1_captured, [1, 2, 3])  # w=1, g=2, b=3
        p2_score = np.dot(p2_captured, [1, 2, 3])
        if p1_score > p2_score:
            return 1
        elif p2_score > p1_score:
            return -1
        else:
            return 0

    if not p1_has_marbles:
        return -1  # Player 1 loses
    if not p2_has_marbles:
        return 1  # Player 2 loses

    # Not terminal
    return None
