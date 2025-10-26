"""Stateless game logic for Zertz.

All functions are pure: same inputs → same outputs.
No side effects, no mutations, no hidden state.

This enables:
- Efficient neural network batching
- Parallelization across multiple cores/GPUs
- Memory-efficient MCTS (store arrays, not objects)
- Reproducible simulations

Architecture:
    BoardConfig: Immutable Rust configuration (constants, dimensions, rules)
    Pure functions: Take (state, config) → return new state or actions

Usage:
    from game.zertz_logic import BoardConfig
    config = BoardConfig.standard_config(rings=37)
    placement, capture = get_valid_actions(board_state, global_state, config)
"""

from typing import Dict, Tuple, List
import numpy as np

# Import Rust implementations
from hiivelabs_zertz_mcts import (
    BoardConfig,
    is_inbounds as rust_is_inbounds,
    get_neighbors as rust_get_neighbors,
    get_jump_destination as rust_get_jump_destination,
    get_regions as rust_get_regions,
    get_open_rings as rust_get_open_rings,
    is_ring_removable as rust_is_ring_removable,
    get_removable_rings as rust_get_removable_rings,
    get_marble_type_at as rust_get_marble_type_at,
    get_supply_index as rust_get_supply_index,
    get_captured_index as rust_get_captured_index,
    get_placement_moves as rust_get_placement_moves,
    get_capture_moves as rust_get_capture_moves,
    get_valid_actions as rust_get_valid_actions,
    apply_placement_action as rust_apply_placement_action,
    apply_capture_action as rust_apply_capture_action,
    is_game_over as rust_is_game_over,
    get_game_outcome as rust_get_game_outcome,
    check_for_isolation_capture as rust_check_isolation,
    ax_rot60,
    ax_mirror_q_axis,
)

from .constants import PLAYER_1_WIN, PLAYER_2_WIN, BOTH_LOSE


# ============================================================================
# PURE HELPER FUNCTIONS (Rust wrappers)
# ============================================================================

def is_inbounds(index: Tuple[int, int], config: BoardConfig) -> bool:
    """Check if index is within board bounds."""
    y, x = index
    return rust_is_inbounds(y, x, config.width)


def get_neighbors(index: Tuple[int, int], config: BoardConfig) -> list:
    """Get list of neighboring indices (filtered to in-bounds only)."""
    y, x = index
    return rust_get_neighbors(y, x, config)


def get_jump_destination(start: Tuple[int, int], cap: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate landing position after capturing marble at cap from start."""
    sy, sx = start
    cy, cx = cap
    return rust_get_jump_destination(sy, sx, cy, cx)


def get_regions(board_state: np.ndarray, config: BoardConfig) -> list:
    """Find all connected regions on the board.

    Returns:
        List of regions, where each region is a list of (y, x) indices
    """
    return rust_get_regions(board_state, config)


def get_open_rings(board_state: np.ndarray, config: BoardConfig) -> list:
    """Get list of empty ring indices across the entire board.

    Args:
        board_state: Board state array
        config: BoardConfig
    """
    return rust_get_open_rings(board_state, config)


def is_ring_removable(index: Tuple[int, int], board_state: np.ndarray, config: BoardConfig) -> bool:
    """Check if ring at index can be removed.

    A ring is removable if:
    1. It's empty (no marble)
    2. Two consecutive neighbors are missing
    """
    y, x = index
    return rust_is_ring_removable(board_state, y, x, config)


def get_removable_rings(board_state: np.ndarray, config: BoardConfig) -> list:
    """Get list of removable ring indices.

    Args:
        board_state: Board state array
        config: BoardConfig
    """
    return rust_get_removable_rings(board_state, config)


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
    return rust_get_marble_type_at(board_state, y, x)


def get_supply_index(marble_type: str, config: BoardConfig) -> int:
    """Get global_state index for marble in supply."""
    return rust_get_supply_index(marble_type)


def get_captured_index(marble_type: str, player: int, config: BoardConfig) -> int:
    """Get global_state index for captured marble for given player."""
    return rust_get_captured_index(player, marble_type)


# ============================================================================
# ACTION TRANSFORMATIONS (kept in Python for now - complex logic)
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
    layout = getattr(config, 'board_layout', None)
    if layout is not None:
        return layout.astype(bool)

    key = (int(config.rings), int(config.width), None)
    cached = _LAYOUT_MASK_CACHE.get(key)
    if cached is None:
        mask = _generate_standard_layout_mask(int(config.rings))
        _LAYOUT_MASK_CACHE[key] = mask
        return mask
    return cached

#todo why do we need this?
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


# ============================================================================
# MODERN TRANSFORMATION SYSTEM (for action replay/notation)
# ============================================================================
# These functions operate on INDIVIDUAL ACTIONS (tuples) for game logic.
# They transform individual action coordinates using axial hexagonal math.
#
# DO NOT CONFUSE WITH: Legacy numpy array transforms in zertz_board.py
# which operate on ENTIRE ACTION MASKS for ML symmetry detection.
#
# Use cases:
# - Modern: Transform action tuples for replay systems and notation parsing
# - Legacy: Transform 3D numpy arrays for neural network training
#
# All axial coordinate transforms now delegate to Rust for performance.
# ============================================================================

def _ax_rot60(q: int, r: int, k: int = 1) -> Tuple[int, int]:
    """Rotate axial coordinate by k * 60° counterclockwise (delegates to Rust)."""
    return ax_rot60(q, r, k)


def _ax_mirror_q_axis(q: int, r: int) -> Tuple[int, int]:
    """Mirror axial coordinate across the q-axis (delegates to Rust)."""
    return ax_mirror_q_axis(q, r)


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


def _transform_coordinate_put(y: int, x: int, rot60_k: int, mirror: bool, mirror_first: bool, config: BoardConfig) -> Tuple[int, int]:
    """Apply rotation/mirror to placement coordinate.

    Args:
        y: Y coordinate
        x: X coordinate
        rot60_k: Number of 60° rotations
        mirror: Whether to apply mirror
        mirror_first: If True, mirror then rotate. If False, rotate then mirror.
        config: Board configuration
    """
    yx_to_ax, ax_to_yx = _build_axial_maps(config)
    key = (int(y), int(x))
    if key not in yx_to_ax:
        raise ValueError(f"Coordinate {(y, x)} is not on the board")

    q, r = yx_to_ax[key]

    # Apply transformations in correct order
    if mirror_first:
        # Mirror first, then rotate (for R{k}M transforms)
        if mirror:
            q, r = _ax_mirror_q_axis(q, r)
        q, r = _ax_rot60(q, r, rot60_k)
    else:
        # Rotate first, then mirror (for MR{k} transforms)
        q, r = _ax_rot60(q, r, rot60_k)
        if mirror:
            q, r = _ax_mirror_q_axis(q, r)

    dest = ax_to_yx.get((q, r))
    if dest is None:
        raise ValueError("Rotated coordinate is not on the board")
    return dest


def _dir_index_map(rot60_k: int, mirror: bool, mirror_first: bool, config: BoardConfig) -> Dict[int, int]:
    """Map direction indices under rotation/mirror."""
    directions = getattr(config, 'directions', None)
    if directions is None:
        directions = config.get_directions()

    mapping = {}
    for idx, (dy, dx) in enumerate(directions):
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
            new_idx = directions.index((new_dy, new_dx))
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
        new_py, new_px = _transform_coordinate_put(py, px, rot60_k, mirror, mirror_first, config)
        if rem_flat == width * width:
            new_rem_flat = rem_flat
        else:
            ry, rx = divmod(int(rem_flat), width)
            new_ry, new_rx = _transform_coordinate_put(ry, rx, rot60_k, mirror, mirror_first, config)
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


# ============================================================================
# VALID ACTIONS (Rust wrappers)
# ============================================================================

def get_placement_moves(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> np.ndarray:
    """Get valid placement moves as boolean array.

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig

    Returns:
        Boolean array of shape (3, width², width² + 1)
    """
    return rust_get_placement_moves(board_state, global_state, config)


def get_capture_moves(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> np.ndarray:
    """Get valid capture moves as boolean array.

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig

    Returns:
        Boolean array of shape (6, width, width)
    """
    return rust_get_capture_moves(board_state, config)


def get_valid_actions(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Get valid actions for current state.

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
    return rust_get_valid_actions(board_state, global_state, config)


# ============================================================================
# ACTION APPLICATION (Rust wrappers with in-place mutation)
# ============================================================================

def apply_placement_action(
    board_state: np.ndarray,
    global_state: np.ndarray,
    action: Tuple[int, int, int],
    config: BoardConfig
) -> None:
    """Apply placement action to state IN-PLACE.

    Args:
        board_state: (L, H, W) spatial state array (MUTATED IN-PLACE)
        global_state: (10,) global state array (MUTATED IN-PLACE)
        action: (type_index, put_loc, rem_loc) placement action
        config: BoardConfig
    """
    type_index, put_loc, rem_loc = action
    dst_y, dst_x = divmod(put_loc, config.width)
    if rem_loc == config.width ** 2:
        remove_y, remove_x = None, None
    else:
        remove_y, remove_x = divmod(rem_loc, config.width)

    rust_apply_placement_action(
        board_state,
        global_state,
        type_index,
        dst_y,
        dst_x,
        remove_y,
        remove_x,
        config
    )


def apply_capture_action(
    board_state: np.ndarray,
    global_state: np.ndarray,
    action: Tuple[int, int, int],
    config: BoardConfig
) -> None:
    """Apply capture action to state IN-PLACE.

    Args:
        board_state: (L, H, W) spatial state array (MUTATED IN-PLACE)
        global_state: (10,) global state array (MUTATED IN-PLACE)
        action: (direction, y, x) capture action
        config: BoardConfig
    """
    direction, start_y, start_x = action
    rust_apply_capture_action(
        board_state,
        global_state,
        start_y,
        start_x,
        direction,
        config
    )


# ============================================================================
# GAME TERMINATION CHECKS (Rust wrappers)
# ============================================================================

def is_game_over(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> bool:
    """Check if game has ended (stateless version).

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig

    Returns:
        True if game is over, False otherwise
    """
    return rust_is_game_over(board_state, global_state, config)


def get_game_outcome(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> int:
    """Determine game outcome from terminal state (stateless version).

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig

    Returns:
        1 if Player 1 wins, -1 if Player 2 wins, 0 for tie, -2 for both lose
    """
    return rust_get_game_outcome(board_state, global_state, config)


def check_for_isolation_capture(
    board_state: np.ndarray,
    global_state: np.ndarray,
    config: BoardConfig
) -> tuple[np.ndarray, np.ndarray, List[tuple[int, int, int]]]:
    """Check for isolated regions and capture marbles (stateless version).

    After a ring is removed, the board may split into multiple disconnected regions.
    If ALL rings in an isolated region are fully occupied (each has a marble),
    then the current player captures all those marbles and removes those rings.

    Args:
        board_state: (L, H, W) spatial state array
        global_state: (10,) global state array
        config: BoardConfig

    Returns:
        Tuple of (updated_board_state, updated_global_state, captured_list)
    """

    # Call Rust implementation
    # Returns: (spatial, global, captured_marbles_list)
    spatial_out, global_out, captured_list = rust_check_isolation(
        board_state, global_state, config
    )

    return spatial_out, global_out, captured_list