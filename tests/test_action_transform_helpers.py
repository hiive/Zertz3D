import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game import zertz_logic  # noqa: E402
from game.zertz_board import ZertzBoard  # noqa: E402


@pytest.mark.parametrize(
    "transform",
    [
        "R0",
        "R60",
        "R120",
        "R180",
        "R240",
        "R300",
        "MR0",
        "MR60",
        "R60M",
        "MR120",
        "R120M",
    ],
)
def test_transform_action_put_matches_canonicalizer(transform):
    board = ZertzBoard(rings=37)
    config = board._get_config()
    canonicalizer = board.canonicalizer
    width = config.width

    marble_idx = 0
    put_y, put_x = board.str_to_index("C3")
    rem_y, rem_x = board.str_to_index("E5")
    action = ("PUT", (marble_idx, put_y * width + put_x, rem_y * width + rem_x))

    new_action = zertz_logic.transform_action(action, transform, config)

    mask = np.zeros((3, width * width, width * width + 1), dtype=int)
    mask[marble_idx, put_y * width + put_x, rem_y * width + rem_x] = 1
    transformed_mask, _, _ = canonicalizer.canonicalize_put_mask(mask, transform)
    m_idx, new_put_flat, new_rem_flat = np.argwhere(transformed_mask == 1)[0]

    assert new_action == ("PUT", (int(m_idx), int(new_put_flat), int(new_rem_flat)))


@pytest.mark.parametrize(
    "transform",
    [
        "R0",
        "R60",
        "R120",
        "R180",
        "MR0",
        "R60M",
    ],
)
def test_transform_action_cap_matches_canonicalizer(transform):
    board = ZertzBoard(rings=37)
    config = board._get_config()
    canonicalizer = board.canonicalizer
    width = config.width

    direction = 0
    y, x = board.str_to_index("D4")
    action = ("CAP", (direction, y, x))

    new_action = zertz_logic.transform_action(action, transform, config)

    mask = np.zeros((6, width, width), dtype=int)
    mask[direction, y, x] = 1
    transformed_mask, _, _ = canonicalizer.canonicalize_capture_mask(mask, transform)
    new_dir, new_y, new_x = np.argwhere(transformed_mask == 1)[0]

    assert new_action == ("CAP", (int(new_dir), int(new_y), int(new_x)))


def test_transform_action_with_translation():
    board = ZertzBoard(rings=37)
    config = board._get_config()

    width = config.width
    action_put = ("PUT", (1, 3 * width + 3, 3 * width + 4))
    translated_put = zertz_logic.transform_action(action_put, "T-1,1", config)
    assert translated_put == ("PUT", (1, 2 * width + 4, 2 * width + 5))

    action_cap = ("CAP", (0, 3, 3))
    translated_cap = zertz_logic.transform_action(action_cap, "T2,-1", config)
    assert translated_cap == ("CAP", (0, 5, 2))
