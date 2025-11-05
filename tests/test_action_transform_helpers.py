import sys
from pathlib import Path

import numpy as np
import hiivelabs_mcts.zertz as zertz
import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    config = board.config
    canonicalizer = board.canonicalizer
    width = config.width

    marble_idx = 0
    put_y, put_x = zertz.algebraic_to_coordinate(config, "C3")
    rem_y, rem_x = zertz.algebraic_to_coordinate(config, "E5")
    # action = ("PUT", (marble_idx, put_y, put_x, rem_y, rem_x))
    action = zertz.ZertzAction.placement(config, marble_idx, put_y, put_x, rem_y, rem_x)

    # new_action = zertz.transform_action(action[0], action[1], transform, config)
    new_action = zertz.transform_action(config, action, transform)
    new_action_tuple = new_action.to_tuple(config)

    mask = np.zeros((3, width, width, width, width), dtype=int)
    mask[marble_idx, put_y, put_x, rem_y, rem_x] = 1
    transformed_mask, _, _ = canonicalizer.canonicalize_put_mask(mask, transform)
    m_idx, new_put_y, new_put_x, new_rem_y, new_rem_x = np.argwhere(transformed_mask == 1)[0]


    assert new_action_tuple == ("PUT", [int(m_idx), int(new_put_y), int(new_put_x), int(new_rem_y), int(new_rem_x)])


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
    config = board.config
    canonicalizer = board.canonicalizer
    width = config.width

    direction = 0
    src_y, src_x = zertz.algebraic_to_coordinate(config, "D4")
    dst_y, dst_x = zertz.get_capture_destination(config, src_y, src_x, direction)
    # action = ("CAP", (direction, y, x))
    action = zertz.ZertzAction.capture(config, src_y, src_x, dst_y, dst_x)

    new_action = zertz.transform_action(config, action, transform)
    new_action_tuple = new_action.to_tuple(config)

    mask = np.zeros((6, width, width), dtype=int)
    mask[direction, src_y, src_x] = 1
    transformed_mask, _, _ = canonicalizer.canonicalize_capture_mask(mask, transform)
    new_direction, new_src_y, new_src_x = np.argwhere(transformed_mask == 1)[0]
    new_dst_y, new_dst_x = zertz.get_capture_destination(config, src_y, src_x, new_direction)

    assert new_action_tuple == ("CAP", [int(new_src_y), int(new_src_x), int(new_dst_y), int(new_dst_x)])


def test_transform_action_with_translation():
    board = ZertzBoard(rings=37)
    config = board.config

    # action_put = ("PUT", (1, 3, 3, 3, 4))
    action_put = zertz.ZertzAction.placement(config, 1, 3, 3, 3, 4)
    translated_put = zertz.transform_action(config, action_put, "T-1,1")
    translated_put_tuple = translated_put.to_tuple(config)
    assert translated_put_tuple == ("PUT", [1, 2, 4, 2, 5])

    # action_cap = ("CAP", (0, 3, 3))
    d_y, d_x = zertz.flat_to_yx(config, "D4")
    action_cap = zertz.ZertzAction.capture(config, 3, 3, 3, 4)
    translated_cap = zertz.transform_action(config, action_cap, "T2,-1", )

    assert translated_cap == ("CAP", [0, 3, 3, d_y, d_x])
