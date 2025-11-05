"""Test position-based loop detection (ignoring marble colors)."""

import pytest
from pathlib import Path

from hiivelabs_mcts import algebraic_to_coordinate

from game.zertz_game import ZertzGame


def test_move_history_normalization():
    """Test the internal normalization function directly."""
    game = ZertzGame(rings=37)
    config = game.board.config

    # Test PUT action normalization - marble color should be ignored
    pos_d4 = algebraic_to_coordinate("D4", config)
    pos_a1 = algebraic_to_coordinate("A1", config)
    put_white = ("PUT", (0, *pos_d4, *pos_a1))  # marble_type=0 (white), dst=10, rem=20
    put_gray = ("PUT", (1, *pos_d4, *pos_a1))   # marble_type=1 (gray), same positions

    norm_white = game._normalize_move_for_loop_detection(put_white)
    norm_gray = game._normalize_move_for_loop_detection(put_gray)

    assert norm_white == norm_gray, "PUT actions with same positions should normalize to same value"
    assert norm_white == ("PUT", (None, *pos_d4, *pos_a1)), "Marble type should be replaced with None"

    # Test CAP action normalization - already position-only
    cap_move = ("CAP", (2, 5, 5))  # direction=2, src=(5,5)
    norm_cap = game._normalize_move_for_loop_detection(cap_move)

    assert norm_cap == cap_move, "CAP actions should not change (already position-only)"

    # Test PASS action normalization
    pass_move = ("PASS", None)
    norm_pass = game._normalize_move_for_loop_detection(pass_move)

    assert norm_pass == pass_move, "PASS actions should not change"


def test_capture_actions_are_position_only():
    """Test that capture actions inherently ignore marble colors."""
    game = ZertzGame(rings=37)

    # Set up board with marbles at specific positions
    # The capture action format is (direction, src_y, src_x)
    # It doesn't include the marble color being moved or captured

    # Two captures from same source in same direction are the same action
    # regardless of what marble is being moved
    cap1 = ("CAP", (0, 3, 3))  # Direction 0 from position (3,3)
    cap2 = ("CAP", (0, 3, 3))  # Same action, even if different marble color

    norm1 = game._normalize_move_for_loop_detection(cap1)
    norm2 = game._normalize_move_for_loop_detection(cap2)

    assert norm1 == norm2, "Identical capture actions should normalize to same value"
    assert norm1 == cap1, "CAP normalization should not modify the action"