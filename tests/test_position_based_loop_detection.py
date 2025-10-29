"""Test position-based loop detection (ignoring marble colors)."""

import pytest
from pathlib import Path
from game.zertz_game import ZertzGame
from game.zertz_player import ReplayZertzPlayer
from game.loaders.sgf_loader import SGFLoader


def test_loop_detection_requires_exact_position_match():
    """Test that loop detection requires exact position repetition."""
    game = ZertzGame(rings=37)

    # Make 8 moves with same 2-move pattern using different positions each time
    # Pattern: P1 at row 1, P2 at row 2
    # Moves 1-2: C1, D2
    # Moves 3-4: E1, F2
    # Moves 5-6: C1, D2 (same as 1-2)
    # Moves 7-8: E1, F2 (same as 3-4)
    game.take_action("PUT", game.str_to_action("PUT w B1")[1])      # P1
    game.take_action("PUT", game.str_to_action("PUT g C2")[1])      # P2
    game.take_action("PUT", game.str_to_action("PUT w D1")[1])      # P1
    game.take_action("PUT", game.str_to_action("PUT g E2")[1])      # P2

    # Repeat the same pattern
    game.take_action("PUT", game.str_to_action("PUT b B2")[1])      # P1 (different color, different pos)
    game.take_action("PUT", game.str_to_action("PUT w D2")[1])      # P2 (different color, different pos)
    game.take_action("PUT", game.str_to_action("PUT g B3")[1])      # P1
    game.take_action("PUT", game.str_to_action("PUT b C3")[1])      # P2

    # No loop should be detected - all positions are different
    assert not game._has_move_loop(), "Loop should not be detected when positions differ"


def test_loop_detection_with_different_removal_positions():
    """Test that different ring removal positions prevent loop detection."""
    game = ZertzGame(rings=37)

    # Make 8 moves where placement/removal combinations never repeat
    # Each move uses a unique combination of (place position, removal position)
    game.take_action("PUT", game.str_to_action("PUT w B1 A1")[1])   # P1: B1/A1
    game.take_action("PUT", game.str_to_action("PUT g C2 B2")[1])   # P2: C2/B2
    game.take_action("PUT", game.str_to_action("PUT w D1 A2")[1])   # P1: D1/A2
    game.take_action("PUT", game.str_to_action("PUT g E2 C3")[1])   # P2: E2/C3
    game.take_action("PUT", game.str_to_action("PUT w F1 A3")[1])   # P1: F1/A3
    game.take_action("PUT", game.str_to_action("PUT g G1 C4")[1])   # P2: G1/C4
    game.take_action("PUT", game.str_to_action("PUT w B4 A4")[1])   # P1: B4/A4
    game.take_action("PUT", game.str_to_action("PUT g C5 D2")[1])   # P2: C5/D2

    # No loop should be detected - all placement & removal combinations are unique
    assert not game._has_move_loop(), "Loop should not be detected with different removals"


def test_move_history_normalization():
    """Test the internal normalization function directly."""
    game = ZertzGame(rings=37)

    # Test PUT action normalization - marble color should be ignored
    put_white = ("PUT", (0, 10, 20))  # marble_type=0 (white), dst=10, rem=20
    put_gray = ("PUT", (1, 10, 20))   # marble_type=1 (gray), same positions

    norm_white = game._normalize_move_for_loop_detection(put_white)
    norm_gray = game._normalize_move_for_loop_detection(put_gray)

    assert norm_white == norm_gray, "PUT actions with same positions should normalize to same value"
    assert norm_white == ("PUT", (None, 10, 20)), "Marble type should be replaced with None"

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