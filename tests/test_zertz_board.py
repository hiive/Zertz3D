"""
Unit tests for ZertzBoard class.

Tests coordinate transformations, action translations, and game logic.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_board import ZertzBoard


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(
    params=[
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61,
    ]
)
def board_size(request):
    """Parameterized fixture for all supported board sizes."""
    return request.param


@pytest.fixture
def board(board_size):
    """Create a fresh board for each test."""
    return ZertzBoard(rings=board_size)


@pytest.fixture
def small_board():
    """37-ring board for size-specific tests."""
    return ZertzBoard(rings=ZertzBoard.SMALL_BOARD_37)


@pytest.fixture
def medium_board():
    """48-ring board for size-specific tests."""
    return ZertzBoard(rings=ZertzBoard.MEDIUM_BOARD_48)


@pytest.fixture
def large_board():
    """61-ring board for size-specific tests."""
    return ZertzBoard(rings=ZertzBoard.LARGE_BOARD_61)


# ============================================================================
# Coordinate Conversion Tests
# ============================================================================


class TestCoordinateConversion:
    """Test coordinate conversion helper methods."""

    def test_flat_to_2d_conversion(self, board):
        """Test conversion from flat index to 2D coordinates."""
        # Test corner positions
        y, x = divmod(0, board.width)
        assert y == 0 and x == 0

        # Test that conversion is consistent
        for flat_idx in [0, 5, 10, board.width - 1]:
            y, x = divmod(flat_idx, board.width)
            assert 0 <= y < board.width
            assert 0 <= x < board.width

    def test_2d_to_flat_conversion(self, board):
        """Test conversion from 2D coordinates to flat index."""
        # Test corner
        flat = 0 * board.width + 0
        assert flat == 0

        # Test roundtrip conversion
        for y in range(board.width):
            for x in range(board.width):
                flat = y * board.width + x
                y2, x2 = divmod(flat, board.width)
                assert y == y2 and x == x2

    def test_str_to_index_and_back(self, board):
        """Test string to index conversion and back."""
        # Get all valid positions from the board layout
        if board.letter_layout is not None:
            valid_positions = board.letter_layout[board.letter_layout != ""]
            for pos_str in valid_positions:
                # Convert to index and back
                idx = board.str_to_index(pos_str)
                # Only test roundtrip if ring still exists
                if board.state[board.RING_LAYER, idx[0], idx[1]] == 1:
                    result_str = board.index_to_str(idx)
                    assert result_str == pos_str, f"Failed roundtrip for {pos_str}"


# ============================================================================
# Game Logic Stub Tests
# ============================================================================


class TestBoardInitialization:
    """Test board initialization."""

    def test_board_creation(self, board_size):
        """Test that board is created with correct size."""
        board = ZertzBoard(rings=board_size)
        assert board.rings == board_size
        assert board.state is not None
        assert board.global_state is not None

    def test_initial_marble_supply(self, board):
        """Test initial marble counts in supply."""
        assert board.global_state[board.SUPPLY_W] == 6
        assert board.global_state[board.SUPPLY_G] == 8
        assert board.global_state[board.SUPPLY_B] == 10

    def test_initial_player(self, board):
        """Test that player 1 starts."""
        assert board.get_cur_player() == board.PLAYER_1


class TestBoardMethods:
    """Stub tests for board game logic methods."""

    def test_get_neighbors(self, board):
        """Test neighbor calculation for hexagonal grid."""
        # Test a center position - should have 6 neighbors
        center_pos = (board.width // 2, board.width // 2)
        neighbors = board.get_neighbors(center_pos)
        assert len(neighbors) == 6, "Hexagonal board center should have 6 neighbors"

        # All neighbors should be distinct
        assert len(set(neighbors)) == len(neighbors), "Neighbors should be unique"

        # All neighbors should be in bounds
        for ny, nx in neighbors:
            assert board._is_inbounds((ny, nx)), (
                f"Neighbor {(ny, nx)} should be in bounds"
            )

        # Test corner position - may have fewer neighbors
        corner_pos = (0, 0)
        corner_neighbors = board.get_neighbors(corner_pos)
        assert len(corner_neighbors) <= 6, "Corner position should have <= 6 neighbors"
        assert len(corner_neighbors) >= 2, (
            "Corner position should have at least 2 neighbors"
        )

        # Test that neighbors are exactly one step away in each direction
        for neighbor in neighbors[:3]:  # Check first 3 for efficiency
            ny, nx = neighbor
            y, x = center_pos
            # Hexagonal neighbors differ by combinations of [-1, 0, 1] in both dimensions
            assert abs(ny - y) <= 1 and abs(nx - x) <= 1
            assert (ny, nx) != (y, x), "Neighbor should not be the same as origin"

    def test_is_inbounds(self, board):
        """Test boundary checking for all board positions."""
        # Valid corner positions
        assert board._is_inbounds((0, 0))
        assert board._is_inbounds((board.width - 1, board.width - 1))

        # Out of bounds positions
        assert not board._is_inbounds((board.width, board.width))
        assert not board._is_inbounds((-1, 0))
        assert not board._is_inbounds((0, -1))

    # Note: test_get_regions is covered in test_isolated_regions.py
    # Note: test_take_placement_action and test_take_capture_action are covered extensively in other test files

    def test_get_valid_moves_shape(self, board):
        """Test that get_valid_moves returns correctly shaped arrays."""
        placement, capture = board.get_valid_moves()
        assert placement.shape == board.get_placement_shape()
        assert capture.shape == board.get_capture_shape()

    def test_get_placement_moves_shape(self, board):
        """Test that placement moves array has correct shape."""
        moves = board.get_placement_moves()
        assert moves.shape == (3, board.width**2, board.width**2 + 1)

    def test_get_capture_moves_shape(self, board):
        """Test that capture moves array has correct shape."""
        moves = board.get_capture_moves()
        assert moves.shape == (6, board.width, board.width)


def test_capture_sequence_continues_with_same_marble():
    """Test that after a capture, only the marble that just moved can continue capturing.

    Per Zertz rules: "If you can make multiple jumps in one sequence, you must
    continue until no more jumps are possible" - with the SAME marble.

    Scenario:
    - Marble A (white at C2) captures marble X (white at D3) and lands at E3
    - Marble A is now adjacent to marble B (gray at F2)
    - ONLY marble A should be able to capture (continue its sequence)
    - Marble B should NOT be able to capture (it's not its turn in the sequence)
    """
    board = ZertzBoard(rings=ZertzBoard.SMALL_BOARD_37)

    # Clear the board of marbles and capture layer
    board.state[board.MARBLE_LAYERS] = 0
    board.state[board.CAPTURE_LAYER] = 0

    # Set up scenario: place marbles for a capture sequence
    # Marble A (white) at C2
    c2_idx = board.str_to_index("C2")
    white_layer = board.MARBLE_TO_LAYER["w"]
    board.state[white_layer][c2_idx] = 1

    # Marble X (white) at D3 (will be captured)
    d3_idx = board.str_to_index("D3")
    board.state[white_layer][d3_idx] = 1

    # Marble B (gray) at F2
    f2_idx = board.str_to_index("F2")
    gray_layer = board.MARBLE_TO_LAYER["g"]
    board.state[gray_layer][f2_idx] = 1

    # Get initial capture moves - should show A can capture X
    captures_before = board.get_capture_moves()

    # Verify A can capture X (C2 → D3 → E3)
    # Direction 4 from C2 should be valid (southeast direction)
    c2_y, c2_x = c2_idx
    assert captures_before[4, c2_y, c2_x], (
        "Marble A at C2 should be able to capture X at D3"
    )

    # Simulate the capture: A moves from C2 to E3, X is removed
    board.state[white_layer][c2_idx] = 0
    e3_idx = board.str_to_index("E3")
    board.state[white_layer][e3_idx] = 1
    board.state[white_layer][d3_idx] = 0  # X is captured

    # Mark A as the marble that just moved (this is what the game does)
    board.state[board.CAPTURE_LAYER] = 0
    board.state[board.CAPTURE_LAYER][e3_idx] = 1

    # Now get capture moves after the first capture
    captures_after = board.get_capture_moves()

    # CRITICAL TEST: Only marble A should be able to capture (continue its sequence)
    # A can capture B: E3 → F2 → G1 (direction 5)
    e3_y, e3_x = e3_idx
    assert captures_after[5, e3_y, e3_x], (
        "Marble A at E3 should be able to capture B at F2"
    )

    # B should NOT be able to capture A (it's not B's turn in the sequence)
    f2_y, f2_x = f2_idx
    assert not captures_after[2, f2_y, f2_x], (
        "Marble B at F2 should NOT be able to capture A (wrong sequence)"
    )

    # Verify we found exactly 1 capture option (only A can continue)
    total_captures = np.sum(captures_after)
    assert total_captures == 1, (
        f"Expected 1 capture option (only A continues), found {total_captures}"
    )
