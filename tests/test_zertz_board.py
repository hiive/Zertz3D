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

@pytest.fixture(params=[
    ZertzBoard.SMALL_BOARD_37,
    ZertzBoard.MEDIUM_BOARD_48,
    ZertzBoard.LARGE_BOARD_61
])
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

    def test_flat_to_2d(self, board):
        """Test conversion from flat index to 2D coordinates."""
        # Test corner positions
        y, x = board._flat_to_2d(0)
        assert y == 0 and x == 0

        # Test that conversion is consistent
        for flat_idx in [0, 5, 10, board.width - 1]:
            y, x = board._flat_to_2d(flat_idx)
            assert 0 <= y < board.width
            assert 0 <= x < board.width

    def test_2d_to_flat(self, board):
        """Test conversion from 2D coordinates to flat index."""
        # Test corner
        flat = board._2d_to_flat(0, 0)
        assert flat == 0

        # Test roundtrip conversion
        for y in range(board.width):
            for x in range(board.width):
                flat = board._2d_to_flat(y, x)
                y2, x2 = board._flat_to_2d(flat)
                assert y == y2 and x == x2

    def test_str_to_index_and_back(self, board):
        """Test string to index conversion and back."""
        # Get all valid positions from the board layout
        if board.letter_layout is not None:
            valid_positions = board.letter_layout[board.letter_layout != '']
            for pos_str in valid_positions:
                # Convert to index and back
                idx = board.str_to_index(pos_str)
                # Only test roundtrip if ring still exists
                if board.state[board.RING_LAYER, idx[0], idx[1]] == 1:
                    result_str = board.index_to_str(idx)
                    assert result_str == pos_str, f"Failed roundtrip for {pos_str}"


# ============================================================================
# Mirror Coordinate Tests
# ============================================================================

class TestMirrorCoordinates:
    """Test mirroring of coordinates."""

    @pytest.mark.parametrize("pos_str,expected_mirror", [
        #        A4  b5  c6  D7
        #      a3  B3  C5  d6  e6
        #    a2  b3  c4  d5  E5  f5
        #  A1  b2  c3  D4  e4  f4  G4
        #    b1  C2  d3  e3  f3  g3
        #      c1  d2  e2  f2  g2
        #        D1  e1  f1  G1
        # 37-ring board: mirror swaps x and y axes
        ("A1", "D7"),  # Corner positions 1
        ("D1", "G4"),  # Corner positions 2
        ("D4", "D4"),  # Center stays at center
        ("A4", "A4"),  # On diagonal
        ("D7", "A1"),  # Reverse of A1
        ("B3", "C5"),  # General position
        ("C2", "E5"),  # Another position
        ("G1", "G1"),  # On diagonal
    ])
    def test_mirror_coords_small_board(self, small_board, pos_str, expected_mirror):
        """Test coordinate mirroring on 37-ring board."""
        # Convert position string to indices
        y, x = small_board.str_to_index(pos_str)

        # Apply mirror transformation
        new_y, new_x = small_board._mirror_coords(y, x)

        # Convert back to string
        result_str = small_board.index_to_str((new_y, new_x))

        assert result_str == expected_mirror, f"Mirror of {pos_str} should be {expected_mirror}, got {result_str}"

    @pytest.mark.parametrize("pos_str,expected_mirror", [
        # 48-ring board: mirror swaps x and y axes
        ("D5", "D5"),  # Center area (stays put)
        ("E4", "E4"),  # Center area (stays put)
        ("A5", "A5"),  # On diagonal
        ("D8", "A2"),  # Off diagonal
        ("H1", "H1"),  # On diagonal
    ])
    def test_mirror_coords_medium_board(self, medium_board, pos_str, expected_mirror):
        """Test coordinate mirroring on 48-ring board."""
        y, x = medium_board.str_to_index(pos_str)
        new_y, new_x = medium_board._mirror_coords(y, x)
        result_str = medium_board.index_to_str((new_y, new_x))
        assert result_str == expected_mirror

    @pytest.mark.parametrize("pos_str,expected_mirror", [
        # 61-ring board: mirror swaps x and y axes (ABCDEFGHJ, skipping I)
        ("A1", "E9"),  # Corner positions
        ("E5", "E5"),  # Center (stays put)
        ("A5", "A5"),  # On diagonal
        ("E8", "B2"),  # Off diagonal
        ("J1", "J1"),  # On diagonal
    ])
    def test_mirror_coords_large_board(self, large_board, pos_str, expected_mirror):
        """Test coordinate mirroring on 61-ring board."""
        y, x = large_board.str_to_index(pos_str)
        new_y, new_x = large_board._mirror_coords(y, x)
        result_str = large_board.index_to_str((new_y, new_x))
        assert result_str == expected_mirror


# ============================================================================
# Rotate Coordinate Tests
# ============================================================================

class TestRotateCoordinates:
    """Test 180-degree rotation of coordinates."""

    @pytest.mark.parametrize("pos_str,expected_rotated", [
        # 37-ring board: Cartesian 180° rotation (point reflection through center)
        ("A1", "G4"),  # Bottom-left to top-right
        ("D4", "D4"),  # Center stays at center
        ("A4", "G1"),  # Top-left to bottom-right
        ("D7", "D1"),  # Top to bottom
        ("B3", "F3"),  # Symmetric reflection
        ("C2", "E5"),  # General position
        ("G1", "A4"),  # Bottom-right to top-left
    ])
    def test_rotate_coords_small_board(self, small_board, pos_str, expected_rotated):
        """Test 180° rotation on 37-ring board."""
        y, x = small_board.str_to_index(pos_str)
        new_y, new_x = small_board._rotate_coords(y, x)
        result_str = small_board.index_to_str((new_y, new_x))
        assert result_str == expected_rotated, f"Rotation of {pos_str} should be {expected_rotated}, got {result_str}"

    @pytest.mark.parametrize("pos_str,expected_rotated", [
        # 48-ring board: Cartesian 180° rotation
        ("D5", "E4"),  # Near center
        ("E4", "D5"),  # Reverse
        ("A5", "H1"),  # Top-left area to bottom-right
        ("D8", "E1"),  # Top to bottom
        ("H1", "A5"),  # Bottom-right to top-left area
    ])
    def test_rotate_coords_medium_board(self, medium_board, pos_str, expected_rotated):
        """Test 180° rotation on 48-ring board."""
        y, x = medium_board.str_to_index(pos_str)
        new_y, new_x = medium_board._rotate_coords(y, x)
        result_str = medium_board.index_to_str((new_y, new_x))
        assert result_str == expected_rotated

    @pytest.mark.parametrize("pos_str,expected_rotated", [
        # 61-ring board: Cartesian 180° rotation
        ("A1", "J5"),  # Bottom-left to top-right
        ("E5", "E5"),  # Center stays at center
        ("A5", "J1"),  # Left to right
        ("E8", "E2"),  # Top to bottom
        ("J1", "A5"),  # Bottom-right to top-left
    ])
    def test_rotate_coords_large_board(self, large_board, pos_str, expected_rotated):
        """Test 180° rotation on 61-ring board."""
        y, x = large_board.str_to_index(pos_str)
        new_y, new_x = large_board._rotate_coords(y, x)
        result_str = large_board.index_to_str((new_y, new_x))
        assert result_str == expected_rotated


# ============================================================================
# Action Translation Tests
# ============================================================================

class TestMirrorAction:
    """Test mirror_action for PUT and CAP action types."""

    def test_mirror_put_action_preserves_shape(self, board):
        """Test that mirror_action preserves PUT action array shape."""
        # Create a dummy PUT action array
        put_actions = np.zeros((3, board.width ** 2, board.width ** 2 + 1), dtype=bool)
        put_actions[0, 5, 10] = True  # Set one action

        mirrored = board.mirror_action('PUT', put_actions)

        assert mirrored.shape == put_actions.shape
        assert mirrored.dtype == put_actions.dtype

    def test_mirror_cap_action_preserves_shape(self, board):
        """Test that mirror_action preserves CAP action array shape."""
        cap_actions = np.zeros((6, board.width, board.width), dtype=bool)
        cap_actions[0, 2, 3] = True

        mirrored = board.mirror_action('CAP', cap_actions)

        assert mirrored.shape == cap_actions.shape
        assert mirrored.dtype == cap_actions.dtype

    def test_mirror_put_action_coordinates(self, small_board):
        """Test that PUT action coordinates are correctly mirrored."""
        # Test a specific PUT action: place marble at B3, remove C5
        put_idx = small_board.str_to_index("B3")
        rem_idx = small_board.str_to_index("C5")

        put_flat = small_board._2d_to_flat(*put_idx)
        rem_flat = small_board._2d_to_flat(*rem_idx)

        # Create action array with this action set
        put_actions = np.zeros((3, small_board.width ** 2, small_board.width ** 2 + 1), dtype=bool)
        put_actions[1, put_flat, rem_flat] = True  # Gray marble

        # Mirror the actions
        mirrored = small_board.mirror_action('PUT', put_actions)

        # Expected mirrored positions: B3 -> C5, C5 -> B3 (mirror swaps x and y)
        expected_put_idx = small_board.str_to_index("C5")
        expected_rem_idx = small_board.str_to_index("B3")

        expected_put_flat = small_board._2d_to_flat(*expected_put_idx)
        expected_rem_flat = small_board._2d_to_flat(*expected_rem_idx)

        # Check that the mirrored action is set at the expected position
        assert mirrored[1, expected_put_flat, expected_rem_flat]


class TestRotateAction:
    """Test rotate_action for PUT and CAP action types."""

    def test_rotate_put_action_preserves_shape(self, board):
        """Test that rotate_action preserves PUT action array shape."""
        put_actions = np.zeros((3, board.width ** 2, board.width ** 2 + 1), dtype=bool)
        put_actions[0, 5, 10] = True

        rotated = board.rotate_action('PUT', put_actions)

        assert rotated.shape == put_actions.shape
        assert rotated.dtype == put_actions.dtype

    def test_rotate_cap_action_preserves_shape(self, board):
        """Test that rotate_action preserves CAP action array shape."""
        cap_actions = np.zeros((6, board.width, board.width), dtype=bool)
        cap_actions[0, 2, 3] = True

        rotated = board.rotate_action('CAP', cap_actions)

        assert rotated.shape == cap_actions.shape
        assert rotated.dtype == cap_actions.dtype

    def test_rotate_put_action_coordinates(self, small_board):
        """Test that PUT action coordinates are correctly rotated 180°."""
        # Test a specific PUT action: place at B3, remove at C5
        put_idx = small_board.str_to_index("B3")
        rem_idx = small_board.str_to_index("C5")

        put_flat = small_board._2d_to_flat(*put_idx)
        rem_flat = small_board._2d_to_flat(*rem_idx)

        # Create action array
        put_actions = np.zeros((3, small_board.width ** 2, small_board.width ** 2 + 1), dtype=bool)
        put_actions[2, put_flat, rem_flat] = True  # Black marble

        # Rotate 180°
        rotated = small_board.rotate_action('PUT', put_actions)

        # Expected rotated positions: B3 -> F3, C5 -> E2 (180° point reflection)
        expected_put_idx = small_board.str_to_index("F3")
        expected_rem_idx = small_board.str_to_index("E2")

        expected_put_flat = small_board._2d_to_flat(*expected_put_idx)
        expected_rem_flat = small_board._2d_to_flat(*expected_rem_idx)

        # Check that the rotated action is set at the expected position
        assert rotated[2, expected_put_flat, expected_rem_flat]


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
            assert board._is_inbounds((ny, nx)), f"Neighbor {(ny, nx)} should be in bounds"

        # Test corner position - may have fewer neighbors
        corner_pos = (0, 0)
        corner_neighbors = board.get_neighbors(corner_pos)
        assert len(corner_neighbors) <= 6, "Corner position should have <= 6 neighbors"
        assert len(corner_neighbors) >= 2, "Corner position should have at least 2 neighbors"

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
        assert moves.shape == (3, board.width ** 2, board.width ** 2 + 1)

    def test_get_capture_moves_shape(self, board):
        """Test that capture moves array has correct shape."""
        moves = board.get_capture_moves()
        assert moves.shape == (6, board.width, board.width)


class TestSymmetryOperations:
    """Test board state symmetry operations."""

    def test_get_rotational_symmetries(self, small_board):
        """Test 180° board state rotation preserves shape and transforms marbles correctly."""
        board = small_board

        # Place a marble at a known position
        test_pos = board.str_to_index('B3')
        white_layer = board.MARBLE_TO_LAYER['w']
        board.state[white_layer][test_pos] = 1

        # Get rotated state
        rotated = board._get_rotational_symmetries()

        # Check shape is preserved
        assert rotated.shape == board.state.shape

        # Check that the marble has moved to the rotated position
        # B3 rotates 180° to F3
        rotated_pos = board.str_to_index('F3')
        assert rotated[white_layer][rotated_pos] == 1, "Marble should be at rotated position"
        assert rotated[white_layer][test_pos] == 0, "Original position should be empty after rotation"

    def test_get_mirror_symmetries(self, small_board):
        """Test board state mirroring preserves shape and transforms marbles correctly."""
        board = small_board

        # Place a marble at a known position
        test_pos = board.str_to_index('B3')
        gray_layer = board.MARBLE_TO_LAYER['g']
        board.state[gray_layer][test_pos] = 1

        # Get mirrored state
        mirrored = board._get_mirror_symmetries()

        # Check shape is preserved
        assert mirrored.shape == board.state.shape

        # Check that the marble has moved to the mirrored position
        # B3 mirrors to C5
        mirrored_pos = board.str_to_index('C5')
        assert mirrored[gray_layer][mirrored_pos] == 1, "Marble should be at mirrored position"
        assert mirrored[gray_layer][test_pos] == 0, "Original position should be empty after mirroring"

    def test_get_state_symmetries(self, board):
        """Test that get_state_symmetries returns mirror, rotate, and mirror+rotate transformations."""
        symmetries = board.get_state_symmetries()

        # Should return 3 symmetries: mirror, rotate, mirror+rotate
        assert len(symmetries) == 3, "Should return 3 symmetries"

        # Each symmetry is a tuple (transform_id, transformed_state)
        # All symmetries should have same shape as original state
        for transform_id, sym_state in symmetries:
            assert isinstance(transform_id, int), "Transform ID should be an integer"
            assert sym_state.shape == board.state.shape, "Symmetry should preserve state shape"


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
    c2_idx = board.str_to_index('C2')
    white_layer = board.MARBLE_TO_LAYER['w']
    board.state[white_layer][c2_idx] = 1

    # Marble X (white) at D3 (will be captured)
    d3_idx = board.str_to_index('D3')
    board.state[white_layer][d3_idx] = 1

    # Marble B (gray) at F2
    f2_idx = board.str_to_index('F2')
    gray_layer = board.MARBLE_TO_LAYER['g']
    board.state[gray_layer][f2_idx] = 1

    # Get initial capture moves - should show A can capture X
    captures_before = board.get_capture_moves()

    # Verify A can capture X (C2 → D3 → E3)
    # Direction 4 from C2 should be valid (southeast direction)
    c2_y, c2_x = c2_idx
    assert captures_before[4, c2_y, c2_x], "Marble A at C2 should be able to capture X at D3"

    # Simulate the capture: A moves from C2 to E3, X is removed
    board.state[white_layer][c2_idx] = 0
    e3_idx = board.str_to_index('E3')
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
    assert captures_after[5, e3_y, e3_x], "Marble A at E3 should be able to capture B at F2"

    # B should NOT be able to capture A (it's not B's turn in the sequence)
    f2_y, f2_x = f2_idx
    assert not captures_after[2, f2_y, f2_x], "Marble B at F2 should NOT be able to capture A (wrong sequence)"

    # Verify we found exactly 1 capture option (only A can continue)
    total_captures = np.sum(captures_after)
    assert total_captures == 1, f"Expected 1 capture option (only A continues), found {total_captures}"