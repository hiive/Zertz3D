"""
Tests for geometric ring removal validation.

Compares the original adjacency-based ring removal logic with the new
geometric collision-detection approach to ensure they produce identical results.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_board import ZertzBoard


@pytest.fixture
def small_board():
    """37-ring board"""
    return ZertzBoard(rings=ZertzBoard.SMALL_BOARD_37)


@pytest.fixture
def medium_board():
    """48-ring board"""
    return ZertzBoard(rings=ZertzBoard.MEDIUM_BOARD_48)


@pytest.fixture
def large_board():
    """61-ring board"""
    return ZertzBoard(rings=ZertzBoard.LARGE_BOARD_61)


class TestGeometricRingRemoval:
    """Test that geometric ring removal matches the original logic."""

    def test_both_methods_agree_on_full_board(self, small_board):
        """On a full board, no rings should be removable by either method."""
        # Check all rings
        for y in range(small_board.width):
            for x in range(small_board.width):
                if small_board.state[small_board.RING_LAYER, y, x] == 1:
                    original = small_board._is_removable((y, x))
                    geometric = small_board._is_removable_geometric((y, x))

                    assert original == geometric, \
                        f"Mismatch at {small_board.index_to_str((y, x))}: " \
                        f"original={original}, geometric={geometric}"

    def test_corner_rings_removable_when_two_neighbors_missing(self, small_board):
        """Test that corner rings are removable when two neighbors are removed."""
        # Remove two neighbors of A4 (top-left corner)
        # A4's neighbors are: A3 (below), B5 (right), B4 (below-right)
        # Remove A3 and B4 to create a gap

        a4_pos = small_board.str_to_index("A4")
        a3_pos = small_board.str_to_index("A3")
        b4_pos = small_board.str_to_index("B4")

        # Remove the two neighbor rings
        small_board.state[small_board.RING_LAYER, a3_pos[0], a3_pos[1]] = 0
        small_board.state[small_board.RING_LAYER, b4_pos[0], b4_pos[1]] = 0

        # Now check if A4 is removable
        original = small_board._is_removable(a4_pos)
        geometric = small_board._is_removable_geometric(a4_pos)

        assert original == geometric is True, \
            f"A4 should be removable: original={original}, geometric={geometric}"

    def test_center_ring_not_removable_on_full_board(self, small_board):
        """Center ring (D4) should not be removable on a full board."""
        d4_pos = small_board.str_to_index("D4")

        original = small_board._is_removable(d4_pos)
        geometric = small_board._is_removable_geometric(d4_pos)

        assert original == geometric is False, \
            f"D4 should not be removable: original={original}, geometric={geometric}"

    def test_ring_with_marble_not_removable(self, small_board):
        """Rings with marbles should not be removable."""
        # Place a marble on A4
        a4_pos = small_board.str_to_index("A4")
        small_board.state[small_board.MARBLE_LAYERS.start, a4_pos[0], a4_pos[1]] = 1

        # Remove neighbors to create gap
        a3_pos = small_board.str_to_index("A3")
        b4_pos = small_board.str_to_index("B4")
        small_board.state[small_board.RING_LAYER, a3_pos[0], a3_pos[1]] = 0
        small_board.state[small_board.RING_LAYER, b4_pos[0], b4_pos[1]] = 0

        # Check removability - should be False because of marble
        original = small_board._is_removable(a4_pos)
        geometric = small_board._is_removable_geometric(a4_pos)

        assert original == geometric is False, \
            f"A4 with marble should not be removable: original={original}, geometric={geometric}"

    @pytest.mark.parametrize("board_fixture", ["small_board", "medium_board", "large_board"])
    def test_exhaustive_comparison_on_full_board(self, request, board_fixture):
        """Exhaustively compare both methods on every ring of a full board."""
        board = request.getfixturevalue(board_fixture)
        mismatches = []

        for y in range(board.width):
            for x in range(board.width):
                if board.state[board.RING_LAYER, y, x] == 1:
                    original = board._is_removable((y, x))
                    geometric = board._is_removable_geometric((y, x))

                    if original != geometric:
                        pos_str = board.index_to_str((y, x))
                        mismatches.append(
                            f"{pos_str}: original={original}, geometric={geometric}"
                        )

        assert len(mismatches) == 0, \
            f"Found {len(mismatches)} mismatches:\n" + "\n".join(mismatches)

    @pytest.mark.parametrize("board_fixture", ["small_board", "medium_board", "large_board"])
    def test_systematic_ring_removal_patterns(self, request, board_fixture):
        """Test various systematic ring removal patterns."""
        board = request.getfixturevalue(board_fixture)

        # Get all ring positions
        ring_positions = []
        for y in range(board.width):
            for x in range(board.width):
                if board.state[board.RING_LAYER, y, x] == 1:
                    ring_positions.append((y, x))

        # Test removing each ring one at a time and checking all others
        for remove_y, remove_x in ring_positions[:10]:  # Test first 10 to keep it manageable
            # Create a copy of the board
            test_board = ZertzBoard(clone=board)

            # Remove this ring
            test_board.state[test_board.RING_LAYER, remove_y, remove_x] = 0

            # Check all remaining rings
            mismatches = []
            for check_y, check_x in ring_positions:
                if (check_y, check_x) == (remove_y, remove_x):
                    continue  # Skip the removed ring

                if test_board.state[test_board.RING_LAYER, check_y, check_x] == 1:
                    original = test_board._is_removable((check_y, check_x))
                    geometric = test_board._is_removable_geometric((check_y, check_x))

                    if original != geometric:
                        pos_str = test_board.index_to_str((check_y, check_x))
                        mismatches.append(
                            f"{pos_str}: original={original}, geometric={geometric}"
                        )

            removed_str = board.index_to_str((remove_y, remove_x))
            assert len(mismatches) == 0, \
                f"After removing {removed_str}, found {len(mismatches)} mismatches:\n" + \
                "\n".join(mismatches)

    def test_edge_ring_patterns(self, small_board):
        """Test specific edge ring patterns that should be removable."""
        # Pattern 1: Remove two consecutive rings on the edge
        # This should make adjacent rings removable

        # Remove G1 and G2 (bottom-right edge)
        g1_pos = small_board.str_to_index("G1")
        g2_pos = small_board.str_to_index("G2")

        small_board.state[small_board.RING_LAYER, g1_pos[0], g1_pos[1]] = 0
        small_board.state[small_board.RING_LAYER, g2_pos[0], g2_pos[1]] = 0

        # Now F2 should be removable (has G1 and G2 missing consecutively)
        f2_pos = small_board.str_to_index("F2")

        original = small_board._is_removable(f2_pos)
        geometric = small_board._is_removable_geometric(f2_pos)

        assert original == geometric, \
            f"F2 removability mismatch: original={original}, geometric={geometric}"

    def test_48_ring_board_specific_patterns(self, medium_board):
        """Test specific patterns on the 48-ring board."""
        # The 48-ring board has D3 symmetry, test a few positions

        # Test a corner position
        a5_pos = medium_board.str_to_index("A5")
        a4_pos = medium_board.str_to_index("A4")
        b5_pos = medium_board.str_to_index("B5")

        # Remove two consecutive neighbors
        medium_board.state[medium_board.RING_LAYER, a4_pos[0], a4_pos[1]] = 0
        medium_board.state[medium_board.RING_LAYER, b5_pos[0], b5_pos[1]] = 0

        # Check A5
        original = medium_board._is_removable(a5_pos)
        geometric = medium_board._is_removable_geometric(a5_pos)

        assert original == geometric, \
            f"A5 removability mismatch: original={original}, geometric={geometric}"

    def test_61_ring_board_specific_patterns(self, large_board):
        """Test specific patterns on the 61-ring board."""
        # The 61-ring board uses ABCDEFGHJ coordinate scheme (skipping I)

        # Test a corner position
        a5_pos = large_board.str_to_index("A5")
        a4_pos = large_board.str_to_index("A4")
        b5_pos = large_board.str_to_index("B5")

        # Remove two consecutive neighbors
        large_board.state[large_board.RING_LAYER, a4_pos[0], a4_pos[1]] = 0
        large_board.state[large_board.RING_LAYER, b5_pos[0], b5_pos[1]] = 0

        # Check A5
        original = large_board._is_removable(a5_pos)
        geometric = large_board._is_removable_geometric(a5_pos)

        assert original == geometric, \
            f"A5 removability mismatch: original={original}, geometric={geometric}"

    @pytest.mark.parametrize("board_fixture", ["small_board", "medium_board", "large_board"])
    def test_random_ring_removal_patterns(self, request, board_fixture):
        """Test random ring removal patterns."""
        import random
        random.seed(42)  # Reproducible tests

        board = request.getfixturevalue(board_fixture)

        # Get all ring positions
        ring_positions = []
        for y in range(board.width):
            for x in range(board.width):
                if board.state[board.RING_LAYER, y, x] == 1:
                    ring_positions.append((y, x))

        # Try 10 different random removal patterns
        for trial in range(10):
            test_board = ZertzBoard(clone=board)

            # Randomly remove 5-10 rings
            num_to_remove = random.randint(5, min(10, len(ring_positions) // 4))
            rings_to_remove = random.sample(ring_positions, num_to_remove)

            for y, x in rings_to_remove:
                test_board.state[test_board.RING_LAYER, y, x] = 0

            # Check all remaining rings
            mismatches = []
            for y, x in ring_positions:
                if test_board.state[test_board.RING_LAYER, y, x] == 1:
                    original = test_board._is_removable((y, x))
                    geometric = test_board._is_removable_geometric((y, x))

                    if original != geometric:
                        pos_str = test_board.index_to_str((y, x))
                        mismatches.append(
                            f"{pos_str}: original={original}, geometric={geometric}"
                        )

            assert len(mismatches) == 0, \
                f"Trial {trial}: found {len(mismatches)} mismatches:\n" + \
                "\n".join(mismatches)


class TestGeometricHelpers:
    """Test the geometric helper methods."""

    def test_yx_to_cartesian_conversion(self, small_board):
        """Test that yx_to_cartesian produces correct coordinates."""
        # D4 is at the center and should be at (0, 0) in centered coordinates
        d4_pos = small_board.str_to_index("D4")
        xc, yc = small_board._yx_to_cartesian(*d4_pos)

        # D4 is at array position (3, 3) for a 7x7 board
        # In axial: q = 3 - 3 = 0, r = 3 - 3 = 0
        # In Cartesian: xc = sqrt(3) * (0 + 0/2) = 0, yc = 1.5 * 0 = 0
        assert abs(xc) < 1e-10 and abs(yc) < 1e-10, \
            f"D4 should be at origin, got ({xc}, {yc})"

    def test_neighbor_distances_are_consistent(self, small_board):
        """Test that all neighbor pairs have consistent distances."""
        sqrt3 = np.sqrt(3)

        # Check a few positions and their neighbors
        test_positions = ["D4", "A4", "G1", "B3"]

        for pos_str in test_positions:
            pos = small_board.str_to_index(pos_str)
            pos_x, pos_y = small_board._yx_to_cartesian(*pos)

            neighbors = small_board.get_neighbors(pos)

            for neighbor in neighbors:
                if (small_board._is_inbounds(neighbor) and
                    small_board.state[small_board.RING_LAYER, neighbor[0], neighbor[1]] == 1):

                    neighbor_x, neighbor_y = small_board._yx_to_cartesian(*neighbor)

                    # Distance between neighbors in a hexagonal grid should be sqrt(3)
                    # (for our unit size of 1.0)
                    dist = np.sqrt((neighbor_x - pos_x)**2 + (neighbor_y - pos_y)**2)

                    # Allow small floating point error
                    assert abs(dist - sqrt3) < 1e-10, \
                        f"Distance from {pos_str} to neighbor should be √3, got {dist}"