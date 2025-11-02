"""
Tests for action space transformation methods.

These methods transform action masks (valid move indicators) under board symmetries,
ensuring that when a board state is rotated/mirrored, the corresponding actions are
also transformed correctly. This is critical for ML/AI applications where state
canonicalization must be accompanied by action transformation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_board import ZertzBoard


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_board():
    """37-ring board for testing."""
    return ZertzBoard(rings=37)


@pytest.fixture
def medium_board():
    """48-ring board for testing."""
    return ZertzBoard(rings=48)


@pytest.fixture
def large_board():
    """61-ring board for testing."""
    return ZertzBoard(rings=61)


# ============================================================================
# Direction Index Map Tests
# ============================================================================


class TestDirectionIndexMap:
    """Test dir_index_map() method for transforming capture directions."""

    def test_identity_preserves_directions(self, small_board):
        """Identity transform (no rotation, no mirror) should map each direction to itself."""
        idx_map = small_board.canonicalizer._dir_index_map(rot60_k=0, mirror=False)

        # All directions should map to themselves
        for i in range(6):
            assert idx_map[i] == i, f"Direction {i} should map to itself under identity"

    def test_rotation_shifts_directions(self, small_board):
        """Rotation by 60° should shift direction indices by 1 (modulo 6)."""
        # Test each rotation angle
        for k in range(1, 6):
            idx_map = small_board.canonicalizer._dir_index_map(rot60_k=k, mirror=False)

            # Each direction should shift by k positions
            for i in range(6):
                expected = (i + k) % 6
                assert idx_map[i] == expected, (
                    f"Rotation by {k*60}° should map direction {i} to {expected}, got {idx_map[i]}"
                )

    def test_180_rotation_inverts_directions(self, small_board):
        """180° rotation should map each direction to its opposite."""
        idx_map = small_board.canonicalizer._dir_index_map(rot60_k=3, mirror=False)

        # Each direction should map to opposite direction (offset by 3)
        for i in range(6):
            expected = (i + 3) % 6
            assert idx_map[i] == expected, (
                f"180° rotation should map direction {i} to {expected}, got {idx_map[i]}"
            )

    def test_mirror_transformation(self, small_board):
        """Mirror transformation should produce valid direction mapping."""
        idx_map = small_board.canonicalizer._dir_index_map(rot60_k=0, mirror=True)

        # Mirror should be an involution (applying twice gives identity)
        # Apply the mapping twice
        double_map = {i: idx_map[idx_map[i]] for i in range(6)}

        # Should return to identity
        for i in range(6):
            assert double_map[i] == i, (
                f"Mirror applied twice should give identity, but direction {i} maps to {double_map[i]}"
            )

    def test_combined_rotation_and_mirror(self, small_board):
        """Combined rotation and mirror should produce valid mapping."""
        # Test a few combinations
        for k in [1, 2, 3]:
            idx_map = small_board.canonicalizer._dir_index_map(rot60_k=k, mirror=True)

            # Should produce a valid permutation (all indices 0-5 present)
            mapped_values = sorted(idx_map.values())
            assert mapped_values == list(range(6)), (
                f"Rotation {k*60}° + mirror should produce valid permutation, got {mapped_values}"
            )

    @pytest.mark.parametrize("board_fixture", ["small_board", "medium_board", "large_board"])
    def test_all_board_sizes(self, request, board_fixture):
        """Direction mapping should work for all board sizes."""
        board = request.getfixturevalue(board_fixture)

        # Test identity
        idx_map = board.canonicalizer._dir_index_map(rot60_k=0, mirror=False)
        for i in range(6):
            assert idx_map[i] == i, f"{board.rings}-ring board: identity failed"

        # Test 60° rotation
        idx_map = board.canonicalizer._dir_index_map(rot60_k=1, mirror=False)
        for i in range(6):
            assert idx_map[i] == (i + 1) % 6, f"{board.rings}-ring board: rotation failed"

    def test_direction_consistency_with_vectors(self, small_board):
        """Direction mapping should be consistent with actual direction vectors."""
        # Get the actual direction vectors
        dirs = small_board.DIRECTIONS

        # Test 60° rotation
        idx_map = small_board.canonicalizer._dir_index_map(rot60_k=1, mirror=False)

        # Each mapped direction should be the rotated version of the original
        for i in range(6):
            original_dir = dirs[i]
            mapped_idx = idx_map[i]

            # The mapped index should point to a valid direction
            assert 0 <= mapped_idx < 6, f"Invalid mapped index: {mapped_idx}"


# ============================================================================
# Capture Mask Transformation Tests
# ============================================================================


class TestTransformCaptureMask:
    """Test transform_capture_mask() method."""

    def test_identity_preserves_mask(self, small_board):
        """Identity transform should preserve capture mask exactly."""
        # Create a simple capture mask with one valid capture
        cap_mask = np.zeros((6, small_board.config.width, small_board.config.width), dtype=bool)

        # Set a valid capture at position (3, 3) in direction 0
        cap_mask[0, 3, 3] = True

        # Apply identity transform
        transformed = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=0, mirror=False)

        # Should be identical
        assert np.array_equal(transformed, cap_mask), "Identity should preserve capture mask"

    def test_preserves_capture_count(self, small_board):
        """Transformation should preserve the total number of valid captures."""
        # Get actual valid capture moves from the board
        small_board.state[1, *small_board.str_to_index("D4")] = 1  # White at center
        small_board.state[2, *small_board.str_to_index("E4")] = 1  # Gray to the right

        cap_mask = small_board.get_capture_moves()
        original_count = np.sum(cap_mask)

        # Test various transformations
        for k in range(6):
            for mirror in [False, True]:
                transformed = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=k, mirror=mirror)
                transformed_count = np.sum(transformed)

                assert transformed_count == original_count, (
                    f"Transform (rot={k*60}°, mirror={mirror}) changed capture count: "
                    f"{original_count} -> {transformed_count}"
                )

    def test_rotation_moves_captures_correctly(self, small_board):
        """Rotating the mask should move captures to rotated positions."""
        # Place marbles to create a capture opportunity
        # White at D4 (center), gray at E4 (to the right)
        small_board.state[1, *small_board.str_to_index("D4")] = 1
        small_board.state[2, *small_board.str_to_index("E4")] = 1

        # Get the state and capture mask
        original_state = small_board.state.copy()
        cap_mask = small_board.get_capture_moves()

        # Rotate both state and mask by 60°
        rotated_state = small_board.canonicalizer.transform_state_hex(original_state, rot60_k=1)
        rotated_mask = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=1, mirror=False)

        # Create a new board with rotated state to get expected mask
        rotated_board = ZertzBoard(clone=small_board)
        rotated_board.state = rotated_state
        expected_mask = rotated_board.get_capture_moves()

        # The rotated mask should match the expected mask
        assert np.array_equal(rotated_mask, expected_mask), (
            "Rotated capture mask doesn't match expected mask from rotated state"
        )

    def test_mirror_transformation_valid(self, small_board):
        """Mirror transformation should produce valid capture mask."""
        # Create a capture scenario
        small_board.state[1, *small_board.str_to_index("A4")] = 1
        small_board.state[2, *small_board.str_to_index("B4")] = 1

        cap_mask = small_board.get_capture_moves()

        # Apply mirror
        mirrored_mask = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=0, mirror=True)

        # Should have same number of captures
        assert np.sum(mirrored_mask) == np.sum(cap_mask), (
            "Mirror transformation changed number of captures"
        )

    def test_empty_mask_remains_empty(self, small_board):
        """Transforming an empty mask should keep it empty."""
        cap_mask = np.zeros((6, small_board.config.width, small_board.config.width), dtype=bool)

        for k in range(6):
            for mirror in [False, True]:
                transformed = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=k, mirror=mirror)
                assert np.sum(transformed) == 0, (
                    f"Transforming empty mask produced non-zero captures (k={k}, mirror={mirror})"
                )

    def test_full_mask_remains_full(self, small_board):
        """Transforming a full mask should preserve the count."""
        # Create mask with captures at all valid board positions
        cap_mask = np.zeros((6, small_board.config.width, small_board.config.width), dtype=bool)

        # Mark all positions that have rings
        for y in range(small_board.config.width):
            for x in range(small_board.config.width):
                if small_board.state[small_board.RING_LAYER, y, x] == 1:
                    # Mark all directions as valid (even if not realistic)
                    cap_mask[:, y, x] = True

        original_count = np.sum(cap_mask)

        # Test transformations preserve count
        for k in [0, 1, 3]:
            transformed = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=k, mirror=False)
            assert np.sum(transformed) == original_count, (
                f"Full mask transformation changed count (k={k})"
            )

    @pytest.mark.parametrize("board_fixture", ["small_board", "medium_board", "large_board"])
    def test_works_for_all_board_sizes(self, request, board_fixture):
        """Capture mask transformation should work for all board sizes."""
        board = request.getfixturevalue(board_fixture)

        # Create a simple mask
        cap_mask = np.zeros((6, board.config.width, board.config.width), dtype=bool)
        cap_mask[0, board.config.width // 2, board.config.width // 2] = True

        # Test identity
        transformed = board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=0, mirror=False)
        assert np.array_equal(transformed, cap_mask), (
            f"{board.rings}-ring board: identity transform failed"
        )


# ============================================================================
# Placement Mask Transformation Tests
# ============================================================================


class TestTransformPutMask:
    """Test transform_put_mask() method."""

    def test_identity_preserves_mask(self, small_board):
        """Identity transform should preserve placement mask exactly."""
        # Get actual placement moves
        put_mask = small_board.get_placement_moves()

        # Apply identity transform
        transformed = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=0, mirror=False)

        # Should be identical
        assert np.array_equal(transformed, put_mask), "Identity should preserve placement mask"

    def test_preserves_placement_count(self, small_board):
        """Transformation should preserve the total number of valid placements."""
        put_mask = small_board.get_placement_moves()
        original_count = np.sum(put_mask)

        # Test various transformations
        for k in range(6):
            for mirror in [False, True]:
                transformed = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=k, mirror=mirror)
                transformed_count = np.sum(transformed)

                assert transformed_count == original_count, (
                    f"Transform (rot={k*60}°, mirror={mirror}) changed placement count: "
                    f"{original_count} -> {transformed_count}"
                )

    def test_no_removal_index_preserved(self, small_board):
        """The special 'no removal' index (W²) should be handled correctly."""
        put_mask = small_board.get_placement_moves()

        # Get count of placements with no removal
        no_removal_idx = small_board.config.width ** 2
        original_no_removal = np.sum(put_mask[:, :, no_removal_idx])

        # Transform
        transformed = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=1, mirror=False)
        transformed_no_removal = np.sum(transformed[:, :, no_removal_idx])

        assert transformed_no_removal == original_no_removal, (
            "No-removal index count changed after transformation"
        )

    def test_rotation_moves_placements_correctly(self, small_board):
        """Rotating the mask should move placements to rotated positions."""
        # Get initial placement moves
        put_mask = small_board.get_placement_moves()

        # Rotate the board state
        rotated_state = small_board.canonicalizer.transform_state_hex(
            small_board.state, rot60_k=1
        )

        # Rotate the placement mask
        rotated_mask = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=1, mirror=False)

        # Create board with rotated state and get expected mask
        rotated_board = ZertzBoard(clone=small_board)
        rotated_board.state = rotated_state
        expected_mask = rotated_board.get_placement_moves()

        # Should match
        assert np.array_equal(rotated_mask, expected_mask), (
            "Rotated placement mask doesn't match expected mask"
        )

    def test_mirror_transformation_valid(self, small_board):
        """Mirror transformation should produce valid placement mask."""
        put_mask = small_board.get_placement_moves()

        # Apply mirror
        mirrored_mask = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=0, mirror=True)

        # Should have same number of placements
        assert np.sum(mirrored_mask) == np.sum(put_mask), (
            "Mirror transformation changed number of placements"
        )

    def test_empty_mask_remains_empty(self, small_board):
        """Transforming an empty mask should keep it empty."""
        put_mask = np.zeros((3, small_board.config.width**2, small_board.config.width**2 + 1), dtype=bool)

        for k in range(6):
            transformed = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=k, mirror=False)
            assert np.sum(transformed) == 0, (
                f"Transforming empty placement mask produced non-zero moves (k={k})"
            )

    def test_preserves_marble_type_separation(self, small_board):
        """Each marble type's moves should be preserved separately."""
        put_mask = small_board.get_placement_moves()

        # Count moves for each marble type
        original_counts = [np.sum(put_mask[i]) for i in range(3)]

        # Transform
        transformed = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=2, mirror=False)
        transformed_counts = [np.sum(transformed[i]) for i in range(3)]

        assert original_counts == transformed_counts, (
            f"Marble type move counts changed: {original_counts} -> {transformed_counts}"
        )

    @pytest.mark.parametrize("board_fixture", ["small_board", "medium_board", "large_board"])
    def test_works_for_all_board_sizes(self, request, board_fixture):
        """Placement mask transformation should work for all board sizes."""
        board = request.getfixturevalue(board_fixture)

        # Get placement moves
        put_mask = board.get_placement_moves()

        # Test identity
        transformed = board.canonicalizer._transform_put_mask(put_mask, rot60_k=0, mirror=False)
        assert np.array_equal(transformed, put_mask), (
            f"{board.rings}-ring board: identity transform failed"
        )

    def test_combined_rotation_and_mirror(self, small_board):
        """Combined rotation and mirror should work correctly."""
        put_mask = small_board.get_placement_moves()
        original_count = np.sum(put_mask)

        # Test several combinations
        for k in [1, 2, 3]:
            transformed = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=k, mirror=True)
            assert np.sum(transformed) == original_count, (
                f"Combined rotation ({k*60}°) + mirror changed placement count"
            )


# ============================================================================
# Integration Tests
# ============================================================================


class TestActionTransformationIntegration:
    """Integration tests for action transformations with actual game scenarios."""

    def test_symmetrical_positions_have_symmetrical_actions(self, small_board):
        """Symmetrically equivalent board states should have equivalent action counts."""
        # Place marbles in a pattern
        small_board.state[1, *small_board.str_to_index("D4")] = 1
        small_board.state[2, *small_board.str_to_index("E4")] = 1
        small_board.state[3, *small_board.str_to_index("D3")] = 1

        # Get action masks
        put_mask = small_board.get_placement_moves()
        cap_mask = small_board.get_capture_moves()

        original_put_count = np.sum(put_mask)
        original_cap_count = np.sum(cap_mask)

        # Rotate state
        rotated_state = small_board.canonicalizer.transform_state_hex(
            small_board.state, rot60_k=1
        )

        # Rotate actions
        rotated_put = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=1)
        rotated_cap = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=1)

        # Create board with rotated state
        rotated_board = ZertzBoard(clone=small_board)
        rotated_board.state = rotated_state

        # Get actions from rotated board
        expected_put = rotated_board.get_placement_moves()
        expected_cap = rotated_board.get_capture_moves()

        # Counts should match
        assert np.sum(rotated_put) == np.sum(expected_put), (
            "Rotated placement mask count doesn't match rotated board"
        )
        assert np.sum(rotated_cap) == np.sum(expected_cap), (
            "Rotated capture mask count doesn't match rotated board"
        )

        # Masks should be identical
        assert np.array_equal(rotated_put, expected_put), (
            "Rotated placement mask doesn't match expected"
        )
        assert np.array_equal(rotated_cap, expected_cap), (
            "Rotated capture mask doesn't match expected"
        )

    def test_inverse_transform_recovers_original(self, small_board):
        """Applying a transform then its inverse should recover the original mask."""
        # Get action masks
        put_mask = small_board.get_placement_moves()
        cap_mask = small_board.get_capture_moves()

        # Apply rotation
        k = 2  # 120°
        rotated_put = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=k)
        rotated_cap = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=k)

        # Apply inverse rotation (360° - 120° = 240° = 4 * 60°)
        inv_k = (6 - k) % 6
        recovered_put = small_board.canonicalizer._transform_put_mask(rotated_put, rot60_k=inv_k)
        recovered_cap = small_board.canonicalizer._transform_capture_mask(rotated_cap, rot60_k=inv_k)

        # Should recover original
        assert np.array_equal(recovered_put, put_mask), (
            "Inverse rotation didn't recover original placement mask"
        )
        assert np.array_equal(recovered_cap, cap_mask), (
            "Inverse rotation didn't recover original capture mask"
        )

    def test_transformation_with_removed_rings(self, small_board):
        """Transformations should work correctly when rings have been removed."""
        # Remove some edge rings
        for pos in ["A4", "D7", "G1"]:
            y, x = small_board.str_to_index(pos)
            small_board.state[small_board.RING_LAYER, y, x] = 0

        # Get action masks
        put_mask = small_board.get_placement_moves()
        cap_mask = small_board.get_capture_moves()

        # Transform
        transformed_put = small_board.canonicalizer._transform_put_mask(put_mask, rot60_k=1)
        transformed_cap = small_board.canonicalizer._transform_capture_mask(cap_mask, rot60_k=1)

        # Should preserve counts
        assert np.sum(transformed_put) == np.sum(put_mask), (
            "Transformation with removed rings changed placement count"
        )
        assert np.sum(transformed_cap) == np.sum(cap_mask), (
            "Transformation with removed rings changed capture count"
        )