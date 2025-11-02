"""
Tests for mask canonicalization methods.

These tests verify the public API for canonicalizing and decanonicalization
capture and put masks using named transforms.
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
# Capture Mask Canonicalization Tests
# ============================================================================


class TestCanonicalizeCaptureMask:
    """Test public API for capture mask canonicalization."""

    def test_canonicalize_with_identity_transform(self, small_board):
        """Test that identity transform preserves capture mask."""
        # Create a simple capture mask
        cap_mask = np.zeros((6, small_board.config.width, small_board.config.width), dtype=bool)
        cap_mask[0, 3, 3] = True

        # Canonicalize with identity
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_capture_mask(cap_mask, "R0")

        assert np.array_equal(canonical, cap_mask), "R0 should preserve mask"
        assert transform == "R0"
        assert inverse == "R0"

    def test_canonicalize_with_rotation(self, small_board):
        """Test canonicalizing with rotation transforms."""
        # Get actual capture mask from board
        small_board.state[1, *small_board.str_to_index("D4")] = 1
        small_board.state[2, *small_board.str_to_index("E4")] = 1
        cap_mask = small_board.get_capture_moves()

        # Canonicalize with R60
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_capture_mask(cap_mask, "R60")

        assert transform == "R60"
        assert inverse == "R300"
        assert np.sum(canonical) == np.sum(cap_mask), "Should preserve capture count"

    def test_canonicalize_with_mirror(self, small_board):
        """Test canonicalizing with mirror transforms."""
        cap_mask = small_board.get_capture_moves()

        # Canonicalize with MR120
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_capture_mask(cap_mask, "MR120")

        assert transform == "MR120"
        assert inverse == "R240M"
        assert np.sum(canonical) == np.sum(cap_mask)

    def test_canonicalize_with_translation(self, small_board):
        """Test canonicalizing with translation transforms."""
        cap_mask = small_board.get_capture_moves()

        # Canonicalize with translation
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_capture_mask(cap_mask, "T1,0")

        assert transform == "T1,0"
        assert inverse == "T-1,0"

    def test_canonicalize_with_combined_transform(self, small_board):
        """Test canonicalizing with combined translation + rotation/mirror."""
        cap_mask = small_board.get_capture_moves()

        # Canonicalize with combined transform
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_capture_mask(
            cap_mask, "T1,0_R60"
        )

        assert transform == "T1,0_R60"
        assert inverse == "R300_T-1,0"

    def test_decanonicalize_capture_mask(self, small_board):
        """Test decanonicalization of capture mask."""
        cap_mask = small_board.get_capture_moves()

        # Canonicalize
        canonical, _, inverse = small_board.canonicalizer.canonicalize_capture_mask(cap_mask, "R120")

        # Decanonicalize
        recovered = small_board.canonicalizer.decanonicalize_capture_mask(canonical, inverse)

        assert np.array_equal(recovered, cap_mask), "Decanonicalize should recover original"

    def test_round_trip_with_various_transforms(self, small_board):
        """Test canonicalize + decanonicalize round trip with various transforms."""
        # Place marbles to create captures
        small_board.state[1, *small_board.str_to_index("D4")] = 1
        small_board.state[2, *small_board.str_to_index("E4")] = 1
        cap_mask = small_board.get_capture_moves()

        # Test only rotation/mirror transforms (no translation) as they're always valid
        test_transforms = [
            "R0", "R60", "R120", "R180", "R240", "R300",
            "MR0", "MR60", "R60M", "R120M"
        ]

        for transform_name in test_transforms:
            canonical, _, inverse = small_board.canonicalizer.canonicalize_capture_mask(
                cap_mask, transform_name
            )
            recovered = small_board.canonicalizer.decanonicalize_capture_mask(canonical, inverse)

            assert np.array_equal(recovered, cap_mask), (
                f"Round trip failed for {transform_name}"
            )

    @pytest.mark.parametrize("board_fixture", ["small_board", "medium_board", "large_board"])
    def test_works_for_all_board_sizes(self, request, board_fixture):
        """Test that mask canonicalization works for all board sizes."""
        board = request.getfixturevalue(board_fixture)

        cap_mask = board.get_capture_moves()

        canonical, transform, inverse = board.canonicalizer.canonicalize_capture_mask(
            cap_mask, "R0"
        )

        assert np.array_equal(canonical, cap_mask)
        assert transform == "R0"
        assert inverse == "R0"


# ============================================================================
# Put Mask Canonicalization Tests
# ============================================================================


class TestCanonicalizePutMask:
    """Test public API for put mask canonicalization."""

    def test_canonicalize_with_identity_transform(self, small_board):
        """Test that identity transform preserves put mask."""
        put_mask = small_board.get_placement_moves()

        # Canonicalize with identity
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_put_mask(put_mask, "R0")

        assert np.array_equal(canonical, put_mask), "R0 should preserve mask"
        assert transform == "R0"
        assert inverse == "R0"

    def test_canonicalize_with_rotation(self, small_board):
        """Test canonicalizing with rotation transforms."""
        put_mask = small_board.get_placement_moves()

        # Canonicalize with R60
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_put_mask(put_mask, "R60")

        assert transform == "R60"
        assert inverse == "R300"
        assert np.sum(canonical) == np.sum(put_mask), "Should preserve placement count"

    def test_canonicalize_with_mirror(self, small_board):
        """Test canonicalizing with mirror transforms."""
        put_mask = small_board.get_placement_moves()

        # Canonicalize with MR120
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_put_mask(put_mask, "MR120")

        assert transform == "MR120"
        assert inverse == "R240M"
        assert np.sum(canonical) == np.sum(put_mask)

    def test_canonicalize_with_translation(self, small_board):
        """Test canonicalizing with translation transforms."""
        put_mask = small_board.get_placement_moves()

        # Canonicalize with translation
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_put_mask(put_mask, "T1,0")

        assert transform == "T1,0"
        assert inverse == "T-1,0"

    def test_canonicalize_with_combined_transform(self, small_board):
        """Test canonicalizing with combined translation + rotation/mirror."""
        put_mask = small_board.get_placement_moves()

        # Canonicalize with combined transform
        canonical, transform, inverse = small_board.canonicalizer.canonicalize_put_mask(
            put_mask, "T1,0_R60"
        )

        assert transform == "T1,0_R60"
        assert inverse == "R300_T-1,0"

    def test_decanonicalize_put_mask(self, small_board):
        """Test decanonicalization of put mask."""
        put_mask = small_board.get_placement_moves()

        # Canonicalize
        canonical, _, inverse = small_board.canonicalizer.canonicalize_put_mask(put_mask, "R120")

        # Decanonicalize
        recovered = small_board.canonicalizer.decanonicalize_put_mask(canonical, inverse)

        assert np.array_equal(recovered, put_mask), "Decanonicalize should recover original"

    def test_round_trip_with_various_transforms(self, small_board):
        """Test canonicalize + decanonicalize round trip with various transforms."""
        put_mask = small_board.get_placement_moves()

        # Test only rotation/mirror transforms (no translation) as they're always valid
        test_transforms = [
            "R0", "R60", "R120", "R180", "R240", "R300",
            "MR0", "MR60", "R60M", "R120M"
        ]

        for transform_name in test_transforms:
            canonical, _, inverse = small_board.canonicalizer.canonicalize_put_mask(
                put_mask, transform_name
            )
            recovered = small_board.canonicalizer.decanonicalize_put_mask(canonical, inverse)

            assert np.array_equal(recovered, put_mask), (
                f"Round trip failed for {transform_name}"
            )

    def test_preserves_marble_type_separation(self, small_board):
        """Test that canonicalization preserves marble type separation."""
        put_mask = small_board.get_placement_moves()

        # Count for each marble type
        original_counts = [np.sum(put_mask[i]) for i in range(3)]

        # Canonicalize
        canonical, _, _ = small_board.canonicalizer.canonicalize_put_mask(put_mask, "R120")
        canonical_counts = [np.sum(canonical[i]) for i in range(3)]

        assert original_counts == canonical_counts, "Should preserve marble type counts"

    def test_preserves_no_removal_index(self, small_board):
        """Test that no-removal index is preserved correctly."""
        put_mask = small_board.get_placement_moves()

        no_removal_idx = small_board.config.width ** 2
        original_no_removal = np.sum(put_mask[:, :, no_removal_idx])

        canonical, _, _ = small_board.canonicalizer.canonicalize_put_mask(put_mask, "R60")
        canonical_no_removal = np.sum(canonical[:, :, no_removal_idx])

        assert canonical_no_removal == original_no_removal, (
            "No-removal index count should be preserved"
        )

    @pytest.mark.parametrize("board_fixture", ["small_board", "medium_board", "large_board"])
    def test_works_for_all_board_sizes(self, request, board_fixture):
        """Test that mask canonicalization works for all board sizes."""
        board = request.getfixturevalue(board_fixture)

        put_mask = board.get_placement_moves()

        canonical, transform, inverse = board.canonicalizer.canonicalize_put_mask(
            put_mask, "R0"
        )

        assert np.array_equal(canonical, put_mask)
        assert transform == "R0"
        assert inverse == "R0"


# ============================================================================
# Integration Tests
# ============================================================================


class TestMaskCanonicalizationIntegration:
    """Integration tests for mask canonicalization with state canonicalization."""

    def test_mask_and_state_use_same_transform(self, small_board):
        """Test that masks and states can be canonicalized with the same transform."""
        # Place marbles
        small_board.state[1, *small_board.str_to_index("D4")] = 1
        small_board.state[2, *small_board.str_to_index("E4")] = 1

        # Get state and masks
        original_state = small_board.state.copy()
        cap_mask = small_board.get_capture_moves()
        put_mask = small_board.get_placement_moves()

        # Canonicalize state
        canonical_state, transform, inverse = small_board.canonicalize_state()

        # Canonicalize masks with the same transform
        canonical_cap, _, _ = small_board.canonicalizer.canonicalize_capture_mask(cap_mask, transform)
        canonical_put, _, _ = small_board.canonicalizer.canonicalize_put_mask(put_mask, transform)

        # Create board with canonical state
        canonical_board = ZertzBoard(clone=small_board)
        canonical_board.state = canonical_state

        # Get masks from canonical board
        expected_cap = canonical_board.get_capture_moves()
        expected_put = canonical_board.get_placement_moves()

        # Canonicalized masks should match masks from canonical state
        assert np.array_equal(canonical_cap, expected_cap), (
            "Canonicalized capture mask should match canonical board's capture mask"
        )
        assert np.array_equal(canonical_put, expected_put), (
            "Canonicalized put mask should match canonical board's put mask"
        )

    def test_decanonicalize_recovers_original_masks(self, small_board):
        """Test that decanonicalization recovers original masks."""
        # Setup
        small_board.state[1, *small_board.str_to_index("D4")] = 1
        small_board.state[2, *small_board.str_to_index("B3")] = 1

        cap_mask = small_board.get_capture_moves()
        put_mask = small_board.get_placement_moves()

        # Canonicalize state
        _, transform, inverse = small_board.canonicalize_state()

        # Canonicalize masks
        canonical_cap, _, _ = small_board.canonicalizer.canonicalize_capture_mask(cap_mask, transform)
        canonical_put, _, _ = small_board.canonicalizer.canonicalize_put_mask(put_mask, transform)

        # Decanonicalize masks
        recovered_cap = small_board.canonicalizer.decanonicalize_capture_mask(canonical_cap, inverse)
        recovered_put = small_board.canonicalizer.decanonicalize_put_mask(canonical_put, inverse)

        # Should recover originals
        assert np.array_equal(recovered_cap, cap_mask), "Should recover original capture mask"
        assert np.array_equal(recovered_put, put_mask), "Should recover original put mask"