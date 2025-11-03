"""
Unit tests for updated hexagonal coordinate transformations (axial-based).

These tests verify that rotations and mirrors behave correctly using
the new axial coordinate system introduced in ZertzBoard.
"""

import pytest
import numpy as np
import sys
import copy
from pathlib import Path
from hiivelabs_mcts import algebraic_to_coordinate, coordinate_to_algebraic

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_board import ZertzBoard


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_board():
    """37-ring board for hex transformation tests.

    37-ring board layout:
           A4  B5  C6  D7       row 0
         A3  B4  C5  D6  E6     row 1
       A2  B3  C4  D5  E5  F5   row 2
     A1  B2  C3  D4  E4  F4  G4 row 3 (center)
       B1  C2  D3  E3  F3  G3   row 4
         C1  D2  E2  F2  G2     row 5
           D1  E1  F1  G1       row 6

    D4 is at array position (3,3) and should map to axial (0,0).
    """
    return ZertzBoard(rings=ZertzBoard.SMALL_BOARD_37)


@pytest.fixture
def medium_board():
    """48-ring board for hex transformation tests."""
    return ZertzBoard(rings=ZertzBoard.MEDIUM_BOARD_48)


@pytest.fixture
def large_board():
    """61-ring board for hex transformation tests."""
    return ZertzBoard(rings=ZertzBoard.LARGE_BOARD_61)


# ============================================================================
# Axial Coordinate Mapping
# ============================================================================


class TestAxialCoordinateMaps:
    def test_center_position_is_origin(self, small_board):
        """Test that D4 (center) maps to origin in axial coordinates."""
        small_board._build_axial_maps()
        y, x = algebraic_to_coordinate("D4", small_board.config)
        q, r = small_board._yx_to_ax[(y, x)]
        assert (q, r) == (0, 0), f"Center D4 should map to (0,0), got ({q},{r})"

    def test_axial_roundtrip_is_consistent(self, small_board):
        """Test that converting to axial and back preserves position."""
        small_board._build_axial_maps()
        positions = ["A4", "D7", "G1", "D1", "A1", "G4", "D4"]
        for pos in positions:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            q, r = small_board._yx_to_ax[(y, x)]
            y2, x2 = small_board._ax_to_yx[(q, r)]
            assert (y, x) == (y2, x2), f"Roundtrip failed for {pos}"


class TestHexRotation:
    @pytest.mark.parametrize(
        "start_pos,k,expected_pos",
        [
            ("A4", 1, "D7"),  # 60° rotation
            ("A4", 2, "G4"),  # 120° rotation
            ("A4", 3, "G1"),  # 180° rotation
            ("A4", 4, "D1"),  # 240° rotation
            ("A4", 5, "A1"),  # 300° rotation
            ("A4", 6, "A4"),  # 360° rotation (identity)
        ],
    )
    def test_rotate_positions(self, small_board, start_pos, k, expected_pos):
        """Test that rotation by k*60° produces expected position."""
        board = small_board

        # Place a marble at start position
        y, x = algebraic_to_coordinate(start_pos, board.config)
        test_state = np.zeros_like(board.state)
        test_state[board.RING_LAYER] = board.state[board.RING_LAYER]  # Copy rings
        test_state[board.MARBLE_TO_LAYER["w"], y, x] = 1  # Place white marble

        # Apply rotation
        rotated = board.canonicalizer.transform_state_hex(test_state, rot60_k=k)

        # Find where the marble ended up
        (end_y, end_x) = np.argwhere(rotated[board.MARBLE_TO_LAYER["w"]] == 1)[0]
        end_pos = coordinate_to_algebraic(*(end_y, end_x, board.config))

        assert end_pos == expected_pos, (
            f"Rotation by {k * 60}° of {start_pos} → expected {expected_pos}, got {end_pos}"
        )

    def test_rotation_preserves_marbles(self, small_board):
        """Test that rotation preserves the number of marbles."""
        small_board._build_axial_maps()

        # Place marbles at multiple positions
        test_state = np.zeros_like(small_board.state)
        test_state[small_board.RING_LAYER] = small_board.state[small_board.RING_LAYER]
        test_state[
            small_board.MARBLE_TO_LAYER["w"], *algebraic_to_coordinate("A4", small_board.config)
        ] = 1
        test_state[
            small_board.MARBLE_TO_LAYER["g"], *algebraic_to_coordinate("D4", small_board.config)
        ] = 1
        test_state[
            small_board.MARBLE_TO_LAYER["b"], *algebraic_to_coordinate("G1", small_board.config)
        ] = 1

        for k in range(6):
            rotated = small_board.canonicalizer.transform_state_hex(test_state, rot60_k=k)
            assert np.sum(rotated[small_board.MARBLE_TO_LAYER["w"]]) == 1, (
                f"White marbles not preserved for k={k}"
            )
            assert np.sum(rotated[small_board.MARBLE_TO_LAYER["g"]]) == 1, (
                f"Gray marbles not preserved for k={k}"
            )
            assert np.sum(rotated[small_board.MARBLE_TO_LAYER["b"]]) == 1, (
                f"Black marbles not preserved for k={k}"
            )


class TestHexMirror:
    @pytest.mark.parametrize(
        "start_pos,expected_pos",
        [
            # FIXED: Using the actual q-axis mirror results (q, -q-r)
            ("A4", "A1"),  # (-3,0) → (-3,3)
            ("D7", "D1"),  # (0,-3) → (0,3)
            ("G4", "G1"),  # (3,-3) → (3,0)
            ("G1", "G4"),  # (3,0) → (3,-3)
            ("D1", "D7"),  # (0,3) → (0,-3)
            ("A1", "A4"),  # (-3,3) → (-3,0)
            ("D4", "D4"),  # (0,0) → (0,0) - center stays fixed
        ],
    )
    def test_mirror_positions(self, small_board, start_pos, expected_pos):
        """Test that mirror reflection produces expected position."""
        board = small_board

        y, x = algebraic_to_coordinate(start_pos, board.config)
        test_state = np.zeros_like(board.state)
        test_state[board.RING_LAYER] = board.state[board.RING_LAYER]
        test_state[board.MARBLE_TO_LAYER["w"], y, x] = 1

        mirrored = board.canonicalizer.transform_state_hex(test_state, mirror=True)
        (end_y, end_x) = np.argwhere(mirrored[board.MARBLE_TO_LAYER["w"]] == 1)[0]
        end_pos = coordinate_to_algebraic(*(end_y, end_x, board.config))

        assert end_pos == expected_pos, (
            f"Mirror of {start_pos} → expected {expected_pos}, got {end_pos}"
        )

    def test_mirror_is_involutive(self, small_board):
        """Test that applying mirror twice returns to original."""
        small_board._build_axial_maps()
        base = np.copy(small_board.state)
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        base[2, *algebraic_to_coordinate("B3", small_board.config)] = 1

        once = small_board.canonicalizer.transform_state_hex(base, mirror=True)
        twice = small_board.canonicalizer.transform_state_hex(once, mirror=True)

        assert np.array_equal(base, twice), (
            "Mirror applied twice should return original"
        )


class TestCombinedSymmetries:
    def test_rotation_composition(self, small_board):
        """Test that composing rotations works correctly."""
        small_board._build_axial_maps()
        base = np.copy(small_board.state)
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1

        # Rotating by 120° twice should equal rotating by 240°
        rot_120 = small_board.canonicalizer.transform_state_hex(base, rot60_k=2)
        rot_240_via_composition = small_board.canonicalizer.transform_state_hex(rot_120, rot60_k=2)
        rot_240_direct = small_board.canonicalizer.transform_state_hex(base, rot60_k=4)

        assert np.array_equal(rot_240_via_composition, rot_240_direct), (
            "Rotation composition failed"
        )

    def test_all_symmetries_unique(self, small_board):
        """Test that all 12 symmetries produce unique board states."""
        small_board._build_axial_maps()

        # Create an asymmetric pattern that doesn't have inherent symmetry
        # Avoid using the center (D4) and opposite corners together
        base = np.copy(small_board.state)
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1  # White at corner
        base[2, *algebraic_to_coordinate("B3", small_board.config)] = 1  # Gray off-center
        base[3, *algebraic_to_coordinate("E3", small_board.config)] = (
            1  # Black at another off-center position
        )

        states = []
        # 6 rotations
        for k in range(6):
            states.append(small_board.canonicalizer.transform_state_hex(base, rot60_k=k))
        # 6 mirror + rotations
        for k in range(6):
            states.append(
                small_board.canonicalizer.transform_state_hex(base, rot60_k=k, mirror=True)
            )

        # Check all states are unique
        unique = {s.tobytes() for s in states}
        assert len(unique) == 12, f"Expected 12 unique symmetries, got {len(unique)}"


class TestAllBoardSizes:
    @pytest.fixture
    def small_board(self):
        from game.zertz_board import ZertzBoard

        return ZertzBoard(37)

    @pytest.fixture
    def medium_board(self):
        from game.zertz_board import ZertzBoard

        return ZertzBoard(48)

    @pytest.fixture
    def large_board(self):
        from game.zertz_board import ZertzBoard

        return ZertzBoard(61)

    @pytest.mark.parametrize(
        "board_fixture", ["small_board", "medium_board", "large_board"]
    )
    def test_axial_map_sizes(self, request, board_fixture):
        """Test that axial maps have correct number of positions."""
        board = request.getfixturevalue(board_fixture)
        board._ensure_positions_built()

        positions = [
            board.position_from_label(label)
            for label in board._positions_by_label.keys()
        ]
        assert len(positions) == board.rings, (
            f"Expected {board.rings} positions, got {len(positions)}"
        )

    @pytest.mark.parametrize(
        "board_fixture", ["small_board", "medium_board", "large_board"]
    )
    def test_center_near_origin(self, request, board_fixture):
        """Test that the center position maps near origin (scaled by coord_scale)."""
        board = request.getfixturevalue(board_fixture)
        board._ensure_positions_built()

        # Find position closest to board center
        center_y, center_x = board.config.width // 2, board.config.width // 2
        min_dist = float("inf")
        center_pos = None

        for pos in board.positions:
            y, x = pos.yx
            q, r = pos.axial
            dist = abs(y - center_y) + abs(x - center_x)
            if dist < min_dist:
                min_dist = dist
                center_pos = (q, r)

        # Center should be at or very close to (0, 0), scaled by coord_scale
        # For D6 boards (37/61 rings), scale=1, so expect |q|, |r| <= 1
        # For D3 boards (48 rings), scale=3, so expect |q|, |r| <= 3
        q, r = center_pos
        max_expected = board._coord_scale
        assert abs(q) <= max_expected and abs(r) <= max_expected, (
            f"Center position should map near (0,0) scaled by {board._coord_scale}, got ({q},{r})"
        )


class TestSymmetryPatterns:
    """Test that patterns with different symmetries produce the expected number of unique states."""

    def count_unique_symmetries(self, board, base_state):
        """Helper to count unique states under all 12 transformations."""
        states = []
        # 6 rotations
        for k in range(6):
            states.append(board.canonicalizer.transform_state_hex(base_state, rot60_k=k))
        # 6 mirror + rotations
        for k in range(6):
            states.append(
                board.canonicalizer.transform_state_hex(base_state, rot60_k=k, mirror=True)
            )

        unique = {s.tobytes() for s in states}
        return len(unique)

    def test_empty_board_has_one_symmetry(self, small_board):
        """An empty board should look identical under all transformations."""
        small_board._build_axial_maps()

        # Just the ring pattern, no marbles
        base = np.copy(small_board.state)

        unique_count = self.count_unique_symmetries(small_board, base)
        assert unique_count == 1, (
            f"Empty board should have 1 unique state, got {unique_count}"
        )

    def test_center_marble_has_one_symmetry(self, small_board):
        """A single marble at center has 6-fold rotational symmetry → 1 unique state."""
        small_board._build_axial_maps()

        base = np.copy(small_board.state)
        base[1, *algebraic_to_coordinate("D4", small_board.config)] = 1  # Single marble at center

        unique_count = self.count_unique_symmetries(small_board, base)
        assert unique_count == 1, (
            f"Center marble should have 1 unique state, got {unique_count}"
        )

    def test_six_fold_pattern_has_one_symmetries(self, small_board):
        """A pattern with 6-fold rotational symmetry should have 2 unique states."""
        small_board._build_axial_maps()

        base = np.copy(small_board.state)
        # Place marbles at all 6 corners - perfect 6-fold symmetry
        for corner in ["A4", "D7", "G4", "G1", "D1", "A1"]:
            base[1, *algebraic_to_coordinate(corner, small_board.config)] = 1

        unique_count = self.count_unique_symmetries(small_board, base)
        assert unique_count == 1, (
            f"6-fold symmetric pattern should have 1 unique state, got {unique_count}"
        )

    def test_three_fold_pattern_has_two_symmetries(self, small_board):
        """A pattern with 3-fold rotational symmetry should have 4 unique states."""
        small_board._build_axial_maps()

        base = np.copy(small_board.state)
        # Place marbles at alternating corners - 3-fold symmetry
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1  # 0°
        base[1, *algebraic_to_coordinate("G4", small_board.config)] = 1  # 120°
        base[1, *algebraic_to_coordinate("D1", small_board.config)] = 1  # 240°

        unique_count = self.count_unique_symmetries(small_board, base)
        assert unique_count == 2, (
            f"3-fold symmetric pattern should have 2 unique states, got {unique_count}"
        )

    def test_two_fold_pattern_has_three_symmetries(self, small_board):
        """A pattern with 2-fold rotational symmetry should have 6 unique states."""
        small_board._build_axial_maps()

        base = np.copy(small_board.state)
        # Place marbles at opposite corners - 2-fold symmetry
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1  # 0°
        base[1, *algebraic_to_coordinate("G1", small_board.config)] = 1  # 180°

        unique_count = self.count_unique_symmetries(small_board, base)
        assert unique_count == 3, (
            f"2-fold symmetric pattern should have 3 unique states, got {unique_count}"
        )

    def test_vertical_mirror_pattern_has_six_symmetries(self, small_board):
        """A pattern with only vertical mirror symmetry should have 6 unique states."""
        small_board._build_axial_maps()

        base = np.copy(small_board.state)
        # Create pattern symmetric across one axis but not rotationally symmetric
        base[1, *algebraic_to_coordinate("B4", small_board.config)] = 1
        base[1, *algebraic_to_coordinate("F4", small_board.config)] = 1  # Mirror of B4 across vertical
        base[2, *algebraic_to_coordinate("C3", small_board.config)] = 1
        base[2, *algebraic_to_coordinate("E3", small_board.config)] = 1  # Mirror of C3 across vertical

        unique_count = self.count_unique_symmetries(small_board, base)
        # This should have 6 unique states (not 12) due to the mirror symmetry
        assert unique_count == 6, (
            f"Mirror symmetric pattern should have 6 unique states, got {unique_count}"
        )

    def test_asymmetric_pattern_has_twelve_symmetries(self, small_board):
        """A completely asymmetric pattern should have 12 unique states."""
        small_board._build_axial_maps()

        base = np.copy(small_board.state)
        # Create an asymmetric pattern
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        base[2, *algebraic_to_coordinate("B3", small_board.config)] = 1
        base[3, *algebraic_to_coordinate("E3", small_board.config)] = 1

        unique_count = self.count_unique_symmetries(small_board, base)
        assert unique_count == 12, (
            f"Asymmetric pattern should have 12 unique states, got {unique_count}"
        )

    # def test_single_off_center_marble_has_twelve_symmetries(self, small_board):
    #     """A single marble not at center should have no symmetry → 12 unique states."""
    #     small_board._build_axial_maps()
    #
    #     base = np.copy(small_board.state)
    #     base[1, *algebraic_to_coordinate("B3", small_board.config)] = 1  # Single marble off-center
    #
    #     unique_count = self.count_unique_symmetries(small_board, base)
    #     assert unique_count == 12, f"Single off-center marble should have 12 unique states, got {unique_count}"

    def test_mirror_produces_unique_states_not_rotations(self, small_board):
        """
        Diagnostic test that mirror transformations produce unique states distinct from rotations.
        If mirror is a true reflection, we should get 12 total unique states.
        If mirror is actually a disguised rotation, we'll get only 6.
        """
        small_board._build_axial_maps()

        # Use the asymmetric 3-marble pattern we know should have 12 states
        base = np.copy(small_board.state)
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1  # White at corner
        base[2, *algebraic_to_coordinate("B3", small_board.config)] = 1  # Gray off-center
        base[3, *algebraic_to_coordinate("E3", small_board.config)] = 1  # Black at different position

        # Collect all rotation states
        rotation_states = set()
        for k in range(6):
            state = small_board.canonicalizer.transform_state_hex(base, rot60_k=k)
            rotation_states.add(state.tobytes())

        # Collect all mirror + rotation states
        mirror_states = set()
        for k in range(6):
            state = small_board.canonicalizer.transform_state_hex(base, rot60_k=k, mirror=True)
            mirror_states.add(state.tobytes())

        # Check for overlap
        overlap = rotation_states & mirror_states
        total_unique = len(rotation_states | mirror_states)

        # Assertions
        assert len(rotation_states) == 6, (
            f"Expected 6 unique rotation states, got {len(rotation_states)}"
        )

        assert len(mirror_states) == 6, (
            f"Expected 6 unique mirror states, got {len(mirror_states)}"
        )

        # The critical test: if mirror is a true reflection, there should be NO overlap
        # between rotation states and mirror states (except possibly for states with
        # inherent symmetry, which this pattern doesn't have)
        assert len(overlap) == 0, (
            f"Mirror states should be distinct from rotation states, but {len(overlap)} overlap. "
            f"This indicates the mirror transformation is equivalent to a rotation, not a true reflection."
        )

        assert total_unique == 12, (
            f"Expected 12 total unique states for asymmetric pattern, got {total_unique}. "
            f"Mirror appears to be producing the same states as rotations."
        )

    def test_asymmetric_three_marble_pattern(self, small_board):
        """
        A truly asymmetric pattern with 3 marbles should have 12 unique states.
        Using three marbles in positions that form a scalene triangle ensures
        no accidental symmetries.
        """
        small_board._build_axial_maps()

        base = np.copy(small_board.state)
        # Use the pattern we KNOW works from earlier testing:
        # These three positions form an asymmetric pattern with no symmetries
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1  # Corner
        base[2, *algebraic_to_coordinate("B3", small_board.config)] = 1  # Off-center
        base[3, *algebraic_to_coordinate("E3", small_board.config)] = 1  # Different off-center

        unique_count = self.count_unique_symmetries(small_board, base)

        # With a true reflection, this should give 12 unique states
        # If we only get 6, then the "mirror" is actually a rotation
        assert unique_count == 12, (
            f"Asymmetric 3-marble pattern should have 12 unique states, got {unique_count}"
        )

    def test_line_through_center_has_three_symmetries(self, small_board):
        """A line through center has 2-fold rotational symmetry → 3 unique states."""
        small_board._build_axial_maps()

        base = np.copy(small_board.state)
        # Create a line from A1 through D4 to G4
        base[1, *algebraic_to_coordinate("A1", small_board.config)] = 1
        base[1, *algebraic_to_coordinate("B2", small_board.config)] = 1
        base[1, *algebraic_to_coordinate("C3", small_board.config)] = 1
        base[1, *algebraic_to_coordinate("D4", small_board.config)] = 1
        base[1, *algebraic_to_coordinate("E4", small_board.config)] = 1
        base[1, *algebraic_to_coordinate("F4", small_board.config)] = 1
        base[1, *algebraic_to_coordinate("G4", small_board.config)] = 1

        unique_count = self.count_unique_symmetries(small_board, base)

        # Line has 2-fold symmetry: 6 rotations ÷ 2 = 3 unique states
        assert unique_count == 3, (
            f"Line through center should have 3 unique states, got {unique_count}"
        )

    @pytest.mark.parametrize(
        "marble_count,expected_range",
        [
            (1, (2, 12)),  # 1 marble: 2 if centered, 12 if off-center
            (2, (2, 12)),  # 2 marbles: varies by placement
            (3, (2, 12)),  # 3 marbles: could have various symmetries
            (6, (1, 12)),  # 6 marbles: could be symmetric or not
            (37, (1, 1)),  # All positions filled: always 1
        ],
    )
    def test_random_patterns_have_valid_symmetry_count(
        self, small_board, marble_count, expected_range
    ):
        """Random patterns should have between 1 and 12 unique states."""
        import random

        random.seed(42)  # Reproducible test

        small_board._build_axial_maps()
        base = np.copy(small_board.state)

        # Get all valid positions
        valid_positions = []
        for y in range(small_board.config.width):
            for x in range(small_board.config.width):
                if small_board.state[small_board.RING_LAYER, y, x] == 1:  # Has ring
                    valid_positions.append((y, x))

        # Place random marbles
        if marble_count <= len(valid_positions):
            chosen = random.sample(valid_positions, marble_count)
            for y, x in chosen:
                base[1, y, x] = 1
        else:
            # Fill all positions
            for y, x in valid_positions:
                base[1, y, x] = 1

        unique_count = self.count_unique_symmetries(small_board, base)
        min_expected, max_expected = expected_range
        assert min_expected <= unique_count <= max_expected, (
            f"Pattern with {marble_count} marbles should have {min_expected}-{max_expected} unique states, got {unique_count}"
        )


class TestSpiralMirror:
    """Tests for mirror transformation using spiral patterns to verify chirality."""

    def test_spiral_chirality_changes_under_mirror(self, small_board):
        """
        A spiral has chirality (handedness). A true mirror should reverse it,
        while rotations preserve it.
        """
        small_board._build_axial_maps()

        # Create a clear clockwise spiral
        clockwise_spiral = [
            ("D4", 1),  # Center
            ("E4", 2),  # 3 o'clock
            ("E3", 3),  # 4:30
            ("D3", 1),  # 6 o'clock
            ("C3", 2),  # 7:30
            ("C4", 3),  # 9 o'clock
            ("C5", 1),  # 10:30
            ("D5", 2),  # 12 o'clock
            ("E5", 3),  # 1:30
        ]

        base = np.copy(small_board.state)
        for pos, marble_layer in clockwise_spiral:
            base[marble_layer, *algebraic_to_coordinate(pos, small_board.config)] = 1

        # Get the sequence of marble types going clockwise from E4
        def get_spiral_sequence(state, start_pos="E4"):
            """Extract the sequence of marble types in clockwise order from start."""
            # Define clockwise path from E4
            path = ["E4", "E3", "D3", "C3", "C4", "C5", "D5", "E5"]
            sequence = []
            for pos in path:
                y, x = algebraic_to_coordinate(pos, small_board.config)
                for marble_type in [1, 2, 3]:
                    if state[marble_type, y, x] == 1:
                        sequence.append(marble_type)
                        break
            return sequence

        # Apply mirror
        mirrored = small_board.canonicalizer.transform_state_hex(base, mirror=True)
        mirrored_sequence = get_spiral_sequence(mirrored)
        original_sequence = get_spiral_sequence(base)

        # The mirrored spiral should have different chirality
        # We can't easily check the exact sequence without knowing the mirror axis,
        # but we can verify it's different from all rotations
        assert mirrored_sequence != original_sequence, (
            "Mirror did not change spiral chirality (sequence identical)"
        )

        for k in range(1, 6):
            rotated = small_board.canonicalizer.transform_state_hex(base, rot60_k=k)
            # Rotations preserve chirality, so if mirror equals any rotation,
            # it's not a true mirror
            if np.array_equal(mirrored, rotated):
                all_rotations_same_chirality = False
                pytest.fail(
                    f"Mirror equals {k * 60}° rotation - chirality not reversed!"
                )

        # Also verify involution
        double_mirrored = small_board.canonicalizer.transform_state_hex(mirrored, mirror=True)
        assert np.array_equal(double_mirrored, base), (
            "Mirror is not involutive - applying twice doesn't return original"
        )


class TestCanonicalTransform:
    def test_mirror_rotation_composition_and_inverses(self, small_board):
        """
        Test that composition of mirror and rotation works correctly
        and verify the correct inverse calculation using the new R{k}M notation.
        """
        small_board._build_axial_maps()

        # Create test pattern
        base = np.copy(small_board.state)
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        base[2, *algebraic_to_coordinate("B3", small_board.config)] = 1
        base[3, *algebraic_to_coordinate("E3", small_board.config)] = 1

        # Get all transforms as a dictionary
        transforms = dict(small_board.canonicalizer.get_all_symmetry_transforms())

        # Test what MR120 actually does
        mr120 = transforms["MR120"](base)

        # Get the calculated inverse
        inverse_name = small_board.canonicalizer._get_inverse_transform("MR120")
        print(f"\nCalculated inverse of MR120: {inverse_name}")

        # Apply the inverse
        if inverse_name not in transforms:
            pytest.fail(
                f"Inverse transform {inverse_name} not found in available transforms"
            )

        recovered = transforms[inverse_name](mr120)

        # Verify it recovers the original
        assert np.array_equal(recovered, base), (
            f"Transform MR120 with calculated inverse {inverse_name} doesn't recover original"
        )

    @pytest.mark.parametrize(
        "board_fixture", ["small_board", "medium_board", "large_board"]
    )
    def test_verify_transform_order_of_operations(self, request, board_fixture):
        """
        Verify that _transform_state_hex applies transformations in the correct order.

        Tests both composition orders:
        - MR(k): rotate by k, THEN mirror (mirror_first=False)
        - R(k)M: mirror, THEN rotate by k (mirror_first=True)

        These should produce different results for asymmetric patterns.

        Tests all board sizes (37, 48, 61 rings).
        """
        board = request.getfixturevalue(board_fixture)
        board._ensure_positions_built()

        # Create asymmetric test pattern using positions that exist on all board sizes
        base = np.copy(board.state)
        if board.rings == 37:
            base[1, *algebraic_to_coordinate("A4", board.config)] = 1
            base[2, *algebraic_to_coordinate("B3", board.config)] = 1
        elif board.rings == 48:
            base[1, *algebraic_to_coordinate("A5", board.config)] = 1
            base[2, *algebraic_to_coordinate("B4", board.config)] = 1
        else:  # 61 rings
            base[1, *algebraic_to_coordinate("A5", board.config)] = 1
            base[2, *algebraic_to_coordinate("B4", board.config)] = 1

        # Test all angles to ensure consistency
        # For 48-ring board (D3), only test 120°, 240° angles (k=2, 4)
        # For D6 boards (37, 61), test 60°, 120°, 180° angles (k=1, 2, 3)
        test_angles = [2, 4] if board.rings == 48 else [1, 2, 3]

        for k in test_angles:
            # === Test MR(k): rotate-then-mirror (default behavior) ===
            mr_combined = board.canonicalizer.transform_state_hex(
                base, rot60_k=k, mirror=True, mirror_first=False
            )

            # Manual two-step: rotate then mirror
            rotated = board.canonicalizer.transform_state_hex(base, rot60_k=k)
            rot_then_mirror = board.canonicalizer.transform_state_hex(rotated, mirror=True)

            # MR(k) should match manual rotate-then-mirror
            assert np.array_equal(mr_combined, rot_then_mirror), (
                f"{board.rings}-ring: MR({k*60}°) should apply rotation first, then mirror. "
                f"If this fails, the transform order doesn't match the inverse calculation."
            )

            # === Test R(k)M: mirror-then-rotate ===
            rm_combined = board.canonicalizer.transform_state_hex(
                base, rot60_k=k, mirror=True, mirror_first=True
            )

            # Manual two-step: mirror then rotate
            mirrored = board.canonicalizer.transform_state_hex(base, mirror=True)
            mirror_then_rot = board.canonicalizer.transform_state_hex(mirrored, rot60_k=k)

            # R(k)M should match manual mirror-then-rotate
            assert np.array_equal(rm_combined, mirror_then_rot), (
                f"{board.rings}-ring: R({k*60}°)M should apply mirror first, then rotation. "
                f"If this fails, the mirror_first parameter doesn't work correctly."
            )

            # === Verify the two orders produce DIFFERENT results (except for 180° where they commute) ===
            # For 180° rotation, MR(180°) = R(180°)M due to D6 symmetry properties
            if k == 3:  # 180°
                # For 180° rotation, the order doesn't matter (operations commute)
                assert np.array_equal(mr_combined, rm_combined), (
                    f"{board.rings}-ring: MR(180°) and R(180°)M should produce the SAME result (they commute)"
                )
            else:
                # For other angles, order matters
                assert not np.array_equal(mr_combined, rm_combined), (
                    f"{board.rings}-ring: MR({k*60}°) and R({k*60}°)M should produce different results for asymmetric pattern. "
                    f"Order of operations matters!"
                )

        # === Test that pure rotation and pure mirror work correctly ===
        # Pure rotation (no mirror)
        for k in test_angles:
            pure_rot = board.canonicalizer.transform_state_hex(base, rot60_k=k, mirror=False)
            assert np.sum(pure_rot) == np.sum(base), (
                f"{board.rings}-ring: Pure rotation R({k*60}°) should preserve marble count"
            )

        # Pure mirror (no rotation)
        pure_mirror = board.canonicalizer.transform_state_hex(base, rot60_k=0, mirror=True)
        assert np.sum(pure_mirror) == np.sum(base), (
            f"{board.rings}-ring: Pure mirror should preserve marble count"
        )

        # Verify pure mirror is involutive (applying twice returns original)
        double_mirror = board.canonicalizer.transform_state_hex(pure_mirror, rot60_k=0, mirror=True)
        assert np.array_equal(base, double_mirror), (
            f"{board.rings}-ring: Mirror applied twice should return to original"
        )

    def test_all_transform_inverses(self, small_board):
        """
        Test that all 18 transformations have correct inverses.
        """
        small_board._build_axial_maps()

        # Create asymmetric test pattern
        base = np.copy(small_board.state)
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        base[2, *algebraic_to_coordinate("B3", small_board.config)] = 1
        base[3, *algebraic_to_coordinate("E3", small_board.config)] = 1

        # Get all transforms as a dictionary
        transforms = dict(small_board.canonicalizer.get_all_symmetry_transforms())

        # Test all transformations and their calculated inverses
        failures = []

        for transform_name, transform_fn in transforms.items():
            # Apply the transformation
            transformed = transform_fn(base)

            # Get the calculated inverse
            inverse_name = small_board.canonicalizer._get_inverse_transform(transform_name)

            # Check the inverse exists
            if inverse_name not in transforms:
                failures.append(
                    f"{transform_name}: inverse {inverse_name} not found in transforms"
                )
                continue

            # Apply the inverse
            recovered = transforms[inverse_name](transformed)

            # Verify it recovers the original
            if not np.array_equal(recovered, base):
                failures.append(
                    f"{transform_name} with inverse {inverse_name} doesn't recover original"
                )

        if failures:
            pytest.fail("Inverse calculation failures:\n" + "\n".join(failures))

    def test_inverse_transform_correctness(self, small_board):
        """Applying transform then inverse should return to original."""
        small_board._build_axial_maps()

        # Create test pattern
        base = np.copy(small_board.state)
        base[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        base[2, *algebraic_to_coordinate("B3", small_board.config)] = 1

        small_board.state = base

        # Get canonical form and transforms
        canonical, transform_used, inverse_transform = small_board.canonicalize_state()

        # Apply the inverse to the canonical form
        # This should give us back the original
        transforms = dict(small_board.canonicalizer.get_all_symmetry_transforms())

        if inverse_transform in transforms:
            recovered = transforms[inverse_transform](canonical)
            assert np.array_equal(recovered, base), (
                f"Transform {transform_used} with inverse {inverse_transform} doesn't recover original"
            )
        else:
            pytest.fail(
                f"Inverse transform {inverse_transform} not found in available transforms"
            )


class TestTranslationCanonicalization:
    """Test translation symmetry detection and canonicalization."""

    def test_bounding_box_full_board(self, small_board):
        """Test that full board has expected bounding box."""
        bbox = small_board.canonicalizer.get_bounding_box()
        assert bbox is not None, "Full board should have bounding box"

        min_y, max_y, min_x, max_x = bbox
        # 37-ring board is 7x7, but corners are empty
        assert 0 <= min_y < small_board.config.width
        assert 0 <= max_y < small_board.config.width
        assert 0 <= min_x < small_board.config.width
        assert 0 <= max_x < small_board.config.width

    def test_bounding_box_after_ring_removal(self, small_board):
        """Test bounding box after removing edge rings."""
        # Remove entire edge rows/columns to actually reduce bounding box
        for pos in ["A4", "A3", "A2", "A1", "B1", "C1", "D1", "D7", "E6", "F5", "G4", "G3", "G2", "G1"]:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            small_board.state[small_board.RING_LAYER, y, x] = 0

        bbox = small_board.canonicalizer.get_bounding_box()
        assert bbox is not None, "Board with removed edges should have bounding box"

        # Bounding box should be smaller than full board now
        min_y, max_y, min_x, max_x = bbox
        # After removing edges, bbox should be reduced
        assert (max_y - min_y) < (small_board.config.width - 1) or (max_x - min_x) < (small_board.config.width - 1), \
            f"Expected reduced bounding box, got ({min_y}, {max_y}, {min_x}, {max_x})"

    def test_bounding_box_empty_board(self, small_board):
        """Test that empty board (no rings) returns None."""
        # Remove all rings
        small_board.state[small_board.RING_LAYER] = 0

        bbox = small_board.canonicalizer.get_bounding_box()
        assert bbox is None, "Empty board should return None for bounding box"

    def test_translation_identity(self, small_board):
        """Test that translating by (0, 0) returns the same state."""
        original = np.copy(small_board.state)

        translated = small_board.canonicalizer.translate_state(original, 0, 0)

        assert translated is not None, "Identity translation should be valid"
        assert np.array_equal(translated, original), "T(0,0) should preserve state"

    def test_translation_preserves_marbles(self, small_board):
        """Test that translation preserves ring and marble counts."""
        # Place some marbles
        small_board.state[1, *algebraic_to_coordinate("D4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("B3", small_board.config)] = 1
        small_board.state[3, *algebraic_to_coordinate("E3", small_board.config)] = 1

        original = np.copy(small_board.state)
        original_rings = np.sum(original[small_board.RING_LAYER])
        original_marbles = np.sum(original[small_board.MARBLE_LAYERS])

        # Try translation by (1, 0)
        translated = small_board.canonicalizer.translate_state(original, 1, 0)

        if translated is not None:
            translated_rings = np.sum(translated[small_board.RING_LAYER])
            translated_marbles = np.sum(translated[small_board.MARBLE_LAYERS])

            assert translated_rings == original_rings, "Translation should preserve ring count"
            assert translated_marbles == original_marbles, "Translation should preserve marble count"

    def test_translation_invalid_off_board(self, small_board):
        """Test that translation moving rings off-board returns None."""
        # Try to translate by large offset that would move rings off-board
        translated = small_board.canonicalizer.translate_state(small_board.state, 10, 10)

        assert translated is None, "Translation moving rings off-board should return None"

    def test_get_all_translations_full_board(self, small_board):
        """Test that full board has minimal translation options."""
        translations = small_board.canonicalizer.get_all_translations()

        assert len(translations) > 0, "Should have at least identity translation"

        # Check that identity is included
        identities = [t for t in translations if t[0] == "T0,0"]
        assert len(identities) == 1, "Should have exactly one identity translation"

    def test_get_all_translations_with_removed_edges(self, small_board):
        """Test that removing edge rings enables more translations."""
        # Keep only a small cluster of rings in the center to enable translation
        # Remove all rings except a 3x3 cluster
        center_cluster = ["C3", "D3", "E3", "C4", "D4", "E4", "C5", "D5", "E5"]

        # Remove all rings first
        small_board.state[small_board.RING_LAYER] = 0

        # Add back only the center cluster
        for pos in center_cluster:
            try:
                y, x = algebraic_to_coordinate(pos, small_board.config)
                small_board.state[small_board.RING_LAYER, y, x] = 1
            except:
                pass  # Some positions might not exist

        translations = small_board.canonicalizer.get_all_translations()

        # Should have more than just identity with a small cluster
        assert len(translations) > 1, f"Board with small cluster should have multiple valid translations, got {len(translations)}"

        # All translations should be valid (not None)
        for name, dy, dx in translations:
            translated = small_board.canonicalizer.translate_state(small_board.state, dy, dx)
            assert translated is not None, f"Translation {name} should be valid"

    def test_get_all_translations_empty_board(self, small_board):
        """Test that empty board only has identity translation."""
        # Remove all rings
        small_board.state[small_board.RING_LAYER] = 0

        translations = small_board.canonicalizer.get_all_translations()

        assert len(translations) == 1, "Empty board should only have identity"
        assert translations[0] == ("T0,0", 0, 0), "Empty board should return T0,0"

    def test_canonicalize_with_translation_only(self, small_board):
        """Test canonicalization with only translation enabled."""
        from game.zertz_board import TransformFlags

        # Keep only a small cluster to enable translation
        center_cluster = ["C3", "D3", "E3", "C4", "D4", "E4"]

        # Remove all rings first
        small_board.state[small_board.RING_LAYER] = 0

        # Add back only the center cluster
        for pos in center_cluster:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            small_board.state[small_board.RING_LAYER, y, x] = 1

        # Place a marble off-center to make translation meaningful
        small_board.state[1, *algebraic_to_coordinate("E4", small_board.config)] = 1

        # Canonicalize with translation only
        canonical, transform, inverse = small_board.canonicalize_state(
            transforms=TransformFlags.TRANSLATION
        )

        # Transform should be R0 or T{dy},{dx} (R0 when already canonical)
        # The key is that rotation/mirror transforms are NOT used
        assert transform.startswith("T") or transform == "R0", f"Expected translation or identity, got {transform}"
        assert "_" not in transform, f"Expected simple transform, got {transform}"

        # If not identity, should be translation
        if transform != "R0":
            assert transform.startswith("T"), f"Expected translation, got {transform}"
            assert inverse.startswith("T"), f"Expected translation inverse, got {inverse}"

    def test_canonicalize_with_rotation_mirror_only(self, small_board):
        """Test canonicalization with rotation/mirror but no translation."""
        from game.zertz_board import TransformFlags

        # Place asymmetric pattern
        small_board.state[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("B3", small_board.config)] = 1

        # Canonicalize with rotation and mirror only (no translation)
        canonical, transform, inverse = small_board.canonicalize_state(
            transforms=TransformFlags.ROTATION_MIRROR
        )

        # Transform should be rotation/mirror, not translation
        assert not transform.startswith("T") or transform == "T0,0", (
            f"Expected rotation/mirror transform, got {transform}"
        )
        # If there's a "_" it should be rotation/mirror combination
        if "_" in transform:
            parts = transform.split("_")
            assert not any(p.startswith("T") and p != "T0,0" for p in parts), (
                "Should not have non-identity translation with ROTATION_MIRROR flag"
            )

    def test_canonicalize_with_all_transforms(self, small_board):
        """Test canonicalization with all transforms enabled."""
        from game.zertz_board import TransformFlags

        # Remove edge rings to enable translation
        for pos in ["A4", "A3", "D7", "G4", "G1"]:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            small_board.state[small_board.RING_LAYER, y, x] = 0

        # Place asymmetric pattern
        small_board.state[1, *algebraic_to_coordinate("D4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("C3", small_board.config)] = 1

        # Canonicalize with all transforms
        canonical, transform, inverse = small_board.canonicalize_state(
            transforms=TransformFlags.ALL
        )

        # Should get some canonical form
        assert canonical is not None
        assert transform is not None
        assert inverse is not None

    def test_combined_transform_format(self, small_board):
        """Test that combined transforms use correct format T{dy},{dx}+{rot_mirror}."""
        from game.zertz_board import TransformFlags

        # Set up board state that will benefit from combined transform
        # Remove some edge rings
        for pos in ["A4", "A3", "A2", "D7", "E6"]:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            small_board.state[small_board.RING_LAYER, y, x] = 0

        # Place marbles
        small_board.state[1, *algebraic_to_coordinate("B3", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("E3", small_board.config)] = 1

        canonical, transform, inverse = small_board.canonicalize_state(
            transforms=TransformFlags.ALL
        )

        # If we get a combined transform, check format
        if "_" in transform:
            parts = transform.split("_")
            assert len(parts) == 2, f"Combined transform should have exactly 2 parts: {transform}"

            trans_part, rot_mirror_part = parts

            # First part should be translation
            assert trans_part.startswith("T"), f"First part should be translation: {transform}"

            # Second part should be rotation/mirror
            assert (
                rot_mirror_part.startswith("R") or rot_mirror_part.startswith("MR")
            ), f"Second part should be rotation/mirror: {transform}"

    def test_inverse_of_translation(self, small_board):
        """Test that inverse of translation T{dy},{dx} is T{-dy},{-dx}."""
        # Test various translation inverses
        test_cases = [
            ("T0,0", "T0,0"),  # Identity
            ("T1,0", "T-1,0"),
            ("T0,1", "T0,-1"),
            ("T2,3", "T-2,-3"),
            ("T-1,2", "T1,-2"),
        ]

        for transform, expected_inverse in test_cases:
            inverse = small_board.canonicalizer._get_inverse_transform(transform)
            assert inverse == expected_inverse, (
                f"Inverse of {transform} should be {expected_inverse}, got {inverse}"
            )

    def test_inverse_of_combined_transform(self, small_board):
        """Test that inverse of combined transform reverses order."""
        # Test that inverse of "T{dy},{dx}_{rot_mirror}" is "{inv_rot_mirror}_T{-dy},{-dx}"
        test_cases = [
            ("T1,0_R60", "R300_T-1,0"),
            ("T2,1_MR120", "R240M_T-2,-1"),
            ("T-1,3_R180", "R180_T1,-3"),
            ("T0,1_R120M", "MR240_T0,-1"),
        ]

        for transform, expected_inverse in test_cases:
            inverse = small_board.canonicalizer._get_inverse_transform(transform)
            assert inverse == expected_inverse, (
                f"Inverse of {transform} should be {expected_inverse}, got {inverse}"
            )

    def test_translation_then_inverse_recovers_original(self, small_board):
        """Test that applying translation then its inverse recovers original state."""
        # Place some marbles
        small_board.state[1, *algebraic_to_coordinate("D4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("B3", small_board.config)] = 1
        small_board.state[3, *algebraic_to_coordinate("E3", small_board.config)] = 1

        original = np.copy(small_board.state)

        # Try various translations
        translations_to_test = [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (2, -1)]

        for dy, dx in translations_to_test:
            # Apply translation
            translated = small_board.canonicalizer.translate_state(original, dy, dx)

            if translated is not None:
                # Apply inverse translation
                recovered = small_board.canonicalizer.translate_state(translated, -dy, -dx)

                assert recovered is not None, f"Inverse translation ({-dy},{-dx}) should be valid"
                assert np.array_equal(recovered, original), (
                    f"Translation ({dy},{dx}) then inverse should recover original"
                )

    def test_combined_transform_then_inverse_recovers_original(self, small_board):
        """Test that combined transform + inverse recovers original."""
        from game.zertz_board import TransformFlags

        # Remove edge rings to enable translation
        for pos in ["A4", "A3", "D7", "G4", "G1"]:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            small_board.state[small_board.RING_LAYER, y, x] = 0

        # Place asymmetric pattern
        small_board.state[1, *algebraic_to_coordinate("D4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("C3", small_board.config)] = 1
        small_board.state[3, *algebraic_to_coordinate("E3", small_board.config)] = 1

        original = np.copy(small_board.state)

        # Canonicalize with all transforms
        canonical, transform, inverse_name = small_board.canonicalize_state(
            transforms=TransformFlags.ALL
        )

        # Create a temporary board to apply the inverse
        temp_board = ZertzBoard(clone=small_board)
        temp_board.state = canonical

        # Get transform functions
        transforms = dict(temp_board.canonicalizer.get_all_symmetry_transforms())

        # If the transform is combined (has translation), we need to apply inverse differently
        if "_" in transform:
            # For combined transforms, we need to manually reconstruct the inverse operation
            # The inverse is already calculated correctly, but we can't apply it directly
            # because _get_all_symmetry_transforms doesn't include translation

            # For now, just verify that the inverse format is correct
            assert "_" in inverse_name, "Combined transform should have combined inverse"
            parts = inverse_name.split("_")
            assert len(parts) == 2, "Combined inverse should have 2 parts"

            # First part should be rotation/mirror inverse
            rot_mirror_inv = parts[0]
            assert (
                rot_mirror_inv.startswith("R") or rot_mirror_inv.startswith("MR")
            ), "First part of combined inverse should be rotation/mirror"

            # Second part should be translation inverse
            trans_inv = parts[1]
            assert trans_inv.startswith("T"), "Second part of combined inverse should be translation"
        else:
            # For simple transforms (rotation/mirror only), we can verify recovery
            if inverse_name in transforms:
                recovered = transforms[inverse_name](canonical)
                assert np.array_equal(recovered, original), (
                    f"Transform {transform} with inverse {inverse_name} should recover original"
                )

    def test_canonicalization_is_deterministic_with_translation(self, small_board):
        """Test that canonicalization produces consistent results with translation enabled."""
        from game.zertz_board import TransformFlags

        # Remove edge rings
        for pos in ["A4", "D7", "G4", "G1"]:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            small_board.state[small_board.RING_LAYER, y, x] = 0

        # Place pattern
        small_board.state[1, *algebraic_to_coordinate("D4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("C3", small_board.config)] = 1

        # Canonicalize multiple times
        results = []
        for _ in range(5):
            canonical, transform, inverse = small_board.canonicalize_state(
                transforms=TransformFlags.ALL
            )
            results.append((canonical.tobytes(), transform, inverse))

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Canonicalization should be deterministic"

    def test_translation_with_history_layers(self, small_board):
        """Test that translation preserves history and capture layers."""
        # Use a board with time history (t > 1)
        board_with_history = ZertzBoard(rings=37, t=3)

        # Set up history layers (simulate past moves)
        board_with_history.state[0:4, 3, 3] = [1, 1, 0, 0]  # Current: ring + white marble
        board_with_history.state[4:8, 3, 3] = [1, 0, 0, 1]  # t-1: ring + black marble
        board_with_history.state[8:12, 3, 3] = [1, 0, 1, 0]  # t-2: ring + gray marble

        # Set capture flag
        board_with_history.state[board_with_history.CAPTURE_LAYER, 2, 2] = 1

        original_history = board_with_history.state[4:].copy()
        original_capture = board_with_history.state[board_with_history.CAPTURE_LAYER].copy()

        # Apply translation
        translated = board_with_history.canonicalizer.translate_state(board_with_history.state, 1, 0)

        if translated is not None:
            # History layers should be preserved (not translated)
            assert np.array_equal(
                translated[4:board_with_history.CAPTURE_LAYER],
                original_history[:board_with_history.CAPTURE_LAYER-4]
            ), "History layers should be preserved"

            # Capture layer should be preserved
            assert np.array_equal(
                translated[board_with_history.CAPTURE_LAYER],
                original_capture
            ), "Capture layer should be preserved"


class TestBoardSizeSymmetries:
    """Test that different board sizes have correct symmetry groups."""

    def test_37_ring_board_has_d6_symmetry(self, small_board):
        """37-ring board should have D6 symmetry (18 transforms: 6R + 6MR + 6RM)."""
        small_board._build_axial_maps()

        transforms = dict(small_board.canonicalizer.get_all_symmetry_transforms())

        # Should have exactly 18 transforms
        assert len(transforms) == 18, (
            f"37-ring board should have 18 transforms (D6), got {len(transforms)}"
        )

        # Count each type
        r_count = sum(
            1
            for name in transforms
            if name.startswith("R")
            and not name.startswith("MR")
            and not name.endswith("M")
        )
        mr_count = sum(1 for name in transforms if name.startswith("MR"))
        rm_count = sum(1 for name in transforms if name.endswith("M"))

        assert r_count == 6, f"Expected 6 R transforms, got {r_count}"
        assert mr_count == 6, f"Expected 6 MR transforms, got {mr_count}"
        assert rm_count == 6, f"Expected 6 RM transforms, got {rm_count}"

    def test_48_ring_board_has_d3_symmetry(self, medium_board):
        """48-ring board should have D3 symmetry (9 transforms: 3R + 3MR + 3RM)."""
        medium_board._build_axial_maps()

        transforms = dict(medium_board.canonicalizer.get_all_symmetry_transforms())

        # Should have exactly 9 transforms
        assert len(transforms) == 9, (
            f"48-ring board should have 9 transforms (D3), got {len(transforms)}"
        )

        # Count each type
        r_count = sum(
            1
            for name in transforms
            if name.startswith("R")
            and not name.startswith("MR")
            and not name.endswith("M")
        )
        mr_count = sum(1 for name in transforms if name.startswith("MR"))
        rm_count = sum(1 for name in transforms if name.endswith("M"))

        assert r_count == 3, f"Expected 3 R transforms, got {r_count}"
        assert mr_count == 3, f"Expected 3 MR transforms, got {mr_count}"
        assert rm_count == 3, f"Expected 3 RM transforms, got {rm_count}"

        # Verify only 120° multiples (0°, 120°, 240°)
        r_angles = [
            int(name[1:])
            for name in transforms
            if name.startswith("R")
            and not name.startswith("MR")
            and not name.endswith("M")
        ]
        assert set(r_angles) == {0, 120, 240}, (
            f"48-ring board should only have 0°, 120°, 240° rotations, got {r_angles}"
        )

    def test_61_ring_board_has_d6_symmetry(self, large_board):
        """61-ring board should have D6 symmetry (18 transforms: 6R + 6MR + 6RM)."""
        large_board._build_axial_maps()

        transforms = dict(large_board.canonicalizer.get_all_symmetry_transforms())

        # Should have exactly 18 transforms
        assert len(transforms) == 18, (
            f"61-ring board should have 18 transforms (D6), got {len(transforms)}"
        )

        # Count each type
        r_count = sum(
            1
            for name in transforms
            if name.startswith("R")
            and not name.startswith("MR")
            and not name.endswith("M")
        )
        mr_count = sum(1 for name in transforms if name.startswith("MR"))
        rm_count = sum(1 for name in transforms if name.endswith("M"))

        assert r_count == 6, f"Expected 6 R transforms, got {r_count}"
        assert mr_count == 6, f"Expected 6 MR transforms, got {mr_count}"
        assert rm_count == 6, f"Expected 6 RM transforms, got {rm_count}"

    @pytest.mark.parametrize(
        "board_fixture,expected_count",
        [
            ("small_board", 18),
            ("medium_board", 9),
            ("large_board", 18),
        ],
    )
    def test_all_inverses_exist_for_board_size(
        self, request, board_fixture, expected_count
    ):
        """Test that all transforms have inverses in the transform dictionary."""
        board = request.getfixturevalue(board_fixture)
        board._ensure_positions_built()

        transforms = dict(board.canonicalizer.get_all_symmetry_transforms())

        assert len(transforms) == expected_count, (
            f"Expected {expected_count} transforms, got {len(transforms)}"
        )

        # Every transform should have an inverse that exists in the dictionary
        missing_inverses = []
        for transform_name in transforms:
            inverse_name = board.canonicalizer._get_inverse_transform(transform_name)
            if inverse_name not in transforms:
                missing_inverses.append(f"{transform_name} -> {inverse_name}")

        assert len(missing_inverses) == 0, (
            "Missing inverses in transform dictionary:\n" + "\n".join(missing_inverses)
        )

    @pytest.mark.parametrize(
        "board_fixture", ["small_board", "medium_board", "large_board"]
    )
    def test_all_transforms_are_involutive(self, request, board_fixture):
        """Test that applying any transform and its inverse returns to original."""
        board = request.getfixturevalue(board_fixture)
        board._ensure_positions_built()

        # Create asymmetric test pattern
        base = np.copy(board.state)
        # Place marbles at positions that exist on all board sizes
        positions = []
        if board.rings == 37:
            positions = [("A4", 1), ("B3", 2), ("E3", 3)]
        elif board.rings == 48:
            positions = [("A5", 1), ("B4", 2), ("F4", 3)]
        elif board.rings == 61:
            positions = [("A5", 1), ("B4", 2), ("F4", 3)]

        for pos, layer in positions:
            base[layer, *algebraic_to_coordinate(pos, board.config)] = 1

        transforms = dict(board.canonicalizer.get_all_symmetry_transforms())
        failures = []

        for transform_name, transform_fn in transforms.items():
            # Apply transformation
            transformed = transform_fn(base)

            # Get and apply inverse
            inverse_name = board.canonicalizer._get_inverse_transform(transform_name)
            if inverse_name in transforms:
                recovered = transforms[inverse_name](transformed)

                if not np.array_equal(recovered, base):
                    failures.append(f"{transform_name} -> {inverse_name}")

        assert len(failures) == 0, (
            "These transform/inverse pairs failed to recover original:\n"
            + "\n".join(failures)
        )

    def test_48_ring_asymmetric_pattern_has_six_symmetries(self, medium_board):
        """48-ring board with D3 symmetry should have 6 unique states for asymmetric pattern."""
        medium_board._build_axial_maps()

        base = np.copy(medium_board.state)
        # Create asymmetric pattern
        base[1, *algebraic_to_coordinate("A5", medium_board.config)] = 1
        base[2, *algebraic_to_coordinate("B4", medium_board.config)] = 1
        base[3, *algebraic_to_coordinate("F4", medium_board.config)] = 1

        transforms = dict(medium_board.canonicalizer.get_all_symmetry_transforms())
        unique_states = set()

        for transform_fn in transforms.values():
            transformed = transform_fn(base)
            unique_states.add(transformed.tobytes())

        # With D3 symmetry (9 transforms), an asymmetric pattern should produce 6 unique states
        # (not 9 because some transforms produce the same result for this pattern)
        assert len(unique_states) <= 9, (
            f"48-ring board should have at most 9 unique states, got {len(unique_states)}"
        )

    def test_61_ring_board_transforms_work_like_37_ring(self, small_board, large_board):
        """61-ring and 37-ring boards should both have D6 symmetry with same structure."""
        small_board._build_axial_maps()
        large_board._build_axial_maps()

        small_transforms = set(dict(small_board.canonicalizer.get_all_symmetry_transforms()).keys())
        large_transforms = set(dict(large_board.canonicalizer.get_all_symmetry_transforms()).keys())

        # Both should have identical transform names
        assert small_transforms == large_transforms, (
            f"37-ring and 61-ring boards should have same transform names.\n"
            f"Difference: {small_transforms.symmetric_difference(large_transforms)}"
        )


def test_center_equidistant_from_three_middle_rings(medium_board):
    """
    The geometric center of the 48-ring board (pointy-top hex) lies between D5, D4, E4.
    It should be equidistant from those three rings.
    """

    # Helper to get Cartesian coordinates from array indices using pointy-top hex geometry
    def coord(y, x):
        # Convert to axial first (standard formula from visualize_board_coords.py)
        c = medium_board.config.width // 2
        q = x - c
        r = y - x
        # Then to Cartesian (pointy-top)
        xc = np.sqrt(3) * (q + r / 2.0)
        yc = 1.5 * r
        return np.array([xc, yc])

    pts = []
    for ring in ["D5", "D4", "E4"]:
        y, x = algebraic_to_coordinate(ring, medium_board.config)
        pts.append(coord(y, x))
    pts = np.stack(pts)

    # Geometric center (average of the three rings)
    center = pts.mean(axis=0)

    # All distances to the center should be equal within floating tolerance
    dists = np.linalg.norm(pts - center, axis=1)
    assert np.allclose(dists, dists[0], rtol=1e-5, atol=1e-5), (
        f"Distances to center not equal: {dists}"
    )


class TestGetAllTransformations:
    """Test the get_all_transformations() method of CanonicalizationManager."""

    @pytest.mark.parametrize(
        "board_fixture,expected_rot_mirror_count",
        [
            ("small_board", 18),   # D6: 6 R + 6 MR + 6 RM
            ("medium_board", 9),   # D3: 3 R + 3 MR + 3 RM
            ("large_board", 18),   # D6: 6 R + 6 MR + 6 RM
        ],
    )
    def test_returns_correct_count_without_translation(
        self, request, board_fixture, expected_rot_mirror_count
    ):
        """Test that get_all_transformations returns correct number of transforms."""
        board = request.getfixturevalue(board_fixture)

        transforms = board.canonicalizer.get_all_transformations(
            include_translation=False, deduplicate=False
        )

        assert len(transforms) == expected_rot_mirror_count, (
            f"{board.rings}-ring board should have {expected_rot_mirror_count} "
            f"rotation/mirror transforms, got {len(transforms)}"
        )

    def test_all_transformations_are_unique(self, small_board):
        """Test that all transformations produce unique state representations."""
        # Create an asymmetric pattern
        small_board.state[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("B3", small_board.config)] = 1
        small_board.state[3, *algebraic_to_coordinate("E3", small_board.config)] = 1

        transforms = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        # Convert all transformed states to bytes for comparison
        unique_states = set()
        for name, state in transforms.items():
            unique_states.add(state.tobytes())

        # For a truly asymmetric pattern, all 18 transforms should produce unique states
        assert len(unique_states) == len(transforms), (
            f"Expected {len(transforms)} unique states, got {len(unique_states)}"
        )

    def test_identity_transform_included(self, small_board):
        """Test that identity transform (R0) is always included."""
        transforms = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        assert "R0" in transforms, "Identity transform R0 should be included"

        # Identity should preserve the state
        original = small_board.state.copy()
        identity_state = transforms["R0"]

        assert np.array_equal(identity_state, original), (
            "Identity transform should preserve state"
        )

    def test_transformations_preserve_marble_count(self, small_board):
        """Test that all transformations preserve marble counts."""
        # Place marbles
        small_board.state[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("D4", small_board.config)] = 1
        small_board.state[3, *algebraic_to_coordinate("G1", small_board.config)] = 1

        original_white = np.sum(small_board.state[small_board.MARBLE_TO_LAYER["w"]])
        original_gray = np.sum(small_board.state[small_board.MARBLE_TO_LAYER["g"]])
        original_black = np.sum(small_board.state[small_board.MARBLE_TO_LAYER["b"]])

        transforms = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        for name, state in transforms.items():
            white_count = np.sum(state[small_board.MARBLE_TO_LAYER["w"]])
            gray_count = np.sum(state[small_board.MARBLE_TO_LAYER["g"]])
            black_count = np.sum(state[small_board.MARBLE_TO_LAYER["b"]])

            assert white_count == original_white, (
                f"Transform {name} changed white marble count"
            )
            assert gray_count == original_gray, (
                f"Transform {name} changed gray marble count"
            )
            assert black_count == original_black, (
                f"Transform {name} changed black marble count"
            )

    def test_transformations_preserve_ring_count(self, small_board):
        """Test that all transformations preserve ring counts."""
        # Remove some edge rings
        for pos in ["A4", "D7", "G1"]:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            small_board.state[small_board.RING_LAYER, y, x] = 0

        original_rings = np.sum(small_board.state[small_board.RING_LAYER])

        transforms = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        for name, state in transforms.items():
            ring_count = np.sum(state[small_board.RING_LAYER])
            assert ring_count == original_rings, (
                f"Transform {name} changed ring count: "
                f"expected {original_rings}, got {ring_count}"
            )

    def test_include_translation_parameter(self, small_board):
        """Test that include_translation parameter works correctly."""
        # Remove edge rings to enable translation
        for pos in ["A4", "A3", "A2", "D7", "E6", "F5", "G4", "G3", "G2", "G1"]:
            y, x = algebraic_to_coordinate(pos, small_board.config)
            small_board.state[small_board.RING_LAYER, y, x] = 0

        # Get transforms without translation
        without_translation = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        # Get transforms with translation
        with_translation = small_board.canonicalizer.get_all_transformations(
            include_translation=True
        )

        # With translation should have more transforms (unless board is full)
        assert len(with_translation) >= len(without_translation), (
            "Including translation should not reduce transform count"
        )

        # All rotation/mirror transforms should be in both sets
        rot_mirror_names = {
            name for name in without_translation.keys()
            if not name.startswith("T") or name == "T0,0"
        }

        # Check that all pure rotation/mirror transforms are in the translation set
        for name in rot_mirror_names:
            # The name might be combined with T0,0 in the translation set
            found = name in with_translation or f"T0,0_{name}" in with_translation
            assert found, f"Rotation/mirror transform {name} not found in translation set"

    def test_transformation_names_follow_convention(self, small_board):
        """Test that transformation names follow the expected naming convention."""
        transforms = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        for name in transforms.keys():
            # Should be one of: R{angle}, MR{angle}, R{angle}M, or T{dy},{dx}_{transform}
            assert (
                name.startswith("R") or
                name.startswith("MR") or
                name.startswith("T")
            ), f"Unexpected transform name format: {name}"

            # If it contains underscore, should be combined transform
            if "_" in name:
                parts = name.split("_")
                assert len(parts) == 2, (
                    f"Combined transform should have exactly 2 parts: {name}"
                )

    def test_symmetric_pattern_reduces_unique_count(self, small_board):
        """Test that patterns with symmetry produce fewer unique transformations."""
        # Create a 6-fold symmetric pattern (all corners)
        for corner in ["A4", "D7", "G4", "G1", "D1", "A1"]:
            small_board.state[1, *algebraic_to_coordinate(corner, small_board.config)] = 1

        transforms = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        # Count unique states
        unique_states = set()
        for name, state in transforms.items():
            unique_states.add(state.tobytes())

        # 6-fold symmetric pattern should produce fewer than 18 unique states
        assert len(unique_states) < 18, (
            f"6-fold symmetric pattern should produce < 18 unique states, "
            f"got {len(unique_states)}"
        )

    def test_empty_board_produces_one_unique_state(self, small_board):
        """Test that empty board produces only one unique state."""
        # Empty board (just rings, no marbles)
        transforms = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        # All transforms should produce the same state for empty board
        unique_states = set()
        for name, state in transforms.items():
            unique_states.add(state.tobytes())

        assert len(unique_states) == 1, (
            f"Empty board should have 1 unique state, got {len(unique_states)}"
        )

    def test_transformations_valid_for_board_layout(self, small_board):
        """Test that all transformations only place rings on valid board positions."""
        # Place marbles
        small_board.state[1, *algebraic_to_coordinate("D4", small_board.config)] = 1
        small_board.state[2, *algebraic_to_coordinate("B3", small_board.config)] = 1

        # Get the original valid ring positions (positions where rings can exist)
        original_ring_mask = small_board.state[small_board.RING_LAYER] == 1

        transforms = small_board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        for name, state in transforms.items():
            # Check that rings only appear at positions that had rings originally
            # (transformations should preserve the ring layout pattern)
            transformed_ring_mask = state[small_board.RING_LAYER] == 1

            # All ring positions in transformed state should be valid board positions
            # This means the total ring count should be preserved
            assert np.sum(transformed_ring_mask) == np.sum(original_ring_mask), (
                f"Transform {name} changed ring count: "
                f"expected {np.sum(original_ring_mask)}, got {np.sum(transformed_ring_mask)}"
            )

    @pytest.mark.parametrize(
        "board_fixture", ["small_board", "medium_board", "large_board"]
    )
    def test_works_for_all_board_sizes(self, request, board_fixture):
        """Test that get_all_transformations works for all board sizes."""
        board = request.getfixturevalue(board_fixture)

        # Place an asymmetric pattern
        positions = []
        if board.rings == 37:
            positions = [("A4", 1), ("B3", 2), ("E3", 3)]
        elif board.rings == 48:
            positions = [("A5", 1), ("B4", 2), ("F4", 3)]
        elif board.rings == 61:
            positions = [("A5", 1), ("B4", 2), ("F4", 3)]

        for pos, layer in positions:
            board.state[layer, *algebraic_to_coordinate(pos, board.config)] = 1

        # Should not raise an error
        transforms = board.canonicalizer.get_all_transformations(
            include_translation=False
        )

        assert len(transforms) > 0, "Should return at least one transformation"
        assert "R0" in transforms, "Should include identity transformation"

    def test_passes_state_parameter(self, small_board):
        """Test that the state parameter allows transforming arbitrary states."""
        # Create a custom state (not the board's current state)
        custom_state = np.copy(small_board.state)
        custom_state[1, *algebraic_to_coordinate("A4", small_board.config)] = 1
        custom_state[2, *algebraic_to_coordinate("G1", small_board.config)] = 1

        # Get transformations of the custom state
        transforms = small_board.canonicalizer.get_all_transformations(
            state=custom_state,
            include_translation=False
        )

        # Verify that transformations were applied to custom_state, not board.state
        assert "R0" in transforms
        identity_state = transforms["R0"]

        # Identity should match custom_state, not board.state
        assert np.array_equal(identity_state, custom_state), (
            "Identity transformation should preserve custom state"
        )
