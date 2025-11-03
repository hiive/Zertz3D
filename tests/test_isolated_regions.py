"""
Tests for isolated region handling according to official Zèrtz rules.

Key Rules:
1. If a move isolates rings, you MAY claim marbles on those rings
2. BUT ONLY IF all rings in the isolated region have marbles (no vacant rings)
3. If isolated region has ANY vacant rings: marbles are NOT captured (the region simply remains on the board)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from hiivelabs_mcts import algebraic_to_coordinate

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_board import ZertzBoard


@pytest.fixture
def small_board():
    """37-ring board"""
    return ZertzBoard(rings=ZertzBoard.SMALL_BOARD_37)


class TestIsolatedRegionCaptureRules:
    """Test that isolated region captures follow official rules."""

    def test_isolated_single_ring_with_marble_captures(self, small_board):
        """Isolating a single ring with marble should capture it."""
        # Manually create a simple scenario: G1 is isolated with a marble
        # Strategy: Remove all of G1's neighbors except one, then remove that last one

        # G1 is at bottom-right corner
        # Its neighbors are: G2 (above), F1 (left), F2 (diagonal upper-left)

        # Place marble on G1
        small_board.state[
            small_board.MARBLE_LAYERS.start, *algebraic_to_coordinate("G1", small_board.config)
        ] = 1  # white

        # Remove G2 and F1 manually to prepare for isolation
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G2", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F1", small_board.config)] = 0

        # Now place a marble somewhere and remove F2 to isolate G1
        # Place on D4 (safe location), remove F2
        put_action = (
            0,
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("D4", small_board.config)),
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("F2", small_board.config)),
        )

        p1_white_before = small_board.global_state[small_board.P1_CAP_W]
        isolated = small_board._take_placement_action(put_action)

        # Should capture the marble on G1
        assert isolated is not None
        assert len(isolated) == 1
        assert isolated[0]["marble"] == "w"
        assert small_board.global_state[small_board.P1_CAP_W] == p1_white_before + 1

        # G1 ring should be removed
        assert (
            small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G1", small_board.config)]
            == 0
        )

    def test_isolated_single_ring_vacant_remains_playable(self, small_board):
        """Isolating a single vacant ring should leave it available for future play."""
        # G1 is vacant (no marble)

        # Remove G2 and F1 manually
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G2", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F1", small_board.config)] = 0

        # Place marble on D4, remove F2 to isolate G1
        put_action = (
            0,
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("D4", small_board.config)),
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("F2", small_board.config)),
        )

        p1_captures_before = small_board.global_state[small_board.P1_CAP_SLICE].copy()
        isolated = small_board._take_placement_action(put_action)

        # Should NOT capture anything (G1 is vacant)
        assert isolated is None or len(isolated) == 0

        # No captures should have occurred
        assert np.array_equal(
            small_board.global_state[small_board.P1_CAP_SLICE], p1_captures_before
        )

        # G1 ring should still exist
        assert (
            small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G1", small_board.config)]
            == 1
        )

        # G1 should appear as an open ring
        assert algebraic_to_coordinate("G1", small_board.config) in small_board._get_open_rings()

    def test_isolated_two_rings_all_occupied_captures(self, small_board):
        """Isolating two rings both with marbles should capture both."""
        # Isolate G1 and G2 together, both with marbles

        # Place marbles on G1 and G2
        small_board.state[
            small_board.MARBLE_LAYERS.start, *algebraic_to_coordinate("G1", small_board.config)
        ] = 1  # white
        small_board.state[
            small_board.MARBLE_LAYERS.start + 1, *algebraic_to_coordinate("G2", small_board.config)
        ] = 1  # grey

        # Remove surrounding rings to isolate G1+G2
        # G1 neighbors: G2 (above), F1 (left), F2 (diagonal)
        # G2 neighbors: G3 (above), F2 (left), F3 (diagonal), G1 (below)
        # To isolate both: remove F1, F2, F3, G3
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F1", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F2", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F3", small_board.config)] = 0

        # Place marble on D4, remove G3 to complete isolation
        put_action = (
            0,
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("D4", small_board.config)),
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("G3", small_board.config)),
        )

        p1_white_before = small_board.global_state[small_board.P1_CAP_W]
        p1_grey_before = small_board.global_state[small_board.P1_CAP_G]

        isolated = small_board._take_placement_action(put_action)

        # Should capture both marbles
        assert isolated is not None
        assert len(isolated) == 2
        assert small_board.global_state[small_board.P1_CAP_W] == p1_white_before + 1
        assert small_board.global_state[small_board.P1_CAP_G] == p1_grey_before + 1

        # Both rings should be removed
        assert (
            small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G1", small_board.config)]
            == 0
        )
        assert (
            small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G2", small_board.config)]
            == 0
        )

    def test_isolated_two_rings_one_vacant_remains_playable(self, small_board):
        """Isolating two rings where one is vacant should leave both available (no capture)."""
        # Isolate G1 and G2, but G1 is vacant

        # Place marble only on G2 (G1 remains vacant)
        small_board.state[
            small_board.MARBLE_LAYERS.start, *algebraic_to_coordinate("G2", small_board.config)
        ] = 1  # white

        # Remove surrounding rings
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F1", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F2", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F3", small_board.config)] = 0

        # Place marble on D4, remove G3 to isolate
        put_action = (
            0,
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("D4", small_board.config)),
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("G3", small_board.config)),
        )

        p1_white_before = small_board.global_state[small_board.P1_CAP_W]
        isolated = small_board._take_placement_action(put_action)

        # Should NOT capture (G1 is vacant)
        assert isolated is None or len(isolated) == 0
        assert small_board.global_state[small_board.P1_CAP_W] == p1_white_before

        # Both rings should remain
        assert (
            small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G1", small_board.config)]
            == 1
        )
        assert (
            small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G2", small_board.config)]
            == 1
        )

        # Marble should still be on G2
        assert (
            small_board.state[
                small_board.MARBLE_LAYERS.start, *algebraic_to_coordinate("G2", small_board.config)
            ]
            == 1
        )

        open_rings = small_board._get_open_rings()
        assert algebraic_to_coordinate("G1", small_board.config) in open_rings
        assert algebraic_to_coordinate("G2", small_board.config) not in open_rings  # occupied

    def test_placement_triggered_isolation_capture(self, small_board):
        """
        Test that placing a marble that creates a fully-occupied isolated region
        triggers immediate capture of that region.

        Rule: If by placing a marble and/or removing a ring, there are any regions
        that are completely full of marbles, the player captures those marbles and
        removes their rings.

        This tests "self-isolation" - placing a marble that immediately gets captured
        because the region it's in becomes fully occupied and isolated.
        """
        # Place marble on D7
        d7_idx = algebraic_to_coordinate("D7", small_board.config)
        small_board.state[small_board.MARBLE_LAYERS.start, *d7_idx] = 1  # white

        # Remove D7's neighbors to isolate it: E6, C6, B5, C5, D6
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("E6", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("C6", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("B5", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("C5", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("D6", small_board.config)] = 0

        # Now place a marble somewhere else to trigger the isolation check
        # Place on C3, remove any removable ring
        put_action = (
            0,  # white marble
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("C3", small_board.config)),
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("G1", small_board.config)),
        )

        p1_white_before = small_board.global_state[small_board.P1_CAP_W]
        isolated = small_board._take_placement_action(put_action)

        # D7 should be captured (it's a fully-occupied isolated region)
        assert (
            small_board.state[small_board.MARBLE_LAYERS.start, *d7_idx] == 0
        ), "D7 marble should be captured (fully-occupied isolated region)"
        assert (
            small_board.state[small_board.RING_LAYER, *d7_idx] == 0
        ), "D7 ring should be removed (fully-occupied isolated region)"

        # Should have captured the marble from D7
        assert (
            small_board.global_state[small_board.P1_CAP_W] == p1_white_before + 1
        ), "Should capture 1 marble from D7"

        # Verify isolated list contains the D7 marble
        assert isolated is not None, "Isolation should occur"
        assert len(isolated) == 1, "Should capture D7 region"
        assert isolated[0]["marble"] == "w", "Captured marble should be white"


class TestPartiallyIsolatedRegionProperties:
    """Test properties of isolated regions that still contain empty rings."""

    def test_isolated_region_appears_in_get_regions(self, small_board):
        """Partially isolated regions should still appear in _get_regions()."""
        # Create isolated region: G1 vacant, separated from main board
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G2", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F1", small_board.config)] = 0

        put_action = (
            0,
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("D4", small_board.config)),
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("F2", small_board.config)),
        )
        small_board._take_placement_action(put_action)

        regions = small_board._get_regions()

        # Should have at least 2 regions (main + frozen)
        assert len(regions) >= 2

        # Find the G1 region
        g1_idx = algebraic_to_coordinate("G1", small_board.config)
        g1_region = None
        for region in regions:
            if g1_idx in region:
                g1_region = region
                break

        # G1 should be in a region
        assert g1_region is not None
        assert len(g1_region) == 1  # Just G1

    def test_partially_isolated_ring_is_placeable(self, small_board):
        """Partially isolated rings should remain valid placement targets."""
        # Create isolated region with G1 vacant
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("G2", small_board.config)] = 0
        small_board.state[small_board.RING_LAYER, *algebraic_to_coordinate("F1", small_board.config)] = 0

        put_action = (
            0,
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("D4", small_board.config)),
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("F2", small_board.config)),
        )
        small_board._take_placement_action(put_action)

        # Get valid moves
        placement_moves, _ = small_board.get_valid_moves()

        # G1 should be a valid placement location
        g1_flat = (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("G1", small_board.config))

        assert placement_moves[:, g1_flat, :].any(), "Isolated ring G1 should remain placeable"


class TestNoIsolation:
    """Test scenarios where no isolation occurs."""

    def test_no_isolation_no_capture(self, small_board):
        """Removing a ring that doesn't isolate anything should not capture."""
        # Place some marbles
        small_board.state[
            small_board.MARBLE_LAYERS.start, *algebraic_to_coordinate("D4", small_board.config)
        ] = 1
        small_board.state[
            small_board.MARBLE_LAYERS.start + 1, *algebraic_to_coordinate("E4", small_board.config)
        ] = 1

        # Remove a ring that doesn't cause isolation (G1 is far from D4/E4)
        put_action = (
            0,
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("C3", small_board.config)),
            (lambda pos: pos[0] * small_board.config.width + pos[1])(algebraic_to_coordinate("G1", small_board.config)),
        )

        p1_captures_before = small_board.global_state[small_board.P1_CAP_SLICE].copy()
        isolated = small_board._take_placement_action(put_action)

        # No isolation should occur
        assert isolated is None

        # No captures
        assert np.array_equal(
            small_board.global_state[small_board.P1_CAP_SLICE], p1_captures_before
        )

        # Marbles should still be on board
        assert (
            small_board.state[
                small_board.MARBLE_LAYERS.start, *algebraic_to_coordinate("D4", small_board.config)
            ]
            == 1
        )
        assert (
            small_board.state[
                small_board.MARBLE_LAYERS.start + 1, *algebraic_to_coordinate("E4", small_board.config)
            ]
            == 1
        )
