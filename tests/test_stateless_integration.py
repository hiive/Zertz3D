"""
Unit tests for stateless logic integration with stateful classes.

Tests that refactored methods in ZertzBoard and ZertzGame properly delegate
to zertz_logic.py functions and produce identical results.

This ensures:
1. Helper function refactoring doesn't change behavior
2. Stateful wrappers correctly call stateless implementations
3. State mutations are correctly applied from stateless function outputs
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame
from game.zertz_board import ZertzBoard
from hiivelabs_mcts import (
    is_inbounds,
    get_neighbors,
    get_jump_destination,
    get_marble_type_at,
    get_supply_index,
    get_captured_index,
    get_valid_actions,
    get_placement_moves,
    get_capture_moves,
    get_regions,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def board():
    """Create a fresh 37-ring board for each test."""
    return ZertzBoard(rings=ZertzBoard.SMALL_BOARD_37)


@pytest.fixture
def game():
    """Create a fresh game for each test."""
    return ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)


# ============================================================================
# Helper Function Delegation Tests
# ============================================================================


class TestHelperFunctionDelegation:
    """Test that stateful helpers correctly delegate to stateless implementations."""

    def test_is_inbounds_delegation(self, board):
        """Test _is_inbounds() produces same results as stateless version."""
        config = board.config

        # Test in-bounds positions
        test_cases = [
            ((0, 0), True),
            ((3, 3), True),
            ((config.width - 1, config.width - 1), True),
            # Out of bounds
            ((-1, 0), False),
            ((0, -1), False),
            ((config.width, 0), False),
            ((0, config.width), False),
            ((config.width, config.width), False),
        ]

        for index, expected in test_cases:
            # Call stateful method
            stateful_result = board._is_inbounds(index)

            # Call stateless function directly
            y, x = index
            stateless_result = is_inbounds(y, x, config.width)

            # They should match
            assert stateful_result == stateless_result == expected, \
                f"Mismatch for index {index}: stateful={stateful_result}, " \
                f"stateless={stateless_result}, expected={expected}"

    def test_get_neighbors_delegation(self, board):
        """Test get_neighbors() produces same results as stateless version."""
        config = board.config

        # Test center position
        test_positions = [
            (0, 0),
            (3, 3),
            (config.width - 1, config.width - 1),
            (2, 4),
        ]

        for pos in test_positions:
            # Call stateful method
            stateful_neighbors = board.get_neighbors(pos)

            # Call stateless function directly
            y, x = pos
            stateless_neighbors = get_neighbors(y, x, config)

            # They should match exactly
            assert stateful_neighbors == stateless_neighbors, \
                f"Neighbor list mismatch for position {pos}"

    def test_get_jump_destination_delegation(self, board):
        """Test get_jump_destination() produces same results as stateless version."""
        # Test jump calculations
        test_cases = [
            ((3, 3), (4, 3), (5, 3)),    # Down
            ((3, 3), (3, 2), (3, 1)),    # Left
            ((3, 3), (2, 2), (1, 1)),    # Up-left diagonal
        ]

        for start, cap, expected_dst in test_cases:
            # Call stateful method
            stateful_dst = board.get_jump_destination(start, cap)

            # Call stateless function directly
            stateless_dst = get_jump_destination(start, cap)

            # They should both match expected
            assert stateful_dst == stateless_dst == expected_dst, \
                f"Jump destination mismatch: start={start}, cap={cap}"

    def test_get_marble_type_at_delegation(self, board):
        """Test get_marble_type_at() produces same results as stateless version."""
        config = board.config

        # Place some marbles on the board
        test_marbles = [
            ((2, 2), 'w'),
            ((3, 3), 'g'),
            ((4, 4), 'b'),
        ]

        for pos, marble_type in test_marbles:
            # Place marble
            marble_layer = board.MARBLE_TO_LAYER[marble_type]
            board.state[marble_layer][pos] = 1

            # Call stateful method
            stateful_type = board.get_marble_type_at(pos)

            # Call stateless function directly
            y, x = pos
            stateless_type = get_marble_type_at(board.state, y, x)

            # They should match
            assert stateful_type == stateless_type == marble_type, \
                f"Marble type mismatch at position {pos}"

    def test_get_supply_index_delegation(self, board):
        """Test _get_supply_index() produces same results as stateless version."""
        config = board.config

        marble_types = ['w', 'g', 'b']
        expected_indices = [
            board.SUPPLY_W,
            board.SUPPLY_G,
            board.SUPPLY_B,
        ]

        for marble_type, expected_idx in zip(marble_types, expected_indices):
            # Call stateful method
            stateful_idx = board._get_supply_index(marble_type)

            # Call stateless function directly
            stateless_idx = get_supply_index(marble_type)

            # They should match
            assert stateful_idx == stateless_idx == expected_idx, \
                f"Supply index mismatch for marble type {marble_type}"

    def test_get_captured_index_delegation(self, board):
        """Test _get_captured_index() produces same results as stateless version."""
        config = board.config

        test_cases = [
            ('w', board.PLAYER_1, board.P1_CAP_W),
            ('g', board.PLAYER_1, board.P1_CAP_G),
            ('b', board.PLAYER_1, board.P1_CAP_B),
            ('w', board.PLAYER_2, board.P2_CAP_W),
            ('g', board.PLAYER_2, board.P2_CAP_G),
            ('b', board.PLAYER_2, board.P2_CAP_B),
        ]

        for marble_type, player, expected_idx in test_cases:
            # Call stateful method
            stateful_idx = board._get_captured_index(marble_type, player)

            # Call stateless function directly
            stateless_idx = get_captured_index(marble_type, player)

            # They should match
            assert stateful_idx == stateless_idx == expected_idx, \
                f"Captured index mismatch for {marble_type}, player {player}"


# ============================================================================
# Move Generation Delegation Tests
# ============================================================================


class TestMoveGenerationDelegation:
    """Test that move generation already properly delegates to stateless logic.

    These tests verify the EXISTING correct delegation (not new refactoring).
    """

    def test_get_valid_actions_delegation(self, board):
        """Test get_valid_actions() uses zertz_logic."""
        config = board.config

        # Call stateful method
        placement_stateful, capture_stateful = board.get_valid_actions()

        # Call stateless function directly
        placement_stateless, capture_stateless = \
            get_valid_actions(
                board.state, board.global_state, config
            )

        # They should match exactly
        assert np.array_equal(placement_stateful, placement_stateless), \
            "Placement moves should match"

        assert np.array_equal(capture_stateful, capture_stateless), \
            "Capture moves should match"

    def test_get_placement_moves_delegation(self, board):
        """Test get_placement_moves() uses zertz_logic."""
        config = board.config

        # Call stateful method
        placement_stateful = board.get_placement_moves()

        # Call stateless function directly
        placement_stateless = get_placement_moves(
            board.state, board.global_state, config
        )

        # They should match
        assert np.array_equal(placement_stateful, placement_stateless), \
            "Placement moves should match stateless implementation"

    def test_get_capture_moves_delegation(self, board):
        """Test get_capture_moves() uses zertz_logic."""
        # Call stateful method
        capture_stateful = board.get_capture_moves()

        # Call stateless function directly
        capture_stateless = get_capture_moves(
            board.state, board.config
        )

        # They should match
        assert np.array_equal(capture_stateful, capture_stateless), \
            "Capture moves should match stateless implementation"


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Test that refactored code maintains backward compatibility."""

    def test_helper_functions_in_action_execution(self, game):
        """Test that refactored helpers work correctly during action execution.

        This tests the full integration: helpers are called during normal
        game play and produce correct results.
        """
        # Execute a simple placement action
        placement, capture = game.get_valid_actions()

        # Find a valid placement action
        placement_indices = np.argwhere(placement)
        assert len(placement_indices) > 0, "Should have valid placements"

        marble_idx, put_y, put_x, rem_y, rem_x = placement_indices[0]

        # Execute action
        # put = put_y * game.board.config.width + put_x
        # rem = rem_y * game.board.config.width + rem_x
        game.take_action("PUT", (marble_idx, put_y, put_x, rem_y, rem_x))

        # Game should still be playable
        assert game.get_game_ended() is None, "Game should still be ongoing"

        # Should be able to get valid moves for next turn
        placement2, capture2 = game.get_valid_actions()
        assert np.any(placement2) or np.any(capture2), \
            "Should have valid moves after action"

    def test_capture_action_uses_refactored_helpers(self, game):
        """Test that capture actions work with refactored helper functions."""
        board = game.board

        # Set up a simple capture scenario
        # Place marbles at positions for a capture
        # Source marble at (3,3), target at (3,4), landing at (3,5)
        board.state[board.MARBLE_TO_LAYER['w']][3, 3] = 1
        board.state[board.MARBLE_TO_LAYER['g']][3, 4] = 1
        board.state[board.RING_LAYER][3, 5] = 1  # Landing spot

        # Get capture moves
        capture = board.get_capture_moves()

        # Should have at least one capture
        if np.any(capture):
            capture_indices = np.argwhere(capture)
            direction, y, x = capture_indices[0]

            # Execute capture using helper to convert to action format
            action = ZertzBoard.capture_indices_to_action(
                direction, y, x, board.config.width, board.DIRECTIONS
            )
            captured = board._take_capture_action(action)

            # Should have captured a marble
            # (Either returned value or state should reflect capture)
            p1_captures = board.global_state[board.P1_CAP_SLICE]
            p2_captures = board.global_state[board.P2_CAP_SLICE]
            total_captures = np.sum(p1_captures) + np.sum(p2_captures)

            assert total_captures > 0, "Should have captured at least one marble"

    def test_regions_function_delegation(self, board):
        """Test _get_regions() correctly delegates to stateless version."""
        config = board.config

        # Call stateful method
        regions_stateful = board._get_regions()

        # Call stateless function directly
        regions_stateless = get_regions(board.state, config)

        # Should have same number of regions
        assert len(regions_stateful) == len(regions_stateless), \
            "Should find same number of regions"

        # Each region should contain same positions (order may differ)
        for region_stateful in regions_stateful:
            # Find matching region in stateless result
            matching = False
            for region_stateless in regions_stateless:
                if set(region_stateful) == set(region_stateless):
                    matching = True
                    break

            assert matching, f"Could not find matching region for {region_stateful}"