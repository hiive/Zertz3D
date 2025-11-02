"""
Unit tests for ZertzBoard error handling and edge cases.

Tests error paths and boundary conditions that aren't covered by other test files:
- Invalid board size initialization
- Invalid placement actions
- Invalid capture actions
- Marble supply exhaustion scenarios
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_board import ZertzBoard


class TestBoardInitializationErrors:
    """Test error handling during board initialization."""

    def test_generate_standard_board_layout_invalid_size(self):
        """Test that generate_standard_layout_mask raises ValueError for invalid size."""
        from hiivelabs_mcts import generate_standard_layout_mask
        with pytest.raises(ValueError, match="Unsupported ring count"):
            # This should fail because 42 is not a valid board size (only 37, 48, 61)
            generate_standard_layout_mask(42, 7)

    def test_unsupported_board_size_without_layout(self):
        """Test that creating board with unsupported size raises ValueError."""
        # Try to create a board with a size that's not in HEX_NUMBERS
        with pytest.raises(ValueError, match="Unsupported board size: 99 rings"):
            ZertzBoard(rings=99)

    def test_supported_hex_number_without_standard_layout(self):
        """Test that boards with valid HEX_NUMBERS but non-standard sizes raise error."""
        # We only support standard board sizes: 37, 48, and 61
        with pytest.raises(ValueError, match="Unsupported board size: 19 rings"):
            ZertzBoard(rings=19)


class TestPlacementActionErrors:
    """Test error handling for invalid placement actions."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_placement_on_occupied_position(self, rings):
        """Test that placing marble on occupied ring raises ValueError (all board sizes)."""
        board = ZertzBoard(rings=rings)

        # Place a white marble at position (3, 3)
        # Action format: (marble_type_idx, put_loc, rem_loc)
        # marble_type_idx: 0=white, 1=gray, 2=black
        # put_loc: flat index = y * width + x
        put_loc = 3 * board.config.width + 3
        rem_loc = board.config.width**2  # No removal

        first_action = (0, put_loc, rem_loc)
        board.take_action(first_action, "PUT")

        # Try to place another marble at the same position
        # Need to switch to a valid action for current player
        board._next_player()  # Switch back to player 1

        second_action = (1, put_loc, rem_loc)  # Gray marble at same position

        with pytest.raises(ValueError, match="Invalid placement: position .* is not an empty ring"):
            board.take_action(second_action, "PUT")

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_placement_on_removed_ring(self, rings):
        """Test that placing marble on removed ring raises ValueError (all board sizes)."""
        board = ZertzBoard(rings=rings)

        # Find a removable ring and remove it
        placement, capture = board.get_valid_moves()
        valid_placements = np.argwhere(placement)

        # Find a placement action that includes a ring removal
        action_with_removal = None
        for action in valid_placements:
            marble_idx, put_loc, rem_loc = action
            if rem_loc != board.config.width**2:  # Has a removal
                action_with_removal = tuple(action)
                break

        if action_with_removal is None:
            pytest.skip("No removable rings found on this board configuration")

        marble_idx, put_loc, rem_loc = action_with_removal
        board.take_action(action_with_removal, "PUT")

        # Switch back to same player
        board._next_player()

        # Try to place a marble on the removed ring
        invalid_action = (0, rem_loc, board.config.width**2)

        with pytest.raises(ValueError, match="Invalid placement: position .* is not an empty ring"):
            board.take_action(invalid_action, "PUT")


class TestMarbleSupplyErrors:
    """Test error handling for marble supply exhaustion scenarios."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_placement_when_marble_type_exhausted_with_pool_remaining(self, rings):
        """Test error when placing marble type that's exhausted but pool has other marbles (all board sizes)."""
        board = ZertzBoard(rings=rings)

        # Exhaust white marbles but leave other marbles in pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 5
        board.global_state[board.SUPPLY_B] = 5

        # Current player has some captured white marbles (but can't use them yet)
        board.global_state[board.P1_CAP_W] = 2

        # Try to place a white marble
        put_loc = 3 * board.config.width + 3
        rem_loc = board.config.width**2
        action = (0, put_loc, rem_loc)  # 0 = white

        with pytest.raises(ValueError, match="No w marbles in supply. Cannot use captured marbles until entire pool is empty"):
            board.take_action(action, "PUT")

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_placement_when_no_captured_marbles_with_empty_pool(self, rings):
        """Test error when placing marble from captured pool but player has none (all board sizes)."""
        board = ZertzBoard(rings=rings)

        # Empty entire supply pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Current player (P1) has no captured white marbles
        board.global_state[board.P1_CAP_W] = 0
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 1

        # Try to place a white marble
        put_loc = 3 * board.config.width + 3
        rem_loc = board.config.width**2
        action = (0, put_loc, rem_loc)  # 0 = white

        with pytest.raises(ValueError, match="No w marbles available in supply or captured by player"):
            board.take_action(action, "PUT")


class TestCaptureActionErrors:
    """Test error handling for invalid capture actions."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_capture_with_no_marble_at_position(self, rings):
        """Test that capturing with no marble at capture position raises ValueError (all board sizes)."""
        board = ZertzBoard(rings=rings)

        # Place marbles to set up an invalid capture scenario
        # Place a white marble at position (3, 3)
        white_layer = board.MARBLE_TO_LAYER["w"]
        board.state[white_layer, 3, 3] = 1

        # Try to capture "through" an empty position
        # Direction 0 is (1, 0) - meaning neighbor is at (y+1, x)
        # This would try to capture the marble at (3+1, 3) = (4, 3)
        # But there's no marble at (4, 3)

        # Convert from capture mask indices to action format using helper
        action = ZertzBoard.capture_indices_to_action(
            direction=0, y=3, x=3, width=board.config.width, directions=board.DIRECTIONS
        )

        with pytest.raises(ValueError, match="Invalid capture: no marble at position"):
            board.take_action(action, "CAP")


class TestBoardEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_index_to_str_for_removed_ring(self, rings):
        """Test that index_to_str returns empty string for removed rings (all board sizes)."""
        board = ZertzBoard(rings=rings)

        # Find a removable ring and remove it
        placement, capture = board.get_valid_moves()
        valid_placements = np.argwhere(placement)

        # Find a placement action that includes a ring removal
        action_with_removal = None
        for action in valid_placements:
            marble_idx, put_loc, rem_loc = action
            if rem_loc != board.config.width**2:  # Has a removal
                action_with_removal = tuple(action)
                break

        if action_with_removal is None:
            pytest.skip("No removable rings found on this board configuration")

        marble_idx, put_loc, rem_loc = action_with_removal

        # Convert rem_loc to (y, x)
        rem_y = rem_loc // board.config.width
        rem_x = rem_loc % board.config.width
        rem_index = (rem_y, rem_x)

        # Take the action to remove the ring
        board.take_action(action_with_removal, "PUT")

        # index_to_str should return empty string for removed ring
        result = board.index_to_str(rem_index)
        assert result == "", f"Expected empty string for removed ring, got '{result}'"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_get_valid_moves_with_captured_marbles(self, rings):
        """Test that placement moves use captured marbles when supply is empty (all board sizes)."""
        board = ZertzBoard(rings=rings)

        # Empty entire supply pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Give current player (P1) some captured marbles
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 0

        # Get valid moves
        placement, capture = board.get_valid_moves()

        # Should have valid placements for white and gray (but not black)
        white_moves = np.any(placement[0])  # White marble type
        gray_moves = np.any(placement[1])   # Gray marble type
        black_moves = np.any(placement[2])  # Black marble type

        assert white_moves, "Should have valid white placements from captured marbles"
        assert gray_moves, "Should have valid gray placements from captured marbles"
        assert not black_moves, "Should not have black placements (none captured)"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_take_placement_action_uses_captured_marble(self, rings):
        """Test that placement action correctly decrements captured marbles when supply empty (all board sizes)."""
        board = ZertzBoard(rings=rings)

        # Empty entire supply pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Give current player (P1) some captured marbles
        initial_captured = 3
        board.global_state[board.P1_CAP_W] = initial_captured

        # Place a white marble
        put_loc = 3 * board.config.width + 3
        rem_loc = board.config.width**2
        action = (0, put_loc, rem_loc)  # 0 = white

        board.take_action(action, "PUT")

        # Verify captured marble count decreased
        assert board.global_state[board.P1_CAP_W] == initial_captured - 1