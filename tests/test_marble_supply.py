"""
Unit tests for marble supply logic.

Tests the rules around when players can/cannot use captured marbles:
- Players MUST use supply marbles when any marble type is available in supply
- Players CAN use captured marbles only when ALL supply marbles are depleted
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


@pytest.fixture
def board():
    """Create a fresh 37-ring board for each test."""
    return ZertzBoard(rings=ZertzBoard.SMALL_BOARD_37)


# ============================================================================
# Marble Supply Tests
# ============================================================================


class TestMarbleSupplyLogic:
    """Test marble supply rules for placement moves."""

    def test_one_marble_type_empty_cannot_use_captured(self, board):
        """Test that players cannot use captured marbles when ANY supply marble is available.

        Per Zèrtz rules: Players must use the general supply first. Only when ALL
        marble types are depleted from the supply can players use their captured marbles.

        Scenario:
        - Supply has: white=0, gray=3, black=5 (white depleted, others available)
        - Player 1 has captured: white=2, gray=1, black=0
        - Expected: Player can only place gray or black (from supply), NOT white (from captured)
        """
        # Set up supply: white depleted, gray and black available
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 3
        board.global_state[board.SUPPLY_B] = 5

        # Set up Player 1 captured marbles: has white available
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 0

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Get placement moves
        moves = board.get_placement_moves()

        # Check that white moves (index 0) are NOT available
        # (even though player has captured white marbles)
        white_moves = moves[0, :, :]
        assert not np.any(white_moves), (
            "White moves should NOT be available when supply has other marbles"
        )

        # Check that gray moves (index 1) ARE available (from supply)
        gray_moves = moves[1, :, :]
        assert np.any(gray_moves), "Gray moves should be available from supply"

        # Check that black moves (index 2) ARE available (from supply)
        black_moves = moves[2, :, :]
        assert np.any(black_moves), "Black moves should be available from supply"

    def test_all_marbles_empty_can_use_captured(self, board):
        """Test that players CAN use captured marbles when ALL supply marbles are depleted.

        Scenario:
        - Supply has: white=0, gray=0, black=0 (all depleted)
        - Player 1 has captured: white=2, gray=1, black=0
        - Expected: Player can place white or gray (from captured), NOT black (none captured)
        """
        # Set up supply: all depleted
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up Player 1 captured marbles
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 0

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Get placement moves
        moves = board.get_placement_moves()

        # Check that white moves (index 0) ARE available (from captured)
        white_moves = moves[0, :, :]
        assert np.any(white_moves), (
            "White moves should be available from captured marbles"
        )

        # Check that gray moves (index 1) ARE available (from captured)
        gray_moves = moves[1, :, :]
        assert np.any(gray_moves), (
            "Gray moves should be available from captured marbles"
        )

        # Check that black moves (index 2) are NOT available (none captured)
        black_moves = moves[2, :, :]
        assert not np.any(black_moves), (
            "Black moves should NOT be available (player has no captured black marbles)"
        )

    def test_supply_to_captured_transition(self, board):
        """Test transition from using supply to using captured marbles.

        Scenario:
        - Start: supply has white=1, gray=0, black=0; Player 1 captured white=2, gray=1, black=1
        - Place white from supply (depletes last supply marble)
        - After: supply has white=0, gray=0, black=0; Player 2's turn
        - Expected before: Can only place white (from supply)
        - Expected after: Player 2 can use their captured marbles (white, gray, or black)
        """
        # Set up initial supply: only one white marble left
        board.global_state[board.SUPPLY_W] = 1
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up Player 1 captured marbles
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 1

        # Set up Player 2 captured marbles
        board.global_state[board.P2_CAP_W] = 1
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 1

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Get placement moves BEFORE transition
        moves_before = board.get_placement_moves()

        # Check that only white is available (from supply)
        white_moves_before = moves_before[0, :, :]
        gray_moves_before = moves_before[1, :, :]
        black_moves_before = moves_before[2, :, :]

        assert np.any(white_moves_before), "White moves should be available from supply"
        assert not np.any(gray_moves_before), (
            "Gray moves should NOT be available (supply empty, captured not allowed yet)"
        )
        assert not np.any(black_moves_before), (
            "Black moves should NOT be available (supply empty, captured not allowed yet)"
        )

        # Simulate placing the last white marble from supply
        board.global_state[board.SUPPLY_W] = 0  # Supply now completely empty

        # Switch to Player 2's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Get placement moves AFTER transition
        moves_after = board.get_placement_moves()

        # Check that Player 2 can use their captured marbles
        white_moves_after = moves_after[0, :, :]
        gray_moves_after = moves_after[1, :, :]
        black_moves_after = moves_after[2, :, :]

        assert np.any(white_moves_after), (
            "White moves should be available from Player 2's captured marbles"
        )
        assert np.any(gray_moves_after), (
            "Gray moves should be available from Player 2's captured marbles"
        )
        assert np.any(black_moves_after), (
            "Black moves should be available from Player 2's captured marbles"
        )

    def test_player_2_uses_correct_captured_pool(self, board):
        """Test that Player 2 uses their own captured pool, not Player 1's.

        Scenario:
        - Supply has: white=0, gray=0, black=0 (all depleted)
        - Player 1 has captured: white=3, gray=0, black=0
        - Player 2 has captured: white=0, gray=2, black=1
        - Expected: Player 2 can place gray or black, NOT white
        """
        # Set up supply: all depleted
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up Player 1 captured marbles
        board.global_state[board.P1_CAP_W] = 3
        board.global_state[board.P1_CAP_G] = 0
        board.global_state[board.P1_CAP_B] = 0

        # Set up Player 2 captured marbles
        board.global_state[board.P2_CAP_W] = 0
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 1

        # Player 2's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Get placement moves
        moves = board.get_placement_moves()

        # Check that white moves are NOT available (Player 2 has no captured white)
        white_moves = moves[0, :, :]
        assert not np.any(white_moves), (
            "White moves should NOT be available (Player 2 has no captured white marbles)"
        )

        # Check that gray moves ARE available (from Player 2's captured)
        gray_moves = moves[1, :, :]
        assert np.any(gray_moves), (
            "Gray moves should be available from Player 2's captured marbles"
        )

        # Check that black moves ARE available (from Player 2's captured)
        black_moves = moves[2, :, :]
        assert np.any(black_moves), (
            "Black moves should be available from Player 2's captured marbles"
        )

    def test_supply_partial_depletion_no_captured_use(self, board):
        """Test that captured marbles cannot be used while any supply marbles remain.

        Scenario:
        - Supply has: white=0, gray=0, black=1 (only black remains)
        - Player 1 has captured: white=1, gray=2, black=0
        - Expected: Player can only place black (from supply), even though they have captured white/gray
        """
        # Set up supply: only black remains
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 1

        # Set up Player 1 captured marbles
        board.global_state[board.P1_CAP_W] = 1
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 0

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Get placement moves
        moves = board.get_placement_moves()

        # Check that only black moves are available
        white_moves = moves[0, :, :]
        gray_moves = moves[1, :, :]
        black_moves = moves[2, :, :]

        assert not np.any(white_moves), (
            "White moves should NOT be available (supply not fully depleted)"
        )
        assert not np.any(gray_moves), (
            "Gray moves should NOT be available (supply not fully depleted)"
        )
        assert np.any(black_moves), "Black moves should be available from supply"
