"""
Unit tests for BOTH_LOSE condition (tournament collaboration rule).

Tests the tournament rule (FAQ #3): If all marbles are placed on the board
and neither player has captured any marbles, both players lose.

This is a unique outcome (BOTH_LOSE = -2) distinct from:
- PLAYER_1_WIN (1)
- PLAYER_2_WIN (-1)
- TIE (0)
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame, PLAYER_1_WIN, PLAYER_2_WIN, BOTH_LOSE
from game.zertz_board import ZertzBoard
from game.zertz_logic import BoardConfig, get_game_outcome


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def game():
    """Create a fresh game for each test."""
    return ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)


# ============================================================================
# BOTH_LOSE Condition Tests (Stateful Game)
# ============================================================================


class TestBothLoseCondition:
    """Test tournament collaboration rule detection."""

    def test_basic_both_lose_detection(self):
        """Test that BOTH_LOSE is detected when board fills with no captures.

        Scenario:
        - Fill all 37 rings with marbles
        - Neither player has captured any marbles
        - Expected outcome: BOTH_LOSE (-2)
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Fill the entire board with marbles (no captures)
        # We need to place marbles on all 37 positions
        marble_types = ['w', 'g', 'b']
        marble_idx = 0

        # Get all valid ring positions
        ring_positions = list(zip(*np.where(game.board.state[game.board.RING_LAYER] == 1)))

        # Place marbles on all rings without removing any rings
        for y, x in ring_positions:
            marble_type = marble_types[marble_idx % 3]
            marble_layer = game.board.MARBLE_TO_LAYER[marble_type]

            # Directly place marble (bypass normal placement rules for test setup)
            game.board.state[marble_layer, y, x] = 1

            marble_idx += 1

        # Verify board is completely full
        board_full = np.all(np.sum(game.board.state[game.board.BOARD_LAYERS], axis=0) != 1)
        assert board_full, "Board should be completely filled with marbles"

        # Verify no captures for either player
        p1_captured = game.board.global_state[game.board.P1_CAP_SLICE]
        p2_captured = game.board.global_state[game.board.P2_CAP_SLICE]
        assert np.all(p1_captured == 0), "Player 1 should have no captures"
        assert np.all(p2_captured == 0), "Player 2 should have no captures"

        # Check outcome
        outcome = game.get_game_ended()
        assert outcome == BOTH_LOSE, f"Expected BOTH_LOSE (-2), got {outcome}"

    def test_both_lose_end_reason_message(self):
        """Test that get_game_end_reason() returns correct message for BOTH_LOSE."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Fill board with no captures
        marble_types = ['w', 'g', 'b']
        marble_idx = 0
        ring_positions = list(zip(*np.where(game.board.state[game.board.RING_LAYER] == 1)))

        for y, x in ring_positions:
            marble_type = marble_types[marble_idx % 3]
            marble_layer = game.board.MARBLE_TO_LAYER[marble_type]
            game.board.state[marble_layer, y, x] = 1
            marble_idx += 1

        # Get end reason
        reason = game.get_game_end_reason()

        # Verify message mentions collaboration
        assert reason is not None, "End reason should not be None"
        assert "collaboration" in reason.lower(), f"Message should mention 'collaboration': {reason}"
        assert "both players lose" in reason.lower() or "both lose" in reason.lower(), \
            f"Message should indicate both players lose: {reason}"

    def test_normal_full_board_win_not_both_lose(self):
        """Test that full board with captures is normal win, not BOTH_LOSE.

        Scenario:
        - Fill all rings
        - Player 1 has 1 captured marble
        - Expected: Normal full board win (last player wins), NOT BOTH_LOSE
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Give Player 1 one captured marble
        game.board.global_state[game.board.P1_CAP_W] = 1

        # Fill the board
        marble_types = ['w', 'g', 'b']
        marble_idx = 0
        ring_positions = list(zip(*np.where(game.board.state[game.board.RING_LAYER] == 1)))

        for y, x in ring_positions:
            marble_type = marble_types[marble_idx % 3]
            marble_layer = game.board.MARBLE_TO_LAYER[marble_type]
            game.board.state[marble_layer, y, x] = 1
            marble_idx += 1

        # Verify board is full
        board_full = np.all(np.sum(game.board.state[game.board.BOARD_LAYERS], axis=0) != 1)
        assert board_full, "Board should be completely filled"

        # Check outcome - should NOT be BOTH_LOSE
        outcome = game.get_game_ended()
        assert outcome != BOTH_LOSE, "Should not be BOTH_LOSE when captures exist"
        assert outcome in [PLAYER_1_WIN, PLAYER_2_WIN], \
            f"Should be normal win (1 or -1), got {outcome}"

    def test_both_lose_edge_case_one_player_has_capture(self):
        """Test edge case: only one player has captures.

        When board fills:
        - Player 1: 1 white marble captured
        - Player 2: 0 captures
        - Expected: Normal win for last player, NOT BOTH_LOSE

        The collaboration rule requires BOTH players to have zero captures.
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Player 1 gets 1 capture, Player 2 has none
        game.board.global_state[game.board.P1_CAP_W] = 1

        # Fill board
        marble_types = ['w', 'g', 'b']
        marble_idx = 0
        ring_positions = list(zip(*np.where(game.board.state[game.board.RING_LAYER] == 1)))

        for y, x in ring_positions:
            marble_type = marble_types[marble_idx % 3]
            marble_layer = game.board.MARBLE_TO_LAYER[marble_type]
            game.board.state[marble_layer, y, x] = 1
            marble_idx += 1

        outcome = game.get_game_ended()

        # Should be normal win, not BOTH_LOSE
        assert outcome != BOTH_LOSE, \
            "Should not be BOTH_LOSE when one player has captures"

    def test_blitz_variant_both_lose(self):
        """Test BOTH_LOSE detection in Blitz variant.

        Blitz uses fewer marbles (5W, 7G, 9B) but same board size.
        The collaboration rule should still apply.
        """
        # Create Blitz game
        blitz_marbles = {"w": 5, "g": 7, "b": 9}
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37, marbles=blitz_marbles)

        # Fill the board
        marble_types = ['w', 'g', 'b']
        marble_idx = 0
        ring_positions = list(zip(*np.where(game.board.state[game.board.RING_LAYER] == 1)))

        for y, x in ring_positions:
            marble_type = marble_types[marble_idx % 3]
            marble_layer = game.board.MARBLE_TO_LAYER[marble_type]
            game.board.state[marble_layer, y, x] = 1
            marble_idx += 1

        # Verify no captures
        p1_captured = game.board.global_state[game.board.P1_CAP_SLICE]
        p2_captured = game.board.global_state[game.board.P2_CAP_SLICE]
        assert np.all(p1_captured == 0), "Player 1 should have no captures"
        assert np.all(p2_captured == 0), "Player 2 should have no captures"

        # Check outcome
        outcome = game.get_game_ended()
        assert outcome == BOTH_LOSE, \
            f"Blitz variant should also detect BOTH_LOSE, got {outcome}"


# ============================================================================
# BOTH_LOSE Condition Tests (Stateless Logic)
# ============================================================================


class TestBothLoseStateless:
    """Test BOTH_LOSE detection in stateless game logic."""

    def test_stateless_both_lose_detection(self):
        """Test get_game_outcome() returns BOTH_LOSE for full board with no captures."""
        config = BoardConfig.standard_config(rings=37)

        # Create a board state with all positions filled
        width = config.width
        board_state = np.zeros((config.t * 4 + 1, width, width), dtype=np.float32)

        # Set all rings to occupied
        board_state[config.ring_layer] = 1

        # Fill with marbles (alternating pattern)
        marble_types = [1, 2, 3]  # w, g, b layer indices
        idx = 0
        for y in range(width):
            for x in range(width):
                if board_state[config.ring_layer, y, x] == 1:
                    marble_layer = marble_types[idx % 3]
                    board_state[marble_layer, y, x] = 1
                    idx += 1

        # Create global state with no captures
        global_state = np.array([0, 0, 0,  # supply (empty)
                                0, 0, 0,  # P1 captures (all zero)
                                0, 0, 0,  # P2 captures (all zero)
                                0],       # current player
                               dtype=np.float32)

        # Call stateless function
        outcome = get_game_outcome(board_state, global_state, config)

        assert outcome == BOTH_LOSE, \
            f"Stateless function should return BOTH_LOSE (-2), got {outcome}"

    def test_stateless_normal_win_with_captures(self):
        """Test that stateless function returns normal win when captures exist."""
        config = BoardConfig.standard_config(rings=37)

        # Create full board
        width = config.width
        board_state = np.zeros((config.t * 4 + 1, width, width), dtype=np.float32)
        board_state[config.ring_layer] = 1

        marble_types = [1, 2, 3]
        idx = 0
        for y in range(width):
            for x in range(width):
                if board_state[config.ring_layer, y, x] == 1:
                    board_state[marble_types[idx % 3], y, x] = 1
                    idx += 1

        # Give Player 1 some captures
        global_state = np.array([0, 0, 0,  # supply
                                2, 1, 0,  # P1 captures (2 white, 1 grey)
                                0, 0, 0,  # P2 captures
                                0],       # current player (P1)
                               dtype=np.float32)

        outcome = get_game_outcome(board_state, global_state, config)

        # Should not be BOTH_LOSE
        assert outcome != BOTH_LOSE, \
            "Should not be BOTH_LOSE when captures exist"

        # Should be normal win (current player loses, so P2 wins)
        assert outcome in [PLAYER_1_WIN, PLAYER_2_WIN], \
            f"Should return normal win value, got {outcome}"

    def test_stateless_48_ring_both_lose(self):
        """Test BOTH_LOSE detection on 48-ring board."""
        config = BoardConfig.standard_config(rings=48)

        # Create full board
        width = config.width
        board_state = np.zeros((config.t * 4 + 1, width, width), dtype=np.float32)
        board_state[config.ring_layer] = 1

        marble_types = [1, 2, 3]
        idx = 0
        for y in range(width):
            for x in range(width):
                if board_state[config.ring_layer, y, x] == 1:
                    board_state[marble_types[idx % 3], y, x] = 1
                    idx += 1

        # No captures
        global_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        outcome = get_game_outcome(board_state, global_state, config)

        assert outcome == BOTH_LOSE, \
            f"48-ring board should also detect BOTH_LOSE, got {outcome}"

    def test_stateless_61_ring_both_lose(self):
        """Test BOTH_LOSE detection on 61-ring board."""
        config = BoardConfig.standard_config(rings=61)

        # Create full board
        width = config.width
        board_state = np.zeros((config.t * 4 + 1, width, width), dtype=np.float32)
        board_state[config.ring_layer] = 1

        marble_types = [1, 2, 3]
        idx = 0
        for y in range(width):
            for x in range(width):
                if board_state[config.ring_layer, y, x] == 1:
                    board_state[marble_types[idx % 3], y, x] = 1
                    idx += 1

        # No captures
        global_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        outcome = get_game_outcome(board_state, global_state, config)

        assert outcome == BOTH_LOSE, \
            f"61-ring board should also detect BOTH_LOSE, got {outcome}"