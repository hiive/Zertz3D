"""
Unit tests for win condition detection.

Tests that the game correctly detects wins in various scenarios:
- Win during chain capture (multi-jump sequence)
- Win immediately after isolated region capture
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame, PLAYER_1_WIN, PLAYER_2_WIN, TIE
from game.zertz_board import ZertzBoard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def game():
    """Create a fresh game for each test."""
    return ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)


# ============================================================================
# Win Condition Tests
# ============================================================================

class TestWinConditions:
    """Test win condition detection in various scenarios."""


    def test_win_during_chain_capture(self):
        """Test that win is detected during a chain capture sequence.

        Scenario:
        - Player 1 has captured: white=2, gray=1, black=5 (needs 1 black for 6-black win)
        - Set up a capture where Player 1 captures a black marble
        - After capture: P1 should have 2/1/6 → WIN! (6 blacks)
        - Game should detect win immediately after the winning capture
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Clear the board to set up our scenario
        board.state[board.MARBLE_LAYERS] = 0
        board.state[board.CAPTURE_LAYER] = 0

        # Set up Player 1's captured marbles (one black away from 6 blacks win)
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 5  # Need 1 more black for win (6 blacks)

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Set up a chain capture scenario:
        # C3 (white) → D4 (black) → E5 (empty) → F5 (black) → G5 (empty)
        # After capturing first black: 5+1=6 blacks → WIN!

        c3_idx = board.str_to_index('C3')
        d4_idx = board.str_to_index('D4')
        e5_idx = board.str_to_index('E5')
        f5_idx = board.str_to_index('F5')

        white_layer = board.MARBLE_TO_LAYER['w']
        black_layer = board.MARBLE_TO_LAYER['b']

        board.state[white_layer][c3_idx] = 1
        board.state[black_layer][d4_idx] = 1
        # E5 is empty
        board.state[black_layer][f5_idx] = 1

        # Verify game is not over yet
        assert game.get_game_ended() is None, "Game should not be over before capture"

        # Execute capture: C3 → D4 → E5 (capture black at D4)
        c3_y, c3_x = c3_idx
        d4_y, d4_x = d4_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (c3_y + dy, c3_x + dx) == (d4_y, d4_x):
                capture_action = (dir_idx, c3_y, c3_x)
                break

        # Take the capture action
        game.take_action('CAP', capture_action)

        # After capture, Player 1 should have 6 blacks → WIN!
        assert board.global_state[board.P1_CAP_B] == 6, "Player 1 should have 6 blacks"

        # Game should be over with Player 1 as winner
        outcome = game.get_game_ended()
        assert outcome == PLAYER_1_WIN, "Player 1 should win with 6 blacks"

    def test_win_after_isolated_region_capture(self):
        """Test that win is detected immediately after isolated region capture.

        Scenario:
        - Player 1 has captured: white=2, gray=2, black=2 (needs 1 of each for 3-of-each win)
        - Create a minimal board with a small isolated region (3 rings, all with marbles)
        - Removing a single connecting ring isolates the region
        - After isolated capture: P1 should have 3/3/3 → WIN!
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Set up Player 1's captured marbles (needs 1 of each for 3-of-each win)
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 2

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Clear the board and set up a minimal topology
        board.state[board.RING_LAYER] = 0

        # Simple topology:
        # Create 2 connected components:
        #   Main: D4-D5-E4 (triangle)
        #   To-isolate: C1-D2-D1 (triangle) connected to main via D3
        # Removing D3 will isolate C1-D2-D1

        d4_idx = board.str_to_index('D4')
        d5_idx = board.str_to_index('D5')
        e4_idx = board.str_to_index('E4')
        d3_idx = board.str_to_index('D3')  # Connection ring (will be removed)
        c1_idx = board.str_to_index('C1')
        d2_idx = board.str_to_index('D2')
        d1_idx = board.str_to_index('D1')

        white_layer = board.MARBLE_TO_LAYER['w']
        gray_layer = board.MARBLE_TO_LAYER['g']
        black_layer = board.MARBLE_TO_LAYER['b']

        # Main region (3 rings)
        board.state[board.RING_LAYER][d4_idx] = 1
        board.state[board.RING_LAYER][d5_idx] = 1
        board.state[board.RING_LAYER][e4_idx] = 1

        # Connection ring
        board.state[board.RING_LAYER][d3_idx] = 1

        # Region to be isolated (3 rings with marbles)
        board.state[board.RING_LAYER][c1_idx] = 1
        board.state[white_layer][c1_idx] = 1  # White marble

        board.state[board.RING_LAYER][d2_idx] = 1
        board.state[gray_layer][d2_idx] = 1  # Gray marble

        board.state[board.RING_LAYER][d1_idx] = 1
        board.state[black_layer][d1_idx] = 1  # Black marble

        # Verify game is not over yet
        assert game.get_game_ended() is None, "Game should not be over before isolation"

        # Player 1 places a marble at E4 and removes D3 (isolating C1-D2-D1)
        e4_y, e4_x = e4_idx
        d3_y, d3_x = d3_idx

        e4_flat = board._2d_to_flat(e4_y, e4_x)
        d3_flat = board._2d_to_flat(d3_y, d3_x)

        # Use white marble for placement (index 0)
        placement_action = (0, e4_flat, d3_flat)

        # Take the placement action
        game.take_action('PUT', placement_action)

        # After isolation, Player 1 should have captured the 3 isolated marbles
        # P1 should now have 3/3/3 → WIN!
        assert board.global_state[board.P1_CAP_W] == 3, "Player 1 should have 3 whites"
        assert board.global_state[board.P1_CAP_G] == 3, "Player 1 should have 3 grays"
        assert board.global_state[board.P1_CAP_B] == 3, "Player 1 should have 3 blacks"

        # Game should be over with Player 1 as winner
        outcome = game.get_game_ended()
        assert outcome == PLAYER_1_WIN, "Player 1 should win with 3 of each color"