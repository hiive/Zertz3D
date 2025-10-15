"""
Unit tests for win condition detection.

Tests that the game correctly detects all game outcomes:
- Player 1 wins (via chain capture and isolated region capture)
- Player 2 wins (via chain capture and isolated region capture)
- Ties (via immobilization with equal captures and loop detection)
"""

import pytest
import sys
from pathlib import Path
import numpy as np

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

        c3_idx = board.str_to_index("C3")
        d4_idx = board.str_to_index("D4")
        e5_idx = board.str_to_index("E5")
        f5_idx = board.str_to_index("F5")

        white_layer = board.MARBLE_TO_LAYER["w"]
        black_layer = board.MARBLE_TO_LAYER["b"]

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
        game.take_action("CAP", capture_action)

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

        d4_idx = board.str_to_index("D4")
        d5_idx = board.str_to_index("D5")
        e4_idx = board.str_to_index("E4")
        d3_idx = board.str_to_index("D3")  # Connection ring (will be removed)
        c1_idx = board.str_to_index("C1")
        d2_idx = board.str_to_index("D2")
        d1_idx = board.str_to_index("D1")

        white_layer = board.MARBLE_TO_LAYER["w"]
        gray_layer = board.MARBLE_TO_LAYER["g"]
        black_layer = board.MARBLE_TO_LAYER["b"]

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
        game.take_action("PUT", placement_action)

        # After isolation, Player 1 should have captured the 3 isolated marbles
        # P1 should now have 3/3/3 → WIN!
        assert board.global_state[board.P1_CAP_W] == 3, "Player 1 should have 3 whites"
        assert board.global_state[board.P1_CAP_G] == 3, "Player 1 should have 3 grays"
        assert board.global_state[board.P1_CAP_B] == 3, "Player 1 should have 3 blacks"

        # Game should be over with Player 1 as winner
        outcome = game.get_game_ended()
        assert outcome == PLAYER_1_WIN, "Player 1 should win with 3 of each color"

    def test_player2_win_during_chain_capture(self):
        """Test that Player 2 win is detected during a chain capture sequence.

        Scenario:
        - Player 2 has captured: white=1, gray=4, black=2 (needs 1 gray for 5-gray win)
        - Set up a capture where Player 2 captures a gray marble
        - After capture: P2 should have 1/5/2 → WIN! (5 grays)
        - Game should detect win immediately after the winning capture
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Clear the board to set up our scenario (same order as test_win_during_chain_capture)
        board.state[board.MARBLE_LAYERS] = 0
        board.state[board.CAPTURE_LAYER] = 0

        # Set up Player 2's captured marbles (one gray away from 5 grays win)
        # Note: Must not meet the 3-of-each win condition!
        board.global_state[board.P2_CAP_W] = 1
        board.global_state[board.P2_CAP_G] = 4  # Need 1 more gray for win (5 grays)
        board.global_state[board.P2_CAP_B] = 2

        # Player 2's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Set up a chain capture scenario:
        # C3 (white) → D4 (gray) → E5 (empty)
        # After capturing gray: 4+1=5 grays → WIN!

        c3_idx = board.str_to_index("C3")
        d4_idx = board.str_to_index("D4")
        e5_idx = board.str_to_index("E5")

        white_layer = board.MARBLE_TO_LAYER["w"]
        gray_layer = board.MARBLE_TO_LAYER["g"]

        board.state[white_layer][c3_idx] = 1
        board.state[gray_layer][d4_idx] = 1
        # E5 is empty

        # Verify game is not over yet
        assert game.get_game_ended() is None, "Game should not be over before capture"

        # Execute capture: C3 → D4 → E5 (capture gray at D4)
        c3_y, c3_x = c3_idx
        d4_y, d4_x = d4_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (c3_y + dy, c3_x + dx) == (d4_y, d4_x):
                capture_action = (dir_idx, c3_y, c3_x)
                break

        # Take the capture action
        game.take_action("CAP", capture_action)

        # After capture, Player 2 should have 5 grays → WIN!
        assert board.global_state[board.P2_CAP_G] == 5, "Player 2 should have 5 grays"

        # Game should be over with Player 2 as winner
        outcome = game.get_game_ended()
        assert outcome == PLAYER_2_WIN, "Player 2 should win with 5 grays"

    def test_player2_win_after_isolated_region_capture(self):
        """Test that Player 2 win is detected after isolated region capture.

        Scenario:
        - Player 2 has captured: white=3, gray=3, black=2 (needs 1 black for 3-of-each win)
        - Create a minimal board with a small isolated region containing a black marble
        - After isolated capture: P2 should have 3/3/3 → WIN!
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Set up Player 2's captured marbles (needs 1 black for 3-of-each win)
        board.global_state[board.P2_CAP_W] = 3
        board.global_state[board.P2_CAP_G] = 3
        board.global_state[board.P2_CAP_B] = 2  # Need 1 more black for win

        # Player 2's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Clear the board and set up a minimal topology
        board.state[board.RING_LAYER] = 0

        # Simple topology:
        # Main: D4-D5-E4 (triangle)
        # To-isolate: C1 (single ring with black marble) connected via D3
        # Removing D3 will isolate C1

        d4_idx = board.str_to_index("D4")
        d5_idx = board.str_to_index("D5")
        e4_idx = board.str_to_index("E4")
        d3_idx = board.str_to_index("D3")  # Connection ring (will be removed)
        c1_idx = board.str_to_index("C1")

        black_layer = board.MARBLE_TO_LAYER["b"]

        # Main region (3 rings)
        board.state[board.RING_LAYER][d4_idx] = 1
        board.state[board.RING_LAYER][d5_idx] = 1
        board.state[board.RING_LAYER][e4_idx] = 1

        # Connection ring
        board.state[board.RING_LAYER][d3_idx] = 1

        # Region to be isolated (1 ring with black marble)
        board.state[board.RING_LAYER][c1_idx] = 1
        board.state[black_layer][c1_idx] = 1

        # Verify game is not over yet
        assert game.get_game_ended() is None, "Game should not be over before isolation"

        # Player 2 places a marble at E4 and removes D3 (isolating C1)
        e4_y, e4_x = e4_idx
        d3_y, d3_x = d3_idx

        e4_flat = board._2d_to_flat(e4_y, e4_x)
        d3_flat = board._2d_to_flat(d3_y, d3_x)

        # Use white marble for placement (index 0)
        placement_action = (0, e4_flat, d3_flat)

        # Take the placement action
        game.take_action("PUT", placement_action)

        # After isolation, Player 2 should have captured the isolated black marble
        # P2 should now have 3/3/3 → WIN!
        assert board.global_state[board.P2_CAP_W] == 3, "Player 2 should have 3 whites"
        assert board.global_state[board.P2_CAP_G] == 3, "Player 2 should have 3 grays"
        assert board.global_state[board.P2_CAP_B] == 3, "Player 2 should have 3 blacks"

        # Game should be over with Player 2 as winner
        outcome = game.get_game_ended()
        assert outcome == PLAYER_2_WIN, "Player 2 should win with 3 of each color"

    def test_tie_via_immobilization_equal_captures(self):
        """Test that tie is detected when both players are immobilized with equal captures.

        Scenario:
        - Both players have equal captured marbles: P1=2/2/2, P2=2/2/2
        - Both players have no marbles left in pool to place
        - Board is full (all rings have marbles)
        - Both players pass consecutively
        - Should result in TIE
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Empty the marble pool so players can't place new marbles
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up equal captured marbles (these count for tie-breaking but can't be placed)
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 2
        board.global_state[board.P2_CAP_W] = 2
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 2

        # Fill all rings with marbles so no placement is possible
        # Even if players have captured marbles, they can't place them
        white_layer = board.MARBLE_TO_LAYER["w"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[white_layer][y, x] = 1

        # Player 1's turn - should pass (no valid moves)
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1
        placement, capture = game.get_valid_actions()
        assert not np.any(placement) and not np.any(capture), (
            "Player 1 should have no valid moves"
        )

        # Player 1 passes
        game.take_action("PASS", None)

        # Now Player 2's turn - should also pass (no valid moves)
        assert board.global_state[board.CUR_PLAYER] == board.PLAYER_2
        placement, capture = game.get_valid_actions()
        assert not np.any(placement) and not np.any(capture), (
            "Player 2 should have no valid moves"
        )

        # Player 2 passes
        game.take_action("PASS", None)

        # After both players pass, game should be tie (equal captures)
        outcome = game.get_game_ended()
        assert outcome == TIE, "Game should end in tie with equal captures"

    def test_tie_via_loop_detection(self):
        """Test that tie is detected when move sequence loops.

        Scenario:
        - Directly create a move history with repeating patterns
        - Loop detection should trigger a tie when last k pairs match preceding k pairs
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Create a repeating move pattern by directly manipulating move_history
        # For k=2, we need 8 moves total (2 pairs repeated twice)
        # Pattern: (action_type, action) tuples
        move_pattern = [
            ("PUT", (0, 10, 5)),  # P1 move
            ("PUT", (1, 15, 8)),  # P2 move
            ("PUT", (0, 12, 7)),  # P1 move
            ("PUT", (1, 18, 11)),  # P2 move
        ]

        # Add the pattern twice to create a loop
        game.move_history = move_pattern + move_pattern

        # Verify loop is detected
        assert game._has_move_loop(), "Loop detection should identify repeating pattern"

        # Verify game returns TIE when loop is detected
        outcome = game.get_game_ended()
        assert outcome == TIE, "Game should end in tie when loop is detected"

    def test_player1_win_4_white_marbles(self):
        """Test that Player 1 wins by capturing 4 white marbles.

        Scenario:
        - Player 1 has captured: white=3, gray=2, black=1 (needs 1 white for 4-white win)
        - Set up a capture where Player 1 captures a white marble
        - After capture: P1 should have 4/2/1 → WIN! (4 whites)
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Clear the board to set up our scenario
        board.state[board.MARBLE_LAYERS] = 0
        board.state[board.CAPTURE_LAYER] = 0

        # Set up Player 1's captured marbles (one white away from 4 whites win)
        board.global_state[board.P1_CAP_W] = 3
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 1

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Set up a capture scenario: B2 (gray) → C3 (white) → D4 (empty)
        b2_idx = board.str_to_index("B2")
        c3_idx = board.str_to_index("C3")
        d4_idx = board.str_to_index("D4")

        gray_layer = board.MARBLE_TO_LAYER["g"]
        white_layer = board.MARBLE_TO_LAYER["w"]

        board.state[gray_layer][b2_idx] = 1
        board.state[white_layer][c3_idx] = 1
        # D4 is empty

        # Verify game is not over yet
        assert game.get_game_ended() is None, "Game should not be over before capture"

        # Execute capture: B2 → C3 → D4 (capture white at C3)
        b2_y, b2_x = b2_idx
        c3_y, c3_x = c3_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (b2_y + dy, b2_x + dx) == (c3_y, c3_x):
                capture_action = (dir_idx, b2_y, b2_x)
                break

        # Take the capture action
        game.take_action("CAP", capture_action)

        # After capture, Player 1 should have 4 whites → WIN!
        assert board.global_state[board.P1_CAP_W] == 4, "Player 1 should have 4 whites"

        # Game should be over with Player 1 as winner
        outcome = game.get_game_ended()
        assert outcome == PLAYER_1_WIN, "Player 1 should win with 4 whites"

    def test_player2_win_4_white_marbles(self):
        """Test that Player 2 wins by capturing 4 white marbles.

        Scenario:
        - Player 2 has captured: white=3, gray=1, black=2 (needs 1 white for 4-white win)
        - Set up a capture where Player 2 captures a white marble
        - After capture: P2 should have 4/1/2 → WIN! (4 whites)
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Clear the board to set up our scenario
        board.state[board.MARBLE_LAYERS] = 0
        board.state[board.CAPTURE_LAYER] = 0

        # Set up Player 2's captured marbles (one white away from 4 whites win)
        board.global_state[board.P2_CAP_W] = 3
        board.global_state[board.P2_CAP_G] = 1
        board.global_state[board.P2_CAP_B] = 2

        # Player 2's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Set up a capture scenario: B2 (black) → C3 (white) → D4 (empty)
        b2_idx = board.str_to_index("B2")
        c3_idx = board.str_to_index("C3")
        d4_idx = board.str_to_index("D4")

        black_layer = board.MARBLE_TO_LAYER["b"]
        white_layer = board.MARBLE_TO_LAYER["w"]

        board.state[black_layer][b2_idx] = 1
        board.state[white_layer][c3_idx] = 1
        # D4 is empty

        # Verify game is not over yet
        assert game.get_game_ended() is None, "Game should not be over before capture"

        # Execute capture: B2 → C3 → D4 (capture white at C3)
        b2_y, b2_x = b2_idx
        c3_y, c3_x = c3_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (b2_y + dy, b2_x + dx) == (c3_y, c3_x):
                capture_action = (dir_idx, b2_y, b2_x)
                break

        # Take the capture action
        game.take_action("CAP", capture_action)

        # After capture, Player 2 should have 4 whites → WIN!
        assert board.global_state[board.P2_CAP_W] == 4, "Player 2 should have 4 whites"

        # Game should be over with Player 2 as winner
        outcome = game.get_game_ended()
        assert outcome == PLAYER_2_WIN, "Player 2 should win with 4 whites"

    def test_board_full_player1_wins(self):
        """Test that Player 1 wins when they fill the last ring on the board.

        Scenario:
        - Board has only one empty ring left
        - Neither player has a win condition yet
        - Player 1 places a marble on the last ring
        - Player 1 should win (last player to move when board is full)
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Set up captured marbles (neither has win condition)
        board.global_state[board.P1_CAP_W] = 1
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 1
        board.global_state[board.P2_CAP_W] = 2
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 2

        # Clear the board and set up a minimal topology with only 3 rings
        board.state[board.RING_LAYER] = 0

        d3_idx = board.str_to_index("D3")
        d4_idx = board.str_to_index("D4")
        d5_idx = board.str_to_index("D5")

        board.state[board.RING_LAYER][d3_idx] = 1
        board.state[board.RING_LAYER][d4_idx] = 1
        board.state[board.RING_LAYER][d5_idx] = 1

        # Fill two rings with marbles, leave one empty
        white_layer = board.MARBLE_TO_LAYER["w"]
        gray_layer = board.MARBLE_TO_LAYER["g"]
        board.state[white_layer][d3_idx] = 1
        board.state[gray_layer][d4_idx] = 1
        # D5 is empty

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Verify game is not over yet
        assert game.get_game_ended() is None, (
            "Game should not be over before board is full"
        )

        # Player 1 places a marble on the last empty ring (D5)
        d5_y, d5_x = d5_idx
        d5_flat = board._2d_to_flat(d5_y, d5_x)

        # Use black marble for placement (index 2), no removal (board is too small)
        placement_action = (2, d5_flat, board.width**2)
        game.take_action("PUT", placement_action)

        # Board should now be full
        assert np.all(np.sum(board.state[board.BOARD_LAYERS], axis=0) != 1), (
            "Board should be full"
        )

        # Game should be over with Player 1 as winner (last to move)
        outcome = game.get_game_ended()
        assert outcome == PLAYER_1_WIN, "Player 1 should win by filling the board"

    def test_board_full_player2_wins(self):
        """Test that Player 2 wins when they fill the last ring on the board.

        Scenario:
        - Board has only one empty ring left
        - Neither player has a win condition yet
        - Player 2 places a marble on the last ring
        - Player 2 should win (last player to move when board is full)
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Set up captured marbles (neither has win condition)
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 1
        board.global_state[board.P2_CAP_W] = 1
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 1

        # Clear the board and set up a minimal topology with only 3 rings
        board.state[board.RING_LAYER] = 0

        d3_idx = board.str_to_index("D3")
        d4_idx = board.str_to_index("D4")
        d5_idx = board.str_to_index("D5")

        board.state[board.RING_LAYER][d3_idx] = 1
        board.state[board.RING_LAYER][d4_idx] = 1
        board.state[board.RING_LAYER][d5_idx] = 1

        # Fill two rings with marbles, leave one empty
        white_layer = board.MARBLE_TO_LAYER["w"]
        black_layer = board.MARBLE_TO_LAYER["b"]
        board.state[white_layer][d3_idx] = 1
        board.state[black_layer][d4_idx] = 1
        # D5 is empty

        # Player 2's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Verify game is not over yet
        assert game.get_game_ended() is None, (
            "Game should not be over before board is full"
        )

        # Player 2 places a marble on the last empty ring (D5)
        d5_y, d5_x = d5_idx
        d5_flat = board._2d_to_flat(d5_y, d5_x)

        # Use gray marble for placement (index 1), no removal
        placement_action = (1, d5_flat, board.width**2)
        game.take_action("PUT", placement_action)

        # Board should now be full
        assert np.all(np.sum(board.state[board.BOARD_LAYERS], axis=0) != 1), (
            "Board should be full"
        )

        # Game should be over with Player 2 as winner (last to move)
        outcome = game.get_game_ended()
        assert outcome == PLAYER_2_WIN, "Player 2 should win by filling the board"

    def test_both_immobilized_player1_has_win_condition(self):
        """Test that Player 1 wins when both are immobilized and P1 has win condition.

        Scenario:
        - Player 1 has 3/3/3 (meets 3-of-each win condition)
        - Player 2 has 2/2/2 (no win condition)
        - Both players pass consecutively
        - Player 1 should win (not tie)
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Empty the marble pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up captured marbles: P1 has win condition, P2 does not
        board.global_state[board.P1_CAP_W] = 3
        board.global_state[board.P1_CAP_G] = 3
        board.global_state[board.P1_CAP_B] = 3
        board.global_state[board.P2_CAP_W] = 2
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 2

        # Fill all rings with marbles so no placement is possible
        white_layer = board.MARBLE_TO_LAYER["w"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[white_layer][y, x] = 1

        # Player 1's turn - should pass (no valid moves)
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1
        game.take_action("PASS", None)

        # Player 2's turn - should also pass (no valid moves)
        game.take_action("PASS", None)

        # After both players pass, Player 1 should win (has 3-of-each)
        outcome = game.get_game_ended()
        assert outcome == PLAYER_1_WIN, (
            "Player 1 should win with 3-of-each when both immobilized"
        )

    def test_both_immobilized_player2_has_win_condition(self):
        """Test that Player 2 wins when both are immobilized and P2 has win condition.

        Scenario:
        - Player 1 has 2/1/2 (no win condition)
        - Player 2 has 4/2/1 (meets 4-white win condition)
        - Both players pass consecutively
        - Player 2 should win (not tie)
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Empty the marble pool
        board.global_state[board.SUPPLY_W] = 0
        board.global_state[board.SUPPLY_G] = 0
        board.global_state[board.SUPPLY_B] = 0

        # Set up captured marbles: P2 has win condition, P1 does not
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 1
        board.global_state[board.P1_CAP_B] = 2
        board.global_state[board.P2_CAP_W] = 4
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 1

        # Fill all rings with marbles so no placement is possible
        gray_layer = board.MARBLE_TO_LAYER["g"]
        ring_positions = np.argwhere(board.state[board.RING_LAYER] == 1)
        for y, x in ring_positions:
            board.state[gray_layer][y, x] = 1

        # Player 1's turn - should pass (no valid moves)
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1
        game.take_action("PASS", None)

        # Player 2's turn - should also pass (no valid moves)
        game.take_action("PASS", None)

        # After both players pass, Player 2 should win (has 4 whites)
        outcome = game.get_game_ended()
        assert outcome == PLAYER_2_WIN, (
            "Player 2 should win with 4 whites when both immobilized"
        )


# ============================================================================
# Blitz Mode Win Condition Tests
# ============================================================================


class TestBlitzWinConditions:
    """Test win condition detection in Blitz variant.

    Blitz uses different win conditions:
    - 2 marbles of each color, or
    - 3 white, or
    - 4 grey, or
    - 5 black marbles
    """

    def test_blitz_player1_win_2_of_each(self):
        """Test that Player 1 wins with 2 of each color in Blitz mode."""
        from game.zertz_game import BLITZ_WIN_CONDITIONS, BLITZ_MARBLES

        game = ZertzGame(
            rings=ZertzBoard.SMALL_BOARD_37,
            marbles=BLITZ_MARBLES,
            win_con=BLITZ_WIN_CONDITIONS,
        )
        board = game.board

        # Clear the board
        board.state[board.MARBLE_LAYERS] = 0
        board.state[board.CAPTURE_LAYER] = 0

        # Set up Player 1 with 2-of-each (one black away from win)
        board.global_state[board.P1_CAP_W] = 2
        board.global_state[board.P1_CAP_G] = 2
        board.global_state[board.P1_CAP_B] = 1

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Set up capture scenario
        b2_idx = board.str_to_index("B2")
        c3_idx = board.str_to_index("C3")
        d4_idx = board.str_to_index("D4")

        white_layer = board.MARBLE_TO_LAYER["w"]
        black_layer = board.MARBLE_TO_LAYER["b"]

        board.state[white_layer][b2_idx] = 1
        board.state[black_layer][c3_idx] = 1

        # Execute capture
        b2_y, b2_x = b2_idx
        c3_y, c3_x = c3_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (b2_y + dy, b2_x + dx) == (c3_y, c3_x):
                capture_action = (dir_idx, b2_y, b2_x)
                break

        game.take_action("CAP", capture_action)

        # Player 1 should have 2/2/2 → WIN in Blitz!
        assert board.global_state[board.P1_CAP_B] == 2
        outcome = game.get_game_ended()
        assert outcome == PLAYER_1_WIN, (
            "Player 1 should win with 2-of-each in Blitz mode"
        )

    def test_blitz_player2_win_3_white(self):
        """Test that Player 2 wins with 3 white marbles in Blitz mode."""
        from game.zertz_game import BLITZ_WIN_CONDITIONS, BLITZ_MARBLES

        game = ZertzGame(
            rings=ZertzBoard.SMALL_BOARD_37,
            marbles=BLITZ_MARBLES,
            win_con=BLITZ_WIN_CONDITIONS,
        )
        board = game.board

        # Clear the board
        board.state[board.MARBLE_LAYERS] = 0
        board.state[board.CAPTURE_LAYER] = 0

        # Set up Player 2 with 2 whites (one away from 3-white win)
        board.global_state[board.P2_CAP_W] = 2
        board.global_state[board.P2_CAP_G] = 1
        board.global_state[board.P2_CAP_B] = 0

        # Player 2's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Set up capture scenario
        b2_idx = board.str_to_index("B2")
        c3_idx = board.str_to_index("C3")

        gray_layer = board.MARBLE_TO_LAYER["g"]
        white_layer = board.MARBLE_TO_LAYER["w"]

        board.state[gray_layer][b2_idx] = 1
        board.state[white_layer][c3_idx] = 1

        # Execute capture
        b2_y, b2_x = b2_idx
        c3_y, c3_x = c3_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (b2_y + dy, b2_x + dx) == (c3_y, c3_x):
                capture_action = (dir_idx, b2_y, b2_x)
                break

        game.take_action("CAP", capture_action)

        # Player 2 should have 3 whites → WIN in Blitz!
        assert board.global_state[board.P2_CAP_W] == 3
        outcome = game.get_game_ended()
        assert outcome == PLAYER_2_WIN, (
            "Player 2 should win with 3 whites in Blitz mode"
        )

    def test_blitz_player1_win_4_grey(self):
        """Test that Player 1 wins with 4 grey marbles in Blitz mode."""
        from game.zertz_game import BLITZ_WIN_CONDITIONS, BLITZ_MARBLES

        game = ZertzGame(
            rings=ZertzBoard.SMALL_BOARD_37,
            marbles=BLITZ_MARBLES,
            win_con=BLITZ_WIN_CONDITIONS,
        )
        board = game.board

        # Clear the board
        board.state[board.MARBLE_LAYERS] = 0
        board.state[board.CAPTURE_LAYER] = 0

        # Set up Player 1 with 3 greys (one away from 4-grey win)
        board.global_state[board.P1_CAP_W] = 1
        board.global_state[board.P1_CAP_G] = 3
        board.global_state[board.P1_CAP_B] = 1

        # Player 1's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_1

        # Set up capture scenario
        b2_idx = board.str_to_index("B2")
        c3_idx = board.str_to_index("C3")

        white_layer = board.MARBLE_TO_LAYER["w"]
        gray_layer = board.MARBLE_TO_LAYER["g"]

        board.state[white_layer][b2_idx] = 1
        board.state[gray_layer][c3_idx] = 1

        # Execute capture
        b2_y, b2_x = b2_idx
        c3_y, c3_x = c3_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (b2_y + dy, b2_x + dx) == (c3_y, c3_x):
                capture_action = (dir_idx, b2_y, b2_x)
                break

        game.take_action("CAP", capture_action)

        # Player 1 should have 4 greys → WIN in Blitz!
        assert board.global_state[board.P1_CAP_G] == 4
        outcome = game.get_game_ended()
        assert outcome == PLAYER_1_WIN, "Player 1 should win with 4 greys in Blitz mode"

    def test_blitz_player2_win_5_black(self):
        """Test that Player 2 wins with 5 black marbles in Blitz mode."""
        from game.zertz_game import BLITZ_WIN_CONDITIONS, BLITZ_MARBLES

        game = ZertzGame(
            rings=ZertzBoard.SMALL_BOARD_37,
            marbles=BLITZ_MARBLES,
            win_con=BLITZ_WIN_CONDITIONS,
        )
        board = game.board

        # Clear the board
        board.state[board.MARBLE_LAYERS] = 0
        board.state[board.CAPTURE_LAYER] = 0

        # Set up Player 2 with 4 blacks (one away from 5-black win)
        board.global_state[board.P2_CAP_W] = 0
        board.global_state[board.P2_CAP_G] = 2
        board.global_state[board.P2_CAP_B] = 4

        # Player 2's turn
        board.global_state[board.CUR_PLAYER] = board.PLAYER_2

        # Set up capture scenario
        b2_idx = board.str_to_index("B2")
        c3_idx = board.str_to_index("C3")

        gray_layer = board.MARBLE_TO_LAYER["g"]
        black_layer = board.MARBLE_TO_LAYER["b"]

        board.state[gray_layer][b2_idx] = 1
        board.state[black_layer][c3_idx] = 1

        # Execute capture
        b2_y, b2_x = b2_idx
        c3_y, c3_x = c3_idx

        for dir_idx, (dy, dx) in enumerate(board.DIRECTIONS):
            if (b2_y + dy, b2_x + dx) == (c3_y, c3_x):
                capture_action = (dir_idx, b2_y, b2_x)
                break

        game.take_action("CAP", capture_action)

        # Player 2 should have 5 blacks → WIN in Blitz!
        assert board.global_state[board.P2_CAP_B] == 5
        outcome = game.get_game_ended()
        assert outcome == PLAYER_2_WIN, (
            "Player 2 should win with 5 blacks in Blitz mode"
        )
