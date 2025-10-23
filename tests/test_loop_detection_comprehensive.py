"""Comprehensive tests for loop detection with actual game states.

Tests verify that loop detection works correctly when players place captured marbles
in repeating patterns after the supply is exhausted.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame, TIE
from game.zertz_board import ZertzBoard


class TestLoopDetectionWithGameStates:
    """Test loop detection with realistic game scenarios."""

    def test_loop_with_captured_marble_placement(self):
        """Test loop detection when players place captured marbles repeatedly.

        This is the specific scenario mentioned: supply empty, players place
        captured marbles creating a loop of board states.
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        board = game.board

        # Empty the supply
        board.global_state[board.SUPPLY_SLICE] = 0

        # Give each player some captured marbles
        board.global_state[board.P1_CAP_SLICE] = [2, 2, 2]  # 2 of each color
        board.global_state[board.P2_CAP_SLICE] = [2, 2, 2]  # 2 of each color

        # Create a board with two open positions that will be used repeatedly
        # Place marbles on all rings except D4 and E4
        white_layer = board.MARBLE_TO_LAYER["w"]
        all_rings = np.argwhere(board.state[board.RING_LAYER] == 1)

        d4_pos = board.str_to_index("D4")
        e4_pos = board.str_to_index("E4")

        for y, x in all_rings:
            if (y, x) != d4_pos and (y, x) != e4_pos:
                board.state[white_layer, y, x] = 1

        # Now simulate a loop: P1 places at D4, P2 at E4, P1 at E4, P2 at D4
        # Then repeat: P1 at D4, P2 at E4, P1 at E4, P2 at D4

        # First pair of moves
        d4_flat = board._2d_to_flat(*d4_pos)
        e4_flat = board._2d_to_flat(*e4_pos)
        no_removal = board.width**2

        # P1 places white at D4 (no removal)
        game.take_action("PUT", (0, d4_flat, no_removal))

        # P2 places white at E4 (no removal)
        game.take_action("PUT", (0, e4_flat, no_removal))

        # Second pair of moves
        # P1 places white at E4 (no removal) - but it's occupied, so use gray at D4
        # Actually, we need to remove marbles to make this work
        board.state[white_layer, d4_pos[0], d4_pos[1]] = 0
        board.state[white_layer, e4_pos[0], e4_pos[1]] = 0

        # P1 places gray at E4
        game.take_action("PUT", (1, e4_flat, no_removal))

        # P2 places gray at D4
        game.take_action("PUT", (1, d4_flat, no_removal))

        # Now repeat the first pair
        board.state[board.MARBLE_TO_LAYER["g"], d4_pos[0], d4_pos[1]] = 0
        board.state[board.MARBLE_TO_LAYER["g"], e4_pos[0], e4_pos[1]] = 0

        # P1 places white at D4 (same as move 1)
        game.take_action("PUT", (0, d4_flat, no_removal))

        # P2 places white at E4 (same as move 2)
        game.take_action("PUT", (0, e4_flat, no_removal))

        # Repeat second pair
        board.state[white_layer, d4_pos[0], d4_pos[1]] = 0
        board.state[white_layer, e4_pos[0], e4_pos[1]] = 0

        # P1 places gray at E4 (same as move 3)
        game.take_action("PUT", (1, e4_flat, no_removal))

        # P2 places gray at D4 (same as move 4)
        game.take_action("PUT", (1, d4_flat, no_removal))

        # Verify loop is detected
        assert game._has_move_loop(), "Loop should be detected after repeating 2 pairs"
        assert game._is_game_over(), "Game should be over when loop detected"
        assert game.get_game_ended() == TIE, "Loop should result in tie"

    def test_loop_detection_pattern_explanation(self):
        """Test and document the exact pattern for loop detection.

        Pattern: [A, B, C, D, A, B, C, D]
        - First pair: (A, B)
        - Second pair: (C, D)
        - Repeat first pair: (A, B)
        - Repeat second pair: (C, D)

        This represents Player 1 making move A, Player 2 making move B,
        then Player 1 making move C, Player 2 making move D,
        then the pattern repeating.
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Manually construct the pattern
        move_a = ("PUT", (0, 10, 20))
        move_b = ("PUT", (1, 11, 21))
        move_c = ("PUT", (2, 12, 22))
        move_d = ("PUT", (0, 13, 23))

        # First occurrence of 2 pairs
        game.move_history.extend([move_a, move_b, move_c, move_d])

        # Not a loop yet (only 4 moves)
        assert not game._has_move_loop()

        # Second occurrence of 2 pairs
        game.move_history.extend([move_a, move_b, move_c, move_d])

        # Now it's a loop (8 moves, last 4 == preceding 4)
        assert game._has_move_loop()

        # Verify the pattern
        assert game.move_history[-4:] == [move_a, move_b, move_c, move_d]
        assert game.move_history[-8:-4] == [move_a, move_b, move_c, move_d]

    def test_simple_two_move_loop(self):
        """Test simplest loop: alternating between two positions.

        Pattern: [A, B, A, B]
        Player 1 does A, Player 2 does B, repeat.
        """
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Set k=1 for simpler test (only need 4 moves)
        game.loop_detection_pairs = 1

        move_a = ("PUT", (0, 10, 20))
        move_b = ("PUT", (1, 11, 21))

        # First pair
        game.move_history.extend([move_a, move_b])
        assert not game._has_move_loop()

        # Repeat the pair
        game.move_history.extend([move_a, move_b])
        assert game._has_move_loop()

    def test_no_loop_with_different_moves(self):
        """Verify that different moves don't trigger false loop detection."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # 8 different moves
        for i in range(8):
            game.move_history.append(("PUT", (0, i, i+10)))

        # Should not detect a loop
        assert not game._has_move_loop()

    def test_loop_with_passes(self):
        """Test that PASS actions are correctly included in loop detection."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Pattern with PASS: [PASS, A, PASS, A] repeated
        move_a = ("PUT", (1, 11, 21))
        move_pass = ("PASS", None)

        # First occurrence
        game.move_history.extend([move_pass, move_a, move_pass, move_a])
        assert not game._has_move_loop()

        # Second occurrence
        game.move_history.extend([move_pass, move_a, move_pass, move_a])
        assert game._has_move_loop()

    def test_near_loop_but_not_quite(self):
        """Test patterns that are almost loops but not quite."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        move_a = ("PUT", (0, 10, 20))
        move_b = ("PUT", (1, 11, 21))
        move_c = ("PUT", (2, 12, 22))
        move_d = ("PUT", (0, 13, 23))

        # Pattern: [A, B, C, D, A, B, C, X] - last move is different
        game.move_history.extend([move_a, move_b, move_c, move_d])
        game.move_history.extend([move_a, move_b, move_c, ("PUT", (1, 99, 99))])

        # Should NOT be a loop (last pair doesn't match)
        assert not game._has_move_loop()

    def test_loop_detection_with_k_equals_3(self):
        """Test loop detection with k=3 (need 12 moves)."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)
        game.loop_detection_pairs = 3

        # Create a pattern of 6 moves
        pattern = [
            ("PUT", (0, i, i+10)) for i in range(6)
        ]

        # Need to repeat it twice to get 12 moves
        game.move_history.extend(pattern)
        assert not game._has_move_loop()  # Only 6 moves

        game.move_history.extend(pattern)
        assert game._has_move_loop()  # Now 12 moves, loop detected

    def test_loop_overrides_other_end_conditions(self):
        """Verify that loop detection is checked before other end conditions."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Set up a loop
        pattern = [("PUT", (0, 10, 20)), ("PUT", (1, 11, 21))]
        game.move_history.extend(pattern * 4)  # 8 moves, creates loop

        # Also set up a win condition (should be overridden by loop)
        game.board.global_state[game.board.P1_CAP_SLICE] = [4, 0, 0]

        # Loop should be detected first
        assert game._has_move_loop()
        assert game.get_game_ended() == TIE
        assert game.get_game_end_reason() == "Move loop detected (repeated position)"

    def test_pass_loop_is_different_from_move_loop(self):
        """Verify that immobilization (2 passes) is distinct from move loop."""
        game = ZertzGame(rings=ZertzBoard.SMALL_BOARD_37)

        # Two consecutive passes
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        # Should detect immobilization, not move loop
        assert game._both_players_immobilized()
        assert not game._has_move_loop()  # Not enough moves for loop (only 2)
        assert game._is_game_over()