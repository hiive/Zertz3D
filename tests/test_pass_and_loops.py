"""Test passing, immobilization, and loop detection."""

import numpy as np
import pytest
import sys
from pathlib import Path

from hiivelabs_mcts import algebraic_to_coordinate

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame, PLAYER_1_WIN, PLAYER_2_WIN, TIE
from game.zertz_board import ZertzBoard
from game.players import RandomZertzPlayer


class TestPassingAndLoops:
    """Test player passing and loop detection."""

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_player_can_pass_when_no_valid_moves(self, rings):
        """Player should be able to pass when they have no valid moves (all board sizes)."""
        game = ZertzGame(rings=rings)
        player = RandomZertzPlayer(game, 1)

        # Create a situation where player has no marbles and no valid moves
        # Set supply to zero
        game.board.global_state[game.board.SUPPLY_SLICE] = 0
        # Set player 1's captured marbles to zero
        game.board.global_state[game.board.P1_CAP_SLICE] = 0

        # Player should return PASS action
        action_type, action = player.get_action()
        assert action_type == "PASS"
        assert action is None

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_pass_action_switches_player(self, rings):
        """PASS action should switch to the other player (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Record initial player
        initial_player = game.get_cur_player_value()

        # Execute a PASS action
        game.take_action("PASS", None)

        # Player should have switched
        assert game.get_cur_player_value() == -initial_player

        # Move should be recorded in history
        assert game.move_history[-1] == ("PASS", None)

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_both_players_immobilized_ends_game(self, rings):
        """Game should end when both players pass consecutively (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Execute two consecutive passes
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        # Game should be over
        assert game._is_game_over()
        assert game._both_players_immobilized()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_immobilization_tie_when_no_winner(self, rings):
        """Game should end in tie when both players immobilized with no winner (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Set up scenario where neither player has winning marbles
        game.board.global_state[game.board.P1_CAP_SLICE] = [
            1,
            1,
            1,
        ]  # Not enough for any win condition
        game.board.global_state[game.board.P2_CAP_SLICE] = [
            2,
            2,
            2,
        ]  # Not enough for any win condition

        # Both players pass
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        # Should be a tie
        assert game.get_game_ended() == TIE

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_immobilization_player1_wins(self, rings):
        """Player 1 should win on immobilization if they have winning marbles (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Give player 1 a winning combination (4 white)
        game.board.global_state[game.board.P1_CAP_SLICE] = [4, 0, 0]
        game.board.global_state[game.board.P2_CAP_SLICE] = [0, 0, 0]

        # Both players pass
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        # Player 1 should win
        assert game.get_game_ended() == PLAYER_1_WIN

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_immobilization_player2_wins(self, rings):
        """Player 2 should win on immobilization if they have winning marbles (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Give player 2 a winning combination (5 grey)
        game.board.global_state[game.board.P1_CAP_SLICE] = [0, 0, 0]
        game.board.global_state[game.board.P2_CAP_SLICE] = [0, 5, 0]

        # Both players pass
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        # Player 2 should win
        assert game.get_game_ended() == PLAYER_2_WIN

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_single_pass_does_not_end_game(self, rings):
        """A single pass should not end the game (all board sizes)."""
        game = ZertzGame(rings=rings)

        # One player passes
        game.take_action("PASS", None)

        # Game should not be over
        assert not game._is_game_over()
        assert game.get_game_ended() is None

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_detection_with_2_pairs(self, rings):
        """Game should detect loop when last 2 move-pairs match preceding 2 pairs (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Create a sequence: A, B, A, B (2 pairs)
        # First pair
        game.move_history.append(("PUT", (0, 10, 20)))
        game.move_history.append(("PUT", (1, 11, 21)))
        # Second pair
        game.move_history.append(("PUT", (2, 12, 22)))
        game.move_history.append(("PUT", (0, 13, 23)))
        # Repeat first pair
        game.move_history.append(("PUT", (0, 10, 20)))
        game.move_history.append(("PUT", (1, 11, 21)))
        # Repeat second pair
        game.move_history.append(("PUT", (2, 12, 22)))
        game.move_history.append(("PUT", (0, 13, 23)))

        # Loop should be detected
        assert game._has_move_loop()
        assert game._is_game_over()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_detection_results_in_tie(self, rings):
        """Loop detection should result in a tie (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Create a loop (4 moves repeated)
        for _ in range(2):
            game.move_history.append(("PUT", (0, 10, 20)))
            game.move_history.append(("PUT", (1, 11, 21)))
            game.move_history.append(("PUT", (2, 12, 22)))
            game.move_history.append(("PUT", (0, 13, 23)))

        # Should be a tie
        assert game.get_game_ended() == TIE

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_not_detected_with_insufficient_moves(self, rings):
        """Loop should not be detected with fewer than 8 moves (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Add only 6 moves (need 8 for k=2 pairs)
        for i in range(6):
            game.move_history.append(("PUT", (0, i, i + 1)))

        # Loop should not be detected
        assert not game._has_move_loop()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_not_detected_with_different_moves(self, rings):
        """Loop should not be detected if moves don't match (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Create 8 different moves
        for i in range(8):
            game.move_history.append(("PUT", (0, i, i + 1)))

        # Loop should not be detected
        assert not game._has_move_loop()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_pass_in_loop_sequence(self, rings):
        """PASS actions should be included in loop detection (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Create a loop with PASS actions
        # First pair
        game.move_history.append(("PASS", None))
        game.move_history.append(("PUT", (1, 11, 21)))
        # Second pair
        game.move_history.append(("PASS", None))
        game.move_history.append(("PUT", (1, 11, 21)))
        # Repeat first pair
        game.move_history.append(("PASS", None))
        game.move_history.append(("PUT", (1, 11, 21)))
        # Repeat second pair
        game.move_history.append(("PASS", None))
        game.move_history.append(("PUT", (1, 11, 21)))

        # Loop should be detected
        assert game._has_move_loop()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_with_captured_marble_placement(self, rings):
        """Test loop detection when players place captured marbles repeatedly.

        This is the specific scenario mentioned: supply empty, players place
        captured marbles creating a loop of board states.
        """
        game = ZertzGame(rings=rings)
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

        d4_pos = algebraic_to_coordinate("D4", board.config)
        e4_pos = algebraic_to_coordinate("E4", board.config)

        for y, x in all_rings:
            if (y, x) != d4_pos and (y, x) != e4_pos:
                board.state[white_layer, y, x] = 1

        # Now simulate a loop: P1 places at D4, P2 at E4, P1 at E4, P2 at D4
        # Then repeat: P1 at D4, P2 at E4, P1 at E4, P2 at D4

        # First pair of moves
        d4_flat = d4_pos[0] * board.config.width + d4_pos[1]
        e4_flat = e4_pos[0] * board.config.width + e4_pos[1]
        no_removal = board.config.width**2

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

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_detection_pattern_explanation(self, rings):
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
        game = ZertzGame(rings=rings)

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

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_simple_two_move_loop(self, rings):
        """Test simplest loop: alternating between two positions.

        Pattern: [A, B, A, B]
        Player 1 does A, Player 2 does B, repeat.
        """
        game = ZertzGame(rings=rings)

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

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_no_loop_with_different_moves(self, rings):
        """Verify that different moves don't trigger false loop detection."""
        game = ZertzGame(rings=rings)

        # 8 different moves
        for i in range(8):
            game.move_history.append(("PUT", (0, i, i+10)))

        # Should not detect a loop
        assert not game._has_move_loop()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_with_passes(self, rings):
        """Test that PASS actions are correctly included in loop detection."""
        game = ZertzGame(rings=rings)

        # Pattern with PASS: [PASS, A, PASS, A] repeated
        move_a = ("PUT", (1, 11, 21))
        move_pass = ("PASS", None)

        # First occurrence
        game.move_history.extend([move_pass, move_a, move_pass, move_a])
        assert not game._has_move_loop()

        # Second occurrence
        game.move_history.extend([move_pass, move_a, move_pass, move_a])
        assert game._has_move_loop()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_near_loop_but_not_quite(self, rings):
        """Test patterns that are almost loops but not quite."""
        game = ZertzGame(rings=rings)

        move_a = ("PUT", (0, 10, 20))
        move_b = ("PUT", (1, 11, 21))
        move_c = ("PUT", (2, 12, 22))
        move_d = ("PUT", (0, 13, 23))

        # Pattern: [A, B, C, D, A, B, C, X] - last move is different
        game.move_history.extend([move_a, move_b, move_c, move_d])
        game.move_history.extend([move_a, move_b, move_c, ("PUT", (1, 99, 99))])

        # Should NOT be a loop (last pair doesn't match)
        assert not game._has_move_loop()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_detection_with_k_equals_3(self, rings):
        """Test loop detection with k=3 (need 12 moves)."""
        game = ZertzGame(rings=rings)
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

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_overrides_other_end_conditions(self, rings):
        """Verify that loop detection is checked before other end conditions."""
        game = ZertzGame(rings=rings)

        # Set up a loop
        pattern = [("PUT", (0, 10, 20)), ("PUT", (1, 11, 21))]
        game.move_history.extend(pattern * 4)  # 8 moves, creates loop

        # Also set up a win condition (should be overridden by loop)
        game.board.global_state[game.board.P1_CAP_SLICE] = [4, 0, 0]

        # Loop should be detected first
        assert game._has_move_loop()
        assert game.get_game_ended() == TIE
        assert game.get_game_end_reason() == "Move loop detected (repeated position)"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_pass_loop_is_different_from_move_loop(self, rings):
        """Verify that immobilization (2 passes) is distinct from move loop."""
        game = ZertzGame(rings=rings)

        # Two consecutive passes
        game.take_action("PASS", None)
        game.take_action("PASS", None)

        # Should detect immobilization, not move loop
        assert game._both_players_immobilized()
        assert not game._has_move_loop()  # Not enough moves for loop (only 2)
        assert game._is_game_over()

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_detection_requires_exact_position_match(self, rings):
        """Test that loop detection requires exact position repetition."""
        game = ZertzGame(rings=rings)

        # Make 8 moves with same 2-move pattern using different positions each time
        # Pattern: P1 at row 1, P2 at row 2
        # Moves 1-2: C1, D2
        # Moves 3-4: E1, F2
        # Moves 5-6: C1, D2 (same as 1-2)
        # Moves 7-8: E1, F2 (same as 3-4)
        game.take_action("PUT", game.str_to_action("PUT w B1")[1])  # P1
        game.take_action("PUT", game.str_to_action("PUT g C2")[1])  # P2
        game.take_action("PUT", game.str_to_action("PUT w D1")[1])  # P1
        game.take_action("PUT", game.str_to_action("PUT g E2")[1])  # P2

        # Repeat the same pattern
        game.take_action("PUT", game.str_to_action("PUT b B2")[1])  # P1 (different color, different pos)
        game.take_action("PUT", game.str_to_action("PUT w D2")[1])  # P2 (different color, different pos)
        game.take_action("PUT", game.str_to_action("PUT g B3")[1])  # P1
        game.take_action("PUT", game.str_to_action("PUT b C3")[1])  # P2

        # No loop should be detected - all positions are different
        assert not game._has_move_loop(), "Loop should not be detected when positions differ"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_loop_detection_with_different_removal_positions(self, rings):
        """Test that different ring removal positions prevent loop detection."""
        game = ZertzGame(rings=rings)

        # Make 8 moves where placement/removal combinations never repeat
        # Each move uses a unique combination of (place position, removal position)
        game.take_action("PUT", game.str_to_action("PUT w B1 A1")[1])  # P1: B1/A1
        game.take_action("PUT", game.str_to_action("PUT g C2 B2")[1])  # P2: C2/B2
        game.take_action("PUT", game.str_to_action("PUT w D1 A2")[1])  # P1: D1/A2
        game.take_action("PUT", game.str_to_action("PUT g E2 C3")[1])  # P2: E2/C3
        game.take_action("PUT", game.str_to_action("PUT w F1 A3")[1])  # P1: F1/A3
        game.take_action("PUT", game.str_to_action("PUT g G1 C4")[1])  # P2: G1/C4
        game.take_action("PUT", game.str_to_action("PUT w B4 A4")[1])  # P1: B4/A4
        game.take_action("PUT", game.str_to_action("PUT g C5 D2")[1])  # P2: C5/D2

        # No loop should be detected - all placement & removal combinations are unique
        assert not game._has_move_loop(), "Loop should not be detected with different removals"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_game_ended_returns_none_when_not_over(self, rings):
        """get_game_ended should return None when game is not over (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Fresh game should not be over
        assert game.get_game_ended() is None

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_move_history_records_all_actions(self, rings):
        """Move history should record all actions including PASS (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Manually add actions to history (without executing them)
        game.move_history.append(("PUT", (0, 15, 20)))
        game.move_history.append(("PASS", None))
        game.move_history.append(("CAP", (2, 3, 4)))

        # All should be in history
        assert len(game.move_history) == 3
        assert game.move_history[0] == ("PUT", (0, 15, 20))
        assert game.move_history[1] == ("PASS", None)
        assert game.move_history[2] == ("CAP", (2, 3, 4))

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_action_to_str_handles_pass(self, rings):
        """action_to_str should handle PASS actions (all board sizes)."""
        game = ZertzGame(rings=rings)

        action_str, action_dict = game.action_to_str("PASS", None)

        assert action_str == "PASS"
        assert action_dict == {"action": "PASS"}

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_realistic_immobilization_scenario(self, rings):
        """Test a realistic scenario leading to immobilization (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Empty the marble supply AND both players' captured marbles
        game.board.global_state[game.board.SUPPLY_SLICE] = 0
        game.board.global_state[game.board.P1_CAP_SLICE] = 0
        game.board.global_state[game.board.P2_CAP_SLICE] = 0

        # Remove all marbles from board so no captures are possible
        game.board.state[game.board.MARBLE_TO_LAYER["w"]] = 0
        game.board.state[game.board.MARBLE_TO_LAYER["g"]] = 0
        game.board.state[game.board.MARBLE_TO_LAYER["b"]] = 0

        # Both players should have no valid moves (no marbles to place, no captures)
        p1_placement, p1_capture = game.get_valid_actions()
        assert not np.any(p1_placement), "Player 1 should have no placement moves"
        assert not np.any(p1_capture), "Player 1 should have no capture moves"

        # Switch player and check again
        game.board._next_player()
        p2_placement, p2_capture = game.get_valid_actions()
        assert not np.any(p2_placement), "Player 2 should have no placement moves"
        assert not np.any(p2_capture), "Player 2 should have no capture moves"

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_clone_preserves_move_history(self, rings):
        """Cloning a game should preserve move history (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Use coordinate strings that exist on all board sizes (D4, C3)
        # D4 exists on all boards (A1-A4, B1-B5+, C1-C6+, D1-D7+)
        action_type1, action1 = game.str_to_action("PUT w D4")
        game.take_action(action_type1, action1)

        game.take_action("PASS", None)

        # Clone the game
        cloned = ZertzGame(clone=game, clone_state=game.board.state)

        # Move history should be preserved with correct content
        assert len(cloned.move_history) == 2
        assert cloned.move_history[0][0] == "PUT"
        assert cloned.move_history[0][1] == action1
        assert cloned.move_history[1] == ("PASS", None)
        assert cloned.move_history == game.move_history
        assert cloned.loop_detection_pairs == game.loop_detection_pairs

    @pytest.mark.parametrize("rings", [
        ZertzBoard.SMALL_BOARD_37,
        ZertzBoard.MEDIUM_BOARD_48,
        ZertzBoard.LARGE_BOARD_61
    ])
    def test_different_k_pairs_for_loop_detection(self, rings):
        """Test that loop detection k parameter works correctly (all board sizes)."""
        game = ZertzGame(rings=rings)

        # Set k=1 (need 4 moves: 2 for last pair, 2 for preceding pair)
        game.loop_detection_pairs = 1

        # Use coordinate strings valid on all board sizes
        _, action_a = game.str_to_action("PUT w D4")
        _, action_b = game.str_to_action("PUT g C3")

        # Create a simple repeating pattern: A, B, A, B
        game.move_history.append(("PUT", action_a))
        game.move_history.append(("PUT", action_b))
        game.move_history.append(("PUT", action_a))
        game.move_history.append(("PUT", action_b))

        # Loop should be detected with k=1
        assert game._has_move_loop()

        # Verify the history contains the correct actions
        assert game.move_history[0] == ("PUT", action_a)
        assert game.move_history[1] == ("PUT", action_b)
        assert game.move_history[2] == ("PUT", action_a)
        assert game.move_history[3] == ("PUT", action_b)

        # Reset and test k=3 (need 12 moves)
        game.loop_detection_pairs = 3
        game.move_history = []

        # Use more coordinate strings valid on all boards
        _, act1 = game.str_to_action("PUT w A1")
        _, act2 = game.str_to_action("PUT g B2")
        _, act3 = game.str_to_action("PUT b C2")
        _, act4 = game.str_to_action("PUT w D3")
        _, act5 = game.str_to_action("PUT g A2")
        _, act6 = game.str_to_action("PUT b B3")

        # Add 12 moves (3 pairs repeated twice) - pattern of 6 moves repeated
        pattern = [
            ("PUT", act1),
            ("PUT", act2),
            ("PUT", act3),
            ("PUT", act4),
            ("PUT", act5),
            ("PUT", act6),
        ]
        game.move_history.extend(pattern)
        game.move_history.extend(pattern)

        # Loop should be detected with k=3
        assert game._has_move_loop()

        # Verify the pattern is correctly repeated
        for i in range(6):
            assert game.move_history[i] == game.move_history[i + 6]
