"""Test passing, immobilization, and loop detection."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame, PLAYER_1_WIN, PLAYER_2_WIN, TIE
from game.zertz_board import ZertzBoard
from game.zertz_player import RandomZertzPlayer


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
