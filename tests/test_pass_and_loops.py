"""Test passing, immobilization, and loop detection."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame, PLAYER_1_WIN, PLAYER_2_WIN, TIE
from game.zertz_player import RandomZertzPlayer


class TestPassingAndLoops:
    """Test player passing and loop detection."""

    def test_player_can_pass_when_no_valid_moves(self):
        """Player should be able to pass when they have no valid moves."""
        game = ZertzGame(rings=37)
        player = RandomZertzPlayer(game, 1)

        # Create a situation where player has no marbles and no valid moves
        # Set supply to zero
        game.board.global_state[game.board.SUPPLY_SLICE] = 0
        # Set player 1's captured marbles to zero
        game.board.global_state[game.board.P1_CAP_SLICE] = 0

        # Player should return PASS action
        action_type, action = player.get_action()
        assert action_type == 'PASS'
        assert action is None

    def test_pass_action_switches_player(self):
        """PASS action should switch to the other player."""
        game = ZertzGame(rings=37)

        # Record initial player
        initial_player = game.get_cur_player_value()

        # Execute a PASS action
        game.take_action('PASS', None)

        # Player should have switched
        assert game.get_cur_player_value() == -initial_player

        # Move should be recorded in history
        assert game.move_history[-1] == ('PASS', None)

    def test_both_players_immobilized_ends_game(self):
        """Game should end when both players pass consecutively."""
        game = ZertzGame(rings=37)

        # Execute two consecutive passes
        game.take_action('PASS', None)
        game.take_action('PASS', None)

        # Game should be over
        assert game._is_game_over()
        assert game._both_players_immobilized()

    def test_immobilization_tie_when_no_winner(self):
        """Game should end in tie when both players immobilized with no winner."""
        game = ZertzGame(rings=37)

        # Set up scenario where neither player has winning marbles
        game.board.global_state[game.board.P1_CAP_SLICE] = [1, 1, 1]  # Not enough for any win condition
        game.board.global_state[game.board.P2_CAP_SLICE] = [2, 2, 2]  # Not enough for any win condition

        # Both players pass
        game.take_action('PASS', None)
        game.take_action('PASS', None)

        # Should be a tie
        assert game.get_game_ended() == TIE

    def test_immobilization_player1_wins(self):
        """Player 1 should win on immobilization if they have winning marbles."""
        game = ZertzGame(rings=37)

        # Give player 1 a winning combination (4 white)
        game.board.global_state[game.board.P1_CAP_SLICE] = [4, 0, 0]
        game.board.global_state[game.board.P2_CAP_SLICE] = [0, 0, 0]

        # Both players pass
        game.take_action('PASS', None)
        game.take_action('PASS', None)

        # Player 1 should win
        assert game.get_game_ended() == PLAYER_1_WIN

    def test_immobilization_player2_wins(self):
        """Player 2 should win on immobilization if they have winning marbles."""
        game = ZertzGame(rings=37)

        # Give player 2 a winning combination (5 grey)
        game.board.global_state[game.board.P1_CAP_SLICE] = [0, 0, 0]
        game.board.global_state[game.board.P2_CAP_SLICE] = [0, 5, 0]

        # Both players pass
        game.take_action('PASS', None)
        game.take_action('PASS', None)

        # Player 2 should win
        assert game.get_game_ended() == PLAYER_2_WIN

    def test_single_pass_does_not_end_game(self):
        """A single pass should not end the game."""
        game = ZertzGame(rings=37)

        # One player passes
        game.take_action('PASS', None)

        # Game should not be over
        assert not game._is_game_over()
        assert game.get_game_ended() is None

    def test_loop_detection_with_2_pairs(self):
        """Game should detect loop when last 2 move-pairs match preceding 2 pairs."""
        game = ZertzGame(rings=37)

        # Create a sequence: A, B, A, B (2 pairs)
        # First pair
        game.move_history.append(('PUT', (0, 10, 20)))
        game.move_history.append(('PUT', (1, 11, 21)))
        # Second pair
        game.move_history.append(('PUT', (2, 12, 22)))
        game.move_history.append(('PUT', (0, 13, 23)))
        # Repeat first pair
        game.move_history.append(('PUT', (0, 10, 20)))
        game.move_history.append(('PUT', (1, 11, 21)))
        # Repeat second pair
        game.move_history.append(('PUT', (2, 12, 22)))
        game.move_history.append(('PUT', (0, 13, 23)))

        # Loop should be detected
        assert game._has_move_loop()
        assert game._is_game_over()

    def test_loop_detection_results_in_tie(self):
        """Loop detection should result in a tie."""
        game = ZertzGame(rings=37)

        # Create a loop (4 moves repeated)
        for _ in range(2):
            game.move_history.append(('PUT', (0, 10, 20)))
            game.move_history.append(('PUT', (1, 11, 21)))
            game.move_history.append(('PUT', (2, 12, 22)))
            game.move_history.append(('PUT', (0, 13, 23)))

        # Should be a tie
        assert game.get_game_ended() == TIE

    def test_loop_not_detected_with_insufficient_moves(self):
        """Loop should not be detected with fewer than 8 moves."""
        game = ZertzGame(rings=37)

        # Add only 6 moves (need 8 for k=2 pairs)
        for i in range(6):
            game.move_history.append(('PUT', (0, i, i+1)))

        # Loop should not be detected
        assert not game._has_move_loop()

    def test_loop_not_detected_with_different_moves(self):
        """Loop should not be detected if moves don't match."""
        game = ZertzGame(rings=37)

        # Create 8 different moves
        for i in range(8):
            game.move_history.append(('PUT', (0, i, i+1)))

        # Loop should not be detected
        assert not game._has_move_loop()

    def test_pass_in_loop_sequence(self):
        """PASS actions should be included in loop detection."""
        game = ZertzGame(rings=37)

        # Create a loop with PASS actions
        # First pair
        game.move_history.append(('PASS', None))
        game.move_history.append(('PUT', (1, 11, 21)))
        # Second pair
        game.move_history.append(('PASS', None))
        game.move_history.append(('PUT', (1, 11, 21)))
        # Repeat first pair
        game.move_history.append(('PASS', None))
        game.move_history.append(('PUT', (1, 11, 21)))
        # Repeat second pair
        game.move_history.append(('PASS', None))
        game.move_history.append(('PUT', (1, 11, 21)))

        # Loop should be detected
        assert game._has_move_loop()

    def test_game_ended_returns_none_when_not_over(self):
        """get_game_ended should return None when game is not over."""
        game = ZertzGame(rings=37)

        # Fresh game should not be over
        assert game.get_game_ended() is None

    def test_move_history_records_all_actions(self):
        """Move history should record all actions including PASS."""
        game = ZertzGame(rings=37)

        # Manually add actions to history (without executing them)
        game.move_history.append(('PUT', (0, 15, 20)))
        game.move_history.append(('PASS', None))
        game.move_history.append(('CAP', (2, 3, 4)))

        # All should be in history
        assert len(game.move_history) == 3
        assert game.move_history[0] == ('PUT', (0, 15, 20))
        assert game.move_history[1] == ('PASS', None)
        assert game.move_history[2] == ('CAP', (2, 3, 4))

    def test_action_to_str_handles_pass(self):
        """action_to_str should handle PASS actions."""
        game = ZertzGame(rings=37)

        action_str, action_dict = game.action_to_str('PASS', None)

        assert action_str == 'PASS'
        assert action_dict == {'action': 'PASS'}

    def test_realistic_immobilization_scenario(self):
        """Test a realistic scenario leading to immobilization."""
        game = ZertzGame(rings=37)

        # Empty the marble supply AND both players' captured marbles
        game.board.global_state[game.board.SUPPLY_SLICE] = 0
        game.board.global_state[game.board.P1_CAP_SLICE] = 0
        game.board.global_state[game.board.P2_CAP_SLICE] = 0

        # Remove all marbles from board so no captures are possible
        game.board.state[game.board.MARBLE_TO_LAYER['w']] = 0
        game.board.state[game.board.MARBLE_TO_LAYER['g']] = 0
        game.board.state[game.board.MARBLE_TO_LAYER['b']] = 0

        # Both players should have no valid moves (no marbles to place, no captures)
        p1_placement, p1_capture = game.get_valid_actions()
        assert not np.any(p1_placement), "Player 1 should have no placement moves"
        assert not np.any(p1_capture), "Player 1 should have no capture moves"

        # Switch player and check again
        game.board._next_player()
        p2_placement, p2_capture = game.get_valid_actions()
        assert not np.any(p2_placement), "Player 2 should have no placement moves"
        assert not np.any(p2_capture), "Player 2 should have no capture moves"

    def test_clone_preserves_move_history(self):
        """Cloning a game should preserve move history."""
        game = ZertzGame(rings=37)

        # Add some moves
        game.take_action('PUT', (0, 15, 20))
        game.take_action('PASS', None)

        # Clone the game
        cloned = ZertzGame(clone=game, clone_state=game.board.state)

        # Move history should be preserved
        assert len(cloned.move_history) == 2
        assert cloned.move_history == game.move_history
        assert cloned.loop_detection_pairs == game.loop_detection_pairs

    def test_different_k_pairs_for_loop_detection(self):
        """Test that loop detection k parameter works correctly."""
        game = ZertzGame(rings=37)

        # Set k=1 (need 4 moves: 2 for last pair, 2 for preceding pair)
        game.loop_detection_pairs = 1

        # Create a simple repeating pattern: A, B, A, B
        game.move_history.append(('PUT', (0, 10, 20)))
        game.move_history.append(('PUT', (1, 11, 21)))
        game.move_history.append(('PUT', (0, 10, 20)))
        game.move_history.append(('PUT', (1, 11, 21)))

        # Loop should be detected with k=1
        assert game._has_move_loop()

        # Reset and test k=3 (need 12 moves)
        game.loop_detection_pairs = 3
        game.move_history = []

        # Add 12 moves (3 pairs repeated twice)
        pattern = [
            ('PUT', (0, 1, 2)),
            ('PUT', (1, 2, 3)),
            ('PUT', (2, 3, 4)),
            ('PUT', (0, 4, 5)),
            ('PUT', (1, 5, 6)),
            ('PUT', (2, 6, 7)),
        ]
        game.move_history.extend(pattern)
        game.move_history.extend(pattern)

        # Loop should be detected with k=3
        assert game._has_move_loop()