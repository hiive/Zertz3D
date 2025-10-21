"""
Comparison tests between Rust and Python MCTS backends.

These tests ensure that both backends produce identical results for all exposed functionality.
"""

import pytest
import numpy as np
from game.zertz_game import ZertzGame
from game.players.mcts_zertz_player import MCTSZertzPlayer
from learner.mcts.backend import HAS_RUST

# Skip all tests if Rust backend not available
pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")


class TestGameLogicComparison:
    """Test that Rust and Python game logic produce identical results."""

    def setup_method(self):
        """Create test games for each backend."""
        np.random.seed(42)
        self.game = ZertzGame(rings=37)

    def test_initial_valid_actions_identical(self):
        """Both backends should return identical valid actions on initial board."""
        # Get valid actions using game's method (Python)
        py_placement, py_capture = self.game.get_valid_actions()

        # Get state for Rust backend
        state = self.game.get_current_state()

        # Import Rust backend
        import hiivelabs_zertz_mcts

        rust_backend = hiivelabs_zertz_mcts.BoardState(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            self.game.board.rings,
            1  # t parameter
        )

        rust_placement, rust_capture = rust_backend.get_valid_actions()

        # Compare placement masks
        np.testing.assert_array_equal(
            py_placement,
            rust_placement,
            err_msg="Placement masks differ between Python and Rust"
        )

        # Compare capture masks
        np.testing.assert_array_equal(
            py_capture,
            rust_capture,
            err_msg="Capture masks differ between Python and Rust"
        )

    def test_valid_actions_after_moves(self):
        """Valid actions should match after several moves."""
        # Play a few moves
        for _ in range(5):
            placement_mask, capture_mask = self.game.get_valid_actions()

            # Find valid action
            if np.any(capture_mask):
                indices = np.where(capture_mask > 0)
                action_type = "CAP"
                action_data = (indices[0][0], indices[1][0], indices[2][0])
            elif np.any(placement_mask):
                indices = np.where(placement_mask > 0)
                action_type = "PUT"
                action_data = (indices[0][0], indices[1][0], indices[2][0])
            else:
                action_type = "PASS"
                action_data = None

            self.game.take_action(action_type, action_data)

        # Now compare valid actions
        py_placement, py_capture = self.game.get_valid_actions()

        # Get Rust valid actions
        import hiivelabs_zertz_mcts

        state = self.game.get_current_state()
        rust_backend = hiivelabs_zertz_mcts.BoardState(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            self.game.board.rings,
            1  # t parameter
        )

        rust_placement, rust_capture = rust_backend.get_valid_actions()

        np.testing.assert_array_equal(py_placement, rust_placement)
        np.testing.assert_array_equal(py_capture, rust_capture)

    def test_placement_state_transitions_identical(self):
        """State after placement should be identical in both backends."""
        # Get initial state
        initial_state = self.game.get_current_state()

        # Get valid placement (shape: 3, width², width²+1)
        placement_mask, _ = self.game.get_valid_actions()
        indices = np.where(placement_mask > 0)

        if indices[0].size == 0:
            pytest.skip("No placement available in this state")

        # Choose first valid placement
        # indices are: (marble_type, dst_flat, remove_flat)
        marble_type = int(indices[0][0])
        dst_flat = int(indices[1][0])
        remove_flat = int(indices[2][0])

        # Apply in Python
        py_game = ZertzGame(rings=37)
        py_game.board.state = np.copy(initial_state['spatial'])
        py_game.board.global_state = np.copy(initial_state['global'])
        py_game.take_action("PUT", (marble_type, dst_flat, remove_flat))
        py_result = py_game.get_current_state()

        # Apply in Rust
        import hiivelabs_zertz_mcts

        rust_backend = hiivelabs_zertz_mcts.BoardState(
            initial_state['spatial'].astype(np.float32),
            initial_state['global'].astype(np.float32),
            self.game.board.rings,
            1  # t parameter
        )

        # Convert flat indices to (y, x)
        dst_y = dst_flat // self.game.board.width
        dst_x = dst_flat % self.game.board.width
        remove_y = remove_flat // self.game.board.width if remove_flat < self.game.board.width**2 else None
        remove_x = remove_flat % self.game.board.width if remove_flat < self.game.board.width**2 else None

        rust_backend.apply_placement(marble_type, dst_y, dst_x, remove_y, remove_x)

        # Compare resulting states
        np.testing.assert_array_almost_equal(
            py_result['spatial'],
            rust_backend.get_spatial(),
            decimal=5,
            err_msg="Spatial states differ after placement"
        )

        np.testing.assert_array_almost_equal(
            py_result['global'],
            rust_backend.get_global(),
            decimal=5,
            err_msg="Global states differ after placement"
        )

    def test_capture_state_transitions_identical(self):
        """State after capture should be identical in both backends."""
        # Set up a capture scenario
        np.random.seed(42)
        game = ZertzGame(rings=37)

        # Play until we get a capture
        max_moves = 50
        for _ in range(max_moves):
            placement_mask, capture_mask = game.get_valid_actions()

            if np.any(capture_mask):
                # Found a capture - save state and test
                pre_capture_state = game.get_current_state()

                indices = np.where(capture_mask > 0)
                direction = int(indices[0][0])
                start_y = int(indices[1][0])
                start_x = int(indices[2][0])

                # Apply in Python
                py_game = ZertzGame(rings=37)
                py_game.board.state = np.copy(pre_capture_state['spatial'])
                py_game.board.global_state = np.copy(pre_capture_state['global'])
                py_game.take_action("CAP", (direction, start_y, start_x))
                py_result = py_game.get_current_state()

                # Apply in Rust
                import hiivelabs_zertz_mcts

                rust_backend = hiivelabs_zertz_mcts.BoardState(
                    pre_capture_state['spatial'].astype(np.float32),
                    pre_capture_state['global'].astype(np.float32),
                    game.board.rings,
                    1  # t parameter
                )

                rust_backend.apply_capture(start_y, start_x, direction)

                # Compare
                np.testing.assert_array_almost_equal(
                    py_result['spatial'],
                    rust_backend.get_spatial(),
                    decimal=5,
                    err_msg="Spatial states differ after capture"
                )

                np.testing.assert_array_almost_equal(
                    py_result['global'],
                    rust_backend.get_global(),
                    decimal=5,
                    err_msg="Global states differ after capture"
                )

                return  # Test passed

            # Make a move to continue
            if np.any(placement_mask):
                indices = np.where(placement_mask > 0)
                game.take_action("PUT", (indices[0][0], indices[1][0], indices[2][0]))
            else:
                game.take_action("PASS", None)

        pytest.skip("No capture encountered in test game")


class TestMCTSBackendComparison:
    """Test that MCTS backends produce comparable results."""

    def setup_method(self):
        """Create test game."""
        np.random.seed(42)
        self.game = ZertzGame(rings=37)

    def test_both_backends_return_valid_actions(self):
        """Both backends should return valid actions."""
        # Python backend
        py_player = MCTSZertzPlayer(
            self.game,
            n=1,
            iterations=50,
            parallel=False,
            verbose=False,
            backend='python'
        )

        py_action = py_player.get_action()
        assert py_action[0] in ["PUT", "CAP", "PASS"]

        # Rust backend
        rust_player = MCTSZertzPlayer(
            self.game,
            n=1,
            iterations=50,
            parallel=False,
            verbose=False,
            backend='rust'
        )

        rust_action = rust_player.get_action()
        assert rust_action[0] in ["PUT", "CAP", "PASS"]

    def test_backends_respect_mandatory_captures(self):
        """Both backends should prioritize captures when available."""
        # Play until we get a capture opportunity
        np.random.seed(123)
        game = ZertzGame(rings=37)

        max_moves = 50
        for _ in range(max_moves):
            _, capture_mask = game.get_valid_actions()

            if np.any(capture_mask):
                # Both backends should return captures
                py_player = MCTSZertzPlayer(
                    game, n=game.get_cur_player_value(),
                    iterations=20, parallel=False, backend='python'
                )
                py_action = py_player.get_action()
                assert py_action[0] == "CAP", "Python backend didn't return mandatory capture"

                rust_player = MCTSZertzPlayer(
                    game, n=game.get_cur_player_value(),
                    iterations=20, parallel=False, backend='rust'
                )
                rust_action = rust_player.get_action()
                assert rust_action[0] == "CAP", "Rust backend didn't return mandatory capture"

                return

            # Continue game
            placement_mask, _ = game.get_valid_actions()
            if np.any(placement_mask):
                from game.zertz_player import RandomZertzPlayer
                player = RandomZertzPlayer(game, game.get_cur_player_value())
                action = player.get_action()
                game.take_action(action[0], action[1])
            else:
                game.take_action("PASS", None)

        pytest.skip("No capture encountered")

    def test_backends_avoid_invalid_actions(self):
        """Neither backend should return invalid actions."""
        for seed in [42, 123, 456, 789]:
            np.random.seed(seed)
            game = ZertzGame(rings=37)

            # Test multiple moves
            for _ in range(10):
                if game.get_game_ended() is not None:
                    break

                placement_mask, capture_mask = game.get_valid_actions()

                # Python backend
                py_player = MCTSZertzPlayer(
                    game, n=game.get_cur_player_value(),
                    iterations=20, parallel=False, backend='python'
                )
                py_action = py_player.get_action()

                # Verify action is valid
                if py_action[0] == "PUT":
                    marble, dst, remove = py_action[1]
                    assert placement_mask[marble, dst, remove] > 0, \
                        f"Python returned invalid placement at seed {seed}"
                elif py_action[0] == "CAP":
                    direction, start_y, start_x = py_action[1]
                    assert capture_mask[direction, start_y, start_x] > 0, \
                        f"Python returned invalid capture at seed {seed}"

                # Rust backend (reset game state first)
                game2 = ZertzGame(rings=37)
                game2.board.state = np.copy(game.board.state)
                game2.board.global_state = np.copy(game.board.global_state)

                rust_player = MCTSZertzPlayer(
                    game2, n=game.get_cur_player_value(),
                    iterations=20, parallel=False, backend='rust'
                )
                rust_action = rust_player.get_action()

                # Verify action is valid
                if rust_action[0] == "PUT":
                    marble, dst, remove = rust_action[1]
                    assert placement_mask[marble, dst, remove] > 0, \
                        f"Rust returned invalid placement at seed {seed}"
                elif rust_action[0] == "CAP":
                    direction, start_y, start_x = rust_action[1]
                    assert capture_mask[direction, start_y, start_x] > 0, \
                        f"Rust returned invalid capture at seed {seed}"

                # Apply Python action to continue
                game.take_action(py_action[0], py_action[1])

    def test_backends_produce_legal_games(self):
        """Complete games using each backend should be legal."""
        for backend in ['python', 'rust']:
            np.random.seed(42)
            game = ZertzGame(rings=37)

            player1 = MCTSZertzPlayer(
                game, n=1,
                iterations=10,  # Small for speed
                parallel=False,
                backend=backend
            )
            player2 = MCTSZertzPlayer(
                game, n=2,
                iterations=10,
                parallel=False,
                backend=backend
            )

            move_count = 0
            max_moves = 100

            while game.get_game_ended() is None and move_count < max_moves:
                current_player = player1 if game.get_cur_player_value() == 1 else player2
                action = current_player.get_action()

                # Verify action is legal
                placement_mask, capture_mask = game.get_valid_actions()

                if action[0] == "PUT":
                    marble, dst, remove = action[1]
                    assert placement_mask[marble, dst, remove] > 0, \
                        f"{backend} backend produced illegal placement"
                elif action[0] == "CAP":
                    direction, start_y, start_x = action[1]
                    assert capture_mask[direction, start_y, start_x] > 0, \
                        f"{backend} backend produced illegal capture"

                game.take_action(action[0], action[1])
                move_count += 1

            # Game should end legally
            assert move_count < max_moves, f"{backend} backend game didn't terminate"


class TestBackendPerformanceComparison:
    """Compare performance characteristics of backends."""

    @pytest.mark.skipif(not HAS_RUST, reason="Requires Rust backend")
    def test_rust_faster_than_python(self):
        """Rust backend should be significantly faster than Python."""
        import time

        np.random.seed(42)
        game = ZertzGame(rings=37)

        # Time Python backend
        py_player = MCTSZertzPlayer(
            game, n=1,
            iterations=100,
            parallel=False,
            verbose=False,
            backend='python'
        )

        start = time.time()
        py_player.get_action()
        py_time = time.time() - start

        # Time Rust backend
        game2 = ZertzGame(rings=37)
        rust_player = MCTSZertzPlayer(
            game2, n=1,
            iterations=100,
            parallel=False,
            verbose=False,
            backend='rust'
        )

        start = time.time()
        rust_action = rust_player.get_action()
        rust_time = time.time() - start

        print(f"\nPython: {py_time:.3f}s, Rust: {rust_time:.3f}s")
        print(f"Speedup: {py_time/rust_time:.1f}x")

        # Rust should be faster (allow 1.5x minimum speedup)
        assert rust_time < py_time / 1.5, \
            f"Rust not significantly faster: {py_time/rust_time:.2f}x speedup"