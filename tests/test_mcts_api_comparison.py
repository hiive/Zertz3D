"""
API Comparison tests between Python and Rust MCTS implementations.

These tests ensure that the public APIs are identical and return compatible data.

NOTE: These tests will become obsolete once the Python MCTS implementation is removed.
They are marked with @pytest.mark.backend_comparison for easy identification and removal.
"""

import pytest
import numpy as np
from game.zertz_game import ZertzGame
from learner.mcts.backend import HAS_RUST

# Skip all tests if Rust backend not available
# Also mark as backend_comparison for eventual removal when Python MCTS is deprecated
pytestmark = [
    pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available"),
    pytest.mark.backend_comparison
]


class TestMCTSSearchAPI:
    """Test that MCTSSearch has identical public API in both backends."""

    def test_search_method_signature(self):
        """Both backends should have identical search() signature."""
        import hiivelabs_zertz_mcts
        from learner.mcts.mcts_tree import MCTSTree

        # Create instances
        rust_search = hiivelabs_zertz_mcts.MCTSSearch()
        py_tree = MCTSTree()

        # Check method exists
        assert hasattr(rust_search, 'search')
        assert callable(rust_search.search)

        # Python uses different structure, but MCTSZertzPlayer wraps it
        # So we test through the player interface

    def test_search_returns_compatible_actions(self):
        """Both backends should return actions in the same format."""
        import hiivelabs_zertz_mcts

        np.random.seed(42)
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        rust_search = hiivelabs_zertz_mcts.MCTSSearch()
        action_type, action_data = rust_search.search(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            rings=37,
            iterations=10,
            t=1,
            verbose=False
        )

        # Check return format
        assert isinstance(action_type, str)
        assert action_type in ["PUT", "CAP", "PASS"]

        if action_type == "PUT":
            assert isinstance(action_data, tuple)
            assert len(action_data) == 3
            marble_type, dst_flat, remove_flat = action_data

            # Validate indices are in valid range
            assert 0 <= marble_type < 3
            assert 0 <= dst_flat < 49  # 7x7 = 49
            assert 0 <= remove_flat <= 49  # 49 positions + 1 for no-removal

        elif action_type == "CAP":
            assert isinstance(action_data, tuple)
            assert len(action_data) == 3
            direction, start_y, start_x = action_data
            assert 0 <= direction < 6
            assert 0 <= start_y < 7
            assert 0 <= start_x < 7

    def test_action_data_matches_valid_actions(self):
        """Returned actions should match positions in valid action masks."""
        import hiivelabs_zertz_mcts

        np.random.seed(42)
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        # Get valid actions from game
        placement_mask, capture_mask = game.get_valid_actions()

        # Get action from Rust MCTS
        rust_search = hiivelabs_zertz_mcts.MCTSSearch()
        action_type, action_data = rust_search.search(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            rings=37,
            iterations=50,
            t=1,
            verbose=False
        )

        # Verify action is valid
        if action_type == "PUT":
            marble_type, dst_flat, remove_flat = action_data
            assert placement_mask[marble_type, dst_flat, remove_flat] > 0, \
                f"Rust returned invalid placement: marble={marble_type}, dst={dst_flat}, remove={remove_flat}"

            # Convert to (y, x) and verify it's a valid ring position
            width = game.board.width
            dst_y = dst_flat // width
            dst_x = dst_flat % width

            # Check this is actually a valid ring position
            assert game.board.state[0, dst_y, dst_x] == 1, \
                f"Destination ({dst_y}, {dst_x}) is not a valid ring position"

        elif action_type == "CAP":
            direction, start_y, start_x = action_data
            assert capture_mask[direction, start_y, start_x] > 0, \
                f"Rust returned invalid capture: dir={direction}, start=({start_y}, {start_x})"

    def test_search_accepts_optional_parameters(self):
        """Rust search should accept optional parameters matching Python API."""
        import hiivelabs_zertz_mcts

        np.random.seed(0)
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        rust_search = hiivelabs_zertz_mcts.MCTSSearch()
        rust_search.set_transposition_table_enabled(True)
        rust_search.set_transposition_lookups(True)
        rust_search.clear_transposition_table()

        # Serial search with optional arguments
        action = rust_search.search(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            rings=37,
            iterations=5,
            t=1,
            max_depth=4,
            time_limit=0.01,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table=False,
            verbose=False,
        )
        assert isinstance(action, tuple)

        # Parallel search with optional arguments
        action_parallel = rust_search.search_parallel(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            rings=37,
            iterations=5,
            t=1,
            max_depth=3,
            time_limit=0.01,
            use_transposition_table=True,
            use_transposition_lookups=True,
            clear_table=False,
            num_threads=2,
            verbose=False,
        )
        assert isinstance(action_parallel, tuple)


class TestNodeAPI:
    """Test that MCTSNode has identical public API in both backends."""

    def test_node_creation(self):
        """Both backends should create nodes with same interface."""
        import hiivelabs_zertz_mcts

        np.random.seed(42)
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        # Rust node creation
        rust_node = hiivelabs_zertz_mcts.BoardState(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            rings=37,
            t=1
        )

        # Check it has expected attributes
        assert hasattr(rust_node, 'spatial')
        assert hasattr(rust_node, 'global')

    def test_node_get_valid_actions(self):
        """Both backends should return identical valid actions."""
        import hiivelabs_zertz_mcts

        np.random.seed(42)
        game = ZertzGame(rings=37)
        state = game.get_current_state()

        # Python valid actions
        py_placement, py_capture = game.get_valid_actions()

        # Rust valid actions
        rust_node = hiivelabs_zertz_mcts.BoardState(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            rings=37,
            t=1
        )
        rust_placement, rust_capture = rust_node.get_valid_actions()

        # Compare
        np.testing.assert_array_equal(py_placement, rust_placement)
        np.testing.assert_array_equal(py_capture, rust_capture)


class TestCoordinateConversion:
    """Test coordinate conversions between flattened and (y, x) formats."""

    def test_flattened_to_yx_conversion(self):
        """Test that flattened indices convert to valid (y, x) coordinates."""
        np.random.seed(42)
        game = ZertzGame(rings=37)
        width = game.board.width  # 7

        # Test all valid ring positions
        for y in range(width):
            for x in range(width):
                if game.board.state[0, y, x] == 1:  # Valid ring
                    # Convert to flattened
                    flat = y * width + x

                    # Convert back
                    y_back = flat // width
                    x_back = flat % width

                    assert y == y_back
                    assert x == x_back

                    # Verify position_from_yx doesn't crash
                    try:
                        pos = game.board.position_from_yx((y, x))
                        # Should succeed for valid ring
                    except ValueError:
                        pytest.fail(f"position_from_yx failed for valid ring at ({y}, {x})")

    def test_placement_action_coordinates(self):
        """Test that placement actions use valid coordinates."""
        import hiivelabs_zertz_mcts

        for seed in [42, 123, 456]:
            np.random.seed(seed)
            game = ZertzGame(rings=37)
            state = game.get_current_state()
            width = game.board.width

            rust_search = hiivelabs_zertz_mcts.MCTSSearch()
            action_type, action_data = rust_search.search(
                state['spatial'].astype(np.float32),
                state['global'].astype(np.float32),
                rings=37,
                iterations=100,
                t=1,
                verbose=False
            )

            if action_type == "PUT":
                marble_type, dst_flat, remove_flat = action_data

                # Convert to (y, x)
                dst_y = dst_flat // width
                dst_x = dst_flat % width

                # Verify these are valid ring positions
                assert game.board.state[0, dst_y, dst_x] == 1, \
                    f"Seed {seed}: Destination ({dst_y}, {dst_x}) from flat={dst_flat} is not a valid ring"

                if remove_flat < width * width:
                    remove_y = remove_flat // width
                    remove_x = remove_flat % width
                    assert game.board.state[0, remove_y, remove_x] == 1, \
                        f"Seed {seed}: Removal ({remove_y}, {remove_x}) from flat={remove_flat} is not a valid ring"


class TestMultipleGames:
    """Test MCTS across multiple game scenarios to catch edge cases."""

    def test_mcts_on_multiple_seeds(self):
        """Run MCTS on multiple random seeds to find coordinate issues."""
        import hiivelabs_zertz_mcts

        failed_seeds = []

        for seed in range(10):
            try:
                np.random.seed(seed)
                game = ZertzGame(rings=37)
                state = game.get_current_state()

                rust_search = hiivelabs_zertz_mcts.MCTSSearch()
                action_type, action_data = rust_search.search(
                    state['spatial'].astype(np.float32),
                    state['global'].astype(np.float32),
                    rings=37,
                    iterations=100,
                    t=1,
                    verbose=False
                )

                # Verify action is valid
                if action_type == "PUT":
                    marble_type, dst_flat, remove_flat = action_data
                    placement_mask, _ = game.get_valid_actions()

                    if placement_mask[marble_type, dst_flat, remove_flat] <= 0:
                        failed_seeds.append((seed, "invalid placement mask"))

                    # Convert and verify coordinates
                    width = game.board.width
                    dst_y = dst_flat // width
                    dst_x = dst_flat % width

                    if game.board.state[0, dst_y, dst_x] != 1:
                        failed_seeds.append((seed, f"invalid dest ring ({dst_y}, {dst_x})"))

                elif action_type == "CAP":
                    direction, start_y, start_x = action_data
                    _, capture_mask = game.get_valid_actions()

                    if capture_mask[direction, start_y, start_x] <= 0:
                        failed_seeds.append((seed, "invalid capture mask"))

            except Exception as e:
                failed_seeds.append((seed, str(e)))

        assert len(failed_seeds) == 0, f"Failed on seeds: {failed_seeds}"

    def test_mcts_returns_valid_positions_after_moves(self):
        """Test MCTS returns valid actions after several moves."""
        import hiivelabs_zertz_mcts

        np.random.seed(1760910995)  # Seed from crash log
        game = ZertzGame(rings=37)

        # Play first move (from crash log): PUT w at E1, remove A4
        # Convert positions to flattened indices
        width = game.board.width
        e1_yx = game.board.str_to_index("E1")
        a4_yx = game.board.str_to_index("A4")
        e1_flat = e1_yx[0] * width + e1_yx[1]
        a4_flat = a4_yx[0] * width + a4_yx[1]

        game.take_action("PUT", (0, e1_flat, a4_flat))

        # Now get MCTS action for player 2
        state = game.get_current_state()
        rust_search = hiivelabs_zertz_mcts.MCTSSearch()
        action_type, action_data = rust_search.search(
            state['spatial'].astype(np.float32),
            state['global'].astype(np.float32),
            rings=37,
            iterations=100,
            t=1,
            verbose=False
        )

        # Verify the action is valid
        if action_type == "PUT":
            marble_type, dst_flat, remove_flat = action_data
            placement_mask, _ = game.get_valid_actions()

            assert placement_mask[marble_type, dst_flat, remove_flat] > 0, \
                f"Invalid placement after first move: marble={marble_type}, dst={dst_flat}, remove={remove_flat}"

            # Convert to coordinates and verify
            width = game.board.width
            dst_y = dst_flat // width
            dst_x = dst_flat % width

            # This should not raise ValueError
            try:
                pos = game.board.position_from_yx((dst_y, dst_x))
            except ValueError as e:
                pytest.fail(f"position_from_yx failed: {e}, coords=({dst_y}, {dst_x}), flat={dst_flat}")
