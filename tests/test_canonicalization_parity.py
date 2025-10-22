import numpy as np
import pytest

hiive_module = pytest.importorskip("hiivelabs_zertz_mcts")

from game.zertz_board import ZertzBoard  # noqa: E402


def _build_sample_state(board: ZertzBoard):
    layout = board.board_layout
    width = board.width
    positions = None

    for y in range(1, width - 1):
        for x in range(1, width - 1):
            if (
                layout[y, x]
                and layout[y, x + 1]
                and layout[y + 1, x]
                and layout[y - 1, x]
            ):
                positions = [(y, x), (y, x + 1), (y + 1, x)]
                shift = (-1, 0)
                # Ensure translation remains on board
                valid_translation = True
                for py, px in positions:
                    ty = py + shift[0]
                    tx = px + shift[1]
                    if ty < 0 or tx < 0 or ty >= width or tx >= width or not layout[ty, tx]:
                        valid_translation = False
                        break
                if valid_translation:
                    return positions, shift

    raise AssertionError("Failed to find interior positions for sample state")


@pytest.mark.parametrize("rings", [37, 48, 61])
def test_canonicalization_matches_python(rings):
    board = ZertzBoard(rings=rings)
    positions, translation = _build_sample_state(board)

    base_state = np.zeros_like(board.state, dtype=np.float32)
    for y, x in positions:
        base_state[board.RING_LAYER, y, x] = 1.0
    base_state[board.MARBLE_LAYERS.start, positions[0][0], positions[0][1]] = 1.0
    base_state[board.MARBLE_LAYERS.start + 1, positions[1][0], positions[1][1]] = 1.0

    translation_name = f"T{translation[0]},{translation[1]}"
    transforms = ["R0", "R60", "MR0", "R60M", translation_name]
    transforms += [f"{translation_name}_R60", f"{translation_name}_MR0"]

    global_state = np.zeros(10, dtype=np.float32)
    t = board.t
    checked = False

    for transform in transforms:
        try:
            variant = board.canonicalizer._apply_transform(base_state, transform)
        except ValueError:
            continue
        variant = np.array(variant, dtype=np.float32, copy=True)
        checked = True

        py_canonical, py_transform, py_inverse = board.canonicalizer.canonicalize_state(variant)
        py_canonical = np.array(py_canonical, dtype=np.float32, copy=False)

        rust_board = hiive_module.BoardState(variant, global_state, rings, t)
        rust_canonical, rust_transform, rust_inverse = rust_board.canonicalize_state()
        rust_canonical = np.array(rust_canonical, dtype=np.float32, copy=False)

        assert np.array_equal(py_canonical, rust_canonical)
        assert rust_transform == py_transform
        assert rust_inverse == py_inverse

    assert checked, "No valid transforms evaluated for canonicalization parity"
