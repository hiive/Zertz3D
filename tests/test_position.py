import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_board import ZertzBoard


@pytest.fixture
def small_board():
    return ZertzBoard(rings=37)


def test_position_from_label_center(small_board):
    pos = small_board.position_from_label('D4')
    assert pos.yx == (3, 3)
    assert pos.label == 'D4'
    assert pos.axial == (0, 0)
    assert pos.cartesian[0] == pytest.approx(0.0, abs=1e-6)
    assert pos.cartesian[1] == pytest.approx(0.0, abs=1e-6)


def test_position_roundtrip_string_and_axial(small_board):
    label = 'A1'
    pos = small_board.position_from_label(label)
    assert small_board.position_from_yx(pos.yx).label == label
    assert small_board.position_from_axial(pos.axial).label == label


def test_position_available_after_ring_removed(small_board):
    pos = small_board.position_from_label('D4')
    small_board.state[small_board.RING_LAYER, pos.y, pos.x] = 0
    # index_to_str returns '' for removed rings, but Position retains label
    assert small_board.index_to_str(pos.yx) == ''
    new_pos = small_board.position_from_yx(pos.yx)
    assert new_pos.label == 'D4'


def test_position_from_axial_roundtrip(small_board):
    center = small_board.position_from_label('D4')
    axial = center.axial
    pos = small_board.position_from_axial(axial)
    assert pos.yx == center.yx
    assert pos.label == 'D4'


def test_position_collection_lazy_build_and_size(small_board):
    collection = small_board.positions
    assert collection._built is False

    pos = collection.get_by_label('D4')
    assert collection._built is True
    assert pos.yx == (3, 3)
    assert len(collection.by_label) == small_board.rings
    assert len(collection.by_yx) == small_board.rings
    assert len(collection.by_axial) == small_board.rings


def test_position_collection_invalidate_rebuilds(small_board):
    collection = small_board.positions
    original = collection.get_by_yx((3, 3))
    assert original.label == 'D4'

    collection.invalidate()
    assert collection._built is False

    rebuilt = collection.get_by_yx((3, 3))
    assert collection._built is True
    assert rebuilt.label == 'D4'
    assert rebuilt is not original


def test_position_collection_axial_and_cartesian_helpers(small_board):
    collection = small_board.positions
    center = collection.get_by_label('D4')

    axial = collection.axial_for_yx(center.yx)
    cart = collection.cartesian_for_yx(center.yx)

    assert axial == center.axial
    assert cart == center.cartesian
