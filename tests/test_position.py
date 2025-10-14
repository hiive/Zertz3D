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


def test_position_collection_rebuilds_on_clone(small_board):
    original_center = small_board.position_from_label('D4')
    assert small_board.positions._built is True

    clone = ZertzBoard(clone=small_board)
    assert clone.positions._built is False

    clone_center = clone.position_from_label('D4')
    assert clone.positions._built is True

    assert clone_center is not original_center
    assert clone_center.board is clone
    assert clone_center.label == original_center.label
    assert hasattr(clone, "_coord_scale")
    assert clone._coord_scale == pytest.approx(small_board._coord_scale)

    clone.state[clone.RING_LAYER, clone_center.y, clone_center.x] = 0
    assert small_board.state[small_board.RING_LAYER, original_center.y, original_center.x] == 1


def test_clone_all_coordinate_lookups_work():
    """Test that all coordinate lookup methods work correctly on cloned boards."""
    original = ZertzBoard(rings=37)

    # Access positions to build cache
    orig_d4 = original.position_from_label('D4')
    orig_axial = orig_d4.axial
    orig_cart = orig_d4.cartesian

    # Clone the board
    clone = ZertzBoard(clone=original)

    # Test lookup by label
    clone_d4_by_label = clone.position_from_label('D4')
    assert clone_d4_by_label.label == 'D4'
    assert clone_d4_by_label.yx == orig_d4.yx
    assert clone_d4_by_label.board is clone

    # Test lookup by yx
    clone_d4_by_yx = clone.position_from_yx(orig_d4.yx)
    assert clone_d4_by_yx.label == 'D4'
    assert clone_d4_by_yx.yx == orig_d4.yx

    # Test lookup by axial
    clone_d4_by_axial = clone.position_from_axial(orig_axial)
    assert clone_d4_by_axial.label == 'D4'
    assert clone_d4_by_axial.axial == orig_axial

    # Test helper methods
    assert clone.positions.axial_for_yx(orig_d4.yx) == orig_axial
    assert clone.positions.label_for_yx(orig_d4.yx) == 'D4'
    cart_clone = clone.positions.cartesian_for_yx(orig_d4.yx)
    assert cart_clone[0] == pytest.approx(orig_cart[0], abs=1e-6)
    assert cart_clone[1] == pytest.approx(orig_cart[1], abs=1e-6)


def test_clone_position_dictionaries_are_independent():
    """Test that cloned board's position dictionaries don't share state with original."""
    original = ZertzBoard(rings=37)

    # Build original's cache
    original.position_from_label('D4')
    orig_by_label_len = len(original.positions.by_label)
    orig_by_yx_len = len(original.positions.by_yx)
    orig_by_axial_len = len(original.positions.by_axial)

    # Clone and build clone's cache
    clone = ZertzBoard(clone=original)
    clone.position_from_label('D4')

    # Verify dictionaries have same size but are different objects
    assert len(clone.positions.by_label) == orig_by_label_len
    assert len(clone.positions.by_yx) == orig_by_yx_len
    assert len(clone.positions.by_axial) == orig_by_axial_len

    assert clone.positions.by_label is not original.positions.by_label
    assert clone.positions.by_yx is not original.positions.by_yx
    assert clone.positions.by_axial is not original.positions.by_axial
    assert clone.positions.yx_to_ax is not original.positions.yx_to_ax
    assert clone.positions.ax_to_yx is not original.positions.ax_to_yx


def test_clone_with_different_board_sizes():
    """Test cloning works correctly for all supported board sizes."""
    for rings in [37, 48, 61]:
        original = ZertzBoard(rings=rings)

        # Access a position to build cache
        positions = list(original.positions.by_label.keys())
        assert len(positions) == rings
        center_label = positions[rings // 2]
        orig_pos = original.position_from_label(center_label)

        # Clone
        clone = ZertzBoard(clone=original)

        # Verify clone has same number of positions
        assert len(clone.positions.by_label) == rings

        # Verify the same position exists and has correct properties
        clone_pos = clone.position_from_label(center_label)
        assert clone_pos.label == orig_pos.label
        assert clone_pos.yx == orig_pos.yx
        assert clone_pos.axial == orig_pos.axial
        assert clone_pos.board is clone


def test_multiple_rounds_of_cloning():
    """Test that cloning works correctly through multiple generations."""
    gen0 = ZertzBoard(rings=37)

    # Build gen0's cache
    gen0_d4 = gen0.position_from_label('D4')
    gen0_axial = gen0_d4.axial

    # Clone to gen1
    gen1 = ZertzBoard(clone=gen0)
    gen1_d4 = gen1.position_from_label('D4')
    assert gen1_d4.axial == gen0_axial
    assert gen1_d4.board is gen1

    # Clone to gen2
    gen2 = ZertzBoard(clone=gen1)
    gen2_d4 = gen2.position_from_label('D4')
    assert gen2_d4.axial == gen0_axial
    assert gen2_d4.board is gen2

    # All should be independent
    assert gen2_d4 is not gen1_d4
    assert gen1_d4 is not gen0_d4
    assert gen2_d4 is not gen0_d4


def test_clone_after_state_modifications():
    """Test cloning after original board state has been modified."""
    original = ZertzBoard(rings=37)

    # Build cache and get a position
    d4 = original.position_from_label('D4')
    d4_yx = d4.yx

    # Modify the original board's state (remove the ring at D4)
    original.state[original.RING_LAYER, d4.y, d4.x] = 0

    # Clone the board (which should copy the modified state)
    clone = ZertzBoard(clone=original)

    # Position collection should still work on clone
    # The position cache is based on initial ring layout, not current state
    clone_d4 = clone.position_from_label('D4')
    assert clone_d4.yx == d4_yx

    # Verify state was copied correctly
    assert clone.state[clone.RING_LAYER, d4.y, d4.x] == 0

    # Restore ring on clone, should not affect original
    clone.state[clone.RING_LAYER, d4.y, d4.x] = 1
    assert original.state[original.RING_LAYER, d4.y, d4.x] == 0


def test_clone_yx_to_ax_and_ax_to_yx_mappings():
    """Test that axial coordinate mapping dictionaries work correctly on cloned boards."""
    original = ZertzBoard(rings=37)

    # Build cache
    d4 = original.position_from_label('D4')
    d4_yx = d4.yx
    d4_axial = d4.axial

    # Get mapping dictionaries
    orig_yx_to_ax = original.positions.yx_to_ax
    orig_ax_to_yx = original.positions.ax_to_yx

    # Clone
    clone = ZertzBoard(clone=original)

    # Get clone's mapping dictionaries
    clone_yx_to_ax = clone.positions.yx_to_ax
    clone_ax_to_yx = clone.positions.ax_to_yx

    # Verify they're different objects but have same content
    assert clone_yx_to_ax is not orig_yx_to_ax
    assert clone_ax_to_yx is not orig_ax_to_yx

    assert clone_yx_to_ax[d4_yx] == d4_axial
    assert clone_ax_to_yx[d4_axial] == d4_yx

    # Verify all mappings match
    assert len(clone_yx_to_ax) == len(orig_yx_to_ax)
    assert len(clone_ax_to_yx) == len(orig_ax_to_yx)

    for yx, axial in orig_yx_to_ax.items():
        assert clone_yx_to_ax[yx] == axial
        assert clone_ax_to_yx[axial] == yx
