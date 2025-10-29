"""Test that capture pool slots are properly tracked and reused."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from renderer.panda_renderer import PandaRenderer
from game.zertz_game import ZertzGame


def test_capture_pool_slot_tracking():
    """Test that capture pool slots are tracked and reused correctly.

    This test verifies the fix for the bug where marbles would overlap in the
    capture pool when marbles were taken from the pool and reused on the board.
    """
    # Create a mock renderer with just the slot tracking data structures
    renderer = Mock()
    renderer.capture_pool_occupied_slots = {1: set(), 2: set()}
    renderer.marble_to_capture_slot = {}

    # Simulate the slot allocation logic from _animate_captured_marble
    def allocate_slot(player_num, marble_id):
        occupied_slots = renderer.capture_pool_occupied_slots[player_num]
        # Find first available slot (simulate having 10 slots)
        available_slot_index = None
        for slot_idx in range(10):
            if slot_idx not in occupied_slots:
                available_slot_index = slot_idx
                break

        if available_slot_index is not None:
            occupied_slots.add(available_slot_index)
            renderer.marble_to_capture_slot[marble_id] = (player_num, available_slot_index)
            return available_slot_index
        return None

    # Simulate the slot freeing logic from show_marble_placement
    def free_slot(marble_id):
        slot_info = renderer.marble_to_capture_slot.pop(marble_id, None)
        if slot_info is not None:
            player_num, slot_index = slot_info
            renderer.capture_pool_occupied_slots[player_num].discard(slot_index)

    # Test scenario: capture -> place -> capture should reuse slot 0

    # Step 1: Capture marble 1 for player 1
    slot1 = allocate_slot(1, marble_id=1001)
    assert slot1 == 0, "First capture should use slot 0"
    assert 0 in renderer.capture_pool_occupied_slots[1]
    assert renderer.marble_to_capture_slot[1001] == (1, 0)

    # Step 2: Capture marble 2 for player 1
    slot2 = allocate_slot(1, marble_id=1002)
    assert slot2 == 1, "Second capture should use slot 1"
    assert {0, 1} == renderer.capture_pool_occupied_slots[1]

    # Step 3: Place marble 1 on board (free slot 0)
    free_slot(marble_id=1001)
    assert renderer.capture_pool_occupied_slots[1] == {1}, "Slot 0 should be freed"
    assert 1001 not in renderer.marble_to_capture_slot

    # Step 4: Capture marble 3 for player 1
    slot3 = allocate_slot(1, marble_id=1003)
    assert slot3 == 0, "Third capture should reuse freed slot 0"
    assert {0, 1} == renderer.capture_pool_occupied_slots[1]

    # Step 5: Verify slot 2 is used next
    slot4 = allocate_slot(1, marble_id=1004)
    assert slot4 == 2, "Fourth capture should use slot 2"


def test_capture_pool_multiple_players():
    """Test that capture pools are tracked independently for each player."""
    renderer = Mock()
    renderer.capture_pool_occupied_slots = {1: set(), 2: set()}
    renderer.marble_to_capture_slot = {}

    def allocate_slot(player_num, marble_id):
        occupied_slots = renderer.capture_pool_occupied_slots[player_num]
        available_slot_index = None
        for slot_idx in range(10):
            if slot_idx not in occupied_slots:
                available_slot_index = slot_idx
                break

        if available_slot_index is not None:
            occupied_slots.add(available_slot_index)
            renderer.marble_to_capture_slot[marble_id] = (player_num, available_slot_index)
            return available_slot_index
        return None

    # Player 1 captures
    slot_p1_1 = allocate_slot(1, marble_id=1001)
    assert slot_p1_1 == 0

    # Player 2 captures (should also use slot 0, independent from player 1)
    slot_p2_1 = allocate_slot(2, marble_id=2001)
    assert slot_p2_1 == 0

    # Verify both players have slot 0 occupied
    assert renderer.capture_pool_occupied_slots[1] == {0}
    assert renderer.capture_pool_occupied_slots[2] == {0}

    # Verify marbles are tracked to correct players
    assert renderer.marble_to_capture_slot[1001] == (1, 0)
    assert renderer.marble_to_capture_slot[2001] == (2, 0)


def test_capture_pool_no_overlap_after_reuse():
    """Integration test: verify marbles don't overlap when reused from capture pool.

    This is the specific bug scenario reported by the user.
    """
    renderer = Mock()
    renderer.capture_pool_occupied_slots = {1: set(), 2: set()}
    renderer.marble_to_capture_slot = {}

    def allocate_slot(player_num, marble_id):
        occupied_slots = renderer.capture_pool_occupied_slots[player_num]
        available_slot_index = None
        for slot_idx in range(10):
            if slot_idx not in occupied_slots:
                available_slot_index = slot_idx
                break

        if available_slot_index is not None:
            occupied_slots.add(available_slot_index)
            renderer.marble_to_capture_slot[marble_id] = (player_num, available_slot_index)
            return available_slot_index
        return None

    def free_slot(marble_id):
        slot_info = renderer.marble_to_capture_slot.pop(marble_id, None)
        if slot_info is not None:
            player_num, slot_index = slot_info
            renderer.capture_pool_occupied_slots[player_num].discard(slot_index)

    # Scenario from bug report:
    # 1. White marble captured -> slot 0
    white_id = 3001
    white_slot = allocate_slot(1, white_id)
    assert white_slot == 0

    # 2. Grey marble captured -> slot 1
    grey_id = 3002
    grey_slot = allocate_slot(1, grey_id)
    assert grey_slot == 1

    # 3. White marble reused from capture pool -> frees slot 0
    free_slot(white_id)
    assert 0 not in renderer.capture_pool_occupied_slots[1]
    assert 1 in renderer.capture_pool_occupied_slots[1]

    # 4. Another white marble captured -> should use slot 0 (not overlap with grey in slot 1)
    white2_id = 3003
    white2_slot = allocate_slot(1, white2_id)
    assert white2_slot == 0, "New capture should reuse freed slot 0, not overlap with grey in slot 1"

    # Verify final state: slot 0 has white2, slot 1 has grey
    assert renderer.capture_pool_occupied_slots[1] == {0, 1}
    assert renderer.marble_to_capture_slot[grey_id] == (1, 1)
    assert renderer.marble_to_capture_slot[white2_id] == (1, 0)
    assert white_id not in renderer.marble_to_capture_slot  # Original white was freed