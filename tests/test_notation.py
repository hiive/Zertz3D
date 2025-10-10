"""
Unit tests for Zèrtz notation conversion.

Tests the action_to_notation() method which converts action dictionaries
to official Zèrtz notation format as specified at:
http://www.gipf.com/zertz/notations/notation.html
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import game modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.zertz_game import ZertzGame


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def game():
    """Create a standard 37-ring game for testing."""
    return ZertzGame(rings=37)


# ============================================================================
# PUT Action Tests (Placement)
# ============================================================================

def test_put_white_no_removal(game):
    """Test placement of white marble without ring removal."""
    action_dict = {
        'action': 'PUT',
        'marble': 'w',
        'dst': 'D4',
        'remove': ''
    }
    assert game.action_to_notation(action_dict) == 'Wd4'


def test_put_gray_no_removal(game):
    """Test placement of gray marble without ring removal."""
    action_dict = {
        'action': 'PUT',
        'marble': 'g',
        'dst': 'C3',
        'remove': ''
    }
    assert game.action_to_notation(action_dict) == 'Gc3'


def test_put_black_no_removal(game):
    """Test placement of black marble without ring removal."""
    action_dict = {
        'action': 'PUT',
        'marble': 'b',
        'dst': 'E5',
        'remove': ''
    }
    assert game.action_to_notation(action_dict) == 'Be5'


def test_put_with_removal(game):
    """Test placement with ring removal."""
    action_dict = {
        'action': 'PUT',
        'marble': 'b',
        'dst': 'D7',
        'remove': 'B2'
    }
    assert game.action_to_notation(action_dict) == 'Bd7,b2'


def test_put_white_with_removal(game):
    """Test white marble placement with ring removal."""
    action_dict = {
        'action': 'PUT',
        'marble': 'w',
        'dst': 'F6',
        'remove': 'C1'
    }
    assert game.action_to_notation(action_dict) == 'Wf6,c1'


def test_put_gray_with_removal(game):
    """Test gray marble placement with ring removal."""
    action_dict = {
        'action': 'PUT',
        'marble': 'g',
        'dst': 'A3',
        'remove': 'G7'
    }
    assert game.action_to_notation(action_dict) == 'Ga3,g7'


# ============================================================================
# CAP Action Tests (Capture)
# ============================================================================

def test_cap_white_marble(game):
    """Test capture of white marble."""
    action_dict = {
        'action': 'CAP',
        'marble': 'b',
        'src': 'E3',
        'dst': 'G3',
        'capture': 'w',
        'cap': 'F3'
    }
    assert game.action_to_notation(action_dict) == 'x e3Wg3'


def test_cap_gray_marble(game):
    """Test capture of gray marble."""
    action_dict = {
        'action': 'CAP',
        'marble': 'w',
        'src': 'D5',
        'dst': 'D3',
        'capture': 'g',
        'cap': 'D4'
    }
    assert game.action_to_notation(action_dict) == 'x d5Gd3'


def test_cap_black_marble(game):
    """Test capture of black marble."""
    action_dict = {
        'action': 'CAP',
        'marble': 'g',
        'src': 'C2',
        'dst': 'E4',
        'capture': 'b',
        'cap': 'D3'
    }
    assert game.action_to_notation(action_dict) == 'x c2Be4'


def test_cap_uppercase_coordinates(game):
    """Test that capture coordinates are properly lowercased."""
    action_dict = {
        'action': 'CAP',
        'marble': 'w',
        'src': 'F5',
        'dst': 'F7',
        'capture': 'w',
        'cap': 'F6'
    }
    assert game.action_to_notation(action_dict) == 'x f5Wf7'


# ============================================================================
# PASS Action Tests
# ============================================================================

def test_pass_action(game):
    """Test PASS action notation."""
    action_dict = {
        'action': 'PASS'
    }
    assert game.action_to_notation(action_dict) == '-'


# ============================================================================
# Case Handling Tests
# ============================================================================

def test_marble_color_uppercase(game):
    """Test that marble colors are uppercased in notation."""
    # Even if input is lowercase, output should be uppercase
    action_dict = {
        'action': 'PUT',
        'marble': 'w',  # lowercase input
        'dst': 'D4',
        'remove': ''
    }
    notation = game.action_to_notation(action_dict)
    assert notation == 'Wd4'
    assert notation[0].isupper()  # First char (marble) is uppercase


def test_coordinates_lowercase(game):
    """Test that coordinates are lowercased in notation."""
    # Even if input is uppercase, output should be lowercase
    action_dict = {
        'action': 'PUT',
        'marble': 'b',
        'dst': 'D4',  # uppercase input
        'remove': 'B2'  # uppercase input
    }
    notation = game.action_to_notation(action_dict)
    assert notation == 'Bd4,b2'
    # All coordinate chars should be lowercase
    assert notation[1:3].islower()  # d4
    assert notation[4:6].islower()  # b2


def test_capture_coordinates_lowercase(game):
    """Test that capture coordinates are lowercased."""
    action_dict = {
        'action': 'CAP',
        'marble': 'b',
        'src': 'E3',  # uppercase input
        'dst': 'G3',  # uppercase input
        'capture': 'w',
        'cap': 'F3'
    }
    notation = game.action_to_notation(action_dict)
    assert notation == 'x e3Wg3'
    # Source and destination should be lowercase
    assert notation[2:4].islower()  # e3
    assert notation[5:7].islower()  # g3


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_removal_string(game):
    """Test that empty removal string produces no comma."""
    action_dict = {
        'action': 'PUT',
        'marble': 'w',
        'dst': 'A1',
        'remove': ''
    }
    notation = game.action_to_notation(action_dict)
    assert notation == 'Wa1'
    assert ',' not in notation


def test_none_removal(game):
    """Test handling of None as removal value."""
    action_dict = {
        'action': 'PUT',
        'marble': 'g',
        'dst': 'B2',
        'remove': None
    }
    notation = game.action_to_notation(action_dict)
    # Should handle None gracefully - either empty string or no comma
    assert notation == 'Gb2' or notation == 'Gb2,'


def test_various_coordinates(game):
    """Test notation with various coordinate positions."""
    test_cases = [
        ('A1', 'Wa1'),
        ('G7', 'Wg7'),
        ('D4', 'Wd4'),
        ('B5', 'Wb5'),
        ('F2', 'Wf2'),
    ]

    for coord, expected in test_cases:
        action_dict = {
            'action': 'PUT',
            'marble': 'w',
            'dst': coord,
            'remove': ''
        }
        assert game.action_to_notation(action_dict) == expected


# ============================================================================
# Parameterized Tests
# ============================================================================

@pytest.mark.parametrize("marble,expected_prefix", [
    ('w', 'W'),
    ('g', 'G'),
    ('b', 'B'),
])
def test_all_marble_colors(game, marble, expected_prefix):
    """Test notation for all marble colors."""
    action_dict = {
        'action': 'PUT',
        'marble': marble,
        'dst': 'D4',
        'remove': ''
    }
    notation = game.action_to_notation(action_dict)
    assert notation.startswith(expected_prefix)
    assert notation == f'{expected_prefix}d4'


@pytest.mark.parametrize("capture_color,expected_char", [
    ('w', 'W'),
    ('g', 'G'),
    ('b', 'B'),
])
def test_all_capture_colors(game, capture_color, expected_char):
    """Test notation for capturing all marble colors."""
    action_dict = {
        'action': 'CAP',
        'marble': 'w',
        'src': 'C3',
        'dst': 'E3',
        'capture': capture_color,
        'cap': 'D3'
    }
    notation = game.action_to_notation(action_dict)
    assert expected_char in notation
    assert notation == f'x c3{expected_char}e3'


# ============================================================================
# Isolation Tests (for Temporal Coupling Issue 2)
# ============================================================================

def test_put_with_single_marble_isolation(game):
    """Test placement with single marble isolation."""
    action_dict = {
        'action': 'PUT',
        'marble': 'b',
        'dst': 'D7',
        'remove': 'B2'
    }

    # Isolation result: single marble captured
    isolation_result = [{'marble': 'w', 'pos': 'A1'}]

    notation = game.action_to_notation(action_dict, isolation_result)
    assert notation == 'Bd7,b2 x Wa1'


def test_put_with_multiple_marble_isolation(game):
    """Test placement with multiple marble isolation."""
    action_dict = {
        'action': 'PUT',
        'marble': 'b',
        'dst': 'D7',
        'remove': 'B2'
    }

    # Isolation result: two marbles captured
    isolation_result = [
        {'marble': 'w', 'pos': 'A1'},
        {'marble': 'w', 'pos': 'A2'}
    ]

    notation = game.action_to_notation(action_dict, isolation_result)
    assert notation == 'Bd7,b2 x Wa1Wa2'


def test_put_with_mixed_color_isolation(game):
    """Test placement isolating marbles of different colors."""
    action_dict = {
        'action': 'PUT',
        'marble': 'g',
        'dst': 'E5',
        'remove': 'C3'
    }

    # Isolation result: different colored marbles
    isolation_result = [
        {'marble': 'w', 'pos': 'A1'},
        {'marble': 'g', 'pos': 'B2'},
        {'marble': 'b', 'pos': 'C1'}
    ]

    notation = game.action_to_notation(action_dict, isolation_result)
    assert notation == 'Ge5,c3 x Wa1Gb2Bc1'


def test_put_no_removal_with_isolation(game):
    """Test placement without ring removal but with isolation."""
    action_dict = {
        'action': 'PUT',
        'marble': 'w',
        'dst': 'D4',
        'remove': ''
    }

    # Isolation result
    isolation_result = [{'marble': 'b', 'pos': 'G1'}]

    notation = game.action_to_notation(action_dict, isolation_result)
    assert notation == 'Wd4 x Bg1'


def test_put_with_empty_isolation_list(game):
    """Test that empty isolation list doesn't add 'x' notation."""
    action_dict = {
        'action': 'PUT',
        'marble': 'w',
        'dst': 'D4',
        'remove': 'B2'
    }

    # Empty isolation list
    isolation_result = []

    notation = game.action_to_notation(action_dict, isolation_result)
    assert notation == 'Wd4,b2'
    assert ' x ' not in notation


def test_put_with_none_marble_in_isolation(game):
    """Test isolation result with None marble (vacant ring removed)."""
    action_dict = {
        'action': 'PUT',
        'marble': 'b',
        'dst': 'D7',
        'remove': 'B2'
    }

    # Isolation result: vacant ring (marble is None) should not appear in notation
    isolation_result = [
        {'marble': 'w', 'pos': 'A1'},
        {'marble': None, 'pos': 'A2'},  # Vacant ring
        {'marble': 'g', 'pos': 'B1'}
    ]

    notation = game.action_to_notation(action_dict, isolation_result)
    # Should only include marbles with actual colors
    assert notation == 'Bd7,b2 x Wa1Gb1'


def test_isolation_coordinates_lowercase(game):
    """Test that isolation coordinates are lowercased."""
    action_dict = {
        'action': 'PUT',
        'marble': 'w',
        'dst': 'D4',
        'remove': 'B2'
    }

    # Isolation with uppercase coordinates (should be lowercased)
    isolation_result = [
        {'marble': 'w', 'pos': 'A1'},
        {'marble': 'G', 'pos': 'B2'}
    ]

    notation = game.action_to_notation(action_dict, isolation_result)
    # All positions should be lowercase
    assert 'a1' in notation.lower()
    assert 'b2' in notation.lower()


def test_capture_action_ignores_isolation(game):
    """Test that CAP actions don't use isolation_result (captures don't cause isolation)."""
    action_dict = {
        'action': 'CAP',
        'marble': 'b',
        'src': 'E3',
        'dst': 'G3',
        'capture': 'w',
        'cap': 'F3'
    }

    # Isolation result should be ignored for captures
    isolation_result = [{'marble': 'w', 'pos': 'A1'}]

    notation = game.action_to_notation(action_dict, isolation_result)
    # Should be normal capture notation without isolation
    assert notation == 'x e3Wg3'
    assert ' x ' not in notation[1:]  # No second 'x' for isolation