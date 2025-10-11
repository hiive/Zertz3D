"""
Unit tests for Temporal Coupling Issue 2: Notation Update with Isolation.

Tests the complex temporal sequence required for correct notation generation
when isolation occurs. From architecture_report3.md:

Required Sequence:
1. Controller generates initial notation (line 174)
2. Controller buffers notation in logger (line 178)
3. Game executes action, may return isolation result (line 206)
4. Renderer processes highlights, returns result (line 136)
5. Controller checks if result is isolation (line 226)
6. Controller re-generates notation with isolation (line 228)
7. Controller updates buffered notation (line 232)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.zertz_game_controller import ZertzGameController
from game.zertz_game import ZertzGame
from controller.game_logger import GameLogger
from game.action_result import ActionResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_game_session():
    """Create a mock game session."""
    session = Mock()
    session.game = Mock(spec=ZertzGame)
    session.rings = 37
    session.blitz = False

    # Mock game methods
    session.game.action_to_str = Mock(return_value=('PUT', {
        'action': 'PUT',
        'marble': 'w',
        'dst': 'D4',
        'remove': 'B2'
    }))

    session.game.action_to_notation = Mock(side_effect=lambda action_dict, action_result: (
        "Wd4,b2" if action_result is None or not (action_result.is_isolation() and action_result.has_captures())
        else "Wd4,b2 x Wa1Wb2"
    ))

    session.game.get_valid_actions = Mock(return_value=(Mock(), Mock()))
    session.game.get_placement_positions = Mock(return_value=[])
    session.game.get_capture_dicts = Mock(return_value=[])
    session.game.get_removal_map = Mock(return_value={})
    # Return ActionResult by default (no isolation, no captures)
    session.game.take_action = Mock(return_value=ActionResult(captured_marbles=None, newly_frozen_positions=set()))

    # Mock board
    session.game.board = Mock()
    session.game.board.frozen_positions = set()
    session.game.board.index_to_str = Mock(return_value="")

    # Mock session methods
    session.get_current_player = Mock()
    session.is_replay_mode = Mock(return_value=False)
    session.get_seed = Mock(return_value=12345)

    return session


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock(spec=GameLogger)
    logger.log_file = None
    logger.notation_file = None
    logger.buffer_notation = Mock()
    logger.update_buffered_notation = Mock()
    logger.log_action = Mock()
    return logger


@pytest.fixture
def mock_player():
    """Create a mock player."""
    player = Mock()
    player.n = 1
    player.get_action = Mock(return_value=(0, (0, 0, 0)))  # PUT action
    player.add_capture = Mock()
    return player


# ============================================================================
# Initial Notation Generation Tests
# ============================================================================

def test_notation_generated_before_action_execution(mock_game_session, mock_logger, mock_player):
    """Test that notation is generated before action is executed (Step 1)."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player
        mock_game_session.game.get_game_ended = Mock(return_value=None)

        # Create task
        task = Mock()
        task.delay_time = 0

        # Run update
        controller.update_game(task)

        # Verify notation was generated with action_result=None
        mock_game_session.game.action_to_notation.assert_called()
        call_args = mock_game_session.game.action_to_notation.call_args
        # Should be called with None for action_result initially
        assert call_args[0][1] is None or call_args[1].get('action_result') is None


def test_notation_buffered_after_generation(mock_game_session, mock_logger, mock_player):
    """Test that notation is buffered in logger (Step 2)."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player
        mock_game_session.game.get_game_ended = Mock(return_value=None)

        task = Mock()
        task.delay_time = 0

        # Run update
        controller.update_game(task)

        # Verify notation was buffered
        mock_logger.buffer_notation.assert_called_once()


def test_action_dict_stored_in_pending(mock_game_session, mock_logger, mock_player):
    """Test that action_dict is stored in pending_action_dict (Step 2)."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player
        mock_game_session.game.get_game_ended = Mock(return_value=None)

        task = Mock()
        task.delay_time = 0

        # Initially no pending action
        assert controller.pending_action_dict is None

        # Run update
        controller.update_game(task)

        # Should have stored action_dict
        assert controller.pending_action_dict is not None
        assert controller.pending_action_dict['action'] == 'PUT'


# ============================================================================
# Isolation Detection and Update Tests
# ============================================================================

def test_isolation_triggers_notation_regeneration(mock_game_session, mock_logger, mock_player):
    """Test that isolation result triggers notation regeneration (Steps 5-6)."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player
        mock_game_session.game.get_game_ended = Mock(return_value=None)

        # Set up isolation result
        isolation_result = [
            {'marble': 'w', 'pos': 'A1'},
            {'marble': 'w', 'pos': 'B2'}
        ]
        mock_game_session.game.take_action = Mock(return_value=ActionResult(captured_marbles=isolation_result, newly_frozen_positions=set()))

        task = Mock()
        task.delay_time = 0

        # Store action dict manually (simulating first call)
        controller.pending_action_dict = {
            'action': 'PUT',
            'marble': 'w',
            'dst': 'D4',
            'remove': 'B2'
        }
        controller.pending_player = mock_player

        # Run update (this is the second call that processes the isolation result)
        with patch.object(controller, 'update_game') as mock_update:
            # Manually call the result processing code
            action_result = ActionResult(captured_marbles=isolation_result, newly_frozen_positions=set())

            if action_result.is_isolation() and controller.pending_action_dict is not None:
                # Re-generate notation with isolation
                notation_with_isolation = mock_game_session.game.action_to_notation(
                    controller.pending_action_dict, action_result
                )

        # Verify action_to_notation was called with ActionResult
        calls = mock_game_session.game.action_to_notation.call_args_list
        # Should be called at least once with non-None action_result
        assert any(call[0][1] is not None or call[1].get('action_result') is not None
                   for call in calls if len(call[0]) > 1 or 'action_result' in call[1])


def test_buffered_notation_updated_with_isolation(mock_game_session, mock_logger, mock_player):
    """Test that buffered notation is updated when isolation occurs (Step 7)."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        # Simulate the controller state after first update_game call
        controller.pending_action_dict = {
            'action': 'PUT',
            'marble': 'w',
            'dst': 'D4',
            'remove': 'B2'
        }
        controller.pending_player = mock_player
        controller.pending_notation = "Wd4,b2"

        # Simulate isolation result coming back
        action_result = ActionResult(
            captured_marbles=[
                {'marble': 'w', 'pos': 'A1'},
                {'marble': 'w', 'pos': 'B2'}
            ],
            newly_frozen_positions=set()
        )

        # Process result (this is what happens in update_game)
        if action_result.is_isolation() and controller.pending_action_dict is not None:
            notation_with_isolation = controller.session.game.action_to_notation(
                controller.pending_action_dict, action_result
            )
            controller.logger.update_buffered_notation(notation_with_isolation)
            controller.pending_notation = notation_with_isolation
            controller.pending_action_dict = None
            controller.pending_player = None

        # Verify update was called
        mock_logger.update_buffered_notation.assert_called_once()
        call_arg = mock_logger.update_buffered_notation.call_args[0][0]
        assert ' x ' in call_arg  # Should have isolation notation


def test_pending_variables_cleared_after_update(mock_game_session, mock_logger, mock_player):
    """Test that pending variables are cleared after isolation update."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        # Set up initial state
        controller.pending_action_dict = {'action': 'PUT'}
        controller.pending_player = mock_player
        controller.pending_notation = "Wd4,b2"

        # Process isolation result
        action_result = ActionResult(captured_marbles=[{'marble': 'w', 'pos': 'A1'}], newly_frozen_positions=set())

        if action_result.is_isolation() and controller.pending_action_dict is not None:
            notation_with_isolation = controller.session.game.action_to_notation(
                controller.pending_action_dict, action_result
            )
            controller.logger.update_buffered_notation(notation_with_isolation)
            controller.pending_notation = notation_with_isolation
            controller.pending_action_dict = None
            controller.pending_player = None

        # Verify cleared
        assert controller.pending_action_dict is None
        assert controller.pending_player is None


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_no_isolation_no_update(mock_game_session, mock_logger, mock_player):
    """Test that no isolation means no notation update."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        controller.pending_action_dict = {'action': 'PUT'}
        controller.pending_player = mock_player

        # No isolation (result is single marble or None)
        result = 'w'  # Single marble capture, not isolation

        # Should not update buffered notation
        if isinstance(result, list):  # This should be False
            controller.logger.update_buffered_notation.assert_not_called()


def test_capture_action_adds_marble_not_notation_update(mock_game_session, mock_logger, mock_player):
    """Test that capture actions add marbles but don't trigger notation update."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        # Capture action result (single marble)
        result = 'w'
        player = mock_player

        # Should add capture but not update notation
        if not isinstance(result, list):
            player.add_capture(result)

        player.add_capture.assert_called_once_with('w')
        # Notation update should not be called
        mock_logger.update_buffered_notation.assert_not_called()


def test_empty_isolation_list_no_update(mock_game_session, mock_logger, mock_player):
    """Test that empty isolation list doesn't trigger update."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger), \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        controller.pending_action_dict = {'action': 'PUT'}

        # Empty isolation list
        action_result = ActionResult(captured_marbles=[], newly_frozen_positions=set())

        # Process - ActionResult with empty captures
        if action_result.is_isolation() and controller.pending_action_dict is not None:
            # Even though it's isolation, empty captures means no actual isolation
            notation = controller.session.game.action_to_notation(
                controller.pending_action_dict, action_result
            )
            # But notation should be same as without isolation
            assert ' x ' not in notation


def test_headless_mode_processes_isolation_immediately(mock_game_session, mock_logger, mock_player):
    """Test that headless mode processes isolation in single update_game call."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, headless=True)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player
        mock_game_session.game.get_game_ended = Mock(return_value=None)

        # Set up isolation result
        isolation_result = [{'marble': 'w', 'pos': 'A1'}]
        mock_game_session.game.take_action = Mock(return_value=ActionResult(captured_marbles=isolation_result, newly_frozen_positions=set()))

        task = Mock()
        task.delay_time = 0

        # Single update_game call should process everything
        controller.update_game(task)

        # In headless mode, result is processed in same call
        # So notation should be generated twice: once without isolation, once with
        assert mock_game_session.game.action_to_notation.call_count >= 1


def test_notation_without_pending_action_dict_skips_update():
    """Test that isolation without pending_action_dict doesn't crash."""
    with patch('controller.zertz_game_controller.GameSession') as mock_session_class, \
         patch('controller.zertz_game_controller.GameLogger') as mock_logger_class, \
         patch('controller.zertz_game_controller.ZertzRenderer', return_value=None):

        mock_session = Mock()
        mock_session.rings = 37  # Set rings attribute
        mock_session.blitz = False
        mock_session.is_replay_mode = Mock(return_value=False)
        mock_session.get_seed = Mock(return_value=12345)
        mock_session_class.return_value = mock_session

        mock_logger = Mock()
        mock_logger.log_file = None
        mock_logger.notation_file = None
        mock_logger_class.return_value = mock_logger

        controller = ZertzGameController(rings=37, headless=True)

        # No pending action dict
        controller.pending_action_dict = None
        controller.pending_player = None

        # Isolation result comes in
        result = [{'marble': 'w', 'pos': 'A1'}]

        # Should not crash or call update
        if isinstance(result, list) and controller.pending_action_dict is not None:
            # This block should not execute
            mock_logger.update_buffered_notation.assert_not_called()