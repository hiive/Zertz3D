"""
Unit tests for Simplified Notation Generation.

Tests that notation is generated in one pass after action execution,
with isolation information included automatically.
"""

import pytest
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.zertz_game_controller import ZertzGameController
from game.zertz_game import ZertzGame
from controller.game_logger import GameLogger
from game.action_result import ActionResult
from renderer.text_renderer import TextRenderer


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

    # action_to_notation now receives ActionResult and generates complete notation in one pass
    def mock_action_to_notation(action_dict, action_result):
        if action_result is None or not action_result.is_isolation():
            return "Wd4,b2"
        # Include isolation in notation
        return "Wd4,b2 x Wa1Wb2"

    session.game.action_to_notation = Mock(side_effect=mock_action_to_notation)

    # Mock render data
    from shared.render_data import RenderData
    session.game.get_render_data = Mock(return_value=RenderData(action_dict={
        'action': 'PUT',
        'marble': 'w',
        'dst': 'D4',
        'remove': 'B2'
    }))

    # Return ActionResult by default (no isolation)
    session.game.take_action = Mock(return_value=ActionResult(
        captured_marbles=None,
        newly_frozen_positions=set()
    ))

    # Mock session methods
    session.get_current_player = Mock()
    session.is_replay_mode = Mock(return_value=False)
    session.get_seed = Mock(return_value=12345)
    session.game.get_game_ended = Mock(return_value=None)

    return session


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock(spec=GameLogger)
    logger.log_file = None
    logger.notation_file = None
    logger.log_action = Mock()
    logger.log_notation = Mock()  # Direct logging, no buffering
    return logger


@pytest.fixture
def mock_player():
    """Create a mock player."""
    player = Mock()
    player.n = 1
    player.get_action = Mock(return_value=('PUT', (0, 0, 0)))
    player.add_capture = Mock()
    return player


@pytest.fixture
def text_renderer():
    """Provide a text renderer that writes to an in-memory buffer."""
    return TextRenderer(stream=StringIO())


# ============================================================================
# One-Pass Notation Generation Tests
# ============================================================================

def test_notation_generated_after_action_execution(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test that notation is generated AFTER action execution with ActionResult."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player

        task = Mock()
        task.delay_time = 0

        # Run update
        controller.update_game(task)

        # Verify action was executed first
        mock_game_session.game.take_action.assert_called_once()

        # Verify notation was generated with ActionResult (not None)
        mock_game_session.game.action_to_notation.assert_called_once()
        call_args = mock_game_session.game.action_to_notation.call_args
        action_result = call_args[0][1]
        # ActionResult should be provided (not None)
        assert action_result is not None
        assert isinstance(action_result, ActionResult)


def test_notation_logged_directly_no_buffering(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test that notation is logged directly without buffering."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player

        task = Mock()
        task.delay_time = 0

        # Run update
        controller.update_game(task)

        # Verify notation was logged directly
        mock_logger.log_notation.assert_called_once()
        # Should be called with complete notation
        notation = mock_logger.log_notation.call_args[0][0]
        assert isinstance(notation, str)
        assert len(notation) > 0


def test_notation_includes_isolation_in_one_pass(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test that notation includes isolation information immediately."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player

        # Set up isolation result
        isolation_result = ActionResult(
            captured_marbles=[
                {'marble': 'w', 'pos': 'A1'},
                {'marble': 'w', 'pos': 'B2'}
            ],
            newly_frozen_positions=set()
        )
        mock_game_session.game.take_action = Mock(return_value=isolation_result)

        task = Mock()
        task.delay_time = 0

        # Run update
        controller.update_game(task)

        # Verify notation was logged with isolation marker
        mock_logger.log_notation.assert_called_once()
        notation = mock_logger.log_notation.call_args[0][0]
        # Should include isolation marker ' x '
        assert ' x ' in notation


def test_no_buffering_attributes_in_controller(mock_game_session, mock_logger, text_renderer):
    """Test that controller no longer has buffering attributes."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)

        # Controller should NOT have these attributes (removed with temporal coupling)
        assert not hasattr(controller, 'pending_action_dict')
        assert not hasattr(controller, 'pending_notation')

        # Controller SHOULD expose new renderer coordination attributes
        assert hasattr(controller, 'waiting_for_renderer')
        assert hasattr(controller, '_completion_queue')


# ============================================================================
# Action Result Processing Tests
# ============================================================================

def test_action_result_stored_for_processing(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test that action_result is stored for later processing."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player

        task = Mock()
        task.delay_time = 0

        # Initially no pending completions
        assert controller._completion_queue == []

        # Run update
        controller.update_game(task)

        # Should have stored action_result for processing
        # (In headless mode, it's processed immediately and cleared)
        # So we verify take_action was called
        mock_game_session.game.take_action.assert_called_once()
        assert controller.waiting_for_renderer is False
        assert controller._completion_queue == []


def test_isolation_captures_added_to_player(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test that isolation captures are added to player's collection."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        # Set up isolation result
        isolation_result = ActionResult(
            captured_marbles=[
                {'marble': 'w', 'pos': 'A1'},
                {'marble': 'b', 'pos': 'B2'}
            ],
            newly_frozen_positions=set()
        )
        mock_game_session.game.take_action = Mock(return_value=isolation_result)
        mock_game_session.get_current_player.return_value = mock_player

        task = Mock()
        task.delay_time = 0

        # Run update - action execution
        controller.update_game(task)

        # In headless mode, result is processed immediately
        # So we need to trigger the processing by calling update_game again
        controller.update_game(task)

        # Verify captures were added
        assert mock_player.add_capture.called


def test_single_capture_added_directly(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test that single capture (CAP action) is added to player."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        # Single capture (not isolation)
        capture_result = ActionResult(
            captured_marbles='w',  # Single marble
            newly_frozen_positions=set()
        )
        mock_game_session.game.take_action = Mock(return_value=capture_result)
        mock_game_session.get_current_player.return_value = mock_player

        task = Mock()
        task.delay_time = 0

        # Run update
        controller.update_game(task)
        controller.update_game(task)  # Process result

        # Verify single capture was added
        mock_player.add_capture.assert_called()


# ============================================================================
# Integration Tests
# ============================================================================

def test_complete_notation_workflow(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test complete notation workflow from action to logging."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player

        task = Mock()
        task.delay_time = 0

        # Run update
        controller.update_game(task)

        # Verify correct call order:
        # 1. Get action from player
        mock_player.get_action.assert_called()

        # 2. Execute action
        mock_game_session.game.take_action.assert_called()

        # 3. Generate notation with result
        mock_game_session.game.action_to_notation.assert_called()

        # 4. Log notation directly
        mock_logger.log_notation.assert_called()


def test_no_regeneration_or_update_calls(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test that notation is generated only once, not regenerated."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player

        task = Mock()
        task.delay_time = 0

        # Run update
        controller.update_game(task)

        # action_to_notation should be called EXACTLY once
        assert mock_game_session.game.action_to_notation.call_count == 1

        # log_notation should be called EXACTLY once
        assert mock_logger.log_notation.call_count == 1


def test_headless_mode_processes_immediately(mock_game_session, mock_logger, mock_player, text_renderer):
    """Test that headless mode processes actions immediately without deferral."""
    with patch('controller.zertz_game_controller.GameSession', return_value=mock_game_session), \
         patch('controller.zertz_game_controller.GameLogger', return_value=mock_logger):

        controller = ZertzGameController(rings=37, renderer_or_factory=text_renderer)
        controller.session = mock_game_session
        controller.logger = mock_logger

        mock_game_session.get_current_player.return_value = mock_player

        task = Mock()
        task.delay_time = 0

        # Single update_game call should complete everything
        controller.update_game(task)

        # All operations should be complete
        assert mock_game_session.game.take_action.called
        assert mock_game_session.game.action_to_notation.called
        assert mock_logger.log_notation.called
