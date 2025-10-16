"""Tests for CompositeRenderer functionality."""

import pytest
from unittest.mock import Mock, MagicMock, call

from renderer.composite_renderer import CompositeRenderer
from shared.interfaces import IRenderer
from shared.render_data import RenderData


class MockRenderer(IRenderer):
    """Mock renderer for testing."""

    def __init__(self, name="mock"):
        self.name = name
        self.run_called = False
        self.reset_board_called = False
        self.execute_action_calls = []
        self.show_isolated_removal_calls = []
        self.report_status_calls = []
        self.attach_update_loop_return = False
        self.attach_update_loop_calls = []

    def run(self):
        self.run_called = True

    def reset_board(self):
        self.reset_board_called = True

    def execute_action(self, player, render_data, action_result, task_delay_time, on_complete):
        self.execute_action_calls.append({
            'player': player,
            'render_data': render_data,
            'action_result': action_result,
            'task_delay_time': task_delay_time,
            'on_complete': on_complete
        })
        if on_complete:
            on_complete(player, action_result)

    def show_isolated_removal(self, player, pos, marble, delay_time):
        self.show_isolated_removal_calls.append({
            'player': player,
            'pos': pos,
            'marble': marble,
            'delay_time': delay_time
        })

    def report_status(self, message):
        self.report_status_calls.append(message)

    def attach_update_loop(self, update_fn, interval):
        self.attach_update_loop_calls.append({
            'update_fn': update_fn,
            'interval': interval
        })
        return self.attach_update_loop_return


class MockRendererWithOptionalMethods(MockRenderer):
    """Mock renderer with optional methods."""

    def __init__(self, name="mock_optional"):
        super().__init__(name)
        self.set_context_highlights_calls = []
        self.clear_context_highlights_calls = []
        self.highlight_context_calls = []
        self.clear_highlight_context_calls = []
        self.show_hover_feedback_calls = []
        self.clear_hover_highlights_called = False
        self.set_selection_callback_calls = []
        self.set_hover_callback_calls = []
        self.apply_context_masks_calls = []

    def set_context_highlights(self, context, positions, color=None, emission=None):
        self.set_context_highlights_calls.append({
            'context': context,
            'positions': positions,
            'color': color,
            'emission': emission
        })

    def clear_context_highlights(self, context=None):
        self.clear_context_highlights_calls.append(context)

    def highlight_context(self, context, positions):
        self.highlight_context_calls.append({
            'context': context,
            'positions': positions
        })

    def clear_highlight_context(self, context=None):
        self.clear_highlight_context_calls.append(context)

    def show_hover_feedback(self, primary=None, secondary=None, supply_colors=None, captured_targets=None):
        self.show_hover_feedback_calls.append({
            'primary': primary,
            'secondary': secondary,
            'supply_colors': supply_colors,
            'captured_targets': captured_targets
        })

    def clear_hover_highlights(self):
        self.clear_hover_highlights_called = True

    def set_selection_callback(self, callback):
        self.set_selection_callback_calls.append(callback)

    def set_hover_callback(self, callback):
        self.set_hover_callback_calls.append(callback)

    def apply_context_masks(self, board, placement_mask, capture_mask):
        self.apply_context_masks_calls.append({
            'board': board,
            'placement_mask': placement_mask,
            'capture_mask': capture_mask
        })


class TestCompositeRenderer:
    """Test suite for CompositeRenderer class."""

    def test_initialization_requires_at_least_one_renderer(self):
        """Test that CompositeRenderer requires at least one renderer."""
        with pytest.raises(ValueError, match="requires at least one renderer"):
            CompositeRenderer([])

    def test_initialization_filters_none_renderers(self):
        """Test that None renderers are filtered out."""
        renderer1 = MockRenderer("r1")
        renderer2 = None
        renderer3 = MockRenderer("r3")

        composite = CompositeRenderer([renderer1, renderer2, renderer3])

        # Should only have 2 renderers
        assert len(composite._renderers) == 2

    def test_initialization_with_single_renderer(self):
        """Test initialization with a single renderer."""
        renderer = MockRenderer()
        composite = CompositeRenderer([renderer])

        assert len(composite._renderers) == 1
        assert composite._renderers[0] == renderer

    def test_initialization_with_multiple_renderers(self):
        """Test initialization with multiple renderers."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRenderer("r2")
        renderer3 = MockRenderer("r3")

        composite = CompositeRenderer([renderer1, renderer2, renderer3])

        assert len(composite._renderers) == 3

    def test_run_forwards_to_all_renderers(self):
        """Test that run() forwards to all renderers."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRenderer("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        composite.run()

        assert renderer1.run_called
        assert renderer2.run_called

    def test_reset_board_forwards_to_all_renderers(self):
        """Test that reset_board() forwards to all renderers."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRenderer("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        composite.reset_board()

        assert renderer1.reset_board_called
        assert renderer2.reset_board_called

    def test_execute_action_with_single_renderer_no_callback(self):
        """Test execute_action with single renderer and no callback."""
        renderer = MockRenderer()
        composite = CompositeRenderer([renderer])

        player = Mock()
        render_data = RenderData({'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
        action_result = Mock()

        composite.execute_action(player, render_data, action_result, 0.5, None)

        assert len(renderer.execute_action_calls) == 1
        call = renderer.execute_action_calls[0]
        assert call['player'] == player
        assert call['render_data'] == render_data
        assert call['task_delay_time'] == 0.5
        assert call['on_complete'] is None

    def test_execute_action_with_single_renderer_with_callback(self):
        """Test execute_action with single renderer and callback."""
        renderer = MockRenderer()
        composite = CompositeRenderer([renderer])

        player = Mock()
        render_data = RenderData({'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
        action_result = Mock()
        callback = Mock()

        composite.execute_action(player, render_data, action_result, 0.5, callback)

        # Callback should be called directly by the single renderer
        callback.assert_called_once_with(player, action_result)

    def test_execute_action_with_multiple_renderers_no_callback(self):
        """Test execute_action with multiple renderers and no callback."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRenderer("r2")
        composite = CompositeRenderer([renderer1, renderer2])

        player = Mock()
        render_data = RenderData({'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
        action_result = Mock()

        composite.execute_action(player, render_data, action_result, 0.5, None)

        assert len(renderer1.execute_action_calls) == 1
        assert len(renderer2.execute_action_calls) == 1
        assert renderer1.execute_action_calls[0]['on_complete'] is None
        assert renderer2.execute_action_calls[0]['on_complete'] is None

    def test_execute_action_with_multiple_renderers_callback_coordination(self):
        """Test execute_action callback coordination with multiple renderers."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRenderer("r2")
        composite = CompositeRenderer([renderer1, renderer2])

        player = Mock()
        render_data = RenderData({'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
        action_result = Mock()
        callback = Mock()

        composite.execute_action(player, render_data, action_result, 0.5, callback)

        # Both renderers should be called
        assert len(renderer1.execute_action_calls) == 1
        assert len(renderer2.execute_action_calls) == 1

        # on_complete should be wrapped callbacks, not the original
        assert renderer1.execute_action_calls[0]['on_complete'] is not None
        assert renderer2.execute_action_calls[0]['on_complete'] is not None

        # Original callback should be called exactly once after both complete
        # (MockRenderer calls on_complete immediately in execute_action)
        assert callback.call_count == 1
        callback.assert_called_with(player, action_result)

    def test_show_isolated_removal_forwards_to_all_renderers(self):
        """Test that show_isolated_removal() forwards to all renderers."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRenderer("r2")
        composite = CompositeRenderer([renderer1, renderer2])

        player = Mock()
        composite.show_isolated_removal(player, "D4", "w", 0.5)

        assert len(renderer1.show_isolated_removal_calls) == 1
        assert len(renderer2.show_isolated_removal_calls) == 1
        assert renderer1.show_isolated_removal_calls[0]['pos'] == "D4"
        assert renderer2.show_isolated_removal_calls[0]['marble'] == "w"

    def test_report_status_forwards_to_all_renderers(self):
        """Test that report_status() forwards to all renderers."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRenderer("r2")
        composite = CompositeRenderer([renderer1, renderer2])

        composite.report_status("Test message")

        assert len(renderer1.report_status_calls) == 1
        assert len(renderer2.report_status_calls) == 1
        assert renderer1.report_status_calls[0] == "Test message"

    def test_attach_update_loop_returns_true_if_any_renderer_handles(self):
        """Test attach_update_loop returns True if any renderer handles it."""
        renderer1 = MockRenderer("r1")
        renderer1.attach_update_loop_return = False
        renderer2 = MockRenderer("r2")
        renderer2.attach_update_loop_return = True  # This one handles it

        composite = CompositeRenderer([renderer1, renderer2])

        update_fn = Mock()
        result = composite.attach_update_loop(update_fn, 0.5)

        assert result is True
        assert len(renderer1.attach_update_loop_calls) == 1
        assert len(renderer2.attach_update_loop_calls) == 1

    def test_attach_update_loop_returns_false_if_no_renderer_handles(self):
        """Test attach_update_loop returns False if no renderer handles it."""
        renderer1 = MockRenderer("r1")
        renderer1.attach_update_loop_return = False
        renderer2 = MockRenderer("r2")
        renderer2.attach_update_loop_return = False

        composite = CompositeRenderer([renderer1, renderer2])

        update_fn = Mock()
        result = composite.attach_update_loop(update_fn, 0.5)

        assert result is False

    def test_attach_update_loop_skips_renderers_without_method(self):
        """Test attach_update_loop handles renderers without the method."""
        # Create a mock without attach_update_loop
        renderer1 = Mock(spec=['run', 'reset_board'])
        renderer2 = MockRenderer("r2")
        renderer2.attach_update_loop_return = True

        composite = CompositeRenderer([renderer1, renderer2])

        update_fn = Mock()
        result = composite.attach_update_loop(update_fn, 0.5)

        # Should still return True from renderer2
        assert result is True
        assert len(renderer2.attach_update_loop_calls) == 1

    def test_set_context_highlights_forwards_to_renderers_with_method(self):
        """Test set_context_highlights forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")  # No set_context_highlights
        renderer2 = MockRendererWithOptionalMethods("r2")  # Has it

        composite = CompositeRenderer([renderer1, renderer2])
        composite.set_context_highlights("test_context", ["A1", "B2"], color=(1, 0, 0))

        # Only renderer2 should receive the call
        assert len(renderer2.set_context_highlights_calls) == 1
        call = renderer2.set_context_highlights_calls[0]
        assert call['context'] == "test_context"
        assert call['positions'] == ["A1", "B2"]
        assert call['color'] == (1, 0, 0)

    def test_clear_context_highlights_forwards_to_renderers_with_method(self):
        """Test clear_context_highlights forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRendererWithOptionalMethods("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        composite.clear_context_highlights("test_context")

        assert len(renderer2.clear_context_highlights_calls) == 1
        assert renderer2.clear_context_highlights_calls[0] == "test_context"

    def test_highlight_context_forwards_to_renderers_with_method(self):
        """Test highlight_context forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRendererWithOptionalMethods("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        composite.highlight_context("test_context", ["A1"])

        assert len(renderer2.highlight_context_calls) == 1

    def test_clear_highlight_context_forwards_to_renderers_with_method(self):
        """Test clear_highlight_context forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRendererWithOptionalMethods("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        composite.clear_highlight_context()

        assert len(renderer2.clear_highlight_context_calls) == 1

    def test_show_hover_feedback_forwards_to_renderers_with_method(self):
        """Test show_hover_feedback forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRendererWithOptionalMethods("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        composite.show_hover_feedback(primary=["A1"], secondary=["B2"])

        assert len(renderer2.show_hover_feedback_calls) == 1
        call = renderer2.show_hover_feedback_calls[0]
        assert call['primary'] == ["A1"]
        assert call['secondary'] == ["B2"]

    def test_clear_hover_highlights_forwards_to_renderers_with_method(self):
        """Test clear_hover_highlights forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRendererWithOptionalMethods("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        composite.clear_hover_highlights()

        assert renderer2.clear_hover_highlights_called

    def test_set_selection_callback_forwards_to_renderers_with_method(self):
        """Test set_selection_callback forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRendererWithOptionalMethods("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        callback = Mock()
        composite.set_selection_callback(callback)

        assert len(renderer2.set_selection_callback_calls) == 1
        assert renderer2.set_selection_callback_calls[0] == callback

    def test_set_hover_callback_forwards_to_renderers_with_method(self):
        """Test set_hover_callback forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRendererWithOptionalMethods("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        callback = Mock()
        composite.set_hover_callback(callback)

        assert len(renderer2.set_hover_callback_calls) == 1
        assert renderer2.set_hover_callback_calls[0] == callback

    def test_apply_context_masks_forwards_to_renderers_with_method(self):
        """Test apply_context_masks forwards to renderers that have it."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRendererWithOptionalMethods("r2")

        composite = CompositeRenderer([renderer1, renderer2])
        board = Mock()
        placement_mask = Mock()
        capture_mask = Mock()
        composite.apply_context_masks(board, placement_mask, capture_mask)

        assert len(renderer2.apply_context_masks_calls) == 1
        call = renderer2.apply_context_masks_calls[0]
        assert call['board'] == board

    def test_multiple_renderers_with_mixed_optional_methods(self):
        """Test composite with multiple renderers having different optional methods."""
        renderer1 = MockRenderer("r1")  # No optional methods
        renderer2 = MockRendererWithOptionalMethods("r2")  # All optional methods
        renderer3 = MockRendererWithOptionalMethods("r3")  # All optional methods

        composite = CompositeRenderer([renderer1, renderer2, renderer3])

        # Call an optional method
        composite.set_context_highlights("ctx", ["A1"])

        # Should forward to renderer2 and renderer3 only
        assert len(renderer2.set_context_highlights_calls) == 1
        assert len(renderer3.set_context_highlights_calls) == 1

    def test_execute_action_callback_coordination_three_renderers(self):
        """Test callback coordination with three renderers."""
        renderer1 = MockRenderer("r1")
        renderer2 = MockRenderer("r2")
        renderer3 = MockRenderer("r3")
        composite = CompositeRenderer([renderer1, renderer2, renderer3])

        player = Mock()
        render_data = RenderData({'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
        action_result = Mock()
        callback = Mock()

        composite.execute_action(player, render_data, action_result, 0.5, callback)

        # All three renderers should be called
        assert len(renderer1.execute_action_calls) == 1
        assert len(renderer2.execute_action_calls) == 1
        assert len(renderer3.execute_action_calls) == 1

        # Callback should be called exactly once after all three complete
        assert callback.call_count == 1