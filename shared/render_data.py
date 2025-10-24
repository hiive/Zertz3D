"""Render data value object for action visualization.

This class encapsulates all data needed by the renderer to visualize a game action,
including the action itself and optional highlight data. It eliminates the need
for higher-level components to perform data transformations or inspect renderer state.
"""


class RenderData:
    """Encapsulates all rendering data for a game action.

    This value object provides a clean interface between the game and renderer layers,
    eliminating the need for orchestrators to:
    - Check renderer configuration (like show_moves)
    - Perform data transformations (array to string conversions)
    - Call multiple separate methods to gather rendering data

    Attributes:
        action_dict: Dictionary containing the action details.
            Format: {'action': 'PUT'|'CAP'|'PASS', 'marble': str, 'dst': str, ...}

        placement_positions: List of placement position dicts with scores for heat-map highlighting.
            Empty list if highlights not requested or no valid placements.
            Format: [{'pos': 'A1', 'score': 0.8}, {'pos': 'B2', 'score': 1.0}, ...]
            OR legacy format: ['A1', 'B2', ...] (when scores not available)

        capture_moves: List of capture move dicts with scores for heat-map highlighting.
            Empty list if highlights not requested or no valid captures.
            Format: [{'action': 'CAP', 'src': 'C4', 'dst': 'E6', 'marble': 'g',
                      'capture': 'D5', 'cap': 'D5', 'score': 0.9}, ...]
            OR legacy format: [{'action': 'CAP', 'src': 'C4', 'dst': 'E6', ...}, ...]
            (when scores not available)

        removal_positions: List of removal position dicts with scores for heat-map highlighting.
            Empty list if highlights not requested.
            Format: [{'pos': 'A1', 'score': 0.7}, {'pos': 'B2', 'score': 0.5}, ...]
            OR legacy format: ['A1', 'B2', ...] (when scores not available)
    """

    def __init__(
        self,
        action_dict,
        placement_positions=None,
        capture_moves=None,
        removal_positions=None,
    ):
        """Initialize render data.

        Args:
            action_dict: Dictionary containing action details
            placement_positions: List of placement position dicts or strings (default: empty list)
            capture_moves: List of capture move dicts (default: empty list)
            removal_positions: List of removal position dicts or strings (default: empty list)
        """
        self.action_dict = action_dict
        self.placement_positions = (
            placement_positions if placement_positions is not None else []
        )
        self.capture_moves = capture_moves if capture_moves is not None else []
        self.removal_positions = (
            removal_positions if removal_positions is not None else []
        )

    def __repr__(self):
        """String representation for debugging."""
        action_type = self.action_dict.get("action", "UNKNOWN")
        highlights = ""
        if self.has_highlights():
            highlights = (
                f", placements={len(self.placement_positions)}, "
                f"captures={len(self.capture_moves)}, "
                f"removals={len(self.removal_positions)}"
            )
        return f"RenderData(action={action_type}{highlights})"

    def has_highlights(self):
        """Check if this render data includes highlight information.

        Returns:
            bool: True if any highlight data is present
        """
        return (
            self.has_placement_highlights()
            or self.has_capture_highlights()
            or self.has_removal_highlights()
        )

    def has_placement_highlights(self):
        """Check if placement highlights are available.

        Returns:
            bool: True if placement positions are available
        """
        return len(self.placement_positions) > 0

    def has_capture_highlights(self):
        """Check if capture highlights are available.

        Returns:
            bool: True if capture moves are available
        """
        return len(self.capture_moves) > 0

    def has_removal_highlights(self):
        """Check if removal highlights are available.

        Returns:
            bool: True if removal positions are available
        """
        return len(self.removal_positions) > 0
