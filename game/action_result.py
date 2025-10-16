"""Action result value object for game actions.

This class encapsulates the complete result of executing a game action,
including captured marbles. It eliminates the need for the controller
to access game internals.

Architecture: Part of Recommendation 1 from architecture_report3.md
"""


class ActionResult:
    """Encapsulates the result of a game action.

    This value object provides a clean interface between the game and controller layers,
    eliminating the need for the controller to access board internals.

    Attributes:
        captured_marbles: The marble(s) captured by the action.
            - For capture actions: single marble color string ('w', 'g', or 'b')
            - For placement with isolation: list of dicts [{'marble': color, 'pos': str}, ...]
            - For placement without isolation: None
            - For PASS: None
    """

    def __init__(self, captured_marbles=None):
        """Initialize action result.

        Args:
            captured_marbles: Marble(s) captured by the action (str, list of dicts, or None)
        """
        self.captured_marbles = captured_marbles

    def __repr__(self):
        """String representation for debugging."""
        return f"ActionResult(captured={self.captured_marbles})"

    def has_captures(self):
        """Check if this action resulted in any captures.

        Returns:
            bool: True if marbles were captured
        """
        if isinstance(self.captured_marbles, list):
            return len(self.captured_marbles) > 0
        return self.captured_marbles is not None

    def is_isolation(self):
        """Check if this result represents isolation (multiple captures).

        Returns:
            bool: True if this is an isolation result (list of captures)
        """
        return isinstance(self.captured_marbles, list)
