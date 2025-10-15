"""Action result value object for game actions.

This class encapsulates the complete result of executing a game action,
including captured marbles and newly frozen positions. It eliminates the need
for the controller to access game internals (game.board.frozen_positions).

Architecture: Part of Recommendation 1 from architecture_report3.md
"""


class ActionResult:
    """Encapsulates the result of a game action.

    This value object provides a clean interface between the game and controller layers,
    eliminating the need for the controller to access board internals or perform
    game logic calculations (like freeze detection).

    Attributes:
        captured_marbles: The marble(s) captured by the action.
            - For capture actions: single marble color string ('w', 'g', or 'b')
            - For placement with isolation: list of dicts [{'marble': color, 'pos': str}, ...]
            - For placement without isolation: None
            - For PASS: None

        newly_frozen_positions: Set of position strings that became frozen due to this action.
            Position strings are in board coordinate format (e.g., 'A1', 'D4').
            Empty set if no positions were frozen.
    """

    def __init__(self, captured_marbles=None, newly_frozen_positions=None):
        """Initialize action result.

        Args:
            captured_marbles: Marble(s) captured by the action (str, list of dicts, or None)
            newly_frozen_positions: Set of position strings that became frozen (set or None)
        """
        self.captured_marbles = captured_marbles
        self.newly_frozen_positions = (
            newly_frozen_positions if newly_frozen_positions is not None else set()
        )

    def __repr__(self):
        """String representation for debugging."""
        frozen_str = (
            f", frozen={len(self.newly_frozen_positions)} positions"
            if self.newly_frozen_positions
            else ""
        )
        return f"ActionResult(captured={self.captured_marbles}{frozen_str})"

    def has_captures(self):
        """Check if this action resulted in any captures.

        Returns:
            bool: True if marbles were captured
        """
        if isinstance(self.captured_marbles, list):
            return len(self.captured_marbles) > 0
        return self.captured_marbles is not None

    def has_freeze(self):
        """Check if this action resulted in any frozen positions.

        Returns:
            bool: True if positions were frozen
        """
        return len(self.newly_frozen_positions) > 0

    def is_isolation(self):
        """Check if this result represents isolation (multiple captures).

        Returns:
            bool: True if this is an isolation result (list of captures)
        """
        return isinstance(self.captured_marbles, list)
