"""Base protocol for game replay loaders."""

from typing import Protocol, Callable


class ReplayLoader(Protocol):
    """Protocol defining the interface for game replay loaders.

    All loader implementations (SGF, notation, transcript) should conform to this interface.
    """

    # Required attributes
    detected_rings: int
    blitz: bool
    player1_name: str | None
    player2_name: str | None

    def load(self) -> tuple[list[dict], list[dict]]:
        """Load and parse the replay file.

        Returns:
            Tuple of (player1_actions, player2_actions) where each action is a dict
        """
        ...

    def set_status_reporter(self, reporter: Callable[[str], None] | None) -> None:
        """Set or update the status reporter callback."""
        ...