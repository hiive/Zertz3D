"""Game logging for Zertz 3D.

Handles logging game actions and notation using pluggable writers.
"""

from typing import Callable

from game.writers import GameWriter, NotationWriter, TranscriptWriter


class GameLogger:
    """Manages multiple game action writers for flexible logging.

    Uses the Strategy pattern to support multiple output formats and destinations
    simultaneously (e.g., replay to file, notation to screen).
    """

    def __init__(
        self,
        writers: list[GameWriter] | None = None,
        status_reporter: Callable[[str], None] | None = None,
    ):
        """Initialize the game logger.

        Args:
            writers: Optional list of GameWriter instances
            status_reporter: Optional callback for status messages
        """
        self.writers: list[GameWriter] = writers if writers is not None else []
        self._status_reporter: Callable[[str], None] | None = status_reporter
        self._game_active = False

    def add_writer(self, writer: GameWriter) -> None:
        """Add a writer to the logger.

        Args:
            writer: GameWriter instance to add
        """
        self.writers.append(writer)

    def remove_writer(self, writer: GameWriter) -> None:
        """Remove a writer from the logger.

        Args:
            writer: GameWriter instance to remove
        """
        if writer in self.writers:
            self.writers.remove(writer)

    def start_game(self, seed: int, rings: int, blitz: bool = False) -> None:
        """Start logging for a new game and write headers to all writers.

        Args:
            seed: Random seed for this game
            rings: Number of rings on the board
            blitz: Whether this is a blitz game
        """
        for writer in self.writers:
            writer.write_header(seed, rings, blitz)
        self._game_active = True

    def end_game(self, game=None) -> None:
        """End logging for the current game, write footers and close all writers.

        Args:
            game: Optional ZertzGame instance for final state
        """
        for writer in self.writers:
            writer.write_footer(game)
            writer.close()
        self._game_active = False

    def log_action(self, player_num: int, action_dict: dict, action_result=None) -> None:
        """Log an action to all writers.

        Args:
            player_num: Player number (1 or 2)
            action_dict: Dictionary containing action details
            action_result: Optional ActionResult (for isolation captures in notation)
        """
        for writer in self.writers:
            writer.write_action(player_num, action_dict, action_result)

    def log_comment(self, message: str) -> None:
        """Log a status/comment message to all writers.

        Args:
            message: Status message to log as a comment
        """
        for writer in self.writers:
            writer.write_comment(message)

    def set_status_reporter(self, reporter: Callable[[str], None] | None) -> None:
        """Set or update the status reporter callback."""
        self._status_reporter = reporter

    def _report(self, message: str | None) -> None:
        if message is None:
            return
        if self._status_reporter is not None:
            self._status_reporter(message)
        else:
            print(message)
