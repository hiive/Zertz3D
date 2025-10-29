"""Game action writers for Zertz3D.

Provides pluggable writer classes that combine formatters with output streams
to log game actions in various formats (notation, transcript, etc.).
"""

from abc import ABC, abstractmethod
from typing import TextIO

from game.formatters import NotationFormatter, TranscriptFormatter


class GameWriter(ABC):
    """Abstract base class for game action writers.

    A GameWriter combines a formatter with an output stream to write
    game actions in a specific format. Subclasses implement format-specific
    headers and action formatting.
    """

    def __init__(self, output: TextIO):
        """Initialize the writer.

        Args:
            output: Output stream to write to (file, stdout, etc.)
        """
        self.output = output
        self._game_started = False

    @abstractmethod
    def write_header(self, seed: int, rings: int, blitz: bool = False, player1_name: str | None = None, player2_name: str | None = None) -> None:
        """Write file header with game metadata.

        Args:
            seed: Random seed for this game
            rings: Number of rings on the board (37, 48, or 61)
            blitz: Whether this is a blitz variant game
            player1_name: Optional name for player 1
            player2_name: Optional name for player 2
        """
        pass

    @abstractmethod
    def write_action(self, player_num: int, action_dict: dict, action_result=None) -> None:
        """Write a game action.

        Args:
            player_num: Player number (1 or 2)
            action_dict: Action dictionary
            action_result: Optional ActionResult (for isolation captures in notation)
        """
        pass

    def write_comment(self, message: str) -> None:
        """Write a status/comment message.

        Default implementation does nothing. Subclasses can override to write comments.

        Args:
            message: Status message to write
        """
        pass

    def write_footer(self, game=None) -> None:
        """Write file footer with final game state (optional).

        Args:
            game: Optional ZertzGame instance for final state
        """
        pass

    def flush(self) -> None:
        """Flush the output stream."""
        self.output.flush()

    def close(self) -> None:
        """Close the output stream (but never close stdout/stderr)."""
        import sys
        # Never close stdout or stderr - they should persist for the entire program
        if self.output in (sys.stdout, sys.stderr):
            return
        if hasattr(self.output, 'close'):
            self.output.close()


class NotationWriter(GameWriter):
    """Writes game actions in official Zèrtz notation format.

    File format:
        37 Blitz          # Header: rings + variant
        Wd4               # Action in notation format
        Ge3,a1            # Another action
        -                 # Pass
    """

    def __init__(self, output: TextIO):
        """Initialize notation writer.

        Args:
            output: Output stream to write to
        """
        super().__init__(output)
        self.formatter = NotationFormatter()

    def write_header(self, seed: int, rings: int, blitz: bool = False, player1_name: str | None = None, player2_name: str | None = None) -> None:
        """Write notation file header.

        Format:
            {rings} [Blitz]
            # Player 1: {name}  (if player1_name provided)
            # Player 2: {name}  (if player2_name provided)

        Args:
            seed: Random seed (not used in notation format)
            rings: Number of rings (37, 48, or 61)
            blitz: Whether this is a blitz game
            player1_name: Optional name for player 1
            player2_name: Optional name for player 2
        """
        variant_text = " Blitz" if blitz else ""
        self.output.write(f"{rings}{variant_text}\n")

        # Write player names if provided
        if player1_name:
            self.output.write(f"# Player 1: {player1_name}\n")
        if player2_name:
            self.output.write(f"# Player 2: {player2_name}\n")

        self._game_started = True
        self.flush()

    def write_action(self, player_num: int, action_dict: dict, action_result=None) -> None:
        """Write action in notation format.

        Args:
            player_num: Player number (not used in notation format)
            action_dict: Action dictionary
            action_result: Optional ActionResult for isolation captures
        """
        notation = self.formatter.action_to_notation(action_dict, action_result)
        self.output.write(f"{notation}\n")
        self.flush()


class TranscriptWriter(GameWriter):
    """Writes game actions in transcript file format.

    File format:
        # Seed: 12345              # Header comments
        # Rings: 37
        # Variant: Blitz
        #
        Player 1: {'action': 'PUT', ...}    # Action with player prefix
        Player 2: {'action': 'CAP', ...}
        #
        # Final game state:        # Footer comments
        # ...
    """

    def __init__(self, output: TextIO):
        """Initialize transcript writer.

        Args:
            output: Output stream to write to
        """
        super().__init__(output)
        self.formatter = TranscriptFormatter()

    def write_header(self, seed: int, rings: int, blitz: bool = False, player1_name: str | None = None, player2_name: str | None = None) -> None:
        """Write transcript file header.

        Format:
            # Seed: {seed}
            # Rings: {rings}
            # Variant: Blitz  (if blitz=True)
            # Player 1: {name}  (if player1_name provided)
            # Player 2: {name}  (if player2_name provided)
            #

        Args:
            seed: Random seed
            rings: Number of rings
            blitz: Whether this is a blitz game
            player1_name: Optional name for player 1
            player2_name: Optional name for player 2
        """
        self.output.write(f"# Seed: {seed}\n")
        self.output.write(f"# Rings: {rings}\n")
        if blitz:
            self.output.write("# Variant: Blitz\n")

        # Write player names if provided
        if player1_name:
            self.output.write(f"# Player 1: {player1_name}\n")
        if player2_name:
            self.output.write(f"# Player 2: {player2_name}\n")

        self.output.write("#\n")
        self._game_started = True
        self.flush()

    def write_action(self, player_num: int, action_dict: dict, action_result=None) -> None:
        """Write action in transcript format.

        Format: "Player {player_num}: {action_dict}"

        Args:
            player_num: Player number (1 or 2)
            action_dict: Action dictionary
            action_result: Not used in transcript format
        """
        transcript_str = self.formatter.action_to_transcript(action_dict)
        self.output.write(f"Player {player_num}: {transcript_str}\n")
        self.flush()

    def write_comment(self, message: str) -> None:
        """Write a status/comment message to transcript file.

        Args:
            message: Status message to write as a comment
        """
        self.output.write(f"# {message}\n")
        self.flush()

    def write_footer(self, game=None) -> None:
        """Write transcript file footer with final game state.

        Args:
            game: ZertzGame instance to get final state from
        """
        if game is None:
            return

        self.output.write("#\n")
        self.output.write("# Final game state:\n")
        self.output.write("# ---------------\n")
        self.output.write("# Board state:\n")

        # Combine board layers for visualization
        board_state = (
            game.board.state[0]
            + game.board.state[1]
            + game.board.state[2] * 2
            + game.board.state[3] * 3
        )
        for row in board_state:
            self.output.write(f"# {row}\n")

        self.output.write("# ---------------\n")
        self.output.write("# Marble supply:\n")
        self.output.write(f"# {game.board.state[-10:-1, 0, 0]}\n")
        self.output.write("# ---------------\n")
        self.flush()