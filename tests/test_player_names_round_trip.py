"""Test that player names round-trip correctly through notation and transcript formats."""

import tempfile
import os
from pathlib import Path

from game.writers import NotationWriter, TranscriptWriter
from game.loaders.notation_loader import NotationLoader
from game.loaders.transcript_loader import TranscriptLoader
from game.zertz_game import ZertzGame
from game.player_config import PlayerConfig
from controller.game_session import GameSession


def test_notation_player_names_round_trip():
    """Test player names are written and read correctly from notation files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_notation.txt', delete=False) as f:
        try:
            # Write a notation file with player names
            writer = NotationWriter(f)
            writer.write_header(seed=12345, rings=37, blitz=False, player1_name="Alice", player2_name="Bob")

            # Write a couple of moves
            writer.write_action(1, {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
            writer.write_action(2, {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': 'A1'})
            f.close()

            # Read the notation file
            loader = NotationLoader(f.name)
            p1_actions, p2_actions = loader.load()

            # Verify player names were loaded
            assert loader.player1_name == "Alice", f"Expected player1_name='Alice', got '{loader.player1_name}'"
            assert loader.player2_name == "Bob", f"Expected player2_name='Bob', got '{loader.player2_name}'"

            # Verify actions were loaded
            assert len(p1_actions) == 1
            assert len(p2_actions) == 1

        finally:
            os.unlink(f.name)


def test_transcript_player_names_round_trip():
    """Test player names are written and read correctly from transcript files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        try:
            # Write a transcript file with player names
            writer = TranscriptWriter(f)
            writer.write_header(seed=12345, rings=37, blitz=False, player1_name="Charlie", player2_name="Diana")

            # Write a couple of moves
            writer.write_action(1, {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''})
            writer.write_action(2, {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': 'A1'})
            f.close()

            # Read the transcript file
            loader = TranscriptLoader(f.name)
            p1_actions, p2_actions = loader.load()

            # Verify player names were loaded
            assert loader.player1_name == "Charlie", f"Expected player1_name='Charlie', got '{loader.player1_name}'"
            assert loader.player2_name == "Diana", f"Expected player2_name='Diana', got '{loader.player2_name}'"

            # Verify actions were loaded
            assert len(p1_actions) == 1
            assert len(p2_actions) == 1

        finally:
            os.unlink(f.name)


def test_notation_without_player_names():
    """Test notation files without player names still work (backward compatibility)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_notation.txt', delete=False) as f:
        try:
            # Write a notation file WITHOUT player names
            f.write("37\n")
            f.write("Wd4\n")
            f.write("Ge3,a1\n")
            f.close()

            # Read the notation file
            loader = NotationLoader(f.name)
            p1_actions, p2_actions = loader.load()

            # Verify player names are None
            assert loader.player1_name is None
            assert loader.player2_name is None

            # Verify actions were loaded
            assert len(p1_actions) == 1
            assert len(p2_actions) == 1

        finally:
            os.unlink(f.name)


def test_transcript_without_player_names():
    """Test transcript files without player names still work (backward compatibility)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        try:
            # Write a transcript file WITHOUT player names
            f.write("# Seed: 12345\n")
            f.write("# Rings: 37\n")
            f.write("#\n")
            f.write("Player 1: {'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}\n")
            f.write("Player 2: {'action': 'PUT', 'marble': 'g', 'dst': 'E3', 'remove': 'A1'}\n")
            f.close()

            # Read the transcript file
            loader = TranscriptLoader(f.name)
            p1_actions, p2_actions = loader.load()

            # Verify player names are None
            assert loader.player1_name is None
            assert loader.player2_name is None

            # Verify actions were loaded
            assert len(p1_actions) == 1
            assert len(p2_actions) == 1

        finally:
            os.unlink(f.name)


def test_notation_with_blitz_and_player_names():
    """Test notation files with both blitz variant and player names."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_notation.txt', delete=False) as f:
        try:
            # Write a notation file with blitz variant and player names
            writer = NotationWriter(f)
            writer.write_header(seed=12345, rings=37, blitz=True, player1_name="Eve", player2_name="Frank")
            f.close()

            # Read the notation file
            loader = NotationLoader(f.name)
            p1_actions, p2_actions = loader.load()

            # Verify blitz flag was set
            assert loader.blitz is True

            # Verify player names were loaded
            assert loader.player1_name == "Eve"
            assert loader.player2_name == "Frank"

        finally:
            os.unlink(f.name)


def test_player_names_with_special_characters():
    """Test player names can contain special characters (spaces, punctuation, etc.)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_notation.txt', delete=False) as f:
        try:
            # Write a notation file with complex player names
            writer = NotationWriter(f)
            writer.write_header(
                seed=12345,
                rings=37,
                blitz=False,
                player1_name="Player One (Beginner)",
                player2_name="AI Bot v2.5"
            )
            f.close()

            # Read the notation file
            loader = NotationLoader(f.name)
            p1_actions, p2_actions = loader.load()

            # Verify player names were loaded correctly
            assert loader.player1_name == "Player One (Beginner)"
            assert loader.player2_name == "AI Bot v2.5"

        finally:
            os.unlink(f.name)


def test_only_player1_name():
    """Test that only player 1 name can be specified."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_notation.txt', delete=False) as f:
        try:
            # Write a notation file with only player 1 name
            writer = NotationWriter(f)
            writer.write_header(seed=12345, rings=37, blitz=False, player1_name="Solo Player", player2_name=None)
            f.close()

            # Read the notation file
            loader = NotationLoader(f.name)
            p1_actions, p2_actions = loader.load()

            # Verify only player1 name was loaded
            assert loader.player1_name == "Solo Player"
            assert loader.player2_name is None

        finally:
            os.unlink(f.name)