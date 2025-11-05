"""Tests for GameSession functionality."""

import pytest
import numpy as np
import random

from controller.game_session import GameSession
from game.constants import (
    STANDARD_MARBLES,
    BLITZ_MARBLES,
    STANDARD_WIN_CONDITIONS,
    BLITZ_WIN_CONDITIONS,
)
from game.players import HumanZertzPlayer
from game.players import ReplayZertzPlayer
from game.players import RandomZertzPlayer


class TestGameSession:
    """Test suite for GameSession class."""

    def test_default_initialization(self):
        """Test default session initialization."""
        session = GameSession()

        assert session.rings == 37
        assert not session.blitz
        assert session.marbles == STANDARD_MARBLES
        assert session.win_condition == STANDARD_WIN_CONDITIONS
        assert session.game is not None
        assert session.player1 is not None
        assert session.player2 is not None
        assert isinstance(session.player1, RandomZertzPlayer)
        assert isinstance(session.player2, RandomZertzPlayer)
        assert session.games_played == 0
        assert not session.replay_mode

    def test_37_ring_initialization(self):
        """Test initialization with 37 rings."""
        session = GameSession(rings=37)

        assert session.rings == 37
        assert session.game.board.rings == 37

    def test_48_ring_initialization(self):
        """Test initialization with 48 rings."""
        session = GameSession(rings=48)

        assert session.rings == 48
        assert session.game.board.rings == 48

    def test_61_ring_initialization(self):
        """Test initialization with 61 rings."""
        session = GameSession(rings=61)

        assert session.rings == 61
        assert session.game.board.rings == 61

    def test_blitz_mode_initialization(self):
        """Test blitz mode initialization."""
        session = GameSession(blitz=True)

        assert session.blitz
        assert session.marbles == BLITZ_MARBLES
        assert session.win_condition == BLITZ_WIN_CONDITIONS

    def test_blitz_mode_only_works_with_37_rings(self):
        """Test that blitz mode only works with 37 rings."""
        # Should work with 37 rings
        session = GameSession(rings=37, blitz=True)
        assert session.blitz

        # Should raise error with 48 rings
        with pytest.raises(ValueError, match="Blitz mode only works with 37 rings"):
            GameSession(rings=48, blitz=True)

        # Should raise error with 61 rings
        with pytest.raises(ValueError, match="Blitz mode only works with 37 rings"):
            GameSession(rings=61, blitz=True)

    def test_seed_initialization(self):
        """Test that seed is properly set."""
        session = GameSession(seed=12345)

        assert session.current_seed == 12345

    def test_seed_affects_rng(self):
        """Test that seed affects random number generation."""
        # Create two sessions with same seed
        session1 = GameSession(seed=12345)
        np_state1 = np.random.get_state()
        random_state1 = random.getstate()

        # Reset and create another session with same seed
        session2 = GameSession(seed=12345)
        np_state2 = np.random.get_state()
        random_state2 = random.getstate()

        # States should be identical
        assert np.array_equal(np_state1[1], np_state2[1])
        assert random_state1 == random_state2

    def test_auto_seed_generation(self):
        """Test that seed is auto-generated when not provided."""
        session = GameSession()

        assert session.current_seed is not None
        assert isinstance(session.current_seed, int)

    def test_get_current_player(self):
        """Test getting the current player."""
        session = GameSession()

        # Should start with player 1
        current = session.get_current_player()
        assert current == session.player1
        assert current.n == 1

    def test_replay_mode_initialization(self):
        """Test replay mode initialization."""
        player1_actions = [
            {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        ]
        player2_actions = [
            {"action": "PUT", "marble": "g", "dst": "E3", "remove": ""}
        ]

        session = GameSession(replay_actions=(player1_actions, player2_actions))

        assert session.replay_mode
        assert isinstance(session.player1, ReplayZertzPlayer)
        assert isinstance(session.player2, ReplayZertzPlayer)
        assert session.current_seed is None  # No seed in replay mode

    def test_replay_mode_no_seed_applied(self):
        """Test that seed is not applied in replay mode."""
        player1_actions = [
            {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        ]
        player2_actions = [
            {"action": "PUT", "marble": "g", "dst": "E3", "remove": ""}
        ]

        # Seed should be ignored in replay mode
        session = GameSession(
            seed=12345, replay_actions=(player1_actions, player2_actions)
        )

        assert session.current_seed is None

    def test_reset_game(self):
        """Test resetting the game."""
        session = GameSession(seed=12345)
        old_game = session.game

        session.reset_game()

        # Should have a new game instance
        assert session.game is not old_game
        assert session.game is not None

        # Should have new player instances
        assert session.player1 is not None
        assert session.player2 is not None

    def test_seed_generation_between_games(self):
        """Test that seed is regenerated between games."""
        session = GameSession(seed=12345)
        first_seed = session.current_seed

        # Reset game (simulating starting a new game)
        session.reset_game()
        second_seed = session.current_seed

        # Seeds should be different but deterministic
        assert second_seed != first_seed
        assert second_seed is not None

        # Reset again
        session.reset_game()
        third_seed = session.current_seed

        # Should get different seed again
        assert third_seed != second_seed
        assert third_seed != first_seed

    def test_seed_generation_deterministic(self):
        """Test that seed generation is deterministic."""
        # Create two sessions with same initial seed
        session1 = GameSession(seed=12345)
        session1.reset_game()
        seed1_second = session1.current_seed

        session2 = GameSession(seed=12345)
        session2.reset_game()
        seed2_second = session2.current_seed

        # Should generate the same second seed
        assert seed1_second == seed2_second

    def test_games_played_counter(self):
        """Test games played counter."""
        session = GameSession()

        assert session.get_games_played() == 0

        session.increment_games_played()
        assert session.get_games_played() == 1

        session.increment_games_played()
        assert session.get_games_played() == 2

    def test_get_seed(self):
        """Test get_seed method."""
        session = GameSession(seed=12345)

        assert session.get_seed() == 12345

    def test_is_replay_mode(self):
        """Test is_replay_mode method."""
        # Non-replay mode
        session = GameSession()
        assert not session.is_replay_mode()

        # Replay mode
        player1_actions = [
            {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        ]
        player2_actions = [
            {"action": "PUT", "marble": "g", "dst": "E3", "remove": ""}
        ]
        replay_session = GameSession(replay_actions=(player1_actions, player2_actions))
        assert replay_session.is_replay_mode()

    def test_partial_replay_mode(self):
        """Test partial replay mode initialization."""
        player1_actions = [
            {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        ]
        player2_actions = [
            {"action": "PUT", "marble": "g", "dst": "E3", "remove": ""}
        ]

        session = GameSession(
            replay_actions=(player1_actions, player2_actions), partial_replay=True
        )

        assert session.replay_mode
        assert session.partial_replay

    def test_switch_to_random_play(self):
        """Test switching from replay to random play."""
        player1_actions = [
            {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        ]
        player2_actions = [
            {"action": "PUT", "marble": "g", "dst": "E3", "remove": ""}
        ]

        session = GameSession(
            replay_actions=(player1_actions, player2_actions), partial_replay=True
        )

        # Get current player and add some captures
        current_player = session.get_current_player()
        current_player.captured = {"w": 2, "g": 1, "b": 0}

        # Switch to random play
        new_current_player = session.switch_to_random_play(current_player)

        # Should now have random players
        assert isinstance(session.player1, RandomZertzPlayer)
        assert isinstance(session.player2, RandomZertzPlayer)
        assert not session.replay_mode

        # Captured marbles should be preserved
        if current_player.n == 1:
            assert session.player1.captured == {"w": 2, "g": 1, "b": 0}
        else:
            assert session.player2.captured == {"w": 2, "g": 1, "b": 0}

    def test_switch_to_random_play_without_partial_replay_raises_error(self):
        """Test that switching to random play without partial_replay flag raises error."""
        player1_actions = [
            {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        ]
        player2_actions = [
            {"action": "PUT", "marble": "g", "dst": "E3", "remove": ""}
        ]

        session = GameSession(
            replay_actions=(player1_actions, player2_actions), partial_replay=False
        )

        current_player = session.get_current_player()

        with pytest.raises(
            ValueError, match="Cannot switch to random play when partial_replay is False"
        ):
            session.switch_to_random_play(current_player)

    def test_status_reporter_callback(self):
        """Test that status reporter callback is called."""
        messages = []

        def reporter(msg):
            messages.append(msg)

        session = GameSession(seed=12345, status_reporter=reporter)

        # Should have received initialization messages
        assert len(messages) > 0
        assert any("Setting Seed" in msg for msg in messages)
        assert any("New game" in msg for msg in messages)

    def test_set_status_reporter_after_init(self):
        """Test setting status reporter after initialization."""
        session = GameSession(seed=12345)

        messages = []

        def reporter(msg):
            messages.append(msg)

        session.set_status_reporter(reporter)

        # Reset game to trigger messages
        session.reset_game()

        assert len(messages) > 0

    def test_status_reporter_shows_blitz_variant(self):
        """Test that status reporter shows blitz variant."""
        messages = []

        def reporter(msg):
            messages.append(msg)

        session = GameSession(blitz=True, status_reporter=reporter)

        # Should mention BLITZ in messages
        assert any("BLITZ" in msg for msg in messages)

    def test_human_player_initialization_player1(self):
        """Test initialization with human player 1."""
        session = GameSession(human_players=(1,))

        assert isinstance(session.player1, HumanZertzPlayer)
        assert isinstance(session.player2, RandomZertzPlayer)

    def test_human_player_initialization_player2(self):
        """Test initialization with human player 2."""
        session = GameSession(human_players=(2,))

        assert isinstance(session.player1, RandomZertzPlayer)
        assert isinstance(session.player2, HumanZertzPlayer)

    def test_human_player_initialization_both(self):
        """Test initialization with both human players."""
        session = GameSession(human_players=(1, 2))

        assert isinstance(session.player1, HumanZertzPlayer)
        assert isinstance(session.player2, HumanZertzPlayer)

    def test_switch_to_random_play_with_human_players(self):
        """Test switching to random/human play preserves human player config."""
        player1_actions = [
            {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        ]
        player2_actions = [
            {"action": "PUT", "marble": "g", "dst": "E3", "remove": ""}
        ]

        session = GameSession(
            replay_actions=(player1_actions, player2_actions),
            partial_replay=True,
            human_players=(1,),
        )

        current_player = session.get_current_player()
        session.switch_to_random_play(current_player)

        # Player 1 should be human, player 2 should be random
        assert isinstance(session.player1, HumanZertzPlayer)
        assert isinstance(session.player2, RandomZertzPlayer)

    def test_reset_game_preserves_human_player_config(self):
        """Test that reset_game preserves human player configuration."""
        session = GameSession(human_players=(1,))

        assert isinstance(session.player1, HumanZertzPlayer)
        assert isinstance(session.player2, RandomZertzPlayer)

        session.reset_game()

        # Should still have same player types
        assert isinstance(session.player1, HumanZertzPlayer)
        assert isinstance(session.player2, RandomZertzPlayer)

    def test_replay_mode_reset_preserves_replay_players(self):
        """Test that reset_game in replay mode preserves replay players."""
        player1_actions = [
            {"action": "PUT", "marble": "w", "dst": "D4", "remove": ""}
        ]
        player2_actions = [
            {"action": "PUT", "marble": "g", "dst": "E3", "remove": ""}
        ]

        session = GameSession(replay_actions=(player1_actions, player2_actions))

        assert isinstance(session.player1, ReplayZertzPlayer)
        assert isinstance(session.player2, ReplayZertzPlayer)

        session.reset_game()

        # Should still have replay players
        assert isinstance(session.player1, ReplayZertzPlayer)
        assert isinstance(session.player2, ReplayZertzPlayer)

    def test_custom_t_value(self):
        """Test initialization with custom loop detection depth."""
        session = GameSession(t=10)

        assert session.t == 10
        assert session.game.t == 10

    def test_variant_settings_preserved_across_reset(self):
        """Test that variant settings are preserved across reset."""
        session = GameSession(blitz=True)

        assert session.marbles == BLITZ_MARBLES
        assert session.win_condition == BLITZ_WIN_CONDITIONS

        session.reset_game()

        # Should still be blitz
        assert session.marbles == BLITZ_MARBLES
        assert session.win_condition == BLITZ_WIN_CONDITIONS
        assert session.game.marbles == BLITZ_MARBLES