"""Game session management for Zertz 3D.

Manages a single game's lifecycle including board state, players, and seed management.
"""

import random
import time
import hashlib
from typing import Callable
import numpy as np

from game.constants import (
    STANDARD_MARBLES,
    BLITZ_MARBLES,
    STANDARD_WIN_CONDITIONS,
    BLITZ_WIN_CONDITIONS,
)
from game.zertz_game import ZertzGame
from game.zertz_player import RandomZertzPlayer, ReplayZertzPlayer, HumanZertzPlayer
from game.players.mcts_zertz_player import MCTSZertzPlayer
from game.player_config import PlayerConfig


class GameSession:
    """Manages a single game's lifecycle (board state, players, current game)."""

    def __init__(
        self,
        rings=37,
        blitz=False,
        seed=None,
        replay_actions=None,
        partial_replay=False,
        t=5,
        status_reporter: Callable[[str], None] | None = None,
        human_players: tuple[int, ...] | None = None,
        player1_config: PlayerConfig | None = None,
        player2_config: PlayerConfig | None = None,
    ):
        """Initialize a game session.

        Args:
            rings: Number of rings on the board (37, 48, or 61)
            blitz: Whether this is a blitz variant
            seed: Random seed for reproducibility (auto-generated if None)
            replay_actions: Tuple of (player1_actions, player2_actions) for replay mode
            partial_replay: If True, continue with random play after replay ends
            t: History depth for loop detection
            human_players: Tuple of player numbers (1, 2) that are human-controlled (deprecated, use player configs)
            player1_config: Configuration for player 1 (default: random)
            player2_config: Configuration for player 2 (default: random)
        """
        self.rings = rings
        self.blitz = blitz
        self.t = t
        self._status_reporter: Callable[[str], None] | None = status_reporter

        # Initialize player configs (use defaults if not provided)
        self.player1_config = player1_config if player1_config is not None else PlayerConfig.random()
        self.player2_config = player2_config if player2_config is not None else PlayerConfig.random()

        # Override with human_players parameter if provided (for backward compatibility)
        # Note: human_players takes precedence over explicitly provided configs
        human_players_set = set(human_players or ())
        if 1 in human_players_set:
            self.player1_config = PlayerConfig.human()
        if 2 in human_players_set:
            self.player2_config = PlayerConfig.human()

        # Set marbles and win conditions based on variant
        if blitz:
            self.marbles = BLITZ_MARBLES
            self.win_condition = BLITZ_WIN_CONDITIONS
        else:
            self.marbles = STANDARD_MARBLES
            self.win_condition = STANDARD_WIN_CONDITIONS

        # Validate blitz mode (only works with 37 rings)
        if self.blitz and self.rings != 37:
            raise ValueError("Blitz mode only works with 37 rings")

        # Replay mode setup
        self.replay_mode = replay_actions is not None
        self.partial_replay = partial_replay
        self.replay_actions = replay_actions
        self.human_players = set(human_players or ())

        # Seed management
        self.current_seed = None
        if not self.replay_mode:
            if seed is None:
                seed = int(time.time())
            self.current_seed = seed
            self._apply_seed(seed)

        # Game state
        self.game = None
        self.player1 = None
        self.player2 = None
        self.games_played = 0

        # Initialize first game
        self.reset_game()

    def _apply_seed(self, seed):
        """Apply a seed to both random number generators.

        Args:
            seed: Random seed value
        """
        self._report(f"-- Setting Seed: {seed}")
        np.random.seed(seed)
        random.seed(seed)

    def _generate_next_seed(self):
        """Generate the next seed deterministically from the current seed using hash.

        Returns:
            int: New seed value
        """
        # Hash the current seed to get a new one
        hash_obj = hashlib.sha256(str(self.current_seed).encode())
        # Take first 8 bytes and convert to integer
        new_seed = int.from_bytes(hash_obj.digest()[:8], byteorder="big")
        # Keep it in a reasonable range (32-bit unsigned int)
        new_seed = new_seed % (2**32)
        return new_seed

    def _create_player_from_config(self, player_num: int, config: PlayerConfig):
        """Create a player from a PlayerConfig.

        Args:
            player_num: Player number (1 or 2)
            config: PlayerConfig describing player type and parameters

        Returns:
            ZertzPlayer: Configured player instance with name set from config
        """
        if config.player_type == "human":
            player = HumanZertzPlayer(self.game, player_num)
        elif config.player_type == "random":
            player = RandomZertzPlayer(self.game, player_num)
        elif config.player_type == "mcts":
            player = MCTSZertzPlayer(
                self.game,
                n=player_num,
                iterations=config.mcts_iterations,
                exploration_constant=config.mcts_exploration,
                max_simulation_depth=config.mcts_max_simulation_depth,
                fpu_reduction=config.mcts_fpu_reduction,
                widening_constant=config.mcts_widening_constant,
                rave_constant=config.mcts_rave_constant,
                use_transposition_table=config.mcts_use_transposition_table,
                use_transposition_lookups=config.mcts_use_transposition_lookups,
                clear_table_each_move=config.mcts_clear_table_each_move,
                time_limit=config.mcts_time_limit,
                verbose=config.mcts_verbose,
                num_workers=config.mcts_num_workers,
                rng_seed=config.rng_seed,
            )
        else:
            raise ValueError(f"Unknown player type: {config.player_type}")

        # Set player name from config if provided
        if config.name is not None:
            player.name = config.name

        return player

    def reset_game(self):
        """Reset the game state for a new game.

        This creates a new game instance and players.
        """
        variant_text = " (BLITZ)" if self.blitz else ""
        self._report(f"** New game{variant_text} **")

        # Generate new seed for non-replay games (only after the first game)
        if (
            not self.replay_mode
            and self.current_seed is not None
            and self.game is not None
        ):
            self.current_seed = self._generate_next_seed()
            self._apply_seed(self.current_seed)

        # Create new game instance
        self.game = ZertzGame(self.rings, self.marbles, self.win_condition, self.t)

        # DEBUG: Log game reset
        game_num = getattr(self, '_game_count', 0) + 1
        self._game_count = game_num
        # print(f"[DEBUG] Starting game #{game_num}, cur_player={self.game.board.get_cur_player()}")

        # Create players based on mode
        if self.replay_mode:
            self._report("-- Replay Mode --")
            player1_actions, player2_actions = self.replay_actions
            self.player1 = ReplayZertzPlayer(self.game, 1, player1_actions)
            self.player2 = ReplayZertzPlayer(self.game, 2, player2_actions)

            # Apply names from configs (loaded from replay file)
            if self.player1_config.name:
                self.player1.name = self.player1_config.name
            if self.player2_config.name:
                self.player2.name = self.player2_config.name
        else:
            # Create players from configs (names are set from config in _create_player_from_config)
            self.player1 = self._create_player_from_config(1, self.player1_config)
            self.player2 = self._create_player_from_config(2, self.player2_config)

    def get_current_player(self):
        """Get the player whose turn it is.

        Returns:
            ZertzPlayer: Current player (player1 or player2)
        """
        p_ix = self.game.get_cur_player_value()
        return self.player1 if p_ix == 1 else self.player2

    def switch_to_random_play(self, current_player):
        """Switch from replay mode to random play (for partial replay).

        Args:
            current_player: The player that was active when replay ended

        Returns:
            ZertzPlayer: The new current player
        """
        if not self.partial_replay:
            raise ValueError(
                "Cannot switch to random play when partial_replay is False"
            )

        self._report("Replay finished - continuing with configured play")

        # Create players from configs
        self.player1 = self._create_player_from_config(1, self.player1_config)
        self.player2 = self._create_player_from_config(2, self.player2_config)

        # Preserve captured marbles
        if current_player.n == 1:
            self.player1.captured = current_player.captured
        else:
            self.player2.captured = current_player.captured

        self.replay_mode = False

        return self.get_current_player()

    def increment_games_played(self):
        """Increment the games played counter."""
        self.games_played += 1

    def get_seed(self):
        """Get the current seed.

        Returns:
            int or None: Current seed value
        """
        return self.current_seed

    def is_replay_mode(self):
        """Check if session is in replay mode.

        Returns:
            bool: True if in replay mode
        """
        return self.replay_mode

    def is_partial_replay(self):
        """Check if session allows partial replay (continuing with random play after replay ends).

        Returns:
            bool: True if partial replay is enabled
        """
        return self.partial_replay

    def get_games_played(self):
        """Get the number of games played.

        Returns:
            int: Number of games played
        """
        return self.games_played

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
