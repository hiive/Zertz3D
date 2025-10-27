"""Player configuration system for Zèrtz."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PlayerType = Literal["random", "mcts", "human"]


@dataclass
class PlayerConfig:
    """Configuration for a single player.

    Attributes:
        player_type: Type of player ('random', 'mcts', or 'human')
        mcts_iterations: MCTS iterations per move (only for MCTS player)
        mcts_exploration: UCB1 exploration constant (default: 1.41)
        mcts_fpu_reduction: First Play Urgency reduction (None = disabled)
        mcts_widening_constant: Progressive widening constant (None = disabled)
        mcts_rave_constant: RAVE constant (None = disabled, 300-3000 = enabled)
        mcts_use_transposition_table: Enable transposition table
        mcts_use_transposition_lookups: Use cached stats for initialization
        mcts_clear_table_each_move: Clear transposition table between moves
        mcts_max_simulation_depth: Max rollout depth (None = full game)
        mcts_time_limit: Max search time per move in seconds (None = no limit)
        mcts_parallel: Use parallel MCTS search
        mcts_num_workers: Number of threads for parallel search
        mcts_verbose: Print search statistics
        rng_seed: Random seed for this player (None = unseeded)
    """

    player_type: PlayerType = "random"

    # MCTS hyperparameters
    mcts_iterations: int = 1000
    mcts_exploration: float = 1.41
    mcts_fpu_reduction: float | None = None
    mcts_widening_constant: float | None = None
    mcts_rave_constant: float | None = None
    mcts_use_transposition_table: bool = True
    mcts_use_transposition_lookups: bool = True
    mcts_clear_table_each_move: bool = True
    mcts_max_simulation_depth: int | None = None
    mcts_time_limit: float | None = None
    mcts_parallel: bool = False
    mcts_num_workers: int = 16
    mcts_verbose: bool = False

    # General player settings
    rng_seed: int | None = None

    @classmethod
    def random(cls, seed: int | None = None) -> PlayerConfig:
        """Create a random player configuration."""
        return cls(player_type="random", rng_seed=seed)

    @classmethod
    def human(cls) -> PlayerConfig:
        """Create a human player configuration."""
        return cls(player_type="human")

    @classmethod
    def mcts(
        cls,
        iterations: int = 1000,
        *,
        exploration: float = 1.41,
        fpu_reduction: float | None = None,
        widening_constant: float | None = None,
        rave_constant: float | None = None,
        use_transposition_table: bool = True,
        use_transposition_lookups: bool = True,
        clear_table_each_move: bool = True,
        max_simulation_depth: int | None = None,
        time_limit: float | None = None,
        parallel: bool = False,
        num_workers: int = 16,
        verbose: bool = False,
        seed: int | None = None,
    ) -> PlayerConfig:
        """Create an MCTS player configuration.

        Args:
            iterations: MCTS iterations per move
            exploration: UCB1 exploration constant (default: √2 ≈ 1.41)
            fpu_reduction: First Play Urgency reduction (None = disabled, 0.2 = moderate)
            widening_constant: Progressive widening constant (None = disabled, 10.0 = moderate)
            rave_constant: RAVE constant (None = disabled, 300-3000 = enabled, 1000 = balanced)
            use_transposition_table: Enable transposition table caching
            use_transposition_lookups: Use cached stats to initialize nodes
            clear_table_each_move: Clear transposition table between moves
            max_simulation_depth: Max rollout depth (None = play to end)
            time_limit: Max search time per move in seconds (None = no limit)
            parallel: Use parallel MCTS search
            num_workers: Number of threads for parallel search
            verbose: Print search statistics after each move
            seed: Random seed for MCTS player (None = unseeded)
        """
        return cls(
            player_type="mcts",
            mcts_iterations=iterations,
            mcts_exploration=exploration,
            mcts_fpu_reduction=fpu_reduction,
            mcts_widening_constant=widening_constant,
            mcts_rave_constant=rave_constant,
            mcts_use_transposition_table=use_transposition_table,
            mcts_use_transposition_lookups=use_transposition_lookups,
            mcts_clear_table_each_move=clear_table_each_move,
            mcts_max_simulation_depth=max_simulation_depth,
            mcts_time_limit=time_limit,
            mcts_parallel=parallel,
            mcts_num_workers=num_workers,
            mcts_verbose=verbose,
            rng_seed=seed,
        )


def parse_player_spec(spec: str) -> PlayerConfig:
    """Parse a player specification string into a PlayerConfig.

    Format:
        TYPE[:PARAM=VALUE,PARAM=VALUE,...]

    Examples:
        "random" -> Random player
        "human" -> Human player
        "mcts" -> MCTS with defaults
        "mcts:iterations=500" -> MCTS with 500 iterations
        "mcts:iterations=1000,exploration=2.0,rave=1000" -> MCTS with custom params

    Supported MCTS parameters:
        - iterations (int): MCTS iterations per move
        - exploration (float): UCB1 exploration constant
        - fpu (float): First Play Urgency reduction
        - widening (float): Progressive widening constant
        - rave (float): RAVE constant
        - transposition (bool): Enable transposition table (1/0/true/false)
        - lookups (bool): Use transposition lookups (1/0/true/false)
        - clear (bool): Clear table each move (1/0/true/false)
        - max_depth (int): Max simulation depth
        - time_limit (float): Time limit in seconds
        - parallel (bool): Use parallel search (1/0/true/false)
        - workers (int): Number of worker threads
        - verbose (bool): Print statistics (1/0/true/false)
        - seed (int): Random seed
    """
    parts = spec.split(":", 1)
    player_type = parts[0].lower()

    if player_type not in ["random", "mcts", "human"]:
        raise ValueError(f"Invalid player type: {player_type}. Must be 'random', 'mcts', or 'human'")

    # Parse parameters if provided
    params = {}
    if len(parts) == 2:
        param_str = parts[1]
        for param_pair in param_str.split(","):
            param_pair = param_pair.strip()
            if not param_pair:
                continue
            if "=" not in param_pair:
                raise ValueError(f"Invalid parameter format: {param_pair}. Expected PARAM=VALUE")
            key, value = param_pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Convert value to appropriate type
            if key in ["iterations", "max_depth", "workers", "seed"]:
                params[key] = int(value)
            elif key in ["exploration", "fpu", "widening", "rave", "time_limit"]:
                params[key] = float(value)
            elif key in ["transposition", "lookups", "clear", "parallel", "verbose"]:
                params[key] = value.lower() in ["1", "true", "yes", "on"]
            else:
                raise ValueError(f"Unknown parameter: {key}")

    # Create player config based on type
    if player_type == "random":
        return PlayerConfig.random(seed=params.get("seed"))
    elif player_type == "human":
        return PlayerConfig.human()
    else:  # mcts
        # Map short parameter names to full names
        mcts_kwargs = {
            "iterations": params.get("iterations", 1000),
            "exploration": params.get("exploration", 1.41),
            "fpu_reduction": params.get("fpu"),
            "widening_constant": params.get("widening"),
            "rave_constant": params.get("rave"),
            "use_transposition_table": params.get("transposition", True),
            "use_transposition_lookups": params.get("lookups", True),
            "clear_table_each_move": params.get("clear", True),
            "max_simulation_depth": params.get("max_depth"),
            "time_limit": params.get("time_limit"),
            "parallel": params.get("parallel", False),
            "num_workers": params.get("workers", 16),
            "verbose": params.get("verbose", False),
            "seed": params.get("seed"),
        }
        return PlayerConfig.mcts(**mcts_kwargs)