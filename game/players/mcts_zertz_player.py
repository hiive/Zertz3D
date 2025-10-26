import numpy as np

from game.zertz_player import ZertzPlayer
from game.constants import BLITZ_WIN_CONDITIONS
from learner.mcts.backend import ensure_rust_backend

import hiivelabs_zertz_mcts


class MCTSZertzPlayer(ZertzPlayer):
    """MCTS-based Zertz player.

    Uses Monte Carlo Tree Search with UCB1 selection and optional transposition table.
    Drop-in replacement for RandomZertzPlayer with same interface.
    """

    def __init__(self, game, n, iterations=1000, exploration_constant=1.41,
                 max_simulation_depth=None, fpu_reduction=None,
                 use_transposition_table=True, use_transposition_lookups=True,
                 time_limit=None, verbose=False, clear_table_each_move=True,
                 parallel='multiprocess', num_workers=16,
                 rng_seed=None, widening_constant=None, progress_callback=None,
                 progress_interval_ms=100):
        """Initialize MCTS player.

        Args:
            game: ZertzGame instance
            n: Player number (1 or 2)
            iterations: MCTS iterations per move (default: 1000)
            exploration_constant: UCB1 exploration (default: √2 ≈ 1.41)
            max_simulation_depth: Max rollout depth (None = play to end)
            fpu_reduction: First Play Urgency reduction (None = disabled, 0.2 = moderate)
            use_transposition_table: Enable symmetry caching
            use_transposition_lookups: Use cached stats to initialize nodes
            time_limit: Max search time per move in seconds (None = no limit)
            verbose: Print search statistics after each move
            clear_table_each_move: Clear transposition table between moves
            parallel: Parallelization mode: False (serial), 'thread' (threaded), 'multiprocess' (uses thread mode)
            num_workers: Number of threads for parallel search (default: 16)
            rng_seed: Optional integer seed used to initialize randomness for reproducible runs
            widening_constant: Progressive widening constant (None = disabled, e.g. 10.0 = moderate)
            progress_callback: Optional callback for MCTS search progress
            progress_interval_ms: Interval in milliseconds for progress updates (default: 100ms)
        """
        super().__init__(game, n)

        # Ensure Rust backend is available
        ensure_rust_backend()

        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_simulation_depth = max_simulation_depth
        self.fpu_reduction = fpu_reduction
        self.widening_constant = widening_constant
        self.use_transposition_table = use_transposition_table
        self.use_transposition_lookups = use_transposition_lookups
        self.time_limit = time_limit
        self.verbose = verbose
        self.clear_table_each_move = clear_table_each_move
        self.parallel = parallel
        self.num_workers = num_workers
        self._last_root_children = 0
        self._last_root_visits = 0
        self._last_root_value = 0.0
        self.rng_seed = rng_seed
        self._rust_seed_initialized = False
        self.progress_callback = progress_callback
        self.progress_interval_ms = progress_interval_ms

        # Note: Rust backend doesn't support multiprocess mode - use thread mode instead
        if self.parallel == 'multiprocess':
            if self.verbose:
                print("Note: Using threaded mode for parallel MCTS")
            self.parallel = 'thread'

        # Create Rust MCTS searcher
        self.rust_mcts = hiivelabs_zertz_mcts.MCTSSearch(
            exploration_constant=exploration_constant,
            widening_constant=widening_constant,
            fpu_reduction=fpu_reduction,
            use_transposition_table=use_transposition_table,
            use_transposition_lookups=use_transposition_lookups,
        )
        self.rust_mcts.set_transposition_table_enabled(use_transposition_table)
        self.rust_mcts.set_transposition_lookups(use_transposition_lookups)

        self.transposition_table = None  # Rust manages its own

    def get_action(self):
        """Select best action using MCTS.

        Returns:
            Action tuple: (action_type, action_data)
        """
        # Optimization: If only one legal move, return it immediately (no search needed)
        placement_mask, capture_mask = self.game.get_valid_actions()

        # Check captures first (they're mandatory)
        c1, c2, c3 = capture_mask.nonzero()
        if c1.size == 1:
            # Exactly one capture - mandatory move, no decision needed
            return ("CAP", (int(c1[0]), int(c2[0]), int(c3[0])))
        elif c1.size > 1:
            # Multiple captures available - need MCTS to decide
            pass
        else:
            # No captures - check placements
            p1, p2, p3 = placement_mask.nonzero()
            if p1.size == 1:
                # Exactly one placement - forced move, no decision needed
                return ("PUT", (int(p1[0]), int(p2[0]), int(p3[0])))
            elif p1.size == 0:
                # No moves available - must pass
                return ("PASS", None)
            # Multiple placements available - need MCTS to decide

        # Run MCTS search
        action = self._search()
        return action

    def get_last_action_scores(self):
        """Get normalized scores for all legal actions from last search.

        Returns:
            Dict mapping action tuples to normalized scores [0.0, 1.0]
        """
        # Get statistics from Rust backend
        child_stats = self.rust_mcts.last_child_statistics()

        # Convert to dictionary with action tuples as keys
        action_scores = {}
        for action_type, action_data, score in child_stats:
            action = (action_type, action_data)
            action_scores[action] = score

        return action_scores

    def _is_blitz_mode(self):
        """Detect if the game is in blitz mode by checking win conditions."""
        # Blitz mode is identified by the first win condition: {"w": 2, "g": 2, "b": 2}
        # Standard mode has: {"w": 3, "g": 3, "b": 3}
        return self.game.win_con == BLITZ_WIN_CONDITIONS

    def _search(self):
        """Run MCTS search using Rust backend."""
        # Get current state
        state = self.game.get_current_state()
        spatial = state['spatial'].astype(np.float32)
        global_state = state['global'].astype(np.float32)

        if self.use_transposition_table and self.clear_table_each_move:
            self.rust_mcts.clear_transposition_table()

        if self.rng_seed is not None and not self._rust_seed_initialized:
            self.rust_mcts.set_seed(self.rng_seed)
            self._rust_seed_initialized = True

        # Detect game mode and pass to Rust
        is_blitz = self._is_blitz_mode()

        rust_kwargs = dict(
            rings=self.game.board.rings,
            iterations=self.iterations,
            t=getattr(self.game.board, 't', 1),
            max_depth=self.max_simulation_depth,
            time_limit=self.time_limit,
            use_transposition_table=self.use_transposition_table,
            use_transposition_lookups=self.use_transposition_lookups,
            clear_table=self.clear_table_each_move,
            verbose=self.verbose,
            blitz=is_blitz,
            progress_callback=self.progress_callback,
            progress_interval_ms=self.progress_interval_ms if self.progress_callback else None,
        )

        # Run search (serial or parallel)
        if self.parallel == 'thread':
            action_type, action_data = self.rust_mcts.search_parallel(
                spatial,
                global_state,
                num_threads=self.num_workers,
                **rust_kwargs,
            )
        else:  # Serial mode
            action_type, action_data = self.rust_mcts.search(
                spatial,
                global_state,
                **rust_kwargs,
            )

        self._last_root_children = self.rust_mcts.last_root_children()
        self._last_root_visits = self.rust_mcts.last_root_visits()
        self._last_root_value = self.rust_mcts.last_root_value()

        return (action_type, action_data)