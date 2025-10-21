import random

import numpy as np

from game.zertz_player import ZertzPlayer
from learner.mcts.mcts_tree import MCTSTree
from learner.mcts.transposition_table import TranspositionTable
from learner.mcts.backend import get_backend, Backend, HAS_RUST

if HAS_RUST:
    import hiivelabs_zertz_mcts


class MCTSZertzPlayer(ZertzPlayer):
    """MCTS-based Zertz player.

    Uses Monte Carlo Tree Search with UCB1 selection and optional transposition table.
    Drop-in replacement for RandomZertzPlayer with same interface.
    Supports both Python and Rust backends.
    """

    def __init__(self, game, n, iterations=1000, exploration_constant=1.41,
                 max_simulation_depth=None, use_transposition_table=True,
                 use_transposition_lookups=True, time_limit=None, verbose=False,
                 clear_table_each_move=True, parallel='multiprocess', num_workers=16,
                 backend='auto', rng_seed=None):
        """Initialize MCTS player.

        Args:
            game: ZertzGame instance
            n: Player number (1 or 2)
            iterations: MCTS iterations per move (default: 1000)
            exploration_constant: UCB1 exploration (default: √2 ≈ 1.41)
            max_simulation_depth: Max rollout depth (None = play to end)
            use_transposition_table: Enable symmetry caching (serial/thread: shared table, multiprocess: per-process tables)
            use_transposition_lookups: Use cached stats to initialize nodes
            time_limit: Max search time per move in seconds (None = no limit)
            verbose: Print search statistics after each move
            clear_table_each_move: Clear transposition table between moves (serial/thread mode only, multiprocess creates fresh tables)
            parallel: Parallelization mode: False (serial), 'thread' (threaded), 'multiprocess' (default)
            num_workers: Number of threads/processes for parallel search (default: 16)
            backend: Backend to use: 'python', 'rust', or 'auto' (default: auto-detect)
            rng_seed: Optional integer seed used to initialize randomness for reproducible runs
        """
        super().__init__(game, n)

        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_simulation_depth = max_simulation_depth
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
        self._python_rng_state = None
        self._python_random_state = None
        self._rust_seed_initialized = False

        if rng_seed is not None:
            python_rng = np.random.RandomState(rng_seed)
            self._python_rng_state = python_rng.get_state()
            python_random = random.Random(rng_seed)
            self._python_random_state = python_random.getstate()

        # Determine backend
        self.backend = get_backend(backend)

        # Initialize appropriate backend
        if self.backend == Backend.RUST:
            # Rust backend doesn't support multiprocess mode - use thread mode instead
            if self.parallel == 'multiprocess':
                if self.verbose:
                    print("Note: Rust backend doesn't support multiprocess mode, using threaded mode")
                self.parallel = 'thread'

            # Create Rust MCTS searcher
            self.rust_mcts = hiivelabs_zertz_mcts.MCTSSearch(
                exploration_constant=exploration_constant,
                progressive_widening=True,
                widening_constant=10.0,
                use_transposition_table=use_transposition_table,
                use_transposition_lookups=use_transposition_lookups,
            )
            self.rust_mcts.set_transposition_table_enabled(use_transposition_table)
            self.rust_mcts.set_transposition_lookups(use_transposition_lookups)

            self.transposition_table = None  # Rust manages its own
        else:
            # Python backend
            self.mcts = MCTSTree()

            # Initialize transposition table (only used in serial/thread modes)
            if use_transposition_table:
                self.transposition_table = TranspositionTable()
            else:
                self.transposition_table = None

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
            return ("CAP", (c1[0], c2[0], c3[0]))
        elif c1.size > 1:
            # Multiple captures available - need MCTS to decide
            pass
        else:
            # No captures - check placements
            p1, p2, p3 = placement_mask.nonzero()
            if p1.size == 1:
                # Exactly one placement - forced move, no decision needed
                return ("PUT", (p1[0], p2[0], p3[0]))
            elif p1.size == 0:
                # No moves available - must pass
                return ("PASS", None)
            # Multiple placements available - need MCTS to decide

        # Route to appropriate backend
        if self.backend == Backend.RUST:
            action = self._rust_search()
        else:
            action = self._python_search()

        return action

    def _rust_search(self):
        """Run MCTS search using Rust backend."""
        # Get current state
        state = self.game.get_current_state()
        spatial = state['spatial'].astype(np.float32)
        global_state = state['global'].astype(np.float32)

        if self.use_transposition_table and self.clear_table_each_move:
            self.rust_mcts.clear_transposition_table()

        if self.use_transposition_table and self.clear_table_each_move:
            self.rust_mcts.clear_transposition_table()

        if self.rng_seed is not None and not self._rust_seed_initialized:
            self.rust_mcts.set_seed(self.rng_seed)
            self._rust_seed_initialized = True

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

    def _python_search(self):
        """Run MCTS search using Python backend."""
        # Clear transposition table if configured
        if self.clear_table_each_move and self.transposition_table:
            self.transposition_table.clear()

        if self.rng_seed is not None:
            original_np_state = np.random.get_state()
            original_random_state = random.getstate()
            if self._python_rng_state is not None:
                np.random.set_state(self._python_rng_state)
            if self._python_random_state is not None:
                random.setstate(self._python_random_state)

        # Run MCTS search (multiprocess, threaded, or serial)
        if self.parallel == 'multiprocess':
            action = self.mcts.search_multiprocess(
                self.game,
                iterations=self.iterations,
                exploration_constant=self.exploration_constant,
                max_simulation_depth=self.max_simulation_depth,
                transposition_table=self.transposition_table,  # Creates per-process tables
                use_transposition_lookups=self.use_transposition_lookups,
                time_limit=self.time_limit,
                verbose=self.verbose,
                num_processes=self.num_workers
            )
        elif self.parallel == 'thread':
            action = self.mcts.search_parallel(
                self.game,
                iterations=self.iterations,
                exploration_constant=self.exploration_constant,
                max_simulation_depth=self.max_simulation_depth,
                transposition_table=self.transposition_table,
                use_transposition_lookups=self.use_transposition_lookups,
                time_limit=self.time_limit,
                verbose=self.verbose,
                num_threads=self.num_workers
            )
        else:  # Serial mode
            action = self.mcts.search(
                self.game,
                iterations=self.iterations,
                exploration_constant=self.exploration_constant,
                max_simulation_depth=self.max_simulation_depth,
                transposition_table=self.transposition_table,
                use_transposition_lookups=self.use_transposition_lookups,
                time_limit=self.time_limit,
                verbose=self.verbose
            )

        self._last_root_children = getattr(self.mcts, "_last_root_children", 0)
        self._last_root_visits = getattr(self.mcts, "_last_root_visits", 0)
        self._last_root_value = getattr(self.mcts, "_last_root_value", 0.0)

        if self.rng_seed is not None:
            self._python_rng_state = np.random.get_state()
            self._python_random_state = random.getstate()
            np.random.set_state(original_np_state)
            random.setstate(original_random_state)

        return action
