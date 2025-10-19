import numpy as np


class TranspositionTable:
    """Cache for canonical board states and their statistics.

    Uses board canonicalization + Zobrist hashing to recognize symmetrically
    equivalent positions. Zobrist hashing reduces key size from 490 bytes to 8 bytes
    and enables much faster lookups.

    Collision detection: Stores canonical state with each entry for verification.

    Stateless: works with (board_state, global_state, config) instead of game objects.
    """

    def __init__(self, width=7, zobrist_seed=42):
        """Initialize transposition table with Zobrist hasher.

        Args:
            width: Board width (7/8/9 for 37/48/61 rings)
            zobrist_seed: Seed for Zobrist random tables (MUST be consistent!)
        """
        from learner.mcts.zobrist_hasher import ZobristHasher

        self.table = {}
        self.hits = 0
        self.misses = 0
        self.collisions = 0  # Track hash collisions
        self.zobrist = ZobristHasher(width=width, seed=zobrist_seed)

    def get_canonical_key(self, board_state, global_state, canonicalizer, config):
        """Get Zobrist hash for canonical state.

        Args:
            board_state: Board state array
            global_state: Global state array
            canonicalizer: Canonicalizer instance
            config: BoardConfig

        Returns:
            (zobrist_hash, canonical_state) tuple for collision detection
        """
        # Canonicalize state first
        canonical_state, _, _ = canonicalizer.canonicalize_state(board_state)

        # Compute Zobrist hash
        zobrist_hash = self.zobrist.hash_state(canonical_state, global_state, config)

        return zobrist_hash, canonical_state

    def lookup(self, board_state, global_state, canonicalizer, config=None):
        """Retrieve cached statistics for canonical state.

        Uses separate chaining to handle hash collisions.

        Args:
            board_state: Board state array
            global_state: Global state array
            canonicalizer: Canonicalizer instance
            config: BoardConfig (optional for backwards compat)

        Returns:
            dict with 'visits' and 'value' or None if not found
        """
        if config is None:
            # Backwards compatibility - extract from canonicalizer
            config = canonicalizer.board._get_config()

        zobrist_hash, canonical_state = self.get_canonical_key(
            board_state, global_state, canonicalizer, config
        )

        if zobrist_hash in self.table:
            # Separate chaining: table[hash] is a list of entries
            chain = self.table[zobrist_hash]

            # Search chain for matching canonical state
            for entry in chain:
                if np.array_equal(entry['canonical_state'], canonical_state):
                    self.hits += 1
                    return {'visits': entry['visits'], 'value': entry['value']}

            # Hash collision: same hash, different state
            # This is fine with separate chaining - just a miss
            self.misses += 1
            return None
        else:
            self.misses += 1
            return None

    def update(self, board_state, global_state, canonicalizer, visits_delta, value_delta, config=None):
        """Update statistics for canonical state.

        Uses separate chaining to handle hash collisions.

        Args:
            board_state: Board state array
            global_state: Global state array
            canonicalizer: Canonicalizer instance
            visits_delta: Visits to add
            value_delta: Value to add
            config: BoardConfig (optional for backwards compat)
        """
        if config is None:
            # Backwards compatibility - extract from canonicalizer
            config = canonicalizer.board._get_config()

        zobrist_hash, canonical_state = self.get_canonical_key(
            board_state, global_state, canonicalizer, config
        )

        if zobrist_hash in self.table:
            # Separate chaining: search chain for matching state
            chain = self.table[zobrist_hash]

            for entry in chain:
                if np.array_equal(entry['canonical_state'], canonical_state):
                    # Found matching entry - update it
                    entry['visits'] += visits_delta
                    entry['value'] += value_delta
                    return

            # Hash collision: same hash, different state
            # Append new entry to chain
            self.collisions += 1
            chain.append({
                'visits': visits_delta,
                'value': value_delta,
                'canonical_state': canonical_state.copy()
            })
        else:
            # New hash - create chain with first entry
            self.table[zobrist_hash] = [{
                'visits': visits_delta,
                'value': value_delta,
                'canonical_state': canonical_state.copy()
            }]

    def get_hit_rate(self):
        """Calculate cache hit rate (for performance monitoring)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    def clear(self):
        """Clear the transposition table (call between games or moves)."""
        self.table.clear()
        self.hits = 0
        self.misses = 0

    def size(self):
        """Get number of positions stored (counting all entries in all chains)."""
        return sum(len(chain) for chain in self.table.values())

    def total_visits(self):
        """Get total visits across all positions."""
        total = 0
        for chain in self.table.values():
            for entry in chain:
                total += entry['visits']
        return total

    def save(self, filepath):
        """Save transposition table to disk using NPZ format.

        Saves chains with Zobrist hashes and canonical states for collision detection.

        Args:
            filepath: Path to save file (will be created/overwritten)
        """
        if not self.table:
            # Empty table - save minimal data
            np.savez_compressed(
                filepath,
                zobrist_hashes=np.array([], dtype=np.int64),
                visits=np.array([], dtype=np.int64),
                values=np.array([], dtype=np.float64),
                canonical_states=np.array([]),
                hits=self.hits,
                misses=self.misses,
                collisions=self.collisions
            )
            return

        # Flatten all chains into arrays
        zobrist_hashes = []
        visits = []
        values = []
        canonical_states = []

        for zobrist_hash, chain in self.table.items():
            for entry in chain:
                zobrist_hashes.append(zobrist_hash)
                visits.append(entry['visits'])
                values.append(entry['value'])
                canonical_states.append(entry['canonical_state'])

        # Stack canonical states into single array for efficient storage
        if canonical_states:
            states_array = np.stack(canonical_states)
        else:
            states_array = np.array([])

        np.savez_compressed(
            filepath,
            zobrist_hashes=np.array(zobrist_hashes, dtype=np.int64),
            visits=np.array(visits, dtype=np.int64),
            values=np.array(values, dtype=np.float64),
            canonical_states=states_array,
            hits=self.hits,
            misses=self.misses,
            collisions=self.collisions
        )

    def load(self, filepath):
        """Load transposition table from disk (NPZ format).

        Reconstructs chains from flattened arrays.

        Args:
            filepath: Path to saved file

        Returns:
            True if loaded successfully, False if file not found
        """
        import os

        if not os.path.exists(filepath):
            return False

        try:
            data = np.load(filepath)

            # Reconstruct table from flattened arrays
            zobrist_hashes = data['zobrist_hashes']
            visits = data['visits']
            values = data['values']
            canonical_states = data['canonical_states']

            self.table = {}
            for i in range(len(zobrist_hashes)):
                zobrist_hash = int(zobrist_hashes[i])

                entry = {
                    'visits': int(visits[i]),
                    'value': float(values[i]),
                    'canonical_state': canonical_states[i] if len(canonical_states) > 0 else None
                }

                # Add to chain (create if first entry for this hash)
                if zobrist_hash in self.table:
                    self.table[zobrist_hash].append(entry)
                else:
                    self.table[zobrist_hash] = [entry]

            self.hits = int(data.get('hits', 0))
            self.misses = int(data.get('misses', 0))
            self.collisions = int(data.get('collisions', 0))

            return True

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False