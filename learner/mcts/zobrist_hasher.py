"""Zobrist hashing for efficient transposition table lookups.

Zobrist hashing assigns random numbers to each feature, then XORs them together
to create a compact hash. This enables:
- Fast lookups (8 bytes vs 490 bytes)
- Incremental updates (XOR in/out changed features)
- Collision detection (store full state for verification)
"""

import numpy as np


class ZobristHasher:
    """Zobrist hash generator for Zertz game states.

    Creates 64-bit hashes from board state features:
    - Ring positions (present/absent)
    - Marble positions (white/gray/black)
    - Captured marbles (per player, per type, per count)
    - Current player
    """

    def __init__(self, width=7, seed=42):
        """Initialize Zobrist random number tables.

        Args:
            width: Board width (7 for 37 rings, 8 for 48, 9 for 61)
            seed: Random seed for reproducibility (MUST be same across runs)
        """
        # Use PCG64 for compatibility with Rust implementation
        # Note: This matches the Rust rand_pcg::Pcg64 generator
        rng = np.random.Generator(np.random.PCG64(seed))

        # Random numbers for ring existence (width × width)
        self.ring_zobrist = rng.integers(0, 2**63, (width, width), dtype=np.uint64)

        # Random numbers for marbles at each position (3 types × width × width)
        self.marble_zobrist = {
            'w': rng.integers(0, 2**63, (width, width), dtype=np.uint64),
            'g': rng.integers(0, 2**63, (width, width), dtype=np.uint64),
            'b': rng.integers(0, 2**63, (width, width), dtype=np.uint64),
        }

        # Random numbers for captured marbles (2 players × 3 types × max 20 each)
        # We hash the count directly, so need numbers for each possible count
        self.captured_zobrist = rng.integers(0, 2**63, (2, 3, 20), dtype=np.uint64)

        # Random numbers for supply marbles (3 types × max 20 each)
        self.supply_zobrist = rng.integers(0, 2**63, (3, 20), dtype=np.uint64)

        # Random number for current player (XOR if player 2)
        self.player_zobrist = rng.integers(0, 2**63, dtype=np.uint64)

        self.width = width

    def hash_state(self, board_state, global_state, config):
        """Compute Zobrist hash for a game state.

        Args:
            board_state: (L, H, W) spatial state array
            global_state: (10,) global state array
            config: BoardConfig

        Returns:
            64-bit hash as Python int
        """
        h = np.uint64(0)

        # Hash rings
        ring_layer = board_state[config.ring_layer]
        for y in range(config.width):
            for x in range(config.width):
                if ring_layer[y, x] == 1:
                    h ^= self.ring_zobrist[y, x]

        # Hash marbles
        marble_type_to_layer = {
            'w': config.marble_to_layer['w'],
            'g': config.marble_to_layer['g'],
            'b': config.marble_to_layer['b'],
        }

        for marble_type, zobrist_table in self.marble_zobrist.items():
            layer = marble_type_to_layer[marble_type]
            for y in range(config.width):
                for x in range(config.width):
                    if board_state[layer, y, x] == 1:
                        h ^= zobrist_table[y, x]

        # Hash supply marbles
        supply = global_state[config.supply_slice]
        for i, count in enumerate(supply):
            count = int(count)
            if count > 0 and count < 20:
                h ^= self.supply_zobrist[i, count]

        # Hash captured marbles (player 1)
        p1_captured = global_state[config.p1_cap_slice]
        for i, count in enumerate(p1_captured):
            count = int(count)
            if count > 0 and count < 20:
                h ^= self.captured_zobrist[0, i, count]

        # Hash captured marbles (player 2)
        p2_captured = global_state[config.p2_cap_slice]
        for i, count in enumerate(p2_captured):
            count = int(count)
            if count > 0 and count < 20:
                h ^= self.captured_zobrist[1, i, count]

        # Hash current player
        if int(global_state[config.cur_player]) == 1:
            h ^= self.player_zobrist

        return int(h)  # Convert to Python int for dict key

    def hash_canonical_state(self, canonical_state, config):
        """Hash a canonical state array directly (faster path).

        Args:
            canonical_state: Already canonicalized state array
            config: BoardConfig

        Returns:
            64-bit hash as Python int
        """
        # For canonical states, we only need to hash the spatial features
        # Global state (captures, supply, player) is extracted separately
        h = np.uint64(0)

        # Hash rings
        ring_layer = canonical_state[config.ring_layer]
        for y in range(config.width):
            for x in range(config.width):
                if ring_layer[y, x] == 1:
                    h ^= self.ring_zobrist[y, x]

        # Hash marbles
        marble_type_to_layer = {
            'w': config.marble_to_layer['w'],
            'g': config.marble_to_layer['g'],
            'b': config.marble_to_layer['b'],
        }

        for marble_type, zobrist_table in self.marble_zobrist.items():
            layer = marble_type_to_layer[marble_type]
            for y in range(config.width):
                for x in range(config.width):
                    if canonical_state[layer, y, x] == 1:
                        h ^= zobrist_table[y, x]

        return int(h)