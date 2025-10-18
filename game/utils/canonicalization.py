"""Board state canonicalization and transformation utilities.

This module provides the CanonicalizationManager class which handles all symmetry
transformations, state canonicalization, and action transformations for Zèrtz boards.
Supports D6 (37/61 rings) and D3 (48 rings) dihedral group symmetries.
"""

from enum import Flag, auto
import numpy as np


class TransformFlags(Flag):
    """Flags for controlling which transforms to use in canonicalization."""
    ROTATION = auto()    # Include rotational symmetries
    MIRROR = auto()      # Include mirror symmetries
    TRANSLATION = auto() # Include translation symmetries

    # Common combinations
    ALL = ROTATION | MIRROR | TRANSLATION
    ROTATION_MIRROR = ROTATION | MIRROR
    NONE = 0


class CanonicalizationManager:
    """Manages board state canonicalization and transformations.

    This class handles all symmetry operations, coordinate transformations,
    and state canonicalization for a Zèrtz board. It operates on board state
    arrays and provides methods to find canonical representations and transform
    actions/masks between different symmetry orientations.

    The manager maintains a reference to the board to access properties like
    width, layout, and layer indices, but does not modify the board state itself.

    Attributes:
        board: Reference to the ZertzBoard instance
    """

    def __init__(self, board):
        """Initialize the canonicalization manager.

        Args:
            board: ZertzBoard instance to manage transformations for
        """
        self.board = board

    # =========================  GEOMETRIC RING REMOVAL  =========================

    def yx_to_cartesian(self, y, x):
        """Convert board indices (y, x) to Cartesian coordinates.

        Uses standard pointy-top hexagonal grid conversion:
        - q = x - center
        - r = y - x
        - xc = sqrt(3) * (q + r/2)
        - yc = 1.5 * r

        Returns:
            tuple: (xc, yc) Cartesian coordinates
        """
        c = self.board.width // 2
        q = x - c
        r = y - x
        sqrt3 = np.sqrt(3)
        xc = sqrt3 * (q + r / 2.0)
        yc = 1.5 * r
        return xc, yc

    def is_removable_geometric(self, index, ring_radius=None):
        """Check if a ring can be removed using geometric collision detection.

        Uses actual Cartesian coordinates and perpendicular distance calculations
        to verify that the ring can be slid out in some direction without colliding
        with any other rings.

        This method validates that the simple adjacency-based heuristic (_is_removable)
        is geometrically correct.

        Args:
            index: (y, x) position of the ring to check
            ring_radius: Radius of a ring (default √3/2 ≈ 0.866 for unit grid)

        Returns:
            bool: True if the ring can be removed
        """
        if ring_radius is None:
            ring_radius = (
                np.sqrt(3) / 2.0
            )  # Correct radius for touching rings in unit hex grid

        y, x = index

        # Ring must be empty (no marble on it)
        if np.sum(self.board.state[self.board.BOARD_LAYERS, y, x]) != 1:
            return False

        # Get Cartesian position of this ring
        ring_x, ring_y = self.yx_to_cartesian(y, x)

        # Get all neighbor positions
        neighbors = self.board.get_neighbors(index)

        # Check each pair of consecutive empty neighbors - each creates a potential slide direction
        for i in range(len(neighbors)):
            curr = neighbors[i]
            next_neighbor = neighbors[(i + 1) % len(neighbors)]

            curr_empty = (
                not self.board._is_inbounds(curr) or self.board.state[self.board.RING_LAYER][curr] == 0
            )
            next_empty = (
                not self.board._is_inbounds(next_neighbor)
                or self.board.state[self.board.RING_LAYER][next_neighbor] == 0
            )

            if curr_empty and next_empty:
                # Found a gap. Calculate the slide direction (angle bisector of the gap)
                # The gap is between directions i and i+1
                dy1, dx1 = self.board.DIRECTIONS[i]
                dy2, dx2 = self.board.DIRECTIONS[(i + 1) % len(self.board.DIRECTIONS)]

                # Convert direction offsets to actual neighbor positions, then to Cartesian vectors
                neighbor1_pos = (y + dy1, x + dx1)
                neighbor2_pos = (y + dy2, x + dx2)

                # Get Cartesian positions of where these neighbors would be
                n1_x, n1_y = self.yx_to_cartesian(*neighbor1_pos)
                n2_x, n2_y = self.yx_to_cartesian(*neighbor2_pos)

                # Direction vectors from ring to neighbor positions
                dir1_x, dir1_y = n1_x - ring_x, n1_y - ring_y
                dir2_x, dir2_y = n2_x - ring_x, n2_y - ring_y

                # Normalize
                norm1 = np.sqrt(dir1_x**2 + dir1_y**2)
                norm2 = np.sqrt(dir2_x**2 + dir2_y**2)
                if norm1 > 0:
                    dir1_x, dir1_y = dir1_x / norm1, dir1_y / norm1
                if norm2 > 0:
                    dir2_x, dir2_y = dir2_x / norm2, dir2_y / norm2

                # Angle bisector (slide direction)
                slide_dx = dir1_x + dir2_x
                slide_dy = dir1_y + dir2_y
                slide_norm = np.sqrt(slide_dx**2 + slide_dy**2)

                if slide_norm > 0:
                    slide_dx /= slide_norm
                    slide_dy /= slide_norm

                    # Check if we can slide out in this direction without hitting other rings
                    if self._can_slide_ring_out(
                        ring_x, ring_y, slide_dx, slide_dy, index, ring_radius
                    ):
                        return True

        return False

    def _can_slide_ring_out(
        self, ring_x, ring_y, slide_dx, slide_dy, ring_index, ring_radius
    ):
        """Check if a ring can be slid out in a given direction.

        A ring is removable if it can slide at least one hex spacing (√3) in some
        direction without colliding with other rings. This represents sliding the
        ring one full hex-position away, which effectively removes it from play.

        Physics of ring collision:
        - In unit hex grid (size=1.0), adjacent centers are √3 apart
        - For touching rings: ring_radius = √3/2 ≈ 0.866
        - Ring diameter = 2 * ring_radius = √3
        - Minimum slide distance = √3 (one hex spacing)
        - Two rings collide if their centers are < √3 apart
        - For slide path: rings collide if perpendicular distance < diameter (2 * ring_radius)

        Args:
            ring_x, ring_y: Cartesian position of the ring to slide
            slide_dx, slide_dy: Normalized direction vector to slide
            ring_index: (y, x) index of the ring being slid (to exclude from checks)
            ring_radius: Radius of rings (should be √3/2 for unit grid)

        Returns:
            bool: True if ring can slide at least √3 distance without collision
        """
        # Minimum slide distance: one hex spacing
        sqrt3 = np.sqrt(3)
        min_slide_distance = sqrt3

        # For a ring to be slideable, no other ring should be within 2*radius
        # (diameter) of the slide path AND within the first √3 of travel

        for y in range(self.board.width):
            for x in range(self.board.width):
                if (y, x) == ring_index:
                    continue  # Don't check against ourselves

                if self.board._is_inbounds((y, x)) and self.board.state[self.board.RING_LAYER, y, x] == 1:
                    # This ring exists, check if it would block the slide
                    other_x, other_y = self.yx_to_cartesian(y, x)

                    # Vector from our ring to the other ring
                    to_other_x = other_x - ring_x
                    to_other_y = other_y - ring_y
                    dist_sq = to_other_x**2 + to_other_y**2

                    # Project onto slide direction
                    projection = to_other_x * slide_dx + to_other_y * slide_dy

                    # Only check rings in front of us AND within one hex spacing
                    # Rings behind us or beyond the minimum slide distance don't matter
                    if projection < 0 or projection > min_slide_distance:
                        continue

                    # Calculate perpendicular distance from the slide path
                    # perp_dist² = |v|² - proj²
                    perp_dist_sq = dist_sq - projection**2

                    # If perpendicular distance < 2*radius, rings would collide during slide
                    # Need at least 2*radius separation (ring diameters)
                    # Use small tolerance for floating point comparison
                    tolerance = 1e-6
                    min_clearance_sq = (2 * ring_radius) ** 2
                    if perp_dist_sq < min_clearance_sq - tolerance:
                        # Would collide during the first √3 of slide
                        return False

        return True

    # =========================  SIMPLE SYMMETRY OPERATIONS  =========================

    def get_rotational_symmetries(self, state=None):
        """Rotate the board 180 degrees.

        Args:
            state: State array to rotate (default: board.state)

        Returns:
            Rotated state array
        """
        # Always copy to ensure immutability of self.state or passed state
        state_to_rotate = np.copy(state if state is not None else self.board.state)
        return np.rot90(np.rot90(state_to_rotate, axes=(1, 2)), axes=(1, 2))

    def get_mirror_symmetries(self, state=None):
        """Flip the board while maintaining adjacency.

        Args:
            state: State array to mirror (default: board.state)

        Returns:
            Mirrored state array
        """
        # Always copy to ensure immutability of self.state or passed state
        mirror_state = np.copy(state if state is not None else self.board.state)
        layers = mirror_state.shape[0]
        for i in range(layers):
            mirror_state[i] = mirror_state[i].T
        return mirror_state

    def get_state_symmetries(self):
        """Return a list of symmetrical states by mirroring and rotating the board.

        Returns:
            List of (index, state) tuples for different symmetries
        """
        # noinspection PyListCreation
        symmetries = []
        symmetries.append((0, self.get_mirror_symmetries()))
        symmetries.append((1, self.get_rotational_symmetries()))
        symmetries.append((2, self.get_rotational_symmetries(symmetries[0][1])))
        return symmetries

    # =========================  COORDINATE TRANSFORMATIONS  =========================

    def flat_to_2d(self, flat_index):
        """Convert flat index to 2D board coordinates (y, x)."""
        return flat_index // self.board.width, flat_index % self.board.width

    def _2d_to_flat(self, y, x):
        """Convert 2D board coordinates to flat index."""
        return y * self.board.width + x

    def mirror_coords(self, y, x):
        """Mirror coordinates by swapping x and y axes."""
        return x, y

    def rotate_coords(self, y, x):
        """Rotate coordinates 180 degrees around board center."""
        # Universal formula that works for both odd and even widths
        return (self.board.width - 1) - y, (self.board.width - 1) - x

    # =========================  ACTION TRANSFORMATIONS  =========================

    def mirror_action(self, action_type, translated):
        """Apply mirror transformation to action array.

        Args:
            action_type: "CAP" or "PUT"
            translated: Action array to transform

        Returns:
            Mirrored action array
        """
        if action_type == "CAP":
            # swap capture direction axes
            temp = np.copy(translated)
            translated[3], translated[1] = temp[1], temp[3]
            translated[4], translated[0] = temp[0], temp[4]

            # transpose location axes
            d = translated.shape[0]
            for i in range(d):
                translated[i] = translated[i].T

        elif action_type == "PUT":
            temp = np.copy(translated)
            _, put, rem = translated.shape
            for p in range(put):
                # Translate the put index
                put_y, put_x = self.flat_to_2d(p)
                new_put_y, new_put_x = self.mirror_coords(put_y, put_x)
                new_p = self._2d_to_flat(new_put_y, new_put_x)
                for r in range(rem - 1):
                    # Translate the rem index
                    rem_y, rem_x = self.flat_to_2d(r)
                    new_rem_y, new_rem_x = self.mirror_coords(rem_y, rem_x)
                    new_r = self._2d_to_flat(new_rem_y, new_rem_x)
                    translated[:, new_p, new_r] = temp[:, p, r]

                # The last rem index is the same
                translated[:, new_p, rem - 1] = translated[:, new_p, rem - 1]

        return translated

    def rotate_action(self, action_type, translated):
        """Apply rotation transformation to action array.

        Args:
            action_type: "CAP" or "PUT"
            translated: Action array to transform

        Returns:
            Rotated action array
        """
        if action_type == "CAP":
            # swap capture direction axes
            temp = np.copy(translated)
            translated[3], translated[0] = temp[0], temp[3]
            translated[4], translated[1] = temp[1], temp[4]
            translated[5], translated[2] = temp[2], temp[5]

            # rotate location axes using universal formula
            temp = np.copy(translated)
            _, y, x = temp.shape
            for i in range(y):
                new_i = (self.board.width - 1) - i
                for j in range(x):
                    new_j = (self.board.width - 1) - j
                    translated[:, new_i, new_j] = temp[:, i, j]

        if action_type == "PUT":
            temp = np.copy(translated)
            _, put, rem = translated.shape
            for p in range(put):
                # Translate the put index
                put_y, put_x = self.flat_to_2d(p)
                new_put_y, new_put_x = self.rotate_coords(put_y, put_x)
                new_p = self._2d_to_flat(new_put_y, new_put_x)
                for r in range(rem - 1):
                    # Translate the rem index
                    rem_y, rem_x = self.flat_to_2d(r)
                    new_rem_y, new_rem_x = self.rotate_coords(rem_y, rem_x)
                    new_r = self._2d_to_flat(new_rem_y, new_rem_x)
                    translated[:, new_p, new_r] = temp[:, p, r]

                # The last rem index is the same
                translated[:, new_p, rem - 1] = translated[:, new_p, rem - 1]

        return translated

    # =========================  AXIAL COORDINATES  =========================

    def build_axial_maps(self):
        """Ensure axial coordinate mappings are built."""
        self.board._ensure_positions_built()

    @staticmethod
    def ax_rot60(q, r, k=1):
        """Rotate (q,r) by k * 60° counterclockwise in axial coords.

        Works for both regular and doubled coordinates (for even-width boards).
        """
        k %= 6
        for _ in range(k):
            q, r = -r, q + r  # 60° CCW
        return q, r

    @staticmethod
    def ax_mirror_q_axis(q, r):
        """Reflect (q,r) across the q-axis (cube: swap y and z)."""
        # In cube coords (x=q, z=r, y=-q-r), mirror over q-axis => (x, z, y)
        return q, -q - r

    def transform_state_hex(self, state, rot60_k=0, mirror=False, mirror_first=False):
        """Apply rotation and/or mirror to the whole SPATIAL state.

        Args:
            state: Board state to transform
            rot60_k: Number of 60° rotations (0-5 for D6, 0,2,4 for D3)
            mirror: Whether to apply mirror transformation
            mirror_first: If True, mirror then rotate. If False, rotate then mirror.
        """
        self.build_axial_maps()
        out = np.zeros_like(state)
        for (y, x), (q, r) in self.board._yx_to_ax.items():
            if mirror_first:
                # Mirror first, then rotate (for R{k}M transforms)
                if mirror:
                    q2, r2 = self.ax_mirror_q_axis(q, r)
                else:
                    q2, r2 = q, r
                q2, r2 = self.ax_rot60(q2, r2, rot60_k)
            else:
                # Rotate first, then mirror (for MR{k} transforms)
                q2, r2 = self.ax_rot60(q, r, rot60_k)
                if mirror:
                    q2, r2 = self.ax_mirror_q_axis(q2, r2)
            dst = self.board._ax_to_yx.get((q2, r2))
            if dst is not None:
                y2, x2 = dst
                out[:, y2, x2] = state[:, y, x]
        return out

    # ======================  D6 / D3 SYMMETRY ENUMERATION  =================

    def get_all_symmetry_transforms(self):
        """Return (name, fn) pairs for all valid symmetries of this board.

        Returns all 12 (D6) or 6 (D3) transformations:
        - R{k}: Pure rotations
        - MR{k}: Rotate then mirror
        - R{k}M: Mirror then rotate
        """
        if self.board.rings in (self.board.SMALL_BOARD_37, self.board.LARGE_BOARD_61):
            # full hex (D6): 6 rotations + 12 mirror combinations
            rot_steps = [0, 1, 2, 3, 4, 5]  # multiples of 60°
        else:
            # alternating 4/5 sides (48): D3 subset
            rot_steps = [0, 2, 4]  # 0°, 120°, 240°

        ops = []
        # Pure rotations
        for k in rot_steps:
            ops.append(
                (
                    f"R{60 * k}",
                    lambda s, kk=k: self.transform_state_hex(
                        s, rot60_k=kk, mirror=False
                    ),
                )
            )
        # Rotate then mirror (MR{k})
        for k in rot_steps:
            ops.append(
                (
                    f"MR{60 * k}",
                    lambda s, kk=k: self.transform_state_hex(
                        s, rot60_k=kk, mirror=True, mirror_first=False
                    ),
                )
            )
        # Mirror then rotate (R{k}M)
        for k in rot_steps:
            ops.append(
                (
                    f"R{60 * k}M",
                    lambda s, kk=k: self.transform_state_hex(
                        s, rot60_k=kk, mirror=True, mirror_first=True
                    ),
                )
            )
        return ops

    def canonical_key(self, state):
        """Lexicographic key over valid cells only (rings+marbles now)."""
        masked = (state[self.board.BOARD_LAYERS] * self.board.board_layout).astype(np.uint8)
        return masked.tobytes()

    # ======================  TRANSLATION CANONICALIZATION  ==================

    def get_bounding_box(self, state=None):
        """Find bounding box of all remaining rings.

        Returns:
            tuple: (min_y, max_y, min_x, max_x) or None if no rings exist
        """
        if state is None:
            state = self.board.state

        # Find all positions with rings
        ring_positions = np.where(state[self.board.RING_LAYER] == 1)

        if len(ring_positions[0]) == 0:
            return None  # No rings remaining

        min_y, max_y = ring_positions[0].min(), ring_positions[0].max()
        min_x, max_x = ring_positions[1].min(), ring_positions[1].max()

        return (min_y, max_y, min_x, max_x)

    def translate_state(self, state, dy, dx):
        """Translate state by (dy, dx) offset.

        Only translates ring and marble data (BOARD_LAYERS), preserving layout validity.
        Returns translated state if valid, None otherwise.
        """
        out = np.zeros_like(state)

        # Translate each position
        for y in range(self.board.width):
            for x in range(self.board.width):
                # Check if source position has a ring
                if state[self.board.RING_LAYER, y, x] == 0:
                    continue

                # Calculate destination
                new_y, new_x = y + dy, x + dx

                # Check if destination is valid on the board layout
                if not (0 <= new_y < self.board.width and 0 <= new_x < self.board.width):
                    return None  # Translation would move rings off-board

                if self.board.board_layout is not None and not self.board.board_layout[new_y, new_x]:
                    return None  # Destination not valid in board layout

                # Copy all board layers (ring + marbles)
                out[self.board.BOARD_LAYERS, new_y, new_x] = state[self.board.BOARD_LAYERS, y, x]

        # Copy history and capture layers unchanged
        # BOARD_LAYERS is slice(0, 4), so it covers 4 layers
        num_board_layers = self.board.BOARD_LAYERS.stop
        if state.shape[0] > num_board_layers:
            out[num_board_layers:] = state[num_board_layers:]

        return out

    def get_all_translations(self, state=None):
        """Generate all valid translation offsets for current board state.

        Returns:
            list: List of (name, dy, dx) tuples for valid translations
        """
        if state is None:
            state = self.board.state

        bbox = self.get_bounding_box(state)
        if bbox is None:
            return [("T0,0", 0, 0)]  # No rings, identity only

        min_y, max_y, min_x, max_x = bbox

        translations = []

        # Try all possible translations that might keep rings on board
        # Limit search to reasonable bounds
        for dy in range(-min_y, self.board.width - max_y):
            for dx in range(-min_x, self.board.width - max_x):
                # Test if this translation is valid
                translated = self.translate_state(state, dy, dx)
                if translated is not None:
                    translations.append((f"T{dy},{dx}", dy, dx))

        return translations

    # ======================  TRANSFORMATION ENUMERATION  ==================

    @staticmethod
    def _transform_simplicity_score(name):
        """Calculate simplicity score for a transform name (lower is simpler).

        Priority order:
        1. Identity (R0)
        2. Pure rotations (R60, R120, etc.)
        3. Mirror-based transforms (MR*, R*M)
        4. Translations (T*)
        5. Combined transforms (T*_*)

        Within categories, prefer:
        - Smaller rotation angles
        - Smaller translation distances
        - Shorter names
        """
        # Identity is simplest
        if name == "R0":
            return (0, 0, 0, 0)

        # Parse transform components
        has_translation = name.startswith("T") or "_" in name
        has_mirror = "M" in name
        has_rotation = "R" in name

        # Extract rotation angle if present
        rotation_angle = 0
        if has_rotation:
            if name.startswith("MR"):
                # MR120 format
                rotation_angle = int(name[2:].split("_")[0].rstrip("M"))
            elif name.startswith("R"):
                # R120 or R120M format
                parts = name[1:].split("_")[0].rstrip("M")
                rotation_angle = int(parts)

        # Extract translation distance if present
        translation_dist = 0
        if has_translation:
            if "_" in name:
                # Combined: T1,2_R60
                trans_part = name.split("_")[0]
            else:
                # Pure translation: T1,2
                trans_part = name

            if trans_part.startswith("T"):
                coords = trans_part[1:]  # Remove "T"
                dy, dx = map(int, coords.split(","))
                translation_dist = abs(dy) + abs(dx)  # Manhattan distance

        # Calculate category score (lower is simpler)
        if not has_translation and not has_mirror and not has_rotation:
            category = 0  # R0 (identity)
        elif not has_translation and not has_mirror:
            category = 1  # Pure rotation (R60, R120, etc.)
        elif not has_translation and has_mirror:
            category = 2  # Mirror-based (MR60, R60M, etc.)
        elif has_translation and not has_rotation and not has_mirror:
            category = 3  # Pure translation (T1,0, etc.)
        else:
            category = 4  # Combined (T1,0_R60, etc.)

        # Return tuple for lexicographic comparison
        # (category, translation_dist, rotation_angle, name_length)
        return (category, translation_dist, rotation_angle, len(name))

    def get_all_transformations(self, state=None, include_translation=True):
        """Get all valid transformation permutations of a state.

        Returns all transformations of the given state under the board's symmetry group.
        Eliminates duplicate states, keeping only the simplest transform name for each.

        For D6 boards (37/61 rings): Up to 18 rotation/mirror transforms (optionally × translations)
        For D3 boards (48 rings): Up to 9 rotation/mirror transforms (optionally × translations)

        Args:
            state: Board state to transform (if None, uses current board state)
            include_translation: If True, includes all valid translations (default: True)

        Returns:
            dict: Mapping of transform names to transformed states (duplicates removed)

        Example:
            >>> manager = board.canonicalizer
            >>> transforms = manager.get_all_transformations()
            >>> for name, transformed_state in transforms.items():
            ...     print(f"{name}: {transformed_state.shape}")
        """
        if state is None:
            state = self.board.state

        # First, collect all transformations (may have duplicates)
        all_transforms = {}

        # Get rotation/mirror operations
        rot_mirror_ops = self.get_all_symmetry_transforms()

        # Get translation operations
        if include_translation:
            translation_ops = self.get_all_translations(state)
        else:
            translation_ops = [("T0,0", 0, 0)]

        # Apply each combination: translate first, then rotate/mirror
        for trans_name, dy, dx in translation_ops:
            # Apply translation
            if dy == 0 and dx == 0:
                translated = state
            else:
                translated = self.translate_state(state, dy, dx)
                if translated is None:
                    continue  # Invalid translation

            # Then apply each rotation/mirror to the translated state
            for rot_mirror_name, rot_mirror_fn in rot_mirror_ops:
                transformed = rot_mirror_fn(translated)

                # Combine transform names
                if trans_name == "T0,0" and rot_mirror_name == "R0":
                    combined_name = "R0"  # Identity
                elif trans_name == "T0,0":
                    combined_name = rot_mirror_name
                elif rot_mirror_name == "R0":
                    combined_name = trans_name
                else:
                    combined_name = f"{trans_name}_{rot_mirror_name}"

                all_transforms[combined_name] = transformed

        # Now eliminate duplicates, keeping only the simplest transform for each unique state
        state_to_simplest = {}  # Maps state_key -> (transform_name, transformed_state)

        for name, transformed in all_transforms.items():
            state_key = transformed.tobytes()

            if state_key not in state_to_simplest:
                # First time seeing this state
                state_to_simplest[state_key] = (name, transformed)
            else:
                # Already seen this state, keep the simpler transform
                current_name, current_state = state_to_simplest[state_key]
                if self._transform_simplicity_score(name) < self._transform_simplicity_score(current_name):
                    state_to_simplest[state_key] = (name, transformed)

        # Return dictionary with only the simplest transforms
        return {name: transformed for name, transformed in state_to_simplest.values()}

    # ======================  MAIN CANONICALIZATION  ==================

    def canonicalize_state(self, transforms=TransformFlags.ALL):
        """
        Return (canonical_state, transform_name, inverse_name).

        Finds the lexicographically smallest representation among all enabled symmetry
        transformations. Transformations are applied in order: translation, then rotation/mirror.

        Args:
            transforms: TransformFlags specifying which transforms to use (default: ALL)

        Returns:
            tuple: (canonical_state, transform_name, inverse_name)
                - canonical_state: The transformed state with minimum lexicographic key
                - transform_name: Name of transform applied (e.g., "T2,1_MR120", "R60", "T1,-1")
                - inverse_name: Inverse transform to map back to original orientation
        """
        best_name = "R0"
        best_state = np.copy(self.board.state)
        best_key = self.canonical_key(best_state)

        # Build list of transform combinations based on flags
        transform_ops = []

        # Get rotation/mirror transforms if enabled
        if transforms & (TransformFlags.ROTATION | TransformFlags.MIRROR):
            rot_mirror_ops = []
            for name, fn in self.get_all_symmetry_transforms():
                # Filter based on flags
                if name.startswith("R") and not name.startswith("MR"):
                    # Pure rotation or mirror-then-rotate (R{k} or R{k}M)
                    if transforms & TransformFlags.ROTATION:
                        if "M" in name:
                            # R{k}M requires both rotation and mirror
                            if transforms & TransformFlags.MIRROR:
                                rot_mirror_ops.append((name, fn))
                        else:
                            # Pure rotation
                            rot_mirror_ops.append((name, fn))
                elif name.startswith("MR"):
                    # Rotate-then-mirror (MR{k})
                    if (transforms & TransformFlags.ROTATION) and (transforms & TransformFlags.MIRROR):
                        rot_mirror_ops.append((name, fn))
        else:
            # No rotation/mirror, just identity
            rot_mirror_ops = [("R0", lambda s: s)]

        # Get translation transforms if enabled
        if transforms & TransformFlags.TRANSLATION:
            translation_ops = self.get_all_translations(self.board.state)
        else:
            # No translation, just identity
            translation_ops = [("T0,0", 0, 0)]

        # Combine: translate FIRST, then rotate/mirror
        for trans_name, dy, dx in translation_ops:
            # Apply translation
            if dy == 0 and dx == 0:
                translated = self.board.state
            else:
                translated = self.translate_state(self.board.state, dy, dx)
                if translated is None:
                    continue  # Invalid translation

            # Then apply each rotation/mirror to the translated state
            for rot_mirror_name, rot_mirror_fn in rot_mirror_ops:
                transformed = rot_mirror_fn(translated)

                # Compute canonical key
                key = self.canonical_key(transformed)

                # Update best if this is lexicographically smaller
                if key < best_key:
                    # Combine transform names
                    if trans_name == "T0,0" and rot_mirror_name == "R0":
                        combined_name = "R0"  # Identity
                    elif trans_name == "T0,0":
                        combined_name = rot_mirror_name
                    elif rot_mirror_name == "R0":
                        combined_name = trans_name
                    else:
                        combined_name = f"{trans_name}_{rot_mirror_name}"

                    best_key = key
                    best_state = transformed
                    best_name = combined_name

        inv = self.get_inverse_transform(best_name)
        return best_state, best_name, inv

    # ======================  INVERSE TRANSFORMS  ===================

    def get_inverse_transform(self, transform_name):
        """
        Get the inverse of a symmetry transform.

        Supports combined transforms with translation:
        - "T{dy},{dx}_{rot_mirror}" → "{inv_rot_mirror}_T{-dy},{-dx}"
        - "T{dy},{dx}" → "T{-dy},{-dx}"
        - Pure rotation/mirror uses existing inversion rules

        Inverse relationships (rotation/mirror only):
        - R(k)⁻¹ = R(-k mod 360°)
        - MR(k)⁻¹ = R(-k mod 360°)M  (rotate-then-mirror inverts to mirror-then-rotate)
        - R(k)M⁻¹ = MR(-k mod 360°)  (mirror-then-rotate inverts to rotate-then-mirror)

        Inverse relationships (combined):
        - (T ∘ R)⁻¹ = R⁻¹ ∘ T⁻¹  (apply inverses in reverse order)

        Args:
            transform_name: String like "R60", "MR120", "R240M", "T2,1", "T1,-1_MR120"

        Returns:
            String naming the inverse transform
        """
        # Check for combined transform (translation + rotation/mirror)
        if "_" in transform_name:
            # Combined transform: "T{dy},{dx}_{rot_mirror}"
            parts = transform_name.split("_")
            if len(parts) != 2:
                raise ValueError(f"Invalid combined transform format: {transform_name}")

            trans_part, rot_mirror_part = parts

            # Invert translation: T{dy},{dx} → T{-dy},{-dx}
            if not trans_part.startswith("T"):
                raise ValueError(f"Expected translation in combined transform: {transform_name}")

            # Extract dy, dx from "T{dy},{dx}"
            coords = trans_part[1:]  # Remove "T"
            dy, dx = map(int, coords.split(","))
            inv_trans = f"T{-dy},{-dx}"

            # Invert rotation/mirror part
            inv_rot_mirror = self.get_inverse_transform(rot_mirror_part)

            # Combine in reverse order: rot_mirror_inv _ trans_inv
            return f"{inv_rot_mirror}_{inv_trans}"

        # Translation-only transform
        elif transform_name.startswith("T"):
            # Extract dy, dx from "T{dy},{dx}"
            coords = transform_name[1:]  # Remove "T"
            dy, dx = map(int, coords.split(","))
            return f"T{-dy},{-dx}"

        # Rotation/mirror-only transforms (existing logic)
        elif transform_name.endswith("M") and not transform_name.startswith("MR"):
            # R{k}M (mirror-then-rotate): inverse is MR{-k} (rotate-then-mirror)
            # Extract angle from name (e.g., "R120M" -> 120)
            angle = int(transform_name[1:-1])  # Strip "R" and "M"
            # Inverse angle in full circle
            inv_angle = (360 - angle) % 360
            return f"MR{inv_angle}"

        elif transform_name.startswith("MR"):
            # MR{k} (rotate-then-mirror): inverse is R{-k}M (mirror-then-rotate)
            # Extract angle from name (e.g., "MR120" -> 120)
            angle = int(transform_name[2:])
            # Inverse angle in full circle
            inv_angle = (360 - angle) % 360
            return f"R{inv_angle}M"

        elif transform_name.startswith("R"):
            # Pure rotation: inverse is opposite rotation
            angle = int(transform_name[1:])
            # Inverse angle in full circle
            inv_angle = (360 - angle) % 360
            return f"R{inv_angle}"

        else:
            raise ValueError(f"Unknown transform name format: {transform_name}")

    # ======================  MASK TRANSFORMATIONS  ===================

    def dir_index_map(self, rot60_k=0, mirror=False):
        """
        Map capture direction indices (0..5) under the same symmetry used for state.
        We transform a direction vector v=(dy,dx) via axial:
           (dq,dr) = (dx, dy-dx)
           rotate/mirror in axial
           back to (dy',dx') with: dx' = dq, dy' = dr + dq
        """
        # Original direction vectors in your order:
        dirs = self.board.DIRECTIONS

        def xform_vec(dy, dx):
            dq, dr = dx, (dy - dx)
            dq, dr = self.ax_rot60(dq, dr, rot60_k)
            if mirror:
                dq, dr = self.ax_mirror_q_axis(dq, dr)
            dx2 = dq
            dy2 = dr + dq
            return (dy2, dx2)

        idx_map = {}
        for i, v in enumerate(dirs):
            vv = xform_vec(*v)
            j = next(k for k, u in enumerate(dirs) if u == vv)
            idx_map[i] = j
        return idx_map

    def transform_capture_mask(self, cap_mask, rot60_k=0, mirror=False):
        """
        cap_mask shape: (6, H, W). Returns transformed mask matching state xform.
        """
        self.build_axial_maps()
        out = np.zeros_like(cap_mask)
        dmap = self.dir_index_map(rot60_k, mirror)

        for (y, x), (q, r) in self.board._yx_to_ax.items():
            q2, r2 = self.ax_rot60(q, r, rot60_k)
            if mirror:
                q2, r2 = self.ax_mirror_q_axis(q2, r2)
            dst = self.board._ax_to_yx.get((q2, r2))
            if dst is None:
                continue
            y2, x2 = dst
            for d in range(6):
                out[dmap[d], y2, x2] = cap_mask[d, y, x]
        return out

    def transform_put_mask(self, put_mask, rot60_k=0, mirror=False):
        """
        put_mask shape: (3, W*W, W*W+1). Applies same symmetry to (put, rem) indices.
        """
        self.build_axial_maps()
        out = np.zeros_like(put_mask)

        # Build flat-index permutation for valid cells
        flat_map = {}  # src_flat -> dst_flat
        for (y, x), (q, r) in self.board._yx_to_ax.items():
            q2, r2 = self.ax_rot60(q, r, rot60_k)
            if mirror:
                q2, r2 = self.ax_mirror_q_axis(q2, r2)
            dst = self.board._ax_to_yx.get((q2, r2))
            if dst is None:
                continue
            y2, x2 = dst
            flat_map[y * self.board.width + x] = y2 * self.board.width + x2

        M, P, R = put_mask.shape
        last = R - 1  # "no ring removed" slot stays put

        for m in range(M):
            for p in range(P):
                p2 = flat_map.get(p)
                if p2 is None:
                    continue
                # all concrete removals
                for r in range(R - 1):
                    r2 = flat_map.get(r)
                    if r2 is not None:
                        out[m, p2, r2] = put_mask[m, p, r]
                # "no remove" column
                out[m, p2, last] = put_mask[m, p, last]

        return out