"""Board state canonicalization and transformation utilities.

This module provides the CanonicalizationManager class which handles all symmetry
transformations, state canonicalization, and action transformations for Zèrtz boards.
Supports D6 (37/61 rings) and D3 (48 rings) dihedral group symmetries.

## Architecture: Python-Rust Delegation

This module delegates performance-critical operations to Rust while maintaining
Python utilities for testing, ML, and convenience.

### Functions Delegated to Rust (MCTS Hot Path):
- ✅ **canonicalize_state()**: Core canonicalization (30-50% MCTS speedup)
- ✅ **transform_state_hex()**: Board rotation/mirror transformations
- ✅ **translate_state()**: Board translation operations
- ✅ **canonical_key()**: Lexicographic key computation
- ✅ **inverse_transform_name()**: Inverse transform computation

### Python-Only Functions (Not Performance Critical):

**Test/Helper Utilities**:
- `get_all_transformations()`: Enumerate all transforms (test utility)
- `get_all_translations()`: Enumerate all translations (delegates to Rust)
- `get_all_symmetry_transforms()`: Enumerate symmetries (test utility)
- `_apply_transform()`: Apply named transform (test/helper)
- `decanonicalize()`: Inverse canonicalization (test/helper)

**ML-Specific Functions (Mask Transformations)**:
- `canonicalize_capture_mask()`: Transform capture action masks for ML
- `canonicalize_put_mask()`: Transform placement action masks for ML
- `_transform_capture_mask()`: Low-level capture mask transformation
- `_transform_put_mask()`: Low-level placement mask transformation
- `_dir_index_map()`: Direction vector transformation for captures

**Convenience/Utility Functions**:
- `flat_to_2d()`, `_2d_to_flat()`: Coordinate conversions
- `get_bounding_box()`: Find ring bounding box
- `_parse_rot_mirror()`: Parse transform strings
- `_transform_simplicity_score()`: Rank transforms by simplicity

### Rust Duplicates (Used by Rust Internally):

Some Python functions have Rust equivalents that are used internally by Rust's
canonicalization system:
- `build_axial_maps()`: Both Python and Rust versions exist
- `get_bounding_box()`: Rust version (`bounding_box()`) used internally

These duplicates are acceptable because:
1. Python versions are used by test utilities and non-critical paths
2. Rust versions are optimized for the MCTS hot path
3. Maintaining both allows gradual migration and testing
"""

import numpy as np

# Import canonicalization functions and TransformFlags from Rust
from hiivelabs_mcts import (
    ax_rot60,
    ax_mirror_q_axis,
    canonicalize_state as rust_canonicalize_state,
    transform_state as rust_transform_state,
    translate_state as rust_translate_state,
    get_bounding_box as rust_get_bounding_box,
    get_translations as rust_get_translations,
    canonical_key as rust_canonical_key,
    inverse_transform_name as rust_inverse_transform_name,
    TransformFlags,
    BoardConfig,
)


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

    # =========================  COORDINATE TRANSFORMATIONS  =========================

    def flat_to_2d(self, flat_index):
        """Convert flat index to 2D board coordinates (y, x)."""
        width = self.board.config.width
        return divmod(flat_index, width)

    def _2d_to_flat(self, y, x):
        """Convert 2D board coordinates to flat index."""
        width = self.board.config.width
        return y * width + x

    # =========================  AXIAL COORDINATES  =========================

    def build_axial_maps(self):
        """Ensure axial coordinate mappings are built."""
        self.board._ensure_positions_built()

    def transform_state_hex(self, state, rot60_k=0, mirror=False, mirror_first=False):
        """Apply rotation and/or mirror to the whole SPATIAL state.

        Delegates to Rust for performance.

        Args:
            state: Board state to transform
            rot60_k: Number of 60° rotations (0-5 for D6, 0,2,4 for D3)
            mirror: Whether to apply mirror transformation
            mirror_first: If True, mirror then rotate. If False, rotate then mirror.
        """
        config = BoardConfig.standard_config(self.board.rings, t=self.board.t)
        return rust_transform_state(state, config, rot60_k, mirror, mirror_first)

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
        """Lexicographic key over valid cells only (rings+marbles now).

        **RUST DUPLICATE**: This function has a Rust implementation that is used by
        Rust's canonicalization for finding lexicographically minimal states.

        Delegates to Rust for performance.
        """
        config = BoardConfig.standard_config(self.board.rings, t=self.board.t)
        return bytes(rust_canonical_key(state, config))

    # ======================  TRANSLATION CANONICALIZATION  ==================

    def translate_state(self, state, dy, dx):
        """Translate state by (dy, dx) offset.

        Only translates ring and marble data (BOARD_LAYERS), preserving layout validity.
        Returns translated state if valid, None otherwise.

        **RUST DUPLICATE**: This function has a Rust implementation that is used by
        Rust's canonicalization. This Python version is kept for:
        - Helper methods (get_all_translations, _apply_transform)
        - Test utilities
        - Direct use by non-canonicalization code

        Delegates to Rust for performance.
        """
        config = BoardConfig.standard_config(self.board.rings, t=self.board.t)
        return rust_translate_state(state, config, dy, dx)

    def get_all_translations(self, state=None):
        """Generate all valid translation offsets for current board state.

        Delegates to Rust for validation of each translation.

        Returns:
            list: List of (name, dy, dx) tuples for valid translations
        """
        if state is None:
            state = self.board.state

        config = BoardConfig.standard_config(self.board.rings, t=self.board.t)
        return rust_get_translations(state, config)

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

    def get_all_transformations(self, state=None, include_translation=True, deduplicate=True):
        """Get all valid transformation permutations of a state.

        Returns all transformations of the given state under the board's symmetry group.
        By default, eliminates duplicate states, keeping only the simplest transform name for each.

        For D6 boards (37/61 rings): 18 rotation/mirror transforms (optionally × translations)
        For D3 boards (48 rings): 9 rotation/mirror transforms (optionally × translations)

        Args:
            state: Board state to transform (if None, uses current board state)
            include_translation: If True, includes all valid translations (default: True)
            deduplicate: If True, eliminates duplicate states keeping simplest transform (default: True)

        Returns:
            dict: Mapping of transform names to transformed states

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

        # If deduplication is disabled, return all transforms
        if not deduplicate:
            return all_transforms

        # Eliminate duplicates, keeping only the simplest transform for each unique state
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

    def canonicalize_state(self, state=None, transforms=TransformFlags.ALL):
        """
        Return (canonical_state, transform_name, inverse_name).

        Finds the lexicographically smallest representation among all enabled symmetry
        transformations. Transformations are applied in order: translation, then rotation/mirror.

        Delegates entirely to Rust for all transform flag combinations for performance.

        Args:
            state: Board state to canonicalize (default: self.board.state)
            transforms: TransformFlags specifying which transforms to use (default: ALL)

        Returns:
            tuple: (canonical_state, transform_name, inverse_name)
                - canonical_state: The transformed state with minimum lexicographic key
                - transform_name: Name of transform applied (e.g., "T2,1_MR120", "R60", "T1,-1")
                - inverse_name: Inverse transform to map back to original orientation
        """
        if state is None:
            state = self.board.state

        # Delegate all canonicalization to Rust for performance
        config = BoardConfig.standard_config(self.board.rings, t=self.board.t)
        canonical_state, transform_name, inverse_name = rust_canonicalize_state(state, config, transforms)
        return canonical_state, transform_name, inverse_name

    # ======================  INVERSE TRANSFORMS  ===================

    @staticmethod
    def _get_inverse_transform(transform_name):
        """
        Get the inverse of a symmetry transform.

        Delegates to Rust for all inverse transform computations.

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
        # Delegate to Rust for inverse computation
        return rust_inverse_transform_name(transform_name)

    def _apply_transform(self, state, transform_name):
        """Apply a named transform to a state.

        Parses the transform name and applies the transformation to the given state.
        Supports all transform types: rotations, mirrors, translations, and combinations.

        Args:
            state: Board state array to transform
            transform_name: Transform name (e.g., "R60", "MR120", "T1,2", "T1,-1_R60")

        Returns:
            Transformed state array

        Examples:
            >>> # Apply 60° rotation
            >>> rotated = manager.apply_transform(state, "R60")
            >>> # Apply translation then mirror
            >>> transformed = manager.apply_transform(state, "T2,1_MR0")
        """
        # Identity transform
        if transform_name == "R0":
            return np.copy(state)

        # Parse combined transform (translation + rotation/mirror)
        if "_" in transform_name:
            # Combined transform can be in two forms:
            # 1. "T{dy},{dx}_{rot_mirror}" (translation first)
            # 2. "{rot_mirror}_T{dy},{dx}" (rotation/mirror first)
            parts = transform_name.split("_")
            if len(parts) != 2:
                raise ValueError(f"Invalid combined transform format: {transform_name}")

            first_part, second_part = parts

            # Determine which form we have
            if first_part.startswith("T"):
                # Form 1: "T{dy},{dx}_{rot_mirror}" (translation first)
                trans_part = first_part
                rot_mirror_part = second_part

                # Extract and apply translation
                coords = trans_part[1:]  # Remove "T"
                dy, dx = map(int, coords.split(","))
                translated = self.translate_state(state, dy, dx)
                if translated is None:
                    raise ValueError(f"Invalid translation {trans_part} for given state")

                # Then apply rotation/mirror to translated state
                return self._apply_transform(translated, rot_mirror_part)

            elif second_part.startswith("T"):
                # Form 2: "{rot_mirror}_T{dy},{dx}" (rotation/mirror first)
                rot_mirror_part = first_part
                trans_part = second_part

                # Apply rotation/mirror first
                rotated = self._apply_transform(state, rot_mirror_part)

                # Then apply translation
                coords = trans_part[1:]  # Remove "T"
                dy, dx = map(int, coords.split(","))
                translated = self.translate_state(rotated, dy, dx)
                if translated is None:
                    raise ValueError(f"Invalid translation {trans_part} for given state")

                return translated

            else:
                raise ValueError(f"Expected translation in combined transform: {transform_name}")

        # Pure translation
        elif transform_name.startswith("T"):
            coords = transform_name[1:]  # Remove "T"
            dy, dx = map(int, coords.split(","))
            translated = self.translate_state(state, dy, dx)
            if translated is None:
                raise ValueError(f"Invalid translation {transform_name} for given state")
            return translated

        # Rotation/mirror transforms
        elif transform_name.endswith("M") and not transform_name.startswith("MR"):
            # R{k}M (mirror-then-rotate)
            angle = int(transform_name[1:-1])  # Strip "R" and "M"
            rot60_k = angle // 60
            return self.transform_state_hex(state, rot60_k=rot60_k, mirror=True, mirror_first=True)

        elif transform_name.startswith("MR"):
            # MR{k} (rotate-then-mirror)
            angle = int(transform_name[2:])
            rot60_k = angle // 60
            return self.transform_state_hex(state, rot60_k=rot60_k, mirror=True, mirror_first=False)

        elif transform_name.startswith("R"):
            # Pure rotation
            angle = int(transform_name[1:])
            rot60_k = angle // 60
            return self.transform_state_hex(state, rot60_k=rot60_k, mirror=False)

        else:
            raise ValueError(f"Unknown transform name format: {transform_name}")

    def decanonicalize(self, canonical_state, inverse_transform_name):
        """Decanonicalize a state by applying its inverse transform.

        Takes a canonical state and the inverse transform name (typically obtained from
        canonicalize_state) and returns the original state before canonicalization.

        This is the complement to canonicalize_state():
        - canonicalize_state() transforms original → canonical using forward transform
        - decanonicalize() transforms canonical → original using inverse transform

        Args:
            canonical_state: The canonical state array
            inverse_transform_name: The inverse transform name (e.g., from canonicalize_state)

        Returns:
            Original state before canonicalization

        Examples:
            >>> # Canonicalize and then decanonicalize
            >>> canonical, forward, inverse = manager.canonicalize_state()
            >>> original = manager.decanonicalize(canonical, inverse)
            >>> # original should match the pre-canonicalized state

            >>> # Apply inverse to canonical state from another board
            >>> original_state = manager.decanonicalize(canonical_state, "MR60_T-2,-1")
        """
        return self._apply_transform(canonical_state, inverse_transform_name)

    @staticmethod
    def _parse_rot_mirror(rot_mirror_name):
        """Parse rotation/mirror transform name into parameters.

        Args:
            rot_mirror_name: Transform like "R60", "MR120", "R60M"

        Returns:
            tuple: (rot60_k, mirror, mirror_first) where rot60_k is number of 60° rotations
        """
        if rot_mirror_name == "R0":
            return 0, False, False
        elif rot_mirror_name.endswith("M") and not rot_mirror_name.startswith("MR"):
            # R{k}M (mirror-then-rotate)
            angle = int(rot_mirror_name[1:-1])
            return angle // 60, True, True
        elif rot_mirror_name.startswith("MR"):
            # MR{k} (rotate-then-mirror)
            angle = int(rot_mirror_name[2:])
            return angle // 60, True, False
        elif rot_mirror_name.startswith("R"):
            # Pure rotation
            angle = int(rot_mirror_name[1:])
            return angle // 60, False, False
        else:
            raise ValueError(f"Unknown rotation/mirror format: {rot_mirror_name}")

    def canonicalize_capture_mask(self, cap_mask, transform_name):
        """Canonicalize a capture mask using a given transform.

        Applies the same transformation to a capture mask that was applied to a state.
        Typically used with the transform returned by canonicalize_state().

        Args:
            cap_mask: Capture mask array (6, H, W)
            transform_name: Transform name from canonicalize_state() (e.g., "R60", "T1,2_MR120")

        Returns:
            tuple: (canonical_mask, transform_name, inverse_name)
                - canonical_mask: The transformed mask
                - transform_name: Same as input (for API consistency)
                - inverse_name: Inverse transform to map back to original

        Examples:
            >>> # Canonicalize state and corresponding mask together
            >>> canonical_state, transform, inverse = manager.canonicalize_state()
            >>> canonical_mask, _, _ = manager.canonicalize_capture_mask(cap_mask, transform)
        """
        # Handle different transform formats by detecting order and applying appropriately
        if transform_name == "R0":
            # Identity
            canonical_mask = np.copy(cap_mask)
        elif "_" in transform_name:
            # Combined transform - detect ordering
            parts = transform_name.split("_")
            if parts[0].startswith("T"):
                # Form: "T{dy},{dx}_{rot_mirror}" - translation first
                trans_part, rot_mirror_part = parts
                coords = trans_part[1:]
                dy, dx = map(int, coords.split(","))

                # Extract rotation/mirror parameters
                rot60_k, mirror, mirror_first = self._parse_rot_mirror(rot_mirror_part)

                # Apply in order: translation first, then rotation/mirror
                canonical_mask = self._transform_capture_mask(cap_mask, rot60_k, mirror, dy, dx, mirror_first)
            elif parts[1].startswith("T"):
                # Form: "{rot_mirror}_T{dy},{dx}" - rotation/mirror first
                rot_mirror_part, trans_part = parts
                coords = trans_part[1:]
                dy, dx = map(int, coords.split(","))

                # Extract rotation/mirror parameters
                rot60_k, mirror, _ = self._parse_rot_mirror(rot_mirror_part)

                # Apply in order: rotation/mirror first, then translation
                # First apply rotation/mirror (no translation)
                intermediate = self._transform_capture_mask(cap_mask, rot60_k, mirror, 0, 0)
                # Then apply translation
                canonical_mask = self._transform_capture_mask(intermediate, 0, False, dy, dx)
            else:
                raise ValueError(f"Expected translation in combined transform: {transform_name}")
        elif transform_name.startswith("T"):
            # Pure translation
            coords = transform_name[1:]
            dy, dx = map(int, coords.split(","))
            canonical_mask = self._transform_capture_mask(cap_mask, 0, False, dy, dx)
        else:
            # Pure rotation/mirror
            rot60_k, mirror, mirror_first = self._parse_rot_mirror(transform_name)
            canonical_mask = self._transform_capture_mask(cap_mask, rot60_k, mirror, 0, 0, mirror_first)

        inverse_name = self._get_inverse_transform(transform_name)
        return canonical_mask, transform_name, inverse_name

    def decanonicalize_capture_mask(self, canonical_mask, inverse_transform_name):
        """Decanonicalize a capture mask by applying its inverse transform.

        Takes a canonical capture mask and applies the inverse transform to recover
        the original mask orientation.

        Args:
            canonical_mask: The canonical capture mask array (6, H, W)
            inverse_transform_name: The inverse transform name (e.g., from canonicalize_capture_mask)

        Returns:
            Original mask before canonicalization

        Examples:
            >>> # Canonicalize and then decanonicalize
            >>> canonical_mask, transform, inverse = manager.canonicalize_capture_mask(mask, "R60")
            >>> original_mask = manager.decanonicalize_capture_mask(canonical_mask, inverse)
        """
        # Parse and apply inverse transform (reuse canonicalize logic)
        result_mask, _, _ = self.canonicalize_capture_mask(canonical_mask, inverse_transform_name)
        return result_mask

    def canonicalize_put_mask(self, put_mask, transform_name):
        """Canonicalize a put mask using a given transform.

        Applies the same transformation to a put mask that was applied to a state.
        Typically used with the transform returned by canonicalize_state().

        Args:
            put_mask: Put mask array (3, W*W, W*W+1)
            transform_name: Transform name from canonicalize_state() (e.g., "R60", "T1,2_MR120")

        Returns:
            tuple: (canonical_mask, transform_name, inverse_name)
                - canonical_mask: The transformed mask
                - transform_name: Same as input (for API consistency)
                - inverse_name: Inverse transform to map back to original

        Examples:
            >>> # Canonicalize state and corresponding mask together
            >>> canonical_state, transform, inverse = manager.canonicalize_state()
            >>> canonical_mask, _, _ = manager.canonicalize_put_mask(put_mask, transform)
        """
        # Handle different transform formats by detecting order and applying appropriately
        if transform_name == "R0":
            # Identity
            canonical_mask = np.copy(put_mask)
        elif "_" in transform_name:
            # Combined transform - detect ordering
            parts = transform_name.split("_")
            if parts[0].startswith("T"):
                # Form: "T{dy},{dx}_{rot_mirror}" - translation first
                trans_part, rot_mirror_part = parts
                coords = trans_part[1:]
                dy, dx = map(int, coords.split(","))

                # Extract rotation/mirror parameters
                rot60_k, mirror, mirror_first = self._parse_rot_mirror(rot_mirror_part)

                # Apply in order: translation first, then rotation/mirror
                canonical_mask = self._transform_put_mask(put_mask, rot60_k, mirror, dy, dx, mirror_first)
            elif parts[1].startswith("T"):
                # Form: "{rot_mirror}_T{dy},{dx}" - rotation/mirror first
                rot_mirror_part, trans_part = parts
                coords = trans_part[1:]
                dy, dx = map(int, coords.split(","))

                # Extract rotation/mirror parameters
                rot60_k, mirror, _ = self._parse_rot_mirror(rot_mirror_part)

                # Apply in order: rotation/mirror first, then translation
                # First apply rotation/mirror (no translation)
                intermediate = self._transform_put_mask(put_mask, rot60_k, mirror, 0, 0)
                # Then apply translation
                canonical_mask = self._transform_put_mask(intermediate, 0, False, dy, dx)
            else:
                raise ValueError(f"Expected translation in combined transform: {transform_name}")
        elif transform_name.startswith("T"):
            # Pure translation
            coords = transform_name[1:]
            dy, dx = map(int, coords.split(","))
            canonical_mask = self._transform_put_mask(put_mask, 0, False, dy, dx)
        else:
            # Pure rotation/mirror
            rot60_k, mirror, mirror_first = self._parse_rot_mirror(transform_name)
            canonical_mask = self._transform_put_mask(put_mask, rot60_k, mirror, 0, 0, mirror_first)

        inverse_name = self._get_inverse_transform(transform_name)
        return canonical_mask, transform_name, inverse_name

    def decanonicalize_put_mask(self, canonical_mask, inverse_transform_name):
        """Decanonicalize a put mask by applying its inverse transform.

        Takes a canonical put mask and applies the inverse transform to recover
        the original mask orientation.

        Args:
            canonical_mask: The canonical put mask array (3, W*W, W*W+1)
            inverse_transform_name: The inverse transform name (e.g., from canonicalize_put_mask)

        Returns:
            Original mask before canonicalization

        Examples:
            >>> # Canonicalize and then decanonicalize
            >>> canonical_mask, transform, inverse = manager.canonicalize_put_mask(mask, "R60")
            >>> original_mask = manager.decanonicalize_put_mask(canonical_mask, inverse)
        """
        # Parse and apply inverse transform (reuse canonicalize logic)
        result_mask, _, _ = self.canonicalize_put_mask(canonical_mask, inverse_transform_name)
        return result_mask

    # ======================  MASK TRANSFORMATIONS  ===================

    def _dir_index_map(self, rot60_k=0, mirror=False, mirror_first=False):
        """
        Map capture direction indices (0..5) under the same symmetry used for state.
        We transform a direction vector v=(dy,dx) via axial:
           (dq,dr) = (dx, dy-dx)
           rotate/mirror in axial
           back to (dy',dx') with: dx' = dq, dy' = dr + dq

        Args:
            rot60_k: Number of 60° rotations
            mirror: Whether to apply mirror transformation
            mirror_first: If True, mirror then rotate. If False, rotate then mirror.
        """
        # Original direction vectors in your order:
        dirs = self.board.DIRECTIONS

        def xform_vec(dy, dx):
            dq, dr = dx, (dy - dx)
            if mirror_first:
                # Mirror first, then rotate (for R{k}M transforms)
                if mirror:
                    dq, dr = ax_mirror_q_axis(dq, dr)
                dq, dr = ax_rot60(dq, dr, rot60_k)
            else:
                # Rotate first, then mirror (for MR{k} transforms)
                dq, dr = ax_rot60(dq, dr, rot60_k)
                if mirror:
                    dq, dr = ax_mirror_q_axis(dq, dr)
            dx2 = dq
            dy2 = dr + dq
            return (dy2, dx2)

        idx_map = {}
        for i, v in enumerate(dirs):
            vv = xform_vec(*v)
            j = next(k for k, u in enumerate(dirs) if u == vv)
            idx_map[i] = j
        return idx_map

    def _transform_capture_mask(self, cap_mask, rot60_k=0, mirror=False, dy=0, dx=0, mirror_first=False):
        """
        cap_mask shape: (6, H, W). Returns transformed mask matching state xform.

        Applies translation first, then rotation/mirror, matching the state transformation order.

        Args:
            cap_mask: Capture mask array (6, H, W)
            rot60_k: Number of 60° rotations
            mirror: Whether to apply mirror transformation
            dy: Translation offset in y direction
            dx: Translation offset in x direction
            mirror_first: If True, mirror then rotate. If False, rotate then mirror.
        """
        self.build_axial_maps()
        out = np.zeros_like(cap_mask)
        dmap = self._dir_index_map(rot60_k, mirror, mirror_first)

        for (y, x), (q, r) in self.board.positions.yx_to_ax.items():
            # Skip positions where ring has been removed
            if self.board.state[self.board.RING_LAYER, y, x] == 0:
                continue

            # Apply translation to get translated position
            y_trans, x_trans = y + dy, x + dx

            # Get axial coordinates of translated position
            if (y_trans, x_trans) not in self.board.positions.yx_to_ax:
                continue  # Translated position is not on the board

            # Skip if the translated position has a removed ring
            if self.board.state[self.board.RING_LAYER, y_trans, x_trans] == 0:
                continue

            q_trans, r_trans = self.board.positions.yx_to_ax[(y_trans, x_trans)]

            # Apply rotation/mirror to translated axial coordinates
            if mirror_first:
                # Mirror first, then rotate (for R{k}M transforms)
                if mirror:
                    q2, r2 = ax_mirror_q_axis(q_trans, r_trans)
                else:
                    q2, r2 = q_trans, r_trans
                q2, r2 = ax_rot60(q2, r2, rot60_k)
            else:
                # Rotate first, then mirror (for MR{k} transforms)
                q2, r2 = ax_rot60(q_trans, r_trans, rot60_k)
                if mirror:
                    q2, r2 = ax_mirror_q_axis(q2, r2)

            # Convert back to (y, x) coordinates
            dst = self.board.positions.ax_to_yx.get((q2, r2))
            if dst is None:
                continue
            y2, x2 = dst

            # Map mask values: original position (y,x) → final position (y2,x2)
            for d in range(6):
                out[dmap[d], y2, x2] = cap_mask[d, y, x]
        return out

    def _transform_put_mask(self, put_mask, rot60_k=0, mirror=False, dy=0, dx=0, mirror_first=False):
        """
        put_mask shape: (3, W*W, W*W+1). Applies same symmetry to (put, rem) indices.

        Applies translation first, then rotation/mirror, matching the state transformation order.

        Args:
            put_mask: Put mask array (3, W*W, W*W+1)
            rot60_k: Number of 60° rotations
            mirror: Whether to apply mirror transformation
            dy: Translation offset in y direction
            dx: Translation offset in x direction
            mirror_first: If True, mirror then rotate. If False, rotate then mirror.
        """
        self.build_axial_maps()
        out = np.zeros_like(put_mask)

        # Build flat-index permutation for valid cells
        flat_map = {}  # src_flat -> dst_flat
        for (y, x) in self.board.positions.yx_to_ax.keys():
            # Skip positions where ring has been removed
            if self.board.state[self.board.RING_LAYER, y, x] == 0:
                continue

            # Apply translation to get translated position
            y_trans, x_trans = y + dy, x + dx

            # Get axial coordinates of translated position
            if (y_trans, x_trans) not in self.board.positions.yx_to_ax:
                continue  # Translated position is not on the board

            # Skip if the translated position has a removed ring
            if self.board.state[self.board.RING_LAYER, y_trans, x_trans] == 0:
                continue

            q_trans, r_trans = self.board.positions.yx_to_ax[(y_trans, x_trans)]

            # Apply rotation/mirror to translated axial coordinates
            if mirror_first:
                # Mirror first, then rotate (for R{k}M transforms)
                if mirror:
                    q2, r2 = ax_mirror_q_axis(q_trans, r_trans)
                else:
                    q2, r2 = q_trans, r_trans
                q2, r2 = ax_rot60(q2, r2, rot60_k)
            else:
                # Rotate first, then mirror (for MR{k} transforms)
                q2, r2 = ax_rot60(q_trans, r_trans, rot60_k)
                if mirror:
                    q2, r2 = ax_mirror_q_axis(q2, r2)

            # Convert back to (y, x) coordinates
            dst = self.board.positions.ax_to_yx.get((q2, r2))
            if dst is None:
                continue
            y2, x2 = dst

            # Map flat indices: original (y,x) → final (y2,x2)
            width = self.board.config.width
            flat_map[y * width + x] = y2 * width + x2

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