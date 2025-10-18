# Board State Canonicalization

## Overview

Canonicalization is the process of converting symmetry-equivalent board states into a single, consistent "canonical" representation. In Zèrtz, the hexagonal board has rotational and reflective symmetry, meaning the same logical game state can appear in multiple physical orientations. Additionally, when edge rings are removed during gameplay, the board can be translated (shifted) while maintaining validity.

## Why Canonicalization Matters

For machine learning and game analysis:
- **State space reduction**: Multiple equivalent states map to one canonical form, reducing the effective state space
- **Efficient learning**: ML models can learn from one orientation and apply to all equivalent positions
- **Consistent comparison**: Two equivalent states always produce the same canonical representation

## Algorithm

The canonicalization algorithm is implemented in `game/zertz_board.py:canonicalize_state()`.

### Process

1. **Generate all symmetries**: Apply every possible symmetry transformation to the current state
   - D6 symmetry (37/61-ring boards): 12 transformations (6 rotations × 2 for mirror)
   - D3 symmetry (48-ring boards): 6 transformations (3 rotations × 2 for mirror)
   - Translation (when edge rings removed): N additional translations per rotation/mirror
   - Combined: Translation applied first, then rotation/mirror

2. **Create comparable keys**: For each transformed state, generate a lexicographic key:
   ```python
   def _canonical_key(self, state):
       """Lexicographic key over valid cells only (rings+marbles now)."""
       masked = (state[self.BOARD_LAYERS] * self.board_layout).astype(np.uint8)
       return masked.tobytes()
   ```
   This creates a byte string representing only the valid board positions.

3. **Select the minimum**: Choose the transformation that produces the lexicographically smallest byte representation:
   ```python
   best_name = "R0"
   best_state = np.copy(self.state)
   best_key = self._canonical_key(best_state)

   for name, fn in self._get_all_symmetry_transforms():
       s2 = fn(self.state)
       key = self._canonical_key(s2)
       if key < best_key:
           best_key, best_state, best_name = key, s2, name
   ```

### Return Values

`canonicalize_state()` returns a tuple:
- `canonical_state`: The transformed state with the lexicographically smallest representation
- `transform_name`: The symmetry transformation applied (e.g., "R120", "MR240", "R60M", "T1,0+R60")
- `inverse_name`: The inverse transformation to map back to the original orientation (e.g., "R300+T-1,0")

## Consistency Guarantee

**Lexicographic ordering is a total ordering** - for any two different byte strings, one is always definitively less than the other.

This guarantees:
- **Uniqueness**: There is always exactly one minimum across all symmetry-equivalent forms
- **Determinism**: No matter which orientation you start with, you always get the same canonical state
- **Independence**: The canonical form is independent of the starting orientation

### Example

Consider a board state and its symmetry-equivalent forms:
- Original state → canonical key = `0x1A2B3C...`
- Rotated 60° → canonical key = `0x2C3D4E...`
- Rotated 120° → canonical key = `0x0F1E2D...` ← **Smallest (canonical)**
- Mirrored → canonical key = `0x3F4E5D...`
- ... (all other symmetries)

The algorithm always selects the 120° rotation because it has the smallest byte representation. If you later encounter any of these equivalent states, the algorithm will test all transformations again and always pick the same winner.

## Symmetry Transformations

### Transform Notation

- `R{angle}`: Pure rotation by angle (0, 60, 120, 180, 240, 300 for D6)
- `MR{angle}`: Rotate by angle, THEN mirror (rotate-then-mirror)
- `R{angle}M`: Mirror THEN rotate by angle (mirror-then-rotate)

### Inverse Relationships

- `R(k)⁻¹ = R(-k)` (pure rotations)
- `MR(k)⁻¹ = R(-k)M` (rotate-then-mirror inverts to mirror-then-rotate)
- `R(k)M⁻¹ = MR(-k)` (mirror-then-rotate inverts to rotate-then-mirror)

## Translation Canonicalization

### Overview

When edge rings are removed during gameplay, the remaining board configuration can be translated (shifted) within the grid while maintaining validity. Translation canonicalization leverages this to further reduce the state space.

### When Translation Is Available

Translation is only possible when:
1. Edge rings have been removed (creating empty space around the remaining board)
2. The remaining rings form a smaller cluster that can fit in multiple positions

At the start of the game, all edge positions are occupied, so translation is not available. As rings are removed through gameplay, translation opportunities emerge.

### How Translation Works

**Bounding Box**: The algorithm first calculates the minimal rectangle containing all remaining rings:

```python
def _get_bounding_box(self):
    """Find minimal rectangle containing all rings."""
    ring_positions = np.argwhere(self.state[self.RING_LAYER] == 1)
    if len(ring_positions) == 0:
        return None

    min_y, min_x = ring_positions.min(axis=0)
    max_y, max_x = ring_positions.max(axis=0)

    return (min_y, max_y, min_x, max_x)
```

**Valid Translations**: The algorithm generates all valid translation offsets that keep the bounding box within the grid:

```python
def _get_all_translations(self):
    """Get all valid translation offsets."""
    bbox = self._get_bounding_box()
    if bbox is None:
        return [(0, 0)]  # No rings, only identity

    min_y, max_y, min_x, max_x = bbox
    bbox_height = max_y - min_y + 1
    bbox_width = max_x - min_x + 1

    translations = []
    for new_min_y in range(self.width - bbox_height + 1):
        for new_min_x in range(self.width - bbox_width + 1):
            dy = new_min_y - min_y
            dx = new_min_x - min_x
            translations.append((dy, dx))

    return translations
```

**Translation Application**: Translation only affects the BOARD_LAYERS (rings + marbles). History and capture layers are preserved as-is:

```python
def _translate_state(self, state, dy, dx):
    """Translate state by offset (dy, dx)."""
    if dy == 0 and dx == 0:
        return np.copy(state)

    result = np.zeros_like(state)

    # Translate only BOARD_LAYERS
    board_portion = state[self.BOARD_LAYERS]

    src_y_start = max(0, -dy)
    src_y_end = min(self.width, self.width - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(self.width, self.width - dx)

    dst_y_start = max(0, dy)
    dst_y_end = min(self.width, self.width + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(self.width, self.width + dx)

    result[self.BOARD_LAYERS, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        board_portion[:, src_y_start:src_y_end, src_x_start:src_x_end]

    # Preserve history and capture layers
    result[self.CAPTURE_LAYER:] = state[self.CAPTURE_LAYER:]

    return result
```

### Transform Composition

When translation is combined with rotation/mirror, **translation is applied FIRST**, then rotation/mirror:

```
Combined Transform = Translate → Rotate/Mirror
```

This order is important for:
1. **Consistency**: Ensures deterministic canonicalization
2. **Inverse calculation**: Makes computing the inverse straightforward
3. **Efficiency**: Translation is cheaper than rotation/mirror

### Combined Transform Notation

Combined transforms use the notation: `T{dy},{dx}+{rot_mirror}`

Examples:
- `T1,0+R60`: Translate by (1,0), then rotate 60°
- `T2,1+MR120`: Translate by (2,1), then rotate 120° and mirror
- `T-1,3+R180`: Translate by (-1,3), then rotate 180°
- `T0,1+R120M`: Translate by (0,1), then mirror and rotate 120°

### Inverse Transforms for Combined Transforms

For combined transforms, the inverse reverses both the operations and their order:

```
If transform = T(dy,dx) → M/R(k)
Then inverse = M/R(k)⁻¹ → T(-dy,-dx)
```

**Examples**:
- `T1,0+R60` → inverse is `R300+T-1,0`
- `T2,1+MR120` → inverse is `R240M+T-2,-1`
- `T-1,3+R180` → inverse is `R180+T1,-3`
- `T0,1+R120M` → inverse is `MR240+T0,-1`

**Pattern**:
- Translation component: `T{dy},{dx}` → `T{-dy},{-dx}`
- Rotation/mirror component: Apply standard inverse rules
- Order: Reverse the sequence (rotation/mirror inverse comes first)

### Transform Control with TransformFlags

The `canonicalize_state()` method accepts a `transforms` parameter to control which symmetries are used:

```python
from game.zertz_board import TransformFlags

# Use all transforms (default)
canonical_state, transform, inverse = board.canonicalize_state(
    transforms=TransformFlags.ALL
)

# Only rotation and mirror (no translation)
canonical_state, transform, inverse = board.canonicalize_state(
    transforms=TransformFlags.ROTATION_MIRROR
)

# Only translation (no rotation/mirror)
canonical_state, transform, inverse = board.canonicalize_state(
    transforms=TransformFlags.TRANSLATION
)

# Only rotation (no mirror or translation)
canonical_state, transform, inverse = board.canonicalize_state(
    transforms=TransformFlags.ROTATION
)

# Only mirror (no rotation or translation)
canonical_state, transform, inverse = board.canonicalize_state(
    transforms=TransformFlags.MIRROR
)

# No transforms (identity only)
canonical_state, transform, inverse = board.canonicalize_state(
    transforms=TransformFlags.NONE
)
```

**TransformFlags enum values**:
- `ROTATION = 1`: Enable rotations (R60, R120, R180, R240, R300)
- `MIRROR = 2`: Enable mirroring (MR0, MR60, MR120, MR180, MR240, MR300 or R0M, R60M, R120M, R180M, R240M, R300M)
- `TRANSLATION = 4`: Enable translations (T{dy},{dx})
- `ROTATION_MIRROR = 3`: ROTATION | MIRROR
- `ALL = 7`: ROTATION | MIRROR | TRANSLATION
- `NONE = 0`: Only identity transform (R0)

### Performance Impact

Translation adds complexity to canonicalization:
- **Without translation**: 12 transforms for D6 (6 rotations × 2 for mirror)
- **With translation**: 12 × N transforms, where N = number of valid translations

However, translation is beneficial because:
1. It only activates when edge rings are removed (mid-to-late game)
2. The number of valid translations is typically small (< 10)
3. The state space reduction often outweighs the computational cost
4. Translation is a simple offset operation (faster than rotation/mirror)

## Querying All Transformations

The `CanonicalizationManager` provides a method to retrieve all valid transformation permutations of a state:

```python
from game.zertz_board import ZertzBoard

board = ZertzBoard(37)
# ... make some moves ...

# Get all unique transformations (without translation)
all_transforms = board.canonicalizer.get_all_transformations(
    state=board.state,
    include_translation=False
)

# Returns: dict mapping transform_name -> transformed_state
# Example: {'R0': state1, 'R60': state2, 'R120': state3, 'MR0': state4, ...}

for transform_name, transformed_state in all_transforms.items():
    print(f"Transform {transform_name}: {transformed_state.shape}")
```

### Deduplication and Simplicity Scoring

The method automatically eliminates duplicate states, keeping only the **simplest** transform name for each unique state:

**Transform Simplicity Priority** (lower is simpler):
1. **Identity** (`R0`)
2. **Pure rotations** (`R60`, `R120`, `R180`, `R240`, `R300`)
3. **Mirror-based** (`MR*`, `R*M`)
4. **Translations** (`T*`)
5. **Combined** (`T*_*`)

**Example**: If both `R60` and `MR300` produce the same state, only `R60` is included (pure rotation is simpler).

### Parameters

- `state` (optional): State to transform. If `None`, uses the board's current state.
- `include_translation` (bool): Whether to include translations. Default: `True`
  - `True`: Returns all valid transforms including translations
  - `False`: Only rotation/mirror symmetries (recommended for counting permutations)

### Use Cases

**Counting permutations**:
```python
# How many unique orientations exist for this state?
transforms = board.canonicalizer.get_all_transformations(
    include_translation=False
)
num_permutations = len(transforms)  # Typically 12 for D6, 6 for D3
```

**Visualization**:
```python
# Generate images for all unique orientations
all_transforms = board.canonicalizer.get_all_transformations(
    state=canonical_state,
    include_translation=False
)

for name, transformed_state in sorted(all_transforms.items()):
    # Render transformed_state with label 'name'
    ...
```

**Policy augmentation for ML**:
```python
# Generate training examples from all equivalent orientations
for transform_name, transformed_state in all_transforms.items():
    # Store (transformed_state, policy, value) as training example
    ...
```

## Usage

```python
from game.zertz_board import ZertzBoard

board = ZertzBoard(37)
# ... make some moves ...

# Get canonical form
canonical_state, transform, inverse = board.canonicalize_state()

print(f"Transform applied: {transform}")
print(f"Inverse transform: {inverse}")

# The canonical_state is now in its standard form
# To map policy outputs back to original orientation, use the inverse transform
```

## Analysis Tool

Use `analyze_canonicalization.py` to visualize how states are canonicalized and analyze transformation permutations:

### Basic Usage

```bash
# Analyze notation file without saving images
uv run analyze_canonicalization.py data/logfiles/game.txt

# Save visualizations
uv run analyze_canonicalization.py --save-images data/logfiles/game.txt

# Custom output directory
uv run analyze_canonicalization.py --save-images --output-dir ./analysis/ data/logfiles/game.txt
```

### Advanced Visualization

```bash
# Show all transformation permutations in images
uv run analyze_canonicalization.py --show-all-transforms data/logfiles/game.txt

# Control grid layout (default: 6 columns)
uv run analyze_canonicalization.py --show-all-transforms --image-columns 4 data/logfiles/game.txt
```

### Features

**Text Output**:
- Move-by-move analysis showing transform applied (e.g., "R120", "MR60")
- Summary table with:
  - **Canonical State ID**: Sequential ID for each unique canonical state
  - **First Seen**: Move number where canonical state first appeared
  - **Source States**: Number of original states mapping to this canonical form
  - **Permutations**: Number of unique transformation permutations for this state
  - **Average**: Mean permutations across all canonical states

**Image Visualization** (when `--save-images` is specified):
- **Grid layout**: Images arranged in rows and columns (default: 6 per row)
- **Canonical state**: Always shown first (left-most), labeled "Canonical (R0)"
- **Original state**: Always shown second, labeled "Original ({transform})"
- **All transforms** (when `--show-all-transforms` is specified): Additional images showing all other unique transformation permutations
- **Background matching**: Blank grid areas use the same background color as the generated images

**Example Output**:
```
Move 5:
  Status: TRANSFORMED (via R120)
  Canonical State ID: #3
--------------------------------------------------------------------------------

================================================================================
CANONICALIZATION SUMMARY
================================================================================

Canonical State ID        First Seen      Source States        Permutations
--------------------------------------------------------------------------------
State #1                  Move 0          1                    12
State #2                  Move 1          1                    12
State #3                  Move 2          4                    6
State #4                  Move 6          2                    12
--------------------------------------------------------------------------------
TOTAL                                     8                    10.5 avg
```

### Command-Line Options

- `notation_file`: Path to notation file (required)
- `--save-images`: Save visualization images for each state
- `--show-all-transforms`: Show all unique transformations in images (implies `--save-images`)
- `--output-dir DIR`: Directory to save images (default: `canonicalization_output`)
- `--image-columns N`: Number of images per row in grid layout (default: 6)

## Technical Details

### Board Layers

The canonical key considers only the board layers (rings and marbles), not global state like marble supply or current player. This is intentional - we want to canonicalize the board position itself, independent of whose turn it is.

```python
masked = (state[self.BOARD_LAYERS] * self.board_layout).astype(np.uint8)
```

### Performance

Canonicalization requires testing all symmetry transformations (12 for D6, 6 for D3), making it O(n × board_size) where n is the number of symmetries. This is acceptable because:
- It's only called when needed (e.g., for ML state storage)
- The board size is small (37-61 positions)
- The operation is vectorized using NumPy

## References

- Implementation: `game/utils/canonicalization.py:CanonicalizationManager`
  - Main method: `canonicalize_state()`
  - Query all transforms: `get_all_transformations()`
  - Transform simplicity: `_transform_simplicity_score()`
- Symmetry transforms: `game/utils/canonicalization.py:_get_all_symmetry_transforms()`
- Translation support:
  - Bounding box: `game/utils/canonicalization.py:_get_bounding_box()`
  - Valid translations: `game/utils/canonicalization.py:_get_all_translations()`
  - Translation application: `game/utils/canonicalization.py:_translate_state()`
- Inverse transforms: `game/utils/canonicalization.py:_get_inverse_transform()`
- Key generation: `game/utils/canonicalization.py:_canonical_key()`
- Transform flags: `game/utils/canonicalization.py:TransformFlags`
- Tests:
  - Translation canonicalization: `tests/test_hex_transforms.py:TestTranslationCanonicalization`
  - Query all transforms: `tests/test_hex_transforms.py:TestGetAllTransformations`
- Analysis tool: `analyze_canonicalization.py`


# TL;DR

## Canonical State Selection Algorithm

The canonicalize_state() method uses a lexicographic minimum approach:

1. **Generate all symmetries**: It applies every possible symmetry transformation to the current state
   - D6 symmetry = 12 transforms for 37/61-ring boards (6 rotations × 2 for mirror)
   - D3 symmetry = 6 transforms for 48-ring boards (3 rotations × 2 for mirror)
   - When edge rings removed: N translations × 12 (D6) or N translations × 6 (D3)
   - Translation applied FIRST, then rotation/mirror

2. **Create comparable keys**: For each transformed state, it generates a _canonical_key():
   ```python
   masked = (state[self.BOARD_LAYERS] * self.board_layout).astype(np.uint8)
   return masked.tobytes()
   ```

3. **Pick the minimum**: It chooses the transformation that produces the lexicographically smallest byte representation:
   ```python
   if key < best_key:
       best_key, best_state, best_name = key, s2, name
   ```

## Why This Guarantees Consistency

Lexicographic ordering is a total ordering - for any two different byte strings, one is always definitively less than the other. This means:

- Given any state, there's always a unique minimum across all symmetry-equivalent forms
- No matter which symmetry-equivalent state you start with, you'll always get the same canonical state (the lexicographic minimum)
- The canonical form is deterministic and independent of the starting orientation

For example, if you have a state and its 180° rotation, both will be tested, and whichever produces the smaller byte string wins. If you later encounter that same state rotated 60°, it will test all transforms again and pick the same winner.

This is a standard technique in game state canonicalization - using lexicographic ordering to break symmetry deterministically.

## Translation Canonicalization Summary

**When available**: Edge rings removed → remaining board can be shifted within the grid

**Transform composition**: `T{dy},{dx}+{rot_mirror}` means translate first, then rotate/mirror
- Example: `T1,0+R60` = translate by (1,0), then rotate 60°

**Inverse composition**: Reverse both operations and their order
- Example: `T1,0+R60`⁻¹ = `R300+T-1,0` (rotate -60°, then translate by (-1,0))

**Control via flags**: Use `TransformFlags` enum to enable/disable translation
```python
# All transforms (rotation + mirror + translation)
board.canonicalize_state(transforms=TransformFlags.ALL)

# Only rotation and mirror (no translation)
board.canonicalize_state(transforms=TransformFlags.ROTATION_MIRROR)

# Only translation (no rotation/mirror)
board.canonicalize_state(transforms=TransformFlags.TRANSLATION)
```

**Performance**: Translation adds N × 12 (or N × 6) transforms where N is typically < 10, but only activates mid-to-late game when edge rings are removed.