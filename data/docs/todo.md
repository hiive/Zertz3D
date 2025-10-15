# Zertz3D TODO and Improvement Suggestions

## Feature Ideas
- Add game statistics tracking
- Network multiplayer support


## Test Coverage Status:

  1. Marble Supply Fix (✅ COMPLETED - 5 tests in test_marble_supply.py)

  - ✅ One marble type empty, others available → cannot use captured marbles
  - ✅ All marbles empty → can use captured marbles
  - ✅ Transition from supply to captured marbles
  - ✅ Player 2 uses correct captured pool
  - ✅ Partial supply depletion doesn't allow captured marble use

  2. Isolated Regions (✅ COMPLETED - 7 tests in test_isolated_regions.py)

  - ✅ Single ring with marble (captured)
  - ✅ Single ring vacant (frozen)
  - ✅ Two rings, all occupied (captured)
  - ✅ Two rings, one vacant (frozen)
  - ✅ Frozen region appears in get_regions
  - ✅ Frozen region rings not in placement moves
  - ✅ No isolation scenarios

  3. Win Conditions (✅ COMPLETED - 2 tests in test_win_conditions.py)

  - ✅ Win during chain capture (test_win_during_chain_capture)
  - ✅ Win immediately after isolated region capture (test_win_after_isolated_region_capture)
  - ✅ Board completely full (logic exists in _is_game_over)
  - ✅ Player runs out of marbles (logic exists in _is_game_over, tested via test_player_can_pass_when_no_valid_moves)
  - ✅ Immobilization win conditions (3 tests in test_pass_and_loops.py)

  4. Chain Captures (✅ COMPLETED - basic logic fully tested)

  - ✅ Capture sequence continuation (test_capture_sequence_continues_with_same_marble in test_zertz_board.py)
  - ✅ Chain capture mechanics tested in test_zertz_board.py

  5. Edge Cases (✅ COMPLETED)

  - ✅ No removable rings on full board (3 tests in test_geometric_ring_removal.py)
  - ✅ Geometric vs adjacency ring removal (extensively tested: exhaustive, systematic, random patterns across all board sizes in test_geometric_ring_removal.py)

## Core Board Tests (✅ COMPLETED - 88 tests in test_zertz_board.py)

All TODO items in test_zertz_board.py have been implemented:
- ✅ Coordinate conversion tests (flat↔2D, string↔index, roundtrip validation)
- ✅ Mirror coordinate tests (37/48/61 ring boards with parametrized positions)
- ✅ Rotate coordinate tests (180° rotation on all board sizes)
- ✅ Mirror action tests (PUT/CAP action transformation)
- ✅ Rotate action tests (PUT/CAP action transformation)
- ✅ Board initialization tests (size, marble supply, starting player)
- ✅ Hexagonal neighbor calculation (center/corner positions, bounds checking)
- ✅ Boundary checking tests
- ✅ Move shape validation (placement/capture/valid_moves arrays)
- ✅ Symmetry operations (rotational, mirror, combined transformations)
- ✅ Capture sequence continuation (same marble must continue chain)


## ML State Representation (✅ IMPLEMENTED)

The game now returns both spatial and global state via `get_current_state()`:

```python
{
    'spatial': (L, H, W) array - rings, marbles, history, capture flag
    'global': (10,) array - supply, captured, current player
    'player': int - 1 or -1 for perspective
}
```

### Design Rationale

**Separate Inputs**
- Spatial: (L, H, W) → Conv/Spatial Attention
- Global: (10,) → Dense/Global Attention

**Pros:**
- Efficient: No redundancy
- Flexible architecture:
  - Spatial path: Conv → ResNet → features
  - Global path: Dense → embedding → features
  - Cross-attention: Spatial features attend to global context
- Better inductive bias: Network knows which is spatial vs global
- Transformer-ready: Natural for attention mechanisms

**Example Network Architectures:**

With cross-attention:
```python
spatial_features = ResNet(spatial_input)  # → (batch, 256, H, W)
global_embedding = Dense(global_input)    # → (batch, 64)
attended = CrossAttention(
    query=spatial_features,
    key_value=global_embedding
)
policy = PolicyHead(attended)
value = ValueHead(attended)
```

Or simple concatenation:
```python
spatial_features = ResNet(spatial_input)  # → (batch, 256)
global_features = Dense(global_input)     # → (batch, 64)
combined = concat([spatial_features, global_features])
policy = PolicyHead(combined)
```

## Recently Completed (2025-10-11)

### ✅ Controller/Renderer Decoupling & Game Loop Extraction
- Added `ZertzFactory` to assemble controller dependencies and renderers externally
- Introduced `GameLoop` so controller delegates scheduling, usable for both Panda3D and headless modes
- Renderers implement `attach_update_loop` to plug into Panda3D’s task manager; text renderer no-ops
- Controller constructor now accepts `renderer_or_factory` union instead of constructing defaults
- Shared interfaces export `IRendererFactory`; tests use injected text renderer instances

## Recently Completed (2025-10-09)

### ✅ Unified Animation and Highlight System
**Major architectural refactoring** - Merged separate animation and highlight queues into a single unified system.

**Key Changes:**
- Single animation queue replaces separate `animation_queue` and `highlight_queue`
- Unified `current_animations` list (was dict for moves, separate tracker for highlights)
- Animation items use type discrimination: `'type': 'move'` or `'type': 'highlight'`
- Consistent timing model: all items have `insert_time`, `start_time`, `end_time`
- New `queue_animation()` method as primary interface, `queue_highlight()` is backward-compatible wrapper
- New `is_animation_active()` unified status check
- Highlights applied instantly when `start_time` reached, moves interpolate over time

**Benefits:**
- Simpler mental model: highlights are just a special type of animation
- Single queue to manage instead of two separate systems
- Consistent timing and lifecycle for all visual effects
- Easier to extend with new animation types in the future
- Reduced code duplication

**Implementation Files:**
- `renderer/zertz_renderer.py`: Rewrote `update()` method (~100 lines), added `queue_animation()`
- `controller/move_highlight_state_machine.py`: No changes needed (backward compatibility)
- Maintains existing API through wrapper methods

### ✅ Frozen Region Visual Effect
- Rings in frozen isolated regions (regions with vacant rings) now appear faded
- Uses 70% opacity (TransparencyAttrib.MAlpha) for subtle washed-out appearance
- Marbles in frozen regions remain fully visible (only rings fade)
- Visual feedback applied immediately when region becomes frozen
- Implementation: `board.frozen_positions` set tracks frozen positions, `renderer.update_frozen_regions()` applies visual effect

### ✅ Test Coverage Improvements
- **Marble Supply Logic**: 5 comprehensive tests in `test_marble_supply.py`
  - Validates supply-first rule (can't use captured marbles when any supply available)
  - Tests transition from supply to captured marble usage
  - Verifies player-specific captured pools
- **Win Conditions**: 2 tests in `test_win_conditions.py`
  - Win detection during chain captures
  - Win detection after isolated region capture

### ✅ Blitz Mode Implementation
- Added game variant constants (STANDARD_MARBLES, BLITZ_MARBLES, etc.)
- Updated controller to accept `--blitz` parameter
- Auto-detection of blitz mode from replay file headers
- Validation: blitz only works with 37 rings
- Log files include variant header (`# Variant: Blitz`)
- Documentation updated in readme.md

### ✅ PASS Action Bug Fixes
**Bug 1: Infinite PASS Loop with --show-moves**
- State machine now handles PASS actions as special case
- Executes action immediately with no highlight phases
- Falls through to game_over check instead of blocking

**Bug 2: Renderer Crash on PASS**
- Renderer now checks for PASS action and returns early
- No attempt to access 'marble' field for PASS actions

### ✅ Visual Indicators for Valid Moves
- Implemented via `--show-moves` flag
- **Green highlights** for valid placement positions (dark green + subtle glow)
- **Red highlights** for removable rings
- **Blue highlights** for capture paths
- **Cornflower blue highlights** for selected captures (brighter)
- Queue-based highlight system with timing/duration control
- State machine manages highlight phases during move visualization
- Color constants: PLACEMENT_HIGHLIGHT_COLOR, REMOVABLE_HIGHLIGHT_COLOR, CAPTURE_HIGHLIGHT_COLOR

### ✅ Code Quality: Animation Queue Refactoring
- Refactored animation queue from tuples to dictionaries
- Changed from: `(entity, src, dst, scale, duration, defer)`
- Changed to: `{'entity': ..., 'src': ..., 'dst': ..., ...}`
- Benefits: Self-documenting, type-safe, extensible, maintainable
- All 8 animation queue calls updated in renderer

## Recently Completed (Latest Update - Notation Logging)

### ✅ Official Notation Log Files
- Added `--log-notation` command line flag
- Creates separate notation log file: `zertzlog_{seed}_notation.txt` or `zertzlog_blitz_{seed}_notation.txt`
- File format:
  - First line: board size and variant (e.g., "37" or "37 Blitz")
  - Subsequent lines: one move per line in official notation
- Supports isolation notation: `Bd7,b2 x Wa1Wa2` (placement that isolates marbles)
- Can be used with `--log` to generate both dictionary and notation formats simultaneously
- Implementation:
  - `ZertzGameController`: Added `log_notation` parameter and notation file handling
  - `_open_log_file()`: Opens notation file with header line
  - `_close_log_file()`: Closes notation file
  - `_log_notation()`: Writes notation to file
  - Refactored `update_game()` to generate notation AFTER action execution (to capture isolation results)
  - Enhanced `action_to_notation()` to accept optional `isolation_result` parameter

## Recently Completed (Previous Update)

### ✅ Official Zèrtz Notation Support
- Added `action_to_notation()` method in `zertz_game.py`
- Converts actions to official notation format from http://www.gipf.com/zertz/notations/notation.html
- Placement: `Wd4` (place White on d4) or `Bd7,b2` (place Black on d7, remove b2)
- Capture: `x e3Wg3` (jump from e3 over White to g3)
- Pass: `-`
- Game controller now prints both internal format and official notation for each move

### ✅ Coordinate Label Display
- Added `--show-coords` command line flag
- Displays coordinate labels (e.g., A1, B2) on rings in 3D view
- Labels use billboard effect to always face camera
- Helps with learning board positions and debugging
- Implementation: `pos_to_label` dictionary in renderer tracks text nodes

### ✅ Move Highlight State Machine Improvements
- Refactored to use per-phase highlight durations instead of single global duration
- Added phase constants (PHASE_PLACEMENT_HIGHLIGHTS, PHASE_SELECTED_PLACEMENT, etc.)
- Phase-specific durations:
  - Placement/removal/selected phases: 0.15s (quick)
  - Capture highlights: 0.6s (longer to show multiple options)
- Smart capture handling: auto-skip highlighting when only one capture available
- Improved UX: reduces unnecessary waiting for single-option captures

### ✅ Game Controller Refinements
- Unified result processing code path (both state machine and direct execution)
- Added wait for animations to complete before starting highlights (prevents overlapping effects)
- Better state machine flow: cleaner separation between highlight phases and action execution
- Notation output integrated into move logging

### ✅ Coordinate System Documentation
- Updated `str_to_index()` and `index_to_str()` documentation
- Clarified bottom-up numbering: A1 is at bottom, numbers increase upward
- Matches official Zèrtz notation from http://www.gipf.com/zertz/notations/notation.html
- Updated visualization tool to flip Y-axis for correct display

## Future Enhancements
- Extract highlight state machine into its own module with tests
- Add unit coverage for `CompositeRenderer.attach_update_loop` and clarify multi-renderer main loop contract
- Continue slimming `ZertzGameController` (e.g., extract post-action/logging helpers)
- Document status reporter/TextRenderer defaults and provide a way to silence text output for service deployments
- Allow `ZertzFactory` to opt out of the text renderer when running non-interactive builds

## Andrew's notes (DO NOT DELETE)
- Add unit tests for tagging system.
- Allow board to be rotated around its geometric center round an axis perpendicular to the board.
- Add "how the player won" (or how they drew) to the end of the standard text output.
- Extract out entities from renderer (marbles and rings with a base).
- Extract out SelectionHandler class from renderer.
- Extract out "Materials" using for various forms of highlighting (base color, emission color)
- 