# Zertz3D TODO and Improvement Suggestions

## Feature Ideas
- Add game statistics tracking
- Network multiplayer support


## Missing Test Coverage:

  1. Marble Supply Fix (logic implemented, tests incomplete)

  - ❌ No test for: "One marble type empty, others available → cannot use captured marbles"
  - ❌ No test for: "All marbles empty → can use captured marbles"
  - ❌ No test for: Transition from supply to captured marbles

  2. Isolated Regions (7 tests in test_isolated_regions.py)

  - ✅ Single ring with marble (captured)
  - ✅ Single ring vacant (frozen)
  - ✅ Two rings, all occupied (captured)
  - ✅ Two rings, one vacant (frozen)
  - ❌ Multiple isolated regions in one move (some capturable, some frozen)
  - ❌ What if main board becomes smaller than an isolated region?
  - ❌ Capture action creating isolation

  3. Win Conditions (logic implemented in _is_game_over, tests incomplete)

  - ❌ Win during chain capture
  - ❌ Win immediately after isolated region capture
  - ✅ Board completely full (logic exists in _is_game_over)
  - ✅ Player runs out of marbles (logic exists in _is_game_over, tested via test_player_can_pass_when_no_valid_moves)
  - ✅ Immobilization win conditions (3 tests in test_pass_and_loops.py)

  4. Chain Captures (basic logic tested, edge cases not)

  - ❌ Long capture sequences (3+ jumps)
  - ❌ Multiple available chain paths (player choice)
  - ❌ CAPTURE_LAYER flag behavior

  5. Edge Cases (partially tested)

  - ✅ No removable rings on full board (3 tests in test_geometric_ring_removal.py)
  - ❌ Ring removal creating multiple frozen regions simultaneously
  - ❌ Geometric vs adjacency ring removal on edge cases


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

## Recently Completed (2025-10-09)

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

## Future Enhancements
