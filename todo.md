# Zertz3D TODO and Improvement Suggestions

## Code Quality Improvements

### ZertzRenderer
1. Visual indicators for valid moves.

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
