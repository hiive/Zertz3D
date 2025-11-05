# Rust Wrapper API Changes Review

## Overview

This document reviews the Zertz3D codebase for compatibility with the major Rust wrapper refactoring in branch `claude/review-rust-wrapper-exports-011CUoRo6T7s6P4C1W6dj3qi` from the `hiivelabs-zertz-mcts` repository.

**Review Date**: 2025-11-05
**Reference Branch**: `claude/review-rust-wrapper-exports-011CUoRo6T7s6P4C1W6dj3qi`
**Current Submodule Commit**: `12537a6`
**Reference Branch Commits**: 50 commits ahead with extensive refactoring

## Summary

The reference branch introduces **breaking API changes** that will require updates to the Zertz3D Python codebase. However, the changes are well-designed with backward compatibility paths and clear migration strategies.

**Impact Level**: 🟡 **MODERATE** - Required changes are straightforward but affect multiple files

---

## Major API Changes

### 1. Module Reorganization

**Current**:
```python
import hiivelabs_zertz_mcts
from hiivelabs_zertz_mcts import MCTSSearch, BoardConfig
```

**New**:
```python
import hiivelabs_mcts
from hiivelabs_mcts.zertz import ZertzMCTS, BoardConfig, BoardState
from hiivelabs_mcts import PLAYER_1, PLAYER_2, TransformFlags
```

**Changes**:
- Module renamed: `hiivelabs_zertz_mcts` → `hiivelabs_mcts`
- Game-specific code moved to submodules: `hiivelabs_mcts.zertz`
- New TicTacToe support added: `hiivelabs_mcts.tictactoe`
- Stub file consolidated: `hiivelabs_zertz_mcts.pyi` → `hiivelabs_mcts.pyi`

---

### 2. MCTS Class Renamed

**Current**:
```python
mcts = hiivelabs_zertz_mcts.MCTSSearch(
    exploration_constant=1.41,
    widening_constant=None,
    fpu_reduction=None,
    rave_constant=None,
)
```

**New**:
```python
from hiivelabs_mcts.zertz import ZertzMCTS

mcts = ZertzMCTS(
    rings=37,  # NOW REQUIRED!
    exploration_constant=1.41,
    widening_constant=None,
    fpu_reduction=None,
    rave_constant=None,
    blitz=False,  # NEW PARAMETER
    t=1,  # NEW PARAMETER
)
```

**Breaking Changes**:
- Class renamed: `MCTSSearch` → `zertz.ZertzMCTS`
- Constructor now **requires** `rings` parameter
- New optional parameters: `blitz`, `t`

---

### 3. BoardConfig Factory Methods

**Current**:
```python
# Direct construction (implementation-dependent)
config = BoardConfig(rings=37, t=1)
```

**New**:
```python
from hiivelabs_mcts.zertz import BoardConfig

# Factory methods (recommended)
config = BoardConfig.standard_config(rings=37, t=1)
config = BoardConfig.blitz_config(rings=37, t=1)
```

**Changes**:
- Direct construction replaced with factory methods
- Clear separation between standard and blitz modes
- More explicit and self-documenting API

---

### 4. Action Representation (New ZertzAction Class)

**Current (still supported but deprecated)**:
```python
# Tuple-based actions
action = ("PUT", (marble_type, dst_flat, rem_flat))
action = ("CAP", (direction, start_y, start_x))
action = ("PASS", None)
```

**New (recommended)**:
```python
from hiivelabs_mcts.zertz import ZertzAction, BoardConfig

config = BoardConfig.standard_config(rings=37)

# Strongly-typed action objects
action = ZertzAction.placement(config, marble_type=0, dst_y=3, dst_x=3, remove_y=None, remove_x=None)
action = ZertzAction.capture(config, src_y=2, src_x=2, dst_y=4, dst_x=4)
action = ZertzAction.pass_action()

# Convert to tuple format for compatibility
action_tuple = action.to_tuple(width=7)
```

**Changes**:
- Type-safe action construction
- Coordinate-based API (y, x) instead of flat indices
- Requires BoardConfig for coordinate conversion
- Old tuple-based API still works but marked `@deprecated`

---

### 5. Action Results (New ZertzActionResult Class)

**Current**:
```python
# apply_placement returns list of captured marbles
captures = apply_placement_action(board_state, global_state, action, config)
# Returns: [(marble_layer, y, x), ...]

# apply_capture returns None
apply_capture_action(board_state, global_state, action, config)
```

**New**:
```python
from hiivelabs_mcts.zertz import BoardState, ZertzAction

board = BoardState(spatial_state, global_state, rings=37)

# New methods return ZertzActionResult
result = board.apply_placement(action)
print(result.result_type())  # "Placement"
print(result.isolation_captures())  # [(marble_layer, y, x), ...] or None

result = board.apply_capture(action)
print(result.result_type())  # "Capture"
print(result.captured_marble())  # (marble_type, y, x) or None
```

**Changes**:
- Structured return values instead of mixed types
- Consistent API for both placement and capture
- Old stateless functions still work (backward compatible)

---

### 6. Game Constants Exported

**Current**:
```python
# Defined in Python constants.py
from game.constants import PLAYER_1_WIN, PLAYER_2_WIN, BOTH_LOSE
```

**New (also available from Rust)**:
```python
# Available from both Python and Rust
from hiivelabs_mcts.zertz import (
    PLAYER_1_WIN,  # = 1
    PLAYER_2_WIN,  # = -1
    TIE,  # = 0
    BOTH_LOSE,  # = -2
    STANDARD_MARBLES,  # = (6, 8, 10)
    BLITZ_MARBLES,  # = (5, 7, 9)
    STANDARD_WIN_CONDITIONS,  # = (3, 4, 5, 6)
    BLITZ_WIN_CONDITIONS,  # = (2, 3, 4, 5)
)
```

**Changes**:
- Game constants now exported from Rust
- Python constants can remain for backward compatibility
- Provides single source of truth

---

## Affected Files in Zertz3D

### Critical Files (Require Updates)

1. **`learner/mcts/backend.py`** (Lines 9-10)
   - Change: `import hiivelabs_zertz_mcts` → `import hiivelabs_mcts`
   - Impact: Backend detection and availability check

2. **`game/players/mcts_zertz_player.py`** (Lines 7, 75)
   - Change: `import hiivelabs_zertz_mcts` → `from hiivelabs_mcts.zertz import ZertzMCTS`
   - Change: `hiivelabs_zertz_mcts.MCTSSearch(...)` → `ZertzMCTS(rings=self.game.board.rings, ...)`
   - Impact: MCTS player initialization

3. **`game/zertz_logic.py`** (Lines 26-48)
   - Change: `from hiivelabs_zertz_mcts import ...` → `from hiivelabs_mcts.zertz import ...`
   - Impact: All stateless game logic functions

4. **`game/zertz_position.py`** (Lines 118-119)
   - Change: `from hiivelabs_zertz_mcts import build_axial_maps, BoardConfig`
   - → `from hiivelabs_mcts.zertz import build_axial_maps, BoardConfig`
   - Impact: Position coordinate mapping

5. **`game/utils/canonicalization.py`** (Lines 58-69)
   - Change: `from hiivelabs_zertz_mcts import ...` → `from hiivelabs_mcts import ...`
   - Impact: Canonicalization and transform operations

6. **`tests/test_rave.py`** (Lines 6)
   - Change: `from hiivelabs_zertz_mcts import MCTSSearch` → `from hiivelabs_mcts.zertz import ZertzMCTS`
   - Impact: RAVE tests

7. **`tests/test_crash_reproduction.py`** (assumed, not read)
   - Likely needs similar import changes

---

## Migration Strategy

### Phase 1: Update Imports (Breaking)

**Priority**: 🔴 HIGH - Required for compatibility

Update all imports across the codebase:

```python
# OLD
import hiivelabs_zertz_mcts
from hiivelabs_zertz_mcts import (
    MCTSSearch,
    BoardConfig,
    # ... other imports
)

# NEW
import hiivelabs_mcts
from hiivelabs_mcts.zertz import (
    ZertzMCTS,
    BoardConfig,
    BoardState,
    ZertzAction,
    ZertzActionResult,
    # ... game-specific imports
)
from hiivelabs_mcts import (
    PLAYER_1,
    PLAYER_2,
    TransformFlags,
    # ... shared imports
)
```

**Files to Update**:
- `learner/mcts/backend.py`
- `game/players/mcts_zertz_player.py`
- `game/zertz_logic.py`
- `game/zertz_position.py`
- `game/utils/canonicalization.py`
- `tests/test_rave.py`
- `tests/test_crash_reproduction.py`

---

### Phase 2: Update MCTS Initialization (Breaking)

**Priority**: 🔴 HIGH - Required for compatibility

In `game/players/mcts_zertz_player.py`:

```python
# OLD (line 75)
self.rust_mcts = hiivelabs_zertz_mcts.MCTSSearch(
    exploration_constant=exploration_constant,
    widening_constant=widening_constant,
    fpu_reduction=fpu_reduction,
    rave_constant=rave_constant,
    use_transposition_table=use_transposition_table,
    use_transposition_lookups=use_transposition_lookups,
)

# NEW
from hiivelabs_mcts.zertz import ZertzMCTS

is_blitz = self._is_blitz_mode()

self.rust_mcts = ZertzMCTS(
    rings=game.board.rings,  # NEW: required parameter
    exploration_constant=exploration_constant,
    widening_constant=widening_constant,
    fpu_reduction=fpu_reduction,
    rave_constant=rave_constant,
    use_transposition_table=use_transposition_table,
    use_transposition_lookups=use_transposition_lookups,
    blitz=is_blitz,  # NEW: set blitz mode at construction
    t=getattr(game.board, 't', 1),  # NEW: time history depth
)
```

**Impact**:
- `rings` parameter must be passed at construction
- Blitz mode detection must happen earlier (before MCTS creation)
- May need to move `_is_blitz_mode()` call or make it static

---

### Phase 3: Update BoardConfig Usage (Recommended)

**Priority**: 🟡 MEDIUM - Recommended but not critical

Replace direct BoardConfig construction with factory methods:

```python
# OLD
config = BoardConfig(rings=37, t=1)

# NEW (recommended)
config = BoardConfig.standard_config(rings=37, t=1)

# For blitz mode
config = BoardConfig.blitz_config(rings=37, t=1)
```

**Files to Update**:
- `game/zertz_logic.py` (if creating configs)
- `game/zertz_position.py` (line 120)
- `game/utils/canonicalization.py` (lines 122, 183, etc.)

---

### Phase 4: Adopt ZertzAction API (Optional)

**Priority**: 🟢 LOW - Optional modernization

The new `ZertzAction` API provides type safety but requires more refactoring. The old tuple-based API will continue to work (marked `@deprecated`).

**Current tuple-based approach (still works)**:
```python
# This still works in the new API
action = ("PUT", (marble_type, dst_flat, rem_flat))
action = ("CAP", (direction, start_y, start_x))
```

**New typed approach (recommended for new code)**:
```python
from hiivelabs_mcts.zertz import ZertzAction, BoardConfig

config = BoardConfig.standard_config(rings=37)

# Create actions with coordinate-based API
action = ZertzAction.placement(
    config,
    marble_type=0,
    dst_y=3,
    dst_x=3,
    remove_y=None,
    remove_x=None
)

# Convert to tuple for backward compatibility
action_tuple = action.to_tuple(width=7)
```

**Decision**: Recommend keeping tuple-based API for now unless type safety becomes a priority.

---

## Testing Requirements

### Unit Tests to Update

1. **Import tests** - Verify new module structure
2. **MCTS initialization tests** - Test new required parameters
3. **BoardConfig tests** - Test factory methods
4. **Action tests** - Test tuple API still works

### Integration Tests

1. **Full game playthrough** - Ensure MCTS player works end-to-end
2. **Replay tests** - Verify transcript/notation loaders still work
3. **Canonicalization tests** - Ensure transforms still work correctly

### Backward Compatibility Tests

1. **Deprecated API** - Verify old tuple-based actions still work
2. **Function signatures** - Ensure stateless functions unchanged

---

## Risk Assessment

### High Risk ⚠️

1. **Import changes break existing code**
   - Mitigation: Update all imports in single commit
   - Testing: Run full test suite after import changes

2. **MCTS initialization requires rings parameter**
   - Mitigation: Ensure `game.board.rings` is always available
   - Testing: Test all player initialization paths

### Medium Risk ⚠️

3. **Blitz mode detection timing**
   - Current: Detected at search time
   - New: Must be set at MCTS construction
   - Mitigation: Move detection to `__init__` or make static
   - Testing: Test both standard and blitz modes

4. **BoardConfig factory methods**
   - Direct construction may still work but is not guaranteed
   - Mitigation: Use factory methods consistently
   - Testing: Test both standard and blitz configs

### Low Risk ✅

5. **Tuple-based actions deprecated**
   - Still supported with backward compatibility
   - Migration: Optional, can be done incrementally
   - Testing: Verify tuple API continues to work

---

## Recommended Implementation Plan

### Step 1: Prepare (No Code Changes)
- [ ] Review this document with team
- [ ] Create feature branch from current main
- [ ] Ensure all tests pass on current code

### Step 2: Update Rust Submodule
- [ ] Update rust submodule to reference branch commit
- [ ] Rebuild Rust extension: `./rust-build.sh`
- [ ] Verify build completes without errors

### Step 3: Update Imports (Single Commit)
- [ ] Update all `import hiivelabs_zertz_mcts` statements
- [ ] Update all `from hiivelabs_zertz_mcts import` statements
- [ ] Update `MCTSSearch` → `ZertzMCTS`
- [ ] Run tests (expect failures)

### Step 4: Fix MCTS Initialization
- [ ] Add `rings` parameter to `ZertzMCTS` construction
- [ ] Move blitz mode detection to `__init__`
- [ ] Add `blitz` and `t` parameters
- [ ] Update tests to match new API

### Step 5: Update BoardConfig Usage
- [ ] Replace direct construction with factory methods
- [ ] Use `standard_config()` or `blitz_config()`
- [ ] Update tests

### Step 6: Verify and Test
- [ ] Run full test suite
- [ ] Test standard and blitz modes
- [ ] Test serial and parallel search
- [ ] Test replay functionality
- [ ] Play test games manually

### Step 7: Optional Modernization
- [ ] Consider adopting `ZertzAction` API for new code
- [ ] Consider using Rust game constants
- [ ] Update documentation

---

## Code Examples

### Before: MCTS Player Initialization
```python
# game/players/mcts_zertz_player.py (current)

def __init__(self, game, n, iterations=1000, ...):
    super().__init__(game, n)

    # ... parameter storage ...

    # Create Rust MCTS searcher (NO rings parameter)
    self.rust_mcts = hiivelabs_zertz_mcts.MCTSSearch(
        exploration_constant=exploration_constant,
        widening_constant=widening_constant,
        fpu_reduction=fpu_reduction,
        rave_constant=rave_constant,
        use_transposition_table=use_transposition_table,
        use_transposition_lookups=use_transposition_lookups,
    )

def _search(self):
    # Detect blitz mode at search time
    is_blitz = self._is_blitz_mode()

    # Pass to search method
    action_type, action_data = self.rust_mcts.search(
        spatial_state,
        global_state,
        rings=self.game.board.rings,  # Passed here
        blitz=is_blitz,  # Passed here
        ...
    )
```

### After: MCTS Player Initialization
```python
# game/players/mcts_zertz_player.py (updated)

from hiivelabs_mcts.zertz import ZertzMCTS

def __init__(self, game, n, iterations=1000, ...):
    super().__init__(game, n)

    # ... parameter storage ...

    # Detect blitz mode early
    is_blitz = self._is_blitz_mode()

    # Create Rust MCTS searcher with rings and blitz
    self.rust_mcts = ZertzMCTS(
        rings=game.board.rings,  # NEW: required
        exploration_constant=exploration_constant,
        widening_constant=widening_constant,
        fpu_reduction=fpu_reduction,
        rave_constant=rave_constant,
        use_transposition_table=use_transposition_table,
        use_transposition_lookups=use_transposition_lookups,
        blitz=is_blitz,  # NEW: set at construction
        t=getattr(game.board, 't', 1),  # NEW: time history
    )

def _search(self):
    # No need to pass rings/blitz to search anymore
    action_type, action_data = self.rust_mcts.search(
        spatial_state,
        global_state,
        iterations=self.iterations,
        ...
    )
```

---

## Benefits of Migration

### 1. Type Safety
- `ZertzAction` and `ZertzActionResult` provide compile-time safety
- Clearer API contracts

### 2. Multi-Game Support
- Modular structure supports TicTacToe and future games
- Shared MCTS infrastructure

### 3. Better Documentation
- Consolidated stub file with comprehensive docstrings
- Factory methods are self-documenting

### 4. Performance
- More efficient coordinate conversions
- Better memory layout for multi-game support

### 5. Maintainability
- Clear separation of concerns (games/ module structure)
- DRY principle applied (BoardConfig helper methods)
- Comprehensive test coverage added

---

## Breaking Changes Summary

| Change | Impact | Migration Effort |
|--------|--------|------------------|
| Module rename | High | Low (find/replace) |
| Class rename `MCTSSearch` → `ZertzMCTS` | High | Low (find/replace) |
| `rings` parameter required | High | Medium (add parameter) |
| `blitz` parameter at construction | Medium | Medium (move detection) |
| BoardConfig factory methods | Low | Low (replace calls) |
| ZertzAction API | None (optional) | High (full refactor) |

---

## Compatibility Matrix

| Feature | Current API | New API | Backward Compatible? |
|---------|-------------|---------|----------------------|
| Module import | `hiivelabs_zertz_mcts` | `hiivelabs_mcts.zertz` | ❌ No |
| MCTS class | `MCTSSearch` | `ZertzMCTS` | ❌ No |
| MCTS initialization | No `rings` param | Requires `rings` | ❌ No |
| BoardConfig | Direct construction | Factory methods | ⚠️ Maybe |
| Action tuples | `("PUT", ...)` | Still works | ✅ Yes |
| ZertzAction | N/A | New API | ✅ Optional |
| Stateless functions | Current signatures | Same signatures | ✅ Yes |
| Game constants | Python only | Also in Rust | ✅ Yes (both work) |

---

## Questions for Resolution

1. **Timing**: When should this migration be prioritized?
   - Recommendation: Before next major feature work

2. **ZertzAction adoption**: Full migration or keep tuple API?
   - Recommendation: Keep tuple API, adopt ZertzAction incrementally

3. **Testing strategy**: How much test coverage needed before merge?
   - Recommendation: Full test suite + manual playtesting

4. **Rollback plan**: How to handle issues post-merge?
   - Recommendation: Feature branch, thorough testing, staged merge

---

## Conclusion

The Rust wrapper changes represent a significant improvement in API design, type safety, and multi-game support. While the changes are breaking, they are well-designed with clear migration paths and backward compatibility where feasible.

**Recommended Action**: Proceed with migration in a dedicated feature branch with comprehensive testing.

**Estimated Effort**: 4-8 hours for implementation + testing

**Risk Level**: MODERATE - Breaking changes but clear migration path

---

## References

- Reference branch: `claude/review-rust-wrapper-exports-011CUoRo6T7s6P4C1W6dj3qi`
- Rust repo: `https://github.com/hiive/hiivelabs-zertz-mcts.git`
- Key commits: 50 commits including major refactoring
- Stub file: `hiivelabs_mcts.pyi` (1693 lines)

---

**Review completed**: 2025-11-05
**Reviewer**: Claude Code
**Status**: ✅ Ready for team review and planning
