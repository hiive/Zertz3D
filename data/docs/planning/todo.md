# Zertz3D TODO and Improvement Suggestions

## Recently Completed (2025-10-22)

### ✅ Blitz Mode Python-Rust Parity
- **Critical Fix**: Restored parity between Python and Rust backends for blitz mode support
- **Problem**: Rust backend supported blitz mode, but Python wasn't passing the mode parameter
- **Changes**:
  - **Rust Backend** (`rust/src/`):
    - Added `blitz` parameter to `MCTSSearch.search()` and `MCTSSearch.search_parallel()` (mcts.rs:108,217)
    - Updated `BoardConfig::new()` to accept `blitz` parameter and use `BoardConfig::blitz()` when appropriate (board.rs:155-178)
    - Both methods now conditionally create `BoardConfig::blitz()` or `BoardConfig::standard()` based on parameter
  - **Python Interface** (`game/players/`):
    - Added `_is_blitz_mode()` helper method to detect mode from win conditions (mcts_zertz_player.py:141-145)
    - Updated `_rust_search()` to detect and pass `blitz=True/False` to Rust search functions (mcts_zertz_player.py:165,177)
    - Mode detection: compares `self.game.win_con` against `BLITZ_WIN_CONDITIONS` constant
- **Testing**:
  - Created comprehensive test suite: `tests/test_blitz_mode_parity.py` (12 tests)
  - Tests cover both Python and Rust backends
  - Verifies mode detection, MCTS search, and parallel search for both standard and blitz modes
  - All tests passing (Python: 4 tests, Rust: 6 tests, Constants: 2 tests)
- **Impact**: Critical parity requirement restored - both backends now correctly use blitz win conditions (2-of-each vs 3-of-each)
- **Files**: rust/src/mcts.rs, rust/src/board.rs, game/players/mcts_zertz_player.py, tests/test_blitz_mode_parity.py

## Recently Completed (2025-01-22)

### ✅ Rust MCTS Code Duplication Elimination
- **Refactoring**: Eliminated ~150 lines of duplicated game logic between rust/src/mcts.rs and rust/src/game.rs
- **Changes**:
  - Added `is_game_over()` and `get_game_outcome()` to game.rs as single source of truth (lines 915-1140)
  - Refactored `is_terminal_state()` in mcts.rs from ~80 lines to 3 lines (delegates to game::is_game_over)
  - Refactored `evaluate_terminal()` in mcts.rs from ~100 lines to ~30 lines (delegates to game::get_game_outcome)
  - Updated imports in mcts.rs to include new functions
- **Benefits**:
  - Single source of truth for game termination logic
  - Matches Python architecture (mcts_tree.py → stateless_logic.py)
  - Easier maintenance (rule changes only in one place)
  - Reduced risk of divergence between MCTS and game logic
- **Files**: rust/src/game.rs (added functions), rust/src/mcts.rs (refactored)
- **Testing**: All 727 tests pass, no behavioral changes
- **Documentation**: See data/docs/architecture/architecture_update_code_duplication_fix.md

## Feature Ideas
- Add game statistics tracking
- Network multiplayer support

## MCTS Enhancement Roadmap (from Architecture Report 9)

### Phase 1: Quick Wins (1 week, 9-13 hours) → 25-40% improvement
**Priority: HIGH** | **Start Date**: TBD

- [ ] **Virtual Loss for Parallel Search** (4-6 hours) - CRITICAL
  - Add `add_virtual_loss()` / `remove_virtual_loss()` / `update_with_real_value()` to `rust/src/node.rs`
  - Add virtual loss mechanism to `rust/src/mcts.rs` selection phase
  - Expected gain: 10-20% better parallel efficiency
  - Files: `rust/src/node.rs:116-147`, `rust/src/mcts.rs` parallel iteration
  - Status: Not started

- [ ] **First Play Urgency (FPU)** (2-3 hours)
  - Replace `f32::INFINITY` with `parent_value - fpu_reduction` in `rust/src/node.rs:152-164`
  - Add adaptive FPU parameter to MCTSSearch (default: 0.2)
  - Expected gain: 5-15% stronger play
  - Files: `rust/src/node.rs` ucb1_score method
  - Status: Not started

- [ ] **Adaptive Exploration Constant** (3-4 hours)
  - Add `adaptive_exploration_constant()` to `rust/src/mcts.rs`
  - Adjust based on: rings_remaining, num_legal_actions, parent_visits
  - Expected gain: 5-10% stronger play
  - Files: `rust/src/mcts.rs` search methods
  - Status: Not started

**Phase 1 ROI**: 9-13 hours → 25-40% stronger play

### Phase 2: RAVE/AMAF (2 weeks, 16-22 hours) → 20-35% improvement
**Priority: MEDIUM** | **Depends on**: Phase 1 completion

- [ ] **RAVE (Rapid Action Value Estimation)** (12-16 hours)
  - Add `rave_visits: AtomicU32`, `rave_total_value: AtomicI32` to `rust/src/node.rs`
  - Implement `rave_ucb_score()` with beta mixing parameter
  - Track actions during simulation for AMAF updates
  - Update backpropagation to update RAVE stats for all actions in path
  - Expected gain: 15-25% stronger play (high branching factor benefit)
  - Memory: +8 bytes per node
  - Files: `rust/src/node.rs`, `rust/src/mcts.rs` simulate/backpropagate
  - Status: Not started

- [ ] **Progressive History Heuristic** (4-6 hours)
  - Add global `HistoryTable` in `rust/src/mcts.rs`
  - Track success rates: `history[action] = (wins, total_tries)`
  - Use for move ordering and initial priors
  - Expected gain: 5-10% stronger play
  - Files: `rust/src/mcts.rs` (new HistoryTable struct)
  - Status: Not started

**Phase 2 ROI**: 16-22 hours → 20-35% stronger play

### Phase 3: Neural Network Integration (3-4 weeks, 40-60 hours) → 2-5× improvement
**Priority: LOW (Research Phase)** | **Depends on**: Self-play data collection

- [ ] **Policy/Value Network** (40-60 hours)
  - Design network architecture (ResNet-based for spatial, dense for global)
  - Implement self-play game generation
  - Train network on self-play data (PyTorch)
  - Replace random rollouts with learned evaluation
  - Add PUCT (predictor + UCT) formula with neural priors
  - Expected gain: 2-5× stronger play
  - Files: New `learner/neural/` directory, integration with MCTS
  - Status: Not started

**Phase 3 ROI**: 40-60 hours → 2-5× stronger play

### Total MCTS Investment
- **Quick Wins**: 9-13 hours → 25-40% improvement ⭐ Highest ROI
- **RAVE/AMAF**: 16-22 hours → 20-35% improvement
- **Neural Networks**: 40-60 hours → 2-5× improvement (long-term)
- **TOTAL**: 65-95 hours for state-of-the-art MCTS

## Backend Infrastructure (from Architecture Report 9.0)

### Completed ✅
- ✅ Dual Python/Rust backends with functional parity
- ✅ Zobrist hashing with PCG64 for both languages
- ✅ Canonicalization with translation + symmetry ordering
- ✅ Transposition tables with shared statistics
- ✅ Comprehensive parity tests (Python ↔ Rust)

### Remaining Tasks
- [ ] **Backend Selector Enhancement** (2-3 hours)
  - Formalize fallback logic: try Rust → fall back to Python
  - Add runtime performance comparison
  - Document backend selection criteria
  - Files: `learner/mcts/backend.py`

- [ ] **Performance Telemetry** (3-4 hours)
  - Add iteration throughput tracking per backend
  - Track per parallelization mode (serial/threaded/multiprocess)
  - Log statistics to structured format (JSON)
  - Files: `rust/src/mcts.rs`, `learner/mcts/mcts_tree.py`

## Testing Enhancements (from Architecture Report 9)

- [ ] **Parallel Correctness Tests** (4-6 hours)
  - Test determinism: serial vs parallel with same seed
  - Test statistical equivalence of search results
  - Test for race conditions in transposition table
  - Files: `tests/test_mcts_parallel.py` (new)

- [ ] **Performance Regression Tests** (2-3 hours)
  - Establish baseline benchmarks
  - 10% tolerance on iteration speed
  - Memory usage caps
  - Files: `tests/test_mcts_performance.py` (new)

- [ ] **Memory Leak Detection** (2-3 hours)
  - Use tracemalloc for Python
  - Valgrind/heaptrack for Rust
  - Test stability over 100+ searches
  - Files: `tests/test_mcts_memory.py` (new)

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

### ~~❌ Frozen Region Visual Effect~~ (REMOVED - based on rules misunderstanding)
- Originally implemented faded appearance for "frozen" isolated regions
- Feature removed after clarifying game rules - isolated regions with vacant rings are simply removed, not frozen
- Related code has been cleaned up from board and renderer

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

## Recently Completed (2025-10-13)

### ✅ Canonicalization & Transposition Alignment
- Rust backend now mirrors Python canonicalization ordering (translation + symmetry) using byte masks for lexicographic comparison
- Added shared canonicalization parity tests (Python ↔ Rust) targeting rotations, mirrors, and translations on all board sizes
- Zobrist hashing infrastructure ported to Rust with identical seed strategy (42) and feature coverage (rings, marbles, supply, captures, current player)
- Rust transposition table now hashes canonical states via Zobrist keys and shares statistics through `Arc<TranspositionEntry>`; Python remains reference implementation with stored canonical states for collision detection

### ✅ MCTS Backend Consistency Tasks
- Node statistics in both implementations now proxy through shared transposition entries, keeping visit/value counts aligned for symmetrical states
- Canonical states are exposed from the Rust backend (`BoardState.canonicalize_state`) for parity checks and future tooling
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
- Add unit tests for ActionVisualizationSequencer (highlight state machine)
- Add unit coverage for `CompositeRenderer.attach_update_loop` and clarify multi-renderer main loop contract
- Continue slimming `ZertzGameController` (e.g., extract post-action/logging helpers)
- Document status reporter/TextRenderer defaults and provide a way to silence text output for service deployments
- Allow `ZertzFactory` to opt out of the text renderer when running non-interactive builds

## Architecture Recommendations (from architecture_report9.md - 2025-10-21)

### MCTS Implementation - Phase 1: Quick Wins (1 week)
**Impact**: 25-40% improvement | **Effort**: Low-Medium

1. **Virtual Loss for Parallel Search** (4-6 hours) - CRITICAL for parallel efficiency
   - Add virtual loss mechanism to prevent thread collision
   - Expected gain: 10-20% better parallel efficiency
   - Implementation: `rust/src/node.rs` add `add_virtual_loss()` / `remove_virtual_loss()`
   - **Priority**: HIGH | **Complexity**: Low

2. **First Play Urgency (FPU)** (2-3 hours) - Better unvisited node handling
   - Replace `f32::INFINITY` with estimated value (parent_value - 0.2)
   - Expected gain: 5-15% stronger play
   - Implementation: `rust/src/node.rs` modify `ucb1_score()`
   - **Priority**: HIGH | **Complexity**: Low

3. **Adaptive Exploration Constant** (3-4 hours) - Phase-aware exploration
   - Dynamically adjust c based on game phase (rings remaining, branching factor)
   - Expected gain: 5-10% stronger play
   - Implementation: `rust/src/mcts.rs` add `adaptive_exploration_constant()`
   - **Priority**: MEDIUM | **Complexity**: Low

**Phase 1 Total**: 9-13 hours → 25-40% improvement

### MCTS Implementation - Phase 2: RAVE/AMAF (2 weeks)
**Impact**: 15-25% improvement | **Effort**: Medium

4. **RAVE (Rapid Action Value Estimation)** (12-16 hours)
   - Add AMAF statistics to nodes (rave_visits, rave_value)
   - Implement RAVE-UCB scoring formula
   - Update backpropagation to track all actions in playout
   - Expected gain: 15-25% stronger play (high branching factor benefit)
   - Implementation:
     - `rust/src/node.rs`: add RAVE fields, `rave_ucb_score()`
     - `rust/src/mcts.rs`: track actions during simulation
   - **Priority**: HIGH | **Complexity**: Medium
   - **Memory**: +8 bytes per node

5. **Progressive History Heuristic** (4-6 hours)
   - Global move success tracking across tree
   - Warm-start priors for move ordering
   - Expected gain: 5-10% stronger play
   - Implementation: Add `HistoryTable` in `rust/src/mcts.rs`
   - **Priority**: MEDIUM | **Complexity**: Low

**Phase 2 Total**: 16-22 hours → 20-35% improvement

### MCTS Implementation - Phase 3: Neural Network Integration (3-4 weeks)
**Impact**: 2-5× improvement | **Effort**: High

6. **Policy/Value Network** (40-60 hours)
   - Train neural network on self-play games
   - Replace random rollouts with learned evaluation
   - Add PUCT (predictor + UCT) formula
   - Expected gain: 2-5× stronger play
   - Technologies: PyTorch, AlphaZero-style architecture
   - **Priority**: LOW (research phase) | **Complexity**: Very High

### Controller/Renderer Architecture (from architecture_report8.md)

**High Priority** (Weeks 1-3) - 12-16 hours total

1. **Variant Configuration Object** (1-2 hours)
   - Create `VariantConfig` dataclass with marble counts, rules, constraints
   - **Impact**: Medium | **Effort**: Low

2. **Extract HoverFeedbackCoordinator** (4-6 hours)
   - Move hover feedback logic (~400 lines) from controller to dedicated class
   - **Impact**: High | **Effort**: Medium

3. **Refactor update_game() Method** (4-6 hours)
   - Break down into `_get_player_action()`, `_execute_action()`, etc.
   - **Impact**: Medium | **Effort**: Medium

**Medium Priority** (Week 4+) - 11-15 hours

4. **Player Factory Pattern** (2-3 hours)
5. **Test Fixture Builders** (3-4 hours)
6. **Placement State Machine** (6-8 hours)

**Low Priority** (Future) - 10-14 hours

7. **Command Pattern for Actions** (4-6 hours)
8. **Event System** (6-8 hours)

### Total Investment Summary

**MCTS Enhancements**:
- Phase 1 (Quick Wins): 9-13 hours → 25-40% improvement
- Phase 2 (RAVE): 16-22 hours → 20-35% improvement
- Phase 3 (Neural Network): 40-60 hours → 2-5× improvement
- **MCTS Total**: 65-95 hours for state-of-the-art MCTS

**Controller/Renderer**:
- High Priority: 12-16 hours
- Medium Priority: 11-15 hours
- Low Priority: 10-14 hours
- **Architecture Total**: 33-45 hours

**GRAND TOTAL**: 98-140 hours for all improvements

## Recently Completed (2025-10-15)

### ✅ Detailed Game End Reasons
- Added `get_game_end_reason()` method to `ZertzGame` class
- Displays specific reasons for game ending:
  - "Captured required marbles: 4 white" (or 5 gray, 6 black, 3 of each)
  - "Both players immobilized: 4 white" (with winning condition)
  - "Both players immobilized with no winner" (tie)
  - "Move loop detected (repeated position)" (tie)
  - "Board completely filled with marbles"
  - "Opponent has no marbles left to place"
- Updated `ZertzGameController._handle_game_ending()` to display reasons
- Comprehensive test suite in `test_game_end_reasons.py` (12 test cases)

### ✅ Capture Flash Animation
- Captured marbles now flash yellow before animating to capture pool
- Works for both direct captures (CAP actions) and isolation captures
- Added `CAPTURE_FLASH_MATERIAL_MOD` with bright yellow highlight (0.9, 0.9, 0.1) and strong glow
- Flash duration: 0.3 seconds before capture animation starts
- Implementation leverages existing `highlighting_manager` infrastructure
- Uses marble's captured key ("captured:id") for proper entity tracking

## Recently Completed (2025-10-15 - Continued)

### ✅ MaterialManager Extraction
- Created `MaterialManager` class in `renderer/panda3d/material_manager.py`
- Centralized all Panda3D material operations:
  - `save_material(entity)` - Save current material properties
  - `restore_material(entity, saved_material)` - Restore saved material
  - `apply_material(entity, material_mod, metallic, roughness)` - Apply MaterialModifier to entity
  - `create_blended_material(...)` - Create blended material for pulsing animations
  - `get_model_from_entity(entity)` - Handle both NodePath and .model objects
- Updated `HighlightingManager` to use MaterialManager:
  - Removed duplicated material creation/application code (4+ places)
  - Removed manual Material object construction
  - Cleaner imports (removed unused LVector4 and Material)
- Benefits:
  - Eliminated ~60 lines of duplicated material handling code
  - Single source of truth for material operations
  - Easier to extend with new material effects
  - Better separation of concerns (highlighting logic vs Panda3D materials)

### ✅ Marble Color Architectural Refactoring
- **Problem**: Marble color was being passed redundantly to every `configure_as_*` method
- **Solution**: Store color as readonly property `zertz_color` in marble object at creation
- **Changes**:
  - Modified `make_marble()` to pass color to marble constructors
  - Updated `_BallBase.__init__()` to accept and store `marble_color` as `self.zertz_color`
  - Removed `color` parameter from `configure_as_supply_marble()`, `configure_as_board_marble()`, `configure_as_captured_marble()`
  - Updated all callsites in `panda_renderer.py` (6 locations)
- **Benefits**:
  - Single source of truth: color set once at creation, never changes
  - Better encapsulation: marble objects are self-contained
  - Cleaner API: no redundant parameters
  - Type safety: color always available as direct property

### ✅ Centralized Logging/Reporting System
- **Architecture**: Implemented GameLogger as sole hub for all text output (hub-and-spoke pattern)
- **Key Changes**:
  - `_report()` method now calls ONLY `self.logger.log_comment(text)` (single responsibility)
  - Logger routes messages to appropriate writers (TranscriptWriter, NotationWriter)
  - Created logger early in initialization (before session creation)
  - Removed duplicate board state printing (now only in `write_footer()`)
- **Writer Architecture**:
  - TranscriptWriter: verbose output with comments (status messages prefixed with `#`)
  - NotationWriter: algebraic notation only (ignores comments via no-op)
  - Both writers implement `write_comment()` for consistent interface
- **Output Modes Fixed**:
  - Headless + file-only: no stdout pollution ✓
  - `--notation-screen`: clean notation only ✓
  - `--transcript-screen`: full output with comments ✓
  - Both screen flags: no duplication ✓

### ✅ Notation File Replay Support
- **Format Auto-Detection**: System detects notation vs transcript format automatically
- **Notation Parsing**:
  - `NotationFormatter.notation_to_action_dict()` parses official Zèrtz notation
  - Handles lowercase coordinates from notation files (e.g., `Gb2,a4`)
  - Converts coordinates to uppercase for game internals (e.g., `B2`, `A4`)
  - Supports all action types: placement (`Wd4`), placement with removal (`Bd7,b2`), capture (`x e3Wg3`), pass (`-`)
- **Testing**: Verified with notation replay in both headless and graphical modes
- **Files Modified**:
  - `game/formatters/notation_formatter.py`: Added uppercase conversion in `_parse_placement()` and `_parse_capture()`

### ✅ Animation Sequencing Improvements
- **Capture Flash Timing**: Yellow flash now conditional on `--highlight-choices` flag
  - Without flag: no flash, captured marble moves immediately after capturing marble lands
  - With flag: marble flashes yellow, then moves to capture pool
  - Flash starts when capturing marble lands (`defer=action_duration`)
- **Capture Animation Sequencing**: Fixed captured marble timing
  - Captured marble now waits for capturing marble to complete jump before moving
  - Base defer time: `action_duration` (capturing marble's jump time)
  - With highlights: additional `CAPTURE_FLASH_DURATION` (0.3s) for flash effect
  - Animation sequence: capturing jump → flash (if enabled) → captured marble moves to pool
- **Code Location**: `panda_renderer.py:_animate_marble_to_capture_pool()` lines 558-590

### ✅ Architecture Report 8
- **Comprehensive Analysis**: 55,000+ word architectural review
- **Overall Score**: 9/10 - Excellent software architecture
- **Report Location**: `/Users/andrewrollings/Dropbox/Hiive/Hiive Games/Zertz3D/architecture_report8.md`
- **Key Findings**:
  - Clean three-tier architecture with no circular dependencies
  - Recent improvements (ActionResult, RenderData, GameLogger, Writers) are high quality
  - Excellent extensibility for board sizes (9/10), output formats (10/10), AI players (9/10)
  - Low technical debt, minimal duplication, well-tested
- **Recommendations Identified**: See "Architecture Recommendations" section below

## README Missing or Incomplete Sections:

  1. Python Version Requirements - Installation section doesn't specify required Python version
  2. Programmatic/API Usage - README only shows command-line usage, but doesn't show how to use the game as a library (creating games, accessing state, implementing custom players)
  3. ML Integration Examples - While state representation is documented, there's no example code showing how to actually use it for ML training
  4. Captured Marble Rules - Game Mechanics doesn't explain that players MUST use their captured marble pool when the general supply is empty (important game rule)
  5. Action Space Details - The action system mentions formats but doesn't clearly document the numpy array shapes for placement (3 × width² × (width²+1)) and capture (6 × width × width) actions
  6. Loop Detection Mechanics - Win Conditions mentions "loop detection" but doesn't explain the k=2 move-pairs mechanism
  7. Immobilization/Auto-Pass - Doesn't explain that players automatically pass when they have no valid moves
  8. License - Currently a placeholder "[Add license information]"
  9. Contributing Guidelines - Currently a placeholder "[Add contribution guidelines]"
  10. Troubleshooting/FAQ - No section for common issues

### Priority
  - Programmatic Usage section with code examples
  - ML Integration Examples
  - Python version requirements
  - Complete game mechanics details (captured marbles, loop detection, immobilization)

## Andrew's notes (DO NOT DELETE)
- Add unit tests for tagging system.
- Allow board to be rotated around its geometric center round an axis perpendicular to the board.
- ✅ Add "how the player won" (or how they drew) to the end of the standard text output.
- Extract out entities from renderer (marbles and rings with a base). (WON'T DO - TOOK DIFFERENT APPROACH)
- ✅ Extract out SelectionHandler class from renderer - COMPLETED: InteractionHelper class in renderer/panda3d/interaction_helper.py handles all mouse picking, collision detection, hover state, and selection callbacks
- ✅ Extract out "Materials" using for various forms of highlighting (base color, emission color) - COMPLETED: MaterialModifier dataclass in renderer/panda3d/material_modifier.py, material constants in shared/materials_modifiers.py, MaterialManager class in renderer/panda3d/material_manager.py handles all Panda3D material operations
- ✅ Extract highlight state machine into its own module - COMPLETED: ActionVisualizationSequencer class in renderer/panda3d/action_sequencer.py handles all multi-phase highlighting sequences (tests still TODO)
- Extract out game configurator (e.g. number of rings/marbles, win conditions, etc.)
- Look for examples of technical debt.
- Edge ring stuff - simpler to add all rings to collection and just check that rather that just add edge rings and if switch.
- Check if get_all_permutations also includes all possible translations? (Even if they are eliminated via grouping.)
- investigate seed 1760840136
- Player classes should use the PLAYER_x constants in their __init__
- ✅ for MCTS player, optimization is when single capture move is only option do it without checking tree.
- unit test to verify MCTS is actually being the right player at each level.
- Tree visualizer.
- Need visual details in graphical mode (which player and what they are thinking)
- ✅ What happens if no edge rings can be removed after placement? How is that notated?
- ✅ Check tournament rule (BOTH_LOSE when board full with zero captures)
- ✅ Verify loop detection (is it possible?!) and consecutive passing.
- Add flag for player moves like --show-coords