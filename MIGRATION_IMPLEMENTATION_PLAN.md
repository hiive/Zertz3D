# Rust Wrapper Migration - Implementation Plan
## For `feat/mcts-trait-abstraction` Branch

**Branch**: `feat/mcts-trait-abstraction`
**Target**: Complete migration to `hiivelabs_mcts` API
**Status**: 90% Complete - Final Steps Required
**Estimated Time**: 1-2 hours

---

## Current Status

### ✅ Already Complete (90% Done!)

The `feat/mcts-trait-abstraction` branch has successfully migrated the Python codebase:

**Python Code Migration** (20 files):
- ✅ All imports updated: `hiivelabs_zertz_mcts` → `hiivelabs_mcts`
- ✅ MCTS class: `MCTSSearch` → `ZertzMCTS`
- ✅ Constructor: `rings` parameter added, `blitz` parameter added
- ✅ Search methods: `rings`, `blitz`, `num_threads` removed from search calls
- ✅ BoardConfig: Factory methods adopted (`standard_config`, `blitz_config`)
- ✅ Action masks: Updated to 5D placement masks (3, H, W, H, W)
- ✅ Canonicalization: All transforms updated
- ✅ Tests: All 13 test files updated

### ⚠️ Remaining Work (10% - Critical Blocker)

**Rust Submodule Mismatch**:
- ❌ Current: Rust exports `hiivelabs_zertz_mcts` (old name)
- ✅ Python imports: `hiivelabs_mcts` (new name)
- **Result**: `ModuleNotFoundError` when running any code

**Current Submodule**: `12537a6` (5 commits old)
**Target Commit**: `525e11c` (latest from reference branch)
**Commits Behind**: 50 commits with new API

---

## Implementation Steps

### Step 1: Update Rust Submodule ⚡ CRITICAL

**Objective**: Update rust submodule to match Python API expectations

**Commands**:
```bash
# Navigate to Zertz3D root
cd /home/user/Zertz3D

# Navigate to rust submodule
cd rust

# Fetch the reference branch
git fetch origin claude/review-rust-wrapper-exports-011CUoRo6T7s6P4C1W6dj3qi

# Checkout the latest commit from reference branch
git checkout 525e11c

# Verify we're at the right commit
git log --oneline -1

# Return to Zertz3D root
cd ..
```

**Expected Output**:
```
525e11c Export Zertz win condition constants to Python
```

**Time Estimate**: 5 minutes

---

### Step 2: Rebuild Rust Extension ⚡ CRITICAL

**Objective**: Compile the updated Rust code with new module name

**Commands**:
```bash
# From Zertz3D root
./rust-build.sh
```

**What to Watch For**:
- Build should complete without errors
- Output should show: `Built wheel for hiivelabs-zertz-mcts`
- Module will now export as `hiivelabs_mcts` (matching Python imports)

**If Build Fails**:
```bash
# Try development build first
./rust-dev.sh

# Check for specific errors in output
# Common issues:
# - Rust version too old: Update Rust with `rustup update`
# - Missing dependencies: Install via apt/brew
```

**Time Estimate**: 5-10 minutes (build time)

---

### Step 3: Verify Module Import 🧪

**Objective**: Confirm the module loads with the new name

**Commands**:
```bash
# Test import
python3 -c "import hiivelabs_mcts; print('✅ Module imported successfully')"

# Check available classes
python3 -c "from hiivelabs_mcts import ZertzMCTS, BoardConfig; print('✅ Classes available')"

# Verify ZertzMCTS constructor
python3 -c "from hiivelabs_mcts import ZertzMCTS; m = ZertzMCTS(rings=37); print('✅ ZertzMCTS works')"
```

**Expected Output**:
```
✅ Module imported successfully
✅ Classes available
✅ ZertzMCTS works
```

**If Import Fails with Submodule Error**:
```python
# You may need to use submodule imports
from hiivelabs_mcts.zertz import ZertzMCTS, BoardConfig
```

**Time Estimate**: 2-3 minutes

---

### Step 4: Check for Import Path Changes 🔍

**Objective**: Determine if imports need `zertz` submodule qualifier

**Test Script**:
```bash
# Test current import pattern (what we have now)
python3 << 'EOF'
try:
    from hiivelabs_mcts import ZertzMCTS, BoardConfig
    print("✅ Direct imports work - no changes needed")
except ImportError as e:
    print(f"❌ Direct imports failed: {e}")
    print("→ Need to add .zertz submodule")

    # Test submodule imports
    try:
        from hiivelabs_mcts.zertz import ZertzMCTS, BoardConfig
        print("✅ Submodule imports work")
    except ImportError as e2:
        print(f"❌ Submodule imports also failed: {e2}")
EOF
```

**Scenario A: Direct imports work** ✅
- No code changes needed
- Proceed to Step 5 (Testing)

**Scenario B: Need submodule imports** ⚠️
- Update imports in 20 files (see Step 4B)
- 15-30 minutes of work

**Time Estimate**: 2 minutes to check

---

### Step 4B: Update Import Paths (If Needed) 📝

**Only required if Step 4 shows submodule imports are needed**

**Files to Update** (5 core files + 13 test files):

**Core Files**:
1. `game/zertz_board.py:5`
2. `game/players/mcts_zertz_player.py:7`
3. `game/zertz_position.py:113`
4. `game/utils/canonicalization.py:57`
5. `learner/mcts/backend.py:9`

**Test Files** (13 files in `tests/`):
- `test_rave.py`, `test_mcts_player.py`, `test_crash_reproduction.py`
- And 10 more (see RUST_WRAPPER_REVIEW.md for full list)

**Change Pattern**:
```python
# OLD (current in feat/mcts-trait-abstraction)
from hiivelabs_mcts import (
    ZertzMCTS,
    BoardConfig,
    # ... other imports
)

# NEW (if submodule is required)
from hiivelabs_mcts.zertz import (
    ZertzMCTS,
    BoardConfig,
    BoardState,
    # ... other game-specific imports
)
from hiivelabs_mcts import (
    TransformFlags,  # Shared imports stay at root
    # ... other shared imports
)
```

**Quick Find/Replace**:
```bash
# Preview changes (dry run)
find . -name "*.py" -type f -exec grep -l "from hiivelabs_mcts import" {} \; | \
    grep -v __pycache__ | \
    head -20

# Use your editor to update each file
# Recommended: Update and test one file at a time
```

**Time Estimate**: 15-30 minutes (if needed)

---

### Step 5: Run Unit Tests 🧪

**Objective**: Verify all tests pass with updated Rust module

**Commands**:
```bash
# Test RAVE functionality (uses ZertzMCTS extensively)
pytest tests/test_rave.py -v

# Test MCTS player
pytest tests/test_mcts_player.py -v

# Test crash reproduction (regression test)
pytest tests/test_crash_reproduction.py -v

# Test canonicalization
pytest tests/test_mask_canonicalization.py -v

# Test action transformations
pytest tests/test_action_transformations.py -v
```

**Expected Results**:
- All tests should pass ✅
- No import errors
- No API signature mismatches

**If Tests Fail**:
- Check error messages for:
  - Import errors → Return to Step 4B
  - API signature mismatches → Check RUST_WRAPPER_REVIEW.md
  - Actual test failures → May need code fixes

**Time Estimate**: 5-10 minutes

---

### Step 6: Run Full Test Suite 🎯

**Objective**: Ensure no regressions across entire codebase

**Commands**:
```bash
# Run all tests
pytest -v

# Run without slow tests (quicker check)
pytest -v -m "not slow"
```

**Expected Results**:
- All tests pass or match pre-migration test status
- No new failures introduced

**Time Estimate**: 5-15 minutes

---

### Step 7: Integration Test - Game Playthrough 🎮

**Objective**: Verify MCTS player works end-to-end in real game

**Commands**:
```bash
# Quick 5-game test with headless mode
uv run main.py \
    --player1 mcts:iterations=100 \
    --player2 random \
    --games 5 \
    --headless

# With MCTS vs MCTS
uv run main.py \
    --player1 mcts:iterations=100 \
    --player2 mcts:iterations=100 \
    --games 3 \
    --headless \
    --stats
```

**Expected Results**:
- Games complete successfully
- No crashes or errors
- MCTS makes valid moves
- Statistics display correctly

**If Games Crash**:
- Check error messages
- Verify action formatting (PUT/CAP tuples)
- Check coordinate conversions

**Time Estimate**: 5 minutes

---

### Step 8: Commit Submodule Update 📦

**Objective**: Record the submodule update in git

**Commands**:
```bash
# From Zertz3D root
git status
# Should show: modified: rust (new commits)

# Stage the submodule update
git add rust

# Commit with clear message
git commit -m "Update rust submodule to hiivelabs_mcts API (50 commits from reference branch)

- Updated from 12537a6 to 525e11c
- Module now exports as hiivelabs_mcts (matches Python imports)
- Includes 50 commits of improvements:
  - Module reorganization
  - ZertzAction/ZertzActionResult classes
  - Game constants exports
  - Coordinate conversion DRY improvements
  - TicTacToe game support
  - Performance optimizations

Completes migration started in previous commits.
All Python code already updated to new API."

# Verify commit
git log --oneline -1
```

**Time Estimate**: 2 minutes

---

### Step 9: Update feat/mcts-trait-abstraction Branch 🚀

**Objective**: Push completed migration to remote

**Commands**:
```bash
# Push to feat/mcts-trait-abstraction
git push origin feat/mcts-trait-abstraction
```

**Note**: You may need to push to a claude/ branch first if you don't have direct push access to feat/mcts-trait-abstraction.

**Alternative** (if direct push fails):
```bash
# Create a claude branch
git checkout -b claude/complete-mcts-migration-011CUpCfEPP4Ac7uFbff1TEN

# Push to claude branch
git push -u origin claude/complete-mcts-migration-011CUpCfEPP4Ac7uFbff1TEN

# Then create a PR to merge into feat/mcts-trait-abstraction
```

**Time Estimate**: 2 minutes

---

## Quick Reference Commands

**Complete Migration in One Go** (if confident):
```bash
cd /home/user/Zertz3D

# Update submodule
cd rust && git fetch origin claude/review-rust-wrapper-exports-011CUoRo6T7s6P4C1W6dj3qi && \
git checkout 525e11c && cd ..

# Rebuild
./rust-build.sh

# Test import
python3 -c "from hiivelabs_mcts import ZertzMCTS; print('✅ Success')"

# Run tests
pytest tests/test_rave.py tests/test_mcts_player.py -v

# Commit
git add rust && git commit -m "Update rust submodule to hiivelabs_mcts API"

# Push
git push origin feat/mcts-trait-abstraction
```

---

## Troubleshooting

### Issue: Module import fails after rebuild

**Symptom**:
```
ModuleNotFoundError: No module named 'hiivelabs_mcts'
```

**Solutions**:
1. Check build output for errors: `./rust-build.sh 2>&1 | grep -i error`
2. Verify Python can find the module: `python3 -c "import sys; print(sys.path)"`
3. Try development build: `./rust-dev.sh`
4. Check if module was installed: `pip list | grep hiivelabs`

---

### Issue: Submodule imports required

**Symptom**:
```python
from hiivelabs_mcts import ZertzMCTS  # Fails
from hiivelabs_mcts.zertz import ZertzMCTS  # Works
```

**Solution**:
- Follow Step 4B to update import paths
- See RUST_WRAPPER_REVIEW.md for complete file list

---

### Issue: Tests fail with API mismatches

**Symptom**:
```
TypeError: search() got an unexpected keyword argument 'rings'
```

**Solution**:
- Check if test still passes `rings` to search method
- Verify MCTS instance was created with `rings` parameter
- See game/players/mcts_zertz_player.py:170-198 for correct pattern

---

### Issue: Build fails with Rust errors

**Symptom**:
```
error: could not compile `hiivelabs-zertz-mcts`
```

**Solutions**:
1. Update Rust: `rustup update`
2. Check Rust version: `rustc --version` (should be 1.70+)
3. Clean build: `cd rust && cargo clean && cd .. && ./rust-build.sh`
4. Check build logs for specific error

---

## Success Criteria

Migration is complete when:

- ✅ Rust submodule updated to commit `525e11c`
- ✅ Rust extension builds without errors
- ✅ Module imports as `hiivelabs_mcts`
- ✅ All unit tests pass
- ✅ Full test suite passes
- ✅ Integration test (game playthrough) succeeds
- ✅ Changes committed to git
- ✅ Changes pushed to remote branch

---

## Post-Migration Notes

### Optional Improvements (Future Work)

1. **Adopt ZertzAction API** (optional - tuple API still works):
   ```python
   from hiivelabs_mcts.zertz import ZertzAction, BoardConfig

   config = BoardConfig.standard_config(rings=37)
   action = ZertzAction.placement(config, marble_type=0, dst_y=3, dst_x=3)
   ```

2. **Use Rust Game Constants** (optional - Python constants still work):
   ```python
   from hiivelabs_mcts.zertz import (
       PLAYER_1_WIN, PLAYER_2_WIN, BOTH_LOSE,
       STANDARD_MARBLES, BLITZ_MARBLES,
   )
   ```

3. **Update Documentation**:
   - Update README.md with new API examples
   - Update development docs with new module structure

---

## Timeline Summary

| Step | Task | Time | Cumulative |
|------|------|------|------------|
| 1 | Update rust submodule | 5 min | 5 min |
| 2 | Rebuild Rust extension | 5-10 min | 10-15 min |
| 3 | Verify module import | 2-3 min | 12-18 min |
| 4 | Check import paths | 2 min | 14-20 min |
| 4B | Update imports (if needed) | 0-30 min | 14-50 min |
| 5 | Run unit tests | 5-10 min | 19-60 min |
| 6 | Run full test suite | 5-15 min | 24-75 min |
| 7 | Integration test | 5 min | 29-80 min |
| 8 | Commit changes | 2 min | 31-82 min |
| 9 | Push to remote | 2 min | 33-84 min |

**Total Time**: 30-90 minutes (depending on whether import path updates are needed)

---

## Contact / Issues

If you encounter issues not covered here:
- Review detailed analysis in `RUST_WRAPPER_REVIEW.md`
- Check reference branch: `claude/review-rust-wrapper-exports-011CUoRo6T7s6P4C1W6dj3qi`
- Review commit history for migration patterns

---

**Implementation Plan Version**: 1.0
**Date**: 2025-11-05
**Branch**: `feat/mcts-trait-abstraction`
**Status**: Ready to execute
