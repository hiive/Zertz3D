# MCTS Hyperparameter Tuning

This directory contains tools for tuning MCTS hyperparameters to optimize performance against random play.

## Quick Start

For a fast test run:

```bash
./scripts/analysis/quick_tune.sh
```

This runs a grid search with 10 games per configuration (should complete in a few minutes).

## Manual Usage

### Random Search (Recommended - more efficient)

Random search is typically more effective than grid search because it:
- Samples more unique values for important parameters
- Better explores continuous parameter spaces
- Finds good solutions faster with the same budget

(See: Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization", 2012)

```bash
uv run python scripts/analysis/tune_mcts_hyperparams.py \
    --method random \
    --games 20 \
    --iterations 1500 \
    --backend rust \
    --samples 30 \
    --seed 42
```

**Random search parameters:**
- `exploration_constant`: uniform[0.3, 3.0]
- `fpu_reduction`: None or uniform[0.0, 0.7]
- `max_simulation_depth`: None (full game)

## Arguments

- `--method {grid,random}`: Search method (default: grid)
- `--games N`: Games per configuration (default: 20)
- `--iterations N`: MCTS iterations per move (default: 1500, **FIXED**)
- `--backend {python,rust}`: MCTS backend (default: rust)
- `--samples N`: Number of random samples for random search (default: 20)
- `--seed N`: RNG seed for reproducibility (default: 42)
- `--output FILE`: Output JSON file (default: data/tuning_results.json)
- `--top N`: Number of top configs to display (default: 5)
- `--quiet`: Suppress progress output

## Output

The script outputs:
1. **Progress updates** during search (unless --quiet)
2. **Top N configurations** ranked by win rate
3. **JSON file** with full results at specified path

### Example Output

```
TOP 5 CONFIGURATIONS (by win rate)

#1: Win rate: 85.0% (17W/3L/0T)
    Exploration constant: 1.410
    FPU reduction: 0.200
    Max depth: Full game
    Avg time/game: 0.298s

#2: Win rate: 80.0% (16W/4L/0T)
    Exploration constant: 2.000
    FPU reduction: 0.300
    Max depth: Full game
    Avg time/game: 0.305s
...
```

## Interpreting Results

- **Win rate**: Higher is better (target: >85% vs random)
- **Avg time/game**: Lower is faster (but don't sacrifice win rate)
- **Exploration constant**:
  - Lower (0.5-1.0): More exploitation, less exploration
  - Higher (2.0-3.0): More exploration, less exploitation
  - √2 ≈ 1.41 is theoretical optimum for UCB1
- **FPU reduction**:
  - None: Standard UCB1 (unvisited nodes have infinite score)
  - 0.1-0.5: Reduces optimism for unvisited nodes
  - Higher values = more pessimistic about unexplored moves

## Recommended Workflow

1. **Quick test** (3-5 minutes):
   ```bash
   ./scripts/analysis/quick_tune.sh
   ```

2. **Full grid search** (15-30 minutes):
   ```bash
   uv run python scripts/analysis/tune_mcts_hyperparams.py \
       --method grid --games 30 --backend rust
   ```

3. **Refine with random search** around best grid result:
   - Adjust ranges in the script based on grid results
   - Run random search with more samples

4. **Validate top configs** (100+ games):
   ```bash
   uv run python scripts/analysis/backend_parity_check.py \
       --games 100 --iterations 1500
   ```

## Tips

- **Start small**: Use 10 games per config for initial exploration
- **Increase games**: Use 30-50 games for final validation
- **Fix iterations**: Keep at 1500 for fair comparison
- **Use same seed**: For reproducible comparisons
- **Check both players**: The script alternates MCTS between P1 and P2
- **Compare backends**: Test final params on both Python and Rust

## Current Baseline

Before tuning:
- **Default params**: exploration=1.41, fpu=None, depth=None
- **Win rate vs random**: ~70% (1500 iterations)

Goal: Find params that achieve >85% win rate vs random.