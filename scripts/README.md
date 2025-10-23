# Script Organization

All utility scripts have been organized into subdirectories under `scripts/` for better project structure.

## Directory Structure

### Training Scripts (`scripts/training/`)
Scripts for training MCTS agents:
- **`train_mcts.py`** - Single-process MCTS training with persistent transposition table
- **`train_mcts_parallel.py`** - Parallel MCTS training with periodic table merging
- **`train_mcts_with_eval.py`** - MCTS training with periodic evaluation tournaments
- **`train_mcts_parallel_with_eval.py`** - Parallel training combined with evaluation

### Visualization Scripts (`scripts/visualization/`)
Scripts for rendering and displaying game data:
- **`render_notation.py`** - Render games from notation files as images
- **`view_elo_history.py`** - Display ELO rating history charts
- **`view_eval_history.py`** - Display evaluation history charts
- **`visualize_board_coords.py`** - Visualize board coordinate system with geometric center

### Analysis/Testing Scripts (`scripts/analysis/`)
Scripts for analyzing game behavior and testing:
- **`analyze_canonicalization.py`** - Analyze board state canonicalization from notation files
- **`backend_parity_check.py`** - Compare Python vs Rust MCTS backend performance
- **`test_single_game.py`** - Quick test to run a single game with MCTS players

### Utilities (`scripts/utilities/`)
General utility scripts:
- **`tournament.py`** - Run tournaments between different player types

## Main Entry Points (Root Directory)
- **`main.py`** - Main game entry point with graphical interface
- **`scratch.py`** - Scratch/testing file (not for production use)