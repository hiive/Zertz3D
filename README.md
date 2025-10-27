# Zèrtz 3D

A sophisticated 3D implementation of the abstract board game Zèrtz using Panda3D, featuring a high-performance Rust-accelerated MCTS AI engine with advanced search techniques (RAVE, transposition tables, parallel search). This project combines clean layered architecture with state-of-the-art game AI to create both a playable game and a research platform.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Full game rules: http://www.gipf.com/zertz/rules/rules.html

**THIS IS A WORK-IN-PROGRESS!**

## Features

### Core Gameplay
- **Multiple Board Sizes**: Play on 37, 48, or 61 ring boards
- **Game Variants**: Standard and Blitz modes with different win conditions
- **Official Notation**: Full support for official Zèrtz notation format
- **Deterministic Gameplay**: Seeded random number generation for reproducible games
- **Replay System**: Record and replay games in transcript or notation format

### AI & Performance
- **Advanced MCTS**: Monte Carlo Tree Search with RAVE, progressive widening, and FPU
- **Rust Acceleration**: High-performance game logic and MCTS engine via PyO3
- **Parallel Search**: Multi-threaded MCTS with virtual loss for lock-free parallelism
- **Transposition Tables**: Position caching with Zobrist hashing
- **Flexible Player Configuration**: Configure AI parameters via command-line

### Visualization & UI
- **3D Rendering**: Panda3D-based visualization with water reflections and dynamic lighting
- **Move Highlighting**: Visual feedback showing valid moves and AI move preferences
- **Human Play Mode**: Interactive mouse-based gameplay
- **Headless Mode**: Run games without rendering for batch simulations
- **Text Rendering**: Console-based output for terminal-only environments

### Development & Testing
- **Comprehensive Tests**: 34 test files covering game logic, edge cases, and integrations
- **Clean Architecture**: Layered design with clear separation of concerns
- **Python-Rust Parity**: Identical game logic in both languages for reliability
- **Statistics Tracking**: Detailed performance metrics and win/loss tracking

## Installation

### Prerequisites

- Python 3.12+
- Rust and Cargo (for Rust acceleration)
- `uv` package manager

### Install Python Dependencies

```bash
uv sync
```

### Build Rust Extension (Recommended)

The Rust extension provides significant performance improvements for MCTS:

```bash
# Development build (with hot reload)
./rust-dev.sh

# Or manual release build
cd rust && uv run python -m maturin develop --release
```

## Usage

### Quick Start

```bash
# Random vs Random (default)
uv run main.py

# Human vs Random
uv run main.py --player1 human

# MCTS vs MCTS
uv run main.py --player1 mcts --player2 mcts

# Strong MCTS with RAVE
uv run main.py --player1 mcts:iterations=5000,rave=1000,parallel=1
```

### Player Configuration

Configure each player using the `--player1` and `--player2` flags with the format:

```
TYPE[:PARAM=VALUE,PARAM=VALUE,...]
```

**Player Types:**
- `random` - Random move selection (default)
- `human` - Manual control via mouse (requires renderer, cannot be used with --headless)
- `mcts` - Monte Carlo Tree Search AI

**MCTS Parameters:**

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `iterations` | MCTS iterations per move | 1000 | 500, 5000, 10000 |
| `exploration` | UCB1 exploration constant | 1.41 | 1.0, 2.0 |
| `fpu` | First Play Urgency reduction | None | 0.2, 0.5 |
| `widening` | Progressive widening constant | None | 10.0, 20.0 |
| `rave` | RAVE constant (300-3000) | None | 1000, 2000 |
| `parallel` | Enable parallel search | 0 | 1 |
| `workers` | Number of worker threads | 16 | 4, 8, 32 |
| `verbose` | Print search statistics | 0 | 1 |
| `seed` | Random seed for this player | None | 12345 |

**MCTS Examples:**

```bash
# Basic MCTS with 1000 iterations (default)
uv run main.py --player1 mcts

# Stronger MCTS with 5000 iterations
uv run main.py --player1 mcts:iterations=5000

# MCTS with RAVE (Rapid Action Value Estimation)
uv run main.py --player1 mcts:iterations=2000,rave=1000

# Parallel MCTS with 8 threads
uv run main.py --player1 mcts:iterations=5000,parallel=1,workers=8

# Advanced MCTS with multiple features
uv run main.py --player1 mcts:iterations=10000,exploration=2.0,rave=1000,fpu=0.2,widening=10,parallel=1,workers=16

# Verbose mode for debugging
uv run main.py --player1 mcts:iterations=1000,verbose=1
```

### Command Line Options

**Board Configuration:**
```bash
uv run main.py --rings 61                    # Board size: 37, 48, or 61 rings (default: 37)
uv run main.py --seed 1234567890             # Random seed for reproducible games
uv run main.py --blitz                       # Blitz variant (37 rings only, fewer marbles)
```

**Game Control:**
```bash
uv run main.py --games 10                    # Number of games to play (default: infinite)
uv run main.py --headless                    # Run without 3D renderer (faster)
uv run main.py --move-duration 0.5           # Duration between moves in seconds (default: 0.5)
uv run main.py --start-delay 2.0             # Delay before first move in seconds (default: 0)
uv run main.py --stats                       # Print timing and win/loss stats
```

**Replay System:**
```bash
uv run main.py --replay path/to/file.txt     # Replay from transcript or notation file
uv run main.py --replay file.txt --partial   # Continue with random play after replay ends
```

**Logging Options:**
```bash
# Transcript format (Python dictionary)
uv run main.py --transcript-file             # Log to zertzlog_<seed>.txt (current dir)
uv run main.py --transcript-file ./logs      # Log to zertzlog_<seed>.txt in ./logs
uv run main.py --transcript-screen           # Output transcript format to screen

# Official Zèrtz notation
uv run main.py --notation-file               # Log official notation to file (current dir)
uv run main.py --notation-file ./logs        # Log official notation to ./logs
uv run main.py --notation-screen             # Output official notation to screen

# Use multiple logging formats simultaneously
uv run main.py --transcript-file --notation-file
uv run main.py --transcript-screen --notation-screen
```

**Visual Options (3D renderer only):**
```bash
uv run main.py --highlight-choices uniform   # Highlight all valid moves equally
uv run main.py --highlight-choices heatmap   # Highlight by AI move preference
uv run main.py --show-coords                 # Display coordinate labels on rings
```

### Common Use Cases

**Tournament Simulations:**
```bash
# Run 100 games between two MCTS players, headless mode, with statistics
uv run main.py --player1 mcts:iterations=5000 --player2 mcts:iterations=5000 \
  --headless --games 100 --stats --notation-file ./tournament_logs
```

**Interactive Play:**
```bash
# Human vs MCTS with move highlighting
uv run main.py --player1 human --player2 mcts:iterations=2000 \
  --highlight-choices heatmap --show-coords
```

**Performance Testing:**
```bash
# Strong parallel MCTS in headless mode
uv run main.py --player1 mcts:iterations=10000,parallel=1,workers=16 \
  --player2 mcts:iterations=10000,parallel=1,workers=16 \
  --headless --games 10 --stats
```

**Debugging & Analysis:**
```bash
# Verbose MCTS with full logging
uv run main.py --player1 mcts:iterations=1000,verbose=1 \
  --transcript-screen --notation-screen --highlight-choices heatmap
```

## Project Structure

```
Zertz3D/
├── main.py                         # Entry point and CLI
├── factory/                        # Wiring for controllers/renderers
│   └── zertz_factory.py            # Creates configured ZertzGameController
├── controller/                     # Game flow control
│   ├── zertz_game_controller.py    # Main game loop coordinator
│   ├── game_session.py             # Session management
│   ├── game_loop.py                # Turn scheduling & pacing
│   ├── game_logger.py              # Transcript/notation logging
│   └── ...                         # Action processing & human input glue
├── game/                           # Core game logic
│   ├── zertz_board.py              # Board state and move validation
│   ├── zertz_game.py               # Rule engine and state management
│   ├── zertz_player.py             # Base player interface
│   ├── players/                    # Player implementations (MCTS, random, replay)
│   ├── loaders/                    # Replay file loaders
│   ├── formatters/                 # Output formatters
│   ├── writers.py                  # Log file writers
│   └── utils/                      # Diagram rendering & helpers
├── learner/                        # Machine learning components
│   ├── mcts/                       # Python MCTS implementation
│   └── backend.py                  # Backend detection (Python vs Rust)
├── renderer/                       # Rendering backends
│   ├── panda_renderer.py           # Panda3D renderer entry point
│   ├── text_renderer.py            # Console/text renderer
│   ├── composite_renderer.py       # Fan-out to multiple renderers
│   └── panda3d/                    # Scene graph helpers, shaders, models
├── shared/                         # Shared DTOs and constants
├── tests/                          # Pytest suite covering rules and integrations
├── rust/                           # hiivelabs_zertz_mcts Rust extension
│   ├── Cargo.toml                  # Rust package manifest
│   └── src/                        # Rust MCTS implementation
├── data/                           # Game data, documentation, logs
├── models/                         # Saved ML checkpoints and assets
├── train_mcts*.py                  # Training and evaluation scripts
└── README.md
```

## Game Mechanics

### Board Representation

The game maintains two separate state representations:

**1. Spatial State (3D array: L × H × W)**
- Layer 0: Ring positions (1 = ring exists, 0 = removed)
- Layers 1-3: Marble positions (white, gray, black)
- Layers 4+: Historical board states (for time t timesteps)
- Last layer: Capture flag (marks marble that must capture)

**2. Global State (1D array: 10 elements)**
- Indices 0-2: Supply pool counts (white, gray, black marbles available)
- Indices 3-5: Player 1 captured marbles (white, gray, black)
- Indices 6-8: Player 2 captured marbles (white, gray, black)
- Index 9: Current player (0 = Player 1, 1 = Player 2)

For ML applications, use `ZertzGame.get_current_state()` which returns both spatial and global state in a dictionary format for complete observability.

### State Canonicalization

The board implements symmetry-aware state canonicalization to reduce the state space for machine learning:

**Symmetry Groups:**
- **37 and 61-ring boards**: D6 dihedral symmetry (18 transforms total)
  - 6 rotations (0°, 60°, 120°, 180°, 240°, 300°)
  - 12 mirror combinations (rotate-then-mirror, mirror-then-rotate)
- **48-ring boards**: D3 dihedral symmetry (9 transforms total)
  - 3 rotations (0°, 120°, 240°)
  - 6 mirror combinations

**Transform Notation:**
- `R{k}`: Pure rotation by k degrees
- `MR{k}`: Rotate by k degrees, then mirror
- `R{k}M`: Mirror, then rotate by k degrees

**Usage:**
```python
canonical_state, transform_name, inverse_transform = board.canonicalize_state()
```

The canonicalization returns:
- The lexicographically smallest equivalent state
- The transform applied to reach canonical form
- The inverse transform (to map policy outputs back to original orientation)

This is particularly useful for neural network training to ensure consistent state representation across rotationally/reflectively equivalent board positions.

### Action System

Two types of actions:
- **Placement (PUT)**: Place a marble and remove a ring
  - Format: `PUT {marble} {dst} {remove}`
  - Example: `PUT g B5 G3`
  - Official notation: `Gb5` or `Gb5,g3` (with ring removal)

- **Capture (CAP)**: Jump over an opponent's marble
  - Format: `CAP {marble} {src} {capture} {dst}`
  - Example: `CAP b C4 g D5 E6`
  - Official notation: `x c4Ge6` (jump from c4 over Gray to e6)

- **Pass (PASS)**: Used when no valid moves available
  - Official notation: `-`

The game now outputs both internal format and official Zèrtz notation (from http://www.gipf.com/zertz/notations/notation.html) for each move.

### Notation Log Files

Use the `--notation-file` flag to create a log file containing moves in official notation format:
- File format: `zertzlog_{seed}_notation.txt` or `zertzlog_blitz_{seed}_notation.txt`
- First line: board size and variant (e.g., "37" or "37 Blitz")
- Subsequent lines: one move per line in official notation
- Supports isolation notation: `Bd7,b2 x Wa1Wa2` (placement that isolates marbles)
- Can be used with `--transcript-file` to generate both formats simultaneously

Similarly, use `--transcript-file` to create dictionary format logs in `zertzlog_{seed}.txt`.

### Win Conditions

A player wins by capturing:
- 3 of each color (3 white, 3 gray, 3 black), OR
- 4 white, OR
- 5 gray, OR
- 6 black

Additional end conditions:
- Last player to place when board is full
- Opponent has no marbles to play
- Loop detection (repeated move patterns result in tie)

## Game Variants

### Standard Mode
The default game mode with standard marble counts and win conditions as described above.

### Blitz Mode
A faster, more aggressive variant designed for quick, tactical play. Enable with `--blitz`.

**Marble counts:**
- 5 white (vs 6 in standard)
- 7 gray (vs 8 in standard)
- 9 black (vs 10 in standard)

**Win conditions:**
- 2 of each color (2 white, 2 gray, 2 black), OR
- 3 white, OR
- 4 gray, OR
- 5 black

**Requirements:**
- Only works with 37-ring boards
- Games are shorter and more unforgiving

## Board Sizes

- **37 rings**: 7×7 hex grid (standard)
- **48 rings**: 8×8 hex grid
- **61 rings**: 9×9 hex grid (uses ABCDEFGHJ coordinate scheme, skipping 'I')

## Replay System

Replay files contain action dictionaries:

```
Player 1: {'action': 'PUT', 'marble': 'g', 'dst': 'B5', 'remove': 'G3'}
Player 2: {'action': 'CAP', 'marble': 'b', 'src': 'C4', 'dst': 'E6', 'capture': 'g', 'cap': 'D5'}
Player 1: {'action': 'PASS'}
```

Board size is automatically detected from coordinates in the replay file.

## Development

### Building the Rust Extension

The Rust extension (`hiivelabs_zertz_mcts`) provides significant performance improvements for MCTS:

```bash
# Development build (faster iteration, less optimized)
cd rust && uv run python -m maturin develop

# Release build (full optimizations, slower compile)
cd rust && uv run python -m maturin develop --release

# Quick development script (with automatic rebuild)
./rust-dev.sh
```

**Requirements:**
- Rust toolchain (install from https://rustup.rs/)
- Cargo (comes with Rust)
- Maturin (included in project dependencies)

**Build Output:**
The extension will be built and installed into your Python environment as `hiivelabs_zertz_mcts`, enabling:
- Fast game logic functions (move generation, state validation)
- Rust MCTS implementation with parallel search
- Zero-copy numpy array integration

**Testing Rust Extension:**
```bash
# Verify the extension is available
uv run python -c "import hiivelabs_zertz_mcts; print('Rust extension loaded!')"

# Run tests that verify Python-Rust parity
uv run pytest tests/test_rust_parity.py -v
```

### Testing

The project includes a comprehensive test suite covering game logic, notation systems, and board mechanics across all supported board sizes.

Run tests with pytest:

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_notation.py -v

# Generate coverage report (excludes Panda3D renderer files)
uv run pytest --cov=game --cov=controller --cov=shared --cov-report=term-missing tests/

# Generate HTML coverage report
uv run pytest --cov=game --cov=controller --cov=shared --cov-report=html tests/
# View at htmlcov/index.html
```

**Test Categories:**
- Game mechanics (placements, captures, isolations)
- Win condition detection
- Notation generation and parsing (official Zèrtz format)
- Board state management
- Pass mechanics and loop detection
- Replay system
- Board size compatibility

### Visual Features

- **Frozen Region Indication**: Isolated regions with vacant rings (unplayable per official rules) are visually distinguished with faded/transparent rings (70% opacity)
- **Move Highlighting**: Use `--highlight-choices` flag to see valid placement positions (green), removable rings (red), and capture paths (blue)
  - Intelligent highlighting: automatically skips highlight phase when only one capture is available
  - Per-phase timing: different durations for placement, removal, and capture highlights
- **Coordinate Labels**: Use `--show-coords` flag to display coordinate labels on rings (e.g., A1, B2) that always face the camera
- **Water Reflections**: Dynamic water plane with custom shaders for realistic reflections
- **Dynamic Lighting**: Directional and ambient lighting for depth and atmosphere
- **Human Player Mode**: Use `--human` flag to control player 1 manually with mouse interaction

### Key Technical Details

- **RNG Separation**: Game logic uses seeded numpy.random for deterministic gameplay, while visual effects (marble rotations) use independent unseeded random instances
- **Coordinate Systems**: Uses bottom-up numbering where A1 is at the bottom of column A (official Zèrtz notation). Custom board layouts use `flattened_letters` array for coordinate mapping (e.g., 61-ring boards skip 'I')
- **Symmetry Transforms**: Implements D6 (37/61 rings) and D3 (48 rings) dihedral group symmetries for state canonicalization
- **ML Integration**: State separated into spatial (L×H×W board features) and global (10-element vector) components for machine learning applications
- **Official Notation**: Game outputs moves in official Zèrtz notation format (e.g., `Wd4`, `x e3Wg3`, `-`) alongside internal dictionary format
- **Unified Animation System**: Single animation queue handles both movement animations and highlight effects (material changes). Type discrimination (`'move'` vs `'highlight'`) allows different processing paths while maintaining consistent timing and lifecycle. Highlights apply instantly; moves interpolate over time
- **Code Architecture**: Both Python and Rust follow delegation pattern - MCTS delegates to pure game logic functions (mcts.rs → game.rs, mcts_tree.py → zertz_logic.py) ensuring single source of truth for all game rules

## Architecture Overview

Zertz3D follows **clean architecture principles** with clear separation of concerns and a layered design pattern. The system combines Python for flexibility with Rust for performance-critical paths.

### Layered Architecture

```
┌─────────────────────────────────────────┐
│      Presentation Layer                 │
│  (main.py, CLI argument parsing)        │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│      Application Layer                  │
│  (Controllers, Session Management)      │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│      Domain Layer                       │
│  (Game Logic, Board State, Players)     │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│      Infrastructure Layer               │
│  (Renderers, File I/O, Rust Backend)   │
└─────────────────────────────────────────┘
```

### Key Design Patterns

- **Factory Pattern**: `ZertzFactory` provides clean dependency injection and object construction
- **Protocol-Based Interfaces**: `IRenderer` protocol enables multiple renderer implementations without inheritance coupling
- **Stateless Game Logic**: Pure functional design in both Python (`zertz_logic.py`) and Rust (`game.rs`) enables parallelization and easy testing
- **Observer Pattern**: Async callbacks for renderer synchronization
- **Strategy Pattern**: Pluggable player implementations (Random, MCTS, Human, Replay)

### Python-Rust Integration

The system uses PyO3 for seamless Python-Rust integration:

```
Python Layer (High-level orchestration)
    │
    ├─▶ zertz_logic.py (thin wrapper)
    │
    ▼
PyO3 FFI (zero-copy array transfer)
    │
    ▼
Rust Layer (performance-critical paths)
    │
    ├─▶ game.rs (pure stateless game logic)
    ├─▶ mcts.rs (MCTS search engine)
    ├─▶ transposition.rs (position caching)
    └─▶ zobrist.rs (hash functions)
```

**Benefits:**
- Zero-copy numpy array transfer between Python and Rust
- 10-100x performance improvement for MCTS
- Absolute parity between Python and Rust implementations
- Fallback to pure Python if Rust extension unavailable

### MCTS Architecture

The MCTS implementation features advanced techniques for competitive AI:

**Core Components:**
- **UCB1 Selection**: Balance exploration vs exploitation
- **RAVE (Rapid Action Value Estimation)**: Faster convergence using all-moves-as-first statistics
- **Progressive Widening**: Focus search on promising moves
- **First Play Urgency (FPU)**: Penalize unexplored nodes
- **Transposition Tables**: Cache position evaluations with Zobrist hashing
- **Virtual Loss**: Lock-free parallel search via optimistic concurrency

**Parallel Architecture:**
```
┌─────────────────────────────────────────┐
│   Thread 1   Thread 2   ...   Thread N  │
│       │          │               │       │
│       └──────────┴───────────────┘       │
│                  │                       │
│                  ▼                       │
│     Shared Root Node (Arc + Atomics)    │
│       visits: AtomicU32                  │
│       value: AtomicU32                   │
│       children: Mutex<Vec<...>>          │
└─────────────────────────────────────────┘
```

Thread-safe statistics with atomic operations and minimal lock contention (mutex only during node expansion).

### Renderer Protocol

Multiple renderer implementations demonstrate the Open/Closed Principle:

- **PandaRenderer**: Full 3D visualization with Panda3D
- **TextRenderer**: Console-based output for terminal environments
- **CompositeRenderer**: Combines multiple renderers (e.g., 3D + text logging)

All renderers implement the `IRenderer` protocol, allowing the controller to work with any renderer without code changes.

## Dependencies

See `pyproject.toml` for complete dependency list.

## License

[GNU Affero General Public License v3.0](LICENSE)

## Contributing

Fork, create a pull request. Wait until I've at least released the first version though! (unless you see a critical bug!)
