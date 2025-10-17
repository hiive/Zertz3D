# Zèrtz 3D

A 3D implementation of the abstract board game Zèrtz using Panda3D. This project combines sophisticated game logic with interactive 3D visualization to create an engaging digital version of the classic strategy game.

Full game rules: http://www.gipf.com/zertz/rules/rules.html

**THIS IS A WORK-IN-PROGRESS!**

## Features

- **Multiple Board Sizes**: Play on 37, 48, or 61 ring boards
- **3D Visualization**: Panda3D rendering with water reflections and dynamic lighting
- **Replay System**: Record and replay games from text files in two formats (transcript/notation)
- **Deterministic Gameplay**: Seeded random number generation for reproducible games
- **Headless Mode**: Run games without rendering for testing or simulation
- **Official Notation**: Full support for official Zèrtz notation format

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

## Usage

### Basic Usage

```bash
uv run main.py
```

### Command Line Options

```bash
# Board Configuration
uv run main.py --rings 61                    # Board size: 37, 48, or 61 rings (default: 37)
uv run main.py --seed 1234567890             # Random seed for reproducible games
uv run main.py --blitz                       # Blitz variant (37 rings only, fewer marbles)

# Replay System
uv run main.py --replay path/to/file.txt     # Replay from transcript or notation file
uv run main.py --replay file.txt --partial   # Continue with random play after replay ends

# Game Control
uv run main.py --games 10                    # Number of games to play (default: infinite)
uv run main.py --headless                    # Run without 3D renderer
uv run main.py --human                       # Control player 1 manually (requires renderer)
uv run main.py --move-duration 0.5           # Duration between moves in seconds (default: 0.5)
uv run main.py --start-delay 2.0             # Delay before first move in seconds (default: 0)

# Logging Options
uv run main.py --transcript-file             # Log to zertzlog_<seed>.txt (current dir)
uv run main.py --transcript-file ./logs      # Log to zertzlog_<seed>.txt in ./logs
uv run main.py --notation-file               # Log official notation to file (current dir)
uv run main.py --notation-file ./logs        # Log official notation to ./logs
uv run main.py --transcript-screen           # Output transcript format to screen
uv run main.py --notation-screen             # Output official notation to screen

# Use multiple logging formats simultaneously
uv run main.py --transcript-file --notation-file
uv run main.py --transcript-screen --notation-screen

# Visual Options (3D renderer only)
uv run main.py --highlight-choices           # Highlight valid moves before each turn
uv run main.py --show-coords                 # Display coordinate labels on rings
```

## Project Structure

```
Zertz3D/
├── game/                           # Core game logic
│   ├── zertz_board.py             # Board state and move validation
│   ├── zertz_game.py              # Game controller and rules
│   ├── zertz_player.py            # Player implementations (Random, Replay, MCTS)
│   ├── zertz_position.py          # Position/state representation
│   ├── action_result.py           # Action result data structure
│   ├── writers.py                 # Log file writers
│   ├── formatters/                # Output formatters
│   │   ├── notation_formatter.py  # Official Zèrtz notation
│   │   └── transcript_formatter.py # Dictionary format
│   ├── loaders/                   # Replay file loaders
│   │   ├── notation_loader.py     # Parse notation files
│   │   └── transcript_loader.py   # Parse dictionary files
│   └── utils/                     # Utility functions
│       └── diagram.py             # ASCII board diagrams
├── renderer/                       # 3D rendering components
│   ├── zertz_renderer.py          # Main Panda3D renderer
│   ├── zertz_models.py            # 3D model classes (marbles, rings)
│   ├── water_node.py              # Water reflection effects
│   ├── animation_manager.py       # Animation queue system
│   ├── highlighting_manager.py    # Move highlighting
│   ├── interaction_helper.py      # Mouse interaction
│   ├── material_modifier.py       # Material state management
│   └── entities/                  # Model entity classes
├── controller/                     # Game flow control
│   ├── zertz_game_controller.py   # Main game loop coordinator
│   ├── game_session.py            # Session management
│   ├── game_loop.py               # Turn execution
│   ├── game_logger.py             # Logging system
│   ├── action_processor.py        # Action execution
│   └── action_text_formatter.py   # Human-readable action text
├── shared/                         # Shared utilities
│   ├── render_data.py             # Data transfer objects
│   ├── constants.py               # Global constants
│   └── materials_modifiers.py     # Material definitions
├── tests/                          # Test suite
│   ├── test_notation.py           # Notation system tests
│   ├── test_win_conditions.py     # Win detection tests
│   ├── test_pass_and_loops.py     # Pass/loop mechanics
│   ├── test_zertz_board.py        # Board logic tests
│   ├── test_zertz_game_methods.py # Game method tests
│   └── ...                        # 20+ additional test files
├── data/                           # Game data and logs
│   ├── docs/                      # Documentation
│   │   └── zertz_rules.md        # Official game rules
│   ├── logfiles/                  # Game replay files
│   └── models/                    # 3D model assets
└── main.py                         # Entry point
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


## Dependencies

See `pyproject.toml` for complete dependency list.

## License

[GNU Affero General Public License v3.0](LICENSE)

## Contributing

Fork, create a pull request. Wait until I've at least released the first version though! (unless you see a critical bug!)