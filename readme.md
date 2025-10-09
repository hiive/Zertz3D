# Zèrtz 3D

A 3D implementation of the abstract board game Zèrtz using Panda3D. This project combines sophisticated game logic with interactive 3D visualization to create an engaging digital version of the classic strategy game.

Full game rules: http://www.gipf.com/zertz/rules/rules.html

## Features

- **Multiple Board Sizes**: Play on 37, 48, or 61 ring boards
- **3D Visualization**: Beautiful Panda3D rendering with water reflections and dynamic lighting
- **Replay System**: Record and replay games from text files
- **Deterministic Gameplay**: Seeded random number generation for reproducible games
- **Headless Mode**: Run games without rendering for testing or simulation
- **AI Players**: Includes random player implementation with extensible player framework

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
# Specify board size (37, 48, or 61 rings)
uv run main.py --rings 61

# Use a specific random seed for reproducible games
uv run main.py --seed 1234567890

# Replay a game from a text file (board size is auto-detected)
uv run main.py --replay path/to/replay.txt

# Run in headless mode (no 3D renderer)
uv run main.py --headless

# Control number of games to play (default: play indefinitely)
uv run main.py --games 10

# Log game actions to file
uv run main.py --log
```

## Project Structure

```
Zertz3D/
├── game/                    # Core game logic
│   ├── zertz_board.py      # Board state and move validation
│   ├── zertz_game.py       # Game controller and rules
│   └── zertz_player.py     # Player implementations
├── renderer/                # 3D rendering components
│   ├── zertz_renderer.py   # Main Panda3D renderer
│   ├── zertz_models.py     # 3D model classes (marbles, rings)
│   └── water_node.py       # Water reflection effects
├── controller/              # Game controller logic
├── tests/                   # Test suite
├── data/                    # Game data and logs
│   ├── docs/               # Documentation
│   └── logfiles/           # Game replay files
└── main.py                  # Entry point
```

## Game Mechanics

### Board Representation

The game uses a layered 3D numpy array (L × H × W) to represent the game state:
- Rings layer: Valid positions on the board
- Marble layers: White, gray, and black marbles
- Supply and capture tracking layers
- Current player state

### Action System

Two types of actions:
- **Placement (PUT)**: Place a marble and remove a ring
  - Format: `PUT {marble} {dst} {remove}`
  - Example: `PUT g B5 G3`

- **Capture (CAP)**: Jump over an opponent's marble
  - Format: `CAP {marble} {src} {capture} {dst}`
  - Example: `CAP b C4 g D5 E6`

- **Pass (PASS)**: Used when no valid moves available

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

Run tests with pytest:

```bash
uv run pytest
```

### Key Technical Details

- **RNG Separation**: Game logic uses seeded numpy.random for deterministic gameplay, while visual effects (marble rotations) use independent unseeded random instances
- **Coordinate Systems**: Custom board layouts use `flattened_letters` array for coordinate mapping (e.g., 61-ring boards skip 'I')
- **Symmetry Transforms**: Implements D6 (37/61 rings) and D3 (48 rings) dihedral group symmetries for state canonicalization
- **ML Integration**: State separated into spatial (L×H×W board features) and global (10-element vector) components for machine learning applications

For more detailed technical documentation, see `CLAUDE.md`.

## Dependencies

Key dependencies:
- Panda3D 1.10.15 - 3D rendering engine
- NumPy 2.3.3 - Game state arrays and calculations
- PyTorch 2.8.0 - For ML/AI integration

See `pyproject.toml` for complete dependency list.

## License

[Add license information]

## Contributing

[Add contribution guidelines]