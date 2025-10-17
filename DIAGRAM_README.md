# Diagram Renderer Utility

A utility for rendering Zèrtz board states from notation files or strings to 2D diagram images.

## Features

- Parse notation files or notation strings
- Execute move sequences and render resulting board states
- Save as PNG, SVG, or display on screen
- Show/hide coordinate labels (with smart color adaptation)
- Edge coordinates mode (show only top/bottom edge labels per column)
- Show/hide removed rings
- Customizable background color with transparency support
- Stop at specific move numbers
- Pixel-based sizing with automatic scaling
- Consistent image dimensions across different board states
- Vector (SVG) and raster (PNG) output formats

## Installation

The diagram renderer requires matplotlib:

```bash
# matplotlib is already in dev dependencies
uv sync --group dev
```

## Usage

### Command Line

Basic usage:

```bash
# Render notation file to PNG (default: 1500x1500 pixels)
python render_notation.py game.txt --output board.png

# Display on screen with coordinates
python render_notation.py game.txt --show --coords

# Render specific move number
python render_notation.py game.txt --output move_15.png --stop-at 15

# Custom size (in pixels) and title
python render_notation.py game.txt --output final.png --width 2400 --title "Final Position"

# Export as SVG (vector format - infinitely scalable)
python render_notation.py game.txt --output board.svg --coords

# Show removed rings (light gray)
python render_notation.py game.txt --output board.png --show-removed --coords

# Edge coordinates only (like official diagrams)
python render_notation.py game.txt --output board.png --edge-coords

# Custom background color (transparent for compositing)
python render_notation.py game.txt --output board.png --coords --bg-color #00000000

# Custom background (opaque white)
python render_notation.py game.txt --output board.png --coords --bg-color #FFFFFF
```

### Sizing

The renderer uses pixel-based sizing with automatic element scaling:

```bash
# Default size (1500x1500 pixels at 150 DPI)
python render_notation.py game.txt --output board.png

# Custom width (height auto-calculated to maintain aspect ratio)
python render_notation.py game.txt --output board.png --width 2400

# Custom dimensions (may distort if not square)
python render_notation.py game.txt --output board.png --width 2400 --height 1800

# Higher DPI for print quality
python render_notation.py game.txt --output board.png --width 3000 --dpi 300
```

All visual elements scale proportionally:
- Ring and marble line widths
- Coordinate text size
- Title text size

### Python API

```python
from game.utils.diagram import render_board_from_notation

# From file, save to PNG (default 1500x1500 pixels)
render_board_from_notation("game.txt", output_path="board.png")

# From string, display on screen
notation = """37
Wd4
Ge3,a1
x d4Ge3"""
render_board_from_notation(notation, show=True, title="Position after 3 moves")

# Stop at specific move
render_board_from_notation("game.txt", output_path="move_10.png", stop_at_move=10)

# Full customization with pixel-based sizing
render_board_from_notation(
    "game.txt",
    output_path="board.png",
    show=False,
    show_coords=True,
    show_removed=True,
    title="Final Position",
    width=2400,          # Width in pixels
    height=2400,         # Height in pixels (optional, maintains aspect ratio if omitted)
    dpi=150,
)

# Edge coordinates with custom background
render_board_from_notation(
    "game.txt",
    output_path="official_style.png",
    edge_coords=True,    # Show only top/bottom edge labels
    show_removed=True,
    bg_color="#00000000", # Transparent background (RGBA)
    title="Position Analysis",
)

# SVG output for vector graphics
render_board_from_notation(
    "game.txt",
    output_path="board.svg",  # Automatically detects SVG format
    show_coords=True,
    title="Vector Board",
)
```

### Advanced Usage

```python
from game.utils.diagram import DiagramRenderer, execute_notation_sequence

# Execute notation and get board state
board = execute_notation_sequence("game.txt", stop_at_move=20)

# Create custom renderer
renderer = DiagramRenderer(show_coords=True, show_removed=False)

# Render with different sizes (figsize in inches = pixels / dpi)
renderer.save_board(board, "board_1500px.png", title="Move 20", figsize=(10, 10), dpi=150)
renderer.save_board(board, "board_3000px.png", title="Move 20", figsize=(20, 20), dpi=150)
renderer.save_board(board, "board.svg", title="Move 20", figsize=(10, 10))

# Show on screen
renderer.show_board(board, title="Current Position")
```

## Notation Format

The utility supports official Zèrtz notation format:

```
37              # Board size (37, 48, or 61)
Wd4             # Place white marble at d4 (no ring removal)
Ge3,a1          # Place gray marble at e3, remove ring a1
x d4Ge3         # Capture: jump from d4 over gray marble to e3
-               # Pass
```

For Blitz variant:

```
37 Blitz        # Board size with variant
Wd4
...
```

## Output

The renderer creates top-down 2D images showing:

- **Rings**: Circular outlines (hollow)
  - Present rings: Solid black, 2pt line width
  - Removed rings (if show_removed): Light gray, 1pt line width
  - Empty positions on present rings: Small inner circle in mid-gray
  - No alpha transparency except on background
- **Marbles**: Filled circles on rings, 20% larger than standard
  - White, gray, and black marbles
  - Dark edge color for definition
- **Coordinates** (if show_coords or edge_coords): Position labels like A1, B2, etc.
  - Regular coords: Black text on empty rings and white marbles, white text on dark marbles, bold, centered on rings
  - Edge coords: Black text only, non-bold, positioned above top edge rings and below bottom edge rings
  - Edge coords shown even when rings are removed
  - Automatically scaled with image size
- **Background**: Light beige by default, customizable via --bg-color flag (supports RGBA transparency)
- **Title** (if specified): Always bold, 1.5x the coordinate font size, scaled proportionally

### Consistent Sizing

The renderer ensures all images from the same board size have identical dimensions:
- Axis limits calculated from ALL possible ring positions (not just existing rings)
- Removed rings don't affect image boundaries
- Perfect for animations or side-by-side comparisons

### Format Support

- **PNG**: Raster format, best for fixed-size displays (default 1500x1500px)
- **SVG**: Vector format, infinitely scalable without quality loss (smaller file size)
- Automatic format detection from file extension

## Examples

```bash
# Create diagram for documentation (high DPI)
uv run render_notation.py examples/example_game.txt \
    --output docs/example_position.png \
    --coords \
    --title "Example Position" \
    --width 2400 \
    --dpi 300

# Generate SVG for web/print
uv run render_notation.py game.txt \
    --output diagram.svg \
    --coords \
    --title "Board Position"

# Generate sequence of images for animation
for i in {1..50}; do
    uv run render_notation.py game.txt \
        --output "frames/frame_$(printf '%03d' $i).png" \
        --stop-at $i \
        --coords \
        --width 1200
done

# Quick preview of a position
uv run render_notation.py game.txt --show --coords

# High-resolution print version (6000px at 300 DPI = 20 inches)
uv run render_notation.py game.txt \
    --output poster.png \
    --width 6000 \
    --dpi 300 \
    --coords \
    --title "Tournament Position"

# Official diagram style (edge coords, transparent background)
uv run render_notation.py game.txt \
    --output official_diagram.png \
    --edge-coords \
    --show-removed \
    --bg-color #00000000 \
    --title "Position Analysis"

# Minimal diagram (white background, edge coords only)
uv run render_notation.py game.txt \
    --output minimal.png \
    --edge-coords \
    --bg-color #FFFFFF
```

## Module Structure

- `game/utils/diagram.py`: Core diagram rendering module
  - `DiagramRenderer`: Main renderer class
  - `render_board_from_notation()`: Convenience function
  - `execute_notation_sequence()`: Execute moves and return board
  - `parse_notation_string()`: Parse notation from string

- `render_notation.py`: Command-line interface

## Customization

The `DiagramRenderer` class can be subclassed to customize appearance:

```python
from game.utils.diagram import DiagramRenderer, execute_notation_sequence

class CustomRenderer(DiagramRenderer):
    # Override colors
    RING_EDGE_COLOR = "#A0522D"      # Sienna (present rings)
    REMOVED_RING_COLOR = "#CCCCCC"   # Lighter gray (removed rings)
    INNER_RING_COLOR = "#666666"     # Darker gray (empty position inner circles)
    BACKGROUND_COLOR = "#FFFFFF"     # White

    # Override sizes (in hex grid units)
    MARBLE_RADIUS = 0.55 * DiagramRenderer.HEX_SIZE

    # Override text styling
    COORD_FONT_SIZE = 24
    COORD_COLOR = "#FF0000"

renderer = CustomRenderer(show_coords=True)
board = execute_notation_sequence("game.txt")
renderer.save_board(board, "custom_style.png")
```

## Technical Details

### Coordinate System

- Uses axial coordinate system with pointy-top hexagons
- 30° counter-clockwise rotation applied
- Y-axis flipped so A1 is at bottom (matches official notation)
- Bottom-up numbering: A1, A2, A3, A4 (increasing upward)

### Scaling

All visual elements scale proportionally with output size:
- `scale_factor = figsize_width / 10.0` (default 10 inches)
- Ring line width: `2.0 * scale_factor`
- Marble line width: `1.5 * scale_factor`
- Coordinate font: `19pt * scale_factor`
- Title font: `28.5pt * scale_factor`

### Border Calculation

Border = `RING_RADIUS + 0.25` (ring radius + 1/4 hex width)
- Shows complete rings at edges
- Minimal whitespace around board

## Limitations

- Only renders final board state (not animations)
  - Use stop_at_move parameter and generate multiple frames for animation
- Requires matplotlib (not available in headless environments without X server)
- Large board sizes (61 rings) automatically scale well with pixel sizing

## See Also

- Official Zèrtz rules: http://www.gipf.com/zertz/rules/rules.html
- Notation specification: http://www.gipf.com/zertz/notations/notation.html