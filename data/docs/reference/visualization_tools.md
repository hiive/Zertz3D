# Diagram Rendering Toolkit

Utilities for turning Zèrtz game data into static diagrams. The core renderer in `game/utils/diagram.py` powers the documentation assets under `data/docs/visualizations/` and drives three dedicated CLI tools.

## Capabilities At A Glance
- Load notation strings, transcript logs, or Boardspace SGF files (`AutoSelectLoader` handles format detection).
- Replay arbitrary prefixes of a game and render the current board.
- Export PNG or SVG with optional transparency, titles, and coordinate labels.
- Scale geometry, line widths, and typography automatically for any pixel dimensions.
- Batch rendering helpers for canonicalisation analysis and move-by-move sequences.

## Environment Setup

Matplotlib is part of the synced development environment; run:

```bash
uv sync
```

All command examples assume execution from the project root so that relative paths resolve correctly.

## CLI Utilities

### 1. `visualize_notation.py` — Single Position Rendering

Render a single board state or inspect the end position of a replay file.

```bash
uv run scripts/visualization/visualize_notation.py data/sgf/sample_game.sgf \
    --output build/sample_position.png \
    --coords \
    --show-removed \
    --show-captures \
    --width 2200 \
    --dpi 300 \
    --title "Sample Position"
```

**Core options**

| Flag | Description |
| --- | --- |
| `INPUT` | Notation/transcript/SGF file, or inline notation string |
| `--output PATH` | Save result to PNG or SVG (omit to skip saving) |
| `--show` | Display an interactive preview window |
| `--coords` / `--edge-coords` | Full ring labels or edge-only style |
| `--show-removed` | Visualise removed rings in muted grey |
| `--show-captures` | Print capture totals for both players |
| `--stop-at N` | Render the position after move `N` (0-indexed) |
| `--width`, `--height` | Pixel dimensions; omit one side to keep aspect ratio |
| `--dpi` | Raster resolution (default 150) |
| `--bg-color HEX` | Background colour (`#RRGGBB` or `#RRGGBBAA`) |
| `--title TEXT` | Caption displayed above the board |

Use `--stop-at` inside a shell loop to export multiple frames for animations or instructional slides.

### 2. `visualize_canonicalization.py` — Symmetry Analysis

Investigate how states collapse under rotational and mirror symmetries. The script prints summary statistics and can emit image grids that juxtapose original and canonicalised boards.

```bash
uv run scripts/visualization/visualize_canonicalization.py data/notation/league_final.txt \
    --save-images \
    --output-dir analysis_output/canonicalisation \
    --image-columns 4 \
    --show-all-transforms \
    --svg
```

Key flags:
- `--save-images` writes a PNG grid for each move; implied when `--show-all-transforms` is set.
- `--show-all-transforms` adds every unique rotation/mirror mapping to the output grid, highlighting symmetry classes.
- `--svg` switches the combined grid output to vector format.
- `--image-columns` controls the width of the combined grid; each move is rendered separately, so `--separate-turns` is ignored.
- `--show-row-dividers` draws subtle horizontal guides between rows in the combined grid output.

Console output includes the number of canonical states, when each state first appeared, and the average number of symmetry permutations that map to the same canonical representative.

### 3. `visualize_game_sequence.py` — Turn-by-Turn Grids

Produce visual narratives of entire matches.

```bash
uv run scripts/visualization/visualize_game_sequence.py data/sgf/tiebreak_game.sgf \
    --save-images \
    --output-dir analysis_output/sequences \
    --image-columns 5 \
    --show-captures
```

Two output modes are available:
- **Combined grid** (default) — images rendered with transparent backgrounds are assembled into a single PNG/SVG mosaic using `GridRenderer`.
- **Separate frames** — pass `--separate-turns` to emit one file per move (named with the move number and notation snippet). Combine with `--svg` for vector frames.

Additional options:
- `--show-captures` overlays live capture totals in each panel.
- `--save-images` toggles between saving and previewing; preview mode displays PNG grids via Pillow.
- `--image-columns` sets the grid width in combined mode.
- `--svg` switches both combined grids and per-move exports to vector output.
- `--show-row-dividers` adds thin separators between grid rows to aid reading dense summaries.

The script labels each panel with the move number, active player name (read from the log when available), and official notation.

## Python API

The CLI tools wrap the same primitives exposed in `game/utils/diagram.py`.

```python
from game.utils.diagram import render_board_from_notation, DiagramRenderer, execute_notation_sequence

# Quick export from a transcript
render_board_from_notation(
    "data/notation/final.txt",
    output_path="build/final_state.png",
    show_coords=True,
    stop_at_move=24,
    width=1800,
    bg_color="#101820",
)

# Custom renderer for publication styling
class PublicationRenderer(DiagramRenderer):
    RING_EDGE_COLOR = "#8B5A2B"
    BACKGROUND_COLOR = "#101820"
    COORD_COLOR = "#F2AA4C"

board = execute_notation_sequence("examples/example_game.txt", stop_at_move=12)
PublicationRenderer(show_coords=True, show_removed=True).save_board(board, "styled.png", width=2000)
```

Helper entry points worth noting:
- `render_board_from_notation(source, **kwargs)` — convenience wrapper (accepts file paths or inline notation strings).
- `execute_notation_sequence(source, stop_at_move=None)` — returns a `ZertzBoard` after applying the requested prefix.
- `action_dict_to_str(action)` — converts parsed dictionaries into canonical strings for applying updates via `ZertzGame`.

## Output Formats & Sizing

- **PNG** honours `--width`, `--height`, and `--dpi`. All rings, marbles, and text scale proportionally to maintain legibility.
- **SVG** ignores the DPI option but preserves exact geometry, making it ideal for print or vector editors.
- Defaults: 1500 × 1500 pixels at 150 DPI with the beige table background (`#F5E6D3`).
- When creating transparency-friendly assets, supply `--bg-color #00000000` (RGBA) and set `transparent=True` in direct API calls.

## Module Structure

- `game/utils/diagram.py` — main rendering engine and coordinate math.
  - `DiagramRenderer` — base class containing drawing routines and styling constants.
  - `render_board_from_notation`, `execute_notation_sequence` — high-level helpers.
- `game/utils/grid_renderer.py` — composes transparent panels into grids for sequence/canonicalisation outputs.
- CLI wrappers in `scripts/visualization/`:
  1. `visualize_notation.py`
  2. `visualize_canonicalization.py`
  3. `visualize_game_sequence.py`

## Customisation Tips

- Extend `DiagramRenderer` to override colours, fonts, or sizing ratios.
- For animation pipelines, run `visualize_notation.py` with `--stop-at` inside a loop, then stitch frames with ffmpeg or imageio.
- The grid-oriented scripts accept the same renderer flags; customise by subclassing `DiagramRenderer` and swapping it in where needed.

## Technical Notes

- Axial coordinates align with official Zèrtz notation (A1 at the lower edge). Hexes are rendered pointy-top with a 30° rotation and inverted y-axis.
- Margins expand just enough to include full rings along the perimeter, keeping diagrams tight without cropping geometry.
- When historical layers are provided (e.g., multi-step tensors for learning pipelines) the renderer understands the block layout used by the Rust MCTS state output.

## Limitations

- Matplotlib requires an available backend; set `MPLBACKEND=Agg` for headless servers.
- PNG previews rely on Pillow for display; ensure it is installed or use `--save-images`.
- Rendering very long matches as combined grids can produce large files; prefer `--separate-turns` for slide decks or animation tooling.
