"""Utility for rendering Zèrtz board states from notation to 2D images.

This module provides functions to visualize game states by:
1. Parsing notation sequences (from string or file)
2. Executing moves on a game board
3. Rendering the resulting board state as a top-down 2D image
4. Saving to PNG or displaying on screen

Example usage:
    # From notation file
    render_board_from_notation("game.txt", output_path="board.png")

    # From notation string
    notation = "37\nWd4\nGe3,a1\nx d4Ge3"
    render_board_from_notation(notation, show=True)

    # Custom styling
    render_board_from_notation(
        "game.txt",
        output_path="board.png",
        show_coords=True,
        size=(1200, 1200),
        title="Final Position"
    )
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from game.zertz_game import ZertzGame, STANDARD_MARBLES, BLITZ_MARBLES, STANDARD_WIN_CONDITIONS, BLITZ_WIN_CONDITIONS
from game.zertz_board import ZertzBoard
from game.formatters import NotationFormatter
from game.loaders import NotationLoader


class DiagramRenderer:
    """Renders Zèrtz board states as 2D diagram images."""

    # Visual constants (matching visualize_board_coords.py)
    HEX_SIZE = 1.0  # Hex grid size
    RING_RADIUS = 0.866 * HEX_SIZE  # Ring radius (sqrt(3)/2 ≈ 0.866)
    MARBLE_RADIUS = 0.6 * HEX_SIZE  # Marble radius (20% bigger: 0.5 * 1.2)

    # Colors
    RING_COLOR = "#8B7355"  # Brown/wood color for rings
    RING_EDGE_COLOR = "#5C4033"  # Darker brown for ring edges
    EMPTY_RING_ALPHA = 0.3  # Transparency for removed rings

    MARBLE_COLORS = {
        "w": "#FFFFFF",  # White
        "g": "#808080",  # Gray
        "b": "#000000",  # Black
    }

    MARBLE_EDGE_COLOR = "#333333"  # Dark edge for all marbles

    COORD_COLOR = "#000000"  # Black text for coordinates (empty rings and white marbles)
    COORD_COLOR_ON_DARK = "#FFFFFF"  # White text for gray/black marbles
    COORD_FONT_SIZE = 19  # 20% bigger: 16 * 1.2 ≈ 19

    BACKGROUND_COLOR = "#F5E6D3"  # Light beige background

    def __init__(self, show_coords: bool = False, show_removed: bool = False):
        """Initialize board renderer.

        Args:
            show_coords: If True, display coordinate labels on rings
            show_removed: If True, show removed rings with transparency
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for board rendering. "
                "Install it with: pip install matplotlib"
            )

        self.show_coords = show_coords
        self.show_removed = show_removed

    def _yx_to_axial(self, y: int, x: int, width: int) -> tuple[float, float]:
        """Convert board (y,x) indices to axial coordinates (q,r).

        Uses Zertz axial convention: q = x - center, r = y - x

        Args:
            y: Row index in board array
            x: Column index in board array
            width: Board width

        Returns:
            (q, r): Axial coordinates
        """
        c = width // 2
        q = x - c
        r = y - x
        return q, r

    def _axial_to_cart(self, q: float, r: float, size: float = 1.0) -> tuple[float, float]:
        """Convert axial (q,r) to Cartesian for a pointy-top hex grid.

        Args:
            q: Axial q coordinate
            r: Axial r coordinate
            size: Hex size (default: 1.0)

        Returns:
            (xc, yc): Cartesian coordinates
        """
        xc = size * np.sqrt(3) * (q + r / 2.0)
        yc = size * 1.5 * r
        return xc, yc

    def _get_hex_position(self, y: int, x: int, width: int) -> tuple[float, float]:
        """Convert board array indices to hexagonal grid coordinates.

        Matches the layout from visualize_board_coords.py:
        1. Convert to axial coordinates
        2. Convert to Cartesian (pointy-top hexagons)
        3. Apply 30° counter-clockwise rotation
        4. Flip Y axis so A1 is at bottom

        Args:
            y: Row index in board array
            x: Column index in board array
            width: Board width

        Returns:
            (px, py): Position in hexagonal coordinate space
        """
        # Convert to axial, then to Cartesian
        q, r = self._yx_to_axial(y, x, width)
        xc, yc = self._axial_to_cart(q, r, size=1.0)

        # Apply 30° counter-clockwise rotation
        theta = np.deg2rad(30)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        px = xc * cos_t - yc * sin_t
        py = xc * sin_t + yc * cos_t

        # Flip Y so numbering increases upward (A1 at bottom)
        py = -py

        return px, py

    def render_board(
        self,
        board: ZertzBoard,
        title: Optional[str] = None,
        figsize: tuple[int, int] = (10, 10),
    ) -> plt.Figure:
        """Render a board state as a matplotlib figure.

        Args:
            board: ZertzBoard instance to render
            title: Optional title for the figure
            figsize: Figure size in inches (width, height)

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.set_facecolor(self.BACKGROUND_COLOR)
        fig.patch.set_facecolor(self.BACKGROUND_COLOR)

        # Get board dimensions
        width = board.width

        # Calculate scale factor for line widths based on figure size
        # Default is 10x10 inches, scale linewidths proportionally
        default_size = 10.0
        scale_factor = figsize[0] / default_size

        # Collect patches for rings and marbles, and text labels
        ring_patches = []
        marble_patches = []
        text_labels = []  # Store (px, py, pos_str, text_color) tuples

        # Iterate through board positions
        for y in range(width):
            for x in range(width):
                # Get position string (empty if not a valid board position)
                pos_str = ""
                try:
                    if board.letter_layout is not None and board.letter_layout[y][x]:
                        pos_str = board.letter_layout[y][x]
                        ring_exists = board.state[board.RING_LAYER, y, x] == 1
                    else:
                        continue
                except IndexError:
                    continue

                if not pos_str:
                    continue

                # Get hexagonal position
                px, py = self._get_hex_position(y, x, width)

                # Draw ring as circle (if exists or show_removed is True)
                if ring_exists or self.show_removed:
                    ring = patches.Circle(
                        (px, py),
                        radius=self.RING_RADIUS,
                        fill=False,  # Rings are hollow (just the outline)
                        edgecolor=self.RING_EDGE_COLOR,
                        linewidth=2 * scale_factor,
                        alpha=1.0 if ring_exists else self.EMPTY_RING_ALPHA,
                        zorder=1,
                    )
                    ring_patches.append(ring)

                # Check for marbles and draw if present
                marble_type = None
                if ring_exists:
                    # Check for marbles (layers 1-3: white, gray, black)
                    for layer_idx, marble_char in enumerate(['w', 'g', 'b'], start=1):
                        if board.state[layer_idx, y, x] == 1:
                            marble_type = marble_char
                            break

                    if marble_type:
                        marble = patches.Circle(
                            (px, py),
                            radius=self.MARBLE_RADIUS,
                            facecolor=self.MARBLE_COLORS[marble_type],
                            edgecolor=self.MARBLE_EDGE_COLOR,
                            linewidth=1.5 * scale_factor,
                            zorder=3,
                        )
                        marble_patches.append(marble)

                # Store text label info for rendering AFTER marbles
                # Use white text on gray/black marbles, black text otherwise
                if self.show_coords and ring_exists:
                    text_color = self.COORD_COLOR_ON_DARK if marble_type in ['g', 'b'] else self.COORD_COLOR
                    text_labels.append((px, py, pos_str, text_color))

        # Add all patches to axis (rings, then marbles, then text on top)
        for patch in ring_patches:
            ax.add_patch(patch)
        for patch in marble_patches:
            ax.add_patch(patch)

        # Render text labels on top of everything
        for px, py, pos_str, text_color in text_labels:
            ax.text(
                px, py,
                pos_str,
                ha='center', va='center',
                fontsize=self.COORD_FONT_SIZE * scale_factor,
                color=text_color,
                zorder=4,  # Above marbles (zorder=3)
                weight='bold',  # Make text bold for better visibility
            )

        # Set axis limits based on ALL possible ring positions (not just existing ones)
        # This ensures consistent image sizes across different board states
        all_x_coords = []
        all_y_coords = []
        for y in range(width):
            for x in range(width):
                try:
                    if board.letter_layout is not None and board.letter_layout[y][x]:
                        px, py = self._get_hex_position(y, x, width)
                        all_x_coords.append(px)
                        all_y_coords.append(py)
                except IndexError:
                    continue

        if all_x_coords and all_y_coords:
            # Border = full ring radius + 1/4 hex width trim
            # This shows complete rings at edges with minimal whitespace
            border = self.RING_RADIUS + 0.25
            ax.set_xlim(min(all_x_coords) - border, max(all_x_coords) + border)
            ax.set_ylim(min(all_y_coords) - border, max(all_y_coords) + border)

        ax.axis('off')

        # Add title if provided (1.5x the coordinate font size, scaled)
        if title:
            ax.set_title(title, fontsize=int(self.COORD_FONT_SIZE * 1.5 * scale_factor), pad=20)

        plt.tight_layout()
        return fig

    def save_board(
        self,
        board: ZertzBoard,
        output_path: Union[str, Path],
        title: Optional[str] = None,
        figsize: tuple[int, int] = (10, 10),
        dpi: int = 150,
    ) -> None:
        """Render and save a board state to a file.

        Args:
            board: ZertzBoard instance to render
            output_path: Path to save the image (PNG/SVG supported)
            title: Optional title for the figure
            figsize: Figure size in inches (width, height)
            dpi: Resolution in dots per inch (for raster formats like PNG)
        """
        fig = self.render_board(board, title=title, figsize=figsize)

        # Detect format from extension - matplotlib handles SVG natively
        output_path = Path(output_path)
        if output_path.suffix.lower() == '.svg':
            # SVG is vector format - no DPI needed, but include for text sizing
            fig.savefig(output_path, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
        else:
            # Raster formats (PNG, JPG, etc.) use DPI
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())

        plt.close(fig)

    def show_board(
        self,
        board: ZertzBoard,
        title: Optional[str] = None,
        figsize: tuple[int, int] = (10, 10),
    ) -> None:
        """Render and display a board state on screen.

        Args:
            board: ZertzBoard instance to render
            title: Optional title for the figure
            figsize: Figure size in inches (width, height)
        """
        fig = self.render_board(board, title=title, figsize=figsize)
        plt.show()


def parse_notation_string(notation_str: str) -> tuple[list[dict], list[dict], int, bool]:
    """Parse a notation string into action lists.

    Args:
        notation_str: Multi-line string with notation format:
            First line: board size + optional "Blitz"
            Remaining lines: notation moves

    Returns:
        Tuple of (player1_actions, player2_actions, rings, blitz)
    """
    lines = notation_str.strip().split('\n')
    if not lines:
        raise ValueError("Empty notation string")

    # Parse header
    header = lines[0].strip()
    parts = header.split()
    try:
        rings = int(parts[0])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid header: expected board size, got '{header}'")

    blitz = len(parts) > 1 and "blitz" in " ".join(parts[1:]).lower()

    # Parse moves
    player1_actions = []
    player2_actions = []
    current_player = 1

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        try:
            action_dict = NotationFormatter.notation_to_action_dict(line)
            if current_player == 1:
                player1_actions.append(action_dict)
            else:
                player2_actions.append(action_dict)
            current_player = 2 if current_player == 1 else 1
        except ValueError as e:
            # Skip invalid notation lines
            continue

    return player1_actions, player2_actions, rings, blitz


def action_dict_to_str(action_dict: dict) -> str:
    """Convert action_dict to action string format.

    Args:
        action_dict: Action dictionary from NotationLoader

    Returns:
        str: Action string in format expected by ZertzGame.str_to_action()
    """
    if action_dict["action"] == "PASS":
        return "PASS"

    if action_dict["action"] == "PUT":
        marble = action_dict["marble"]
        dst = action_dict["dst"]
        remove = action_dict.get("remove", "")
        if remove:
            return f"PUT {marble} {dst} {remove}"
        else:
            return f"PUT {marble} {dst}"

    if action_dict["action"] == "CAP":
        src = action_dict["src"]
        dst = action_dict["dst"]
        capture = action_dict["capture"]
        return f"CAP {src} {capture} {dst}"

    raise ValueError(f"Unknown action type: {action_dict.get('action')}")


def execute_notation_sequence(
    notation_input: Union[str, Path],
    stop_at_move: Optional[int] = None,
) -> ZertzBoard:
    """Execute a notation sequence and return the resulting board state.

    Args:
        notation_input: Either a file path or a notation string
        stop_at_move: Optional move number to stop at (0-indexed total moves)

    Returns:
        ZertzBoard instance with the final state
    """
    # Determine if input is a file or string
    if isinstance(notation_input, (str, Path)):
        path = Path(notation_input)
        if path.exists() and path.is_file():
            # Load from file
            loader = NotationLoader(str(path))
            player1_actions, player2_actions = loader.load()
            rings = loader.detected_rings
            blitz = loader.blitz
            marbles = BLITZ_MARBLES if blitz else STANDARD_MARBLES
            win_condition = BLITZ_WIN_CONDITIONS if blitz else STANDARD_WIN_CONDITIONS
        else:
            # Treat as notation string
            player1_actions, player2_actions, rings, blitz = parse_notation_string(str(notation_input))
            marbles = BLITZ_MARBLES if blitz else STANDARD_MARBLES
            win_condition = BLITZ_WIN_CONDITIONS if blitz else STANDARD_WIN_CONDITIONS
    else:
        raise TypeError("notation_input must be a string or Path")

    # Create game
    game = ZertzGame(rings, marbles, win_condition, t=5)

    # Execute moves
    move_count = 0

    for i in range(max(len(player1_actions), len(player2_actions))):
        # Player 1 move
        if i < len(player1_actions):
            if stop_at_move is not None and move_count >= stop_at_move:
                break

            action_dict = player1_actions[i]
            action_str = action_dict_to_str(action_dict)
            action_type, action = game.str_to_action(action_str)
            game.take_action(action_type, action)
            move_count += 1

        # Player 2 move
        if i < len(player2_actions):
            if stop_at_move is not None and move_count >= stop_at_move:
                break

            action_dict = player2_actions[i]
            action_str = action_dict_to_str(action_dict)
            action_type, action = game.str_to_action(action_str)
            game.take_action(action_type, action)
            move_count += 1

        # Check if game ended
        if game.get_game_ended() is not None:
            break

    return game.board


def render_board_from_notation(
    notation_input: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    show_coords: bool = True,
    show_removed: bool = False,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    dpi: int = 150,
    stop_at_move: Optional[int] = None,
) -> None:
    """Render a board state from notation sequence.

    This is the main entry point for the utility. It parses notation,
    executes the moves, and renders the final board state.

    Args:
        notation_input: Either a file path or a notation string
        output_path: Optional path to save the image (PNG recommended)
        show: If True, display the image on screen
        show_coords: If True, display coordinate labels on rings
        show_removed: If True, show removed rings with transparency
        title: Optional title for the figure
        width: Output width in pixels (if None, defaults to 1500)
        height: Output height in pixels (if None, defaults to 1500)
        dpi: Resolution for saved image (default: 150)
        stop_at_move: Optional move number to stop at (0-indexed)

    Width/Height behavior:
        - Both None: Use default (1500x1500 pixels at 150 DPI = 10x10 inches)
        - Width only: Calculate height to maintain aspect ratio
        - Height only: Calculate width to maintain aspect ratio
        - Both specified: Use both (may distort aspect ratio)

    Example:
        # From file, save to PNG with default size (1500x1500 pixels)
        render_board_from_notation("game.txt", output_path="board.png")

        # Custom width in pixels (height auto-calculated)
        render_board_from_notation("game.txt", output_path="board.png", width=1200)

        # Custom dimensions in pixels
        render_board_from_notation("game.txt", output_path="board.png", width=1800, height=1800)

        # From string, display on screen
        notation = "37\\nWd4\\nGe3,a1"
        render_board_from_notation(notation, show=True, title="Position after 2 moves")

        # Stop at specific move
        render_board_from_notation("game.txt", output_path="move_10.png", stop_at_move=10)
    """
    # Calculate figsize based on width/height parameters (convert pixels to inches)
    default_pixels = 1500  # Default: 10 inches * 150 DPI

    if width is None and height is None:
        # Both omitted: use defaults
        width_inches = default_pixels / dpi
        height_inches = default_pixels / dpi
        figsize = (width_inches, height_inches)
    elif width is not None and height is None:
        # Width only: maintain aspect ratio (board is square)
        width_inches = width / dpi
        figsize = (width_inches, width_inches)
    elif width is None and height is not None:
        # Height only: maintain aspect ratio (board is square)
        height_inches = height / dpi
        figsize = (height_inches, height_inches)
    else:
        # Both specified: use as-is
        width_inches = width / dpi
        height_inches = height / dpi
        figsize = (width_inches, height_inches)

    # Execute notation to get board state
    board = execute_notation_sequence(notation_input, stop_at_move=stop_at_move)

    # Create renderer
    renderer = DiagramRenderer(show_coords=show_coords, show_removed=show_removed)

    # Save or show
    if output_path:
        renderer.save_board(board, output_path, title=title, figsize=figsize, dpi=dpi)

    if show:
        renderer.show_board(board, title=title, figsize=figsize)

    if not output_path and not show:
        # Default to showing if no output specified
        renderer.show_board(board, title=title, figsize=figsize)