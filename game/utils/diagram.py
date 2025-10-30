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
        internal_coords=True,
        edge_coords=True,
        width=1200,
        height=1200,
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

from game.zertz_game import ZertzGame
from game.constants import STANDARD_MARBLES, BLITZ_MARBLES, STANDARD_WIN_CONDITIONS, BLITZ_WIN_CONDITIONS
from game.zertz_board import ZertzBoard
from game.formatters import NotationFormatter
from game.loaders import AutoSelectLoader


class DiagramRenderer:
    """Renders Zèrtz board states as 2D diagram images."""

    # Visual constants (matching visualize_board_coords.py)
    HEX_SIZE = 1.0  # Hex grid size
    RING_RADIUS = 0.866 * HEX_SIZE  # Ring radius (sqrt(3)/2 ≈ 0.866)
    MARBLE_RADIUS = 0.6 * HEX_SIZE  # Marble radius (20% bigger: 0.5 * 1.2)

    # Colors
    RING_COLOR = "#000000"  # Brown/wood color for rings
    RING_EDGE_COLOR = "#000000"  # Black for present ring edges
    REMOVED_RING_COLOR = "#AAAAAA"  # Light gray for removed rings
    INNER_RING_COLOR = "#888888"  # Mid-to-dark gray for inner rings on empty positions
    INNER_RING_RATIO = 0.4  # Inner ring radius as ratio of outer ring (1/3 marble radius)

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

    def __init__(self, internal_coords: bool = False, edge_coords: bool = False, show_removed: bool = False, show_captures: bool = False, bg_color: Optional[str] = None):
        """Initialize board renderer.

        Args:
            internal_coords: If True, display coordinate labels centered on rings
            edge_coords: If True, display coordinate labels at top/bottom edges of columns
            show_removed: If True, show removed rings with transparency
            show_captures: If True, display player capture counts at bottom corners
            bg_color: Background color in #RRGGBB or #RRGGBBAA format (default: #F5E6D3)

        Note:
            internal_coords and edge_coords can both be enabled simultaneously.
            When both are enabled, edge positions show edge coords (outside the ring),
            while non-edge positions show internal coords (centered on ring).
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for board rendering. "
                "Install it with: pip install matplotlib"
            )

        self.internal_coords = internal_coords
        self.edge_coords = edge_coords
        self.show_removed = show_removed
        self.show_captures = show_captures
        self.bg_color = bg_color if bg_color is not None else self.BACKGROUND_COLOR

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
        width: int = 1024,
        height: int = 1024,
        transparent: bool = False,
    ) -> plt.Figure:
        """Render a board state as a matplotlib figure.

        Args:
            board: ZertzBoard instance to render
            title: Optional title for the figure
            width: Figure width in pixels (default: 1024)
            height: Figure height in pixels (default: 1024)
            transparent: If True, use transparent background (default: False)

        Returns:
            matplotlib Figure object
        """
        # Convert pixels to inches for matplotlib (using DPI=100)
        DPI = 100
        figsize = (width / DPI, height / DPI)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')

        # Set background colors (or transparent if requested)
        if transparent:
            ax.set_facecolor('none')
            fig.patch.set_facecolor('none')
        else:
            ax.set_facecolor(self.bg_color)
            fig.patch.set_facecolor(self.bg_color)

        # Get board dimensions
        board_width = board.width

        # Calculate scale factor for line widths based on figure size
        # Default is 1024 pixels, scale linewidths proportionally
        default_size = 1024.0
        scale_factor = width / default_size

        # Collect patches for rings and marbles, and text labels
        ring_patches = []
        marble_patches = []
        text_labels = []  # Store (px, py, pos_str, text_color, va, weight) tuples

        # If edge_coords is enabled, find the top and bottom edge of each column
        edge_ring_positions = {}  # Maps (y, x) -> 'top' or 'bottom'
        if self.edge_coords:
            # First pass: find all valid ring positions grouped by column (x)
            columns = {}  # Maps x -> list of y values
            for y in range(board_width):
                for x in range(board_width):
                    try:
                        if board.letter_layout is not None and board.letter_layout[y][x]:
                            if x not in columns:
                                columns[x] = []
                            columns[x].append(y)
                    except IndexError:
                        continue

            # For each column, find min and max y and mark as top/bottom edges
            for x, y_values in columns.items():
                if y_values:
                    min_y = min(y_values)
                    max_y = max(y_values)
                    edge_ring_positions[(min_y, x)] = 'top'
                    edge_ring_positions[(max_y, x)] = 'bottom'

        # Iterate through board positions
        for y in range(board_width):
            for x in range(board_width):
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
                px, py = self._get_hex_position(y, x, board_width)

                # Draw ring as circle (if exists or show_removed is True)
                if ring_exists or self.show_removed:
                    # Use solid black for present rings, light gray for removed rings (no alpha)
                    ring_color = self.RING_EDGE_COLOR if ring_exists else self.REMOVED_RING_COLOR
                    # Present rings get higher zorder so they draw on top of removed rings
                    ring_zorder = 1 if ring_exists else 0.5
                    # Removed rings get thinner lines
                    ring_linewidth = (2 * scale_factor) if ring_exists else (1 * scale_factor)
                    ring = patches.Circle(
                        (px, py),
                        radius=self.RING_RADIUS,
                        fill=False,  # Rings are hollow (just the outline)
                        edgecolor=ring_color,
                        linewidth=ring_linewidth,
                        zorder=ring_zorder,
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
                    else:
                        # Empty ring - add inner ring with thinner line in mid-to-dark gray
                        inner_ring = patches.Circle(
                            (px, py),
                            radius=self.RING_RADIUS * self.INNER_RING_RATIO,
                            fill=False,
                            edgecolor=self.INNER_RING_COLOR,
                            linewidth=1 * scale_factor,  # Thinner than outer ring
                            zorder=2,
                        )
                        ring_patches.append(inner_ring)

                # Store text label info for rendering AFTER marbles
                # Use white text on gray/black marbles, black text otherwise

                # Check if we should show this coordinate
                # Edge coords take precedence over internal coords at edge positions
                show_this_coord = False
                edge_position = None

                if self.edge_coords and (y, x) in edge_ring_positions:
                    # Show edge coords at top/bottom of columns
                    show_this_coord = True
                    edge_position = edge_ring_positions[(y, x)]
                    if edge_position == 'top':
                        va = 'bottom'  # Position label above ring (text bottom aligns to point above ring)
                        label_py = py + self.RING_RADIUS * 1.3  # Position above ring
                    else:  # 'bottom'
                        va = 'top'  # Position label below ring (text top aligns to point below ring)
                        label_py = py - self.RING_RADIUS * 1.3  # Position below ring
                elif self.internal_coords and ring_exists:
                    # Show internal coords, centered on ring
                    show_this_coord = True
                    va = 'center'

                if show_this_coord:
                    if edge_position:
                        # Edge coords always black, non-bold
                        text_color = self.COORD_COLOR
                        text_labels.append((px, label_py, pos_str, text_color, va, 'normal'))
                    else:
                        # Regular coords: white on dark marbles, black otherwise, bold
                        text_color = self.COORD_COLOR_ON_DARK if (ring_exists and marble_type in ['g', 'b']) else self.COORD_COLOR
                        text_labels.append((px, py, pos_str, text_color, 'center', 'bold'))

        # Add all patches to axis (rings, then marbles, then text on top)
        for patch in ring_patches:
            ax.add_patch(patch)
        for patch in marble_patches:
            ax.add_patch(patch)

        # Render text labels on top of everything
        for px, py, pos_str, text_color, va, weight in text_labels:
            ax.text(
                px, py,
                pos_str,
                ha='center', va=va,
                fontsize=self.COORD_FONT_SIZE * scale_factor,
                color=text_color,
                zorder=4,  # Above marbles (zorder=3)
                weight=weight,
            )

        # Collect board extent for axis limits
        all_x_coords = []
        all_y_coords = []
        for y in range(board_width):
            for x in range(board_width):
                try:
                    if board.letter_layout is not None and board.letter_layout[y][x]:
                        px, py = self._get_hex_position(y, x, board_width)
                        all_x_coords.append(px)
                        all_y_coords.append(py)
                except IndexError:
                    continue

        # Render capture counts if enabled
        if self.show_captures and all_x_coords and all_y_coords:
            # Calculate positions for capture displays (bottom corners, outside board)
            min_x = min(all_x_coords)
            max_x = max(all_x_coords)
            min_y = min(all_y_coords)

            # Size of capture indicator marbles (2/3 of board marbles)
            capture_marble_radius = self.MARBLE_RADIUS * 0.67

            # Horizontal spacing between marble types (tighter packing)
            horizontal_spacing = capture_marble_radius * 2.2

            # Y position (closer to the board now)
            capture_y = min_y - self.RING_RADIUS * 1.3

            # Extract capture counts from global_state
            # Player 1: indices 3-5 (w, g, b)
            # Player 2: indices 6-8 (w, g, b)
            p1_captures = {
                'w': int(board.global_state[board.P1_CAP_W]),
                'g': int(board.global_state[board.P1_CAP_G]),
                'b': int(board.global_state[board.P1_CAP_B]),
            }
            p2_captures = {
                'w': int(board.global_state[board.P2_CAP_W]),
                'g': int(board.global_state[board.P2_CAP_G]),
                'b': int(board.global_state[board.P2_CAP_B]),
            }

            # Render Player 1 captures (bottom left, angled at 30 degrees)
            # Calculate vertical offset for 30-degree slope alignment
            vertical_offset = horizontal_spacing * 0.577  # tan(30°) ≈ 0.577

            p1_base_x = min_x
            for idx, marble_type in enumerate(['w', 'g', 'b']):
                count = p1_captures[marble_type]
                marble_x = p1_base_x + (idx * horizontal_spacing)

                # Black marble (idx=2) stays at base position, others offset upward
                marble_y = capture_y + ((2 - idx) * vertical_offset)

                # Draw marble circle
                marble = patches.Circle(
                    (marble_x, marble_y),
                    radius=capture_marble_radius,
                    facecolor=self.MARBLE_COLORS[marble_type],
                    edgecolor=self.MARBLE_EDGE_COLOR,
                    linewidth=1.5 * scale_factor,
                    zorder=5,
                )
                ax.add_patch(marble)

                # Draw count text inside marble (white on dark, black on light)
                text_color = self.COORD_COLOR_ON_DARK if marble_type in ['g', 'b'] else self.COORD_COLOR
                ax.text(
                    marble_x, marble_y,
                    str(count),
                    ha='center', va='center',
                    fontsize=self.COORD_FONT_SIZE * 0.9 * scale_factor,
                    color=text_color,
                    zorder=6,
                    weight='bold',
                )

            # Render Player 2 captures (bottom right, angled at 30 degrees)
            p2_base_x = max_x
            for idx, marble_type in enumerate(['w', 'g', 'b']):
                count = p2_captures[marble_type]
                # Reverse order: start from right and go left
                marble_x = p2_base_x - (idx * horizontal_spacing)

                # Black marble (idx=2) stays at base position, others offset upward
                marble_y = capture_y + ((2 - idx) * vertical_offset)

                # Draw marble circle
                marble = patches.Circle(
                    (marble_x, marble_y),
                    radius=capture_marble_radius,
                    facecolor=self.MARBLE_COLORS[marble_type],
                    edgecolor=self.MARBLE_EDGE_COLOR,
                    linewidth=1.5 * scale_factor,
                    zorder=5,
                )
                ax.add_patch(marble)

                # Draw count text inside marble (white on dark, black on light)
                text_color = self.COORD_COLOR_ON_DARK if marble_type in ['g', 'b'] else self.COORD_COLOR
                ax.text(
                    marble_x, marble_y,
                    str(count),
                    ha='center', va='center',
                    fontsize=self.COORD_FONT_SIZE * 0.9 * scale_factor,
                    color=text_color,
                    zorder=6,
                    weight='bold',
                )

        if all_x_coords and all_y_coords:
            # Border = full ring radius + 1/4 hex width trim
            # This shows complete rings at edges with minimal whitespace
            border = self.RING_RADIUS + 0.25

            # When edge_coords is enabled, add extra space at bottom and top for labels
            if self.edge_coords:
                border_top = border + self.RING_RADIUS * 0.6  # Extra space for top labels
                border_bottom = border + self.RING_RADIUS * 0.6  # Extra space for bottom labels
                ax.set_xlim(min(all_x_coords) - border, max(all_x_coords) + border)
                ax.set_ylim(min(all_y_coords) - border_bottom, max(all_y_coords) + border_top)
            else:
                ax.set_xlim(min(all_x_coords) - border, max(all_x_coords) + border)
                ax.set_ylim(min(all_y_coords) - border, max(all_y_coords) + border)

        ax.axis('off')

        # Add title if provided (1.5x the coordinate font size, scaled, always bold)
        if title:
            ax.set_title(title, fontsize=int(self.COORD_FONT_SIZE * 1.5 * scale_factor), pad=20, weight='bold')

        plt.tight_layout()
        return fig

    def save_board(
        self,
        board: ZertzBoard,
        output_path: Union[str, Path],
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        transparent: bool = False,
    ) -> None:
        """Render and save a board state to a file.

        Args:
            board: ZertzBoard instance to render
            output_path: Path to save the image (PNG/SVG supported)
            title: Optional title for the figure
            width: Figure width in pixels (if None, defaults to 1024 or matches height)
            height: Figure height in pixels (if None, defaults to 1024 or matches width)
            transparent: If True, use transparent background (default: False)
        """
        # Handle optional width/height - default to 1024, or match the provided dimension
        default_pixels = 1024
        if width is None and height is None:
            output_width = default_pixels
            output_height = default_pixels
        elif width is not None and height is None:
            output_width = width
            output_height = width
        elif width is None and height is not None:
            output_width = height
            output_height = height
        else:
            output_width = width
            output_height = height

        fig = self.render_board(board, title=title, width=output_width, height=output_height, transparent=transparent)

        # Detect format from extension - matplotlib handles SVG natively
        # Use fixed DPI=100 internally for saving
        output_path = Path(output_path)
        if transparent:
            # Transparent background
            if output_path.suffix.lower() == '.svg':
                fig.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
            else:
                fig.savefig(output_path, dpi=100, bbox_inches='tight', transparent=True)
        else:
            # Solid background
            if output_path.suffix.lower() == '.svg':
                fig.savefig(output_path, format='svg', bbox_inches='tight', facecolor=fig.get_facecolor())
            else:
                fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())

        plt.close(fig)

    def show_board(
        self,
        board: ZertzBoard,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Render and display a board state on screen.

        Args:
            board: ZertzBoard instance to render
            title: Optional title for the figure
            width: Figure width in pixels (if None, defaults to 1024 or matches height)
            height: Figure height in pixels (if None, defaults to 1024 or matches width)
        """
        # Handle optional width/height - default to 1024, or match the provided dimension
        default_pixels = 1024
        if width is None and height is None:
            output_width = default_pixels
            output_height = default_pixels
        elif width is not None and height is None:
            output_width = width
            output_height = width
        elif width is None and height is not None:
            output_width = height
            output_height = height
        else:
            output_width = width
            output_height = height

        fig = self.render_board(board, title=title, width=output_width, height=output_height)
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
        except ValueError:
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
            # Load from file (auto-detects format: notation, transcript, or SGF)
            loader = AutoSelectLoader(str(path))
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

    # Execute moves (respecting turn order, not strict alternation)
    # Chain captures result in multiple consecutive actions for one player
    move_count = 0
    p1_idx = 0
    p2_idx = 0

    while p1_idx < len(player1_actions) or p2_idx < len(player2_actions):
        if stop_at_move is not None and move_count >= stop_at_move:
            break

        # Check whose turn it is (matches main replay system behavior)
        current_player = game.get_cur_player_value()

        if current_player == 1:
            # Player 1's turn
            if p1_idx >= len(player1_actions):
                raise ValueError(f"Player 1 out of actions (move {move_count}, expected player 1)")

            action_dict = player1_actions[p1_idx]
            p1_idx += 1
        else:
            # Player 2's turn
            if p2_idx >= len(player2_actions):
                raise ValueError(f"Player 2 out of actions (move {move_count}, expected player 2)")

            action_dict = player2_actions[p2_idx]
            p2_idx += 1

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
    internal_coords: bool = True,
    edge_coords: bool = False,
    show_removed: bool = False,
    show_captures: bool = False,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    dpi: int = 150,
    stop_at_move: Optional[int] = None,
    bg_color: Optional[str] = None,
) -> None:
    """Render a board state from notation sequence.

    This is the main entry point for the utility. It parses notation,
    executes the moves, and renders the final board state.

    Args:
        notation_input: Either a file path or a notation string
        output_path: Optional path to save the image (PNG recommended)
        show: If True, display the image on screen
        internal_coords: If True, display coordinate labels centered on rings
        edge_coords: If True, display coordinate labels at top/bottom edges of columns
        show_removed: If True, show removed rings with transparency
        show_captures: If True, display player capture counts at bottom corners
        title: Optional title for the figure
        width: Output width in pixels (if None, defaults to 1024)
        height: Output height in pixels (if None, defaults to 1024)
        dpi: DEPRECATED - kept for backward compatibility but ignored (fixed at 100 internally)
        stop_at_move: Optional move number to stop at (0-indexed)
        bg_color: Background color in #RRGGBB or #RRGGBBAA format (default: #F5E6D3)

    Width/Height behavior:
        - Both None: Use default (1024x1024 pixels)
        - Width only: Height set to match width (square image)
        - Height only: Width set to match height (square image)
        - Both specified: Use both (may distort aspect ratio if not equal)

    Example:
        # From file, save to PNG with default size (1024x1024 pixels)
        render_board_from_notation("game.txt", output_path="board.png")

        # Custom width in pixels (height auto-set to match)
        render_board_from_notation("game.txt", output_path="board.png", width=1200)

        # Custom dimensions in pixels
        render_board_from_notation("game.txt", output_path="board.png", width=1800, height=1800)

        # From string, display on screen
        notation = "37\\nWd4\\nGe3,a1"
        render_board_from_notation(notation, show=True, title="Position after 2 moves")

        # Stop at specific move
        render_board_from_notation("game.txt", output_path="move_10.png", stop_at_move=10)
    """
    # Convert to pixel dimensions (DPI parameter is deprecated but kept for compatibility)
    default_pixels = 1024

    if width is None and height is None:
        # Both omitted: use defaults
        output_width = default_pixels
        output_height = default_pixels
    elif width is not None and height is None:
        # Width only: maintain aspect ratio (board is square)
        output_width = width
        output_height = width
    elif width is None and height is not None:
        # Height only: maintain aspect ratio (board is square)
        output_width = height
        output_height = height
    else:
        # Both specified: use as-is
        output_width = width
        output_height = height

    # Execute notation to get board state
    board = execute_notation_sequence(notation_input, stop_at_move=stop_at_move)

    # Create renderer
    renderer = DiagramRenderer(internal_coords=internal_coords, edge_coords=edge_coords, show_removed=show_removed, show_captures=show_captures, bg_color=bg_color)

    # Save or show
    if output_path:
        renderer.save_board(board, output_path, title=title, width=output_width, height=output_height)

    if show:
        renderer.show_board(board, title=title, width=output_width, height=output_height)

    if not output_path and not show:
        # Default to showing if no output specified
        renderer.show_board(board, title=title, width=output_width, height=output_height)