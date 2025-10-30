#!/usr/bin/env python3
"""
Utility for combining multiple board diagrams into grid layouts.

Supports both PNG and SVG formats with transparent backgrounds for individual images
and proper background color for the combined grid.
"""

import tempfile
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


class GridRenderer:
    """Handles combining multiple board diagrams into grid layouts."""

    def __init__(self, image_columns: int = 6, background_color: Tuple[int, int, int] | str | None = None, show_row_dividers: bool = False):
        """
        Initialize GridRenderer.

        Args:
            image_columns: Number of images per row in grid (default: 6)
            background_color: RGB tuple for grid background (default: dark gray)
            show_row_dividers: If True, draw thin dividers between rows (default: False)
        """
        self.image_columns = image_columns
        self.show_row_dividers = show_row_dividers

        if isinstance(background_color, str):
            bg_hex = background_color.lstrip('#')
            bg_rgb = tuple(int(bg_hex[i:i + 2], 16) for i in (0, 2, 4))
            background_color = bg_rgb
        self.background_color = background_color

    def create_svg_grid(self, svg_files: List[str], output_path: str) -> None:
        """
        Combine multiple SVG files into a single grid layout.

        Args:
            svg_files: List of paths to individual SVG files
            output_path: Path where combined SVG should be saved
        """
        if not svg_files:
            raise ValueError("No SVG files provided")

        # Parse first SVG to get dimensions
        tree = ET.parse(svg_files[0])
        root = tree.getroot()

        # Extract viewBox or width/height
        viewbox = root.get('viewBox')
        if viewbox:
            _, _, svg_width, svg_height = map(float, viewbox.split())
        else:
            svg_width = float(root.get('width', 1024))
            svg_height = float(root.get('height', 1024))

        # Calculate grid dimensions
        num_files = len(svg_files)
        num_cols = min(self.image_columns, num_files)
        num_rows = (num_files + num_cols - 1) // num_cols  # Ceiling division

        total_width = num_cols * svg_width
        total_height = num_rows * svg_height

        # Create new SVG root with background
        combined_svg = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'viewBox': f'0 0 {total_width} {total_height}',
            'width': str(total_width),
            'height': str(total_height)
        })

        # Add background rectangle
        bg_rect = ET.SubElement(combined_svg, 'rect', {
            'width': str(total_width),
            'height': str(total_height),
            'fill': self._rgb_to_hex(self.background_color)
        })

        # Add each SVG in grid position
        for idx, svg_file in enumerate(svg_files):
            row = idx // num_cols
            col = idx % num_cols
            x_offset = col * svg_width
            y_offset = row * svg_height

            # Parse this SVG
            tree = ET.parse(svg_file)
            svg_root = tree.getroot()

            # Create a group with transform for this position
            g = ET.SubElement(combined_svg, 'g', {
                'transform': f'translate({x_offset}, {y_offset})'
            })

            # Copy all children from source SVG to group
            for child in svg_root:
                g.append(child)

        # Add row dividers (horizontal lines between rows) if enabled
        if self.show_row_dividers:
            divider_color = 'rgba(0,0,0,0.2)'
            divider_width = 1
            for row_idx in range(1, num_rows):
                y_pos = row_idx * svg_height
                ET.SubElement(combined_svg, 'line', {
                    'x1': '0',
                    'y1': str(y_pos),
                    'x2': str(total_width),
                    'y2': str(y_pos),
                    'stroke': divider_color,
                    'stroke-width': str(divider_width)
                })

        # Save combined SVG
        tree = ET.ElementTree(combined_svg)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

    def create_png_grid(self, png_files: List[str], output_path: str) -> None:
        """
        Combine multiple PNG files into a single grid layout.

        Args:
            png_files: List of paths to individual PNG files
            output_path: Path where combined PNG should be saved
        """
        if not png_files:
            raise ValueError("No PNG files provided")

        # Load all images
        images = [Image.open(png_file) for png_file in png_files]

        # Calculate grid dimensions
        num_images = len(images)
        num_cols = min(self.image_columns, num_images)
        num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division

        img_width = images[0].width
        img_height = images[0].height
        total_width = num_cols * img_width
        total_height = num_rows * img_height

        # Create combined image with background color (RGBA for transparency support)
        # self.background_color is already a tuple from __init__
        combined_img = Image.new('RGBA', (total_width, total_height), self.background_color + (255,))

        # Paste images in grid layout
        for idx, img in enumerate(images):
            row = idx // num_cols
            col = idx % num_cols
            x_offset = col * img_width
            y_offset = row * img_height

            # Handle transparency
            if img.mode == 'RGBA':
                # Paste with alpha channel
                combined_img.paste(img, (x_offset, y_offset), img)
            else:
                # Convert to RGBA if needed
                img_rgba = img.convert('RGBA')
                combined_img.paste(img_rgba, (x_offset, y_offset))

        # Draw row dividers (horizontal lines between rows) if enabled
        if self.show_row_dividers:
            draw = ImageDraw.Draw(combined_img, 'RGBA')
            divider_color = (0, 0, 0, 51)  # RGBA(0,0,0,0.2) -> alpha=51/255≈0.2
            for row_idx in range(1, num_rows):
                y_pos = row_idx * img_height
                draw.line([(0, y_pos), (total_width, y_pos)], fill=divider_color, width=1)

        # Convert back to RGB for saving (flattens transparency onto background)
        final_img = Image.new('RGB', (total_width, total_height), self.background_color)
        final_img.paste(combined_img, (0, 0), combined_img)

        # Save combined image
        final_img.save(output_path)

    def create_grid_from_temp_files(
        self,
        temp_files: List[str],
        output_path: str,
        svg: bool = False
    ) -> None:
        """
        Create a grid from temporary files and clean them up.

        Args:
            temp_files: List of temporary file paths
            output_path: Path where combined image should be saved
            svg: If True, treat as SVG files; otherwise PNG
        """
        try:
            if svg:
                self.create_svg_grid(temp_files, output_path)
            else:
                self.create_png_grid(temp_files, output_path)
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink()
                except Exception:
                    pass  # Ignore cleanup errors

    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int] | str | None) -> str:
        if rgb is None:
            return "#00000000"
        elif isinstance(rgb, str):
            return rgb
        """Convert RGB tuple to hex color string."""
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'