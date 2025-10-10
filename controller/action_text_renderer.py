"""Move display formatting for Zertz 3D.

Formats valid moves and game state information for display.
"""

import numpy as np


class ActionTextRenderer:
    """Formats valid moves and game state for display."""

    def format_valid_actions(self, game, player):
        """Format valid move information for the current player.

        Args:
            game: ZertzGame instance
            player: The current player

        Returns:
            str: Formatted text displaying valid moves
        """
        placement, capture = game.get_valid_actions()

        lines = []
        lines.append(f"\n--- Valid Moves for Player {player.n} ---")

        # Count valid moves
        num_placement = np.sum(placement)
        num_capture = np.sum(capture)

        lines.append(f"Placement moves: {num_placement}")
        lines.append(f"Capture moves: {num_capture}")

        # If there are captures, they are mandatory
        if num_capture > 0:
            lines.append("CAPTURES ARE MANDATORY")
            capture_lines = self._format_capture_actions(game, capture)
            lines.extend(capture_lines)
        else:
            # Show placement information
            placement_lines = self._format_placement_actions(game, placement)
            lines.extend(placement_lines)

        # Show current captured marbles
        lines.append(f"\nPlayer {player.n} captured: {player.captured}")
        lines.append("---" + "-" * 30 + "\n")

        return "\n".join(lines)

    def _format_capture_actions(self, game, capture):
        """Format capture moves for display.

        Args:
            game: ZertzGame instance
            capture: Capture array from get_valid_actions()

        Returns:
            list[str]: Lines of formatted capture move text
        """
        lines = []
        capture_positions = np.argwhere(capture)

        if len(capture_positions) > 0:
            lines.append("Capture moves available:")
            for i, (direction, y, x) in enumerate(capture_positions[:10]):  # Show up to 10
                try:
                    _, action_dict = game.action_to_str("CAP", (direction, y, x))
                    marble = action_dict['marble']
                    src = action_dict['src']
                    dst = action_dict['dst']
                    captured = action_dict['capture']
                    cap_pos = action_dict['cap']
                    lines.append(f"  - CAP {marble} {src} -> {dst} capturing {captured} at {cap_pos}")
                except (IndexError, KeyError):
                    # Fallback to basic info if conversion fails
                    src_str = game.board.index_to_str((y, x))
                    lines.append(f"  - Jump from {src_str} (direction {direction})")

            if len(capture_positions) > 10:
                lines.append(f"  ... and {len(capture_positions) - 10} more")

        return lines

    def _format_placement_actions(self, game, placement):
        """Format placement moves for display.

        Args:
            game: ZertzGame instance
            placement: Placement array from get_valid_actions()

        Returns:
            list[str]: Lines of formatted placement move text
        """
        lines = []
        placement_positions = np.argwhere(placement)

        if len(placement_positions) > 0:
            # Group by (marble, destination) to find unique placement positions
            marble_types = ['w', 'g', 'b']
            placements = {}  # {(marble, dst_str): [list of removal positions]}
            removals_set = set()  # Track all possible removals

            for marble_idx, dst, rem in placement_positions:
                marble = marble_types[marble_idx]
                dst_y = dst // game.board.width
                dst_x = dst % game.board.width
                dst_str = game.board.index_to_str((dst_y, dst_x))

                key = (marble, dst_str)
                if key not in placements:
                    placements[key] = []

                if rem != game.board.width ** 2:
                    rem_y = rem // game.board.width
                    rem_x = rem % game.board.width
                    rem_str = game.board.index_to_str((rem_y, rem_x))
                    placements[key].append(rem_str)
                    removals_set.add(rem_str)

            lines.append(f"\nPlacement positions ({len(placements)} unique):")
            for i, ((marble, dst_str), _) in enumerate(list(placements.items())[:10]):
                lines.append(f"  - PUT {marble} at {dst_str}")
            if len(placements) > 10:
                lines.append(f"  ... and {len(placements) - 10} more")

            if removals_set:
                lines.append(f"\nRings available to remove ({len(removals_set)}):")
                removals_list = sorted(list(removals_set))
                # Show first 10
                lines.append(f"  {', '.join(removals_list[:10])}")
                if len(removals_list) > 10:
                    lines.append(f"  ... and {len(removals_list) - 10} more")

        return lines