"""Official Zèrtz notation formatter.

Converts between action dictionaries and official Zèrtz notation format
as specified at http://www.gipf.com/zertz/notations/notation.html
"""

import re


class NotationFormatter:
    """Converts actions to/from official Zèrtz notation."""

    # Letter-to-index mapping (A=0, B=1, ..., H=7, J=8, skipping I)
    _LETTER_TO_INDEX = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8
    }

    # Index-to-letter mapping
    _INDEX_TO_LETTER = {v: k for k, v in _LETTER_TO_INDEX.items()}

    @staticmethod
    def _calculate_cap_position(src: str, dst: str) -> str:
        """Calculate the middle position between src and dst.

        Works for all board sizes including 61-ring (which uses ABCDEFGHJ, skipping I).

        Args:
            src: Source position (e.g., "c2", "H3")
            dst: Destination position (e.g., "e4", "J3")

        Returns:
            str: Middle position in same case as input (e.g., "d3", "J3")
        """
        # Remember original case
        preserve_case = src[0].islower()

        # Parse source and destination (work in uppercase)
        src_letter, src_num = src[0].upper(), int(src[1:])
        dst_letter, dst_num = dst[0].upper(), int(dst[1:])

        # Calculate middle position
        src_idx = NotationFormatter._LETTER_TO_INDEX[src_letter]
        dst_idx = NotationFormatter._LETTER_TO_INDEX[dst_letter]

        mid_idx = (src_idx + dst_idx) // 2
        mid_num = (src_num + dst_num) // 2

        result = NotationFormatter._INDEX_TO_LETTER[mid_idx] + str(mid_num)

        # Return in original case
        return result.lower() if preserve_case else result

    @staticmethod
    def action_to_notation(action_dict: dict, action_result=None) -> str:
        """Convert action_dict to official Zèrtz notation.

        Notation format from http://www.gipf.com/zertz/notations/notation.html:
        - Placement: [Color][coord] or [Color][coord],[removed_coord]
          Examples: "Wd4" or "Bd7,b2"
        - Placement with isolation: [Color][coord],[removed_coord] x [Color][pos]...
          Example: "Bd7,b2 x Wa1Wa2"
        - Capture: x [src][captured_color][dst]
          Example: "x e3Wg3"
        - Pass: "-"

        Args:
            action_dict: Dictionary with action details
            action_result: Optional ActionResult object (for isolation captures)

        Returns:
            str: Notation string
        """
        if action_dict["action"] == "PASS":
            return "-"

        if action_dict["action"] == "PUT":
            # Convert marble color to uppercase
            marble = action_dict["marble"].upper()
            # Convert destination to lowercase
            dst = action_dict["dst"].lower()

            # Check if a ring was removed
            if action_dict["remove"]:
                remove = action_dict["remove"].lower()
                notation = f"{marble}{dst},{remove}"
            else:
                notation = f"{marble}{dst}"

            # Add isolation captures if any
            if action_result and action_result.is_isolation():
                isolated_parts = []
                for removal in action_result.captured_marbles:
                    if removal["marble"]:
                        color = removal["marble"].upper()
                        pos = removal["pos"].lower()
                        isolated_parts.append(f"{color}{pos}")
                if isolated_parts:
                    notation += " x " + "".join(isolated_parts)

            return notation

        elif action_dict["action"] == "CAP":
            # Convert to lowercase and get captured marble as uppercase
            src = action_dict["src"].lower()
            dst = action_dict["dst"].lower()
            captured = action_dict["capture"].upper()
            return f"x {src}{captured}{dst}"

        return ""

    @staticmethod
    def notation_to_action_dict(notation_str: str) -> dict:
        """Parse official Zèrtz notation to action dictionary.

        Args:
            notation_str: Notation string (e.g., "Wd4", "x c2We3", "-")

        Returns:
            dict: Action dictionary with fields:
                - action: "PASS", "PUT", or "CAP"
                - marble: Marble color (for PUT only)
                - dst: Destination position (for PUT/CAP)
                - remove: Removed ring position (for PUT, empty string if none)
                - src: Source position (for CAP only)
                - capture: Captured marble color (for CAP only)
                - cap: Captured ring position (for CAP only, calculated from src/dst)
        """
        notation_str = notation_str.strip()

        # Handle pass
        if notation_str == "-":
            return {"action": "PASS"}

        # Handle capture: "x c2We3" or "x e3Wg3"
        if notation_str.startswith("x "):
            return NotationFormatter._parse_capture(notation_str)

        # Handle placement: "Wd4" or "Bd7,g4" or "Wf2,d4 x Gd5"
        # Strip off isolation captures if present (we don't store them in action dict)
        if " x " in notation_str:
            notation_str = notation_str.split(" x ")[0]

        return NotationFormatter._parse_placement(notation_str)

    @staticmethod
    def _parse_placement(notation_str):
        """Parse placement notation: 'Wd4' or 'Bd7,g4'

        Args:
            notation_str: Placement notation string

        Returns:
            dict: Action dictionary for PUT action
        """
        # Pattern: [Color][position] or [Color][position],[remove_position]
        # Color is single letter (W/G/B), positions are letter+number
        match = re.match(r"([WGB])([a-jA-J]\d+)(?:,([a-jA-J]\d+))?", notation_str)
        if not match:
            raise ValueError(f"Invalid placement notation: {notation_str}")

        color = match.group(1).lower()
        dst = match.group(2).lower()  # Keep lowercase per GIPF notation rules
        remove = match.group(3).lower() if match.group(3) else ""  # Keep lowercase

        return {
            "action": "PUT",
            "marble": color,
            "dst": dst,
            "remove": remove,
        }

    @staticmethod
    def _parse_capture(notation_str):
        """Parse capture notation: 'x c2We3'

        Args:
            notation_str: Capture notation string

        Returns:
            dict: Action dictionary for CAP action.
                 'cap' position is calculated from src/dst using string interpolation.
        """
        # Remove "x " prefix
        notation_str = notation_str[2:].strip()

        # Pattern: [src_pos][captured_color][dst_pos]
        # Positions are letter+number, color is single letter
        match = re.match(r"([a-jA-J]\d+)([WGB])([a-jA-J]\d+)", notation_str)
        if not match:
            raise ValueError(f"Invalid capture notation: x {notation_str}")

        src = match.group(1).lower()  # Keep lowercase per GIPF notation rules
        captured_color = match.group(2).lower()
        dst = match.group(3).lower()  # Keep lowercase per GIPF notation rules

        # Calculate cap position from src and dst using string interpolation
        # Pass lowercase coordinates so cap is also lowercase
        cap = NotationFormatter._calculate_cap_position(src, dst)

        # cap position calculated from src/dst geometry
        return {
            "action": "CAP",
            "src": src,
            "dst": dst,
            "capture": captured_color,
            "cap": cap,
        }