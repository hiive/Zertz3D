"""Transcript action formatter for Zertz3D.

Converts between action dictionaries and transcript file string format.
Transcript format: "{'action': 'PUT', 'marble': 'w', 'dst': 'D4', 'remove': ''}"
"""


class TranscriptFormatter:
    """Converts actions to/from transcript file string format."""

    @staticmethod
    def action_to_transcript(action_dict: dict) -> str:
        """Convert action_dict to transcript string format.

        Args:
            action_dict: Dictionary with action details

        Returns:
            str: String representation of action dict (e.g., "{'action': 'PUT', ...}")
        """
        return str(action_dict)

    @staticmethod
    def transcript_to_action_dict(transcript_str: str) -> dict:
        """Parse transcript string to action dictionary.

        Args:
            transcript_str: String representation of action dict

        Returns:
            dict: Action dictionary

        Raises:
            ValueError: If the string cannot be safely parsed
            SyntaxError: If the string is not valid Python
        """
        import ast
        return ast.literal_eval(transcript_str)