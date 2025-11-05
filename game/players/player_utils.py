"""Utilities for working with player information."""


def format_player_name(player_num: int, player_name: str | None = None) -> str:
    """Format a player display name with optional custom name.

    Args:
        player_num: Player number (1 or 2)
        player_name: Optional custom player name

    Returns:
        Formatted string like "Player 1" or "Player 1 (Alice)"

    Examples:
        >>> format_player_name(1)
        'Player 1'
        >>> format_player_name(1, "Alice")
        'Player 1 (Alice)'
        >>> format_player_name(2, None)
        'Player 2'
    """
    base = f"Player {player_num}"
    if player_name:
        return f"{base} ({player_name})"
    return base