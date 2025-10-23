"""Game constants shared across modules.

This module contains outcome constants and other shared game values
used by both stateful (ZertzGame) and stateless (zertz_logic) implementations.
"""

# Game outcome constants
PLAYER_1_WIN = 1
PLAYER_2_WIN = -1
TIE = 0
BOTH_LOSE = -2  # Tournament rule: both players lose (collaboration detected)

# Win condition sets
STANDARD_WIN_CONDITIONS = [{"w": 3, "g": 3, "b": 3}, {"w": 4}, {"g": 5}, {"b": 6}]
BLITZ_WIN_CONDITIONS = [{"w": 2, "g": 2, "b": 2}, {"w": 3}, {"g": 4}, {"b": 5}]

# Marble supply configurations
STANDARD_MARBLES = {"w": 6, "g": 8, "b": 10}
BLITZ_MARBLES = {"w": 5, "g": 7, "b": 9}