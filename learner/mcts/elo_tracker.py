"""ELO rating system for tracking player strength.

Standard chess-style ELO with configurable K-factor.
"""

import json
from pathlib import Path
from datetime import datetime


class EloTracker:
    """Track and update ELO ratings for players."""

    def __init__(self, k_factor=32, initial_rating=1500):
        """Initialize ELO tracker.

        Args:
            k_factor: Rating adjustment speed (32 for new players, 16 for established)
            initial_rating: Starting rating for new players (1500 is standard)
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}  # player_id -> current rating
        self.history = []  # List of games with ratings

    def get_rating(self, player_id):
        """Get current rating for player (creates entry if new)."""
        if player_id not in self.ratings:
            self.ratings[player_id] = self.initial_rating
        return self.ratings[player_id]

    def expected_score(self, rating_a, rating_b):
        """Calculate expected score (probability of winning) for player A.

        Args:
            rating_a: Player A's rating
            rating_b: Player B's rating

        Returns:
            Expected score for player A (0.0 to 1.0)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, player_a_id, player_b_id, outcome):
        """Update ratings based on game outcome.

        Args:
            player_a_id: Identifier for player A
            player_b_id: Identifier for player B
            outcome: Game result from player A's perspective:
                     1.0 = A won, 0.5 = draw, 0.0 = A lost

        Returns:
            dict with rating changes and new ratings
        """
        # Get current ratings
        rating_a = self.get_rating(player_a_id)
        rating_b = self.get_rating(player_b_id)

        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        # Calculate actual scores (outcome is from A's perspective)
        actual_a = outcome
        actual_b = 1.0 - outcome

        # Update ratings
        new_rating_a = rating_a + self.k_factor * (actual_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_b - expected_b)

        # Store new ratings
        self.ratings[player_a_id] = new_rating_a
        self.ratings[player_b_id] = new_rating_b

        # Record in history
        game_record = {
            'timestamp': datetime.now().isoformat(),
            'player_a': player_a_id,
            'player_b': player_b_id,
            'outcome': outcome,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b,
            'rating_a_change': new_rating_a - rating_a,
            'rating_b_change': new_rating_b - rating_b,
        }
        self.history.append(game_record)

        return game_record

    def record_game(self, player1_id, player2_id, game_outcome):
        """Record a game and update ratings.

        Args:
            player1_id: Player 1 identifier
            player2_id: Player 2 identifier
            game_outcome: Result as returned by game.get_game_ended():
                         1 = player 1 won, -1 = player 2 won, 0 = tie

        Returns:
            dict with rating changes
        """
        # Convert game outcome to player 1's score
        if game_outcome == 1:
            score = 1.0  # Player 1 won
        elif game_outcome == -1:
            score = 0.0  # Player 2 won
        else:
            score = 0.5  # Tie

        return self.update_ratings(player1_id, player2_id, score)

    def get_leaderboard(self, top_n=None):
        """Get sorted list of players by rating.

        Args:
            top_n: Return only top N players (None = all)

        Returns:
            List of (player_id, rating) tuples sorted by rating
        """
        leaderboard = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            leaderboard = leaderboard[:top_n]
        return leaderboard

    def save(self, filepath):
        """Save ratings and history to JSON file.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'k_factor': self.k_factor,
            'initial_rating': self.initial_rating,
            'ratings': self.ratings,
            'history': self.history,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath):
        """Load ratings and history from JSON file.

        Args:
            filepath: Path to saved file

        Returns:
            True if loaded successfully, False if file not found
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return False

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.k_factor = data['k_factor']
        self.initial_rating = data['initial_rating']
        self.ratings = data['ratings']
        self.history = data['history']

        return True

    def print_summary(self):
        """Print summary of current ratings."""
        if not self.ratings:
            print("No players rated yet.")
            return

        print("\n" + "="*60)
        print("ELO Ratings Leaderboard")
        print("="*60)

        leaderboard = self.get_leaderboard()
        for rank, (player_id, rating) in enumerate(leaderboard, 1):
            games_played = sum(
                1 for game in self.history
                if game['player_a'] == player_id or game['player_b'] == player_id
            )
            print(f"{rank:2d}. {player_id:30s} {rating:7.1f} ({games_played} games)")

        print("="*60)
        print(f"Total games: {len(self.history)}")
        print("="*60 + "\n")