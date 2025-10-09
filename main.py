"""Main entry point for Zertz 3D game."""

import argparse
from controller.zertz_game_controller import ZertzGameController


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zertz 3D Game')
    parser.add_argument('--replay', type=str, help='Path to replay file (board size auto-detected)')
    parser.add_argument('--rings', type=int, default=37, help='Number of rings on the board (default: 37, ignored if --replay is used)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible games (ignored if --replay is used)')
    parser.add_argument('--log', action='store_true', help='Log game actions to zertzlog_<seed>.txt (ignored if --replay is used)')
    parser.add_argument('--partial', action='store_true', help='Continue with random play after replay ends (only with --replay)')
    parser.add_argument('--headless', action='store_true', help='Run without 3D renderer')
    parser.add_argument('--games', type=int, help='Number of games to play (default: play indefinitely)')
    parser.add_argument('--show-moves', action='store_true', help='Show valid moves before each turn')
    parser.add_argument('--blitz', action='store_true', help='Use blitz variant (37 rings only, fewer marbles, lower win thresholds)')
    args = parser.parse_args()

    game = ZertzGameController(rings=args.rings, replay_file=args.replay, seed=args.seed,
                                log_to_file=args.log, partial_replay=args.partial, headless=args.headless,
                                max_games=args.games, show_moves=args.show_moves, blitz=args.blitz)
    game.run()