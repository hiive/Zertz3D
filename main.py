"""Main entry point for Zertz 3D game."""

import argparse

from factory import ZertzFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Zertz 3D Game")
    parser.add_argument(
        "--replay", type=str, help="Path to replay file (board size auto-detected)"
    )
    parser.add_argument(
        "--rings",
        type=int,
        default=37,
        help="Number of rings on the board (default: 37, ignored if --replay is used)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible games (ignored if --replay is used)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log game actions to zertzlog_<seed>.txt (ignored if --replay is used)",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Continue with random play after replay ends (only with --replay)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without 3D renderer"
    )
    parser.add_argument(
        "--games", type=int, help="Number of games to play (default: play indefinitely)"
    )
    parser.add_argument(
        "--highlight-choices",
        action="store_true",
        help="Highlight valid moves and actions before each turn",
    )
    parser.add_argument(
        "--show-coords",
        action="store_true",
        help="Show coordinate labels on rings in 3D view",
    )
    parser.add_argument(
        "--log-notation",
        action="store_true",
        help="Log game moves using official Zèrtz notation",
    )
    parser.add_argument(
        "--blitz",
        action="store_true",
        help="Use blitz variant (37 rings only, fewer marbles, lower win thresholds)",
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=0.5,
        help="Duration between moves in seconds (default: 0.666)",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Control player 1 manually (requires interactive renderer)",
    )
    args = parser.parse_args()

    factory = ZertzFactory()
    controller = factory.create_controller(
        rings=args.rings,
        replay_file=args.replay,
        seed=args.seed,
        log_to_file=args.log,
        partial_replay=args.partial,
        headless=args.headless,
        max_games=args.games,
        highlight_choices=args.highlight_choices,
        show_coords=args.show_coords,
        log_notation=args.log_notation,
        blitz=args.blitz,
        move_duration=args.move_duration,
        human_players=(1,) if args.human else None,
    )
    controller.run()


if __name__ == "__main__":
    main()
