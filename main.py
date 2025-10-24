"""Main entry point for Zèrtz 3D game."""

import argparse

from factory import ZertzFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Zèrtz 3D Game")
    parser.add_argument(
        "--replay", type=str, help="Path to transcript/notationre file (board size auto-detected)"
    )
    parser.add_argument(
        "--rings",
        type=int,
        choices=[37, 48, 61],
        default=37,
        help="Board size: 37, 48, or 61 rings (default: 37)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible games (ignored if --replay is used)",
    )
    parser.add_argument(
        "--transcript-file",
        nargs="?",
        const=".",
        default=None,
        metavar="DIR",
        help="Log game actions to zertzlog_<seed>.txt in DIR (default: current directory, ignored if --replay is used)",
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
        "--notation-file",
        nargs="?",
        const=".",
        default=None,
        metavar="DIR",
        help="Log game moves using official Zèrtz notation to file in DIR (default: current directory, ignored if --replay is used)",
    )
    parser.add_argument(
        "--transcript-screen",
        action="store_true",
        help="Output transcript format game actions to screen",
    )
    parser.add_argument(
        "--notation-screen",
        action="store_true",
        help="Output official Zèrtz notation to screen",
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
    parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="Delay before first move in seconds (Panda renderer only, default: 0)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Track and report statistics for each game",
    )
    parser.add_argument(
        "--mcts-player2",
        type=int,
        nargs='?',
        const=100,
        default=None,
        metavar="ITERATIONS",
        help="Use MCTS player for player 2 with N iterations (default: 100 if not specified)",
    )
    args = parser.parse_args()

    factory = ZertzFactory()
    controller = factory.create_controller(
        rings=args.rings,
        replay_file=args.replay,
        seed=args.seed,
        log_to_file=args.transcript_file,
        log_to_screen=args.transcript_screen,
        log_notation_to_file=args.notation_file,
        log_notation_to_screen=args.notation_screen,
        partial_replay=args.partial,
        headless=args.headless,
        max_games=args.games,
        highlight_choices=args.highlight_choices,
        show_coords=args.show_coords,
        blitz=args.blitz,
        move_duration=args.move_duration,
        human_players=(1,) if args.human else None,
        start_delay=args.start_delay,
        track_statistics=args.stats,
        mcts_player2_iterations=args.mcts_player2,
    )
    controller.run()

    # Print timing statistics if enabled
    if args.stats:
        controller.print_statistics()


if __name__ == "__main__":
    main()

