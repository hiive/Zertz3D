"""Main entry point for Zèrtz 3D game."""

import argparse

from factory import ZertzFactory
from game.player_config import parse_player_spec, PlayerConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zèrtz 3D Game",
        epilog="""
Player Configuration:
  Use --player1 and --player2 to configure each player with the format:
    TYPE[:PARAM=VALUE,PARAM=VALUE,...]

  Types:
    random          - Random move selection
    human           - Manual control (requires renderer)
    mcts            - Monte Carlo Tree Search AI

  MCTS Parameters (examples):
    iterations=N    - MCTS iterations per move (default: 1000)
    exploration=X   - UCB1 exploration constant (default: 1.41)
    fpu=X           - First Play Urgency reduction (e.g., 0.2)
    widening=X      - Progressive widening constant (e.g., 10.0)
    rave=X          - RAVE constant (300-3000, e.g., 1000)
    workers=N       - Number of worker threads (default: 16)
    verbose=1       - Print search statistics
    seed=N          - Random seed for reproducibility

  Examples:
    --player1 human
    --player2 mcts:iterations=500,workers=1
    --player1 mcts:iterations=1000,exploration=2.0,rave=1000
    --player2 mcts:iterations=2000,workers=8,verbose=1
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--player1",
        type=str,
        default="random",
        metavar="SPEC",
        help="Player 1 configuration (default: random). See --help for format."
    )
    parser.add_argument(
        "--player2",
        type=str,
        default="random",
        metavar="SPEC",
        help="Player 2 configuration (default: random). See --help for format."
    )
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
        choices=["uniform", "heatmap"],
        default=None,
        metavar="MODE",
        help="Highlight valid moves before each turn: 'uniform' (all equal) or 'heatmap' (proportional to AI score)",
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
    args = parser.parse_args()

    # Parse player configurations
    try:
        player1_config = parse_player_spec(args.player1)
        player2_config = parse_player_spec(args.player2)
    except ValueError as e:
        parser.error(f"Invalid player configuration: {e}")
        return

    # Determine which players are human (for renderer interaction)
    human_players = []
    if player1_config.player_type == "human":
        human_players.append(1)
    if player2_config.player_type == "human":
        human_players.append(2)

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
        human_players=tuple(human_players) if human_players else None,
        start_delay=args.start_delay,
        track_statistics=args.stats,
        player1_config=player1_config,
        player2_config=player2_config,
    )
    controller.run()

    # Print timing statistics if enabled
    if args.stats:
        controller.print_statistics()


if __name__ == "__main__":
    main()

