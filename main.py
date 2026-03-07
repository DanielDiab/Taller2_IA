import sys
from optparse import OptionParser
from typing import TypedDict

import view.text_display as text_display
import world.layout as layout_module
from world.runner import run_adversarial_mode, run_csp_mode
from view.display import AdversarialDisplay, CspDisplay


class CommandArgs(TypedDict):
    mode: str
    algorithm: str
    layout: layout_module.DroneLayout
    display: AdversarialDisplay | CspDisplay
    depth: int
    random_probability: float
    num_games: int


def read_command(argv: list[str]) -> CommandArgs:
    """
    Processes the command used to run main from the command line.
    """
    usage_str = """
    USAGE:      python main.py -m MODE -a ALGORITHM_OR_AGENT -l LAYOUT [options]

    MODES:
      csp          - CSP planning mode (drone assignment)
      adversarial  - Adversarial game mode (Minimax, AlphaBeta, Expectimax)

    EXAMPLES:
      python main.py -m csp -a backtracking_mrv_lcv -l twin_bases
      python main.py -m adversarial -a MinimaxAgent -d 3 -l small_hunt -q -n 5
    """
    parser = OptionParser(usage_str, add_help_option=False)
    parser.add_option("--help", action="help", help="Show this message and exit")

    parser.add_option(
        "--mode",
        "-m",
        dest="mode",
        help="Execution mode: csp or adversarial (REQUIRED)",
    )
    parser.add_option(
        "-a",
        dest="algorithm",
        help="CSP: algorithm name. Adversarial: agent class name (REQUIRED)",
    )
    parser.add_option(
        "-p",
        type="float",
        dest="random_probability",
        default=0.0,
        help="Probability of hunter taking a random action (0.0-1.0) [Default: %default]",
    )
    parser.add_option(
        "-d",
        "--depth",
        type="int",
        dest="depth",
        default=1,
        help="Search depth for adversarial agents [Default: %default]",
    )
    parser.add_option(
        "-l", "--layout", dest="layout", help="Layout file to load (REQUIRED)"
    )
    parser.add_option(
        "-t",
        "--textGraphics",
        action="store_true",
        dest="textGraphics",
        default=False,
        help="Display output as text only",
    )
    parser.add_option(
        "-q",
        "--quietTextGraphics",
        action="store_true",
        dest="quietGraphics",
        default=False,
        help="Generate minimal output and no graphics",
    )
    parser.add_option(
        "-z",
        "--zoom",
        type="float",
        dest="zoom",
        default=1.0,
        help="Zoom the size of the graphics window [Default: %default]",
    )
    parser.add_option(
        "-x",
        "--frameTime",
        dest="frameTime",
        type="float",
        default=0.1,
        help="Time to delay between frames [Default: %default]",
    )
    parser.add_option(
        "-n",
        "--numGames",
        type="int",
        dest="numGames",
        default=1,
        help="Number of games to run [Default: %default]",
    )

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception("Command line input not understood: " + str(otherjunk))

    if not options.mode:
        parser.error("-m/--mode is required")
    if options.mode not in ("csp", "adversarial"):
        parser.error("Invalid mode '%s'. Choose: csp or adversarial" % options.mode)
    if not options.algorithm:
        parser.error("-a is required")
    if not options.layout:
        parser.error("-l/--layout is required")

    assert options.mode is not None
    assert options.algorithm is not None
    assert options.layout is not None

    layout = layout_module.get_layout(options.layout)
    if layout is None:
        raise Exception("The layout " + options.layout + " cannot be found")

    display = None

    if options.mode == "csp":
        if options.quietGraphics:
            display = text_display.CspNullGraphics()
        elif options.textGraphics:
            text_display.sleep_time = options.frameTime
            display = text_display.CspGraphics(speed=options.frameTime)
        else:
            import view.graphics_display as graphics_display
            display = graphics_display.CspGraphics(options.zoom, options.frameTime)
    elif options.mode == "adversarial":
        if options.quietGraphics:
            display = text_display.AdversarialNullGraphics()
        elif options.textGraphics:
            text_display.sleep_time = options.frameTime
            display = text_display.TextAdversarialGraphics()
        else:
            import view.graphics_display as graphics_display
            display = graphics_display.VisualAdversarialGraphics(
                options.zoom, options.frameTime
            )

    assert display is not None

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Mode: {options.mode}")
    print(f"  Layout: {options.layout}")
    if options.mode == "csp":
        print(f"  Algorithm: {options.algorithm}")
    elif options.mode == "adversarial":
        print(f"  Agent: {options.algorithm}")
        print(f"  Depth: {options.depth}")
        print(f"  Random probability: {options.random_probability}")
        print(f"  Games: {options.numGames}")
    print("=" * 60)

    return CommandArgs(
        mode=options.mode,
        num_games=options.numGames,
        layout=layout,
        display=display,
        algorithm=options.algorithm,
        depth=options.depth,
        random_probability=options.random_probability,
    )


if __name__ == "__main__":
    args = read_command(sys.argv[1:])
    mode = args.get("mode")
    num_games = args.get("num_games", 1)
    if mode == "csp":
        display = args["display"]
        assert isinstance(display, CspDisplay)
        run_csp_mode(
            layout=args["layout"],
            algorithm=args["algorithm"],
            display=display,
        )
    elif mode == "adversarial":
        display = args["display"]
        assert isinstance(display, AdversarialDisplay)
        run_adversarial_mode(
            layout=args["layout"],
            agent_type=args["algorithm"],
            depth=args["depth"],
            random_probability=args["random_probability"],
            display=display,
            num_games=num_games,
        )
