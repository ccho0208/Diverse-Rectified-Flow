"""Run the shared demonstrations locally or inside a Vista GPU allocation."""

import argparse
from dataclasses import replace
from pathlib import Path

from .config import DemoConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", help="JSON configuration; otherwise use default settings")
    run.add_argument("--output", required=True)
    run.add_argument("--device", help="auto, cpu, cuda, cuda:0, or mps")
    run.add_argument("--seed", type=int)
    run.add_argument("--experiments", nargs="+", choices=("gmm8", "two_disks", "rings"))
    run.add_argument("--ring-components", type=int, choices=(2, 8))
    run.add_argument("--resume", action="store_true")
    render = commands.add_parser("render")
    render.add_argument("--input", required=True, help="experiment directory with display.npz/report.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            from .visualize import render_demo
            render_demo(Path(args.input))
            return
        config = DemoConfig.load(args.config) if args.config else DemoConfig()
        updates = {key: getattr(args, key) for key in ("device", "seed", "ring_components")
                   if getattr(args, key) is not None}
        if args.experiments is not None:
            updates["experiments"] = tuple(args.experiments)
        config = replace(config, **updates)
        from .runner import run_demos
        run_demos(config, args.output, resume=args.resume)
    except InterruptedError as error:
        print(str(error), flush=True)
        raise SystemExit(75)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
