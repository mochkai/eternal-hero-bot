"""Command-line interface: ``ehbot run`` and ``ehbot detect``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from eternal_hero_bot.action.navigation import plan
from eternal_hero_bot.brain.rules import load_brain_config
from eternal_hero_bot.debug import render_detect
from eternal_hero_bot.device.adb import AdbDevice
from eternal_hero_bot.device.base import Device
from eternal_hero_bot.runtime import Runtime
from eternal_hero_bot.skills import build_skills

logger = logging.getLogger(__name__)

DEFAULT_MINUTES = 20.0
DEFAULT_DETECT_OUT = Path("detect.png")
DEFAULT_CONFIG = Path(__file__).with_name("default_config.toml")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The configured parser with ``run`` and ``detect`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="ehbot",
        description="Auto-gatherer for Eternal Hero on BlueStacks via adb. "
        "Set EHBOT_ADB / EHBOT_DEVICE to override the adb path and device serial.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="gather loop")
    run_p.add_argument(
        "--minutes",
        type=float,
        default=DEFAULT_MINUTES,
        help=f"how long to run (default {DEFAULT_MINUTES:g})",
    )
    run_p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="brain/skill config TOML (default: packaged default_config.toml)",
    )
    det_p = sub.add_parser("detect", help="write an annotated minimap debug image")
    det_p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DETECT_OUT,
        help=f"output image path (default {DEFAULT_DETECT_OUT})",
    )
    return parser


def detect(device: Device, out: Path) -> None:
    """Screenshot once, plan, and save the annotated debug image.

    Args:
        device: Device to screenshot.
        out: Where to write the PNG.
    """
    img = device.screenshot()
    nodes, walk, path = plan(img, [])
    render_detect(img, nodes, walk, path).save(out)
    logger.info(
        "nodes: %s path len: %s",
        [(round(x), round(y)) for x, y in nodes],
        len(path) if path else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ehbot`` console script.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    device = AdbDevice()
    if args.command == "detect":
        detect(device, args.out)
    else:
        cfg = load_brain_config(args.config)
        runtime = Runtime(
            device,
            brain=cfg.brain,
            skills=build_skills(cfg.skill_options),
            max_consecutive_failures=cfg.max_consecutive_failures,
        )
        try:
            runtime.run(args.minutes)
        except KeyboardInterrupt:
            logger.info("stopped")
    return 0
