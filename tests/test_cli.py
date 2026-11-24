"""Tests for the command-line parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from eternal_hero_bot.cli import build_parser
from eternal_hero_bot.config import DeviceConfig


def test_run_defaults() -> None:
    """``run`` defaults to 20 minutes."""
    args = build_parser().parse_args(["run"])
    assert args.command == "run"
    assert args.minutes == 20.0


def test_detect_out() -> None:
    """``detect --out`` takes a path, defaulting to detect.png."""
    assert build_parser().parse_args(["detect"]).out == Path("detect.png")
    assert build_parser().parse_args(["detect", "--out", "x.png"]).out == Path("x.png")


def test_command_required() -> None:
    """A subcommand is mandatory."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_device_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """EHBOT_ADB / EHBOT_DEVICE override the adb defaults."""
    monkeypatch.setenv("EHBOT_ADB", "/usr/bin/adb")
    monkeypatch.setenv("EHBOT_DEVICE", "emulator-5556")
    cfg = DeviceConfig.from_env()
    assert (cfg.adb, cfg.serial) == ("/usr/bin/adb", "emulator-5556")
