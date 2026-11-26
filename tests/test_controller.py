"""Tests for the action Controller."""

from __future__ import annotations

import pytest

from eternal_hero_bot.action.controller import Controller
from eternal_hero_bot.core.actions import Move, Swipe, Tap, TapNamed, Wait
from eternal_hero_bot.core.arrays import RGBImage
from eternal_hero_bot.core.config import SCREEN
from tests.helpers import blank_screen


class FakeDevice:
    """Records device calls."""

    def __init__(self) -> None:
        """Create an empty recorder."""
        self.taps: list[tuple[int, int]] = []
        self.swipes: list[tuple[int, int, int, int, int]] = []

    def screenshot(self) -> RGBImage:
        """Return a blank screen."""
        return blank_screen()

    def tap(self, x: int, y: int) -> None:
        """Record a tap."""
        self.taps.append((x, y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int) -> None:
        """Record a swipe."""
        self.swipes.append((x1, y1, x2, y2, ms))


def _controller() -> tuple[Controller, FakeDevice, list[float]]:
    dev = FakeDevice()
    sleeps: list[float] = []
    ctrl = Controller(device=dev, assets=None, screen=SCREEN, sleep=sleeps.append)
    return ctrl, dev, sleeps


def test_move_executes_joystick_swipe() -> None:
    """Move becomes a normalised joystick swipe."""
    ctrl, dev, _ = _controller()
    ctrl.execute(Move(dx=0.0, dy=-10.0, ms=300))
    assert dev.swipes == [
        (SCREEN.joy_x, SCREEN.joy_y, SCREEN.joy_x, SCREEN.joy_y - SCREEN.joy_r, 300)
    ]


def test_tap() -> None:
    """Tap forwards screen coordinates."""
    ctrl, dev, _ = _controller()
    ctrl.execute(Tap(10, 20))
    assert dev.taps == [(10, 20)]


def test_tap_named_revive_fallback() -> None:
    """TapNamed('revive_button') falls back to the fixed revive coordinate."""
    ctrl, dev, _ = _controller()
    ctrl.execute(TapNamed("revive_button"))
    assert dev.taps == [SCREEN.revive]


def test_tap_named_unknown_raises() -> None:
    """Unknown named taps raise ValueError without assets."""
    ctrl, _, _ = _controller()
    with pytest.raises(ValueError, match="unknown named tap"):
        ctrl.execute(TapNamed("missing"))


def test_swipe() -> None:
    """Swipe forwards coordinates and duration."""
    ctrl, dev, _ = _controller()
    ctrl.execute(Swipe(1, 2, 3, 4, 50))
    assert dev.swipes == [(1, 2, 3, 4, 50)]


def test_wait() -> None:
    """Wait calls the sleep function."""
    ctrl, _, sleeps = _controller()
    ctrl.execute(Wait(1.5))
    assert sleeps == [1.5]
