"""Device layer: the :class:`Device` protocol and the joystick helper.

The bot only talks to the device through the small :class:`Device` protocol, so
tests can substitute a fake that serves synthetic screenshots.
"""

from __future__ import annotations

import math
from typing import Protocol

from eternal_hero_bot.core.arrays import RGBImage
from eternal_hero_bot.core.config import SCREEN, ScreenGeometry


class Device(Protocol):
    """Minimal interface the bot needs from an Android device."""

    def screenshot(self) -> RGBImage:
        """Capture the screen.

        Returns:
            The screen as an ``H x W x 3`` integer RGB array.
        """
        ...

    def tap(self, x: int, y: int) -> None:
        """Tap the screen at ``(x, y)``.

        Args:
            x: Screen x coordinate.
            y: Screen y coordinate.
        """
        ...

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int) -> None:
        """Swipe from ``(x1, y1)`` to ``(x2, y2)`` over ``ms`` milliseconds.

        Args:
            x1: Start x.
            y1: Start y.
            x2: End x.
            y2: End y.
            ms: Duration in milliseconds.
        """
        ...


def push_joystick(
    device: Device, dx: float, dy: float, ms: int, screen: ScreenGeometry = SCREEN
) -> None:
    """Hold the virtual joystick in direction ``(dx, dy)`` for ``ms`` milliseconds.

    Args:
        device: Device to send the swipe to.
        dx: Direction x component (any magnitude).
        dy: Direction y component (any magnitude).
        ms: Hold duration in milliseconds.
        screen: Screen geometry holding the joystick position and push radius.
    """
    length = math.hypot(dx, dy) or 1
    tx = screen.joy_x + dx / length * screen.joy_r
    ty = screen.joy_y + dy / length * screen.joy_r
    device.swipe(screen.joy_x, screen.joy_y, int(tx), int(ty), ms)
