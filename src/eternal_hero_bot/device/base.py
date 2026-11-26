"""Device layer: the :class:`Device` protocol.

The bot only talks to the device through the small :class:`Device` protocol, so
tests can substitute a fake that serves synthetic screenshots.
"""

from __future__ import annotations

from typing import Protocol

from eternal_hero_bot.core.arrays import RGBImage


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
