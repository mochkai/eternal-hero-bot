"""Controller: execute :class:`~eternal_hero_bot.core.actions.Action` values against a device."""

from __future__ import annotations

import math
from collections.abc import Callable

from eternal_hero_bot.core.actions import Action, Move, Swipe, Tap, TapNamed, Wait
from eternal_hero_bot.core.config import SCREEN, ScreenGeometry
from eternal_hero_bot.device.base import Device


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


class Controller:
    """Translates high-level actions into device calls."""

    def __init__(
        self,
        device: Device,
        assets: object | None,
        screen: ScreenGeometry,
        sleep: Callable[[float], None],
    ) -> None:
        """Create a controller.

        Args:
            device: Device to send input to.
            assets: Optional assets/manifest for named tap resolution.
            screen: Screen geometry for joystick and fallback coordinates.
            sleep: Sleep function used by :class:`~eternal_hero_bot.core.actions.Wait`.
        """
        self._device = device
        self._assets = assets
        self._screen = screen
        self._sleep = sleep

    def execute(self, action: Action) -> None:
        """Execute an action against the device.

        Args:
            action: Action to execute.

        Raises:
            ValueError: If a named tap cannot be resolved.
        """
        match action:
            case Move(dx=dx, dy=dy, ms=ms):
                push_joystick(self._device, dx, dy, ms, self._screen)
            case Tap(x=x, y=y):
                self._device.tap(x, y)
            case TapNamed(name=name):
                x, y = self._resolve_named(name)
                self._device.tap(x, y)
            case Swipe(x1=x1, y1=y1, x2=x2, y2=y2, ms=ms):
                self._device.swipe(x1, y1, x2, y2, ms)
            case Wait(seconds=seconds):
                self._sleep(seconds)

    def _resolve_named(self, name: str) -> tuple[int, int]:
        """Resolve a named tap to screen coordinates.

        Args:
            name: Manifest region or template name.

        Returns:
            Screen ``(x, y)`` coordinates.

        Raises:
            ValueError: If the name cannot be resolved.
        """
        # parity: fall back to the fixed Revive coordinate when no manifest region exists.
        if name == "revive_button":
            return self._screen.revive
        raise ValueError(f"unknown named tap: {name!r}")
