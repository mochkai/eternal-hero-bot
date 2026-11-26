"""adb-backed :class:`~eternal_hero_bot.device.base.Device` for BlueStacks."""

from __future__ import annotations

import io
import subprocess

import numpy as np
from PIL import Image

from eternal_hero_bot.core.arrays import RGBImage
from eternal_hero_bot.core.config import DeviceConfig


class AdbDevice:
    """A :class:`~eternal_hero_bot.device.base.Device` backed by adb (e.g. BlueStacks)."""

    def __init__(self, config: DeviceConfig | None = None) -> None:
        """Create the device wrapper.

        Args:
            config: adb settings; defaults to :meth:`DeviceConfig.from_env`.
        """
        self.config = config or DeviceConfig.from_env()

    def adb(self, *args: str, capture: bool = False) -> bytes:
        """Run an adb command against the configured device.

        Args:
            *args: adb arguments after ``-s <serial>``.
            capture: Return stdout when true, otherwise an empty bytes object.

        Returns:
            The command's stdout if ``capture`` is set, else ``b""``.

        Raises:
            FileNotFoundError: If the adb executable does not exist.
            subprocess.TimeoutExpired: If the command exceeds the configured timeout.
        """
        cmd = [self.config.adb, "-s", self.config.serial, *args]
        result = subprocess.run(cmd, capture_output=True, timeout=self.config.timeout, check=False)
        return result.stdout if capture else b""

    def screenshot(self) -> RGBImage:
        """Capture the screen via ``screencap``.

        Returns:
            The screen as an ``H x W x 3`` int64 RGB array.
        """
        png = self.adb("exec-out", "screencap", "-p", capture=True)
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(np.int64)

    def tap(self, x: int, y: int) -> None:
        """Tap the screen at ``(x, y)``.

        Args:
            x: Screen x coordinate.
            y: Screen y coordinate.
        """
        self.adb("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int) -> None:
        """Swipe from ``(x1, y1)`` to ``(x2, y2)`` over ``ms`` milliseconds.

        Args:
            x1: Start x.
            y1: Start y.
            x2: End x.
            y2: End y.
            ms: Duration in milliseconds.
        """
        self.adb("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(ms))
