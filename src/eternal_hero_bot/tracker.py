"""Hero world-position tracking by phase correlation of consecutive minimaps."""

from __future__ import annotations

import math

import numpy as np

from eternal_hero_bot.arrays import FloatArray
from eternal_hero_bot.config import SCREEN, STUCK, ScreenGeometry


class Tracker:
    """Tracks the hero's world position by phase-correlating consecutive minimaps.

    The minimap is north-up and centred on the hero, so the shift between two
    frames equals the hero's movement in minimap pixels.

    Attributes:
        win: Soft circular window applied before correlation (hero arrow zeroed).
        prev: The previous frame, or None before the first update / after reset.
        pos: Accumulated world position ``(x, y)`` in minimap pixels.
        max_step_px: Per-frame shifts larger than this are discarded as glitches.
    """

    def __init__(
        self, screen: ScreenGeometry = SCREEN, max_step_px: int = STUCK.max_step_px
    ) -> None:
        """Build the correlation window and start at the world origin.

        Args:
            screen: Screen geometry (minimap size, hero arrow radius).
            max_step_px: Per-frame shifts larger than this are discarded as glitches.
        """
        n, r = screen.mm_size, screen.mm_r
        yy, xx = np.mgrid[0:n, 0:n]
        d = np.hypot(xx - r, yy - r)
        self.win: FloatArray = np.clip((r - 6 - d) / 10, 0, 1)  # soft circular window
        self.win[d < screen.player_r + 3] = 0  # ignore the hero arrow (it rotates)
        self.prev: FloatArray | None = None
        self.pos: FloatArray = np.zeros(2)
        self.max_step_px = max_step_px

    def update(self, gray: FloatArray) -> float:
        """Feed a new minimap frame and accumulate the detected shift.

        Args:
            gray: Grayscale (or float mask) minimap frame, same shape as the window.

        Returns:
            Distance moved since the last frame (0.0 on the first frame or a glitch).
        """
        if self.prev is None:
            self.prev = gray
            return 0.0
        a = (self.prev - self.prev.mean()) * self.win
        b = (gray - gray.mean()) * self.win
        # Normalised cross-power spectrum; its inverse FFT peaks at the shift
        cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
        cross /= np.abs(cross) + 1e-9
        corr = np.fft.ifft2(cross).real
        yi, xi = np.unravel_index(corr.argmax(), corr.shape)
        n = corr.shape[0]
        # Map wrapped peak indices to signed shifts
        x = int(xi) - n if xi > n // 2 else int(xi)
        y = int(yi) - n if yi > n // 2 else int(yi)
        self.prev = gray
        if math.hypot(x, y) > self.max_step_px:  # implausible: tracking glitch, ignore
            return 0.0
        self.pos += (x, y)
        return math.hypot(x, y)

    def reset(self) -> None:
        """Forget the previous frame (e.g. after a teleport/respawn)."""
        self.prev = None
