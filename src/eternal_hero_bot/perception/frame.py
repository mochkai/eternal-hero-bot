"""A captured screenshot with metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from eternal_hero_bot.core.arrays import RGBImage


@dataclass
class Frame:
    """A single captured screenshot.

    Attributes:
        image: Full-screen RGB image as ``H x W x 3`` int64.
        t: Capture time in seconds.
        index: Monotonic frame index within the runtime.
    """

    image: RGBImage
    t: float
    index: int
    _u8: NDArray[np.uint8] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def u8(self) -> NDArray[np.uint8]:
        """A cached uint8 view of the image for cv2 APIs.

        Returns:
            The image cast to ``uint8``.
        """
        if self._u8 is None:
            self._u8 = self.image.astype(np.uint8)
        return self._u8
