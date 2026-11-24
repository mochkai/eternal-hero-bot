"""Shared numpy type aliases."""

import numpy as np
from numpy.typing import NDArray

type RGBImage = NDArray[np.int64]
"""An ``H x W x 3`` RGB image with integer channels (signed, so differences are safe)."""

type Mask = NDArray[np.bool_]
"""A boolean ``H x W`` mask."""

type FloatArray = NDArray[np.float64]
"""A float64 array (grayscale frames, positions)."""

type Point = tuple[int, int]
"""An ``(x, y)`` pixel position."""

type Offset = tuple[float, float]
"""A ``(dx, dy)`` offset relative to the hero on the minimap."""
