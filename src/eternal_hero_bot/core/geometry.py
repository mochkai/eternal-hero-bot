"""Geometry primitives and resolution scaling helpers.

The reference calibration is 960x1080 portrait, but scaling helpers support any
resolution. Callers pick the reference size appropriate for their assets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle in pixel coordinates.

    Attributes:
        x: Left edge.
        y: Top edge.
        w: Width.
        h: Height.
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        """Exclusive right edge."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Exclusive bottom edge."""
        return self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        """Centre pixel ``(x, y)``."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    def scale(self, factor: float) -> Box:
        """Scale box coordinates and size by ``factor``.

        Args:
            factor: Uniform scale factor.

        Returns:
            A scaled box with integer pixel coordinates.
        """
        return Box(
            x=round(self.x * factor),
            y=round(self.y * factor),
            w=max(1, round(self.w * factor)),
            h=max(1, round(self.h * factor)),
        )


def uniform_scale(
    frame_w: int,
    frame_h: int,
    ref_w: int,
    ref_h: int,
    *,
    tolerance: float = 0.01,
) -> float:
    """Compute the uniform scale factor from reference to frame pixels.

    The v1 manifest scaling mode is uniform: frame aspect ratio must match the
    reference within ``tolerance``. The factor is ``frame_w / ref_w``.

    Args:
        frame_w: Observed frame width in pixels.
        frame_h: Observed frame height in pixels.
        ref_w: Reference width in pixels.
        ref_h: Reference height in pixels.
        tolerance: Maximum allowed relative aspect-ratio difference.

    Returns:
        The uniform scale factor.

    Raises:
        ValueError: If the frame aspect ratio differs from the reference by more
            than ``tolerance``.
    """
    if ref_w <= 0 or ref_h <= 0:
        raise ValueError("reference dimensions must be positive")
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame dimensions must be positive")
    ref_ratio = ref_w / ref_h
    frame_ratio = frame_w / frame_h
    if abs(frame_ratio - ref_ratio) / ref_ratio > tolerance:
        raise ValueError(
            f"frame aspect ratio {frame_w}x{frame_h} does not match "
            f"reference {ref_w}x{ref_h} within {tolerance:.1%}"
        )
    return frame_w / ref_w


def scale_point(x: int, y: int, factor: float) -> tuple[int, int]:
    """Scale a point from reference to frame pixels.

    Args:
        x: Reference x coordinate.
        y: Reference y coordinate.
        factor: Uniform scale factor.

    Returns:
        The scaled ``(x, y)`` point.
    """
    return (round(x * factor), round(y * factor))


def scale_scalar(value: int | float, factor: float) -> int:
    """Scale a scalar distance from reference to frame pixels.

    Args:
        value: Reference scalar value.
        factor: Uniform scale factor.

    Returns:
        The scaled integer value, at least 1 for positive inputs.
    """
    scaled = round(value * factor)
    if value > 0:
        return max(1, scaled)
    return scaled
