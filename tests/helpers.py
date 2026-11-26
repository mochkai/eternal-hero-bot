"""Synthetic-screen helpers shared by the tests (no emulator needed)."""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.core.arrays import RGBImage
from eternal_hero_bot.core.config import COLOURS, SCREEN

HEIGHT, WIDTH = 1080, 960
HANDLE_RGB = (140, 87, 64)
KEY_RGB = (178, 106, 35)
WALL_RGB = (30, 38, 56)
METAL_RGB = (200, 200, 200)


def screen_xy(dx: int, dy: int) -> tuple[int, int]:
    """Screen pixel of a hero-relative minimap offset.

    Args:
        dx: Offset x from the hero.
        dy: Offset y from the hero.

    Returns:
        Screen ``(x, y)``.
    """
    return SCREEN.mm_cx + dx, SCREEN.mm_cy + dy


def paint(img: RGBImage, dx: int, dy: int, w: int, h: int, rgb: tuple[int, int, int]) -> None:
    """Fill a ``w x h`` rectangle whose top-left is at hero offset ``(dx, dy)``.

    Args:
        img: Screen image, modified in place.
        dx: Left edge offset from the hero.
        dy: Top edge offset from the hero.
        w: Width in pixels.
        h: Height in pixels.
        rgb: Fill colour.
    """
    x, y = screen_xy(dx, dy)
    img[y : y + h, x : x + w] = rgb


def blank_screen() -> RGBImage:
    """A black 960x1080 screen whose minimap square is all walkable ground.

    Returns:
        The synthetic screen.
    """
    img = np.zeros((HEIGHT, WIDTH, 3), np.int64)
    r = SCREEN.mm_r
    paint(img, -r, -r, 2 * r, 2 * r, COLOURS.ground_rgb)
    return img
