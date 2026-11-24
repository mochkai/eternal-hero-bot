"""Debug rendering: the minimap next to the walkable mask, with nodes and path."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from eternal_hero_bot.arrays import Mask, Offset, Point, RGBImage
from eternal_hero_bot.config import SCREEN, ScreenGeometry
from eternal_hero_bot.vision import minimap

SCALE = 3
"""Upscale factor for the debug image."""
NODE_MARK_R = 12
"""Radius of the magenta circle drawn around each node (output pixels)."""


def render_detect(
    img: RGBImage,
    nodes: list[Offset],
    walk: Mask,
    path: list[Point] | None,
    screen: ScreenGeometry = SCREEN,
) -> Image.Image:
    """Build the debug image: minimap | wall mask, nodes in magenta and path in green.

    Args:
        img: Full-screen RGB screenshot.
        nodes: Node offsets relative to the hero.
        walk: Walkable mask (crop pixels).
        path: BFS path in crop pixels, if any.
        screen: Screen geometry.

    Returns:
        The side-by-side annotated image.
    """
    k, r = SCALE, screen.mm_r
    src = Image.fromarray(minimap(img, screen).astype("uint8")).resize(
        (2 * r * k,) * 2, Image.Resampling.NEAREST
    )
    m = np.where(walk[..., None], 190, 30).astype("uint8").repeat(3, 2)
    msk = Image.fromarray(m).resize(src.size, Image.Resampling.NEAREST)
    for im_ in (src, msk):
        d = ImageDraw.Draw(im_)
        for dx, dy in nodes:
            x, y = (r + dx) * k, (r + dy) * k
            d.ellipse(
                [x - NODE_MARK_R, y - NODE_MARK_R, x + NODE_MARK_R, y + NODE_MARK_R],
                outline=(255, 0, 255),
                width=3,
            )
        if path:
            d.line([(x * k, y * k) for x, y in path], fill=(0, 220, 0), width=3)
    out = Image.new("RGB", (src.width * 2, src.height))
    out.paste(src, (0, 0))
    out.paste(msk, (src.width, 0))
    return out
