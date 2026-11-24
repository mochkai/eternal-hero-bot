"""Tests for the detect debug image."""

from __future__ import annotations

from eternal_hero_bot.arrays import RGBImage
from eternal_hero_bot.bot import plan
from eternal_hero_bot.config import SCREEN
from eternal_hero_bot.debug import SCALE, render_detect
from tests.helpers import HANDLE_RGB, paint


def test_render_detect_side_by_side(screen: RGBImage) -> None:
    """The image is minimap | mask at SCALE, with the path drawn in green."""
    paint(screen, 50, -2, 4, 4, HANDLE_RGB)
    nodes, walk, path = plan(screen, [])
    assert len(nodes) == 1
    assert path is not None
    out = render_detect(screen, nodes, walk, path)
    side = SCREEN.mm_size * SCALE
    assert out.size == (2 * side, side)
    r = SCREEN.mm_r * SCALE
    assert out.getpixel((r + 15, r)) == (0, 220, 0)


def test_plan_skips_listed_nodes(screen: RGBImage) -> None:
    """Nodes within the skip radius are dropped and no path is planned."""
    paint(screen, 50, -2, 4, 4, HANDLE_RGB)
    nodes, _, path = plan(screen, [(52.0, 0.0)])
    assert nodes == []
    assert path is None
