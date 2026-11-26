"""Tests for minimap pixel analysis."""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.core.arrays import RGBImage
from eternal_hero_bot.core.config import SCREEN
from eternal_hero_bot.perception.vision import (
    find_nodes,
    icon_center,
    is_dead,
    minimap,
    walkable_mask,
)
from tests.helpers import HANDLE_RGB, KEY_RGB, METAL_RGB, WALL_RGB, paint

R = SCREEN.mm_r


def test_minimap_crop_shape(screen: RGBImage) -> None:
    """The crop is a 2R square."""
    assert minimap(screen).shape == (2 * R, 2 * R, 3)


def test_walkable_mask_ground_and_wall(screen: RGBImage) -> None:
    """Ground is walkable, a wall block (plus margin) is not, outside the disc is not."""
    paint(screen, 30, -10, 20, 20, WALL_RGB)
    walk = walkable_mask(screen)
    assert walk.shape == (2 * R, 2 * R)
    assert walk[R, R + 20]  # open ground
    assert not walk[R, R + 40]  # inside wall
    assert not walk[R, R + 29]  # wall margin
    assert not walk[0, 0]  # corner is outside the minimap disc


def test_walkable_mask_cone_colour(screen: RGBImage) -> None:
    """The darker view-cone shade still counts as ground."""
    paint(screen, -40, 20, 20, 20, (69, 84, 100))
    assert walkable_mask(screen)[R + 30, R - 30]


def test_walkable_mask_hero_always_standable(screen: RGBImage) -> None:
    """The hero arrow disc is walkable even when painted as wall."""
    paint(screen, -5, -5, 10, 10, WALL_RGB)
    assert walkable_mask(screen)[R, R]


def test_find_nodes_detects_handle_and_rejects_key(screen: RGBImage) -> None:
    """Tool-handle blobs become nodes; bright low-blue key blobs are rejected."""
    paint(screen, 40, 0, 4, 4, HANDLE_RGB)
    paint(screen, -50, -30, 4, 4, HANDLE_RGB)
    paint(screen, 0, 50, 4, 4, KEY_RGB)
    nodes = find_nodes(screen)
    assert len(nodes) == 2
    found = sorted((round(dx), round(dy)) for dx, dy, _ in nodes)
    # Centroid of a 4x4 block starting at offset o is o + 1.5
    assert found == [(-48, -28), (42, 2)]
    assert all(size == 16 for *_, size in nodes)


def test_find_nodes_ignores_small_specks_and_hero(screen: RGBImage) -> None:
    """Clusters under 4 px and handle pixels on the hero arrow are ignored."""
    paint(screen, 40, 0, 1, 3, HANDLE_RGB)
    paint(screen, -2, -2, 4, 4, HANDLE_RGB)
    assert find_nodes(screen) == []


def test_icon_center_includes_metal(screen: RGBImage) -> None:
    """The icon centre moves from the handle toward the attached grey metal head."""
    paint(screen, 40, 0, 4, 4, HANDLE_RGB)
    paint(screen, 44, 0, 4, 4, METAL_RGB)
    sub = minimap(screen)
    handle = np.zeros((2 * R, 2 * R), bool)
    handle[R : R + 4, R + 40 : R + 44] = True
    cx, cy = icon_center(sub, handle, R + 41.5, R + 1.5)
    assert cx == R + 43.5
    assert cy == R + 1.5


def test_icon_center_falls_back_when_empty(screen: RGBImage) -> None:
    """With too few icon pixels the handle centroid is returned unchanged."""
    sub = minimap(screen)
    handle = np.zeros((2 * R, 2 * R), bool)
    assert icon_center(sub, handle, 50.0, 60.0) == (50.0, 60.0)


def test_is_dead(screen: RGBImage) -> None:
    """The green Revive button at the probe pixel means dead."""
    assert not is_dead(screen)
    x, y = SCREEN.death_probe
    screen[y, x] = (150, 200, 80)
    assert is_dead(screen)
