"""Pixel analysis of screenshots: minimap crop, walkable mask, nodes and death dialog.

Everything here is pure numpy over an RGB screenshot, so it is cheap and needs no
model or API calls.
"""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.core.arrays import Mask, RGBImage
from eternal_hero_bot.core.config import COLOURS, PATHING, SCREEN, Colours, ScreenGeometry

ICON_R = 11
"""Half-size of the window searched for a node icon's metal around its handle."""
CLUSTER_PX = 7
"""Handle pixels within this distance of a cluster's centroid join that cluster."""
MIN_CLUSTER = 4
"""Clusters with fewer pixels are noise."""


def is_dead(img: RGBImage, screen: ScreenGeometry = SCREEN) -> bool:
    """Detect the 'You died' dialog by its green Revive button.

    Args:
        img: Full-screen RGB screenshot.
        screen: Screen geometry holding the probe pixel.

    Returns:
        True if the probe pixel has the Revive button's green.
    """
    x, y = screen.death_probe
    r, g, b = img[y, x]
    return bool(g > 170 and r > 120 and b < 110 and g > r)


def minimap(img: RGBImage, screen: ScreenGeometry = SCREEN) -> RGBImage:
    """Return the square minimap crop centred on the hero.

    Args:
        img: Full-screen RGB screenshot.
        screen: Screen geometry of the minimap.

    Returns:
        A ``2R x 2R x 3`` view into ``img``.
    """
    cx, cy, r = screen.mm_cx, screen.mm_cy, screen.mm_r
    return img[cy - r : cy + r, cx - r : cx + r]


def dilate(mask: Mask, r: int) -> Mask:
    """Binary dilation of ``mask`` by a square of radius ``r`` (numpy only, wraps at edges).

    Args:
        mask: Boolean mask.
        r: Square structuring-element radius.

    Returns:
        The dilated mask.
    """
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def _disc_d2(screen: ScreenGeometry) -> np.ndarray:
    """Squared distance of every minimap-crop pixel from the hero.

    Args:
        screen: Screen geometry of the minimap.

    Returns:
        A ``2R x 2R`` integer array.
    """
    n, r = screen.mm_size, screen.mm_r
    yy, xx = np.mgrid[0:n, 0:n]
    d2: np.ndarray = (xx - r) ** 2 + (yy - r) ** 2
    return d2


def walkable_mask(
    img: RGBImage,
    screen: ScreenGeometry = SCREEN,
    colours: Colours = COLOURS,
    wall_margin: int = PATHING.wall_margin,
) -> Mask:
    """Boolean grid of walkable minimap pixels (True = can walk).

    Dark navy pixels are walls. Thin dark icon outlines are closed over first so
    icons/text don't become fake walls, then walls are grown by ``wall_margin``.

    Args:
        img: Full-screen RGB screenshot.
        screen: Screen geometry of the minimap.
        colours: Ground palette.
        wall_margin: Pixels of clearance kept from walls.

    Returns:
        A ``2R x 2R`` mask in minimap-crop coordinates.
    """
    mm = minimap(img, screen)
    # Walkable ground is one flat palette colour; the hero's view cone darkens it slightly
    d_ground = np.sqrt(((mm - np.array(colours.ground_rgb)) ** 2).sum(2))
    d_cone = np.sqrt(((mm - np.array(colours.cone_rgb)) ** 2).sum(2))
    ground = (d_ground < colours.ground_tol) | (d_cone < colours.cone_tol)
    # Close 1px speckles (anti-aliasing), then keep a margin from walls
    ground = ~dilate(~dilate(ground, 1), 1)
    wall = dilate(~ground, wall_margin)
    d2 = _disc_d2(screen)
    walk: Mask = ~wall & (d2 < screen.mm_r**2)
    walk[d2 < screen.player_r**2] = True  # hero arrow is always standable
    return walk


def _handle_mask(px: RGBImage) -> Mask:
    """Brown-orange tool-handle pixels (~145,90,65); ground is blue-grey (b >= r).

    Args:
        px: RGB array of any leading shape.

    Returns:
        Boolean mask over the leading shape.
    """
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    m: Mask = (r > 105) & (r - b > 50) & (g > r * 0.4) & (g < r * 0.8)
    return m


def _cluster(mask: Mask) -> list[list[int]]:
    """Greedy clustering: merge pixels within ``CLUSTER_PX`` of a cluster's centroid.

    Args:
        mask: Pixels to cluster.

    Returns:
        Clusters as ``[sum_x, sum_y, count]`` accumulators.
    """
    clusters: list[list[int]] = []
    ys, xs = np.nonzero(mask)
    for py, px in zip(ys.tolist(), xs.tolist(), strict=True):
        for c in clusters:
            if abs(c[0] / c[2] - px) < CLUSTER_PX and abs(c[1] / c[2] - py) < CLUSTER_PX:
                c[0] += px
                c[1] += py
                c[2] += 1
                break
        else:
            clusters.append([px, py, 1])
    return clusters


def _is_key(sub: RGBImage, cx: float, cy: float) -> bool:
    """Whether the cluster at ``(cx, cy)`` is a bright, low-blue dungeon-key icon.

    Handles are muted brown; key icons share the hue but have almost no blue.

    Args:
        sub: Minimap crop.
        cx: Cluster centroid x.
        cy: Cluster centroid y.

    Returns:
        True if the cluster should be rejected as a key.
    """
    patch = sub[max(0, int(cy) - 3) : int(cy) + 4, max(0, int(cx) - 3) : int(cx) + 4].reshape(-1, 3)
    pm = _handle_mask(patch)
    if pm.any():
        mr, _, mb = patch[pm].mean(0)
        if mb < mr * 0.3:
            return True
    return False


def find_nodes(img: RGBImage, screen: ScreenGeometry = SCREEN) -> list[tuple[float, float, int]]:
    """Return node candidates relative to the hero on the minimap.

    Nodes are found by clustering brown-orange tool-handle pixels inside the minimap.

    Args:
        img: Full-screen RGB screenshot.
        screen: Screen geometry of the minimap.

    Returns:
        A list of ``(dx, dy, size)`` with ``size`` the handle pixel count.
    """
    sub = minimap(img, screen)
    mask = _handle_mask(sub)
    d2 = _disc_d2(screen)
    mask &= (d2 < screen.mm_r**2) & (d2 > screen.player_r**2)

    out: list[tuple[float, float, int]] = []
    for sx, sy, cnt in _cluster(mask):
        if cnt < MIN_CLUSTER:
            continue
        cx, cy = sx / cnt, sy / cnt
        if _is_key(sub, cx, cy):
            continue
        cx, cy = icon_center(sub, mask, cx, cy)
        out.append((cx - screen.mm_r, cy - screen.mm_r, cnt))
    return out


def icon_center(sub: RGBImage, handle: Mask, cx: float, cy: float) -> tuple[float, float]:
    """Centre of the whole tool icon (handle + neutral-grey metal) near a handle cluster.

    The handle sits off to one side of the node, so the icon's centre is a better target.

    Args:
        sub: Minimap crop.
        handle: Handle-pixel mask over ``sub``.
        cx: Handle cluster centroid x.
        cy: Handle cluster centroid y.

    Returns:
        The icon centre ``(x, y)`` in crop pixels, or ``(cx, cy)`` if too few pixels.
    """
    y0, x0 = max(0, int(cy) - ICON_R), max(0, int(cx) - ICON_R)
    p = sub[y0 : int(cy) + ICON_R + 1, x0 : int(cx) + ICON_R + 1]
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    metal = (p.mean(2) > 140) & (np.abs(r - b) < 25) & (np.abs(r - g) < 20)
    m = metal | handle[y0 : y0 + p.shape[0], x0 : x0 + p.shape[1]]
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return cx, cy
    return float(xs.mean() + x0), float(ys.mean() + y0)
