"""Breadth-first path search over the minimap walkable mask."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from eternal_hero_bot.core.arrays import Mask, Offset, Point
from eternal_hero_bot.core.config import PATHING, SCREEN

NEIGHBOURS: tuple[Point, ...] = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
"""8-connected neighbourhood, in the order the search expands it."""


def bfs(
    walk: Mask,
    targets: list[Offset],
    centre: int = SCREEN.mm_r,
    goal_r: int = PATHING.goal_r,
) -> list[Point] | None:
    """Shortest 8-connected path from the hero to the nearest reachable target.

    Targets are ``(dx, dy)`` relative to the hero; a small disc around each is made
    walkable since icons sit on their node.

    Args:
        walk: Square walkable mask in minimap-crop pixels (not modified).
        targets: Target offsets relative to the hero.
        centre: Hero position on both axes of the crop (the minimap radius).
        goal_r: Radius of the walkable goal disc around each target.

    Returns:
        The path as ``[(x, y), ...]`` in minimap-crop pixels from the hero to the
        goal, or None if no target is reachable.
    """
    walk = walk.copy()
    n = walk.shape[0]
    goal = np.zeros_like(walk)
    yy, xx = np.mgrid[0:n, 0:n]
    for dx, dy in targets:
        disc = (xx - (centre + dx)) ** 2 + (yy - (centre + dy)) ** 2 < goal_r**2
        walk |= disc
        goal |= disc
    start = (centre, centre)
    prev: dict[Point, Point | None] = {start: None}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if goal[y, x]:
            # Walk the predecessor chain back to the start
            path: list[Point] = []
            node: Point | None = (x, y)
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        for ddx, ddy in NEIGHBOURS:
            nx, ny = x + ddx, y + ddy
            if 0 <= nx < n and 0 <= ny < n and walk[ny, nx] and (nx, ny) not in prev:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return None


def nearest_node(nodes: list[Offset], x: int, y: int, centre: int = SCREEN.mm_r) -> Offset:
    """The node whose centre is closest to minimap-crop pixel ``(x, y)``.

    Args:
        nodes: Node offsets relative to the hero (non-empty).
        x: Crop pixel x.
        y: Crop pixel y.
        centre: Hero position on both axes of the crop.

    Returns:
        The closest node offset.

    Raises:
        ValueError: If ``nodes`` is empty.
    """
    return min(nodes, key=lambda n: math.hypot(centre + n[0] - x, centre + n[1] - y))
