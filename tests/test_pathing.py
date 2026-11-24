"""Tests for BFS path planning."""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.config import PATHING, SCREEN
from eternal_hero_bot.pathing import bfs, nearest_node

R = SCREEN.mm_r
N = 2 * R


def test_bfs_routes_around_wall() -> None:
    """A vertical wall between hero and target forces a detour around its end."""
    walk = np.ones((N, N), bool)
    walk[R - 30 : R + 31, R + 20 : R + 23] = False
    path = bfs(walk, [(60.0, 0.0)])
    assert path is not None
    assert path[0] == (R, R)
    assert all(walk[y, x] or (x - R - 60) ** 2 + (y - R) ** 2 < PATHING.goal_r**2 for x, y in path)
    gx, gy = path[-1]
    assert (gx - R - 60) ** 2 + (gy - R) ** 2 < PATHING.goal_r**2
    # It must leave the straight line to get past the wall
    assert max(abs(y - R) for _, y in path) > 30


def test_bfs_does_not_modify_input() -> None:
    """The caller's mask is left untouched."""
    walk = np.zeros((N, N), bool)
    walk[R, R] = True
    before = walk.copy()
    bfs(walk, [(30.0, 0.0)])
    assert (walk == before).all()


def test_bfs_unreachable_returns_none() -> None:
    """A hero boxed in by a closed ring of wall cannot reach anything."""
    ring = np.ones((N, N), bool)
    ring[R - 5 : R + 6, R - 5 : R + 6] = False
    ring[R - 4 : R + 5, R - 4 : R + 5] = True
    assert bfs(ring, [(60.0, 0.0)]) is None


def test_nearest_node() -> None:
    """The node closest to the given crop pixel is chosen."""
    nodes = [(10.0, 0.0), (-40.0, 5.0)]
    assert nearest_node(nodes, R - 38, R + 4) == (-40.0, 5.0)
