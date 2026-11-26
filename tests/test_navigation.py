"""Tests for BFS path planning over the walkable mask."""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.action.navigation import bfs, nearest_node
from eternal_hero_bot.core.config import PATHING, SCREEN

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


# --- Navigator / WorldMemory ---

from eternal_hero_bot.action.navigation import Navigator, WorldMemory  # noqa: E402
from eternal_hero_bot.core.state import MinimapView  # noqa: E402


def _view(
    walk: np.ndarray | None = None, hero_pos: tuple[float, float] = (0.0, 0.0)
) -> MinimapView:
    if walk is None:
        walk = np.ones((N, N), bool)
    return MinimapView(walk=walk, hero_pos=hero_pos)


def test_navigator_step_toward_node() -> None:
    """Navigator returns a Move aimed toward a visible node."""
    nav = Navigator()
    action = nav.step_toward(_view(), [(50.0, 0.0)], now=1000.0)
    assert action is not None
    assert action.dx > 0
    assert abs(action.dy) <= 1
    assert 250 <= action.ms <= 1000


def test_navigator_explores_when_no_nodes() -> None:
    """With no targets, navigator explores with EXPLORE_MS."""
    nav = Navigator()
    action = nav.step_toward(_view(), [], now=1000.0)
    assert action is not None
    assert action.ms == 1500


def test_navigator_stuck_returns_none_and_learns_wall() -> None:
    """A hero that never moves causes stuck handling and returns None."""
    nav = Navigator()
    view = _view()
    targets = [(50.0, 0.0)]
    action = None
    for _ in range(4):
        action = nav.step_toward(view, targets, now=1000.0)
    assert action is None
    assert nav.memory.walls


def test_navigator_reset_clears_memory_keeps_hist() -> None:
    """reset() clears world memory but preserves stuck history (parity)."""
    nav = Navigator()
    nav.memory.walls[(1, 1)] = 9999.0
    nav.hist.append(np.zeros(2))
    nav.reset()
    assert nav.memory.walls == {}
    assert len(nav.hist) == 1


def test_world_memory_blocked_view_and_expire() -> None:
    """WorldMemory projects walls hero-relative and expires old entries."""
    mem = WorldMemory(walls={(5, 0): 10.0, (6, 0): 1.0}, skips=[(1.0, 2.0, 1.0)])
    mem.expire(5.0)
    assert list(mem.walls) == [(5, 0)]
    assert mem.skips == []
    blocked = mem.blocked_view(np.zeros(2))
    assert blocked[SCREEN.mm_r, SCREEN.mm_r + 5]
    assert blocked.sum() == 1
