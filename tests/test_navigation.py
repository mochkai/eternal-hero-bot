"""Tests for BFS path planning over the walkable mask."""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.action.navigation import bfs, nearest_node
from eternal_hero_bot.core.config import PATHING, SCREEN, STUCK

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


def test_navigator_blocked_push_corrects_course_next_step() -> None:
    """One push that goes nowhere learns a wall ahead and turns away on the very next step.

    Before issue #18 this took four steps (three still frames of net-movement stuck
    detection, then a ``None`` replan frame) before the heading changed.
    """
    nav = Navigator()
    view = _view()
    targets = [(50.0, 0.0)]
    first = nav.step_toward(view, targets, now=1000.0)
    assert first is not None and first.dy == 0 and first.dx > 0
    second = nav.step_toward(view, targets, now=1000.0)  # hero did not move
    assert second is not None
    assert nav.memory.walls
    assert abs(second.dy) > 0  # heading changed to go round the learned wall


def test_navigator_immobile_hero_still_hits_stuck_handling() -> None:
    """Back-to-back blocked pushes fall through to net-movement stuck handling (None)."""
    nav = Navigator()
    view = _view()
    targets = [(50.0, 0.0)]
    actions = [nav.step_toward(view, targets, now=1000.0) for _ in range(6)]
    assert None in actions
    assert sum(nav.memory.detours.values()) >= 1


def test_navigator_immobile_hero_eventually_skips_node() -> None:
    """A hero that can never move ends up skipping the node area rather than looping."""
    nav = Navigator()
    view = _view()
    targets = [(50.0, 0.0)]
    for _ in range(60):
        nav.step_toward(view, targets, now=1000.0)
    assert nav.memory.skips


def test_navigator_small_progress_not_judged_blocked() -> None:
    """A push that covers most of its expected distance does not learn a wall."""
    nav = Navigator()
    targets = [(50.0, 0.0)]
    first = nav.step_toward(_view(), targets, now=1000.0)
    assert first is not None
    nav.step_toward(_view(hero_pos=(9.0, 0.0)), [(41.0, 0.0)], now=1001.0)
    assert not nav.memory.walls


def _arrow_frame(ring: bool = False) -> np.ndarray:
    """Open walkable mask as vision reports it: the arrow disc is always walkable.

    Args:
        ring: Also draw the arrow's dilated edge as a 1px wall ring just outside the disc.

    Returns:
        A ``2R x 2R`` mask.
    """
    walk = np.ones((N, N), bool)
    yy, xx = np.mgrid[0:N, 0:N]
    d2 = (xx - R) ** 2 + (yy - R) ** 2
    if ring:
        walk[(d2 >= SCREEN.player_r**2) & (d2 < (SCREEN.player_r + 1) ** 2)] = False
    return walk


def test_navigator_fills_arrow_disc_from_previous_frame() -> None:
    """A wall seen last frame stays known once the hero arrow covers it."""
    nav = Navigator()
    walk = _arrow_frame()
    walk[:, R + 17 : R + 19] = False  # wall 17px east, well outside the arrow
    nav._fill_hidden(walk, np.array([0.0, 0.0]))
    # Next frame: hero moved 9px east; the wall is 8px away, hidden under the arrow
    out = nav._fill_hidden(_arrow_frame(), np.array([9.0, 0.0]))
    assert not out[R, R + 8]
    assert out[R, R + 2]  # open ground under the arrow stays walkable


def test_navigator_arrow_edge_not_remembered_as_wall() -> None:
    """The arrow's own edge ring is not carried forward as a fake wall ahead of the hero."""
    nav = Navigator()
    out = None
    for k in range(5):
        out = nav._fill_hidden(_arrow_frame(ring=True), np.array([6.0 * k, 0.0]))
    assert out is not None
    yy, xx = np.mgrid[0:N, 0:N]
    arrow = (xx - R) ** 2 + (yy - R) ** 2 < SCREEN.player_r**2
    assert out[arrow].all()


def test_navigator_single_block_wall_expires_sooner() -> None:
    """Walls from one blocked push expire well before confirmed (stuck) walls."""
    nav = Navigator()
    targets = [(50.0, 0.0)]
    nav.step_toward(_view(), targets, now=1000.0)
    nav.step_toward(_view(), targets, now=1001.0)
    assert nav.memory.walls
    assert max(nav.memory.walls.values()) < 1001.0 + STUCK.block_seconds


def test_navigator_stale_push_not_judged() -> None:
    """A push from long ago (gather pre-empted meanwhile) does not learn a wall."""
    nav = Navigator()
    targets = [(50.0, 0.0)]
    nav.step_toward(_view(), targets, now=1000.0)
    nav.step_toward(_view(), targets, now=1100.0)
    assert not nav.memory.walls


def test_navigator_reset_clears_push_and_mask_memory() -> None:
    """reset() forgets the last push and remembered mask (positions jump on revive)."""
    nav = Navigator()
    nav.step_toward(_view(), [(50.0, 0.0)], now=1000.0)
    nav.reset()
    action = nav.step_toward(_view(), [(50.0, 0.0)], now=1000.0)
    assert not nav.memory.walls
    assert action is not None and action.dy == 0


# --- line of sight / partial paths ---

from eternal_hero_bot.action.navigation import aim_index, line_clear  # noqa: E402


def test_line_clear() -> None:
    """A segment is clear unless it crosses a non-walkable pixel."""
    walk = np.ones((N, N), bool)
    assert line_clear(walk, (R, R), (R + 20, R + 10))
    walk[R + 5, R + 10] = False
    assert not line_clear(walk, (R, R), (R + 20, R + 10))


def test_aim_index_open_ground_is_lookahead() -> None:
    """In the open the steering point is exactly the lookahead point (golden parity)."""
    walk = np.ones((N, N), bool)
    path = bfs(walk, [(40.0, 17.0)])
    assert path is not None
    assert aim_index(walk, path, PATHING.lookahead) == PATHING.lookahead


def test_aim_index_does_not_cut_corners() -> None:
    """Round a wall corner the hero aims at a point it can actually walk straight to."""
    walk = np.ones((N, N), bool)
    walk[R - 20 : R + 6, R + 3 : R + 20] = False  # block east/north-east of the hero
    # Path: 7px south, then east along the block's southern face
    path = [(R, R + k) for k in range(8)] + [(R + k, R + 7) for k in range(1, 10)]
    i = aim_index(walk, path, PATHING.lookahead)
    assert i < PATHING.lookahead
    assert line_clear(walk, path[0], path[i])


def test_bfs_partial_heads_toward_unreachable_target() -> None:
    """With no route to the target, partial BFS returns a path to the closest reachable cell."""
    walk = np.ones((N, N), bool)
    walk[:, R + 20 : R + 23] = False  # wall right across the minimap
    assert bfs(walk, [(60.0, 0.0)]) is None
    path = bfs(walk, [(60.0, 0.0)], partial=True)
    assert path is not None
    end_x, end_y = path[-1]
    assert end_x == R + 19 and abs(end_y - R) <= 1


def test_bfs_partial_none_when_no_progress_possible() -> None:
    """Partial BFS gives up when nothing reachable is closer than the hero already is."""
    ring = np.ones((N, N), bool)
    ring[R - 5 : R + 6, R - 5 : R + 6] = False
    ring[R - 4 : R + 5, R - 4 : R + 5] = True
    path = bfs(ring, [(60.0, 0.0)], partial=True)
    assert path is not None  # can still shuffle 4px east inside the ring
    ring[R - 4 : R + 5, R - 4 : R + 5] = False
    ring[R, R] = True
    assert bfs(ring, [(60.0, 0.0)], partial=True) is None


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
