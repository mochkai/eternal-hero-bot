"""Navigation: BFS pathfinding, world memory, and the Navigator movement policy."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from eternal_hero_bot.core.actions import Move
from eternal_hero_bot.core.arrays import FloatArray, Mask, Offset, Point
from eternal_hero_bot.core.config import (
    PATHING,
    SCREEN,
    STUCK,
    PathingConfig,
    ScreenGeometry,
    StuckConfig,
)
from eternal_hero_bot.core.state import MinimapView

logger = logging.getLogger(__name__)

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

EXPLORE_MS = 1500
"""Joystick hold time when exploring with no nodes in view."""
MIN_PUSH_MS = 250
"""Shortest joystick hold per step."""
MAX_PUSH_MS = 1000
"""Longest joystick hold per step."""
WALL_STAMP_R = 2
"""Half-size of the square stamped into world memory per blocked path pixel."""
FALLBACK_STEPS = 20
"""Number of points on the straight-line fallback path."""


def open_start(walk: Mask, centre: int = SCREEN.mm_r) -> None:
    """Force the 5x5 cells around the hero walkable so a path can always start.

    Args:
        walk: Walkable mask, modified in place.
        centre: Hero position on both axes of the crop.
    """
    walk[centre - 2 : centre + 3, centre - 2 : centre + 3] = True


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


@dataclass
class WorldMemory:
    """Obstacles and skipped areas remembered in world coordinates.

    World coordinates are minimap pixels relative to the tracker's origin, so the
    memory stays valid while the hero moves.

    Attributes:
        walls: World cell -> expiry time of a learned invisible wall.
        skips: ``(world x, world y, expiry)`` of skipped node areas.
        detours: Coarse target area -> number of blocked detours toward it.
    """

    walls: dict[Point, float] = field(default_factory=dict)
    skips: list[tuple[float, float, float]] = field(default_factory=list)
    detours: dict[Point, int] = field(default_factory=dict)

    def clear(self) -> None:
        """Forget everything (after death the map position is meaningless)."""
        self.walls.clear()
        self.skips.clear()
        self.detours.clear()

    def expire(self, now: float) -> None:
        """Drop walls and skips whose expiry has passed.

        Args:
            now: Current time in seconds.
        """
        self.walls = {k: t for k, t in self.walls.items() if t > now}
        self.skips = [s for s in self.skips if s[2] > now]

    def blocked_view(self, pos: FloatArray, screen: ScreenGeometry = SCREEN) -> Mask:
        """Project remembered walls into the current hero-relative minimap crop.

        Args:
            pos: Hero world position ``(x, y)``.
            screen: Screen geometry.

        Returns:
            A ``2R x 2R`` mask of remembered walls.
        """
        n, r = screen.mm_size, screen.mm_r
        px, py = pos
        blocked = np.zeros((n, n), bool)
        for wx, wy in self.walls:
            x, y = int(wx - px + r), int(wy - py + r)
            if 0 <= x < n and 0 <= y < n:
                blocked[y, x] = True
        return blocked

    def skip_offsets(self, pos: FloatArray) -> list[Offset]:
        """Skipped areas as offsets relative to the hero.

        Args:
            pos: Hero world position ``(x, y)``.

        Returns:
            Hero-relative ``(dx, dy)`` of each skipped area.
        """
        px, py = pos
        return [(sx - px, sy - py) for sx, sy, _ in self.skips]


class Navigator:
    """Movement policy extracted from Bot: stuck/fallback/explore/world memory.

    Attributes:
        memory: World-coordinate obstacles and skipped areas.
        hist: Recent hero positions used for stuck detection.
    """

    def __init__(
        self,
        screen: ScreenGeometry = SCREEN,
        pathing: PathingConfig = PATHING,
        stuck: StuckConfig = STUCK,
    ) -> None:
        """Create a navigator with empty world memory.

        Args:
            screen: Screen geometry.
            pathing: Path planning parameters.
            stuck: Stuck detection and world-memory parameters.
        """
        self.screen = screen
        self.pathing = pathing
        self.stuck = stuck
        self.memory = WorldMemory()
        self.hist: list[FloatArray] = []

    def reset(self) -> None:
        """Clear world memory. Stuck history is kept (parity)."""
        self.memory.clear()

    def step_toward(self, view: MinimapView, targets: list[Offset], now: float) -> Move | None:
        """Plan and return the next move toward targets, or explore if none.

        This is the body of today's ``Bot.step`` after the death check, kept in the
        same order for gather parity.

        Args:
            view: Current minimap view (walkable mask and hero position).
            targets: Hero-relative objective offsets.
            now: Current time in seconds.

        Returns:
            A :class:`~eternal_hero_bot.core.actions.Move` action, or ``None`` when
            stuck handling requires replanning on the next frame.
        """
        r = self.screen.mm_r
        pos = np.array(view.hero_pos, dtype=np.float64)
        px, py = float(pos[0]), float(pos[1])
        self.memory.expire(now)

        blocked = self.memory.blocked_view(pos, self.screen)
        skip_rel = self.memory.skip_offsets(pos)

        nodes = [
            (dx, dy)
            for dx, dy in targets
            if not any(math.hypot(dx - sx, dy - sy) < self.stuck.skip_radius for sx, sy in skip_rel)
        ]
        walk = np.array(view.walk, copy=True) & ~blocked
        open_start(walk, r)
        path = bfs(walk, nodes, r, self.pathing.goal_r) if nodes else None

        stuck = self._is_stuck(pos)
        if path is not None and stuck:
            self._handle_stuck(nodes, path, px, py, now)
            return None  # replan on a fresh frame

        if nodes and path is None:
            # No ground path found: head straight for the nearest node; walls get learned
            tx, ty = min(nodes, key=lambda p: math.hypot(*p))
            path = [(r, r)] + [
                (int(r + tx * t / FALLBACK_STEPS), int(r + ty * t / FALLBACK_STEPS))
                for t in range(1, FALLBACK_STEPS + 1)
            ]
        if not nodes or path is None:
            logger.info("no nodes in view; exploring")
            return self.explore(walk)

        lookahead = self.pathing.lookahead
        gx, gy = path[-1][0] - r, path[-1][1] - r
        wx, wy = path[min(lookahead, len(path) - 1)]
        dx, dy = wx - r, wy - r
        seg = min(len(path), lookahead)
        ms = int(min(MAX_PUSH_MS, max(MIN_PUSH_MS, seg / self.pathing.px_per_sec * 1000)))
        logger.info(
            "pos (%+.0f,%+.0f) node (%+d,%+d) path %d -> step (%+d,%+d) %dms walls %d",
            px,
            py,
            gx,
            gy,
            len(path),
            dx,
            dy,
            ms,
            len(self.memory.walls),
        )
        return Move(dx=float(dx), dy=float(dy), ms=ms)

    def explore(self, walk: Mask) -> Move:
        """Wander toward the most open of 8 compass directions.

        Args:
            walk: Current walkable mask (crop pixels).

        Returns:
            A move action in the most open direction.
        """
        r, player_r = self.screen.mm_r, self.screen.player_r

        def openness(a: int) -> int:
            """Count walkable samples along the ray at angle ``a * 45`` degrees."""
            ang = a * math.pi / 4
            return sum(
                bool(walk[int(r + d * math.sin(ang)), int(r + d * math.cos(ang))])
                for d in range(player_r, r - 2, 3)
            )

        best = max(range(8), key=openness)
        ang = best * math.pi / 4
        return Move(dx=math.cos(ang), dy=math.sin(ang), ms=EXPLORE_MS)

    def _is_stuck(self, pos: FloatArray) -> bool:
        """Record the current position and test for little *net* recent movement.

        Args:
            pos: Current hero world position.

        Returns:
            True if the hero moved less than ``stuck_px * stuck_steps`` over the
            last ``stuck_steps`` steps.
        """
        steps = self.stuck.stuck_steps
        self.hist.append(pos.copy())
        self.hist = self.hist[-(steps + 1) :]
        net = math.hypot(*(self.hist[-1] - self.hist[0])) if len(self.hist) > steps else 99.0
        return net < self.stuck.stuck_px * steps

    def _handle_stuck(
        self, nodes: list[Offset], path: list[Point], px: float, py: float, now: float
    ) -> None:
        """Learn an invisible wall ahead, or skip the target area after repeated detours.

        Args:
            nodes: Visible node offsets.
            path: The current BFS path (crop pixels).
            px: Hero world x.
            py: Hero world y.
            now: Current time in seconds.
        """
        r = self.screen.mm_r
        mem = self.memory
        tx, ty = nearest_node(nodes, *path[-1], centre=r)
        cell = self.stuck.area_cell
        key = (int((tx + px) // cell), int((ty + py) // cell))
        mem.detours[key] = mem.detours.get(key, 0) + 1
        if mem.detours[key] > self.stuck.max_detours:
            logger.info(
                "giving up on area around node (%+.0f,%+.0f) for %ds",
                tx,
                ty,
                self.stuck.skip_seconds,
            )
            mem.skips.append((tx + px, ty + py, now + self.stuck.skip_seconds))
        else:
            # Remember the stretch ahead as an invisible wall (world coords)
            expiry = now + self.stuck.block_seconds
            for x, y in path[2 : self.pathing.lookahead + 6]:
                for ddy in range(-WALL_STAMP_R, WALL_STAMP_R + 1):
                    for ddx in range(-WALL_STAMP_R, WALL_STAMP_R + 1):
                        mem.walls[(int(x + ddx - r + px), int(y + ddy - r + py))] = expiry
            logger.info("blocked ahead (area detour %d); replanning", mem.detours[key])
        self.hist.clear()
