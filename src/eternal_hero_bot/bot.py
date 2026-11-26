"""The gather loop: plan to the nearest node, learn invisible walls, revive on death."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from eternal_hero_bot.core.arrays import FloatArray, Mask, Offset, Point, RGBImage
from eternal_hero_bot.core.config import (
    PATHING,
    SCREEN,
    STUCK,
    PathingConfig,
    ScreenGeometry,
    StuckConfig,
)
from eternal_hero_bot.device import Device, push_joystick
from eternal_hero_bot.pathing import bfs, nearest_node
from eternal_hero_bot.tracker import Tracker
from eternal_hero_bot.vision import find_nodes, is_dead, walkable_mask

logger = logging.getLogger(__name__)

REVIVE_WAIT_S = 3.0
"""Seconds to wait after tapping Revive."""
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


def plan(
    img: RGBImage,
    skip: list[Offset],
    blocked: Mask | None = None,
    screen: ScreenGeometry = SCREEN,
    pathing: PathingConfig = PATHING,
) -> tuple[list[Offset], Mask, list[Point] | None]:
    """Find nodes, drop skipped ones, and path to the nearest reachable one.

    Args:
        img: Full-screen RGB screenshot.
        skip: Hero-relative offsets of nodes to ignore.
        blocked: Extra wall mask learned from places the hero got stuck.
        screen: Screen geometry.
        pathing: Pathing parameters.

    Returns:
        ``(nodes, walk, path)``: remaining node offsets, the walkable mask used and
        the BFS path (None if no node is reachable or none were found).
    """
    nodes = [
        (dx, dy)
        for dx, dy, _ in find_nodes(img, screen)
        if not any(math.hypot(dx - sx, dy - sy) < pathing.skip_match_r for sx, sy in skip)
    ]
    walk = walkable_mask(img, screen, wall_margin=pathing.wall_margin)
    if blocked is not None:
        walk &= ~blocked
        open_start(walk, screen.mm_r)
    path = bfs(walk, nodes, screen.mm_r, pathing.goal_r) if nodes else None
    return nodes, walk, path


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


class Bot:
    """The gather loop, one :meth:`step` per screenshot.

    The hero's world position is tracked from minimap motion. When pushing doesn't
    move the hero, the path stretch ahead is stored as a wall in *world* coordinates
    (so it survives the hero sliding around) and we replan around it. After
    ``max_detours`` failures near one node, that whole area is skipped for
    ``skip_seconds``.
    """

    def __init__(
        self,
        device: Device,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        screen: ScreenGeometry = SCREEN,
        pathing: PathingConfig = PATHING,
        stuck: StuckConfig = STUCK,
    ) -> None:
        """Create a bot with a fresh tracker and empty world memory.

        Args:
            device: Device to screenshot and send input to.
            clock: Returns the current time in seconds (injectable for tests).
            sleep: Sleeps for the given seconds (injectable for tests).
            screen: Screen geometry.
            pathing: Path planning parameters.
            stuck: Stuck detection and world-memory parameters.
        """
        self.device = device
        self.clock = clock
        self.sleep = sleep
        self.screen = screen
        self.pathing = pathing
        self.stuck = stuck
        self.tracker = Tracker(screen, stuck.max_step_px)
        self.memory = WorldMemory()
        self.hist: list[FloatArray] = []

    def run(self, minutes: float) -> None:
        """Run :meth:`step` until ``minutes`` have elapsed.

        Args:
            minutes: How long to gather for.
        """
        end = self.clock() + minutes * 60
        while self.clock() < end:
            self.step()

    def step(self) -> None:
        """Take one screenshot and act on it (revive, replan, explore or move)."""
        r = self.screen.mm_r
        img = self.device.screenshot()
        if is_dead(img, self.screen):
            logger.info("died -> revive")
            self.device.tap(*self.screen.revive)
            self.sleep(REVIVE_WAIT_S)
            self.tracker = Tracker(self.screen, self.stuck.max_step_px)
            self.memory.clear()
            return

        walk_raw = walkable_mask(img, self.screen, wall_margin=self.pathing.wall_margin)
        self.tracker.update(walk_raw.astype(float))
        now = self.clock()
        pos = self.tracker.pos
        px, py = pos
        self.memory.expire(now)

        # Project remembered walls / skips into the current hero-relative view
        blocked = self.memory.blocked_view(pos, self.screen)
        skip_rel = self.memory.skip_offsets(pos)

        nodes = [
            (dx, dy)
            for dx, dy, _ in find_nodes(img, self.screen)
            if not any(math.hypot(dx - sx, dy - sy) < self.stuck.skip_radius for sx, sy in skip_rel)
        ]
        walk = walk_raw & ~blocked
        open_start(walk, r)
        path = bfs(walk, nodes, r, self.pathing.goal_r) if nodes else None

        # Always record history first (even with no path), then test for being stuck
        stuck = self._is_stuck()
        if path is not None and stuck:
            self._handle_stuck(nodes, path, px, py, now)
            return  # replan on a fresh frame

        if nodes and path is None:
            # No ground path found: head straight for the nearest node; walls get learned
            tx, ty = min(nodes, key=lambda p: math.hypot(*p))
            path = [(r, r)] + [
                (int(r + tx * t / FALLBACK_STEPS), int(r + ty * t / FALLBACK_STEPS))
                for t in range(1, FALLBACK_STEPS + 1)
            ]
        if not nodes or path is None:
            logger.info("no nodes in view; exploring")
            self._explore(walk)
            return

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
        push_joystick(self.device, dx, dy, ms, self.screen)

    def _is_stuck(self) -> bool:
        """Record the current position and test for little *net* recent movement.

        Net movement over the last few steps catches back-and-forth jitter too.

        Returns:
            True if the hero moved less than ``stuck_px * stuck_steps`` over the
            last ``stuck_steps`` steps.
        """
        steps = self.stuck.stuck_steps
        self.hist.append(self.tracker.pos.copy())
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

    def _explore(self, walk: Mask) -> None:
        """Wander toward the most open of 8 compass directions.

        Args:
            walk: Current walkable mask (crop pixels).
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
        push_joystick(self.device, math.cos(ang), math.sin(ang), EXPLORE_MS, self.screen)
