"""Navigation: BFS pathfinding, world memory, and the Navigator movement policy."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from eternal_hero_bot.core.actions import Move
from eternal_hero_bot.core.arrays import FloatArray, Mask, Offset, Point, RGBImage
from eternal_hero_bot.core.config import (
    PATHING,
    SCREEN,
    STUCK,
    PathingConfig,
    ScreenGeometry,
    StuckConfig,
)
from eternal_hero_bot.core.state import MinimapView
from eternal_hero_bot.perception.vision import find_nodes, walkable_mask

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
BLOCKED_FRACTION = 0.35
"""A push that covers less than this share of its expected distance hit something."""
BLOCKED_MIN_EXPECTED_PX = 6.0
"""Pushes expected to move less than this are too short to judge (tracker noise)."""
BLOCKED_STAMP_DEPTH: tuple[int, ...] = (3, 4, 5)
"""Distances ahead of the hero (crop pixels) stamped as wall after a blocked push."""
BLOCKED_STAMP_HALF_WIDTH = 8
"""Half-width, across the heading, of the wall stamped after a blocked push."""
BLOCKED_WALL_SECONDS = 60
"""How long a wall learned from a single blocked push is remembered.

Shorter than ``StuckConfig.block_seconds`` because one slow push (combat, a swipe hitch,
minimap lag) can be a false alarm, and a false wall must not seal a corridor for long.
"""
BLOCKED_SKIP_COUNT = 20
"""Blocked pushes toward one node area before that area is skipped (unreachable)."""
BLOCKED_STALE_S = 2.0
"""Slack past a push's own duration after which it is too old to judge."""
ARROW_RING_PX = 3
"""Pixels beyond the arrow radius never remembered as wall (the arrow's dilated edge)."""
PARTIAL_MIN_GAIN_PX = 3.0
"""A partial path must end at least this much closer to the target than the hero is."""


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
    *,
    partial: bool = False,
) -> list[Point] | None:
    """Shortest 8-connected path from the hero to the nearest reachable target.

    Targets are ``(dx, dy)`` relative to the hero; a small disc around each is made
    walkable since icons sit on their node.

    Args:
        walk: Square walkable mask in minimap-crop pixels (not modified).
        targets: Target offsets relative to the hero.
        centre: Hero position on both axes of the crop (the minimap radius).
        goal_r: Radius of the walkable goal disc around each target.
        partial: When no target is reachable, return the path to the reachable
            cell closest to any target instead (so the hero works its way around
            walls toward nodes whose ground route is not visible yet).

    Returns:
        The path as ``[(x, y), ...]`` in minimap-crop pixels from the hero to the
        goal, or None if no target is reachable (and, with ``partial``, no reachable
        cell is at least ``PARTIAL_MIN_GAIN_PX`` closer to a target than the hero).
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
            return _unwind(prev, (x, y))
        for ddx, ddy in NEIGHBOURS:
            nx, ny = x + ddx, y + ddy
            if 0 <= nx < n and 0 <= ny < n and walk[ny, nx] and (nx, ny) not in prev:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    if not partial or not targets:
        return None

    def gap(cell: Point) -> float:
        """Distance from ``cell`` to the closest target centre."""
        return min(math.hypot(centre + dx - cell[0], centre + dy - cell[1]) for dx, dy in targets)

    # BFS visits cells in path-length order, so ``min`` keeps the shortest route on ties
    best = min(prev, key=gap)
    if gap(best) > gap(start) - PARTIAL_MIN_GAIN_PX:
        return None
    return _unwind(prev, best)


def _unwind(prev: dict[Point, Point | None], end: Point) -> list[Point]:
    """Walk a BFS predecessor chain back from ``end`` to the start.

    Args:
        prev: Predecessor of every visited cell (``None`` for the start).
        end: Cell to unwind from.

    Returns:
        The path from the start to ``end``, inclusive.
    """
    path: list[Point] = []
    node: Point | None = end
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


def line_clear(walk: Mask, a: Point, b: Point) -> bool:
    """Whether the straight segment from ``a`` to ``b`` stays on walkable pixels.

    Args:
        walk: Walkable mask in crop pixels.
        a: Segment start ``(x, y)``.
        b: Segment end ``(x, y)``.

    Returns:
        True if every pixel sampled along the segment is walkable.
    """
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    for k in range(1, steps + 1):
        x = round(a[0] + (b[0] - a[0]) * k / steps)
        y = round(a[1] + (b[1] - a[1]) * k / steps)
        if not walk[y, x]:
            return False
    return True


def aim_index(walk: Mask, path: list[Point], lookahead: int) -> int:
    """Index of the path point to steer at: the lookahead point, unless that cuts a corner.

    Steering straight at ``path[lookahead]`` drives the hero into a wall when the path
    bends around a corner. This picks the farthest point up to ``lookahead`` that the
    hero can see in a straight line, so the push follows the path around obstacles.
    In open ground the result is exactly ``min(lookahead, len(path) - 1)``.

    Args:
        walk: Walkable mask the path was planned on.
        path: Planned path in crop pixels, starting at the hero.
        lookahead: Preferred steering distance along the path.

    Returns:
        The index into ``path`` to aim at (at least 1 when the path has two points).
    """
    far = min(lookahead, len(path) - 1)
    # Path cells and their 1px corridor count as clear: that covers the goal disc and
    # diagonal squeezes between wall pixels, while a real corner cut strays further
    los = walk.copy()
    n = walk.shape[0]
    for x, y in path:
        los[max(0, y - 1) : min(n, y + 2), max(0, x - 1) : min(n, x + 2)] = True
    for i in range(far, 1, -1):
        if line_clear(los, path[0], path[i]):
            return i
    return min(1, far)


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
        blocks: Coarse target area -> number of single blocked pushes toward it.
    """

    walls: dict[Point, float] = field(default_factory=dict)
    skips: list[tuple[float, float, float]] = field(default_factory=list)
    detours: dict[Point, int] = field(default_factory=dict)
    blocks: dict[Point, int] = field(default_factory=dict)

    def clear(self) -> None:
        """Forget everything (after death the map position is meaningless)."""
        self.walls.clear()
        self.skips.clear()
        self.detours.clear()
        self.blocks.clear()

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


@dataclass(frozen=True)
class _Push:
    """A path-following push, remembered to judge on the next frame whether it was blocked.

    Attributes:
        start: Hero world position when the push was planned.
        ux: Unit heading x.
        uy: Unit heading y.
        expected: Distance in minimap pixels the push should cover in the open.
        deadline: Time after which the push is too old to judge.
        target: World position of the node the push heads for.
    """

    start: FloatArray
    ux: float
    uy: float
    expected: float
    deadline: float
    target: tuple[float, float]


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
        self._last_push: _Push | None = None
        self._blocked_streak = 0
        self._prev_walk: tuple[Mask, FloatArray] | None = None
        n, pr = screen.mm_size, screen.player_r
        yy, xx = np.mgrid[0:n, 0:n]
        d2 = (xx - screen.mm_r) ** 2 + (yy - screen.mm_r) ** 2
        self._arrow: Mask = d2 < pr**2
        self._hidden: Mask = d2 < (pr + pathing.wall_margin + ARROW_RING_PX) ** 2

    def reset(self) -> None:
        """Clear world memory, the last push and the remembered mask. Stuck history is kept."""
        self.memory.clear()
        self._last_push = None
        self._blocked_streak = 0
        self._prev_walk = None

    def _fill_hidden(self, walk: Mask, pos: FloatArray) -> Mask:
        """Fill the ground hidden under the hero arrow from the previous frame's mask.

        The vision layer marks the whole arrow disc walkable because the arrow covers
        the ground, which hides walls the hero is standing next to; paths then run
        straight into them. The previous (already filled) mask, shifted by the hero's
        movement, still shows what lies under the arrow now.

        The arrow's own edge can read as a thin wall ring just outside the disc, so the
        remembered mask never takes the arrow or that ring from the current frame: they
        keep the remembered value (unknown = walkable). Otherwise the ring on the leading
        side would be carried forward as a fake wall right ahead of the hero.

        Args:
            walk: Walkable mask of this frame (crop pixels).
            pos: Hero world position this frame.

        Returns:
            ``walk`` with the arrow disc taken from the previous frame where known.
        """
        prev, self._prev_walk = self._prev_walk, None
        out = walk
        shifted = np.ones_like(walk)
        if prev is not None:
            prev_walk, prev_pos = prev
            sx, sy = (round(float(v)) for v in pos - prev_pos)
            n = walk.shape[0]
            if math.hypot(sx, sy) <= self.stuck.max_step_px:
                # Pixel (x, y) now was pixel (x + sx, y + sy) last frame; unseen = walkable
                ys, ye = max(0, -sy), min(n, n - sy)
                xs, xe = max(0, -sx), min(n, n - sx)
                shifted[ys:ye, xs:xe] = prev_walk[ys + sy : ye + sy, xs + sx : xe + sx]
                out = walk.copy()
                out[self._arrow] = shifted[self._arrow]
        stored = out.copy()
        stored[self._hidden] = shifted[self._hidden]
        self._prev_walk = (stored, pos.copy())
        return out

    def step_toward(self, view: MinimapView, targets: list[Offset], now: float) -> Move | None:
        """Plan and return the next move toward targets, or explore if none.

        Each frame: judge whether the last push was blocked (learning a wall at once),
        fill the ground hidden under the hero arrow from the previous frame, plan with
        BFS (partial toward the node when no full route is visible), then steer at the
        farthest path point in line of sight.

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
        self._check_last_push(pos, now)

        blocked = self.memory.blocked_view(pos, self.screen)
        skip_rel = self.memory.skip_offsets(pos)

        nodes = [
            (dx, dy)
            for dx, dy in targets
            if not any(math.hypot(dx - sx, dy - sy) < self.stuck.skip_radius for sx, sy in skip_rel)
        ]
        walk = self._fill_hidden(np.array(view.walk, dtype=bool), pos) & ~blocked
        open_start(walk, r)
        # Partial: with no visible ground route, work around walls toward the node anyway
        path = bfs(walk, nodes, r, self.pathing.goal_r, partial=True) if nodes else None

        if nodes and path is None:
            # Nothing reachable gets closer: head straight for the nearest node; walls get learned
            tx, ty = min(nodes, key=lambda p: math.hypot(*p))
            path = [(r, r)] + [
                (int(r + tx * t / FALLBACK_STEPS), int(r + ty * t / FALLBACK_STEPS))
                for t in range(1, FALLBACK_STEPS + 1)
            ]

        stuck = self._is_stuck(pos)
        if path is not None and stuck:
            self._handle_stuck(nodes, path, px, py, now)
            return None  # replan on a fresh frame

        if not nodes or path is None:
            logger.info("no nodes in view; exploring")
            return self.explore(walk)

        lookahead = self.pathing.lookahead
        gx, gy = path[-1][0] - r, path[-1][1] - r
        aim = aim_index(walk, path, lookahead)
        wx, wy = path[aim]
        dx, dy = wx - r, wy - r
        # Around a corner, push only as far as the visible point so the next frame turns
        seg = min(len(path), lookahead) if aim == min(lookahead, len(path) - 1) else aim
        ms = int(min(MAX_PUSH_MS, max(MIN_PUSH_MS, seg / self.pathing.px_per_sec * 1000)))
        tx, ty = nearest_node(nodes, *path[-1], centre=r)
        self._remember_push(pos, dx, dy, ms, now, (tx + px, ty + py))
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

    def _remember_push(
        self,
        pos: FloatArray,
        dx: float,
        dy: float,
        ms: int,
        now: float,
        target: tuple[float, float],
    ) -> None:
        """Record a path-following push so the next frame can tell if it was blocked.

        Args:
            pos: Hero world position when the push was planned.
            dx: Push direction x (any magnitude).
            dy: Push direction y (any magnitude).
            ms: Push duration in milliseconds.
            now: Time the push was planned, in seconds.
            target: World position of the node the push heads for.
        """
        length = math.hypot(dx, dy)
        if length == 0:
            # Standing on the node: nothing to judge, and the area proved reachable
            self._last_push = None
            area = self.stuck.area_cell
            self.memory.blocks.pop((int(target[0] // area), int(target[1] // area)), None)
            return
        expected = ms / 1000 * self.pathing.px_per_sec
        self._last_push = _Push(
            start=pos.copy(),
            ux=dx / length,
            uy=dy / length,
            expected=expected,
            deadline=now + ms / 1000 + BLOCKED_STALE_S,
            target=target,
        )

    def _check_last_push(self, pos: FloatArray, now: float) -> None:
        """Learn a wall straight ahead as soon as one push fails to make progress.

        Net-movement stuck detection needs several still frames; a single push that
        moves the hero far less than expected along its heading already shows the way
        is blocked (typically an invisible wall), so a short wall is stamped across
        the heading and the caller replans this same frame.

        Args:
            pos: Current hero world position.
            now: Current time in seconds.
        """
        last, self._last_push = self._last_push, None
        if last is None or now > last.deadline:
            return  # nothing pushed, or so long ago (e.g. gather was pre-empted) it says nothing
        ux, uy, expected = last.ux, last.uy, last.expected
        if expected < BLOCKED_MIN_EXPECTED_PX:
            return
        progress = float((pos[0] - last.start[0]) * ux + (pos[1] - last.start[1]) * uy)
        if progress >= BLOCKED_FRACTION * expected:
            self._blocked_streak = 0
            return
        self._blocked_streak += 1
        mem = self.memory
        area = self.stuck.area_cell
        tx, ty = last.target
        key = (int(tx // area), int(ty // area))
        mem.blocks[key] = mem.blocks.get(key, 0) + 1
        if mem.blocks[key] > BLOCKED_SKIP_COUNT:
            # Blocked again and again on the way to this node: treat it as unreachable
            logger.info(
                "giving up on node at (%+.0f,%+.0f) after %d blocked pushes for %ds",
                tx,
                ty,
                mem.blocks[key],
                self.stuck.skip_seconds,
            )
            mem.skips.append((tx, ty, now + self.stuck.skip_seconds))
            mem.blocks[key] = 0
            self.hist.clear()
            return
        expiry = now + BLOCKED_WALL_SECONDS
        w = BLOCKED_STAMP_HALF_WIDTH
        hx, hy = float(pos[0]), float(pos[1])
        for d in BLOCKED_STAMP_DEPTH:
            for s in range(-w, w + 1):
                # (-uy, ux) is the heading turned 90 degrees: the wall runs across it
                cell = (round(hx + ux * d - uy * s), round(hy + uy * d + ux * s))
                self.memory.walls[cell] = expiry
        if self._blocked_streak == 1:
            # The wall was handled; don't let net-movement stuck detection count it as a
            # detour. Repeated blocks with no progress in between fall through to it, so a
            # hero that cannot move at all still ends up skipping the node area.
            self.hist.clear()
        logger.info(
            "push blocked (%.1f of %.1fpx toward %+.2f,%+.2f); learned wall, replanning",
            progress,
            expected,
            ux,
            uy,
        )

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
        ``(nodes, walk, path)``: remaining node offsets, the walkable mask used and the BFS
        path (None if no node is reachable or none were found).
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
