"""Deterministic fake device: a seeded synthetic world rendered as hero-centred minimaps.

`FakeWorld` implements the :class:`~eternal_hero_bot.device.base.Device` protocol without any
emulator. It backs a seeded synthetic world bitmap (walkable ground, wall cells,
handle-coloured nodes) and renders a 960x1080 frame whose minimap square shows the world
around the hero. Joystick swipes move the hero; every device call and sleep is appended to
`trace` for golden device-call traces.

No live device is used; everything is pure numpy and a seeded RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from eternal_hero_bot.core.arrays import Point, RGBImage
from eternal_hero_bot.core.config import COLOURS, PATHING, SCREEN, ScreenGeometry

HEIGHT, WIDTH = 1080, 960
"""Synthetic full-screen frame size (BlueStacks 960x1080 portrait)."""

GROUND_RGB: tuple[int, int, int] = COLOURS.ground_rgb
"""Walkable minimap ground colour."""
WALL_RGB: tuple[int, int, int] = (30, 38, 56)
"""Wall cell colour (dark navy, not walkable ground)."""
HANDLE_RGB: tuple[int, int, int] = (140, 87, 64)
"""Brown-orange tool-handle colour marking a gatherable node."""
DEATH_RGB: tuple[int, int, int] = (150, 200, 80)
"""Green colour at the death probe marking a 'You died' frame."""

NODE_PX = 4
"""Side length in pixels of a rendered node handle block."""

TraceEvent = dict[str, int | float | str]
"""One recorded event: ``kind`` plus numeric/string fields."""


@dataclass
class FakeWorld:
    """A seeded synthetic world driven as a fake :class:`Device`.

    Attributes:
        seed: Seed for the deterministic RNG used to build the world.
        world_size: Side length in pixels of the square world bitmap.
        screen: Screen geometry (minimap placement, joystick, revive, death probe).
        hero: Hero world position ``(x, y)`` in integer world pixels.
        nodes: World pixels of handle-coloured nodes.
        invisible_walls: World cells that block movement but render as ground.
        death_ticks: Screenshot tick indices that render a death frame.
        speed: World pixels per millisecond moved per unit direction.
        time: Current fake clock time in seconds.
        trace: Recorded events (``tap``/``swipe``/``sleep``) in call order.
    """

    seed: int = 0
    world_size: int = 256
    screen: ScreenGeometry = SCREEN
    hero: Point = field(default_factory=lambda: (0, 0))
    nodes: list[Point] = field(default_factory=list)
    invisible_walls: list[Point] = field(default_factory=list)
    death_ticks: set[int] = field(default_factory=set)
    speed: float = PATHING.px_per_sec / 1000.0
    time: float = 1000.0
    trace: list[TraceEvent] = field(default_factory=list)

    world: RGBImage = field(init=False)
    invisible: np.ndarray = field(init=False)
    tick: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Build the seeded world bitmap and invisible-wall mask."""
        ws = self.world_size
        self.world = np.full((ws, ws, 3), GROUND_RGB, np.int64)
        self.invisible = np.zeros((ws, ws), bool)
        for wx, wy in self.invisible_walls:
            if 0 <= wx < ws and 0 <= wy < ws:
                self.invisible[wy, wx] = True
        # Place hero inside the world if not given.
        if self.hero == (0, 0):
            self.hero = (ws // 2, ws // 2)

    # -- Device protocol ----------------------------------------------------
    def screenshot(self) -> RGBImage:
        """Render the hero-centred minimap into a 960x1080 frame.

        Returns:
            A fresh ``H x W x 3`` int64 RGB array; the minimap square samples the
            world around the hero, everything outside it is black. On a death tick
            the death probe pixel is set green.
        """
        s = self.screen
        r = s.mm_r
        n = 2 * r
        crop = np.full((n, n, 3), GROUND_RGB, np.int64)
        hx, hy = self.hero
        for i in range(n):
            wy = hy + (i - r)
            for j in range(n):
                wx = hx + (j - r)
                crop[i, j] = self._cell(wx, wy)
        # Draw handle-coloured nodes on top, hero-relative.
        for nx, ny in self.nodes:
            px = r + (nx - hx)
            py = r + (ny - hy)
            x_lo, x_hi = max(0, px), min(n, px + NODE_PX)
            y_lo, y_hi = max(0, py), min(n, py + NODE_PX)
            if x_hi > x_lo and y_hi > y_lo:
                crop[y_lo:y_hi, x_lo:x_hi] = HANDLE_RGB
        img = np.zeros((HEIGHT, WIDTH, 3), np.int64)
        x0 = s.mm_cx - r
        y0 = s.mm_cy - r
        img[y0 : y0 + n, x0 : x0 + n] = crop
        if self.tick in self.death_ticks:
            dx, dy = s.death_probe
            img[dy, dx] = DEATH_RGB
        self.tick += 1
        return img

    def tap(self, x: int, y: int) -> None:
        """Record a tap.

        Args:
            x: Screen x coordinate.
            y: Screen y coordinate.
        """
        self.trace.append({"kind": "tap", "x": int(x), "y": int(y)})

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int) -> None:
        """Record a swipe and, if it starts on the joystick, move the hero.

        The hero moves by ``direction * ms * speed`` world pixels, where ``direction``
        is the unit swipe direction, unless a wall or invisible-wall cell blocks the
        way; it stops at the last open cell.

        Args:
            x1: Start x.
            y1: Start y.
            x2: End x.
            y2: End y.
            ms: Duration in milliseconds.
        """
        x1, y1, x2, y2, ms = int(x1), int(y1), int(x2), int(y2), int(ms)
        self.trace.append({"kind": "swipe", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "ms": ms})
        s = self.screen
        if (x1, y1) != (s.joy_x, s.joy_y):
            return
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        dist = ms * self.speed
        hx, hy = self.hero
        nx = round(hx + ux * dist)
        ny = round(hy + uy * dist)
        self.hero = self._slide(hx, hy, nx, ny)

    # -- fake clock ---------------------------------------------------------
    def clock(self) -> float:
        """Return the current fake time in seconds."""
        return self.time

    def sleep(self, seconds: float) -> None:
        """Record a sleep and advance the fake clock.

        Args:
            seconds: Seconds to sleep.
        """
        seconds = float(seconds)
        self.trace.append({"kind": "sleep", "seconds": seconds})
        self.time += seconds

    # -- world --------------------------------------------------------------
    def _cell(self, wx: int, wy: int) -> tuple[int, int, int]:
        """Colour of world cell ``(wx, wy)``; invisible walls render as ground."""
        if 0 <= wx < self.world_size and 0 <= wy < self.world_size:
            r, g, b = self.world[wy, wx].tolist()
            return (int(r), int(g), int(b))
        return WALL_RGB

    def _blocked(self, wx: int, wy: int) -> bool:
        """Whether world cell ``(wx, wy)`` blocks hero movement."""
        if wx < 0 or wy < 0 or wx >= self.world_size or wy >= self.world_size:
            return True
        if self.world[wy, wx].tolist() == list(WALL_RGB):
            return True
        return bool(self.invisible[wy, wx])

    def _slide(self, hx: int, hy: int, nx: int, ny: int) -> Point:
        """Move from ``(hx, hy)`` toward ``(nx, ny)`` stopping before blocked cells."""
        dx = nx - hx
        dy = ny - hy
        n = max(abs(dx), abs(dy))
        if n == 0:
            return (hx, hy)
        cx, cy = hx, hy
        for k in range(1, n + 1):
            tx = hx + round(dx * k / n)
            ty = hy + round(dy * k / n)
            if self._blocked(tx, ty):
                break
            cx, cy = tx, ty
        return (cx, cy)

    def add_wall_rect(self, x: int, y: int, w: int, h: int) -> None:
        """Paint a wall rectangle into the world bitmap.

        Args:
            x: Left edge world x.
            y: Top edge world y.
            w: Width.
            h: Height.
        """
        self.world[y : y + h, x : x + w] = WALL_RGB

    # -- scenario constructors ---------------------------------------------
    @classmethod
    def open_field(cls, seed: int = 0, *, node_count: int = 6) -> FakeWorld:
        """An open field with handle-coloured nodes and no walls.

        Args:
            seed: RNG seed for node placement.
            node_count: Number of nodes to scatter.

        Returns:
            A `FakeWorld` for the open-field scenario.
        """
        ws = 256
        world = cls(seed=seed, world_size=ws)
        rng = np.random.default_rng(seed)
        cx, cy = ws // 2, ws // 2
        nodes: list[Point] = []
        for _ in range(node_count):
            for _ in range(100):
                nx = int(rng.integers(cx - 120, cx + 120))
                ny = int(rng.integers(cy - 120, cy + 120))
                if abs(nx - cx) < 24 and abs(ny - cy) < 24:
                    continue
                nodes.append((nx, ny))
                break
        world.nodes = nodes
        world.hero = (cx, cy)
        return world

    @classmethod
    def invisible_wall(cls, seed: int = 0) -> FakeWorld:
        """An open field with an invisible wall guarding a node.

        The node sits on the far side of a one-cell invisible barrier that renders as
        ground but blocks movement, forcing stuck detection, wall learning and, after
        repeated detours, area skipping.

        Args:
            seed: RNG seed (kept for reproducibility).

        Returns:
            A `FakeWorld` for the invisible-wall scenario.
        """
        ws = 256
        world = cls(seed=seed, world_size=ws)
        cx, cy = ws // 2, ws // 2
        world.hero = (cx, cy)
        # Node just east of a vertical invisible barrier; no ground wall is drawn.
        world.nodes = [(cx + 40, cy)]
        barrier_x = cx + 20
        world.invisible_walls = [(barrier_x, cy + dy) for dy in range(-60, 61)]
        world.invisible = np.zeros((ws, ws), bool)
        for wx, wy in world.invisible_walls:
            world.invisible[wy, wx] = True
        return world

    @classmethod
    def death_mid_run(cls, seed: int = 0, *, death_tick: int = 30) -> FakeWorld:
        """An open field with nodes where a death frame is injected mid-run.

        Args:
            seed: RNG seed for node placement.
            death_tick: Screenshot tick index that renders a death frame.

        Returns:
            A `FakeWorld` for the death scenario.
        """
        world = cls.open_field(seed, node_count=6)
        world.death_ticks = {death_tick}
        return world


GOLDEN_TICKS = 80
"""Number of `Runtime` ticks per golden scenario."""

SCENARIOS: tuple[str, ...] = ("open_field", "invisible_wall", "death")
"""Golden scenario names; golden files are `tests/golden/gather_<name>.json`."""


def scenario(name: str, seed: int = 0) -> FakeWorld:
    """Build a `FakeWorld` for a named golden scenario.

    Args:
        name: One of `SCENARIOS`.
        seed: RNG seed.

    Returns:
        The scenario's `FakeWorld`.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "open_field":
        return FakeWorld.open_field(seed)
    if name == "invisible_wall":
        return FakeWorld.invisible_wall(seed)
    if name == "death":
        return FakeWorld.death_mid_run(seed)
    raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")


def run_scenario(world: FakeWorld, ticks: int = GOLDEN_TICKS) -> list[TraceEvent]:
    """Run the runtime on `world` for `ticks` steps and return the trace.

    Args:
        world: The fake device world.
        ticks: Number of `Runtime` ticks.

    Returns:
        The recorded event list (`tap`/`swipe`/`sleep`, in order).
    """
    from eternal_hero_bot.runtime import FixedBrain, Runtime

    runtime = Runtime(world, brain=FixedBrain(), clock=world.clock, sleep=world.sleep)
    for _ in range(ticks):
        runtime.tick()
    return list(world.trace)
