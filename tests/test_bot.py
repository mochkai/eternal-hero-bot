"""Tests for the gather loop using a fake device."""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.action.controller import push_joystick
from eternal_hero_bot.action.navigation import bfs
from eternal_hero_bot.bot import Bot, WorldMemory
from eternal_hero_bot.core.arrays import RGBImage
from eternal_hero_bot.core.config import SCREEN, STUCK
from eternal_hero_bot.perception.vision import find_nodes
from tests.helpers import HANDLE_RGB, blank_screen, paint


class FakeDevice:
    """A :class:`~eternal_hero_bot.device.base.Device` serving a fixed screen, recording input."""

    def __init__(self, img: RGBImage) -> None:
        """Create the fake.

        Args:
            img: Screen returned by every :meth:`screenshot`.
        """
        self.img = img
        self.taps: list[tuple[int, int]] = []
        self.swipes: list[tuple[int, int, int, int, int]] = []

    def screenshot(self) -> RGBImage:
        """Return the fixed screen.

        Returns:
            A copy of the screen.
        """
        return self.img.copy()

    def tap(self, x: int, y: int) -> None:
        """Record a tap.

        Args:
            x: Screen x.
            y: Screen y.
        """
        self.taps.append((x, y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int) -> None:
        """Record a swipe.

        Args:
            x1: Start x.
            y1: Start y.
            x2: End x.
            y2: End y.
            ms: Duration.
        """
        self.swipes.append((x1, y1, x2, y2, ms))


def _bot(img: RGBImage) -> tuple[Bot, FakeDevice]:
    """Build a bot on a fake device with a frozen clock and no-op sleep.

    Args:
        img: Screen the fake device serves.

    Returns:
        The bot and its device.
    """
    dev = FakeDevice(img)
    return Bot(dev, clock=lambda: 1000.0, sleep=lambda _s: None), dev


def test_step_pushes_joystick_toward_node() -> None:
    """With a node due east, one step pushes the joystick east."""
    img = blank_screen()
    paint(img, 50, -2, 4, 4, HANDLE_RGB)
    bot, dev = _bot(img)
    bot.step()
    assert len(dev.swipes) == 1
    x1, y1, x2, y2, ms = dev.swipes[0]
    assert (x1, y1) == (SCREEN.joy_x, SCREEN.joy_y)
    assert x2 == SCREEN.joy_x + SCREEN.joy_r
    assert abs(y2 - SCREEN.joy_y) <= 1
    assert 250 <= ms <= 1000
    assert dev.taps == []


def test_step_revives_when_dead() -> None:
    """On the death dialog the bot taps Revive, clears its memory and resets tracking."""
    img = blank_screen()
    x, y = SCREEN.death_probe
    img[y, x] = (150, 200, 80)
    bot, dev = _bot(img)
    bot.memory.walls[(1, 1)] = 9999.0
    bot.tracker.prev = np.zeros((SCREEN.mm_size, SCREEN.mm_size))
    bot.tracker.pos[:] = (7.0, -3.0)
    bot.step()
    assert dev.taps == [SCREEN.revive]
    assert dev.swipes == []
    assert bot.memory.walls == {}
    assert bot.tracker.prev is None
    assert (bot.tracker.pos == 0).all()


def test_step_explores_without_nodes() -> None:
    """With no nodes in view the bot still pushes the joystick (exploring)."""
    bot, dev = _bot(blank_screen())
    bot.step()
    assert len(dev.swipes) == 1
    assert dev.swipes[0][4] == 1500


def test_stuck_learns_wall_ahead() -> None:
    """A hero that never moves eventually gets a learned wall on its path."""
    img = blank_screen()
    paint(img, 50, -2, 4, 4, HANDLE_RGB)
    bot, dev = _bot(img)
    for _ in range(4):
        bot.step()
    assert bot.memory.walls
    assert len(dev.swipes) == 3  # the stuck step replans instead of pushing


def test_world_memory_expire_and_project() -> None:
    """Expired entries are dropped; live walls project relative to the hero."""
    mem = WorldMemory(walls={(5, 0): 10.0, (6, 0): 1.0}, skips=[(1.0, 2.0, 1.0)])
    mem.expire(5.0)
    assert list(mem.walls) == [(5, 0)]
    assert mem.skips == []
    blocked = mem.blocked_view(np.zeros(2))
    assert blocked[SCREEN.mm_r, SCREEN.mm_r + 5]
    assert blocked.sum() == 1


def test_push_joystick_normalises_direction() -> None:
    """The swipe length is the joystick radius regardless of input magnitude."""
    dev = FakeDevice(blank_screen())
    push_joystick(dev, 0.0, -10.0, 300)
    assert dev.swipes == [
        (SCREEN.joy_x, SCREEN.joy_y, SCREEN.joy_x, SCREEN.joy_y - SCREEN.joy_r, 300)
    ]


def test_history_recorded_even_without_path() -> None:
    """Position history grows on every live step, including explore steps with no path."""
    bot, _ = _bot(blank_screen())
    bot.step()
    bot.step()
    assert len(bot.hist) == 2


def test_repeated_detours_skip_node_area() -> None:
    """Past max_detours the node's area is skipped instead of stamping more walls."""
    img = blank_screen()
    paint(img, 50, -2, 4, 4, HANDLE_RGB)
    bot, dev = _bot(img)
    nodes = [(dx, dy) for dx, dy, _ in find_nodes(img)]
    assert len(nodes) == 1
    tx, ty = nodes[0]
    path = bfs(np.ones((SCREEN.mm_size, SCREEN.mm_size), bool), nodes)
    assert path is not None
    cell = STUCK.area_cell
    key = (int(tx // cell), int(ty // cell))
    bot.memory.detours[key] = STUCK.max_detours
    bot._handle_stuck(nodes, path, 0.0, 0.0, 1000.0)
    assert bot.memory.skips == [(tx, ty, 1000.0 + STUCK.skip_seconds)]
    assert bot.memory.walls == {}
    # The skipped node is filtered out, so the next step explores
    bot.step()
    assert len(dev.swipes) == 1
    assert dev.swipes[0][4] == 1500
