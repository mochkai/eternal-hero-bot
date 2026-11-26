"""The gather loop: perceive -> navigate -> control, with revive on death."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

from eternal_hero_bot.action.controller import Controller
from eternal_hero_bot.action.navigation import (  # noqa: F401
    EXPLORE_MS,
    FALLBACK_STEPS,
    MAX_PUSH_MS,
    MIN_PUSH_MS,
    WALL_STAMP_R,
    Navigator,
    bfs,
    nearest_node,
    open_start,
)
from eternal_hero_bot.action.navigation import (
    WorldMemory as WorldMemory,
)
from eternal_hero_bot.core.actions import Tap, Wait
from eternal_hero_bot.core.arrays import FloatArray, Mask, Offset, Point, RGBImage
from eternal_hero_bot.core.config import (
    PATHING,
    SCREEN,
    STUCK,
    PathingConfig,
    ScreenGeometry,
    StuckConfig,
)
from eternal_hero_bot.core.state import ScreenKind
from eternal_hero_bot.device.base import Device
from eternal_hero_bot.perception.frame import Frame
from eternal_hero_bot.perception.perceiver import DefaultPerceiver
from eternal_hero_bot.perception.tracker import Tracker
from eternal_hero_bot.perception.vision import find_nodes, walkable_mask

logger = logging.getLogger(__name__)

REVIVE_WAIT_S = 3.0
"""Seconds to wait after tapping Revive."""


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


class Bot:
    """The gather loop, one :meth:`step` per screenshot.

    The bot wires perception, navigation and control: it perceives a frame into a
    :class:`~eternal_hero_bot.core.state.GameState`, asks the
    :class:`~eternal_hero_bot.action.navigation.Navigator` for the next action, and
    executes it through the :class:`~eternal_hero_bot.action.controller.Controller`.
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
        """Create a bot with a fresh perceiver, navigator and controller.

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
        self.perceiver = DefaultPerceiver(screen=screen, pathing=pathing, stuck=stuck)
        self.navigator = Navigator(screen=screen, pathing=pathing, stuck=stuck)
        self.controller = Controller(device=device, assets=None, screen=screen, sleep=sleep)
        self._frame_index = 0

    @property
    def tracker(self) -> Tracker:
        """The perceiver's phase-correlation tracker (parity/test access)."""
        return self.perceiver.tracker

    @tracker.setter
    def tracker(self, value: Tracker) -> None:
        """Replace the perceiver tracker (parity/test access)."""
        self.perceiver._tracker = value

    @property
    def memory(self) -> WorldMemory:
        """The navigator's world memory (parity/test access)."""
        return self.navigator.memory

    @memory.setter
    def memory(self, value: WorldMemory) -> None:
        """Replace navigator world memory (parity/test access)."""
        self.navigator.memory = value

    @property
    def hist(self) -> list[FloatArray]:
        """Navigator stuck-detection history (parity/test access)."""
        return self.navigator.hist

    @hist.setter
    def hist(self, value: list[FloatArray]) -> None:
        """Replace navigator stuck-detection history (parity/test access)."""
        self.navigator.hist = value

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
        img = self.device.screenshot()
        frame = Frame(image=img, t=self.clock(), index=self._frame_index)
        self._frame_index += 1
        state = self.perceiver.perceive(frame)

        if state.screen == ScreenKind.DEAD:
            logger.info("died -> revive")
            # parity: fixed revive coordinate when no manifest region exists.
            self.controller.execute(Tap(*self.screen.revive))
            self.controller.execute(Wait(REVIVE_WAIT_S))
            self.navigator.reset()
            return

        if state.minimap is None:
            return

        targets = [(obj.dx, obj.dy) for obj in state.objectives]
        action = self.navigator.step_toward(state.minimap, targets, now=self.clock())
        if action is not None:
            self.controller.execute(action)

    def _handle_stuck(
        self, nodes: list[Offset], path: list[Point], px: float, py: float, now: float
    ) -> None:
        """Delegate to navigator stuck handling (parity/test access).

        Args:
            nodes: Visible node offsets.
            path: The current BFS path (crop pixels).
            px: Hero world x.
            py: Hero world y.
            now: Current time in seconds.
        """
        self.navigator._handle_stuck(nodes, path, px, py, now)
