"""Tests for the runtime main loop: gather parity through the skills and lifecycle."""

from __future__ import annotations

import numpy as np
import pytest

from eternal_hero_bot.action.navigation import bfs
from eternal_hero_bot.core.actions import Move
from eternal_hero_bot.core.config import SCREEN, STUCK
from eternal_hero_bot.core.outcomes import SkillStatus
from eternal_hero_bot.perception.vision import find_nodes
from eternal_hero_bot.runtime import FixedBrain, Runtime
from tests.helpers import HANDLE_RGB, FakeDevice, blank_screen, paint


def _runtime(dev: FakeDevice) -> Runtime:
    """Build a runtime on a fake device with a frozen clock and no-op sleep.

    Args:
        dev: Fake device serving the synthetic screens.

    Returns:
        The runtime with the fixed brain and default skills.
    """
    return Runtime(dev, brain=FixedBrain(), clock=lambda: 1000.0, sleep=lambda _s: None)


def _dead_device() -> FakeDevice:
    """A fake device whose fixed screen shows the death dialog.

    Returns:
        The fake device with a green death-probe pixel.
    """
    img = blank_screen()
    x, y = SCREEN.death_probe
    img[y, x] = (150, 200, 80)
    return FakeDevice(img)


def test_tick_pushes_joystick_toward_node() -> None:
    """One runtime tick on a world frame pushes the joystick toward a visible node."""
    img = blank_screen()
    paint(img, 50, -2, 4, 4, HANDLE_RGB)
    dev = FakeDevice(img)
    rt = _runtime(dev)
    result = rt.tick()
    assert result.status is SkillStatus.RUNNING
    (action,) = result.actions
    assert isinstance(action, Move)
    assert len(dev.swipes) == 1
    x1, y1, x2, y2, ms = dev.swipes[0]
    assert (x1, y1) == (SCREEN.joy_x, SCREEN.joy_y)
    assert x2 == SCREEN.joy_x + SCREEN.joy_r
    assert abs(y2 - SCREEN.joy_y) <= 1
    assert 250 <= ms <= 1000
    assert dev.taps == []


def test_tick_revives_when_dead() -> None:
    """On the death screen the runtime runs the forced revive skill to completion."""
    dev = _dead_device()
    rt = _runtime(dev)
    rt.navigator.memory.walls[(1, 1)] = 9999.0
    rt.perceiver.tracker.prev = np.zeros((SCREEN.mm_size, SCREEN.mm_size))
    rt.perceiver.tracker.pos[:] = (7.0, -3.0)
    result = rt.tick()
    assert dev.taps == [SCREEN.revive]
    assert dev.swipes == []
    assert rt.navigator.memory.walls == {}
    assert rt.perceiver.tracker.prev is None
    assert (rt.perceiver.tracker.pos == 0).all()
    assert result.status is SkillStatus.DONE


def test_tick_explores_without_nodes() -> None:
    """With no nodes in view the runtime still pushes the joystick (exploring)."""
    dev = FakeDevice(blank_screen())
    rt = _runtime(dev)
    result = rt.tick()
    assert len(dev.swipes) == 1
    assert dev.swipes[0][4] == 1500
    assert result.status is SkillStatus.RUNNING


def test_stuck_learns_wall_ahead() -> None:
    """A hero that never moves gets a learned wall after one push, then stuck replans."""
    img = blank_screen()
    paint(img, 50, -2, 4, 4, HANDLE_RGB)
    dev = FakeDevice(img)
    rt = _runtime(dev)
    rt.tick()
    assert not rt.memory.walls
    rt.tick()
    assert rt.memory.walls  # the first push went nowhere: wall learned straight away
    for _ in range(6):
        rt.tick()
    # Back-to-back blocks fall through to net-movement stuck handling (a counted detour)
    assert sum(rt.memory.detours.values()) >= 1


def test_history_recorded_even_without_path() -> None:
    """Position history grows on every live step, including explore steps with no path."""
    dev = FakeDevice(blank_screen())
    rt = _runtime(dev)
    rt.tick()
    rt.tick()
    assert len(rt.navigator.hist) == 2


def test_repeated_detours_skip_node_area() -> None:
    """Past max_detours the node's area is skipped instead of stamping more walls."""
    img = blank_screen()
    paint(img, 50, -2, 4, 4, HANDLE_RGB)
    dev = FakeDevice(img)
    rt = _runtime(dev)
    nodes = [(dx, dy) for dx, dy, _ in find_nodes(img)]
    assert len(nodes) == 1
    tx, ty = nodes[0]
    path = bfs(np.ones((SCREEN.mm_size, SCREEN.mm_size), bool), nodes)
    assert path is not None
    cell = STUCK.area_cell
    key = (int(tx // cell), int(ty // cell))
    rt.memory.detours[key] = STUCK.max_detours
    rt.navigator._handle_stuck(nodes, path, 0.0, 0.0, 1000.0)
    assert rt.memory.skips == [(tx, ty, 1000.0 + STUCK.skip_seconds)]
    assert rt.memory.walls == {}
    # The skipped node is filtered out, so the next step explores
    result = rt.tick()
    assert len(dev.swipes) == 1
    assert dev.swipes[0][4] == 1500
    assert result.status is SkillStatus.RUNNING


def test_tick_timeout_marks_failed() -> None:
    """A skill that exceeds its timeout is aborted and reported as FAILED."""
    dev = FakeDevice(blank_screen())
    brain = FixedBrain()
    rt = Runtime(
        dev,
        brain=brain,
        config={"gather": {"timeout_s": 10}},
        clock=lambda: 0.0,
        sleep=lambda _s: None,
    )
    first = rt.tick()
    assert first.status is SkillStatus.RUNNING
    rt.clock = lambda: 11.0
    second = rt.tick()
    assert second.status is SkillStatus.FAILED
    assert second.reason == "timeout"
    assert brain.last_outcome is SkillStatus.FAILED
    assert rt._current is None
    assert rt._consecutive_failures == 1


def test_tick_exception_marks_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside a skill step aborts it and marks the activation FAILED."""

    def _boom(*args: object, **kwargs: object) -> Move:
        """Always raise, simulating a skill crash."""
        raise RuntimeError("boom")

    dev = FakeDevice(blank_screen())
    brain = FixedBrain()
    rt = Runtime(dev, brain=brain, clock=lambda: 1000.0, sleep=lambda _s: None)
    monkeypatch.setattr(rt.navigator, "step_toward", _boom)
    first = rt.tick()
    assert first.status is SkillStatus.FAILED
    assert first.reason == "error"
    assert brain.last_outcome is SkillStatus.FAILED
    assert rt._consecutive_failures == 1
    assert rt._current is None

    second = rt.tick()
    assert second.status is SkillStatus.FAILED
    assert rt._consecutive_failures == 2
    third = rt.tick()
    assert third.status is SkillStatus.FAILED
    assert rt._consecutive_failures == 3
    assert "gather" in rt._disabled

    fourth = rt.tick()
    assert fourth.status is SkillStatus.RUNNING
    assert fourth.actions == ()
