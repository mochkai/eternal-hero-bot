"""Tests for the runtime main loop: gather parity through the skills and lifecycle."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from eternal_hero_bot.action.navigation import bfs
from eternal_hero_bot.core.actions import Move
from eternal_hero_bot.core.config import SCREEN, STUCK
from eternal_hero_bot.core.outcomes import SkillStatus
from eternal_hero_bot.perception.vision import find_nodes
from eternal_hero_bot.runtime import PHASES, FixedBrain, Runtime, StepTiming
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


_LOGGER = "eternal_hero_bot.runtime"


def _step_records(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    """Collect the structured per-step timing payloads from captured DEBUG records.

    Args:
        caplog: Pytest log capture fixture.

    Returns:
        The ``step_timing`` dicts, in emission order.
    """
    return [
        r.__dict__["step_timing"]
        for r in caplog.records
        if r.levelno == logging.DEBUG and "step_timing" in r.__dict__
    ]


def _assert_timing_fields(payload: dict[str, object]) -> None:
    """Assert every phase and the total are present as non-negative floats.

    Args:
        payload: A ``step_timing`` dict from a log record.
    """
    phases = payload["phases_ms"]
    assert isinstance(phases, dict)
    assert set(phases) == {*PHASES, "total"}
    for value in phases.values():
        assert isinstance(value, float)
        assert value >= 0.0


def test_tick_logs_step_timing(caplog: pytest.LogCaptureFixture) -> None:
    """Every tick emits a DEBUG timing record with frame, skill, status and all phases."""
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    rt = _runtime(FakeDevice(blank_screen()))
    rt.tick()
    rt.tick()
    payloads = _step_records(caplog)
    assert [p["frame"] for p in payloads] == [0, 1]
    for payload in payloads:
        assert payload["skill"] == "gather"
        assert payload["status"] == "running"
        _assert_timing_fields(payload)
    messages = [r.getMessage() for r in caplog.records if "step_timing" in r.__dict__]
    for name in (*PHASES, "total"):
        assert f"{name}=" in messages[0]
    assert "frame=0" in messages[0]
    assert "skill=gather" in messages[0]
    assert isinstance(rt.last_timing, StepTiming)
    assert rt.last_timing.frame == 1
    assert rt.last_timing.total_ms >= max(rt.last_timing.phases_ms.values())


def test_tick_logs_summary_every_n(caplog: pytest.LogCaptureFixture) -> None:
    """An INFO summary with per-phase mean/max appears every ``timing_log_every`` ticks."""
    caplog.set_level(logging.INFO, logger=_LOGGER)
    rt = Runtime(
        FakeDevice(blank_screen()),
        brain=FixedBrain(),
        clock=lambda: 1000.0,
        sleep=lambda _s: None,
        timing_log_every=2,
    )

    def summaries() -> list[dict[str, object]]:
        """Return the structured summary payloads captured so far."""
        return [
            r.__dict__["step_timing_summary"]
            for r in caplog.records
            if r.levelno == logging.INFO and "step_timing_summary" in r.__dict__
        ]

    rt.tick()
    assert summaries() == []
    rt.tick()
    (summary,) = summaries()
    assert summary["steps"] == 2
    for name in (*PHASES, "total"):
        stats = summary[name]
        assert isinstance(stats, dict)
        assert isinstance(stats["mean"], float)
        assert 0.0 <= stats["mean"] <= stats["max"]
    rt.tick()
    assert len(summaries()) == 1  # window was reset
    rt.tick()
    assert len(summaries()) == 2


def test_early_return_tick_still_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Done and disabled-skill early returns still log timing, unreached phases as 0."""
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    rt = _runtime(_dead_device())
    result = rt.tick()
    assert result.status is SkillStatus.DONE
    (payload,) = _step_records(caplog)
    assert payload["skill"] == "revive"
    assert payload["status"] == "done"
    _assert_timing_fields(payload)

    caplog.clear()
    rt._disabled.add("revive")
    result = rt.tick()
    assert result.actions == ()
    (payload,) = _step_records(caplog)
    assert payload["frame"] == 1
    assert payload["skill"] == "revive"
    _assert_timing_fields(payload)
    phases = payload["phases_ms"]
    assert isinstance(phases, dict)
    assert phases["act"] == 0.0
    assert rt.last_timing is not None
    assert rt.last_timing.phases_ms["skill"] == 0.0


def test_timing_log_every_must_be_positive() -> None:
    """A non-positive summary window is rejected."""
    with pytest.raises(ValueError, match="timing_log_every"):
        Runtime(FakeDevice(blank_screen()), brain=FixedBrain(), timing_log_every=0)
