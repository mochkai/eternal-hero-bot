"""Tests for DefaultPerceiver (perception -> GameState)."""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.core.config import SCREEN
from eternal_hero_bot.core.state import ScreenKind
from eternal_hero_bot.perception.frame import Frame
from eternal_hero_bot.perception.perceiver import DefaultPerceiver
from tests.helpers import HANDLE_RGB, blank_screen, paint


def test_perceive_world_frame() -> None:
    """A live frame produces a WORLD state with minimap and node objectives."""
    img = blank_screen()
    paint(img, 50, -2, 4, 4, HANDLE_RGB)
    p = DefaultPerceiver()
    state = p.perceive(Frame(image=img, t=1.0, index=0))
    assert state.screen == ScreenKind.WORLD
    assert state.map_epoch == 0
    assert state.minimap is not None
    assert state.minimap.walk.shape == (SCREEN.mm_size, SCREEN.mm_size)
    assert not state.minimap.walk.flags.writeable
    assert len(state.objectives) == 1
    obj = state.objectives[0]
    assert obj.kind == "node"
    assert obj.size == 16
    assert abs(obj.dx - 52.0) < 1.0
    assert abs(obj.dy - 0.0) < 2.0


def test_perceive_dead_frame_resets_once() -> None:
    """First DEAD frame resets tracker/bumps epoch; second DEAD does not bump again."""
    img = blank_screen()
    x, y = SCREEN.death_probe
    img[y, x] = (150, 200, 80)
    p = DefaultPerceiver()
    p.tracker.prev = np.zeros((SCREEN.mm_size, SCREEN.mm_size))
    p.tracker.pos[:] = (7.0, -3.0)

    dead1 = p.perceive(Frame(image=img, t=1.0, index=0))
    assert dead1.screen == ScreenKind.DEAD
    assert dead1.minimap is None
    assert dead1.objectives == ()
    assert dead1.map_epoch == 1
    assert p.tracker.prev is None
    assert (p.tracker.pos == 0).all()

    dead2 = p.perceive(Frame(image=img, t=2.0, index=1))
    assert dead2.screen == ScreenKind.DEAD
    assert dead2.map_epoch == 1  # not bumped again


def test_perceive_dead_does_not_update_tracker() -> None:
    """A DEAD frame does not feed the tracker a new minimap."""
    p = DefaultPerceiver()
    live = blank_screen()
    p.perceive(Frame(image=live, t=1.0, index=0))
    assert p.tracker.prev is not None
    prev_before = p.tracker.prev.copy()

    dead = blank_screen()
    x, y = SCREEN.death_probe
    dead[y, x] = (150, 200, 80)
    p.perceive(Frame(image=dead, t=2.0, index=1))
    # after first dead, tracker was reset so prev is None
    assert p.tracker.prev is None
    _ = prev_before


def test_reset_bumps_epoch_and_clears_tracker() -> None:
    """reset() creates a fresh tracker and increments map_epoch."""
    p = DefaultPerceiver()
    p.perceive(Frame(image=blank_screen(), t=1.0, index=0))
    assert p.tracker.prev is not None
    p.reset()
    assert p.map_epoch == 1
    assert p.tracker.prev is None
    assert (p.tracker.pos == 0).all()


def test_world_after_dead_clears_dead_seen() -> None:
    """A WORLD frame after DEAD is perceived normally and clears dead-seen state."""
    p = DefaultPerceiver()
    dead = blank_screen()
    x, y = SCREEN.death_probe
    dead[y, x] = (150, 200, 80)
    p.perceive(Frame(image=dead, t=1.0, index=0))
    assert p.map_epoch == 1

    world = p.perceive(Frame(image=blank_screen(), t=2.0, index=1))
    assert world.screen == ScreenKind.WORLD
    assert world.map_epoch == 1
