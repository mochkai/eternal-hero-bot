"""Tests for phase-correlation position tracking."""

from __future__ import annotations

import numpy as np

from eternal_hero_bot.arrays import FloatArray
from eternal_hero_bot.config import SCREEN
from eternal_hero_bot.tracker import Tracker

N = SCREEN.mm_size


def _pattern() -> FloatArray:
    """A deterministic blocky mask with enough structure to correlate.

    Returns:
        A float ``N x N`` mask.
    """
    rng = np.random.default_rng(0)
    m = np.zeros((N, N))
    for _ in range(30):
        x, y = rng.integers(20, N - 20, 2)
        m[y - 5 : y + 5, x - 8 : x + 8] = 1
    return m


def test_first_frame_reports_no_motion() -> None:
    """The first update only stores the frame."""
    t = Tracker()
    assert t.update(_pattern()) == 0.0
    assert (t.pos == 0).all()


def test_translated_mask_moves_hero() -> None:
    """Map content moving up-left means the hero moved down-right."""
    m = _pattern()
    t = Tracker()
    t.update(m)
    moved = t.update(np.roll(m, (-3, -5), (0, 1)))
    assert moved == np.hypot(5, 3)
    assert t.pos.tolist() == [5.0, 3.0]


def test_large_jump_is_ignored() -> None:
    """Shifts beyond max_step_px are treated as glitches."""
    m = _pattern()
    t = Tracker(max_step_px=2)
    t.update(m)
    assert t.update(np.roll(m, -5, 1)) == 0.0
    assert (t.pos == 0).all()


def test_reset_forgets_previous_frame() -> None:
    """After reset the next frame is treated as a first frame."""
    m = _pattern()
    t = Tracker()
    t.update(m)
    t.reset()
    assert t.update(np.roll(m, -5, 1)) == 0.0
