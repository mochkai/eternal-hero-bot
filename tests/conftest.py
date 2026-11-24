"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from eternal_hero_bot.arrays import RGBImage
from tests.helpers import blank_screen


@pytest.fixture
def screen() -> RGBImage:
    """Fixture: a blank ground-filled synthetic screen.

    Returns:
        The synthetic screen.
    """
    return blank_screen()
