"""Simulated wall-around and course-correction regressions (issue #18).

Each scenario runs the full runtime on a `FakeWorld` and must reach the node within a
tick budget. Budgets carry headroom over the current results; ``main`` before issue #18
needed 24 / 27 / 29 ticks for the visible-wall cases and never reached the node behind
the long invisible wall, so every budget below fails on the old navigator.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.sim import FakeWorld, ticks_to_reach

CASES: dict[str, tuple[Callable[[], FakeWorld], int]] = {
    "wall_detour": (FakeWorld.wall_detour, 18),  # now 11
    "u_pocket": (FakeWorld.u_pocket, 18),  # now 10
    "wide_wall": (FakeWorld.wide_wall, 22),  # now 13
    "invisible_wall": (lambda: FakeWorld.invisible_wall(0), 55),  # now 40
}
"""Scenario name -> (world factory, tick budget)."""


@pytest.mark.parametrize("name", list(CASES))
def test_reaches_node_around_walls(name: str) -> None:
    """The hero gets round the walls and reaches the node within the budget.

    Args:
        name: Scenario name in `CASES`.
    """
    make, budget = CASES[name]
    ticks = ticks_to_reach(make(), budget)
    assert ticks is not None, f"{name}: node not reached in {budget} ticks"
