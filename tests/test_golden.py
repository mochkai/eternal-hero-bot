"""Golden device-call traces: current `Bot` must reproduce recorded runs on `FakeWorld`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.sim import GOLDEN_TICKS, SCENARIOS, TraceEvent, run_scenario, scenario

GOLDEN_DIR = Path(__file__).parent / "golden"
"""Directory of `gather_*.json` golden trace fixtures."""


def _golden_path(name: str) -> Path:
    """Path to a scenario's golden fixture.

    Args:
        name: Scenario name.

    Returns:
        `tests/golden/gather_<name>.json`.
    """
    return GOLDEN_DIR / f"gather_{name}.json"


@pytest.mark.parametrize("name", SCENARIOS)
def test_golden_trace(name: str) -> None:
    """The current `Bot` reproduces the recorded device-call trace exactly.

    Args:
        name: Golden scenario name.
    """
    payload = json.loads(_golden_path(name).read_text())
    assert payload["scenario"] == name
    assert payload["ticks"] == GOLDEN_TICKS
    world = scenario(name, seed=payload["seed"])
    events: list[TraceEvent] = run_scenario(world, GOLDEN_TICKS)
    assert events == payload["events"]
