"""Tests for the skill framework: GatherSkill, ReviveSkill and the registry."""

from __future__ import annotations

import numpy as np
import pytest

from eternal_hero_bot.action.navigation import Navigator
from eternal_hero_bot.core.actions import Move, TapNamed, Wait
from eternal_hero_bot.core.config import SCREEN
from eternal_hero_bot.core.outcomes import SkillStatus, StepResult
from eternal_hero_bot.core.state import GameState, MinimapView, Objective, ScreenKind
from eternal_hero_bot.skills import GatherSkill, ReviveSkill, build_skills
from eternal_hero_bot.skills.base import Skill, SkillContext
from eternal_hero_bot.skills.revive import REVIVE_WAIT_S


def _state(
    screen: ScreenKind, minimap: MinimapView | None = None, objectives: tuple[Objective, ...] = ()
) -> GameState:
    """Build a minimal game state for skill tests.

    Args:
        screen: Screen kind to build the state with.
        minimap: Optional minimap view (defaults to None).
        objectives: Visible objectives for the state.

    Returns:
        The game state.
    """
    return GameState(
        t=0.0,
        frame_index=0,
        screen=screen,
        map_epoch=0,
        minimap=minimap,
        objectives=objectives,
        resources={},
        detections={},
    )


def _world_state(objectives: tuple[Objective, ...] = ()) -> GameState:
    """Build a world game state with a full walkable minimap.

    Args:
        objectives: Visible objectives.

    Returns:
        The world game state.
    """
    return _state(
        ScreenKind.WORLD,
        minimap=MinimapView(
            walk=np.ones((SCREEN.mm_size, SCREEN.mm_size), bool),
            hero_pos=(0.0, 0.0),
        ),
        objectives=objectives,
    )


def _ctx() -> SkillContext:
    """Build a skill context backed by a fresh navigator and frozen clock.

    Returns:
        The context handed to skills by the runtime.
    """
    return SkillContext(navigator=Navigator(), clock=lambda: 0.0, options={})


def test_gather_can_run_only_on_world() -> None:
    """Gather runs on world screens and refuses dead or unknown screens."""
    skill = GatherSkill()
    assert skill.can_run(_state(ScreenKind.WORLD))
    assert not skill.can_run(_state(ScreenKind.DEAD))
    assert not skill.can_run(_state(ScreenKind.UNKNOWN))


def test_gather_never_done() -> None:
    """Gather runs until it is preempted, fails, or times out."""
    assert GatherSkill().is_done(_world_state()) is False


def test_gather_step_pushes_toward_node() -> None:
    """Gather turns a node objective into a joystick move."""
    state = _world_state((Objective(kind="node", dx=50.0, dy=0.0, size=16),))
    result = GatherSkill().step(state, _ctx())
    assert result.status is SkillStatus.RUNNING
    assert isinstance(result, StepResult)
    (action,) = result.actions
    assert isinstance(action, Move)
    assert action.dx > 0
    assert 250 <= action.ms <= 1000


def test_gather_step_explores_without_nodes() -> None:
    """Gather explores with the explore move when no node is visible."""
    result = GatherSkill().step(_world_state(), _ctx())
    assert result.status is SkillStatus.RUNNING
    (action,) = result.actions
    assert isinstance(action, Move)
    assert action.ms == 1500


def test_gather_step_without_minimap_does_nothing() -> None:
    """Gather does nothing when the state carries no minimap."""
    result = GatherSkill().step(_state(ScreenKind.WORLD), _ctx())
    assert result.status is SkillStatus.RUNNING
    assert result.actions == ()


def test_gather_options_timeout_and_max_steps() -> None:
    """Gather tunes its timeout and step budget from its options table."""
    assert GatherSkill().timeout_s is None
    assert GatherSkill({}).max_steps is None
    assert GatherSkill({"timeout_s": 5}).timeout_s == 5.0
    assert GatherSkill({"timeout_s": 0}).timeout_s is None  # 0 means unlimited
    assert GatherSkill({"max_steps": 3}).max_steps == 3


def test_gather_rejects_unknown_options() -> None:
    """Gather rejects option keys it does not recognise."""
    with pytest.raises(ValueError, match="unknown options"):
        GatherSkill({"treasure": {}})


def test_revive_runs_on_dead_and_finishes() -> None:
    """Revive is only for dead screens, runs the revive tap and ends DONE."""
    dead = _state(ScreenKind.DEAD)
    live = _state(ScreenKind.WORLD)
    skill = ReviveSkill()
    assert skill.can_run(dead)
    assert not skill.can_run(live)
    assert skill.is_done(dead) is False
    assert skill.is_done(live) is True
    result = skill.step(dead, _ctx())
    assert result.status is SkillStatus.DONE
    assert len(result.actions) == 2
    first, second = result.actions
    assert isinstance(first, TapNamed)
    assert isinstance(second, Wait)
    assert first.name == "revive_button"
    assert second.seconds == REVIVE_WAIT_S


def test_revive_rejects_options() -> None:
    """Revive takes no options yet and rejects any it is given."""
    with pytest.raises(ValueError, match="unknown options"):
        ReviveSkill({"timeout_s": 3})


def test_build_skills_registers_gather_and_revive() -> None:
    """The registry builds all registered skills, including stubs, from defaults."""
    skills = build_skills({})
    assert {
        "gather",
        "revive",
        "find_treasure",
        "hunt",
        "complete_altar",
        "daily_tasks",
        "spend_resource",
    } <= set(skills)
    assert isinstance(skills["gather"], GatherSkill)
    assert isinstance(skills["revive"], ReviveSkill)


def test_build_skills_rejects_unknown_skill() -> None:
    """The registry refuses configuration for skills it does not know."""
    with pytest.raises(ValueError, match="unknown skill"):
        build_skills({"treasure": {}})


def test_gather_satisfies_skill_protocol() -> None:
    """Gather structurally satisfies the skill protocol."""
    skill: Skill = GatherSkill()
    assert skill.name == "gather"
