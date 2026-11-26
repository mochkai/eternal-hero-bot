"""Tests for RuleBrain, conditions, config loading, and stub registration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eternal_hero_bot.brain.base import BrainContext
from eternal_hero_bot.brain.conditions import parse_condition
from eternal_hero_bot.brain.rules import Rule, RuleBrain, load_brain_config
from eternal_hero_bot.core.geometry import Box
from eternal_hero_bot.core.outcomes import Decision, SkillStatus
from eternal_hero_bot.core.state import (
    Detection,
    GameState,
    MinimapView,
    Objective,
    ResourceReading,
    ScreenKind,
)
from eternal_hero_bot.skills import build_skills
from eternal_hero_bot.skills.stubs import (
    CompleteAltarSkill,
    DailyTasksSkill,
    FindTreasureSkill,
    HuntSkill,
    SpendResourceSkill,
)


def _state(
    *,
    screen: ScreenKind = ScreenKind.WORLD,
    resources: dict[str, ResourceReading] | None = None,
    detections: dict[str, tuple[Detection, ...]] | None = None,
    objectives: tuple[Objective, ...] = (),
) -> GameState:
    """Build a minimal game state for brain tests."""
    minimap = None
    if screen == ScreenKind.WORLD:
        minimap = MinimapView(walk=np.ones((10, 10), bool), hero_pos=(0.0, 0.0))
    return GameState(
        t=0.0,
        frame_index=0,
        screen=screen,
        map_epoch=0,
        minimap=minimap,
        objectives=objectives,
        resources=resources or {},
        detections=detections or {},
    )


def _ctx(
    *,
    now: float = 0.0,
    current: str | None = None,
    current_rule: str | None = None,
    current_interruptible: bool = True,
    last_run: dict[str, float] | None = None,
    can_run: dict[str, bool] | None = None,
) -> BrainContext:
    can = can_run or {}

    def _can(name: str, state: GameState) -> bool:
        del state
        return can.get(name, True)

    return BrainContext(
        now=now,
        current=current,
        current_rule=current_rule,
        current_interruptible=current_interruptible,
        can_run=_can,
        last_run=last_run or {},
    )


def _rule(name: str, skill: str, when: dict[str, object], **kw: object) -> Rule:
    return Rule(name=name, skill=skill, when=parse_condition(when), **kw)  # type: ignore[arg-type]


def test_rule_order_first_matching_wins() -> None:
    """Rules are evaluated in priority order; first passing rule wins."""
    brain = RuleBrain(
        rules=(
            _rule("high", "hunt", {"screen": "world"}),
            _rule("low", "gather", {"screen": "world"}),
        ),
        default="gather",
    )
    decision = brain.decide(_state(), _ctx())
    assert decision.skill == "hunt"
    assert decision.rule == "high"


def test_default_when_no_rule_matches() -> None:
    """Default skill is chosen when no rule passes."""
    brain = RuleBrain(
        rules=(_rule("dead-only", "revive", {"screen": "dead"}, force=True),),
        default="gather",
    )
    decision = brain.decide(_state(screen=ScreenKind.WORLD), _ctx())
    assert decision == Decision(skill="gather", args={}, rule="default", force=False)


def test_rule_skipped_when_can_run_false() -> None:
    """A rule is skipped when ctx.can_run is False for its skill."""
    brain = RuleBrain(
        rules=(
            _rule("stub", "daily_tasks", {"screen": "world"}),
            _rule("fallback", "gather", {"screen": "world"}),
        ),
        default="gather",
    )
    decision = brain.decide(_state(), _ctx(can_run={"daily_tasks": False, "gather": True}))
    assert decision.skill == "gather"
    assert decision.rule == "fallback"


def test_cooldown_blocks_rule_until_elapsed() -> None:
    """cooldown_s prevents a rule from matching until the cooldown expires."""
    brain = RuleBrain(
        rules=(_rule("spend", "spend_resource", {"resource_at_max": "gold"}, cooldown_s=300.0),),
        default="gather",
    )
    state = _state(resources={"gold": ResourceReading("gold", 500, 500, 1.0)})
    d1 = brain.decide(state, _ctx(now=100.0))
    assert d1.rule == "spend"
    brain.notify(d1, SkillStatus.DONE, now=100.0)

    d2 = brain.decide(state, _ctx(now=399.0))
    assert d2.rule == "default"
    d3 = brain.decide(state, _ctx(now=400.0))
    assert d3.rule == "spend"


def test_failure_cooldown_used_on_failed() -> None:
    """failure_cooldown_s applies on failure/timeout; defaults to cooldown_s."""
    state = _state(resources={"gold": ResourceReading("gold", 500, 500, 1.0)})
    brain = RuleBrain(
        rules=(
            _rule(
                "spend",
                "spend_resource",
                {"resource_at_max": "gold"},
                cooldown_s=300.0,
                failure_cooldown_s=900.0,
            ),
        ),
        default="gather",
    )
    d1 = brain.decide(state, _ctx(now=0.0))
    brain.notify(d1, SkillStatus.FAILED, now=0.0)
    assert brain.decide(state, _ctx(now=899.0)).rule == "default"
    assert brain.decide(state, _ctx(now=900.0)).rule == "spend"


def test_failure_cooldown_defaults_to_cooldown_s() -> None:
    """When failure_cooldown_s is omitted, cooldown_s applies on failure."""
    state = _state(resources={"gold": ResourceReading("gold", 500, 500, 1.0)})
    brain = RuleBrain(
        rules=(_rule("spend", "spend_resource", {"resource_at_max": "gold"}, cooldown_s=300.0),),
        default="gather",
    )
    d1 = brain.decide(state, _ctx(now=0.0))
    brain.notify(d1, SkillStatus.FAILED, now=0.0)
    assert brain.decide(state, _ctx(now=299.0)).rule == "default"
    assert brain.decide(state, _ctx(now=300.0)).rule == "spend"


def test_higher_priority_preempts_interruptible() -> None:
    """A higher-priority rule preempts an interruptible running skill."""
    brain = RuleBrain(
        rules=(
            _rule("revive", "revive", {"screen": "dead"}, force=True),
            _rule("gather", "gather", {"screen": "world"}),
        ),
        default="gather",
    )
    # gather running, then death happens
    d = brain.decide(
        _state(screen=ScreenKind.DEAD),
        _ctx(current="gather", current_rule="gather", current_interruptible=True),
    )
    assert d.skill == "revive"
    assert d.rule == "revive"
    assert d.force is True


def test_higher_priority_does_not_preempt_non_interruptible_without_force() -> None:
    """Non-interruptible skill continues when higher rule lacks force."""
    brain = RuleBrain(
        rules=(
            _rule("daily", "daily_tasks", {"screen": "world"}),
            _rule("gather", "gather", {"screen": "world"}),
        ),
        default="gather",
    )
    d = brain.decide(
        _state(screen=ScreenKind.WORLD),
        _ctx(current="gather", current_rule="gather", current_interruptible=False),
    )
    assert d.skill == "gather"
    assert d.rule == "gather"


def test_force_preempts_non_interruptible() -> None:
    """A force rule preempts even a non-interruptible running skill."""
    brain = RuleBrain(
        rules=(
            _rule("revive", "revive", {"screen": "dead"}, force=True),
            _rule("gather", "gather", {"screen": "world"}),
        ),
        default="gather",
    )
    d = brain.decide(
        _state(screen=ScreenKind.DEAD),
        _ctx(current="gather", current_rule="gather", current_interruptible=False),
    )
    assert d.skill == "revive"
    assert d.force is True


def test_min_run_s_prevents_lower_priority_thrash() -> None:
    """min_run_s keeps a lower rule from displacing the current rule too soon."""
    brain = RuleBrain(
        rules=(_rule("sticky", "gather", {"detected": "node"}, min_run_s=10.0),),
        default="gather",
    )
    visible = _state(detections={"node": (Detection("node", Box(0, 0, 1, 1), 0.9),)})
    hidden = _state(detections={})
    # activate at t=0
    d0 = brain.decide(visible, _ctx(now=0.0))
    assert d0.rule == "sticky"
    # at t=5 detection flickers away; min_run_s keeps sticky
    d1 = brain.decide(
        hidden, _ctx(now=5.0, current="gather", current_rule="sticky", current_interruptible=True)
    )
    assert d1.rule == "sticky"
    # at t=10 min_run_s satisfied, lower/default can take over
    d2 = brain.decide(
        hidden, _ctx(now=10.0, current="gather", current_rule="sticky", current_interruptible=True)
    )
    assert d2.rule == "default"


def test_same_activation_keeps_running() -> None:
    """When the same rule/skill is chosen, the decision keeps the current activation."""
    brain = RuleBrain(rules=(_rule("g", "gather", {"screen": "world"}),), default="gather")
    state = _state()
    d = brain.decide(state, _ctx(current="gather", current_rule="g", current_interruptible=True))
    assert d.skill == "gather"
    assert d.rule == "g"


def test_condition_unknown_key_raises() -> None:
    """Unknown condition keys are errors."""
    with pytest.raises(ValueError, match="unknown condition key"):
        parse_condition({"screen": "world", "bogus": True})
    with pytest.raises(ValueError, match="unknown condition key"):
        parse_condition({"typo": "world"})


def test_condition_unknown_nested_key_raises() -> None:
    """Unknown keys inside predicate tables are errors."""
    with pytest.raises(ValueError, match="unknown key"):
        parse_condition({"resource_gte": {"name": "gold", "value": 1, "extra": 2}})
    with pytest.raises(ValueError, match="unknown key"):
        parse_condition({"since_last_run": {"skill": "x", "s": 1, "nope": 1}})


def test_config_unknown_keys_raise(tmp_path: Path) -> None:
    """Unknown keys in config/rule tables are errors."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text('[brain]\ndefault = "gather"\nbogus = 1\n')
    with pytest.raises(ValueError, match="unknown key"):
        load_brain_config(cfg)

    cfg.write_text(
        '[brain]\ndefault = "gather"\n[[brain.rules]]\n'
        'name="r"\nskill="gather"\nwhen={screen="world"}\nextra=1\n'
    )
    with pytest.raises(ValueError, match="unknown key"):
        load_brain_config(cfg)

    cfg.write_text('[brain]\ndefault="gather"\n[unknown_section]\n')
    with pytest.raises(ValueError, match="unknown key"):
        load_brain_config(cfg)


def test_stubs_registered_and_cannot_run() -> None:
    """Stub skills are registered and report can_run False."""
    skills = build_skills({})
    for name, cls in [
        ("find_treasure", FindTreasureSkill),
        ("hunt", HuntSkill),
        ("complete_altar", CompleteAltarSkill),
        ("daily_tasks", DailyTasksSkill),
        ("spend_resource", SpendResourceSkill),
    ]:
        assert isinstance(skills[name], cls)
        assert skills[name].can_run(_state()) is False


def test_default_config_loads() -> None:
    """The packaged default config loads and configures the RuleBrain."""
    from eternal_hero_bot.cli import DEFAULT_CONFIG

    cfg = load_brain_config(DEFAULT_CONFIG)
    assert cfg.brain.default == "gather"
    assert cfg.max_consecutive_failures == 3
    names = [r.name for r in cfg.brain.rules]
    assert "revive-on-death" in names
    assert names[0] == "revive-on-death"
    # skills options present
    assert "gather" in cfg.skill_options
    assert "spend_resource" in cfg.skill_options


def test_default_config_golden_decision() -> None:
    """Default config picks revive on death and gather on world when stubs can't run."""
    from eternal_hero_bot.cli import DEFAULT_CONFIG

    cfg = load_brain_config(DEFAULT_CONFIG)
    skills = build_skills(cfg.skill_options)

    def can_run(name: str, state: GameState) -> bool:
        skill = skills.get(name)
        return skill is not None and skill.can_run(state)

    ctx = BrainContext(
        now=0.0,
        current=None,
        current_rule=None,
        current_interruptible=True,
        can_run=can_run,
        last_run={},
    )
    d_dead = cfg.brain.decide(_state(screen=ScreenKind.DEAD), ctx)
    assert d_dead.skill == "revive"
    assert d_dead.force is True
    d_world = cfg.brain.decide(_state(screen=ScreenKind.WORLD), ctx)
    assert d_world.skill == "gather"


def test_cli_accepts_config(tmp_path: Path) -> None:
    """The run subcommand accepts --config."""
    from eternal_hero_bot.cli import build_parser

    cfg = tmp_path / "c.toml"
    cfg.write_text('[brain]\ndefault="gather"\n')
    args = build_parser().parse_args(["run", "--config", str(cfg)])
    assert args.config == cfg


def test_cli_default_config_path() -> None:
    """The run subcommand defaults --config to the packaged default_config.toml."""
    from eternal_hero_bot.cli import DEFAULT_CONFIG, build_parser

    args = build_parser().parse_args(["run"])
    assert args.config == DEFAULT_CONFIG
    assert DEFAULT_CONFIG.is_file()


def test_condition_reads_unknown_value_as_false() -> None:
    """Conditions reading an unknown/missing value evaluate to False."""
    ctx = BrainContext(
        now=0.0,
        current=None,
        current_rule=None,
        current_interruptible=True,
        can_run=lambda n, s: True,
        last_run={},
    )
    state = _state(resources={})  # no gold reading
    assert parse_condition({"resource_at_max": "gold"}).evaluate(state, ctx) is False
    assert (
        parse_condition({"resource_gte": {"name": "gold", "value": 1}}).evaluate(state, ctx)
        is False
    )
    assert (
        parse_condition({"resource_lt": {"name": "gold", "value": 1}}).evaluate(state, ctx) is False
    )
    assert (
        parse_condition({"resource_frac_gte": {"name": "gold", "frac": 0.5}}).evaluate(state, ctx)
        is False
    )

    state_none = _state(resources={"gold": ResourceReading("gold", None, 500, 0.0)})
    assert (
        parse_condition({"resource_gte": {"name": "gold", "value": 1}}).evaluate(state_none, ctx)
        is False
    )


def test_condition_predicates() -> None:
    """Core condition predicates evaluate as specified."""
    ctx = BrainContext(
        now=100.0,
        current=None,
        current_rule=None,
        current_interruptible=True,
        can_run=lambda n, s: True,
        last_run={"daily": 50.0},
    )
    state = _state(
        screen=ScreenKind.WORLD,
        resources={
            "gold": ResourceReading("gold", 500, 500, 1.0),
            "labor": ResourceReading("labor", 5, 100, 1.0),
        },
        detections={"chest": (Detection("chest", Box(0, 0, 1, 1), 0.9),)},
        objectives=(Objective("node", 1, 2, 3),),
    )
    assert parse_condition({"screen": "world"}).evaluate(state, ctx) is True
    assert parse_condition({"screen": "dead"}).evaluate(state, ctx) is False
    assert parse_condition({"resource_at_max": "gold"}).evaluate(state, ctx) is True
    assert (
        parse_condition({"resource_gte": {"name": "labor", "value": 5}}).evaluate(state, ctx)
        is True
    )
    assert (
        parse_condition({"resource_lt": {"name": "labor", "value": 10}}).evaluate(state, ctx)
        is True
    )
    assert (
        parse_condition({"resource_frac_gte": {"name": "gold", "frac": 1.0}}).evaluate(state, ctx)
        is True
    )
    assert parse_condition({"detected": "chest"}).evaluate(state, ctx) is True
    assert parse_condition({"detected": "missing"}).evaluate(state, ctx) is False
    assert parse_condition({"objective": "node"}).evaluate(state, ctx) is True
    assert parse_condition({"objective": "chest"}).evaluate(state, ctx) is False
    assert (
        parse_condition({"since_last_run": {"skill": "daily", "s": 60}}).evaluate(state, ctx)
        is False
    )
    assert (
        parse_condition({"since_last_run": {"skill": "daily", "s": 40}}).evaluate(state, ctx)
        is True
    )
    assert (
        parse_condition({"since_last_run": {"skill": "never", "s": 999}}).evaluate(state, ctx)
        is True
    )
    assert (
        parse_condition({"all": [{"screen": "world"}, {"resource_at_max": "gold"}]}).evaluate(
            state, ctx
        )
        is True
    )
    assert (
        parse_condition({"any": [{"screen": "dead"}, {"resource_at_max": "gold"}]}).evaluate(
            state, ctx
        )
        is True
    )
    assert parse_condition({"not": {"screen": "dead"}}).evaluate(state, ctx) is True
