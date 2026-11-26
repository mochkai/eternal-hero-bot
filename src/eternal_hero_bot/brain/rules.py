"""Rule-based brain: priority-ordered rules with cooldowns and preemption."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eternal_hero_bot.brain.base import BrainContext
from eternal_hero_bot.brain.conditions import Condition, parse_condition
from eternal_hero_bot.core.outcomes import Decision, SkillStatus
from eternal_hero_bot.core.state import GameState

__all__ = ["Rule", "RuleBrain", "load_brain_config"]


@dataclass(frozen=True)
class Rule:
    """A single priority rule.

    Attributes:
        name: Rule name for logging and cooldown bookkeeping.
        skill: Skill to run when the rule matches.
        when: Condition that must hold.
        args: Skill arguments passed to ``skill.start``.
        cooldown_s: Cooldown after a successful finish.
        failure_cooldown_s: Cooldown after failure/timeout; defaults to ``cooldown_s``.
        force: Whether this rule preempts a non-interruptible running skill.
        min_run_s: Minimum run time before a lower-priority rule may displace it.
    """

    name: str
    skill: str
    when: Condition
    args: Mapping[str, object] = field(default_factory=dict)
    cooldown_s: float = 0.0
    failure_cooldown_s: float | None = None
    force: bool = False
    min_run_s: float = 0.0


class RuleBrain:
    """Priority-ordered rule brain with cooldowns and preemption.

    Rules are evaluated in list order; the first rule whose condition holds, that
    is not on cooldown, and whose skill can run on the current state wins. If no
    rule passes, the default skill is chosen.

    Attributes:
        last_run: Skill name to last finish time in seconds.
    """

    def __init__(self, *, rules: Sequence[Rule], default: str) -> None:
        """Create a rule brain.

        Args:
            rules: Rules in priority order (first = highest priority).
            default: Skill name to use when no rule matches.
        """
        self._rules: tuple[Rule, ...] = tuple(rules)
        self._default = default
        self._rule_index: dict[str, int] = {rule.name: i for i, rule in enumerate(self._rules)}
        self._cooldown_until: dict[str, float] = {}
        self.last_run: dict[str, float] = {}
        self._activation_started_at: dict[str, float] = {}
        self._current_rule: str | None = None

    @property
    def rules(self) -> tuple[Rule, ...]:
        """The configured rules in priority order."""
        return self._rules

    @property
    def default(self) -> str:
        """The default skill name."""
        return self._default

    def decide(self, state: GameState, ctx: BrainContext) -> Decision:
        """Decide the next skill for the current game state.

        Args:
            state: Current immutable game state.
            ctx: Context for the current tick.

        Returns:
            The decision to run.
        """
        self._sync_current(ctx)
        candidate = self._first_passing(state, ctx)
        if candidate is None:
            candidate = self._default_decision()

        if ctx.current is None:
            return self._activate(candidate, ctx)

        # Same activation: keep running.
        if candidate.rule == ctx.current_rule and candidate.skill == ctx.current:
            return candidate

        current_idx = self._rule_index.get(ctx.current_rule or "", len(self._rules))
        candidate_idx = self._rule_index.get(candidate.rule, len(self._rules))

        if candidate_idx < current_idx:
            # Higher-priority preemption (spec §5.3).
            if ctx.current_interruptible or candidate.force:
                return self._activate(candidate, ctx)
            return self._keep_current(ctx)

        # Candidate is lower-priority or default. min_run_s on the current rule
        # prevents thrashing to lower-priority rules (spec §5.3).
        current_rule = self._rule_by_name(ctx.current_rule)
        min_run_s = current_rule.min_run_s if current_rule is not None else 0.0
        started = self._activation_started_at.get(ctx.current_rule or "", ctx.now)
        if ctx.now - started < min_run_s:
            return self._keep_current(ctx)

        return self._activate(candidate, ctx)

    def notify(self, decision: Decision, outcome: SkillStatus, now: float) -> None:
        """Record a finished activation for cooldown and last-run bookkeeping.

        Args:
            decision: The decision that produced the outcome.
            outcome: Skill status the activation ended with.
            now: Current time in seconds.
        """
        self.last_run[decision.skill] = now
        rule = self._rule_by_name(decision.rule)
        if rule is None:
            return
        cooldown = rule.cooldown_s
        if outcome is SkillStatus.FAILED:
            cooldown = (
                rule.cooldown_s if rule.failure_cooldown_s is None else rule.failure_cooldown_s
            )
        if cooldown > 0:
            self._cooldown_until[rule.name] = now + cooldown
        if self._current_rule == rule.name:
            self._current_rule = None

    def _sync_current(self, ctx: BrainContext) -> None:
        """Sync internal current-rule tracking with the runtime's view.

        Args:
            ctx: Context for the current tick.
        """
        if ctx.current is None:
            self._current_rule = None
            return
        if ctx.current_rule != self._current_rule:
            self._current_rule = ctx.current_rule
            if ctx.current_rule is not None:
                self._activation_started_at.setdefault(ctx.current_rule, ctx.now)

    def _first_passing(self, state: GameState, ctx: BrainContext) -> Decision | None:
        """Return the highest-priority rule that passes all checks.

        Args:
            state: Current immutable game state.
            ctx: Context for the current tick.

        Returns:
            Decision for the first passing rule, or None.
        """
        for rule in self._rules:
            if self._rule_passes(rule, state, ctx):
                return Decision(skill=rule.skill, args=rule.args, rule=rule.name, force=rule.force)
        return None

    def _rule_passes(self, rule: Rule, state: GameState, ctx: BrainContext) -> bool:
        """Check whether a rule passes condition, cooldown and can_run.

        Args:
            rule: Rule to evaluate.
            state: Current immutable game state.
            ctx: Context for the current tick.

        Returns:
            True when the rule may be chosen.
        """
        if ctx.now < self._cooldown_until.get(rule.name, 0.0):
            return False
        if not rule.when.evaluate(state, ctx):
            return False
        return ctx.can_run(rule.skill, state)

    def _current_still_valid(self, state: GameState, ctx: BrainContext) -> bool:
        """Whether the current rule still passes its checks.

        Args:
            state: Current immutable game state.
            ctx: Context for the current tick.

        Returns:
            True when the current rule can still be chosen.
        """
        if ctx.current_rule is None:
            return False
        rule = self._rule_by_name(ctx.current_rule)
        if rule is None:
            # Default or unknown rule - valid as long as it is the running skill.
            return ctx.current == self._default
        return self._rule_passes(rule, state, ctx)

    def _rule_by_name(self, name: str | None) -> Rule | None:
        """Look up a rule by name.

        Args:
            name: Rule name.

        Returns:
            The rule, or None.
        """
        if name is None:
            return None
        idx = self._rule_index.get(name)
        if idx is None:
            return None
        return self._rules[idx]

    def _default_decision(self) -> Decision:
        """Build the default decision.

        Returns:
            Decision for the default skill.
        """
        return Decision(skill=self._default, args={}, rule="default", force=False)

    def _activate(self, decision: Decision, ctx: BrainContext) -> Decision:
        """Record activation start time for a decision and return it.

        Args:
            decision: Decision to activate.
            ctx: Context for the current tick.

        Returns:
            The same decision.
        """
        if decision.rule != ctx.current_rule or decision.skill != ctx.current:
            self._activation_started_at[decision.rule] = ctx.now
            self._current_rule = decision.rule
        return decision

    def _keep_current(self, ctx: BrainContext) -> Decision:
        """Return a decision that keeps the current skill running.

        Args:
            ctx: Context for the current tick.

        Returns:
            Decision for the currently running skill/rule.
        """
        rule_name = ctx.current_rule or "default"
        rule = self._rule_by_name(rule_name)
        force = rule.force if rule is not None else False
        args = rule.args if rule is not None else {}
        return Decision(skill=ctx.current or self._default, args=args, rule=rule_name, force=force)


@dataclass(frozen=True)
class BrainConfig:
    """Parsed brain configuration from TOML.

    Attributes:
        brain: The configured rule brain.
        max_consecutive_failures: Consecutive failures before disabling a skill.
        skill_options: Per-skill option tables from ``[skills.<name>]``.
    """

    brain: RuleBrain
    max_consecutive_failures: int
    skill_options: Mapping[str, Mapping[str, object]]


def load_brain_config(source: str | Path | Mapping[str, Any]) -> BrainConfig:
    """Load brain and skill configuration from a TOML file or mapping.

    Args:
        source: Path to a TOML file, or an already-parsed mapping.

    Returns:
        The parsed brain configuration.

    Raises:
        ValueError: If the configuration is malformed or contains unknown keys.
        OSError: If the TOML file cannot be read.
    """
    if isinstance(source, Mapping):
        raw: Mapping[str, Any] = source
    else:
        path = Path(source)
        with path.open("rb") as f:
            raw = tomllib.load(f)

    _reject_unknown_keys(raw, frozenset({"brain", "skills"}), "config")
    brain_table = _expect_table(raw.get("brain", {}), "brain")
    _reject_unknown_keys(
        brain_table,
        frozenset({"default", "max_consecutive_failures", "rules"}),
        "brain",
    )

    default = _expect_str_value(brain_table.get("default", "gather"), "brain.default")
    max_failures = _expect_int_value(
        brain_table.get("max_consecutive_failures", 3), "brain.max_consecutive_failures"
    )
    if max_failures < 1:
        raise ValueError("brain.max_consecutive_failures: must be >= 1")

    rules_raw = brain_table.get("rules", [])
    if not isinstance(rules_raw, list):
        raise ValueError("brain.rules: expected array of tables")
    rules = tuple(_parse_rule(item, i) for i, item in enumerate(rules_raw))

    skills_table = _expect_table(raw.get("skills", {}), "skills")
    skill_options: dict[str, Mapping[str, object]] = {}
    for name, value in skills_table.items():
        if not isinstance(name, str):
            raise ValueError("skills: expected string keys")
        skill_options[name] = _expect_table(value, f"skills.{name}")

    return BrainConfig(
        brain=RuleBrain(rules=rules, default=default),
        max_consecutive_failures=max_failures,
        skill_options=skill_options,
    )


_RULE_KEYS = frozenset(
    {"name", "skill", "when", "args", "cooldown_s", "failure_cooldown_s", "force", "min_run_s"}
)


def _parse_rule(raw: object, index: int) -> Rule:
    """Parse a single rule table.

    Args:
        raw: Raw rule table.
        index: Rule index for error messages.

    Returns:
        The parsed rule.

    Raises:
        ValueError: If the rule is malformed.
    """
    path = f"brain.rules[{index}]"
    table = _expect_table(raw, path)
    _reject_unknown_keys(table, _RULE_KEYS, path)
    for required in ("name", "skill", "when"):
        if required not in table:
            raise ValueError(f"{path}: missing required key {required!r}")

    name = _expect_str_value(table["name"], f"{path}.name")
    skill = _expect_str_value(table["skill"], f"{path}.skill")
    when = parse_condition(table["when"], path=f"{path}.when")
    args = _expect_table(table.get("args", {}), f"{path}.args")
    cooldown_s = _expect_float_value(table.get("cooldown_s", 0.0), f"{path}.cooldown_s")
    failure_raw = table.get("failure_cooldown_s")
    failure_cooldown_s = (
        None
        if failure_raw is None
        else _expect_float_value(failure_raw, f"{path}.failure_cooldown_s")
    )
    force = _expect_bool_value(table.get("force", False), f"{path}.force")
    min_run_s = _expect_float_value(table.get("min_run_s", 0.0), f"{path}.min_run_s")
    if cooldown_s < 0:
        raise ValueError(f"{path}.cooldown_s: must be >= 0")
    if failure_cooldown_s is not None and failure_cooldown_s < 0:
        raise ValueError(f"{path}.failure_cooldown_s: must be >= 0")
    if min_run_s < 0:
        raise ValueError(f"{path}.min_run_s: must be >= 0")

    return Rule(
        name=name,
        skill=skill,
        when=when,
        args=dict(args),
        cooldown_s=cooldown_s,
        failure_cooldown_s=failure_cooldown_s,
        force=force,
        min_run_s=min_run_s,
    )


def _reject_unknown_keys(table: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    """Reject unknown keys in a table.

    Args:
        table: Table to validate.
        allowed: Allowed key names.
        path: Dotted path for error messages.

    Raises:
        ValueError: If unknown keys are present.
    """
    unknown = set(table.keys()) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {sorted(unknown)}")


def _expect_table(value: object, path: str) -> Mapping[str, Any]:
    """Expect a mapping value.

    Args:
        value: Raw value.
        path: Dotted path for error messages.

    Returns:
        The mapping.

    Raises:
        ValueError: If the value is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected table, got {type(value).__name__}")
    bad = [k for k in value if not isinstance(k, str)]
    if bad:
        raise ValueError(f"{path}: expected string keys")
    return value


def _expect_str_value(value: object, path: str) -> str:
    """Expect a string value.

    Args:
        value: Raw value.
        path: Dotted path for error messages.

    Returns:
        The string.

    Raises:
        ValueError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected string, got {type(value).__name__}")
    return value


def _expect_int_value(value: object, path: str) -> int:
    """Expect an integer value (not bool).

    Args:
        value: Raw value.
        path: Dotted path for error messages.

    Returns:
        The integer.

    Raises:
        ValueError: If the value is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected integer, got {type(value).__name__}")
    return value


def _expect_float_value(value: object, path: str) -> float:
    """Expect a numeric value (int or float, not bool).

    Args:
        value: Raw value.
        path: Dotted path for error messages.

    Returns:
        The number as float.

    Raises:
        ValueError: If the value is not numeric.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected number, got {type(value).__name__}")
    return float(value)


def _expect_bool_value(value: object, path: str) -> bool:
    """Expect a boolean value.

    Args:
        value: Raw value.
        path: Dotted path for error messages.

    Returns:
        The boolean.

    Raises:
        ValueError: If the value is not a boolean.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{path}: expected boolean, got {type(value).__name__}")
    return value
