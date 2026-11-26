"""Shared outcome vocabulary: skill lifecycle results and brain decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from eternal_hero_bot.core.actions import Action


class SkillStatus(StrEnum):
    """Lifecycle status of a skill activation.

    Attributes:
        RUNNING: The skill is still going and may emit actions.
        DONE: The skill finished (or was told to stop) without failing.
        FAILED: The skill failed, timed out or hit its step budget.
    """

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class StepResult:
    """Outcome of a single skill step.

    Attributes:
        status: Outcome of the step; ``RUNNING`` while the skill keeps going.
        actions: Actions to execute in order, empty when the step asks for nothing.
        reason: Machine-readable reason, set when the outcome is not ``RUNNING``.
    """

    status: SkillStatus
    actions: tuple[Action, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class Decision:
    """What the brain decided for the current tick.

    Attributes:
        skill: Name of the skill to run next.
        args: Arguments for the skill, as produced by the deciding rule.
        rule: Rule that made the decision, or ``"default"``.
        force: Whether the decision preempts a non-interruptible running skill.
    """

    skill: str
    args: Mapping[str, object] = field(default_factory=dict)
    rule: str = ""
    force: bool = False
