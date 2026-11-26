"""Stub skills for future behaviors.

Each stub is registered, has ``can_run -> False``, and documents its intent. Follow-up
issues will implement them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, cast

from eternal_hero_bot.core.outcomes import SkillStatus, StepResult
from eternal_hero_bot.core.state import GameState
from eternal_hero_bot.skills.base import SkillContext
from eternal_hero_bot.skills.registry import register

_STUB_OPTIONS: frozenset[str] = frozenset({"timeout_s", "max_steps"})


class _StubSkill:
    """Base for registered stubs: never runnable, never completes on its own."""

    name: ClassVar[str]
    timeout_s: float | None = None
    max_steps: int | None = None

    def __init__(self, options: Mapping[str, object] | None = None) -> None:
        """Create the stub, applying common timeout/step options.

        Args:
            options: Options table for the skill; recognised keys are ``timeout_s`` and
                ``max_steps``. ``timeout_s = 0`` means unlimited.

        Raises:
            ValueError: If the options table holds unrecognised keys.
        """
        table = dict(options) if options else {}
        unknown = sorted(set(table) - _STUB_OPTIONS)
        if unknown:
            raise ValueError(f"{self.name}: unknown options {unknown}")
        raw_timeout = table.get("timeout_s")
        seconds = float(cast(float, raw_timeout)) if raw_timeout is not None else 0.0
        self.timeout_s = seconds if seconds > 0 else None
        raw_max = table.get("max_steps")
        self.max_steps = int(cast(int, raw_max)) if raw_max is not None else None

    @property
    def interruptible(self) -> bool:
        """Stubs are interruptible by default."""
        return True

    def can_run(self, state: GameState) -> bool:
        """Stubs cannot run until implemented.

        Args:
            state: Current immutable game state.

        Returns:
            Always False.
        """
        return False

    def start(self, state: GameState, ctx: SkillContext, args: Mapping[str, object]) -> None:
        """Start a stub activation (no-op).

        Args:
            state: Current immutable game state.
            ctx: Services for the current activation.
            args: Arguments for the skill, as produced by the deciding rule.
        """

    def is_done(self, state: GameState) -> bool:
        """Stubs report done only if somehow started.

        Args:
            state: Current immutable game state.

        Returns:
            True, so an accidental activation ends immediately.
        """
        return True

    def step(self, state: GameState, ctx: SkillContext) -> StepResult:
        """Return a no-op done result.

        Args:
            state: Current immutable game state.
            ctx: Services for the current activation.

        Returns:
            A DONE result with no actions.
        """
        return StepResult(status=SkillStatus.DONE, actions=(), reason="stub")

    def abort(self, reason: str) -> None:
        """Abort the stub activation.

        Args:
            reason: Machine-readable abort reason.
        """


@register("find_treasure")
class FindTreasureSkill(_StubSkill):
    """Find and open treasure chests.

    Intent: detect chest objectives (minimap marker + world template), navigate with
    ``Navigator``, tap open, confirm the reward popup. Follow-up issue: implement.
    """

    name: ClassVar[str] = "find_treasure"


@register("hunt")
class HuntSkill(_StubSkill):
    """Hunt monsters for drops/XP.

    Intent: detect monster objectives, approach to attack range, rely on auto-attack or
    tap attack, retreat when hero HP is low (HP bar region). Follow-up issue: implement.
    """

    name: ClassVar[str] = "hunt"


@register("complete_altar")
class CompleteAltarSkill(_StubSkill):
    """Complete altar objectives.

    Intent: find the altar objective, navigate, open its dialog, perform its requirement,
    claim. Follow-up issue: implement.
    """

    name: ClassVar[str] = "complete_altar"


@register("daily_tasks")
class DailyTasksSkill(_StubSkill):
    """Complete and claim daily tasks.

    Intent: open the daily-task menu, OCR task progress, claim completed rewards, pick
    tasks other skills can satisfy (feeds brain conditions). Follow-up issue: implement.
    """

    name: ClassVar[str] = "daily_tasks"


@register("spend_resource")
class SpendResourceSkill(_StubSkill):
    """Spend a resource that is at or near its cap.

    Intent: run the per-resource spend flow from spec §13, DONE when the resource drops
    below max. The rule passes ``args={"resource": "<name>"}``. Follow-up issue: implement.
    """

    name: ClassVar[str] = "spend_resource"
