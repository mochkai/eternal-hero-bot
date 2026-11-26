"""The runtime: the main loop wiring perception, brain, skills and action."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

from eternal_hero_bot.action.controller import Controller
from eternal_hero_bot.action.navigation import Navigator, WorldMemory
from eternal_hero_bot.brain.base import Brain, BrainContext
from eternal_hero_bot.core.arrays import FloatArray
from eternal_hero_bot.core.config import (
    PATHING,
    SCREEN,
    STUCK,
    PathingConfig,
    ScreenGeometry,
    StuckConfig,
)
from eternal_hero_bot.core.outcomes import Decision, SkillStatus, StepResult
from eternal_hero_bot.core.state import GameState, ScreenKind
from eternal_hero_bot.device.base import Device
from eternal_hero_bot.perception.frame import Frame
from eternal_hero_bot.perception.perceiver import DefaultPerceiver
from eternal_hero_bot.perception.tracker import Tracker
from eternal_hero_bot.skills.base import Skill, SkillContext
from eternal_hero_bot.skills.registry import build_skills

logger = logging.getLogger(__name__)


class FixedBrain:
    """Minimal fixed brain kept for backward compatibility.

    Death is handled by a forced revive decision; every other frame gathers.
    Prefer :class:`~eternal_hero_bot.brain.rules.RuleBrain` for configurable behavior.
    """

    last_decision: Decision | None = None
    last_outcome: SkillStatus | None = None
    last_notified_at: float = 0.0

    def decide(self, state: GameState, ctx: BrainContext) -> Decision:
        """Decide the next skill for the current game state.

        Args:
            state: Current immutable game state.
            ctx: Context for the current tick.

        Returns:
            The revive decision on a dead screen, else the gather decision.
        """
        del ctx
        if state.screen == ScreenKind.DEAD:
            return Decision(skill="revive", args={}, rule="revive-on-death", force=True)
        return Decision(skill="gather", args={}, rule="default")

    def notify(self, decision: Decision, outcome: SkillStatus, now: float) -> None:
        """Record the finished activation.

        Args:
            decision: The decision that produced the outcome.
            outcome: Skill status the activation ended with.
            now: Current time in seconds.
        """
        self.last_decision = decision
        self.last_outcome = outcome
        self.last_notified_at = now


#: Per-tick phases, in execution order; every :class:`StepTiming` reports all of them.
PHASES: tuple[str, ...] = ("capture", "perceive", "decide", "skill", "act")


@dataclass(frozen=True)
class StepTiming:
    """Wall-clock processing time of one runtime tick, measured with ``perf_counter``.

    Phases not reached on a tick (early returns) are reported as ``0.0``.

    Attributes:
        frame: Index of the frame captured on this tick.
        skill: Running or decided skill name, if any.
        status: Step status value (``running``/``done``/``failed``), or ``error`` when
            the tick raised.
        phases_ms: Duration of each phase in :data:`PHASES`, in milliseconds.
        total_ms: Duration of the whole tick, in milliseconds.
    """

    frame: int
    skill: str | None
    status: str
    phases_ms: Mapping[str, float]
    total_ms: float

    def as_dict(self) -> dict[str, object]:
        """Flatten the timing into a dict suitable for structured logging.

        Returns:
            Frame, skill name, status, and ``phases_ms`` mapping each phase plus
            ``total`` to milliseconds (nested, since ``skill`` is both a field and a phase).
        """
        return {
            "frame": self.frame,
            "skill": self.skill,
            "status": self.status,
            "phases_ms": {**self.phases_ms, "total": self.total_ms},
        }


@dataclass
class _PhaseTimer:
    """Mutable per-tick recorder of phase durations (seconds, from ``perf_counter``).

    Attributes:
        frame: Index of the frame captured on this tick.
        skill: Running or decided skill name, once known.
        durations: Seconds spent in each completed phase.
    """

    frame: int
    skill: str | None = None
    durations: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the enclosed block as phase ``name`` (recorded even on early return/raise).

        Args:
            name: Phase name, one of :data:`PHASES`.

        Yields:
            None; the block's duration is added to :attr:`durations` on exit.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = self.durations.get(name, 0.0) + time.perf_counter() - start

    def finish(self, status: str, total_s: float) -> StepTiming:
        """Freeze the recorded durations into a :class:`StepTiming`.

        Args:
            status: Step status value, or ``error`` when the tick raised.
            total_s: Whole-tick duration in seconds.

        Returns:
            The immutable timing, with unreached phases reported as 0 ms.
        """
        return StepTiming(
            frame=self.frame,
            skill=self.skill,
            status=status,
            phases_ms={p: self.durations.get(p, 0.0) * 1000.0 for p in PHASES},
            total_ms=total_s * 1000.0,
        )


def summarize_timings(window: list[StepTiming]) -> dict[str, dict[str, float]]:
    """Compute per-phase mean and max (in ms) over a window of step timings.

    Args:
        window: Non-empty list of step timings.

    Returns:
        Mapping of each phase (plus ``total``) to ``{"mean": ms, "max": ms}``.

    Raises:
        ValueError: If ``window`` is empty.
    """
    if not window:
        raise ValueError("cannot summarize an empty timing window")
    summary: dict[str, dict[str, float]] = {}
    for name in (*PHASES, "total"):
        values = [t.total_ms if name == "total" else t.phases_ms[name] for t in window]
        summary[name] = {"mean": sum(values) / len(values), "max": max(values)}
    return summary


class Runtime:
    """Wires perception, brain, skills and action into the gather main loop.

    Attributes:
        device: Device used to capture frames and send actions.
        clock: Returns the current time in seconds (injectable for tests).
        sleep: Sleeps for the given seconds (injectable for tests).
        screen: Screen geometry.
        pathing: Pathing parameters.
        stuck: Stuck detection parameters.
        brain: Decision maker for the current game state.
        max_consecutive_failures: Failures in a row before a skill is disabled.
        perceiver: Perception layer producing immutable game states.
        navigator: Movement policy used by skills.
        controller: Action executor turning actions into device calls.
        timing_log_every: Ticks per INFO timing summary window.
        last_timing: Processing-time stats of the most recent tick, if any.
    """

    def __init__(
        self,
        device: Device,
        *,
        brain: Brain,
        skills: Mapping[str, Skill] | None = None,
        config: Mapping[str, Mapping[str, object]] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        screen: ScreenGeometry = SCREEN,
        pathing: PathingConfig = PATHING,
        stuck: StuckConfig = STUCK,
        max_consecutive_failures: int = 3,
        timing_log_every: int = 100,
    ) -> None:
        """Create the runtime with fresh perception, navigation and control layers.

        Args:
            device: Device to screenshot and send input to.
            brain: Brain deciding which skill runs on each game state.
            skills: Registered skills to run; defaults to the ones built from ``config``.
            config: Skill options keyed by skill name (the ``[skills.<name>]`` tables).
            clock: Returns the current time in seconds (injectable for tests).
            sleep: Sleeps for the given seconds (injectable for tests).
            screen: Screen geometry.
            pathing: Path planning parameters.
            stuck: Stuck detection parameters.
            max_consecutive_failures: Consecutive failures before a skill is disabled.
            timing_log_every: Log an INFO summary (per-phase mean/max in ms) every this
                many ticks, then reset the window. Per-tick timings log at DEBUG.

        Raises:
            ValueError: If ``timing_log_every`` is less than 1.
        """
        if timing_log_every < 1:
            raise ValueError(f"timing_log_every must be >= 1, got {timing_log_every}")
        self.device = device
        self.clock = clock
        self.sleep = sleep
        self.screen = screen
        self.pathing = pathing
        self.stuck = stuck
        self.brain = brain
        self.max_consecutive_failures = max_consecutive_failures
        self._config = dict(config) if config is not None else {}
        self._skills: dict[str, Skill] = (
            dict(skills) if skills is not None else build_skills(self._config)
        )
        self.perceiver = DefaultPerceiver(screen=screen, pathing=pathing, stuck=stuck)
        self.navigator = Navigator(screen=screen, pathing=pathing, stuck=stuck)
        self.controller = Controller(device=device, assets=None, screen=screen, sleep=sleep)
        self._current: Skill | None = None
        self._current_name: str | None = None
        self._current_rule: str | None = None
        self._interruptible: bool = False
        self._map_epoch: int = 0
        self._frame_index: int = 0
        self._started_at: float = 0.0
        self._steps: int = 0
        self._last_run: dict[str, float] = {}
        self._consecutive_failures: int = 0
        self._disabled: set[str] = set()
        self._last_decision: Decision | None = None
        self.timing_log_every = timing_log_every
        self.last_timing: StepTiming | None = None
        self._timing_window: list[StepTiming] = []

    def run(self, minutes: float) -> None:
        """Run :meth:`tick` until ``minutes`` have elapsed.

        Args:
            minutes: How long to gather for.
        """
        end = self.clock() + minutes * 60
        while self.clock() < end:
            self.tick()

    def tick(self) -> StepResult:
        """Run one perceive -> decide -> skill -> act cycle and log its processing time.

        Phase durations are measured with ``time.perf_counter`` only (never the injected
        clock), stored in :attr:`last_timing` and logged; see :meth:`_record_timing`.

        Returns:
            The outcome of the executed step, with the actions to execute (if any).
        """
        timer = _PhaseTimer(frame=self._frame_index)
        status = "error"
        start = time.perf_counter()
        try:
            result = self._tick(timer)
            status = result.status.value
            return result
        finally:
            self._record_timing(timer.finish(status, time.perf_counter() - start))

    def _tick(self, timer: _PhaseTimer) -> StepResult:
        """Run one perceive -> decide -> skill -> act cycle, timing each phase.

        Args:
            timer: Recorder for this tick's phase durations and skill name.

        Returns:
            The outcome of the executed step, with the actions to execute (if any).
        """
        with timer.phase("capture"):
            image = self.device.screenshot()
        frame = Frame(image=image, t=self.clock(), index=self._frame_index)
        self._frame_index += 1
        with timer.phase("perceive"):
            state = self.perceiver.perceive(frame)
            if state.map_epoch != self._map_epoch:
                self.navigator.reset()
                self._map_epoch = state.map_epoch

        with timer.phase("decide"):
            decision = self.brain.decide(state, self._brain_context())
            self._last_decision = decision
            timer.skill = decision.skill
            skill = self._select(decision)
        if skill is None:
            # The decided skill is disabled for this session; nothing to do this tick.
            return StepResult(status=SkillStatus.RUNNING, actions=())

        # Skill planning: lifecycle checks plus skill.step (where Navigator pathfinding runs).
        with timer.phase("skill"):
            if self._current is not skill:
                if self._current is not None:
                    self._current.abort("preempted")
                self._current = skill
                self._current_name = skill.name
                self._interruptible = skill.interruptible
                self._current_rule = decision.rule
                self._started_at = self.clock()
                self._steps = 0
                skill.start(state, self._skill_context(), decision.args)

            current = self._current
            if current is None:
                return StepResult(status=SkillStatus.FAILED, actions=(), reason="no-skill")
            timer.skill = current.name

            if current.is_done(state):
                self._consecutive_failures = 0
                self._finish(decision, SkillStatus.DONE)
                return StepResult(status=SkillStatus.DONE, actions=(), reason="done")

            reason: str | None = None
            if (
                current.timeout_s is not None
                and self.clock() - self._started_at > current.timeout_s
            ) or (current.max_steps is not None and self._steps >= current.max_steps):
                reason = "timeout"
            if reason is not None:
                return self._fail(decision, reason)

            ctx = self._skill_context()
            try:
                result = current.step(state, ctx)
            except Exception:
                logger.exception("skill %r failed", current.name)
                current.abort("error")
                return self._fail(decision, "error")

            if result.status is SkillStatus.FAILED:
                current.abort(result.reason or "error")
                return self._fail(decision, result.reason or "error")

        with timer.phase("act"):
            for action in result.actions:
                self.controller.execute(action)

        if result.status is SkillStatus.DONE:
            self._consecutive_failures = 0
            self._finish(decision, SkillStatus.DONE)
            return StepResult(status=SkillStatus.DONE, actions=result.actions, reason="done")

        self._steps += 1
        self._consecutive_failures = 0
        return StepResult(status=SkillStatus.RUNNING, actions=result.actions)

    def _record_timing(self, timing: StepTiming) -> None:
        """Store a tick's timing, log it at DEBUG, and log a windowed INFO summary.

        Every :attr:`timing_log_every` ticks an INFO record with per-phase mean/max (ms)
        over the window is emitted and the window is reset. The records carry their
        structured data in ``extra={"step_timing": ...}`` (per tick) and
        ``extra={"step_timing_summary": ...}`` (window summary).

        Args:
            timing: Processing-time stats of the tick that just ran.
        """
        self.last_timing = timing
        if logger.isEnabledFor(logging.DEBUG):
            phases = " ".join(f"{p}={timing.phases_ms[p]:.1f}ms" for p in PHASES)
            logger.debug(
                "step frame=%d skill=%s status=%s %s total=%.1fms",
                timing.frame,
                timing.skill,
                timing.status,
                phases,
                timing.total_ms,
                extra={"step_timing": timing.as_dict()},
            )
        self._timing_window.append(timing)
        if len(self._timing_window) >= self.timing_log_every:
            summary = summarize_timings(self._timing_window)
            steps = len(self._timing_window)
            self._timing_window = []
            text = " ".join(
                f"{name}={s['mean']:.1f}/{s['max']:.1f}ms" for name, s in summary.items()
            )
            logger.info(
                "step timing over %d steps (mean/max) %s",
                steps,
                text,
                extra={"step_timing_summary": {"steps": steps, **summary}},
            )

    @property
    def tracker(self) -> Tracker:
        """The perceiver's phase-correlation tracker (parity/test access)."""
        return self.perceiver.tracker

    @property
    def memory(self) -> WorldMemory:
        """The navigator's world memory (parity/test access)."""
        return self.navigator.memory

    @property
    def hist(self) -> list[FloatArray]:
        """Navigator stuck-detection history (parity/test access)."""
        return self.navigator.hist

    def _select(self, decision: Decision) -> Skill | None:
        """Pick the skill to run, skipping skills disabled for this session.

        Args:
            decision: The brain decision for this tick.

        Returns:
            The selected skill, or None when it is disabled or unknown.
        """
        if decision.skill in self._disabled:
            return None
        return self._skills.get(decision.skill)

    def _skill_context(self) -> SkillContext:
        """Build the context handed to the running skill.

        Returns:
            Context with the navigator, clock and options of the current activation.
        """
        name = self._current_name or ""
        return SkillContext(
            navigator=self.navigator,
            clock=self.clock,
            options=self._config.get(name) or {},
        )

    def _brain_context(self) -> BrainContext:
        """Build the context handed to the brain for the current tick.

        Returns:
            Brain context with the running activation's state.
        """
        last_run = getattr(self.brain, "last_run", self._last_run)
        if not isinstance(last_run, dict):
            last_run = self._last_run
        return BrainContext(
            now=self.clock(),
            current=self._current_name,
            current_rule=self._current_rule,
            current_interruptible=self._interruptible,
            can_run=self._can_run,
            last_run=last_run,
        )

    def _can_run(self, name: str, state: GameState) -> bool:
        """Whether the named skill may run on the given state.

        Args:
            name: Skill name.
            state: Current immutable game state.

        Returns:
            True when the skill applies to this state.
        """
        skill = self._skills.get(name)
        return skill is not None and skill.can_run(state)

    def _finish(self, decision: Decision, outcome: SkillStatus) -> None:
        """Notify the brain that the current activation ended and clear it.

        Args:
            decision: The decision that produced the outcome.
            outcome: Skill status the activation ended with.
        """
        now = self.clock()
        self._last_run[decision.skill] = now
        brain_last_run = getattr(self.brain, "last_run", None)
        if isinstance(brain_last_run, dict):
            brain_last_run[decision.skill] = now
        self.brain.notify(decision, outcome, now)
        self._current = None
        self._current_name = None

    def _fail(self, decision: Decision, reason: str) -> StepResult:
        """Abort the running skill as failed, then handle consecutive-failure bookkeeping.

        Args:
            decision: The decision that produced the failure.
            reason: Machine-readable failure reason.

        Returns:
            The failed step result.
        """
        current = self._current
        if current is None:
            return StepResult(status=SkillStatus.FAILED, actions=(), reason=reason)
        current.abort(reason)
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_consecutive_failures:
            if self._current_name is not None:
                self._disabled.add(self._current_name)
            logger.warning(
                "skill %s disabled after %d consecutive failures",
                self._current_name,
                self._consecutive_failures,
            )
        self._finish(decision, SkillStatus.FAILED)
        return StepResult(status=SkillStatus.FAILED, actions=(), reason=reason)
