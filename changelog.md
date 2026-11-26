# Changelog

## [Unreleased]

### Added
- Per-step processing-time stats (#19): `Runtime.tick` times the `capture`, `perceive`,
  `decide`, `skill` (planning/pathfinding) and `act` phases plus `total` with
  `time.perf_counter` (injected clock untouched, golden traces unchanged). Each tick logs a
  DEBUG line and every `timing_log_every` ticks (default 100) an INFO per-phase mean/max
  summary, both with structured `extra` fields; the last tick's `StepTiming` is exposed as
  `Runtime.last_timing`.
- Architecture refactor spec (`docs/specs/architecture-refactor.md`): skills system, rule-based brain,
  perception/decision/action/device layers, reference image library, and recording hook. Spec only;
  no behaviour change yet.
- Step 1 of the architecture refactor (§11): `tests/sim.py` adds `FakeWorld`, a seeded fake
  `Device` rendering hero-centred minimap frames from a synthetic world (ground, walls, handle-coloured
  nodes, invisible-wall cells, injected death ticks) with a fake clock and sleep; `swipe` from the
  joystick moves the hero. Golden device-call traces recorded from current `Bot` in
  `tests/golden/gather_*.json`, asserted by `tests/test_golden.py`. `.gitignore`: add
  `!src/eternal_hero_bot/assets/**/*.png` and ignore `assets/raw/`.
- Step 2 of the architecture refactor (§11): created `eternal_hero_bot.core` with moved
  `arrays.py` and `config.py` (tuned values unchanged), plus new `state.py` (`ScreenKind`,
  `Objective`, `ResourceReading`, `Detection`, `MinimapView`, `GameState`), `actions.py`
  (`Move`, `Tap`, `TapNamed`, `Swipe`, `Wait`, `Action`), and `geometry.py` (`Box` and
  reference/frame scaling helpers). Updated imports across `src/` and `tests/`. Added
  `tests/test_layers.py` to enforce the spec layer import allow-list with `ast` for packages
  present at this step.

- Step 3 of the architecture refactor (§11): created `eternal_hero_bot.perception` with moved
  `vision.py` and `tracker.py` (behaviour unchanged, numpy code verbatim); created
  `eternal_hero_bot.action` with `navigation.py` moved verbatim from `pathing.py`
  (`bfs`, `nearest_node`, `NEIGHBOURS`); split `device.py` into `eternal_hero_bot.device`
  package: `base.py` (`Device` protocol, signature unchanged, + `push_joystick`) and
  `adb.py` (`AdbDevice`, unchanged). `Bot` still drives. Imports in `bot.py`, `cli.py`,
  `debug.py`, `tests/sim.py` and tests updated; `tests/test_pathing.py` renamed
  `tests/test_navigation.py`; no compatibility shims. `tests/test_layers.py` updated for the
  new packages (`device`, `perception`, `action`) and allows numpy/PIL where required by
  `AdbDevice`, `vision`, `tracker`, `navigation`.
- Step 4 of the architecture refactor (§11): added `perception.frame.Frame` and
  `perception.perceiver` (`Perceiver` Protocol, `DefaultPerceiver`) producing immutable
  `GameState` snapshots (world/dead screen classification, tracker reset and `map_epoch`
  bump on first dead frame, node objectives, read-only minimap walk). Added
  `action.controller.Controller` executing `Action` values against a `Device` and moved
  `push_joystick` to the action layer. Extracted `Navigator` (stuck/fallback/explore and
  `WorldMemory`) into `action.navigation`; `plan()` remains available for detect. Refactored
  `Bot.step` to the perceive → navigator → controller wiring while preserving gather parity.
  Added unit tests for perceiver, controller, and navigator.
- Step 5 of the architecture refactor (§11): added the skill framework (`skills/` with the
  `Skill` protocol, `SkillContext`, `StepResult`/`SkillStatus` in `core/outcomes.py`, the
  `@register` decorator and `build_skills(config)`), ported the gather loop into the pure
  `GatherSkill` and the death flow into the `ReviveSkill`, and added the `Runtime` main loop
  with a minimal fixed brain (`revive-on-death` / `gather`). The `Bot` class was deleted and
  `ehbot run` now drives the runtime through perception, brain, skill and action. Golden
  device-call traces still pass through the runtime; ported bot tests and new lifecycle tests
  cover timeout, exception → FAILED and disable after N consecutive failures.
- Step 6 of the architecture refactor (§11): added the `brain/` package with the `Brain`/`BrainContext`
  protocol, `RuleBrain` (priority-ordered rules, cooldown/failure cooldown, preemption, `force`,
  `min_run_s`), and condition predicates with strict TOML parsing (unknown keys raise errors).
  Added packaged `default_config.toml` and `ehbot run --config FILE`. Registered stub skills
  `find_treasure`, `hunt`, `complete_altar`, `daily_tasks`, and `spend_resource` (`can_run=False`);
  default config keeps stub rules safely skipped until implemented. Golden gather traces remain identical.

## 0.2.0 - 2026-11-24

### Changed
- Restructured `gather.py` into the uv-managed `eternal_hero_bot` package (src layout).
- Run the bot with `uv run ehbot run` / `uv run ehbot detect` (or `python -m eternal_hero_bot`).
- Tuning constants moved from `gather.py` into frozen dataclasses in `config.py`.
- Requires Python 3.12 or newer (was 3.10+).
- Progress is written through the `logging` module instead of `print`.
- Ctrl+C stops `ehbot run` cleanly.
- Dependencies are managed with uv (`uv sync`) instead of `requirements.txt`.

### Added
- `ehbot detect --out` to choose the debug image path.
- `EHBOT_ADB` and `EHBOT_DEVICE` environment variables to override the adb path and device serial.
- Development tooling: ruff (lint and format), mypy strict, pytest with synthetic-image tests, and pre-commit hooks.

### Removed
- `gather.py` and `requirements.txt`.

## 0.1.0 - 2026-11-23

### Added
- `gather.py`: minimap-based auto-gatherer for Eternal Hero on BlueStacks via adb.
  - Node detection from tool-handle colours (ore, logs, gems), dungeon-key filtering.
  - Walkable-ground mask from the minimap palette and BFS pathfinding.
  - Hero position tracking via phase correlation; learned invisible walls and node skipping.
  - Automatic revive on death.
  - `detect` debug mode writing an annotated `detect.png`.
- README, requirements and changelog.
