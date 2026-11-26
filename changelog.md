# Changelog

## [Unreleased]

### Added
- Architecture refactor spec (`docs/specs/architecture-refactor.md`): skills system, rule-based brain,
  perception/decision/action/device layers, reference image library, and recording hook. Spec only;
  no behaviour change yet.

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
