# Changelog

## 0.1.0 - 2026-11-23

### Added
- `gather.py`: minimap-based auto-gatherer for Eternal Hero on BlueStacks via adb.
  - Node detection from tool-handle colours (ore, logs, gems), dungeon-key filtering.
  - Walkable-ground mask from the minimap palette and BFS pathfinding.
  - Hero position tracking via phase correlation; learned invisible walls and node skipping.
  - Automatic revive on death.
  - `detect` debug mode writing an annotated `detect.png`.
- README, requirements and changelog.
