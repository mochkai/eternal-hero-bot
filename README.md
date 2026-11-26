# eternal-hero-bot

Auto-gatherer for **Eternal Hero** running in BlueStacks (Windows), driven from WSL over adb.
It reads the minimap, finds resource nodes, walks the hero to them with the on-screen joystick,
and lets the game's proximity gathering do the rest. Pure pixel analysis: no AI/API calls.

## Requirements

- BlueStacks with **Settings → Advanced → Android Debug Bridge** enabled
- BlueStacks' bundled adb at `C:\Program Files\BlueStacks_nxt\HD-Adb.exe` (override with `EHBOT_ADB`)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync                      # creates .venv and installs the pinned dependencies
```

## Usage

```bash
uv run ehbot run --minutes 20   # gather for 20 minutes (default; Ctrl+C to stop)
uv run ehbot detect             # debug: writes detect.png (minimap | walkable mask, nodes, path)
uv run ehbot detect --out x.png # choose the debug image path
uv run python -m eternal_hero_bot run   # equivalent module invocation
```

Game setup the bot expects:

- Minimap at **max size** and default zoom (`SCREEN.mm_cx`, `mm_cy`, `mm_r` assume this)
- Joystick at its default position (`SCREEN.joy_x`, `joy_y`)
- Hero has labor points (each gather costs 10)

## Configuration

- **Environment variables**
  - `EHBOT_ADB`: path to the adb executable (default: BlueStacks' bundled `HD-Adb.exe`)
  - `EHBOT_DEVICE`: adb device serial of the BlueStacks instance (default `emulator-5554`)
- **Tuned constants** live in `src/eternal_hero_bot/core/config.py` as frozen dataclasses:
  `ScreenGeometry` (minimap and joystick positions), `Colours` (walkable palette),
  `PathingConfig` (lookahead, wall margin) and `StuckConfig` (stuck and skip timings).
  They were calibrated for 960x1080 portrait; changing them changes the bot's behaviour.

## How it works

1. **Nodes**: clusters of brown-orange tool-handle pixels on the minimap (pickaxe = ore,
   saw = logs, shovel = gems). Bright low-blue orange (dungeon key icons) is rejected.
2. **Walkable ground**: the flat mid grey-blue minimap colour `COLOURS.ground_rgb` (plus its darker
   shade under the hero's view cone). The lighter bands, bright outlines and dark navy are walls.
3. **Pathing**: BFS over the walkable mask to the nearest reachable node; the hero is steered
   toward a point `PATHING.lookahead` px along the path.
4. **Position tracking**: phase correlation of consecutive walkable masks gives the hero's
   world position, so learned obstacles persist while the hero moves.
5. **Stuck handling**: little net movement over `StuckConfig.stuck_steps` steps marks the stretch
   ahead as an invisible wall (remembered `block_seconds`); after `max_detours` failures the
   node's area is skipped for `skip_seconds`.
6. **Death**: the Revive button is tapped automatically.

## Development

```bash
uv run ruff check        # lint
uv run ruff format       # format
uv run mypy              # strict type check
uv run pytest            # tests (synthetic images, no device needed)
uv run pre-commit install  # run ruff and mypy automatically on commit
```

## Project layout

```
src/eternal_hero_bot/
  core/
    arrays.py    shared numpy type aliases
    config.py    tuned constants as frozen dataclasses
    state.py     immutable game-state snapshots (perception output)
    actions.py   action union skills emit; controller executes
    geometry.py  Box and reference/frame scaling helpers
  device.py      adb wrapper for BlueStacks and the joystick helper
  vision.py      minimap crop, walkable mask, node and death-dialog detection
  pathing.py     breadth-first path search over the walkable mask
  tracker.py     hero position tracking by phase correlation
  bot.py         the gather loop: planning, wall learning, revive
  debug.py       annotated detect.png rendering
  cli.py         the ehbot command (run, detect)
  __main__.py    enables python -m eternal_hero_bot
tests/           pytest suite with synthetic screenshots and golden traces
```

## Known limitations

- Fishing spots are not detected (fish icons have no orange handle).
- Changing minimap size or zoom requires recalibrating the geometry/colour values in `core/config.py`.
- No count of successful gathers; watch the in-game log (bottom-left) to confirm.
