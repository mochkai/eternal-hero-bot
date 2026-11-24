# eternal-hero-bot

Auto-gatherer for **Eternal Hero** running in BlueStacks (Windows), driven from WSL over adb.
It reads the minimap, finds resource nodes, walks the hero to them with the on-screen joystick,
and lets the game's proximity gathering do the rest. Pure pixel analysis: no AI/API calls.

## Requirements

- BlueStacks with **Settings → Advanced → Android Debug Bridge** enabled
- BlueStacks' bundled adb at `C:\Program Files\BlueStacks_nxt\HD-Adb.exe` (path set in `ADB` in `gather.py`)
- Python 3.10+, `numpy`, `pillow`

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python gather.py run 20    # gather for 20 minutes (Ctrl+C to stop)
.venv/bin/python gather.py detect    # debug: writes detect.png (minimap | walkable mask, nodes, path)
```

Game setup the script expects:

- Minimap at **max size** and default zoom (geometry constants `MM_CX`, `MM_CY`, `MM_R` assume this)
- Joystick at its default position (`JOY_X`, `JOY_Y`)
- Hero has labor points (each gather costs 10)

## How it works

1. **Nodes**: clusters of brown-orange tool-handle pixels on the minimap (pickaxe = ore,
   saw = logs, shovel = gems). Bright low-blue orange (dungeon key icons) is rejected.
2. **Walkable ground**: the flat mid grey-blue minimap colour `GROUND_RGB` (plus its darker
   shade under the hero's view cone). The lighter bands, bright outlines and dark navy are walls.
3. **Pathing**: BFS over the walkable mask to the nearest reachable node; the hero is steered
   toward a point `LOOKAHEAD` px along the path.
4. **Position tracking**: phase correlation of consecutive walkable masks gives the hero's
   world position, so learned obstacles persist while the hero moves.
5. **Stuck handling**: little net movement over `STUCK_STEPS` steps marks the stretch ahead as
   an invisible wall (remembered `BLOCK_SECONDS`); after `MAX_DETOURS` failures the node's area
   is skipped for `SKIP_SECONDS`.
6. **Death**: the Revive button is tapped automatically.

## Known limitations

- Fishing spots are not detected (fish icons have no orange handle).
- Changing minimap size or zoom requires recalibrating the geometry/colour constants.
- No count of successful gathers; watch the in-game log (bottom-left) to confirm.
