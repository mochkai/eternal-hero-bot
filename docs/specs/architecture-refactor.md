# Spec: architecture refactor (skills, brain, perception/action layers)

Status: **draft, awaiting approval**. No code under `src/` or `tests/` changes until approved.
Branch: `refactor/skills-architecture` (off `feat/uv-package`), draft PR against `feat/uv-package`.
Target version: 0.3.0.

## 1. Current state (brief)

`eternal_hero_bot` 0.2.0 is a gather-only bot. 31 tests, ruff, mypy strict: all green.

| Module | Today |
|---|---|
| `config.py` | Frozen dataclasses of tuned constants (`ScreenGeometry`, `Colours`, `PathingConfig`, `StuckConfig`, `DeviceConfig`), calibrated for 960x1080 portrait. |
| `device.py` | `Device` Protocol (`screenshot`, `tap`, `swipe`), `AdbDevice`, `push_joystick`. |
| `vision.py` | Minimap crop, walkable mask, node detection (handle colour clusters), `is_dead`. |
| `pathing.py` | 8-connected BFS, `nearest_node`. |
| `tracker.py` | Hero world position from phase correlation of consecutive walkable masks. |
| `bot.py` | `Bot.step()` does everything: screenshot, death check and revive, tracking, world memory (walls, skips, detours), planning, the straight-line fallback, explore, joystick push. Also `plan()` (used by `detect`). |
| `debug.py`, `cli.py` | `detect.png` rendering; `ehbot run` / `ehbot detect`. |

The problem: perception, decision and action are fused in `Bot.step`, so there is no room for a
second activity, and nothing reads the HUD.

## 2. Decisions

Taken from the brief, plus mine. ⚠ marks a point where I disagree or add something.

1. **Branching.** Work happens on `refactor/skills-architecture` off `feat/uv-package`, with the draft PR targeting `feat/uv-package`.
   ⚠ `feat/uv-package` is **not** the GitHub default branch: `main` is, and `feat/uv-package` is open PR #1.
   Stacking on it is still right. Once #1 merges, retarget this PR to `main`.
2. **Template matching uses `opencv-python-headless`** (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`, multi-scale). I agree, for these reasons:
   - It runs SIMD/FFT-optimised normalised cross-correlation. It supports masks, so it can match non-rectangular icons on varying backgrounds.
   - It also gives us `connectedComponentsWithStats`, which the digit reader uses for segmentation.
   - A numpy version would be roughly 10–100x slower per template and would duplicate well-tested code.
   - The headless wheel has no GUI or Qt libraries, so it is safe on WSL.

   Costs: a ~45 MB wheel, and stubs that are not perfect under mypy strict.
   Mitigation: `cv2` is imported in exactly **one** module (`perception/cv.py`). That module converts types at the boundary, and any `type: ignore` lives only there.
   Pillow stays for I/O, and the existing numpy vision code stays as it is (parity).
3. **Number reading** matches digit templates against the game font, with no Tesseract.
   It sits behind an `OCR` Protocol so the reader can be swapped later.
4. **Assets manifest is TOML**, read with stdlib `tomllib` (no new dependency). `tomllib` cannot write, so CLI helpers *append* new entries as text and never rewrite the file (§8.4).
5. **Gather parity is mandatory.** Verification uses a golden device-call trace (§10).
6. **Conventions:** Google-style docstrings everywhere, `changelog.md` updated, and ruff + mypy strict + pytest green after every step. Tests use fakes and never a live device.
7. ⚠ **Revive becomes its own skill**, triggered by a forced brain rule. Today it is hard-coded at the top of `Bot.step`. Observable behaviour stays the same (§10).
8. ⚠ **A `spend_resource` stub skill is added** next to the four you asked for, because the "resource at max → spend" rule needs a target. What "spend" means is an open question (§13).
9. **Recorder:** the interface plus a minimal JSONL+PNG recorder are **in scope**. A `watch` mode records without acting.
   ⚠ Capturing *your* inputs during manual play (via `adb shell getevent`) is designed here but deferred. It is unverified on BlueStacks and is not small.
10. ⚠ **`.gitignore` has `*.png`.** This would silently drop every template, and hatchling also respects `.gitignore` when building the wheel. We add `!src/eternal_hero_bot/assets/**/*.png` and ignore `assets/raw/`.
    The repo is **public**, so raw screenshots (player name, account details) never get committed. Only small curated crops do.
11. **No compatibility shims.** `vision.py`, `pathing.py`, `tracker.py` and `bot.py` move, and test imports are updated. Nothing outside this repo imports the package.

## 3. Target package layout

```
src/eternal_hero_bot/
  core/                     # shared vocabulary; imports nothing from the layers
    arrays.py               # (moved) numpy aliases
    config.py               # (moved) tuned dataclasses, unchanged values
    state.py                # GameState, Objective, ResourceReading, Detection, ScreenKind
    actions.py              # Action union: Move, Tap, TapNamed, Swipe, Wait
    outcomes.py             # Decision, SkillStatus, StepResult (shared by brain, skills, recording)
    geometry.py             # Box, scaling helpers (reference px <-> frame px)
  device/
    base.py                 # Device Protocol (unchanged signature) + Frame capture helper
    adb.py                  # AdbDevice (moved)
  perception/
    frame.py                # Frame(image int64, t, index) + cached uint8 view for cv2
    minimap.py              # (from vision.py) minimap, walkable_mask, find_nodes, icon_center, dilate
    screens.py              # is_dead + template-based screen classification
    tracker.py              # (moved) Tracker
    cv.py                   # the only cv2 import: match_template(), components()
    templates.py            # TemplateMatcher (multi-scale, region-limited)
    ocr.py                  # OCR Protocol, DigitTemplateOCR, number parsing
    perceiver.py            # Perceiver Protocol + DefaultPerceiver -> GameState
  assets/                   # package data, versioned with the code
    manifest.toml
    templates/{buttons,icons,nodes,screens}/*.png
    fonts/<font>/{0..9,k,m,slash,comma,dot}.png
  assets_lib.py             # manifest loading, validation, scaling (used by perception + controller)
  brain/
    base.py                 # Brain Protocol, Decision, BrainContext
    rules.py                # RuleBrain (priority list, cooldowns, default skill)
    conditions.py           # condition predicates + TOML parsing
  skills/
    base.py                 # Skill Protocol, SkillContext, StepResult, SkillStatus
    registry.py             # @register, build_skills(config)
    gather.py               # GatherSkill (parity port)
    revive.py               # ReviveSkill (parity port)
    stubs.py                # treasure, hunt, altar, daily, spend_resource (registered, can_run=False)
  action/
    controller.py           # Controller: Action -> Device calls (push_joystick lives here)
    navigation.py           # bfs, nearest_node, WorldMemory, Navigator (stuck/fallback/explore)
  recording/
    base.py                 # Recorder Protocol, StepRecord, NullRecorder
    jsonl.py                # JsonlPngRecorder
  runtime.py                # Runtime: the main loop wiring every layer
  default_config.toml       # brain rules + per-skill options
  debug.py, cli.py, __main__.py
```

### Layer boundaries (enforced by a test)

`tests/test_layers.py` parses every module's imports with `ast` and checks this allow-list:

| Package | May import |
|---|---|
| `core` | stdlib, numpy |
| `device` | `core` |
| `perception` | `core`, `assets_lib` (**not** `device`: it receives a `Frame`) |
| `brain` | `core` |
| `skills` | `core`, `action.navigation` (**not** `device`, `perception`, `controller`) |
| `action` | `core`, `assets_lib`, `device` (controller only) |
| `recording` | `core` |
| `runtime`, `cli`, `debug` | anything |

Skills are pure. They read a `GameState` and return `Action` values; they never touch the device.
This keeps them unit-testable, makes every action recordable, and lets the controller be faked.

## 4. Core types (public interfaces)

```python
# core/state.py
class ScreenKind(StrEnum):
    WORLD = "world"      # minimap visible, hero controllable
    DEAD = "dead"        # death dialog
    UNKNOWN = "unknown"  # anything else (popups, menus): future skills/close-popup handle it

@dataclass(frozen=True)
class Objective:
    kind: str                    # "node" today; later "treasure", "monster", "altar"
    dx: float; dy: float         # hero-relative minimap px (same convention as today)
    size: int = 0                # detector-specific weight (node handle pixel count)

@dataclass(frozen=True)
class ResourceReading:
    name: str
    value: int | None            # None = unreadable this frame
    max: int | None              # from "a/b" counters or static manifest max
    confidence: float
    @property
    def at_max(self) -> bool: ...    # value is not None and max is not None and value >= max

@dataclass(frozen=True)
class Detection:
    name: str; box: Box; score: float        # frame px

@dataclass(frozen=True)
class MinimapView:
    walk: Mask                   # walkable_mask(..., wall_margin) read-only (writeable=False)
    hero_pos: tuple[float, float]  # Tracker.pos after this frame's update

@dataclass(frozen=True)
class GameState:
    t: float; frame_index: int
    screen: ScreenKind
    map_epoch: int               # bumps when the tracker is reset (death/respawn)
    minimap: MinimapView | None  # None unless screen == WORLD
    objectives: tuple[Objective, ...]
    resources: Mapping[str, ResourceReading]
    detections: Mapping[str, tuple[Detection, ...]]
```

```python
# core/actions.py: everything a skill can ask for
@dataclass(frozen=True) class Move:     dx: float; dy: float; ms: int    # joystick push
@dataclass(frozen=True) class Tap:      x: int; y: int                   # frame px
@dataclass(frozen=True) class TapNamed: name: str                        # manifest region/template centre
@dataclass(frozen=True) class Swipe:    x1: int; y1: int; x2: int; y2: int; ms: int
@dataclass(frozen=True) class Wait:     seconds: float
type Action = Move | Tap | TapNamed | Swipe | Wait
```

## 5. Layers

### 5.1 Device

`Device` Protocol is unchanged (`screenshot() -> RGBImage`, `tap`, `swipe`). `AdbDevice` is unchanged.
Device errors (`TimeoutExpired`, a missing adb) still propagate and stop the run, as they do today.
A retry policy is a follow-up.

### 5.2 Perception

```python
class Perceiver(Protocol):
    def perceive(self, frame: Frame) -> GameState: ...
    def reset(self) -> None: ...          # new tracker, epoch += 1
```

`DefaultPerceiver(screen, pathing, stuck, assets: Assets | None, ocr: OCR | None)` does the following per frame:

1. `is_dead(img)` → `DEAD`. On the first DEAD frame it resets the tracker (`Tracker(...)`, so `pos=0` and `prev=None`) and bumps `map_epoch`. It returns no minimap and no objectives, and does **not** update the tracker. This matches today.
2. Otherwise it classifies the screen. If the manifest has `screens.*` templates, a template match decides the screen; if not, every non-dead frame is `WORLD` (parity for v1).
3. On `WORLD`, it computes `walk_raw = walkable_mask(img, wall_margin)`, calls `tracker.update(walk_raw.astype(float))`, and sets the objectives to `find_nodes(img)` as `Objective("node", dx, dy, size)`.
4. If the manifest declares resources, it reads each counter region with the `OCR` (§8.3).
5. It runs any manifest templates flagged `detect = true` within their search regions → `detections`.

The existing numpy functions are moved verbatim; their behaviour is unchanged.

```python
class TemplateMatcher:
    def __init__(self, assets: Assets) -> None: ...
    def find(self, frame: Frame, name: str, *, region: str | None = None) -> list[Detection]: ...
    def best(self, frame: Frame, name: str) -> Detection | None: ...

class OCR(Protocol):
    def read(self, frame: Frame, region: str) -> OCRResult: ...    # OCRResult(text, value, max, confidence)
```

### 5.3 Decision (brain)

```python
@dataclass(frozen=True)
class Decision:
    skill: str
    args: Mapping[str, object]
    rule: str                    # rule name, or "default"
    force: bool = False          # preempt even a non-interruptible skill

@dataclass(frozen=True)
class BrainContext:
    now: float
    current: str | None                  # running skill name
    current_rule: str | None
    current_interruptible: bool
    can_run: Callable[[str, GameState], bool]
    last_run: Mapping[str, float]        # skill -> last finish time (for conditions)

class Brain(Protocol):
    def decide(self, state: GameState, ctx: BrainContext) -> Decision: ...
    def notify(self, decision: Decision, outcome: SkillStatus, now: float) -> None: ...  # cooldown bookkeeping
```

`RuleBrain` works like this:

- **Rules are ordered; the list order is the priority.** Each tick the brain picks the **first** rule that passes three checks:
  - its `when` condition holds;
  - it is not on cooldown;
  - `ctx.can_run(skill, state)` is true.
- If no rule passes, the result is the `default` skill.
- **Preemption.** Suppose the chosen rule sits above the running skill's rule, and names a different skill. It replaces the running skill if `ctx.current_interruptible` is true or the rule has `force = true`. Otherwise the running skill continues.
- **Cooldowns.** `cooldown_s` starts when a skill started by that rule finishes or fails. `failure_cooldown_s` (default: `cooldown_s`) applies on failure or timeout.
- **`min_run_s`** (optional) keeps a lower rule from being displaced again too soon, which avoids thrash.

The brain sits behind a Protocol, so an `ImitationBrain` (§9.3) can replace `RuleBrain` without touching the other layers.

### 5.4 Skills

```python
# SkillStatus and StepResult live in core/outcomes.py (as does Decision), so brain and
# recording can use them without importing skills.
class SkillStatus(StrEnum): RUNNING = "running"; DONE = "done"; FAILED = "failed"

@dataclass(frozen=True)
class StepResult:
    status: SkillStatus
    actions: tuple[Action, ...] = ()
    reason: str = ""

@dataclass
class SkillContext:              # shared services the runtime hands to skills
    navigator: Navigator
    clock: Callable[[], float]
    options: Mapping[str, object]        # [skills.<name>] table from config

class Skill(Protocol):
    name: ClassVar[str]
    timeout_s: float | None              # wall-clock budget per activation (None = unlimited)
    max_steps: int | None
    @property
    def interruptible(self) -> bool: ...
    def can_run(self, state: GameState) -> bool: ...
    def start(self, state: GameState, ctx: SkillContext, args: Mapping[str, object]) -> None: ...
    def is_done(self, state: GameState) -> bool: ...     # state-derived completion, checked before step
    def step(self, state: GameState, ctx: SkillContext) -> StepResult: ...
    def abort(self, reason: str) -> None: ...            # release state; must not emit actions
```

**Lifecycle (run by `Runtime`)**

1. When the brain picks a skill different from the current one: `current.abort("preempted")` (if any), then `new.start(state, ctx, args)`. Start time and step count are reset.
2. Each tick:
   - If `is_done(state)` → DONE.
   - Else, if `timeout_s` or `max_steps` is exceeded → `abort("timeout")` → FAILED.
   - Else → `step()`, and the runtime executes the returned actions in order.
3. If `step` raises, the runtime logs the traceback, calls `abort("error")` and treats the outcome as FAILED. After `max_consecutive_failures` (config, default 3) the skill is disabled for the session and a warning is logged.
4. On DONE or FAILED, the runtime calls `brain.notify(...)` and clears the current skill. The next tick decides afresh.

**Registration and configuration.** `@register("gather")` on the class adds a factory to `registry.SKILLS`.
`build_skills(config)` builds every skill, passing it its own `[skills.<name>]` options table.
Config keys that are not recognised raise an error at startup.

**Skills in this refactor**

| Skill | Status | Behaviour |
|---|---|---|
| `gather` | full port | `can_run` is `screen == WORLD`. Never DONE, interruptible. Its options are only `timeout_s`/`max_steps` (default none). |
| `revive` | full port | `can_run` is `screen == DEAD`. Step 1: `TapNamed("revive_button")` (falls back to `SCREEN.revive` when the manifest has no such region), then `Wait(3.0)`, then DONE. Not interruptible. |
| `find_treasure` | stub | Intent: detect chest objectives (minimap marker + world template), navigate with `Navigator`, tap open, confirm the reward popup. |
| `hunt` | stub | Intent: detect monster objectives, approach to attack range, rely on auto-attack or tap attack, retreat when hero HP is low (HP bar region). |
| `complete_altar` | stub | Intent: find the altar objective, navigate, open its dialog, perform its requirement, claim. |
| `daily_tasks` | stub | Intent: open the daily-task menu, OCR task progress, claim completed rewards, pick tasks other skills can satisfy (feeds brain conditions). |
| `spend_resource` | stub (arg `resource`) | Intent: run the per-resource spend flow from §13, DONE when the resource drops below max. |

Every stub has `can_run → False` and a docstring describing its intent, and gets a follow-up issue.

### 5.5 Action

```python
class Controller:
    def __init__(self, device: Device, assets: Assets | None, screen: ScreenGeometry,
                 sleep: Callable[[float], None]) -> None: ...
    def execute(self, action: Action) -> None: ...
    # Move -> push_joystick (unchanged maths); Tap/Swipe -> device; TapNamed -> manifest
    # region centre scaled to the frame; Wait -> sleep
```

`Navigator` (in `action/navigation.py`) is the movement policy pulled out of `Bot`, so treasure, hunt and altar can reuse it.

```python
class Navigator:
    memory: WorldMemory          # walls, skips, detours (moved unchanged)
    hist: list[FloatArray]       # stuck history
    def reset(self) -> None: ...                       # memory.clear(); hist kept (see §10 note)
    def step_toward(self, view: MinimapView, targets: list[Offset], now: float) -> Action | None: ...
    def explore(self, walk: Mask) -> Move: ...
```

`step_toward` is the body of today's `Bot.step` after the death check, kept in exactly the same order. In outline:

1. Expire memory, then filter targets against skips using `skip_radius`.
2. Build `walk = walk_raw & ~blocked` and call `open_start`, then run BFS.
3. Record history, then check whether the hero is stuck. If stuck, run `_handle_stuck` and return `None`, which means "replan next frame".
4. If no path was found, use the straight-line fallback.
5. If there are no nodes, explore. Otherwise return a `Move` using lookahead with `ms` clamped to [250, 1000].

`plan()` stays available for `detect`.

## 6. Main loop data flow

```
            ┌────────── Runtime.tick() ───────────────────────────────────────────────┐
 Device ──► capture: Frame(device.screenshot(), clock(), i)                           │
            │   state = perceiver.perceive(frame)                  # perception        │
            │   if state.map_epoch changed: ctx.navigator.reset()                      │
            │   decision = brain.decide(state, BrainContext(...))   # decision         │
            │   switch skill if needed (abort / start)                                 │
            │   result = skill.is_done / timeout / skill.step(state, ctx)  # skill     │
            │   for a in result.actions: controller.execute(a)    # action ──► Device  │
            │   recorder.record(StepRecord(frame, state, decision, result, "bot"))  ◄── hook
            │   on DONE/FAILED: brain.notify(...)                                      │
            └──────────────────────────────────────────────────────────────────────────┘
Runtime.run(minutes): tick until clock() >= start + minutes*60   (same as Bot.run)
Runtime.watch(minutes): capture → perceive → record(source="human"); no brain/skill/controller
```

There is one screenshot per tick and at most one joystick push per gather tick, the same as today.

## 7. Brain rule config (`default_config.toml`, overridable with `ehbot run --config FILE`)

```toml
[brain]
default = "gather"
max_consecutive_failures = 3

# Order = priority. First matching rule wins.
[[brain.rules]]
name = "revive-on-death"
skill = "revive"
force = true
when = { screen = "dead" }

[[brain.rules]]
name = "spend-gold-at-max"          # worked example: resource at max -> spend before gathering more
skill = "spend_resource"
args = { resource = "gold" }
cooldown_s = 300                    # don't retry for 5 min after it finishes/fails
failure_cooldown_s = 900
when = { resource_at_max = "gold" }

[[brain.rules]]
name = "daily-tasks-after-reset"
skill = "daily_tasks"
cooldown_s = 3600
when = { all = [ { screen = "world" }, { since_last_run = { skill = "daily_tasks", s = 21600 } } ] }

[[brain.rules]]
name = "hunt-when-labor-low"
skill = "hunt"
when = { resource_lt = { name = "labor", value = 10 } }

[skills.gather]
timeout_s = 0          # 0 = unlimited

[skills.spend_resource]
timeout_s = 120
```

**How the worked example plays out.** Gather is running. The gold counter reads `500/500`, so `ResourceReading.at_max` is true.

- `spend-gold-at-max` ranks above the default and gather is interruptible, so the runtime aborts gather and starts `spend_resource(resource="gold")`.
- When spend reports DONE (or `is_done` sees gold below max), the rule's 300 s cooldown starts and the brain falls back to `gather`.
- If spend fails, it is not retried for 900 s, so the bot keeps gathering instead of looping.
- Today the stub's `can_run` is False, so the rule is skipped. That is safe until the skill is real.

**Condition predicates.** These are parsed once at load, and unknown keys are errors. A condition that reads an unknown value (`value is None`) is **False**.

| Predicate | Meaning |
|---|---|
| `screen = "<kind>"` | `state.screen` equals the kind |
| `resource_at_max = "<name>"` | reading `at_max` |
| `resource_gte` / `resource_lt = {name, value}` | numeric compare |
| `resource_frac_gte = {name, frac}` | value/max ≥ frac |
| `detected = "<template>"` | at least one detection of that template |
| `objective = "<kind>"` | at least one objective of that kind |
| `since_last_run = {skill, s}` | skill hasn't finished in `s` seconds (or never) |
| `all = [...]`, `any = [...]`, `not = {...}` | combinators |

## 8. Reference image library

### 8.1 Layout

```
assets/raw/                               # repo root, git-ignored: your full screenshots
  hud/ nodes/ buttons/ icons/ max/ treasure/ monsters/ altar/ daily/ popups/ digits/
src/eternal_hero_bot/assets/              # package data, committed
  manifest.toml
  templates/buttons/revive.png
  templates/buttons/revive.mask.png       # optional: white = compare, black = ignore
  templates/icons/gold.png
  templates/nodes/ore.png
  templates/screens/death_dialog.png
  fonts/hud/0.png … 9.png  k.png m.png slash.png comma.png dot.png
```

**Versioning.** `manifest.toml` has `schema` (the loader's format version) and `version` (the content version, bumped whenever templates change).
Every recording stores the manifest version (§9). An `--assets DIR` flag lets you point the bot at a working copy outside the package.

### 8.2 Manifest schema (example)

```toml
schema = 1
version = "2026.12.0"

[reference]                 # resolution every box/template below was captured at
width = 960
height = 1080
density_dpi = 240           # informational; confirm with `adb shell wm density`

[scaling]
mode = "uniform"            # v1: frame must have the reference aspect ratio (±1%);
                            # boxes scale by frame_w/ref_w, templates are resized by the same factor
search_scales = [0.95, 1.0, 1.05]   # extra multipliers tried around the uniform factor
default_threshold = 0.85

[regions]                   # [x, y, w, h] in reference px
hud_gold      = { box = [600, 20, 120, 28] }
hud_labor     = { box = [760, 20, 120, 28] }
death_dialog  = { box = [200, 500, 560, 250] }
revive_button = { box = [400, 625, 160, 52], tap = true }   # TapNamed("revive_button") target

[templates.revive]
file = "templates/buttons/revive.png"
mask = "templates/buttons/revive.mask.png"
search = "death_dialog"     # region to search (default: full frame; slower)
threshold = 0.88
detect = true               # run every frame -> state.detections["revive"]

[templates.death_dialog]
file = "templates/screens/death_dialog.png"
search = "death_dialog"
screen = "dead"             # a match classifies the frame as ScreenKind.DEAD

[fonts.hud]
dir = "fonts/hud"
glyphs = { "/" = "slash", "," = "comma", "." = "dot" }   # filename stems for non-digit glyphs
threshold = 0.75
text_rgb = [255, 255, 255]  # binarisation: pixels within text_tol of text_rgb are ink
text_tol = 60
suffixes = { k = 1_000, m = 1_000_000 }

[resources.gold]
region = "hud_gold"
font = "hud"
format = "current/max"      # "current" | "current/max"
icon = "templates/icons/gold.png"

[resources.labor]
region = "hud_labor"
font = "hud"
format = "current"
max = 100                   # static cap when the HUD doesn't show it
```

The loader (`assets_lib.load_assets(path) -> Assets`) validates the manifest at startup:

- every file exists;
- every box fits inside the reference frame;
- every template is smaller than its search region;
- all references (`search`, `region`, `font`) resolve.

**Digit reading algorithm.**

1. Crop the region and binarise it by `text_rgb`/`text_tol`.
2. Segment glyphs with connected components, merging components that overlap in x.
3. Resize each glyph to the template height and score it against every glyph template with `TM_CCOEFF_NORMED`.
4. Join the best-scoring glyphs into a string. Confidence is the lowest glyph score.
5. Parse the string (`1.2k`, `1,234`, `120/500`).

If confidence is below the font threshold, `value=None`.

### 8.3 CLI helpers

| Command | Does |
|---|---|
| `ehbot capture [--out assets/raw/] [--name hud/main__gold-1234]` | One lossless `screencap` PNG at device resolution, plus a `.json` sidecar (resolution, density, time). |
| `ehbot assets grid RAW.png [--step 20]` | Writes `RAW.grid.png` with a labelled coordinate grid, so boxes can be read off it. |
| `ehbot assets crop RAW.png --box x,y,w,h --as templates/buttons/revive [--region revive_button]` | Saves the crop into the assets dir and **appends** a `[templates.*]` (and optional `[regions.*]`) block to `manifest.toml`. Refuses if the key already exists. |
| `ehbot assets digits RAW.png --region hud_gold --text 1234/5000 --font hud` | Segments the glyphs in a region and saves each one as the font template for the matching character in `--text`. |
| `ehbot assets check` | Validates the manifest (as above). |
| `ehbot assets test RAW.png [--out test.png]` | Runs every template and OCR region against a screenshot, prints scores and readings, and writes an annotated image. This is the main tool for tuning thresholds. |

The existing `ehbot detect` stays.

## 9. Recording and the future learned brain

### 9.1 Hook

```python
@dataclass(frozen=True)
class StepRecord:
    frame: Frame
    state: GameState
    decision: Decision | None            # None in watch mode
    result: StepResult | None            # actions actually executed
    source: Literal["bot", "human"]

class Recorder(Protocol):
    def record(self, rec: StepRecord) -> None: ...
    def close(self) -> None: ...
```

The `Runtime` takes `recorder: Recorder = NullRecorder()`, and the recorder is called once per tick after the actions run.

### 9.2 Format (`JsonlPngRecorder`, in scope)

```
recordings/2026-12-01T10-00-00/
  meta.json       {"format": 1, "bot_version": "0.3.0", "assets_version": "2026.12.0",
                   "mode": "bot"|"watch", "resolution": [960, 1080], "config": {...resolved TOML...}}
  frames/000123.png                    # every Nth frame (--frame-every, default 1)
  steps.jsonl     one line per tick:
  {"i": 123, "t": 1764583200.12, "frame": "frames/000123.png" | null,
   "state": {"screen": "world", "map_epoch": 0, "hero_pos": [12.0, -3.0],
             "objectives": [{"kind": "node", "dx": 50.0, "dy": 0.0, "size": 16}],
             "resources": {"gold": {"value": 500, "max": 500, "confidence": 0.93}},
             "detections": {"revive": [{"box": [400, 625, 160, 52], "score": 0.91}]}},
   "decision": {"skill": "gather", "rule": "default", "args": {}},
   "status": "running",
   "actions": [{"type": "move", "dx": 12.0, "dy": -1.0, "ms": 857}],
   "source": "bot"}
```

Masks are not stored, because they can be recomputed from frames. Usage: `ehbot run --record DIR [--frame-every N]` and `ehbot watch --record DIR --minutes M`.
Watch mode records frames and state with `actions: []`.

### 9.3 Future (design only)

- **Human input capture.** An `InputSource` Protocol yields `Action`s with timestamps. The planned implementation streams `adb shell getevent -lt`, maps raw touch coordinates to frame px, and turns down/move/up sequences into `Tap`/`Swipe`/`Move` (a swipe starting on the joystick becomes a `Move`).
  In watch mode those actions fill `actions` with `source="human"`.
  It is unverified whether BlueStacks keymapping (WASD) shows up in `getevent`; that needs a spike.
- **`ImitationBrain`** implements `Brain`. A first version learns skill selection (state features → `Decision`), which needs only recordings of bot runs annotated by you.
  A later version learns low-level actions (frame → `Move`), wrapped as a `learned_control` skill, so the layer boundaries hold.
  Training happens offline from `recordings/`. Model files are versioned with the `assets_version` and resolution they were trained on.

## 10. Gather parity: how it's verified

1. **Golden trace (the main guarantee).** This lands in step 1, *before* any code moves.
   - `tests/sim.py` adds `FakeWorld`, a `Device` backed by a seeded synthetic world bitmap (ground, walls, handle-coloured nodes). `screenshot()` renders the hero-centred minimap into a 960x1080 frame. `swipe()` from the joystick moves the hero by direction × ms × speed, unless it hits a wall or an "invisible wall" cell. Death frames can be injected at given ticks.
   - Today's `Bot` runs on 3 scripted scenarios for 80 ticks each, using a fake clock and sleep: an open field with nodes, an invisible wall that forces wall learning and then area skipping, and a death mid-run.
   - Every device call (`tap`/`swipe` with exact args and `ms`) and every `sleep` is written to `tests/golden/gather_*.json`.
   - After every later step, `Runtime` with `default_config.toml` (only `revive` + default `gather` active) must reproduce **identical** call lists.
2. **Ported unit tests.** All 31 existing tests keep their assertions; only imports and constructors change.
   - `test_bot.py` → `test_gather.py` + `test_navigation.py` + `test_runtime.py`. For example, "revive clears memory and resets tracker" becomes one `Runtime.tick()` on a dead frame. That tick must produce `taps == [SCREEN.revive]`, no swipes, `perceiver` tracker `prev is None`, `pos == 0`, and empty `navigator.memory.walls` before the next gather step.
   - The vision, tracker, pathing, debug and CLI tests move with their modules.
3. **Known quirks are preserved, not fixed.**
   - The stuck history `hist` is not cleared on death.
   - `plan()` uses `skip_match_r=10` while the loop uses `skip_radius=18`.
   - Revive taps the fixed `SCREEN.revive` coordinate when no manifest region exists.

   Each is noted in code with a `# parity:` comment, so it can be fixed separately later.
4. **Live smoke test (you).** `ehbot run --minutes 10 --record recordings/parity` on the real game. It should behave the same as 0.2.0, and the recording gives us real frames for later regression tests.

## 11. Migration plan (each step is one PR-sized commit, green on ruff + mypy + pytest)

| # | Step | Verifies |
|---|---|---|
| 1 | `tests/sim.py` FakeWorld + golden traces recorded from current `Bot`; `.gitignore` fixes | Golden files exist; old code passes them |
| 2 | Create `core/` (arrays, config moved; `state.py`, `actions.py`, `geometry.py` new) + `tests/test_layers.py` | Existing tests, layer test |
| 3 | Move `vision`/`tracker` → `perception/`, `pathing` → `action/navigation.py`, `AdbDevice` → `device/`; `Bot` still drives | All tests + golden |
| 4 | `DefaultPerceiver` → `GameState`; `Controller` + `Action`s; `Navigator` extracted; `Bot.step` becomes perceive → navigator → controller | Golden; new perceiver/controller/navigator unit tests |
| 5 | Skill framework, registry, `GatherSkill`, `ReviveSkill`, `Runtime` with a minimal fixed brain; delete `Bot`; CLI `run` uses `Runtime` | Golden via Runtime; ported bot tests; lifecycle tests (timeout, exception → FAILED, disable after N) |
| 6 | `RuleBrain`, conditions, `default_config.toml`, `--config`; stubs registered | Rule order, cooldown, preemption, `force`, unknown-key errors; golden still identical |
| 7 | `opencv-python-headless` dep; `assets_lib` (manifest, validation, scaling); `perception/cv.py`, `TemplateMatcher`; empty packaged manifest | Synthetic templates pasted into frames at 1.0/0.95/1.05 scale; manifest validation errors |
| 8 | `OCR` Protocol + `DigitTemplateOCR`; resources into `GameState` | Digits rendered from a 5x7 bitmap font in tests: `1234`, `120/500`, `1.2k`, low-confidence → None |
| 9 | CLI: `capture`, `assets grid/crop/digits/check/test` | Fake device + tmp dirs; crop append refuses duplicates |
| 10 | `Recorder`, `JsonlPngRecorder`, `run --record`, `watch` | Round-trip: records parse; frame-every N |
| 11 | README, changelog, version 0.3.0; follow-up issues for the 5 stubs + input capture + ImitationBrain | Docs review |

Once your reference material arrives, a follow-up PR fills in the real manifest and templates and replaces `is_dead`'s probe pixel with the death-dialog template. That swap is behind parity checks on your recordings.

## 12. Test strategy, risks, out of scope

**Tests.**
- Fakes only: `FakeDevice` (records calls), `FakeWorld` (sim), `FakePerceiver` (scripted `GameState`s for brain and skill tests), and synthetic templates and digits drawn with numpy and Pillow.
- No real game images in CI until you provide them. After that, `assets test` fixtures come from `assets/raw` crops with ground-truth values in the filenames (§13).

**Risks.**

| Risk | Mitigation |
|---|---|
| Template matching is brittle under animation, glow or lighting | Masks, per-template thresholds, search regions, `assets test` tuning loop |
| Digits are outlined/gradient or proportional | Colour-tolerance binarisation, per-font threshold, `None` on low confidence (conditions go False, so they fail safe) |
| The BlueStacks resolution or window changes | `screencap` is resolution-exact; startup check of the frame aspect ratio against the manifest; clear error |
| cv2 stubs vs mypy strict, wheel size | Single wrapper module; pin `>=4.10` (numpy 2 support) |
| int64 frames are wasteful for cv2 | `Frame.u8` cached view; legacy code keeps int64 (parity) |
| Brain thrash between skills | Cooldowns, `min_run_s`, `interruptible` |
| Parity drift during the move | Golden trace gate at every step |

**Out of scope.**
- Real implementations of treasure, hunt, altar, daily and spend.
- Human input capture.
- Training any model.
- Non-uniform or anchored scaling (only a reserved `anchor` field).
- Moving `ScreenGeometry` and `Colours` into the manifest.
- Fishing nodes.
- A GUI cropping tool.
- Device retry and reconnect.

## 13. What we need from you

**Capture settings.**
- Keep BlueStacks at **960x1080 portrait**, the resolution everything is calibrated for.
- Run `adb shell wm size` and `adb shell wm density` and send me the output.
- Minimap at max size, default zoom, joystick at its default position.
- Capture with adb, not a Windows screenshot, so the pixels are exact:
  ```bash
  "$EHBOT_ADB" -s emulator-5554 exec-out screencap -p > assets/raw/hud/main__gold-1234_labor-80.png
  ```
  (`ehbot capture` will do this once step 9 lands.)
- Put the files in **`assets/raw/<category>/`**, named `<what>__<state-or-values>.png`. Values in the name become OCR ground truth.

**Checklist.**

- [ ] `hud/`: the main HUD with every resource counter visible, 3+ screenshots with different values.
- [ ] `max/`: each capped resource **at its maximum** (and how it looks when full: colour change, "MAX" label?).
- [ ] `digits/`: counters that together show **0–9**, plus every separator or suffix the game uses (`/`, `,`, `.`, `k`, `m`, …). Put the exact text in the filename.
- [ ] `nodes/`: each gatherable node (ore, logs, gems, fish spot) in the main view and on the minimap, in both available and depleted/cooldown states.
- [ ] `icons/`: each resource icon (HUD and inventory, if they differ).
- [ ] `buttons/`: every button the bot will press. That includes Revive (death dialog) and popup close/X and confirm buttons, and any buttons in the "spend" flows. Take each screenshot with the button visible, and pressed or disabled too if that state exists.
- [ ] `popups/`: anything that interrupts play (level-up, offers, disconnect, rewards).
- [ ] `treasure/`: a chest in the world and on the minimap, the open/interaction prompt, the reward screen.
- [ ] `monsters/`: monsters in view and on the minimap, the attack UI (if any), hero and monster HP bars, low-HP state.
- [ ] `altar/`: an altar in the world and on the minimap, its dialog, requirement display, completed state.
- [ ] `daily/`: the button that opens daily tasks, the task list (in-progress, claimable and claimed states), the claim button, the reward popup.
- [ ] Optional: a ~10-minute `ehbot run --record` after step 10, for regression frames.

**Open questions (only you can answer).**

1. Which resources does the HUD show, and which have a **maximum**? Is that max shown (`120/500`) or fixed?
2. What does **"spend"** mean for each capped resource (craft, upgrade, sell, deposit, a specific NPC or menu)? What is the exact tap sequence?
3. **Labor points**: when labor is below 10 (gathering does nothing), should the bot wait, switch to hunt or treasure, or stop?
4. **Priorities**: what order do you want? For example, revive > spend-at-max > daily tasks > treasure in view > altar > hunt > gather.
5. Does the game **abbreviate** large numbers (`1.2k`, `3M`) or use thousands separators?
6. Is **960x1080** the only resolution you'll run at, or do we need others (which affects scaling work)?
7. **Hunting**: is combat automatic on proximity (like gathering) or button-driven? Which deaths or risks are acceptable?
8. **Daily tasks**: when is the daily reset (time and timezone), and which tasks are realistically automatable?
9. **Manual play** for future recording: do you play with the mouse or with BlueStacks keyboard mapping (WASD)?
10. Are you OK with the `opencv-python-headless` dependency (~45 MB) and with committing small game-art crops to this **public** repo?
11. Should this PR be retargeted to `main` once PR #1 merges, or should `feat/uv-package` become the long-lived base?
