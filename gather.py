"""Auto-gatherer for Eternal Hero on BlueStacks via adb.

Reads the minimap, finds resource-node icons (orange tool handles), plans a
wall-avoiding path on the minimap (dark pixels = walls), steers the hero along it
with the virtual joystick, and revives on death. Gathering itself is triggered by
the game on proximity. No AI/API calls: pure pixel analysis, so it costs no tokens.

Usage:
    python gather.py detect          # annotate minimap + walls + path -> detect.png (debug)
    python gather.py run [minutes]   # gather loop (default 20 minutes)
"""

import io
import math
import subprocess
import sys
import time
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

ADB = "/mnt/c/Program Files/BlueStacks_nxt/HD-Adb.exe"
DEV = ["-s", "emulator-5554"]

# Screen geometry (960x1080 portrait, minimap at max size)
MM_CX, MM_CY, MM_R = 127, 292, 98   # minimap centre (= hero) and usable radius
JOY_X, JOY_Y, JOY_R = 480, 825, 90  # joystick centre and push distance
PLAYER_R = 11                       # hero arrow radius on minimap
REVIVE = (480, 651)

GROUND_RGB = np.array([81, 97, 114])  # walkable minimap colour (mid grey-blue)
GROUND_TOL = 13
CONE_RGB = np.array([69, 84, 100])    # walkable colour under the hero's view cone
CONE_TOL = 9
WALL_MARGIN = 1        # px of clearance kept from walls
GOAL_R = 12           # path may end this close to a node (nodes sit at ground edges)
LOOKAHEAD = 12         # px along the path to aim at
PX_PER_SEC = 14        # rough hero speed in minimap px per second


def adb(*args: str, capture: bool = False) -> bytes:
    """Run an adb command against the BlueStacks device."""
    r = subprocess.run([ADB, *DEV, *args], capture_output=True, timeout=30)
    return r.stdout if capture else b""


def screenshot() -> np.ndarray:
    """Capture the screen as an RGB numpy array."""
    png = adb("exec-out", "screencap", "-p", capture=True)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)


def is_dead(img: np.ndarray) -> bool:
    """Detect the 'You died' dialog by its green Revive button."""
    r, g, b = img[651, 400]
    return g > 170 and r > 120 and b < 110 and g > r


def minimap(img: np.ndarray) -> np.ndarray:
    """Return the square minimap crop centred on the hero (2R x 2R x 3)."""
    return img[MM_CY - MM_R:MM_CY + MM_R, MM_CX - MM_R:MM_CX + MM_R]


def _shift_any(m: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation of mask m by a square of radius r (numpy only)."""
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def walkable_mask(img: np.ndarray) -> np.ndarray:
    """Boolean grid of walkable minimap pixels (True = can walk).

    Dark navy pixels are walls. Thin dark icon outlines are closed over first so
    icons/text don't become fake walls, then walls are grown by WALL_MARGIN.
    """
    mm = minimap(img)
    # Walkable ground is one flat palette colour; the hero's view cone darkens it slightly
    d_ground = np.sqrt(((mm - GROUND_RGB) ** 2).sum(2))
    d_cone = np.sqrt(((mm - CONE_RGB) ** 2).sum(2))
    ground = (d_ground < GROUND_TOL) | (d_cone < CONE_TOL)
    # Close 1px speckles (anti-aliasing), then keep a margin from walls
    ground = ~_shift_any(~_shift_any(ground, 1), 1)
    wall = _shift_any(~ground, WALL_MARGIN)
    n = 2 * MM_R
    yy, xx = np.mgrid[0:n, 0:n]
    d2 = (xx - MM_R) ** 2 + (yy - MM_R) ** 2
    walk = ~wall & (d2 < MM_R ** 2)
    walk[d2 < PLAYER_R ** 2] = True  # hero arrow is always standable
    return walk


def find_nodes(img: np.ndarray) -> list[tuple[float, float, int]]:
    """Return node candidates as (dx, dy, size) relative to the hero on the minimap.

    Nodes are found by clustering brown-orange tool-handle pixels inside the minimap.
    """
    sub = minimap(img)
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    # Brown-orange handles (~145,90,65); minimap ground is blue-grey (b >= r)
    mask = (r > 105) & (r - b > 50) & (g > r * 0.4) & (g < r * 0.8)
    n = 2 * MM_R
    yy, xx = np.mgrid[0:n, 0:n]
    d2 = (xx - MM_R) ** 2 + (yy - MM_R) ** 2
    mask &= (d2 < MM_R ** 2) & (d2 > PLAYER_R ** 2)

    # Greedy clustering: merge pixels within 7px of a cluster's centroid
    clusters: list[list] = []
    for py, px in zip(*np.nonzero(mask)):
        for c in clusters:
            if abs(c[0] / c[2] - px) < 7 and abs(c[1] / c[2] - py) < 7:
                c[0] += px; c[1] += py; c[2] += 1
                break
        else:
            clusters.append([px, py, 1])
    out = []
    for sx, sy, cnt in clusters:
        if cnt < 4:
            continue
        cx, cy = sx / cnt, sy / cnt
        # Reject bright, low-blue orange (dungeon key icons): handles are muted brown
        patch = sub[max(0, int(cy) - 3):int(cy) + 4, max(0, int(cx) - 3):int(cx) + 4].reshape(-1, 3)
        pr, pg, pb = patch[:, 0], patch[:, 1], patch[:, 2]
        pm = (pr > 105) & (pr - pb > 50) & (pg > pr * 0.4) & (pg < pr * 0.8)
        if pm.any():
            mr, _, mb = patch[pm].mean(0)
            if mb < mr * 0.3:
                continue
        cx, cy = icon_center(sub, mask, cx, cy)
        out.append((cx - MM_R, cy - MM_R, cnt))
    return out


def icon_center(sub: np.ndarray, handle: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Centre of the whole tool icon (handle + neutral-grey metal) near a handle cluster.

    The handle sits off to one side of the node, so the icon's centre is a better target.
    """
    R = 11
    y0, x0 = max(0, int(cy) - R), max(0, int(cx) - R)
    p = sub[y0:int(cy) + R + 1, x0:int(cx) + R + 1]
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    metal = (p.mean(2) > 140) & (np.abs(r - b) < 25) & (np.abs(r - g) < 20)
    m = metal | handle[y0:y0 + p.shape[0], x0:x0 + p.shape[1]]
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return cx, cy
    return xs.mean() + x0, ys.mean() + y0


def bfs(walk: np.ndarray, targets: list[tuple[float, float]]) -> list[tuple[int, int]] | None:
    """Shortest 8-connected path from the hero to the nearest reachable target.

    Targets are (dx, dy) relative to the hero; a small disc around each is made
    walkable since icons sit on their node. Returns path as [(x, y), ...] in
    minimap-crop pixels, or None if no target is reachable.
    """
    walk = walk.copy()
    n = walk.shape[0]
    goal = np.zeros_like(walk)
    yy, xx = np.mgrid[0:n, 0:n]
    for dx, dy in targets:
        disc = (xx - (MM_R + dx)) ** 2 + (yy - (MM_R + dy)) ** 2 < GOAL_R ** 2
        walk |= disc
        goal |= disc
    start = (MM_R, MM_R)
    prev = {start: None}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if goal[y, x]:
            path = []
            node = (x, y)
            while node:
                path.append(node)
                node = prev[node]
            return path[::-1]
        for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + ddx, y + ddy
            if 0 <= nx < n and 0 <= ny < n and walk[ny, nx] and (nx, ny) not in prev:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return None


def push(dx: float, dy: float, ms: int) -> None:
    """Hold the joystick in direction (dx, dy) for ms milliseconds."""
    L = math.hypot(dx, dy) or 1
    tx, ty = JOY_X + dx / L * JOY_R, JOY_Y + dy / L * JOY_R
    adb("shell", "input", "swipe", str(JOY_X), str(JOY_Y), str(int(tx)), str(int(ty)), str(ms))


def plan(img: np.ndarray, skip: list[tuple[float, float]], blocked: np.ndarray | None = None):
    """Find nodes, drop skipped ones, and path to the nearest reachable one.

    `blocked` is an extra wall mask learned from places the hero got stuck.
    """
    nodes = [(dx, dy) for dx, dy, _ in find_nodes(img)
             if not any(math.hypot(dx - sx, dy - sy) < 10 for sx, sy in skip)]
    walk = walkable_mask(img)
    if blocked is not None:
        walk &= ~blocked
        walk[MM_R - 2:MM_R + 3, MM_R - 2:MM_R + 3] = True  # keep the start cell open
    return nodes, walk, (bfs(walk, nodes) if nodes else None)


def nearest_node(nodes: list[tuple[float, float]], x: int, y: int) -> tuple[float, float]:
    """The node whose centre is closest to minimap-crop pixel (x, y)."""
    return min(nodes, key=lambda n: math.hypot(MM_R + n[0] - x, MM_R + n[1] - y))


def detect() -> None:
    """Save detect.png: minimap | wall mask with nodes (magenta) and path (green)."""
    img = screenshot()
    nodes, walk, path = plan(img, [])
    k = 3
    src = Image.fromarray(minimap(img).astype("uint8")).resize((2 * MM_R * k,) * 2, Image.NEAREST)
    m = np.where(walk[..., None], 190, 30).astype("uint8").repeat(3, 2)
    msk = Image.fromarray(m).resize(src.size, Image.NEAREST)
    for im_ in (src, msk):
        d = ImageDraw.Draw(im_)
        for dx, dy in nodes:
            x, y = (MM_R + dx) * k, (MM_R + dy) * k
            d.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(255, 0, 255), width=3)
        if path:
            d.line([(x * k, y * k) for x, y in path], fill=(0, 220, 0), width=3)
    out = Image.new("RGB", (src.width * 2, src.height))
    out.paste(src, (0, 0)); out.paste(msk, (src.width, 0))
    out.save("detect.png")
    print("nodes:", [(round(x), round(y)) for x, y in nodes], "path len:", len(path) if path else None)


STUCK_PX = 2.0         # moved less than this (minimap px) in a step = not making progress
STUCK_STEPS = 3        # steps of little net movement before marking the way ahead as blocked
MAX_DETOURS = 5        # blocked detours toward one node area before skipping it
SKIP_RADIUS = 18       # skipping a node also skips nodes this close to it (minimap px)
SKIP_SECONDS = 180
BLOCK_SECONDS = 300    # how long learned invisible walls are remembered
MAX_STEP_PX = 25       # larger per-frame jumps are treated as tracking errors


class Tracker:
    """Tracks the hero's world position by phase-correlating consecutive minimaps.

    The minimap is north-up and centred on the hero, so the shift between two
    frames equals the hero's movement in minimap pixels.
    """

    def __init__(self) -> None:
        n = 2 * MM_R
        yy, xx = np.mgrid[0:n, 0:n]
        d = np.hypot(xx - MM_R, yy - MM_R)
        self.win = np.clip((MM_R - 6 - d) / 10, 0, 1)  # soft circular window
        self.win[d < PLAYER_R + 3] = 0                  # ignore the hero arrow (it rotates)
        self.prev: np.ndarray | None = None
        self.pos = np.zeros(2)

    def update(self, gray: np.ndarray) -> float:
        """Feed a new minimap (grayscale); returns distance moved since last frame."""
        if self.prev is None:
            self.prev = gray
            return 0.0
        a = (self.prev - self.prev.mean()) * self.win
        b = (gray - gray.mean()) * self.win
        F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
        F /= np.abs(F) + 1e-9
        c = np.fft.ifft2(F).real
        y, x = np.unravel_index(c.argmax(), c.shape)
        n = c.shape[0]
        x = x - n if x > n // 2 else x
        y = y - n if y > n // 2 else y
        self.prev = gray
        if math.hypot(x, y) > MAX_STEP_PX:  # implausible: tracking glitch, ignore
            return 0.0
        self.pos += (x, y)
        return math.hypot(x, y)

    def reset(self) -> None:
        """Forget the previous frame (e.g. after a teleport/respawn)."""
        self.prev = None


def run(minutes: float) -> None:
    """Main loop: path to nearest reachable node, learn invisible walls, revive on death.

    The hero's world position is tracked from minimap motion. When pushing doesn't
    move the hero, the path stretch ahead is stored as a wall in *world* coordinates
    (so it survives the hero sliding around) and we replan around it. After
    MAX_DETOURS failures near one node, that whole area is skipped for SKIP_SECONDS.
    """
    end = time.time() + minutes * 60
    trk = Tracker()
    walls: dict[tuple[int, int], float] = {}           # world cell -> expiry
    skips: list[tuple[float, float, float]] = []       # (world x, world y, expiry)
    detours: dict[tuple[int, int], int] = {}            # coarse target area -> count
    still = 0
    hist: list[np.ndarray] = []
    n = 2 * MM_R
    while time.time() < end:
        img = screenshot()
        if is_dead(img):
            print("died -> revive", flush=True)
            adb("shell", "input", "tap", *map(str, REVIVE))
            time.sleep(3)
            trk = Tracker(); walls.clear(); skips.clear(); detours.clear(); still = 0
            continue

        walk_raw = walkable_mask(img)
        moved = trk.update(walk_raw.astype(float))
        now = time.time()
        px, py = trk.pos
        walls = {k: t for k, t in walls.items() if t > now}
        skips = [s for s in skips if s[2] > now]

        # Project remembered walls / skips into the current hero-relative view
        blocked = np.zeros((n, n), bool)
        for (wx, wy) in walls:
            x, y = int(wx - px + MM_R), int(wy - py + MM_R)
            if 0 <= x < n and 0 <= y < n:
                blocked[y, x] = True
        skip_rel = [(sx - px, sy - py) for sx, sy, _ in skips]

        nodes = [(dx, dy) for dx, dy, _ in find_nodes(img)
                 if not any(math.hypot(dx - sx, dy - sy) < SKIP_RADIUS for sx, sy in skip_rel)]
        walk = walk_raw & ~blocked
        walk[MM_R - 2:MM_R + 3, MM_R - 2:MM_R + 3] = True
        path = bfs(walk, nodes) if nodes else None

        # Stuck = little *net* movement over the last few steps (catches back-and-forth jitter)
        hist.append(trk.pos.copy())
        hist = hist[-(STUCK_STEPS + 1):]
        net = math.hypot(*(hist[-1] - hist[0])) if len(hist) > STUCK_STEPS else 99.0
        still = STUCK_STEPS if net < STUCK_PX * STUCK_STEPS else 0
        if path is not None and still >= STUCK_STEPS:
            tx, ty = nearest_node(nodes, *path[-1])
            key = (int((tx + px) // 15), int((ty + py) // 15))
            detours[key] = detours.get(key, 0) + 1
            if detours[key] > MAX_DETOURS:
                print(f"giving up on area around node ({tx:+.0f},{ty:+.0f}) for {SKIP_SECONDS}s", flush=True)
                skips.append((tx + px, ty + py, now + SKIP_SECONDS))
            else:
                # Remember the stretch ahead as an invisible wall (world coords)
                for x, y in path[2:LOOKAHEAD + 6]:
                    for ddy in range(-2, 3):
                        for ddx in range(-2, 3):
                            walls[(int(x + ddx - MM_R + px), int(y + ddy - MM_R + py))] = now + BLOCK_SECONDS
                print(f"blocked ahead (area detour {detours[key]}); replanning", flush=True)
            still = 0; hist.clear()
            continue  # replan on a fresh frame

        if nodes and path is None:
            # No ground path found: head straight for the nearest node; walls get learned
            tx, ty = min(nodes, key=lambda p: math.hypot(*p))
            path = [(MM_R, MM_R)] + [(int(MM_R + tx * t / 20), int(MM_R + ty * t / 20)) for t in range(1, 21)]
        if not nodes:
            print("no nodes in view; exploring", flush=True)
            # Wander toward the most open direction
            best = max(range(8), key=lambda a: sum(
                walk[int(MM_R + r * math.sin(a * math.pi / 4)), int(MM_R + r * math.cos(a * math.pi / 4))]
                for r in range(PLAYER_R, MM_R - 2, 3)))
            push(math.cos(best * math.pi / 4), math.sin(best * math.pi / 4), 1500)
            continue
        gx, gy = path[-1][0] - MM_R, path[-1][1] - MM_R
        wx, wy = path[min(LOOKAHEAD, len(path) - 1)]
        dx, dy = wx - MM_R, wy - MM_R
        seg = min(len(path), LOOKAHEAD)
        ms = int(min(1000, max(250, seg / PX_PER_SEC * 1000)))
        print(f"pos ({px:+.0f},{py:+.0f}) node ({gx:+},{gy:+}) path {len(path)} "
              f"-> step ({dx:+},{dy:+}) {ms}ms walls {len(walls)}", flush=True)
        push(dx, dy, ms)


if __name__ == "__main__":
    if sys.argv[1] == "detect":
        detect()
    else:
        run(float(sys.argv[2]) if len(sys.argv) > 2 else 20)
