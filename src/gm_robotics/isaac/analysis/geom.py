"""Pure-numpy layout geometry for the Isaac GM nav analysis: forced edit banks, robot-frame bearings, pixel counts."""
import numpy as np

# Mirrors isaac/gm_nav_env.py (that module imports Isaac Lab at top level, so it cannot be imported without it).
ARENA, FLAG_LIM, TOWER_LIM, START_LIM, BOX_LIM = 5.0, 4.2, 4.4, 4.0, 4.4
CAM_X = 0.62
HALF_FOV = np.deg2rad(50.2)          # focal 10, aperture 24 -> 100.4 deg HFOV
GM_PAIR_OFF = 0.65                   # middle of the GM training offset range 0.5-0.8 m
# Behind the east wall, in the 0.6 m gap before the neighbouring arena: occluded by the 2.5 m wall from anywhere inside.
HIDE_FLAG, HIDE_TOWER = np.array([5.5, -2.0]), np.array([5.5, 2.0])
EDITS = ("base", "flag_swap", "tower_swap", "no_flag", "no_tower", "box_move")


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def cam_xy(start, yaw):
    return start + CAM_X * np.stack([np.cos(yaw), np.sin(yaw)], -1)


def robot_frame(p, start, yaw):
    """Point p (arena xy) in the robot base frame (x forward, y left)."""
    d = p - start
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def cam_bearing_dist(p, start, yaw):
    c = cam_xy(start, yaw)
    d = p - c
    return wrap(np.arctan2(d[..., 1], d[..., 0]) - yaw), np.linalg.norm(d, axis=-1)


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def pix_counts(img):
    """Flag-banner (green) and tower (orange) pixels by colour threshold on uint8 RGB; same rule as oracle_gm_nav.py."""
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    flag = (g > 90) & (g > r + 40) & (g > b + 40)
    tower = (r > 150) & (g > 100) & (r - b > 80)
    return flag.sum((-1, -2)), tower.sum((-1, -2))


def _inside(p, lim):
    return (np.abs(p) <= lim).all(-1)


def _beside(p, other, cam):
    """p shifted GM_PAIR_OFF perpendicular to the camera ray, toward the other candidate (stays in view, no occlusion)."""
    ray = p - cam
    perp = np.stack([-ray[..., 1], ray[..., 0]], -1) / np.linalg.norm(ray, axis=-1, keepdims=True)
    side = np.sign(((other - p) * perp).sum(-1, keepdims=True))
    side[side == 0] = 1
    return p + GM_PAIR_OFF * side * perp


def sample_edit_bank(n, seed, kind="visible", k=4096):
    """Conflict-type base scenes (flag u, tower v, both in view) plus the per-edit object positions.

    kind 'visible': flag and tower at 1.2-2.5 m from the camera (the flag banner is >= ~40 px at 2 m, ~1 px at 3 m), so
      the flag stays perceptible after the flag swap; flag-tower separation is then 1.5-3 m, below the 3-5 m of the eval.
    kind 'conflict': flag at 1.2-2.5 m, tower anywhere in view with the eval's 3-5 m flag-tower separation; the flag
      swap then puts the flag where it renders to few pixels.
    kind 'far': flag at 3.5-6 m inside the field of view (renders to ~0 px), tower at 1.2-2.5 m.
    The tower is kept >= 1.5 m off the start->flag segment (the conflict-eval rule) and vice versa >= 1.0 m."""
    rng = np.random.default_rng(seed)
    out = {k_: [] for k_ in ("start", "yaw", "flag", "tower", "flag_moved", "tower_moved", "boxes", "box0_moved",
                             "flag_yaw", "box_yaw")}
    got = 0
    fd = (3.5, 6.0) if kind == "far" else (1.2, 2.5)
    td = (1.2, 7.0) if kind == "conflict" else (1.2, 2.5)
    lim_b = np.deg2rad(42)
    while got < n:
        s = rng.uniform(-START_LIM, START_LIM, (k, 2))
        yaw = rng.uniform(-np.pi, np.pi, k)
        c = cam_xy(s, yaw)
        bf, bt = rng.uniform(-lim_b, lim_b, k), rng.uniform(-lim_b, lim_b, k)
        df, dt = rng.uniform(*fd, k), rng.uniform(*td, k)
        u = c + df[:, None] * np.stack([np.cos(yaw + bf), np.sin(yaw + bf)], -1)
        v = c + dt[:, None] * np.stack([np.cos(yaw + bt), np.sin(yaw + bt)], -1)
        u2, v2 = _beside(v, u, c), _beside(u, v, c)   # flag moved beside the tower; tower moved beside the flag
        ok = _inside(u, FLAG_LIM) & _inside(v, TOWER_LIM) & _inside(u2, FLAG_LIM) & _inside(v2, TOWER_LIM)
        ok &= (seg_dist(v, s, u) >= 1.5) & (seg_dist(u, s, v) >= 1.0)
        if kind == "conflict":
            sep = np.linalg.norm(u - v, axis=-1)
            ok &= (sep >= 3.0) & (sep <= 5.0)
        ok &= (np.linalg.norm(s - v, axis=-1) >= 1.0) & (np.linalg.norm(s - u, axis=-1) >= 1.0)
        idx = np.flatnonzero(ok)
        for i in idx:
            if got >= n:
                break
            objs = [s[i], u[i], v[i], u2[i], v2[i]]
            rays = [(c[i], q) for q in (u[i], v[i], u2[i], v2[i])]
            boxes = []
            # box0 in view at 1.5-4.5 m so the nuisance edit is a visible pixel change of comparable size.
            for j in range(4):
                for _ in range(400):
                    if j in (0, 3):
                        bb, bd = rng.uniform(-lim_b, lim_b), rng.uniform(1.5, 4.5)
                        b = c[i] + bd * np.array([np.cos(yaw[i] + bb), np.sin(yaw[i] + bb)])
                    else:
                        b = rng.uniform(-BOX_LIM, BOX_LIM, 2)
                    if not _inside(b, BOX_LIM):
                        continue
                    if min(np.linalg.norm(b - q) for q in objs + boxes) < 1.2:
                        continue
                    if min(seg_dist(b, a_, q_) for a_, q_ in rays) < 0.7:
                        continue
                    if j == 3 and np.linalg.norm(b - boxes[0]) < 1.0:
                        continue
                    boxes.append(b)
                    break
                else:
                    boxes = None
                    break
            if boxes is None:
                continue
            out["start"].append(s[i]); out["yaw"].append(yaw[i]); out["flag"].append(u[i]); out["tower"].append(v[i])
            out["flag_moved"].append(u2[i]); out["tower_moved"].append(v2[i])
            out["boxes"].append(np.stack(boxes[:3])); out["box0_moved"].append(boxes[3])
            out["flag_yaw"].append(rng.uniform(-np.pi, np.pi)); out["box_yaw"].append(rng.uniform(-np.pi, np.pi, 3))
            got += 1
    return {k_: np.asarray(v_, np.float32) for k_, v_ in out.items()}


def edit_layouts(b):
    """Per-edit full layouts {edit: dict(start, yaw, flag, tower, boxes, flag_yaw, box_yaw)} from an edit bank."""
    n = len(b["start"])
    base = dict(start=b["start"], yaw=b["yaw"], flag=b["flag"], tower=b["tower"], boxes=b["boxes"],
                flag_yaw=b["flag_yaw"], box_yaw=b["box_yaw"])
    out = {"base": base}
    out["flag_swap"] = {**base, "flag": b["flag_moved"]}
    out["tower_swap"] = {**base, "tower": b["tower_moved"]}
    out["no_flag"] = {**base, "flag": np.tile(HIDE_FLAG, (n, 1)).astype(np.float32)}
    out["no_tower"] = {**base, "tower": np.tile(HIDE_TOWER, (n, 1)).astype(np.float32)}
    bx = b["boxes"].copy()
    bx[:, 0] = b["box0_moved"]
    out["box_move"] = {**base, "boxes": bx}
    return out


def probe_targets(lay, flag_px, tower_px, vis_px=5):
    """Robot-frame geometry of flag and tower plus visibility labels for the probe banks."""
    s, yaw = lay["start"], lay["yaw"]
    out = {}
    for name, p, px in (("flag", lay["flag"], flag_px), ("tower", lay["tower"], tower_px)):
        rf = robot_frame(p, s, yaw)
        b, d = cam_bearing_dist(p, s, yaw)
        out[f"{name}_rx"], out[f"{name}_ry"] = rf[:, 0], rf[:, 1]
        out[f"{name}_bearing"], out[f"{name}_dist"] = b, d
        out[f"{name}_px"] = px
        out[f"{name}_visible"] = (px >= vis_px).astype(np.float32)
    return out
