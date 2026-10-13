"""Ground-truth-calibrated agent and target localization for black-background Attribute Maze (maze_colorsobjects_duel)."""
# Walls come from set_object return codes, the pixel->grid map from objects placed at known cells, and the agent
# cell from intersecting the motion masks of a move and its opposite; the agent sprite is never colour-matched.
# Texture backgrounds break the colour-based object localizer used for calibration, so this is black-only.
from __future__ import annotations

from collections import deque

import numpy as np


def bfs_distances(walkable: set, target: tuple[int, int]) -> dict:
    """Full single-source BFS distance from `target` over the empirically-probed grid."""
    if target not in walkable:
        return {}
    dist = {target: 0}
    q = deque([target])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n in walkable and n not in dist:
                dist[n] = dist[(x, y)] + 1
                q.append(n)
    return dist


def first_action_set(dist_from_target: dict, cell: tuple[int, int], action_deltas: dict) -> set[int] | None:
    """All actions whose successor cell strictly decreases distance-to-target (more than one on a tie)."""
    if cell not in dist_from_target:
        return None
    base = dist_from_target[cell]
    best = None
    per_action = {}
    for a, delta in action_deltas.items():
        if delta is None:
            continue
        succ = (cell[0] + delta[0], cell[1] + delta[1])
        d = dist_from_target.get(succ)
        if d is None:
            continue
        per_action[a] = d
        if best is None or d < best:
            best = d
    if best is None or best >= base:
        return None  # no action makes progress (shouldn't happen if cell != target and reachable)
    return {a for a, d in per_action.items() if d == best}


def _obs_hwc(obs) -> np.ndarray:
    rgb = obs["rgb"][0] if isinstance(obs, dict) else obs[0]
    return np.asarray(rgb, dtype=np.float64) / 255.0


def _motion_mask(img_ref: np.ndarray, img_cur: np.ndarray, thresh: float = 0.12) -> np.ndarray:
    return np.abs(img_cur - img_ref).sum(axis=-1) > thresh


def _mask_centroid(mask: np.ndarray, min_px: int = 1) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) < min_px:
        return None
    return float(xs.mean()), float(ys.mean())


def probe_wall_grid(base_api, x_range, y_range) -> set[tuple[int, int]]:
    """Empirically reconstruct the walkable cell set via set_object return codes."""
    base_api.set_object([1], [0], [0], [0])  # clear both objects first
    base_api.set_object([2], [0], [0], [0])
    walkable = set()
    for x in x_range:
        for y in y_range:
            rc = base_api.set_object([1], [x], [y], [1])[0]
            if rc == 0:  # WALKABLE_RC
                walkable.add((x, y))
    base_api.set_object([1], [0], [0], [0])  # leave clean
    return walkable


def calibrate_pixel_to_grid(base_api, env, walkable: set, obj1: str, obj2: str, refresh_action: int = 0):
    """Fit an affine pixel(+bias) -> grid map by placing object 1 at known cells and reading back its pixel centroid.

    The env has no ``.observe()``, so one ``.step()`` renders each sample and the state is restored afterwards.
    Returns (affine matrix, last observed frame) or (None, None)."""
    from gm_geometry.probes.choice_point import localize_dual_objects

    sample_cells = list(walkable)[:12]
    if len(sample_cells) < 3:
        return None, None
    snapshot = base_api.get_state()
    grid_pts, pix_pts = [], []
    last_obs = None
    for cell in sample_cells:
        rc = base_api.set_object([1], [cell[0]], [cell[1]], [1])[0]
        if rc != 0:
            continue
        obs, _rew, _done, _info = env.step(np.array([refresh_action]))
        base_api.set_state(snapshot)  # undo any agent movement from that step immediately
        last_obs = obs
        rgb = obs["rgb"][0] if isinstance(obs, dict) else obs[0]
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float64) / 255.0
        loc = localize_dual_objects(chw, obj1=obj1, obj2=obj2)
        xy = loc.get("gem_xy") or loc.get("line_xy") or loc.get("yellow_xy") or loc.get("red_xy")
        if xy is None:
            continue
        grid_pts.append(cell)
        pix_pts.append(xy)
    base_api.set_object([1], [0], [0], [0])
    if len(grid_pts) < 3:
        return None, None
    P = np.asarray(pix_pts, dtype=np.float64)
    Pb = np.concatenate([P, np.ones((len(pix_pts), 1))], axis=1)
    Grid = np.asarray(grid_pts, dtype=np.float64)
    Winv, *_ = np.linalg.lstsq(Pb, Grid, rcond=None)
    return Winv, last_obs  # pixel(+bias) -> grid


def _calibrate_actions_from(base_api, env, img_ref, snapshot, lin, k_probe, actions) -> dict:
    """Test a subset of actions for grid displacement from one fixed reference frame/snapshot."""
    k_lo, k_hi = k_probe

    def bbox_after(a: int, k: int):
        base_api.set_state(snapshot)
        obs = None
        for _ in range(k):
            obs = env.step(np.array([a]))[0]
        mask = _motion_mask(img_ref, _obs_hwc(obs))
        if not mask.any():
            return None
        ys, xs = np.where(mask)
        return xs.min(), xs.max(), ys.min(), ys.max()

    out: dict[int, tuple[int, int] | None] = {}
    for a in actions:
        b_lo, b_hi = bbox_after(a, k_lo), bbox_after(a, k_hi)
        if b_lo is None or b_hi is None:
            out[a] = None
            continue
        xlo0, xlo1, ylo0, ylo1 = b_lo
        xhi0, xhi1, yhi0, yhi1 = b_hi
        growth = {"right": xhi1 - xlo1, "left": xlo0 - xhi0, "down": yhi1 - ylo1, "up": ylo0 - yhi0}
        best_dir = max(growth, key=lambda k: growth[k])
        if growth[best_dir] <= 0:
            out[a] = None
            continue
        dpix = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}[best_dir]
        grid_delta = np.asarray(dpix, dtype=np.float64) @ lin
        if abs(grid_delta[0]) < 1e-6 and abs(grid_delta[1]) < 1e-6:
            out[a] = None
        elif abs(grid_delta[0]) >= abs(grid_delta[1]):
            out[a] = (1, 0) if grid_delta[0] > 0 else (-1, 0)
        else:
            out[a] = (0, 1) if grid_delta[1] > 0 else (0, -1)
    base_api.set_state(snapshot)
    return out


def calibrate_action_deltas(base_api, env, obs0, calib: np.ndarray, k_probe: tuple[int, int] = (2, 6)) -> dict:
    """Measure each action's grid displacement from bounding-box growth at two repeat counts.

    The agent spawns in a corner, so after testing from spawn it moves to an interior cell and retests the
    still-unresolved actions; action semantics are env-wide."""
    snapshot = base_api.get_state()
    img0 = _obs_hwc(obs0)
    lin = calib[:2, :]  # (2,2): pixel-delta -> grid-delta, dropping calib's bias row

    deltas = _calibrate_actions_from(base_api, env, img0, snapshot, lin, k_probe, range(15))

    unresolved = [a for a, d in deltas.items() if d is None]
    known_action = next((a for a, d in deltas.items() if d is not None), None)
    if unresolved and known_action is not None:
        base_api.set_state(snapshot)
        moved_obs = None
        for _ in range(3):
            moved_obs = env.step(np.array([known_action]))[0]
        moved_snapshot = base_api.get_state()
        refined = _calibrate_actions_from(
            base_api, env, _obs_hwc(moved_obs), moved_snapshot, lin, k_probe, unresolved
        )
        for a, d in refined.items():
            if d is not None:
                deltas[a] = d

    base_api.set_state(snapshot)
    return deltas


def locate_agent_cell(base_api, env, obs0, action_deltas: dict, calib: np.ndarray, walkable: set):
    """Per-level agent grid cell.

    A move and its exact opposite from the same snapshot agree on the vacated start cell and disagree on the new
    one, so intersecting their motion masks isolates the start footprint. Every opposite pair is tried in turn."""
    snapshot = base_api.get_state()
    img0 = _obs_hwc(obs0)

    by_delta: dict[tuple[int, int], int] = {}
    for a, d in action_deltas.items():
        if d is not None and d not in by_delta:  # first (lowest-index) action per direction
            by_delta[d] = a

    for d, a_pos in by_delta.items():
        a_neg = by_delta.get((-d[0], -d[1]))
        if a_neg is None:
            continue
        base_api.set_state(snapshot)
        mask_pos = _motion_mask(img0, _obs_hwc(env.step(np.array([a_pos]))[0]))
        base_api.set_state(snapshot)
        mask_neg = _motion_mask(img0, _obs_hwc(env.step(np.array([a_neg]))[0]))
        base_api.set_state(snapshot)
        old_xy = _mask_centroid(mask_pos & mask_neg)
        if old_xy is None:
            continue
        pix_aug = np.array([old_xy[0], old_xy[1], 1.0])
        est = pix_aug @ calib
        cand = (int(round(est[0])), int(round(est[1])))
        if cand in walkable:
            return cand
    return None
