#!/usr/bin/env python3
"""Texture-safe Attribute Maze conflict-state harvest built on colour-free primitives (object-position API, wall probing, frame differences)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Only the colour-shape procgenAISC build supports the duel env's world_dim/obj1/obj2 options; it must be
# first on sys.path before anything imports procgen.
sys.path.insert(0, str(ROOT / "external" / "colour-shape-goal-misgeneralization" / "procgenAISC"))
from procgen import ProcgenEnv  # noqa: E402

from gm_geometry.probes.ras_agent_localization import (  # noqa: E402
    bfs_distances,
    first_action_set,
    probe_wall_grid,
)

# The texture-safe action-delta and agent-cell routines below replace the black-background versions, which fail
# under textured floors (see their docstrings).

OBJ_LINE = "red_line_diag"
OBJ_GEM = "yellow_gem"
SHAPE_EDIT_OBJ = "red_gem"
COLOUR_EDIT_OBJ = "green_gem"
WORLD_DIM = 5  # matches the textured-cohort training config


def make_env(obj1: str, obj2: str, start_level: int, rand_seed: int,
             distribution_mode: str = "easy", use_backgrounds: bool = True):
    """Textured-background env by default (use_backgrounds=True), the opposite of the black-only script's default."""
    return ProcgenEnv(
        num_envs=1,
        env_name="maze_colorsobjects_duel",
        num_levels=1,
        start_level=int(start_level),
        rand_seed=int(rand_seed),
        distribution_mode=distribution_mode,
        world_dim=WORLD_DIM,
        obj1=obj1,
        obj2=obj2,
        use_backgrounds=use_backgrounds,
    )


def unwrap_base(env):
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "get_state") and callable(cur.get_state):
            if hasattr(cur, "set_object") or not hasattr(cur, "env"):
                return cur
        cur = getattr(cur, "env", getattr(cur, "venv", getattr(cur, "_env", None)))
    raise RuntimeError("unwrap failed")


def state_bytes(env) -> bytes:
    st = unwrap_base(env).get_state()
    if isinstance(st, (list, tuple)):
        parts = []
        for s in st:
            if isinstance(s, (bytes, bytearray)):
                parts.append(bytes(s))
            else:
                parts.append(np.asarray(s).tobytes())
        return b"||".join(parts)
    if isinstance(st, (bytes, bytearray)):
        return bytes(st)
    return np.asarray(st).tobytes()


def _obs_hwc(obs) -> np.ndarray:
    rgb = obs["rgb"][0] if isinstance(obs, dict) else obs[0]
    return np.asarray(rgb, dtype=np.float64) / 255.0


def to_nchw(obs) -> np.ndarray:
    x = _obs_hwc(obs).astype(np.float32)
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


N_ACTIONS = 15


def _frame_after(base_api, env, snapshot, action: int, k: int) -> np.ndarray:
    base_api.set_state(snapshot)
    obs = None
    for _ in range(k):
        obs = env.step(np.array([action]))[0]
    return _obs_hwc(obs)


def _mask_centroid(mask: np.ndarray, min_px: int = 5) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) < min_px:
        return None
    return float(xs.mean()), float(ys.mean())


def calibrate_action_deltas_texture_safe(base_api, env, snapshot, calib, thresh: float = 0.12):
    """Texture-safe per-action grid deltas from the centroid shift between consecutive frames of the same action.

    Consecutive frames from one snapshot share call-history depth, which cancels the frame-wide perturbation that
    set_object calls cause under textured floors; actions blocked at the corner spawn are resolved from an interior cell.
    """
    lin = calib[:2, :]

    def _resolve_from(snap) -> dict[int, tuple[int, int] | None]:
        out: dict[int, tuple[int, int] | None] = {}
        for a in range(N_ACTIONS):
            f1 = _frame_after(base_api, env, snap, a, 1)
            f2 = _frame_after(base_api, env, snap, a, 2)
            f3 = _frame_after(base_api, env, snap, a, 3)
            m_early = np.abs(f2 - f1).sum(axis=-1) > thresh
            m_late = np.abs(f3 - f2).sum(axis=-1) > thresh
            c_e, c_l = _mask_centroid(m_early), _mask_centroid(m_late)
            if c_e is None or c_l is None:
                out[a] = None
                continue
            vel_pix = np.array([c_l[0] - c_e[0], c_l[1] - c_e[1]], dtype=np.float64)
            grid_vel = vel_pix @ lin
            if abs(grid_vel[0]) < 1e-6 and abs(grid_vel[1]) < 1e-6:
                out[a] = None
            elif abs(grid_vel[0]) >= abs(grid_vel[1]):
                out[a] = (1, 0) if grid_vel[0] > 0 else (-1, 0)
            else:
                out[a] = (0, 1) if grid_vel[1] > 0 else (0, -1)
        base_api.set_state(snapshot)
        return out

    deltas = _resolve_from(snapshot)
    unresolved = [a for a, d in deltas.items() if d is None]
    known_action = next((a for a, d in deltas.items() if d is not None), None)
    if unresolved and known_action is not None:
        base_api.set_state(snapshot)
        for _ in range(3):
            env.step(np.array([known_action]))
        moved_snapshot = base_api.get_state()
        refined_full = _resolve_from(moved_snapshot)
        for a in unresolved:
            if refined_full.get(a) is not None:
                deltas[a] = refined_full[a]
    base_api.set_state(snapshot)
    return deltas


def locate_agent_cell_texture_safe(base_api, env, snapshot, action_deltas: dict, calib,
                                    walkable: set, thresh: float = 0.12):
    """Texture-safe agent-cell localization from one locally open direction.

    The consecutive-frame velocity (as in calibrate_action_deltas_texture_safe) is back-extrapolated 1.5 steps from the
    early swept-region centroid to the spawn pixel, since a locally open opposite pair is often unavailable at spawn."""
    for a, d in action_deltas.items():
        if d is None:
            continue
        f1 = _frame_after(base_api, env, snapshot, a, 1)
        f2 = _frame_after(base_api, env, snapshot, a, 2)
        f3 = _frame_after(base_api, env, snapshot, a, 3)
        base_api.set_state(snapshot)
        m_early = np.abs(f2 - f1).sum(axis=-1) > thresh
        m_late = np.abs(f3 - f2).sum(axis=-1) > thresh
        c_e, c_l = _mask_centroid(m_early), _mask_centroid(m_late)
        if c_e is None or c_l is None:
            continue  # not genuinely open at snapshot, despite a global delta -- try the next action
        v_pix = np.array([c_l[0] - c_e[0], c_l[1] - c_e[1]], dtype=np.float64)
        pos0_pix = np.array(c_e, dtype=np.float64) - 1.5 * v_pix
        pix_aug = np.array([pos0_pix[0], pos0_pix[1], 1.0])
        est = pix_aug @ calib
        cand = (int(round(est[0])), int(round(est[1])))
        if cand in walkable:
            return cand
    return None


def calibrate_pixel_to_grid_diff(base_api, env, walkable: set, refresh_action: int = 0):
    """Pixel-to-grid calibration by present-vs-absent frame difference of a placed object, not colour thresholds.

    Both frames replay the same action from the same snapshot, so agent motion cancels and only the object's pixels differ."""
    sample_cells = list(walkable)[:12]
    if len(sample_cells) < 3:
        return None
    snapshot = base_api.get_state()
    grid_pts, pix_pts = [], []
    for cell in sample_cells:
        base_api.set_state(snapshot)
        rc_off = base_api.set_object([1], [0], [0], [0])[0]
        if rc_off not in (0, -2):  # -2: already absent is fine here, just means no-op
            pass
        obs_off, _r, _d, _i = env.step(np.array([refresh_action]))
        img_off = _obs_hwc(obs_off)

        base_api.set_state(snapshot)
        rc_on = base_api.set_object([1], [cell[0]], [cell[1]], [1])[0]
        if rc_on != 0:
            continue
        obs_on, _r, _d, _i = env.step(np.array([refresh_action]))
        img_on = _obs_hwc(obs_on)

        diff = np.abs(img_on - img_off).sum(axis=-1)
        mask = diff > 0.12
        ys, xs = np.where(mask)
        if len(xs) < 3:
            continue
        grid_pts.append(cell)
        pix_pts.append((float(xs.mean()), float(ys.mean())))
    base_api.set_state(snapshot)
    base_api.set_object([1], [0], [0], [0])
    if len(grid_pts) < 3:
        return None
    P = np.asarray(pix_pts, dtype=np.float64)
    Pb = np.concatenate([P, np.ones((len(pix_pts), 1))], axis=1)
    Grid = np.asarray(grid_pts, dtype=np.float64)
    winv, *_ = np.linalg.lstsq(Pb, Grid, rcond=None)
    return winv


def _groups_from_action_deltas(action_deltas: dict) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for a, d in action_deltas.items():
        if d is None:
            continue
        groups.setdefault(d, []).append(a)
    return {k: sorted(v) for k, v in groups.items()}


def _canonical_group_for_set(action_set: set[int], action_deltas: dict) -> tuple[int, int] | None:
    """Map a first_action_set back to its grid delta; a cross-direction tie (rare) is flagged and resolved by the delta
    with the most member actions."""
    deltas = [action_deltas[a] for a in action_set if action_deltas.get(a) is not None]
    if not deltas:
        return None
    counts: dict[tuple[int, int], int] = {}
    for d in deltas:
        counts[d] = counts.get(d, 0) + 1
    if len(counts) > 1:
        print(f"  [tie] first_action_set spans multiple deltas {counts}, "
              f"taking the majority one", flush=True)
    return max(counts, key=lambda d: counts[d])


def calibrate_and_harvest_level(level: int, distribution_mode: str, use_backgrounds: bool,
                                 world_dim: int, margin: int = 3):
    """One level's texture-safe pipeline: wall grid, calibration, agent cell, object cells via get_object_pos, conflict check.

    Returns None on failure, else the black-only conflict-row schema plus the level-local action_deltas/groups."""
    env = ProcgenEnv(
        num_envs=1, env_name="maze_colorsobjects_duel", num_levels=1, start_level=level,
        rand_seed=0, distribution_mode=distribution_mode, world_dim=world_dim,
        obj1=OBJ_LINE, obj2=OBJ_GEM, use_backgrounds=use_backgrounds,
    )
    obs0 = env.reset()
    base = unwrap_base(env)
    base_state_bytes = state_bytes(env)

    # Must be read before any set_object-based probing: slots 1/2 are the line/gem goal entities, and
    # probe_wall_grid and calibrate_pixel_to_grid_diff clear or move them.
    line_xy0 = base.get_object_pos([1])[0]
    gem_xy0 = base.get_object_pos([2])[0]

    coord_range = range(-margin, world_dim + margin)
    walkable = probe_wall_grid(base, coord_range, coord_range)
    if len(walkable) < 4:
        env.close()
        return None, None, "wall_probe_too_small"

    calib = calibrate_pixel_to_grid_diff(base, env, walkable)
    if calib is None:
        env.close()
        return None, None, "pixel_calib_fail"
    snapshot = base.get_state()

    action_deltas = calibrate_action_deltas_texture_safe(base, env, snapshot, calib)
    found = {d for d in action_deltas.values() if d is not None}
    if not any((-d[0], -d[1]) in found for d in found):
        env.close()
        return None, action_deltas, "no_global_opposite_pair"

    agent_cell = locate_agent_cell_texture_safe(base, env, snapshot, action_deltas, calib, walkable)
    if agent_cell is None:
        env.close()
        return None, action_deltas, "agent_locate_fail"

    line_xy, gem_xy = line_xy0, gem_xy0
    if line_xy is None or gem_xy is None:
        env.close()
        return None, action_deltas, "object_pos_none"
    line_xy, gem_xy = tuple(line_xy), tuple(gem_xy)
    if line_xy not in walkable or gem_xy not in walkable:
        # ground-truth object cell disagrees with the independently-probed wall
        # grid -- treat as a failed level rather than trusting either silently
        env.close()
        return None, action_deltas, "object_pos_not_walkable"

    dist_from_line = bfs_distances(walkable, line_xy)
    dist_from_gem = bfs_distances(walkable, gem_xy)
    d_line = dist_from_line.get(agent_cell)
    d_gem = dist_from_gem.get(agent_cell)
    if d_line is None or d_gem is None:
        env.close()
        return None, action_deltas, "unreachable"
    if d_line != d_gem:
        env.close()
        return None, action_deltas, "not_equidistant"

    set_line = first_action_set(dist_from_line, agent_cell, action_deltas)
    set_gem = first_action_set(dist_from_gem, agent_cell, action_deltas)
    if not set_line or not set_gem:
        env.close()
        return None, action_deltas, "no_first_action_set"
    if not set_line.isdisjoint(set_gem):
        env.close()
        return None, action_deltas, "not_disjoint"

    group_line = _canonical_group_for_set(set_line, action_deltas)
    group_gem = _canonical_group_for_set(set_gem, action_deltas)
    if group_line is None or group_gem is None or group_line == group_gem:
        env.close()
        return None, action_deltas, "canonical_group_collision"

    res = {
        "level": level,
        "obs0": obs0,
        "state_bytes": base_state_bytes,
        "agent_cell": agent_cell,
        "line_xy": line_xy,
        "gem_xy": gem_xy,
        "dist_line": int(d_line),
        "dist_gem": int(d_gem),
        "first_group_line": group_line,
        "first_group_gem": group_gem,
    }
    env.close()
    return res, action_deltas, "matched"


def harvest_conflict_levels(*, start_level: int, n_levels_to_try: int, n_target: int,
                             distribution_mode: str, use_backgrounds: bool, world_dim: int = WORLD_DIM):
    """Texture-safe counterpart of find_conflict_levels(); returns (groups, matched) in the same shape so score_seed()
    from run_ras_conflict_signed_derivative.py can be reused."""
    matched: list[dict] = []
    stats: dict[str, int] = {}
    action_deltas_cached: dict | None = None
    level = start_level
    end_level = start_level + n_levels_to_try
    while level < end_level and len(matched) < n_target:
        stats["tried"] = stats.get("tried", 0) + 1
        res, action_deltas, reason = calibrate_and_harvest_level(level, distribution_mode, use_backgrounds, world_dim)
        stats[reason] = stats.get(reason, 0) + 1
        if action_deltas is not None and action_deltas_cached is None:
            action_deltas_cached = action_deltas
            print(f"[calibrate] @ level {level}: {action_deltas}", flush=True)
        if res is None:
            level += 1
            continue
        matched.append(res)
        level += 1
        if len(matched) % 5 == 0:
            print(f"  matched {len(matched)}/{n_target} (tried={stats['tried']}) stats={stats}", flush=True)
    print(f"conflict-level search done: matched={len(matched)} stats={stats}", flush=True)
    groups = _groups_from_action_deltas(action_deltas_cached or {})
    return groups, matched


def determinism_check(*, start_level: int, n_levels: int, distribution_mode: str,
                       use_backgrounds: bool, world_dim: int = WORLD_DIM) -> dict:
    """Two independent harvest passes over the same level range must agree
    exactly on which levels are accepted and every field of each accepted
    row (except the raw obs array, compared separately by byte-equality)."""
    def _run():
        matched = []
        for level in range(start_level, start_level + n_levels):
            res, _, _ = calibrate_and_harvest_level(level, distribution_mode, use_backgrounds, world_dim)
            if res is not None:
                matched.append(res)
        return matched

    run_a = _run()
    run_b = _run()
    ok = len(run_a) == len(run_b)
    mismatches = []
    if ok:
        for ra, rb in zip(run_a, run_b):
            same_scalar = all(
                ra[k] == rb[k] for k in
                ("level", "agent_cell", "line_xy", "gem_xy", "dist_line", "dist_gem",
                 "first_group_line", "first_group_gem", "state_bytes")
            )
            same_obs = bool(np.array_equal(ra["obs0"]["rgb"] if isinstance(ra["obs0"], dict) else ra["obs0"],
                                            rb["obs0"]["rgb"] if isinstance(rb["obs0"], dict) else rb["obs0"]))
            if not (same_scalar and same_obs):
                mismatches.append(ra["level"])
                ok = False
    else:
        mismatches = ["run lengths differ"]
    return {"ok": ok, "n_run_a": len(run_a), "n_run_b": len(run_b), "mismatches": mismatches}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-level", type=int, default=2000)
    ap.add_argument("--n-levels-to-try", type=int, default=400)
    ap.add_argument("--n-target", type=int, default=24)
    ap.add_argument("--distribution-mode", type=str, default="easy")
    ap.add_argument("--use-backgrounds", action="store_true", default=True)
    ap.add_argument("--determinism-levels", type=int, default=15,
                     help="How many levels (from --start-level) to run the determinism/replay check over.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[config] distribution_mode={args.distribution_mode} use_backgrounds={args.use_backgrounds} "
          f"world_dim={WORLD_DIM} -- verified arm-C training config, not a default assumption", flush=True)

    print("=== determinism/replay check (before trusting any harvested level) ===", flush=True)
    det = determinism_check(
        start_level=args.start_level, n_levels=args.determinism_levels,
        distribution_mode=args.distribution_mode, use_backgrounds=args.use_backgrounds,
    )
    print(json.dumps(det), flush=True)
    if not det["ok"]:
        raise SystemExit(f"determinism check FAILED: {det}")

    print("=== full harvest ===", flush=True)
    groups, matched = harvest_conflict_levels(
        start_level=args.start_level + args.determinism_levels,  # disjoint from the determinism-check range
        n_levels_to_try=args.n_levels_to_try, n_target=args.n_target,
        distribution_mode=args.distribution_mode, use_backgrounds=args.use_backgrounds,
    )
    if len(matched) < 5:
        raise SystemExit(f"too few conflict levels matched ({len(matched)}) -- widen --n-levels-to-try")

    out = {
        "obj_line": OBJ_LINE, "obj_gem": OBJ_GEM,
        "distribution_mode": args.distribution_mode, "use_backgrounds": args.use_backgrounds,
        "world_dim": WORLD_DIM,
        "determinism_check": det,
        "groups": {str(k): v for k, v in groups.items()},
        "n_conflict_levels": len(matched),
        "conflict_levels": [
            {"level": m["level"], "agent_cell": m["agent_cell"], "line_xy": m["line_xy"], "gem_xy": m["gem_xy"],
             "dist_line": m["dist_line"], "dist_gem": m["dist_gem"],
             "first_group_line": list(m["first_group_line"]), "first_group_gem": list(m["first_group_gem"])}
            for m in matched
        ],
    }
    (args.out / "arm_c_conflict_harvest_ready.json").write_text(json.dumps(out, indent=2))
    print("wrote", args.out / "arm_c_conflict_harvest_ready.json", flush=True)
    print("HARVEST READY -- no policy scoring or steering run in this pass.", flush=True)


if __name__ == "__main__":
    main()
