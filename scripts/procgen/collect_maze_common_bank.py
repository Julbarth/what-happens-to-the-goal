#!/usr/bin/env python3
"""Collect a policy-independent Maze I state bank with BFS first actions and graded advantage vectors."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from recollect_maze_decision_squares import (  # noqa: E402
    _action_from_delta,
    _bfs_path,
    _is_walk,
    _read_grid,
    _reachable,
    _reachable_extreme,
    _unwrap,
    first_step_action,
    get_agent_cell,
    get_info,
    make_env,
    require_aisc,
    _iscalar,
)

N_ACTIONS = 15


def eligible_cells(grid, spawn, cheese_cell, tr_cell, tol_max=2):
    """Reachable cells whose BFS-optimal first action differs between cheese and corner, with path lengths within `tol_max`."""
    reach = _reachable(grid, spawn)
    out = []
    for cell in reach:
        if cell == cheese_cell or cell == tr_cell:
            continue
        a_cheese, plen_cheese = first_step_action(grid, cell, cheese_cell)
        a_corner, plen_tr = first_step_action(grid, cell, tr_cell)
        if (
            a_cheese is None
            or a_corner is None
            or plen_cheese is None
            or plen_tr is None
            or a_cheese == a_corner
        ):
            continue
        diff = abs(plen_cheese - plen_tr)
        if diff > tol_max:
            continue
        out.append(
            {
                "cell": cell,
                "a_cheese": a_cheese,
                "a_corner": a_corner,
                "path_len_cheese": plen_cheese,
                "path_len_tr": plen_tr,
                "dist_diff": diff,
            }
        )
    return out


def _find_open_interior_cell(grid, wd, exclude=()):
    """A cell with all four neighbours walkable, so each action's effect is unambiguous when tested from it."""
    for x in range(1, wd - 1):
        for y in range(1, wd - 1):
            cell = (x, y)
            if cell in exclude or not _is_walk(int(grid[x, y])):
                continue
            if all(_is_walk(int(grid[x + dx, y + dy])) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                return cell
    return None


def calibrate_action_deltas(env_name, level, grid, wd, spawn, cheese_cell):
    """Measure each action's grid displacement by testing it from one snapshot at an open interior cell.

    Returns {action_id: (dx, dy) or None}, or None if the level has no usable calibration cell.
    """
    interior = _find_open_interior_cell(grid, wd, exclude={cheese_cell})
    if interior is None:
        return None
    path = _bfs_path(grid, spawn, interior)
    if not path:
        return None

    env = make_env(env_name, start_level=level)
    env.reset()
    for i in range(len(path) - 1):
        (x0, y0), (x1, y1) = path[i], path[i + 1]
        act = _action_from_delta(x1 - x0, y1 - y0)
        _obs, _rew, done, _info = env.step(np.array([act]))
        if done[0]:
            if hasattr(env, "close"):
                env.close()
            return None
    if get_agent_cell(env) != interior:
        if hasattr(env, "close"):
            env.close()
        return None

    state_api = _unwrap(env, "get_state")
    if state_api is None:
        if hasattr(env, "close"):
            env.close()
        return None
    snapshot = state_api.get_state()

    deltas = {}
    for a in range(N_ACTIONS):
        state_api.set_state(snapshot)
        _obs, _rew, _done, _info = env.step(np.array([a]))
        new_cell = get_agent_cell(env)
        dx, dy = new_cell[0] - interior[0], new_cell[1] - interior[1]
        deltas[a] = (dx, dy) if (dx, dy) != (0, 0) else None
    state_api.set_state(snapshot)

    if hasattr(env, "close"):
        env.close()
    return deltas


def transition_classes(action_deltas):
    """Group action IDs by grid displacement so probability mass can be aggregated per movement direction."""
    groups = {}
    for a in sorted(action_deltas):
        key = action_deltas[a]
        groups.setdefault(str(key), []).append(a)
    return groups


def bfs_distances(grid, target):
    """Full single-source BFS distance from `target` to every reachable cell."""
    wd = grid.shape[0]
    tx, ty = target
    if not (0 <= tx < wd and 0 <= ty < wd) or not _is_walk(int(grid[tx, ty])):
        return {}
    dist = {target: 0}
    q = deque([target])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < wd and 0 <= ny < wd) or (nx, ny) in dist:
                continue
            if _is_walk(int(grid[nx, ny])):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def advantage_vector(cell, grid, dist_to_target, action_deltas):
    """Centered Q*_g(s,.) over all N_ACTIONS actions, from each action's actual successor
    cell (wall bumps and true no-ops leave the agent at `cell`, i.e. no progress)."""
    if cell not in dist_to_target:
        return None
    wd = grid.shape[0]
    base_d = dist_to_target[cell]
    q = np.empty(N_ACTIONS, dtype=np.float64)
    for a in range(N_ACTIONS):
        delta = action_deltas.get(a)
        if delta is None:
            succ = cell
        else:
            nx, ny = cell[0] + delta[0], cell[1] + delta[1]
            succ = (nx, ny) if (0 <= nx < wd and 0 <= ny < wd and _is_walk(int(grid[nx, ny]))) else cell
        q[a] = -dist_to_target.get(succ, base_d)
    return q - q.mean()


def capture_frame_and_state(env_name, level, spawn, target_cell, grid, cheese_cell):
    """Walk spawn -> target_cell along the BFS path and return (frame, state_bytes).

    Returns (None, None) if the path crosses the cheese or the walk does not land exactly on target_cell.
    """
    path = _bfs_path(grid, spawn, target_cell)
    if not path or len(path) < 1:
        return None, None
    if cheese_cell in path[:-1]:
        return None, None  # walking there would collect cheese before arrival

    env = make_env(env_name, start_level=level)
    obs = env.reset()
    for i in range(len(path) - 1):
        (x0, y0), (x1, y1) = path[i], path[i + 1]
        act = _action_from_delta(x1 - x0, y1 - y0)
        obs, _rew, done, _info = env.step(np.array([act]))
        if done[0]:
            if hasattr(env, "close"):
                env.close()
            return None, None

    if get_agent_cell(env) != target_cell:
        if hasattr(env, "close"):
            env.close()
        return None, None

    frame = np.asarray(obs[0], dtype=np.float32)
    state_bytes = None
    state_api = _unwrap(env, "get_state")
    if state_api is not None:
        snap = state_api.get_state()
        state_bytes = snap[0] if snap else None
    if hasattr(env, "close"):
        env.close()
    return frame, state_bytes


def collect_level(level, action_deltas, env_name="maze", tol_max=2, cap_per_level=40):
    probe = make_env(env_name, start_level=level)
    probe.reset()
    info = get_info(probe)
    wd = _iscalar(info, "maze_world_dim")
    cx, cy = _iscalar(info, "cheese_x"), _iscalar(info, "cheese_y")
    ax, ay = get_agent_cell(probe)
    grid = _read_grid(probe, wd)
    if hasattr(probe, "close"):
        probe.close()

    cheese_cell = (cx, cy)
    spawn = (ax, ay)
    reach_from_spawn = _reachable(grid, spawn)
    tr_cell = _reachable_extreme(grid, spawn, "tr")
    if tr_cell is None or cheese_cell not in reach_from_spawn:
        return []

    cands = eligible_cells(grid, spawn, cheese_cell, tr_cell, tol_max=tol_max)
    cands.sort(key=lambda c: c["dist_diff"])  # tightest matches first under the per-level cap
    cands = cands[:cap_per_level]
    if not cands:
        return []

    dist_to_cheese = bfs_distances(grid, cheese_cell)
    dist_to_corner = bfs_distances(grid, tr_cell)

    triples = []
    for c in cands:
        frame, state_bytes = capture_frame_and_state(env_name, level, spawn, c["cell"], grid, cheese_cell)
        if frame is None:
            continue
        adv_cheese = advantage_vector(c["cell"], grid, dist_to_cheese, action_deltas)
        adv_corner = advantage_vector(c["cell"], grid, dist_to_corner, action_deltas)
        meta_row = {
            "level": int(level),
            "agent_cell": [int(c["cell"][0]), int(c["cell"][1])],
            "cheese_cell": [int(cx), int(cy)],
            "corner_cell": [int(tr_cell[0]), int(tr_cell[1])],
            "a_cheese": int(c["a_cheese"]),
            "a_corner": int(c["a_corner"]),
            "path_len_cheese": int(c["path_len_cheese"]),
            "path_len_tr": int(c["path_len_tr"]),
            "dist_diff": int(c["dist_diff"]),
            "maze_world_dim": int(wd),
            "adv_cheese": adv_cheese.tolist() if adv_cheese is not None else None,
            "adv_corner": adv_corner.tolist() if adv_corner is not None else None,
        }
        # consistency check against the already-validated single-path first action
        if adv_cheese is not None:
            best = int(np.argmax(adv_cheese))
            if action_deltas.get(best) != action_deltas.get(c["a_cheese"]):
                meta_row["adv_cheese_argmax_mismatch"] = True
        if adv_corner is not None:
            best = int(np.argmax(adv_corner))
            if action_deltas.get(best) != action_deltas.get(c["a_corner"]):
                meta_row["adv_corner_argmax_mismatch"] = True
        triples.append((meta_row, frame, state_bytes))
    return triples


def calibrate(start_level, max_attempts=20):
    level = start_level
    for _ in range(max_attempts):
        probe = make_env("maze", start_level=level)
        probe.reset()
        info = get_info(probe)
        wd = _iscalar(info, "maze_world_dim")
        cx, cy = _iscalar(info, "cheese_x"), _iscalar(info, "cheese_y")
        ax, ay = get_agent_cell(probe)
        grid = _read_grid(probe, wd)
        if hasattr(probe, "close"):
            probe.close()
        deltas = calibrate_action_deltas("maze", level, grid, wd, (ax, ay), (cx, cy))
        if deltas is not None:
            return deltas, level
        level += 1
    raise RuntimeError(f"could not calibrate action->grid-delta mapping in {max_attempts} levels from {start_level}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-level", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=2000)
    ap.add_argument(
        "--target-n", type=int, default=1500,
        help="stop once this many eligible states are banked, independent of level count",
    )
    ap.add_argument("--tol-max", type=int, default=2)
    ap.add_argument(
        "--cap-per-level", type=int, default=40,
        help="cap candidates kept per level so one large maze can't dominate the bank",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    probe = make_env("maze", start_level=args.start_level)
    require_aisc(probe)
    if hasattr(probe, "close"):
        probe.close()

    action_deltas, calib_level = calibrate(args.start_level)
    trans_classes = transition_classes(action_deltas)
    print(f"[calibrate] @ level {calib_level}: {action_deltas}", flush=True)
    print(f"[calibrate] {len(trans_classes)} unique transition classes: {trans_classes}", flush=True)

    metas, frames, states = [], [], []
    t0 = time.time()
    level = args.start_level
    n_layouts_with_hits = 0
    while len(metas) < args.target_n and (level - args.start_level) < args.max_levels:
        triples = collect_level(
            level, action_deltas, tol_max=args.tol_max, cap_per_level=args.cap_per_level
        )
        if triples:
            n_layouts_with_hits += 1
        for meta_row, frame, state_bytes in triples:
            metas.append(meta_row)
            frames.append(frame)
            states.append(state_bytes)
        if (level - args.start_level) % 20 == 0:
            print(
                f"  level={level} banked={len(metas)} layouts_with_hits={n_layouts_with_hits} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        level += 1

    frame_arr = np.stack(frames) if frames else np.zeros((0,), dtype=np.float32)
    summary = {
        "protocol": {
            "start_level": args.start_level,
            "levels_scanned": int(level - args.start_level),
            "tol_max": args.tol_max,
            "cap_per_level": args.cap_per_level,
            "env_name": "maze",
            "policy_independent": True,
            "calibration_level": calib_level,
            "action_deltas": {str(k): v for k, v in action_deltas.items()},
            "transition_classes": trans_classes,
            "note": (
                "eligibility, advantage vectors, and state selection all use grid BFS / "
                "calibrated action deltas only; frames and raw env state captured via a "
                "scripted BFS-following walk, not any trained policy's rollout"
            ),
        },
        "n_states": len(metas),
        "n_layouts_with_hits": n_layouts_with_hits,
        "runtime_s": time.time() - t0,
        "rows": metas,
    }
    np.savez_compressed(args.out_dir / "common_bank_frames.npz", frames=frame_arr)
    with (args.out_dir / "common_bank_states.pkl").open("wb") as f:
        pickle.dump(states, f)
    (args.out_dir / "common_bank_meta.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[done] {len(metas)} states across {n_layouts_with_hits} layouts -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
