#!/usr/bin/env python3
"""Persist Maze I graph-exact decision-square geometry rows (positions, path lengths, first actions, outcome)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def _add_paths(root: Path) -> None:
    for p in (
        root / "src",
        root / "external" / "procgenAISC",
        root / "vendored" / "train-procgen-pytorch",
        root / "external" / "train-procgen-pytorch",
        root / "run_scripts",
        root / "scripts" / "procgen",
    ):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)


_add_paths(ROOT)

from common.env.procgen_wrappers import (  # noqa: E402
    ScaledFloatFrame,
    TransposeFrame,
    VecExtractDictObs,
    VecNormalize,
)
from procgen import ProcgenEnv  # noqa: E402

from gm_geometry.integrations.colour_shape import load_impala_policy_from_env  # noqa: E402
from gm_geometry.procgen import unwrap_base as _unwrap_base  # noqa: E402
from gm_geometry.utils import device as default_device  # noqa: E402
from validate_langosco_maze_i_200m import (  # noqa: E402
    chw_to_hwc01 as _chw_to_hwc01,
    in_top_right,
    motion_agent_xy,
)

# procgen cardinals: LEFT=1 DOWN=3 UP=5 RIGHT=7
A_LEFT, A_DOWN, A_UP, A_RIGHT = 1, 3, 5, 7
WALL_OBJ, SPACE_OBJ, GOAL_OBJ = 51, 100, 2


def make_env(env_name: str, *, start_level: int, num_threads: int = 2):
    venv = ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=1,
        start_level=start_level,
        distribution_mode="hard",
        num_threads=num_threads,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False, ret=False)
    venv = TransposeFrame(venv)
    return ScaledFloatFrame(venv)


def load_policy(checkpoint: Path, device: str):
    train_root = None
    for cand in (
        ROOT / "vendored" / "train-procgen-pytorch",
        ROOT / "external" / "train-procgen-pytorch",
    ):
        if (cand / "hyperparams/procgen/config.yml").exists():
            train_root = cand
            break
    if train_root is None:
        raise FileNotFoundError("train-procgen-pytorch hyperparams not found")
    return load_impala_policy_from_env(checkpoint, "maze", device, train_root=train_root)


def _unwrap(env, attr):
    return _unwrap_base(env, attr, on_missing="none")


def get_info(env):
    base = _unwrap(env, "get_info")
    infos = base.get_info()
    return infos[0] if isinstance(infos, (list, tuple)) else infos


def _iscalar(info, key):
    return int(np.asarray(info[key]).reshape(-1)[0])


def get_maze_obj(env, x, y):
    return int(_unwrap(env, "get_maze_obj").get_maze_obj([x], [y])[0])


def get_agent_cell(env):
    return tuple(_unwrap(env, "get_agent_cell").get_agent_cell()[0])


def _is_walk(t):
    return t == SPACE_OBJ or t == GOAL_OBJ


def _read_grid(env, wd):
    return np.array([[get_maze_obj(env, x, y) for y in range(wd)] for x in range(wd)], dtype=np.int32)


def _reachable(grid, start):
    wd = grid.shape[0]
    q = deque([start])
    seen = {start}
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < wd and 0 <= ny < wd) or (nx, ny) in seen:
                continue
            if _is_walk(int(grid[nx, ny])):
                seen.add((nx, ny))
                q.append((nx, ny))
    return seen


def _reachable_extreme(grid, start, which="tr"):
    reach = _reachable(grid, start)
    if not reach:
        return None
    if which == "tr":
        return max(reach, key=lambda c: (c[0] + c[1], c[0], c[1]))
    raise ValueError(which)


def _bfs_path(grid, start, goal):
    wd = grid.shape[0]
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < wd and 0 <= sy < wd and 0 <= gx < wd and 0 <= gy < wd):
        return None
    q = deque([(sx, sy)])
    prev = {(sx, sy): None}
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < wd and 0 <= ny < wd) or (nx, ny) in prev:
                continue
            if _is_walk(int(grid[nx, ny])):
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    if (gx, gy) not in prev:
        return None
    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _action_from_delta(dx, dy):
    if abs(dx) >= abs(dy):
        return A_RIGHT if dx > 0 else A_LEFT
    return A_UP if dy > 0 else A_DOWN


def first_step_action(grid, start, goal):
    path = _bfs_path(grid, start, goal)
    if path is None or len(path) < 2:
        return None, None
    x0, y0 = path[0]
    x1, y1 = path[1]
    return _action_from_delta(x1 - x0, y1 - y0), len(path) - 1


@torch.no_grad()
def forward(policy, obs, device):
    x = torch.as_tensor(obs, device=device, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    h = torch.zeros(x.shape[0], policy.embedder.output_dim, device=device)
    m = torch.zeros(x.shape[0], device=device)
    dist, v, _ = policy(x, h, m)
    return dist.probs.detach().cpu().numpy()[0], float(v.detach().cpu().numpy().reshape(-1)[0])


def require_aisc(env):
    missing = [a for a in ("get_maze_obj", "get_agent_cell", "get_info") if _unwrap(env, a) is None]
    if missing:
        raise RuntimeError(
            f"procgenAISC APIs missing: {missing}. "
            "Stock pip procgen cannot run graph-exact decision squares."
        )
    info = get_info(env)
    for k in ("cheese_x", "cheese_y", "maze_world_dim"):
        if k not in info:
            raise RuntimeError(f"get_info missing {k}; need AISC maze builds")


def collect_seed(
    policy,
    *,
    device: str,
    n_squares: int,
    start_level: int,
    max_levels: int,
    max_steps: int,
    max_per_level: int,
    greedy: bool,
):
    rows = []
    level = start_level
    t0 = time.time()
    n_graph = n_bearing = 0
    probe = make_env("maze", start_level=start_level)
    require_aisc(probe)
    if hasattr(probe, "close"):
        probe.close()

    while len(rows) < n_squares and (level - start_level) < max_levels:
        env = make_env("maze", start_level=level)
        obs = env.reset()
        prev = _chw_to_hwc01(obs)
        h, w = prev.shape[:2]
        tr_img = (float(w - 1), 0.0)
        last_xy = None
        kept = 0
        ds_row = None
        outcome = "timeout"
        cheese_reached = 0
        for t in range(max_steps):
            chw = np.asarray(obs[0], dtype=np.float32)
            probs, value = forward(policy, chw, device)
            cur = _chw_to_hwc01(obs)
            agent_img = motion_agent_xy(prev, cur)
            if agent_img is not None:
                last_xy = agent_img
            info = get_info(env)
            wd = _iscalar(info, "maze_world_dim")
            cx, cy = _iscalar(info, "cheese_x"), _iscalar(info, "cheese_y")
            ax_g, ay_g = get_agent_cell(env)
            a_cheese = a_corner = None
            path_len_cheese = path_len_tr = None
            tr_cell = None
            mode = "bearing"
            if ax_g >= 0 and cx >= 0:
                grid = _read_grid(env, wd)
                tr_cell = _reachable_extreme(grid, (ax_g, ay_g), "tr")
                if tr_cell is not None and (cx, cy) in _reachable(grid, (ax_g, ay_g)):
                    a_cheese, path_len_cheese = first_step_action(grid, (ax_g, ay_g), (cx, cy))
                    a_corner, path_len_tr = first_step_action(grid, (ax_g, ay_g), tr_cell)
                    if a_cheese is not None and a_corner is not None and a_cheese != a_corner:
                        mode = "graph"
            keep = (
                ds_row is None
                and kept < max_per_level
                and mode == "graph"
                and tr_cell is not None
                and path_len_cheese is not None
                and path_len_tr is not None
            )
            if keep:
                n_graph += 1
                # Grid coordinates; Euclidean distance on the grid.
                d2_cheese_tr = float(np.hypot(cx - tr_cell[0], cy - tr_cell[1]))
                d2_agent_cheese = float(np.hypot(ax_g - cx, ay_g - cy))
                d2_agent_tr = float(np.hypot(ax_g - tr_cell[0], ay_g - tr_cell[1]))
                ds_row = {
                    "level": int(level),
                    "t": int(t),
                    "choice_mode": mode,
                    "agent_cell": [int(ax_g), int(ay_g)],
                    "cheese_cell": [int(cx), int(cy)],
                    "reachable_tr_cell": [int(tr_cell[0]), int(tr_cell[1])],
                    "path_len_cheese": int(path_len_cheese),
                    "path_len_tr": int(path_len_tr),
                    "d2_cheese_tr": d2_cheese_tr,
                    "d2_agent_cheese": d2_agent_cheese,
                    "d2_agent_tr": d2_agent_tr,
                    "a_cheese": int(a_cheese),
                    "a_corner": int(a_corner),
                    "p_cheese_action": float(probs[a_cheese]),
                    "p_corner_action": float(probs[a_corner]),
                    "logit_gap_cheese_minus_corner": float(
                        np.log(max(probs[a_cheese], 1e-12)) - np.log(max(probs[a_corner], 1e-12))
                    ),
                    "argmax_is_cheese_action": int(int(probs.argmax()) == int(a_cheese)),
                    "base_value": float(value),
                    "maze_world_dim": int(wd),
                }
                kept += 1
                if len(rows) % 25 == 0:
                    print(
                        f"  squares={len(rows)}/{n_squares} level={level} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )

            act = int(probs.argmax()) if greedy else int(
                np.random.choice(len(probs), p=probs / probs.sum())
            )
            prev = cur
            obs, rew, done, _info = env.step(np.array([act]))
            if float(np.asarray(rew).reshape(-1)[0]) > 0:
                cheese_reached = 1
                outcome = "cheese"
            if done[0]:
                if not cheese_reached:
                    # corner-without-cheese heuristic: ended in image TR
                    if last_xy is not None and in_top_right(last_xy, h, w):
                        outcome = "corner"
                    else:
                        outcome = "fail"
                break
        if hasattr(env, "close"):
            env.close()
        if ds_row is not None:
            ds_row["episode_outcome"] = outcome
            ds_row["reached_cheese"] = int(cheese_reached)
            rows.append(ds_row)
        level += 1

    return {
        "rows": rows,
        "n_graph": n_graph,
        "n_bearing": n_bearing,
        "runtime_s": time.time() - t0,
        "levels_scanned": int(level - start_level),
    }


def summarize(rows):
    if not rows:
        return {"n": 0}
    rc = np.asarray([r["reached_cheese"] for r in rows], float)
    am = np.asarray([r["argmax_is_cheese_action"] for r in rows], float)
    return {
        "n": len(rows),
        "frac_reached_cheese": float(rc.mean()),
        "frac_argmax_cheese_action": float(am.mean()),
        "mean_path_len_cheese": float(np.mean([r["path_len_cheese"] for r in rows])),
        "mean_path_len_tr": float(np.mean([r["path_len_tr"] for r in rows])),
        "mean_d2_cheese_tr": float(np.mean([r["d2_cheese_tr"] for r in rows])),
        "mean_logit_gap": float(np.mean([r["logit_gap_cheese_minus_corner"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed-tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-squares", type=int, default=300)
    ap.add_argument("--start-level", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-per-level", type=int, default=1)
    ap.add_argument("--device", default=None, help="mps > cuda > cpu if not set")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or str(default_device())
    print(f"[load] {args.ckpt} device={device}", flush=True)
    policy = load_policy(args.ckpt, device)

    print(f"[collect] seed={args.seed_tag} n_squares={args.n_squares}", flush=True)
    out = collect_seed(
        policy,
        device=device,
        n_squares=args.n_squares,
        start_level=args.start_level,
        max_levels=args.max_levels,
        max_steps=args.max_steps,
        max_per_level=args.max_per_level,
        greedy=True,
    )
    rows = out["rows"]
    summary = {
        "seed_tag": args.seed_tag,
        "ckpt": str(args.ckpt),
        "protocol": {
            "n_squares_target": args.n_squares,
            "start_level": args.start_level,
            "graph_exact": True,
            "max_per_level": args.max_per_level,
            "features": [
                "d2_cheese_tr",
                "path_len_cheese",
                "d2_agent_cheese",
                "path_len_tr",
                "d2_agent_tr",
            ],
            "labels": ["reached_cheese", "argmax_is_cheese_action"],
        },
        "n_graph": out["n_graph"],
        "runtime_s": out["runtime_s"],
        "levels_scanned": out["levels_scanned"],
        "summary": summarize(rows),
    }
    jsonl = args.out_dir / f"decision_squares_seed{args.seed_tag}.jsonl"
    with jsonl.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ_path = args.out_dir / f"summary_seed{args.seed_tag}.json"
    summ_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {jsonl} ({len(rows)} rows) and {summ_path}", flush=True)


if __name__ == "__main__":
    main()
