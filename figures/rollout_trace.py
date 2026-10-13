"""Record one greedy OOD episode (frames + agent positions) for a single checkpoint, for visualization."""
from __future__ import annotations

from paths import ROOT
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch



# The in-repo colour-shape procgen build is x86_64. ATTR_PROCGEN_ROOT points at a native build of the
# same source. Importing it before any gm_geometry module pins sys.modules["procgen"], since
# gm_geometry.integrations.colour_shape imports procgen from the in-repo path on import.
if os.environ.get("ATTR_PROCGEN_ROOT"):
    sys.path.insert(0, os.environ["ATTR_PROCGEN_ROOT"])
    import procgen  # noqa: E402,F401
sys.path.insert(0, str(ROOT / "scripts" / "procgen"))

from gm_geometry.probes.procgen_agent_state import raw_state  # noqa: E402
from gm_geometry.probes.coin_location import get_info  # noqa: E402
from gm_geometry.probes.ras_gt_bank import GOAL_GEM, GOAL_LINE, cells_of, parse_agent, parse_grid  # noqa: E402


def state_api(env):
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if callable(getattr(cur, "get_state", None)):
            return cur
        cur = getattr(cur, "env", getattr(cur, "venv", getattr(cur, "_env", None)))
    raise RuntimeError("no get_state under the env wrappers")


def grid_dim(raw: bytes) -> int:
    """Side length of the square procgen Grid, found by its serialized int32 (w, h, w*h) header."""
    hits = [d for d in range(3, 65) if raw.find(np.array([d, d, d * d], dtype="<i4").tobytes()) != -1]
    if not hits:
        raise ValueError("no square grid header in state")
    return max(hits)


def to_u8(obs) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    x = np.transpose(x, (1, 2, 0))
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


@torch.no_grad()
def grid_episode(policy, env, device, max_steps):
    """Greedy rollout in a grid-step maze env; agent cell read from the serialized game state."""
    api = state_api(env)
    obs = env.reset()
    raw0 = raw_state(api)
    dim = grid_dim(raw0)
    cells = [parse_agent(raw0)]
    frames = [to_u8(obs)]
    info_start = get_info(env)
    hires0 = np.asarray(info_start["rgb"], dtype=np.uint8).copy()
    hires_last = hires0
    hidden = np.zeros((1, policy.embedder.output_dim), dtype=np.float32)
    total, done_flag = 0.0, False
    for _ in range(max_steps):
        dist, _v, h = policy(torch.as_tensor(obs, dtype=torch.float32, device=device),
                             torch.as_tensor(hidden, device=device), torch.zeros(1, device=device))
        hidden = h.cpu().numpy()
        obs, rew, done, info = env.step(dist.probs.argmax(-1).cpu().numpy())
        info0 = info[0] if isinstance(info, (list, tuple)) else info
        raw_r = info0.get("env_reward") if isinstance(info0, dict) else None
        total += float(raw_r) if raw_r is not None else float(np.asarray(rew)[0])
        if bool(np.asarray(done)[0]):
            done_flag = True
            break
        cells.append(parse_agent(raw_state(api)))
        frames.append(to_u8(obs))
        hires_last = np.asarray(get_info(env)["rgb"], dtype=np.uint8).copy()
    return {"cells": np.asarray(cells), "frame0": frames[0], "frames": np.stack(frames),
            "hires0": hires0, "hires_last": hires_last, "grid_dim": dim, "raw0": raw0,
            "cheese": [int(info_start.get("cheese_x", -1)), int(info_start.get("cheese_y", -1))], "reward": total, "done": done_flag, "steps": len(cells)}


def run_maze(ckpt, level, device, max_steps):
    from validate_langosco_maze_i_200m import load_policy, make_env
    policy = load_policy(Path(ckpt), device)
    env = make_env("maze", num_levels=1, start_level=level, extra={"render_mode": "rgb_array"})
    r = grid_episode(policy, env, device, max_steps)
    r["targets"] = {"goal": r.pop("cheese")}
    r.pop("raw0")
    return r


def run_attr(ckpt, level, device, max_steps, use_backgrounds):
    from gm_geometry.integrations.colour_shape.paths import ensure_imports
    from gm_geometry.integrations.colour_shape.policy import load_impala_policy, load_recurrent_flag
    ensure_imports()
    from procgen import ProcgenEnv
    from common.env.procgen_wrappers import (ScaledFloatFrame, TransposeFrame, VecExtractDictObs,
                                             VecNormalize)
    # Same construction as gm_geometry.integrations.colour_shape.envs.make_duel_env, plus render_mode
    # for a 512 px frame; rendering does not touch dynamics.
    env = ProcgenEnv(num_envs=1, env_name="maze_colorsobjects_duel", num_levels=1, start_level=level,
                     distribution_mode="easy", use_backgrounds=use_backgrounds, world_dim=5,
                     obj1="red_line_diag", obj2="yellow_gem", num_threads=1, render_mode="rgb_array")
    env = ScaledFloatFrame(TransposeFrame(VecNormalize(VecExtractDictObs(env, "rgb"), ob=False)))
    policy = load_impala_policy(Path(ckpt), env.observation_space.shape[0], env.action_space.n,
                                load_recurrent_flag()).to(device)
    r = grid_episode(policy, env, device, max_steps)
    _, _, grid = parse_grid(r.pop("raw0"))
    r.pop("cheese")
    r["targets"] = {"goal": list(cells_of(grid, GOAL_LINE)[0]), "proxy": list(cells_of(grid, GOAL_GEM)[0])}
    return r


def run_coin(ckpt, level, device, max_steps):
    from validate_langosco_coinrun_200m import load_policy
    from coin_ood_paths import rollout
    policy = load_policy(Path(ckpt), device)
    r = rollout(policy, device, level, max_steps=max_steps)
    return {"positions": np.asarray(r["positions"]), "frames": np.stack(r["frames"]),
            "coin": np.asarray(r["coin"]), "end_x": r["end_x"], "coin_reach": r["coin_reach"],
            "steps": r["n"], "collect_px": np.asarray(r["collect_px"] or (-1, -1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["maze", "coin", "attr"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--use-backgrounds", action="store_true")
    ap.add_argument("--scan", type=int, default=0,
                    help="roll levels [level, level+scan) and write outcomes only (level search)")
    args = ap.parse_args()
    device = "cpu"
    if args.scan:
        rows = []
        for lv in range(args.level, args.level + args.scan):
            if args.env == "maze":
                r = run_maze(args.ckpt, lv, device, args.max_steps)
            elif args.env == "attr":
                r = run_attr(args.ckpt, lv, device, args.max_steps, args.use_backgrounds)
            else:
                r = run_coin(args.ckpt, lv, device, args.max_steps)
                r["cells"] = r["positions"]
            rows.append({"level": lv, "reward": float(r.get("reward", 10.0 * r.get("coin_reach", 0))),
                         "steps": int(r["steps"]), "last": np.asarray(r["cells"])[-1].tolist()})
        Path(args.out).write_text(json.dumps(rows))
        print("DONE scan", len(rows))
        return
    if args.env == "maze":
        r = run_maze(args.ckpt, args.level, device, args.max_steps)
    elif args.env == "attr":
        r = run_attr(args.ckpt, args.level, device, args.max_steps, args.use_backgrounds)
    else:
        r = run_coin(args.ckpt, args.level, device, args.max_steps)
    meta = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
    meta.update(ckpt=args.ckpt, level=args.level, env=args.env)
    np.savez_compressed(args.out, **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps(meta))
    print("DONE", json.dumps(meta))


if __name__ == "__main__":
    main()
