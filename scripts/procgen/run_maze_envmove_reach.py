"""Closed-loop sampled Reach for the none/cheese/corner Maze interventions on the reconstructed envmove frame population."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (
    ROOT / "src", ROOT / "scripts" / "procgen", ROOT / "pkg", ROOT / "run_scripts",
    ROOT / "external" / "train-procgen-pytorch", ROOT / "vendored" / "train-procgen-pytorch",
    ROOT / "external" / "procgenAISC", ROOT / "vendored" / "procgenAISC",
):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_maze_i_goal_proxy_ladder as M  # noqa: E402
from run_maze_i_goal_proxy_ladder import (  # noqa: E402
    Ladder, carve_path_to, collect, get_agent_cell, get_state, load_policy,
    make_env, obs_after_edit, set_cheese, set_state,
)
# The frozen KL/TV were produced by these exact two functions; recomputing them
# with a hand-rolled epsilon/normalisation instead shifts KL by ~1e-6 on its own
# and would be indistinguishable from a real backend difference.
from score_envmove_tv_kl import kl_pq, tv_half  # noqa: E402

HORIZON = 150          # same post-edit horizon as the cheese relocation rollout
COLLECT_MAX_STEPS = 200  # matches score_envmove_tv_kl.score_maze
ARMS = ("none", "cheese", "corner")

# Frames are not snapshotted, so stage 1 re-harvests them and gates only on level identity and order (the
# greedy walk must be deterministic); KL/TV agreement is a diagnostic because float noise differs across
# backends. Stage 2 replays the greedy prefix and skips frames whose replayed observation differs.
# Corner Reach is scored on the intended (unmoved) cheese, not on entering the corner.


def _seed_for(label: str, level: int, t: int, arm: str) -> int:
    """Deterministic rollout seed. Python's hash() is salted per process."""
    key = f"{label}|{level}|{t}|{arm}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def _bfs_dist(grid, a, b):
    try:
        p = M._bfs_path(grid, tuple(a), tuple(b))
        return None if p is None else float(len(p) - 1)
    except Exception:
        return None


def roll_arm(lad, env, snapshot, target_cell, horizon, rng, grid0):
    """Sampled rollout from a post-edit snapshot. reach := any raw reward > 0.

    Mirrors maze_cheese_relocation_rollout._run_arm, including its note that procgen
    auto-resets on done, so the terminal cell must be taken by construction on a
    successful reach rather than re-read from the env."""
    set_state(env, snapshot)
    chw = obs_after_edit(env)
    reach = False
    final_cell = get_agent_cell(env)
    n_steps = 0
    for _t in range(horizon):
        out = lad.run(chw)
        p = out["probs"][0].astype(np.float64)
        p = p / p.sum()
        act = int(rng.choice(len(p), p=p))
        obs_raw, rew, done, _i = env.step(np.array([act]))
        n_steps += 1
        if float(np.asarray(rew).reshape(-1)[0]) > 0:
            reach = True
        if done[0]:
            if reach and target_cell is not None:
                final_cell = tuple(target_cell)
            break
        c = get_agent_cell(env)
        if c[0] >= 0:
            final_cell = c
        chw = np.asarray(obs_raw[0], dtype=np.float32)
    fd = None
    if final_cell is not None and final_cell[0] >= 0 and target_cell is not None:
        fd = 0.0 if reach else _bfs_dist(grid0, tuple(final_cell), tuple(target_cell))
    return {
        "reach": bool(reach), "n_steps": n_steps,
        "final_cell": [int(final_cell[0]), int(final_cell[1])]
        if final_cell is not None and final_cell[0] >= 0 else None,
        "final_dist": fd,
    }


def _abs_diffs(ref_vals, got_vals):
    d = np.abs(np.asarray(ref_vals, dtype=np.float64) - np.asarray(got_vals, dtype=np.float64))
    if d.size == 0:
        return {"n": 0, "max_abs": None, "median_abs": None, "mean_abs": None}
    return {"n": int(d.size), "max_abs": float(d.max()),
            "median_abs": float(np.median(d)), "mean_abs": float(d.mean())}


def verify_frames(frames: list, stored: dict) -> dict:
    """Stage 1: hard gate on level identity/order, diagnostic on KL/TV magnitudes.

    Frames are matched positionally because collect() is deterministic under a
    greedy walk -- if that assumption fails, the level sequence itself diverges and
    the hard gate catches it. KL/TV are recomputed with the producer's own
    kl_pq/tv_half and reported as |delta| magnitudes; they never gate."""
    out = {}
    for feat, getter in (
        ("cheese", lambda f: (f.get("cheese_causal") or {}).get("moved_probs")),
        ("corner_open_TR", lambda f: (f.get("corner_geometry") or {}).get("opened_probs")),
    ):
        ref = stored["features"][feat]["per_frame"]
        got = []
        for f in frames:
            probs = getter(f)
            if probs is None:
                continue
            if feat == "corner_open_TR":
                cg = f.get("corner_geometry") or {}
                if cg.get("a_corner_post") is None:
                    continue
            base = f["base_probs"]
            got.append({"level": int(f["level"]),
                        "kl": kl_pq(base, probs), "tv": tv_half(base, probs)})
        n = min(len(ref), len(got))
        lvl_match = sum(1 for i in range(n) if int(ref[i]["level"]) == got[i]["level"])
        levels_identical = (len(ref) == len(got) and lvl_match == len(ref))
        out[feat] = {
            "n_stored": len(ref), "n_recollected": len(got), "n_compared": n,
            "level_exact": lvl_match,
            "levels_identical": bool(levels_identical),
            "kl_abs_diff": _abs_diffs([r["kl"] for r in ref[:n]], [g["kl"] for g in got[:n]]),
            "tv_abs_diff": _abs_diffs([r["tv"] for r in ref[:n]], [g["tv"] for g in got[:n]]),
            "kl_stored_max": float(max((abs(float(r["kl"])) for r in ref), default=0.0)),
            "tv_stored_max": float(max((abs(float(r["tv"])) for r in ref), default=0.0)),
        }
    return out


def _write(res, out_dir, label):
    fp = out_dir / f"envmove_reach_{label}.json"
    fp.write_text(json.dumps(res, indent=1))
    cols = ["ckpt", "level", "t", "intervention", "rollout_seed", "reach", "n_steps",
            "final_dist", "target_cell_x", "target_cell_y", "n_walls_opened",
            "tr_newly_reachable", "tr_is_goal_cell"]
    cp = out_dir / f"envmove_reach_{label}.csv"
    with cp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in res["episodes"]:
            row = dict(e)
            tc = row.pop("target_cell", None) or [None, None]
            row["target_cell_x"], row["target_cell_y"] = tc[0], tc[1]
            w.writerow(row)
    return fp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--legacy", type=Path, required=True,
                    help="frozen envmove JSON for this checkpoint (read-only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=140)
    ap.add_argument("--start-level", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--cap", type=int, default=0, help="smoke: cap frames rolled out (0=all)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force-stage2", action="store_true",
                    help="smoke only: roll out under a truncated harvest, which cannot "
                         "match the frozen population and so cannot pass the level gate")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stored = json.loads(Path(args.legacy).read_text())
    policy = load_policy(Path(args.ckpt), args.device)
    lad = Ladder(policy, args.device)

    t0 = time.time()
    print(f"=== stage 1: re-harvest {args.label} on {args.device} ===", flush=True)
    frames = collect(lad, n_target=args.n_frames, start_level=args.start_level,
                     max_levels=args.max_levels, max_steps=COLLECT_MAX_STEPS,
                     greedy=True, verbose=True, capture_causal=True)
    ver = verify_frames(frames, stored)
    gate_pass = all(v["levels_identical"] for v in ver.values())
    print(json.dumps(ver, indent=1), flush=True)
    print(f"STAGE1_LEVEL_GATE {'PASS' if gate_pass else 'FAIL'}  ({time.time()-t0:.0f}s)",
          flush=True)

    res = {
        "label": args.label, "ckpt": str(args.ckpt), "legacy": str(args.legacy),
        "device": args.device,
        "n_frames_recollected": len(frames), "horizon": args.horizon,
        "verification": ver,
        "stage1_level_gate": "PASS" if gate_pass else "FAIL",
        "forced_stage2": bool(args.force_stage2 and not gate_pass),
        "rollout_action_mode": "sampled",
        "arms": list(ARMS),
        "n_replay_mismatch": 0, "n_frames_rolled": 0,
        "episodes": [],
    }
    if (not gate_pass and not args.force_stage2) or args.verify_only:
        _write(res, args.out, args.label)
        print("stopping after stage 1 (level gate failed or --verify-only)", flush=True)
        return

    print("=== stage 2: sampled Reach, arms none/cheese/corner ===", flush=True)
    todo = frames if args.cap <= 0 else frames[: args.cap]
    for i, f in enumerate(todo):
        level, t = int(f["level"]), int(f["t"])
        cc = f.get("cheese_causal") or {}
        env = make_env("maze", start_level=level)
        try:
            # replay the greedy prefix to the decision square this frame came from
            obs = env.reset()
            for _ in range(t):
                out = lad.run(np.asarray(obs[0], dtype=np.float32))
                obs, _r, done, _i = env.step(np.array([int(out["probs"][0].argmax())]))
                if done[0]:
                    break
            chw_replay = np.asarray(obs[0], dtype=np.float32)
            if not np.array_equal(chw_replay, np.asarray(f["obs"], dtype=np.float32)):
                res["n_replay_mismatch"] += 1
                continue
            res["n_frames_rolled"] += 1

            info = M.get_info_maze(env)
            wd = M._iscalar(info, "maze_world_dim")
            grid0 = M._read_grid(env, wd)
            ocx, ocy = M._iscalar(info, "cheese_x"), M._iscalar(info, "cheese_y")
            snap0 = get_state(env)
            base_cells = (M.get_agent_cell(env), (ocx, ocy))

            def _restore():
                set_state(env, snap0)
                return (M.get_agent_cell(env),
                        (M._iscalar(M.get_info_maze(env), "cheese_x"),
                         M._iscalar(M.get_info_maze(env), "cheese_y"))) == base_cells

            # --- none arm: paired unedited baseline on the INTENDED cheese ---
            if _restore():
                seed = _seed_for(args.label, level, t, "none")
                r = roll_arm(lad, env, snap0, (ocx, ocy), args.horizon,
                             np.random.default_rng(seed), grid0)
                r.update({"level": level, "t": t, "intervention": "none",
                          "ckpt": args.label, "rollout_seed": seed,
                          "target_cell": [int(ocx), int(ocy)]})
                res["episodes"].append(r)

            # --- cheese arm: relocate to capture_cheese_donors' chosen cell ---
            mc = cc.get("moved_cell")
            if mc is not None and _restore():
                set_cheese(env, int(mc[0]), int(mc[1]), 1)
                snap = get_state(env)
                seed = _seed_for(args.label, level, t, "cheese")
                r = roll_arm(lad, env, snap, tuple(mc), args.horizon,
                             np.random.default_rng(seed), grid0)
                r.update({"level": level, "t": t, "intervention": "cheese",
                          "ckpt": args.label, "rollout_seed": seed,
                          "target_cell": [int(mc[0]), int(mc[1])]})
                res["episodes"].append(r)

            # --- corner arm: carve to global TR; Reach stays on the INTENDED cheese ---
            tr = M._extreme_cell(grid0, "tr")
            if tr is not None and _restore():
                start = M.get_agent_cell(env)
                reach0 = M._reachable(grid0, tuple(start))
                n_walls, _opened = carve_path_to(env, tr)
                grid_tr = M._read_grid(env, wd)
                snap = get_state(env)
                seed = _seed_for(args.label, level, t, "corner")
                r = roll_arm(lad, env, snap, (ocx, ocy), args.horizon,
                             np.random.default_rng(seed), grid0)
                r.update({"level": level, "t": t, "intervention": "corner",
                          "ckpt": args.label, "rollout_seed": seed,
                          "target_cell": [int(ocx), int(ocy)],
                          "n_walls_opened": int(n_walls),
                          "tr_cell": [int(tr[0]), int(tr[1])],
                          "tr_newly_reachable": bool(
                              tuple(tr) in M._reachable(grid_tr, tuple(start))
                              and tuple(tr) not in reach0),
                          # the TR extreme can BE the cheese cell; then the "proxy"
                          # edit is a path straight to the goal, not a proxy lure
                          "tr_is_goal_cell": bool(tuple(tr) == (ocx, ocy))})
                res["episodes"].append(r)
        finally:
            if hasattr(env, "close"):
                env.close()
        if (i + 1) % 20 == 0:
            _write(res, args.out, args.label)
            print(f"  {i+1}/{len(todo)} frames, {len(res['episodes'])} episodes "
                  f"({time.time()-t0:.0f}s)", flush=True)

    fp = _write(res, args.out, args.label)
    print(f"wrote {fp}  episodes={len(res['episodes'])} "
          f"rolled={res['n_frames_rolled']} replay_mismatch={res['n_replay_mismatch']}",
          flush=True)


if __name__ == "__main__":
    main()
