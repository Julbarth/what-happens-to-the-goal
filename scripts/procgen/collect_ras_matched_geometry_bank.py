#!/usr/bin/env python3
"""Policy-independent, graph-distance-matched Attribute Maze (maze_colorsobjects_duel) state bank."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "pkg", ROOT / "src"):
    if _p.exists():
        sys.path.insert(0, str(_p))

from gm_geometry.probes.ras_agent_localization import (  # noqa: E402
    bfs_distances,
    calibrate_action_deltas,
    calibrate_pixel_to_grid,
    first_action_set,
    locate_agent_cell,
    probe_wall_grid,
)

N_ACTIONS = 15
WALKABLE_RC = 0
BLOCKED_RC = -2
# The env exposes neither the wall grid nor the agent cell: walls are recovered by probing set_object return codes
# (rc=-2 on WALL), and the agent cell by intersecting the motion masks of two opposite actions.
# All trained checkpoints use world_dim=5, distribution_mode="easy"; backgrounds only affect rendering at scoring time.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-level", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=400)
    ap.add_argument("--target-n", type=int, default=150)
    ap.add_argument("--dist-tolerance", type=int, default=0, help="0 = exact graph-distance match")
    ap.add_argument("--world-dim", type=int, default=5,
                     help="5 matches the training config of our and the published Attribute Maze "
                          "checkpoints (5x5 world).")
    ap.add_argument("--distribution-mode", default="easy",
                     help="'easy' matches training (see --world-dim).")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cs_procgen = ROOT / "external" / "colour-shape-goal-misgeneralization" / "procgenAISC"
    if not cs_procgen.exists():
        raise RuntimeError(f"colour-shape procgenAISC not found at {cs_procgen} (see external/README.md)")
    sys.path.insert(0, str(cs_procgen))
    sys.path.insert(0, str(ROOT / "external" / "train-procgen-pytorch"))
    from procgen import ProcgenEnv
    from common.env.procgen_wrappers import VecExtractDictObs

    def make_env(level):
        venv = ProcgenEnv(
            num_envs=1, env_name="maze_colorsobjects_duel", num_levels=1, start_level=level,
            distribution_mode=args.distribution_mode, world_dim=args.world_dim,
            obj1="red_line_diag", obj2="yellow_gem",
        )
        return VecExtractDictObs(venv, "rgb")

    def unwrap(env):
        cur = env
        seen = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if hasattr(cur, "set_object"):
                return cur
            cur = getattr(cur, "venv", None) or getattr(cur, "env", None)
        return None

    margin = 3  # generous bound around the playable area; out-of-range cells just probe as blocked
    coord_range = range(-margin, args.world_dim + margin)

    rows = []
    level = args.start_level
    n_layouts_with_hits = 0
    action_deltas: dict | None = None
    calib_level = None
    while len(rows) < args.target_n and (level - args.start_level) < args.max_levels:
        env = make_env(level)
        obs0 = env.reset()
        base = unwrap(env)
        if base is None:
            level += 1
            if hasattr(env, "close"):
                env.close()
            continue

        walkable = probe_wall_grid(base, coord_range, coord_range)
        if len(walkable) < 4:
            level += 1
            if hasattr(env, "close"):
                env.close()
            continue

        calib, _calib_obs = calibrate_pixel_to_grid(base, env, walkable, "red_line_diag", "yellow_gem")
        if calib is None:
            level += 1
            if hasattr(env, "close"):
                env.close()
            continue

        # action_deltas is an env-wide constant (get_combos()'s button semantics don't
        # vary by level) but its pixel->grid conversion needs THIS level's calib, so it
        # can only be computed once calib is available -- cache and reuse afterward.
        if action_deltas is None:
            cand = calibrate_action_deltas(base, env, obs0, calib)
            found = {d for d in cand.values() if d is not None}
            # locate_agent_cell needs an exact opposite pair (d, -d) to intersect
            # motion masks -- ">=2 distinct directions" isn't sufficient (e.g. UP+RIGHT
            # alone can never form a pair).
            has_opposite_pair = any((-d[0], -d[1]) in found for d in found)
            if has_opposite_pair:
                action_deltas = cand
                calib_level = level
                print(f"[calibrate] @ level {calib_level}: {action_deltas}", flush=True)
            else:
                print(f"[calibrate] @ level {level}: failed (no opposite pair among {found}), retrying", flush=True)
                level += 1
                if hasattr(env, "close"):
                    env.close()
                continue

        agent_cell = locate_agent_cell(base, env, obs0, action_deltas, calib, walkable)
        if agent_cell is None:
            level += 1
            if hasattr(env, "close"):
                env.close()
            continue

        # enumerate reachable-pair candidates with disjoint first-action sets and
        # (near-)equal graph distance from the agent
        dist_from_agent_targets = {}
        found_pair = None
        cells = list(walkable)
        for i, xa in enumerate(cells):
            if xa == agent_cell:
                continue
            if xa not in dist_from_agent_targets:
                dist_from_agent_targets[xa] = bfs_distances(walkable, xa)
            da = dist_from_agent_targets[xa].get(agent_cell)
            if da is None:
                continue
            aa = first_action_set(dist_from_agent_targets[xa], agent_cell, action_deltas)
            if not aa:
                continue
            for xb in cells[i + 1 :]:
                if xb == agent_cell:
                    continue
                if xb not in dist_from_agent_targets:
                    dist_from_agent_targets[xb] = bfs_distances(walkable, xb)
                db = dist_from_agent_targets[xb].get(agent_cell)
                if db is None or abs(da - db) > args.dist_tolerance:
                    continue
                ab = first_action_set(dist_from_agent_targets[xb], agent_cell, action_deltas)
                if not ab or not aa.isdisjoint(ab):
                    continue
                found_pair = (xa, xb, da, db, sorted(aa), sorted(ab))
                break
            if found_pair:
                break

        if found_pair is not None:
            xa, xb, da, db, aa, ab = found_pair
            base_state = base.get_state()
            rows.append(
                {
                    "level": int(level),
                    "agent_cell": list(agent_cell),
                    "x_a": list(xa),
                    "x_b": list(xb),
                    "dist_a": int(da),
                    "dist_b": int(db),
                    "first_actions_a": aa,
                    "first_actions_b": ab,
                    "world_dim": args.world_dim,
                    "raw_state": base_state[0],
                }
            )
            n_layouts_with_hits += 1
        level += 1
        if hasattr(env, "close"):
            env.close()
        if (level - args.start_level) % 10 == 0:
            print(f"  level={level} banked={len(rows)}", flush=True)

    states = [r.pop("raw_state") for r in rows]
    meta = {
        "protocol": {
            "start_level": args.start_level,
            "levels_scanned": int(level - args.start_level),
            "dist_tolerance": args.dist_tolerance,
            "world_dim": args.world_dim,
            "calibration_level": calib_level,
            "action_deltas": {str(k): v for k, v in (action_deltas or {}).items()},
            "note": (
                "wall grid via set_object return-code probing; agent cell via pixel "
                "calibration cross-checked against the probed grid; policy-independent, "
                "no trained checkpoint involved in state selection"
            ),
        },
        "n_geometries": len(rows),
        "n_layouts_with_hits": n_layouts_with_hits,
        "rows": rows,
    }
    (args.out_dir / "ras_matched_geometry_meta.json").write_text(json.dumps(meta, indent=2))
    with (args.out_dir / "ras_matched_geometry_states.pkl").open("wb") as f:
        pickle.dump(states, f)
    print(f"[done] {len(rows)} geometries -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
