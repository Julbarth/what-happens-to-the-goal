#!/usr/bin/env python3
"""Build the canonical Attribute-Maze conflict bank from game state only, with fatal execution-based validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from gm_geometry.probes.fisher import make_env  # noqa: E402
from gm_geometry.probes.procgen_agent_state import raw_state  # noqa: E402
from gm_geometry.probes.ras_agent_localization import probe_wall_grid  # noqa: E402
from gm_geometry.probes.ras_gt_bank import (  # noqa: E402
    LAYOUT_KEYS, N_ACTIONS, bfs, canonical_hash, parse_agent, parse_grid, state_record, successor,
    walkable_cells, walls_key,
)

# Every accepted conflict state must pass gates G1-G6 in validate() (targets, walls, all 15 action successors,
# progress sets, shortest paths, edit invariance) or the build aborts; one process builds one background.

OBJ_LINE, OBJ_GEM = "red_line_diag", "yellow_gem"
SHAPE_EDIT_OBJ, COLOUR_EDIT_OBJ = "red_gem", "green_gem"   # line's shape ablated / gem's colour ablated
WORLD_DIM, PROBE_MARGIN = 5, 3
REWARD_SIGN = {"line": 1.0, "gem": -1.0}


def state_api(env):
    """The wrapped object exposing get_state/set_state/set_object/get_object_pos."""
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if callable(getattr(cur, "get_state", None)) and (hasattr(cur, "set_object") or not hasattr(cur, "env")):
            return cur
        cur = getattr(cur, "env", getattr(cur, "venv", getattr(cur, "_env", None)))
    raise RuntimeError("could not find the procgen state API under the env wrappers")
CODE_FILES = ("src/gm_geometry/probes/ras_gt_bank.py", "src/gm_geometry/probes/procgen_agent_state.py",
              "scripts/build_ras_gt_bank.py")


def builder_code_hash() -> str:
    """sha256 over the builder's own source files (also found under pkg/ and run_scripts/ in packaged copies)."""
    h = hashlib.sha256()
    for rel in CODE_FILES:
        cand = [ROOT / rel, ROOT / rel.replace("src/", "pkg/").replace("scripts/", "run_scripts/")]
        h.update(next(p for p in cand if p.exists()).read_bytes())
    return h.hexdigest()


class GateError(RuntimeError):
    pass


def gate(cond: bool, msg: str) -> None:
    if not cond:
        raise GateError(msg)


def _step(env, a: int):
    obs, rew, done, _ = env.step(np.array([a]))
    return obs, float(np.asarray(rew).reshape(-1)[0]), bool(np.asarray(done).reshape(-1)[0])


def validate(rec: dict, env, api, st0, bg: str) -> None:
    level = rec["level"]
    agent, line, gem = tuple(rec["agent"]), tuple(rec["line"]), tuple(rec["gem"])
    _, _, grid = parse_grid(raw_state(api))
    walkable = walkable_cells(grid)

    gate(tuple(api.get_object_pos([1])[0]) == line and tuple(api.get_object_pos([2])[0]) == gem,
         f"G1 level {level}: get_object_pos disagrees with grid")

    env_p = make_env(OBJ_LINE, OBJ_GEM, level, 0, "easy", bg == "texture")
    env_p.reset()
    rng = range(-PROBE_MARGIN, WORLD_DIM + PROBE_MARGIN)
    probed = probe_wall_grid(state_api(env_p), rng, rng)
    env_p.close()
    gate(probed == walkable, f"G2 level {level}: probe walkable {sorted(probed ^ walkable)} differ from grid")

    executed: dict[int, tuple] = {}
    for a in range(N_ACTIONS):
        api.set_state(st0)
        _, r, done = _step(env, a)
        want = successor(agent, a, walkable)
        if want in (line, gem):
            tgt = "line" if want == line else "gem"
            gate(done and np.sign(r) == REWARD_SIGN[tgt], f"G3 level {level} a={a}: expected to reach {tgt}")
            executed[a] = want
        else:
            gate(not done, f"G3 level {level} a={a}: episode ended unexpectedly")
            got = parse_agent(raw_state(api))
            gate(got == want, f"G3 level {level} a={a}: moved to {got}, engine semantics predict {want}")
            executed[a] = got

    for name, tgt, other, key in (("line", line, gem, "A_line"), ("gem", gem, line, "A_gem")):
        dist = bfs(walkable - {other}, tgt)
        a_exec = [a for a in range(N_ACTIONS) if dist.get(executed[a]) == dist[agent] - 1]
        gate(a_exec == rec[key], f"G4 level {level}: executed {name} progress set {a_exec} != stored {rec[key]}")

        api.set_state(st0)
        path, cur = rec[f"path_{name}"], agent
        gate(len(path) == rec[f"d_{name}"], f"G5 level {level}: {name} path length != distance")
        for i, a in enumerate(path):
            cur = successor(cur, a, walkable)
            _, r, done = _step(env, a)
            if i < len(path) - 1:
                gate(not done and parse_agent(raw_state(api)) == cur, f"G5 level {level}: {name} path off at step {i}")
            else:
                gate(done and np.sign(r) == REWARD_SIGN[name] and cur == tgt,
                     f"G5 level {level}: {name} path did not end on {name} at step {len(path)}")
    api.set_state(st0)

    for obj1, obj2, tag in ((SHAPE_EDIT_OBJ, OBJ_GEM, "shape"), (OBJ_LINE, COLOUR_EDIT_OBJ, "colour")):
        env_e = make_env(obj1, obj2, level, 0, "easy", bg == "texture")
        env_e.reset()
        e = state_record(raw_state(state_api(env_e)), level, bg)
        env_e.close()
        gate(all(e[k] == rec[k] for k in ("agent", "line", "gem", "walls")),
             f"G6 level {level}: {tag}-edit world changed agent/targets/walls")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--background", choices=["black", "texture"], required=True)
    ap.add_argument("--ranges", type=str, default="2000:800,5000:1500", help="start:count,...")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--build-id", type=str, default="a")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bg = args.background
    ranges = [tuple(int(v) for v in r.split(":")) for r in args.ranges.split(",")]

    accepted, obs_list, counts = [], [], {"scanned": 0, "conflict": 0, "agent_cells": {}}
    per_level = {k: [] for k in ("level", "d_line", "d_gem", "equidistant", "disjoint", "conflict",
                                 "state_key", "walls_key")}
    t0 = time.time()
    for start, count in ranges:
        for level in range(start, start + count):
            env = make_env(OBJ_LINE, OBJ_GEM, level, 0, "easy", bg == "texture")
            obs0 = env.reset()
            api = state_api(env)
            st0 = api.get_state()
            rec = state_record(raw_state(api), level, bg)
            counts["scanned"] += 1
            key = str(rec["agent"])
            counts["agent_cells"][key] = counts["agent_cells"].get(key, 0) + 1
            for k, v in (("level", level), ("d_line", rec["d_line"] if rec["d_line"] is not None else -1),
                         ("d_gem", rec["d_gem"] if rec["d_gem"] is not None else -1),
                         ("equidistant", rec["equidistant"]), ("disjoint", rec["disjoint"]),
                         ("conflict", rec["conflict"]), ("state_key", rec["state_key"]),
                         ("walls_key", walls_key(rec))):
                per_level[k].append(v)
            if rec["conflict"]:
                validate(rec, env, api, st0, bg)
                rgb = np.asarray(obs0["rgb"] if isinstance(obs0, dict) else obs0).reshape(64, 64, 3)
                rec["obs_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
                accepted.append(rec)
                obs_list.append(rgb)
                counts["conflict"] += 1
            env.close()
            if level % 200 == 0:
                print(f"[{bg}/{args.build_id}] level {level}: accepted {len(accepted)} ({time.time() - t0:.0f}s)", flush=True)

    out = {
        "bank": "ras_conflict_bank_gt_v2", "background": bg, "build_id": args.build_id, "ranges": ranges,
        "builder_code_sha256": builder_code_hash(),
        "counts": counts, "gates_passed": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "bank_sha256": canonical_hash(accepted),
        "layout_sha256": canonical_hash(accepted, LAYOUT_KEYS),
        "records": accepted,
    }
    (args.out / f"ras_gt_bank_{bg}_{args.build_id}.json").write_text(json.dumps(out))
    np.savez_compressed(args.out / f"ras_gt_bank_{bg}_{args.build_id}_levels.npz",
                        **{k: np.array(v) for k, v in per_level.items()})
    np.savez_compressed(args.out / f"ras_gt_bank_{bg}_{args.build_id}_obs.npz",
                        levels=np.array([r["level"] for r in accepted]), obs=np.stack(obs_list))
    print(f"[{bg}/{args.build_id}] DONE accepted={len(accepted)} scanned={counts['scanned']} "
          f"bank_sha256={out['bank_sha256']} layout_sha256={out['layout_sha256']} agent_cells={counts['agent_cells']}",
          flush=True)


if __name__ == "__main__":
    main()
