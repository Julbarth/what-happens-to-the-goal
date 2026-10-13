"""Search for a Maze state shared by all GM checkpoints on which they agree on all four corner-edit arms; save frames + traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paths import DATA_ROOT, ROOT, TABLES

for p in (ROOT / "external/procgenAISC", ROOT / "external/train-procgen-pytorch", ROOT / "scripts" / "procgen"):
    sys.path.insert(0, str(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=int, default=1000000)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--labels", nargs="+", default=["1080_gm", "1081_gm", "1082_gm"])
    ap.add_argument("--min-size", type=int, default=13)
    ap.add_argument("--max-size", type=int, default=17)
    ap.add_argument("--max-t", type=int, default=30)
    ap.add_argument("--min-steps", type=int, default=6, help="shortest accepted path to the arm target")
    ap.add_argument("--rollout-steps", type=int, default=100)
    ap.add_argument("--out", type=Path, default=TABLES / "maze_search")
    a = ap.parse_args()
    import torch
    import maze_corner_open as M
    M.use_prebuilt_procgen()
    import run_maze_i_goal_proxy_ladder as L

    torch.set_num_threads(1)
    # Batch-1 CPU forwards as in the source runs, so greedy argmax ties resolve identically.
    lads = {l: L.Ladder(L.load_policy(DATA_ROOT / f"checkpoints/maze/{l}.pth", "cpu"), "cpu") for l in a.labels}
    a.out.mkdir(parents=True, exist_ok=True)

    def rollout(env, lab):
        traj, obs, done_at = [list(map(int, L.get_agent_cell(env)))], L.obs_after_edit(env), -1
        for k in range(a.rollout_steps):
            _o, _r, done, _i = env.step(np.array([int(lads[lab].run(obs)["probs"][0].argmax())]))
            if done[0]:
                done_at = k + 1
                break
            obs = np.asarray(_o[0], dtype=np.float32)
            traj.append(list(map(int, L.get_agent_cell(env))))
        return traj, done_at

    def first_hit(traj, cell):
        return next((k for k, c in enumerate(traj) if tuple(c) == tuple(cell)), -1)

    def try_state(env, t):
        """All four arms for every checkpoint from the current state; (meta, arrays) if every checkpoint shows the behaviour."""
        info = L.get_info_maze(env)
        wd = L._iscalar(info, "maze_world_dim")
        grid0 = L._read_grid(env, wd)
        ag = tuple(map(int, L.get_agent_cell(env)))
        cheese = [L._iscalar(info, "cheese_x"), L._iscalar(info, "cheese_y")]
        old = L._reachable_extreme(grid0, ag, "tr")
        new = L._extreme_cell(grid0, "tr")
        st = L.get_state(env)
        common = dict(is_walk=L._is_walk, wall=L.WALL_OBJ, bfs_path=L._bfs_path,
                      reach_extreme=L._reachable_extreme, min_sc=3)
        plan_tr = M.build_extension(grid0, ag, old, True, **common)
        bl = L._reachable_extreme(grid0, ag, "bl")
        plan_bl = M.build_extension(grid0, ag, bl, False, **common,
                                    match_len=len(plan_tr["cells"]) if "cells" in plan_tr else None)
        if "target" not in plan_tr or "target" not in plan_bl or tuple(new) == tuple(old):
            return None
        meta = dict(t=t, wd=wd, agent=list(ag), old=list(map(int, old)), new=list(map(int, new)), cheese=cheese,
                    offset_target=list(map(int, plan_tr["target"])), blnull_target=list(map(int, plan_bl["target"])))
        arrays = {}

        def edit(arm):
            L.set_state(env, st)
            if arm == "global":
                return L.carve_path_to(env, new)[1]
            if arm in ("offset", "blnull"):
                cells = (plan_tr if arm == "offset" else plan_bl)["cells"]
                for x, y in cells:
                    L.set_maze_obj(env, x, y, L.SPACE_OBJ)
                return cells
            return []

        for arm in ("base", "global", "offset", "blnull"):
            cells = edit(arm)
            arrays[f"{arm}_cells"] = np.asarray(cells).reshape(-1, 2)
            arrays[f"{arm}_frame"] = L.obs_after_edit(env)
            target = {"base": old, "global": new, "offset": plan_tr["target"], "blnull": old}[arm]
            for lab in a.labels:
                edit(arm)
                tr, dn = rollout(env, lab)
                h = first_hit(tr, target)
                ho = first_hit(tr, old) if arm in ("global", "offset") else -1
                hbl = first_hit(tr, plan_bl["cells"][0]) if arm == "blnull" else -1
                # Shown behaviour: reach the arm's target before the cheese ends the episode, without passing the old corner or entering the BL corridor.
                ok = h >= a.min_steps and (dn < 0 or dn > h) and (ho < 0 or ho > h) and hbl < 0
                meta[f"{arm}_{lab}"] = {"hit": h, "hit_old": ho, "hit_bl": hbl, "done": dn, "ok": bool(ok)}
                arrays[f"{arm}_traj_{lab}"] = np.asarray(tr)
                if not ok:
                    L.set_state(env, st)
                    return None
        L.set_state(env, st)
        return meta, arrays

    summary = []
    for lv in range(a.first, a.first + a.n):
        env = L.make_env("maze", start_level=lv)
        env.reset()
        info = L.get_info_maze(env)
        wd = L._iscalar(info, "maze_world_dim")
        grid0 = L._read_grid(env, wd)
        walk = np.argwhere(np.vectorize(L._is_walk)(grid0))
        size = int(walk[:, 0].max() - walk[:, 0].min() + 1)
        rec = {"level": lv, "size": size}
        if not (a.min_size <= size <= a.max_size):
            summary.append(rec)
            print(json.dumps(rec), flush=True)
            env.close()
            continue
        # Shared prefix: steps on which all checkpoints pick the same greedy action from the same state.
        obs, prefix = L.obs_after_edit(env), 0
        st0 = L.get_state(env)
        for k in range(a.max_t):
            acts = {int(lads[l].run(obs)["probs"][0].argmax()) for l in a.labels}
            if len(acts) > 1:
                break
            _o, _r, done, _i = env.step(np.array([acts.pop()]))
            if done[0]:
                break
            obs = np.asarray(_o[0], dtype=np.float32)
            prefix = k + 1
        rec["prefix"] = prefix
        rec["all_ok"] = False
        for t in range(2, prefix + 1, 2):
            L.set_state(env, st0)
            o = L.obs_after_edit(env)
            for _ in range(t):
                _o, _r, done, _i = env.step(np.array([int(lads[a.labels[0]].run(o)["probs"][0].argmax())]))
                o = np.asarray(_o[0], dtype=np.float32)
            r = try_state(env, t)
            if r is not None:
                rec.update(r[0], all_ok=True)
                np.savez_compressed(a.out / f"L{lv}.npz", **r[1], meta=np.array(json.dumps(rec)))
                break
        print(json.dumps({k: rec[k] for k in ("level", "size", "all_ok")}), flush=True)
        summary.append(rec)
        env.close()
    (a.out / "summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
