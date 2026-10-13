"""Maze world-edit KL, signed follow and reach on the shared 128-state conflict sample (cheese and corner edits)."""
# CPU only. Run from the procgen/ directory:
#   python figures/maze_shared_bank_worldedit.py harvest --work <dir>
#   for c in gm_1080 gm_1081 gm_1082 nongm_1090 nongm_1091 nongm_1092; do
#     python figures/maze_shared_bank_worldedit.py score --ckpt $c --work <dir>; done
#   python figures/maze_shared_bank_worldedit.py summarize --work <dir> \
#     --groups <data/probes/maze_edit_action_groups_20260925>
#   ... frames --work <dir> --level 1000003   (example-figure frames of one shared-sample state)
# One process per checkpoint because looping procgen scoring in one process leaks memory.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from paths import DATA_ROOT, ROOT, TABLES

for p in (ROOT / "scripts" / "procgen", ROOT / "external" / "procgenAISC"):
    sys.path.insert(0, str(p))

# maze.cpp zeroes vy whenever vx != 0, so LEFT+UP/LEFT+DOWN move left and 9-14 are no-ops.
DIR_ACTIONS = {1: (0, 1, 2), 3: (3,), 5: (5,), 7: (6, 7, 8)}
CKPTS = {"gm_1080": "1080_gm.pth", "gm_1081": "1081_gm.pth", "gm_1082": "1082_gm.pth",
         "nongm_1090": "1090_teach.pth", "nongm_1091": "1091_teach.pth", "nongm_1092": "1092_teach.pth"}
HORIZON = 150
OUT_CSV = TABLES / "maze_shared128_per_state.csv"
SEED_CSV = TABLES / "maze_shared128_per_seed.csv"


def use_prebuilt_procgen() -> None:
    import procgen.env as pe
    lib = Path(pe.__file__).parent / ".build" / "relwithdebinfo"
    if (lib / "libenv.dylib").exists() or (lib / "libenv.so").exists():
        pe.build = lambda debug=False, package=False: str(lib)


def kl(p, q):
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1.0); q = np.clip(np.asarray(q, np.float64), 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def probs(pol, chw):
    import torch
    x = torch.as_tensor(np.asarray(chw, np.float32)).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(pol.fc_policy(pol.embedder(x)), -1)[0].numpy().astype(np.float64)


def harvest(args) -> None:
    """Harvest the shared bank exactly as maze_edit_action_groups.py and keep each frame's env state."""
    import torch
    use_prebuilt_procgen()
    import run_maze_fisher_decision_fork as fork
    from run_maze_i_goal_proxy_ladder import _reachable_extreme, first_step_action, get_state, load_policy
    torch.set_num_threads(4)
    extra = []
    orig = fork._present_gone_relocated

    def triple(env, cx, cy, grid, agent_cell):
        snap = get_state(env)
        out = orig(env, cx, cy, grid, agent_cell)
        if out is not None:
            bl = _reachable_extreme(grid, agent_cell, "bl")
            extra.append({"state": snap, "a_reloc": first_step_action(grid, agent_cell, bl)[0]})
        return out

    fork._present_gone_relocated = triple
    warm = load_policy(args.ckpt_dir / CKPTS["gm_1080"], "cpu")
    frames, meta = fork.harvest_decision_fork(warm, device="cpu", n_frames=128, start_level=1_000_000,
                                              max_levels=8000, max_steps=200, source_name="gm_rollouts")
    assert len(frames) == len(extra) == 128
    keep = ("level", "t", "obs_a", "obs_reloc", "reloc_xy", "agent_cell", "cheese_xy", "a_cheese", "a_corner")
    rows = [{**{k: f[k] for k in keep}, **e} for f, e in zip(frames, extra)]
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "bank.pkl").write_bytes(pickle.dumps({"rows": rows, "meta": meta}))
    print("harvest", meta, flush=True)


def _seed(label, level, arm):
    return int.from_bytes(hashlib.blake2b(f"{label}|{level}|{arm}".encode(), digest_size=4).digest(), "big")


def rollout(pol, env, snap, rng):
    """Sampled rollout from a restored snapshot; reach := any raw procgen reward > 0, as in run_maze_envmove_reach."""
    from run_maze_i_goal_proxy_ladder import obs_after_edit, set_state
    set_state(env, snap)
    chw = obs_after_edit(env)
    for _ in range(HORIZON):
        p = probs(pol, chw); p = p / p.sum()
        obs, rew, done, _ = env.step(np.array([int(rng.choice(15, p=p))]))
        if float(np.asarray(rew).reshape(-1)[0]) > 0:
            return 1
        if done[0]:
            return 0
        chw = np.asarray(obs[0], np.float32)
    return 0


def score(args) -> None:
    import torch
    use_prebuilt_procgen()
    import run_maze_i_goal_proxy_ladder as M
    torch.set_num_threads(2)
    bank = pickle.loads((args.work / "bank.pkl").read_bytes())["rows"]
    pol = M.load_policy(args.ckpt_dir / CKPTS[args.ckpt], "cpu")
    out = []
    for r in bank:
        env = M.make_env("maze", start_level=r["level"])
        try:
            env.reset()
            M.set_state(env, r["state"])
            obs0 = M.obs_after_edit(env)
            info = M.get_info_maze(env)
            wd = M._iscalar(info, "maze_world_dim")
            grid0 = M._read_grid(env, wd)
            start = M.get_agent_cell(env)
            p0 = probs(pol, r["obs_a"])
            row = {"ckpt": args.ckpt, "level": r["level"], "t": r["t"],
                   "replay_obs_maxdiff": float(np.abs(obs0 - r["obs_a"]).max())}
            # cheese edit: relocation to the reachable bl extreme (the bank's obs_reloc)
            p_c = probs(pol, r["obs_reloc"])
            a = int(r["a_reloc"])
            row.update(a_reloc=a, kl_cheese=kl(p0, p_c), tv_cheese=0.5 * float(np.abs(p_c - p0).sum()),
                       sf_cheese=float((p_c - p0)[list(DIR_ACTIONS[a])].sum()),
                       sf_cheese_single=float(p_c[a] - p0[a]))
            # corner edit: carve to the global top-right extreme, as capture_corner_geometry
            tr = M._extreme_cell(grid0, "tr")
            n_walls, _ = M.carve_path_to(env, tr)
            grid_tr = M._read_grid(env, wd)
            a_post, _ = M.first_step_action(grid_tr, start, tr)
            snap_tr = M.get_state(env)
            p_o = probs(pol, M.obs_after_edit(env))
            row.update(n_walls_opened=n_walls, a_corner_post=-1 if a_post is None else int(a_post),
                       kl_corner=kl(p0, p_o), tv_corner=0.5 * float(np.abs(p_o - p0).sum()),
                       sf_corner=np.nan if a_post is None else float((p_o - p0)[list(DIR_ACTIONS[int(a_post)])].sum()),
                       sf_corner_single=np.nan if a_post is None else float(p_o[a_post] - p0[a_post]))
            # cheese relocation snapshot, built from the base state
            M.set_state(env, r["state"])
            M.set_cheese(env, int(r["reloc_xy"][0]), int(r["reloc_xy"][1]), 1)
            row["reloc_obs_maxdiff"] = float(np.abs(M.obs_after_edit(env) - r["obs_reloc"]).max())
            snap_c = M.get_state(env)
            if not args.no_reach:
                for arm, snap in (("none", r["state"]), ("cheese", snap_c), ("corner", snap_tr)):
                    row[f"reach_{arm}"] = rollout(pol, env, snap, np.random.default_rng(_seed(args.ckpt, r["level"], arm)))
            out.append(row)
        finally:
            env.close()
    args.work.joinpath(f"scored_{args.ckpt}.json").write_text(json.dumps(out))
    print(args.ckpt, "done", len(out), flush=True)


def frames(args) -> None:
    """Original, cheese-moved and top-right-opened frames of one shared-sample state, for the example figure."""
    use_prebuilt_procgen()
    import run_maze_i_goal_proxy_ladder as M
    r = next(r for r in pickle.loads((args.work / "bank.pkl").read_bytes())["rows"] if r["level"] == args.level)
    env = M.make_env("maze", start_level=r["level"])
    env.reset()
    M.set_state(env, r["state"])
    wd = M._iscalar(M.get_info_maze(env), "maze_world_dim")
    M.carve_path_to(env, M._extreme_cell(M._read_grid(env, wd), "tr"))
    hwc = lambda x: np.clip(np.round(np.transpose(x, (1, 2, 0)) * 255.0), 0, 255).astype(np.uint8)
    np.savez_compressed(args.work / "frames_maze_shared128.npz", maze_cheese_before=hwc(r["obs_a"]),
                        maze_cheese_after=hwc(r["obs_reloc"]), maze_corner_after=hwc(M.obs_after_edit(env)))
    (args.work / "frames_maze_shared128.json").write_text(json.dumps(
        {"level": r["level"], "t": r["t"], "agent_cell": list(r["agent_cell"]), "cheese_before": list(r["cheese_xy"]),
         "cheese_after": list(r["reloc_xy"])}, indent=1))
    env.close()


def summarize(args) -> None:
    rows = []
    for c in CKPTS:
        rows += json.loads((args.work / f"scored_{c}.json").read_text())
    ref = {}
    with open(args.groups / "per_state.csv") as f:
        for r in csv.DictReader(f):
            ref[(r["ckpt"], int(r["level"]))] = r
    for r in rows:
        g = ref[(r["ckpt"], r["level"])]
        r["kl_cheese_ref"] = float(g["kl"])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    metrics = [k for k in rows[0] if k.startswith(("kl_", "tv_", "sf_", "reach_"))]
    seeds = []
    for c in CKPTS:
        rr = [r for r in rows if r["ckpt"] == c]
        seeds.append({"ckpt": c, "cohort": "GM" if c.startswith("gm") else "Non-GM", "n": len(rr),
                      **{m: float(np.nanmean([r[m] for r in rr])) for m in metrics if m in rr[0]},
                      "max_rel_kl_cheese_err": float(max(abs(r["kl_cheese"] - r["kl_cheese_ref"]) / max(r["kl_cheese_ref"], 1e-12) for r in rr)),
                      "max_replay_obs_diff": float(max(r["replay_obs_maxdiff"] for r in rr)),
                      "max_reloc_obs_diff": float(max(r["reloc_obs_maxdiff"] for r in rr)),
                      "n_corner_sf": int(np.sum([np.isfinite(r["sf_corner"]) for r in rr]))})
    with open(SEED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seeds[0])); w.writeheader(); w.writerows(seeds)
    for coh in ("GM", "Non-GM"):
        ss = [s for s in seeds if s["cohort"] == coh]
        print(coh, {m: f"{np.mean([s[m] for s in ss]):.4f}+-{np.std([s[m] for s in ss], ddof=1):.4f}"
                    for m in metrics if m in ss[0]})
    print({s["ckpt"]: (s["max_rel_kl_cheese_err"], s["max_replay_obs_diff"], s["max_reloc_obs_diff"]) for s in seeds})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["harvest", "score", "frames", "summarize"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--ckpt", choices=list(CKPTS))
    ap.add_argument("--ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/maze")
    ap.add_argument("--groups", type=Path)
    ap.add_argument("--no-reach", action="store_true")
    ap.add_argument("--level", type=int, default=1_000_003)
    args = ap.parse_args()
    {"harvest": harvest, "score": score, "frames": frames, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
