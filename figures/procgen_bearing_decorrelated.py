"""Maze cheese bearing and Coin coin bearing probes on frames where the goal bearing is decorrelated from the agent's own position."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from paths import DATA_ROOT, ROOT, TABLES

sys.path.insert(0, str(ROOT / "external" / "train-procgen-pytorch"))
sys.path.insert(0, str(ROOT / "scripts" / "procgen"))
sys.modules.setdefault("gym", types.ModuleType("gym"))
from check_maze_probes_grouped_independent import (  # noqa: E402
    layout_groups, make_policy, ridge_auc, site_acts, split_masks)
from run_coinrun_concept_probes_grouped import (  # noqa: E402
    all_site_features, build_policy, load_checkpoint, make_splits, obs_hashes, pixel_features, r2_split)

MAZE_SITES = ["block1_gap", "block2_gap", "block3_gap", "pre_fc", "fc"]
COIN_SITES = ["block1", "block2", "block3_gap", "block3", "fc"]
SITE_NAME = dict(zip(MAZE_SITES, ["h1", "h2", "h3", "h_pre", "h_fc"])) | dict(zip(COIN_SITES, ["h1", "h2", "h3", "h_pre", "h_fc"]))


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def ci95(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def balance_within(key, y, seed):
    """Subsample the majority bearing class inside every key value, so key carries no information about y."""
    rng = np.random.default_rng(seed)
    keep = []
    for u in np.unique(key):
        pos, neg = np.where((key == u) & (y == 1))[0], np.where((key == u) & (y == 0))[0]
        k = min(len(pos), len(neg))
        if k:
            keep += list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, k, replace=False))
    return np.sort(np.array(keep))


def run_maze(args, dev, out):
    z = np.load(args.maze_frames)
    obs = z["obs_u8"].astype(np.float32) / 255.0
    level, agent = z["level"], z["agent_cell"]
    groups = layout_groups(level, z["frame_hash"])
    ip = np.where(z["cheese_present"] == 1)[0]
    y_all = z["y_cheese_bearing"]
    # The bearing label is "cheese right of agent", so agent x alone predicts it (r=-0.54); balancing classes
    # inside every agent cell leaves position with no information about the label, linear or not.
    cell = agent[:, 0] * 100 + agent[:, 1]
    bal = ip[balance_within(cell[ip], y_all[ip], args.balance_seed)]
    stats_ = {"n_present": int(len(ip)), "n_balanced": int(len(bal)), "pos_rate_balanced": float(y_all[bal].mean())}
    for nm, rows in (("full", ip), ("balanced", bal)):
        for j, ax in enumerate("xy"):
            stats_[f"corr_agent_{ax}_{nm}"] = float(np.corrcoef(agent[rows, j], y_all[rows])[0, 1])

    arms = [(f"gm_{t}", "GM", args.maze_ckpt_dir / f"{t}_gm.pth", None) for t in ("1080", "1081", "1082")]
    arms += [(f"nongm_{t}", "Non-GM", args.maze_ckpt_dir / f"{t}_teach.pth", None) for t in ("1090", "1091", "1092")]
    arms += [(f"init_{i}", "init", None, 1000 + i) for i in range(3)]
    feats = {}
    for name, _, ck, s in arms:
        feats[name], _ = site_acts(make_policy(ck, s), obs, dev)
    ax = agent.astype(float)
    refs = {"agent_xy": {"ref": np.stack([ax[:, 0] / 24, ax[:, 1] / 24], 1)},
            "agent_cell_onehot": {"ref": np.concatenate([np.eye(25)[agent[:, 0]], np.eye(25)[agent[:, 1]]], 1)},
            "pixel64": {"ref": obs.reshape(len(obs), -1).astype(np.float64)}}
    sets = [(n, c, feats[n]) for n, c, _, _ in arms] + [(k, "reference", v) for k, v in refs.items()]

    raw = []
    for s in args.seeds:
        test_all = split_masks(groups, s)
        for subset, rows in (("full", ip), ("balanced", bal)):
            te, tr = test_all[rows], ~test_all[rows]
            y = y_all[rows]
            for n, coh, F in sets:
                for site, X in F.items():
                    auc, _ = ridge_auc(X[rows][tr], y[tr], X[rows][te], y[te])
                    raw.append({"env": "maze", "subset": subset, "seed": s, "arm": n, "cohort": coh, "site": site,
                                "score": auc, "n_train": int(tr.sum()), "n_test": int(te.sum())})
        # Global probe fit on all present training frames, scored only on the balanced held-out frames.
        te_full, tr_full = test_all[ip], ~test_all[ip]
        bal_te = np.isin(ip, bal) & te_full
        for n, coh, F in sets:
            for site, X in F.items():
                _, sc = ridge_auc(X[ip][tr_full], y_all[ip][tr_full], X[ip][te_full], y_all[ip][te_full])
                auc = float(roc_auc_score(y_all[ip][bal_te], sc[bal_te[te_full]]))
                raw.append({"env": "maze", "subset": "full_fit_balanced_test", "seed": s, "arm": n, "cohort": coh,
                            "site": site, "score": auc, "n_train": int(tr_full.sum()), "n_test": int(bal_te.sum())})
    return raw, stats_


def run_coin(args, dev, out):
    fr = dict(np.load(args.coin_frames, allow_pickle=False))
    levels = fr["level"].astype(np.int64)
    splits = make_splits(levels, obs_hashes(fr["obs"]), fr["coin_on_screen"].astype(np.int64), args.coin_split_seeds, 0.35)
    bx = fr["coin_bearing_x"].astype(np.float64)
    prog = fr["progress"].astype(np.float64)
    axw, ayw = fr["agent_x"].astype(np.float64), fr["agent_y"].astype(np.float64)
    xb = np.digitize(axw, np.quantile(axw, np.linspace(0, 1, 9)[1:-1]))
    yb = np.digitize(ayw, np.quantile(ayw, np.linspace(0, 1, 5)[1:-1]))
    nuis = {"progress": prog[:, None],
            "progress_agent_xy": np.stack([prog, axw, ayw, fr["level_width"].astype(np.float64)], 1),
            "progress_agent_bins": np.concatenate([prog[:, None], np.eye(8)[xb], np.eye(4)[yb]], 1)}

    def resid(P, tr):
        return bx - Ridge(alpha=1e-3).fit(P[tr], bx[tr]).predict(P)

    nets = [(n, "GM", args.coin_ckpt_dir / f"{n}.pth") for n in ("6033_gm_20M", "6037_gm_120M", "6051_gm_55M")]
    nets += [(n, "Non-GM", args.coin_ckpt_dir / f"{n}.pth") for n in ("7001_teach", "7002_teach", "7003_teach")]
    nets += [(f"random_init_{i}", "init", i) for i in range(3)]
    feats = {}
    for n, coh, p in nets:
        pol = build_policy(None, dev, seed=p) if coh == "init" else load_checkpoint(p, dev)
        f = all_site_features(pol, fr["obs"], dev)
        feats[n] = {k: f[k].astype(np.float64) for k in COIN_SITES}
    refs = {"agent_xy": {"ref": np.stack([axw, ayw], 1)},
            "agent_bins": {"ref": np.concatenate([np.eye(8)[xb], np.eye(4)[yb]], 1)},
            "pixels16": {"ref": pixel_features(fr["obs"], 16).astype(np.float64)}}
    sets = [(n, c, feats[n]) for n, c, _ in nets] + [(k, "reference", v) for k, v in refs.items()]
    raw, stats_ = [], {"n_frames": int(len(bx))}
    for nn, P in nuis.items():
        ys = [resid(P, sp["train"]) for sp in splits]
        stats_[f"corr_agent_x_{nn}"] = float(np.mean([np.corrcoef(axw, y)[0, 1] for y in ys]))
        for k, sp in enumerate(splits):
            for n, coh, F in sets:
                for site, X in F.items():
                    raw.append({"env": "coin", "subset": nn, "seed": sp["seed"], "arm": n, "cohort": coh, "site": site,
                                "score": r2_split(X, ys[k], sp["train"], sp["test"]),
                                "n_train": int(len(sp["train"])), "n_test": int(len(sp["test"]))})
    return raw, stats_


def summarise(raw):
    """Per-network mean over split seeds, then cohort mean, SD and 95% t-CI across networks."""
    keys = sorted({(r["env"], r["subset"], r["cohort"], r["site"]) for r in raw})
    rows = []
    for env, sub, coh, site in keys:
        arms = sorted({r["arm"] for r in raw if (r["env"], r["subset"], r["cohort"], r["site"]) == (env, sub, coh, site)})
        for arm in (arms if coh == "reference" else [None]):
            sel = [a for a in arms if arm is None or a == arm]
            per = [np.mean([r["score"] for r in raw if (r["env"], r["subset"], r["arm"], r["site"]) == (env, sub, a, site)]) for a in sel]
            rows.append({"env": env, "subset": sub, "cohort": coh if arm is None else arm, "site": SITE_NAME.get(site, site),
                         "mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1)) if len(per) > 1 else float("nan"),
                         "ci95": ci95(per), "per_network": ";".join(f"{v:.3f}" for v in per)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maze-frames", type=Path, required=True)
    ap.add_argument("--maze-ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/head_rewire/maze")
    ap.add_argument("--coin-frames", type=Path, required=True)
    ap.add_argument("--coin-ckpt-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--coin-split-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--balance-seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=TABLES / "procgen_bearing_decorrelated")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device()
    raw_m, st_m = run_maze(args, dev, args.out_dir)
    raw_c, st_c = run_coin(args, dev, args.out_dir)
    raw = raw_m + raw_c
    for env in ("maze", "coin"):
        rs = [r for r in raw if r["env"] == env]
        with open(args.out_dir / f"raw_{env}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    summ = summarise(raw)
    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    (args.out_dir / "stats.json").write_text(json.dumps({"maze": st_m, "coin": st_c}, indent=1))
    print(json.dumps({"maze": st_m, "coin": st_c}, indent=1))


if __name__ == "__main__":
    main()
