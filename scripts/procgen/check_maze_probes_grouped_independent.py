#!/usr/bin/env python3
"""Independent check of run_maze_probes_grouped.py: own activations, level-grouped splits, confound controls, pixel cues."""
from __future__ import annotations

import os
import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "train-procgen-pytorch"))
sys.modules.setdefault("gym", types.ModuleType("gym"))
from common.model import ImpalaModel  # noqa: E402
from common.policy import CategoricalPolicy  # noqa: E402

SITES = ["block1_gap", "block2_gap", "block3_gap", "pre_fc", "fc"]
CONCEPTS = ["cheese_present", "cheese_bearing", "corner_bearing", "cheese_bearing_corner1"]
WD_PX = 64 / 25


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def make_policy(ckpt, init_seed):
    # Init draws follow the torch.manual_seed(1000+i) -> CategoricalPolicy(ImpalaModel) order of the reference init arms.
    if init_seed is not None:
        torch.manual_seed(init_seed)
        np.random.seed(init_seed)
    pol = CategoricalPolicy(ImpalaModel(in_channels=3), False, 15)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        pol.load_state_dict(sd)
    return pol.eval()


@torch.no_grad()
def site_acts(pol, obs, device, bs=256):
    """Manual forward through the trunk; asserts the reconstructed logits equal the policy's own."""
    pol.to(device)
    e = pol.embedder
    out = {k: [] for k in SITES}
    max_err = 0.0
    for s in range(0, len(obs), bs):
        x = torch.as_tensor(obs[s:s + bs], dtype=torch.float32, device=device)
        h1 = e.block1(x)
        h2 = e.block2(h1)
        h3 = e.block3(h2)
        r3 = torch.relu(h3)
        pre = r3.flatten(1)
        fc = torch.relu(e.fc(pre))
        logits = pol.fc_policy(fc)
        dist, _, _ = pol(x, torch.zeros(len(x), 256, device=device), torch.zeros(len(x), device=device))
        max_err = max(max_err, float((torch.log_softmax(logits, 1) - dist.logits).abs().max()))
        for k, v in (("block1_gap", h1.mean((2, 3))), ("block2_gap", h2.mean((2, 3))),
                     ("block3_gap", r3.mean((2, 3))), ("pre_fc", pre), ("fc", fc)):
            out[k].append(v.cpu().numpy().astype(np.float64))
    assert max_err < 1e-4, f"reconstructed logits disagree with policy: {max_err}"
    return {k: np.concatenate(v) for k, v in out.items()}, max_err


def ridge_auc(xtr, ytr, xte, yte, alpha=1.0):
    """Closed-form ridge on {0,1} labels, train-centred, alpha fixed; held-out AUC."""
    mu = xtr.mean(0)
    xc = xtr - mu
    d = xc.shape[1]
    if d > len(xc):
        w = xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(len(xc)), ytr - ytr.mean())
    else:
        w = np.linalg.solve(xc.T @ xc + alpha * np.eye(d), xc.T @ (ytr - ytr.mean()))
    s = (xte - mu) @ w
    return float(roc_auc_score(yte, s)), s


def layout_groups(level, hashes):
    """Union levels that share any pixel-identical frame, so duplicate layouts cannot straddle the split."""
    parent = {L: L for L in np.unique(level)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    first = {}
    for L, h in zip(level, hashes):
        if h in first:
            parent[find(L)] = find(first[h])
        else:
            first[h] = L
    return np.array([find(L) for L in level])


def split_masks(groups, seed, frac=0.3):
    rng = np.random.default_rng(7000 + seed)
    g = rng.permutation(np.unique(groups))
    test_g = set(g[: int(round(frac * len(g)))].tolist())
    return np.array([x in test_g for x in groups])


def bins_of(agent, grid, wd=25):
    bx = np.minimum(grid - 1, (agent[:, 0] / (wd - 1) * grid).astype(int))
    by = np.minimum(grid - 1, (agent[:, 1] / (wd - 1) * grid).astype(int))
    return bx * grid + by


def within_bin(y, s, b):
    vals, w = [], []
    for u in np.unique(b):
        m = b == u
        if 0 < y[m].sum() < m.sum():
            vals.append(roc_auc_score(y[m], s[m]))
            w.append(m.sum())
    if not vals:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.average(vals, weights=w)), len(vals)


def location_holdout_masks(bins, frac, seed, y):
    """Position-bin hold-out split, used only to reproduce the paper's Delta."""
    rng = np.random.default_rng(seed)
    ub = sorted(int(b) for b in np.unique(bins) if b >= 0)
    n_hold = max(1, int(round(frac * len(ub))))

    def both(side):
        m = np.isin(bins, list(side))
        return m.any() and y[m].min() < 0.5 < y[m].max()
    hold = None
    for _ in range(128):
        cand = set(int(b) for b in rng.choice(ub, size=n_hold, replace=False))
        if both([b for b in ub if b not in cand]) and both(cand):
            hold = cand
            break
    return ~np.isin(bins, sorted(hold)), np.isin(bins, sorted(hold)), sorted(hold)


def ci95(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, default=DATA_ROOT / "probes/maze_probe_frames_20260924/maze_probe_frames.npz")
    ap.add_argument("--ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/head_rewire/maze")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "probes/maze_probes_grouped_independent_check_20260924")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device()

    z = np.load(args.frames)
    obs_u8 = z["obs_u8"]
    obs = obs_u8.astype(np.float32) / 255.0
    level, agent, cc = z["level"], z["agent_cell"], z["cheese_cell"]
    present = z["cheese_present"].astype(int)
    y_cb, y_co = z["y_cheese_bearing"], z["y_corner_bearing"]
    assert len(level) == 2787 and (present == 0).sum() == 800 and (z["wd"] == 25).all()
    groups = layout_groups(level, z["frame_hash"])
    n_merged = len(np.unique(level)) - len(np.unique(groups))

    ip = np.where(present == 1)[0]
    # corner=1 holds TR bearing fixed, so any cheese-bearing signal left here is cheese-relative, not corner-relative.
    corner1 = ip[y_co[ip] == 1]
    rows_of = {"cheese_present": (np.arange(len(level)), present.astype(float)),
               "cheese_bearing": (ip, y_cb[ip]), "corner_bearing": (np.arange(len(level)), y_co),
               "cheese_bearing_corner1": (corner1, y_cb[corner1])}

    arms = [(f"gm_{t}", "GM", args.ckpt_dir / f"{t}_gm.pth", None) for t in ("1080", "1081", "1082")]
    arms += [(f"corrected_{t}", "Non-GM", args.ckpt_dir / f"{t}_teach.pth", None) for t in ("1090", "1091", "1092")]
    arms += [(f"init_{i}", "init", None, 1000 + i) for i in range(3)]

    patch = np.random.default_rng(42).integers(0, 2, size=len(level)).astype(bool)
    obs_p = obs.copy()
    obs_p[patch, :, 2:10, 2:10] = np.array([1.0, 0.0, 1.0], dtype=np.float32)[:, None, None]

    feats, feats_p, logit_err = {}, {}, {}
    for name, _, ck, s in arms:
        pol = make_policy(ck, s)
        feats[name], logit_err[name] = site_acts(pol, obs, dev)
        feats_p[name], _ = site_acts(pol, obs_p, dev)

    ax = agent.astype(float)
    ref_feats = {
        "agent_xy": np.stack([ax[:, 0] / 24, ax[:, 1] / 24, ax[:, 0], ax[:, 1], np.full(len(ax), 25.0)], 1),
        "agent_x_only": ax[:, :1],
        "agent_cell_onehot": np.concatenate([np.eye(25)[agent[:, 0]], np.eye(25)[agent[:, 1]]], 1),
        "pixel16": torch.nn.functional.avg_pool2d(torch.as_tensor(obs), 4).numpy().reshape(len(obs), -1).astype(np.float64),
        "pixel64": obs.reshape(len(obs), -1).astype(np.float64),
    }

    # --- 1-3: grouped probes ------------------------------------------------------------------
    raw = []
    for s in args.seeds:
        test_all = split_masks(groups, s)
        for c, (idx, y) in rows_of.items():
            te, tr = test_all[idx], ~test_all[idx]
            assert not set(level[idx][te]) & set(level[idx][tr])
            fsets = [(n, coh, feats[n]) for n, coh, _, _ in arms] + [(k, "reference", {k: v}) for k, v in ref_feats.items()]
            for n, coh, F in fsets:
                for site, X in F.items():
                    Xi = X[idx]
                    auc, sc = ridge_auc(Xi[tr], y[tr], Xi[te], y[te])
                    row = {"seed": s, "arm": n, "cohort": coh, "site": site, "concept": c, "auc": auc,
                           "n_train": int(tr.sum()), "n_test": int(te.sum()), "pos_rate_train": float(y[tr].mean()),
                           "pos_rate_test": float(y[te].mean())}
                    for g in (4, 8, 25):
                        b = bins_of(agent[idx][te], g)
                        wb, wbw, nb = within_bin(y[te], sc, b)
                        row[f"wb{g}"], row[f"wb{g}_w"], row[f"wb{g}_nbins"] = wb, wbw, nb
                    raw.append(row)
        with open(args.out_dir / f"raw_seed{s}.csv", "w", newline="") as f:
            rs = [r for r in raw if r["seed"] == s]
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)

    def ckpt_mean(arm, site, c, key="auc"):
        return float(np.nanmean([r[key] for r in raw if r["arm"] == arm and r["site"] == site and r["concept"] == c]))

    summary = {"logit_max_abs_err": logit_err, "n_layout_groups_merged": int(n_merged),
               "n_corner1_frames": int(len(corner1)), "corner1_pos_rate": float(y_cb[corner1].mean()), "cohorts": {}, "reference": {}}
    for c in rows_of:
        for site in SITES:
            for coh in ("GM", "Non-GM", "init"):
                names = [a for a, h, _, _ in arms if h == coh]
                cell = {}
                for key in ("auc", "wb4", "wb8", "wb25", "wb8_w"):
                    v = [ckpt_mean(a, site, c, key) for a in names]
                    cell[key] = {"per_ckpt": v, "mean": float(np.mean(v)), "ci95": ci95(v)}
                summary["cohorts"][f"{c}|{site}|{coh}"] = cell
        for k in ref_feats:
            summary["reference"][f"{c}|{k}"] = {key: ckpt_mean(k, k, c, key) for key in ("auc", "wb4", "wb8", "wb25")}

    # --- producer comparison ------------------------------------------------------------------
    prod = {}
    with open(DATA_ROOT / "probes/maze_probes_grouped_20260924/checkpoint_level.csv") as f:
        for r in csv.DictReader(f):
            prod[(r["arm"], r["site"], r["concept"])] = (float(r["auc"]), float(r["within_bin_auc"]))
    cmp = []
    for n, _, _, _ in arms:
        for site in ("block3_gap", "pre_fc", "fc"):
            for c in ("cheese_present", "cheese_bearing"):
                mine = ckpt_mean(n, site, c)
                cmp.append({"arm": n, "site": site, "concept": c, "mine": mine, "producer": prod[(n, site, c)][0],
                            "abs_diff": abs(mine - prod[(n, site, c)][0]),
                            "wb4_mine": ckpt_mean(n, site, c, "wb4"), "wb4_producer": prod[(n, site, c)][1]})
    for k, pk in (("agent_xy", "agent_xy"), ("pixel16", "pixel16")):
        for c in ("cheese_present", "cheese_bearing", "corner_bearing"):
            p = prod[({"agent_xy": "agent_position", "pixel16": "pixel_floor"}[k], pk, c)]
            cmp.append({"arm": k, "site": k, "concept": c, "mine": ckpt_mean(k, k, c), "producer": p[0],
                        "abs_diff": abs(ckpt_mean(k, k, c) - p[0]), "wb4_mine": ckpt_mean(k, k, c, "wb4"), "wb4_producer": p[1]})
    with open(args.out_dir / "producer_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp[0]))
        w.writeheader()
        w.writerows(cmp)
    summary["producer_cmp_max_abs_diff_ckpt_sites"] = float(max(r["abs_diff"] for r in cmp if r["arm"] not in ref_feats))
    summary["producer_cmp_n_over_0.01"] = int(sum(r["abs_diff"] > 0.01 for r in cmp))

    # --- 4: counterfactual edit check ------------------------------------------------------------
    twin = {(level[i], tuple(agent[i])): i for i in range(len(level)) if present[i] == 0}
    ndiff, outside, win_pos, win_neg, pair_lvl = [], 0, [], [], []
    oi = obs_u8.astype(int)
    for i in ip:
        j = twin[(level[i], tuple(agent[i]))]
        d = (oi[i] != oi[j]).any(0)
        r_, c_ = np.where(d)
        x, yy = cc[i]
        c0, r0 = int(np.floor(x * WD_PX)), int(np.floor((24 - yy) * WD_PX))
        ndiff.append(int(d.sum()))
        outside += int(not (r_.min() >= r0 and r_.max() <= r0 + 3 and c_.min() >= c0 and c_.max() <= c0 + 3))
        r0, c0 = min(r0, 60), min(c0, 60)
        win_pos.append(obs[i][:, r0:r0 + 4, c0:c0 + 4].ravel())
        win_neg.append(obs[j][:, r0:r0 + 4, c0:c0 + 4].ravel())
        pair_lvl.append(groups[i])
    Xw = np.concatenate([np.array(win_pos), np.array(win_neg)]).astype(np.float64)
    yw = np.r_[np.ones(len(win_pos)), np.zeros(len(win_neg))]
    gw = np.r_[pair_lvl, pair_lvl]
    cell_aucs = []
    for s in args.seeds:
        te = split_masks(gw, s)
        cell_aucs.append(ridge_auc(Xw[~te], yw[~te], Xw[te], yw[te])[0])
    # the two sprite colours that never occur in any cheese-absent frame
    palette = [np.array([253, 155, 37]), np.array([254, 231, 98])]
    exact = sum((oi == c[None, :, None, None]).all(1) for c in palette).sum((1, 2))
    tr_mask = z["cheese_is_tr"][ip]
    pix16_by_type = {}
    for s in args.seeds[:5]:
        te = split_masks(groups, s)
        X = ref_feats["pixel16"]
        _, sc = ridge_auc(X[~te], present[~te].astype(float), X[te], present[te].astype(float))
        yt = present[te]
        is_tr_te = z["cheese_is_tr"][te]
        for name, m in (("tr_vs_absent", (yt == 0) | is_tr_te), ("nontr_vs_absent", (yt == 0) | ~is_tr_te)):
            pix16_by_type.setdefault(name, []).append(roc_auc_score(yt[m], sc[m]))
    summary["edit_audit"] = {
        "every_present_has_absent_twin": True,
        "n_pixels_changed_min_median_max": [int(np.min(ndiff)), float(np.median(ndiff)), int(np.max(ndiff))],
        "n_edits_outside_cheese_cell_window": int(outside),
        "cheese_cell_window_probe_auc_mean": float(np.mean(cell_aucs)), "cheese_cell_window_probe_auc_min": float(np.min(cell_aucs)),
        "cheese_palette_pixel_count_auc": float(roc_auc_score(present, exact)),
        "cheese_palette_count_absent_max": int(exact[present == 0].max()),
        "cheese_palette_count_present_min": int(exact[present == 1].min()),
        "pixel16_auc_by_cheese_type": {k: float(np.mean(v)) for k, v in pix16_by_type.items()},
        "frac_present_cheese_is_tr": float(tr_mask.mean()),
    }

    # --- 5: provenance (original position-bin split, patched vs unpatched) ----------------------
    dvi = json.loads((DATA_ROOT / "probes/maze_layerwise_probe_redo_20260913/delta_vs_init_fcfix_20260922.json").read_text())["arms"]
    full = json.loads((DATA_ROOT / "probes/maze_layerwise_probe_redo_20260913/full_fcfix_20260922.json").read_text())["arms"]
    pb = bins_of(agent, 4)
    prov = []
    for c in ("cheese_present", "cheese_bearing", "corner_bearing"):
        idx, y = rows_of[c]
        fit, hold, hb = location_holdout_masks(pb[idx], 0.3, 0, y)
        for site in SITES:
            res = {}
            for tag, F in (("unpatched", feats), ("patched", feats_p)):
                res[tag] = {n: ridge_auc(F[n][site][idx][fit], y[fit], F[n][site][idx][hold], y[hold])[0] for n, _, _, _ in arms}
            for n, coh, _, _ in arms:
                if coh == "init":
                    continue
                init_u = np.mean([res["unpatched"][f"init_{i}"] for i in range(3)])
                init_p = np.mean([res["patched"][f"init_{i}"] for i in range(3)])
                prov.append({"arm": n, "cohort": coh, "site": site, "concept": c, "hold_bins": str(hb),
                             "auc_unpatched": res["unpatched"][n], "auc_ref_bootstrap_file": dvi[n][site][c]["auc_trained_recomputed"],
                             "auc_patched": res["patched"][n], "auc_ref_full_file": full[n][site][c]["auc_held_out"],
                             "delta_unpatched_point": res["unpatched"][n] - init_u, "delta_patched_point": res["patched"][n] - init_p,
                             "paper_delta_mean": dvi[n][site][c]["delta_mean"]})
    with open(args.out_dir / "provenance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov[0]))
        w.writeheader()
        w.writerows(prov)
    P = lambda k1, k2: float(max(abs(r[k1] - r[k2]) for r in prov))
    summary["provenance"] = {
        "max|auc_unpatched - bootstrap_file.auc_trained_recomputed|": P("auc_unpatched", "auc_ref_bootstrap_file"),
        "max|auc_patched - full_file.auc_held_out|": P("auc_patched", "auc_ref_full_file"),
        "max|delta_unpatched_point - paper_delta_mean|": P("delta_unpatched_point", "paper_delta_mean"),
        "max|delta_patched_point - paper_delta_mean|": P("delta_patched_point", "paper_delta_mean"),
        "max|auc_patched - auc_unpatched|": P("auc_patched", "auc_unpatched"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "cohorts"}, indent=2))


if __name__ == "__main__":
    main()
