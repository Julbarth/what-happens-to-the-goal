#!/usr/bin/env python3
"""Maze availability probes (Fig 4 / Table 4 / A.4) on a level-grouped, position-bin-stratified split, gated on reproducing reference AUCs."""
from __future__ import annotations

import os
import argparse
import csv
import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "external" / "train-procgen-pytorch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
# common/misc_util.py imports gym only for a logger helper; gym may not be installed and
# nothing on the model/policy path touches it.
sys.modules.setdefault("gym", types.ModuleType("gym"))

from common.model import ImpalaModel  # noqa: E402
from common.policy import CategoricalPolicy  # noqa: E402
from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc  # noqa: E402

LAYERS = ["block1_gap", "block2_gap", "block3_gap", "pre_fc", "fc"]
CONCEPTS = ["cheese_present", "cheese_bearing", "corner_bearing"]
SENSITIVITY = ["corner_bearing_trainbal"]
BIN_GRID = 4
PATCH_ROWS = slice(2, 10)
PATCH_COLS = slice(2, 10)
PATCH_COLOR = np.array([1.0, 0.0, 1.0], dtype=np.float32)
ACTION_SIZE = 15
N_FOLDS = 10

# Phase 1 (gate) scores magenta-patched frames with the position-bin split and fixed ridge and requires every held-out
# AUC to match --gate-ref within --gate-tol. Phase 2 uses unpatched frames with level-disjoint StratifiedGroupKFold splits
# (strata = position bin x label), drops test frames identical to a train frame, and permutes labels at level granularity.


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def build_policy(ckpt: Path | None, init_seed: int | None) -> CategoricalPolicy:
    """Same construction as load_impala_policy[_from_env] (hard-500: recurrent=False, 3 channels, 15 actions)."""
    if init_seed is not None:
        torch.manual_seed(init_seed)
        np.random.seed(init_seed)
    policy = CategoricalPolicy(ImpalaModel(in_channels=3), False, ACTION_SIZE)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        policy.load_state_dict(state)
    policy.eval()
    return policy


class MultiLayerCapture:
    """Same hooks as maze_probe_redo.MultiLayerCapture: GAP(h1), GAP(h2), GAP(ReLU(h3)), fc input, ReLU(fc)."""

    def __init__(self, policy):
        emb = policy.embedder
        self.policy = policy
        self.buf: dict[str, np.ndarray] = {}
        self.handles = [
            emb.block1.register_forward_hook(self._hook("block1_gap", relu=False)),
            emb.block2.register_forward_hook(self._hook("block2_gap", relu=False)),
            emb.block3.register_forward_hook(self._hook("block3_gap", relu=True)),
            emb.fc.register_forward_pre_hook(lambda _m, inp: self.buf.__setitem__("pre_fc", inp[0].detach().cpu().numpy())),
            emb.fc.register_forward_hook(lambda _m, _i, out: self.buf.__setitem__("fc", torch.relu(out).detach().cpu().numpy())),
        ]

    def _hook(self, name, relu):
        def fn(_m, _i, out):
            o = torch.relu(out) if relu else out
            self.buf[name] = o.detach().mean(dim=(2, 3)).cpu().numpy()
        return fn

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def activations(policy, obs: np.ndarray, device, batch_size: int = 128) -> dict[str, np.ndarray]:
    policy.to(device)
    cap = MultiLayerCapture(policy)
    out: dict[str, list] = {k: [] for k in LAYERS}
    try:
        for s in range(0, len(obs), batch_size):
            x = torch.as_tensor(obs[s:s + batch_size], dtype=torch.float32, device=device)
            hidden = torch.zeros(x.shape[0], policy.embedder.output_dim, device=device)
            masks = torch.zeros(x.shape[0], device=device)
            cap.buf = {}
            policy(x, hidden, masks)
            for k in LAYERS:
                out[k].append(cap.buf[k].copy())
    finally:
        cap.close()
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def pixel_features(obs: np.ndarray, size: int = 16) -> np.ndarray:
    t = torch.nn.functional.interpolate(torch.as_tensor(obs, dtype=torch.float32), size=(size, size), mode="area")
    return t.numpy().reshape(len(t), -1)


def location_holdout_masks(location_bin, *, holdout_frac=0.3, seed=0, y_binary=None):
    """Verbatim copy of collect_maze_i_cheese_probe_grid.location_holdout_masks (that module imports procgen)."""
    rng = np.random.default_rng(seed)
    bins = sorted(int(b) for b in np.unique(location_bin) if b >= 0)
    n_hold = max(1, int(round(holdout_frac * len(bins))))

    def _both(side):
        if y_binary is None or not side:
            return True
        m = np.isin(location_bin, list(side))
        if not m.any():
            return False
        v = y_binary[m]
        return bool(v.min() < 0.5 and v.max() > 0.5)

    hold_bins = None
    for _ in range(128):
        cand = set(int(b) for b in rng.choice(bins, size=n_hold, replace=False))
        if _both([b for b in bins if b not in cand]) and _both(cand):
            hold_bins = cand
            break
    if hold_bins is None:
        hold_bins = set(int(b) for b in rng.choice(bins, size=n_hold, replace=False))
    fit_bins = [b for b in bins if b not in hold_bins]
    fit = np.isin(location_bin, fit_bins) | (location_bin < 0)
    hold = np.isin(location_bin, sorted(hold_bins))
    return fit, hold, fit_bins, sorted(hold_bins)


def ridge_scores_batched(xtr, ytr, xte, alpha=1.0):
    """fit_ridge for many label columns at once (float64); returns test scores (n_te, k)."""
    xtr = np.asarray(xtr, dtype=np.float64)
    mu = xtr.mean(axis=0, keepdims=True)
    xc = xtr - mu
    ym = ytr.mean(axis=0, keepdims=True)
    a = xc.T @ xc + alpha * np.eye(xtr.shape[1])
    w = np.linalg.solve(a, xc.T @ (ytr - ym))
    return (np.asarray(xte, dtype=np.float64) - mu) @ w + ym


def auc_columns(y, s):
    """Mann-Whitney AUC per score column with average tie ranks (same statistic as roc_auc)."""
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    r = rankdata(s, axis=0)
    return (r[pos].sum(axis=0) - npos * (npos + 1) / 2) / (npos * nneg)


def within_bin_auc(y, s, bins):
    vals, n_frames = [], 0
    for b in np.unique(bins):
        m = bins == b
        if 0 < y[m].sum() < m.sum():
            vals.append(roc_auc(y[m], s[m]))
            n_frames += int(m.sum())
    return (float(np.mean(vals)) if vals else float("nan")), len(vals), n_frames


def agent_position_bin(agent_cell, wd, grid=BIN_GRID):
    denom = np.maximum(1, wd - 1)
    bx = np.minimum(grid - 1, (agent_cell[:, 0] / denom * grid).astype(int))
    by = np.minimum(grid - 1, (agent_cell[:, 1] / denom * grid).astype(int))
    return bx * grid + by


def grouped_split(level, bins, y, seed, holdout_frac):
    """Level-disjoint split stratified on (position bin, label); holds out round(holdout_frac*10) of 10 folds."""
    strata = bins * 2 + y.astype(int)
    fold = np.empty(len(y), dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), strata, groups=level)):
            fold[te] = k
    return fold < int(round(holdout_frac * N_FOLDS))


def level_permuted_labels(y_tr, lev_tr, n, rng):
    """(n_tr, n) label matrix: each train level gets a random donor level's labels, randomly ordered."""
    levels = np.unique(lev_tr)
    idx = {L: np.where(lev_tr == L)[0] for L in levels}
    out = np.empty((len(y_tr), n))
    for k in range(n):
        donors = rng.permutation(levels)
        for L, D in zip(levels, donors):
            src = y_tr[idx[D]]
            m = len(idx[L])
            out[idx[L], k] = rng.permutation(src)[:m] if len(src) >= m else rng.choice(src, m, replace=True)
    return out


def load_frames(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["obs"] = d["obs_u8"].astype(np.float32) / 255.0
    recomputed = agent_position_bin(d["agent_cell"], d["wd"])
    assert np.array_equal(recomputed, d["pos_bin"]), "pos_bin in dump disagrees with agent_position_bin"
    return d


def concept_rows(fr: dict) -> dict:
    n = len(fr["level"])
    present = fr["cheese_present"].astype(bool)
    idx_c = np.where(present)[0]
    return {
        "cheese_present": (np.arange(n), present.astype(np.float64)),
        "cheese_bearing": (idx_c, fr["y_cheese_bearing"][idx_c]),
        "corner_bearing": (np.arange(n), fr["y_corner_bearing"]),
    }


def arm_list(args):
    arms = [(f"gm_{t}", "GM", args.ckpt_dir / f"{t}_gm.pth", None) for t in args.gm_tags]
    arms += [(f"corrected_{t}", "Non-GM", args.ckpt_dir / f"{t}_teach.pth", None) for t in args.nongm_tags]
    arms += [(f"init_{i}", "init", None, 1000 + i) for i in range(args.n_init_seeds)]
    return arms


def run_gate(args, fr, arms, device, out_dir):
    ref = json.loads(args.gate_ref.read_text())
    boot = json.loads(args.gate_boot_ref.read_text()) if args.gate_boot_ref.exists() else {"arms": {}}
    n = len(fr["level"])
    patch = np.random.default_rng(42).integers(0, 2, size=n).astype(bool)
    obs_p = fr["obs"].copy()
    for i in np.where(patch)[0]:
        obs_p[i][:, PATCH_ROWS, PATCH_COLS] = PATCH_COLOR[:, None, None]
    cd = concept_rows(fr)
    cd["patch_present"] = (np.arange(n), patch.astype(np.float64))
    bins = fr["pos_bin"]
    splits = {}
    for c, (idx, y) in cd.items():
        fit, hold, _, hold_bins = location_holdout_masks(bins[idx], holdout_frac=0.3, seed=0, y_binary=y)
        splits[c] = (fit, hold, hold_bins)

    def score(X, c, with_null=True):
        idx, y = cd[c]
        fit, hold, _ = splits[c]
        Xc = X[idx]
        w, b = fit_ridge(Xc[fit], y[fit], alpha=1.0)
        auc = float(roc_auc(y[hold], predict_ridge(Xc[hold], w, b)))
        out = {"auc": auc, "n_fit": int(fit.sum()), "n_hold": int(hold.sum()),
               "pos_rate_fit": float(y[fit].mean()), "pos_rate_hold": float(y[hold].mean())}
        if with_null:
            rng = np.random.default_rng(0)
            ys = np.stack([rng.permutation(y[fit]) for _ in range(128)], axis=1)
            null = auc_columns(y[hold], ridge_scores_batched(Xc[fit], ys, Xc[hold]))
            out["null_percentile"] = float((null <= auc).mean())
        return out

    cells = []
    pix = pixel_features(obs_p)
    for c in cd:
        s = score(pix, c, with_null=False)
        r = ref["pixel_floor"][c]
        cells.append({"arm": "pixel_floor", "layer": "pixel16", "concept": c, "auc_new": s["auc"],
                      "auc_ref": r["auc_held_out"], "abs_diff": abs(s["auc"] - r["auc_held_out"]),
                      "n_fit_match": s["n_fit"] == r["n_fit"] and s["n_hold"] == r["n_hold"]})
    acts_unpatched = {}
    for name, cohort, ckpt, init_seed in arms:
        pol = build_policy(ckpt, init_seed)
        acts_p = activations(pol, obs_p, device)
        acts_u = activations(pol, fr["obs"], device)
        acts_unpatched[name] = acts_u
        for layer in LAYERS:
            for c in cd:
                s = score(acts_p[layer], c)
                r = ref["arms"][name][layer][c]
                row = {"arm": name, "layer": layer, "concept": c, "auc_new": s["auc"], "auc_ref": r["auc_held_out"],
                       "abs_diff": abs(s["auc"] - r["auc_held_out"]),
                       "pct_new": s["null_percentile"], "pct_ref": r["shuffle_null_percentile"],
                       "n_fit_match": s["n_fit"] == r["n_fit"] and s["n_hold"] == r["n_hold"]
                       and abs(s["pos_rate_fit"] - r["pos_rate_fit"]) < 1e-9}
                b = boot["arms"].get(name, {}).get(layer, {}).get(c)
                if b is not None:
                    su = score(acts_u[layer], c, with_null=False)
                    row["auc_unpatched_new"] = su["auc"]
                    row["auc_unpatched_ref"] = b["auc_trained_recomputed"]
                    row["abs_diff_unpatched"] = abs(su["auc"] - b["auc_trained_recomputed"])
                cells.append(row)
        print(f"[gate] {name}: max |dAUC| so far {max(x['abs_diff'] for x in cells):.2e}", flush=True)
        del acts_p

    diffs = np.array([x["abs_diff"] for x in cells])
    ud = np.array([x["abs_diff_unpatched"] for x in cells if "abs_diff_unpatched" in x])
    pd_ = np.array([abs(x["pct_new"] - x["pct_ref"]) for x in cells if "pct_new" in x])
    summary = {
        "n_cells": len(cells), "tol": args.gate_tol,
        "max_abs_diff_auc": float(diffs.max()), "median_abs_diff_auc": float(np.median(diffs)),
        "n_cells_over_tol": int((diffs > args.gate_tol).sum()),
        "all_split_counts_match": bool(all(x["n_fit_match"] for x in cells)),
        "unpatched_vs_bootstrap_n": int(len(ud)),
        "unpatched_vs_bootstrap_max_abs_diff": float(ud.max()) if len(ud) else None,
        "null_percentile_max_abs_diff": float(pd_.max()), "null_percentile_mean_abs_diff": float(pd_.mean()),
        "hold_bins_seed0": {c: splits[c][2] for c in splits},
        "passed": bool(diffs.max() <= args.gate_tol and all(x["n_fit_match"] for x in cells)),
    }
    (out_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "gate_cells.csv", "w", newline="") as f:
        keys = sorted({k for x in cells for k in x})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(cells)
    print(json.dumps(summary, indent=2), flush=True)
    return summary, acts_unpatched


def run_grouped(args, fr, arms, acts, out_dir):
    cd = concept_rows(fr)
    level, bins, hashes = fr["level"], fr["pos_bin"], fr["frame_hash"]
    ax = fr["agent_cell"].astype(np.float64)
    wd = fr["wd"].astype(np.float64)
    pos_feats = np.stack([ax[:, 0] / (wd - 1), ax[:, 1] / (wd - 1), ax[:, 0], ax[:, 1], wd], axis=1)
    pix = pixel_features(fr["obs"])
    per_seed_dir = out_dir / "per_seed"
    per_seed_dir.mkdir(exist_ok=True)

    splits, balance = {}, []
    for c in CONCEPTS + SENSITIVITY:
        base = "corner_bearing" if c == "corner_bearing_trainbal" else c
        idx, y = cd[base]
        for s in args.split_seeds:
            test = grouped_split(level[idx], bins[idx], y, s, args.holdout_frac)
            train = ~test
            if c == "corner_bearing_trainbal":
                rng = np.random.default_rng(10_000 + s)
                tr = np.where(train)[0]
                pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
                k = min(len(pos), len(neg))
                keep = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
                train = np.zeros_like(train)
                train[keep] = True
            dup = test & np.isin(hashes[idx], hashes[idx][~test])
            test_kept = test & ~dup
            lv_tr, lv_te = set(level[idx][~test]), set(level[idx][test])
            assert not (lv_tr & lv_te), "level on both sides"
            null_y = level_permuted_labels(y[train], level[idx][train], args.n_shuffles,
                                           np.random.default_rng(args.null_seed + 1000 * s + CONCEPTS.index(base)))
            splits[(c, s)] = (train, test_kept, null_y)
            tb = np.bincount(bins[idx][train], minlength=16) / train.sum()
            eb = np.bincount(bins[idx][test_kept], minlength=16) / test_kept.sum()
            balance.append({
                "concept": c, "split_seed": s, "n_train": int(train.sum()), "n_train_pos": int(y[train].sum()),
                "n_train_neg": int((1 - y[train]).sum()), "pos_rate_train": float(y[train].mean()),
                "n_test_before_dedupe": int(test.sum()), "n_test_dup_dropped": int(dup.sum()),
                "n_test": int(test_kept.sum()), "n_test_pos": int(y[test_kept].sum()),
                "n_test_neg": int((1 - y[test_kept]).sum()), "pos_rate_test": float(y[test_kept].mean()),
                "n_levels_train": len(lv_tr), "n_levels_test": len(lv_te),
                "pos_bin_tv_distance": float(0.5 * np.abs(tb - eb).sum()),
            })
    with open(out_dir / "split_balance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(balance[0]))
        w.writeheader()
        w.writerows(balance)

    def probe(X, c, s):
        base = "corner_bearing" if c == "corner_bearing_trainbal" else c
        idx, y = cd[base]
        train, test, null_y = splits[(c, s)]
        Xi = X[idx]
        w_, b_ = fit_ridge(Xi[train], y[train], alpha=1.0)
        sc = predict_ridge(Xi[test], w_, b_)
        auc = float(roc_auc(y[test], sc))
        wb, nb, nbf = within_bin_auc(y[test], sc, bins[idx][test])
        null = auc_columns(y[test], ridge_scores_batched(Xi[train], null_y, Xi[test]))
        return auc, wb, nb, nbf, null

    rows, nulls = [], {}
    feature_sets = [(name, cohort, acts[name]) for name, cohort, _, _ in arms]
    feature_sets += [("pixel_floor", "reference", {"pixel16": pix}), ("agent_position", "reference", {"agent_xy": pos_feats})]
    for s in args.split_seeds:
        seed_rows = []
        for name, cohort, A in feature_sets:
            for layer, X in A.items():
                for c in CONCEPTS + SENSITIVITY:
                    auc, wb, nb, nbf, null = probe(X, c, s)
                    nulls[f"{name}|{layer}|{c}|{s}"] = null
                    seed_rows.append({"arm": name, "cohort": cohort, "site": layer, "concept": c, "split_seed": s,
                                      "auc": auc, "within_bin_auc": wb, "n_within_bins": nb,
                                      "n_within_bin_frames": nbf, "null_mean": float(null.mean()),
                                      "null_p95": float(np.quantile(null, 0.95)),
                                      "null_percentile": float((null <= auc).mean())})
        with open(per_seed_dir / f"raw_split_seed{s}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
            w.writeheader()
            w.writerows(seed_rows)
        rows += seed_rows
        print(f"[grouped] split seed {s} done", flush=True)
    np.savez_compressed(out_dir / "null_draws.npz", **{k.replace("|", "__"): v for k, v in nulls.items()})
    return rows, nulls, balance


def aggregate(args, rows, nulls, out_dir):
    """Checkpoint level: average AUC over split seeds; null = split-averaged permutation draws."""
    keys = sorted({(r["arm"], r["cohort"], r["site"], r["concept"]) for r in rows})
    ck = []
    for arm, cohort, site, c in keys:
        rr = [r for r in rows if (r["arm"], r["site"], r["concept"]) == (arm, site, c)]
        auc = float(np.mean([r["auc"] for r in rr]))
        null = np.mean([nulls[f"{arm}|{site}|{c}|{s}"] for s in args.split_seeds], axis=0)
        ck.append({"arm": arm, "cohort": cohort, "site": site, "concept": c, "auc": auc,
                   "auc_split_sd": float(np.std([r["auc"] for r in rr], ddof=1)),
                   "within_bin_auc": float(np.nanmean([r["within_bin_auc"] for r in rr])),
                   "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
                   "null_percentile": float((null <= auc).mean()), "above_p95": bool(auc > np.quantile(null, 0.95))})
    with open(out_dir / "checkpoint_level.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ck[0]))
        w.writeheader()
        w.writerows(ck)
    return ck


def summarise(ck, out_dir):
    from scipy import stats
    t = float(stats.t.ppf(0.975, 2))
    get = lambda arm_pred, site, c: [x for x in ck if arm_pred(x["arm"]) and x["site"] == site and x["concept"] == c]
    ref = lambda name, c, key: [x for x in ck if x["arm"] == name and x["concept"] == c][0][key]
    cells = []
    for c in CONCEPTS + SENSITIVITY:
        for site in LAYERS:
            init = get(lambda a: a.startswith("init_"), site, c)
            init_auc = np.array([x["auc"] for x in init])
            for cohort, pref in (("GM", "gm_"), ("Non-GM", "corrected_")):
                tr = get(lambda a: a.startswith(pref), site, c)
                a = np.array([x["auc"] for x in tr])
                d = a - init_auc.mean()
                cells.append({
                    "concept": c, "site": site, "cohort": cohort,
                    "ckpts": [x["arm"] for x in tr], "auc_ckpt": a.tolist(),
                    "auc_mean": float(a.mean()), "auc_sd": float(a.std(ddof=1)), "auc_ci95": float(t * a.std(ddof=1) / np.sqrt(3)),
                    "init_draws": init_auc.tolist(), "init_mean": float(init_auc.mean()), "init_sd": float(init_auc.std(ddof=1)),
                    "delta_ckpt": d.tolist(), "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
                    "delta_ci95": float(t * d.std(ddof=1) / np.sqrt(3)),
                    "null_p95_ckpt": [x["null_p95"] for x in tr], "null_percentile_ckpt": [x["null_percentile"] for x in tr],
                    "null_mean": float(np.mean([x["null_mean"] for x in tr])),
                    "n_above_p95": int(sum(x["above_p95"] for x in tr)),
                    "decodable": bool(all(x["above_p95"] for x in tr)),
                    "init_n_above_p95": int(sum(x["above_p95"] for x in init)),
                    "within_bin_ckpt": [x["within_bin_auc"] for x in tr],
                    "within_bin_mean": float(np.mean([x["within_bin_auc"] for x in tr])),
                    "within_bin_sd": float(np.std([x["within_bin_auc"] for x in tr], ddof=1)),
                    "within_bin_init_mean": float(np.mean([x["within_bin_auc"] for x in init])),
                    "pixel_floor_auc": ref("pixel_floor", c, "auc"),
                    "agent_position_auc": ref("agent_position", c, "auc"),
                    "agent_position_within_bin_auc": ref("agent_position", c, "within_bin_auc"),
                })
    (out_dir / "cells.json").write_text(json.dumps(cells, indent=2))
    return cells


def fmt(x):
    return "nan" if not np.isfinite(x) else (f"{x:.2e}" if 0 < abs(x) < 0.01 else f"{x:.3f}")


def write_table(cells, balance, out_dir, gate):
    site_name = {"block1_gap": "block1 (GAP h1, 16-d)", "block2_gap": "block2 (GAP h2, 32-d)",
                 "block3_gap": "block3 (GAP ReLU h3, 32-d)", "pre_fc": "h_pre (2048-d)", "fc": "h_fc (256-d)"}
    L = ["# Maze probes, level-grouped split (5 split seeds averaged per checkpoint)", "",
         f"Gate: reproduced full_fcfix_20260922.json position-bin AUCs, max |diff| = {fmt(gate['max_abs_diff_auc'])} "
         f"over {gate['n_cells']} cells (tol {gate['tol']}).", "",
         "AUC and Delta are mean ± SD across 3 checkpoints (ddof=1); the 95% CI half-width (t, df=2) is in its own column. "
         "Null p95 = checkpoint mean of the split-averaged level-permutation null's 95th percentile. "
         "Decodable = all 3 checkpoints above their own null p95. "
         "Within-bin = held-out AUC computed inside each agent-position bin with both classes, averaged over bins.", ""]
    for c in CONCEPTS + SENSITIVITY:
        L += [f"## {c}", "", "| site | cohort | AUC trained (mean ± SD) | AUC 95% CI ± | 3 ckpts | init (mean ± SD) | ΔAUC (mean ± SD) | ΔAUC 95% CI ± | null p95 | null pct (3 ckpts) | decodable | within-bin trained (mean ± SD) | within-bin init | agent-pos AUC / within-bin |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in [x for x in cells if x["concept"] == c]:
            L.append(
                f"| {site_name[x['site']]} | {x['cohort']} | {fmt(x['auc_mean'])} ± {fmt(x['auc_sd'])} | {fmt(x['auc_ci95'])} | "
                f"{', '.join(fmt(v) for v in x['auc_ckpt'])} | {fmt(x['init_mean'])} ± {fmt(x['init_sd'])} | "
                f"{fmt(x['delta_mean'])} ± {fmt(x['delta_sd'])} | {fmt(x['delta_ci95'])} | {fmt(float(np.mean(x['null_p95_ckpt'])))} | "
                f"{', '.join(fmt(v) for v in x['null_percentile_ckpt'])} | {'yes' if x['decodable'] else 'no'} ({x['n_above_p95']}/3) | "
                f"{fmt(x['within_bin_mean'])} ± {fmt(x['within_bin_sd'])} | {fmt(x['within_bin_init_mean'])} | "
                f"{fmt(x['agent_position_auc'])} / {fmt(x['agent_position_within_bin_auc'])} |")
        L.append("")
    L += ["## Reference probes (input-level baselines, same grouped splits unless noted)", "",
          "The presence edit changes only 2–5 pixels, all inside the cheese cell's 4×4 pixel window. "
          "The 16-cell pixel-average probe is therefore a deliberately weak linear baseline, not a floor on what the input contains.", "",
          "| reference | cheese presence | cheese bearing | corner bearing | source |", "|---|---|---|---|---|"]
    rc = {c: next(x for x in cells if x["concept"] == c) for c in CONCEPTS}
    L.append("| 16-cell pixel-average probe (16×16 area-pooled pixels, 768-d) | "
             + " | ".join(fmt(rc[c]["pixel_floor_auc"]) for c in CONCEPTS) + " | this run |")
    L.append("| agent position (x, y, x/(wd−1), y/(wd−1), wd) | "
             + " | ".join(fmt(rc[c]["agent_position_auc"]) for c in CONCEPTS) + " | this run |")
    L.append("| agent position, within-bin | "
             + " | ".join(fmt(rc[c]["agent_position_within_bin_auc"]) for c in CONCEPTS) + " | this run |")
    L.append("| cheese-cell raw-pixel probe (48 raw pixels of the cheese cell, linear ridge) | 0.988 | – | – | independent check, 10 grouped splits (min 0.982) |")
    L.append("| sprite-colour count (pixels of the two cheese sprite colours) | 1.000 | – | – | independent check |")
    L.append("")
    L += ["Corner-bearing-fixed test (independent check): cheese bearing on the subset with corner bearing = 1 "
          "(1,498 frames, 65% positive), where the corner-relative answer is constant.", "",
          "| site | GM | Non-GM | init | agent (x, y) |", "|---|---|---|---|---|",
          "| block3 | 0.668 | 0.942 | 0.491 | 0.703 |", "| h_fc | 0.646 | 0.948 | 0.614 | 0.703 |", ""]
    L += ["## Class balance per split", "", "| concept | seed | train n (pos/neg) | pos rate train | test n (pos/neg) | pos rate test | dup dropped | levels train/test | bin TV |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b in balance:
        L.append(f"| {b['concept']} | {b['split_seed']} | {b['n_train']} ({b['n_train_pos']}/{b['n_train_neg']}) | {b['pos_rate_train']:.3f} | "
                 f"{b['n_test']} ({b['n_test_pos']}/{b['n_test_neg']}) | {b['pos_rate_test']:.3f} | {b['n_test_dup_dropped']} | "
                 f"{b['n_levels_train']}/{b['n_levels_test']} | {b['pos_bin_tv_distance']:.3f} |")
    (out_dir / "table_mean_sd.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=Path, default=DATA_ROOT / "probes/maze_probe_frames_20260924/maze_probe_frames.npz")
    ap.add_argument("--ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/head_rewire/maze")
    ap.add_argument("--gm-tags", nargs="+", default=["1080", "1081", "1082"])
    ap.add_argument("--nongm-tags", nargs="+", default=["1090", "1091", "1092"])
    ap.add_argument("--n-init-seeds", type=int, default=3)
    ap.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--n-shuffles", type=int, default=128)
    ap.add_argument("--null-seed", type=int, default=0)
    ap.add_argument("--gate-ref", type=Path, default=DATA_ROOT / "probes/maze_layerwise_probe_redo_20260913/full_fcfix_20260922.json")
    ap.add_argument("--gate-boot-ref", type=Path, default=DATA_ROOT / "probes/maze_layerwise_probe_redo_20260913/delta_vs_init_fcfix_20260922.json")
    ap.add_argument("--gate-tol", type=float, default=1e-3)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--table-only", action="store_true", help="rewrite table_mean_sd.md from saved cells.json / split_balance.csv")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "probes/maze_probes_grouped_20260924")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    t0 = time.time()
    if args.table_only:
        cells = json.loads((args.out_dir / "cells.json").read_text())
        with open(args.out_dir / "split_balance.csv") as f:
            balance = [{k: (float(v) if k.startswith(("pos_rate", "pos_bin")) else int(v) if k != "concept" else v)
                        for k, v in r.items()} for r in csv.DictReader(f)]
        write_table(cells, balance, args.out_dir, json.loads((args.out_dir / "gate_summary.json").read_text()))
        return
    fr = load_frames(args.frames)
    arms = arm_list(args)
    print(f"[frames] {len(fr['level'])} frames, {len(np.unique(fr['level']))} levels, device {device}", flush=True)
    gate, acts = run_gate(args, fr, arms, device, args.out_dir)
    if not gate["passed"]:
        raise SystemExit("GATE FAILED: original position-bin AUCs not reproduced; grouped analysis not run")
    if args.gate_only:
        return
    dims = {k: int(v.shape[1]) for k, v in acts[arms[0][0]].items()}
    rows, nulls, balance = run_grouped(args, fr, arms, acts, args.out_dir)
    ck = aggregate(args, rows, nulls, args.out_dir)
    cells = summarise(ck, args.out_dir)
    write_table(cells, balance, args.out_dir, gate)
    (args.out_dir / "run_meta.json").write_text(json.dumps({
        "site_dims": dims, "args": {k: str(v) for k, v in vars(args).items()},
        "arms": [(a, c, str(p), s) for a, c, p, s in arms], "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
