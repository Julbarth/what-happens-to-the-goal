#!/usr/bin/env python3
"""Coin availability probes with level-grouped splits, a train-only residualiser, init draws and a pixel baseline."""

from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "procgen", ROOT / "external" / "train-procgen-pytorch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.model import ImpalaModel  # noqa: E402
from common.policy import CategoricalPolicy  # noqa: E402
from head_rewire_layer_features import block3_flat_from_policy  # noqa: E402

# Canonical sites shared with Maze / Attribute Maze (maze_probe_redo.MultiLayerCapture); "pre_fc" is
# the pre-ReLU vec(H3) site, kept only for comparison with the earlier site set.
HOOKS = ("block1", "block2", "block3_gap", "block3", "fc", "pre_fc")
CANONICAL = ("block1", "block2", "block3_gap", "block3", "fc")
SITE_MAP = {
    "block1": ("H(1)", "GAP(embedder.block1 output); no ReLU follows block1 in the architecture; 16x32x32 -> 16-d"),
    "block2": ("H(2)", "GAP(embedder.block2 output); no ReLU follows block2; 32x16x16 -> 32-d"),
    "block3_gap": ("H(3)", "GAP(ReLU(embedder.block3 output)), 32x8x8 -> 32-d"),
    "block3": ("h_pre", "flatten(ReLU(block3)) = embedder.fc forward-pre-hook input, 2048-d"),
    "fc": ("h_fc", "ReLU(embedder.fc(h_pre)) = input of fc_policy / fc_value, 256-d"),
    "pre_fc": ("vec H(3) pre-ReLU (legacy)", "flatten(embedder.block3 output) before ReLU, 2048-d"),
}
CONCEPTS = ("presence", "bearing", "end_wall")
# Per-concept seeds for the level-permutation null.
NULL_SEED = {"presence": 3, "bearing": 2, "end_wall": 1}
IN_CHANNELS, N_ACTIONS = 3, 15
GM = ("6033_gm_20M", "6037_gm_120M", "6051_gm_55M")
NONGM = ("7001_teach", "7002_teach", "7003_teach")
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
CKPT_DIR = DATA_ROOT / "checkpoints/coin"


def default_device() -> torch.device:
    return torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )


def build_policy(state_dict: dict | None, device: torch.device, seed: int | None = None):
    """Impala + CategoricalPolicy; random init consumes torch RNG exactly as `load_random_policy`."""
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    policy = CategoricalPolicy(ImpalaModel(in_channels=IN_CHANNELS), False, N_ACTIONS)
    if state_dict is not None:
        policy.load_state_dict(state_dict)
    return policy.to(device).eval()


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    return build_policy(state, device)


@torch.no_grad()
def all_site_features(policy, obs: np.ndarray, device: torch.device, batch: int = 64) -> dict:
    """Features at all sites from one policy forward, with numerical checks on h_pre and h_fc."""
    emb = policy.embedder
    buf: dict = {}
    handles = [
        emb.block1.register_forward_hook(lambda _m, _i, o: buf.__setitem__("block1", o.mean(dim=(2, 3)))),
        emb.block2.register_forward_hook(lambda _m, _i, o: buf.__setitem__("block2", o.mean(dim=(2, 3)))),
        emb.block3.register_forward_hook(lambda _m, _i, o: buf.__setitem__("h3_raw", o)),
        emb.fc.register_forward_pre_hook(lambda _m, inp: buf.__setitem__("fc_in", inp[0])),
        emb.register_forward_hook(lambda _m, _i, o: buf.__setitem__("h_fc", o)),
    ]
    out = {h: [] for h in HOOKS}
    try:
        for i in range(0, len(obs), batch):
            chunk = obs[i : i + batch]
            x = torch.as_tensor(chunk, dtype=torch.float32, device=device)
            hid = torch.zeros(len(chunk), emb.output_dim, device=device)
            dist, value, _ = policy(x, hid, torch.zeros(len(chunk), device=device))
            h3 = buf["h3_raw"]
            h_pre = torch.relu(h3).flatten(1)
            h_fc = torch.relu(emb.fc(buf["fc_in"]))
            # h_pre must be fc's actual input; h_fc must reproduce the policy's logits and value.
            assert torch.allclose(h_pre, buf["fc_in"], atol=1e-5)
            assert torch.allclose(h_fc, buf["h_fc"], atol=1e-5)
            assert torch.allclose(torch.log_softmax(policy.fc_policy(h_fc), 1), dist.logits, atol=1e-5)
            assert torch.allclose(policy.fc_value(h_fc).reshape(-1), value, atol=1e-5)
            legacy = block3_flat_from_policy(policy, chunk, str(device))
            assert np.allclose(legacy, h_pre.cpu().numpy(), atol=1e-5)
            out["block1"].append(buf["block1"].cpu().numpy())
            out["block2"].append(buf["block2"].cpu().numpy())
            out["block3_gap"].append(torch.relu(h3).mean(dim=(2, 3)).cpu().numpy())
            out["block3"].append(h_pre.cpu().numpy())
            out["fc"].append(h_fc.cpu().numpy())
            out["pre_fc"].append(h3.flatten(1).cpu().numpy())
    finally:
        for hd in handles:
            hd.remove()
    return {h: np.concatenate(v, axis=0).astype(np.float32) for h, v in out.items()}


def obs_hashes(obs: np.ndarray) -> np.ndarray:
    return np.array([hashlib.blake2b(o.tobytes(), digest_size=16).hexdigest() for o in obs])


def make_splits(levels, hashes, y_presence, seeds, test_frac) -> list[dict]:
    """Level-grouped splits, then drop test frames pixel-identical to any train frame."""
    splits = []
    for s in seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=int(s))
        tr, te = next(gss.split(np.zeros(len(levels)), groups=levels))
        assert not set(levels[tr]) & set(levels[te])
        train_hash = set(hashes[tr])
        keep = np.array([h not in train_hash for h in hashes[te]])
        te_kept = te[keep]
        for part in (tr, te_kept):
            if len(np.unique(y_presence[part])) < 2:
                raise SystemExit(f"split seed {s}: presence single-class on one side")
        splits.append(
            {
                "seed": int(s),
                "train": tr,
                "test": te_kept,
                "n_train": int(len(tr)),
                "n_test_before_dedup": int(len(te)),
                "n_test_dropped_dup": int((~keep).sum()),
                "n_test": int(len(te_kept)),
                "n_levels_train": int(len(np.unique(levels[tr]))),
                "n_levels_test": int(len(np.unique(levels[te_kept]))),
                "presence_pos_train": float(y_presence[tr].mean()),
                "presence_pos_test": float(y_presence[te_kept].mean()),
            }
        )
    return splits


def auc_split(X, y, tr, te) -> float:
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))


def r2_split(X, y, tr, te) -> float:
    pred = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
    yte = y[te]
    return float(1.0 - np.sum((yte - pred) ** 2) / (np.sum((yte - yte.mean()) ** 2) + 1e-12))


def residualise_train_only(y, proxy, tr) -> np.ndarray:
    """Residual of y on proxy, with the linear fit estimated on train rows only."""
    p = proxy.reshape(-1, 1)
    reg = Ridge(alpha=1e-3).fit(p[tr], y[tr])
    return y - reg.predict(p)


def level_mean_permutation(y, levels, rng, binarise: bool) -> np.ndarray:
    """Permute per-level mean labels across levels (thresholded at 0.5 for binary)."""
    uniq = np.unique(levels)
    vals = np.array([y[levels == g].mean() for g in uniq])
    rng.shuffle(vals)
    new = dict(zip(uniq, vals))
    out = np.array([new[g] for g in levels])
    return (out >= 0.5).astype(np.int64) if binarise else out


def targets(frames: dict, splits: list[dict]) -> dict:
    """Per-split target vectors; bearing depends on the split through its residualiser."""
    bx = frames["coin_bearing_x"].astype(np.float64)
    prog = frames["progress"].astype(np.float64)
    return {
        "presence": [frames["coin_on_screen"].astype(np.int64)] * len(splits),
        "bearing": [residualise_train_only(bx, prog, sp["train"]) for sp in splits],
        "end_wall": [frames["dist_to_level_end"].astype(np.float64)] * len(splits),
    }


def null_targets(tgts: dict, levels: np.ndarray, n_shuffles: int) -> dict:
    """[concept][shuffle][split] permuted targets; one level permutation per shuffle, shared by splits."""
    out = {}
    for c in CONCEPTS:
        rows = []
        for i in range(n_shuffles):
            per_split = []
            for y in tgts[c]:
                rng = np.random.default_rng([NULL_SEED[c], i])
                per_split.append(level_mean_permutation(y, levels, rng, binarise=c == "presence"))
            rows.append(per_split)
        out[c] = rows
    return out


def score(c, X, y, sp) -> float:
    return auc_split(X, y, sp["train"], sp["test"]) if c == "presence" else r2_split(X, y, sp["train"], sp["test"])


def evaluate(feats: dict, tgts: dict, splits: list[dict], nulls: dict | None) -> tuple[list[dict], dict]:
    """Per-split real scores and (optionally) per-shuffle split-averaged null scores."""
    rows, null_means = [], {}
    for h, X in feats.items():
        for c in CONCEPTS:
            for k, sp in enumerate(splits):
                rows.append({"site": h, "concept": c, "split_seed": sp["seed"], "score": score(c, X, tgts[c][k], sp)})
            if nulls is None:
                continue
            nm = []
            for per_split in nulls[c]:
                v = [score(c, X, per_split[k], sp) for k, sp in enumerate(splits)]
                nm.append(float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"))
            null_means[(h, c)] = nm
    return rows, null_means


def pixel_features(obs: np.ndarray, size: int = 16) -> np.ndarray:
    """Block-averaged raw RGB (size x size per channel)."""
    n, c, h, w = obs.shape
    hs, ws = h // size, w // size
    x = obs[:, :, : hs * size, : ws * size].reshape(n, c, size, hs, size, ws).mean(axis=(3, 5))
    return x.reshape(n, -1).astype(np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# Frame sources differ by level parity: even levels were rolled out by 6033@20M, odd levels by 6033@200M.
SUBSETS = {
    "all": lambda lv: np.ones(len(lv), dtype=bool),
    "src20M_even": lambda lv: lv % 2 == 0,
    "src200M_odd": lambda lv: lv % 2 == 1,
}


def prepare_subsets(frames: dict, args) -> dict:
    levels_all = frames["level"].astype(np.int64)
    hashes_all = obs_hashes(frames["obs"])
    out = {}
    for name, fn in SUBSETS.items():
        m = fn(levels_all)
        sub = {k: v[m] for k, v in frames.items() if v.shape[:1] == levels_all.shape}
        levels = sub["level"].astype(np.int64)
        splits = make_splits(levels, hashes_all[m], sub["coin_on_screen"].astype(np.int64), args.split_seeds, args.test_frac)
        tgts = targets(sub, splits)
        nulls = null_targets(tgts, levels, args.n_shuffles) if name == "all" else None
        out[name] = {"mask": m, "splits": splits, "tgts": tgts, "nulls": nulls, "n_frames": int(m.sum()),
                     "n_levels": int(len(np.unique(levels)))}
    return out


def run(args) -> None:
    device = torch.device(args.device) if args.device else default_device()
    frames = dict(np.load(args.frames, allow_pickle=False))
    subsets = prepare_subsets(frames, args)
    out = args.out_dir
    (out / "per_network").mkdir(parents=True, exist_ok=True)
    split_rows = [
        {"subset": n, "n_frames": d["n_frames"], "n_levels": d["n_levels"],
         **{k: v for k, v in sp.items() if k not in ("train", "test")}}
        for n, d in subsets.items() for sp in d["splits"]
    ]
    write_csv(out / "splits.csv", split_rows)
    print(f"device={device}")
    for r in split_rows:
        print("  split", r)

    networks = [("raw_pixels", "pixels", None)]
    networks += [(f"random_init_{args.init_seed0 + i}", None, args.init_seed0 + i) for i in range(args.n_inits)]
    networks += [(lab, Path(p), None) for lab, p in (s.split(":", 1) for s in args.checkpoints)]
    for name, path, seed in networks:
        t0 = time.time()
        if path == "pixels":
            feats = {"pixels": pixel_features(frames["obs"], args.pixel_size)}
        else:
            pol = build_policy(None, device, seed=seed) if path is None else load_checkpoint(path, device)
            feats = all_site_features(pol, frames["obs"], device)
            del pol
        rows, null_rows = [], []
        for sname, d in subsets.items():
            sub_feats = {h: X[d["mask"]] for h, X in feats.items()}
            r, nm = evaluate(sub_feats, d["tgts"], d["splits"], d["nulls"])
            rows += [{"subset": sname, **x} for x in r]
            null_rows += [
                {"subset": sname, "site": h, "concept": c, "shuffle": i, "null_score": v}
                for (h, c), vals in nm.items() for i, v in enumerate(vals)
            ]
        write_csv(out / "per_network" / f"{name}_scores.csv", rows)
        write_csv(out / "per_network" / f"{name}_null.csv", null_rows)
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "frames": str(args.frames),
                "split": "GroupShuffleSplit by level, test_size=%.2f of levels, per subset; test frames pixel-identical to a train frame dropped" % args.test_frac,
                "split_seeds": list(args.split_seeds),
                "subsets": {"all": "all 320 frames", "src20M_even": "even levels, rolled out by 6033@20M (model_20054016 = 6033_gm_20M)", "src200M_odd": "odd levels, rolled out by 6033@200M"},
                "residualiser": "Ridge(alpha=1e-3) of coin_bearing_x on progress, fit on train rows of each split",
                "probes": "LogisticRegression(C=1, max_iter=2000) for presence; Ridge(alpha=1) for bearing/end_wall; no scaler",
                "decodable": "all cohort checkpoints > max of init draws AND mean delta > 0 (every concept)",
                "null": "secondary; per-level mean label permuted across levels (presence thresholded at 0.5); same grouped splits; statistic = mean over splits; subset 'all' only",
                "pixel_baseline": f"block means {args.pixel_size}x{args.pixel_size}x3",
                "n_shuffles": args.n_shuffles, "n_inits": args.n_inits, "init_seed0": args.init_seed0,
                "checkpoints": args.checkpoints, "device": str(device), "site_map": SITE_MAP, "torch": torch.__version__,
            },
            indent=2,
        )
    )


def fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.2e}" if abs(x) < 0.01 and x != 0 else f"{x:.3f}"


def ci95(v: np.ndarray) -> float:
    if len(v) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def load_scores(out: Path, names: list[str]) -> tuple[dict, dict]:
    """(network, subset, site, concept) -> split-mean score / split SD, and null arrays for subset 'all'."""
    mean, null = {}, {}
    for n in names:
        acc: dict = {}
        for r in csv.DictReader((out / "per_network" / f"{n}_scores.csv").open()):
            acc.setdefault((n, r["subset"], r["site"], r["concept"]), []).append(float(r["score"]))
        for k, v in acc.items():
            a = np.asarray(v)
            mean[k] = (float(np.nanmean(a)), float(np.nanstd(a, ddof=1)))
        nacc: dict = {}
        for r in csv.DictReader((out / "per_network" / f"{n}_null.csv").open()):
            nacc.setdefault((n, r["subset"], r["site"], r["concept"]), []).append(float(r["null_score"]))
        for k, v in nacc.items():
            a = np.asarray(v)
            null[k] = a[np.isfinite(a)]
    return mean, null


def cell(mean, null, inits, cks, subset, h, c) -> dict:
    init = np.array([mean[(n, subset, h, c)][0] for n in inits])
    tr = np.array([mean[(k, subset, h, c)][0] for k in cks])
    delta = tr - init.mean()
    p95 = [float(np.percentile(null[(k, subset, h, c)], 95)) if (k, subset, h, c) in null else float("nan") for k in cks]
    beats = [t >= q for t, q in zip(tr, p95)] if np.isfinite(p95).all() else None
    return {
        "subset": subset, "site": h, "concept": c, "n_ckpts": len(cks),
        "trained_mean": float(tr.mean()), "trained_sd": float(tr.std(ddof=1)) if len(tr) > 1 else float("nan"),
        "trained_ckpts": tr.tolist(), "init_mean": float(init.mean()), "init_sd": float(init.std(ddof=1)),
        "init_max": float(init.max()), "n_above_init_max": int(np.sum(tr > init.max())),
        "delta_mean": float(delta.mean()), "delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
        "delta_ci95": ci95(delta), "delta_ckpts": delta.tolist(), "trained_ci95": ci95(tr),
        "null_p95_mean": float(np.mean(p95)), "beats_null": f"{sum(beats)}/{len(cks)}" if beats is not None else "n/a",
        "decodable": bool(np.all(tr > init.max()) and delta.mean() > 0),
        # The init-separation rule can pass a target no network decodes (R2 < 0 for trained and init alike).
        "all_above_chance": bool(np.all(tr > (0.5 if c == "presence" else 0.0))),
        "split_sd_mean": float(np.mean([mean[(k, subset, h, c)][1] for k in cks])),
    }


def dec(r: dict) -> str:
    if not r["decodable"]:
        return "N"
    return "Y" if r["all_above_chance"] else "Y* (below chance)"


def summarise(args) -> None:
    out = args.out_dir
    inits = [f"random_init_{args.init_seed0 + i}" for i in range(args.n_inits)]
    mean, null = load_scores(out, inits + list(GM + NONGM) + ["raw_pixels"])
    cohorts = {"GM": GM, "Non-GM": NONGM, "GM-without-6033": GM[1:]}
    splits = list(csv.DictReader((out / "splits.csv").open()))
    n_test_levels = {
        s: [int(r["n_levels_test"]) for r in splits if r["subset"] == s] for s in SUBSETS
    }

    cells = []
    for subset in SUBSETS:
        for c in CONCEPTS:
            for h in HOOKS:
                for cohort, cks in cohorts.items():
                    cells.append({"cohort": cohort, "paper_site": SITE_MAP[h][0], **cell(mean, null, inits, cks, subset, h, c)})
    pixel = {
        (s, c): mean[("raw_pixels", s, "pixels", c)] for s in SUBSETS for c in CONCEPTS
    }
    (out / "summary.json").write_text(json.dumps(
        {"site_map": SITE_MAP, "cells": cells, "n_test_levels": n_test_levels,
         "pixel_baseline": {f"{s}|{c}": {"mean": v[0], "split_sd": v[1]} for (s, c), v in pixel.items()}},
        indent=2))
    write_csv(out / "cells.csv", [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in cells])

    rule = ("Decodable (every concept): all checkpoints of the cohort > max of the 8 init draws and mean Δ > 0. "
            "Null p95 / beats-null (level-permutation, 64 shuffles) is secondary and computed on all frames only. "
            "Y* = passes the rule although some checkpoint is at or below chance (AUC 0.5 / R² 0).")
    idx = {(r["subset"], r["cohort"], r["site"], r["concept"]): r for r in cells}
    lines = [rule, "", "Main table: all 320 frames, canonical sites.", "",
             "| concept | site | cohort | trained mean ± SD [ckpts] | init mean ± SD (max) | Δ mean ± SD [ckpts] (95% CI) | ckpts > init max | decodable | null p95 | beats null |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONCEPTS:
        for h in CANONICAL:
            for cohort in ("GM", "Non-GM"):
                r = idx[("all", cohort, h, c)]
                lines.append(
                    f"| {c} | {r['paper_site']} | {cohort} | {fmt(r['trained_mean'])} ± {fmt(r['trained_sd'])} "
                    f"[{', '.join(fmt(x) for x in r['trained_ckpts'])}] | {fmt(r['init_mean'])} ± {fmt(r['init_sd'])} ({fmt(r['init_max'])}) | "
                    f"{fmt(r['delta_mean'])} ± {fmt(r['delta_sd'])} [{', '.join(fmt(x) for x in r['delta_ckpts'])}] (±{fmt(r['delta_ci95'])}) | "
                    f"{r['n_above_init_max']}/3 | {dec(r)} | {fmt(r['null_p95_mean'])} | {r['beats_null']} |"
                )
    lines += ["", "Raw-pixel baseline (16x16x3 block means, 768-d), mean over splits (split SD):", "",
              "| concept | all | src20M_even | src200M_odd |", "|---|---|---|---|"]
    for c in CONCEPTS:
        lines.append(f"| {c} | " + " | ".join(f"{fmt(pixel[(s, c)][0])} ({fmt(pixel[(s, c)][1])})" for s in SUBSETS) + " |")
    lines += ["", "Source strata, h_pre and h_fc (test levels per split: " +
              "; ".join(f"{s} {min(v)}-{max(v)}" for s, v in n_test_levels.items()) + ").", "",
              "| concept | site | subset | GM [ckpts] | Non-GM [ckpts] | GM-without-6033 [ckpts] | init mean (max) | pixel | decodable GM / Non-GM / GM-w/o-6033 | GM − Non-GM |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONCEPTS:
        for h in ("block3", "fc"):
            for s in SUBSETS:
                g, ng, g2 = (idx[(s, k, h, c)] for k in ("GM", "Non-GM", "GM-without-6033"))
                lines.append(
                    f"| {c} | {SITE_MAP[h][0]} | {s} | {fmt(g['trained_mean'])} [{', '.join(fmt(x) for x in g['trained_ckpts'])}] | "
                    f"{fmt(ng['trained_mean'])} [{', '.join(fmt(x) for x in ng['trained_ckpts'])}] | "
                    f"{fmt(g2['trained_mean'])} [{', '.join(fmt(x) for x in g2['trained_ckpts'])}] | "
                    f"{fmt(g['init_mean'])} ({fmt(g['init_max'])}) | {fmt(pixel[(s, c)][0])} | "
                    f"{dec(g)} / {dec(ng)} / {dec(g2)} | "
                    f"{fmt(g['trained_mean'] - ng['trained_mean'])} |"
                )
    (out / "table_cells.md").write_text("\n".join(lines) + "\n")

    if args.prev_dir and (args.prev_dir / "summary.json").exists():
        prev = {(r["concept"], r["site"], r["cohort"]): r for r in json.loads((args.prev_dir / "summary.json").read_text())["cells"]}
        prev_init = {}
        for n in inits:
            for r in csv.DictReader((args.prev_dir / "per_network" / f"{n}_scores.csv").open()):
                prev_init.setdefault((r["site"], r["concept"]), {}).setdefault(n, []).append(float(r["score"]))
        comp = ["Previous grouped run (legacy sites) → harmonised run (canonical sites). Same frames/splits/probes; "
                "decodable = init-separation rule applied to both.", "",
                "| concept | site (prev → new) | cohort | trained prev → new | init prev → new (max) | Δ prev → new | decodable prev → new |",
                "|---|---|---|---|---|---|---|"]
        pairs = [("block1", "block1"), ("block2", "block2"), ("pre_fc", "block3_gap"), ("block3", "block3"), ("fc", "fc")]
        for c in CONCEPTS:
            for ps, ns in pairs:
                pim = np.array([np.mean(v) for v in prev_init[(ps, c)].values()])
                for cohort in ("GM", "Non-GM"):
                    p, n = prev[(c, ps, cohort)], idx[("all", cohort, ns, c)]
                    pdec = bool(np.all(np.array(p["trained_ckpts"]) > pim.max()) and p["delta_mean"] > 0)
                    comp.append(
                        f"| {c} | {SITE_MAP[ps][0]} → {SITE_MAP[ns][0]} | {cohort} | {fmt(p['trained_mean'])} → {fmt(n['trained_mean'])} | "
                        f"{fmt(p['init_mean'])} ({fmt(pim.max())}) → {fmt(n['init_mean'])} ({fmt(n['init_max'])}) | "
                        f"{fmt(p['delta_mean'])} → {fmt(n['delta_mean'])} | {'Y' if pdec else 'N'} → {'Y' if n['decodable'] else 'N'} |"
                    )
        (out / "prev_vs_harmonised.md").write_text("\n".join(comp) + "\n")
        print((out / "prev_vs_harmonised.md").read_text())
    print((out / "table_cells.md").read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames", type=Path, default=DATA_ROOT / "probes/coinrun_concept/frames_v1.npz")
    p.add_argument(
        "--checkpoints", nargs="+", default=[f"{k}:{CKPT_DIR / (k + '.pth')}" for k in GM + NONGM],
        help="label:path pairs; labels must be the six GM/NONGM names for the summary stage",
    )
    p.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--test-frac", type=float, default=0.35)
    p.add_argument("--n-shuffles", type=int, default=64)
    p.add_argument("--n-inits", type=int, default=8)
    p.add_argument("--init-seed0", type=int, default=0)
    p.add_argument("--pixel-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=DATA_ROOT / "probes/coinrun_concept_grouped_canonical_20260924")
    p.add_argument("--prev-dir", type=Path, default=DATA_ROOT / "probes/coinrun_concept_grouped_20260924",
                   help="Grouped run with the legacy Coin sites, for the site-harmonisation comparison.")
    p.add_argument("--summarise-only", action="store_true")
    args = p.parse_args()
    if not args.summarise_only:
        run(args)
    summarise(args)


if __name__ == "__main__":
    main()
