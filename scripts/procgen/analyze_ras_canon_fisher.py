#!/usr/bin/env python3
"""Seed-level summaries, 95% t-CIs and matched-state comparisons for run_ras_canon_fisher.py rows (Tables 7/8, Attribute Maze)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = ("Q", "G", "D_F", "g", "ghat", "norm_delta", "S", "KL_t1", "KL_over_halfQ", "dC_finite",
           "Q_oldmask", "g_oldmask", "frac_delta_sq_on_inactive")
STATE_METRICS = ("trF_over_d", "G_rand_median", "cos_F_shape_colour", "entropy", "C_base")
TAP_OLD = {"h_pre": "block3", "h_fc": "fc"}
# Texture Non-GM = seeds 4361-63 (texture-trained); seeds 4371-73 (black-trained)
# evaluated on the texture rendering form a supplementary row only.
CELLS = {
    "black": {"GM": "gm", "Non-GM": "deco", "init": "init"},
    "texture": {"GM": "gmtex", "Non-GM": "decotex", "suppl-blacktrained-control": "deco", "init": "init"},
}
PAIRS = {"swap": ("shape_swap", "colour_swap"), "ablation": ("shape", "colour")}
PAIR = PAIRS["swap"]
PAIR_ROWS: dict = {}
# Ablation keys carry an explicit suffix so a bare "shape"/"colour" key can never be read as a swap value.
EDIT_KEY = {"shape": "shape_ablation", "colour": "colour_ablation", "shape_swap": "shape_swap", "colour_swap": "colour_swap"}
SUBSETS = {
    "all": lambda r: True,
    "eval50": lambda r: r["split_role"] == "eval",
    "d>1": lambda r: not r["is_d1"],
    "sign+1": lambda r: r["axis_sign"] == 1,
    "sign-1": lambda r: r["axis_sign"] == -1,
    "d>1,sign+1": lambda r: (not r["is_d1"]) and r["axis_sign"] == 1,
    "d>1,sign-1": lambda r: (not r["is_d1"]) and r["axis_sign"] == -1,
}


def load_rows(d: Path) -> list[dict]:
    rows = []
    for f in sorted(d.glob("rows_*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                for k, v in r.items():
                    if k in ("state_key", "state_hash", "background", "cohort", "tap", "edit", "split_role",
                             "ckpt_label", "ckpt_path", "ckpt_sha256", "canonical_bank_sha256", "code_sha256",
                             "A_line", "A_gem"):
                        continue
                    if v in ("True", "False"):
                        r[k] = v == "True"
                    else:
                        try:
                            r[k] = float(v)
                        except ValueError:
                            pass
                r["seed"] = int(r["seed"])
                r["axis_sign"] = int(r["axis_sign"])
                rows.append(r)
    return rows


def ci95(vals) -> dict:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    n = len(v)
    m = float(v.mean()) if n else float("nan")
    if n < 2:
        return {"mean": m, "lo": float("nan"), "hi": float("nan"), "n": n, "seed_values": v.tolist()}
    h = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return {"mean": m, "lo": m - h, "hi": m + h, "half": h, "sd": float(v.std(ddof=1)), "n": n,
            "seed_values": v.tolist()}


def seed_stat(rows, metric, stat="median") -> float:
    v = np.array([r[metric] for r in rows if isinstance(r.get(metric), float) and np.isfinite(r[metric])])
    if not len(v):
        return float("nan")
    return float(np.median(v) if stat == "median" else np.mean(v))


def index(rows):
    idx = defaultdict(list)
    for r in rows:
        idx[(r["background"], r["cohort"], r["seed"], r["tap"], r["edit"])].append(r)
    return idx


def cohort_table(idx, rows, stat="median") -> dict:
    out = {}
    seeds = defaultdict(set)
    for r in rows:
        seeds[(r["background"], r["cohort"])].add(r["seed"])
    for bg, cells in CELLS.items():
        for label, coh in cells.items():
            for tap in ("h_pre", "h_fc"):
                for edit in PAIR:
                    for sub, pred in SUBSETS.items():
                        per_seed = {}
                        for s in sorted(seeds[(bg, coh)]):
                            rs = [r for r in idx[(bg, coh, s, tap, edit)] if pred(r)]
                            per_seed[s] = {m: seed_stat(rs, m, stat) for m in METRICS + STATE_METRICS}
                            per_seed[s]["n_states"] = len(rs)
                            per_seed[s]["frac_g_neg"] = float(np.mean([r["g"] < 0 for r in rs])) if rs else float("nan")
                        agg = {m: ci95([per_seed[s][m] for s in per_seed]) for m in METRICS + STATE_METRICS + ("frac_g_neg",)}
                        out[f"{bg}|{label}|{tap}|{EDIT_KEY[edit]}|{sub}"] = {"per_seed": per_seed, "cohort": agg}
    return out


def sign_stratified(tab: dict) -> dict:
    """Per seed, 0.5 * (stat on sign +1 + stat on sign -1); then the across-seed CI."""
    out = {}
    for key, v in tab.items():
        bg, label, tap, edit, sub = key.split("|")
        if sub not in ("sign+1", "d>1,sign+1"):
            continue
        other = tab[f"{bg}|{label}|{tap}|{edit}|{sub.replace('+1', '-1')}"]
        name = "signstrat" if sub == "sign+1" else "d>1,signstrat"
        per_seed = {s: {m: 0.5 * (v["per_seed"][s][m] + other["per_seed"][s][m]) for m in METRICS}
                    for s in v["per_seed"]}
        out[f"{bg}|{label}|{tap}|{edit}|{name}"] = {
            "per_seed": per_seed, "cohort": {m: ci95([per_seed[s][m] for s in per_seed]) for m in METRICS}}
    return out


def paired(idx, rows) -> dict:
    """Matched-state comparisons, per seed median of the per-state paired quantity, CI across seeds:
    background (same checkpoint, same state_key, texture vs black) and cue (shape vs colour, same state)."""
    out = {}
    ck = sorted({(r["cohort"], r["seed"]) for r in rows})
    for coh, s in ck:
        for tap in ("h_pre", "h_fc"):
            for edit in PAIR:
                b = {r["state_key"]: r for r in idx[("black", coh, s, tap, edit)]}
                t = {r["state_key"]: r for r in idx[("texture", coh, s, tap, edit)]}
                keys = sorted(set(b) & set(t))
                if not keys:
                    continue
                out.setdefault(f"bg|{coh}|{tap}|{EDIT_KEY[edit]}", {})[s] = {
                    "n_matched": len(keys),
                    "median_log10_Q_tex_over_black": float(np.median([np.log10(t[k]["Q"] / b[k]["Q"]) for k in keys])),
                    "median_g_tex_minus_black": float(np.median([t[k]["g"] - b[k]["g"] for k in keys])),
                    "spearman_Q": float(stats.spearmanr([b[k]["Q"] for k in keys], [t[k]["Q"] for k in keys])[0]),
                    "spearman_g": float(stats.spearmanr([b[k]["g"] for k in keys], [t[k]["g"] for k in keys])[0]),
                }
            for bg in ("black", "texture"):
                sh = {r["state_key"]: r for r in idx[(bg, coh, s, tap, PAIR[0])]}
                co = {r["state_key"]: r for r in idx[(bg, coh, s, tap, PAIR[1])]}
                keys = sorted(set(sh) & set(co))
                if not keys:
                    continue
                out.setdefault(f"cue|{bg}|{coh}|{tap}", {})[s] = {
                    "n": len(keys),
                    "median_log10_Q_shape_over_colour": float(np.median([np.log10(sh[k]["Q"] / co[k]["Q"]) for k in keys])),
                    "frac_Q_shape_gt_colour": float(np.mean([sh[k]["Q"] > co[k]["Q"] for k in keys])),
                    "frac_g_shape_neg_and_g_colour_pos": float(np.mean([(sh[k]["g"] < 0) and (co[k]["g"] > 0) for k in keys])),
                    "median_g_shape_minus_g_colour": float(np.median([sh[k]["g"] - co[k]["g"] for k in keys])),
                    "median_log10_G_shape_over_colour": float(np.median([np.log10(sh[k]["G"] / co[k]["G"]) for k in keys])),
                    "median_log10_DF_shape_over_colour": float(np.median([np.log10(sh[k]["D_F"] / co[k]["D_F"]) for k in keys])),
                }
                pr = PAIR_ROWS.get((bg, coh, s, tap))
                if pr:
                    out[f"cue|{bg}|{coh}|{tap}"][s].update({
                        "median_cos_F": float(np.median([r["cos_F_swap"] for r in pr])),
                        "frac_cos_F_neg": float(np.mean([r["cos_F_swap"] < 0 for r in pr])),
                        "median_cos_E": float(np.median([r["cos_E_swap"] for r in pr])),
                        "median_Q_dcomp": float(np.median([r["Q_dcomp"] for r in pr])),
                        "median_g_dcomp": float(np.median([r["g_dcomp"] for r in pr])),
                        "frac_g_dcomp_pos": float(np.mean([r["g_dcomp"] > 0 for r in pr])),
                    })
    agg = {}
    for key, per_seed in out.items():
        ms = next(iter(per_seed.values())).keys()
        agg[key] = {"per_seed": per_seed, "cohort": {m: ci95([per_seed[s][m] for s in per_seed]) for m in ms}}
    return agg


def figure_json(tab: dict) -> dict:
    """Per-background, per-tap cohort summaries in the ras.{black,texture} layout read by the figure builders."""
    out = {}
    for bg, gm, ng in (("black", "GM", "Non-GM"), ("texture", "GM", "Non-GM")):
        out[bg] = {"source": "run_ras_canon_fisher.py on frozen canonical bank 0f45ca67 (196 states)"}
        for tap, old in TAP_OLD.items():
            d = {}
            for m in ("Q", "G", "D_F", "g", "ghat"):
                for edit, src in zip(("shape", "colour"), PAIR):
                    e = {}
                    for coh_key, label in (("gm", gm), ("nongm", ng), ("init", "init")):
                        c = tab[f"{bg}|{label}|{tap}|{EDIT_KEY[src]}|all"]["cohort"][m]
                        e[f"{coh_key}_seed_values"] = c["seed_values"]
                        e[f"{coh_key}_mean"] = c["mean"]
                        e[f"{coh_key}_ci95_half"] = c.get("half")
                    d[f"{m}_{edit}"] = e
            out[bg][old] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--primary", choices=list(PAIRS), default="swap")
    ap.add_argument("--pair-dir", type=Path, default=None, help="rows_swap_pair/ (cos_F, D_COMP; swap mode)")
    args = ap.parse_args()
    global PAIR
    PAIR = PAIRS[args.primary]
    if args.primary == "swap" and args.pair_dir:
        for f in sorted(args.pair_dir.glob("rows_swap_pair_*.csv")):
            for r in csv.DictReader(open(f)):
                for k in ("cos_F_swap", "cos_E_swap", "Q_dcomp", "g_dcomp"):
                    r[k] = float(r[k])
                PAIR_ROWS.setdefault((r["background"], r["cohort"], int(r["seed"]), r["tap"]), []).append(r)
    rows = load_rows(args.rows_dir)
    idx = index(rows)
    tab = cohort_table(idx, rows)
    tab_mean = cohort_table(idx, rows, stat="mean")
    res = {"primary_edits": dict(zip(("shape", "colour"), PAIR)), "n_rows": len(rows), "stat": "per seed: median over states; cohort: mean of seed medians, 95% t-CI (n seeds)",
           "canonical_bank_sha256": sorted({r["canonical_bank_sha256"] for r in rows}),
           "table": tab, "table_seedmean": tab_mean, "signstrat": sign_stratified(tab), "paired": paired(idx, rows),
           "figure_ras": figure_json(tab)}
    args.out.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", args.out, "rows", len(rows))


if __name__ == "__main__":
    main()
