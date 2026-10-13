"""Attribute Maze tug of war on u_comp: per-cue projections, base-state position and native choice (black, h_fc)."""
from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
import canon_geometry_core as cg  # noqa: E402
from canon_geometry_run import BANK_DIR, PRIMARY_SPLIT_FILE, load_splits, load_state_table  # noqa: E402

CKPTS = {"gm": ("6313", "6315", "6317"), "deco": ("4371", "4372", "4373"), "init": ("3000", "3001", "3002")}
BEHAV_SUFFIX = {"gm": "gm", "deco": "teach"}


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_contrast(logits: np.ndarray, a_line, a_gem) -> np.ndarray:
    """log sum_{A_line} pi - log sum_{A_gem} pi per row (softmax normaliser cancels)."""
    out = np.empty(len(logits))
    for i, z in enumerate(logits):
        out[i] = np.logaddexp.reduce(z[a_line[i]]) - np.logaddexp.reduce(z[a_gem[i]])
    return out


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """P(score_goal > score_proxy), ties 1/2; NaN unless both classes are present."""
    m = np.isfinite(score) & np.isfinite(y)
    s, t = score[m], y[m].astype(bool)
    n1, n0 = t.sum(), (~t).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_within_sign(score, y, sign) -> float:
    """Mean of the two within-orientation AUCs, so a global UP-vs-RIGHT bias cannot carry the score."""
    v = [auc(score[sign == s], y[sign == s]) for s in (1, -1)]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def fisher_alpha(u: np.ndarray, X: np.ndarray, W: np.ndarray, p: np.ndarray) -> np.ndarray:
    """alpha_F = u^T F X / u^T F u with each state's own F = W^T (diag p - p p^T) W."""
    return np.array([cg.fisher_bilinear(W, p[i], u, X[i]) / cg.fisher_bilinear(W, p[i], u, u)
                     for i in range(len(X))])


def load_cell(fisher_dir: Path, cohort: str, seed: str, st: dict) -> dict:
    npz = fisher_dir / "geom_swap" / f"geom_swap_black_{cohort}_{seed}.npz"
    z = np.load(npz, allow_pickle=False)
    pos = {str(k): i for i, k in enumerate(z["state_key"])}
    if set(pos) != set(st["state_key"]):
        raise SystemExit(f"{npz}: state set differs from the bank")
    o = np.array([pos[k] for k in st["state_key"]])
    W, b = z["W_pi"].astype(np.float64), z["b_pi"].astype(np.float64)
    h = z["h_base_h_fc"].astype(np.float64)[o]
    logits = h @ W.T + b
    err = float(np.abs(logits - z["logits"][o]).max())
    if err > 1e-3:
        raise SystemExit(f"{npz}: W_pi h + b deviates from stored logits by {err}")
    return {"h": h, "ds": z["delta_shape_swap_h_fc"].astype(np.float64)[o],
            "dc": z["delta_colour_swap_h_fc"].astype(np.float64)[o], "W": W, "b": b,
            "p": z["p"].astype(np.float64)[o], "logits": logits, "src": {str(npz.relative_to(ROOT)): sha(npz)},
            "logit_err": err}


def load_native(behav_dir: Path, cohort: str, seed: str, st: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Unedited rollout outcome (1 goal/line, 0 proxy/gem, NaN timeout) and base P(A_line) from the rollout rows."""
    f = behav_dir / "raw" / f"world_edit_rows_black_{seed}_{BEHAV_SUFFIX[cohort]}.csv.gz"
    r = pd.read_csv(f)
    r = r[r.edit == "none"].set_index("state_key").loc[st["state_key"]]
    y = r.outcome.map({"slot1_line_obj": 1.0, "slot2_gem_obj": 0.0}).to_numpy(dtype=float)
    return y, r.pb_line.to_numpy(float), {str(f.relative_to(ROOT)): sha(f)}


def analyze(cell: dict, st: dict, fit_idx: np.ndarray, axis: str) -> dict:
    sign = st["sign"].astype(float)
    dcomp = cell["dc"] - cell["ds"]
    if axis == "unit_mean":
        u = cg.shared_axis(dcomp, st["sign"], fit_idx)
    else:
        fp, fm = fit_idx[st["sign"][fit_idx] == 1], fit_idx[st["sign"][fit_idx] == -1]
        u = dcomp[fp].mean(0) - dcomp[fm].mean(0)
        u /= np.linalg.norm(u)
    W, p = cell["W"], cell["p"]
    out = {"u": u}
    for name, X in (("star", cell["ds"]), ("p", cell["dc"])):
        out[f"aF_raw_{name}"] = fisher_alpha(u, X, W, p)
        out[f"aE_raw_{name}"] = X @ u
    # shape swap favours the gem cell v (-sigma u side), colour swap the line cell u (+sigma u side)
    for m in ("F", "E"):
        out[f"a{m}_star"] = -sign * out[f"a{m}_raw_star"]
        out[f"a{m}_p"] = sign * out[f"a{m}_raw_p"]
        out[f"x{m}_mid"] = 0.5 * (out[f"a{m}_star"] - out[f"a{m}_p"])
    out["xE_abs"] = sign * (cell["h"] @ u)
    aL, aG = st["A_line"], st["A_gem"]
    Wb = lambda H: H @ W.T + cell["b"]  # noqa: E731
    out["C_base"] = group_contrast(cell["logits"], aL, aG)
    out["C_star"] = group_contrast(Wb(cell["h"] + cell["ds"]), aL, aG)
    out["C_p"] = group_contrast(Wb(cell["h"] + cell["dc"]), aL, aG)
    out["C_mid"] = out["C_base"] - 0.5 * (out["C_star"] + out["C_p"])
    # fraction of the way from the proxy-consensus state to the goal-consensus state, read off u
    out["tF"] = out["aF_star"] / (out["aF_star"] + out["aF_p"])
    out["C_interp"] = out["C_star"] + out["tF"] * (out["C_p"] - out["C_star"])
    ae, af = cg.axis_loadings(dcomp, u, W, p)
    out["cosF_dcomp_u"], out["cosE_dcomp_u"] = sign * af, sign * ae
    return out


def seed_summary(o: dict, y_roll: np.ndarray | None, sign: np.ndarray, mask: np.ndarray) -> dict:
    s = {}
    for m in ("F", "E"):
        a_s, a_p = o[f"a{m}_star"][mask], o[f"a{m}_p"][mask]
        s[f"{m}_mean_a_star"] = float(a_s.mean())
        s[f"{m}_mean_a_p"] = float(a_p.mean())
        s[f"{m}_ratio_of_means"] = float(abs(a_p.mean()) / abs(a_s.mean()))
        r = np.abs(a_p) / np.abs(a_s)
        s[f"{m}_median_state_ratio"] = float(np.median(r))
        s[f"{m}_geomean_state_ratio"] = float(np.exp(np.mean(np.log(r))))
        s[f"{m}_frac_p_gt_star"] = float(np.mean(np.abs(a_p) > np.abs(a_s)))
        s[f"{m}_frac_a_star_pos"] = float(np.mean(a_s > 0))
        s[f"{m}_frac_a_p_pos"] = float(np.mean(a_p > 0))
        s[f"{m}_mean_x_mid"] = float(o[f"x{m}_mid"][mask].mean())
        s[f"{m}_frac_proxy_side"] = float(np.mean(o[f"x{m}_mid"][mask] < 0))
    s["mean_C_base"] = float(o["C_base"][mask].mean())
    s["mean_C_star"] = float(o["C_star"][mask].mean())
    s["mean_C_p"] = float(o["C_p"][mask].mean())
    s["mean_C_mid"] = float(o["C_mid"][mask].mean())
    s["frac_C_mid_proxy_side"] = float(np.mean(o["C_mid"][mask] < 0))
    s["frac_first_action_goal"] = float(np.mean(o["C_base"][mask] > 0))
    s["median_tF"] = float(np.median(o["tF"][mask]))
    s["C_interp_sign_agree"] = float(np.mean(np.sign(o["C_interp"][mask]) == np.sign(o["C_base"][mask])))
    s["pearson_C_interp_vs_C_base"] = float(stats.pearsonr(o["C_interp"][mask], o["C_base"][mask])[0])
    s["spearman_aF_star_vs_aF_p"] = float(stats.spearmanr(o["aF_star"][mask], o["aF_p"][mask])[0])
    s["cosF_dcomp_u"] = float(np.nanmean(o["cosF_dcomp_u"][mask]))
    s["cosE_dcomp_u"] = float(np.nanmean(o["cosE_dcomp_u"][mask]))
    s["pearson_xF_mid_vs_C_base"] = float(stats.pearsonr(o["xF_mid"][mask], o["C_base"][mask])[0])
    s["spearman_xF_mid_vs_C_base"] = float(stats.spearmanr(o["xF_mid"][mask], o["C_base"][mask])[0])
    s["spearman_xE_mid_vs_C_base"] = float(stats.spearmanr(o["xE_mid"][mask], o["C_base"][mask])[0])
    y_first = (o["C_base"] > 0).astype(float)
    labels = {"first": y_first}
    if y_roll is not None:
        labels["roll"] = y_roll
        yr = y_roll[mask]
        s["n_roll_goal"] = int(np.nansum(yr == 1))
        s["n_roll_proxy"] = int(np.nansum(yr == 0))
        s["n_roll_timeout"] = int(np.isnan(yr).sum())
    scores = {"xF_mid": o["xF_mid"], "xE_mid": o["xE_mid"], "C_mid": o["C_mid"], "xE_abs": o["xE_abs"],
              "aF_star": o["aF_star"], "neg_aF_p": -o["aF_p"], "aE_star": o["aE_star"], "neg_aE_p": -o["aE_p"]}
    for ln, y in labels.items():
        for sn, sc in scores.items():
            s[f"auc_{ln}_{sn}"] = auc(sc[mask], y[mask])
            s[f"aucws_{ln}_{sn}"] = auc_within_sign(sc[mask], y[mask], sign[mask])
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fisher-dir", type=Path, default=DATA_ROOT / "probes/canon_fisher_20260924")
    ap.add_argument("--behav-dir", type=Path, default=DATA_ROOT / "probes/canon_behav_20260924")
    ap.add_argument("--geom-dir", type=Path, default=DATA_ROOT / "probes/canon_geometry_20260924/swap")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "probes/am_axis_tug_of_war_20260925")
    ap.add_argument("--cohorts", nargs="+", default=list(CKPTS))
    ap.add_argument("--axes", nargs="+", default=["unit_mean", "diff_of_means"],
                    help="unit_mean = canon_geometry_core.shared_axis (primary); diff_of_means = steering-style")
    args = ap.parse_args()
    (args.out_dir / "per_seed").mkdir(parents=True, exist_ok=True)

    st = load_state_table(BANK_DIR / "ras_gt_bank_canonical.json")
    sp = load_splits(BANK_DIR / "splits", st["state_key"], st["partner"])[PRIMARY_SPLIT_FILE]
    fit_idx = np.flatnonzero(sp["fit"])
    masks = {"eval50": sp["eval"], "all196": np.ones(len(st["sign"]), bool)}
    role = np.where(sp["fit"], "fit", "eval")

    seed_rows, checks, srcs = [], [], {}
    for cohort in args.cohorts:
        for seed in CKPTS[cohort]:
            cell = load_cell(args.fisher_dir, cohort, seed, st)
            srcs.update(cell["src"])
            y_roll = None
            if cohort in BEHAV_SUFFIX:
                y_roll, pb_line, bs = load_native(args.behav_dir, cohort, seed, st)
                srcs.update(bs)
                pl = np.array([cell["p"][i][st["A_line"][i]].sum() for i in range(len(pb_line))])
                checks.append({"cell": f"{cohort}_{seed}", "max_abs_pb_line_geom_minus_rollout":
                               float(np.abs(pl - pb_line).max())})
            for axis in args.axes:
                o = analyze(cell, st, fit_idx, axis)
                if axis == "unit_mean":
                    ref = json.loads((args.geom_dir / "per_seed" / f"summary_{cohort}_{seed}_black_h_fc.json")
                                     .read_text())["summary"]["all"]["heldout"][PRIMARY_SPLIT_FILE]
                    checks.append({"cell": f"{cohort}_{seed}", "axis_cos_f_eval_repro":
                                   float(np.nanmean(o["cosF_dcomp_u"][sp["eval"]])),
                                   "axis_cos_f_eval_saved": ref["axis_cos_f_eval"],
                                   "logit_err": cell["logit_err"]})
                    rows = []
                    for i, k in enumerate(st["state_key"]):
                        r = {"cohort": cohort, "seed": seed, "background": "black", "tap": "h_fc", "axis": axis,
                             "state_key": k, "split_role": role[i], "sign": int(st["sign"][i]),
                             "d": int(st["d"][i]), "layout": int(st["layout"][i]),
                             "native_rollout_goal": "" if y_roll is None or np.isnan(y_roll[i]) else int(y_roll[i]),
                             "native_rollout_outcome": "" if y_roll is None else
                             ("timeout" if np.isnan(y_roll[i]) else ("goal" if y_roll[i] == 1 else "proxy"))}
                        for f in ("aF_star", "aF_p", "aE_star", "aE_p", "xF_mid", "xE_mid", "xE_abs",
                                  "C_base", "C_star", "C_p", "C_mid", "tF", "C_interp", "cosF_dcomp_u", "cosE_dcomp_u"):
                            r[f] = float(o[f][i])
                        rows.append(r)
                    fn = args.out_dir / "per_seed" / f"per_state_{cohort}_{seed}_black_h_fc.csv"
                    with open(fn, "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(rows)
                for mname, m in masks.items():
                    seed_rows.append({"cohort": cohort, "seed": seed, "axis": axis, "mask": mname,
                                      **seed_summary(o, y_roll, st["sign"], m)})

    df = pd.DataFrame(seed_rows)
    df.to_csv(args.out_dir / "per_seed_summary.csv", index=False)
    summary = {}
    metrics = [c for c in df.columns if c not in ("cohort", "seed", "axis", "mask")]
    for (cohort, axis, mname), g in df.groupby(["cohort", "axis", "mask"]):
        summary.setdefault(axis, {}).setdefault(mname, {})[cohort] = {
            mt: {**cg.t_ci(g[mt].tolist()), "per_seed": dict(zip(g.seed, map(float, g[mt])))} for mt in metrics}
    prov = {"canonical_bank_sha256": st["canonical_bank_sha256"], "primary_split": PRIMARY_SPLIT_FILE,
            "split_sha256": sp["split_sha256"], "n_fit": int(sp["fit"].sum()), "n_eval": int(sp["eval"].sum()),
            "background": "black", "tap": "h_fc", "source_files_sha256": srcs,
            "code_sha256": {"scripts/am_axis_tug_of_war.py": sha(Path(__file__))},
            "checks": checks,
            "conventions": {
                "u": "unit_mean: unit sign-folded mean of d_comp/||d_comp|| over the 146 fit states "
                     "(canon_geometry_core.shared_axis); sigma*u points toward the line (goal-shape) cell",
                "a_star": "shape-swap displacement projected on u, positive = toward the gem cell v it favours",
                "a_p": "colour-swap displacement projected on u, positive = toward the line cell u it favours",
                "aF": "u^T F(s) Delta / u^T F(s) u at the base state; aE = u_hat^T Delta",
                "x_mid": "(a_star - a_p)/2 = base position minus the midpoint of the two consensus states along u, "
                         "positive = goal side (algebraic identity with a_star, a_p)",
                "C": "log sum_{A_line} pi - log sum_{A_gem} pi; C_mid = C_base - (C_star + C_p)/2, exact readout",
                "tF": "a_star/(a_star + a_p): base position between proxy-consensus (0) and goal-consensus (1)",
                "C_interp": "C_star + tF (C_p - C_star), the base contrast predicted by linear interpolation on u",
                "native_rollout": "canon_behav_20260924 edit=none outcome, timeouts excluded",
                "first": "label C_base > 0 (first-action goal preference)",
                "ci": "t-distribution 95% CI across the 3 checkpoints, sd ddof=1"}}
    (args.out_dir / "summary.json").write_text(json.dumps({"provenance": prov, "summary": summary}, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
