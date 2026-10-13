"""Ablation displacements projected on the fit-only swap axis u_swap: is the signed competition axis a swap artefact?"""
from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import canon_geometry_core as cg  # noqa: E402
import canon_geometry_io as gio  # noqa: E402
from canon_geometry_report import NATIVE, _num  # noqa: E402
from canon_geometry_run import (BANK_DIR, PRIMARY_SPLIT_FILE, check_rows_meta, load_splits,  # noqa: E402
                                load_state_table, sha256_file)

VECS = ("d_comp", "delta_shape", "delta_colour")


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def fisher_cos(u: np.ndarray, v: np.ndarray, J: np.ndarray, p: np.ndarray) -> np.ndarray:
    """cos_F(u, v) under each state's own F, for fixed vectors u, v."""
    n = p.shape[0]
    Jb = np.broadcast_to(J, (n,) + J.shape) if J.ndim == 2 else J
    out = np.full(n, np.nan)
    for i in range(n):
        uu = cg.fisher_bilinear(Jb[i], p[i], u, u)
        vv = cg.fisher_bilinear(Jb[i], p[i], v, v)
        if uu > cg.EPS and vv > cg.EPS:
            out[i] = cg.fisher_bilinear(Jb[i], p[i], u, v) / np.sqrt(uu * vv)
    return out


def project(vecs: dict, u: np.ndarray, sign: np.ndarray, J, p, masks: dict) -> dict:
    """(a) mean cos_F^2 with u; (b) fraction with sign * cos_F > 0; (c) mean sign * cos_E, per vector and mask."""
    out = {}
    for name, X in vecs.items():
        ae, af = cg.axis_loadings(X, u, J, p)
        for mname, m in masks.items():
            out[f"{name}/{mname}/frac_par_F"] = float(np.nanmean(af[m] ** 2))
            out[f"{name}/{mname}/sign_agree_F"] = float(np.mean((sign * af)[m] > 0))
            out[f"{name}/{mname}/cos_E_signed"] = float(np.nanmean((sign * ae)[m]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fisher-dir", type=Path, default=DATA_ROOT / "probes/canon_fisher_20260924")
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT / "probes/canon_geometry_20260924/ablation_supplementary")
    ap.add_argument("--fig", type=Path, default=ROOT / "paper_figures/figS_ras_ablation_on_swap_axis.png")
    args = ap.parse_args()

    st = load_state_table(BANK_DIR / "ras_gt_bank_canonical.json")
    splits = load_splits(BANK_DIR / "splits", st["state_key"], st["partner"])
    sp = splits[PRIMARY_SPLIT_FILE]
    fit_idx = np.flatnonzero(sp["fit"])
    masks = {"eval50": sp["eval"], "all196": np.ones(len(st["sign"]), bool)}
    sign = st["sign"]

    u_swap, srcs = {}, {}
    for cell in gio.iter_cells(args.fisher_dir, st["state_key"], "swap"):
        check_rows_meta(cell, st)
        key = (cell["cohort"], cell["seed"], cell["background"], cell["tap"])
        u_swap[key] = cg.shared_axis(cell["delta_colour"] - cell["delta_shape"], sign, fit_idx)
        srcs.update(cell["source_files"])

    rows = []
    for cell in gio.iter_cells(args.fisher_dir, st["state_key"], "ablation"):
        check_rows_meta(cell, st)
        key = (cell["cohort"], cell["seed"], cell["background"], cell["tap"])
        srcs.update(cell["source_files"])
        J, p = cell["J"], cell["p"]
        ds, dc = cell["delta_shape"], cell["delta_colour"]
        vecs = {"d_comp": dc - ds, "delta_shape": ds, "delta_colour": dc}
        us = u_swap[key]
        ua = cg.shared_axis(vecs["d_comp"], sign, fit_idx)
        r = {"cohort": key[0], "seed": key[1], "background": key[2], "tap": key[3],
             "checkpoint_sha256": cell["checkpoint_sha256"], "native": (key[0], key[2]) in NATIVE}
        r.update({f"swapaxis/{k}": v for k, v in project(vecs, us, sign, J, p, masks).items()})
        r.update({f"ablaxis/{k}": v for k, v in project(vecs, ua, sign, J, p, masks).items()})
        cf = fisher_cos(ua, us, J, p)
        for mname, m in masks.items():
            r[f"cosF_uabl_uswap/{mname}"] = float(np.nanmean(cf[m]))
        r["cosE_uabl_uswap"] = float(ua @ us)
        rows.append(r)
        print("done", key, flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "axis_projection_per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["cohort"], r["background"], r["tap"])].append(r)

    def ms(vals) -> str:
        c = cg.t_ci(vals)
        return f"{_num(c['mean'], 3)} ± {_num(c['sd'], 2)}" if c["n"] >= 2 else f"{_num(c['mean'], 3)} (n=1)"

    lines = ["# Ablation displacements on the fit-only swap axis (supplementary robustness check)", "",
             "u_swap = sign-folded unit mean of swap D_COMP over the 146 primary fit states, per cohort x bg x tap x "
             "seed; u_abl = the same from ablation D_COMP. frac_par_F = mean cos_F(x, u)^2 under each state's own F; "
             "sign_agree = fraction of states with sign * cos_F(x, u) > 0 (expected ~1 for d_comp and Delta_colour, "
             "~0 for Delta_shape, since removing the line shape pushes away from the line); cos_E = mean sign * "
             "cos_E(x, u). Cells: mean ± SD across seeds (ddof = 1); values below 0.01 in scientific notation. "
             "`*` = cross-rendering (supplementary). eval50 = primary eval states; all196 = whole bank.", ""]
    for mname in masks:
        for axis in ("swapaxis", "ablaxis"):
            lines.append(f"## {'u_swap' if axis == 'swapaxis' else 'u_abl'}, {mname}")
            lines.append("")
            hdr = ["cohort", "bg", "tap"] + [f"{v} {s}" for v in VECS for s in ("frac_par_F", "sign_agree", "cos_E")]
            if axis == "ablaxis":
                hdr += ["cos_F(u_abl, u_swap)", "cos_E(u_abl, u_swap)"]
            lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for (coh, bg, tap), rs in sorted(groups.items()):
                row = [coh if (coh, bg) in NATIVE else coh + "*", bg, tap]
                for v in VECS:
                    for s, k in (("frac_par_F", "frac_par_F"), ("sign_agree", "sign_agree_F"), ("cos_E", "cos_E_signed")):
                        row.append(ms([r[f"{axis}/{v}/{mname}/{k}"] for r in rs]))
                if axis == "ablaxis":
                    row += [ms([r[f"cosF_uabl_uswap/{mname}"] for r in rs]), ms([r["cosE_uabl_uswap"] for r in rs])]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    per_seed = ["## Per-seed d_comp on u_swap, eval50 (frac_par_F / sign_agree)", "",
                "| cohort | bg | tap | seed | frac_par_F | sign_agree |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["cohort"], r["background"], r["tap"], r["seed"])):
        if r["native"] or r["cohort"] == "init":
            per_seed.append(f"| {r['cohort']} | {r['background']} | {r['tap']} | {r['seed']} | "
                            f"{_num(r['swapaxis/d_comp/eval50/frac_par_F'], 3)} | "
                            f"{_num(r['swapaxis/d_comp/eval50/sign_agree_F'], 3)} |")
    code = {f: sha256_file(ROOT / f) for f in ("scripts/canon_geometry_axis_projection.py",
                                               "scripts/canon_geometry_core.py", "scripts/canon_geometry_io.py",
                                               "scripts/canon_geometry_run.py", "scripts/canon_geometry_report.py")}
    prov = {"canonical_bank_sha256": st["canonical_bank_sha256"], "primary_split_sha256": sp["split_sha256"],
            "code_sha256": code, "source_files": srcs}
    lines += per_seed + ["", "Provenance: `axis_projection_provenance.json`; raw per-seed rows: "
                         "`axis_projection_per_seed.csv`."]
    (args.out_dir / "axis_projection_mean_sd.md").write_text("\n".join(lines) + "\n")
    (args.out_dir / "axis_projection_provenance.json").write_text(json.dumps(prov, indent=1))

    keys = [k for k in sorted(groups) if (k[0], k[1]) in NATIVE]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    colors = {"d_comp": "#0072B2", "delta_shape": "#D55E00", "delta_colour": "#009E73"}
    panels = [("swapaxis", "frac_par_F", "A  Fisher-energy fraction on u_swap (eval50)"),
              ("swapaxis", "sign_agree_F", "B  sign agreement with u_swap (eval50)"),
              ("swapaxis", "cos_E_signed", "C  signed Euclidean cos with u_swap (eval50)"),
              (None, None, "D  cos_F(u_abl, u_swap) under state F (eval50)")]
    for ax, (axis, stat, title) in zip(axes.ravel(), panels):
        for x, k in enumerate(keys):
            rs = groups[k]
            series = [("u", [r["cosF_uabl_uswap/eval50"] for r in rs], "#444444", 0.0)] if axis is None else [
                (v, [r[f"{axis}/{v}/eval50/{stat}"] for r in rs], colors[v], off)
                for v, off in zip(VECS, (-0.22, 0.0, 0.22))]
            for v, vals, c, off in series:
                ax.scatter(np.full(len(vals), x + off - 0.05), vals, s=14, color=c, alpha=0.7)
                ci = cg.t_ci(vals)
                if ci["n"] >= 2:
                    ax.errorbar(x + off + 0.05, ci["mean"], yerr=ci["sd"], fmt="o", ms=5, color=c, capsize=3,
                                label=v if x == 0 else None)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{k[0]}\n{k[1][:3]}\n{k[2]}" for k in keys], fontsize=9)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("value (seeds; mean ± SD)", fontsize=11)
        ax.tick_params(labelsize=10)
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle("[ABLATION edits (supplementary)] Ablation displacements on the fit-only SWAP axis u_swap; "
                 "native cohorts + init (ablation Δ_shape expected sign-opposite)", fontsize=12)
    plt.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=150, bbox_inches="tight")
    fig.savefig(args.fig.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out_dir / "axis_projection_mean_sd.md", args.fig)


if __name__ == "__main__":
    main()
