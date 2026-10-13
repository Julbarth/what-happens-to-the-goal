"""Appendix Attribute Maze probe figure (fig:attribute-layerwise-probes) from the seeded rollouts, with the 3-draw random-init band."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from palette import PAL  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

SEEDED = DATA_ROOT / "probes/goal_identity_loci_v2_rseed_20260925_seeded"
SITES = ["h1", "h2", "h3", "h_pre", "h_fc"]
RAW_SITES = ["block1_gap", "block2_gap", "block3_gap", "pre_fc", "fc"]
SITE_LABEL = [r"$h^{(1)}$", r"$h^{(2)}$", r"$h^{(3)}$", r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$"]
CONCEPTS = [("Line presence", "line_present"), ("Line role", "goal_is_line"),
            ("Yellow presence", "yellow_present"), ("Yellow role", "goal_is_yellow")]
ROWS = [("Textured", "texture"), ("Black", "black")]
T2 = 4.303  # t_{0.975, df=2}: 3 checkpoints per cohort, and 3 init draws
COH = {"GM": PAL["gm"], "Non-GM": PAL["nongm"]}
# Geometry and style match the other appendix probe figures; the script
# writes to the same fig_appendix_attribute_probes_final path the appendix includes.
PW, PH, WSP, HSP = 3.0, 2.35, 0.62, 0.55
LEFT, RIGHT, TOP, BOTTOM = 0.95, 0.15, 0.55, 0.85
ROWLAB = 0.3
STYLE = dict(line_w=2.2, mean_ms=6.5, pt_s=11, band_a=0.18, pt_a=0.75, font=11, title=12, sup=14)
INIT_LS = (0, (3.5, 1.8))


def load(bg_dir: Path) -> tuple[dict, dict]:
    """Trained AUC per checkpoint (mean over 3 rollout seeds) and per-draw init means pooled over all 6 frame sets."""
    trained: dict = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(bg_dir / "per_checkpoint.csv")):
        if r["site"] in RAW_SITES and r["cohort"] in COH:
            s = SITES[RAW_SITES.index(r["site"])]
            trained[r["concept"]][r["cohort"]].setdefault(s, []).append(float(r["trained_auc"]))
    ctrl = json.load(open(bg_dir / "summary.json"))["controls"]
    init = {k: {SITES[i]: np.array(list(ctrl[rs][k]["init_draw_means"].values())) for i, rs in enumerate(RAW_SITES)}
            for _, k in CONCEPTS}
    for k, d in init.items():
        for i, rs in enumerate(RAW_SITES):
            assert len(d[SITES[i]]) == 3 and abs(d[SITES[i]].mean() - ctrl[rs][k]["init_mean"]) < 1e-9
    return trained, init


def draw(ax, cells: dict, init: dict | None) -> dict:
    x = np.arange(len(SITES))
    ax._band_lo = 1.0
    plotted = {}
    if init is not None:
        im = np.array([init[s].mean() for s in SITES])
        ih = np.array([T2 * init[s].std(ddof=1) / np.sqrt(3) for s in SITES])
        ilo, ihi = np.clip(im - ih, 0, 1), np.clip(im + ih, 0, 1)
        ax.fill_between(x, ilo, ihi, color=PAL["init"], alpha=0.16, lw=0, zorder=1)
        ax.plot(x, im, ls=INIT_LS, color=PAL["init"], lw=1.6, zorder=2)
        ax._band_lo = float(ilo.min())
        plotted["init"] = im
    for k, (coh, col) in enumerate(COH.items()):
        vals = [np.asarray(cells[coh][s], float) for s in SITES]
        assert all(len(v) == 3 for v in vals), (coh, [len(v) for v in vals])
        m = np.array([v.mean() for v in vals])
        h = np.array([T2 * v.std(ddof=1) / np.sqrt(3) for v in vals])
        lo, hi = np.clip(m - h, 0, 1), np.clip(m + h, 0, 1)
        ax.fill_between(x, lo, hi, color=col, alpha=STYLE["band_a"], lw=0)
        ax._band_lo = min(ax._band_lo, float(lo.min()))
        ax.plot(x, m, "-o", color=col, lw=STYLE["line_w"], ms=STYLE["mean_ms"], mec="white", mew=1.0, zorder=4)
        off = -0.08 if k == 0 else 0.08
        for i, v in enumerate(vals):
            ax.scatter(np.full(3, i + off), v, s=STYLE["pt_s"], color=col, alpha=STYLE["pt_a"], lw=0, zorder=3)
        plotted[coh] = m
    ax.set_xticks(x, SITE_LABEL)
    ax.set_xlim(-0.4, len(SITES) - 0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=STYLE["font"])
    return plotted


def render(data: dict, out_stem: Path) -> dict:
    nrow, ncol = len(ROWS), len(CONCEPTS)
    W = LEFT + ROWLAB + ncol * PW + (ncol - 1) * WSP + RIGHT
    H = TOP + nrow * PH + (nrow - 1) * HSP + BOTTOM
    fig = plt.figure(figsize=(W, H))
    means, axes = {}, []
    for ri, (rlab, bg) in enumerate(ROWS):
        trained, init = data[bg]
        for ci, (ptitle, k) in enumerate(CONCEPTS):
            l = (LEFT + ROWLAB + ci * (PW + WSP)) / W
            b = (BOTTOM + (nrow - 1 - ri) * (PH + HSP)) / H
            ax = fig.add_axes((l, b, PW / W, PH / H))
            axes.append(ax)
            ini = None if init is None else init[k]
            means[f"{rlab}|{ptitle}"] = {c: list(map(float, v)) for c, v in draw(ax, trained[k], ini).items()}
            if ri == 0:
                ax.set_title(ptitle, fontsize=STYLE["title"], color=PAL["ink"])
            if ci == 0:
                ax.set_ylabel("held-out AUC", fontsize=STYLE["font"])
                fig.text(0.12 / W, b + PH / H / 2, rlab, rotation=90, va="center", ha="left",
                         fontsize=STYLE["title"], color=PAL["ink"], fontweight="bold")
    floor = min(0.4, np.floor(min(a._band_lo for a in axes) * 10) / 10)
    for a in axes:
        a.set_ylim(floor, 1.02)
    handles = [Line2D([0], [0], color=c, marker="o", lw=STYLE["line_w"], ms=STYLE["mean_ms"], mec="white", label=k)
               for k, c in COH.items()]
    handles += [Line2D([0], [0], color=PAL["init"], ls=INIT_LS, lw=1.6, label="random init (3 draws)"),
                Patch(color="#999999", alpha=0.3, lw=0, label="95% CI")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=STYLE["font"],
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Attribute Maze", fontsize=STYLE["sup"], fontweight="bold", color=PAL["ink"], y=1 - 0.12 / H)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out_stem.with_suffix(".{png,pdf}"))
    return means


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--black", type=Path, default=SEEDED)
    ap.add_argument("--texture", type=Path, default=SEEDED / "texture")
    ap.add_argument("--out-stem", type=Path, default=OUT_DIR / "fig_appendix_attribute_probes_final")
    ap.add_argument("--values-out", type=Path, default=TABLES / "plotted_values_attribute.json")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    data = {"black": load(args.black), "texture": load(args.texture)}
    means = render(data, args.out_stem)
    args.values_out.write_text(json.dumps({"sites": SITES, "means": means}, indent=1))
    print("wrote", args.values_out)


if __name__ == "__main__":
    main()
