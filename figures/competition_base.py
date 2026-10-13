"""Main-text Sec. 3.4 figure: within-state swap cue cosines (black cohorts) plus h_fc d_comp heatmaps for GM black."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paths import DATA_ROOT, OUT_DIR, ROOT, TABLES

sys.path.insert(0, str(ROOT / "scripts" / "procgen"))
from palette import PAL  # noqa: E402
from canon_geometry_run import BANK_DIR, load_state_table  # noqa: E402

GEOM = DATA_ROOT / "probes/canon_geometry_20260924"
FS_TICK, FS_LABEL, FS_TITLE = 7.0, 7.5, 8.0
LW = 0.7
# Mean marker, cap size and seed-dot style; main_competition.py overrides them with the shared main-figure style.
MS, CAP, SEED_S, SEED_A, ELW = 2.6, 1.6, 2.0, 0.45, LW
T975_DF2 = 4.302652729911275

# Main text is black background and A.2.2 swaps only; texture and ablations stay in the appendix.
COHORTS = [("gm", "GM", "gm"), ("deco", "Non-GM", "nongm"), ("init", "random init", "init")]
GROUPS = [("cos_e", "$\\cos_E$"), ("cos_f", "$\\cos_F$")]
GM_BLACK_SEEDS = ["6313", "6315", "6317"]


def t_ci(v: list[float]) -> tuple[float, float]:
    a = np.asarray(v, float)
    return float(a.mean()), float(T975_DF2 * a.std(ddof=1) / np.sqrt(len(a)))


def within_state(tap: str) -> dict:
    out: dict = {}
    for c in json.loads((GEOM / "swap" / "summaries.json").read_text()):
        if c["tap"] == tap and c["background"] == "black":
            out.setdefault(c["cohort"], []).append(c["summary"]["all"])
    return out


def panel_within(ax, tap: str, record: dict) -> None:
    cells = within_state(tap)
    width = len(COHORTS) + 1.0
    for g, (key, _) in enumerate(GROUPS):
        for j, (co, name, col) in enumerate(COHORTS):
            vals = [s[key] for s in cells[co]]
            assert len(vals) == 3, (co, key, len(vals))
            m, h = t_ci(vals)
            x = g * width + j
            ax.scatter(np.full(3, x - 0.2), vals, s=SEED_S, color=PAL[col], alpha=SEED_A, lw=0, zorder=3)
            ax.errorbar(x + 0.08, m, yerr=h, fmt="o", ms=MS, mfc=PAL[col], mec=PAL[col], mew=LW,
                        color=PAL[col], capsize=CAP, capthick=LW, elinewidth=ELW, zorder=4)
            record[f"swap|{key}|{name}|black"] = {"mean": round(m, 3), "ci95": round(h, 3),
                                                   "seeds": [round(v, 3) for v in vals]}
    ax.axhline(0, color="#CFCFCF", lw=0.6, zorder=1)
    ax.set_xticks([g * width + (len(COHORTS) - 1) / 2 for g in range(len(GROUPS))], [g[1] for g in GROUPS])
    ax.set_xlim(-0.7, (len(GROUPS) - 1) * width + len(COHORTS) - 0.3)
    ax.set_ylim(-1.1, 0.6)
    ax.set_yticks([-1, -0.5, 0, 0.5])
    ax.set_ylabel(r"cosine$(\Delta_\star,\Delta_p)$", fontsize=FS_LABEL, labelpad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, length=2, width=LW, pad=1.5)
    ax.tick_params(axis="x", labelsize=FS_LABEL)
    ax.set_title("Within a state (swap)", fontsize=FS_TITLE, fontweight="bold", pad=3)


def panel_heat(ax, key: str, tap: str, st: dict, title: str, with_ylabels: bool, cmap):
    mats = []
    for s in GM_BLACK_SEEDS:
        z = np.load(GEOM / "swap" / "per_seed" / f"matrices_gm_{s}_black_{tap}.npz")
        if list(z["state_key"]) != st["state_key"]:
            raise SystemExit(f"state order mismatch: gm {s}")
        mats.append(z[key])
    m = np.mean(np.stack(mats), axis=0)
    cut = int((st["sign"] == 1).sum())
    n = len(st["sign"])
    im = ax.imshow(m, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
    ax.set_anchor("N" if with_ylabels else "NW")
    ax.axhline(cut - 0.5, color="black", lw=LW)
    ax.axvline(cut - 0.5, color="black", lw=LW)
    centers = [(cut - 1) / 2, cut + (n - cut - 1) / 2]
    ax.set_xticks(centers, ["UP", "RIGHT"], fontsize=FS_TICK)
    ax.set_yticks(centers, ["UP", "RIGHT"] if with_ylabels else [], fontsize=FS_TICK, rotation=90, va="center")
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=3)
    same = np.concatenate([m[:cut, :cut][np.triu_indices(cut, 1)], m[cut:, cut:][np.triu_indices(n - cut, 1)]])
    return im, float(same.mean()), m.shape


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tap", default="h_fc", choices=["h_pre", "h_fc"])
    ap.add_argument("--out-stem", type=Path, default=OUT_DIR / "fig_main_sec3_4")
    ap.add_argument("--values-out", type=Path, default=TABLES / "fig_main_sec3_4_values.json")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans", "axes.linewidth": LW,
                         "axes.edgecolor": PAL["ink"], "text.color": PAL["ink"], "xtick.color": PAL["ink"],
                         "ytick.color": PAL["ink"], "axes.labelcolor": PAL["ink"]})
    cmap = plt.get_cmap("RdBu_r").copy()
    # Diagonal (self-similarity) is NaN by construction; grey keeps it distinct from kappa = 0 (white).
    cmap.set_bad("#BDBDBD")
    st = load_state_table(BANK_DIR / "ras_gt_bank_canonical.json")
    fig = plt.figure(figsize=(5.5, 1.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], left=0.085, right=0.935, top=0.87, bottom=0.235,
                          wspace=0.2)
    record: dict = {"source": str(GEOM.relative_to(ROOT)), "tap": args.tap, "edits": "swap", "background": "black"}
    ax0 = fig.add_subplot(gs[0])
    panel_within(ax0, args.tap, record)
    im, e_same, shape = panel_heat(fig.add_subplot(gs[1]), "kappa_e", args.tap, st, "Across states, $\\kappa_E$",
                                   True, cmap)
    ax_f = fig.add_subplot(gs[2])
    _, f_same, _ = panel_heat(ax_f, "kappa_f", args.tap, st, "Across states, $\\kappa_F$",
                              False, cmap)
    record["heatmap_gm_black_seedmean_same_sign"] = {"kappa_e": round(e_same, 3), "kappa_f": round(f_same, 3),
                                                     "n_states": shape[0]}
    ax_f.apply_aspect()
    pos = ax_f.get_position()
    cb = fig.colorbar(im, cax=fig.add_axes((pos.x1 + 0.012, pos.y0, 0.011, pos.height)), ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=FS_TICK, length=1.5, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.solids.set_rasterized(True)
    handles = [Line2D([], [], ls="", marker="o", ms=2.6, mfc=PAL[c], mec=PAL[c], mew=LW, label=name)
               for _, name, c in COHORTS]
    ax0.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
               fontsize=FS_TICK, handletextpad=0.1, columnspacing=0.8, borderaxespad=0, handlelength=1.0)
    fig.savefig(args.out_stem.with_suffix(".png"), dpi=400)
    fig.savefig(args.out_stem.with_suffix(".pdf"), dpi=600)
    plt.close(fig)
    args.values_out.write_text(json.dumps(record, indent=1))
    print("wrote", args.out_stem, args.values_out)


if __name__ == "__main__":
    main()
