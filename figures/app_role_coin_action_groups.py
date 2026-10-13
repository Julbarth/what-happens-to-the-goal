"""Coin action-group figure at 0.8 linewidth (4.4 in), drawn natively at that width so fonts stay at paper size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import style as fs  # noqa: E402

from paths import OUT_DIR, TABLES

GROUPS = ("star", "right", "jump", "other")
NAMES = {"star": r"$A_\star$", "right": r"$A_\rightarrow$", "jump": "jump", "other": "other"}
COHS = (("GM", "GM"), ("NonGM", "Non-GM"), ("init", "init"))


def point(ax, x, s, col, ms=fs.MS):
    ax.scatter(x + np.array([-0.05, 0, 0.05]), s["per_seed"], color=col, zorder=2, **fs.SEED)
    lo, hi = s["ci95"]
    ax.errorbar(x, s["mean"], yerr=[[s["mean"] - lo], [hi - s["mean"]]], fmt="o", color=col, ms=ms, lw=fs.LW_ERR,
                mec="white", mew=0.4, zorder=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--values", type=Path, default=TABLES / "coin_action_groups_values.json")
    ap.add_argument("--width", type=float, default=0.8 * fs.W_IN)
    a = ap.parse_args()
    fs.style()
    v = json.loads(a.values.read_text())
    w, h = a.width, 3.25
    fig = plt.figure(figsize=(w, h))
    left, gapx, right, pw = 0.42, 0.42, 0.06, None
    pw = (w - left - gapx - right) / 2
    ph, bottom, gapy = 1.02, 0.56, 0.5
    A = [[fig.add_axes([(left + c * (pw + gapx)) / w, (bottom + (1 - r) * (ph + gapy)) / h, pw / w, ph / h])
          for c in range(2)] for r in range(2)]
    for ax, metric, title, ylab, ref, L in (
            (A[0][0], "dmass", "Change per group", r"$\Delta\pi(A_g)$", 0.0, "a"),
            (A[0][1], "tv_over_mass", "Share of change / prior mass", r"TV share$/\pi(A_g)$", 1.0, "b")):
        for gi, g in enumerate(GROUPS):
            for ci, (key, name) in enumerate(COHS):
                point(ax, gi + (ci - 1) * 0.24, v["cohort"][key][f"{metric}_{g}"], fs.COH[name])
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([NAMES[g] for g in GROUPS])
        ax.axhline(ref, zorder=1, **fs.REF)
        ax.set_ylabel(ylab)
        fs.panel_title(ax, title)
        fs.panel_letter(ax, L, dx=-24)
        fs.finish(ax)
    sw = v["distance_sweep"]
    ds = sorted(sw, key=int)
    for ax, metric, title, ylab, ref, L in (
            (A[1][0], "dmass_star", r"Change on $A_\star$ vs distance", r"$\Delta\pi(A_\star)$", 0.0, "c"),
            (A[1][1], "tv_over_mass_star", r"Ratio for $A_\star$ vs distance", r"TV share$/\pi(A_\star)$", 1.0, "d")):
        for ci, (key, name) in enumerate(COHS):
            xs = np.array([int(d) for d in ds]) + (ci - 1) * 0.14
            ax.plot(xs, [sw[d][key][metric]["mean"] for d in ds], color=fs.COH[name], lw=0.8, alpha=0.7, zorder=2)
            for x, d in zip(xs, ds):
                point(ax, x, sw[d][key][metric], fs.COH[name])
        ax.set_xticks([int(d) for d in ds])
        ax.set_xlabel("coin cells left of agent")
        ax.axhline(ref, zorder=1, **fs.REF)
        ax.set_ylabel(ylab)
        fs.panel_title(ax, title)
        fs.panel_letter(ax, L, dx=-24)
        fs.finish(ax)
    hs = [fs.cohort_handle("GM"), fs.cohort_handle("Non-GM"), fs.init_handle(), fs.ci_handle(),
          Line2D([], [], **fs.REF)]
    fs.legend_row(fig, hs, ["GM", "Non-GM", "random init", fs.CI_LABEL, "reference"], y_in=0.06, handlelength=1.4,
                  columnspacing=0.9)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"app_role/fig_app_role_coin_action_groups.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
