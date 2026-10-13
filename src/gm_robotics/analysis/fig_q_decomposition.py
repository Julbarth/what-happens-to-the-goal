"""Q = G * D_F decomposition at h_fc across the robot tasks, with the paper's Procgen values as reference panels."""
import argparse
import os

import matplotlib
from gm_robotics.paths import ROOT as RUN_ROOT
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = str(RUN_ROOT)
GM, GMP, CT, CTP, RND, INK, MUTE = "#7A1520", "#C45C58", "#3D8B8B", "#5FA8A0", "#8E8E8E", "#2A2A2A", "#6E6E6E"
COL = {("gm", "goal"): GM, ("gm", "proxy"): GMP, ("control", "goal"): CT, ("control", "proxy"): CTP, ("init", "goal"): RND, ("init", "proxy"): RND}
ROBOT = [("LR", "LocationReach", f"{ROOT}/results/analysis/worldedit_LR_per_seed.csv", None, [("goal_move", "goal")]),
         ("PTM", "PushToMarker", f"{ROOT}/results/analysis/worldedit_PTM_per_seed.csv", None, [("marker_move", "goal")]),
         ("AT", "AttributeTable", f"{ROOT}/results/analysis/worldedit_AT_per_seed.csv", None, [("shape_swap", "goal"), ("colour_swap", "proxy")]),
         ("SZ", "SortZones", f"{ROOT}/results/analysis/worldedit_SZ_per_seed.csv", None, [("mark_swap", "goal"), ("colour_swap", "proxy")]),
         ("Isaac", "Isaac quadruped", f"{ROOT}/results/isaac_analysis/worldedit_per_seed.csv", "visible", [("flag_swap", "goal"), ("tower_swap", "proxy")])]
# Paper Table 7, h_fc, mean of 3 seed medians.
# Attribute Maze = black background, canonical 196-state conflict bank, A.2.2 swaps. Values are transcribed, not recomputed.
PAPER = [("Maze", {("gm", "goal"): (0.727, 1.59e-3), ("control", "goal"): (1.141, 1.667), ("init", "goal"): (0.962, 6.24e-9)}),
         ("Coin", {("gm", "goal"): (1.350, 0.0192), ("control", "goal"): (1.784, 0.2073), ("init", "goal"): (1.014, 5.55e-8)}),
         ("Attribute Maze", {("gm", "goal"): (17.165, 0.066), ("control", "goal"): (10.268, 0.395), ("init", "goal"): (0.914, 2.10e-7),
                             ("gm", "proxy"): (13.759, 0.142), ("control", "proxy"): (2.121, 0.013), ("init", "proxy"): (1.015, 2.11e-7)})]
XL, YL = (1e-10, 3e2), (0.4, 40)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK})


def iso_q(ax, label=False):
    xs = np.logspace(np.log10(XL[0]), np.log10(XL[1]), 50)
    for e in range(-10, 4, 2):
        ax.plot(xs, 10.0 ** e / xs, ":", color=RND, lw=0.9, zorder=1)
        if label and e in (-8, -4, 0):
            x = 10.0 ** e / YL[0] * 1.1
            ax.text(x, YL[0] * 1.05, f"Q=10$^{{{e}}}$" if e else "Q=1", fontsize=10, color=MUTE, ha="left", va="bottom")


def style(ax, title, first):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XL)
    ax.set_ylim(YL)
    ax.set_xticks([1e-9, 1e-6, 1e-3, 1e0])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r"$D_F$")
    if first:
        ax.set_ylabel(r"$G$ at $h_{\mathrm{fc}}$")


def robot_panel(ax, path, bank, edits):
    df = pd.read_csv(path)
    if bank:
        df = df[df.bank == bank]
    rows = []
    for edit, role in edits:
        mk = "^" if role == "goal" else "s"
        ini = df[(df.edit == edit) & (df.cohort == "init")].G_h_fc.mean()
        # Init actor heads are scaled to ~0.01, so init D_F ~ 1e-9 by construction: the init cluster sits far left.
        for coh in ("init", "control", "gm"):
            sub = df[(df.edit == edit) & (df.cohort == coh)]
            col = COL[(coh, role)]
            face = col if role == "goal" else "white"
            x, y = sub.D_F_h_fc.to_numpy(), sub.G_h_fc.to_numpy()
            ax.scatter(x, y, s=22, marker=mk, color=face, edgecolor=col, alpha=0.55, lw=0.9, zorder=2)
            ax.scatter(x.mean(), y.mean(), s=110, marker=mk, color=face, edgecolor=col, lw=2.0, zorder=3)
            rows.append(dict(edit=edit, role=role, cohort=coh, n=len(sub), G=y.mean(), D_F=x.mean(), Q=sub.Q_h_fc.mean(),
                             GxD_of_means=y.mean() * x.mean(), G_init=ini))
    return rows


def paper_panel(ax, vals):
    for (coh, role), (g, d) in vals.items():
        col = COL[(coh, role)]
        ax.scatter(d, g, s=110, marker="^" if role == "goal" else "s", color=col if role == "goal" else "white", edgecolor=col, lw=2.0, zorder=3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=f"{ROOT}/figures/analysis")
    p.add_argument("--dpi", type=int, default=150)
    a = p.parse_args()
    fig = plt.figure(figsize=(17, 8.2))
    w, h, x0, gap = 0.163, 0.33, 0.06, 0.028
    rows = []
    for k, (key, name, path, bank, edits) in enumerate(ROBOT):
        ax = fig.add_axes([x0 + k * (w + gap), 0.56, w, h])
        iso_q(ax, label=(k == 0))
        rows += [dict(env=key, **r) for r in robot_panel(ax, path, bank, edits)]
        style(ax, name, k == 0)
    fig.text(x0, 0.40, "Paper reference (Procgen, Table 7)", fontsize=14, weight="bold", color=MUTE)
    for k, (name, vals) in enumerate(PAPER):
        ax = fig.add_axes([x0 + k * (w + gap), 0.075, w, h * 0.85])
        iso_q(ax)
        paper_panel(ax, vals)
        style(ax, name, k == 0)
        for (coh, role), (g, d) in vals.items():
            rows.append(dict(env=f"paper_{name}", edit=role, role=role, cohort=coh, n=3, G=g, D_F=d, Q=np.nan, GxD_of_means=g * d))
    h_ = [Line2D([], [], marker="o", ls="", color=GM, ms=10, label="GM"), Line2D([], [], marker="o", ls="", color=CT, ms=10, label="control"),
          Line2D([], [], marker="o", ls="", color=RND, ms=10, label="random network"),
          Line2D([], [], marker="^", ls="", color=INK, ms=10, label="goal edit"),
          Line2D([], [], marker="s", ls="", mfc="white", color=INK, ms=10, label="proxy edit"),
          Line2D([], [], color=RND, ls=":", lw=1, label="iso-Q (every 100x)")]
    fig.legend(handles=h_, loc="center left", bbox_to_anchor=(0.62, 0.24), ncol=2, frameon=False, fontsize=13)
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/q_decomposition.png", dpi=a.dpi, facecolor="white")
    pd.DataFrame(rows).to_csv(f"{a.out}/q_decomposition_values.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
