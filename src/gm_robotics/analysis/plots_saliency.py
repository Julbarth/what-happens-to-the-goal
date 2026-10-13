"""Fisher input saliency figures (example maps and region mass) for AttributeTable, LocationReach and SortZones."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from gm_robotics.analysis import common as C
from gm_robotics.analysis import plots_paper as PP

P = C.PAL
SDIR = os.path.join(C.RESULTS, "saliency")
ENVS = {"AT": "AttributeTable", "LR": "LocationReach", "SZ": "SortZones", "PTM": "PushToMarker"}
PAIR0 = {"AT": (2, 2), "LR": (1, 1), "SZ": (0, 0), "PTM": (0, 0)}
# Mid-episode frame step of saliency_frames.py per env (PTM has a 100-step horizon).
STEP = {"AT": 10, "LR": 10, "SZ": 10, "PTM": 20}
RING = {"goal": "#2DBE4E", "proxy": "#F28E1C"}
REG_LABEL = {"AT": [("goal", "goal"), ("proxy", "proxy")], "SZ": [("goal", "goal"), ("proxy", "proxy")],
             "LR": [("goal", "goal"), ("proxy", "proxy"), ("mirror", "other")],
             "PTM": [("goal", "goal"), ("proxy", "proxy"), ("cube", "cube"), ("mirror", "other")]}


def log_rel(S):
    return np.log10(np.maximum(S, 1e-30) / S.mean())


def overlay(ax, rgb, S, vmin=-1, vmax=2):
    grey = rgb.astype(float).mean(-1) / 255
    ax.imshow(grey, cmap="gray", vmin=0, vmax=2.6, interpolation="nearest")
    L = log_rel(S)
    a = np.clip((L - vmin) / (vmax - vmin), 0, 1)
    return ax.imshow(np.clip(L, vmin, vmax), cmap="magma", vmin=vmin, vmax=vmax, alpha=a, interpolation="nearest")


def rings(ax, z, i, radius, env):
    ax.add_patch(Circle(z["px_goal"][i] - 0.5, radius, fill=False, ec=RING["goal"], lw=1.6))
    # The LocationReach proxy is a location without a visual token: dashed ring at the projected training corner.
    ax.add_patch(Circle(z["px_proxy"][i] - 0.5, radius, fill=False, ec=RING["proxy"], lw=1.6,
                        ls="--" if env in ("LR", "PTM") else "-"))


def examples(env, frames, n_rows=4):
    z = np.load(os.path.join(SDIR, f"frames_{env}.npz"))
    m = np.load(os.path.join(SDIR, f"maps_{env}.npz"))
    radius = pd.read_csv(os.path.join(SDIR, f"saliency_{env}_per_state.csv"), nrows=1).radius.iloc[0]
    sg, sc = PAIR0[env]
    if frames == "start":
        rgb, key = z["rgb0"], "start"
    else:
        rgb, key = z[f"mid_gm_s{sg}_rgb"], "mid_gm"
    fig, axes = plt.subplots(n_rows, 3, figsize=(5.4, 1.85 * n_rows))
    for i in range(n_rows):
        ax = axes[i, 0]
        ax.imshow(rgb[i], interpolation="nearest")
        for j, (coh, s) in enumerate((("gm", sg), ("control", sc)), start=1):
            im = overlay(axes[i, j], rgb[i], m[f"{coh}_s{s}_{key}"][i])
        for ax in axes[i]:
            rings(ax, z, i, radius, env)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True), sp.set_color(P["ink"]), sp.set_linewidth(0.8)
    for j, (t, c) in enumerate((("state", P["ink"]), ("GM", P["gm"]), ("Non-GM", P["nongm"]))):
        axes[0, j].set_title(t, color=c, fontsize=10)
    fig.subplots_adjust(wspace=0.05, hspace=0.06, right=0.88)
    cax = fig.add_axes([0.9, 0.3, 0.018, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.solids.set_alpha(1)
    cb.set_label(r"$\log_{10} S(x)\,/\,\bar S$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    h = [Line2D([], [], marker="o", ms=8, mfc="none", mec=RING["goal"], ls="", mew=1.6, label="goal"),
         Line2D([], [], marker="o", ms=8, mfc="none", mec=RING["proxy"], ls="", mew=1.6, label="proxy")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.45, 0.115))
    PP.save(fig, f"fig_saliency_examples_{env}" + ("" if frames == "start" else f"_step{STEP[env]}"))


def mass_panel(envs, name="fig_saliency_mass"):
    """Rows: start state, step 10. Columns: env. Per region, seed dots and mean with 95% t-CI per cohort."""
    fig, axes = plt.subplots(2, len(envs), figsize=(3.3 * len(envs), 5.6), squeeze=False)
    for c, env in enumerate(envs):
        ps = pd.read_csv(os.path.join(SDIR, f"saliency_{env}_per_seed.csv"))
        regs = REG_LABEL[env]
        for r, fs in enumerate(("start", "mid")):
            ax = axes[r, c]
            for k, (reg, lab) in enumerate(regs):
                for dx, (coh, col) in zip((-0.25, 0.0, 0.25), (("gm", P["gm"]), ("control", P["nongm"]),
                                                               ("init", P["init"]))):
                    v = ps[(ps.frames == fs) & (ps.cohort == coh)][f"mass_{reg}"].to_numpy()
                    face = col if reg == "goal" else "white"
                    ax.scatter(np.full(len(v), k + dx - 0.07), v, s=16, color=face, edgecolor=col, lw=1, zorder=3)
                    mn, lo, hi = PP.ci(v)
                    ax.errorbar(k + dx + 0.03, mn, yerr=[[lo], [hi]], fmt="_", ms=12, mew=2.2, color=col, lw=1.6,
                                zorder=4)
            ax.set_ylim(bottom=0)
            ax.set_xticks(range(len(regs)), [l for _, l in regs])
            ax.set_xlim(-0.6, len(regs) - 0.4)
            if c == 0:
                ax.set_ylabel(("start state" if fs == "start" else f"step {STEP[env]}") + "\nsaliency mass fraction")
            if r == 0:
                ax.set_title(ENVS[env], fontsize=10)
    h = [Line2D([], [], marker="o", ls="", color=P["gm"], label="GM"),
         Line2D([], [], marker="o", ls="", color=P["nongm"], label="Non-GM"),
         Line2D([], [], marker="o", ls="", color=P["init"], label="init"),
         Line2D([], [], marker="o", ls="", color=P["ink"], label="goal"),
         Line2D([], [], marker="o", ls="", mfc="white", color=P["ink"], label="proxy / other"),
         Line2D([], [], marker="_", ms=12, mew=2.2, ls="", color=P["mute"], label="mean ± 95% CI")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    PP.save(fig, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=["AT", "LR", "SZ"])
    p.add_argument("--n_rows", type=int, default=4)
    p.add_argument("--mass_name", default="fig_saliency_mass")
    a = p.parse_args()
    for env in a.envs:
        for fs in ("start", "mid"):
            examples(env, fs, a.n_rows)
    mass_panel(a.envs, a.mass_name)


if __name__ == "__main__":
    main()
