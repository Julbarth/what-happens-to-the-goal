"""Competition geometry figures for AttributeTable and SortZones: cue cosines, d_comp cosine matrices, tug of war on u_comp."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from gm_robotics.analysis import common as C
from gm_robotics.analysis import plots_paper as PP

P = C.PAL
GDIR = os.environ.get("GEOM_DIR", C.RESULTS)
ENVS = {"AT": "AttributeTable", "SZ": "SortZones"}
GOAL_CUE = {"AT": "shape", "SZ": "cross"}
COH = {"gm": "GM", "control": "Non-GM", "init": "init"}


def seed_mean_matrices(env, cohort):
    """Seed-mean cosE / cosF over the fit states, re-ordered by goal side (left first) and, within a side, by
    candidate separation |u - v|. axis_geometry.py stores each matrix sorted by a stable argsort of -sign, which is
    undone first so every seed gets the same order before averaging."""
    meta = np.load(os.path.join(GDIR, "axis", f"meta_{env}.npz"))
    files = sorted(f for f in os.listdir(os.path.join(GDIR, "axis")) if f.startswith(f"{env}_{cohort}_s"))
    out = {"cosE": [], "cosF": []}
    for f in files:
        z = np.load(os.path.join(GDIR, "axis", f))
        fit = z["fit"]
        sign = meta["sign"][fit].astype(int)
        old = np.argsort(-sign, kind="stable")
        assert np.array_equal(z["sign_sorted"], sign[old])
        inv = np.argsort(old)
        sep = np.linalg.norm(meta["u"][fit] - meta["v"][fit], axis=1)
        new = np.lexsort((sep, -sign))
        for k in out:
            m = z[k].astype(np.float32)[inv][:, inv]
            out[k].append(m[new][:, new])
    sign_new = sign[new]
    return {k: np.mean(v, 0) for k, v in out.items()}, sign_new, len(files)


def draw_matrix(ax, M, sign, ylabels=True):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    nl = int((sign > 0).sum())
    ax.axhline(nl - 0.5, color="k", lw=1.0), ax.axvline(nl - 0.5, color="k", lw=1.0)
    ax.set_xticks([])
    ax.set_yticks([nl / 2, nl + (len(sign) - nl) / 2], ["left", "right"] if ylabels else ["", ""], rotation=90,
                  va="center")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
    return im


def main_figure(env):
    """Same four panels as the paper's competition figure: (a) within state, (b, c) GM Euclidean / Fisher
    matrices, (d) tug of war."""
    ps = pd.read_csv(os.path.join(GDIR, f"axis_{env}_per_seed.csv"))
    st = pd.read_csv(os.path.join(GDIR, f"axis_{env}_per_state.csv"))
    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.05], width_ratios=[1, 1, 1, 1.05], hspace=0.12, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    sub = {c: ps[ps.cohort == c] for c in ("gm", "control", "init")}
    PP.cohort_points(ax, [0, 1], {c: [s.cosE_within.to_numpy(), s.cosF_within.to_numpy()] for c, s in sub.items()})
    ax.set_xticks([0, 1], [r"$\cos_E$", r"$\cos_F$"])
    ax.axhline(0, color=P["init"], lw=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("cosine")
    PP.panel_label(ax, 0, "Within a state")
    mats, sign, _ = seed_mean_matrices(env, "gm")
    for j, (k, t) in enumerate((("cosE", "Euclidean"), ("cosF", "Fisher"))):
        ax = fig.add_subplot(gs[0, 1 + j])
        im = draw_matrix(ax, mats[k], sign, ylabels=j == 0)
        PP.panel_label(ax, 1 + j, t)
    cax = fig.add_subplot(gs[1, 1:3])
    fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    ax = fig.add_subplot(gs[0, 3])
    allv = []
    for coh, col in (("init", P["init"]), ("control", P["nongm"]), ("gm", P["gm"])):
        s = st[st.cohort == coh]
        g, pr = s.alpha_goal.abs().to_numpy(), s.alpha_prox.abs().to_numpy()
        allv += [g, pr]
        ax.scatter(g, pr, s=5, color=col, alpha=0.3, lw=0)
        pg = ps[ps.cohort == coh]
        mg, lg, hg = PP.ci(pg.pull_goal)
        mp, lp, hp = PP.ci(pg.pull_proxy)
        ax.errorbar(mg, mp, xerr=[[min(lg, 0.95 * mg)], [hg]], yerr=[[min(lp, 0.95 * mp)], [hp]], fmt="o", ms=6,
                    color=col, mec="white", mew=0.8, lw=1.6, zorder=4)
    v = np.concatenate(allv)
    v = v[v > 0]
    lim = (np.quantile(v, 0.01) / 2, v.max() * 2)
    ax.plot(lim, lim, "--", color=P["mute"], lw=1)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel(rf"{GOAL_CUE[env]} pull $|\alpha_F(\Delta_\star)|$")
    ax.set_ylabel(r"colour pull $|\alpha_F(\Delta_p)|$")
    PP.panel_label(ax, 3, "Tug of war")
    h = [Line2D([], [], marker="o", ls="", color=P["gm"], label="GM"),
         Line2D([], [], marker="o", ls="", color=P["nongm"], label="Non-GM"),
         Line2D([], [], marker="o", ls="", color=P["init"], label="init"),
         Line2D([], [], marker="o", color=P["mute"], label="mean ± 95% CI")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    PP.save(fig, f"fig_competition_paper_{env}")


def matrices_all(env):
    """Supplementary: seed-mean matrices for GM, Non-GM and init, Euclidean (top) and Fisher (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.0))
    for j, coh in enumerate(("gm", "control", "init")):
        mats, sign, n = seed_mean_matrices(env, coh)
        for i, (k, lab) in enumerate((("cosE", r"$\kappa_E$ (Euclidean)"), ("cosF", r"$\kappa_F$ (Fisher)"))):
            im = draw_matrix(axes[i, j], mats[k], sign, ylabels=j == 0)
            if j == 0:
                axes[i, j].set_ylabel(lab)
        axes[0, j].set_title(COH[coh], color={"gm": P["gm"], "control": P["nongm"], "init": P["init"]}[coh],
                             fontsize=10)
    fig.subplots_adjust(right=0.86, wspace=0.08, hspace=0.08)
    cax = fig.add_axes([0.885, 0.25, 0.018, 0.5])
    fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1]).set_label("seed-mean cosine", fontsize=8)
    PP.save(fig, f"fig_competition_matrices_all_{env}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=["AT", "SZ"])
    a = p.parse_args()
    for env in a.envs:
        main_figure(env)
        matrices_all(env)


if __name__ == "__main__":
    main()
