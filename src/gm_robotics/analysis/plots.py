"""Robotics GM analysis figures (paper palette: burgundy GM, teal Non-GM, grey init; goal solid, proxy soft/open)."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C

P = C.PAL
COH = {"gm": ("GM", P["gm"]), "control": ("Non-GM", P["nongm"]), "init": ("init", P["init"])}
GOAL_EDITS = {"shape_swap", "shape_ablate", "goal_move", "goal_to_corner", "corner_to_goal"}
plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white", "axes.edgecolor": P["ink"],
                     "axes.labelcolor": P["ink"], "xtick.color": P["ink"], "ytick.color": P["ink"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def ci_err(v):
    m, lo, hi = C.ci95(np.asarray(v, float))
    return m, (m - lo if np.isfinite(lo) else 0), (hi - m if np.isfinite(hi) else 0)


def worldedit(env):
    f = os.path.join(C.RESULTS, f"worldedit_{env}_per_seed.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    edits = [e for e in ["shape_swap", "colour_swap", "shape_ablate", "colour_ablate", "goal_move", "goal_to_corner",
                         "corner_to_goal", "distractor_move"] if e in set(df.edit)]
    cols = [("KL", "KL"), ("SF", "SF"), ("Q_h_fc", r"$Q$ at $h_{\mathrm{fc}}$"), ("Q_h_pre", r"$Q$ at $h_{\mathrm{pre}}$")]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 3.0))
    for ax, (c, lab) in zip(axes, cols):
        for j, coh in enumerate(["gm", "control", "init"]):
            for i, e in enumerate(edits):
                v = df[(df.edit == e) & (df.cohort == coh)][c].to_numpy()
                if not len(v):
                    continue
                m, lo, hi = ci_err(v)
                x = i + (j - 1) * 0.25
                filled = e in GOAL_EDITS
                col = COH[coh][1]
                ax.errorbar(x, m, yerr=[[lo], [hi]], fmt="o", ms=5, color=col, mfc=col if filled else "white",
                            mec=col, capsize=2, label=COH[coh][0] if i == 0 else None)
                ax.scatter(np.full(len(v), x + 0.08), v, s=6, color=col, alpha=0.5)
        ax.set_xticks(range(len(edits)))
        ax.set_xticklabels([e.replace("_", "\n") for e in edits], fontsize=7)
        ax.set_title(lab, fontsize=9)
        if c.startswith(("KL", "Q")):
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, f"worldedit_{env}.png"), dpi=160)
    plt.close(fig)
    # G vs D_F per seed (paper Fig. fisher-decomposition).
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
    for ax, site in zip(axes, ["h_pre", "h_fc"]):
        for _, r in df.iterrows():
            col = COH[r.cohort][1]
            ax.scatter(r[f"D_F_{site}"], r[f"G_{site}"], s=22, color=col if r.edit in GOAL_EDITS else "white",
                       edgecolor=col, marker="^" if r.edit in GOAL_EDITS else "s")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel(r"$D_F$"), ax.set_ylabel(r"$G$"), ax.set_title(site.replace("h_", r"$h_{\mathrm{") + "}}$", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, f"worldedit_G_vs_DF_{env}.png"), dpi=160)
    plt.close(fig)


def probes(env):
    f = os.path.join(C.RESULTS, f"probes_{env}_per_seed.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    sites = ["conv1", "conv2", "conv3", "h_pre", "h_fc", "a_hid"]
    tg = list(dict.fromkeys(df.target))
    fig, axes = plt.subplots(1, len(tg), figsize=(2.6 * len(tg), 2.8), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), tg):
        for coh in ["gm", "control"]:
            ms = np.array([ci_err(df[(df.target == t) & (df.cohort == coh) & (df.site == s)].value) for s in sites])
            x = np.arange(len(sites))
            ax.plot(x, ms[:, 0], "-o", ms=3, color=COH[coh][1], label=COH[coh][0])
            ax.fill_between(x, ms[:, 0] - ms[:, 1], ms[:, 0] + ms[:, 2], color=COH[coh][1], alpha=0.18, lw=0)
        mi = [df[(df.target == t) & (df.cohort == "init") & (df.site == s)].value.mean() for s in sites]
        ax.plot(range(len(sites)), mi, "--", color=P["init"], label="init")
        for coh, ls in (("pixels", ":"), ("proprio", "-.")):
            v = df[(df.target == t) & (df.cohort == coh)].value
            if len(v):
                ax.axhline(v.mean(), ls=ls, color=P["null"], lw=1, label=coh)
        ax.set_xticks(range(len(sites)))
        ax.set_xticklabels(["c1", "c2", "c3", "pre", "fc", "act"], fontsize=7)
        metric = df[df.target == t].metric.iloc[0]
        ax.set_title(t.replace("_", " "), fontsize=8)
        ax.set_ylabel("AUC" if metric == "auc" else r"$R^2$", fontsize=8)
        ax.set_ylim(0.4 if metric == "auc" else -0.2, 1.02)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, f"probes_{env}.png"), dpi=160)
    plt.close(fig)


def steering():
    f = os.path.join(C.RESULTS, "steer_AT_per_state.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    df["goal"] = (df.outcome == "goal").astype(float)
    per = df.groupby(["cohort", "seed", "arm", "beta"]).goal.mean().reset_index()
    base = per[per.arm == "baseline"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharey=True)
    for r, coh in enumerate(["gm", "control"]):
        b = base[base.cohort == coh].goal
        for c, arms in enumerate([[("local", "-", "corrective"), ("random", ":", "random")],
                                  [("shared", "-", "sign-gated"), ("fisher_proj", "--", "Fisher-proj."),
                                   ("unsigned", ":", "unsigned"), ("fixed_local", "-.", "local fixed")]]):
            ax = axes[r, c]
            m, lo, hi = ci_err(b)
            ax.axhspan(m - lo, m + hi, color=P["init"], alpha=0.2, lw=0)
            ax.axhline(m, ls=":", color=P["init"])
            for arm, ls, lab in arms:
                sub = per[(per.cohort == coh) & (per.arm == arm)]
                if not len(sub):
                    continue
                bs = sorted(sub.beta.unique()) if arm != "random" else sorted(sub.beta.unique())
                ms = np.array([ci_err(sub[sub.beta == x].groupby("seed").goal.mean()) for x in bs])
                col = COH[coh][1]
                ax.plot(bs, ms[:, 0], ls, marker="o", ms=3, color=col, label=lab)
                ax.fill_between(bs, ms[:, 0] - ms[:, 1], ms[:, 0] + ms[:, 2], color=col, alpha=0.15, lw=0)
            ax.axvline(0, color=P["mute"], lw=0.5)
            ax.set_xlabel(r"dose $\beta$")
            ax.set_ylabel(r"goal reach $\hat J^\star$")
            ax.set_title(f"{COH[coh][0]}, {'state-local' if c == 0 else 'fixed at $s_0$'}", fontsize=9)
            ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "steer_AT_dose.png"), dpi=160)
    plt.close(fig)


def rewiring():
    f = os.path.join(C.RESULTS, "rewire_LR", "closed_loop_per_episode.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    df["goal"] = (df.outcome == "goal").astype(float)
    per = df.groupby(["cohort", "seed", "arm", "set"]).goal.mean().reset_index()
    sets = [s for s in ["offpath", "out_region", "in_region"] if s in set(per.set)]
    arms = ["native", "fc_head", "remix", "pre", "untrained"]
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.0), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), sets):
        for j, coh in enumerate(["gm", "control"]):
            for i, arm in enumerate(arms):
                v = per[(per.set == s) & (per.cohort == coh) & (per.arm == arm)].goal
                if not len(v):
                    continue
                m, lo, hi = ci_err(v)
                x = i + (j - 0.5) * 0.3
                col = COH[coh][1]
                ax.errorbar(x, m, yerr=[[lo], [hi]], fmt="o", color=col, capsize=2, label=COH[coh][0] if i == 0 else None)
                ax.scatter(np.full(len(v), x + 0.07), v, s=6, color=col, alpha=0.5)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(["native", "fc", "remix", "pre", "untr."], fontsize=7)
        ax.set_title(s.replace("_", " "), fontsize=9)
        ax.set_ylim(-0.03, 1.03)
    np.atleast_1d(axes)[0].set_ylabel(r"goal reach $\hat J^\star$")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "rewire_LR.png"), dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["worldedit", "probes", "steering", "rewiring"])
    a = p.parse_args()
    C.ensure_dirs()
    for w in a.which:
        if w in ("worldedit", "probes"):
            for env in ("AT", "LR"):
                globals()[w](env)
        else:
            globals()[w]()


if __name__ == "__main__":
    main()
