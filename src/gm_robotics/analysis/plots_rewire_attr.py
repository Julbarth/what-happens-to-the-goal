"""AttributeTable / SortZones rewiring figures: goal and proxy reach per arm, and cue-decoupled target selection."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C
from gm_robotics.analysis.plots_paper import P, ci, save

ENVS = {"AT": "AttributeTable", "SZ": "SortZones"}
HUE = {"gm": (P["gm"], P["gm_proxy"]), "control": (P["nongm"], P["nongm_proxy"]), "init": (P["init"], "#C8C8C8")}
NAME = {"gm": "GM", "control": "Non-GM", "init": "init"}
ARMS = [("native", "native"), ("fc_head", r"head on $h_{fc}$"), ("pre", r"from $h_{pre}$"), ("remix", "random remix"),
        ("untrained", "untrained"), ("pre_masked", r"$h_{pre}$, goal cue masked")]


def bars(ax, x, s, coh, w=0.13):
    for v, c, off in ((s.P_u, HUE[coh][0], -w / 2), (s.P_v, HUE[coh][1], w / 2)):
        if not len(v):
            continue
        m, lo, hi = ci(v)
        ax.bar(x + off, m, w * 0.92, color=c, yerr=[[lo], [hi]], ecolor=P["ink"], capsize=1.5, lw=0)
        ax.scatter(np.full(len(v), x + off), v, s=5, color=P["ink"], zorder=3, lw=0)


def fig_reach(ps):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    for j, env in enumerate(ENVS):
        ax = axes[j]
        d = ps[(ps.env == env) & (ps.cond == "standard")]
        for a_i, (arm, _) in enumerate(ARMS):
            for k, coh in enumerate(("gm", "control", "init")):
                bars(ax, a_i + (k - 1) * 0.3, d[(d.arm == arm) & (d.cohort == coh)], coh)
        rw = d[d.arm == "rand_walk"]
        if len(rw):
            ax.axhline(rw.P_u.mean(), color=P["null"], lw=0.8, ls=":")
        ax.set_xticks(range(len(ARMS)), [a[1] for a in ARMS], rotation=30, ha="right", fontsize=8)
        ax.set_title(ENVS[env], fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("P(first contact)")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (P["gm"], P["gm_proxy"], P["nongm"], P["nongm_proxy"], P["init"], "#C8C8C8")]
    fig.legend(h, ["GM goal", "GM proxy", "Non-GM goal", "Non-GM proxy", "init goal", "init proxy"], loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.tight_layout()
    save(fig, "fig_rewire_attr")


def fig_cues(ps):
    conds = {"AT": [("standard", "standard"), ("nogoal_sph", "no cube"), ("nogoal_cube", "two cubes"),
                    ("noproxy_blue", "no red"), ("noproxy_red", "two red")],
             "SZ": [("standard", "standard"), ("nogoal", "no cross"), ("noproxy_green", "two green"),
                    ("noproxy_purple", "no green")]}
    series = [("gm", "native", "o", True), ("gm", "pre", "o", True), ("control", "pre", "s", True),
              ("init", "pre", "^", True), ("gm", "pre_masked", "o", False)]
    lab = {("gm", "native"): "GM native", ("gm", "pre"): r"GM $h_{pre}$", ("control", "pre"): r"Non-GM $h_{pre}$",
           ("init", "pre"): r"init $h_{pre}$", ("gm", "pre_masked"): r"GM $h_{pre}$ masked"}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.2), sharey="row")
    for j, env in enumerate(ENVS):
        d = ps[ps.env == env]
        for i, col in enumerate(("P_u_dec", "D")):
            ax = axes[i, j]
            for s_i, (coh, arm, mk, filled) in enumerate(series):
                xs = np.arange(len(conds[env])) + (s_i - 2) * 0.12
                for x, (c, _) in zip(xs, conds[env]):
                    v = d[(d.cohort == coh) & (d.arm == arm) & (d.cond == c)][col].dropna()
                    if not len(v):
                        continue
                    m, lo, hi = ci(v)
                    color = HUE[coh][0] if not (coh == "gm" and arm == "native") else HUE["gm"][1]
                    ax.errorbar(x, m, yerr=[[lo], [hi]], fmt=mk, color=color, mfc=color if filled else "white",
                                ms=5, capsize=2, lw=1, label=lab[(coh, arm)] if x == xs[0] else None)
            if col == "P_u_dec":
                ax.axhline(0.5, color=P["mute"], lw=0.8, ls=":")
            ax.set_xticks(range(len(conds[env])), [c[1] for c in conds[env]], fontsize=8)
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(ENVS[env], fontsize=10)
        axes[0, 0].set_ylabel("P(goal location | decisive)")
        axes[1, 0].set_ylabel("decisive fraction $D$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout()
    save(fig, "fig_rewire_attr_cues")


def main():
    ps = pd.read_csv(os.path.join(C.RESULTS, "rewire_attr", "per_seed.csv"))
    fig_reach(ps)
    fig_cues(ps)


if __name__ == "__main__":
    main()
