"""Paper-style PushToMarker figures: native behaviour, world-edit outside / inside, probes by depth, steering dose and rewiring with chance checks."""
import argparse
import glob
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from gm_robotics.analysis import common as C
from gm_robotics.analysis import plots_paper as PP
from gm_robotics.analysis.plots_paper import P, ci

PTM_FIG = os.path.join(C.FIGURES, "ptm")
SUMM = os.path.join(C.RESULTS, "ptm")
HUE = {"gm": (P["gm"], P["gm_proxy"]), "control": (P["nongm"], P["nongm_proxy"]), "init": (P["init"], "#C8C8C8")}
NAME = {"gm": "GM", "control": "Non-GM", "init": "init"}


def save(fig, name):
    """Paper-style copy plus a copy in figures/analysis/ptm/."""
    PP.save(fig, name)
    os.makedirs(PTM_FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        shutil.copy(os.path.join(PP.OUT, f"{name}.{ext}"), os.path.join(PTM_FIG, f"{name}.{ext}"))


def bar(ax, x, v, color, w, dots=True, log=False):
    v = np.asarray(v, float)
    m, lo, hi = ci(v)
    if log:
        # log axis: a t-interval reaching below zero is drawn to the axis floor
        lo = max(0.0, min(lo, m - 1e-3))
    ax.bar(x, m, w * 0.92, color=color, yerr=[[lo], [hi]], ecolor=P["ink"], capsize=2, lw=0, zorder=2)
    if dots:
        ax.scatter(np.full(len(v), x), v, s=6, color=P["ink"], lw=0, zorder=3)


# ---------------------------------------------------------------- native behaviour

def behaviour(eval_glob):
    """Goal / proxy rates per seed from the evaluate_newenvs CSVs (det): each cohort on its training distribution
    and both on the out_marker conflict set."""
    rows = []
    for f in sorted(glob.glob(eval_glob)):
        b = os.path.basename(f)[:-4]
        coh = "gm" if "_gm_" in b else "control"
        d = pd.read_csv(f)
        d = d[d["mode"] == "det"]
        for s, g in d.groupby("set"):
            rows.append(dict(cohort=coh, run=b, seed=int(b.split("_s")[-1]), set=s, goal=g.success.mean(), proxy=g.proxy.mean()))
    df = pd.DataFrame(rows)
    keep = df.run.str.contains("m50_PushToMarker_gm") | df.run.isin(
        ["eval_m50_PushToMarker_control_s0", "eval_m100_PushToMarker_control_s1", "eval_m100_PushToMarker_control_s2"])
    df = df[keep]
    os.makedirs(SUMM, exist_ok=True)
    df.to_csv(os.path.join(SUMM, "behaviour_per_seed.csv"), index=False)
    panels = [("training distribution", {"gm": "in_marker", "control": "control"}),
              ("conflict (marker off-centre)", {"gm": "out_marker", "control": "out_marker"})]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6), sharey=True)
    w = 0.36
    for ax, (title, sets) in zip(axes, panels):
        for k, coh in enumerate(("gm", "control")):
            s = df[(df.cohort == coh) & (df.set == sets[coh])]
            bar(ax, k - w / 2, s.goal, HUE[coh][0], w)
            bar(ax, k + w / 2, s.proxy, HUE[coh][1], w)
        ax.set_xticks([0, 1], ["GM", "Non-GM"])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("fraction of episodes")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (P["gm"], P["gm_proxy"], P["nongm"], P["nongm_proxy"])]
    fig.legend(h, ["GM goal", "GM proxy", "Non-GM goal", "Non-GM proxy"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_behaviour_PTM")
    lines = ["## Native behaviour (evaluate_newenvs, 500 episodes det per run; mean [95% t-CI] over seeds)", "",
             "| cohort | set | goal | proxy | per-seed goal | per-seed proxy |", "|---|---|---|---|---|---|"]
    for (coh, st), g in df.groupby(["cohort", "set"]):
        g = g.sort_values("seed")
        lines.append(f"| {NAME[coh]} | {st} | {C.fmt_ci(*C.ci95(g.goal.to_numpy()))} | {C.fmt_ci(*C.ci95(g.proxy.to_numpy()))} | "
                     + " / ".join(f"{v:.3f}" for v in g.goal) + " | " + " / ".join(f"{v:.3f}" for v in g.proxy) + " |")
    open(os.path.join(SUMM, "behaviour.md"), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- world edits

EDITS = [("marker_move", "Marker (goal)"), ("marker_to_centre", "Marker to centre"), ("marker_hide", "Marker hidden"),
         ("cube_yaw", "Cube yaw (nuisance)")]
WE_COLS = [("KL", "KL"), ("Q_h_fc", "$Q$"), ("G_h_fc", "$G$"), ("D_F_h_fc", "$D_F$")]


def worldedit():
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.6))
    for r, (tag, sub, lab) in enumerate((("PTM", "all", "start"), ("PTMhover", "marker_visible", "hover"))):
        f = os.path.join(SUMM, f"worldedit_{tag}_per_seed.csv")
        if not os.path.exists(f):
            continue
        ps = pd.read_csv(f)
        ps = ps[ps.subset == sub]
        for c, (col, clab) in enumerate(WE_COLS):
            ax = axes[r, c]
            w = 0.26
            for e_i, (e, _) in enumerate(EDITS):
                for k, coh in enumerate(("gm", "control", "init")):
                    v = ps[(ps.edit == e) & (ps.cohort == coh)][col]
                    if len(v):
                        bar(ax, e_i + (k - 1) * w, v, HUE[coh][0], w, log=col != "G_h_fc")
            if col != "G_h_fc":
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 1e3)
            else:
                ax.axhline(1, color=P["mute"], lw=0.8, ls=":")
            ax.set_xticks(range(len(EDITS)), [e[1] for e in EDITS], rotation=30, ha="right", fontsize=7)
            if r == 0:
                ax.set_title(clab, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{lab} states")
    h = [plt.Rectangle((0, 0), 1, 1, color=HUE[c][0]) for c in ("gm", "control", "init")]
    fig.legend(h, ["GM", "Non-GM", "init"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_worldedit_PTM")


# ---------------------------------------------------------------- probes

SITES = ["conv1", "conv2", "conv3", "h_pre", "h_fc", "a_hid"]


def probes():
    f = os.path.join(C.RESULTS, "probes_PTM_per_seed.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    tg = [("marker_x", "marker x"), ("marker_y", "marker y"), ("push_dx", "marker - cube, x"), ("push_dy", "marker - cube, y"),
          ("cube_x", "cube x"), ("cube_y", "cube y")]
    fig, axes = plt.subplots(1, len(tg), figsize=(13, 2.6), sharey=True)
    x = np.arange(len(SITES))
    for ax, (t, lab) in zip(axes, tg):
        d = df[df.target == t]
        for coh in ("gm", "control"):
            ys = [d[(d.cohort == coh) & (d.site == s)].value.to_numpy() for s in SITES]
            m = np.array([ci(y)[0] for y in ys])
            lo = np.array([ci(y)[1] for y in ys])
            hi = np.array([ci(y)[2] for y in ys])
            ax.plot(x, m, "-o", color=HUE[coh][0], ms=3.5, lw=1.4, label=NAME[coh])
            ax.fill_between(x, m - lo, m + hi, color=HUE[coh][0], alpha=0.2, lw=0)
        ini = [d[(d.cohort == "init") & (d.site == s)].value.mean() for s in SITES]
        ax.plot(x, ini, "--", color=P["init"], lw=1.2, label="init")
        for coh, ls in (("pixels", ":"), ("proprio", (0, (1, 3)))):
            v = d[d.cohort == coh].value
            if len(v):
                ax.axhline(v.mean(), color=P["null"], ls=ls, lw=1, label=coh)
        ax.set_xticks(x, ["c1", "c2", "c3", "pre", "fc", "act"], fontsize=7)
        ax.set_title(lab, fontsize=9)
        ax.set_ylim(-0.1, 1.0)
    axes[0].set_ylabel("held-out $R^2$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_probes_PTM")


# ---------------------------------------------------------------- steering

def steering():
    f = os.path.join(SUMM, "steer_per_seed.csv")
    if not os.path.exists(f):
        return
    ps = pd.read_csv(f)
    doses = sorted(ps[ps.kind == "corrective"].dose.unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.6), sharex=True, sharey=True)
    for c, coh in enumerate(("gm", "control")):
        base = ps[(ps.cohort == coh) & (ps.kind == "baseline")]
        for r, (col, lab) in enumerate((("goal", "goal reach"), ("proxy", "cube at centre"))):
            ax = axes[r, c]
            m, lo, hi = ci(base[col])
            ax.axhspan(m - lo, m + hi, color=P["init"], alpha=0.2, lw=0)
            ax.axhline(m, color=P["mute"], ls=":", lw=1)
            for kind, ls, mk, fill in (("corrective", "-", "o", True), ("anticorrective", "--", "o", False),
                                       ("random", ":", "s", True)):
                ys = [ps[(ps.cohort == coh) & (ps.kind == kind) & (ps.dose == b)][col].to_numpy() for b in doses]
                mm = np.array([ci(y)[0] for y in ys])
                l_ = np.array([ci(y)[1] for y in ys])
                h_ = np.array([ci(y)[2] for y in ys])
                col_ = HUE[coh][0] if kind != "random" else P["null"]
                ax.plot(doses, mm, ls=ls, marker=mk, color=col_, mfc=col_ if fill else "white", ms=4, lw=1.3, label=kind)
                ax.fill_between(doses, mm - l_, mm + h_, color=col_, alpha=0.15, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.02, 1.02)
            if r == 0:
                ax.set_title(NAME[coh], fontsize=10)
            if c == 0:
                ax.set_ylabel(lab)
            if r == 1:
                ax.set_xlabel(r"dose $\beta$")
    h = [Line2D([], [], color=P["ink"], ls="-", marker="o", label="corrective"),
         Line2D([], [], color=P["ink"], ls="--", marker="o", mfc="white", label="reversed"),
         Line2D([], [], color=P["null"], ls=":", marker="s", label="random")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_steer_PTM")


# ---------------------------------------------------------------- rewiring

ARMS = [("native", "native"), ("fc_head", r"head on $h_{fc}$"), ("pre", r"from $h_{pre}$"), ("remix", "random remix"),
        ("untrained", "untrained"), ("pre_masked", r"$h_{pre}$, marker masked")]


def rewiring():
    f = os.path.join(SUMM, "rewire_per_seed.csv")
    if not os.path.exists(f):
        return
    ps = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    for ax, sname, title in ((axes[0], "conflict", "conflict (marker on the push ray)"),
                             (axes[1], "ctrlset", "control distribution")):
        d = ps[(ps.set == sname) & (ps.cond == "standard")]
        w = 0.13
        for a_i, (arm, _) in enumerate(ARMS):
            for k, coh in enumerate(("gm", "control", "init")):
                s = d[(d.arm == arm) & (d.cohort == coh)]
                if len(s):
                    x0 = a_i + (k - 1) * 0.3
                    bar(ax, x0 - w / 2, s.goal, HUE[coh][0], w)
                    bar(ax, x0 + w / 2, s.proxy, HUE[coh][1], w)
        ph = d[(d.arm == "pre") & (d.cohort == "gm")].phantom_pool
        if len(ph):
            ax.axhline(ph.mean(), color=P["null"], lw=0.8, ls=":")
        ax.set_xticks(range(len(ARMS)), [a[1] for a in ARMS], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("fraction of episodes")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (P["gm"], P["gm_proxy"], P["nongm"], P["nongm_proxy"], P["init"], "#C8C8C8")]
    fig.legend(h, ["GM goal", "GM proxy", "Non-GM goal", "Non-GM proxy", "init goal", "init proxy"], loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.tight_layout()
    save(fig, "fig_rewire_PTM")
    # chance checks for the h_pre rewire
    conds = [("conflict", "standard", "goal", "conflict"), ("conflict", "nomarker", "goal", "marker hidden"),
             ("conflict", "decoupled", "goal", "drawn elsewhere:\nscored marker"),
             ("conflict", "decoupled", "reach_m2", "drawn elsewhere:\ndrawn marker"),
             ("ctrlset", "standard", "goal", "control dist."), ("ctrlset", "nomarker", "goal", "control dist.,\nhidden"),
             ("ctrlset", "standard", "phantom_pool", "control dist.,\nphantom")]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    w = 0.2
    for i, (sn, cond, col, _) in enumerate(conds):
        for k, (coh, arm) in enumerate((("gm", "pre"), ("control", "pre"), ("gm", "pre_masked"), ("init", "pre"))):
            s = ps[(ps.set == sn) & (ps.cond == cond) & (ps.cohort == coh) & (ps.arm == arm)]
            if len(s) and col in s:
                c = HUE[coh][0] if arm == "pre" else HUE[coh][1]
                bar(ax, i + (k - 1.5) * w, s[col], c, w)
    ax.set_xticks(range(len(conds)), [c[3] for c in conds], fontsize=7)
    ax.set_ylabel("fraction of episodes")
    ax.set_ylim(0, 1.05)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (P["gm"], P["nongm"], P["gm_proxy"], P["init"])]
    fig.legend(h, [r"GM $h_{pre}$", r"Non-GM $h_{pre}$", r"GM $h_{pre}$, marker masked", r"init $h_{pre}$"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_rewire_checks_PTM")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["behaviour", "worldedit", "probes", "steering", "rewiring"])
    p.add_argument("--eval_glob", default=os.path.join(C.ROOT, "results", "ptm50", "eval_m*_PushToMarker_*.csv"))
    a = p.parse_args()
    for w in a.which:
        if w == "behaviour":
            behaviour(a.eval_glob)
        else:
            globals()[w]()


if __name__ == "__main__":
    main()
