"""Isaac quadruped figures: push-pull, nuisance vs tower, seeking vs perceiving, steering doses, rewiring."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gm_robotics.isaac.analysis import common_isaac as C

P = C.PAL
R = C.RESULTS
OUT = C.FIGURES
COHS = (("gm", P["gm"], "GM"), ("control", P["nongm"], "Non-GM"), ("init", P["init"], "init"))
GOAL_GREEN, PROXY_ORANGE = "#2E9E44", "#E08A1E"
plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.edgecolor": P["ink"], "axes.labelcolor": P["ink"], "xtick.color": P["ink"],
                     "ytick.color": P["ink"], "text.color": P["ink"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def ci(v):
    m, lo, hi = C.ci95(np.asarray(v, float))
    return m, (m - lo if np.isfinite(lo) else 0.0), (hi - m if np.isfinite(hi) else 0.0)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def label(ax, k, title):
    ax.set_title(title, fontsize=10)
    ax.text(-0.16, 1.06, f"({chr(97 + k)})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def push_pull(form="", name="fig_push_pull"):
    st = pd.read_csv(os.path.join(R, "push_pull", "isaac_per_state.csv"))
    ps = pd.read_csv(os.path.join(R, "push_pull", "isaac_per_seed.csv"))
    ps = ps[ps.subset == "all"]
    fx, fy = f"S_fav{form}", f"S_comp{form}"
    panels = [(b, e) for b in ("visible", "conflict") for e in ("flag_swap", "tower_swap")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    rng = np.random.default_rng(0)
    for k, (bk, ed) in enumerate(panels):
        ax = axes[k]
        sub_all = st[(st.bank == bk) & (st.edit == ed) & (st.cohort != "init")]
        lim = np.nanquantile(np.abs(sub_all[[fx, fy]].to_numpy()), 0.95) * 1.15
        ax.add_patch(plt.Rectangle((0, -lim), lim, lim, color="#F2F2F2", zorder=0))
        ax.axhline(0, color=P["mute"], ls="--", lw=0.8), ax.axvline(0, color=P["mute"], ls="--", lw=0.8)
        for coh, col, _ in COHS:
            sub = st[(st.bank == bk) & (st.edit == ed) & (st.cohort == coh)]
            if len(sub) > 700:
                sub = sub.iloc[rng.choice(len(sub), 700, replace=False)]
            ax.scatter(sub[fx].clip(-lim, lim), sub[fy].clip(-lim, lim), s=4, color=col, alpha=0.15, lw=0)
            sd = ps[(ps.bank == bk) & (ps.edit == ed) & (ps.cohort == coh)]
            ax.scatter(sd[fx], sd[fy], marker="D", s=36, color=col, ec="white", lw=0.8, zorder=4)
            mf, lf, hf = ci(sd[fx])
            mc, lc, hc = ci(sd[fy])
            ax.errorbar(mf, mc, xerr=[[lf], [hf]], yerr=[[lc], [hc]], fmt="o", ms=5, color=col, mfc="white", mew=1.4,
                        lw=1.4, zorder=5)
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$S_{\mathrm{fav}}$")
        if k == 0:
            ax.set_ylabel(r"$S_{\mathrm{comp}}$")
        et = "Flag swap (favours tower spot)" if ed == "flag_swap" else "Tower swap (favours flag spot)"
        label(ax, k, f"{et}\n{bk} bank")
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for _, c, l in COHS]
    h += [Line2D([], [], marker="D", ls="", color=P["mute"], label="seed mean"),
          Line2D([], [], marker="o", color=P["mute"], mfc="white", label="mean ± 95% CI"),
          Patch(color="#F2F2F2", label="push-pull quadrant")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    save(fig, name)


def push_pull_sensitivity():
    sm = pd.read_csv(os.path.join(R, "push_pull", "isaac_summary.csv"))
    sm = sm[(sm.subset == "all") & (sm.bank == "visible")]
    forms = [("", "oracle cmd"), ("_cone15", "cone 15°"), ("_cone30", "cone 30°"), ("_cone45", "cone 45°")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for k, ed in enumerate(("flag_swap", "tower_swap")):
        ax = axes[k]
        for j, (coh, col, lab) in enumerate(COHS):
            r = sm[(sm.edit == ed) & (sm.cohort == coh)]
            if not len(r):
                continue
            r = r.iloc[0]
            x = np.arange(len(forms)) + (j - 1) * 0.25
            y = [r[f"dl{f}"] for f, _ in forms]
            e = [[r[f"dl{f}"] - r[f"dl{f}_lo"] for f, _ in forms], [r[f"dl{f}_hi"] - r[f"dl{f}"] for f, _ in forms]]
            ax.errorbar(x, y, yerr=e, fmt="o", color=col, capsize=0, lw=1.4, label=lab)
        ax.axhline(0, color=P["mute"], lw=0.8, ls="--")
        ax.set_xticks(range(len(forms)), [f for _, f in forms])
        label(ax, k, ("Flag swap" if ed == "flag_swap" else "Tower swap") + r": $\Delta\ell = S_{fav}-S_{comp}$")
    axes[0].set_ylabel(r"$\Delta\ell$ (visible bank)")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_push_pull_sensitivity")


def nuisance():
    f = os.path.join(R, "nuisance", "nuisance_per_seed.csv")
    if not os.path.exists(f):
        return
    ps = pd.read_csv(f)
    edits = [("tower_swap", "tower swap\n(orange)"), ("grey_tower_swap", "grey tower\nswap"), ("decoy_move", "grey decoy\nmoved"),
             ("decoy_insert", "grey decoy\nadded"), ("recolour_tower", "tower\nrecoloured")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for k, (col_, ylab) in enumerate((("KL_excess", "KL − re-render KL"), ("S_to", r"$S$ toward moved object's new spot"))):
        ax = axes[k]
        for j, (coh, col, lab) in enumerate(COHS[:2]):
            for i, (e, _) in enumerate(edits):
                v = ps[(ps.kind == "visible") & (ps.edit == e) & (ps.cohort == coh)][col_].dropna()
                if not len(v):
                    continue
                m, lo, hi = ci(v)
                x = i + (j - 0.5) * 0.3
                ax.bar(x, m, 0.28, color=col, alpha=0.85)
                ax.errorbar(x, m, yerr=[[lo], [hi]], color=P["ink"], lw=1.1, capsize=0)
                ax.scatter(np.full(len(v), x), v, s=9, color="white", ec=P["ink"], lw=0.6, zorder=4)
        ax.axhline(0, color=P["mute"], lw=0.8)
        ax.set_xticks(range(len(edits)), [l for _, l in edits], fontsize=8)
        if k == 0:
            ax.set_yscale("symlog", linthresh=0.5)
        ax.set_ylabel(ylab)
        label(ax, k, "Policy change (visible bank)" if k == 0 else "Directed response")
    axes[1].legend(handles=[Patch(color=c, label=l) for _, c, l in COHS[:2]], loc="upper right")
    fig.tight_layout()
    save(fig, "fig_nuisance_vs_tower")


def seeking():
    d = os.path.join(R, "trace_summary")
    if not os.path.exists(os.path.join(d, "trace_per_seed.csv")):
        return
    import glob
    from gm_robotics.isaac.analysis.trace_analyze_isaac import canon
    ps = pd.read_csv(os.path.join(d, "trace_per_seed.csv"))
    tb = pd.read_csv(os.path.join(d, "follow_per_seed_tbin.csv"))
    fig = plt.figure(figsize=(14, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.1])
    # (a, b) example paths in the canonical frame, native bank, seed 0 of each cohort
    for k, coh in enumerate(("control", "gm")):
        ax = fig.add_subplot(gs[0, k])
        f = os.path.join(R, "trace", f"conflict_main_{coh}_s0.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            for i in range(min(40, len(z["start"]))):
                s, u, v = z["start"][i], z["lay_base_flag"][i], z["lay_base_tower"][i]
                L = np.linalg.norm(u - s)
                for acting, ls, col in (("no_tower", "-", P["mute"]), ("base", "-", P["gm"] if coh == "gm" else P["nongm"])):
                    q = canon(z[f"{acting}_xy"][i], s, u, v) / L
                    ax.plot(q[:, 0], q[:, 1], ls, color=col, lw=0.7, alpha=0.5)
                vv = canon(v[None], s, u, v)[0] / L
                ax.scatter(*vv, marker="s", s=10, color=PROXY_ORANGE, zorder=3, alpha=0.6)
            ax.scatter([1], [0], marker="^", s=50, color=GOAL_GREEN, zorder=4)
            ax.scatter([0], [0], marker="o", s=30, color=P["ink"], zorder=4)
            ax.set_aspect("equal"), ax.set_xlim(-0.5, 2.0), ax.set_ylim(-1.0, 2.0)
            ax.set_xlabel("along start→flag (flag distance = 1)")
            if k == 0:
                ax.set_ylabel("toward tower side")
        label(ax, k, f"{'Non-GM' if coh == 'control' else 'GM'} s0: with tower (colour) vs without (grey)")
    # (c) paired path changes
    ax = fig.add_subplot(gs[0, 2])
    mets = [("diff_dev_max", "max lateral\ndev. (m)"), ("diff_min_d_tower", "min dist\nto tower (m)"),
            ("diff_steps_near_tower", "steps within\n1 m / 10")]
    for j, (coh, col, lab) in enumerate(COHS[:2]):
        for i, (c, _) in enumerate(mets):
            v = ps[(ps.bank == "native") & (ps.cohort == coh)][c] / (10 if "steps" in c else 1)
            m, lo, hi = ci(v)
            x = i + (j - 0.5) * 0.3
            ax.bar(x, m, 0.28, color=col, alpha=0.85)
            ax.errorbar(x, m, yerr=[[lo], [hi]], color=P["ink"], lw=1.1)
            ax.scatter(np.full(len(v), x), v, s=9, color="white", ec=P["ink"], lw=0.6, zorder=4)
    ax.axhline(0, color=P["mute"], lw=0.8)
    ax.set_xticks(range(len(mets)), [l for _, l in mets], fontsize=8)
    ax.set_ylabel("with tower − without (paired)")
    label(ax, 2, "Path change caused by the tower (native eval)")
    # (d) along-trajectory pulls by time bin
    ax = fig.add_subplot(gs[0, 3])
    bins = list(tb.tbin.unique())
    for j, (coh, col, lab) in enumerate(COHS[:2]):
        for q, (c, mk) in enumerate((("tower_pull", "s"), ("flag_pull", "^"))):
            ys = [ci(tb[(tb.bank == "native") & (tb.cohort == coh) & (tb.tbin == b)][c]) for b in bins]
            x = np.arange(len(bins)) + (j - 0.5) * 0.2 + (q - 0.5) * 0.08
            ax.errorbar(x, [y[0] for y in ys], yerr=[[y[1] for y in ys], [y[2] for y in ys]], fmt=mk + ("-" if q == 0 else "--"),
                        color=col, ms=5, lw=1.2, mfc=col if q == 0 else "white")
    ax.axhline(0, color=P["mute"], lw=0.8, ls="--")
    ax.set_xticks(range(len(bins)), bins)
    ax.set_ylabel("pull on heading (log support)")
    label(ax, 3, "Object pull along the trajectory")
    h = [Patch(color=c, label=l) for _, c, l in COHS[:2]] + [Line2D([], [], marker="s", color=P["mute"], label="tower pull"),
                                                           Line2D([], [], marker="^", ls="--", mfc="white", color=P["mute"],
                                                                  label="flag pull")]
    ax.legend(handles=h, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_seeking_vs_perceiving")


def steering():
    f = os.path.join(R, "steer", "steer_per_seed.csv")
    if not os.path.exists(f):
        return
    ps = pd.read_csv(f)
    banks = [("conflict", "conflict bank"), ("native", "native eval")]
    fams = [("local", "random", "state-local $d_{comp}$"), ("flag", "random_flag", "flag-only $h_{base}-h_{flag\\,swap}$"),
            ("pre", "random_pre", "$d_{comp}$ at $h_{pre}$")]
    cohs = [c for c in ("gm", "control") if c in ps.cohort.unique()]
    fig, axes = plt.subplots(len(cohs), 4, figsize=(15, 3.2 * len(cohs)), squeeze=False)
    for r, coh in enumerate(cohs):
        col = P["gm"] if coh == "gm" else P["nongm"]
        for k, (bk, btxt) in enumerate(banks):
            ax = axes[r, k]
            sub = ps[(ps.cohort == coh) & (ps.bank == bk) & (ps.plan != "phantom")]
            base = sub[sub.arm == "baseline"].goal
            if len(base):
                m, lo, hi = ci(base)
                ax.axhspan(m - lo, m + hi, color="#E6E6E6", zorder=0)
                ax.axhline(m, color=P["mute"], ls=":", lw=1)
            for (arm, rnd, txt), ls in zip(fams, ("-", "--", ":")):
                for sgn, mfc, tag in ((1, col, "corrective"), (-1, "white", "reversed")):
                    a = sub[(sub.arm == arm) & (np.sign(sub.beta) == sgn)]
                    if not len(a):
                        continue
                    g = a.groupby("beta").goal.apply(list)
                    xs = np.abs(g.index.to_numpy())
                    ys = [ci(v) for v in g]
                    ax.errorbar(xs, [y[0] for y in ys], yerr=[[y[1] for y in ys], [y[2] for y in ys]], fmt="o" + ls,
                                color=col, mfc=mfc, ms=4, lw=1.2)
                a = sub[sub.arm == rnd]
                if len(a):
                    g = a.groupby("beta").goal.apply(list)
                    ys = [ci(v) for v in g]
                    ax.errorbar(g.index.to_numpy(), [y[0] for y in ys], yerr=[[y[1] for y in ys], [y[2] for y in ys]],
                                fmt="x" + ls, color=P["null"], ms=5, lw=1.0)
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel(r"dose $\beta$")
            if k == 0:
                ax.set_ylabel(f"{'GM' if coh == 'gm' else 'Non-GM'}\nflag reached first")
            label(ax, r * 4 + k, f"State-local pushes, {btxt}")
        ax = axes[r, 2]
        sub = ps[(ps.cohort == coh) & (ps.bank == "conflict")]
        for arm, mk in (("fixed_local", "o"), ("sign_gated", "s"), ("fisher_proj", "D"), ("unsigned", "^")):
            a = sub[sub.arm == arm]
            if not len(a):
                continue
            g = a.groupby("beta").goal.apply(list)
            ys = [ci(v) for v in g]
            ax.errorbar(g.index.to_numpy(), [y[0] for y in ys], yerr=[[y[1] for y in ys], [y[2] for y in ys]], fmt=mk + "-",
                        ms=4, lw=1.1, label=arm.replace("_", " "))
        base = sub[sub.arm == "baseline"].goal
        if len(base):
            m, lo, hi = ci(base)
            ax.axhspan(m - lo, m + hi, color="#E6E6E6", zorder=0)
        ax.axvline(0, color=P["mute"], lw=0.6)
        ax.set_ylim(-0.03, 1.03), ax.set_xlabel(r"dose $\beta$ (negative = reversed)")
        if len(ax.get_legend_handles_labels()[0]):
            ax.legend(fontsize=7, loc="upper left")
        label(ax, r * 4 + 2, "Shared axis (held at $s_0$), conflict bank")
        ax = axes[r, 3]
        bk = "conflict" if r == 0 else "native"
        sub = ps[(ps.cohort == "gm") & (ps.bank == bk)]
        for arm, ls in (("local", "-"), ("flag", "--")):
            for plan, val, col, mk, mfc, lab in (("full", "goal", P["gm"], "o", P["gm"], "standard: flag spot"),
                                                 ("phantom", "goal", P["mute"], "o", "white", "flag removed: flag spot"),
                                                 ("decoupled", "goal", GOAL_GREEN, "s", "white", "flag moved to tower: old spot"),
                                                 ("decoupled", "goal_object", GOAL_GREEN, "^", GOAL_GREEN, "flag moved to tower: flag")):
                a_ = sub[(sub.plan == plan) & (sub.arm == arm) & (sub.beta > 0)]
                if not len(a_) or val not in a_:
                    continue
                g = a_.groupby("beta")[val].apply(list)
                ys = [ci(v) for v in g]
                ax.errorbar(g.index.to_numpy(), [y[0] for y in ys], yerr=[[y[1] for y in ys], [y[2] for y in ys]],
                            fmt=mk + ls, color=col, mfc=mfc, ms=4, lw=1.1, label=lab if arm == "local" else None)
        ax.set_xscale("log", base=2), ax.set_ylim(-0.03, 1.03), ax.set_xlabel(r"dose $\beta$ (solid $d_{comp}$, dashed flag-only)")
        if len(ax.get_legend_handles_labels()[0]):
            ax.legend(fontsize=6, loc="center right")
        label(ax, r * 4 + 3, f"GM chance controls, {bk}")
        if coh == "control":
            axes[r, 2].text(0.5, 0.5, "shared-axis arms run for GM only", transform=axes[r, 2].transAxes, ha="center",
                            color=P["mute"])
    h = [Line2D([], [], marker="o", ls="-", color=P["ink"], label="$d_{comp}$ at $h_{fc}$"),
         Line2D([], [], marker="o", ls="--", color=P["ink"], label="flag-only at $h_{fc}$"),
         Line2D([], [], marker="o", ls=":", color=P["ink"], label="$d_{comp}$ at $h_{pre}$"),
         Line2D([], [], marker="o", ls="", color=P["ink"], mfc="white", label="reversed"),
         Line2D([], [], marker="x", ls="", color=P["null"], label="norm-matched random"),
         Patch(color="#E6E6E6", label="unsteered (95% CI)")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout()
    save(fig, "fig_steering_dose")


def rewiring():
    f = os.path.join(R, "rewire", "rewire_per_seed.csv")
    if not os.path.exists(f):
        return
    ps = pd.read_csv(f)
    arms = [("native", "none", "native"), ("untrained", "none", "untrained\nhead"), ("fc_head", "oracle", "h_fc head\noracle"),
            ("pre", "oracle", "h_pre\noracle"), ("remix", "oracle", "remix\noracle"), ("fc_head", "teacher", "h_fc head\nteacher"),
            ("pre", "teacher", "h_pre\nteacher"), ("remix", "teacher", "remix\nteacher")]
    sets = [("conflict", "conflict bank"), ("native", "native eval"), ("conflict_noflag", "conflict bank, flag removed")]
    fig, axes = plt.subplots(1, len(sets), figsize=(15, 3.4), sharey=True)
    for k, (s, stxt) in enumerate(sets):
        ax = axes[k]
        for j, (coh, col, lab) in enumerate(COHS[:2]):
            for i, (arm, tgt, _) in enumerate(arms):
                v = ps[(ps.set == s) & (ps.cohort == coh) & (ps.arm == arm) & (ps.target == tgt)]["goal"]
                if not len(v):
                    continue
                m, lo, hi = ci(v)
                x = i + (j - 0.5) * 0.32
                ax.errorbar(x, m, yerr=[[lo], [hi]], fmt="o", color=col, ms=6, lw=1.5)
                ax.scatter(np.full(len(v), x + 0.08), v, s=8, color=col, alpha=0.5, lw=0)
                ph = ps[(ps.set == s) & (ps.cohort == coh) & (ps.arm == arm) & (ps.target == tgt)]["phantom"]
                ax.scatter([x], [ph.mean()], marker="_", s=60, color=P["mute"], zorder=5)
        ax.set_xticks(range(len(arms)), [l for _, _, l in arms], fontsize=7)
        ax.set_ylim(-0.03, 1.03)
        if k == 0:
            ax.set_ylabel("flag spot reached first")
        label(ax, k, stxt)
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for _, c, l in COHS[:2]] + \
        [Line2D([], [], marker="_", ls="", color=P["mute"], ms=10, label="phantom spot reached first (mean)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    save(fig, "fig_rewiring")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=["push_pull", "nuisance", "seeking", "steering", "rewiring"])
    a = p.parse_args()
    if "push_pull" in a.only:
        push_pull()
        push_pull("_cone30", "fig_push_pull_cone30")
        push_pull_sensitivity()
    if "nuisance" in a.only:
        nuisance()
    if "seeking" in a.only:
        seeking()
    if "steering" in a.only:
        steering()
    if "rewiring" in a.only:
        rewiring()


if __name__ == "__main__":
    main()
