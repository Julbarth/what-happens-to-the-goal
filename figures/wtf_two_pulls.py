"""WalkToFlag two-pulls figure: per-state goal and proxy pull vectors in command space, their joint sign and distance strata."""
from paths import OUT_DIR, TABLES
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import style as fs  # noqa: E402
from style import PAL, REF  # noqa: E402

DATA = TABLES
FIG = OUT_DIR / "app_role"
COH = (("gm", "GM"), ("control", "Non-GM"), ("init", "init"))
NULL_Q, TAU_MIN = 0.95, 0.1


def ci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def stratum(d):
    """Distance strata: within the visible bank by the sign of d_flag - d_tower, and the two far-separated banks."""
    dd = d.d_flag - d.d_tower
    return np.select([(d.bank == "visible") & (dd < -0.3), (d.bank == "visible") & (dd > 0.3), d.bank == "conflict",
                      d.bank == "far"], ["vis_flag_closer", "vis_tower_closer", "conflict_flag_closer",
                                         "far_tower_closer"], "vis_equidistant")


def per_seed(v):
    """tau = 95th percentile of the |nuisance shift| projections (box move toward the flag, re-render toward the tower),
    floored at 0.1 sigma so near-constant random networks do not pass on numerical noise."""
    rows = []
    for (c, s), x in v.groupby(["cohort", "seed"]):
        tau = max(np.quantile(np.abs(np.r_[x.follow_box_to_flag, x.follow_rr_to_tower]), NULL_Q), TAU_MIN)
        for st, y in [("all", x)] + list(x.groupby("stratum")):
            F, T = y.follow_flag_to_flag, y.follow_tower_to_tower
            rows.append(dict(cohort=c, seed=s, stratum=st, n=len(y), tau=tau, f_goal=F.mean(), f_proxy=T.mean(),
                             goal_toward=(F > tau).mean(), goal_away=(F < -tau).mean(),
                             proxy_toward=(T > tau).mean(), proxy_away=(T < -tau).mean(),
                             both_toward=((F > tau) & (T > tau)).mean(),
                             null_both=((y.follow_box_to_flag > tau) & (y.follow_rr_to_tower > tau)).mean(),
                             goal_on_proxy=y.follow_flag_to_tower.mean(), proxy_on_goal=y.follow_tower_to_flag.mean()))
    return pd.DataFrame(rows)


def summarise(ps):
    out = []
    for (c, st), g in ps.groupby(["cohort", "stratum"]):
        r = dict(cohort=c, stratum=st, n_per_seed=int(g.n.mean()))
        for k in ("f_goal", "f_proxy", "goal_toward", "goal_away", "proxy_toward", "proxy_away", "both_toward", "null_both",
                  "goal_on_proxy", "proxy_on_goal"):
            r[k], r[k + "_lo"], r[k + "_hi"] = ci(g[k])
            r[k + "_seeds"] = ";".join(f"{x:.3f}" for x in g[k])
        out.append(r)
    return pd.DataFrame(out)


def figure(v, ps):
    fs.style()
    fig = plt.figure(figsize=(fs.W_IN, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.2], wspace=0.45)
    ax, bx, cx = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) per-state pull vectors projected toward each object's own command.
    lim, ylim = (-3, 8), (-3, 13)
    for c, name in COH[:2]:
        x = v[v.cohort == c]
        ax.scatter(x.follow_flag_to_flag.clip(*lim), x.follow_tower_to_tower.clip(*ylim), color=fs.COH[name], **fs.STATE)
    for c, name in COH:
        g = ps[(ps.cohort == c) & (ps.stratum == "all")]
        col = fs.COH[name]
        mx, my = ci(g.f_goal), ci(g.f_proxy)
        ax.errorbar(mx[0], my[0], xerr=[[mx[0] - mx[1]], [mx[2] - mx[0]]], yerr=[[my[0] - my[1]], [my[2] - my[0]]],
                    fmt="o", ms=fs.MS + 0.6, color=col, mec="white", mew=0.4, elinewidth=fs.LW_ERR, zorder=5)
        ax.scatter(g.f_goal, g.f_proxy, s=9, color=col, marker="D", lw=0, alpha=0.7, zorder=4)
    ax.axhline(0, **REF)
    ax.axvline(0, **REF)
    ax.set_xlim(*lim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"goal pull $f_\star$ ($\sigma$)")
    ax.set_ylabel(r"proxy pull $f_p$ ($\sigma$)")
    fs.finish(ax)

    # (b) fraction of states whose pull points toward its own object beyond the nuisance null.
    keys = (("goal_toward", "goal toward", True), ("goal_away", "goal away", True),
            ("proxy_toward", "proxy toward", False), ("both_toward", "both toward", True), ("null_both", "null", None))
    for i, (k, lab, filled) in enumerate(keys):
        for j, (c, name) in enumerate(COH[:2]):
            g = ps[(ps.cohort == c) & (ps.stratum == "all")][k].to_numpy()
            m, lo, hi = ci(g)
            x = i + (j - 0.5) * 0.4
            col = PAL["null"] if filled is None else fs.COH[name]
            soft = PAL["gm_proxy"] if name == "GM" else PAL["nongm_proxy"]
            bx.bar(x, m, width=0.38, color=soft if filled is False else col, lw=0)
            bx.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=PAL["ink"], elinewidth=fs.LW_ERR)
            bx.scatter(np.full(len(g), x), g, s=4, color=PAL["ink"], alpha=0.5, lw=0, zorder=4)
    bx.set_xticks(range(len(keys)))
    bx.set_xticklabels([l for _, l, _ in keys], fontsize=fs.FS_TICK, rotation=35, ha="right", rotation_mode="anchor")
    bx.set_ylim(0, 1)
    bx.set_ylabel("fraction of states")
    fs.finish(bx)

    # (c) goal pull by distance stratum (GM and Non-GM), proxy pull for GM.
    strata = (("vis_flag_closer", "near, $d_\\star{<}d_p$"), ("vis_tower_closer", "near, $d_\\star{>}d_p$"),
              ("conflict_flag_closer", "$d_\\star{\\ll}d_p$"), ("far_tower_closer", "$d_\\star{\\gg}d_p$"))
    for j, (c, name) in enumerate(COH[:2]):
        for k, filled in (("goal_toward", True), ("proxy_toward", False)):
            vals = [ci(ps[(ps.cohort == c) & (ps.stratum == s)][k]) for s, _ in strata]
            xs = np.arange(len(strata)) + (j - 0.5) * 0.12 + (0.03 if not filled else -0.03)
            m = np.array([q[0] for q in vals])
            col = fs.COH[name]
            cx.errorbar(xs, m, yerr=[m - [q[1] for q in vals], [q[2] for q in vals] - m], fmt="none", ecolor=col,
                        elinewidth=fs.LW_ERR)
            cx.plot(xs, m, ls="-" if filled else (0, (2, 1.2)), color=col, lw=fs.LW_DATA * 0.8, marker="o", ms=fs.MS,
                    mfc=col if filled else "white", mec=col, mew=0.9)
    cx.set_xticks(range(len(strata)))
    cx.set_xticklabels([l for _, l in strata], fontsize=fs.FS_TICK, rotation=35, ha="right", rotation_mode="anchor")
    cx.set_ylim(0, 1)
    cx.set_ylabel("fraction toward own object")
    fs.finish(cx)

    for a, k, t in ((ax, "a", "pull vectors"), (bx, "b", "joint sign"), (cx, "c", "by distance")):
        a.set_title(t, fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)
        a.annotate(f"({k})", (0, 1), xycoords="axes fraction", xytext=(fs.LETTER_DX, fs.TITLE_PAD),
                   textcoords="offset points", fontsize=fs.FS_LETTER, fontweight="bold", va="bottom")
    h = [Line2D([], [], color=fs.COH[n], marker="o", ls="", ms=fs.MS, label=n) for n in ("GM", "Non-GM", "init")]
    h += [Line2D([], [], color=PAL["ink"], marker="o", mfc=PAL["ink"], ls="-", ms=fs.MS, lw=0.8, label="goal"),
          Line2D([], [], color=PAL["ink"], marker="o", mfc="white", ls=(0, (2, 1.2)), ms=fs.MS, lw=0.8, label="proxy"),
          Patch(color=PAL["null"], label="null")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=fs.FS_LEG, bbox_to_anchor=(0.5, -0.2),
               handlelength=1.5, columnspacing=1.2)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_app_role_wtf_two_pulls.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    argparse.ArgumentParser().parse_args()
    files = sorted(glob.glob(str(DATA / "pull_vectors_*_s*.csv")))
    v = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    v["stratum"] = stratum(v)
    ps = per_seed(v)
    ps.to_csv(DATA / "two_pulls_per_seed.csv", index=False)
    sm = summarise(ps)
    sm.to_csv(DATA / "two_pulls_summary.csv", index=False)
    figure(v, ps)
    (TABLES / "fig_app_role_wtf_two_pulls_values.json").write_text(json.dumps(
        {"sources": files, "summary": sm.to_dict("records")}, indent=1, default=str))


if __name__ == "__main__":
    main()
