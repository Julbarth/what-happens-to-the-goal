"""WalkToFlag object-competition figure: one panel per claim of the object-competition subsection, plainly labelled axes."""
from __future__ import annotations

from paths import TABLES
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from wtf_base import COH, DD_E, MIN_STATES, PAL, REF, ci, fs, grid, load, save, share, titled  # noqa: E402

STRATA = (("vis_flag_closer", "near,\nflag\ncloser"), ("vis_tower_closer", "near,\ntower\ncloser"),
          ("conflict_flag_closer", "flag\nmuch\ncloser"), ("far_tower_closer", "tower\nmuch\ncloser"))
ALL3 = COH + (("init", "init"),)


def pulls(ax, ps):
    """Claim 1: mean pull of each object along the command that walks to it, per cohort (seed dots, 95% t-CI)."""
    out = {}
    for i, (c, name) in enumerate(ALL3):
        for k, (key, filled) in enumerate((("f_goal", True), ("f_proxy", False))):
            g = ps[(ps.cohort == c) & (ps.stratum == "all")][key].to_numpy()
            m, lo, hi = ci(g)
            out[f"{name}_{key}"] = [m, lo, hi]
            x = i + (k - 0.5) * 0.36
            col = fs.COH[name]
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=col, elinewidth=fs.LW_ERR, zorder=3)
            ax.plot(x, m, "o", ms=fs.MS + 1.2, mfc=col if filled else "white", mec=col, mew=1.0, zorder=4)
            ax.scatter(np.full(len(g), x + 0.09), g, s=5, color=col, alpha=0.5, lw=0, zorder=4)
    ax.axhline(0, **REF)
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for _, n in ALL3])
    for t, (_, n) in zip(ax.get_xticklabels(), ALL3):
        t.set_color(fs.COH[n])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel(r"pull toward the object, $f$ (policy s.d.)")
    fs.finish(ax)
    return out


def by_stratum(ax, ps):
    """Claim 2: GM fraction of states pulled toward the flag and toward the tower, by distance stratum."""
    out = {}
    for key, filled, lab in (("goal_toward", True, "flag"), ("proxy_toward", False, "tower")):
        v = [ci(ps[(ps.cohort == "gm") & (ps.stratum == s)][key]) for s, _ in STRATA]
        m = np.array([q[0] for q in v])
        xs = np.arange(len(STRATA)) + (-0.06 if filled else 0.06)
        col = PAL["gm"]
        ax.errorbar(xs, m, yerr=[m - [q[1] for q in v], [q[2] for q in v] - m], fmt="none", ecolor=col,
                    elinewidth=fs.LW_ERR)
        ax.plot(xs, m, ls="-" if filled else (0, (2, 1.2)), color=col, lw=fs.LW_DATA * 0.8, marker="o",
                ms=fs.MS + 0.6, mfc=col if filled else "white", mec=col, mew=0.9)
        out[lab] = [list(q) for q in v]
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels([l for _, l in STRATA], fontsize=fs.FS_TICK - 0.5)
    ax.set_xlim(-0.4, len(STRATA) - 0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("GM: fraction of states\npulled toward the object")
    fs.finish(ax)
    return out


def joint(ax, ps):
    """Claim 3: both pulls toward their own object on the same state, against the nuisance null."""
    keys = (("goal_toward", "flag\ntoward"), ("goal_away", "flag\naway"), ("proxy_toward", "tower\ntoward"),
            ("both_toward", "both\ntoward"), ("null_both", "null"))
    out = {}
    for i, (k, lab) in enumerate(keys):
        for j, (c, name) in enumerate(COH):
            g = ps[(ps.cohort == c) & (ps.stratum == "all")][k].to_numpy()
            m, lo, hi = ci(g)
            out[f"{name}_{k}"] = [m, lo, hi]
            x = i + (j - 0.5) * 0.38
            col = PAL["null"] if k == "null_both" else fs.COH[name]
            ax.bar(x, m, width=0.36, color=col, lw=0)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=PAL["ink"], elinewidth=fs.LW_ERR)
            ax.scatter(np.full(len(g), x), g, s=3, color=PAL["ink"], alpha=0.5, lw=0, zorder=4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([l for _, l in keys], fontsize=fs.FS_TICK - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of states")
    fs.finish(ax)
    return out


def share_gap(ax, r):
    """Claim 4: proxy share of the removal pull against the distance gap; the nearer-object rule is the dotted step."""
    out = {}
    for c, name in COH:
        d = r[r.cohort == c].assign(b=pd.cut(r.d_flag - r.d_tower, DD_E))
        g = d.groupby(["seed", "b"], observed=False).agg(pf=("pull_flag", "mean"), pt=("pull_tower", "mean"),
                                                         n=("pull_flag", "size")).reset_index()
        g["share"] = share(g.pf, g.pt)
        g.loc[g.n < MIN_STATES, "share"] = np.nan
        sm = pd.DataFrame([dict(x=b.mid, **dict(zip(("mean", "lo", "hi"), ci(s.share))))
                           for b, s in g.groupby("b", observed=False)])
        col = fs.COH[name]
        ax.fill_between(sm.x, sm.lo.clip(0, 1), sm.hi.clip(0, 1), color=col, alpha=fs.CI_ALPHA, lw=0)
        ax.plot(sm.x, sm["mean"], color=col, lw=fs.LW_DATA, marker="o", ms=fs.MS)
        out[name] = sm.to_dict("records")
    ax.step([-5, 0, 5], [0, 1, 1], where="post", color=PAL["null"], lw=fs.LW_REF * 1.4, ls=(0, (1, 1.5)))
    ax.text(-4.8, 0.06, "nearer-object rule", fontsize=fs.FS_TICK - 1, color=PAL["null"], va="bottom")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-5, 5)
    ax.set_xlabel(r"$d_\star - d_p$ (m)")
    ax.text(0.03, 0.55, "\u2190 flag closer", transform=ax.transAxes, fontsize=fs.FS_TICK - 1, ha="left")
    ax.text(0.97, 0.55, "tower closer \u2192", transform=ax.transAxes, fontsize=fs.FS_TICK - 1, ha="right")
    ax.set_ylabel(r"tower share of the pull, $P_p/(P_p+P_\star)$")
    fs.finish(ax)
    return out


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    fs.style()
    v, ps, r = load()
    fig, A = grid(2, 2, 1.9, 0.75, 0.78, 0.24, 0.62, left=0.62)
    vals = {"pulls": pulls(A[0][0], ps), "by_stratum": by_stratum(A[0][1], ps), "joint": joint(A[1][0], ps),
            "share_gap": share_gap(A[1][1], r)}
    for ax, L, t in ((A[0][0], "a", "Tower pulls strongly, flag weakly"), (A[0][1], "b", "Flag pulls only when near"),
                     (A[1][0], "c", "The two pulls add"), (A[1][1], "d", "Tower wins even when farther")):
        titled(ax, L, t)
    h = [Line2D([], [], ls="", marker="o", ms=fs.MS + 1, color=PAL["ink"], label="flag ($f_\\star$)"),
         Line2D([], [], ls="", marker="o", ms=fs.MS + 1, mfc="white", mec=PAL["ink"], label="tower ($f_p$)"),
         Line2D([], [], color=fs.COH["GM"], lw=1.4, label="GM"), Line2D([], [], color=fs.COH["Non-GM"], lw=1.4, label="Non-GM"),
         Line2D([], [], color=fs.COH["init"], lw=1.4, label="init"), Patch(color=PAL["null"], label="nuisance null"),
         Line2D([], [], color=PAL["mute"], marker="o", ms=fs.MS, label=fs.CI_LABEL)]
    fs.legend_row(fig, h, [x.get_label() for x in h], y_in=0.03, handlelength=1.4, columnspacing=0.9)
    save(fig, "variant_wtf_4_claims")
    (TABLES / "wtf_panels_values.json").write_text(json.dumps(vals, indent=1, default=float))


if __name__ == "__main__":
    main()
