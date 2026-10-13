"""Recomposition variants for the WalkToFlag object-competition figures: balanced panel heights and one merged GM / Non-GM figure."""
from __future__ import annotations

from paths import OUT_DIR, TABLES
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

WTF = TABLES
import style as fs  # noqa: E402
from style import PAL, REF  # noqa: E402
from wtf_two_pulls import ci, per_seed, stratum  # noqa: E402
from wtf_role import MIN_STATES, binned, share  # noqa: E402

DATA = WTF / "data"
COH = (("gm", "GM"), ("control", "Non-GM"))
STRATA = (("vis_flag_closer", "near\n$<$"), ("vis_tower_closer", "near\n$>$"),
          ("conflict_flag_closer", "far\n$\\ll$"), ("far_tower_closer", "far\n$\\gg$"))
E_MAP = np.arange(1.5, 8.01, 0.75)
DD_E = np.r_[-5, -3, -2, -1, 0, 1, 2, 3, 5]


def grid(ncol, nrow, side, gap_x, gap_y, top, bottom, left=0.42):
    """nrow x ncol square axes of `side` inches, starting `left` inches from the figure edge."""
    w = fs.W_IN
    h = top + nrow * side + (nrow - 1) * gap_y + bottom
    fig = plt.figure(figsize=(w, h))
    x0 = left
    return fig, [[fig.add_axes([(x0 + c * (side + gap_x)) / w, (bottom + (nrow - 1 - r) * (side + gap_y)) / h,
                                side / w, side / h]) for c in range(ncol)] for r in range(nrow)]


def titled(ax, letter, title):
    ax.set_title(rf"$\bf{{({letter})}}$ {title}", loc="left", fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"app_role/{stem}.{ext}", dpi=300)
    plt.close(fig)


def load():
    v = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(DATA / "pull_vectors_*_s*.csv")))], ignore_index=True)
    v["stratum"] = stratum(v)
    r = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(DATA / "removal_pull_*_s*.csv")))], ignore_index=True)
    return v, per_seed(v), r


def scatter(ax, v, ps, cohorts, annotate=False):
    lim, ylim = (-3, 8), (-3, 13)
    for c, name in cohorts:
        x = v[v.cohort == c]
        ax.scatter(x.follow_flag_to_flag.clip(*lim), x.follow_tower_to_tower.clip(*ylim), color=fs.COH[name], **fs.STATE)
    for c, name in cohorts + ((("init", "init"),) if len(cohorts) > 1 else ()):
        g = ps[(ps.cohort == c) & (ps.stratum == "all")]
        mx, my = ci(g.f_goal), ci(g.f_proxy)
        ax.errorbar(mx[0], my[0], xerr=[[mx[0] - mx[1]], [mx[2] - mx[0]]], yerr=[[my[0] - my[1]], [my[2] - my[0]]],
                    fmt="o", ms=fs.MS + 0.6, color=fs.COH[name], mec="white", mew=0.4, elinewidth=fs.LW_ERR, zorder=5)
    if annotate:
        c = cohorts[0][0]
        g = ps[(ps.cohort == c) & (ps.stratum == "all")]
        txt = (f"toward flag {g.goal_toward.mean():.2f}\ntoward tower {g.proxy_toward.mean():.2f}\n"
               f"both {g.both_toward.mean():.2f} (null {g.null_both.mean():.3f})")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=fs.FS_TICK - 0.5,
                color=PAL["ink"], linespacing=1.15)
    ax.axhline(0, **REF)
    ax.axvline(0, **REF)
    ax.set_xlim(*lim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"flag pull $f_\star$ ($\sigma$)")
    ax.set_ylabel(r"tower pull $f_p$ ($\sigma$)")
    fs.finish(ax)


def joint_sign(ax, ps):
    keys = (("goal_toward", "flag\n+"), ("goal_away", "flag\n\u2212"), ("proxy_toward", "tower\n+"),
            ("both_toward", "both\n+"), ("null_both", "null"))
    for i, (k, lab) in enumerate(keys):
        for j, (c, name) in enumerate(COH):
            g = ps[(ps.cohort == c) & (ps.stratum == "all")][k].to_numpy()
            m, lo, hi = ci(g)
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


def by_stratum(ax, ps, cohorts):
    for j, (c, name) in enumerate(cohorts):
        for k, filled in (("goal_toward", True), ("proxy_toward", False)):
            vals = [ci(ps[(ps.cohort == c) & (ps.stratum == s)][k]) for s, _ in STRATA]
            off = (j - 0.5) * 0.12 if len(cohorts) > 1 else 0.0
            xs = np.arange(len(STRATA)) + off + (0.03 if not filled else -0.03)
            m = np.array([q[0] for q in vals])
            col = fs.COH[name]
            ax.errorbar(xs, m, yerr=[m - [q[1] for q in vals], [q[2] for q in vals] - m], fmt="none", ecolor=col,
                        elinewidth=fs.LW_ERR)
            ax.plot(xs, m, ls="-" if filled else (0, (2, 1.2)), color=col, lw=fs.LW_DATA * 0.8, marker="o", ms=fs.MS,
                    mfc=col if filled else "white", mec=col, mew=0.9)
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels([l for _, l in STRATA], fontsize=fs.FS_TICK - 0.5)
    ax.set_xlabel(r"$d_\star$ relative to $d_p$")
    ax.set_xlim(-0.4, len(STRATA) - 0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction pulled toward it")
    fs.finish(ax)


def share_map(ax, r, c, name, ylabel=True):
    d = r[r.cohort == c].assign(fb=pd.cut(r.d_flag, E_MAP), tb=pd.cut(r.d_tower, E_MAP))
    g = d.groupby(["seed", "fb", "tb"], observed=False).agg(pf=("pull_flag", "mean"), pt=("pull_tower", "mean"),
                                                            n=("pull_flag", "size")).reset_index()
    g["share"] = share(g.pf, g.pt)
    g.loc[g.n < MIN_STATES, "share"] = np.nan
    M = np.full((len(E_MAP) - 1, len(E_MAP) - 1), np.nan)
    for (fb, tb), s in g.groupby(["fb", "tb"], observed=False):
        if s.share.notna().sum() == s.seed.nunique() and s.share.notna().any():
            M[E_MAP.searchsorted(tb.left), E_MAP.searchsorted(fb.left)] = s.share.mean()
    cmap = LinearSegmentedColormap.from_list(c, ["#FFFFFF", fs.COH[name]])
    ax.imshow(M, origin="lower", extent=(E_MAP[0], E_MAP[-1], E_MAP[0], E_MAP[-1]), cmap=cmap, vmin=0, vmax=1,
              aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text((E_MAP[j] + E_MAP[j + 1]) / 2, (E_MAP[i] + E_MAP[i + 1]) / 2,
                        f"{M[i, j]:.2f}"[1:] if M[i, j] < 1 else "1", ha="center", va="center", fontsize=4.8,
                        color="white" if M[i, j] > 0.6 else PAL["ink"])
    ax.plot([E_MAP[0], E_MAP[-1]], [E_MAP[0], E_MAP[-1]], **REF)
    ax.set_xlim(1.5, 7.5)
    ax.set_ylim(1.5, 7.5)
    ax.set_xticks([2, 4, 6])
    ax.set_yticks([2, 4, 6])
    ax.set_xlabel(r"flag distance $d_\star$ (m)")
    if ylabel:
        ax.set_ylabel(r"tower distance $d_p$ (m)")
    fs.finish(ax)
    return M


def share_line(ax, r):
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
    ax.step([-5, 0, 5], [0, 1, 1], where="post", color=PAL["null"], lw=fs.LW_REF, ls=(0, (1, 1.5)))
    ax.axhline(0.5, zorder=0, **REF)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-5, 5)
    ax.set_xlabel(r"$d_\star - d_p$ (m), flag closer $<0$")
    ax.set_ylabel(r"proxy share $s_p$")
    fs.finish(ax)


def pull_distance(ax, r):
    """Flag removal pull against flag distance: the near (<= 3.1 m) and far banks are drawn as separate segments."""
    edges = np.r_[1.75, 2.25, 2.75, 3.1, 4.05, 4.75, 5.4, 6.6]
    for c, name in COH + (("init", "init"),):
        _g, sm = binned(r[r.cohort == c], "d_flag", "pull_flag", edges)
        col = fs.COH[name]
        for part in (sm[sm.x < 3.3], sm[sm.x >= 3.3]):
            ax.fill_between(part.x, part.lo, part.hi, color=col, alpha=fs.CI_ALPHA, lw=0)
            ax.plot(part.x, part["mean"], color=col, lw=fs.LW_DATA, marker="o", ms=fs.MS)
    ax.axhline(0, zorder=0, **REF)
    ax.set_xlabel(r"flag distance $d_\star$ (m)")
    ax.set_ylabel(r"flag removal pull $P_\star$")
    fs.finish(ax)


def cohort_legend(fig, extra=()):
    h = [Line2D([], [], color=fs.COH[n], marker="o", ls="", ms=fs.MS + 0.6, label=n) for n in ("GM", "Non-GM", "init")]
    h += list(extra)
    fs.legend_row(fig, h, [x.get_label() for x in h], y_in=0.03, handlelength=1.6)


LINE_KEYS = (Line2D([], [], color=PAL["ink"], marker="o", ls="-", ms=fs.MS, lw=0.8, label="flag"),
             Line2D([], [], color=PAL["ink"], marker="o", mfc="white", ls=(0, (2, 1.2)), ms=fs.MS, lw=0.8, label="tower"))


def variant_two_pulls(v, ps):
    fig, A = grid(3, 1, 1.3, 0.52, 0, 0.22, 0.6)
    A = A[0]
    scatter(A[0], v, ps, COH)
    titled(A[0], "a", "Pull of each object")
    joint_sign(A[1], ps)
    titled(A[1], "b", "Direction of the pull")
    by_stratum(A[2], ps, COH)
    titled(A[2], "c", "By distance")
    cohort_legend(fig, LINE_KEYS + (Patch(color=PAL["null"], label="null"),))
    save(fig, "variant_wtf_1_two_pulls_balanced")


def variant_distance(r):
    fig, A = grid(4, 1, 0.94, 0.42, 0, 0.22, 0.6, left=0.36)
    A = A[0]
    share_map(A[0], r, "gm", "GM")
    titled(A[0], "a", "GM share")
    share_map(A[1], r, "control", "Non-GM", ylabel=False)
    titled(A[1], "b", "Non-GM share")
    share_line(A[2], r)
    titled(A[2], "c", "Share vs. gap")
    pull_distance(A[3], r)
    titled(A[3], "d", "Flag pull")
    cohort_legend(fig, (Line2D([], [], color=PAL["null"], lw=fs.LW_REF, ls=(0, (1, 1.5)), label="nearer object"),))
    save(fig, "variant_wtf_2_distance_balanced")


def variant_merged(v, ps, r):
    """Rows GM and Non-GM, columns pull scatter (with quadrant shares), toward-own-object by stratum, proxy-share map."""
    fig, A = grid(3, 2, 1.24, 0.52, 0.52, 0.22, 0.6, left=0.62)
    for row, (c, name) in enumerate(COH):
        scatter(A[row][0], v, ps, ((c, name),), annotate=True)
        by_stratum(A[row][1], ps, ((c, name),))
        share_map(A[row][2], r, c, name)
        b = A[row][0].get_position()
        fig.text(0.012, b.y0 + b.height / 2, name, rotation=90, ha="left", va="center",
                 fontsize=fs.FS_TITLE, color=fs.COH[name])
    for ax, L, t in zip([a for row in A for a in row], "abcdef",
                        ("Pull of each object", "By distance", "Proxy share of pull") * 2):
        titled(ax, L, t)
    h = list(LINE_KEYS) + [Line2D([], [], color=PAL["mute"], marker="o", ls="", ms=fs.MS + 0.6, mec="white",
                                  label=fs.CI_LABEL)]
    fs.legend_row(fig, h, [x.get_label() for x in h], y_in=0.03, handlelength=1.6)
    save(fig, "variant_wtf_3_merged")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    fs.style()
    v, ps, r = load()
    variant_two_pulls(v, ps)
    variant_distance(r)
    variant_merged(v, ps, r)
    ref = pd.read_csv(DATA / "two_pulls_per_seed.csv")
    same = np.allclose(ref.sort_values(["cohort", "seed", "stratum"]).f_proxy.to_numpy(),
                       ps.sort_values(["cohort", "seed", "stratum"]).f_proxy.to_numpy())
    (TABLES / "wtf_base_values.json").write_text(json.dumps({"per_seed_matches_saved": bool(same)}))
    print("per-seed pulls match saved two_pulls_per_seed.csv:", same)


if __name__ == "__main__":
    main()
