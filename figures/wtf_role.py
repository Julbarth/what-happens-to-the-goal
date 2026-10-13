"""Appendix role figures for WalkToFlag: object pull vs distance, distance-binned pull share, nuisance ablation (ghost-free banks)."""
from paths import OUT_DIR, ROBOTICS_ROOT, TABLES
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats

import style as fs  # noqa: E402
from style import PAL, REF  # noqa: E402

DATA = TABLES
FIG = OUT_DIR / "app_role"
NUIS = ROBOTICS_ROOT / "results/isaac_analysis/ghost_check/fixed/nuisance/nuisance_per_seed.csv"
COH = (("gm", "GM"), ("control", "Non-GM"))
MIN_STATES = 15  # per seed and bin; smaller bins are dropped


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def letter(ax, k, title):
    ax.set_title(title, fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)
    ax.annotate(f"({k})", (0, 1), xycoords="axes fraction", xytext=(fs.LETTER_DX, fs.TITLE_PAD), textcoords="offset points",
                fontsize=fs.FS_LETTER, fontweight="bold", va="bottom")


def binned(df, xcol, ycol, edges):
    """Per-seed bin means (bins with >= MIN_STATES states), then mean and 95% t-CI over seeds."""
    df = df.assign(bin=pd.cut(df[xcol], edges))
    g = df.groupby(["cohort", "seed", "bin"], observed=False)[ycol].agg(["mean", "size"]).reset_index()
    g.loc[g["size"] < MIN_STATES, "mean"] = np.nan
    rows = []
    for (c, b), s in g.groupby(["cohort", "bin"], observed=False):
        m, lo, hi = ci(s["mean"])
        rows.append(dict(cohort=c, bin=str(b), x=b.mid, n_seeds=int(s["mean"].notna().sum()),
                         n_states_per_seed=int(s["size"].mean()), mean=m, lo=lo, hi=hi))
    return g, pd.DataFrame(rows)


def fig_pull_vs_distance(r):
    """(a) goal pull vs distance to the flag, (b) proxy pull vs distance to the tower; GM, Non-GM, init."""
    fs.style()
    fig, axs = plt.subplots(1, 2, figsize=(fs.W_IN, 1.9), sharey=True)
    edges = {"pull_flag": np.r_[1.75, 2.25, 2.75, 3.1, 4.05, 4.75, 5.4, 6.6],
             "pull_tower": np.r_[1.75, 2.25, 2.75, 3.1, 3.55, 4.5, 5.5, 6.5, 7.55]}
    out, per_seed = [], []
    for ax, (k, ycol, xcol, filled, xl, title) in zip(axs, (
            ("a", "pull_flag", "d_flag", True, r"$d_\star$ (m)", "flag (goal)"),
            ("b", "pull_tower", "d_tower", False, r"$d_p$ (m)", "tower (proxy)"))):
        for c, name in COH + (("init", "init"),):
            g, sm = binned(r[r.cohort == c], xcol, ycol, edges[ycol])
            g["quantity"], sm["quantity"] = ycol, ycol
            per_seed.append(g)
            out.append(sm)
            col = fs.COH[name]
            # Gap between the near banks (<= 3.1 m) and the far ones: do not draw a line across it.
            for part in (sm[sm.x < 3.3], sm[sm.x >= 3.3]):
                ax.fill_between(part.x, part.lo, part.hi, color=col, alpha=fs.CI_ALPHA, lw=0)
                ax.plot(part.x, part["mean"], color=col, lw=fs.LW_DATA)
                ax.plot(part.x, part["mean"], "o", ms=fs.MS, mfc=col if filled else "white", mec=col, mew=0.9)
        ax.axhline(0, zorder=0, **REF)
        ax.set_xlabel(xl)
        letter(ax, k, title)
        fs.finish(ax)
    axs[0].set_ylabel(r"$P_o$")
    h = [Line2D([], [], color=fs.COH[n], lw=fs.LW_DATA, label=n) for n in ("GM", "Non-GM", "init")]
    axs[1].legend(handles=h, frameon=False, loc="upper right", fontsize=fs.FS_LEG, handlelength=1.4)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_app_role_wtf_pull_distance")
    pd.concat(per_seed).to_csv(DATA / "fig_pull_distance_per_seed.csv", index=False)
    sm = pd.concat(out)
    sm.to_csv(DATA / "fig_pull_distance_summary.csv", index=False)
    return sm


def share(pf, pt):
    """Proxy share of the positive pull, computed on bin means."""
    pf, pt = np.clip(pf, 0, None), np.clip(pt, 0, None)
    tot = pf + pt
    return np.where(tot > 0, pt / np.where(tot > 0, tot, 1), np.nan)


def fig_distance_map(r):
    """(a, b) proxy share of the pull over (d_flag, d_tower) for GM and Non-GM; (c) proxy share vs d_flag - d_tower."""
    fs.style()
    fig = plt.figure(figsize=(fs.W_IN, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.38)
    axs = [fig.add_subplot(gs[0, i]) for i in (0, 1)]
    axl = fig.add_subplot(gs[0, 2])
    e = np.arange(1.5, 8.01, 0.75)
    cells = []
    for ax, (c, name), k in zip(axs, COH, "ab"):
        d = r[r.cohort == c].assign(fb=pd.cut(r.d_flag, e), tb=pd.cut(r.d_tower, e))
        g = d.groupby(["seed", "fb", "tb"], observed=False).agg(pf=("pull_flag", "mean"), pt=("pull_tower", "mean"),
                                                                n=("pull_flag", "size")).reset_index()
        g["share"] = share(g.pf, g.pt)
        g.loc[g.n < MIN_STATES, "share"] = np.nan
        g["cohort"] = c
        cells.append(g)
        M = np.full((len(e) - 1, len(e) - 1), np.nan)
        for (fb, tb), s in g.groupby(["fb", "tb"], observed=False):
            if s.share.notna().sum() == s.seed.nunique() and s.share.notna().any():
                M[e.searchsorted(tb.left), e.searchsorted(fb.left)] = s.share.mean()
        cmap = LinearSegmentedColormap.from_list(c, ["#FFFFFF", fs.COH[name]])
        ax.imshow(M, origin="lower", extent=(e[0], e[-1], e[0], e[-1]), cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text((e[j] + e[j + 1]) / 2, (e[i] + e[i + 1]) / 2, f"{M[i, j]:.2f}"[1:] if M[i, j] < 1 else "1",
                            ha="center", va="center", fontsize=4.6, color="white" if M[i, j] > 0.6 else PAL["ink"])
        ax.plot([e[0], e[-1]], [e[0], e[-1]], **REF)
        ax.set_xlim(1.5, 7.5)
        ax.set_ylim(1.5, 7.5)
        ax.set_xticks([2, 4, 6])
        ax.set_yticks([2, 4, 6])
        ax.set_xlabel(r"$d_\star$ (m)")
        if k == "a":
            ax.set_ylabel(r"$d_p$ (m)")
        letter(ax, k, name)
        fs.finish(ax)
    # (c) share vs Delta d, per seed bin means.
    dd_e = np.r_[-5, -3, -2, -1, 0, 1, 2, 3, 5]
    lines = []
    for c, name in COH:
        d = r[r.cohort == c].assign(b=pd.cut(r.d_flag - r.d_tower, dd_e))
        g = d.groupby(["seed", "b"], observed=False).agg(pf=("pull_flag", "mean"), pt=("pull_tower", "mean"),
                                                         n=("pull_flag", "size")).reset_index()
        g["share"] = share(g.pf, g.pt)
        g.loc[g.n < MIN_STATES, "share"] = np.nan
        rows = []
        for b, s in g.groupby("b", observed=False):
            m, lo, hi = ci(s.share)
            rows.append(dict(cohort=c, bin=str(b), x=b.mid, n_states_per_seed=int(s.n.mean()), mean=m, lo=lo, hi=hi))
        sm = pd.DataFrame(rows)
        lines.append(sm)
        col = fs.COH[name]
        axl.fill_between(sm.x, sm.lo.clip(0, 1), sm.hi.clip(0, 1), color=col, alpha=fs.CI_ALPHA, lw=0)
        axl.plot(sm.x, sm["mean"], color=col, lw=fs.LW_DATA, marker="o", ms=fs.MS, label=name)
        g["cohort"] = c
        cells.append(g.rename(columns={"b": "dd_bin"}))
    axl.step([-5, 0, 5], [0, 1, 1], where="post", color=PAL["null"], lw=fs.LW_REF, ls=(0, (1, 1.5)), label="nearest")
    axl.axhline(0.5, zorder=0, **REF)
    axl.axvline(0, zorder=0, **REF)
    axl.set_ylim(-0.03, 1.03)
    axl.set_xlim(-5, 5)
    axl.set_xlabel(r"$d_\star - d_p$ (m)")
    axl.set_ylabel(r"$s_p$")
    axl.legend(frameon=False, fontsize=fs.FS_LEG, loc="upper left", bbox_to_anchor=(0.0, 0.62), handlelength=1.4)
    letter(axl, "c", "proxy share")
    fs.finish(axl)
    save(fig, "fig_app_role_wtf_distance_map")
    pd.concat(cells).to_csv(DATA / "fig_distance_map_per_seed.csv", index=False)
    sm = pd.concat(lines)
    sm.to_csv(DATA / "fig_distance_map_share_summary.csv", index=False)
    return sm


def fig_nuisance():
    """KL above the matched re-render null for the tower swap and four nuisance edits, visible and conflict banks."""
    fs.style()
    ps = pd.read_csv(NUIS)
    edits = (("tower_swap", "tower\nswap"), ("recolour_tower", "tower\ngrey"), ("grey_tower_swap", "grey tower\nswap"),
             ("decoy_move", "decoy\nmove"), ("decoy_insert", "decoy\nadd"))
    fig, axs = plt.subplots(1, 2, figsize=(fs.W_IN, 1.95), sharey=True)
    rows = []
    for ax, (bank, title), k in zip(axs, (("visible", "visible bank"), ("conflict", "conflict bank")), "ab"):
        for i, (e, lab) in enumerate(edits):
            for j, (c, name) in enumerate(COH + (("init", "init"),)):
                v = ps[(ps.kind == bank) & (ps.edit == e) & (ps.cohort == c)].KL_excess.to_numpy()
                m, lo, hi = ci(v)
                rows.append(dict(bank=bank, edit=e, cohort=c, n_seeds=len(v), mean=m, lo=lo, hi=hi,
                                 seeds=";".join(f"{x:.4f}" for x in v)))
                x = i + (j - 1) * 0.26
                col = fs.COH[name]
                ax.bar(x, m, width=0.24, color=col, lw=0)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=PAL["ink"], elinewidth=fs.LW_ERR)
                ax.scatter(np.full(len(v), x), v, s=4, color=PAL["ink"], alpha=0.5, lw=0, zorder=4)
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.6)
        ax.set_ylim(-0.3, 100)
        ax.axhline(0, color=PAL["ink"], lw=fs.LW_AX)
        ax.set_xticks(range(len(edits)))
        ax.set_xticklabels([l for _, l in edits], fontsize=fs.FS_TICK)
        letter(ax, k, title)
        fs.finish(ax)
    axs[0].set_ylabel(r"$\mathrm{KL} - \mathrm{KL}_{\mathrm{null}}$")
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(color=fs.COH[n], label=n) for n in ("GM", "Non-GM", "init")], frameon=False,
                  fontsize=fs.FS_LEG, loc="upper right", handlelength=1.0)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_app_role_wtf_nuisance")
    sm = pd.DataFrame(rows)
    sm.to_csv(DATA / "fig_nuisance_summary.csv", index=False)
    return sm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=["pull", "map", "nuisance"])
    a = p.parse_args()
    files = sorted(glob.glob(str(DATA / "removal_pull_*_s*.csv")))
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    vals = {"sources": {"removal_pull": files, "nuisance": str(NUIS)}}
    if "pull" in a.only:
        vals["pull_distance"] = fig_pull_vs_distance(r).to_dict("records")
    if "map" in a.only:
        vals["share_vs_dd"] = fig_distance_map(r).to_dict("records")
    if "nuisance" in a.only:
        vals["nuisance"] = fig_nuisance().to_dict("records")
    (TABLES / "fig_app_role_wtf_values.json").write_text(json.dumps(vals, indent=1, default=str))


if __name__ == "__main__":
    main()
