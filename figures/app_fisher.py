"""Appendix Fisher chapter: per-seed Q, G, D_F for all eight environments, the G-vs-D_F figure and the LaTeX table rows."""
from __future__ import annotations
from paths import OUT_DIR, ROBOTICS_ROOT, TABLES

# Run from the procgen/ directory:
#   python figures/app_fisher.py            # figure + table rows from the cache
#   python figures/app_fisher.py --extract  # rebuild the cache first
# --extract reads the robot per-seed CSVs in robotics code (worldedit_{PTM,LR,AT,SZ}_per_seed.csv, isaac ghost_check/fixed/
# worldedit_per_seed.csv, bank visible) and the three Procgen summaries in the released
# data/probes/ archive (canon_mcfisher_20260924/summary_mcfisher.json,
# coin_fisher_canon_20260924/reaggregated_paper_convention.json, canon_fisher_20260924/summary_canon_swap.json),
# fetched to --procgen-dir first. Every value is a per-seed median over the evaluation states.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GR = ROBOTICS_ROOT
CACHE = TABLES / "fisher_seed_values_all8.csv"
OUT = OUT_DIR / "app_fisher/fig_app_fisher_g_vs_df"
# Two-sided 97.5% Student-t quantiles by degrees of freedom, tabulated so scipy is not needed.
T975 = {2: 4.302653, 3: 3.182446, 4: 2.776445}
COH = {"gm": "GM", "control": "Non-GM", "nongm": "Non-GM", "init": "init", "GM": "GM", "Non-GM": "Non-GM"}

# (env, edit key in source, edit label, role); order is the order of the table and the figure.
ROBOT = [
    ("PushToMarker", GR / "results/analysis/worldedit_PTM_per_seed.csv", [("marker_move", "Marker", "goal")]),
    ("LocationReach", GR / "results/analysis/worldedit_LR_per_seed.csv", [("goal_move", "Ball", "goal")]),
    ("WalkToFlag", GR / "results/isaac_analysis/ghost_check/fixed/worldedit_per_seed.csv",
     [("flag_swap", "Flag", "goal"), ("tower_swap", "Tower", "proxy")]),
    ("AttributeTable", GR / "results/analysis/worldedit_AT_per_seed.csv",
     [("shape_swap", "Shape", "goal"), ("colour_swap", "Colour", "proxy")]),
    ("SortZones", GR / "results/analysis/worldedit_SZ_per_seed.csv",
     [("mark_swap", "Cross", "goal"), ("colour_swap", "Colour", "proxy")]),
]
# The three groups of the world-edit appendix (app:world-edit), in its order.
GROUPS = [("Goal inert", ["Maze", "LocationReach"]),
          ("Goal reaches, does not direct", ["PushToMarker", "Coin"]),
          ("Goal enters and loses", ["Attribute Maze", "AttributeTable", "SortZones", "WalkToFlag"])]
# Row order of the appendix table and the ratio figure: the order of the world-edit table (tab:world-edit-dependence).
TABLE_ORDER = ["Maze", "LocationReach", "PushToMarker", "Coin", "WalkToFlag", "Attribute Maze", "AttributeTable",
               "SortZones"]
# The AttributeTable Non-GM cohort of the paper is seeds 3, 4, 5. Seed 2 stays in the cache and is dropped here.
DROP = [("AttributeTable", "Non-GM", 2)]


def cohort_filter(df: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(df), bool)
    for env, coh, seed in DROP:
        keep &= ~((df.env == env) & (df.cohort == coh) & (df.seed == seed))
    return df[keep]


def extract(procgen_dir: Path) -> pd.DataFrame:
    import json
    rows = []
    for env, path, edits in ROBOT:
        d = pd.read_csv(path)
        if "bank" in d:
            d = d[d.bank == "visible"]
        for key, label, role in edits:
            for _, r in d[d.edit == key].iterrows():
                for site in ("h_pre", "h_fc"):
                    rows.append(dict(env=env, edit=label, role=role, cohort=COH[r.cohort], seed=int(r.seed), site=site,
                                     Q=r[f"Q_{site}"], G=r[f"G_{site}"], D_F=r[f"D_F_{site}"]))
    mz = json.loads((procgen_dir / "canon_mcfisher_20260924_summary_mcfisher.json").read_text())["figure"]["maze"]
    for site, tap in (("h_pre", "block3"), ("h_fc", "fc")):
        for c in ("gm", "nongm", "init"):
            for i in range(3):
                rows.append(dict(env="Maze", edit="Cheese", role="goal", cohort=COH[c], seed=i, site=site,
                                 **{k: mz[tap][k][f"{c}_seed_values"][i] for k in ("Q", "G", "D_F")}))
    co = json.loads((procgen_dir / "coin_fisher_canon_20260924_reaggregated_paper_convention.json").read_text())
    for site in ("h_pre", "h_fc"):
        for c in ("GM", "Non-GM", "init"):
            for i, v in enumerate(co[site][c]["per_ckpt"].values()):
                rows.append(dict(env="Coin", edit="Coin", role="goal", cohort=COH[c], seed=i, site=site,
                                 Q=v["Q_median"], G=v["G_median"], D_F=v["D_F_median"]))
    am = json.loads((procgen_dir / "canon_fisher_20260924_summary_canon_swap.json").read_text())["figure_ras"]["black"]
    for site, tap in (("h_pre", "block3"), ("h_fc", "fc")):
        for e, label, role in (("shape", "Shape", "goal"), ("colour", "Colour", "proxy")):
            for c in ("gm", "nongm", "init"):
                for i in range(3):
                    rows.append(dict(env="Attribute Maze", edit=label, role=role, cohort=COH[c], seed=i, site=site,
                                     **{k: am[tap][f"{k}_{e}"][f"{c}_seed_values"][i] for k in ("Q", "G", "D_F")}))
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df


def fmt(v: np.ndarray) -> str:
    return f"${np.mean(v):.3f}\\pm{np.std(v, ddof=1):.2f}$"


def table_rows(df: pd.DataFrame) -> str:
    """Body of the appendix table: mean +- SD (ddof=1) over seeds, h_pre then h_fc."""
    lines = []
    for env in TABLE_ORDER:
        sub = df[df.env == env]
        for edit in dict.fromkeys(sub.edit):
            for coh in ("GM", "Non-GM", "init"):
                s = sub[(sub.edit == edit) & (sub.cohort == coh)]
                cells = [fmt(s[s.site == site].sort_values("seed")[k].to_numpy())
                         for site in ("h_pre", "h_fc") for k in ("Q", "G", "D_F")]
                lines.append(f"{env} & {edit} & {coh} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def ci_log(v: np.ndarray) -> tuple[float, float, float]:
    """Geometric mean and 95% t-interval computed on log values, so the interval stays positive on log axes."""
    lv = np.log(v)
    m, se = lv.mean(), lv.std(ddof=1) / np.sqrt(len(lv))
    h = T975[len(lv) - 1] * se
    return float(np.exp(m)), float(np.exp(m - h)), float(np.exp(m + h))


def figure(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import style as fs

    fs.style()
    d = df[df.site == "h_fc"]
    fig = plt.figure(figsize=(fs.W_IN, 3.75))
    gs = fig.add_gridspec(2, 4, left=0.075, right=0.985, top=0.885, bottom=0.17, wspace=0.42, hspace=0.75)
    order = [env for _, envs in GROUPS for env in envs]
    pos = {env: divmod(i, 4) for i, env in enumerate(order)}
    axes = {}
    mk = {"goal": "o", "proxy": "s"}
    soft = {"GM": fs.PAL["gm_proxy"], "Non-GM": fs.PAL["nongm_proxy"]}
    for env, (r, c) in pos.items():
        ax = axes[env] = fig.add_subplot(gs[r, c])
        sub = d[d.env == env]
        init_g = sub[sub.cohort == "init"].G
        ax.axhline(init_g.mean(), **fs.REF)
        for (edit, role), se in sub[sub.cohort != "init"].groupby(["edit", "role"], sort=False):
            for coh in ("GM", "Non-GM"):
                s = se[se.cohort == coh]
                col = fs.COH[coh] if role == "goal" else soft[coh]
                ax.scatter(s.D_F, s.G, color=col, marker=mk[role], **fs.SEED)
                gx, xlo, xhi = ci_log(s.D_F.to_numpy())
                gy, ylo, yhi = ci_log(s.G.to_numpy())
                ax.plot([xlo, xhi], [gy, gy], color=col, lw=fs.LW_ERR, zorder=3)
                ax.plot([gx, gx], [ylo, yhi], color=col, lw=fs.LW_ERR, zorder=3)
                if role == "goal":
                    ax.plot([gx], [gy], marker="o", color=col, ms=fs.MS, mec="white", mew=0.4, zorder=4, ls="")
                else:
                    ax.plot([gx], [gy], marker="s", mfc="white", mec=col, ms=fs.MS, mew=0.9, zorder=4, ls="")
        ax.set_xscale("log")
        ax.set_yscale("log")
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xl = (xl[0] / 2, xl[1] * 2)
        yl = (min(yl[0], 0.6), max(yl[1] * 1.25, 2.5))
        xs = np.geomspace(*xl, 50)
        for q in 10.0 ** np.arange(-6, 5):
            ax.plot(xs, q / xs, ls=":", lw=fs.LW_REF, color="#BBBBBB", zorder=0)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        import matplotlib.ticker as mt
        dec = np.arange(np.ceil(np.log10(xl[0])), np.floor(np.log10(xl[1])) + 1)
        # Panels are narrow, so more than two decade labels collide.
        ax.set_xticks(10.0 ** dec[::1 if len(dec) <= 2 else int(np.ceil(len(dec) / 2))])
        ax.xaxis.set_minor_formatter(mt.NullFormatter())
        ax.yaxis.set_major_locator(mt.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
        ax.yaxis.set_minor_formatter(mt.NullFormatter())
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        fs.finish(ax)
        ax.set_title(env, fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)
        ax.set_xlabel(r"$D_F$", labelpad=1)
        if c == 0:
            ax.set_ylabel(r"$G$", labelpad=1)
    fig.canvas.draw()
    for gname, envs in GROUPS:
        boxes = [axes[e].get_position() for e in envs]
        x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
        y = max(b.y1 for b in boxes) + 0.062
        fig.text((x0 + x1) / 2, y, gname, fontsize=fs.FS_TITLE, weight="bold", ha="center", va="bottom")
        fig.add_artist(Line2D([x0, x1], [y - 0.008] * 2, color=fs.PAL["mute"], lw=0.5))
    hs = [Line2D([], [], color=fs.COH["GM"], marker="o", ls="", ms=fs.MS, label="GM goal edit"),
          Line2D([], [], color=fs.COH["Non-GM"], marker="o", ls="", ms=fs.MS, label="Non-GM goal edit"),
          Line2D([], [], mec=fs.PAL["gm_proxy"], mfc="white", marker="s", ls="", ms=fs.MS, mew=0.9, label="GM proxy edit"),
          Line2D([], [], mec=fs.PAL["nongm_proxy"], mfc="white", marker="s", ls="", ms=fs.MS, mew=0.9,
                 label="Non-GM proxy edit"),
          Line2D([], [], **fs.REF, label="init $G$"),
          Line2D([], [], ls=":", lw=fs.LW_REF, color="#BBBBBB", label="constant $Q$")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, frameon=False, fontsize=fs.FS_LEG,
               handletextpad=0.3, columnspacing=0.9)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)


# Rows of the ratio figure, in the order of the world-edit ratio figure (fig:worldedit-ratios).
RATIO_ROWS = [
    ("Goal inert", "Maze, cheese moved", "Maze", "Cheese"),
    ("Goal inert", "LocationReach, ball moved", "LocationReach", "Ball"),
    ("Goal reaches, does not direct", "PushToMarker, marker moved", "PushToMarker", "Marker"),
    ("Goal reaches, does not direct", "Coin, coin moved", "Coin", "Coin"),
    ("Goal enters and loses", "WalkToFlag, flag moved", "WalkToFlag", "Flag"),
    ("Goal enters and loses", "WalkToFlag, tower moved", "WalkToFlag", "Tower"),
    ("Goal enters and loses", "Attribute Maze, shape swap", "Attribute Maze", "Shape"),
    ("Goal enters and loses", "Attribute Maze, colour swap", "Attribute Maze", "Colour"),
    ("Goal enters and loses", "AttributeTable, shape swap", "AttributeTable", "Shape"),
    ("Goal enters and loses", "AttributeTable, colour swap", "AttributeTable", "Colour"),
    ("Goal enters and loses", "SortZones, cross swap", "SortZones", "Cross"),
    ("Goal enters and loses", "SortZones, colour swap", "SortZones", "Colour"),
]
RATIO_OUT = OUT_DIR / "app_fisher/fig_app_fisher_ratios"


def ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    """GM / Non-GM at h_fc for Q, D_F and G: ratio of cohort means, and each GM seed over the Non-GM mean."""
    d = df[df.site == "h_fc"]
    rows = []
    for group, label, env, edit in RATIO_ROWS:
        s = d[(d.env == env) & (d.edit == edit)]
        for k in ("Q", "D_F", "G"):
            g = s[s.cohort == "GM"].sort_values("seed")[k].to_numpy()
            n = s[s.cohort == "Non-GM"].sort_values("seed")[k].to_numpy()
            rows.append(dict(group=group, row=label, env=env, edit=edit, role=s.role.iloc[0], metric=k,
                             R=g.mean() / n.mean(), seed_ratios=np.round(g / n.mean(), 6).tolist(),
                             gm_mean=g.mean(), nongm_mean=n.mean()))
    return pd.DataFrame(rows)


def ratio_figure(t: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import style as fs

    fs.style()
    labels = [r[1] for r in RATIO_ROWS]
    y, ys = 0.0, {}
    for _, lab, *_ in RATIO_ROWS:
        ys[lab] = -y
        y += 1.0
    h = 0.78 + 0.155 * y
    fig = plt.figure(figsize=(fs.W_IN, h))
    left, w, gap, y0, top = 1.62, 1.05, 0.22, 0.62, 0.16
    X = lambda x: x / fs.W_IN
    col = {"goal": fs.PAL["gm_goal"], "proxy": fs.PAL["gm_proxy"]}
    names = {"Q": "$Q$, GM ÷ Non-GM", "D_F": "$D_F$, GM ÷ Non-GM", "G": "$G$, GM ÷ Non-GM"}
    lims = {"Q": (2e-4, 300), "D_F": (5e-4, 50), "G": (0.2, 10)}
    ticks = {"Q": [1e-3, 1e-1, 10], "D_F": [1e-3, 1e-1, 10], "G": [0.3, 1, 3, 10]}
    axes = []
    for i, k in enumerate(("Q", "D_F", "G")):
        ax = fig.add_axes([X(left + i * (w + gap)), y0 / h, X(w), (h - y0 - top) / h],
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        for _, r in t[t.metric == k].iterrows():
            c, yy = col[r.role], ys[r.row]
            per = np.asarray(r.seed_ratios)
            ax.scatter(per, np.full(len(per), yy), s=7, color=c, alpha=0.4, lw=0, zorder=4)
            goal = r.role == "goal"
            ax.plot([r.R], [yy], "o", ms=fs.MS + 1.0, mfc=c if goal else "white", mec=c, mew=0.4 if goal else 1.1,
                    zorder=5)
        ax.axvline(1.0, **fs.REF, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(*lims[k])
        ax.set_xticks(ticks[k])
        ax.set_xticklabels([f"{v:g}" for v in ticks[k]])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.spines["left"].set_visible(False)
        fs.finish(ax)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(names[k])
        if i:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].set_yticks([ys[l] for l in labels])
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(-y + 0.3, 0.7)
    c = fs.PAL["gm"]
    hl = [Line2D([], [], ls="", marker="o", ms=fs.MS + 1, color=c, mec=c, mew=0.4),
          Line2D([], [], ls="", marker="o", ms=fs.MS + 1, mfc="white", mec=fs.PAL["gm_proxy"], mew=1.1),
          Line2D([], [], ls="", marker="o", ms=2.4, color=c, alpha=0.4, mec="none")]
    fig.legend(hl, ["goal edit", "proxy edit", "single GM seed"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=fs.FS_LEG, handletextpad=0.3, columnspacing=1.4,
               borderaxespad=0)
    for ext in ("pdf", "png"):
        fig.savefig(RATIO_OUT.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--procgen-dir", type=Path, default=TABLES / "_procgen_src")
    ap.add_argument("--rows-out", type=Path, default=TABLES / "tab_fisher_all8_rows.tex")
    a = ap.parse_args()
    df = cohort_filter(extract(a.procgen_dir) if a.extract else pd.read_csv(CACHE))
    a.rows_out.write_text(table_rows(df) + "\n")
    figure(df)
    t = ratio_table(df)
    t.to_csv(TABLES / "fisher_ratios.csv", index=False)
    ratio_figure(t)


if __name__ == "__main__":
    main()
