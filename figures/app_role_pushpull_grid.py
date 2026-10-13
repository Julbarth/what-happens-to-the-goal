"""Appendix role figure: S_fav / S_comp push-pull per environment (rows) and swap (columns), in the Attribute Maze figure format."""
from __future__ import annotations

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import style as fs  # noqa: E402
from style import MS, PAL, REF  # noqa: E402

GR = ROBOTICS_ROOT
AM_ROSTER = {"black": {"GM": ["gm_6313", "gm_6315", "gm_6317"], "Non-GM": ["deco_4371", "deco_4372", "deco_4373"]},
             "texture": {"GM": ["gm_6301", "gm_6302", "gm_6303"], "Non-GM": ["deco_4361", "deco_4362", "deco_4363"]}}
# AttributeTable Non-GM seed 2 is dropped everywhere in the paper.
ROBOT_SEEDS = {"AT": {"GM": [2, 3, 4], "Non-GM": [3, 4, 5]}, "SZ": {"GM": [0, 1, 2], "Non-GM": [0, 1, 2]},
               "WTF": {"GM": [0, 1, 2], "Non-GM": [0, 1, 2]}}
COH = {"GM": "gm", "Non-GM": "control"}


def am_rows(probes: Path, bg: str) -> pd.DataFrame:
    """Per-state S for the canonical Attribute Maze swaps; seed means average the two conflict signs as in the paper."""
    res = json.loads((probes / f"canon_cue_20260924/ras_cue_competition_canon_swap_{bg}.json").read_text())["results"]
    out = []
    for c, labs in AM_ROSTER[bg].items():
        for lab in labs:
            for r in res[lab]:
                out.append(dict(cohort=c, seed=lab, swap="goal" if r["edit"] == "shape" else "proxy",
                                S_fav=r["S_fav"], S_comp=r["S_comp"], sign=r["axis_sign"]))
    return pd.DataFrame(out)


def robot_rows(env: str) -> pd.DataFrame:
    if env == "WTF":
        d = pd.read_csv(GR / "results/isaac_analysis/ghost_check/fixed/push_pull/isaac_per_state.csv")
        d = d[d.bank == "visible"].rename(columns={"S_fav_cone30": "Sf", "S_comp_cone30": "Sc"})
        d = d.drop(columns=["S_fav", "S_comp"]).rename(columns={"Sf": "S_fav", "Sc": "S_comp"})
        goal = "flag_swap"
    else:
        d = pd.read_csv(GR / f"results/analysis/push_pull/{env}_per_state.csv")
        # AttributeTable on the strict bank (opposite sides, >= 45 deg apart from the start pose), SortZones at decision frames.
        d = d[d.bank == ("strict" if env == "AT" else "decision")]
        goal = "shape_swap" if env == "AT" else "mark_swap"
    out = []
    for c, seeds in ROBOT_SEEDS[env].items():
        s = d[(d.cohort == COH[c]) & d.seed.isin(seeds)]
        out.append(pd.DataFrame(dict(cohort=c, seed=s.seed, swap=np.where(s.edit == goal, "goal", "proxy"),
                                     S_fav=s.S_fav, S_comp=s.S_comp, sign=1)))
    return pd.concat(out, ignore_index=True)


def seed_means(sub: pd.DataFrame) -> pd.DataFrame:
    g = sub.groupby(["seed", "sign"])[["S_fav", "S_comp"]].mean().groupby("seed").mean()
    return g


def ci_half(v) -> float:
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def limits(df: pd.DataFrame, q: float):
    """Row limits from the 1-99% state range (WalkToFlag states are heavy-tailed) plus every seed-mean CI."""
    xs, ys = [0.0], [0.0]
    for (_, _), sub in df.groupby(["swap", "cohort"]):
        xs += list(np.quantile(sub.S_fav, [q, 1 - q]))
        ys += list(np.quantile(sub.S_comp, [q, 1 - q]))
        m = seed_means(sub)
        xs += [m.S_fav.mean() - ci_half(m.S_fav), m.S_fav.mean() + ci_half(m.S_fav)]
        ys += [m.S_comp.mean() - ci_half(m.S_comp), m.S_comp.mean() + ci_half(m.S_comp)]
    px, py = 0.06 * (max(xs) - min(xs)), 0.06 * (max(ys) - min(ys))
    return (min(xs) - px, max(xs) + px), (min(ys) - py, max(ys) + py)


def panel(ax, df: pd.DataFrame, swap: str, xl, yl) -> dict:
    ax.fill_between([0, xl[1]], yl[0], 0, color=PAL["mute"], alpha=0.06, lw=0, zorder=0)
    out = {}
    for c in ("GM", "Non-GM"):
        sub = df[(df.swap == swap) & (df.cohort == c)]
        col = fs.COH[c]
        ax.scatter(np.clip(sub.S_fav, *xl), np.clip(sub.S_comp, *yl), s=1.6, facecolor=col, edgecolor="none",
                   alpha=0.10, zorder=2, rasterized=True)
        m = seed_means(sub)
        ax.scatter(m.S_fav, m.S_comp, s=14, marker="D", facecolor=col, edgecolor="white", linewidth=0.4, zorder=4)
        ax.errorbar(m.S_fav.mean(), m.S_comp.mean(), xerr=ci_half(m.S_fav), yerr=ci_half(m.S_comp), fmt="o",
                    ms=MS - 0.6, color=col, mfc="white", mec=col, mew=0.8, elinewidth=fs.LW_ERR, capsize=1.5,
                    zorder=5)
        out[c] = {"seeds": [str(s) for s in m.index], "S_fav": m.S_fav.tolist(), "S_comp": m.S_comp.tolist()}
    ax.axhline(0, zorder=1, **REF)
    ax.axvline(0, zorder=1, **REF)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    fs.finish(ax)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", type=Path, required=True, help="local copy of data/probes (canon_cue_20260924)")
    ap.add_argument("--rows", nargs="+", default=["am_black", "am_texture", "AT", "SZ", "WTF"])
    ap.add_argument("--panel", type=float, default=1.22, help="panel height in inches")
    ap.add_argument("--q", type=float, default=0.01, help="state quantile clipped at each axis end (0.05 for WalkToFlag)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "app_role/fig_app_role_pushpull_grid")
    a = ap.parse_args()
    fs.FS_TICK = fs.FS_LEG = fs.FS_SUB = 7.0
    fs.style()
    names = {"am_black": ("Attribute Maze, black", "shape", "colour"),
             "am_texture": ("Attribute Maze, texture", "shape", "colour"),
             "AT": ("AttributeTable", "shape", "colour"), "SZ": ("SortZones", "cross", "colour"),
             "WTF": ("WalkToFlag", "flag", "tower")}
    # Two environments per row (goal swap, proxy swap each), so panels stay close to square.
    n_rows = (len(a.rows) + 1) // 2
    top_in, bot_in, gap_in = 0.42, 0.36, 0.62
    H = n_rows * a.panel + (n_rows - 1) * gap_in + top_in + bot_in
    fig = plt.figure(figsize=(fs.W_IN, H))
    left, right, inner, mid = 0.075, 0.99, 0.035, 0.085
    pw = (right - left - 2 * inner - mid) / 4
    vals = {}
    for i, key in enumerate(a.rows):
        r, half = divmod(i, 2)
        y0 = (H - top_in - (r + 1) * a.panel - r * gap_in) / H
        df = am_rows(a.probes, key.split("_")[1]) if key.startswith("am_") else robot_rows(key)
        xl, yl = limits(df, 0.05 if key == "WTF" else a.q)
        title, gname, pname = names[key]
        axs = []
        for j, (swap, sname) in enumerate((("goal", gname), ("proxy", pname))):
            x0 = left + half * (2 * pw + inner + mid) + j * (pw + inner)
            ax = fig.add_axes([x0, y0, pw, a.panel / H])
            vals[f"{key}/{swap}"] = panel(ax, df, swap, xl, yl)
            if j == 1:
                ax.tick_params(labelleft=False)
            axs.append(ax)
            ax.set_xlabel(r"$S_{\mathrm{fav}}$", labelpad=1)
        axs[0].set_ylabel(r"$S_{\mathrm{comp}}$", labelpad=1)
        fs.panel_title(axs[0], None, sub=f"{gname} swap")
        fs.panel_title(axs[1], None, sub=f"{pname} swap")
        axs[0].annotate(title, xy=(1 + inner / pw / 2, 1), xycoords="axes fraction", xytext=(0, 3 + fs.FS_SUB * 1.35),
                        textcoords="offset points", ha="center", va="bottom", fontsize=fs.FS_TITLE + 0.5,
                        color=PAL["ink"], fontweight=fs.TITLE_WEIGHT)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=2.2, color=fs.COH["GM"], alpha=0.5, mec="none"),
         Line2D([], [], ls="", marker="o", ms=2.2, color=fs.COH["Non-GM"], alpha=0.5, mec="none"),
         Line2D([], [], ls="", marker="D", ms=MS, color=fs.COH["GM"], mec="white", mew=0.4),
         Line2D([], [], ls="", marker="D", ms=MS, color=fs.COH["Non-GM"], mec="white", mew=0.4),
         Line2D([], [], color=PAL["mute"], lw=fs.LW_ERR, marker="o", ms=MS - 0.6, mfc="white", mec=PAL["mute"])]
    if len(a.rows) % 2:
        fig.legend(h, ["GM state", "Non-GM state", "GM seed mean", "Non-GM seed mean", fs.CI_LABEL],
                   loc="center", bbox_to_anchor=(left + 3 * pw + inner + mid, (bot_in + a.panel / 2) / H),
                   frameon=False, fontsize=fs.FS_LEG, ncol=1, handletextpad=0.4, labelspacing=0.7)
    else:
        fs.legend_row(fig, h, ["GM state", "Non-GM state", "GM seed mean", "Non-GM seed mean", fs.CI_LABEL], y_in=0.03)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"), dpi=300)
    fig.savefig(a.out.with_suffix(".png"), dpi=200)
    (TABLES / f"{a.out.name}_values.json").write_text(json.dumps(vals, indent=1))
    print("wrote", a.out, f"{fs.W_IN} x {H:.2f} in")


if __name__ == "__main__":
    main()
