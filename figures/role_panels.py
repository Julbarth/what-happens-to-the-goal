"""Goal-role panels and loaders (Maze detour/new corner, Coin r_star, Attribute Maze push-pull), reused by main_goal_role.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from role_base import FS, PAL, W_IN, ci_half, style  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

DETOUR_JSON = DATA_ROOT / "probes/maze_detour_definition_check_20260925/result_summary.json"
NEW_DIR = DATA_ROOT / "probes/maze_corner_new_vs_old_20260925"
OFF_DIR = DATA_ROOT / "probes/maze_corner_offscreen_20260925"
COIN_JSON = DATA_ROOT / "probes/coin_edit_action_groups_20260925/summary.json"
AM_JSON = DATA_ROOT / "probes/canon_cue_20260924/ras_cue_competition_canon_swap_black.json"
MAZE_CKPTS = {"GM": ["1080_gm", "1081_gm", "1082_gm"], "Non-GM": ["1090_teach", "1091_teach", "1092_teach"]}
# Same black roster as the appendix figure.
AM_ROSTER = {"GM": ["gm_6313", "gm_6315", "gm_6317"], "Non-GM": ["deco_4371", "deco_4372", "deco_4373"]}
AGENTS = ("GM", "Non-GM")
COIN_COH = {"GM": "GM", "Non-GM": "NonGM", "init": "init"}
COL = {"GM": PAL["gm"], "Non-GM": PAL["nongm"], "init": PAL["init"]}
REF = dict(color=PAL["mute"], lw=0.6, ls=(0, (3, 2)))
SEED = dict(s=5, alpha=0.35, lw=0, zorder=3)
MS, MS_OPEN = 3.4, 4.6


def load_maze() -> dict:
    """Per-seed (goal-side outcome, alternative outcome) for the three Maze conditions."""
    d = json.loads(DETOUR_JSON.read_text())["summary"]["D_text"]
    new = json.loads((NEW_DIR / "summary.json").read_text())["per_ckpt"]
    off = json.loads((OFF_DIR / "summary.json").read_text())["per_ckpt"]
    out = {}
    for c, labs in MAZE_CKPTS.items():
        k = {"GM": "GM", "Non-GM": "NonGM"}[c]
        out[("Detour", c)] = (d[k]["cat_corner"]["vals"], d[k]["cat_cheese_only"]["vals"])
        out[("New corner", c)] = ([new[l]["reach_new_first"] for l in labs], [new[l]["reach_old_first"] for l in labs])
        out[("Off-screen", c)] = ([off[l]["tr_all_reach_new_first"] for l in labs],
                                  [off[l]["tr_all_reach_old_first"] for l in labs])
    return {k: (np.asarray(a, float), np.asarray(b, float)) for k, (a, b) in out.items()}


def load_coin() -> dict:
    c = json.loads(COIN_JSON.read_text())["cohort"]
    return {k: np.asarray(c[v]["tv_over_mass_star"]["per_seed"], float) for k, v in COIN_COH.items()}


def seed_mean(rows: list[dict], met: str) -> float:
    return float(np.mean([np.mean([r[met] for r in rows if r["axis_sign"] == s]) for s in (1, -1)]))


def maze_panel(ax, maze: dict):
    groups = ["Detour", "New corner", "Off-screen"]
    off = {"GM": 0.19, "Non-GM": -0.19}
    for gi, g in enumerate(groups):
        y0 = -gi
        for c in AGENTS:
            y = y0 + off[c]
            goal, alt = maze[(g, c)]
            ax.plot([alt.mean(), goal.mean()], [y, y], color=COL[c], lw=1.8, alpha=0.2, solid_capstyle="butt", zorder=2)
            for v, filled in ((alt, False), (goal, True)):
                ax.scatter(v, np.full_like(v, y), color=COL[c], **SEED)
                ax.errorbar(v.mean(), y, xerr=ci_half(v), fmt="none", ecolor=COL[c], elinewidth=0.7, capsize=0,
                            zorder=4)
            # Open marker is larger and underneath, so goal = alternative still reads as a ring around a dot.
            ax.plot(alt.mean(), y, "o", ms=MS_OPEN, mfc="white", mec=COL[c], mew=0.9, zorder=4.5)
            ax.plot(goal.mean(), y, "o", ms=MS, color=COL[c], mec="white", mew=0.4, zorder=5)
    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels(groups)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-2.5, 0.5)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("fraction of states")


def coin_panel(ax, coin: dict, rng):
    ax.axhline(1, zorder=1, **REF)
    for i, (c, v) in enumerate(coin.items()):
        ax.scatter(i - 0.14 + rng.uniform(-0.05, 0.05, len(v)), v, color=COL[c], **SEED)
        ax.errorbar(i + 0.12, v.mean(), yerr=ci_half(v), fmt="o", ms=MS, color=COL[c], mec="white", mew=0.4,
                    elinewidth=0.7, capsize=0, zorder=5)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 7.6)
    ax.set_yticks([0, 1, 3, 5, 7])
    ax.set_xticks(range(len(coin)))
    ax.set_xticklabels(list(coin))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"$r_\star$", labelpad=3, rotation=0, va="center", fontsize=FS + 1)


def am_limits(res: dict) -> tuple:
    """Shared row limits exactly as in the appendix script: all black states, both edits, both cohorts."""
    pts = [(r["S_fav"], r["S_comp"]) for rost in AM_ROSTER.values() for lab in rost for r in res[lab]]
    xs, ys = np.array(pts).T
    pad = 0.08
    return ((min(xs.min(), 0) - pad * np.ptp(xs), max(xs.max(), 0) + pad * np.ptp(xs)),
            (min(ys.min(), 0) - pad * np.ptp(ys), max(ys.max(), 0) + pad * np.ptp(ys)))


def am_panel(ax, res: dict, edit: str, xlim, ylim, ylabel: bool) -> dict:
    ax.fill_between([0, xlim[1]], ylim[0], 0, color=PAL["mute"], alpha=0.06, lw=0, zorder=0)
    stats = {}
    for c, labs in AM_ROSTER.items():
        seeds = []
        for lab in labs:
            rr = [r for r in res[lab] if r["edit"] == edit]
            ax.scatter([r["S_fav"] for r in rr], [r["S_comp"] for r in rr], s=1.6, facecolor=COL[c],
                       edgecolor="none", alpha=0.12, zorder=2, rasterized=True)
            seeds.append((seed_mean(rr, "S_fav"), seed_mean(rr, "S_comp")))
        sx, sy = np.array(seeds).T
        ax.scatter(sx, sy, s=13, facecolor=COL[c], edgecolor="white", linewidth=0.35, marker="D", zorder=4)
        ax.errorbar(sx.mean(), sy.mean(), xerr=ci_half(sx), yerr=ci_half(sy), fmt="o", ms=2.4, color=COL[c],
                    mec="white", mew=0.35, capsize=1.3, capthick=0.6, lw=0.7, zorder=5)
        stats[c] = {"seed_S_fav": sx.tolist(), "seed_S_comp": sy.tolist()}
    ax.axhline(0.0, color=PAL["mute"], lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax.axvline(0.0, color=PAL["mute"], lw=0.5, ls=(0, (3, 2)), zorder=0)
    # Two lines instead of one: at this size the one-line version spans the panel and covers states.
    ax.text(xlim[1] * 0.97, ylim[0] * 0.05, "favoured ↑,\ncompetitor ↓", fontsize=5.0, color=PAL["mute"],
            ha="right", va="top", style="italic", linespacing=1.1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 4, 8])
    ax.set_yticks([0, -4, -8])
    ax.set_yticklabels(["0", "−4", "−8"])
    ax.set_xlabel(r"$S_{\mathrm{fav}}$")
    if ylabel:
        ax.set_ylabel(r"$S_{\mathrm{comp}}$", labelpad=0)
    return stats


def handles() -> list:
    """Legend entries in column-major order (matplotlib fills legends column by column)."""
    g = dict(ls="none", color=PAL["mute"])
    blank = (Line2D([0], [0], ls="none"), "")
    return [(Patch(color=PAL["gm"]), "GM"),
            (Line2D([0], [0], marker="o", ms=MS, mec="white", mew=0.4, **g), "corner / new"),
            (Patch(color=PAL["nongm"]), "Non-GM"),
            (Line2D([0], [0], marker="o", ms=MS_OPEN, mfc="white", mec=PAL["mute"], mew=0.9, ls="none"), "cheese / old"),
            (Patch(color=PAL["init"]), "init"),
            (Line2D([0], [0], marker="o", ms=1.4, alpha=0.4, mew=0, **g), "state"),
            (Line2D([0], [0], marker="o", ms=2.4, alpha=0.45, mew=0, **g), "seed"),
            (Line2D([0], [0], marker="D", ms=3.2, mec="white", mew=0.35, **g), "seed mean"),
            (Line2D([0], [0], marker="o", ms=MS, color=PAL["mute"], lw=0.7, mec="white", mew=0.4), "mean ± 95% CI"),
            (Patch(color=PAL["mute"], alpha=0.14, lw=0), "push–pull"),
            (Line2D([0], [0], **REF), r"$r_\star = 1$ (no preferred direction)"), blank]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0, help="jitter seed for the coin seed dots")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "fig_role_panels")
    args = ap.parse_args()

    style()
    plt.rcParams.update({"xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5})
    maze, coin = load_maze(), load_coin()
    res = json.loads(AM_JSON.read_text())["results"]
    xlim, ylim = am_limits(res)

    w, h = W_IN, args.height
    fig = plt.figure(figsize=(w, h))
    # Panel boxes in inches (x0, width); exact placement keeps the page width fixed at 5.5 in.
    y0, yh = 0.63, 1.10
    boxes = [(0.64, 0.98), (1.96, 0.92), (3.22, 1.0), (4.39, 1.0)]
    axes = [fig.add_axes([x / w, y0 / h, bw / w, yh / h]) for x, bw in boxes]

    maze_panel(axes[0], maze)
    coin_panel(axes[1], coin, np.random.default_rng(args.seed))
    am = {e: am_panel(axes[2 + j], res, e, xlim, ylim, j == 0) for j, e in enumerate(("shape", "colour"))}

    y1, y2 = 1.985 / h, 1.865 / h
    title = dict(ha="center", va="top", fontweight="bold")
    mid = lambda i: (boxes[i][0] + boxes[i][1] / 2) / w
    fig.text(mid(0), y1, "Maze: detour and", **title)
    fig.text(mid(0), y2, "new corner", **title)
    fig.text(mid(1), y1, "Coin: where the coin's", **title)
    fig.text(mid(1), y2, "influence goes", **title)
    fig.text((boxes[2][0] + boxes[3][0] + boxes[3][1]) / 2 / w, y1, "Attribute Maze: push–pull", **title)
    for i, e in ((2, "shape"), (3, "colour")):
        fig.text(mid(i), y2, f"Black · {e} swap", fontsize=FS - 1, **title)

    hs = handles()
    fig.legend(*zip(*hs), loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.4, columnspacing=1.2, handletextpad=0.4, labelspacing=0.2, fontsize=FS - 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", args.out.with_suffix(".pdf"))

    vals = {"maze": {f"{g}|{c}": {"goal_side": a.tolist(), "alternative": b.tolist(),
                                  "mean": [a.mean(), b.mean()], "ci95_half": [ci_half(a), ci_half(b)]}
                     for (g, c), (a, b) in maze.items()},
            "coin_r_star": {c: {"per_seed": v.tolist(), "mean": v.mean(), "ci95_half": ci_half(v)} for c, v in coin.items()},
            "am_black": am, "am_limits": [list(xlim), list(ylim)]}
    (TABLES / "role_panels_values.json").write_text(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
