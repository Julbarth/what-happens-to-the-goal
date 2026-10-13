"""Appendix figure: side-view Coin conflict traces (3 GM + 3 Non-GM seeds per level) in the robot trace-figure style."""
from __future__ import annotations

from paths import OUT_DIR, TABLES
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import robot_traces as R  # noqa: E402

S, PAL, COH, COH_LABEL = R.S, R.PAL, R.COH, R.COH_LABEL
OUTCOME = {"coin": "goal", "end_wall": "proxy", "died": "neither", "timed_out": "neither"}
SEEDS = dict(gm=[6033, 6037, 6051], control=[7001, 7002, 7003])
WALL, LAVA = {15, 16}, {17, 18}


def load_coin(data: Path) -> R.Task:
    """Bank = first 20 levels from 1,000,000 with the coin >= 5 cells left of the end-wall marker (reset state only)."""
    bank = pd.read_csv(data / "coin_bank.csv")
    terr = np.load(data / "coin_bank.npz")
    eps = {}
    for c, ss in SEEDS.items():
        for s in ss:
            st = pd.read_csv(data / f"coin_steps_{c}_s{s}.csv")
            ep = pd.read_csv(data / f"coin_episodes_{c}_s{s}.csv").sort_values("bank_idx")
            eps[(c, s)] = [R.Ep(st[st.bank_idx == k].sort_values("t")[["x", "y"]].to_numpy(), OUTCOME[o], None)
                           for k, o in zip(ep.bank_idx, ep.outcome)]
    coin = bank[["coin_x", "coin_y"]].to_numpy(float) + 0.5
    end = bank[["end_x", "end_y"]].to_numpy(float) + 0.5

    def layout(i, seed=None):
        g = terr[f"terrain_{int(bank.level[i])}"]

        def draw(ax):
            img = np.zeros(g.shape)
            img[np.isin(g, list(WALL))] = 1
            img[np.isin(g, list(LAVA))] = 2
            ax.imshow(img, origin="lower", extent=(0, g.shape[1], 0, g.shape[0]), interpolation="nearest",
                      cmap=ListedColormap(["white", "#D9D9D9", "#F2B8A8"]), vmin=0, vmax=2, zorder=1)
            ax.plot(*coin[i], "o", ms=4.2, color=R.OBJ["yellow"], mec="none", zorder=3)
            ax.axvline(end[i][0] + 0.5, color=PAL["mute"], lw=0.6, ls=(0, (2, 1.5)), zorder=1.5)
        return dict(goal=coin[i], proxy=end[i], draw=draw, grid=g)

    t = R.Task("Coin", "Coin: coin (goal) vs end of level (proxy)", SEEDS, eps, layout,
               [(0, None), (1, None), (2, None)], arena="coin", ring=0.9, scale=5.0)
    t.bank = bank
    return t


def coin_window(task: R.Task, i: int, seeds: dict):
    """Crop to the walkable part of the level: x up to just past the end wall, y up to the highest trace/object."""
    L = task.layout(i)
    xmax = L["proxy"][0] + 2.5
    ys = [L["goal"][1], L["proxy"][1]] + [task.eps[(c, s)][i].path[:, 1].max() for c in seeds for s in seeds[c]]
    return (0, xmax), (0, max(ys) + 2.5)


def draw_coin_example(ax, task: R.Task, i: int, seeds: dict, label: str, window, lw=R.TRACE_LW):
    rows = R.draw_traces(ax, task, i, seeds, lw=lw)
    ax.set_xlim(*window[0])
    ax.set_ylim(*window[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.annotate(label, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 1.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=S.FS_SUB, color=PAL["ink"])
    return rows


def top_left_bar(ax, length: float, label: str):
    """Coin levels have open sky at the top left, so the bar sits there instead of over the floor."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xa, ya = x0 + 0.04 * (x1 - x0) + 1.0, y1 - 0.06 * (y1 - y0)
    ax.plot([xa, xa + length], [ya, ya], color=PAL["mute"], lw=0.8, solid_capstyle="butt", zorder=2)
    ax.annotate(label, xy=(xa + length, ya), xytext=(1.5, 0), textcoords="offset points", ha="left",
                va="center", fontsize=S.FS_TICK - 1.5, color=PAL["mute"])


def shared_window(task: R.Task, idx, seeds: dict):
    """One window for all shown levels so every panel has the same tiles-per-inch."""
    w = [coin_window(task, i, seeds) for i in idx]
    return (0, max(x[0][1] for x in w)), (0, max(x[1][1] for x in w))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=TABLES / "coin")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "app_traces")
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    S.style()
    task = load_coin(a.data_dir)
    ep = R.endpoints(task)
    R.rates(task).to_csv(TABLES / "coin_conflict_rates.csv", index=False)
    ep.to_csv(TABLES / "coin_conflict_endpoints.csv", index=False)

    n = a.n_examples
    assert n == 3, "2 x 2 layout: three examples + the all-levels panel"
    win = shared_window(task, range(n), task.seeds)
    fig = plt.figure(figsize=(S.W_IN, 3.7))
    gs = fig.add_gridspec(2, 2, left=0.01, right=0.985, top=0.92, bottom=0.10, wspace=0.10, hspace=0.22)
    rows = []
    cells = [gs[0, 0], gs[0, 1], gs[1, 0]]
    for k in range(n):
        ax = fig.add_subplot(cells[k])
        rows += draw_coin_example(ax, task, k, task.seeds, f"start state {k + 1}", win)
        top_left_bar(ax, task.scale, f"{task.scale:.0f} tiles")
        if R.disagree(task, k, None):
            ax.annotate("*", xy=(1, 1), xycoords="axes fraction", xytext=(-2, -1), textcoords="offset points",
                        ha="right", va="top", fontsize=S.FS_TITLE + 1, fontweight="bold", color=PAL["gm"])
    axp = fig.add_subplot(gs[1, 1])
    R.draw_population(axp, task, ep)
    fig.text(0.5, 0.995, task.title, ha="center", va="top", fontsize=S.FS_TITLE, color=PAL["ink"])
    R.legend(fig, False)
    out = a.out_dir / "fig_app_traces_Coin"
    fig.savefig(out.with_suffix(".png"), dpi=a.dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    pd.DataFrame(rows).assign(level=lambda d: task.bank.level.to_numpy()[d.scene]).to_csv(
        TABLES / "coin_shown_examples.csv", index=False)


if __name__ == "__main__":
    main()
