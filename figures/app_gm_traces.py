"""Compose the GM vs Non-GM same-level OOD trace figure from rollout_trace.py outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

GM, NONGM = "#7A1520", "#3D8B8B"
STROKE = [pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()]


def cell_px(cells, dim, size):
    s = size / dim
    c = np.asarray(cells, dtype=float)
    return np.stack([(c[:, 0] + 0.5) * s, size - (c[:, 1] + 0.5) * s], 1)


def draw_grid(ax, d, color, crop_cells=None, extra_end=None, ring=None):
    img = d["hires0"]
    size, dim = img.shape[0], int(json.loads(str(d["meta"]))["grid_dim"])
    ax.imshow(img, interpolation="nearest")
    cells = [tuple(c) for c in d["cells"].tolist()]
    if extra_end is not None:
        cells.append(tuple(extra_end))
    # Consecutive duplicates (NOOPs, wall bumps, corner dwelling) carry no path information.
    path = [cells[0]] + [c for p, c in zip(cells, cells[1:]) if c != p]
    xy = cell_px(path, dim, size)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.4, solid_capstyle="round", path_effects=STROKE, zorder=3)
    ax.scatter(*xy[0], s=28, facecolor="white", edgecolor=color, lw=1.6, zorder=4)
    ax.scatter(*xy[-1], s=22, marker="s", facecolor=color, edgecolor="white", lw=0.8, zorder=4)
    if ring is not None:
        r = cell_px([ring], dim, size)[0]
        ax.scatter(*r, s=150, facecolor="none", edgecolor="#E8B400", lw=1.8, zorder=5)
    if crop_cells is not None:
        lo, hi = crop_cells
        s = size / dim
        ax.set_xlim(lo * s, (hi + 1) * s)
        ax.set_ylim(size - lo * s, size - (hi + 1) * s)
    ax.set_xticks([]), ax.set_yticks([])


def draw_coin(ax, d, color, xlim, ylim):
    p = d["positions"]
    coin = d["coin"]
    end_x = float(json.loads(str(d["meta"]))["end_x"])
    ax.plot(p[:, 0], p[:, 1], color=color, lw=2.2, solid_capstyle="round")
    ax.scatter(*p[0], s=28, facecolor="white", edgecolor=color, lw=1.6, zorder=4)
    ax.scatter(*p[-1], s=36, marker="s", facecolor=color, edgecolor="white", lw=1.0, zorder=4)
    ax.scatter(*coin, s=150, marker="*", facecolor="#E8B400", edgecolor="#2A2A2A", lw=0.8, zorder=5)
    ax.axvline(end_x, color="#5A5A5A", ls="--", lw=1.0)
    ax.set_xlim(*xlim), ax.set_ylim(*ylim)
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#8E8E8E")
    ins = inset_axes(ax, width="32%", height="32%", loc="lower right", borderpad=0.4)
    ins.imshow(d["frames"][-1], interpolation="nearest")
    ins.set_xticks([]), ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(color), sp.set_linewidth(1.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="directory with {maze,coin,attr}_{gm,nongm}.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t = Path(args.traces)
    L = {k: np.load(t / f"{k}.npz") for k in
         ["maze_gm", "maze_nongm", "coin_gm", "coin_nongm", "attr_gm", "attr_nongm"]}

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.5), gridspec_kw=dict(wspace=0.06, hspace=0.06))
    maze_cells = np.concatenate([L["maze_gm"]["cells"], L["maze_nongm"]["cells"]])
    crop = (int(maze_cells.min()), int(maze_cells.max()))
    goal_maze = json.loads(str(L["maze_nongm"]["meta"]))["targets"]["goal"]
    draw_grid(axes[0, 0], L["maze_gm"], GM, crop, ring=goal_maze)
    draw_grid(axes[1, 0], L["maze_nongm"], NONGM, crop, extra_end=goal_maze, ring=goal_maze)

    pos = np.concatenate([L["coin_gm"]["positions"], L["coin_nongm"]["positions"]])
    xlim = (pos[:, 0].min() - 0.8, float(json.loads(str(L["coin_gm"]["meta"]))["end_x"]) + 1.0)
    ylim = (pos[:, 1].min() - 1.0, pos[:, 1].max() + 1.2)
    draw_coin(axes[0, 1], L["coin_gm"], GM, xlim, ylim)
    draw_coin(axes[1, 1], L["coin_nongm"], NONGM, xlim, ylim)
    for ax in axes[:, 1]:
        ax.set_aspect("equal", adjustable="box")

    for k, color in (("attr_gm", GM), ("attr_nongm", NONGM)):
        tg = json.loads(str(L[k]["meta"]))["targets"]
        reached = tg["proxy"] if json.loads(str(L[k]["meta"]))["reward"] < 0 else tg["goal"]
        draw_grid(axes[0 if k == "attr_gm" else 1, 2], L[k], color, extra_end=reached)

    for ax, title in zip(axes[0], ["Maze", "Coin", "Attribute Maze"]):
        ax.set_title(title, fontsize=11)
    for ax, lab in zip(axes[:, 0], ["GM", "Non-GM"]):
        ax.set_ylabel(lab, fontsize=11, color=GM if lab == "GM" else NONGM)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
