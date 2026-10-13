"""Maze corner-edit arms, LocationReach off-path episodes, path density and first-reach counts for the role appendix."""
from __future__ import annotations

import argparse
import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

import style as fs  # noqa: E402
from style import PAL  # noqa: E402
from role_maze_lr_base import CKPT, LEVEL, ONPATH, SCALE, TRACES, cell_px, frame_axes, grid, titled, truncate  # noqa: E402

from paths import OUT_DIR, TABLES

GOAL, PROXY = "#1b7837", "#c2410c"
CUBE_Y = "#D4B000"
WS = ((-0.20, -0.25), (0.20, 0.25))   # ball sampling workspace, gm_robotics/envs/location_reach.py WS_LO/WS_HI
LIM = 0.28
def grey_floor(fr):
    """Corridor pixels redrawn as dark grey, as in the grey-floor levels; this level's pale floor has
    little contrast to the sand. Floor = pixels near the median non-sand colour, so cheese and agent keep their colours."""
    f = fr.reshape(-1, 3)
    vals, counts = np.unique((f * 255 // 16).astype(int), axis=0, return_counts=True)
    sand = vals[counts.argmax()] * 16 / 255
    non_sand = np.abs(f - sand).sum(1) > 0.25
    med = np.median(f[non_sand], 0)
    floor = non_sand & (np.abs(f - med).sum(1) < 0.5)
    lum = f[floor].mean(1)
    out = f.copy()
    out[floor] = (0.32 + 0.6 * (lum - lum.mean()))[:, None]
    return np.clip(out, 0, 1).reshape(fr.shape)


def maze(ax, z, arm):
    """High-contrast look: ink trace for the unedited maze, red for edits, ringed old corner, starred new extreme."""
    k = f"L{LEVEL}"
    m = json.loads(str(z[f"{k}_meta"]))
    wd = m["wd"]
    cs = 64 / wd * SCALE
    fr = grey_floor(np.kron(np.transpose(z[f"{k}_{arm}_frame"], (1, 2, 0)), np.ones((SCALE, SCALE, 1))))
    ax.imshow(fr, interpolation="nearest")
    if arm != "base":
        for c in z[f"{k}_{arm}_cells"]:
            x, y = cell_px(c, wd)
            ax.add_patch(mpatches.Rectangle((x - cs / 2, y - cs / 2), cs, cs, fill=False, ec="white", lw=0.45, zorder=2))
    target = {"base": m["old"], "global": m["new"], "offset": m["offset_target"], "blnull": m["old"]}[arm]
    p = np.array([cell_px(c, wd) for c in truncate(z[f"{k}_{arm}_traj"], target)])
    col = PAL["ink"] if arm in ("base", "blnull") else PAL["gm_proxy"]
    ax.plot(p[:, 0], p[:, 1], color="white", lw=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.plot(p[:, 0], p[:, 1], color=col, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    ax.plot(*cell_px(m["agent"], wd), "o", ms=3.4, mfc="white", mec=PAL["ink"], mew=0.6, zorder=8)
    o = cell_px(m["old"], wd)
    ax.plot(*o, "o", ms=7.5, mfc="none", mec="white", mew=1.6, zorder=6)
    ax.plot(*o, "o", ms=7.5, mfc="none", mec=PAL["ink"], mew=0.7, zorder=6)
    if arm in ("global", "offset"):
        ax.plot(*cell_px(target, wd), "*", ms=8.5, mfc=PAL["gm_proxy"], mec="white", mew=0.5, zorder=7)
    if arm == "blnull":
        x, y = cell_px(z[f"{k}_blnull_cells"][0], wd)
        ax.add_patch(mpatches.Rectangle((x - cs / 2, y - cs / 2), cs, cs, fill=False, ec=PAL["gm_proxy"], lw=0.9,
                                        ls=(0, (2, 1.2)), zorder=5))
    ax.set_xlim(-0.5, fr.shape[1] - 0.5)
    ax.set_ylim(fr.shape[0] - 0.5, -0.5)
    frame_axes(ax)


def table(ax, corner):
    ax.add_patch(mpatches.Rectangle(WS[0], WS[1][0] - WS[0][0], WS[1][1] - WS[0][1], fill=False, ec=PAL["mute"], lw=0.6,
                                    ls=(0, (2, 1.5)), zorder=1))
    ax.plot(*corner[:2], "o", ms=7.5, mfc="none", mec="white", mew=1.6, zorder=6)
    ax.plot(*corner[:2], "o", ms=7.5, mfc="none", mec=PAL["ink"], mew=0.7, zorder=6)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    frame_axes(ax)


def episode(ax, zl, i):
    table(ax, zl["corner"])
    ax.add_patch(plt.Rectangle(zl["dxy"][i] - 0.015, 0.03, 0.03, color=CUBE_Y, alpha=0.85, lw=0))
    ax.plot(*zl["control"][i, :, :2].T, color=fs.COH["Non-GM"], lw=1.2, zorder=3)
    ax.plot(*zl["gm"][i, :, :2].T, color=fs.COH["GM"], lw=1.2, zorder=4)
    ax.plot(*zl["goal"][i, :2], "o", ms=5, color=GOAL, mec="white", mew=0.6, zorder=5)
    ax.plot(*zl["gm"][i, 0, :2], "o", color="white", mec=PAL["ink"], ms=3.4, mew=0.6, zorder=8)


def density(ax, zls):
    """GM gripper positions at every step of every valid off-path episode of all seeds, log-scaled 2-D histogram."""
    pts = np.concatenate([z["gm"][z["valid"].astype(bool), :, :2].reshape(-1, 2) for z in zls])
    table(ax, zls[0]["corner"])
    e = np.linspace(-LIM, LIM, 57)
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[e, e])
    cmap = LinearSegmentedColormap.from_list("gm", ["#F6E9E9", "#D9A3A3", PAL["gm"], "#3A0A0F"])
    ax.imshow(np.where(H.T > 0, H.T, np.nan), origin="lower", extent=(-LIM, LIM, -LIM, LIM), cmap=cmap,
              norm=LogNorm(vmin=1, vmax=H.max()), interpolation="nearest", zorder=0)
    return int(sum(z["valid"].sum() for z in zls)), int(len(pts))


def first_reached(ax):
    """Stacked counts per seed: corner first, ball first, neither (500 valid episodes per seed and cohort)."""
    rows = []
    x = 0
    ticks, labels = [], []
    for coh, name in (("gm", "GM"), ("control", "Non-GM")):
        for s in (1, 2, 3):
            d = pd.read_csv(ONPATH / f"offpath_s{s}.csv")
            d = d[(d.valid == 1) & (d.cohort == coh)]
            n = {o: int((d.outcome == o).sum()) for o in ("corner", "goal", "neither")}
            rows.append(dict(cohort=name, seed=s, **n))
            b = 0
            for o, col in (("corner", PROXY), ("goal", GOAL), ("neither", "#BBBBBB")):
                ax.bar(x, n[o], bottom=b, width=0.75, color=col, lw=0)
                b += n[o]
            ticks.append(x)
            labels.append(str(s))
            x += 1
        ax.text(x - 2, -0.13, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fs.FS_TICK,
                color=fs.COH[name])
        x += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=fs.FS_TICK - 0.5)
    ax.set_ylim(0, 500)
    ax.set_yticks([0, 250, 500])
    ax.tick_params(axis="y", pad=0.5)
    fs.finish(ax)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1, help="LocationReach seed of the two example episodes")
    a = ap.parse_args()
    fs.style()
    z = np.load(TRACES)
    zls = [np.load(ONPATH / f"offpath_paths_s{s}.npz") for s in (1, 2, 3)]
    zl = zls[a.seed - 1]
    fig, A = grid(4, 2, 1.15, 0.25, 0.38, 0.0, 0.22, 0.62)
    for ax, arm, L, t in zip(A[0], ("base", "global", "offset", "blnull"), "abcd",
                             ("Unedited", "Corner opened", "Offset extreme", "Bottom-left edit")):
        maze(ax, z, arm)
        titled(ax, L, t)
    eps = [int(i) for i in np.where(zl["valid"])[0][:2]]  # first two valid episodes, no selection on outcome
    for ax, i, L, t in zip(A[1][:2], eps, "ef", ("Episode 1", "Episode 2")):
        episode(ax, zl, i)
        titled(ax, L, t)
    n_ep, n_pts = density(A[1][2], zls)
    titled(A[1][2], "g", "All GM paths")
    rows = first_reached(A[1][3])
    titled(A[1][3], "h", "Reached first")
    h1 = [Line2D([], [], color=PAL["ink"], lw=1.4, label="GM, unedited"),
          Line2D([], [], color=PAL["gm_proxy"], lw=1.4, label="GM, edited"),
          Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec=PAL["ink"], mew=0.8, label="old corner / LR corner"),
          Line2D([], [], ls="", marker="*", ms=7, mfc=PAL["gm_proxy"], mec="white", label="new extreme"),
          Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=PAL["ink"], mew=0.6, label="start")]
    h2 = [Line2D([], [], color=PAL["gm"], lw=1.4, label="GM"), Line2D([], [], color=PAL["nongm"], lw=1.4, label="Non-GM"),
          Line2D([], [], ls="", marker="o", ms=4.5, color=GOAL, label="ball"),
          Line2D([], [], ls="", marker="s", ms=4.5, color=CUBE_Y, label="distractor cube"),
          Line2D([], [], color=PAL["mute"], lw=0.7, ls=(0, (2, 1.5)), label="ball workspace"),
          mpatches.Patch(color=PROXY, label="corner first"), mpatches.Patch(color=GOAL, label="ball first")]
    fs.legend_row(fig, h1, [h.get_label() for h in h1], y_in=0.2, handlelength=1.5)
    fs.legend_row(fig, h2, [h.get_label() for h in h2], y_in=0.03, handlelength=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"app_role/fig_role_maze_lr_panels.{ext}", dpi=300)
    plt.close(fig)
    (TABLES / "role_maze_lr_panels_values.json").write_text(json.dumps(
        {"maze_level": LEVEL, "t": 8, "checkpoint_shown": CKPT, "floor": "corridor pixels redrawn grey", "lr_episodes": eps,
         "lr_seed": a.seed, "density_episodes": n_ep, "density_points": n_pts, "first_reached": rows}, indent=1))


if __name__ == "__main__":
    main()
