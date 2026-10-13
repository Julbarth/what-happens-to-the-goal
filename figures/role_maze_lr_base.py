"""Shared helpers and candidate layouts for the Maze corner-edit / LocationReach off-path appendix figure."""
from __future__ import annotations

import argparse
import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style as fs  # noqa: E402
from style import PAL  # noqa: E402

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES

ONPATH = (ROBOTICS_ROOT / "results/onpath")
CKPT = "1081_gm"   # all three GM checkpoints give identical traces on every arm of this state
TRACES = TABLES / f"maze_traces_L1000048_t8_{CKPT}.npz"
GOAL, PROXY = "#1b7837", "#c2410c"   # ring convention of the bank-examples figures
DASH = (0, (2.2, 1.4))
LEVEL = 1000048   # t = 8: shared eligible state, all GM checkpoints agree on every arm, both main traces >= 8 steps
SCALE = 8


def cell_px(c, wd):
    cs = 64 / wd * SCALE
    return (c[0] + 0.5) * cs, (wd - 1 - c[1] + 0.5) * cs


def frame_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color(PAL["mute"])


def ring(ax, xy, r, colour, dashed, lw=1.1, z=6):
    ax.add_patch(mpatches.Circle(xy, r, fill=False, ec="white", lw=lw + 1.0, zorder=z))
    ax.add_patch(mpatches.Circle(xy, r, fill=False, ec=colour, lw=lw, ls=DASH if dashed else "-", zorder=z + 1))


def truncate(traj, cell):
    """Trace up to the first arrival at the arm's target cell; the 100-step rollout keeps moving after it."""
    hit = [i for i, c in enumerate(traj.tolist()) if list(c) == list(cell)]
    return traj[: hit[0] + 1] if hit else traj


def maze_panel(ax, z, arm):
    """arm in base / global / offset / blnull: frame, carved cells, GM greedy trace, cheese (goal) and corner rings."""
    k = f"L{LEVEL}"
    m = json.loads(str(z[f"{k}_meta"]))
    wd = m["wd"]
    cs = 64 / wd * SCALE
    fr = np.kron(np.transpose(z[f"{k}_{arm}_frame"], (1, 2, 0)), np.ones((SCALE, SCALE, 1)))
    ax.imshow(fr, interpolation="nearest")
    if arm != "base":
        for c in z[f"{k}_{arm}_cells"]:
            x, y = cell_px(c, wd)
            ax.add_patch(mpatches.Rectangle((x - cs / 2, y - cs / 2), cs, cs, fill=False, ec="white", lw=0.4, zorder=2))
    traj = truncate(z[f"{k}_{arm}_traj"], {"base": m["old"], "global": m["new"], "offset": m.get("offset_target"),
                                           "blnull": m["old"]}[arm])
    p = np.array([cell_px(c, wd) for c in traj])
    ax.plot(p[:, 0], p[:, 1], color="white", lw=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.plot(p[:, 0], p[:, 1], color=PAL["gm"], lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    ax.plot(*cell_px(m["agent"], wd), "o", ms=3.4, mfc="white", mec=PAL["ink"], mew=0.6, zorder=8)
    ring(ax, cell_px(m["cheese"], wd), 1.15 * cs, GOAL, False)
    target = {"base": m["old"], "global": m["new"], "offset": m.get("offset_target"), "blnull": m["old"]}[arm]
    ring(ax, cell_px(target, wd), 1.0 * cs, PROXY, True)
    if arm in ("global", "offset"):
        ring(ax, cell_px(m["old"], wd), 1.0 * cs, PAL["mute"], True, lw=0.8)
    if arm == "blnull":
        tip = z[f"{k}_blnull_cells"][0]
        x, y = cell_px(tip, wd)
        ax.add_patch(mpatches.Rectangle((x - cs / 2, y - cs / 2), cs, cs, fill=False, ec=PAL["mute"], lw=0.8, ls=DASH,
                                        zorder=5))
    ax.set_xlim(-0.5, fr.shape[1] - 0.5)
    ax.set_ylim(fr.shape[0] - 0.5, -0.5)
    frame_axes(ax)
    return m


LR_LIM = 0.27


def table(ax, corner):
    ring(ax, corner[:2], 0.025, PROXY, True)
    ax.set_xlim(-LR_LIM, LR_LIM)
    ax.set_ylim(-LR_LIM, LR_LIM)
    ax.set_aspect("equal")
    ax.add_patch(mpatches.Rectangle((-0.2, -0.25), 0.4, 0.5, fill=False, ec=PAL["mute"], lw=0.4, ls=(0, (1, 1.5))))
    frame_axes(ax)


def lr_episode(ax, z, i):
    table(ax, z["corner"])
    ax.add_patch(plt.Rectangle(z["dxy"][i] - 0.015, 0.03, 0.03, color="#D4B000", alpha=0.8, lw=0))
    ax.plot(*z["control"][i, :, :2].T, color=fs.COH["Non-GM"], lw=1.2)
    ax.plot(*z["gm"][i, :, :2].T, color=fs.COH["GM"], lw=1.2)
    ax.plot(*z["goal"][i, :2], "o", color=GOAL, ms=4.0, zorder=4)
    ring(ax, z["goal"][i, :2], 0.025, GOAL, False)
    ax.plot(*z["gm"][i, 0, :2], "o", color="white", mec=PAL["ink"], ms=3.4, mew=0.6, zorder=8)


def lr_overlay(ax, z, n=60):
    table(ax, z["corner"])
    idx = np.where(z["valid"])[0][:n]  # first n valid episodes in order, no selection on outcome
    for i in idx:
        ax.plot(*z["gm"][i, :, :2].T, color=fs.COH["GM"], lw=0.4, alpha=0.35)
    ax.plot(*z["goal"][idx, :2].T, "o", color=GOAL, ms=1.6, alpha=0.8, lw=0)
    return [int(i) for i in idx]


def lr_first(ax):
    """Fraction of off-path episodes reaching corner or ball first, per seed (s1-s3)."""
    vals = {}
    for k, (coh, name) in enumerate((("gm", "GM"), ("control", "Non-GM"))):
        col = fs.COH[name]
        per = []
        for s in (1, 2, 3):
            d = pd.read_csv(ONPATH / f"offpath_s{s}.csv")
            d = d[(d.valid == 1) & (d.cohort == coh)]
            per.append(((d.outcome == "corner").mean(), (d.outcome == "goal").mean()))
        per = np.array(per)
        vals[name] = {"corner_first": per[:, 0].tolist(), "ball_first": per[:, 1].tolist()}
        x0 = 1.5 * k
        for cf, bf in per:
            ax.plot([x0, x0 + 0.7], [cf, bf], color=col, lw=0.7, alpha=0.6)
            ax.plot(x0, cf, "o", ms=3.4, mfc="white", mec=col, mew=0.9)
            ax.plot(x0 + 0.7, bf, "o", ms=3.4, color=col)
        ax.text(x0 + 0.35, -0.3, name, ha="center", va="top", color=col, fontsize=fs.FS_TICK, clip_on=False,
                transform=ax.get_xaxis_transform())
    ax.set_xticks([0, 0.7, 1.5, 2.2])
    ax.set_xticklabels(["corner", "ball", "corner", "ball"], fontsize=fs.FS_TICK - 0.5)
    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", ".5", "1"])
    
    fs.finish(ax)
    return vals


def grid(ncol, nrow, side, gap_x, gap_y, left, top, bottom, row_label_w=0.0):
    """Figure with nrow x ncol square axes of `side` inches."""
    w = fs.W_IN
    h = top + nrow * side + (nrow - 1) * gap_y + bottom
    fig = plt.figure(figsize=(w, h))
    used = row_label_w + ncol * side + (ncol - 1) * gap_x
    x0 = row_label_w + (w - used) / 2 + left
    axes = [[fig.add_axes([(x0 + c * (side + gap_x)) / w, (bottom + (nrow - 1 - r) * (side + gap_y)) / h, side / w, side / h])
             for c in range(ncol)] for r in range(nrow)]
    return fig, axes


def titled(ax, letter, title):
    """Bold letter and short title on one left-aligned line, so narrow panels never collide."""
    ax.set_title(rf"$\bf{{({letter})}}$ {title}", loc="left", fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)


def row_label(fig, ax, text):
    b = ax.get_position()
    fig.text(b.x0 - 0.012, b.y0 + b.height / 2, text, rotation=90, ha="right", va="center", fontsize=fs.FS_TITLE)


def legend(fig, with_nongm=True):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=PAL["gm"], lw=1.4, label="GM"),
         Line2D([], [], color=PAL["nongm"], lw=1.4, label="Non-GM"),
         Line2D([], [], ls="", marker="o", ms=5.5, mfc="none", mec=GOAL, mew=1.0, label="goal (cheese, ball)"),
         Line2D([], [], ls="", marker="o", ms=5.5, mfc="none", mec=PROXY, mew=1.0, label="proxy place (corner)"),
         Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=PAL["ink"], mew=0.6, label="start")]
    if not with_nongm:
        h.pop(1)
    fs.legend_row(fig, h, [x.get_label() for x in h], y_in=0.03, handlelength=1.6)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"app_role/{stem}.{ext}", dpi=300)
    plt.close(fig)


MAZE_T = {"base": "Unedited", "global": "Corner opened", "offset": "Offset extreme",
          "blnull": "Bottom-left corridor"}


def variant_a(z, zl):
    """One environment per row, 3 square panels each."""
    fig, A = grid(3, 2, 1.45, 0.22, 0.42, 0.0, 0.24, 0.34, row_label_w=0.22)
    for ax, arm, L in zip(A[0], ("base", "global", "offset"), "abc"):
        maze_panel(ax, z, arm)
        titled(ax, L, MAZE_T[arm])
    eps = [int(i) for i in np.where(zl["valid"])[0][:1]]
    lr_episode(A[1][0], zl, eps[0])
    titled(A[1][0], "d", "Off-path episode")
    lr_overlay(A[1][1], zl)
    titled(A[1][1], "e", "60 GM paths")
    lr_first(A[1][2])
    titled(A[1][2], "f", "Reached first")
    row_label(fig, A[0][0], "Maze")
    row_label(fig, A[1][0], "LocationReach")
    legend(fig)
    save(fig, "variant_role_maze_lr_a_rows")


def variant_b(z, zl):
    """2 x 4 equal squares, with the bottom-left corridor control as the optional 4th Maze panel."""
    fig, A = grid(4, 2, 1.18, 0.2, 0.36, 0.0, 0.22, 0.34)
    for ax, arm, L in zip(A[0], ("base", "global", "offset", "blnull"), "abcd"):
        maze_panel(ax, z, arm)
        titled(ax, L, MAZE_T[arm])
    eps = [int(i) for i in np.where(zl["valid"])[0][:2]]
    lr_episode(A[1][0], zl, eps[0])
    titled(A[1][0], "e", "Off-path episode 1")
    lr_episode(A[1][1], zl, eps[1])
    titled(A[1][1], "f", "Off-path episode 2")
    lr_overlay(A[1][2], zl)
    titled(A[1][2], "g", "60 GM paths")
    lr_first(A[1][3])
    titled(A[1][3], "h", "Reached first")
    legend(fig)
    save(fig, "variant_role_maze_lr_b_grid2x4")


def variant_c(z, zl):
    """Compact: one row, two Maze and two LocationReach squares; numbers move to the caption."""
    fig, A = grid(4, 1, 1.18, 0.2, 0.0, 0.0, 0.22, 0.34)
    A = A[0]
    maze_panel(A[0], z, "base")
    titled(A[0], "a", "Maze, unedited")
    maze_panel(A[1], z, "global")
    titled(A[1], "b", "Maze, corner opened")
    eps = [int(i) for i in np.where(zl["valid"])[0][:1]]
    lr_episode(A[2], zl, eps[0])
    titled(A[2], "c", "LR, off-path episode")
    lr_overlay(A[3], zl)
    titled(A[3], "d", "LR, 60 GM paths")
    legend(fig)
    save(fig, "variant_role_maze_lr_c_compact")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1, help="LocationReach seed for the example panels")
    a = ap.parse_args()
    fs.style()
    z = np.load(TRACES)
    zl = np.load(ONPATH / f"offpath_paths_s{a.seed}.npz")
    variant_a(z, zl)
    variant_b(z, zl)
    variant_c(z, zl)
    m = json.loads(str(z[f"L{LEVEL}_meta"]))
    vals = {"maze_level": LEVEL, "maze_meta": m,
            "checkpoint_shown": CKPT,
            "maze_steps_to_target": {arm: int(len(truncate(z[f"L{LEVEL}_{arm}_traj"], t)) - 1) for arm, t in
                                     (("base", m["old"]), ("global", m["new"]), ("offset", m["offset_target"]),
                                      ("blnull", m["old"]))},
            "lr_seed": a.seed, "lr_episodes": [int(i) for i in np.where(zl["valid"])[0][:2]]}
    (TABLES / "role_maze_lr_base_values.json").write_text(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
