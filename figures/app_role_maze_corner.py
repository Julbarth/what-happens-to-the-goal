"""Appendix role figure: four Maze corner-edit arms with identical GM traces, and LocationReach off-path episodes as a point cloud."""
from __future__ import annotations

import argparse
import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import style as fs  # noqa: E402
from style import PAL  # noqa: E402
from role_maze_lr_panels import CUBE_Y, GOAL, density, episode, grey_floor  # noqa: E402
from role_maze_lr_base import ONPATH, SCALE, frame_axes, grid, titled, truncate  # noqa: E402

from paths import OUT_DIR, TABLES

LABELS = ("1080_gm", "1081_gm", "1082_gm")
THR = 0.03


def cell_px(c, wd):
    cs = 64 / wd * SCALE
    return (c[0] + 0.5) * cs, (wd - 1 - c[1] + 0.5) * cs


def maze(ax, z, m, arm):
    wd = m["wd"]
    cs = 64 / wd * SCALE
    fr = grey_floor(np.kron(np.transpose(z[f"{arm}_frame"], (1, 2, 0)), np.ones((SCALE, SCALE, 1))))
    ax.imshow(fr, interpolation="nearest")
    if arm != "base":
        for c in z[f"{arm}_cells"]:
            x, y = cell_px(c, wd)
            ax.add_patch(mpatches.Rectangle((x - cs / 2, y - cs / 2), cs, cs, fill=False, ec="white", lw=0.45, zorder=2))
    target = {"base": m["old"], "global": m["new"], "offset": m["offset_target"], "blnull": m["old"]}[arm]
    p = np.array([cell_px(c, wd) for c in truncate(z[f"{arm}_traj_{LABELS[0]}"], target)])
    col = PAL["ink"] if arm in ("base", "blnull") else PAL["gm_proxy"]
    ax.plot(p[:, 0], p[:, 1], color="white", lw=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.plot(p[:, 0], p[:, 1], color=col, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    ax.plot(*cell_px(m["agent"], wd), "o", ms=3.4, mfc="white", mec=PAL["ink"], mew=0.6, zorder=8)
    ax.add_patch(mpatches.Circle(cell_px(m["cheese"], wd), 0.75 * cs, fill=False, ec="white", lw=1.8, zorder=5))
    ax.add_patch(mpatches.Circle(cell_px(m["cheese"], wd), 0.75 * cs, fill=False, ec=GOAL, lw=0.9, zorder=5))
    o = cell_px(m["old"], wd)
    ax.plot(*o, "o", ms=7.5, mfc="none", mec="white", mew=1.6, zorder=6)
    ax.plot(*o, "o", ms=7.5, mfc="none", mec=PAL["ink"], mew=0.7, zorder=6)
    if arm in ("global", "offset"):
        ax.plot(*cell_px(target, wd), "*", ms=8.5, mfc=PAL["gm_proxy"], mec="white", mew=0.5, zorder=7)
    if arm == "blnull":
        x, y = cell_px(z["blnull_cells"][0], wd)
        ax.add_patch(mpatches.Rectangle((x - cs / 2, y - cs / 2), cs, cs, fill=False, ec=PAL["gm_proxy"], lw=0.9,
                                        ls=(0, (2, 1.2)), zorder=5))
    # Crop the sand margin below-left of the maze; the opened corner (23, 23) stays in view.
    lo = min((wd - m["size"]) // 2 - 1, int(z["blnull_cells"].min()) - 1)
    ax.set_xlim(lo * cs - 0.5, wd * cs - 0.5)
    ax.set_ylim((wd - lo) * cs - 0.5, -0.5)
    frame_axes(ax)


def checkpoint_agreement(z, m):
    """Per arm: every checkpoint reaches the arm's target, and the three traces up to arrival are identical."""
    out = {}
    for arm in ("base", "global", "offset", "blnull"):
        tr = [z[f"{arm}_traj_{l}"][: m[f"{arm}_{l}"]["hit"] + 1] for l in LABELS]
        out[arm] = {"hit": [m[f"{arm}_{l}"]["hit"] for l in LABELS], "ok": [m[f"{arm}_{l}"]["ok"] for l in LABELS],
                    "identical_traces": all(np.array_equal(tr[0], t) for t in tr[1:])}
    return out


def offpath_points(seeds):
    """Per valid off-path episode and cohort: closest approach to the corner and to the ball over the episode (cm)."""
    rows = []
    for s in seeds:
        z = np.load(ONPATH / f"offpath_paths_s{s}.npz")
        v = z["valid"].astype(bool)
        for coh in ("gm", "control"):
            P = z[coh][v]
            dc = np.linalg.norm(P - z["corner"], axis=-1)
            dg = np.linalg.norm(P - z["goal"][v][:, None], axis=-1)
            hc, hg = (dc < THR).any(1), (dg < THR).any(1)
            tc = np.where(hc, (dc < THR).argmax(1), 99)
            tg = np.where(hg, (dg < THR).argmax(1), 99)
            t_first = np.minimum(tc, tg)
            idx = np.arange(len(P))
            at = np.clip(t_first, 0, P.shape[1] - 1)
            for i in idx:
                rows.append(dict(seed=s, cohort=coh, min_d_corner=100 * dc[i].min(), min_d_ball=100 * dg[i].min(),
                                 t_first=int(t_first[i]), first=("corner" if tc[i] < tg[i] else "ball" if tg[i] < tc[i] else "neither"),
                                 d_other_at_first=100 * (dg[i, at[i]] if tc[i] < tg[i] else dc[i, at[i]])))
    return rows


def cloud(ax, rows):
    """One point per off-path episode (all seeds): closest approach to the corner vs to the ball, cohort means ringed."""
    for coh, name in (("control", "Non-GM"), ("gm", "GM")):
        r = [x for x in rows if x["cohort"] == coh]
        x = np.array([q["min_d_corner"] for q in r])
        y = np.array([q["min_d_ball"] for q in r])
        ax.scatter(x, y, s=2.0, color=fs.COH[name], alpha=0.3, lw=0, rasterized=True, zorder=2)
        ax.plot(x.mean(), y.mean(), "o", ms=6, mfc=fs.COH[name], mec="white", mew=1.0, zorder=4)
    ax.axvline(100 * THR, **fs.REF)
    ax.axhline(100 * THR, **fs.REF)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 36)
    ax.set_aspect("equal")
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([0, 15, 30])
    ax.set_xlabel("closest to corner (cm)", fontsize=fs.FS_TICK, labelpad=1)
    ax.set_ylabel("closest to ball (cm)", fontsize=fs.FS_TICK, labelpad=1)
    fs.finish(ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=1000140, help="maze from maze_shared_state_search.py with identical traces")
    ap.add_argument("--seed", type=int, default=1, help="LocationReach seed of the two example episodes")
    a = ap.parse_args()
    fs.style()
    z = np.load(TABLES / f"maze_search/L{a.level}.npz")
    m = json.loads(str(z["meta"]))
    agree = checkpoint_agreement(z, m)
    zls = [np.load(ONPATH / f"offpath_paths_s{s}.npz") for s in (1, 2, 3)]
    zl = zls[a.seed - 1]
    fig, A = grid(4, 2, 1.15, 0.25, 0.38, 0.0, 0.22, 0.62)
    for ax, arm, L, t in zip(A[0], ("base", "global", "offset", "blnull"), "abcd",
                             ("Unedited", "Corner opened", "Offset extreme", "Bottom-left edit")):
        maze(ax, z, m, arm)
        titled(ax, L, t)
    eps = [int(i) for i in np.where(zl["valid"])[0][:2]]  # first two valid episodes, no selection on outcome
    for ax, i, L, t in zip(A[1][:2], eps, "ef", ("Episode 1", "Episode 2")):
        episode(ax, zl, i)
        titled(ax, L, t)
    n_ep, n_pts = density(A[1][2], zls)
    titled(A[1][2], "g", "All GM paths")
    rows = offpath_points((1, 2, 3))
    p = A[1][3].get_position()
    W, H = fig.get_size_inches()
    A[1][3].set_position([p.x0 + 0.26 / W, p.y0 + 0.2 / H, p.width - 0.26 / W, p.height - 0.2 / H])
    cloud(A[1][3], rows)
    titled(A[1][3], "h", "Closest reach")
    h1 = [Line2D([], [], color=PAL["ink"], lw=1.4, label="GM, unedited"),
          Line2D([], [], color=PAL["gm_proxy"], lw=1.4, label="GM, edited"),
          Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec=PAL["ink"], mew=0.8, label="old corner / LR corner"),
          Line2D([], [], ls="", marker="*", ms=7, mfc=PAL["gm_proxy"], mec="white", label="new extreme"),
          Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=PAL["ink"], mew=0.6, label="start"),
          Line2D([], [], ls="", marker="o", ms=5.5, mfc="none", mec=GOAL, mew=0.9, label="cheese")]
    h2 = [Line2D([], [], color=PAL["gm"], lw=1.4, label="GM"), Line2D([], [], color=PAL["nongm"], lw=1.4, label="Non-GM"),
          Line2D([], [], ls="", marker="o", ms=4.5, color=GOAL, label="ball"),
          Line2D([], [], ls="", marker="s", ms=4.5, color=CUBE_Y, label="distractor cube"),
          Line2D([], [], color=PAL["mute"], lw=0.7, ls=(0, (2, 1.5)), label="ball workspace"),
          Line2D([], [], ls="", marker="o", ms=5, mfc=PAL["mute"], mec="white", label="cohort mean"),
          Line2D([], [], **fs.REF, label="3 cm reach radius")]
    fs.legend_row(fig, h1, [h.get_label() for h in h1], y_in=0.2, handlelength=1.5)
    fs.legend_row(fig, h2, [h.get_label() for h in h2], y_in=0.03, handlelength=1.3, columnspacing=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"app_role/fig_app_role_maze_corner_lr.{ext}", dpi=300)
    plt.close(fig)
    summ = {}
    for coh in ("gm", "control"):
        r = [x for x in rows if x["cohort"] == coh]
        summ[coh] = {"n": len(r), "mean_min_d_corner_cm": float(np.mean([x["min_d_corner"] for x in r])),
                     "mean_min_d_ball_cm": float(np.mean([x["min_d_ball"] for x in r])),
                     "corner_first": float(np.mean([x["first"] == "corner" for x in r])),
                     "ball_first": float(np.mean([x["first"] == "ball" for x in r]))}
    (TABLES / "app_role_maze_corner_values.json").write_text(json.dumps(
        {"maze_level": a.level, "maze_meta": m, "checkpoint_agreement": agree, "lr_episodes": eps, "lr_seed": a.seed,
         "density_episodes": n_ep, "density_points": n_pts, "closest_approach": summ}, indent=1))


if __name__ == "__main__":
    main()
