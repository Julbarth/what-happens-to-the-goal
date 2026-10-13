"""Figure variants for the PushToMarker evaluation configurations: real camera frames, a text-free top-down schematic, and a hybrid."""
from __future__ import annotations

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES
import argparse
import json
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import style as fs  # noqa: E402
from style import PAL  # noqa: E402

GR = ROBOTICS_ROOT / "results/analysis"
GOAL, PROXY, GHOST = "#1b7837", "#c2410c", "#8a8a8a"
DASH = (0, (2.2, 1.4))
DISC_GREEN = (0.10, 0.75, 0.15)
CUBE = "#3A5FCD"
CENTRE = np.array([-0.08, 0.0])
MARKER_R = 0.035                 # push_to_marker.py marker_radius
CUBE_HALF = 0.02                 # push_to_marker.py cube_half
OFF_DEG, SWEEP_CM = 75.0, (3, 5, 7, 9, 11)
TABLE = "#EFE3D3"
TITLES_ROLL = ("Drawn", "Hidden", "Conflict", "Decoupled")
TITLES_FRAME = ("Moved", "Disc on path", "Disc off path", "Hover")


def T(xy):
    """World (x, y) to the camera orientation: image right = +y, image up = -x."""
    xy = np.asarray(xy, float)
    return np.stack([xy[..., 1], -xy[..., 0]], -1)


def axes_grid(ncol, nrow, side, gx, gy, top, bottom):
    w = fs.W_IN
    h = top + nrow * side + (nrow - 1) * gy + bottom
    fig = plt.figure(figsize=(w, h))
    x0 = (w - ncol * side - (ncol - 1) * gx) / 2
    return fig, [[fig.add_axes([(x0 + c * (side + gx)) / w, (bottom + (nrow - 1 - r) * (side + gy)) / h, side / w,
                                side / h]) for c in range(ncol)] for r in range(nrow)]


def titled(ax, letter, title):
    ax.set_title(rf"$\bf{{({letter})}}$ {title}", loc="left", fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color(PAL["mute"])


def ring(ax, xy, r, colour, style, lw=1.1, z=6):
    ls = {"solid": "-", "dash": DASH, "dot": (0, (1, 1.3))}[style]
    ax.add_patch(mp.Circle(xy, r, fill=False, ec="white", lw=lw + 1.1, zorder=z))
    ax.add_patch(mp.Circle(xy, r, fill=False, ec=colour, lw=lw, ls=ls, zorder=z + 1))


def sweep_xy(cube_xy, d, side=1):
    """Disc positions of the off-path test, rotated OFF_DEG from the cube-to-centre ray, as in the sweep evaluation."""
    ray = CENTRE - cube_xy
    ray = ray / np.linalg.norm(ray)
    th = np.deg2rad(OFF_DEG) * side
    c, s = np.cos(th), np.sin(th)
    return cube_xy + d * np.array([c * ray[0] - s * ray[1], s * ray[0] + c * ray[1]])


# ---------------------------------------------------------------- schematic
def table(ax, ctr, view):
    ax.set_facecolor(TABLE)
    ax.set_xlim(ctr[0] - view, ctr[0] + view)
    ax.set_ylim(ctr[1] - view, ctr[1] + view)
    ax.set_aspect("equal")
    bare(ax)
    ring(ax, T(CENTRE), MARKER_R, PROXY, "dash")
    ax.plot(*T(CENTRE), "+", color=PROXY, ms=3.5, mew=0.8, zorder=7)


def cube(ax, xy):
    ax.add_patch(mp.Rectangle((xy[0] - CUBE_HALF, xy[1] - CUBE_HALF), 2 * CUBE_HALF, 2 * CUBE_HALF, color=CUBE, zorder=5))


def marker(ax, xy, kind):
    """kind: goal (white marker), hidden (not drawn), green (control disc)."""
    if kind == "goal":
        ax.add_patch(mp.Circle(xy, MARKER_R, color="white", ec=PAL["mute"], lw=0.3, zorder=3))
        ring(ax, xy, MARKER_R, GOAL, "solid")
    elif kind == "hidden":
        ring(ax, xy, MARKER_R, GHOST, "dot", lw=0.8)
    else:
        ax.add_patch(mp.Circle(xy, MARKER_R, color=DISC_GREEN, ec="white", lw=0.6, zorder=3))


def paths(ax, stem, cond, i):
    """Cube paths of all three seeds per cohort (native policies); the scene is one where all three agree."""
    for coh, name in (("control", "Non-GM"), ("gm", "GM")):
        for s in (0, 1, 2):
            p = T(np.load(GR / f"rewire_ptm/paths/{coh}_s{s}_native_{stem}_{cond}.npz")["cube"][i])
            ax.plot(*p.T, color=fs.COH[name], lw=0.9, alpha=0.85, zorder=4, solid_capstyle="round")


def gripper(ax, xy):
    ax.plot(*xy, marker="x", ms=6, mew=1.2, color="white", zorder=9)
    ax.plot(*xy, marker="x", ms=6, mew=0.7, color=PAL["ink"], zorder=10)


def path_points(stem, conds, i):
    return np.concatenate([T(np.load(GR / f"rewire_ptm/paths/{coh}_s{s}_native_{stem}_{cond}.npz")["cube"][i])
                           for coh in ("control", "gm") for s in (0, 1, 2) for cond in conds])


def fit_view(pts, pad):
    """Centre and half-width of the square that holds every point, so no panel shows mostly empty table."""
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    return (lo + hi) / 2, (hi - lo).max() / 2


def schematic_rollouts(row, ctrl, conf, i, j):
    m, c = T(ctrl["m"][i]), T(ctrl["cube"][i, :2])
    cw = T(conf["cube"][j, :2])
    pts_a = np.vstack([path_points("ctrlset", ("standard", "nomarker"), i), m, c, T(CENTRE)])
    pts_c = np.vstack([path_points("conflict", ("standard", "decoupled"), j), T(conf["m"][j]), T(conf["m2"][j]), cw,
                       T(CENTRE)])
    (ca, va), (cc, vc) = fit_view(pts_a, MARKER_R + 0.012), fit_view(pts_c, MARKER_R + 0.012)
    # One half-width for the whole row keeps a single scale bar valid for all four panels.
    view = max(va, vc)
    for ax, cond, kind in ((row[0], "standard", "goal"), (row[1], "nomarker", "hidden")):
        table(ax, ca, view)
        marker(ax, m, kind)
        cube(ax, c)
        paths(ax, "ctrlset", cond, i)
    for ax, cond, mk in ((row[2], "standard", conf["m"][j]), (row[3], "decoupled", conf["m2"][j])):
        table(ax, cc, view)
        marker(ax, T(mk), "goal")
        cube(ax, cw)
        paths(ax, "conflict", cond, j)
    scale(row[0], view, 5)


def schematic_frames(row, conf, j):
    cw, m, m2 = conf["cube"][j, :2], T(conf["m"][j]), T(conf["m2"][j])
    c = T(cw)
    pts = np.array([c, T(CENTRE), m, m2, T(sweep_xy(cw, SWEEP_CM[-1] / 100))])
    ctr = (pts.min(0) + pts.max(0)) / 2
    view = (pts.max(0) - pts.min(0)).max() / 2 + MARKER_R + 0.01
    for ax in row:
        table(ax, ctr, view)
        ax.plot(*np.c_[c, T(CENTRE)], color=PAL["mute"], lw=0.5, ls=(0, (1, 1.5)), zorder=2)
    marker(row[0], m, "hidden")
    marker(row[0], m2, "goal")
    row[0].annotate("", xy=m2, xytext=m, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=PAL["ink"],
                                                         shrinkA=8, shrinkB=11), zorder=8)
    marker(row[1], m, "green")
    # Ends of the 3 to 11 cm sweep; the intermediate discs would overlap at this scale.
    for d in (SWEEP_CM[0], SWEEP_CM[-1]):
        marker(row[2], T(sweep_xy(cw, d / 100)), "green")
    marker(row[3], m, "goal")
    for ax in row:
        cube(ax, c)
    gripper(row[3], c)
    scale(row[0], view, 5)


def scale(ax, view, cm):
    x0, x1 = ax.get_xlim()
    y0, _ = ax.get_ylim()
    # Bottom right: the example layouts keep that corner of the table empty, so the bar never sits on a marker.
    x, y = x1 - 0.07 * (x1 - x0) - cm / 100, y0 + 0.07 * (x1 - x0)
    ax.plot([x, x + cm / 100], [y, y], color=PAL["ink"], lw=1.0, solid_capstyle="butt", zorder=9)
    ax.text(x + cm / 200, y + 0.03 * (x1 - x0), f"{cm} cm", ha="center", va="bottom", fontsize=fs.FS_TICK - 1)


def legend_schematic(fig, y_in, paths_too=True, gripper_too=True):
    h = []
    if paths_too:
        h += [Line2D([], [], color=PAL["gm"], lw=1.3, label="GM cube path"),
              Line2D([], [], color=PAL["nongm"], lw=1.3, label="Non-GM cube path")]
    h += [Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=GOAL, mew=1.1, label="marker (goal)"),
          Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=GHOST, mew=0.8, label="marker hidden"),
          Line2D([], [], ls="", marker="o", ms=6, mfc=DISC_GREEN, mec="white", label="green disc"),
          Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=PROXY, mew=1.1, label="centre (proxy)"),
          Line2D([], [], ls="", marker="s", ms=4, color=CUBE, label="cube"),
          Line2D([], [], ls="", marker="x", ms=4.5, mew=0.9, color=PAL["ink"], label="gripper")][:None if gripper_too else -1]
    k = 4
    fs.legend_row(fig, h[:k], [x.get_label() for x in h[:k]], y_in=y_in + 0.17, handlelength=1.4)
    fs.legend_row(fig, h[k:], [x.get_label() for x in h[k:]], y_in=y_in, handlelength=1.4)


# ---------------------------------------------------------------- frames
def project(K, E, xyz):
    pc = E[:3, :3] @ xyz + E[:3, 3]
    uv = K @ pc
    return uv[:2] / uv[2]


def frame_panel(ax, F, key, crop):
    # The decoupled conflict draws only the moved marker, which is the moved-marker frame without the old ring.
    img = F[f"rgb_{'moved' if key == 'dec' else key}"]
    ax.imshow(img, interpolation="lanczos")
    (u0, u1), (v0, v1) = crop
    ax.set_xlim(u0, u1)
    ax.set_ylim(v1, v0)
    bare(ax)


def px_radius(K, E, xy, r):
    a = project(K, E, np.r_[xy, 0.0])
    b = project(K, E, np.r_[xy[0], xy[1] + r, 0.0])
    return np.linalg.norm(b - a)


def frames_overlay(ax, F, key, conf, ctrl, i, j):
    """Minimal rings: centre dashed orange, marker solid green, hidden marker dotted grey."""
    K, E = F["K"], F["extrinsic_cv"]
    pc = project(K, E, np.r_[CENTRE, 0.0])
    r = px_radius(K, E, CENTRE, MARKER_R) * 1.25
    ring(ax, pc, r, PROXY, "dash", lw=1.0)
    src = ctrl if key.startswith("ctrl") else conf
    k = i if key.startswith("ctrl") else j
    if key in ("ctrl_hidden", "conf_hidden", "hover_hidden"):
        ring(ax, project(K, E, np.r_[src["m"][k], 0.0]), r, GHOST, "dot", lw=0.9)
    elif key in ("ctrl_drawn", "conf", "hover"):
        ring(ax, project(K, E, np.r_[src["m"][k], 0.0]), r, GOAL, "solid", lw=1.0)
    elif key == "dec":
        ring(ax, project(K, E, np.r_[conf["m2"][j], 0.0]), r, GOAL, "solid", lw=1.0)
    elif key == "moved":
        a, b = project(K, E, np.r_[conf["m"][j], 0.0]), project(K, E, np.r_[conf["m2"][j], 0.0])
        ring(ax, a, r, GHOST, "dot", lw=0.9)
        ring(ax, b, r, GOAL, "solid", lw=1.0)
        for col, lw in (("white", 2.0), (PAL["ink"], 0.9)):
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=lw, color=col, shrinkA=6,
                                                            shrinkB=11), zorder=9)


def crop_box(F, conf, ctrl, i, j, pad=46):
    """Square crop around the conflict scene only: the single-frame tests never show the control layout."""
    K, E = F["K"], F["extrinsic_cv"]
    cw = conf["cube"][j, :2]
    pts = [CENTRE, conf["m"][j], conf["m2"][j], cw] + [sweep_xy(cw, d / 100) for d in (SWEEP_CM[0], SWEEP_CM[-1])]
    uv = np.array([project(K, E, np.r_[p, 0.0]) for p in pts])
    lo, hi = uv.min(0) - pad, uv.max(0) + pad
    c, s = (lo + hi) / 2, (hi - lo).max() / 2
    return (c[0] - s, c[0] + s), (c[1] - s, c[1] + s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctrlset-index", type=int, default=3,
                    help="first layout in the analysed stratum with all 3 seeds per cohort agreeing and cube and marker"
                         " >= 10 cm from the centre (index 1 qualifies too but is too cramped to read)")
    ap.add_argument("--conflict-index", type=int, default=2,
                    help="first conflict scene on which all 3 seeds per cohort agree (standard and decoupled)")
    ap.add_argument("--frames", type=Path, default=TABLES / "ptm_frames")
    a = ap.parse_args()
    fs.style()
    ctrl = np.load(GR / "rewire_ptm/ctrlset_scenes.npz")
    conf = np.load(GR / "rewire_ptm/conflict_scenes.npz")
    i, j = a.ctrlset_index, a.conflict_index
    side, gx, gy = 1.2, 0.12, 0.34

    # (B) schematic
    fig, (r0, r1) = axes_grid(4, 2, side, gx, gy, 0.22, 0.5)
    schematic_rollouts(r0, ctrl, conf, i, j)
    schematic_frames(r1, conf, j)
    for ax, L, t in zip(r0 + r1, "abcdefgh", TITLES_ROLL + TITLES_FRAME):
        titled(ax, L, t)
    legend_schematic(fig, 0.03)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"app_role/variant_ptm_configs_B_schematic.{ext}", dpi=300)
    plt.close(fig)

    have = (a.frames / "ptm_white.npz").exists() and (a.frames / "ptm_green.npz").exists()
    if have:
        W, G = np.load(a.frames / "ptm_white.npz"), np.load(a.frames / "ptm_green.npz")
        F = {k: W[k] for k in W.files} | {k: G[k] for k in G.files if k.startswith("rgb_")}
        box = crop_box(F, conf, ctrl, i, j)
        keys_roll = ("ctrl_drawn", "ctrl_hidden", "conf", "dec")
        keys_frame = ("moved", "conf_green", "mlat_green", "hover")
        # (A) frames only: rollout set frames show the start of each rollout set, single frames the edited state.
        fig, (q0, q1) = axes_grid(4, 2, side, gx, gy, 0.22, 0.42)
        for ax, key, L, t in zip(q0 + q1, keys_roll + keys_frame, "abcdefgh",
                                 TITLES_ROLL + TITLES_FRAME):
            frame_panel(ax, F, key, box)
            frames_overlay(ax, F, key, conf, ctrl, i, j)
            titled(ax, L, t)
        legend_schematic(fig, 0.03, paths_too=False, gripper_too=False)
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"app_role/variant_ptm_configs_A_frames.{ext}", dpi=300)
        plt.close(fig)
        # (C) hybrid: schematic rollouts with paths on top, real frames for the single-frame tests below.
        fig, (h0, h1) = axes_grid(4, 2, side, gx, gy, 0.22, 0.5)
        schematic_rollouts(h0, ctrl, conf, i, j)
        for ax, key in zip(h1, keys_frame):
            frame_panel(ax, F, key, box)
            frames_overlay(ax, F, key, conf, ctrl, i, j)
        for ax, L, t in zip(h0 + h1, "abcdefgh", TITLES_ROLL + TITLES_FRAME):
            titled(ax, L, t)
        legend_schematic(fig, 0.03, gripper_too=False)
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"app_role/variant_ptm_configs_C_hybrid.{ext}", dpi=300)
        plt.close(fig)
    cw = conf["cube"][j, :2]
    ang = np.degrees(np.arccos(np.dot(conf["m2"][j] - cw, CENTRE - cw) /
                               (np.linalg.norm(conf["m2"][j] - cw) * np.linalg.norm(CENTRE - cw))))
    (TABLES / "ptm_configs_values.json").write_text(json.dumps({
        "ctrlset_index": i, "conflict_index": j, "frames_rendered": bool(have),
        "conflict_marker_to_cube_cm": float(100 * np.linalg.norm(conf["m"][j] - cw)),
        "cube_to_centre_cm": float(100 * np.linalg.norm(CENTRE - cw)),
        "marker_move_cm": float(100 * np.linalg.norm(conf["m2"][j] - conf["m"][j])),
        "decoupled_angle_deg": float(ang)}, indent=1))


if __name__ == "__main__":
    main()
