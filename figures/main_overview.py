"""Figure 1: where each GM regime loses the intended goal along network depth, and which interventions recover it."""
import argparse
from pathlib import Path

import matplotlib

from paths import OUT_DIR, ROBOTICS_ROOT
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import overview_base as d  # noqa: E402

GOAL, PROXY, INK, MUTE, BLOCK = "#2f8a8a", "#9b2233", "#222222", "#8a8a8a", "#dfe3e6"
BG = ("#eef3f3", "#f4f1f1", "#f3f3f3")
FS = 7
ROW = 0.6
TH = 0.5  # thumbnail side
VS = 0.82  # vertical scale of network blocks and labels relative to overview_base
ENVS = ["maze", "ptm", "coin", "attr"]  # shallowest to deepest goal; also keeps both target substitutions adjacent
Y0, TOP_PAD = 0.72, 0.85
W, H = 7.4, Y0 + ROW * (len(ENVS) - 1) + TOP_PAD
X_REC0, X_REC1 = 5.05, 7.2
YC = {e: Y0 + ROW * (len(ENVS) - 1 - i) for i, e in enumerate(ENVS)}
ROLE = {"maze": "Target substitution", "ptm": "Target substitution", "coin": "Policy shortcut",
        "attr": "Cue competition"}
ENV = {"maze": "Maze", "ptm": "PushToMarker", "coin": "Coin", "attr": "Attribute Maze"}
OUTCOME = {"maze": "top-right\ncorner", "ptm": "table\ncentre", "coin": "move\nright", "attr": "yellow\ncue"}
X_IN, X_PRE, X_FC, X_HEAD, X_OUT = 1.2, 1.8, 2.4, 3.0, 3.74
HW = {"pre": 0.045, "fc": 0.04, "head": 0.04}
LANE = 0.055
# Depth band edges. Each band must contain the mark that ends the goal's path: Maze's cross before B1 (lost before
# the policy), PushToMarker's and Coin's after pi but before B2 (read, not used), Attribute Maze's arrow past B2.
B0, B1, B2, B3 = X_PRE - 0.28, X_FC + 0.2, X_HEAD + 0.44, X_OUT + 0.38
_BARS_RIGHT = X_HEAD + HW["head"] + 0.02 + 0.16
CROSS_X = {2: (X_FC + HW["fc"] + B1) / 2, 3: (_BARS_RIGHT + B2) / 2}  # keyed by the blocked segment index
# Goal segments input->pre, pre->fc, fc->head, head->outcome as (lw, alpha, dashed), None = blocked here.
# Widths are a schematic grading: PushToMarker reaches pi more thinly than Coin.
GOAL_SEGS = {"maze": [(1.1, 0.45, True), (1.1, 0.45, True), None, None],
             "ptm": [(2.4, 1, False), (1.0, 1, False), (0.45, 1, False), None],
             "coin": [(2.4, 1, False), (1.2, 1, False), (0.8, 1, False), None],
             "attr": [(1.4, 1, False), (1.4, 1, False), (1.4, 1, False), (1.0, 1, False)]}
# GM goal reach (native, after, control rewire). Sources:
#  steer = corrective beta = 1 at h_fc, tab:steering-local (Attribute Maze black background).
#  rewire = new readout from h_pre, tab:rewire-main. Maze and Attribute Maze use the oracle-target rows, as
#    fig:rewire-main does; Coin uses the teacher-target row. PushToMarker is the h_pre DAgger rewire after three
#    rounds, tab:maze-dagger, from data/robotics/results/analysis/rewire_ptm_dagger/summary.csv
#    (site pre, mode dagger, round 3, set conflict: gm 0.661, control 0.816); its native is tab:rewire-main.
REC = {"maze": {"steer": (0.001, 0.002, None), "rewire": (0.219, 0.116, 0.308)},
       "ptm": {"steer": (0.000, 0.000, None), "rewire": (0.000, 0.661, 0.816)},
       "coin": {"steer": (0.018, 0.016, None), "rewire": (0.316, 0.843, 0.888)},
       "attr": {"steer": (0.160, 0.953, None), "rewire": (0.358, 0.871, 0.908)}}
NATIVE_NONGM = {"maze": 0.347, "ptm": 0.889, "coin": 0.896, "attr": 0.696}
FLOOR = {"maze": 0.226, "ptm": 0.135, "coin": 0.646, "attr": 0.840}  # higher of random trunk and chance, same rows
# Steering has no control bar, so it counts as recovered only where its CI clears baseline and random (bold in
# tab:steering-local). Rewire verdicts are those of tab:rewire-main / tab:maze-dagger, re-derived in check_ok.
OK = {"maze": {"steer": "none", "rewire": "none"}, "ptm": {"steer": "none", "rewire": "partial"},
      "coin": {"steer": "none", "rewire": "full"}, "attr": {"steer": "full", "rewire": "full"}}
REC_FRAC = 0.9  # recovered threshold of the Baselines paragraph

PTM_DIR = (ROBOTICS_ROOT / "results/analysis/rewire_ptm")
PTM_CENTRE = np.array([-0.08, 0.0])  # push_to_marker.CENTRE, the GM training marker position (proxy)


def check_ok():
    """Rewire verdict by the appendix rule: recovered at >= REC_FRAC of min(native Non-GM, Non-GM rewire), partial when
    clearly above the floor. The CI part of the partial rule is taken from the tables, so this checks only the means."""
    for e in ENVS:
        nat, aft, ctl = REC[e]["rewire"]
        ref = min(NATIVE_NONGM[e], ctl)
        verdict = "full" if aft >= REC_FRAC * ref else "partial" if aft > FLOOR[e] else "none"
        assert OK[e]["rewire"] == verdict, (e, verdict)


def ptm_frame(idx):
    """Agent camera frame (128 x 128) of one conflict scene, plus the marker and centre in pixel coordinates.

    The world-to-pixel map is affine, fitted on marker detections in the start frames themselves (top-down camera).
    """
    rgb = np.load(PTM_DIR / "eval_start_frames.npz")["rgb"]
    sc = np.load(PTM_DIR / "conflict_scenes.npz")
    m = sc["m"][: len(rgb)]
    yy, xx = np.mgrid[:128, :128]
    P, M = [], []
    for i in range(len(rgb)):
        g = m[i] @ [[0, 160], [170, 0]] + [63, 69]  # rough hand map used only to window the detection
        f = rgb[i].astype(int)
        w = (f.min(-1) > 225) & (np.ptp(f, -1) < 20) & ((xx - g[0]) ** 2 + (yy - g[1]) ** 2 < 144)
        if 15 <= w.sum() <= 60:
            P.append([xx[w].mean(), yy[w].mean()])
            M.append(m[i])
    A = np.linalg.lstsq(np.c_[np.array(M), np.ones(len(M))], np.array(P), rcond=None)[0]
    to_px = lambda p: np.r_[p, 1] @ A  # noqa: E731
    return rgb[idx], to_px(m[idx]), to_px(PTM_CENTRE)


def ptm_thumb(ax, cx, cy, size, idx, crop):
    """Crop of the agent view around marker and centre, so the two rings do not touch at thumbnail size."""
    img, mk, ce = ptm_frame(idx)
    x0p, y0p = crop[0], crop[1]
    s = crop[2]
    sub = img[y0p:y0p + s, x0p:x0p + s]
    x0, y0 = cx - size / 2, cy - size / 2
    ax.imshow(sub, extent=(x0, x0 + size, y0, y0 + size), zorder=2, interpolation="nearest")
    ax.add_patch(Rectangle((x0, y0), size, size, fc="none", ec=d.PAL["ink"], lw=0.5, zorder=3))
    u = lambda p: (x0 + (p[0] - x0p + 0.5) / s * size, y0 + (1 - (p[1] - y0p + 0.5) / s) * size)  # noqa: E731
    r = 0.085 * size
    ax.add_patch(Circle(u(mk), r, fc="none", ec="white", lw=2.0, zorder=4))
    ax.add_patch(Circle(u(mk), r, fc="none", ec=d.PAL["goal_ring"], lw=1.1, zorder=4))
    # The lighter proxy red of the Procgen thumbnails vanishes on the orange table, so the figure's proxy red is used.
    ax.add_patch(Circle(u(ce), r, fc="none", ec=PROXY, lw=1.4, ls=(0, (1.8, 0.9)), zorder=4))


def arr(ax, p, q, color, lw, alpha=1.0, dashed=False, z=5):
    """Straight arrow with one fixed head geometry; a dashed shaft keeps a solid head."""
    head = 0.07
    if dashed:
        ax.plot([p[0], q[0] - head], [p[1], q[1]], color=color, lw=lw, alpha=alpha, ls=(0, (2.2, 1.6)), zorder=z,
                solid_capstyle="butt")
        p = (q[0] - head - 0.005, q[1])
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=0.28,head_width=0.14", mutation_scale=9 + 1.2 * lw,
                                 color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0, zorder=z,
                                 capstyle="butt", joinstyle="miter"))


def block(ax, x, y, hw, hh, fc=BLOCK):
    ax.add_patch(Rectangle((x - hw, y - hh), 2 * hw, 2 * hh, fc=fc, ec="#9aa3a8", lw=0.5, zorder=3))


def cross(ax, x, y, s=0.045, color=PROXY):
    for sx in (1, -1):
        ax.plot([x - s, x + s], [y - sx * s, y + sx * s], color=color, lw=1.4, zorder=7, solid_capstyle="round")


def network(ax, env):
    """One row of the inside panel. The cross sits inside the depth band where the goal's path ends."""
    y = YC[env]
    for k in range(3):
        block(ax, X_IN - 0.06 + 0.03 * k, y - 0.03 * VS + 0.03 * VS * k, 0.13, 0.16 * VS, fc="#e9ecee")
    in_edge = X_IN - 0.06 + 0.06 + 0.13
    block(ax, X_PRE, y, HW["pre"], 0.25 * VS)
    block(ax, X_FC, y, HW["fc"], 0.16 * VS)
    block(ax, X_HEAD, y, HW["head"], 0.16 * VS)
    bars_right = X_HEAD + HW["head"] + 0.02 + 0.16
    for k, b in enumerate([0.10, 0.16, 0.05, 0.03, 0.07]):
        ax.add_patch(Rectangle((X_HEAD + HW["head"] + 0.02, y + (0.11 - 0.055 * k - 0.018) * VS), b, 0.036 * VS, fc="#555555",
                               lw=0, zorder=3))
    starts = [in_edge, X_PRE + HW["pre"], X_FC + HW["fc"], bars_right + 0.04]
    ends = [X_PRE - HW["pre"], X_FC - HW["fc"], X_HEAD - HW["head"], X_OUT - 0.26]
    gy = y + LANE if env == "attr" else y
    for i, seg in enumerate(GOAL_SEGS[env]):
        if seg is None:
            cross(ax, CROSS_X[i], gy)
            break
        lw, al, ds = seg
        arr(ax, (starts[i], gy), (ends[i], gy), GOAL, lw, al, ds)
    if env == "attr":
        for i in range(4):
            arr(ax, (starts[i], y - LANE), (ends[i], y - LANE), PROXY, 2.6)
    ax.text(X_OUT, y, OUTCOME[env], ha="center", va="center", fontsize=FS, color=PROXY, fontweight="bold",
            zorder=6, linespacing=1.0)


def header(fig, ax, env):
    """Role in bold, then the environment in muted type, as one visual label."""
    x, y = X_IN - 0.2, YC[env] + 0.245
    t = ax.text(x, y, ROLE[env], ha="left", va="bottom", fontsize=FS, fontweight="bold", color=INK)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    x1 = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x1, y, f" · {ENV[env]}", ha="left", va="bottom", fontsize=FS, color=MUTE)


def recovery(fig):
    for env, yc in YC.items():
        ax = fig.add_axes([X_REC0 / W, (yc - 0.19) / H, (X_REC1 - X_REC0) / W, 0.43 / H])
        for i, lab in enumerate(("steer", "rewire")):
            nat, aft, ctl = REC[env][lab]
            ok = OK[env][lab]
            ax.bar(i - 0.18, nat, 0.32, color="#c9c9c9", lw=0)
            ax.bar(i + 0.18, aft, 0.32, color=GOAL if ok == "full" else "white", ec=GOAL, lw=0.9,
                   hatch="//////" if ok == "partial" else None)
            if ctl is not None:
                ax.plot([i + 0.0, i + 0.36], [ctl, ctl], color=INK, lw=1.1, ls=(0, (2, 1)))
            # Above the control line where there is one, so the label never sits on the dashes.
            ax.text(i + 0.18, max(aft, ctl or 0) + 0.04, f"{aft:.2f}".lstrip("0") if aft >= 0.005 else "0", ha="center",
                    va="bottom", fontsize=FS - 2.5, color=GOAL if ok != "none" else MUTE, zorder=6)
        ax.set_xticks([0, 1], ["steer", "rewire"], fontsize=FS - 0.5)
        ax.set_ylim(0, 1.15)
        ax.set_xlim(-0.55, 1.55)
        ax.set_yticks([0, 0.5, 1], ["0", "", "1"], fontsize=FS - 1)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.5)
        ax.tick_params(length=2, width=0.5, pad=1)
        ax.set_ylabel("goal reach", fontsize=FS - 1, labelpad=1)


def legend(ax):
    y = 0.09
    x = 1.25
    for c, lw, al, ds, lab in ((GOAL, 1.4, 1, False, "true goal"), (PROXY, 2.6, 1, False, "proxy"),
                               (GOAL, 1.1, 0.45, True, "not found")):
        arr(ax, (x, y), (x + 0.24, y), c, lw, al, ds)
        ax.text(x + 0.29, y, lab, fontsize=FS - 0.5, va="center", color=INK)
        x += 0.95
    cross(ax, x + 0.05, y, s=0.035)
    ax.text(x + 0.13, y, "not used", fontsize=FS - 0.5, va="center", color=INK)
    rows = (((("#c9c9c9", "#c9c9c9", None), "native"), ((GOAL, GOAL, None), "recovered"),
             (("white", GOAL, "//////"), "partly")),
            ((("white", GOAL, None), "not recovered"), (None, "control rewire")))
    for r, row in enumerate(rows):
        x = X_REC0 + 0.15
        cols = iter((0.88, 0.72))
        yy = H - 0.25 - 0.14 * r
        for spec, lab in row:
            if spec is None:
                ax.plot([x - 0.04, x + 0.1], [yy, yy], color=INK, lw=1.1, ls=(0, (2, 1)))
            else:
                ax.add_patch(Rectangle((x - 0.03, yy - 0.035), 0.1, 0.07, fc=spec[0], ec=spec[1], lw=0.8,
                                       hatch=spec[2]))
            ax.text(x + 0.12, yy, lab, fontsize=FS - 0.5, va="center", color=INK)
            x += next(cols, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "fig1_overview")
    ap.add_argument("--ptm-scene", type=int, default=29,
                    help="index into rewire_ptm/eval_start_frames.npz (conflict set, standard condition)")
    ap.add_argument("--ptm-crop", type=int, nargs=3, default=[17, 21, 72], metavar=("X0", "Y0", "SIDE"),
                    help="pixel crop of the 128 x 128 agent view")
    args = ap.parse_args()
    check_ok()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    thumbs = d.load_thumbs()
    top = H - 0.06
    zb, zt = 0.27, YC[ENVS[0]] + 0.36
    for (x0, x1, lab), fc in zip(((B0, B1, "represented"), (B1, B2, "read by the policy"), (B2, B3, "goal role")), BG):
        ax.add_patch(FancyBboxPatch((x0 + 0.01, zb), x1 - x0 - 0.02, zt - zb, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=fc, ec="none", zorder=0))
        ax.text((x0 + x1) / 2, zb + 0.03, lab, ha="center", va="bottom", fontsize=FS - 1, color=MUTE, style="italic")
    ax.text(0.45, top, "Outside", ha="center", va="top", fontsize=FS + 1, fontweight="bold", color=INK)
    ax.text(0.45, top - 0.17, "all fail alike", ha="center", va="top", fontsize=FS - 0.5, color=MUTE)
    ax.text((X_IN + X_OUT) / 2, top, "Inside: how far does the goal get?", ha="center", va="top", fontsize=FS + 1,
            fontweight="bold", color=INK)
    ax.text((X_REC0 + X_REC1) / 2, top, "Recover", ha="center", va="top", fontsize=FS + 1, fontweight="bold", color=INK)
    for x, s in ((X_IN, "input"), (X_PRE, r"$h_{\mathrm{pre}}$"), (X_FC, r"$h_{\mathrm{fc}}$"), (X_HEAD + 0.1, r"$\pi$")):
        ax.text(x, top - 0.2, s, ha="center", va="top", fontsize=FS + 0.5, color=INK)
    ax.plot([4.5, 4.5], [zb, top - 0.05], color="#cccccc", lw=0.6)
    for env, y in YC.items():
        if env == "ptm":
            ptm_thumb(ax, 0.45, y, TH, args.ptm_scene, args.ptm_crop)
        else:
            d.thumb(ax, thumbs[env], 0.45, y, TH, env)
        header(fig, ax, env)
        network(ax, env)
    recovery(fig)
    legend(ax)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
