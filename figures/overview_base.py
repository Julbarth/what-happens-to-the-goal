"""Figure 1 base layouts (A, B): where each GM regime loses the intended goal along depth, and which interventions recover it."""

import argparse
from pathlib import Path

import matplotlib

from paths import DATA_ROOT, OUT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from PIL import Image


# ---------------------------------------------------------------------------
# Every number on the figure, with the paper table or appendix section it comes from.
# ---------------------------------------------------------------------------
N = {
    # -- h_pre station: linear probe availability (app:probes) --------------
    # Maze counterfactual cheese presence AUC at h_pre, GM vs Non-GM.
    "maze_probe_auc_gm": 0.544,
    "maze_probe_auc_nongm": 0.997,
    # Coin residualized bearing, absolute held-out R^2 at h_pre.
    "coin_probe_r2_gm": 0.523,
    "coin_probe_r2_nongm": 0.527,
    # Attribute Maze (black) training-induced Delta AUC at block3, the site that
    # feeds h_pre = vec(ReLU(h3)). Absolute AUCs are not in the appendix.
    "attr_probe_line_role_dauc_gm": 0.186,
    "attr_probe_line_role_dauc_nongm": 0.126,
    "attr_probe_yellow_role_dauc_gm": 0.279,
    # -- h_fc station: world-edit Fisher sensitivity Q (tab:fisher-decomposition)
    "maze_q_hfc_gm": 0.001,  # l.3135
    "maze_q_hfc_nongm": 1.689,  # l.3141
    "coin_q_hfc_gm": 0.021,  # l.3177
    "coin_q_hfc_nongm": 0.339,  # l.3183
    "attr_q_hfc_shape_gm": 1.097,  # l.3239
    "attr_q_hfc_shape_nongm": 3.569,  # l.3245
    "attr_q_hfc_colour_gm": 2.126,  # l.3259
    # Within-state Fisher cosine between shape and colour swaps, GM black, h_fc
    # (tab:attribute-within-state-cue-geometry; -0.942 at h_pre).
    "attr_cosF_hfc": -0.919,
    # -- steering at h_fc, beta = 1, GM intended-goal reach (tab:steering-local)
    "maze_steer_base": 0.000,  # l.4666
    "maze_steer_corr": 0.000,  # 0 of 48 states
    "maze_steer_n": 48,  # l.4605
    "coin_steer_base": 0.362,  # l.4671
    "coin_steer_corr": 0.382,  # l.4671
    "attr_steer_base": 0.160,  # black
    "attr_steer_corr": 0.953,  # black
    # -- rewiring, GM reach and sampled floor, teacher targets (tab:rewire-main)
    "maze_rw_pre_gm": 0.089, "maze_rw_pre_floor": 0.244, "maze_rw_pre_nongm": 0.277,  # l.5301-5310
    "maze_rw_fc_gm": 0.048, "maze_rw_fc_floor": 0.242, "maze_rw_fc_nongm": 0.222,  # l.5301-5310
    "coin_rw_pre_gm": 0.843, "coin_rw_pre_floor": 0.626, "coin_rw_pre_nongm": 0.888,  # l.5288-5297
    "coin_rw_fc_gm": 0.666, "coin_rw_fc_floor": 0.630, "coin_rw_fc_nongm": 0.881,  # l.5288-5297
    "attr_rw_pre_gm": 0.689, "attr_rw_pre_floor": 0.451, "attr_rw_pre_nongm": 0.684,  # black
    "attr_rw_fc_gm": 0.273, "attr_rw_fc_floor": 0.420, "attr_rw_fc_nongm": 0.653,  # black
}

# A rewire counts as recovery when GM recovers at least half of the Non-GM
# rewire's gain over the sampled floor. Coin h_fc (0.14) and all Maze/Attr-h_fc
# arms fall far below this; Coin/Attr h_pre (0.83, 1.02) far above it.
RECOVERY_THRESHOLD = 0.5
# A steering arm counts as recovery when corrective reach gains > 0.2 over baseline.
STEER_THRESHOLD = 0.2


def band_widths():
    """GM signal relative to Non-GM at each depth station [obs, h_pre, h_fc, pi].

    obs: identical observations for both cohorts (=1 by construction).
    h_pre: probe availability ratio (AUC above chance, R^2, or Delta AUC).
    h_fc: world-edit Fisher Q ratio; pi reuses h_fc because the policy head
    reads h_fc directly, so Q at h_fc is the policy's own sensitivity.
    """
    maze_pre = (N["maze_probe_auc_gm"] - 0.5) / (N["maze_probe_auc_nongm"] - 0.5)
    coin_pre = N["coin_probe_r2_gm"] / N["coin_probe_r2_nongm"]
    attr_pre = N["attr_probe_line_role_dauc_gm"] / N["attr_probe_line_role_dauc_nongm"]
    attr_pre_proxy = N["attr_probe_yellow_role_dauc_gm"] / N["attr_probe_line_role_dauc_nongm"]
    maze_fc = N["maze_q_hfc_gm"] / N["maze_q_hfc_nongm"]
    coin_fc = N["coin_q_hfc_gm"] / N["coin_q_hfc_nongm"]
    attr_fc = N["attr_q_hfc_shape_gm"] / N["attr_q_hfc_shape_nongm"]
    attr_fc_proxy = N["attr_q_hfc_colour_gm"] / N["attr_q_hfc_shape_nongm"]
    clip = lambda v: float(np.clip(v, 0.0, 1.0))  # noqa: E731
    return {
        "maze": [1.0, clip(maze_pre), clip(maze_fc), clip(maze_fc)],
        "coin": [1.0, clip(coin_pre), clip(coin_fc), clip(coin_fc)],
        "attr": [1.0, clip(attr_pre), clip(attr_fc), clip(attr_fc)],
        "attr_proxy": [1.0, clip(attr_pre_proxy), clip(attr_fc_proxy), clip(attr_fc_proxy)],
    }


def recovered_fraction(env, site):
    gm, fl, ng = (N[f"{env}_rw_{site}_{k}"] for k in ("gm", "floor", "nongm"))
    return (gm - fl) / (ng - fl) if ng - fl > 1e-9 else float("nan")


def interventions(env):
    """[(kind, success, label)] for steering at h_fc, rewire from h_pre, rewire from h_fc."""
    b, c = N[f"{env}_steer_base"], N[f"{env}_steer_corr"]
    if env == "maze":
        steer_lbl = f"{round(c * N['maze_steer_n'])}/{N['maze_steer_n']}"
    else:
        steer_lbl = f"{b:.2f}→{c:.2f}"
    out = [("steer", c - b > STEER_THRESHOLD, steer_lbl)]
    for site in ("pre", "fc"):
        frac = recovered_fraction(env, site)
        gm, fl = N[f"{env}_rw_{site}_gm"], N[f"{env}_rw_{site}_floor"]
        ok = bool(np.isfinite(frac) and frac >= RECOVERY_THRESHOLD)
        out.append((f"rw_{site}", ok, f"{gm:.2f}", f"{fl:.2f}"))
    return out


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PAL = {
    "gm_goal": "#7A1520", "gm_proxy": "#C45C58", "nongm": "#3D8B8B",
    "init": "#8E8E8E", "ink": "#2A2A2A", "mute": "#6E6E6E", "paper": "#FFFFFF",
    "goal_ring": "#E69F00",
}
FS = 7.5
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS, "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "axes.linewidth": 0.6,
})

REGIMES = [
    ("maze", "Functional target\nsubstitution", "e.g., Maze", "maze.png"),
    ("coin", "Policy\nshortcut", "e.g., Coin", "coin.png"),
    ("attr", "Cue\ncompetition", "e.g., Attribute Maze", "attr.png"),
]
STATIONS = ["obs", r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$", r"$\pi$"]
SEGMENTS = ["representation", "readout", "decision"]


def load_thumbs():
    """Crops of existing renders (paths and boxes are fixed; see comments)."""
    t = {}
    # GM first action at the fork (burgundy arrow) and cheese (gold ring), fig_24 top-left panel.
    t["maze"] = Image.open(DATA_ROOT / "figure_inputs/fig_24_maze_cheese_edit_panels.png").convert("RGB").crop((86, 86, 376, 376))
    # Coin relocated one step left of the agent, t=0 frame of the GM strip; GM then runs right for 200 steps.
    t["coin"] = Image.open(DATA_ROOT / "figure_inputs/fig_22_coin_envmove_paths.png").convert("RGB").crop((40, 64, 256, 280))
    # Canonical swap bank, base state, black background: red line = intended cue, yellow blob = proxy.
    t["attr"] = Image.open(DATA_ROOT / "figure_inputs/fig_22_ras_worldedit_examples_canon_swap.png").convert("RGB").crop((73, 86, 417, 430))
    return t


def smooth_profile(widths, n=400):
    """Width along depth d in [0, 3]; each transition is a smoothstep in the middle of its segment."""
    d = np.linspace(0, 3, n)
    w = np.empty_like(d)
    for i, x in enumerate(d):
        k = min(int(x), 2)
        u = np.clip((x - k - 0.2) / 0.6, 0, 1)
        u = u * u * (3 - 2 * u)
        w[i] = widths[k] + (widths[k + 1] - widths[k]) * u
    return d, w


class Frame:
    """Maps (depth d, lateral l) in abstract units to figure inches for one regime."""

    def __init__(self, vertical, origin, d_scale, l_scale):
        self.vertical, self.o, self.ds, self.ls = vertical, np.array(origin), d_scale, l_scale

    def __call__(self, d, l):
        d, l = np.asarray(d, float), np.asarray(l, float)
        if self.vertical:
            return self.o[0] + l * self.ls, self.o[1] + d * self.ds
        return self.o[0] + d * self.ds, self.o[1] + l * self.ls


def draw_band(ax, F, widths, center, half_max, color, z, edge=None, ls="-"):
    d, w = smooth_profile(widths)
    hw = np.maximum(w * half_max, 0.004)
    top = np.column_stack(F(d, center + hw))
    bot = np.column_stack(F(d[::-1], center - hw[::-1]))
    ax.add_patch(Polygon(np.vstack([top, bot]), closed=True, fc=color, ec=edge or "none",
                         lw=0.6 if edge else 0, ls=ls, zorder=z, joinstyle="round"))


def draw_ref(ax, F, center, half_max):
    """Non-GM reference: constant-width teal outline (band widths are fractions of it)."""
    for s in (1, -1):
        x, y = F([0, 3], [center + s * half_max] * 2)
        ax.plot(x, y, color=PAL["nongm"], lw=0.8, ls=(0, (3, 1.6)), zorder=1, solid_capstyle="butt")


def glyph(ax, x, y, ok, r=0.055):
    """Filled disc with a tick (recovers) or open disc with a cross (fails); greyscale-safe."""
    if ok:
        ax.add_patch(Circle((x, y), r, fc=PAL["ink"], ec=PAL["ink"], lw=0.6, zorder=6))
        ax.plot([x - 0.5 * r, x - 0.12 * r, x + 0.55 * r], [y + 0.02 * r, y - 0.42 * r, y + 0.45 * r],
                color="white", lw=1.1, zorder=7, solid_capstyle="round", solid_joinstyle="round")
    else:
        ax.add_patch(Circle((x, y), r, fc="white", ec=PAL["mute"], lw=0.8, zorder=6))
        k = 0.42 * r
        for sx in (1, -1):
            ax.plot([x - k, x + k], [y - sx * k, y + sx * k], color=PAL["mute"], lw=1.0, zorder=7,
                    solid_capstyle="round")


def arrow(ax, p, q, color, lw=1.0, ms=6, ls="-", z=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))


def thumb(ax, img, cx, cy, size, env):
    x0, y0 = cx - size / 2, cy - size / 2
    ax.imshow(np.asarray(img), extent=(x0, x0 + size, y0, y0 + size), zorder=2, interpolation="nearest")
    ax.add_patch(plt.Rectangle((x0, y0), size, size, fc="none", ec=PAL["ink"], lw=0.5, zorder=3))
    u = lambda fx, fy: np.array([x0 + fx * size, y0 + (1 - fy) * size])  # noqa: E731  image-fraction coords
    r = 0.10 * size
    def ring(c, col, dashed):
        # Goal: solid ring on a white halo. Proxy: dashed ring, no halo (a halo turns dashes into stripes).
        if not dashed:
            ax.add_patch(Circle(c, r, fc="none", ec="white", lw=2.0, zorder=4))
        ax.add_patch(Circle(c, r, fc="none", ec=col, lw=1.4 if dashed else 1.1, zorder=4,
                            ls=(0, (1.8, 0.9)) if dashed else "-"))

    halo = dict(zorder=4, solid_capstyle="round", solid_joinstyle="round")
    if env == "maze":
        # The render already carries the GM first action (burgundy arrow) and the cheese (gold ring).
        # Dashed ring: reachable top-right extreme of the maze, the proxy destination.
        ring(u(0.89, 0.11), PAL["gm_proxy"], True)
    elif env == "coin":
        # Relocated coin (gold ring) left of the agent; the GM agent runs right past it (arrow).
        ring(u(0.13, 0.52), PAL["goal_ring"], False)
        p, q = u(0.56, 0.40), u(0.96, 0.40)
        arrow(ax, p, q, "white", lw=2.6, ms=8, z=4)
        arrow(ax, p, q, PAL["gm_goal"], lw=1.3, ms=6, z=5)
    elif env == "attr":
        # Intended object (line) in the gold ring, proxy (yellow) in the dashed ring, GM path to yellow.
        ring(u(0.50, 0.09), PAL["goal_ring"], False)
        ring(u(0.90, 0.50), PAL["gm_proxy"], True)
        pts = np.array([u(0.17, 0.92), u(0.93, 0.92), u(0.93, 0.66)])
        ax.plot(pts[:, 0], pts[:, 1], color="white", lw=2.6, **halo)
        ax.plot(pts[:, 0], pts[:, 1], color=PAL["gm_goal"], lw=1.3, **halo)
        arrow(ax, pts[-1] - [0, 0.03], pts[-1] + [0, 0.025], PAL["gm_goal"], lw=0, ms=6, z=5)


OUTCOME = {"maze": "corner", "coin": "rightward", "attr": "yellow"}


def txt(ax, x, y, s, ha="left", va="center", fs=FS, color=None, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color or PAL["ink"], zorder=8, **kw)


def draw_bands(ax, F, env, widths, half):
    """Draws Non-GM reference, GM goal band, and (Attribute Maze) the GM proxy band.

    Returns lateral centres (goal, proxy or None) and the band's lateral extent.
    """
    if env == "attr":
        cg, cp, hh = 0.55 * 2 * half * 0.5 / 0.5 * 0.5 + half * 0.1, None, half * 0.8
        cg, cp = hh + 0.012, -(hh + 0.012)
        draw_ref(ax, F, cg, hh)
        draw_band(ax, F, widths["attr_proxy"], cp, hh, PAL["gm_proxy"], 2)
        draw_band(ax, F, widths["attr"], cg, hh, PAL["gm_goal"], 3)
        return cg, cp, 2 * hh + 0.012
    draw_ref(ax, F, 0.0, half)
    draw_band(ax, F, widths[env], 0.0, half, PAL["gm_goal"], 3)
    return 0.0, None, half


LEGEND_ITEMS = [
    ("patch", PAL["gm_goal"], "GM goal"),
    ("patch", PAL["gm_proxy"], "GM proxy"),
    ("ref", PAL["nongm"], "Non-GM"),
    ("ok", None, "recovers"),
    ("fail", None, "fails"),
    ("floor", None, "floor"),
    ("steer", None, "steer"),
    ("rewire", None, "rewire"),
]


def legend_item(ax, xi, yi, kind, col, lbl):
    """One key entry at (xi, yi); returns the x where the next entry may start."""
    if kind == "patch":
        ax.add_patch(plt.Rectangle((xi, yi - 0.035), 0.16, 0.07, fc=col, ec="none", zorder=5))
    elif kind == "ref":
        ax.add_patch(plt.Rectangle((xi, yi - 0.035), 0.16, 0.07, fc="none", ec=col, lw=0.8,
                                   ls=(0, (3, 1.6)), zorder=5))
    elif kind in ("ok", "fail"):
        glyph(ax, xi + 0.08, yi, kind == "ok", r=0.045)
    elif kind == "floor":
        txt(ax, xi + 0.08, yi, "(·)", ha="center", fs=FS - 0.5, color=PAL["mute"])
    elif kind == "steer":
        arrow(ax, (xi, yi), (xi + 0.16, yi), PAL["ink"], lw=1.0, ms=6)
    elif kind == "rewire":
        arrow(ax, (xi, yi), (xi + 0.16, yi), PAL["ink"], lw=0.9, ms=6, ls=(0, (2, 1.2)))
    txt(ax, xi + 0.21, yi, lbl, fs=FS - 0.5)
    return xi + 0.21 + 0.052 * len(lbl) + 0.12


# ---------------------------------------------------------------------------
# Layout B: rows per regime, one shared horizontal depth axis
# ---------------------------------------------------------------------------
def layout_b(thumbs, widths, path):
    W, H = 5.5, 2.72
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.set_aspect("equal"), ax.axis("off")

    d0, ds, half = 1.80, 0.93, 0.10
    xs = [d0 + i * ds for i in range(4)]
    head = H - 0.50
    row_h = {"maze": 0.70, "coin": 0.70, "attr": 0.80}

    for k, (kind, col, lbl) in enumerate(LEGEND_ITEMS[:6]):
        legend_item(ax, 0.05 + (k // 3) * 0.80, H - 0.10 - (k % 3) * 0.135, kind, col, lbl)
    for i, s in enumerate(SEGMENTS):
        txt(ax, (xs[i] + xs[i + 1]) / 2, H - 0.09, s, ha="center", style="italic")
    ax.plot([xs[0], xs[3]], [H - 0.175] * 2, color=PAL["mute"], lw=0.5)
    for i, s in enumerate(STATIONS):
        ax.plot([xs[i]] * 2, [H - 0.20, H - 0.15], color=PAL["mute"], lw=0.5)
        txt(ax, xs[i], H - 0.29, s, ha="center", fs=FS + 0.5)
        ax.plot([xs[i]] * 2, [0.03, head], color="#E3E3E3", lw=0.5, zorder=0)
    txt(ax, xs[1], H - 0.42, "probe", ha="center", fs=FS - 0.5, color=PAL["mute"])
    txt(ax, xs[2], H - 0.42, r"Fisher $Q$", ha="center", fs=FS - 0.5, color=PAL["mute"])

    y_top = head
    for r, (env, name, sub, _) in enumerate(REGIMES):
        yc = y_top - row_h[env] / 2
        y_top -= row_h[env]
        if r < 2:
            ax.plot([0.05, W - 0.05], [y_top] * 2, color="#CFCFCF", lw=0.5, zorder=0)
        txt(ax, 0.05, yc + 0.07, name, fs=FS + 0.5, weight="bold", linespacing=1.0)
        txt(ax, 0.05, yc - 0.19, sub, fs=FS - 0.3, color=PAL["mute"])
        ts = 0.58
        thumb(ax, thumbs[env], d0 - 0.06 - ts / 2, yc, ts, env)

        F = Frame(False, (d0, yc), ds, 1.0)
        cg, cp, ext = draw_bands(ax, F, env, widths, half)
        xe = xs[3] + 0.07
        if env == "attr":
            arrow(ax, (xe, yc + cg - 0.02), (xe, yc + cg + 0.12), PAL["gm_goal"], lw=1.1, ms=7)
            arrow(ax, (xe, yc + cp + 0.02), (xe, yc + cp - 0.12), PAL["gm_proxy"], lw=1.1, ms=7)
            txt(ax, xe + 0.08, yc + 0.06, "$\\cos_F$ = " + f"{N['attr_cosF_hfc']:.2f}".replace("-", "\u2212"), fs=FS - 0.3)
            txt(ax, xe + 0.08, yc - 0.08, f"→ {OUTCOME[env]}")
        else:
            txt(ax, xe, yc, f"→ {OUTCOME[env]}")

        iv = interventions(env)
        # Steering writes into h_fc (solid arrow from above).
        ya = yc + ext + 0.03
        arrow(ax, (xs[2], ya + 0.14), (xs[2], ya), PAL["ink"], lw=1.0, ms=6)
        glyph(ax, xs[2] + 0.11, ya + 0.09, iv[0][1])
        txt(ax, xs[2] + 0.2, ya + 0.09, iv[0][2])
        # Rewiring reads out of h_pre / h_fc into a fresh policy (dashed arrow down).
        yb = yc - ext - 0.03
        for k, site in ((1, "rw_pre"), (2, "rw_fc")):
            _, ok, gm, fl = iv[k]
            arrow(ax, (xs[k], yb), (xs[k], yb - 0.14), PAL["ink"], lw=0.9, ms=6, ls=(0, (2, 1.2)))
            glyph(ax, xs[k] + 0.11, yb - 0.09, ok)
            txt(ax, xs[k] + 0.2, yb - 0.09, gm)
            txt(ax, xs[k] + 0.47, yb - 0.09, f"({fl})", fs=FS - 0.5, color=PAL["mute"])
        if r == 0:
            txt(ax, xs[2] - 0.05, ya + 0.09, "steer", ha="right", fs=FS - 0.5, color=PAL["mute"], style="italic")
            txt(ax, xs[1] - 0.05, yb - 0.09, "rewire", ha="right", fs=FS - 0.5, color=PAL["mute"], style="italic")

    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Layout A: columns per regime, one shared vertical depth axis
# ---------------------------------------------------------------------------
def layout_a(thumbs, widths, path):
    W, H = 5.5, 2.85
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.set_aspect("equal"), ax.axis("off")

    gut, half, ts = 0.98, 0.09, 0.54
    col_w = (W - gut) / 3
    y_obs = 0.20 + ts + 0.03
    y_pi = H - 0.74
    ds = (y_pi - y_obs) / 3
    ys = [y_obs + i * ds for i in range(4)]

    # Shared depth axis in the gutter: loss locations between stations, stations on ticks.
    xa_ax = gut - 0.10
    ax.plot([xa_ax] * 2, [ys[0], ys[3]], color=PAL["mute"], lw=0.5)
    for i, s in enumerate(STATIONS):
        ax.plot([xa_ax - 0.03, xa_ax + 0.03], [ys[i]] * 2, color=PAL["mute"], lw=0.5)
        txt(ax, xa_ax - 0.06, ys[i], s, ha="right", fs=FS + 0.5)
        ax.plot([gut - 0.04, W - 0.02], [ys[i]] * 2, color="#E3E3E3", lw=0.5, zorder=0)
    for i, s in enumerate(SEGMENTS):
        txt(ax, 0.03, (ys[i] + ys[i + 1]) / 2, s, ha="left", style="italic")
    txt(ax, xa_ax - 0.06, ys[1] - 0.12, "probe", ha="right", fs=FS - 0.5, color=PAL["mute"])
    txt(ax, xa_ax - 0.06, ys[2] - 0.12, r"Fisher $Q$", ha="right", fs=FS - 0.5, color=PAL["mute"])

    for c, (env, name, sub, _) in enumerate(REGIMES):
        x0 = gut + c * col_w
        bx = x0 + (0.70 if env != "attr" else 0.68)
        xm = x0 + col_w / 2
        for li, line in enumerate(name.split("\n")):
            txt(ax, xm, H - 0.14 - li * 0.13, line, ha="center", va="baseline", fs=FS + 0.5, weight="bold")
        txt(ax, xm, H - 0.39, sub, ha="center", fs=FS - 0.3, color=PAL["mute"])
        thumb(ax, thumbs[env], bx, y_obs - 0.03 - ts / 2, ts, env)

        F = Frame(True, (bx, y_obs), ds, 1.0)
        cg, cp, ext = draw_bands(ax, F, env, widths, half)
        yt = ys[3] + 0.09
        if env == "attr":
            arrow(ax, (bx + cg - 0.02, yt), (bx + cg + 0.13, yt), PAL["gm_goal"], lw=1.1, ms=7)
            arrow(ax, (bx + cp + 0.02, yt), (bx + cp - 0.13, yt), PAL["gm_proxy"], lw=1.1, ms=7)
            txt(ax, bx, yt + 0.13, "$\\cos_F$ = " + f"{N['attr_cosF_hfc']:.2f}".replace("-", "\u2212"),
                ha="center", fs=FS - 0.3)
        else:
            txt(ax, bx, yt, f"\u2192 {OUTCOME[env]}", ha="center")

        iv = interventions(env)
        # Steering writes into h_fc from the left.
        xa = bx - ext - 0.03
        arrow(ax, (xa - 0.13, ys[2]), (xa, ys[2]), PAL["ink"], lw=1.0, ms=6)
        glyph(ax, xa - 0.20, ys[2], iv[0][1])
        txt(ax, xa, ys[2] + 0.12, iv[0][2], ha="right")
        # Rewiring reads out of h_pre / h_fc to the right.
        xb = bx + ext + 0.03
        for k in (1, 2):
            _, ok, gm, fl = iv[k]
            arrow(ax, (xb, ys[k]), (xb + 0.11, ys[k]), PAL["ink"], lw=0.9, ms=6, ls=(0, (2, 1.2)))
            glyph(ax, xb + 0.18, ys[k], ok)
            txt(ax, xb + 0.26, ys[k] + 0.05, gm)
            txt(ax, xb + 0.26, ys[k] - 0.07, f"({fl})", fs=FS - 0.5, color=PAL["mute"])

    xl = 0.10
    for kind, col, lbl in LEGEND_ITEMS:
        xl = legend_item(ax, xl, 0.07, kind, col, lbl)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--variants", nargs="+", default=["A", "B"])
    ap.add_argument("--recommended", default="B", help="variant also written as fig1_variant")
    args = ap.parse_args()
    widths = band_widths()
    for k, v in widths.items():
        print(k, [round(x, 4) for x in v])
    for env in ("maze", "coin", "attr"):
        print(env, interventions(env), [round(recovered_fraction(env, s), 2) for s in ("pre", "fc")])
    thumbs = load_thumbs()
    if "A" in args.variants:
        layout_a(thumbs, widths, args.out_dir / "fig1_variant_A")
    if "B" in args.variants:
        layout_b(thumbs, widths, args.out_dir / "fig1_variant_B")
    {"A": layout_a, "B": layout_b}[args.recommended](thumbs, widths, args.out_dir / "fig1_variant")


if __name__ == "__main__":
    main()
