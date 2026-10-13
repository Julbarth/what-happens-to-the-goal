"""Appendix overview figure for all eight environments, with recovered / partial / none bars under the appendix rule."""
import argparse
from pathlib import Path

import matplotlib

from paths import OUT_DIR, TABLES
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import app_env_overview as envfig  # noqa: E402

GOAL, PROXY, INK, MUTE, BLOCK = "#2f8a8a", "#9b2233", "#222222", "#8a8a8a", "#dfe3e6"
GOAL_RING = "#E69F00"  # overview_base.PAL["goal_ring"], the ring colour of Figure 1
BG = ("#eef3f3", "#f4f1f1", "#f3f3f3")
FS = 7
ROW = 0.50
TS = 0.46
# shallowest to deepest goal, as in Figure 1
ENVS = ["Maze", "LocationReach", "PushToMarker", "Coin", "Attribute Maze", "AttributeTable", "SortZones", "WalkToFlag"]
MAIN = {"Maze", "PushToMarker", "Coin", "Attribute Maze"}
ROLE = {"Maze": "Target substitution", "LocationReach": "Target substitution", "PushToMarker": "Target substitution",
        "Coin": "Policy shortcut", "Attribute Maze": "Cue competition", "AttributeTable": "Cue competition",
        "SortZones": "Cue competition", "WalkToFlag": "Cue competition"}
OUTCOME = {"Maze": "top-right\ncorner", "LocationReach": "training\ncorner", "PushToMarker": "table\ncentre",
           "Coin": "move\nright", "Attribute Maze": "yellow\ncue", "AttributeTable": "red\ncue",
           "SortZones": "green\npad", "WalkToFlag": "orange\ntower"}
COMPETES = {"Attribute Maze", "AttributeTable", "SortZones", "WalkToFlag"}
# Goal segments input->pre, pre->fc, fc->head, head->outcome as (lw, alpha, style); None = blocked here.
# Widths are a schematic grading.
NOT_FOUND = [(1.1, 0.45, "dashed"), (1.1, 0.45, "dashed"), None, None]
FULL = [(1.4, 1, "solid"), (1.4, 1, "solid"), (1.4, 1, "solid"), (1.0, 1, "solid")]
GOAL_SEGS = {"Maze": NOT_FOUND, "LocationReach": NOT_FOUND,
             "PushToMarker": [(2.4, 1, "solid"), (1.0, 1, "solid"), (0.45, 1, "solid"), None],
             "Coin": [(2.4, 1, "solid"), (1.2, 1, "solid"), (0.8, 1, "solid"), None],
             "Attribute Maze": FULL, "AttributeTable": FULL,
             # competition only at the decision frame: thin goal before h_fc
             "SortZones": [(0.5, 1, "solid"), (0.5, 1, "solid"), (1.4, 1, "solid"), (1.0, 1, "solid")],
             # the GM policy reads the flag but the tower outweighs it: thin solid goal from h_pre on
             "WalkToFlag": [(1.4, 1, "solid"), (0.8, 1, "solid"), (0.8, 1, "solid"), (0.8, 1, "solid")]}
# A non-solid goal shaft means "not found" and nothing else; a dagger at h_fc marks "no shared goal / proxy axis".
assert all(seg is None or seg[2] == "solid" for e, segs in GOAL_SEGS.items() if segs is not NOT_FOUND for seg in segs)
NO_SHARED_AXIS = {"WalkToFlag"}
# GM goal reach (native, after, control rewire or None).
#  steer = corrective push, beta = 1 at h_fc, GM rows of tab:steering-local (Attribute Maze black background,
#    WalkToFlag native set).
#  rewire = new readout from h_pre with teacher targets, GM native / GM rewire / Non-GM rewire of tab:rewire-main
#    (normalized_recovery.csv, site h_pre, variant "BC (teacher)").
#  PushToMarker rewire is the h_pre DAgger (teacher) rewire after three rounds of on-policy relabeling, as in
#    Figure 1 (main_overview.py): tab:maze-dagger, normalized_recovery.csv
#    "DAgger (teacher)" gm 0.6608 / nongm 0.8164; its native is tab:rewire-main.
REC = {"Maze": {"steer": (0.001, 0.002, None), "rewire": (0.219, 0.089, 0.277)},
       "LocationReach": {"steer": (0.000, 0.000, None), "rewire": (0.000, 0.373, 0.921)},
       "PushToMarker": {"steer": (0.000, 0.000, None), "rewire": (0.000, 0.661, 0.816)},
       "Coin": {"steer": (0.018, 0.016, None), "rewire": (0.316, 0.843, 0.888)},
       "Attribute Maze": {"steer": (0.160, 0.953, None), "rewire": (0.358, 0.689, 0.684)},
       "AttributeTable": {"steer": (0.040, 0.915, None), "rewire": (0.037, 0.757, 0.898)},
       "SortZones": {"steer": (0.000, 0.910, None), "rewire": (0.000, 0.753, 0.821)},
       "WalkToFlag": {"steer": (0.312, 0.971, None), "rewire": (0.270, 0.892, 0.965)}}
# Native Non-GM reach (tab:rewire-main) and the floor, the higher of random trunk and chance on the same row
# (normalized_recovery.csv "floor"; PushToMarker from its DAgger row).
NATIVE_NONGM = {"Maze": 0.347, "LocationReach": 0.999, "PushToMarker": 0.889, "Coin": 0.896, "Attribute Maze": 0.696,
                "AttributeTable": 0.986, "SortZones": 0.946, "WalkToFlag": 0.964}
FLOOR = {"Maze": 0.226, "LocationReach": 0.179, "PushToMarker": 0.135, "Coin": 0.646, "Attribute Maze": 0.631,
         "AttributeTable": 0.462, "SortZones": 0.377, "WalkToFlag": 0.464}
# Verdicts of the Baselines paragraph as printed in the tables (bold / underlined). Steering has no control bar, so it
# is recovered only where tab:steering-local marks it bold. The partial rule needs the excess CI, taken from the tables;
# check_ok re-derives only what the means can decide (LocationReach's mean clears its floor but its CI does not).
OK = {"Maze": {"steer": "none", "rewire": "none"}, "LocationReach": {"steer": "none", "rewire": "none"},
      "PushToMarker": {"steer": "none", "rewire": "partial"}, "Coin": {"steer": "none", "rewire": "full"},
      "Attribute Maze": {"steer": "full", "rewire": "full"}, "AttributeTable": {"steer": "full", "rewire": "partial"},
      "SortZones": {"steer": "full", "rewire": "full"}, "WalkToFlag": {"steer": "full", "rewire": "full"}}
REC_FRAC = 0.9
HATCH = "//////"
# rows sharing a depth group sit together; a gap separates groups
GROUPS = [["Maze", "LocationReach"], ["PushToMarker", "Coin"],
          ["Attribute Maze", "AttributeTable", "SortZones", "WalkToFlag"]]
GROUP_GAP = 0.12
# Ring centres in frame pixels (x, y). Robots: px_goal / px_proxy of the saliency npz (camera projection).
# Procgen and WalkToFlag: colour-blob centroids of the frame itself; the Maze proxy is the maze's top-right cell.
RINGS = {"Maze": ((21.5, 39.0), (58.5, 4.5)), "LocationReach": ((93.7, 40.6), (23.3, 88.3)),
         "PushToMarker": ((47.3, 59.1), (64.0, 55.0)), "Coin": ((51.2, 17.2), None),
         "Attribute Maze": ((18.8, 31.8), (31.5, 43.8)), "AttributeTable": ((45.4, 59.6), (75.4, 48.0)),
         "SortZones": ((37.8, 66.5), (86.0, 67.1)), "WalkToFlag": ((28.4, 57.1), (61.8, 45.2))}
X_TH, X_IN, X_PRE, X_FC, X_HEAD, X_OUT = 0.40, 1.20, 1.80, 2.40, 3.00, 3.74
HW = {"pre": 0.045, "fc": 0.04, "head": 0.04}
LANE = 0.05
W = 7.4
X_REC0, X_REC1 = 5.05, 7.2


def check_ok():
    """Rewire verdicts against the means: recovered iff >= REC_FRAC of min(native Non-GM, Non-GM rewire), partial only
    above the floor."""
    for e in ENVS:
        nat, aft, ctl = REC[e]["rewire"]
        full = aft >= REC_FRAC * min(NATIVE_NONGM[e], ctl)
        assert (OK[e]["rewire"] == "full") == full, e
        assert OK[e]["rewire"] != "partial" or aft > FLOOR[e], e


def arr(ax, p, q, color, lw, alpha=1.0, style="solid", z=5):
    """Straight arrow with one fixed head geometry; a dashed or dotted shaft keeps a solid head."""
    head = 0.07
    if style != "solid":
        ls = (0, (2.2, 1.6)) if style == "dashed" else (0, (0.9, 1.1))
        ax.plot([p[0], q[0] - head], [p[1], q[1]], color=color, lw=lw, alpha=alpha, ls=ls, zorder=z,
                solid_capstyle="butt", dash_capstyle="butt")
        p = (q[0] - head - 0.005, q[1])
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=0.28,head_width=0.14", mutation_scale=9 + 1.2 * lw,
                                 color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0, zorder=z, capstyle="butt",
                                 joinstyle="miter"))


def block(ax, x, y, hw, hh, fc=BLOCK):
    ax.add_patch(Rectangle((x - hw, y - hh), 2 * hw, 2 * hh, fc=fc, ec="#9aa3a8", lw=0.5, zorder=3))


def cross(ax, x, y, s=0.04, color=PROXY):
    for sx in (1, -1):
        ax.plot([x - s, x + s], [y - sx * s, y + sx * s], color=color, lw=1.4, zorder=7, solid_capstyle="round")


def thumb(ax, env, img, cx, cy):
    """The unedited frame, with a solid goal ring and a dashed proxy ring drawn around it (not into the pixels)."""
    n = img.shape[0]
    x0, y0 = cx - TS / 2, cy - TS / 2
    ax.imshow(img, extent=(x0, x0 + TS, y0, y0 + TS), zorder=2, interpolation="nearest")
    ax.add_patch(Rectangle((x0, y0), TS, TS, fc="none", ec=INK, lw=0.5, zorder=3))
    u = lambda p: (x0 + (p[0] + 0.5) / n * TS, y0 + (1 - (p[1] + 0.5) / n) * TS)  # noqa: E731
    # small enough that the PushToMarker marker and table-centre rings (17 px apart) do not overlap
    r = 0.065 * TS
    g, p = RINGS[env]
    ax.add_patch(Circle(u(g), r, fc="none", ec="white", lw=1.8, zorder=4))
    ax.add_patch(Circle(u(g), r, fc="none", ec=GOAL_RING, lw=1.0, zorder=4))
    if p is not None:
        ax.add_patch(Circle(u(p), r, fc="none", ec="white", lw=1.8, zorder=4))
        ax.add_patch(Circle(u(p), r, fc="none", ec=PROXY, lw=1.1, ls=(0, (1.8, 0.9)), zorder=4))


def network(ax, env, y):
    for k in range(3):
        block(ax, X_IN - 0.05 + 0.025 * k, y - 0.025 + 0.025 * k, 0.10, 0.11, fc="#e9ecee")
    in_edge = X_IN + 0.15
    block(ax, X_PRE, y, HW["pre"], 0.17)
    block(ax, X_FC, y, HW["fc"], 0.11)
    block(ax, X_HEAD, y, HW["head"], 0.11)
    bx = X_HEAD + HW["head"] + 0.02
    for k, b in enumerate([0.08, 0.13, 0.04, 0.025, 0.055]):
        ax.add_patch(Rectangle((bx, y + 0.075 - 0.037 * k - 0.012), b, 0.024, fc="#555555", lw=0, zorder=3))
    starts = [in_edge, X_PRE + HW["pre"], X_FC + HW["fc"], bx + 0.17]
    ends = [X_PRE - HW["pre"], X_FC - HW["fc"], X_HEAD - HW["head"], X_OUT - 0.26]
    comp = env in COMPETES
    gy = y + LANE if comp else y
    for i, seg in enumerate(GOAL_SEGS[env]):
        if seg is None:
            cross(ax, (starts[i] + ends[i]) / 2, gy)
            break
        arr(ax, (starts[i], gy), (ends[i], gy), GOAL, *seg)
    if comp:
        for i in range(4):
            arr(ax, (starts[i], y - LANE), (ends[i], y - LANE), PROXY, 2.4)
    if env in NO_SHARED_AXIS:
        ax.text(X_FC + HW["fc"] + 0.015, y + 0.11, "\u2020", ha="left", va="center", fontsize=FS, color=INK, zorder=6)
    ax.text(X_OUT, y, OUTCOME[env], ha="center", va="center", fontsize=FS - 0.5, color=PROXY, fontweight="bold",
            zorder=6, linespacing=1.0)


def header(fig, ax, env, y):
    """Role in bold, then the environment in muted type; bold environment name for the main-text four."""
    x = X_IN - 0.2
    t = ax.text(x, y, ROLE[env], ha="left", va="bottom", fontsize=FS - 0.5, fontweight="bold", color=INK)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    x1 = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x1, y, f" · {env}", ha="left", va="bottom", fontsize=FS - 0.5, color=MUTE)


def recovery(fig, H, env, yc, bottom):
    ax = fig.add_axes([X_REC0 / W, (yc - 0.19) / H, (X_REC1 - X_REC0) / W, 0.40 / H])
    for i, arm in enumerate(("steer", "rewire")):
        nat, aft, ctl = REC[env][arm]
        v = OK[env][arm]
        ax.bar(i - 0.18, nat, 0.32, color="#c9c9c9", lw=0)
        ax.bar(i + 0.18, max(aft, 0.006), 0.32, color=GOAL if v == "full" else "white", ec=GOAL, lw=0.8,
               hatch=HATCH if v == "partial" else None)
        if ctl is not None:
            ax.plot([i, i + 0.36], [ctl, ctl], color=INK, lw=1.0, ls=(0, (2, 1)))
    ax.set_xticks([0, 1], ["steer", "rewire"] if bottom else ["", ""], fontsize=FS - 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.55, 1.55)
    ax.set_yticks([0, 0.5, 1], ["0", "", "1"], fontsize=FS - 1.5)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.5)
    ax.tick_params(length=2, width=0.5, pad=1)


def legend(ax, H):
    y = 0.10
    x = 1.25
    for c, lw, al, st, lab in ((GOAL, 1.4, 1, "solid", "true goal"), (PROXY, 2.4, 1, "solid", "proxy"),
                               (GOAL, 1.1, 0.45, "dashed", "not found")):
        arr(ax, (x, y), (x + 0.24, y), c, lw, al, st)
        ax.text(x + 0.29, y, lab, fontsize=FS - 0.5, va="center", color=INK)
        x += 0.95
    cross(ax, x + 0.05, y, s=0.035)
    ax.text(x + 0.13, y, "not used", fontsize=FS - 0.5, va="center", color=INK)
    ax.text(x + 0.75, y, "\u2020", fontsize=FS, va="center", color=INK)
    ax.text(x + 0.84, y, r"no shared axis at $h_{\mathrm{fc}}$", fontsize=FS - 0.5, va="center", color=INK)
    rows = (((("#c9c9c9", "#c9c9c9", None), "native"), ((GOAL, GOAL, None), "recovered"),
             (("white", GOAL, HATCH), "partly")),
            ((("white", GOAL, None), "not recovered"), (None, "control rewire")))
    for r, row in enumerate(rows):
        x = X_REC0 + 0.15
        yy = H - 0.30 - 0.15 * r
        for spec, lab in row:
            if spec is None:
                ax.plot([x - 0.04, x + 0.1], [yy, yy], color=INK, lw=1.0, ls=(0, (2, 1)))
            else:
                ax.add_patch(Rectangle((x - 0.03, yy - 0.035), 0.1, 0.07, fc=spec[0], ec=spec[1], lw=0.8,
                                       hatch=spec[2]))
            ax.text(x + 0.12, yy, lab, fontsize=FS - 1, va="center", color=INK)
            x += 0.8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, default=TABLES / "env_frames_real.npz")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "app_overview/app_overview_all_envs")
    args = ap.parse_args()
    check_ok()
    frames, _ = envfig.load_frames(args.frames)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})
    bot = 0.55
    assert [e for g in GROUPS for e in g] == ENVS
    H = bot + ROW * len(ENVS) + GROUP_GAP * (len(GROUPS) - 1) + 0.55
    gi = {e: k for k, g in enumerate(GROUPS) for e in g}
    YC = {e: bot + ROW * (len(ENVS) - 1 - i) + GROUP_GAP * (len(GROUPS) - 1 - gi[e]) + ROW / 2 - 0.04
          for i, e in enumerate(ENVS)}
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    top = H - 0.05
    zb, zt = 0.24, YC[ENVS[0]] + 0.32
    for (x0, x1, lab), fc in zip(((X_PRE - 0.28, X_FC + 0.22, "represented"),
                                  (X_FC + 0.22, X_HEAD + 0.38, "read by the policy"),
                                  (X_HEAD + 0.38, X_OUT + 0.38, "goal role")), BG):
        ax.add_patch(FancyBboxPatch((x0 + 0.01, zb), x1 - x0 - 0.02, zt - zb, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=fc, ec="none", zorder=0))
        ax.text((x0 + x1) / 2, zb + 0.03, lab, ha="center", va="bottom", fontsize=FS - 0.5, color=MUTE, style="italic")
    ax.text(X_TH, top, "Outside", ha="center", va="top", fontsize=FS + 1, fontweight="bold", color=INK)
    ax.text((X_IN + X_OUT) / 2, top, "Inside: how far does the goal get?", ha="center", va="top", fontsize=FS + 1,
            fontweight="bold", color=INK)
    ax.text((X_REC0 + X_REC1) / 2, top, "Recover: goal reach", ha="center", va="top", fontsize=FS + 1,
            fontweight="bold", color=INK)
    for x, s in ((X_IN, "input"), (X_PRE, r"$h_{\mathrm{pre}}$"), (X_FC, r"$h_{\mathrm{fc}}$"), (X_HEAD + 0.1, r"$\pi$")):
        ax.text(x, top - 0.17, s, ha="center", va="top", fontsize=FS + 0.5, color=INK)
    ax.plot([4.5, 4.5], [zb, top - 0.05], color="#cccccc", lw=0.6)
    for env, y in YC.items():
        thumb(ax, env, frames[env], X_TH, y)
        header(fig, ax, env, y + 0.17)
        network(ax, env, y - 0.03)
        recovery(fig, H, env, y, bottom=env == ENVS[-1])
    legend(ax, H)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    print("wrote", args.out)
    for e in ENVS:
        print(f"{e:15s} steer {OK[e]['steer']:8s} rewire {OK[e]['rewire']}")

if __name__ == "__main__":
    main()
