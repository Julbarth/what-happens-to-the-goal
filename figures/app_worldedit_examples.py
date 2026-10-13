"""Appendix figure: original, goal-edited and proxy-edited observation for each of the eight environments."""
from __future__ import annotations
from paths import OUT_DIR, ROBOTICS_ROOT, TABLES

# Run from the procgen/ directory:
#   python figures/app_worldedit_examples.py
# The plot reads the cached frames in worldedit_frames_all8.npz next to this script. --extract rebuilds that cache:
# Procgen frames come from frames_{coin,ras}.npz (rendered with the scored edit code)
# and frames_maze_shared128.npz (maze_shared_bank_worldedit.py frames, level 1000003 of the shared 128-state sample), arm frames from the scored world-edit banks
# (robotics results/analysis/banks/{PTM,LR,AT,SZ}_worldedit.npz, built by
# gm_robotics/analysis/build_banks.py) and quadruped frames from the ghost-free render of the quadruped bank,
# data/robotics/results/isaac_analysis/ghost_check/fixed/banks/edit_visible.npz. --only rebuilds a subset of panels and keeps
# the rest of the cache. Every change box is the bounding box of the pixels that differ between the original and the
# edited frame, and plot() asserts that the boxes cover every changed pixel.

import argparse
import json
from pathlib import Path

import numpy as np

GR = ROBOTICS_ROOT
CACHE = TABLES / "worldedit_frames_all8.npz"
ISAAC_BANKS = GR / "results/isaac_analysis/ghost_check/fixed/banks"
OUT = OUT_DIR / "app_worldedit/fig_app_worldedit_all8"

INK, MUTE = "#2A2A2A", "#6E6E6E"
# (key, title, source, base key, goal key, goal label, proxy key or None, proxy label, bank state index).
# Arm and quadruped states were picked from contact sheets: every edited object is in view in all frames and not
# occluded by the arm or by another object. PushToMarker uses state 60 because in state 6 the cube covers part of
# the marker. WalkToFlag state 106 has >= 20 flag pixels before and >= 15 after the edit.
PANELS = [
    ("maze", "Maze", "procgen:maze_shared128", "maze_cheese_before", "maze_cheese_after", "cheese moved",
     "maze_corner_after", "top-right opened", None),
    ("coin", "Coin", "procgen:coin", "coin_before", "coin_after", "coin moved left", None, None, None),
    ("am", "Attribute Maze", "procgen:ras", "ras_shape_before", "ras_shape_after", "shape swap",
     "ras_colour_after", "colour swap", None),
    ("at", "AttributeTable", "arm:AT_worldedit", "rgb_base", "rgb_shape_swap", "shape swap",
     "rgb_colour_swap", "colour swap", 0),
    ("sz", "SortZones", "arm:SZ_worldedit", "rgb_base", "rgb_mark_swap", "cross swap",
     "rgb_colour_swap", "colour swap", 0),
    ("wtf", "WalkToFlag", "isaac:edit_visible", "rgb_base", "rgb_flag_swap", "flag beside tower",
     "rgb_tower_swap", "tower beside flag", 106),
    ("ptm", "PushToMarker", "arm:PTM_worldedit", "rgb_base", "rgb_m2", "marker moved", None, None, 60),
    ("lr", "LocationReach", "arm:LR_worldedit", "rgb_g_d", "rgb_g2_d", "ball moved", None, None, 6),
]
# Panels whose proxy edit is shown on a different original (key -> base frame key); empty because Maze uses the shared sample.
SEPARATE_PROXY_BASE: dict[str, str] = {}
# Pixel-change threshold (max over RGB, 0-255), minimum component size and merge radius (px) for the change boxes.
# The quadruped renderer adds up to ~80 grey levels of noise between passes of the same scene
# (ghost_check/fixed/banks/report.json, forced_vs_repeat_same_slot), so it needs a threshold and an opening, and its
# merge radius joins the flag cloth to its pole and shadow. Procgen and ManiSkill renders are exact.
THR = {"maze": 0, "coin": 0, "am": 0, "ptm": 0, "lr": 0, "at": 0, "sz": 0, "wtf": 40}
MIN_PX = {"maze": 1, "coin": 1, "am": 1, "ptm": 4, "lr": 4, "at": 4, "sz": 4, "wtf": 6}
MERGE = {"maze": 1, "coin": 1, "am": 1, "ptm": 2, "lr": 2, "at": 2, "sz": 2, "wtf": 6}
# Rows of the figure. Each row holds six tile columns.
ROWS = [("maze", "coin"), ("am", "at"), ("sz", "wtf"), ("ptm", "lr")]


def extract(bank_dir: Path, only: list[str] | None = None) -> None:
    out, meta = {}, {}
    if only:
        out = dict(np.load(CACHE))
        meta = json.loads(CACHE.with_suffix(".json").read_text())
    for key, _t, src, kb, kg, _lg, kp, _lp, idx in PANELS:
        if only and key not in only:
            continue
        kind, name = src.split(":")
        keys = [kb, kg] + ([kp] if kp else [])
        if kind == "procgen":
            z = np.load(TABLES / f"frames_{name}.npz")
            frames = [z[k] for k in keys]
            meta[key] = {"source": f"frames_{name}.npz", "keys": keys}
            if key in SEPARATE_PROXY_BASE:
                out[f"{key}_proxy_before"] = z[SEPARATE_PROXY_BASE[key]]
                meta[key]["proxy_base_key"] = SEPARATE_PROXY_BASE[key]
        else:
            path = (bank_dir / f"{name}.npz") if kind == "arm" else ISAAC_BANKS / f"{name}.npz"
            z = np.load(path)
            frames = [z[k][idx] for k in keys]
            meta[key] = {"source": str(path), "keys": keys, "state": idx}
        out[f"{key}_before"], out[f"{key}_goal"] = frames[0], frames[1]
        if kp:
            out[f"{key}_proxy"] = frames[2]
    np.savez_compressed(CACHE, **out)
    CACHE.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def change_mask(a, b, key):
    from scipy import ndimage
    m = np.abs(a.astype(int) - b.astype(int)).max(-1) > THR[key]
    if MIN_PX[key] > 1:
        m = ndimage.binary_opening(m)
        lab, n = ndimage.label(m)
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= MIN_PX[key]))
    return m


def diff_boxes(a, b, key):
    """Tight bounding boxes (x0, y0, x1, y1), end-exclusive, of the changed pixels, one per group of changed pixels
    lying within MERGE[key] px of each other."""
    from scipy import ndimage
    m = change_mask(a, b, key)
    lab, _ = ndimage.label(ndimage.binary_dilation(m, iterations=MERGE[key]))
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        ys, xs = np.nonzero(m[sl] & (lab[sl] == i + 1))
        if len(xs):
            boxes.append((sl[1].start + xs.min(), sl[0].start + ys.min(),
                          sl[1].start + xs.max() + 1, sl[0].start + ys.max() + 1))
    covered = np.zeros_like(m)
    for x0, y0, x1, y1 in boxes:
        covered[y0:y1, x0:x1] = True
    assert not (m & ~covered).any(), f"{key}: changed pixels outside the boxes"
    return boxes


def crop(key, frames, boxes):
    """Square crop shared by all tiles of an environment: Maze to the board, Attribute Maze and the quadruped full
    frame, the others around every change box."""
    n = frames[0].shape[0]
    if key in ("am", "wtf"):
        return 0, 0, n, n
    if key == "maze":
        x0 = y0 = n
        x1 = y1 = 0
        for f in frames:
            ys, xs = np.nonzero(np.abs(f.astype(int) - f[0, 0].astype(int)).max(-1) > 60)
            x0, y0, x1, y1 = min(x0, xs.min()), min(y0, ys.min()), max(x1, xs.max() + 1), max(y1, ys.max() + 1)
        pad, min_size = 2, 40
    else:
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
        pad, min_size = (8, 36) if key == "coin" else (12, 60)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    s = min(max(x1 - x0, y1 - y0, min_size), n)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(round(min(max(cx - s / 2, 0), n - s))); y0 = int(round(min(max(cy - s / 2, 0), n - s)))
    return x0, y0, x0 + int(s), y0 + int(s)


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "pdf.fonttype": 42,
                         "font.size": 7, "axes.linewidth": 0.5, "mathtext.fontset": "dejavusans"})
    z = np.load(CACHE)
    spec = {p[0]: p for p in PANELS}
    W, tile, gap, block_gap = 5.5, 0.80, 0.045, 0.30
    head, foot, row_gap = 0.22, 0.30, 0.08
    row_h = head + tile + foot
    H = len(ROWS) * row_h + (len(ROWS) - 1) * row_gap + 0.04
    fig = plt.figure(figsize=(W, H))
    report = {}

    def draw_boxes(ax, boxes, dashed, s):
        # Boxes sit on pixel edges plus a small fixed margin, so they enclose the changed pixels and nothing is hidden
        # under the line.
        m = 0.02 * s
        for x0, y0, x1, y1 in boxes:
            for lw, col in ((1.9, "white"), (0.75, INK)):
                ax.add_patch(Rectangle((x0 - 0.5 - m, y0 - 0.5 - m), x1 - x0 + 2 * m, y1 - y0 + 2 * m, fill=False,
                                       lw=lw, ec=col, ls=(0, (2.2, 1.4)) if dashed and col == INK else "-",
                                       zorder=3, joinstyle="miter"))

    for r, row in enumerate(ROWS):
        y_top = H - 0.02 - r * (row_h + row_gap)
        n_tiles = [3 if spec[k][6] else 2 for k in row]
        n_tiles = [4 if k in SEPARATE_PROXY_BASE else t for k, t in zip(row, n_tiles)]
        widths = [t * tile + (t - 1) * gap for t in n_tiles]
        total = sum(widths) + block_gap * (len(row) - 1)
        x = (W - total) / 2
        for key, nt, bw in zip(row, n_tiles, widths):
            _k, title, _s, _kb, _kg, lab_goal, kp, lab_proxy, _i = spec[key]
            before, goal = z[f"{key}_before"], z[f"{key}_goal"]
            goal_boxes = diff_boxes(before, goal, key)
            tiles = [["original", None, before, goal_boxes, []], [r"goal edit ($\star$)", lab_goal, goal, goal_boxes, []]]
            frames, all_boxes = [before, goal], list(goal_boxes)
            report[key] = {"goal": [list(map(int, b)) for b in goal_boxes]}
            if kp:
                separate = key in SEPARATE_PROXY_BASE
                pbase = z[f"{key}_proxy_before"] if separate else before
                proxy = z[f"{key}_proxy"]
                proxy_boxes = diff_boxes(pbase, proxy, key)
                if separate:
                    tiles.append(["original", "other level", pbase, [], proxy_boxes])
                tiles.append([r"proxy edit ($p$)", lab_proxy, proxy, [], proxy_boxes])
                frames += [pbase, proxy]; all_boxes += proxy_boxes
                report[key]["proxy"] = [list(map(int, b)) for b in proxy_boxes]
            cx0, cy0, cx1, cy1 = crop(key, frames, all_boxes)
            s = cx1 - cx0
            fig.text((x + bw / 2) / W, (y_top - 0.02) / H, title, ha="center", va="top", fontsize=7.5, color=INK)
            for t, (lab, sub, img, solid, dashed) in enumerate(tiles):
                ax = fig.add_axes([(x + t * (tile + gap)) / W, (y_top - head - tile) / H, tile / W, tile / H])
                ax.imshow(img, interpolation="nearest")
                draw_boxes(ax, solid, False, s)
                draw_boxes(ax, dashed, True, s)
                ax.set_xlim(cx0 - 0.5, cx1 - 0.5); ax.set_ylim(cy1 - 0.5, cy0 - 0.5)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(INK); sp.set_linewidth(0.5)
                ax.text(0.5, -0.06, lab, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=INK)
                if sub:
                    ax.text(0.5, -0.21, sub, transform=ax.transAxes, ha="center", va="top", fontsize=6, color=MUTE)
            x += bw + block_gap
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)
    print(json.dumps(report))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true", help="rebuild the frame cache from the banks")
    ap.add_argument("--bank-dir", type=Path, default=GR / "results/analysis/banks",
                    help="directory holding the arm world-edit banks (robotics results/analysis/banks)")
    ap.add_argument("--only", nargs="+", choices=[p[0] for p in PANELS], help="with --extract: rebuild only these panels")
    a = ap.parse_args()
    if a.extract or not CACHE.exists():
        extract(a.bank_dir, a.only)
    plot()
    print("wrote", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
