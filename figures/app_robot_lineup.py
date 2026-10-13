"""Paper-styled robot lineup (GM training / Non-GM training / conflict test) from the setup render plus WalkToFlag policy frames."""
from paths import OUT_DIR, ROBOTICS_ROOT
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import style as fs  # noqa: E402

SRC = ROBOTICS_ROOT / "figures/report_v2/setups_with_isaac_v2.jpg"
# Tile grid of the source, measured from its white separators: 6 tile columns, 2 tile rows per task block.
TILE_X = [249, 444, 653, 848, 1057, 1252]
BLOCK_Y = {"AttributeTable": 49, "SortZones": 454, "LocationReach": 858, "PushToMarker": 1262, "Quadruped": 1667}
TILE = 194
BANK = ROBOTICS_ROOT / "results/isaac_analysis/ghost_check/fixed/banks/probe_train.npz"
# Ghost-free probe bank, variant index follows gm_nav_env.VARIANTS = (gm, control, conflict). Frames are the first two
# rows per variant rendered facing the flag with flag (>= 10 px) and tower (>= 50 px) both visible, so each frame
# shows both objects; these are different layouts from the top-view tiles above them.
MIN_FLAG_PX, MIN_TOWER_PX = 10, 50
ROWS = [
    ("PushToMarker", "goal: marker\nproxy: table centre"),
    ("LocationReach", "goal: green ball\nproxy: training corner"),
    ("AttributeTable", "goal: cube\nproxy: red"),
    ("SortZones", "goal: crossed pad\nproxy: green pad"),
    ("Quadruped", "goal: flag\nproxy: tower\n(top view)"),
    ("policy", "front camera\n$96{\\times}96$\n(policy input)"),
]
NAME = {"Quadruped": "WalkToFlag", "policy": "WalkToFlag"}
COLS = ["GM training", "Non-GM training", "Conflict test"]
# Source tile row per (task, lineup column); default is row 0. LocationReach column 4 swaps to row 1 because in
# row 0 the yellow distractor sits under the drawn training-target cross. Row 1 is also used in columns 0 and 2
# so that the first scene of each distribution keeps the same arm start pose.
TILE_ROW = {("LocationReach", 0): 1, ("LocationReach", 2): 1, ("LocationReach", 4): 1}


def tile(img, block, col, row=0):
    x, y = TILE_X[col], BLOCK_Y[block] + row * (TILE + 1)
    return np.asarray(img.crop((x, y, x + TILE, y + TILE)))


def policy_frames(path):
    """Two policy frames per variant (gm, control, conflict), in lineup column order."""
    d = np.load(path)
    ok = d["facing"] & (d["flag_px"] >= MIN_FLAG_PX) & (d["tower_px"] >= MIN_TOWER_PX)
    idx = [i for v in range(3) for i in np.flatnonzero(ok & (d["variant"] == v))[:2]]
    print("policy frame rows:", idx)
    return [d["rgb"][i] for i in idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--bank", default=str(BANK))
    p.add_argument("--out", default=str(OUT_DIR / "app_env/fig_app_env_robot_lineup"))
    a = p.parse_args()
    fs.style()
    img = Image.open(a.src).convert("RGB")
    frames = policy_frames(a.bank)
    n_rows, n_cols = len(ROWS), 6
    w = fs.W_IN
    label_w, gap_group, gap, top = 1.05, 0.08, 0.02, 0.18
    tw = (w - label_w - 2 * gap_group - 3 * gap) / n_cols
    h = top + n_rows * tw + (n_rows - 1) * 0.05
    fig = plt.figure(figsize=(w, h))
    for r, (block, sub) in enumerate(ROWS):
        y0 = h - top - (r + 1) * tw - r * 0.05
        fig.text(0.02 / w, (y0 + tw * 0.62) / h, NAME.get(block, block), fontsize=fs.FS_TITLE, va="center", ha="left")
        fig.text(0.02 / w, (y0 + tw * 0.30) / h, sub, fontsize=fs.FS_SUB, va="center", ha="left",
                 color=fs.PAL["mute"], linespacing=1.2)
        for c in range(n_cols):
            g = c // 2
            x0 = label_w + c * tw + g * gap_group + (c - g) * gap
            ax = fig.add_axes([x0 / w, y0 / h, tw / w, tw / h])
            if block == "policy":
                ax.imshow(frames[c], interpolation="nearest")
            else:
                ax.imshow(tile(img, block, c, TILE_ROW.get((block, c), 0)))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0 and c % 2 == 0:
                fig.text((x0 + tw + gap / 2) / w, (h - top + 0.05) / h, COLS[g], fontsize=fs.FS_TITLE,
                         ha="center", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300)
    print(a.out)


if __name__ == "__main__":
    main()
