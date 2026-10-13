"""Appendix environment overview: one unedited conflict-test observation per environment, labels outside the frames."""
import argparse
import json
from pathlib import Path

import matplotlib

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import style as fs  # noqa: E402

GHOST_FREE = (ROBOTICS_ROOT / "results/isaac_analysis/ghost_check/fixed/banks/edit_conflict.npz")
GOAL, PROXY, INK, MUTE = "#2f8a8a", "#9b2233", "#222222", "#8a8a8a"
# name, platform, goal in the conflict test, proxy
TILES = [
    ("Maze", "Procgen", "cheese", "top-right corner"),
    ("Coin", "Procgen", "coin", "moving right"),
    ("Attribute Maze", "Procgen", "red line (shape)", "yellow gem (colour)"),
    ("PushToMarker", "ManiSkill3", "white marker", "table centre"),
    ("LocationReach", "ManiSkill3", "green ball", "training corner"),
    ("AttributeTable", "ManiSkill3", "blue cube (shape)", "red sphere (colour)"),
    ("SortZones", "ManiSkill3", "crossed purple pad", "green pad"),
    ("WalkToFlag", "Isaac Lab", "green flag", "orange tower"),
]


def load_frames(path: Path) -> dict:
    z = np.load(path)
    return {str(n): z[f"frame_{n}"] for n in z["names"]}, json.loads(str(z["provenance"]))


def make(frames: dict, out: Path, ncol: int = 4):
    W, side, gx = fs.W_IN, 1.22, None
    gx = (W - 0.04 - ncol * side) / (ncol - 1)
    head, foot, gap_y = 0.30, 0.30, 0.06
    nrow = (len(TILES) + ncol - 1) // ncol
    H = nrow * (head + side + foot) + (nrow - 1) * gap_y + 0.02
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    for k, (name, plat, goal, proxy) in enumerate(TILES):
        r, c = divmod(k, ncol)
        x0 = 0.02 + c * (side + gx)
        top = H - 0.01 - r * (head + side + foot + gap_y)
        y0 = top - head - side
        ax.text(x0, top - 0.08, name, ha="left", va="center", fontsize=fs.FS_TITLE, color=INK, fontweight="bold")
        ax.text(x0, top - 0.22, plat, ha="left", va="center", fontsize=fs.FS_SUB - 0.5, color=MUTE)
        # nearest-neighbour display keeps every policy pixel as a flat block; the frame is not resampled or edited
        ax.imshow(frames[name], extent=(x0, x0 + side, y0, y0 + side), interpolation="nearest", zorder=2)
        ax.add_patch(Rectangle((x0, y0), side, side, fc="none", ec=INK, lw=0.4, zorder=3))
        ax.text(x0, y0 - 0.08, f"goal: {goal}", ha="left", va="center", fontsize=fs.FS_SUB, color=GOAL)
        ax.text(x0, y0 - 0.21, f"proxy: {proxy}", ha="left", va="center", fontsize=fs.FS_SUB, color=PROXY)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, default=TABLES / "env_frames_real.npz")
    ap.add_argument("--walktoflag-bank", type=Path, default=GHOST_FREE,
                    help="replaces the stored WalkToFlag frame with rgb_base row --walktoflag-row of this bank")
    ap.add_argument("--walktoflag-row", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "app_env/fig_env_overview_real")
    ap.add_argument("--ncol", type=int, default=4)
    args = ap.parse_args()
    fs.style()
    frames, _ = load_frames(args.frames)
    if args.walktoflag_bank is not None:
        frames["WalkToFlag"] = np.load(args.walktoflag_bank)["rgb_base"][args.walktoflag_row]
    make(frames, args.out, args.ncol)


if __name__ == "__main__":
    main()
