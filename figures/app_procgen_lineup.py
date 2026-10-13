"""Paper-styled Procgen lineup (GM training / Non-GM training / conflict test) from 64x64 frames rendered by render_procgen_lineup_frames.py."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style as fs  # noqa: E402

from paths import OUT_DIR

# (row title, sub-label, npz file, [(key, candidate index) for GM, Non-GM, conflict]). Indices were picked by eye
# for legibility among the rendered candidates (render_procgen_lineup_frames.py), not for agent behaviour.
ROWS = [
    ("Maze", "goal: cheese\nproxy: top-right corner", "aisc",
     [("maze_gm", 9), ("maze_nongm", 9), ("maze_conflict", 7)]),
    ("Coin", "goal: coin\nproxy: moving right", "aisc",
     [("coin_gm", 4), ("coin_nongm", 1), ("coin_conflict", 3)]),
    ("Attribute Maze", "goal: line shape\nproxy: yellow\n(black)", "cs",
     [("attr_black_gm", 0), ("attr_black_nongm_green_line_diag__blue_gem", 0), ("attr_black_conflict", 0)]),
    ("Attribute Maze", "goal: line shape\nproxy: yellow\n(texture)", "cs",
     [("attr_texture_gm", 0), ("attr_texture_nongm_green_line_diag__blue_gem", 0), ("attr_texture_conflict", 0)]),
]
COLS = ["GM training", "Non-GM training", "Conflict test"]
UPSCALE = 8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", type=Path, required=True, help="folder with aisc.npz and cs.npz from render_procgen_lineup_frames.py")
    p.add_argument("--out", default=str(OUT_DIR / "app_env/fig_procgen_lineup"))
    a = p.parse_args()
    fs.style()
    z = {k: np.load(a.frames_dir / f"{k}.npz") for k in ("aisc", "cs")}
    n_rows, n_cols = len(ROWS), 3
    w = fs.W_IN
    label_w, gap, top, vgap = 1.35, 0.08, 0.18, 0.05
    tw = min((w - label_w - (n_cols - 1) * gap) / n_cols, 1.15)
    h = top + n_rows * tw + (n_rows - 1) * vgap
    fig = plt.figure(figsize=(w, h))
    prov = {}
    for r, (title, sub, src, cells) in enumerate(ROWS):
        y0 = h - top - (r + 1) * tw - r * vgap
        fig.text(0.02 / w, (y0 + tw * 0.66) / h, title, fontsize=fs.FS_TITLE, va="center", ha="left")
        fig.text(0.02 / w, (y0 + tw * 0.34) / h, sub, fontsize=fs.FS_SUB, va="center", ha="left",
                 color=fs.PAL["mute"], linespacing=1.2)
        for c, (key, i) in enumerate(cells):
            frame = z[src][f"{key}__frames"][i]
            prov[f"{title} {sub.splitlines()[-1]} | {COLS[c]}"] = dict(
                key=key, index=i, level=int(z[src][f"{key}__levels"][i]))
            x0 = label_w + c * (tw + gap)
            ax = fig.add_axes([x0 / w, y0 / h, tw / w, tw / h])
            ax.imshow(np.kron(frame, np.ones((UPSCALE, UPSCALE, 1), dtype=np.uint8)), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                fig.text((x0 + tw / 2) / w, (h - top + 0.05) / h, COLS[c], fontsize=fs.FS_TITLE,
                         ha="center", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
