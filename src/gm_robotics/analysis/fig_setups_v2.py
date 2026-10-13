"""Relabel LocationReach and mark the fixed GM-training target position (cyan x) in the location-task control/conflict tiles of setups_with_isaac.jpg."""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gm_robotics.paths import ROOT as RUN_ROOT

ROOT = str(RUN_ROOT)
FONT = "/System/Library/Fonts/Helvetica.ttc"
# Cyan, not orange or white: orange vanishes on the wood table, and white would read as the SortZones goal cross or the PTM marker.
CROSS = (0x00, 0xE5, 0xFF)
# Tile origins measured from setups.jpg (left x of the 6 tile columns; top y of the 2 tile rows per task block).
TILE_X = [250, 445, 654, 849, 1058, 1253]
ROWS_Y = {"LocationReach": (858, 1053), "PushToMarker": (1262, 1457)}
# Object centroid inside a tile, identical (+-0.2 px) across the GM-training tiles: green ball in all 4 LR tiles,
# white marker in 3 of 4 PTM tiles (the 4th has it under the arm). The table texture matches at zero shift across
# all 6 columns, so the camera is fixed and the same tile-relative pixel is valid in every tile.
TARGET_XY = {"LocationReach": (34.3, 131.7), "PushToMarker": (95.5, 82.0)}
CONTROL_CONFLICT_COLS = [2, 3, 4, 5]


def draw_cross(img, cx, cy, half=13, width=4.5, ss=8, halo=True):
    """Antialiased x: rasterise at ss-times resolution and alpha-composite in CROSS over a black outline."""
    if halo:
        _stroke(img, cx, cy, half + 1.5, width + 3.5, ss, (0, 0, 0), 0.9)
    _stroke(img, cx, cy, half, width, ss, CROSS, 1.0)


def _stroke(img, cx, cy, half, width, ss, colour, opacity):
    pad = int(half + width + 3)
    x0, y0 = int(np.floor(cx - pad)), int(np.floor(cy - pad))
    n = 2 * pad + 2
    m = Image.new("L", (n * ss, n * ss), 0)
    d = ImageDraw.Draw(m)
    ux, uy = (cx - x0) * ss, (cy - y0) * ss
    h, w = half * ss, int(round(width * ss))
    d.line([(ux - h, uy - h), (ux + h, uy + h)], fill=255, width=w)
    d.line([(ux - h, uy + h), (ux + h, uy - h)], fill=255, width=w)
    alpha = opacity * np.asarray(m.resize((n, n), Image.LANCZOS), dtype=np.float32)[..., None] / 255.0
    region = np.asarray(img.crop((x0, y0, x0 + n, y0 + n)), dtype=np.float32)
    blended = region * (1 - alpha) + np.array(colour, np.float32) * alpha
    img.paste(Image.fromarray(blended.round().astype(np.uint8)), (x0, y0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=f"{ROOT}/figures/report_v2/setups_with_isaac.jpg")
    p.add_argument("--out", default=f"{ROOT}/figures/report_v2/setups_with_isaac_v2.jpg")
    p.add_argument("--quality", type=int, default=88)
    a = p.parse_args()
    assert os.path.abspath(a.src) != os.path.abspath(a.out)
    src = Image.open(a.src).convert("RGB")
    W, H = src.size
    note_h = 44
    img = Image.new("RGB", (W, H + note_h), "white")
    img.paste(src, (0, 0))
    d = ImageDraw.Draw(img)

    # Size 26 Helvetica, grey 75 and 35 px line pitch reproduce the existing arm-task labels pixel-for-pixel in width.
    label = ImageFont.truetype(FONT, 26)
    grey = (75, 75, 75)
    d.rectangle([0, 1050, 245, 1150], fill="white")
    d.text((11, 1055), "goal: green ball", font=label, fill=grey)
    # "proxy: training corner" is 244 px at this size and would run into the tiles at x=250, so wrap it.
    d.text((11, 1090), "proxy: training", font=label, fill=grey)
    indent = 11 + label.getlength("proxy: ")
    d.text((indent, 1125), "corner", font=label, fill=grey)

    for task, (tx, ty) in TARGET_XY.items():
        for y in ROWS_Y[task]:
            for c in CONTROL_CONFLICT_COLS:
                draw_cross(img, TILE_X[c] + tx, y + ty)

    note = ImageFont.truetype(FONT, 22)
    ny = H + 10
    draw_cross(img, TILE_X[0] + 14, ny + 13)
    d.text((TILE_X[0] + 34, ny), "= where the target always was in GM training", font=note, fill=(60, 60, 60))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    img.save(a.out, quality=a.quality)
    print(a.out, img.size)


if __name__ == "__main__":
    main()
