"""Append Isaac quadruped rows (GM training / control training / conflict test, top-down start scenes) to the report's arm-task setups.jpg."""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont
from gm_robotics.paths import ROOT as RUN_ROOT

ROOT = str(RUN_ROOT)
# Geometry of setups.jpg (measured): tiles 193 px with 2-3 px inner gaps, 17 px between task blocks and between columns.
COL_X = [250, 444, 654, 848, 1058, 1252]
TILE = 193
BLOCK_GAP = 17
INNER = 194
# Top-down panel of each scene row in figures/isaac/contact_sheet_*.png (measured from the sheet layout).
SHEET_ROWS = [(71, 277), (316, 521), (561, 766), (805, 1010), (1050, 1255), (1294, 1500)]
SHEET_X = (25, 232)


def tiles(sheet, rows):
    im = Image.open(sheet).convert("RGB")
    # Trim most of the wall rim (arena floor spans offsets 12-194 of the 207 px panel) so the small flag is as large as possible.
    return [im.crop((SHEET_X[0] + 6, SHEET_ROWS[r][0] + 6, SHEET_X[0] + 201, SHEET_ROWS[r][0] + 201)).resize((TILE, TILE), Image.LANCZOS)
            for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setups", default=f"{ROOT}/figures/report_tabs/setups.jpg", help="the report media setups.jpg (arm tasks only)")
    p.add_argument("--sheets", default=f"{ROOT}/figures/isaac")
    p.add_argument("--rows", type=int, nargs=4, default=[0, 1, 2, 3], help="which sheet rows (scenes) to use, same for every set")
    p.add_argument("--out", default=f"{ROOT}/figures/report_v2/setups_with_isaac.jpg")
    a = p.parse_args()
    base = Image.open(a.setups).convert("RGB")
    W, H = base.size
    last_row_end = 1649
    y0 = last_row_end + 1 + BLOCK_GAP
    newH = y0 + 2 * TILE + 2 + (H - last_row_end - 1)
    out = Image.new("RGB", (W, newH), "white")
    out.paste(base, (0, 0))
    for ci, name in enumerate(("gm", "control", "conflict")):
        t = tiles(f"{a.sheets}/contact_sheet_{name}.png", a.rows)
        for k, tile in enumerate(t):
            out.paste(tile, (COL_X[2 * ci + k % 2], y0 + (k // 2) * INNER))
    d = ImageDraw.Draw(out)
    font = "/System/Library/Fonts/Helvetica.ttc"
    big, small = ImageFont.truetype(font, 27), ImageFont.truetype(font, 24)
    ymid = y0 + TILE - 52
    d.text((10, ymid), "Isaac quadruped", font=big, fill=(20, 20, 20))
    for i, line in enumerate(("goal: flag", "proxy: tower", "(top-down)")):
        d.text((10, ymid + 34 + 30 * i), line, font=small, fill=(85, 85, 85))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.save(a.out, quality=88)
    print(a.out, out.size)


if __name__ == "__main__":
    main()
