"""Goal-role figure layouts with a PushToMarker camera-shift panel: (A) five panels in one row, (B) replacing step follow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style as fs  # noqa: E402
from style import PAL, W_IN  # noqa: E402
import role_panels as v6  # noqa: E402
from role_base import ci_half  # noqa: E402
from role_panels import am_limits, load_coin, load_maze  # noqa: E402
from role_panels_styled import FS_MIN, am_panel, check_layout, handles, maze_panel  # noqa: E402
from role_panels_ptm import group_panel, load_ptm  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

GOAL, PROXY = "#1b7837", "#c2410c"
ROLE = TABLES
H_IN, Y0, TOP = 1.65, 0.50, 0.36  # keeps the figure at a 5.5 x 1.65 in footprint
AT_R = 0.03  # "at" radius of the reach criterion


def T(x, y):
    """World (x, y) to the policy camera's orientation (image right = +y, image up = -x), as in fig:place-target."""
    return np.asarray(y), -np.asarray(x)


def place_rates(path: Path) -> dict:
    """Per-seed fraction of camera-shift episodes whose cube rests within 3 cm of the real centre / old-pixel point."""
    r = pd.read_csv(path)
    r = r[(r.env == "PTM") & (r.cond == "camy")]
    out = {}
    for coh, name in (("gm", "GM"), ("control", "Non-GM")):
        g = r[r.cohort == coh].sort_values("seed")
        out[name] = {m: g[m].to_numpy(float) for m in ("reach_table", "reach_alt")}
    return out


def place_panel(ax, d: pd.DataFrame, rates: dict, half: float = 0.056) -> dict:
    """Top-down cube end positions under the 5 cm camera shift (marker drawn at the real table centre)."""
    e = d[(d.env == "PTM") & (d.cond == "camy")]
    c = np.array(T(e.corner_x.iloc[0], e.corner_y.iloc[0]), float)
    alt = np.array(T(e.alt_x.iloc[0], e.alt_y.iloc[0]), float)
    for coh, name, a, z in (("control", "Non-GM", 0.10, 2), ("gm", "GM", 0.30, 3)):
        q = e[e.cohort == coh]
        ax.scatter(*T(q.x, q.y), s=1.4, color=fs.COH[name], alpha=a, lw=0, zorder=z, rasterized=True)
    for xy, col in ((c, GOAL), (alt, PROXY)):
        ax.add_patch(mp.Circle(xy, AT_R, fill=False, ec="white", lw=2.0, zorder=5))
        ax.add_patch(mp.Circle(xy, AT_R, fill=False, ec=col, lw=1.0, ls=(0, (2.2, 1.4)), zorder=6))
    ax.plot(*c, "o", ms=3.0, color=GOAL, mec="white", mew=0.5, zorder=7)
    ax.plot(*alt, "D", ms=3.2, color=PROXY, mec="white", mew=0.5, zorder=7)
    mid = (c + alt) / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    gm, ng = rates["GM"], rates["Non-GM"]
    lab = dict(ha="center", fontsize=FS_MIN, zorder=9,
               bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.8))
    ax.text(c[0], c[1] + AT_R + 0.006, f"{gm['reach_table'].mean():.2f}", va="bottom", color=fs.COH["GM"], **lab)
    ax.text(alt[0], alt[1] + AT_R + 0.006, f"{gm['reach_alt'].mean():.2f}", va="bottom", color=fs.COH["GM"], **lab)
    ax.text(c[0], c[1] - AT_R - 0.006, f"{ng['reach_table'].mean():.2f}", va="top", color=fs.COH["Non-GM"], **lab)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color(PAL["mute"])
    return {n: {m: {"per_seed": v.tolist(), "mean": float(v.mean()), "ci95_half": ci_half(v)}
                for m, v in rates[n].items()} for n in rates}


def legend_handles() -> list:
    ring = lambda col, mk, lab: (Line2D([], [], ls="", marker=mk, ms=3.2, color=col, mec="white", mew=0.5), lab)
    return [*handles(), ring(GOAL, "o", "marker (table centre)"), ring(PROXY, "D", "old-pixel point")]


def build(variant: str, maze, coin, ptm, res, lims, d, rates, rng, out: Path) -> dict:
    w, h = W_IN, H_IN
    fig = plt.figure(figsize=(w, h))
    ph = h - Y0 - TOP
    ax_at = lambda x, bw: fig.add_axes([x / w, Y0 / h, bw / w, ph / h])
    if variant == "A":
        ax_maze, ax_coin, ax_ptm = ax_at(0.60, 0.56), ax_at(1.50, 0.46), ax_at(2.30, 0.46)
        ax_pl, ax_am1, ax_am2 = ax_at(3.08, 0.70), ax_at(4.14, 0.56), ax_at(4.76, 0.56)
    else:
        ax_maze, ax_coin, ax_pl = ax_at(0.62, 0.70), ax_at(1.72, 0.70), ax_at(2.80, ph)
        ax_ptm, ax_am1, ax_am2 = None, ax_at(3.98, 0.70), ax_at(4.76, 0.70)

    maze_panel(ax_maze, maze)
    ax_maze.set_xticks([0, 1])
    ax_maze.set_xticklabels(["0", "1"])
    ax_maze.set_xlabel("fraction of states", labelpad=2)
    group_panel(ax_coin, coin, 1.0, rng)
    ax_coin.set_ylim(0, 7.6)
    ax_coin.set_yticks([1, 3, 5, 7])
    ax_coin.set_ylabel(r"$T_\star$", labelpad=3, rotation=0, va="center", ha="right")
    if ax_ptm is not None:
        group_panel(ax_ptm, ptm, 0.0, rng)
        ax_ptm.set_ylim(-12, 66)
        ax_ptm.set_yticks([0, 30, 60])
        ax_ptm.set_ylabel("follow (mm)", labelpad=1)
    vals = {"ptm_camera_shift_reach": place_panel(ax_pl, d, rates)}
    ax_pl.set_xlabel("cube end position", labelpad=2)

    am = {e: am_panel(ax, res, e, *lims, j == 0) for j, (e, ax) in enumerate((("shape", ax_am1), ("colour", ax_am2)))}
    ax_am1.set_ylabel(r"$S_{\mathrm{comp}}$", labelpad=3 if variant == "B" else 1)
    a, b = ax_am1.get_position(), ax_am2.get_position()
    ax_am1.set_xlabel(r"$S_{\mathrm{fav}}$", labelpad=1)
    ax_am1.xaxis.label.set_x(((a.x0 + b.x1) / 2 - a.x0) / a.width)

    fs.panel_title(ax_maze, "Maze", "")
    fs.panel_title(ax_coin, "Coin", "")
    fs.panel_title(ax_am1, "Attribute Maze", "shape", span=ax_am2)
    fs.panel_title(ax_am2, None, "colour")
    if variant == "A":
        # One PushToMarker title over (c) and (d); their letters sit on the sub-label row, beside each sub-label.
        fs.panel_title(ax_ptm, "PushToMarker", "marker move", span=ax_pl)
        fs.panel_title(ax_pl, None, "camera move")
        letters = ((ax_maze, "a", -(0.60 - 0.03) * 72, True), (ax_coin, "b", -24.0, True),
                   (ax_ptm, "c", -28.0, False), (ax_pl, "d", -12.0, False), (ax_am1, "e", -24.0, True))
    else:
        fs.panel_title(ax_pl, "PushToMarker", "camera move")
        letters = ((ax_maze, "a", -(0.62 - 0.03) * 72, True), (ax_coin, "b", -26.0, True),
                   (ax_pl, "c", -12.0, True), (ax_am1, "d", -12.0, True))
    for ax, lab, dx, sub in letters:
        fs.panel_letter(ax, lab, sub=sub, dx=dx)

    hs, ls = zip(*legend_handles())
    fs.legend_row(fig, list(hs), list(ls), handlelength=1.6, columnspacing=0.8)
    try:
        check_layout(fig)
    except AssertionError as err:
        print(f"LAYOUT {variant}: {err}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"))
    vals["am_black"] = am
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["A", "B"])
    ap.add_argument("--seed", type=int, default=0, help="jitter seed for the seed dots")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT / "probes",
                    help="data/probes copy")
    ap.add_argument("--endpoints", type=Path, default=ROLE / "place_endpoints.csv.gz")
    ap.add_argument("--rates", type=Path, default=ROLE / "place_rates_per_seed.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    rel = lambda p: args.data_root / p.relative_to(DATA_ROOT / "probes")
    for k in ("DETOUR_JSON", "NEW_DIR", "OFF_DIR", "COIN_JSON", "AM_JSON"):
        setattr(v6, k, rel(getattr(v6, k)))

    fs.FS_TICK = fs.FS_LEG = fs.FS_SUB = max(fs.FS_SUB, FS_MIN)
    fs.style()
    maze, coin, ptm = load_maze(), load_coin(), load_ptm()
    maze = {k: v for k, v in maze.items() if k[0] != "Off-screen"}
    res = json.loads(v6.AM_JSON.read_text())["results"]
    lims = am_limits(res)
    d, rates = pd.read_csv(args.endpoints), place_rates(args.rates)
    vals = {}
    for v in args.variants:
        vals[v] = build(v, maze, coin, ptm, res, lims, d, rates, np.random.default_rng(args.seed), args.out_dir
                        / f"fig3_ptm_{v}")
    (args.out_dir / "fig3_ptm_values.json").write_text(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
