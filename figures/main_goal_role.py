"""Main goal-role figure: Maze, Coin, PushToMarker cube-end smear under frame shifts, and Attribute Maze push-pull."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from role_layout import GOAL, H_IN, PROXY, ROLE, TOP, Y0, fs, v6
from role_layout import (FS_MIN, W_IN, am_limits, am_panel, check_layout, ci_half, group_panel,
                                  handles, load_coin, load_maze, load_ptm, maze_panel)
from paths import DATA_ROOT, OUT_DIR, TABLES

CLIP = (-0.6, 1.6)
# (row label, condition, cohorts). GM without a marker isolates the proxy; under the camera shift the marker is drawn
# at the real centre for both cohorts (the main-text test), and only there does Non-GM have a target.
ROWS = (("camera", "camy", ("gm", "control")), ("base", "shifty_nomarker", ("gm",)),
        ("image", "imgu8_nomarker", ("gm",)))
NAME = {"gm": "GM", "control": "Non-GM"}


def rates(r: pd.DataFrame, cond: str, coh: str) -> dict:
    g = r[(r.env == "PTM") & (r.cond == cond) & (r.cohort == coh)].sort_values("seed")
    return {m: g[m].to_numpy(float) for m in ("reach_table", "reach_alt")}


def smear_panel(ax, d: pd.DataFrame, r: pd.DataFrame, rng) -> dict:
    """Cube end positions projected on the real-centre -> old-point axis (0 = centre, 1 = old point), all seeds pooled.

    Numbers above each strip are the per-seed-mean fraction of episodes ending within 3 cm of that point."""
    y, ticks, out = 0.0, [], {}
    lab = dict(ha="center", va="center", fontsize=FS_MIN, zorder=9,
               bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
    for row, cond, cohs in ROWS:
        y_row = []
        for coh in cohs:
            name = NAME[coh]
            e = d[(d.env == "PTM") & (d.cond == cond) & (d.cohort == coh)]
            a = np.c_[e.alt_x - e.corner_x, e.alt_y - e.corner_y]
            p = np.c_[e.x - e.corner_x, e.y - e.corner_y]
            f = (p * a).sum(1) / (a * a).sum(1)
            ax.scatter(np.clip(f, *CLIP), y + rng.uniform(-0.07, 0.07, len(f)), s=0.7, color=fs.COH[name],
                       alpha=0.22, lw=0, rasterized=True, zorder=3)
            q = rates(r, cond, coh)
            marks = (("reach_table", 0.0), ("reach_alt", 1.0)) if coh == "gm" else (("reach_table", 0.0),)
            for m, x in marks:
                ax.text(x, y + 0.27, f"{q[m].mean():.2f}", color=fs.COH[name], **lab)
            out[f"{cond}/{name}"] = {m: {"per_seed": v.tolist(), "mean": float(v.mean()), "ci95_half": ci_half(v)}
                                     for m, v in q.items()} | {"median_proj": float(np.median(f)), "n": int(len(f))}
            y_row.append(y)
            y -= 0.58
        ticks.append((np.mean(y_row), row))
    ax.axvline(0, color=GOAL, lw=0.9, ls=(0, (2.2, 1.4)), zorder=1)
    ax.axvline(1, color=PROXY, lw=0.9, ls=(0, (2.2, 1.4)), zorder=1)
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks])
    ax.set_ylim(y + 0.58 - 0.17, 0.5)
    ax.set_xlim(CLIP[0] - 0.08, CLIP[1] + 0.08)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["real centre", "old point"])
    ax.spines["left"].set_visible(False)
    fs.finish(ax)
    ax.tick_params(axis="y", length=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0, help="jitter seed for the seed dots and smear strips")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT / "probes",
                    help="data/probes copy")
    ap.add_argument("--endpoints", type=Path, default=ROLE / "place_endpoints.csv.gz")
    ap.add_argument("--rates", type=Path, default=ROLE / "place_rates_per_seed.csv")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "fig_main_goal_role")
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
    d, r = pd.read_csv(args.endpoints), pd.read_csv(args.rates)
    rng = np.random.default_rng(args.seed)

    w, h = W_IN, H_IN
    fig = plt.figure(figsize=(w, h))
    ph = h - Y0 - TOP
    ax_at = lambda x, bw: fig.add_axes([x / w, Y0 / h, bw / w, ph / h])
    x_maze, x_coin, x_pl = 0.36, 1.24, 2.30
    ax_maze, ax_coin, ax_pl = ax_at(x_maze, 0.60), ax_at(x_coin, 0.60), ax_at(x_pl, 1.36)
    ax_am1, ax_am2 = ax_at(3.98, 0.70), ax_at(4.76, 0.70)

    maze_panel(ax_maze, maze)
    ax_maze.set_yticklabels(["detour", "new\ncorner"])
    ax_maze.set_xticks([0, 1])
    ax_maze.set_xticklabels(["0", "1"])
    ax_maze.set_xlabel("fraction of states", labelpad=2)
    group_panel(ax_coin, coin, 1.0, rng)
    ax_coin.set_ylim(0, 7.6)
    ax_coin.set_yticks([1, 3, 5, 7])
    ax_coin.set_ylabel(r"$T_\star$", labelpad=3, rotation=0, va="center", ha="right")
    vals = {"ptm_shift_place": smear_panel(ax_pl, d, r, rng)}
    ax_pl.set_xlabel("cube end position", labelpad=2)

    am = {e: am_panel(ax, res, e, *lims, j == 0) for j, (e, ax) in enumerate((("shape", ax_am1), ("colour", ax_am2)))}
    ax_am1.set_ylabel(r"$S_{\mathrm{comp}}$", labelpad=3)
    a, c = ax_am1.get_position(), ax_am2.get_position()
    ax_am1.set_xlabel(r"$S_{\mathrm{fav}}$", labelpad=1)
    ax_am1.xaxis.label.set_x(((a.x0 + c.x1) / 2 - a.x0) / a.width)

    fs.panel_title(ax_maze, "Maze", "")
    fs.panel_title(ax_coin, "Coin", "")
    fs.panel_title(ax_pl, "PushToMarker", "frame shifts")
    fs.panel_title(ax_am1, "Attribute Maze", "shape", span=ax_am2)
    fs.panel_title(ax_am2, None, "colour")
    for ax, lab, x_in in ((ax_maze, "a", x_maze), (ax_coin, "b", x_coin), (ax_pl, "c", x_pl), (ax_am1, "d", None)):
        dx = -12.0 if x_in is None else -(x_in - {"a": 0.03, "b": 1.00, "c": 1.92}[lab]) * 72
        fs.panel_letter(ax, lab, sub=True, dx=dx)

    ring = lambda col, lab: (Line2D([], [], color=col, lw=0.9, ls=(0, (2.2, 1.4))), lab)
    hs, ls = zip(*[*handles(), ring(GOAL, "real table centre"), ring(PROXY, "old pixel / posture")])
    fs.legend_row(fig, list(hs), list(ls), handlelength=1.6, columnspacing=0.8)
    check_layout(fig)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    vals["am_black"] = am
    (TABLES / "main_goal_role_values.json").write_text(json.dumps(vals, indent=1))
    print("wrote", args.out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
