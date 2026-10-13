"""Goal-role figure variant with Coin T* beside PushToMarker signed step follow; provides load_ptm and group_panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style as fs  # noqa: E402
from style import MS, REF, SEED, W_IN  # noqa: E402
import role_panels as v6  # noqa: E402
from role_base import ci_half  # noqa: E402
from role_panels import COL, am_limits, load_coin, load_maze  # noqa: E402
from role_panels_styled import FS_MIN, am_panel, check_layout, handles, maze_panel  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

PTM_CSV = TABLES / "robot_step_follow_per_seed.csv"
PTM_COH = {"GM": "gm", "Non-GM": "control", "init": "init"}


def load_ptm() -> dict:
    """Per-seed mean signed step follow (mm) of the PushToMarker marker move, standard bank (not the hover bank)."""
    d = pd.read_csv(PTM_CSV)
    d = d[(d.task == "PTM") & (d.edit == "marker_move")]
    return {k: d[d.cohort == v].sort_values("seed").step_follow_mm.to_numpy(float) for k, v in PTM_COH.items()}


def group_panel(ax, vals: dict, ref: float, rng) -> None:
    """Seed dots left, mean ± 95% CI right, one column per cohort; category names come from the legend colours."""
    ax.axhline(ref, zorder=1, **REF)
    for i, (c, v) in enumerate(vals.items()):
        ax.scatter(i - 0.16 + rng.uniform(-0.05, 0.05, len(v)), v, color=COL[c], zorder=3, **SEED)
        ax.errorbar(i + 0.14, v.mean(), yerr=ci_half(v), fmt="o", ms=MS, color=COL[c], mec="white", mew=0.4,
                    elinewidth=fs.LW_ERR, capsize=0, zorder=5)
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_xticks([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--height", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=0, help="jitter seed for the seed dots in (b)")
    ap.add_argument("--letters", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "fig_role_panels_ptm")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT / "probes",
                    help="data/probes copy")
    args = ap.parse_args()
    # The role_panels loaders read module-level paths, so repoint them at the chosen data root.
    rel = lambda p: args.data_root / p.relative_to(DATA_ROOT / "probes")
    for k in ("DETOUR_JSON", "NEW_DIR", "OFF_DIR", "COIN_JSON", "AM_JSON"):
        setattr(v6, k, rel(getattr(v6, k)))

    fs.FS_TICK = fs.FS_LEG = fs.FS_SUB = max(fs.FS_SUB, FS_MIN)
    fs.style()
    maze, coin, ptm = load_maze(), load_coin(), load_ptm()
    maze = {k: v for k, v in maze.items() if k[0] != "Off-screen"}
    res = json.loads(v6.AM_JSON.read_text())["results"]
    xlim, ylim = am_limits(res)

    w, h = W_IN, args.height
    fig = plt.figure(figsize=(w, h))
    y0 = 0.42
    yh = h - y0 - 0.28
    # (a) and (c) boxes match role_panels_styled.py; (b) splits its 1.86..2.88 slot into two narrow axes, each with its own y-axis.
    boxes = [(0.60, 0.98), (1.86, 0.36), (2.55, 0.36), (3.22, 1.07), (4.39, 1.07)]
    axes = [fig.add_axes([x / w, y0 / h, bw / w, yh / h]) for x, bw in boxes]
    ax_maze, ax_coin, ax_ptm, ax_am1, ax_am2 = axes

    maze_panel(ax_maze, maze)
    rng = np.random.default_rng(args.seed)
    group_panel(ax_coin, coin, 1.0, rng)
    ax_coin.set_ylim(0, 7.6)
    ax_coin.set_yticks([1, 3, 5, 7])
    ax_coin.set_ylabel(r"$T_\star$", labelpad=2, rotation=0, va="center", ha="right")
    ax_coin.set_xlabel("Coin", labelpad=3)
    group_panel(ax_ptm, ptm, 0.0, rng)
    ax_ptm.set_ylim(-12, 66)
    ax_ptm.set_yticks([0, 30, 60])
    ax_ptm.set_ylabel("follow (mm)", labelpad=1)
    ax_ptm.set_xlabel("PushToMarker", labelpad=3)

    am = {e: am_panel(axes[3 + j], res, e, xlim, ylim, j == 0) for j, e in enumerate(("shape", "colour"))}
    a, b = ax_am1.get_position(), ax_am2.get_position()
    ax_am1.set_xlabel(r"$S_{\mathrm{fav}}$", labelpad=1)
    ax_am1.xaxis.label.set_x(((a.x0 + b.x1) / 2 - a.x0) / a.width)

    fs.panel_title(ax_maze, "Maze", "")
    # The question is wider than the (b) slot between the (b) and (c) letters, so it wraps onto the sub-label row,
    # which is free there and lets the longer half run under the (c) letter.
    fs.panel_title(ax_coin, "Is the weak", "influence directed?", span=ax_ptm)
    p, q = ax_coin.get_position(), ax_ptm.get_position()
    for t in [t for t in ax_coin.texts if t.get_text() == "influence directed?"]:
        t.xy = (((p.x0 + q.x1) / 2 - p.x0) / p.width, 1)
        t.set_fontsize(fs.FS_TITLE)
    fs.panel_title(ax_am1, "Attribute Maze", "shape swap", span=ax_am2)
    fs.panel_title(ax_am2, None, "colour swap")
    if args.letters:
        for ax, lab, dx in ((ax_maze, "a", -(boxes[0][0] - 0.03) * 72), (ax_coin, "b", -18.0),
                            (ax_am1, "c", fs.LETTER_DX - 6)):
            fs.panel_letter(ax, lab, sub=True, dx=dx)

    fs.legend_row(fig, *zip(*handles()))
    check_layout(fig)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", args.out.with_suffix(".pdf"))

    summ = lambda d: {c: {"per_seed": v.tolist(), "mean": float(v.mean()), "ci95_half": ci_half(v)} for c, v in d.items()}
    vals = {"maze": {f"{g}|{c}": {"goal_side": a.tolist(), "alternative": b.tolist(),
                                  "mean": [a.mean(), b.mean()], "ci95_half": [ci_half(a), ci_half(b)]}
                     for (g, c), (a, b) in maze.items()},
            "coin_t_star": summ(coin), "ptm_step_follow_mm": summ(ptm), "am_black": am,
            "am_limits": [list(xlim), list(ylim)]}
    (TABLES / "role_panels_ptm_values.json").write_text(json.dumps(vals, indent=1))
    for k in ("coin_t_star", "ptm_step_follow_mm"):
        for c, s in vals[k].items():
            print(f"{k:20s} {c:7s} seeds={np.round(s['per_seed'], 3).tolist()} mean={s['mean']:.3f} ±{s['ci95_half']:.3f}")


if __name__ == "__main__":
    main()
