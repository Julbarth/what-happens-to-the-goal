"""Goal-role figure in the shared main-figure style; provides maze_panel, am_panel, handles and check_layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

import style as fs  # noqa: E402
from style import MS, MS_OPEN, PAL, REF, SEED, W_IN  # noqa: E402
from role_base import ci_half  # noqa: E402
from role_panels import AGENTS, AM_JSON, AM_ROSTER, COL, am_limits, load_coin, load_maze, seed_mean  # noqa: E402
from steering_base import polys_overlap, text_poly  # noqa: E402

from paths import OUT_DIR, TABLES

# Print minimum at \linewidth; the shared style's 6.5pt ticks, legend and sub-labels fall under it.
FS_MIN = 7.0

MAZE_ROWS = (("Detour", "detour"), ("New corner", "new corner"))


def maze_panel(ax, maze: dict):
    off = {"GM": 0.18, "Non-GM": -0.18}
    for gi, (g, _) in enumerate(MAZE_ROWS):
        for c in AGENTS:
            y = -gi + off[c]
            goal, alt = maze[(g, c)]
            ax.plot([alt.mean(), goal.mean()], [y, y], color=COL[c], lw=1.8, alpha=0.2, solid_capstyle="butt", zorder=2)
            for v in (alt, goal):
                ax.scatter(v, np.full_like(v, y), color=COL[c], zorder=3, **SEED)
                ax.errorbar(v.mean(), y, xerr=ci_half(v), fmt="none", ecolor=COL[c], elinewidth=fs.LW_ERR, capsize=0,
                            zorder=4)
            # Open marker is larger and underneath, so goal = alternative still reads as a ring around a dot.
            ax.plot(alt.mean(), y, "o", ms=MS_OPEN, mfc="white", mec=COL[c], mew=0.9, zorder=4.5)
            ax.plot(goal.mean(), y, "o", ms=MS, color=COL[c], mec="white", mew=0.4, zorder=5)
    ax.set_yticks([0, -1])
    ax.set_yticklabels([lab for _, lab in MAZE_ROWS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-1.45, 0.45)
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("fraction of states")


def coin_panel(ax, coin: dict, rng):
    ax.axhline(1, zorder=1, **REF)
    for i, (c, v) in enumerate(coin.items()):
        ax.scatter(i - 0.14 + rng.uniform(-0.05, 0.05, len(v)), v, color=COL[c], zorder=3, **SEED)
        ax.errorbar(i + 0.12, v.mean(), yerr=ci_half(v), fmt="o", ms=MS, color=COL[c], mec="white", mew=0.4,
                    elinewidth=fs.LW_ERR, capsize=0, zorder=5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 7.6)
    # No 0 label: at this panel height it would collide with the 1 that marks the reference line.
    ax.set_yticks([1, 3, 5, 7])
    ax.set_xticks(range(len(coin)))
    ax.set_xticklabels(list(coin))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"$T_\star$", labelpad=2, rotation=0, va="center", ha="right")


def am_panel(ax, res: dict, edit: str, xlim, ylim, ylabel: bool) -> dict:
    """Same marks as role_panels.am_panel, without the in-axes quadrant annotation."""
    ax.fill_between([0, xlim[1]], ylim[0], 0, color=PAL["mute"], alpha=0.06, lw=0, zorder=0)
    stats = {}
    for c, labs in AM_ROSTER.items():
        seeds = []
        for lab in labs:
            rr = [r for r in res[lab] if r["edit"] == edit]
            ax.scatter([r["S_fav"] for r in rr], [r["S_comp"] for r in rr], s=fs.STATE["s"], facecolor=COL[c],
                       edgecolor="none", alpha=0.12, zorder=2, rasterized=True)
            seeds.append((seed_mean(rr, "S_fav"), seed_mean(rr, "S_comp")))
        sx, sy = np.array(seeds).T
        ax.scatter(sx, sy, s=13, facecolor=COL[c], edgecolor="white", linewidth=0.35, marker="D", zorder=4)
        ax.errorbar(sx.mean(), sy.mean(), xerr=ci_half(sx), yerr=ci_half(sy), fmt="o", ms=2.4, color=COL[c],
                    mec="white", mew=0.35, capsize=0, lw=fs.LW_ERR, zorder=5)
        stats[c] = {"seed_S_fav": sx.tolist(), "seed_S_comp": sy.tolist()}
    ax.axhline(0.0, zorder=0, **REF)
    ax.axvline(0.0, zorder=0, **REF)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 4, 8])
    ax.set_yticks([0, -4, -8])
    ax.set_yticklabels(["0", "−4", "−8"] if ylabel else [])
    if ylabel:
        ax.set_ylabel(r"$S_{\mathrm{comp}}$", labelpad=0)
    return stats


def check_layout(fig) -> None:
    """Assert every visible text is >= FS_MIN, inside the figure, and overlaps no other text or the legend."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    leg = fig.legends[0]
    for t in texts:
        assert t.get_fontsize() >= FS_MIN - 1e-9, f"{t.get_text()!r} at {t.get_fontsize()}pt"
    polys = [(t.get_text(), text_poly(t, r), any(t is u for u in leg.get_texts())) for t in texts]
    fb = fig.bbox
    for n, p, _ in polys:
        assert (p[:, 0].min() >= fb.x0 - 0.5 and p[:, 0].max() <= fb.x1 + 0.5
                and p[:, 1].min() >= fb.y0 - 0.5 and p[:, 1].max() <= fb.y1 + 0.5), f"{n!r} clipped"
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys_overlap(polys[i][1], polys[j][1]):
                raise AssertionError(f"overlap: {polys[i][0]!r} vs {polys[j][0]!r}")
    e = leg.get_window_extent(r)
    box = np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]])
    for n, p, in_leg in polys:
        assert in_leg or not polys_overlap(p, box), f"{n!r} hits legend"
    print(f"layout ok: {len(polys)} text boxes, all >= {FS_MIN}pt, no overlaps")


def handles() -> list:
    return [(fs.cohort_handle("GM"), "GM"), (fs.cohort_handle("Non-GM"), "Non-GM"), (fs.init_handle(), "init"),
            (fs.ci_handle(), fs.CI_LABEL)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--height", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=0, help="jitter seed for the coin seed dots")
    ap.add_argument("--letters", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "fig_role_panels_styled")
    args = ap.parse_args()

    fs.FS_TICK = fs.FS_LEG = fs.FS_SUB = max(fs.FS_SUB, FS_MIN)
    fs.style()
    maze, coin = load_maze(), load_coin()
    maze = {k: v for k, v in maze.items() if k[0] != "Off-screen"}
    res = json.loads(AM_JSON.read_text())["results"]
    xlim, ylim = am_limits(res)

    w, h = W_IN, args.height
    fig = plt.figure(figsize=(w, h))
    # Panel boxes in inches (x0, width); the two Attribute Maze panels share y, so the second drops its tick labels.
    # Top 0.28 in holds the two-line Attribute Maze title; the panels take whatever height is left.
    y0 = 0.42
    yh = h - y0 - 0.28
    boxes = [(0.60, 0.98), (1.86, 1.02), (3.22, 1.07), (4.39, 1.07)]
    axes = [fig.add_axes([x / w, y0 / h, bw / w, yh / h]) for x, bw in boxes]

    maze_panel(axes[0], maze)
    coin_panel(axes[1], coin, np.random.default_rng(args.seed))
    am = {e: am_panel(axes[2 + j], res, e, xlim, ylim, j == 0) for j, e in enumerate(("shape", "colour"))}
    # One S_fav label centred under both Attribute Maze panels instead of one per panel.
    a, b = axes[2].get_position(), axes[3].get_position()
    axes[2].set_xlabel(r"$S_{\mathrm{fav}}$", labelpad=1)
    axes[2].xaxis.label.set_x(((a.x0 + b.x1) / 2 - a.x0) / a.width)

    # Every title reserves the sub-label line so the one-row titles share a baseline.
    fs.panel_title(axes[0], "Maze", "")
    fs.panel_title(axes[1], "Coin", "")
    fs.panel_title(axes[2], "Attribute Maze", "shape swap", span=axes[3])
    fs.panel_title(axes[3], None, "colour swap")
    if args.letters:
        # Maze's category labels sit left of its (hidden) spine, so (a) goes to the figure edge instead.
        for ax, lab, dx in ((axes[0], "a", -(boxes[0][0] - 0.03) * 72), (axes[1], "b", fs.LETTER_DX),
                            (axes[2], "c", fs.LETTER_DX - 6)):
            fs.panel_letter(ax, lab, sub=True, dx=dx)

    fs.legend_row(fig, *zip(*handles()))
    check_layout(fig)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", args.out.with_suffix(".pdf"))

    vals = {"maze": {f"{g}|{c}": {"goal_side": a.tolist(), "alternative": b.tolist(),
                                  "mean": [a.mean(), b.mean()], "ci95_half": [ci_half(a), ci_half(b)]}
                     for (g, c), (a, b) in maze.items()},
            "coin_r_star": {c: {"per_seed": v.tolist(), "mean": v.mean(), "ci95_half": ci_half(v)} for c, v in coin.items()},
            "am_black": am, "am_limits": [list(xlim), list(ylim)]}
    (TABLES / "role_panels_styled_values.json").write_text(json.dumps(vals, indent=1))


if __name__ == "__main__":
    main()
