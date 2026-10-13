"""Main-text steering figure (Sec. 4.1): local push at beta = 1 per environment, plus the Attribute Maze shared-axis compression at beta = 0.25."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

from paths import DATA_ROOT, OUT_DIR, ROOT, TABLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "procgen"))
from palette import PAL  # noqa: E402
from plot_steering_paper import load_am, load_coin, load_global, load_maze  # noqa: E402

FS_TICK, FS_LAB, FS_TITLE = 7.0, 7.5, 8.0
W_IN, H_IN = 5.5, 2.0
BETA = 1.0
BETA_SHARED = 0.25
CONDS = [("baseline", "none"), ("corrective", "toward goal"), ("anticorrective", "away"), ("random", "random")]
SHARED_CONDS = [("baseline", "none"), ("full_local_fixed", "local"), ("sign_gated", "shared axis"),
                ("fisher_projected", "Fisher-projected"), ("ungated", "unsigned")]
COH = {"GM": PAL["gm"], "Non-GM": PAL["nongm"]}
OFFSET = {"GM": -0.14, "Non-GM": 0.14}
LW = 0.9


def seeds_at(d: dict, arm: str, beta: float) -> np.ndarray:
    return np.array(d["baseline"][0.0] if arm == "baseline" else d[arm][beta])


def mean_ci(v: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% t-interval across checkpoints, clipped to [0, 1] since reach is a proportion."""
    m = float(v.mean())
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, max(0.0, m - h), min(1.0, m + h)


def panel(ax, get, cohorts: list[str], conds: list, beta: float, title: str, record: dict) -> None:
    x = np.arange(len(conds))
    for cohort in cohorts:
        col, d = COH[cohort], get(cohort)
        rec = record.setdefault(title, {"beta": beta}).setdefault(cohort, {})
        off = OFFSET[cohort] if len(cohorts) > 1 else 0.0
        for i, (arm, _) in enumerate(conds):
            v = seeds_at(d, arm, beta)
            m, lo, hi = mean_ci(v)
            rec[arm] = {"mean": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)], "seeds": [round(s, 3) for s in v]}
            xi = x[i] + off
            ax.scatter(np.full(len(v), xi + 0.06), v, s=4, color=col, alpha=0.35, lw=0, zorder=3)
            ax.vlines(xi, lo, hi, color=col, lw=LW, zorder=4)
            ax.plot(xi, m, "o", color=col, ms=3.6, mec="white", mew=0.4, zorder=5)
    ax.set_xticks(x, [lab for _, lab in conds], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(conds) - 0.5)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, length=2.2, width=0.6, pad=1.5)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", color=PAL["ink"], pad=3)


def text_poly(t, r) -> np.ndarray:
    """Display-space corners of a text's true (rotated) box. Axis-aligned extents of rotated labels overlap spuriously."""
    rot = t.get_rotation()
    if rot % 360 == 0:
        e = t.get_window_extent(r)
        return np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]])
    assert t.get_rotation_mode() == "anchor", "only anchor-mode rotation is checked exactly"
    t.set_rotation(0)
    e = t.get_window_extent(r)
    t.set_rotation(rot)
    ax_ = {"left": e.x0, "right": e.x1, "center": (e.x0 + e.x1) / 2}[t.get_ha()]
    ay_ = {"top": e.y1, "bottom": e.y0}.get(t.get_va(), (e.y0 + e.y1) / 2)
    c = np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]]) - [ax_, ay_]
    th = np.deg2rad(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return c @ R.T + [ax_, ay_]


def polys_overlap(a: np.ndarray, b: np.ndarray, tol: float = 0.5) -> bool:
    """Separating-axis test for two convex quadrilaterals; touching within tol pixels is not an overlap."""
    for poly in (a, b):
        for k in range(4):
            edge = poly[(k + 1) % 4] - poly[k]
            n = np.array([-edge[1], edge[0]]) / np.hypot(*edge)
            pa, pb = a @ n, b @ n
            if pa.max() <= pb.min() + tol or pb.max() <= pa.min() + tol:
                return False
    return True


def check_layout(fig, axes) -> None:
    """Assert equal, aligned axes and no overlapping text boxes (tick labels, titles, legend, y label)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [a.get_position() for a in axes]
    for b in boxes[1:]:
        assert np.isclose(b.width, boxes[0].width) and np.isclose(b.height, boxes[0].height), "unequal axes"
        assert np.isclose(b.y0, boxes[0].y0) and np.isclose(b.y1, boxes[0].y1), "misaligned axes"
    texts = []
    for i, a in enumerate(axes):
        texts += [(f"ax{i}:xtick:{t.get_text()}", t) for t in a.get_xticklabels() if t.get_visible() and t.get_text()]
        texts += [(f"ax{i}:ytick:{t.get_text()}", t) for t in a.get_yticklabels() if t.get_visible() and t.get_text()]
        texts.append((f"ax{i}:title", a.title))
    texts.append(("ylabel", axes[0].yaxis.label))
    texts += [(f"legend:{t.get_text()}", t) for t in fig.legends[0].get_texts()]
    ext = [(n, text_poly(t, r)) for n, t in texts]
    fb = fig.bbox
    for n, p in ext:
        assert (p[:, 0].min() >= fb.x0 - 0.5 and p[:, 0].max() <= fb.x1 + 0.5
                and p[:, 1].min() >= fb.y0 - 0.5 and p[:, 1].max() <= fb.y1 + 0.5), f"{n} clipped"
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            if polys_overlap(ext[i][1], ext[j][1]):
                raise AssertionError(f"overlap: {ext[i][0]} vs {ext[j][0]}")
    e = fig.legends[0].get_window_extent(r)
    leg = np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]])
    for n, p in ext:
        if not n.startswith("legend"):
            assert not polys_overlap(p, leg), f"{n} hits legend"
    print(f"layout ok: {len(axes)} axes {boxes[0].width * W_IN:.3f} x {boxes[0].height * H_IN:.3f} in, "
          f"{len(ext)} text boxes, no overlaps")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canon", type=Path, default=DATA_ROOT / "probes/canon_steer_20260924/swap")
    ap.add_argument("--maze", type=Path,
                    default=DATA_ROOT / "probes/maze_local_steering_hfc_20260923/maze_local_steering_hfc.json")
    ap.add_argument("--coin", type=Path,
                    default=DATA_ROOT / "probes/coin_local_steering_canon_20260924/coin_local_steering_canon.json")
    ap.add_argument("--out-stem", type=Path, default=OUT_DIR / "fig_main_sec4_1")
    ap.add_argument("--values-out", type=Path, default=TABLES / "fig_main_sec4_1_values.json")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6,
                         "axes.edgecolor": PAL["ink"], "text.color": PAL["ink"], "xtick.color": PAL["ink"],
                         "ytick.color": PAL["ink"]})
    am, maze, coin = load_am(args.canon, "eval50"), load_maze(args.maze), load_coin(args.coin)
    glob = load_global(args.canon, "primary")
    both = ["GM", "Non-GM"]
    panels = [("(a) Maze", lambda c: maze[c], both, CONDS, BETA),
              ("(b) Coin", lambda c: coin[c], both, CONDS, BETA),
              ("(c) Attribute Maze", lambda c: am[("black", c)], both, CONDS, BETA),
              ("(d) Shared axis", lambda c: glob["black"], ["GM"], SHARED_CONDS, BETA_SHARED)]
    fig, axes = plt.subplots(1, len(panels), figsize=(W_IN, H_IN), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.40, wspace=0.12)
    record: dict = {}
    for ax, (title, get, cohorts, conds, beta) in zip(axes, panels):
        panel(ax, get, cohorts, conds, beta, title, record)
    axes[0].set_ylabel("intended-goal reach", fontsize=FS_LAB, labelpad=2)
    handles = [Line2D([0], [0], color=COH["GM"], marker="o", ms=3.6, mec="white", lw=0, label="GM"),
               Line2D([0], [0], color=COH["Non-GM"], marker="o", ms=3.6, mec="white", lw=0, label="Non-GM"),
               Line2D([0], [0], color=PAL["null"], lw=LW, label="95% CI across checkpoints"),
               Line2D([0], [0], color=PAL["null"], marker="o", ms=2.0, alpha=0.5, lw=0, label="single checkpoint")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=FS_LAB, handletextpad=0.4,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    check_layout(fig, axes)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    args.values_out.write_text(json.dumps({"values": record}, indent=1))
    print("wrote", args.out_stem.with_suffix(".{png,pdf}"), "and", args.values_out)


if __name__ == "__main__":
    main()
