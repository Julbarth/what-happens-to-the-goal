"""Main-text Sec. 4.1 dose-response figure: state-local steering in Maze, PushToMarker, Coin, Attribute Maze (black), plus the shared-axis compression, GM over Non-GM."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

from paths import DATA_ROOT, OUT_DIR, ROOT, TABLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "procgen"))
import style as fs  # noqa: E402
from steering_base import polys_overlap, text_poly  # noqa: E402
from palette import PAL  # noqa: E402
from plot_steering_paper import ARMS, load_am, load_coin, load_maze  # noqa: E402

# Ticks and legend floored at 7pt (the shared style's 6.5pt falls under the print minimum at \linewidth).
FS_TICK, FS_LAB, FS_TITLE, FS_LEG = max(fs.FS_TICK, 7.0), fs.FS, fs.FS_TITLE, max(fs.FS_LEG, 7.0)
W_IN, H_IN = fs.W_IN, 1.9
LW, MS = fs.LW_DATA, fs.MS - 0.6
COHORTS = ("GM", "Non-GM")
CSV_COHORT = {"GM": "GM", "Non-GM": "deco"}
HUE = {"GM": (PAL["gm"], PAL["gm_proxy"]), "Non-GM": (PAL["nongm"], PAL["nongm_proxy"])}
# Same arm -> (colour role, line style, marker) mapping as plot_steering_paper.COMP_ARMS, minus full local and random.
# (a-c) and (d) arms share markers ("corrective" vs "fixed at s0"), so they need separate legend rows to stay unambiguous.
LOCAL_LABELS = {"corrective": "corrective", "anticorrective": "reversed", "random": "random"}
SHARED_ARMS = (("full_local_fixed", r"fixed at $s_0$", "hue", "--", "o"),
               ("sign_gated", "sign-gated", "soft", "-", "D"),
               ("fisher_projected", "Fisher-projected", "soft", "--", "D"),
               ("ungated", "unsigned", "ink", ":", "v"))

# Layout in inches; every panel gets the same box.
L, R, T, B_LEG, B_AX = 0.62, 0.04, 0.18, 0.25, 0.27
GAP_X, GAP_Y = 0.17, 0.07


def load_global_cohort(canon: Path, population: str, cohort: str, bg: str) -> dict:
    """plot_steering_paper.load_global for one cohort and background: {arm: {signed beta: [per-seed reach]}}."""
    out: dict = defaultdict(lambda: defaultdict(list))
    seen: set = set()
    for r in csv.DictReader(open(canon / "global" / "global_per_seed.csv")):
        if (r["population"] != population or r["cohort"] != cohort or r["background"] != bg
                or r["sign"] != "all" or r["d_stratum"] != "all"):
            continue
        arm, b = r["arm"], float(r["beta"])
        if arm == "baseline":
            # Baseline rows repeat per beta; keep one per seed.
            if r["seed"] not in seen:
                seen.add(r["seed"])
                out["baseline"][0.0].append(float(r["reach"]))
            continue
        anti = arm.endswith("_anti")
        out[arm[:-5] if anti else arm][-b if anti else b].append(float(r["reach"]))
    return out


def mean_ci(v) -> tuple[float, float, float]:
    """Mean and 95% t-interval across checkpoints, clipped to [0, 1] since reach is a proportion."""
    if isinstance(v, tuple):
        return v
    v = np.asarray(v, float)
    m = float(v.mean())
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, max(0.0, m - h), min(1.0, m + h)


def style_ax(ax) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, length=2, width=fs.LW_AX, pad=1.5, colors=PAL["ink"])


def baseline_band(ax, base, rec: dict) -> None:
    m, lo, hi = mean_ci(base)
    rec["baseline"] = {"mean": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    if not isinstance(base, tuple):
        rec["baseline"]["seeds"] = [round(s, 3) for s in base]
    ax.axhspan(lo, hi, color=PAL["init"], alpha=0.15, lw=0, zorder=1)
    ax.axhline(m, color=PAL["init"], lw=0.6, ls=":", zorder=1)


def local_panel(ax, data: dict, cohort: str, rec: dict) -> None:
    """Appendix _panel restyled: same arm encodings, thin lines, 95% CI tunnels."""
    hue, soft = HUE[cohort]
    baseline_band(ax, data["baseline"][0.0], rec)
    style = {"corrective": (hue, "-", "o"), "anticorrective": (soft, "--", "s"), "random": (PAL["null"], "-.", "^")}
    for arm in ARMS:
        betas = sorted(data[arm])
        ci = np.array([mean_ci(data[arm][b]) for b in betas])
        rec[arm] = {str(b): {"mean": round(c[0], 3), "ci95": [round(c[1], 3), round(c[2], 3)]} for b, c in zip(betas, ci)}
        c, ls, mk = style[arm]
        ax.fill_between(betas, ci[:, 1], ci[:, 2], color=c, alpha=fs.CI_ALPHA, lw=0, zorder=2)
        ax.plot(betas, ci[:, 0], color=c, ls=ls, marker=mk, ms=MS, lw=LW, zorder=3,
                mfc="white" if arm == "anticorrective" else c, mec=c, mew=0.7)
    bmax = max(max(data[a]) for a in ARMS)
    ax.set_xlim(0, bmax * 1.04)
    ticks = [0, 2, 4] if bmax > 2 else [0, 1, 2] if bmax > 1 else [0, 0.5, 1]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    style_ax(ax)


def load_values(path: Path, title: str, cohort: str) -> dict:
    """Panel data from a saved values JSON, as precomputed (mean, lo, hi) tuples that mean_ci passes through.

    The JSON keeps every plotted mean and 95% t-CI, so the per-seed Procgen steering CSVs are not needed."""
    rec = json.load(open(path))["values"][title][cohort]
    out: dict = defaultdict(dict)
    b = rec["baseline"]
    out["baseline"][0.0] = (b["mean"], *b["ci95"])
    for arm, pts in rec.items():
        if arm != "baseline":
            for x, c in pts.items():
                if float(x) != 0.0:
                    out[arm][float(x)] = (c["mean"], *c["ci95"])
    return out


def robot_local_data(ps: pd.DataFrame, env: str, cohort: str) -> dict:
    """Robot per-seed CSV in the {arm: {beta: [per-seed reach]}} shape of the Procgen loaders; reversed = local at -beta."""
    q = ps[(ps.env == env) & (ps.axis == "none") & (ps.cohort == {"GM": "gm", "Non-GM": "control"}[cohort])]
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in q.sort_values("seed").itertuples():
        if r.arm == "baseline":
            out["baseline"][0.0].append(r.reach)
        elif r.arm == "random":
            out["random"][r.beta].append(r.reach)
        elif r.arm == "local":
            out["corrective" if r.beta > 0 else "anticorrective"][abs(r.beta)].append(r.reach)
    return out


def shared_panel(ax, d: dict, cohort: str, rec: dict) -> None:
    """Appendix fig_compression restyled: signed dose, baseline anchors beta = 0."""
    hue, soft = HUE[cohort]
    col = {"hue": hue, "soft": soft, "ink": PAL["ink"]}
    base = d["baseline"][0.0]
    baseline_band(ax, base, rec)
    for arm, _, role, ls, mk in SHARED_ARMS:
        xs = sorted(list(d[arm]) + [0.0])
        ci = np.array([mean_ci(base if x == 0.0 else d[arm][x]) for x in xs])
        rec[arm] = {str(x): {"mean": round(c[0], 3), "ci95": [round(c[1], 3), round(c[2], 3)]} for x, c in zip(xs, ci)}
        c = col[role]
        ax.fill_between(xs, ci[:, 1], ci[:, 2], color=c, alpha=0.07, lw=0, zorder=2)
        ax.plot(xs, ci[:, 0], color=c, ls=ls, marker=mk, ms=MS, lw=LW, mec=c, zorder=3)
    ax.axvline(0, color=PAL["ink"], lw=0.5, zorder=1)
    ax.set_xlim(-1.08, 1.08)
    ax.set_xticks([-1, 0, 1], ["−1", "0", "1"])
    style_ax(ax)


def pending_panel(ax, _d, _cohort, _rec) -> None:
    ax.text(0.5, 0.5, "updating", transform=ax.transAxes, ha="center", va="center",
            fontsize=FS_LAB, color="#c0392b", style="italic")
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    style_ax(ax)


def check_layout(fig, grid) -> None:
    """Assert equal, row/column-aligned axes and no overlapping or clipped text (ticks, titles, labels, legends)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pos = [[a.get_position() for a in row] for row in grid]
    w0, h0 = pos[0][0].width, pos[0][0].height
    for i, row in enumerate(pos):
        for j, p in enumerate(row):
            assert np.isclose(p.width, w0) and np.isclose(p.height, h0), f"unequal axes ({i},{j})"
            assert np.isclose(p.y0, row[0].y0), f"row {i} misaligned at col {j}"
            assert np.isclose(p.x0, pos[0][j].x0), f"col {j} misaligned at row {i}"
    texts = []
    for i, row in enumerate(grid):
        for j, a in enumerate(row):
            tag = f"ax{i}{j}"
            texts += [(f"{tag}:xtick:{t.get_text()}", t) for t in a.get_xticklabels() if t.get_visible() and t.get_text()]
            texts += [(f"{tag}:ytick:{t.get_text()}", t) for t in a.get_yticklabels() if t.get_visible() and t.get_text()]
            for kind, t in (("title", a.title), ("xlabel", a.xaxis.label), ("ylabel", a.yaxis.label)):
                if t.get_text():
                    texts.append((f"{tag}:{kind}", t))
    texts += [(f"fig:{t.get_text()}", t) for t in fig.texts]
    for k, leg in enumerate(fig.legends):
        texts += [(f"legend{k}:{t.get_text()}", t) for t in leg.get_texts()]
    ext = [(n, text_poly(t, r)) for n, t in texts]
    fb = fig.bbox
    for n, p in ext:
        assert (p[:, 0].min() >= fb.x0 - 0.5 and p[:, 0].max() <= fb.x1 + 0.5
                and p[:, 1].min() >= fb.y0 - 0.5 and p[:, 1].max() <= fb.y1 + 0.5), f"{n} clipped"
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            if polys_overlap(ext[i][1], ext[j][1]):
                raise AssertionError(f"overlap: {ext[i][0]} vs {ext[j][0]}")
    boxes = []
    for leg in fig.legends:
        e = leg.get_window_extent(r)
        boxes.append(np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]]))
    for a in (a for row in grid for a in row):
        e = a.get_window_extent(r)
        axp = np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]])
        for lb in boxes:
            assert not polys_overlap(axp, lb), "legend overlaps an axes"
    for n, p in ext:
        if not n.startswith("legend"):
            for lb in boxes:
                assert not polys_overlap(p, lb), f"{n} hits legend"
    print(f"layout ok: {sum(map(len, grid))} axes {w0 * W_IN:.3f} x {h0 * H_IN:.3f} in, "
          f"{len(ext)} text boxes, no overlaps or clipping")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canon", type=Path, default=DATA_ROOT / "probes/canon_steer_20260924/swap")
    ap.add_argument("--maze-json", type=Path,
                    default=DATA_ROOT / "probes/maze_local_steering_hfc_large_20260925/maze_local_steering_hfc_large.json")
    ap.add_argument("--coin-json", type=Path,
                    default=DATA_ROOT / "probes/coin_local_steering_realcoin_20260925/coin_local_steering_realcoin.json")
    ap.add_argument("--procgen-values", type=Path, default=None,
                    help="draw the Procgen panels from this values JSON (4-panel render) instead of --canon/--maze-json/--coin-json")
    ap.add_argument("--robot-csv", type=Path, default=TABLES / "robot_steering_per_seed.csv")
    ap.add_argument("--coin-pending", action="store_true",
                    help="draw (b) empty with a pending note instead of the Coin data")
    ap.add_argument("--population", default="eval50", choices=["eval50", "all196"])
    ap.add_argument("--out-stem", type=Path, default=OUT_DIR / "fig_main_sec4_1_dose")
    ap.add_argument("--values-out", type=Path, default=TABLES / "fig_main_sec4_1_dose_values.json")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans", "axes.linewidth": 0.6,
                         "axes.edgecolor": PAL["ink"], "text.color": PAL["ink"], "xtick.color": PAL["ink"],
                         "ytick.color": PAL["ink"]})
    if args.procgen_values:
        pv = args.procgen_values
        maze = {c: load_values(pv, "(a) Maze", c) for c in COHORTS}
        coin = {c: load_values(pv, "(b) Coin", c) for c in COHORTS}
        am = {("black", c): load_values(pv, "(c) Attribute Maze", c) for c in COHORTS}
        shared = {c: load_values(pv, "(d) Shared axis", c) for c in COHORTS}
    else:
        am, maze, coin = load_am(args.canon, args.population), load_maze(args.maze_json), load_coin(args.coin_json)
        shared = {c: load_global_cohort(args.canon, "primary", CSV_COHORT[c], "black") for c in COHORTS}
    ptm = {c: robot_local_data(pd.read_csv(args.robot_csv), "PTM", c) for c in COHORTS}
    cols = [("(a) Maze", lambda c: maze[c], local_panel),
            ("(b) PushToMarker", lambda c: ptm[c], local_panel),
            ("(c) Coin", lambda c: coin[c], pending_panel if args.coin_pending else local_panel),
            ("(d) Attribute Maze", lambda c: am[("black", c)], local_panel),
            ("(e) Shared axis", lambda c: shared[c], shared_panel)]
    n = len(cols)

    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
    w = (W_IN - L - R - (n - 1) * GAP_X) / n
    h = (H_IN - T - B_LEG - B_AX - GAP_Y) / 2
    grid, record = [], {}
    for i, cohort in enumerate(COHORTS):
        row = []
        y0 = B_LEG + B_AX + (1 - i) * (h + GAP_Y)
        for j, (title, get, fn) in enumerate(cols):
            ax = fig.add_axes([(L + j * (w + GAP_X)) / W_IN, y0 / H_IN, w / W_IN, h / H_IN])
            fn(ax, get(cohort), cohort, record.setdefault(title, {}).setdefault(cohort, {}))
            if i == 0:
                letter, env = title[1], title[4:]
                # Five panels leave no room for centred titles beside the letters, so titles start right after them.
                dx = -9 if j else fs.LETTER_DX
                fs.panel_letter(ax, letter, dx=dx)
                ax.annotate(env, xy=(0, 1), xycoords="axes fraction", xytext=(dx + 13.5, fs.TITLE_PAD),
                            textcoords="offset points", ha="left", va="bottom", fontsize=fs.FS_TITLE,
                            fontweight=fs.TITLE_WEIGHT, color=PAL["ink"])
                ax.tick_params(labelbottom=False)
            if j:
                ax.tick_params(labelleft=False)
            row.append(ax)
        grid.append(row)
    for i, cohort in enumerate(COHORTS):
        p = grid[i][0].get_position()
        fig.text(0.08 / W_IN, (p.y0 + p.y1) / 2, cohort, rotation=90, rotation_mode="anchor", ha="center",
                 va="center", fontsize=FS_TITLE, fontweight=fs.TITLE_WEIGHT, color=HUE[cohort][0])
    # One y label for both rows and one x label per dose scale, instead of repeating them on every panel.
    top, bot = grid[0][0].get_position(), grid[1][0].get_position()
    fig.text(0.30 / W_IN, (bot.y0 + top.y1) / 2, "goal reach", rotation=90, rotation_mode="anchor", ha="center", va="center",
             fontsize=FS_LAB)
    y_xlab = (B_LEG + 0.005) / H_IN
    for lab, a0, a1 in ((r"dose $\beta$", grid[1][0], grid[1][n - 2]), (r"signed dose $\beta$", grid[1][n - 1], grid[1][n - 1])):
        fig.text((a0.get_position().x0 + a1.get_position().x1) / 2, y_xlab, lab, ha="center", va="bottom",
                 fontsize=FS_LAB)

    ink = PAL["ink"]
    h_local = [Line2D([], [], color=ink, ls="-", marker="o", ms=MS, lw=LW),
               Line2D([], [], color=ink, ls="--", marker="s", ms=MS, lw=LW, mfc="white"),
               Line2D([], [], color=PAL["null"], ls="-.", marker="^", ms=MS, lw=LW)]
    h_shared = [Line2D([], [], color=ink, ls=ls, marker=mk, ms=MS, lw=LW) for _, _, _, ls, mk in SHARED_ARMS]
    x_mid = (L + (W_IN - L - R) / 2) / W_IN
    rows = [(fs.legend_row(fig, h_local, [LOCAL_LABELS[a] for a in ARMS], y_in=0.125, x=x_mid, fontsize=FS_LEG), "(a–d)"),
            (fs.legend_row(fig, h_shared, [lab for _, lab, _, _, _ in SHARED_ARMS], y_in=0.015, x=x_mid, fontsize=FS_LEG), "(e)")]
    fig.canvas.draw()
    x_left = min(leg.get_window_extent().x0 for leg, _ in rows) / fig.dpi
    for leg, lab in rows:
        e = leg.get_window_extent()
        fig.text((x_left - 0.04) / W_IN, (e.y0 + e.y1) / 2 / fig.dpi / H_IN, lab, ha="right", va="center",
                 fontsize=FS_LEG, fontweight="bold")

    check_layout(fig, grid)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_stem.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)
    args.values_out.write_text(json.dumps({"maze_json": str(args.maze_json), "coin_json": str(args.coin_json),
                                           "robot_csv": str(args.robot_csv),
                                           "procgen_values": str(args.procgen_values),
                                           "band": "95% t-CI across checkpoints", "values": record}, indent=1))
    print("wrote", args.out_stem.with_suffix(".{png,pdf}"), "and", args.values_out)


if __name__ == "__main__":
    main()
