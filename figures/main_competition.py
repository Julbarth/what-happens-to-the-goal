"""Main-text Sec. 3.4 four-panel figure (within-state cosines, kappa_E/kappa_F heatmaps, tug of war) on one GridSpec, shared main-figure style."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from paths import OUT_DIR, ROOT, TABLES

sys.path.insert(0, str(ROOT / "scripts" / "procgen"))
import style as fs  # noqa: E402
from palette import PAL  # noqa: E402
from canon_geometry_run import BANK_DIR, load_state_table  # noqa: E402
import competition_base as m3  # noqa: E402
import competition_tug_of_war as tw  # noqa: E402

FS_TICK, FS_LABEL, FS_TITLE = fs.FS_TICK, fs.FS, fs.FS_TITLE
LW = fs.LW_AX
IDENTITY = fs.REF
m3.FS_TICK, m3.FS_LABEL, m3.FS_TITLE, m3.LW = FS_TICK, FS_LABEL, FS_TITLE, LW
m3.MS, m3.CAP, m3.SEED_S, m3.SEED_A, m3.ELW = fs.MS, 0, fs.SEED["s"], fs.SEED["alpha"], fs.LW_ERR
TUG = [("gm", "GM", "gm"), ("deco", "Non-GM", "nongm")]

# Layout in inches. Every axes shares one y0..y1; heatmap columns are as wide as they are tall so they stay square.
# At 8pt bold a ~1.05 in panel cannot hold "(b) Across states: raw", so R titles (b)/(c) are short and one
# "across states" label spans both.
# Variant R makes all four panels square and equal ("p4": None and "sq" columns share the leftover width).
W = 5.5
LAYOUTS = {
    "S": dict(H=1.9, top=1.71, cols=[("left", 0.47), ("p1", 0.95), ("gap", 0.22), ("hE", "sq"), ("gap", 0.07),
                                     ("hF", "sq"), ("gap", 0.05), ("cb", 0.07), ("gap", 0.54), ("p4", None),
                                     ("right", 0.10)], side=1.14),
    "R": dict(H=1.9, top=1.62, cols=[("left", 0.45), ("p1", "sq"), ("gap", 0.15), ("hE", "sq"), ("gap", 0.04),
                                     ("hF", "sq"), ("gap", 0.03), ("cb", 0.05), ("gap", 0.53), ("p4", "sq"),
                                     ("right", 0.05)], side=None),
}
# Main: four equal squares; the one colorbar lies horizontally under (b)+(c) so no column is spent on it.
LAYOUTS["main"] = dict(H=1.82, top=1.60, cols=[("left", 0.41), ("p1", "sq"), ("gap", 0.13), ("hE", "sq"),
                                              ("gap", 0.03), ("hF", "sq"), ("gap", 0.34), ("p4", "sq"),
                                              ("right", 0.03)], side=None, cb_below=(0.06, 0.045))
SQUARE = {"R": ("p1", "kE", "kF", "p4"), "main": ("p1", "kE", "kF", "p4"), "S": ("kE", "kF")}
EQUAL_BOTTOM = {"R": ("p1", "kE", "kF", "cbar", "p4"), "main": ("p1", "kE", "kF", "p4"),
                "S": ("p1", "kE", "kF", "cbar", "p4")}
STEMS = {"R": "fig_main_sec3_4_4panel_R", "S": "fig_main_sec3_4_4panel_S", "main": "fig_main_sec3_4"}
TITLES = {"S": ("Within a state", r"Across, $\kappa_E$", r"Across, $\kappa_F$", "Tug of war"),
          "R": ("(a) Within a state", "(b) Raw", "(c) Policy", "(d) Tug of war"),
          "main": ("Within a state", "Euclidean", "Fisher", "Tug of war")}
# Main variant: letters sit outside the panels, apart from the title. (c) abuts (b), so it sits over its own corner.
LETTERS = {"main": (("a", fs.LETTER_DX - 4), ("b", -12), ("c", -2), ("d", fs.LETTER_DX - 4))}


def layout(fig, variant: str):
    L = LAYOUTS[variant]
    cols = L["cols"]
    fixed = sum(w for _, w in cols if isinstance(w, float))
    n_sq = sum(w == "sq" for _, w in cols)
    side = L["side"] if L["side"] is not None else (W - fixed) / n_sq
    rest = W - fixed - n_sq * side
    widths = [w if isinstance(w, float) else side if w == "sq" else rest for _, w in cols]
    top, H = L["top"], L["H"]
    rows = [side, *L.get("cb_below", ())]
    gs = fig.add_gridspec(len(rows), len(cols), width_ratios=widths, height_ratios=rows, left=0, right=1, wspace=0,
                          hspace=0, top=top / H, bottom=(top - sum(rows)) / H)
    idx = {name: i for i, (name, _) in enumerate(cols)}
    cells = {name: gs[0, i] for name, i in idx.items() if name in ("p1", "hE", "hF", "cb", "p4")}
    if "cb_below" in L:
        cells["cb"] = gs[2, idx["hE"]:idx["hF"] + 1]
    return cells


def set_title(ax, title: str) -> None:
    ax.set_title("")
    fs.panel_title(ax, title)


def finish(ax, title: str) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, length=2, width=LW, pad=1.5)
    set_title(ax, title)


def panel_ratio(ax, summary: dict, record: dict, title: str) -> None:
    """Colour/shape pull ratio (ratio of means over the 50 eval states) per seed, with mean and 95% t-CI."""
    lo_edge, hi_edge = 0.012, 15.0
    for i, (co, name, col) in enumerate(TUG):
        v = summary["eval50"][co]["F_ratio_of_means"]
        seeds = list(v["per_seed"].values())
        assert len(seeds) == 3
        m, h = m3.t_ci(seeds)
        assert abs(m - v["mean"]) < 1e-9 and abs(m - h - v["lo"]) < 1e-6, co
        ax.scatter(np.full(3, i - 0.2), seeds, s=2.0, color=PAL[col], alpha=0.45, lw=0, zorder=3)
        # A negative lower CI bound has no log position; the whisker runs to the axis edge and is clipped there.
        lo = max(m - h, lo_edge * 1e-2)
        ax.errorbar(i + 0.08, m, yerr=[[m - lo], [m + h - m]], fmt="o", ms=fs.MS, mfc=PAL[col], mec=PAL[col],
                    mew=LW, color=PAL[col], capsize=0, elinewidth=fs.LW_ERR, zorder=4)
        if m - h <= lo_edge:
            ax.plot(i + 0.08, lo_edge, marker="v", ms=3.2, color=PAL[col], mew=0, clip_on=False, zorder=5)
        record[f"ratio|{name}"] = {"mean": round(m, 3), "ci95": round(h, 3), "lo": round(m - h, 3),
                                   "hi": round(m + h, 3), "seeds": [round(s, 3) for s in seeds],
                                   "lower_whisker_clipped": bool(m - h <= lo_edge)}
    ax.axhline(1.0, zorder=1, **IDENTITY)
    ax.set_yscale("log")
    ax.set_ylim(lo_edge, hi_edge)
    ax.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks([0, 1], [n for _, n, _ in TUG])
    ax.set_xlim(-0.55, 1.45)
    ax.set_ylabel(r"$\alpha_F(\Delta_p)\,/\,\alpha_F(\Delta_\star)$", fontsize=FS_LABEL, labelpad=1.5)
    finish(ax, title)
    ax.tick_params(axis="x", labelsize=FS_LABEL)


def panel_scatter(ax, df, summary: dict, record: dict, title: str, edge_arrows: bool = False,
                  labels: tuple[str, str] = (r"shape $\alpha_F(\Delta_\star)$", r"colour $\alpha_F(\Delta_p)$")) -> None:
    """Per-state shape vs colour pull on log-log axes; non-positive or sub-edge values sit on the lower edge."""
    lo_edge, hi_edge = 0.1, 70.0
    ev = df[df.split_role == "eval"]
    ax.plot([lo_edge, hi_edge], [lo_edge, hi_edge], zorder=1, **IDENTITY)
    for co, name, col in TUG:
        g = ev[ev.cohort == co]
        x, y = g.aF_star.to_numpy(), g.aF_p.to_numpy()
        clip = (x < lo_edge) | (y < lo_edge)
        ax.scatter(np.clip(x, lo_edge, None), np.clip(y, lo_edge, None), s=fs.STATE["s"], color=PAL[col], alpha=0.35, lw=0,
                   zorder=2, rasterized=True, clip_on=False)
        sx, sy = summary["eval50"][co]["F_mean_a_star"], summary["eval50"][co]["F_mean_a_p"]
        ax.errorbar(sx["mean"], sy["mean"],
                    xerr=[[sx["mean"] - max(sx["lo"], lo_edge)], [sx["hi"] - sx["mean"]]],
                    yerr=[[sy["mean"] - max(sy["lo"], lo_edge)], [sy["hi"] - sy["mean"]]], fmt="o", ms=fs.MS,
                    mfc=PAL[col], mec=PAL["paper"], mew=0.4, color=PAL[col], capsize=0, elinewidth=fs.LW_ERR,
                    zorder=4)
        if edge_arrows and sy["lo"] < lo_edge:
            ax.plot(sx["mean"], lo_edge, marker="v", ms=3.2, color=PAL[col], mew=0, clip_on=False, zorder=5)
        if edge_arrows and sx["lo"] < lo_edge:
            ax.plot(lo_edge, sy["mean"], marker="<", ms=3.2, color=PAL[col], mew=0, clip_on=False, zorder=5)
        record[f"scatter|{name}"] = {"n_points": int(len(g)), "n_clipped_to_edge": int(clip.sum()),
                                     "n_nonpositive": int(((x <= 0) | (y <= 0)).sum()),
                                     "a_star_mean": round(sx["mean"], 3), "a_p_mean": round(sy["mean"], 3),
                                     "n_x_below_edge": int((x < lo_edge).sum()), "n_y_below_edge": int((y < lo_edge).sum()),
                                     "a_star_ci": [round(sx["lo"], 3), round(sx["hi"], 3)],
                                     "a_p_ci": [round(sy["lo"], 3), round(sy["hi"], 3)],
                                     "a_p_lo_clipped": bool(sy["lo"] < lo_edge)}
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_edge, hi_edge)
    ax.set_ylim(lo_edge, hi_edge)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([1, 10], ["1", "10"])
        axis.set_minor_locator(NullLocator())
    ax.set_xlabel(labels[0], fontsize=FS_LABEL, labelpad=1.0)
    ax.set_ylabel(labels[1], fontsize=FS_LABEL, labelpad=1.0)
    finish(ax, title)


def build(variant: str, df, summary: dict, st: dict, tap: str, cmap, record: dict):
    fig = plt.figure(figsize=(W, LAYOUTS[variant]["H"]))
    cells = layout(fig, variant)
    titles = TITLES[variant]
    ax1 = fig.add_subplot(cells["p1"])
    m3.panel_within(ax1, tap, record)
    if variant == "main":
        ax1.set_ylim(-1.1, 0.6)
        ax1.set_yticks([-1, -0.5, 0, 0.5], ["\u22121", "\u22120.5", "0", "0.5"])
        ax1.set_ylabel("cosine", fontsize=FS_LABEL, labelpad=1.0)
    set_title(ax1, titles[0])
    axE, axF = fig.add_subplot(cells["hE"]), fig.add_subplot(cells["hF"])
    im, e_same, shape = m3.panel_heat(axE, "kappa_e", tap, st, titles[1], True, cmap)
    _, f_same, _ = m3.panel_heat(axF, "kappa_f", tap, st, titles[2], False, cmap)
    for ax, t in ((axE, titles[1]), (axF, titles[2])):
        set_title(ax, t)
        # The cell is already square in inches, so 'auto' fills it without apply_aspect shrinking the box.
        ax.set_aspect("auto")
    record["heatmap_gm_black_seedmean_same_sign"] = {"kappa_e": round(e_same, 3), "kappa_f": round(f_same, 3),
                                                     "n_states": shape[0]}
    horiz = "cb_below" in LAYOUTS[variant]
    if horiz:
        axE.set_xticks([])
        axF.set_xticks([])
    cb = fig.colorbar(im, cax=fig.add_subplot(cells["cb"]), ticks=[-1, 0, 1],
                      orientation="horizontal" if horiz else "vertical")
    cb.ax.tick_params(labelsize=FS_TICK, length=1.5, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.solids.set_rasterized(True)
    ax4 = fig.add_subplot(cells["p4"])
    if variant == "R":
        panel_ratio(ax4, summary, record, titles[3])
    else:
        kw = dict(labels=(r"shape pull $\alpha_F(\Delta_\star)$", r"colour pull $\alpha_F(\Delta_p)$")) \
            if variant == "main" else {}
        panel_scatter(ax4, df, summary, record, titles[3], edge_arrows=variant == "main", **kw)
    # The colour/shape pull ratios are quoted in the caption, not the legend.
    for co, n, _ in TUG:
        record[f"legend_ratio|{n}"] = summary["eval50"][co]["F_ratio_of_means"]["mean"]
    for (lab, dx), ax in zip(LETTERS.get(variant, ()), (ax1, axE, axF, ax4)):
        fs.panel_letter(ax, lab, dx=dx)
    fs.legend_row(fig, [fs.cohort_handle("GM"), fs.cohort_handle("Non-GM"), fs.init_handle(), fs.ci_handle()],
                  ["GM", "Non-GM", "init", fs.CI_LABEL])
    if variant == "R":
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        tb = max(a.title.get_window_extent(r).y1 for a in (axE, axF)) / fig.dpi
        bE, bF = axE.get_position(), axF.get_position()
        fig.text((bE.x0 + bF.x1) / 2, (tb + 0.03) / fig.get_figheight(), "across states", ha="center",
                 va="bottom", fontsize=FS_LABEL)
    return fig, (ax1, axE, axF, cb.ax, ax4)


def check_text(fig, min_gap_in: float = 0.01) -> None:
    """Every visible text box stays inside the figure and at least min_gap_in from every other text box."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in fig.findobj(plt.Text):
        if not t.get_visible() or not t.get_text().strip() or t.get_figure(root=True) is not fig:
            continue
        b = t.get_window_extent(r)
        if b.width == 0:
            continue
        boxes.append((t.get_text(), b.x0 / fig.dpi, b.y0 / fig.dpi, b.x1 / fig.dpi, b.y1 / fig.dpi, t.axes))
    for name, x0, y0, x1, y1, _ in boxes:
        assert x0 >= 0 and y0 >= 0 and x1 <= fig.get_figwidth() and y1 <= fig.get_figheight(), ("outside", name, round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
    bad = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            dx = max(b[1] - a[3], a[1] - b[3])
            dy = max(b[2] - a[4], a[2] - b[4])
            if dx < min_gap_in and dy < min_gap_in:
                bad.append((a[0], b[0], round(dx, 3), round(dy, 3)))
    for name, x0, y0, x1, y1, own in boxes:
        for ax in fig.axes:
            if ax is own:
                continue
            e = ax.get_window_extent(r)
            ax0, ay0, ax1_, ay1 = e.x0 / fig.dpi, e.y0 / fig.dpi, e.x1 / fig.dpi, e.y1 / fig.dpi
            if max(ax0 - x1, x0 - ax1_) < min_gap_in and max(ay0 - y1, y0 - ay1) < min_gap_in:
                bad.append((name, "axes", round(ax0, 3), round(ay0, 3)))
    print(f"checked {len(boxes)} text boxes, {len(bad)} collisions", bad)
    assert not bad, bad


def check_alignment(fig, axes, variant: str) -> dict:
    fig.canvas.draw()
    boxes = {}
    for name, ax in zip(("p1", "kE", "kF", "cbar", "p4"), axes):
        b = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        boxes[name] = [round(b.x0, 4), round(b.y0, 4), round(b.x1, 4), round(b.y1, 4)]
        print(f"[{variant}] {name:5s} x0={b.x0:.4f} y0={b.y0:.4f} x1={b.x1:.4f} y1={b.y1:.4f} "
              f"w={b.width:.4f} h={b.height:.4f} in")
    y0s = {boxes[k][1] for k in EQUAL_BOTTOM[variant]}
    y1s = {boxes[k][3] for k in EQUAL_BOTTOM[variant]}
    assert len(y0s) == 1 and len(y1s) == 1, (y0s, y1s)
    square = SQUARE[variant]
    for k in square:
        x0, y0, x1, y1 = boxes[k]
        assert abs((x1 - x0) - (y1 - y0)) < 1e-3, (k, boxes[k])
    if variant in ("R", "main"):
        w = [boxes[k][2] - boxes[k][0] for k in square]
        assert max(w) - min(w) < 1e-3, boxes
    check_text(fig)
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tap", default="h_fc", choices=["h_fc"])
    ap.add_argument("--variants", nargs="+", default=["main", "R", "S"], choices=["main", "R", "S"])
    ap.add_argument("--fig-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--values-out", type=Path, default=TABLES / "main_competition_values.json")
    args = ap.parse_args()
    tw.style()
    cmap = plt.get_cmap("RdBu_r").copy()
    # Diagonal (self-similarity) is NaN by construction; grey keeps it distinct from kappa = 0 (white).
    cmap.set_bad("#BDBDBD")
    st = load_state_table(BANK_DIR / "ras_gt_bank_canonical.json")
    summary = json.loads((tw.DATA / "summary.json").read_text())["summary"]["unit_mean"]
    df = tw.load(summary)
    out: dict = {"sources": [str(m3.GEOM.relative_to(ROOT)), str((tw.DATA / "summary.json").relative_to(ROOT))],
                 "tap": args.tap, "edits": "swap", "background": "black", "tug_states": "eval50"}
    for v in args.variants:
        record: dict = {}
        fig, axes = build(v, df, summary, st, args.tap, cmap, record)
        record["axes_bbox_in"] = check_alignment(fig, axes, v)
        stem = args.fig_dir / STEMS[v]
        fig.savefig(stem.with_suffix(".png"), dpi=400)
        fig.savefig(stem.with_suffix(".pdf"), dpi=600)
        plt.close(fig)
        out[v] = record
    args.values_out.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", args.fig_dir, args.values_out)


if __name__ == "__main__":
    main()
