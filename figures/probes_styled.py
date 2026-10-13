"""Styled probe figure (all sites, 95% t bands, markers) with one-line titles, a one-row legend and sparse ticks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

import probes_panels as v2  # noqa: E402
from probes_panels import COH, CTRL_LS, INIT_LS, PAL, fs  # noqa: E402
from probes_bands import R2_LIM, auc_of  # noqa: E402

from paths import OUT_DIR, TABLES

FS_TICK = 6.0
FS_LAB = 6.0
FS_TITLE = 6.5
FS_SUB = 6.0
FS_LEG = 6.0
FS_MIN = 6.0
MS = 3.0
MS_INIT = 2.0
LW_INIT = 0.8
H_IN = 1.75
BASE_FS = 6.0


def set_font_size(fs_pt: float) -> None:
    """Ticks, labels, legend and subtitles at fs_pt; titles 0.5 pt larger."""
    global FS_TICK, FS_LAB, FS_TITLE, FS_SUB, FS_LEG, FS_MIN
    FS_TICK = FS_LAB = FS_SUB = FS_LEG = FS_MIN = fs_pt
    FS_TITLE = fs_pt + 0.5

# Labelled ticks at chance and ceiling only (plus R² 0.5). The AUC axis keeps an unlabelled tick at 0.75 so
# every panel has a tick at the same three heights without printing two-decimal labels.
R2_MAJOR = [0.0, 0.5, 1.0]
AUC_MAJOR = [0.5, 1.0]
AUC_MINOR = [0.75]


def short_num(v: float, _pos=None) -> str:
    return f"{v:g}"


def draw_series(ax, x, ms, col, ls, lim, hollow=frozenset(), init=False, band_alpha=fs.CI_ALPHA) -> list[float]:
    """Mean line, shaded 95% t band, and a marker at every site (open and smaller for random-init networks)."""
    x = np.asarray(x, float)
    m = np.array([a for a, _ in ms])
    h = np.array([b for _, b in ms])
    lo, hi = m - h, m + h
    ax.fill_between(x, np.clip(lo, *lim), np.clip(hi, *lim), color=col, alpha=band_alpha, lw=0,
                    zorder=1 if init else 2)
    ax.plot(x, m, ls=ls, color=col, lw=LW_INIT if init else fs.LW_DATA, zorder=3 if init else 4)
    for i in range(len(x)):
        if init:
            ax.plot(x[i], m[i], "o", ms=MS_INIT, zorder=3.5, mfc="white", mec=col, mew=0.7)
        else:
            open_ = i in hollow
            ax.plot(x[i], m[i], "o", ms=MS, zorder=5, mfc="white" if open_ else col, mec=col if open_ else "white",
                    mew=0.8 if open_ else 0.35)
    return list(lo) + list(hi)


def draw_panel(ax, P: dict, letter: str, xlim: tuple[float, float]) -> dict:
    auc = P["metric"] == "AUC"
    clip_lo = 0.0 if auc else -np.inf
    ax.axhline(0.5 if auc else 0.0, ls="-", color="#CFCFCF", lw=0.7, zorder=0.5)
    edges = draw_series(ax, P["x"], P["s"]["init"], PAL["init"], INIT_LS, (clip_lo, 1.0), init=True, band_alpha=0.14)
    for c, col in COH.items():
        edges += draw_series(ax, P["x"], P["s"][c], col, "-", (clip_lo, 1.0), P.get("hollow", frozenset()))
    if P["ctrl"] is not None:
        ax.axhline(P["ctrl"], ls=CTRL_LS, color=PAL["null"], lw=0.9, zorder=2.5)
        edges.append(P["ctrl"])
    if auc:
        ax.set_ylim(auc_of(R2_LIM[0]), auc_of(R2_LIM[1]))
        ax.set_yticks(AUC_MAJOR)
        ax.set_yticks(AUC_MINOR, minor=True)
    else:
        ax.set_ylim(*R2_LIM)
        ax.set_yticks(R2_MAJOR)
    ax.yaxis.set_major_formatter(FuncFormatter(short_num))
    ax.set_ylabel(P["metric"], fontsize=FS_LAB, labelpad=1)
    ax.set_xticks(P["x"], P["ticks"])
    ax.set_xlim(*xlim)
    ax.tick_params(labelsize=FS_TICK, length=2, width=fs.LW_AX, pad=1.2)
    ax.tick_params(axis="y", which="minor", length=2, width=fs.LW_AX)
    ax.tick_params(axis="x", pad=0.8)
    ax.annotate(P["sub"], xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 2.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=FS_SUB, color=PAL["ink"])
    ax.annotate(f"$\\mathbf{{({letter})}}$ {P['env']}", xy=(0.5, 1), xycoords="axes fraction",
                xytext=(0, 2.5 + FS_SUB * 1.25), textcoords="offset points", ha="center", va="bottom",
                fontsize=FS_TITLE, color=PAL["ink"])
    return {"min_edge": float(min(edges)), "clipped_below_axis": bool(min(edges) < ax.get_ylim()[0])}


def check_layout(fig) -> None:
    """Every visible text >= FS_MIN, inside the figure, and no two text boxes overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        assert t.get_fontsize() >= FS_MIN - 1e-9, f"{t.get_text()!r} at {t.get_fontsize()}pt"
        b = t.get_window_extent(r)
        assert fig.bbox.x0 - 0.5 <= b.x0 and b.x1 <= fig.bbox.x1 + 0.5 and fig.bbox.y0 - 0.5 <= b.y0 \
            and b.y1 <= fig.bbox.y1 + 0.5, f"{t.get_text()!r} clipped"
        boxes.append((t.get_text(), b))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                raise AssertionError(f"overlap: {boxes[i][0]!r} vs {boxes[j][0]!r}")
    print(f"layout ok: {len(boxes)} text boxes, all >= {FS_MIN}pt, no overlaps")


def render(panels: list[dict], out_stem: Path, h_in: float, xlim: tuple[float, float], top_in: float | None = None,
           bottom_in: float | None = None, legend_y_in: float = 0.02) -> dict:
    """top_in / bottom_in override the font-scaled top and bottom margins (in)."""
    W = fs.W_IN
    fig, axes = plt.subplots(1, len(panels), figsize=(W, h_in))
    # Text-driven margins (y labels, two title lines, x ticks + legend) grow in proportion to the font.
    k = FS_LAB / BASE_FS
    top_in = 0.27 * k if top_in is None else top_in
    bottom_in = 0.40 * k if bottom_in is None else bottom_in
    fig.subplots_adjust(left=0.27 * k / W, right=1 - 0.03 / W, top=1 - top_in / h_in, bottom=bottom_in / h_in,
                        wspace=0.30 * k)
    info = {P["env"]: draw_panel(ax, P, "abcd"[i], xlim) for i, (ax, P) in enumerate(zip(axes, panels))}
    g = PAL["mute"]
    handles = [Line2D([], [], color=COH["GM"], lw=fs.LW_DATA, marker="o", ms=MS, mec="white", mew=0.35),
               Line2D([], [], color=COH["Non-GM"], lw=fs.LW_DATA, marker="o", ms=MS, mec="white", mew=0.35),
               Line2D([], [], color=PAL["init"], lw=LW_INIT, ls=INIT_LS, marker="o", ms=MS_INIT, mfc="white",
                      mec=PAL["init"], mew=0.7),
               Line2D([], [], color=PAL["null"], lw=0.9, ls=CTRL_LS),
               (Patch(color=g, alpha=0.3, lw=0), Line2D([], [], color=g, lw=fs.LW_DATA, marker="o", ms=MS,
                                                        mec="white", mew=0.35))]
    labels = ["GM", "Non-GM", "random init", "input control", fs.CI_LABEL]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y_in / h_in), ncol=len(handles),
               frameon=False, fontsize=FS_LEG, handlelength=2.0, handleheight=0.6, handletextpad=0.35,
               columnspacing=1.1, borderaxespad=0, borderpad=0)
    check_layout(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out_stem.with_suffix(".{png,pdf}"))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=v2.DATA, help="gm-geometry data/probes root (Procgen panels)")
    ap.add_argument("--robot", type=Path, default=v2.ROBOT_CSV, help="robotics code probe_audit PTM_per_seed.csv")
    ap.add_argument("--robot-probe", default="ridge_sel", help="probe_audit probe column (ridge_sel = paper default)")
    ap.add_argument("--font-size", type=float, default=BASE_FS,
                    help="ticks, axis labels, legend (pt); titles are 0.5 pt larger")
    ap.add_argument("--height", type=float, default=None,
                    help="figure height (in); default keeps the 6 pt plot-area height and adds the larger text margins")
    ap.add_argument("--top-margin", type=float, default=None, help="top margin (in); default 0.27 in x font/6")
    ap.add_argument("--bottom-margin", type=float, default=None, help="bottom margin (in); default 0.40 in x font/6")
    ap.add_argument("--legend-y", type=float, default=0.02, help="legend baseline above the figure bottom (in)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-name", default="fig_probes_styled")
    ap.add_argument("--values-out", type=Path, default=TABLES / "probes_styled_values.json")
    args = ap.parse_args()
    fs.style()
    set_font_size(args.font_size)
    panels = v2.build_panels(args)
    h_in = args.height or H_IN + (0.27 + 0.40) * (args.font_size / BASE_FS - 1)
    info = render(panels, args.out_dir / args.out_name, h_in, (-0.25, 4.25), args.top_margin, args.bottom_margin,
                  args.legend_y)
    rec = {P["env"]: {"metric": P["metric"], "sites": P["ticks"], "control": P["ctrl"], **info[P["env"]],
                      **{c: [[round(m, 4), round(h, 4)] for m, h in P["s"][c]] for c in ("GM", "Non-GM", "init")}}
           for P in panels}
    args.values_out.write_text(json.dumps({"note": "per site [mean, 95% t half-width]; bands clipped to [0, 1] for AUC",
                                           "y_scale": "AUC axis = 0.5 + R2 axis / 2", "panels": rec}, indent=1))
    print("wrote", args.values_out)


if __name__ == "__main__":
    main()
