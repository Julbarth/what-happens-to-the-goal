"""Probe figure panels with shaded 95% t bands and a marker at every site; variant b keeps only h_pre and h_fc."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import probes_panels as v2  # noqa: E402
from probes_panels import COH, CTRL_LS, INIT_LS, PAL, check_layout, control_label, fs  # noqa: E402

from paths import OUT_DIR, TABLES

# R² and AUC share one vertical scale via AUC = 0.5 + R²/2, so chance (R² 0, AUC 0.5), the midpoint and the
# ceiling sit at the same height in every panel and gridlines line up across metrics.
R2_LIM = (-0.5, 1.02)
R2_TICKS = [0.0, 0.5, 1.0]


def auc_of(r2: float) -> float:
    return 0.5 + r2 / 2


def keep_sites(P: dict, idx: list[int]) -> dict:
    """Restrict a panel to the sites at positions idx, re-indexed to 0..len(idx)-1."""
    Q = dict(P)
    Q["x"] = list(range(len(idx)))
    Q["ticks"] = [P["ticks"][i] for i in idx]
    Q["s"] = {c: [v[i] for i in idx] for c, v in P["s"].items()}
    Q["hollow"] = {idx.index(i) for i in P.get("hollow", ()) if i in idx}
    return Q


def only_pre_fc(panels: list[dict]) -> list[dict]:
    out = []
    for P in panels:
        idx = [P["ticks"].index(t) for t in (r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$")]
        out.append(keep_sites(P, idx))
    return out


def draw_series(ax, x, ms, col, ls, lim, hollow=frozenset(), init=False, band_alpha=fs.CI_ALPHA) -> list[float]:
    """Mean line, shaded 95% t band, and a marker at every site (open for init and for de-emphasised sites)."""
    x = np.asarray(x, float)
    m = np.array([a for a, _ in ms])
    h = np.array([b for _, b in ms])
    lo, hi = m - h, m + h
    edges = list(lo) + list(hi)
    ax.fill_between(x, np.clip(lo, *lim), np.clip(hi, *lim), color=col, alpha=band_alpha, lw=0,
                    zorder=1 if init else 2)
    ax.plot(x, m, ls=ls, color=col, lw=fs.LW_DATA if not init else 1.0, zorder=3 if init else 4)
    for i in range(len(x)):
        open_ = init or i in hollow
        ax.plot(x[i], m[i], "o", ms=fs.MS, zorder=5, mfc="white" if open_ else col, mec=col if open_ else "white",
                mew=0.9 if open_ else 0.4)
    return edges


def draw_panel(ax, P: dict, letter: str, xlim: tuple[float, float], letter_dx: float) -> dict:
    auc = P["metric"] == "AUC"
    clip_lo = 0.0 if auc else -np.inf
    ax.axhline(0.5 if auc else 0.0, ls="-", color="#CFCFCF", lw=0.8, zorder=0.5)
    edges = draw_series(ax, P["x"], P["s"]["init"], PAL["init"], INIT_LS, (clip_lo, 1.0), init=True, band_alpha=0.16)
    for c, col in COH.items():
        edges += draw_series(ax, P["x"], P["s"][c], col, "-", (clip_lo, 1.0), P.get("hollow", frozenset()))
    if P["ctrl"] is not None:
        ax.axhline(P["ctrl"], ls=CTRL_LS, color=PAL["null"], lw=1.0, zorder=2.5)
        edges.append(P["ctrl"])
    if auc:
        ax.set_ylim(auc_of(R2_LIM[0]), auc_of(R2_LIM[1]))
        ax.set_yticks([auc_of(t) for t in R2_TICKS])
    else:
        ax.set_ylim(*R2_LIM)
        ax.set_yticks(R2_TICKS)
    ax.set_ylabel(P["metric"], fontsize=fs.FS, labelpad=1)
    ax.set_xticks(P["x"], P["ticks"])
    ax.set_xlim(*xlim)
    fs.finish(ax)
    fs.panel_title(ax, P["env"], P["sub"])
    fs.panel_letter(ax, letter, sub=True, dx=letter_dx)
    ylo = ax.get_ylim()[0]
    return {"min_edge": float(min(edges)), "clipped_below_axis": bool(min(edges) < ylo)}


def render(panels: list[dict], out_stem: Path, h_in: float, xlim: tuple[float, float]) -> dict:
    W = v2.W_IN
    fig, axes = plt.subplots(1, len(panels), figsize=(W, h_in))
    fig.subplots_adjust(left=0.40 / W, right=1 - 0.04 / W, top=1 - 0.33 / h_in, bottom=0.58 / h_in, wspace=0.48)
    info = {}
    for i, (ax, P) in enumerate(zip(axes, panels)):
        info[P["env"]] = draw_panel(ax, P, "abcd"[i], xlim, -24)
    g = PAL["mute"]
    handles = [fs.cohort_handle("GM", line=True), fs.cohort_handle("Non-GM", line=True),
               Line2D([], [], color=PAL["init"], lw=1.0, ls=INIT_LS, marker="o", ms=fs.MS, mfc="white",
                      mec=PAL["init"], mew=0.9),
               Line2D([], [], color=PAL["null"], lw=1.0, ls=CTRL_LS),
               (Patch(color=g, alpha=0.3, lw=0), Line2D([], [], color=g, lw=fs.LW_DATA, marker="o", ms=fs.MS,
                                                        mec="white", mew=0.4))]
    labels = ["GM agents", "Non-GM agents", "random-init network", control_label(panels), fs.CI_LABEL]
    fs.legend_row(fig, handles, labels, y_in=0.03, ncol=3, columnspacing=1.4)
    check_layout(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out_stem.with_suffix(".{png,pdf}"))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["a", "b"], default="a", help="a: all sites, b: h_pre and h_fc only")
    ap.add_argument("--data", type=Path, default=v2.DATA, help="gm-geometry data/probes root (Procgen panels)")
    ap.add_argument("--robot", type=Path, default=v2.ROBOT_CSV, help="robotics code probe_audit PTM_per_seed.csv")
    ap.add_argument("--robot-probe", default="ridge_sel", help="probe_audit probe column (ridge_sel = paper default)")
    ap.add_argument("--height", type=float, default=v2.H_IN)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-name", default=None, help="default fig_main_probes_v3<variant>")
    ap.add_argument("--values-out", type=Path, default=None, help="default plotted_values_v3<variant>.json here")
    args = ap.parse_args()
    name = args.out_name or f"fig_main_probes_v3{args.variant}"
    values_out = args.values_out or TABLES / f"probes_bands_{args.variant}_values.json"
    fs.FS_TICK = fs.FS_LEG = fs.FS_SUB = max(fs.FS_SUB, v2.FS_MIN)
    fs.style()
    panels = v2.build_panels(args)
    if args.variant == "b":
        panels, xlim = only_pre_fc(panels), (-0.35, 1.35)
    else:
        xlim = (-0.3, 4.3)
    info = render(panels, args.out_dir / name, args.height, xlim)
    rec = {P["env"]: {"metric": P["metric"], "sites": P["ticks"], "control": P["ctrl"], **info[P["env"]],
                      **{c: [[round(m, 4), round(h, 4)] for m, h in P["s"][c]] for c in ("GM", "Non-GM", "init")}}
           for P in panels}
    values_out.write_text(json.dumps({"note": "per site [mean, 95% t half-width]; bands clipped to [0, 1] for AUC",
                                      "y_scale": "AUC axis = 0.5 + R2 axis / 2", "panels": rec}, indent=1))
    print("wrote", values_out)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
