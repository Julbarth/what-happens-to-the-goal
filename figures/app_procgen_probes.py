"""Appendix Maze and Coin probe figures with their baselines: 3 random-init draws (band) and the input control (line)."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from palette import PAL  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

SITES = ["h1", "h2", "h3", "h_pre", "h_fc"]
SITE_LABEL = [r"$h^{(1)}$", r"$h^{(2)}$", r"$h^{(3)}$", r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$"]
MAZE_SITES = ["block1_gap", "block2_gap", "block3_gap", "pre_fc", "fc"]
COIN_SITES = ["block1", "block2", "block3_gap", "block3", "fc"]  # producer keys, see protocol.json site_map
T2 = 4.303  # t_{0.975, df=2}: 3 checkpoints per cohort, and 3 init draws
COH = {"GM": PAL["gm"], "Non-GM": PAL["nongm"]}
# Geometry and style copied from app_attribute_probes.py so the three Procgen probe figures match.
PW, PH, WSP = 3.0, 2.35, 0.62
LEFT, RIGHT, TOP, BOTTOM = 0.95, 0.15, 0.55, 0.85
STYLE = dict(line_w=2.2, mean_ms=6.5, pt_s=11, band_a=0.18, pt_a=0.75, font=11, title=12, sup=14)
INIT_LS = (0, (3.5, 1.8))
CTRL_LS = (0, (1, 1.6))
# Coin random-init networks: the first three of the eight draws the producer evaluated (init seeds 0, 1, 2),
# so every environment reports three draws.
COIN_INIT_DRAWS = 3


def load_maze(d: Path, ind: Path) -> dict:
    """Per-checkpoint and per-init-draw AUC (mean over 5 split seeds), plus the input controls."""
    cells: dict = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(d / "checkpoint_level.csv")):
        if r["site"] in MAZE_SITES and r["cohort"] in ("GM", "Non-GM", "init"):
            s = SITES[MAZE_SITES.index(r["site"])]
            cells[r["concept"]][r["cohort"]].setdefault(s, []).append(float(r["auc"]))
    ref = {(r["concept"]): float(r["auc"]) for r in csv.DictReader(open(d / "checkpoint_level.csv"))
           if r["arm"] == "agent_position"}
    audit = json.load(open(ind / "summary.json"))["edit_audit"]
    ctrl = {"cheese_present": (audit["cheese_cell_window_probe_auc_mean"], "cheese-cell pixels", (0.42, False)),
            "cheese_bearing": (ref["cheese_bearing"], "agent position"),
            "corner_bearing": (ref["corner_bearing"], "agent position")}
    return {"cells": cells, "ctrl": ctrl}


def load_coin(d: Path) -> dict:
    """Trained per-checkpoint scores from cells.csv (subset all) and per-draw init scores from per_network/."""
    cells: dict = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(d / "cells.csv")):
        if r["subset"] == "all" and r["site"] in COIN_SITES and r["cohort"] in COH:
            cells[r["concept"]][r["cohort"]][SITES[COIN_SITES.index(r["site"])]] = ast.literal_eval(r["trained_ckpts"])
    for i in range(COIN_INIT_DRAWS):
        acc: dict = defaultdict(list)
        for r in csv.DictReader(open(d / f"per_network/random_init_{i}_scores.csv")):
            if r["subset"] == "all" and r["site"] in COIN_SITES:
                acc[(r["concept"], SITES[COIN_SITES.index(r["site"])])].append(float(r["score"]))
        for (c, s), v in acc.items():
            assert len(v) == 5, (i, c, s)
            cells[c]["init"].setdefault(s, []).append(float(np.mean(v)))
    pix = defaultdict(list)
    for r in csv.DictReader(open(d / "per_network/raw_pixels_scores.csv")):
        if r["subset"] == "all":
            pix[r["concept"]].append(float(r["score"]))
    pos = [float(r["r2"]) for r in csv.DictReader(open(d / "agent_position_control_bearing.csv"))
           if r["features"] == "agent_xy"]
    ctrl = {"presence": (float(np.mean(pix["presence"])), "raw pixels", (0.40, True)),
            "bearing": (float(np.mean(pos)), "agent position", (0.36, True))}
    return {"cells": cells, "ctrl": ctrl, "pixels": {k: float(np.mean(v)) for k, v in pix.items()}}


def load_decorrelated(path: Path) -> dict:
    """Agent-position-decorrelated bearing probes (procgen_bearing_decorrelated.py): per-network means and position control."""
    out: dict = {}
    for r in csv.DictReader(open(path)):
        key = (r["env"], r["subset"])
        if r["cohort"] in ("GM", "Non-GM", "init"):
            out.setdefault(key, defaultdict(dict))[r["cohort"]][r["site"]] = [float(v) for v in r["per_network"].split(";")]
        elif r["cohort"] == "agent_xy":
            out.setdefault(("ctrl",) + key, float(r["mean"]))
    return out


def band(v) -> tuple[float, float]:
    v = np.asarray(v, float)
    assert len(v) == 3, v
    return float(v.mean()), float(T2 * v.std(ddof=1) / np.sqrt(3))


def draw(ax, cells: dict, ctrl: tuple, metric: str, tag: str | None = None) -> dict:
    x = np.arange(len(SITES))
    clip = (0, 1) if metric == "AUC" else (-np.inf, 1)
    plotted = {}
    im, ih = map(np.array, zip(*[band(cells["init"][s]) for s in SITES]))
    ax.fill_between(x, np.clip(im - ih, *clip), np.clip(im + ih, *clip), color=PAL["init"], alpha=0.16, lw=0, zorder=1)
    ax.plot(x, im, ls=INIT_LS, color=PAL["init"], lw=1.6, zorder=2)
    plotted["init"] = im.tolist()
    for k, (coh, col) in enumerate(COH.items()):
        vals = [np.asarray(cells[coh][s], float) for s in SITES]
        m, h = map(np.array, zip(*[band(v) for v in vals]))
        ax.fill_between(x, np.clip(m - h, *clip), np.clip(m + h, *clip), color=col, alpha=STYLE["band_a"], lw=0)
        ax.plot(x, m, "-o", color=col, lw=STYLE["line_w"], ms=STYLE["mean_ms"], mec="white", mew=1.0, zorder=4)
        off = -0.08 if k == 0 else 0.08
        for i, v in enumerate(vals):
            ax.scatter(np.full(3, i + off), v, s=STYLE["pt_s"], color=col, alpha=STYLE["pt_a"], lw=0, zorder=3)
        plotted[coh] = m.tolist()
    # ctrl = (value, label[, (label x in axes fraction, label above the line)]), placed to clear the data lines.
    ax.axhline(ctrl[0], ls=CTRL_LS, color=PAL["null"], lw=1.4, zorder=2.5)
    xf, above = ctrl[2] if len(ctrl) > 2 else (0.0, False)
    ax.annotate(ctrl[1], xy=(xf, ctrl[0]), xycoords=("axes fraction", "data"), xytext=(3, 3 if above else -3),
                textcoords="offset points", ha="left", va="bottom" if above else "top", fontsize=STYLE["font"] - 2, color=PAL["null"])
    plotted["input_control"] = ctrl[0]
    if tag:
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, ha="left", va="top", fontsize=STYLE["font"] - 2, color=PAL["ink"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["null"], lw=0.8))
    ax.set_xticks(x, SITE_LABEL)
    ax.set_xlim(-0.4, len(SITES) - 0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=STYLE["font"])
    ax.set_ylabel(f"held-out {metric}", fontsize=STYLE["font"])
    return plotted


def render(title: str, panels: list[tuple], out_stem: Path) -> dict:
    """Each panel is (title, cells, ctrl, metric, ylim[, tag]), where a tag marks a decorrelated variant of a probe."""
    ncol = len(panels)
    W = LEFT + ncol * PW + (ncol - 1) * WSP + RIGHT
    H = TOP + PH + BOTTOM
    fig = plt.figure(figsize=(W, H))
    means = {}
    for ci, (ptitle, cells, ctrl, metric, ylim, *tag) in enumerate(panels):
        ax = fig.add_axes(((LEFT + ci * (PW + WSP)) / W, BOTTOM / H, PW / W, PH / H))
        means[ptitle] = draw(ax, cells, ctrl, metric, tag[0] if tag else None)
        ax.set_ylim(*ylim)
        ax.set_title(ptitle, fontsize=STYLE["title"], color=PAL["ink"])
    handles = [Line2D([0], [0], color=c, marker="o", lw=STYLE["line_w"], ms=STYLE["mean_ms"], mec="white", label=k)
               for k, c in COH.items()]
    handles += [Line2D([0], [0], color=PAL["init"], ls=INIT_LS, lw=1.6, label="random init (3 draws)"),
                Line2D([0], [0], color=PAL["null"], ls=CTRL_LS, lw=1.4, label="input control"),
                Patch(color="#999999", alpha=0.3, lw=0, label="95% CI")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=STYLE["font"] - (1 if ncol < 3 else 0),
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.8)
    fig.suptitle(title, fontsize=STYLE["sup"], fontweight="bold", color=PAL["ink"], y=1 - 0.12 / H)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out_stem.with_suffix(".{png,pdf}"))
    return means


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maze", type=Path, default=DATA_ROOT / "probes/maze_probes_grouped_20260924",
                    help="see the README data section")
    ap.add_argument("--maze-indep", type=Path, default=DATA_ROOT / "probes/maze_probes_grouped_independent_check_20260924")
    ap.add_argument("--coin", type=Path, default=DATA_ROOT / "probes/coinrun_concept_grouped_canonical_20260924")
    ap.add_argument("--decorrelated", type=Path, default=TABLES / "procgen_bearing_decorrelated/summary.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--values-out", type=Path, default=TABLES / "plotted_values_procgen_baselines.json")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    mz, cn = load_maze(args.maze, args.maze_indep), load_coin(args.coin)
    dc = load_decorrelated(args.decorrelated)
    auc_lim = (0.4, 1.02)
    out = {"sites": SITES, "maze": render("Maze", [
        ("Cheese presence", mz["cells"]["cheese_present"], mz["ctrl"]["cheese_present"], "AUC", auc_lim),
        ("Cheese bearing", mz["cells"]["cheese_bearing"], mz["ctrl"]["cheese_bearing"], "AUC", auc_lim),
        ("Cheese bearing, decorrelated", dc[("maze", "balanced")], (dc[("ctrl", "maze", "balanced")], "agent position", (0.40, False)),
         "AUC", auc_lim, "position-balanced set"),
        ("Corner bearing", mz["cells"]["corner_bearing"], mz["ctrl"]["corner_bearing"], "AUC", auc_lim)],
        args.out_dir / "fig_appendix_maze_probes_baselines")}
    out["coin"] = render("Coin", [
        ("Coin presence", cn["cells"]["presence"], cn["ctrl"]["presence"], "AUC", auc_lim),
        ("Coin bearing", cn["cells"]["bearing"], cn["ctrl"]["bearing"], "R²", (-1.0, 1.02)),
        ("Coin bearing, decorrelated", dc[("coin", "progress_agent_xy")], (dc[("ctrl", "coin", "progress_agent_xy")], "agent position", (0.36, True)),
         "R²", (-1.0, 1.02), "residualized on position")],
        args.out_dir / "fig_appendix_coin_probes_baselines")
    out["coin_raw_pixels"] = cn["pixels"]
    out["coin_init_draws"] = {c: {s: cn["cells"][c]["init"][s] for s in SITES} for c in ("presence", "bearing")}
    out["maze_init_draws"] = {c: {s: mz["cells"][c]["init"][s] for s in SITES}
                              for c in ("cheese_present", "cheese_bearing", "corner_bearing")}
    args.values_out.write_text(json.dumps(out, indent=1))
    print("wrote", args.values_out)


if __name__ == "__main__":
    main()
