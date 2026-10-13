#!/usr/bin/env python3
"""Diagnostic: GM cross-state d_comp heatmaps under raw, bearing, action-pair and distance state orderings."""
from __future__ import annotations


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from paths import DATA_ROOT, OUT_DIR

OUT = OUT_DIR
DATA_DIR = DATA_ROOT / "probes/ras_dcomp_canonical_taps_groundtruth_20260923"
ORDERINGS_OUT = OUT / "heatmap_orderings"

from palette import PAL  # noqa: E402

GM_SEEDS = ["6313", "6315", "6317"]
TAPS = ["h_pre", "h_fc"]
METRICS = ["euclidean", "fisher"]
MATRIX_KEY = {"euclidean": "kappa_e_matrix", "fisher": "kappa_f_matrix"}
METRIC_LABEL = {"euclidean": r"$\kappa_E$", "fisher": r"$\kappa_F$"}
TAP_LABEL = {"h_pre": r"$h_{\mathrm{pre}}$", "h_fc": r"$h_{\mathrm{fc}}$"}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 12,
        "axes.titlesize": 15,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": PAL["paper"],
        "axes.facecolor": PAL["paper"],
    })


def load_avg(tap: str, key: str) -> np.ndarray:
    mats = [np.load(DATA_DIR / f"dcomp_{tap}_gm_{s}.npz")[key] for s in GM_SEEDS]
    return np.mean(np.stack(mats, axis=0), axis=0)


def load_spatial() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """spatial (n,6), dist_line (n,), dist_gem (n,) from seed 6313; these are identical across seeds."""
    d = np.load(DATA_DIR / "dcomp_h_pre_gm_6313.npz")
    return d["spatial"], d["dist_line"], d["dist_gem"]


# Each ordering returns (order, boundaries, group_labels).
# group_labels[k] = (label_string, count) for the k-th group along `order`,
# aligned with `boundaries` (len(group_labels) == len(boundaries)+1).

def ordering_raw(n: int, *_args) -> tuple[np.ndarray, list[int], list[tuple[str, int]]]:
    return np.arange(n), [], [("raw harvest order", n)]


def ordering_bearing(spatial: np.ndarray, dist_line: np.ndarray, dist_gem: np.ndarray):
    line_dir = [tuple(r) for r in spatial[:, :2].astype(int)]
    return _group_and_sort(line_dir, dist_line, dist_gem,
                            label_fn=lambda k: f"line dir {k}")


def ordering_action_pair(spatial: np.ndarray, dist_line: np.ndarray, dist_gem: np.ndarray):
    keys = [(tuple(r[:2].astype(int)), tuple(r[2:4].astype(int))) for r in spatial]
    return _group_and_sort(keys, dist_line, dist_gem,
                            label_fn=lambda k: f"line{k[0]} gem{k[1]}")


def ordering_distance(spatial: np.ndarray, dist_line: np.ndarray, dist_gem: np.ndarray):
    keys = list(dist_line)
    return _group_and_sort(keys, dist_line, dist_gem,
                            label_fn=lambda k: f"$d_{{\\mathrm{{line}}}}$={k}")


def _group_and_sort(keys, dist_line, dist_gem, label_fn):
    n = len(keys)
    # stable group order = first-occurrence order, sub-sorted within group by (dist_line, dist_gem)
    seen = []
    for k in keys:
        if k not in seen:
            seen.append(k)
    group_of = {k: gi for gi, k in enumerate(seen)}
    idx = sorted(range(n), key=lambda i: (group_of[keys[i]], dist_line[i], dist_gem[i]))
    order = np.array(idx)
    ordered_groups = [group_of[keys[i]] for i in order]
    boundaries = [i for i in range(1, n) if ordered_groups[i] != ordered_groups[i - 1]]
    counts = {g: 0 for g in seen}
    for k in keys:
        counts[k] += 1
    group_labels = [(label_fn(k), counts[k]) for k in seen]
    return order, boundaries, group_labels


ORDERINGS = {
    "raw": ("Raw harvest order (no sort)", ordering_raw),
    "bearing": ("Bearing (line first-move direction), current", ordering_bearing),
    "action_pair": ("Full ordered action pair (line, gem)", ordering_action_pair),
    "distance": (r"Distance ($d_{\mathrm{line}}$ primary)", ordering_distance),
}


def within_between_means(matrix: np.ndarray, order: np.ndarray, boundaries: list[int]) -> dict:
    """Mean kappa within each group and between each pair of groups of the reordered matrix."""
    cuts = [0] + boundaries + [matrix.shape[0]]
    n_groups = len(cuts) - 1
    m = matrix[np.ix_(order, order)]
    out = {"within": [], "between": []}
    for g in range(n_groups):
        a, b = cuts[g], cuts[g + 1]
        block = m[a:b, a:b]
        iu = np.triu_indices(b - a, k=1)
        vals = block[iu]
        out["within"].append(float(np.mean(vals)) if len(vals) else float("nan"))
    for g1 in range(n_groups):
        for g2 in range(g1 + 1, n_groups):
            a1, b1 = cuts[g1], cuts[g1 + 1]
            a2, b2 = cuts[g2], cuts[g2 + 1]
            block = m[a1:b1, a2:b2]
            out["between"].append({"groups": (g1, g2), "mean": float(np.mean(block))})
    return out


def build_ordering_figure(name: str, title: str, order_fn) -> dict:
    spatial, dist_line, dist_gem = load_spatial()
    n = spatial.shape[0]
    if name == "raw":
        order, boundaries, group_labels = order_fn(n)
    else:
        order, boundaries, group_labels = order_fn(spatial, dist_line, dist_gem)

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.5))
    report = {"ordering": name, "n_groups": len(group_labels), "group_labels": group_labels, "cells": {}}
    im = None
    for row, metric in enumerate(METRICS):
        for col, tap in enumerate(TAPS):
            ax = axes[row, col]
            avg = load_avg(tap, MATRIX_KEY[metric])
            avg_sorted = avg[np.ix_(order, order)]
            im = ax.imshow(avg_sorted, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            for b in boundaries:
                ax.axhline(b - 0.5, color="black", lw=2.2, zorder=5)
                ax.axvline(b - 0.5, color="black", lw=2.2, zorder=5)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5)
            cuts = [0] + boundaries + [n]
            centers = [(cuts[i] + cuts[i + 1] - 1) / 2 for i in range(len(cuts) - 1)]
            if len(centers) <= 12:
                labels_short = [f"{lab} (n={cnt})" for lab, cnt in group_labels]
                ax.set_xticks(centers)
                ax.set_xticklabels(labels_short, rotation=90, fontsize=6.5)
                ax.set_yticks(centers)
                ax.set_yticklabels(labels_short, fontsize=6.5)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(TAP_LABEL[tap], fontsize=15, fontweight="bold", color=PAL["ink"], pad=8)
            if col == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=14, color=PAL["ink"])

            wb = within_between_means(avg, order, boundaries)
            report["cells"][f"{metric}_{tap}"] = wb

    fig.suptitle(title, fontsize=15, fontweight="bold", color=PAL["ink"])
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label(r"seed-mean $\kappa$ (GM, n=3 seeds)", fontsize=11)

    ORDERINGS_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = ORDERINGS_OUT / f"ordering_{name}.{ext}"
        fig.savefig(p)
        print("wrote", p)
    plt.close(fig)
    return report


def main() -> None:
    import json
    all_reports = {}
    for name, (title, order_fn) in ORDERINGS.items():
        print(f"=== {name}: {title} ===")
        report = build_ordering_figure(name, title, order_fn)
        print(f"  {report['n_groups']} groups: {[lab for lab, _ in report['group_labels']]}")
        all_reports[name] = report
    out_json = ORDERINGS_OUT / "ordering_within_between_report.json"
    out_json.write_text(json.dumps(all_reports, indent=2))
    print("wrote", out_json)


if __name__ == "__main__":
    main()
