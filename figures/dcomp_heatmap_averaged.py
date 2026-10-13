#!/usr/bin/env python3
"""GM seed-averaged cross-state d_comp heatmaps (Euclidean/Fisher x post-ReLU h_pre/h_fc), states grouped by line bearing."""
from __future__ import annotations


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from paths import DATA_ROOT, OUT_DIR

OUT = OUT_DIR
DATA_DIR = DATA_ROOT / "probes/ras_dcomp_canonical_taps_groundtruth_20260923"

from palette import PAL  # noqa: E402
from dcomp_heatmap_orderings import ordering_bearing  # noqa: E402

GM_SEEDS = ["6313", "6315", "6317"]
TAPS = ["h_pre", "h_fc"]
METRICS = ["euclidean", "fisher"]
MATRIX_KEY = {"euclidean": "kappa_e_matrix", "fisher": "kappa_f_matrix"}
METRIC_LABEL = {"euclidean": r"$\kappa_E$ (Euclidean)", "fisher": r"$\kappa_F$ (Fisher)"}
TAP_LABEL = {"h_pre": r"$h_{\mathrm{pre}}$", "h_fc": r"$h_{\mathrm{fc}}$"}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 13,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": PAL["paper"],
        "axes.facecolor": PAL["paper"],
    })


def load_seed_matrices(tap: str, key: str) -> list[np.ndarray]:
    return [np.load(DATA_DIR / f"dcomp_{tap}_gm_{s}.npz")[key] for s in GM_SEEDS]


def main() -> None:
    apply_style()

    # Cellwise averaging is only valid if every seed stores its states in the same order.
    levels = [np.load(DATA_DIR / f"dcomp_h_pre_gm_{s}.npz")["levels"] for s in GM_SEEDS]
    identical = all(np.array_equal(levels[0], lv) for lv in levels[1:])
    if not identical:
        raise SystemExit("REFUSING to build averaged heatmap: GM seeds' levels arrays are not "
                          "identical -- cellwise averaging would be invalid.")
    print(f"[check] identical state order across the 3 GM seeds used here: {identical}")

    ref = np.load(DATA_DIR / "dcomp_h_pre_gm_6313.npz")
    n_states = ref["same_bearing_matrix"].shape[0]
    order, boundaries, group_labels = ordering_bearing(ref["spatial"], ref["dist_line"], ref["dist_gem"])
    print(f"[check] {len(group_labels)} same-first-move-direction groups found, "
          f"boundaries at {boundaries} (2026-09-23: was 2 groups/1 boundary under the "
          f"old, invalid bank; now 4 groups, drawn explicitly rather than assumed): "
          f"{[lab for lab, _ in group_labels]}")

    # Opposite-bearing block pairs come from the group directions, not from block positions.
    import re
    dirs = [tuple(int(x) for x in re.findall(r"-?\d+", lab)) for lab, _ in group_labels]
    opposite_block_pairs = [
        (g1, g2) for g1 in range(len(dirs)) for g2 in range(len(dirs))
        if g1 < g2 and dirs[g1] == (-dirs[g2][0], -dirs[g2][1])
    ]
    print(f"[check] opposite-bearing block pairs: "
          f"{[(dirs[g1], dirs[g2]) for g1, g2 in opposite_block_pairs]}")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 11.2))
    im = None
    cuts = [0] + boundaries + [n_states]
    centers = [(cuts[i] + cuts[i + 1] - 1) / 2 for i in range(len(cuts) - 1)]
    tick_labels = [f"{lab}\n(n={cnt})" for lab, cnt in group_labels]
    for row, metric in enumerate(METRICS):
        for col, tap in enumerate(TAPS):
            ax = axes[row, col]
            mats = load_seed_matrices(tap, MATRIX_KEY[metric])
            avg = np.mean(np.stack(mats, axis=0), axis=0)
            avg_sorted = avg[np.ix_(order, order)]
            im = ax.imshow(avg_sorted, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
            for b in boundaries:
                ax.axhline(b - 0.5, color="black", lw=2.2, zorder=5)
                ax.axvline(b - 0.5, color="black", lw=2.2, zorder=5)
            ax.set_xlim(-0.5, n_states - 0.5)
            ax.set_ylim(n_states - 0.5, -0.5)
            ax.set_xticks(centers)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
            ax.set_yticks(centers)
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(TAP_LABEL[tap], fontsize=16, fontweight="bold", color=PAL["ink"], pad=10)
            if col == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=14, color=PAL["ink"])

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label(r"seed-mean $\kappa$ (GM, n=3 seeds)", fontsize=13)

    out_path = OUT / "fig_appendix_ras_dcomp_heatmap_averaged_2x2"
    for ext in ("png", "pdf"):
        p = out_path.with_suffix(f".{ext}")
        fig.savefig(p)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
