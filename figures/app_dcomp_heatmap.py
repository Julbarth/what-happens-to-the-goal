#!/usr/bin/env python3
"""Seed-averaged cross-state d_comp heatmaps (Euclidean/Fisher x h_pre/h_fc) on the canonical bank, swap or ablation edits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import DATA_ROOT, OUT_DIR, ROOT

sys.path.insert(0, str(ROOT / "scripts" / "procgen"))
from dcomp_heatmap_averaged import apply_style  # noqa: E402
from palette import PAL  # noqa: E402
from canon_geometry_run import BANK_DIR, load_state_table  # noqa: E402

TAPS = ("h_pre", "h_fc")
METRICS = (("kappa_e", r"$\kappa_E$ (Euclidean)"), ("kappa_f", r"$\kappa_F$ (Fisher)"))
TAP_LABEL = {"h_pre": r"$h_{\mathrm{pre}}$", "h_fc": r"$h_{\mathrm{fc}}$"}
COHORTS = {
    "gm_black": ("gm", "black", ["6313", "6315", "6317"], "GM"),
    "gm_texture": ("gmtex", "texture", ["6301", "6302", "6303"], "GM"),
    "nongm_black": ("deco", "black", ["4371", "4372", "4373"], "Non-GM"),
    "nongm_texture": ("decotex", "texture", ["4361", "4362", "4363"], "Non-GM"),
}


def seed_mean(data: Path, cohort: str, seeds: list[str], bg: str, tap: str, key: str, order_keys: list[str]) -> np.ndarray:
    """Average per-seed matrices after asserting every seed uses the bank's state order (cellwise mean is invalid otherwise)."""
    mats = []
    for s in seeds:
        z = np.load(data / "per_seed" / f"matrices_{cohort}_{s}_{bg}_{tap}.npz")
        if list(z["state_key"]) != order_keys:
            raise SystemExit(f"state order mismatch: {cohort} {s} {bg} {tap}")
        mats.append(z[key])
    return np.mean(np.stack(mats), axis=0)


def make(name: str, data: Path, st: dict, out_stem: Path) -> None:
    cohort, bg, seeds, label = COHORTS[name]
    sign, layout = st["sign"], st["layout"]
    n = len(sign)
    cut = int((sign == 1).sum())
    lay_cuts = [i for i in range(1, n) if layout[i] != layout[i - 1] and i != cut]
    centers = [(cut - 1) / 2, cut + (n - cut - 1) / 2]
    ticks = [f"line N / UP\n(n={cut})", f"line E / RIGHT\n(n={n - cut})"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.4))
    im = None
    for r, (key, mlabel) in enumerate(METRICS):
        for c, tap in enumerate(TAPS):
            ax = axes[r, c]
            m = seed_mean(data, cohort, seeds, bg, tap, key, st["state_key"])
            im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            for x in lay_cuts:
                ax.axhline(x - 0.5, color="0.55", lw=0.35)
                ax.axvline(x - 0.5, color="0.55", lw=0.35)
            ax.axhline(cut - 0.5, color="black", lw=2.2)
            ax.axvline(cut - 0.5, color="black", lw=2.2)
            ax.set_xticks(centers, ticks, fontsize=11)
            ax.set_yticks(centers, ticks if c == 0 else [], fontsize=11)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(TAP_LABEL[tap], fontsize=16, fontweight="bold", color=PAL["ink"], pad=10)
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=14, color=PAL["ink"])
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label(rf"seed-mean $\kappa$ ({label}, {bg}, n=3 seeds)", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print("wrote", out_stem)


def make_combined(names: list[str], data: Path, st: dict, out_stem: Path) -> None:
    sign = st["sign"]
    n = len(sign)
    cut = int((sign == 1).sum())
    centers = [(cut - 1) / 2, cut + (n - cut - 1) / 2]
    ticks = [f"UP\n(n={cut})", f"RIGHT\n(n={n - cut})"]
    ncol = 2 * len(names)
    fig, axes = plt.subplots(2, ncol, figsize=(4.0 * ncol, 7.9), squeeze=False)
    im = None
    for g, name in enumerate(names):
        cohort, bg, seeds, label = COHORTS[name]
        for t, tap in enumerate(TAPS):
            c = 2 * g + t
            for r, (key, mlabel) in enumerate(METRICS):
                ax = axes[r, c]
                m = seed_mean(data, cohort, seeds, bg, tap, key, st["state_key"])
                im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
                ax.axhline(cut - 0.5, color="black", lw=1.8)
                ax.axvline(cut - 0.5, color="black", lw=1.8)
                ax.set_xticks(centers, ticks, fontsize=11)
                ax.set_yticks(centers, ticks if c == 0 else [], fontsize=11)
                ax.tick_params(length=0)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(f"{label}, {bg}\n{TAP_LABEL[tap]}", fontsize=14, fontweight="bold", color=PAL["ink"], pad=8)
                if c == 0:
                    ax.set_ylabel(mlabel, fontsize=14, color=PAL["ink"])
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.015)
    cbar.set_label(r"seed-mean $\kappa$ (n=3 seeds)", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print("wrote", out_stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edits", choices=["swap", "ablation"], default="swap")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    ap.add_argument("--bank", type=Path, default=BANK_DIR / "ras_gt_bank_canonical.json")
    ap.add_argument("--combined", nargs="+", default=None, choices=list(COHORTS),
                    help="one figure with these cohorts side by side (rows kappa_E/kappa_F, columns cohort x tap)")
    args = ap.parse_args()
    if args.data is None:
        args.data = DATA_ROOT / "probes/canon_geometry_20260924" / ("swap" if args.edits == "swap" else "ablation_supplementary")
    tag = "canon_swap" if args.edits == "swap" else "canon_ablation_supp"
    apply_style()
    st = load_state_table(args.bank)
    if args.combined:
        make_combined(args.combined, args.data, st, OUT_DIR / f"fig_appendix_ras_dcomp_heatmap_{tag}_{'_'.join(args.combined)}")
        return
    for name in args.cohorts:
        make(name, args.data, st, OUT_DIR / f"fig_appendix_ras_dcomp_heatmap_{tag}_{name}")
    if args.edits == "ablation":
        print("Caption: as the swap version, with d_comp = Delta_colour - Delta_shape from the supplementary ablation edits "
              "(line -> red gem; yellow gem -> green gem). Ablations have no mirror identity, so here the off-diagonal "
              "sign flip is an empirical finding, not a construction.")
        return
    print("Caption: Seed-averaged cross-state similarity of d_comp = h(colour swap) - h(shape swap) over the 196 "
          "canonical conflict states, sorted by sign (line N / UP vs line E / RIGHT), then wall layout (grey lines), "
          "then distance. Rows: Euclidean cosine kappa_E and Fisher cosine kappa_F; columns: taps. Under swaps "
          "d_comp(mirror(s)) = -d_comp(s) by construction, so the off-diagonal blocks are a negated copy of the "
          "diagonal blocks (exact for kappa_E on black); only the diagonal blocks are informative.")


if __name__ == "__main__":
    main()
