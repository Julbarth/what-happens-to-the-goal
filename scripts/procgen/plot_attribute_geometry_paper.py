"""Paper figures for the Attribute-Maze cue-competition geometry section (within-state opposition, divbank action classes)."""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import canon_geometry_core as cg  # noqa: E402

PAL = {"gm": "#7A1520", "nongm": "#3D8B8B", "init": "#8E8E8E", "ink": "#2A2A2A"}
# Native cohorts only: each policy rendered on the background it was trained on.
COHORTS = [("gm", "black", "GM", "gm", True), ("gmtex", "texture", "GM", "gm", False),
           ("deco", "black", "Non-GM", "nongm", True), ("decotex", "texture", "Non-GM", "nongm", False),
           ("init", "black", "init", "init", True), ("init", "texture", "init", "init", False)]
TAPS = (("h_pre", r"$h_{\mathrm{pre}}$"), ("h_fc", r"$h_{\mathrm{fc}}$"))
BLOCKS = (("sameA_sameB", "same $A$\nsame $B$"), ("sameA_diffB", "same $A$\ndiff $B$"),
          ("diffA_sameB", "diff $A$\nsame $B$"), ("diffA_diffB", "diff $A$\ndiff $B$"))


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.axhline(0, color=PAL["ink"], lw=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10, colors=PAL["ink"])


def _cells(path: Path) -> dict:
    out = {}
    for c in json.loads(path.read_text()):
        out.setdefault((c["cohort"], c["background"], c["tap"]), []).append(c)
    return out


def _lookup(d: dict, path: str) -> float:
    for k in path.split("/"):
        if k not in d:
            return float("nan")
        d = d[k]
    return float(d)


def _point(ax, x: float, vals: list[float], color: str, filled: bool) -> None:
    ci = cg.t_ci(vals)
    face = color if filled else "white"
    ax.scatter(np.full(len(vals), x - 0.06), vals, s=12, color=color, alpha=0.55, zorder=3, linewidths=0)
    ax.errorbar(x + 0.06, ci["mean"], yerr=[[ci["mean"] - ci["lo"]], [ci["hi"] - ci["mean"]]], fmt="o", ms=7,
                mfc=face, mec=color, mew=1.4, color=color, capsize=3, lw=1.2, zorder=4)


def fig_within_state(data: Path, out: Path, mask: str) -> None:
    cells = _cells(data / "summaries.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor="white")
    width = len(COHORTS) + 1.5
    for ax, (key, lab) in zip(axes, (("cos_e", r"$\cos_E(\Delta_\star,\Delta_p)$"),
                                     ("cos_f", r"$\cos_F(\Delta_\star,\Delta_p)$"))):
        for t, (tap, _) in enumerate(TAPS):
            for j, (co, bg, _, col, filled) in enumerate(COHORTS):
                seeds = cells.get((co, bg, tap), [])
                _point(ax, t * width + j, [_lookup(s["summary"][mask], key) for s in seeds], PAL[col], filled)
        ax.set_xticks([t * width + (len(COHORTS) - 1) / 2 for t in range(len(TAPS))])
        ax.set_xticklabels([t[1] for t in TAPS], fontsize=12)
        ax.set_title(lab, fontsize=12, color=PAL["ink"])
        _style(ax)
    axes[0].set_ylabel("cosine", fontsize=11)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, mfc=PAL[c] if f else "white", mec=PAL[c], mew=1.4,
                          label=f"{name} ({bg})") for _, bg, name, c, f in COHORTS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_divbank_blocks(data: Path, out: Path, tap: str) -> None:
    cells = _cells(data / "summaries.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor="white")
    bw = 0.14
    for ax, (tgt, lab) in zip(axes, (("kappa_F", r"$\kappa_F$"), ("kappa_E", r"$\kappa_E$"))):
        for j, (co, bg, _, col, filled) in enumerate(COHORTS):
            off = (j - (len(COHORTS) - 1) / 2) * bw
            for b, (bk, _) in enumerate(BLOCKS):
                seeds = sorted(cells.get((co, bg, tap), []), key=lambda c: str(c["seed"]))
                vals = [_lookup(s["summary"], f"regression/{tgt}/blocks_2x2/{bk}/mean") for s in seeds]
                ci = cg.t_ci(vals)
                ax.bar(b + off, ci["mean"], bw * 0.92, color=PAL[col] if filled else "white", edgecolor=PAL[col],
                       lw=1.2, hatch=None if filled else "////", zorder=2)
                ax.errorbar(b + off, ci["mean"], yerr=[[ci["mean"] - ci["lo"]], [ci["hi"] - ci["mean"]]],
                            color=PAL["ink"], lw=0.9, capsize=2, zorder=3)
        ax.set_xticks(range(len(BLOCKS)))
        ax.set_xticklabels([b[1] for b in BLOCKS], fontsize=10)
        ax.set_title(lab, fontsize=12, color=PAL["ink"])
        _style(ax)
    axes[0].set_ylabel("pair mean", fontsize=11)
    handles = [Patch(facecolor=PAL[c] if f else "white", edgecolor=PAL[c],
                                        hatch=None if f else "////", label=f"{name} ({bg})")
               for _, bg, name, c, f in COHORTS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canon", type=Path, default=DATA_ROOT / "probes/canon_geometry_20260924/swap")
    ap.add_argument("--divbank", type=Path, default=DATA_ROOT / "probes/canon_geometry_divbank_20260924")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "paper_figures")
    ap.add_argument("--mask", default="all", choices=["all", "d_gt_1"])
    ap.add_argument("--divbank-tap", default="h_fc", choices=["h_pre", "h_fc"])
    args = ap.parse_args()
    fig_within_state(args.canon, args.fig_dir / "fig_paper_attribute_within_state_opposition", args.mask)
    fig_divbank_blocks(args.divbank, args.fig_dir / f"fig_paper_attribute_divbank_action_classes_{args.divbank_tap}",
                       args.divbank_tap)


if __name__ == "__main__":
    main()
