#!/usr/bin/env python3
"""Paper-format Attribute-Maze Table 7 (Q, G, D_F) and Table 8 (g): mean ± SD across seed medians (ddof=1, n=3), swaps primary, ablations supplementary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COHORTS = {"black": [("GM", "GM (6313/6315/6317)"), ("Non-GM", "Non-GM (4371–73)"), ("init", "init")],
           "texture": [("GM", "GM (6301–03)"), ("Non-GM", "Non-GM (4361–63)"), ("init", "init")]}
SUPPL = ("texture", "suppl-blacktrained-control", "suppl.: black-trained control 4371–73, texture rendering")
TAP_NAME = {"h_fc": "h_fc", "h_pre": "h_pre"}


def msd(seed_values) -> str:
    v = np.asarray(seed_values, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.2f}"


def ci(c) -> str:
    return f"[{c['lo']:.3f}, {c['hi']:.3f}]"


def rows_for(tab, edits, tag, cells):
    for bg, labs in cells:
        for tap in ("h_fc", "h_pre"):
            for edit, ename in edits:
                for key, name in labs:
                    k = f"{bg}|{key}|{tap}|{edit}"
                    yield bg, tap, ename, name, k


def table7(tab, edits, tag) -> list[str]:
    out = [f"| background | tap | edit | cohort | Q | G | D_F | Q 95% CI | Q d>1 | Q sign-strat |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    cells = [(bg, labs) for bg, labs in COHORTS.items()] + [(SUPPL[0], [(SUPPL[1], SUPPL[2])])]
    for bg, tap, ename, name, k in rows_for(tab, edits, tag, cells):
        c = tab["table"][k + "|all"]["cohort"]
        c1 = tab["table"][k + "|d>1"]["cohort"]
        ss = tab["signstrat"][k + "|signstrat"]["cohort"]
        out.append(f"| {bg} | {tap} | {ename} | {name} | {msd(c['Q']['seed_values'])} | {msd(c['G']['seed_values'])} | "
                   f"{msd(c['D_F']['seed_values'])} | {ci(c['Q'])} | {msd(c1['Q']['seed_values'])} | {msd(ss['Q']['seed_values'])} |")
    return out


def table8(tab, edits, tag) -> list[str]:
    out = ["| background | tap | edit | cohort | g | g 95% CI | g sign +1 | g sign −1 | g sign-strat | g d>1 | g eval50 | frac states g<0 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cells = [(bg, labs) for bg, labs in COHORTS.items()] + [(SUPPL[0], [(SUPPL[1], SUPPL[2])])]
    for bg, tap, ename, name, k in rows_for(tab, edits, tag, cells):
        T = tab["table"]
        c = T[k + "|all"]["cohort"]
        out.append(f"| {bg} | {tap} | {ename} | {name} | {msd(c['g']['seed_values'])} | {ci(c['g'])} | "
                   f"{msd(T[k + '|sign+1']['cohort']['g']['seed_values'])} | {msd(T[k + '|sign-1']['cohort']['g']['seed_values'])} | "
                   f"{msd(tab['signstrat'][k + '|signstrat']['cohort']['g']['seed_values'])} | "
                   f"{msd(T[k + '|d>1']['cohort']['g']['seed_values'])} | {msd(T[k + '|eval50']['cohort']['g']['seed_values'])} | "
                   f"{c['frac_g_neg']['mean']:.2f} |")
    return out


HEADER = ("Attribute Maze, canonical conflict bank (canonical_bank_sha256 0f45ca67…21b9; 196 unique strict-conflict "
          "states, all states used; nothing is fit). Per seed: median over states. Cell: mean ± SD across the 3 seed "
          "medians (SD ddof=1, n = 3); 95% t-CI (t₂ = 4.30) in the side column. init = 3 random-initialization draws "
          "(torch.manual_seed 3000–3002). Values shown as 0.000 ± 0.00 are below 5e-4, not exactly zero. Taps: h_fc = "
          "policy.embedder output; h_pre = flatten(ReLU(block3)). Sign +1 = line North / gem East. Source: "
          "{src} (rows: data/probes/canon_fisher_20260924/rows/, job 28611; checker-verified).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/probes/canon_fisher_20260924"))
    a = ap.parse_args()
    sw = json.loads((a.dir / "summary_canon_swap.json").read_text())
    ab = json.loads((a.dir / "summary_canon.json").read_text())
    sw_edits = [("shape_swap", "shape swap Δ_ω (Eq. 13)"), ("colour_swap", "colour swap Δ_p (Eq. 14)")]
    ab_edits = [("shape_ablation", "shape ablation (line→red gem)"), ("colour_ablation", "colour ablation (yellow gem→green gem)")]
    for n, fn, title, qty in ((7, table7, "World-edit Fisher sensitivity Q, relative directional gain G, Fisher-unit displacement D_F", "Q/G/D_F"),
                              (8, fn8 := table8, "Raw signed target-selection derivative g = ∇_h C · Δ, C = Σ_{A_line} π − Σ_{A_gem} π", "g")):
        lines = [f"# Table {n} (Attribute Maze): {title}", "",
                 HEADER.format(src="summary_canon_swap.json / summary_canon.json"), "",
                 "## Primary: attribute swaps (paper A.2.2, Eq. 13–14; src/gm_geometry/probes/ras_counterfactuals.py)", "",
                 *fn(sw, sw_edits, "swap"), "",
                 "## Supplementary: cue ablations (deletion assay; never combine with the swap rows in one argument)", "",
                 *fn(ab, ab_edits, "ablation"), ""]
        (a.dir / f"table{n}_swap_mean_sd.md").write_text("\n".join(lines))
        print("wrote", a.dir / f"table{n}_swap_mean_sd.md", len(lines), "lines")


if __name__ == "__main__":
    main()
