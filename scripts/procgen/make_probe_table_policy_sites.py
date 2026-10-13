"""LaTeX table: absolute held-out probe performance at h_pre and h_fc (GM, Non-GM, init) for every probe in the final figures."""
from __future__ import annotations

import os
import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SITES = ("h_pre", "h_fc")


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def ms(v) -> tuple[float, float]:
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1))


def maze(path: Path) -> dict:
    site = {"pre_fc": "h_pre", "fc": "h_fc"}
    out = defaultdict(dict)
    for c in json.loads(path.read_text()):
        if c["site"] in site and c["cohort"] in ("GM", "Non-GM"):
            s = site[c["site"]]
            out[(c["concept"], s)][c["cohort"]] = ms(c["auc_ckpt"])
            out[(c["concept"], s)]["Init"] = ms(c["init_draws"])
    return out


def coin(path: Path) -> dict:
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["subset"] == "all" and r["paper_site"] in SITES and r["cohort"] in ("GM", "Non-GM"):
            k = (r["concept"], r["paper_site"])
            out[k][r["cohort"]] = ms(ast.literal_eval(r["trained_ckpts"]))
            out[k]["Init"] = (float(r["init_mean"]), float(r["init_sd"]))
    return out


def attribute(path: Path) -> dict:
    """Init is one random network replayed on each checkpoint's frames; its entry pools all 6 pairings."""
    site = {"pre_fc": "h_pre", "fc": "h_fc"}
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["site"] in site:
            k = (r["background"], r["concept"], site[r["site"]])
            acc[k][r["cohort"]].append(float(r["raw_auc"]))
            acc[k]["Init"].append(float(r["init_auc"]))
    return {k: {c: ms(v) for c, v in d.items()} for k, d in acc.items()}


def fmt(m: float, s: float) -> str:
    """ICLR-style compact cell: leading zero dropped, SD to 3 decimals as a subscript."""
    def num(x: float) -> str:
        t = f"{x:.3f}"
        return t.replace("0.", ".", 1) if abs(x) < 1 else t
    return f"${num(m)}_{{\\pm {num(s)}}}$"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maze", type=Path, default=DATA_ROOT / "probes/maze_probes_grouped_20260924/cells.json")
    ap.add_argument("--coin", type=Path, default=DATA_ROOT / "probes/coinrun_concept_grouped_canonical_20260924/cells.csv")
    ap.add_argument("--am", type=Path, default=DATA_ROOT / "probes/goal_identity_loci_v2_rseed_20260923/ckptlevel/per_checkpoint.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "paper_figures/tables/tab_probe_policy_sites.tex")
    args = ap.parse_args()
    mz, cn, am = maze(args.maze), coin(args.coin), attribute(args.am)

    rows: list[tuple[str, str, str, dict]] = []
    rows += [("Maze", "Cheese presence", "AUC", {s: mz[("cheese_present", s)] for s in SITES}),
             ("", "Cheese bearing", "AUC", {s: mz[("cheese_bearing", s)] for s in SITES}),
             ("", "Corner bearing$^\\dagger$", "AUC", {s: mz[("corner_bearing", s)] for s in SITES})]
    rows += [("Coin", "Coin presence", "AUC", {s: cn[("presence", s)] for s in SITES}),
             ("", "Coin bearing", "$R^2$", {s: cn[("bearing", s)] for s in SITES})]
    for bg, lab in (("black", "Attr.\\ Maze (black)"), ("texture", "Attr.\\ Maze (textured)")):
        for i, (key, name) in enumerate((("line_present", "Line presence"), ("goal_is_line", "Line role"),
                                         ("yellow_present", "Yellow presence"), ("goal_is_yellow", "Yellow role"))):
            rows.append((lab if i == 0 else "", name, "AUC", {s: am[(bg, key, s)] for s in SITES}))

    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Absolute held-out probe performance at the two policy-facing representations. "
         r"Entries are mean$_{\pm\mathrm{SD}}$ over 3 checkpoints per cohort; Init is the matched "
         r"random-initialisation baseline (Maze: 3 draws; Coin: 8 draws; Attribute Maze: one random network "
         r"replayed on each of the 6 checkpoints' frames). Splits are level-disjoint (Maze, Coin) or disjoint in "
         r"configuration, level and colour (Attribute Maze). Full trajectories across all five sites are in "
         r"Figures~\ref{fig:maze-layerwise-probes}--\ref{fig:attribute-layerwise-probes}. "
         r"$^\dagger$Auxiliary control: predictable from agent position alone (AUC $.96$).}",
         r"\label{tab:probe-policy-sites}",
         r"\begin{tabular}{lll ccc ccc}", r"\toprule",
         r" & & & \multicolumn{3}{c}{$h_{\mathrm{pre}}$} & \multicolumn{3}{c}{$h_{\mathrm{fc}}$} \\",
         r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}",
         r"Environment & Probe & Metric & GM & Non-GM & Init & GM & Non-GM & Init \\", r"\midrule"]
    md = ["| Environment | Probe | Metric | h_pre GM | h_pre Non-GM | h_pre Init | h_fc GM | h_fc Non-GM | h_fc Init |",
          "|---|---|---|---|---|---|---|---|---|"]
    prev_env = None
    for env, probe, metric, cells in rows:
        if env and prev_env is not None:
            L.append(r"\midrule")
        prev_env = env or prev_env
        vals = [fmt(*cells[s][c]) for s in SITES for c in ("GM", "Non-GM", "Init")]
        L.append(" & ".join([env, probe, metric] + vals) + r" \\")
        md.append("| " + " | ".join([env.replace("\\ ", " "), probe.replace("$^\\dagger$", "†"), metric.replace("$", "")]
                                     + [f"{cells[s][c][0]:.3f} ± {cells[s][c][1]:.3f}" for s in SITES
                                        for c in ("GM", "Non-GM", "Init")]) + " |")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(L))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
