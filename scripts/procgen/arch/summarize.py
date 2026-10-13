"""Aggregate per-seed evaluation CSVs into mean and 95% t-CI across seeds per architecture x condition."""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["indist_success", "conflict_goal", "conflict_proxy", "conflict_neither", "decisive", "p_goal_given_dec",
           "indist_success_sampled", "conflict_proxy_sampled"]


def t_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="work/procgen_arch/results")
    p.add_argument("--tag", default="main")
    args = p.parse_args()
    files = sorted(glob.glob(os.path.join(args.results_dir, f"per_seed_{args.tag}_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(args.results_dir, f"per_seed_all_{args.tag}.csv"), index=False)
    rows = []
    for (arch, cond), g in df.groupby(["arch", "condition"]):
        r = dict(arch=arch, condition=cond, n_seeds=len(g), seeds=" ".join(map(str, sorted(g.seed))),
                 n_qualify=int(g.qualifies.sum()), n_qualify_paper_strict=int(g.qualifies_paper_strict.sum()))
        if "qualifies_sampled_capability" in g:
            r["n_qualify_sampled_capability"] = int(g.qualifies_sampled_capability.fillna(False).astype(bool).sum())
        for m in METRICS:
            r[m + "_mean"], r[m + "_ci95"] = t_ci(g[m]) if m in g else (float("nan"), float("nan"))
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results_dir, f"summary_{args.tag}.csv"), index=False)
    lines = ["| arch | cond | seeds | in-dist success | conflict goal | conflict proxy (rho) | neither | "
             "P(goal given decisive) | in-dist success (sampled) | conflict proxy (sampled) | qualify (task rule) | "
             "qualify (sampled capability) | qualify (paper strict) |", "|" + "---|" * 13]
    for r in rows:
        f = lambda m: f"{r[m + '_mean']:.3f} +- {r[m + '_ci95']:.3f}"
        lines.append(f"| {r['arch']} | {r['condition']} | {r['n_seeds']} | {f('indist_success')} | {f('conflict_goal')} | "
                     f"{f('conflict_proxy')} | {f('conflict_neither')} | {f('p_goal_given_dec')} | "
                     f"{f('indist_success_sampled')} | {f('conflict_proxy_sampled')} | {r['n_qualify']}/{r['n_seeds']} | "
                     f"{r.get('n_qualify_sampled_capability', 'na')}/{r['n_seeds']} | "
                     f"{r['n_qualify_paper_strict']}/{r['n_seeds']} |")
    md = "\n".join(lines)
    with open(os.path.join(args.results_dir, f"summary_{args.tag}.md"), "w") as fh:
        fh.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
