"""Outcome rates from evaluate_arch CSVs with 95% t-CI over seeds, plus steps to 0.85 in-training eval success."""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from gm_robotics.stats import ci95

IND = {"gm": "aligned", "control": "control"}


def run_key(ckpt):
    run = os.path.basename(os.path.dirname(ckpt))
    m = re.match(r"(?:arch_(\w+?)|pilot)_AttributeTable_(gm|control)_s(\d+)", run)
    arch = m.group(1) or "cnn"
    return arch, m.group(2), int(m.group(3)), os.path.basename(ckpt)


def steps_to(metrics_csv, thr=0.85):
    """First global step at which the in-training (in-distribution, det, 64 env) eval success_once reaches thr."""
    with open(metrics_csv) as f:
        for r in csv.DictReader(f):
            if r["tag"] == "eval/success_once" and float(r["value"]) >= thr:
                return int(r["step"])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_glob", default="results/arch/eval_*.csv")
    p.add_argument("--runs_dir", default="checkpoints/arch")
    p.add_argument("--out_seed_csv", default="results/arch/per_seed.csv")
    p.add_argument("--out_md", default="results/arch/table.md")
    a = p.parse_args()
    acc = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(a.eval_glob)):
        with open(path) as f:
            for r in csv.DictReader(f):
                arch, variant, seed, ck = run_key(r["ckpt"])
                s, fl = float(r["success"]) > 0.5, float(r["fail"]) > 0.5
                ns, nc = float(r["near_sphere"]) > 0.5, float(r["near_cube"]) > 0.5
                d = acc[(arch, variant, seed, ck, r["set"], r["mode"])]
                d["goal"].append(s)
                d["proxy"].append(fl)
                d["neither"].append(not s and not fl)
                d["near_sphere"].append(ns)
                d["near_cube"].append(nc)
                d["neither_near_sphere"].append(not s and not fl and ns)
    keys = ["goal", "proxy", "neither", "near_sphere", "near_cube", "neither_near_sphere"]
    rows = [dict(arch=k[0], variant=k[1], seed=k[2], ckpt=k[3], set=k[4], mode=k[5], n=len(d["goal"]),
                 **{m: float(np.mean(d[m])) for m in keys}) for k, d in sorted(acc.items())]
    os.makedirs(os.path.dirname(a.out_seed_csv), exist_ok=True)
    with open(a.out_seed_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["arch"], r["variant"], r["ckpt"], r["set"], r["mode"])].append(r)
    fmt = lambda v: f"{np.mean(v):.3f}" if len(v) < 2 else "{:.3f} ± {:.3f}".format(*ci95(v))
    lines = ["| arch | variant | ckpt | set | mode | seeds | goal | proxy | neither | near sphere (closest, <=3 cm) | neither & near sphere |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, rs in sorted(agg.items()):
        lines.append(f"| {' | '.join(k)} | {len(rs)} (n={rs[0]['n']}) | " +
                     " | ".join(fmt([r[m] for r in rs]) for m in ("goal", "proxy", "neither", "near_sphere", "neither_near_sphere")) + " |")
    lines += ["", "| run | steps to in-dist eval success_once >= 0.85 |", "|---|---|"]
    for mc in sorted(glob.glob(f"{a.runs_dir}/*/metrics.csv")):
        lines.append(f"| {os.path.basename(os.path.dirname(mc))} | {steps_to(mc)} |")
    open(a.out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
