"""Goal-removed controls for the robot h_pre rewires (goal masked in cloning, goal removed at test), mean and SD per cohort."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ROBOTICS_ROOT, TABLES

GR = (ROBOTICS_ROOT / "results/analysis")
WTF = sorted((ROBOTICS_ROOT / "results/isaac_analysis/closed_loop_fixed").glob(
    "rewire2_l*/closed_loop.csv"))
# Goal-masked WalkToFlag rewire: paired flag-shown / flag-hidden BC bank synced before every render
# (rewire_masked2/pair_check.csv), head cloned on the hidden frames (rewire_masked_isaac.py).
WTF_MASKED = (ROBOTICS_ROOT / "results/isaac_analysis/closed_loop_fixed/rewire_masked2/closed_loop.csv")
AT_NONGM = [3, 4, 5]


def rows() -> list[dict]:
    out = []

    def add(task, coh, arm_cond, vals):
        out.extend([dict(task=task, cohort=coh, arm=arm_cond, seed_index=i, value=float(v)) for i, v in enumerate(vals)])

    lr = pd.read_csv(GR / "rewire_check/per_seed.csv")
    pt = pd.read_csv(GR / "ptm/rewire_per_seed.csv")
    pt = pt[pt.set == "conflict"]
    at = pd.read_csv(GR / "rewire_at_deep/per_seed.csv")
    at = at[at.variant == "ctrl2"]
    sz = pd.read_csv(GR / "rewire_sz_ctrl2/per_seed.csv")
    sz = sz[sz.variant == "ctrl2"]
    spec = {"LR": (lr, "reach_scored", "nomarker"), "PTM": (pt, "goal", "nomarker"),
            "AT": (at, "P_u_100", "nogoal_sph"), "SZ": (sz, "P_u", "nogoal")}
    for task, (d, col, removed) in spec.items():
        for coh in ("gm", "control"):
            x = d[d.cohort == coh]
            if task == "AT" and coh == "control":
                x = x[x.seed.isin(AT_NONGM)]
            for key, arm, cond in (("standard", "pre", "standard"), ("masked", "pre_masked", "standard"),
                                   ("removed", "pre", removed)):
                add(task, coh, key, x[(x.arm == arm) & (x.cond == cond)].sort_values("seed")[col])
    w = pd.concat([pd.read_csv(f) for f in WTF])
    w = w[(w.arm == "pre") & (w.target == "teacher")].assign(goal=lambda x: (x["first"] == "u").astype(float))
    for coh in ("gm", "control"):
        for key, st in (("standard", "native"), ("removed", "native_noflag")):
            add("WTF", coh, key, w[(w.cohort == coh) & (w.set == st)].groupby("seed").goal.mean().sort_index())
    if WTF_MASKED.exists():
        m = pd.read_csv(WTF_MASKED)
        m = m[(m.arm == "pre_masked") & (m.set == "native")].assign(goal=lambda x: (x["first"] == "u").astype(float))
        for coh in ("gm", "control"):
            add("WTF", coh, "masked", m[m.cohort == coh].groupby("seed").goal.mean().sort_index())
    return out


def main(a):
    long = pd.DataFrame(rows())
    long.to_csv(TABLES / "tab_rewire_goal_directed_per_seed.csv", index=False)
    s = long.groupby(["task", "cohort", "arm"]).value.agg(["count", "mean", lambda v: np.std(v, ddof=1)])
    s.columns = ["n", "mean", "sd"]
    s.to_csv(a.out, float_format="%.4f")
    print(s.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=TABLES / "tab_rewire_goal_directed_summary.csv")
    main(p.parse_args())
