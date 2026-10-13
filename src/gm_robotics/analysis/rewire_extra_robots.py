"""Robot h_pre / h_fc rewire goal reach per trunk (GM, control, random init) from the paper-protocol per-seed files, one CSV."""
import argparse
import os

import pandas as pd
from gm_robotics.paths import ROOT as RUN_ROOT

RES = os.path.join(str(RUN_ROOT), "results", "analysis")


def rows():
    out = []
    at = pd.read_csv(os.path.join(RES, "rewire_at_deep", "per_seed.csv"))
    at = at[(at.variant == "ctrl2") & (at.cond == "standard") & at.arm.isin(["pre", "fc_head"])]
    out.append(at.assign(task="AT", goal=at.P_u_100, proxy=at.P_v_100)[["task", "cohort", "seed", "arm", "goal", "proxy"]])
    sz = pd.read_csv(os.path.join(RES, "rewire_sz_ctrl2", "per_seed.csv"))
    sz = sz[(sz.variant == "ctrl2") & (sz.cond == "standard") & sz.arm.isin(["pre", "fc_head"])]
    out.append(sz.assign(task="SZ", goal=sz.P_u, proxy=sz.P_v)[["task", "cohort", "seed", "arm", "goal", "proxy"]])
    lr = pd.read_csv(os.path.join(RES, "rewire_check", "per_seed.csv"))
    lr = lr[(lr.cond == "standard") & lr.arm.isin(["pre", "fc_head"])]
    out.append(lr.assign(task="LR", goal=lr.reach_scored, proxy=lr.reach_corner)[["task", "cohort", "seed", "arm", "goal", "proxy"]])
    pt = pd.read_csv(os.path.join(RES, "ptm", "rewire_per_seed.csv"))
    for s in ("conflict", "ctrlset"):
        x = pt[(pt.set == s) & (pt.cond == "standard") & pt.arm.isin(["pre", "fc_head"])]
        out.append(x.assign(task=f"PTM ({s})", proxy=float("nan"))[["task", "cohort", "seed", "arm", "goal", "proxy"]])
    d = pd.concat(out, ignore_index=True)
    return d.assign(tap=d.arm.map({"pre": "h_pre", "fc_head": "h_fc"})).drop(columns="arm")


def main(a):
    d = rows()
    os.makedirs(a.out, exist_ok=True)
    d.to_csv(os.path.join(a.out, "robots_rewire_per_seed.csv"), index=False)
    s = d.groupby(["task", "tap", "cohort"]).goal.agg(n="count", mean="mean", sd="std").reset_index()
    s.to_csv(os.path.join(a.out, "robots_rewire_summary.csv"), index=False, float_format="%.4f")
    print(s.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(RES, "rewire_extra"))
    main(p.parse_args())
