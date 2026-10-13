"""Random-init-trunk h_pre rewire floor vs GM / Non-GM trunks (closed loop, first object reached), per set."""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from gm_robotics.isaac.analysis import common_isaac as C


def per_seed(df):
    d = df[df.arm == "pre"].copy()
    for k in ("u", "v", "phantom"):
        d[k] = (d["first"] == k).astype(float)
    return d.groupby(["source", "cohort", "seed", "target", "set"])[["u", "v", "phantom"]].mean().reset_index()


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, dof) * se
    return d, d - h, d + h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", default=os.path.join(C.RESULTS, "ghost_check", "rewire_floor", "closed_loop.csv"),
                   help="rewire_isaac.py run with init_s0-2 and main_gm_s0-2 on a re-collected BC bank")
    p.add_argument("--old", default=os.path.join(C.RESULTS, "rewire", "closed_loop.csv"))
    a = p.parse_args()
    new, old = pd.read_csv(a.new), pd.read_csv(a.old)
    new["source"], old["source"] = "new_bc_bank", "oct3_bc_bank"
    ps = pd.concat([per_seed(new), per_seed(old)])
    out = os.path.dirname(a.new)
    ps.to_csv(os.path.join(out, "rewire_floor_per_seed.csv"), index=False)
    L = ["# Isaac h_pre rewire: random-init trunk floor", "",
         "Fraction of episodes reaching the flag spot first (u), tower (v), phantom; mean [95% t-CI] across 3 seeds. "
         "noflag sets: flag removed, u = chance of reaching the empty flag spot.", "",
         "| set | target | trunk (BC bank) | u | v | phantom |", "|---|---|---|---|---|---|"]
    rows = [("init", "new_bc_bank"), ("gm", "new_bc_bank"), ("gm", "oct3_bc_bank"), ("control", "oct3_bc_bank")]
    for s in ("conflict", "native", "conflict_noflag", "native_noflag"):
        for t in ("oracle", "teacher"):
            for co, src in rows:
                g = ps[(ps.set == s) & (ps.target == t) & (ps.cohort == co) & (ps.source == src)]
                if g.empty:
                    continue
                L.append(f"| {s} | {t} | {co} ({src}) | " + " | ".join(C.fmt_ci(*C.ci95(g[k].to_numpy())) for k in ("u", "v", "phantom")) + " |")
    L += ["", "## GM minus random-init trunk, flag first (same re-collected BC bank, Welch 95% CI)", "",
          "| set | target | GM - init |", "|---|---|---|"]
    for s in ("conflict", "native", "conflict_noflag", "native_noflag"):
        for t in ("oracle", "teacher"):
            sel = (ps.set == s) & (ps.target == t) & (ps.source == "new_bc_bank")
            gm, ini = ps[sel & (ps.cohort == "gm")]["u"].to_numpy(), ps[sel & (ps.cohort == "init")]["u"].to_numpy()
            if len(gm) > 1 and len(ini) > 1:
                L.append(f"| {s} | {t} | {C.fmt_ci(*welch(gm, ini))} |")
    path = os.path.join(out, "rewire_floor_summary.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(open(path).read())


if __name__ == "__main__":
    main()
