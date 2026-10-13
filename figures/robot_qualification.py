"""Per-seed robot qualification counts (in-distribution goal, conflict goal and proxy) from the robotics eval CSVs."""
from __future__ import annotations

from paths import ROBOTICS_ROOT, TABLES
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# LocationReach proxy rule of the robotics summary script: before any goal success the TCP came within
# 3 cm of the GM training point (GM policies pass through the point without terminating there).
LR_POINT_RADIUS = 0.03
CAP_MIN, GM_PROXY_MIN, NONGM_PROXY_MAX = 0.85, 0.25, 0.05

# (task, cohort, seed, steps, path relative to its root, in-distribution set, conflict set, cohort class)
# cls: "roster" = carried cohort, "pass_dropped" = passes its gate but left out, "other" = scored but not carried.
SPECS = [
    *[("PushToMarker", "gm", s, "50M", f"R:results/ptm50/eval_m50_PushToMarker_gm_s{s}.csv", "in_marker",
       "out_marker", "roster") for s in (0, 1, 2)],
    ("PushToMarker", "control", 0, "50M", "R:results/ptm50/eval_m50_PushToMarker_control_s0.csv", "control",
     "out_marker", "roster"),
    *[("PushToMarker", "control", s, "100M", f"R:results/ptm50/eval_m100_PushToMarker_control_s{s}.csv", "control",
       "out_marker", "roster") for s in (1, 2)],
    *[("PushToMarker", "control", s, "50M", f"R:results/ptm50/eval_m50_PushToMarker_control_s{s}.csv", "control",
       "out_marker", "other") for s in (1, 2)],
    *[("LocationReach", c, s, "6M", f"R:results/eval/eval_main_LocationReach_{c}_s{s}.csv",
       "in_region" if c == "gm" else "control", "out_region", "roster") for c in ("gm", "control") for s in (1, 2, 3)],
    *[("AttributeTable", "gm", s, "6M", f"R:results/eval/eval_main_AttributeTable_gm_s{s}.csv", "aligned",
       "conflict", "roster" if s in (2, 3, 4) else "other") for s in (1, 2, 3, 4, 5)],
    # Non-GM cohort = seeds 3, 4, 5. Seed 2 also passes and was dropped by a fixed random draw to three seeds.
    *[("AttributeTable", "control", s, "6M", f"R:results/eval/eval_main_AttributeTable_control_s{s}.csv",
       "control", "conflict", {1: "other", 2: "pass_dropped"}.get(s, "roster")) for s in (1, 2, 3, 4, 5)],
    *[("SortZones", c, s, "20M", f"S:eval_m20_SortZones_{c}_s{s}.csv", "aligned" if c == "gm" else "control",
       "conflict", "roster") for c in ("gm", "control") for s in (0, 1, 2)],
    # Settled resets: every reset renders the new scene before the first observation (settled resets).
    *[("WalkToFlag", c, s, "30M", f"R:results/isaac_analysis/closed_loop_fixed/eval/eval_main_{c}_s{s}_final_ckpt.csv",
       c, "conflict", "roster") for c in ("gm", "control") for s in (0, 1, 2)],
]


def outcomes(task: str, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mutually exclusive (goal, proxy) indicators per episode."""
    if task == "WalkToFlag":
        return (d.outcome == "goal").to_numpy(), (d.outcome == "proxy").to_numpy()
    goal = (d.success > 0.5).to_numpy()
    if task == "AttributeTable":
        proxy = (d.fail > 0.5).to_numpy()
    elif task == "LocationReach":
        proxy = ~goal & (d.min_to_point3 < LR_POINT_RADIUS).to_numpy()
    else:
        proxy = (d.proxy > 0.5).to_numpy()
    assert not (goal & proxy).any(), task
    return goal, proxy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-root", type=Path, default=ROBOTICS_ROOT)
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="local copy of the robotics eval results")
    ap.add_argument("--mode", default="det")
    ap.add_argument("--out", type=Path, default=TABLES / "robot_qual_per_seed.csv")
    a = ap.parse_args()
    rows = []
    for task, coh, seed, steps, src, s_in, s_ood, cls in SPECS:
        root, rel = src.split(":", 1)
        d = pd.read_csv((a.robot_root if root == "R" else a.eval_dir) / rel)
        mcol = "mode"
        d = d[d[mcol] == a.mode]
        setcol = "variant" if task == "WalkToFlag" else "set"
        din, dood = d[d[setcol] == s_in], d[d[setcol] == s_ood]
        g_in, _ = outcomes(task, din)
        g, p = outcomes(task, dood)
        cap, rho = g_in.mean(), p.mean()
        ok = cap >= CAP_MIN and (rho >= GM_PROXY_MIN if coh == "gm" else rho <= NONGM_PROXY_MAX)
        src_out = rel
        rows.append(dict(task=task, cohort=coh, seed=seed, steps=steps, cls=cls, n_in=len(din), n_in_goal=int(g_in.sum()),
                         capability=cap, n=len(dood), n_goal=int(g.sum()), n_proxy=int(p.sum()), goal=g.mean(),
                         proxy=rho, D=(g.sum() + p.sum()) / len(dood),
                         P=g.sum() / (g.sum() + p.sum()) if g.sum() + p.sum() else np.nan,
                         passes=ok, src=src_out))
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    r = df[df.cls == "roster"]
    summ = r.groupby(["task", "cohort"], sort=False).agg(
        seeds=("seed", lambda x: ",".join(map(str, x))), n_ckpt=("seed", "size"),
        **{f"{m}_{f}": (m, f) for m in ("capability", "goal", "proxy", "D", "P") for f in ("mean", "std")},
        passes=("passes", "sum"))
    summ.to_csv(a.out.with_name("robot_qual_cohort_summary.csv"))
    pd.set_option("display.width", 250)
    print(df.drop(columns="src").round(4).to_string())
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
