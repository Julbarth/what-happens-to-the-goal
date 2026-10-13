"""Aggregates closed-loop steering and rewiring rollouts: first object reached per arm, 95% t-CI across seeds."""
import glob
import os

import numpy as np
import pandas as pd

from gm_robotics.isaac.analysis import common_isaac as C


def fmt(sub, c):
    return C.fmt_ci(*C.ci95(sub[c].to_numpy()), 3)


def first_cols(df):
    for k, name in (("u", "goal"), ("goalobj", "goal_object"), ("v", "tower"), ("phantom", "phantom"), ("none", "neither")):
        df[name] = (df["first"] == k).astype(float)
    df["fall"] = (df.outcome == "fall").astype(float)
    return df


def steering():
    d = os.path.join(C.RESULTS, "steer")
    fs = [f for f in glob.glob(os.path.join(d, "*.csv")) if "per_seed" not in f and "summary" not in f]
    if not fs:
        return
    df = first_cols(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True))
    vals = ["goal", "goal_object", "tower", "phantom", "neither", "fall"]
    # random arms: mean over draws within seed first
    ps = df.groupby(["plan", "bank", "cohort", "seed", "arm", "beta"])[vals + ["drift", "dnorm0", "dflag0"]].mean().reset_index()
    ps.to_csv(os.path.join(d, "steer_per_seed.csv"), index=False)
    with open(os.path.join(d, "steer_summary.md"), "w") as fh:
        fh.write("# Isaac steering (closed loop, first object reached; mean [95% t-CI] across 3 seeds; random arms averaged "
                 "over draws within a seed). plan phantom: acting copy has the flag removed, steering copies unchanged; "
                 "goal there = flag spot reached geometrically. phantom = 90 deg rotated flag spot (chance control).\n\n")
        fh.write(f"Max per-step copy pose gap after resync: {df.drift.max():.2e} m\n\n")
        fh.write("decoupled: acting copy has the flag moved beside the tower (flag-swap scene), steering copies unchanged; goal = "
                 "empty original flag spot u reached first, goal_object = the moved flag reached first.\n\n")
        fh.write("| plan | bank | cohort | arm | beta | goal (spot u) | goal object | tower | phantom | neither |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for key, sub in ps.groupby(["plan", "bank", "cohort", "arm", "beta"], sort=True):
            fh.write("| " + " | ".join(str(k) for k in key) + " | " + " | ".join(fmt(sub, c) for c in vals[:5]) + " |\n")
    # compact standard / phantom / decoupled comparison for the GM agents
    gm = ps[ps.cohort == "gm"]
    rows = []
    for (bank, arm, beta), sub in gm.groupby(["bank", "arm", "beta"]):
        r = dict(bank=bank, arm=arm, beta=beta)
        for plan, tag in (("full", "std"), ("phantom", "phantom"), ("decoupled", "dec")):
            q = sub[sub.plan == plan]
            if len(q):
                r[f"{tag}_u"] = fmt(q, "goal")
                if plan == "decoupled":
                    r["dec_goal_object"] = fmt(q, "goal_object")
                    r["dec_tower"] = fmt(q, "tower")
        rows.append(r)
    cmp_ = pd.DataFrame(rows)
    cmp_ = cmp_[cmp_.filter(like="phantom").notna().any(axis=1) | cmp_.filter(like="dec").notna().any(axis=1)] if len(cmp_.columns) > 4 else cmp_
    cmp_.to_csv(os.path.join(d, "steer_chance_compare.csv"), index=False)
    with open(os.path.join(d, "steer_chance_compare.md"), "w") as fh:
        fh.write("# GM steering: flag spot u reached first, standard vs phantom (flag removed) vs decoupled (flag moved beside the "
                 "tower), and the decoupled goal-object reach (mean [95% t-CI] over 3 seeds)\n\n")
        cols = [c for c in ("std_u", "phantom_u", "dec_u", "dec_goal_object", "dec_tower") if c in cmp_.columns]
        fh.write("| bank | arm | beta | " + " | ".join(cols) + " |\n|---|---|---|" + "---|" * len(cols) + "\n")
        for _, r in cmp_.iterrows():
            fh.write(f"| {r.bank} | {r.arm} | {r.beta} | " + " | ".join(str(r.get(c, "")) if pd.notna(r.get(c)) else "" for c in cols) + " |\n")
    print(open(os.path.join(d, "steer_summary.md")).read()[:6000])


def rewiring():
    d = os.path.join(C.RESULTS, "rewire")
    f = os.path.join(d, "closed_loop.csv")
    if not os.path.exists(f):
        return
    df = first_cols(pd.read_csv(f))
    vals = ["goal", "tower", "phantom", "neither", "fall"]
    ps = df.groupby(["set", "cohort", "seed", "arm", "target"])[vals + ["val_loss", "n_trainable"]].mean().reset_index()
    ps.to_csv(os.path.join(d, "rewire_per_seed.csv"), index=False)
    with open(os.path.join(d, "rewire_summary.md"), "w") as fh:
        fh.write("# Isaac rewiring (BC on control-agent rollouts, control layouts; closed loop first object reached; mean "
                 "[95% t-CI] across 3 seeds). noflag sets: flag removed, goal = flag spot reached geometrically (chance).\n\n")
        fh.write("| set | cohort | arm | target | params | val loss | goal | tower | phantom | neither |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for key, sub in ps.groupby(["set", "cohort", "arm", "target"], sort=True):
            fh.write("| " + " | ".join(str(k) for k in key) + f" | {int(sub.n_trainable.mean())} | {sub.val_loss.mean():.3f} | "
                     + " | ".join(fmt(sub, c) for c in vals[:4]) + " |\n")
    print(open(os.path.join(d, "rewire_summary.md")).read())


if __name__ == "__main__":
    steering()
    rewiring()
