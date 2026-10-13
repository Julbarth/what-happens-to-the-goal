"""Seed-level summaries (95% t-CI) of the h_fc steering runs (AT, SZ, LR, PTM) and the LR rewiring."""
import argparse
import os

import pandas as pd
from gm_robotics.analysis import common as C
from gm_robotics.analysis.recovery_stats import outcome_stats, write


def steer_summary(env, title):
    f = os.path.join(C.RESULTS, f"steer_{env}_per_state.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    # outcome_stats pools the random-arm repeats within a seed before the seed-level CI.
    per, summ = outcome_stats(df, ["arm", "beta"])
    per.to_csv(os.path.join(C.RESULTS, f"steer_{env}_per_seed.csv"), index=False)
    summ = summ.sort_values(["cohort", "arm", "beta"])
    summ.to_csv(os.path.join(C.RESULTS, f"steer_{env}_summary.csv"), index=False)
    write(summ, ["arm", "beta"], os.path.join(C.RESULTS, f"steer_{env}_summary.md"), title)
    print(open(os.path.join(C.RESULTS, f"steer_{env}_summary.md")).read())


def rewire_summary():
    f = os.path.join(C.RESULTS, "rewire_LR", "closed_loop_per_episode.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    per, summ = outcome_stats(df, ["set", "arm"])
    per.to_csv(os.path.join(C.RESULTS, "rewire_LR_per_seed.csv"), index=False)
    summ = summ.sort_values(["set", "arm", "cohort"])
    summ.to_csv(os.path.join(C.RESULTS, "rewire_LR_summary.csv"), index=False)
    write(summ, ["set", "arm"], os.path.join(C.RESULTS, "rewire_LR_summary.md"),
          "LocationReach rewiring, closed loop (first-contact scoring; proxy = training corner)")
    print(open(os.path.join(C.RESULTS, "rewire_LR_summary.md")).read())


STEER_TITLES = {
    "AT": "AttributeTable steering at h_fc (200 eval states, deterministic mean actions)",
    "SZ": "SortZones steering at h_fc (200 eval states, deterministic mean actions)",
    "LR": "LocationReach state-local steering at h_fc (200 off-path states, deterministic mean actions)",
    "PTM": "PushToMarker state-local steering at h_fc (200 conflict states, deterministic mean actions; reach = cube on "
           "the marker, proxy = cube at the table centre, final state at T = 100)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["steer", "rewire"], choices=["steer", "rewire"])
    p.add_argument("--envs", nargs="+", default=list(STEER_TITLES), choices=list(STEER_TITLES))
    a = p.parse_args()
    if "steer" in a.which:
        for e in a.envs:
            steer_summary(e, STEER_TITLES[e])
    if "rewire" in a.which:
        rewire_summary()


if __name__ == "__main__":
    main()
