"""Seed-level summaries (95% t-CI) of steering and rewiring outcomes, including the reach = D * P(goal|decisive) split."""
import os

import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C


def outcome_stats(df, keys):
    g = df.groupby(keys + ["cohort", "seed"]).outcome
    per = pd.DataFrame({"reach": g.apply(lambda o: (o == "goal").mean()),
                        "proxy": g.apply(lambda o: (o == "proxy").mean()),
                        "D": g.apply(lambda o: o.isin(["goal", "proxy"]).mean())}).reset_index()
    per["P_goal_dec"] = per.reach / per.D.replace(0, np.nan)
    rows = []
    for k, sub in per.groupby(keys + ["cohort"]):
        r = dict(zip(keys + ["cohort"], k if isinstance(k, tuple) else (k,)), n_seeds=len(sub))
        for c in ("reach", "proxy", "D", "P_goal_dec"):
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(sub[c].to_numpy())
        rows.append(r)
    return per, pd.DataFrame(rows)


def write(summ, keys, path, title):
    cols = ["reach", "proxy", "D", "P_goal_dec"]
    with open(path, "w") as f:
        f.write(f"# {title}\n\nMean and 95% t-CI across seeds.\n\n| " + " | ".join(keys + ["cohort", "n"] + cols) +
                " |\n|" + "---|" * (len(keys) + 2 + len(cols)) + "\n")
        for _, r in summ.iterrows():
            f.write("| " + " | ".join(str(r[k]) for k in keys + ["cohort"]) + f" | {r['n_seeds']} | " +
                    " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi']) for c in cols) + " |\n")


def main():
    f = os.path.join(C.RESULTS, "steer_AT_per_state.csv")
    if os.path.exists(f):
        df = pd.read_csv(f)
        # Random arm: average the repeats within a seed before the seed-level CI.
        per, summ = outcome_stats(df, ["arm", "beta"])
        per.to_csv(os.path.join(C.RESULTS, "steer_AT_per_seed.csv"), index=False)
        summ.sort_values(["cohort", "arm", "beta"]).to_csv(os.path.join(C.RESULTS, "steer_AT_summary.csv"), index=False)
        write(summ.sort_values(["cohort", "arm", "beta"]), ["arm", "beta"], os.path.join(C.RESULTS, "steer_AT_summary.md"),
              "AttributeTable steering at h_fc (200 eval states, deterministic mean actions)")
    f = os.path.join(C.RESULTS, "rewire_LR", "closed_loop_per_episode.csv")
    if os.path.exists(f):
        df = pd.read_csv(f)
        per, summ = outcome_stats(df, ["set", "arm"])
        per.to_csv(os.path.join(C.RESULTS, "rewire_LR_per_seed.csv"), index=False)
        summ.sort_values(["set", "arm", "cohort"]).to_csv(os.path.join(C.RESULTS, "rewire_LR_summary.csv"), index=False)
        write(summ.sort_values(["set", "arm", "cohort"]), ["set", "arm"], os.path.join(C.RESULTS, "rewire_LR_summary.md"),
              "LocationReach rewiring, closed loop (first-contact scoring; proxy = training point)")
    for p in ("steer_AT_summary.md", "rewire_LR_summary.md"):
        if os.path.exists(os.path.join(C.RESULTS, p)):
            print(open(os.path.join(C.RESULTS, p)).read())


if __name__ == "__main__":
    main()
