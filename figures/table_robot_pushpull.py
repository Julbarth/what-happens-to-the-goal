"""Robot push-pull per cohort (mean, SD, 95% t-CI over seeds) from the robotics per-seed CSVs, AttributeTable Non-GM seeds 3-5."""
from paths import ROBOTICS_ROOT
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

COLS = ["S_fav", "S_comp", "dl", "pushpull", "null_pushpull"]


def load(gr):
    at = pd.read_csv(os.path.join(gr, "results/analysis/push_pull/AT_per_seed.csv"))
    # AT Non-GM seed 2 is dropped everywhere in the paper (default_rng(0).choice([2,3,4,5], 3) = [4, 5, 3]).
    at = at[~((at.cohort == "control") & (at.seed == 2))]
    sz = pd.read_csv(os.path.join(gr, "results/analysis/push_pull/SZ_per_seed.csv"))
    w = pd.read_csv(os.path.join(gr, "results/isaac_analysis/ghost_check/fixed/push_pull/isaac_per_seed.csv"))
    w = w[w.subset == "all"]
    forms = [w.assign(env="WTF", form="oracle")]
    for k in (15, 30, 45):
        f = w.drop(columns=["S_fav", "S_comp", "dl", "pushpull"]).rename(
            columns={f"{c}_cone{k}": c for c in ("S_fav", "S_comp", "dl", "pushpull")})
        forms.append(f.assign(env="WTF", form=f"cone{k}"))
    w = pd.concat(forms, ignore_index=True)
    at, sz = at.assign(form="probit"), sz.assign(form="probit")
    keep = ["env", "form", "bank", "edit", "cohort", "seed"] + COLS
    return pd.concat([at[keep], sz[keep], w[keep]], ignore_index=True)


def summarise(df):
    rows = []
    for key, s in df.groupby(["env", "form", "bank", "edit", "cohort"], sort=False):
        n = len(s)
        r = dict(zip(["env", "form", "bank", "edit", "cohort"], key), n=n, seeds=" ".join(map(str, s.seed)),
                 seeds_in_quadrant=int(((s.S_fav > 0) & (s.S_comp < 0)).sum()))
        for c in COLS:
            m, sd = s[c].mean(), s[c].std(ddof=1)
            h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            r.update({c: m, c + "_sd": sd, c + "_lo": m - h, c + "_hi": m + h})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gr", default=str(ROBOTICS_ROOT))
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_pushpull"))
    a = p.parse_args()
    df = load(a.gr)
    df.to_csv(a.out + "_per_seed.csv", index=False)
    s = summarise(df)
    s.to_csv(a.out + "_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(s[["env", "form", "bank", "edit", "cohort", "n", "seeds", "S_fav", "S_fav_sd", "S_comp", "S_comp_sd", "dl", "dl_sd",
                 "pushpull", "null_pushpull", "seeds_in_quadrant"]].round(3).to_string())


if __name__ == "__main__":
    main()
