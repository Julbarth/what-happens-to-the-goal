"""95% intervals for the Attribute Maze GM outcome AUC (signed midpoint distance xF_mid), from the saved per-state CSVs."""
import argparse, glob, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n-boot", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.data}/per_seed/per_state_gm_*_black_h_fc.csv"))])
df = df[(df.axis == "unit_mean") & df.native_rollout_outcome.isin(["goal", "proxy"])].copy()
df["y"] = (df.native_rollout_outcome == "goal").astype(int)
rng = np.random.default_rng(a.seed)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def auc(g):
    return roc_auc_score(g.y, g.xF_mid) if g.y.nunique() == 2 else np.nan


def auc_ws(g):
    return np.nanmean([auc(h) for _, h in g.groupby("sign")])


res = {}
for split in ["all196", "eval50"]:
    d = df if split == "all196" else df[df.split_role == "eval"]
    seeds = {s: g.reset_index(drop=True) for s, g in d.groupby("seed")}
    for name, f in [("pooled", auc), ("within_orientation", auc_ws)]:
        point = np.array([f(g) for g in seeds.values()])
        boots = []
        for _ in range(a.n_boot):
            # Resample states within each seed, stratified by sign so both orientations stay present.
            vals = []
            for g in seeds.values():
                idx = np.concatenate([rng.choice(ix, len(ix)) for ix in g.groupby("sign").indices.values()])
                vals.append(f(g.iloc[idx]))
            boots.append(np.nanmean(vals))
        boots = np.array(boots)
        m, sd = point.mean(), point.std(ddof=1)
        h = stats.t.ppf(0.975, 2) * sd / np.sqrt(3)
        res[f"{split}|{name}"] = {"mean": m, "per_seed": point.tolist(),
                                  "boot_states_within_seeds": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                                  "t_across_seeds": [m - h, m + h],
                                  "n_goal_per_seed": [int(g.y.sum()) for g in seeds.values()],
                                  "n_states_per_seed": [len(g) for g in seeds.values()]}
        print(split, name, json.dumps(res[f"{split}|{name}"], default=float))
json.dump({"source": "data/probes/am_axis_tug_of_war_20260925/per_seed/per_state_gm_*_black_h_fc.csv", "score": "xF_mid",
           "n_boot": a.n_boot, "rng_seed": a.seed, "results": res}, open(a.out, "w"), indent=1, default=float)
