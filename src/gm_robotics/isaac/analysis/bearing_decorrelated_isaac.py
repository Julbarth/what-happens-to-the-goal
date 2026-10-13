"""WalkToFlag flag/tower bearing probes restricted to layouts where the two bearings are uncorrelated (ghost-free banks)."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
# Variant codes follow isaac/gm_nav_env.py VARIANTS = ("gm", "control", "conflict").
SUBSETS = {"all": (0, 1, 2), "nongm_conflict": (1, 2), "conflict": (2,)}


def targets(b):
    v, fv, tv = b["variant"], b["flag_visible"].astype(bool), b["tower_visible"].astype(bool)
    out = {}
    for name, keep in SUBSETS.items():
        m = np.isin(v, keep)
        out[f"flag_bearing|{name}"] = (b["flag_bearing"], "r2", fv & m)
        out[f"tower_bearing|{name}"] = (b["tower_bearing"], "r2", tv & m)
    return out


def bearing_corr(b):
    both = (b["flag_visible"] > 0.5) & (b["tower_visible"] > 0.5)
    rows = []
    for name, keep in SUBSETS.items():
        m = both & np.isin(b["variant"], keep)
        rows.append(dict(subset=name, n_both_visible=int(m.sum()),
                         r=float(np.corrcoef(b["flag_bearing"][m], b["tower_bearing"][m])[0, 1])))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds_gm", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seeds_control", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    from gm_robotics.isaac.analysis import common_isaac as C
    from gm_robotics.isaac.analysis import probes_isaac as PI
    MP = PI.MP
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cpu")
    tr, te = C.load_bank("probe_train"), C.load_bank("probe_test")
    corr = pd.DataFrame([dict(split="train", **r) for r in bearing_corr(tr)] + [dict(split="test", **r) for r in bearing_corr(te)])
    corr.to_csv(os.path.join(a.out_dir, "bearing_corr.csv"), index=False)
    print(corr.to_string(index=False), flush=True)
    ttr, tte = targets(tr), targets(te)
    counts = pd.DataFrame([dict(target=k, n_train=int(ttr[k][2].sum()), n_test=int(tte[k][2].sum())) for k in ttr])
    counts.to_csv(os.path.join(a.out_dir, "state_counts.csv"), index=False)
    print(counts.to_string(index=False), flush=True)
    dfs = []
    out = os.path.join(a.out_dir, "pixels_s0.csv")
    if not os.path.exists(out):
        pd.DataFrame(MP.run_feature_set(dict(cohort="pixels", seed=0), {"input": MP.pixel_blocks(tr["rgb"])},
                                        {"input": MP.pixel_blocks(te["rgb"])}, ttr, tte)).to_csv(out, index=False)
    dfs.append(pd.read_csv(out))
    for cohort, seed, ck in C.agent_list(a.seeds_gm, a.seeds_control, a.init_seeds):
        out = os.path.join(a.out_dir, f"{cohort}_s{seed}.csv")
        if not os.path.exists(out):
            agent = C.build_agent(ck, init_seed=seed, dev=dev)
            rows = MP.run_feature_set(dict(cohort=cohort, seed=seed), PI.features(agent, tr, dev), PI.features(agent, te, dev),
                                      ttr, tte)
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{cohort} s{seed} done ({ck})", flush=True)
        dfs.append(pd.read_csv(out))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(a.out_dir, "per_seed.csv"), index=False)
    rows = []
    for (cohort, site, tname), g in df.groupby(["cohort", "site", "target"]):
        v = g.sort_values("seed")["value"].to_numpy()
        m, lo, hi = C.ci95(v)
        rows.append(dict(cohort=cohort, site=site, target=tname, n=len(v), mean=m, sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                         lo=lo, hi=hi, seeds=" ".join(f"{x:.4f}" for x in v)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    print(summ.pivot_table(index=["target", "site"], columns="cohort", values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
