"""Isaac quadruped layerwise linear probes (flag/tower bearing, distance, visibility) with random-init and raw-pixel baselines."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gm_robotics.isaac.analysis import common_isaac as C
from gm_robotics.analysis import probes as MP   # analysis/probes.py: RidgeX, run_feature_set (alpha by CV on train only), pixel_blocks

SITES = ("conv1", "conv2", "conv3", "h_pre", "h_fc", "a_hid")
POOL = 4   # conv maps pooled to 4x4 rather than 1x1: a global mean would erase bearing by construction


class RidgeGlobal(MP.RidgeX):
    """RidgeX with one global scale instead of per-unit standardisation: rarely firing ReLU units have near-zero train
    sd, and dividing by it makes single held-out activations explode."""

    def __init__(self, x):
        self.mu = x.mean(0)
        self.keep = np.ones(x.shape[1], bool)
        xc = x - self.mu
        self.sd = np.full(x.shape[1], np.sqrt((xc ** 2).mean()) + 1e-12)
        self.U, self.s, self.Vt = np.linalg.svd((xc / self.sd).astype(np.float64), full_matrices=False)


MP.RidgeX = RidgeGlobal


def targets(b):
    """{name: (y, kind, mask)}; bearing/distance probes only on states where the object renders (>= 5 px)."""
    n = len(b["start"])
    ones = np.ones(n, bool)
    fv, tv = b["flag_visible"].astype(bool), b["tower_visible"].astype(bool)
    return {"flag_visible": (b["flag_visible"], "auc", ones),
            "flag_bearing": (b["flag_bearing"], "r2", fv), "flag_dist": (b["flag_dist"], "r2", fv),
            "tower_visible": (b["tower_visible"], "auc", ones),
            "tower_bearing": (b["tower_bearing"], "r2", tv), "tower_dist": (b["tower_dist"], "r2", tv)}


@torch.no_grad()
def features(agent, bank, dev, bs=256):
    out = {s: [] for s in SITES}
    for sl in C.batched(len(bank["state"]), bs):
        t = C.taps(agent, C.to_t(bank["rgb"][sl], dev), C.to_t(bank["state"][sl], dev))
        for s in SITES:
            x = t[s]
            x = F.adaptive_avg_pool2d(x.cpu(), POOL).flatten(1) if x.ndim == 4 else x.cpu()
            out[s].append(x.cpu().numpy().astype(np.float32))
    return {s: np.concatenate(v) for s, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    C.ensure_dirs()
    dev = C.device()
    tr, te = C.load_bank("probe_train"), C.load_bank("probe_test")
    ttr, tte = targets(tr), targets(te)
    raw_dir = os.path.join(C.RESULTS, "probe_raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_ctrl = os.path.join(raw_dir, "input_controls.csv")
    if not os.path.exists(out_ctrl):
        rows = MP.run_feature_set(dict(cohort="pixels", seed=0), {"input": MP.pixel_blocks(tr["rgb"])},
                                  {"input": MP.pixel_blocks(te["rgb"])}, ttr, tte)
        pd.DataFrame(rows).to_csv(out_ctrl, index=False)
    dfs = [pd.read_csv(out_ctrl)]
    for cohort, seed, ck in C.agent_list(a.seeds_gm, a.seeds_control, a.init_seeds):
        out = os.path.join(raw_dir, f"{cohort}_s{seed}.csv")
        if not os.path.exists(out):
            agent = C.build_agent(ck, init_seed=seed, dev=dev)
            rows = MP.run_feature_set(dict(cohort=cohort, seed=seed), features(agent, tr, dev), features(agent, te, dev),
                                      ttr, tte)
            pd.DataFrame(rows).to_csv(out, index=False)
        df = pd.read_csv(out)
        print(f"{cohort} s{seed}: " + ", ".join(
            f"{t}@h_fc {g.loc[g.site == 'h_fc', 'value'].iloc[0]:.3f}" for t, g in df.groupby("target")), flush=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(C.RESULTS, "probes_per_seed.csv"), index=False)
    rows = []
    for (cohort, site, tname, metric), g in df.groupby(["cohort", "site", "target", "metric"]):
        m, lo, hi = C.ci95(g["value"].to_numpy())
        rows.append(dict(cohort=cohort, site=site, target=tname, metric=metric, n=len(g), mean=m, lo=lo, hi=hi))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(C.RESULTS, "probes_summary.csv"), index=False)
    piv = summ.assign(v=[C.fmt_ci(r.mean, r.lo, r.hi) for r in summ.itertuples()]).pivot_table(
        index=["target", "site"], columns="cohort", values="v", aggfunc="first")
    n_tr, n_te = {k: int(v[2].sum()) for k, v in ttr.items()}, {k: int(v[2].sum()) for k, v in tte.items()}
    with open(os.path.join(C.RESULTS, "probes_summary.md"), "w") as f:
        f.write("# Isaac quadruped linear probes (held-out bank; AUC or R2; mean and 95% t-CI across seeds)\n\n")
        f.write(f"States (train / test): {n_tr} / {n_te}\n\n```\n" + piv.to_string() + "\n```\n")
    print(piv.to_string())


if __name__ == "__main__":
    main()
