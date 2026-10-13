"""Layerwise linear probes for goal identity / location with random-init, raw-pixel and proprio baselines."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C

SITES = ("conv1", "conv2", "conv3", "h_pre", "h_fc", "a_hid")
ALPHAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def targets(bank, env):
    """{name: (y, kind, mask)}; kind 'auc' for binary, 'r2' for continuous."""
    n = len(bank["state"])
    ones = np.ones(n, bool)
    if env == "AT":
        u, v, cr, sr = bank["u"], bank["v"], bank["cube_red"].astype(bool), bank["sphere_red"].astype(bool)
        one_red = cr ^ sr
        red_xy = np.where(cr[:, None], u, v)
        return {"cube_x": (u[:, 0], "r2", ones), "cube_y": (u[:, 1], "r2", ones),
                "cube_left": ((u[:, 1] > v[:, 1]).astype(float), "auc", ones),
                "cube_is_red": (cr.astype(float), "auc", ones),
                "red_x": (red_xy[:, 0], "r2", one_red), "red_y": (red_xy[:, 1], "r2", one_red),
                "red_left": ((red_xy[:, 1] > np.where(cr[:, None], v, u)[:, 1]).astype(float), "auc", one_red)}
    if env == "SZ":
        u, v, cg, cube = bank["u"], bank["v"], bank["cross_green"].astype(bool), bank["cube"]
        green = np.where(cg[:, None], u, v)
        return {"cross_x": (u[:, 0], "r2", ones), "cross_y": (u[:, 1], "r2", ones),
                "cross_left": ((u[:, 1] > 0).astype(float), "auc", ones),
                "cross_is_green": (cg.astype(float), "auc", ones),
                "green_x": (green[:, 0], "r2", ones), "green_y": (green[:, 1], "r2", ones),
                "green_left": ((green[:, 1] > 0).astype(float), "auc", ones),
                "cube_x": (cube[:, 0], "r2", ones), "cube_y": (cube[:, 1], "r2", ones)}
    if env == "PTM":
        from gm_robotics.analysis.envs_forced import PTM_C
        m, cube, tcp = bank["marker"], bank["cube"][:, :2], bank["tcp_env"]
        return {"marker_x": (m[:, 0], "r2", ones), "marker_y": (m[:, 1], "r2", ones),
                "marker_r": (np.linalg.norm(m - PTM_C, axis=1), "r2", ones),
                "cube_x": (cube[:, 0], "r2", ones), "cube_y": (cube[:, 1], "r2", ones),
                "push_dx": (m[:, 0] - cube[:, 0], "r2", ones), "push_dy": (m[:, 1] - cube[:, 1], "r2", ones),
                "marker_bearing_x": (m[:, 0] - tcp[:, 0], "r2", ones), "marker_bearing_y": (m[:, 1] - tcp[:, 1], "r2", ones)}
    g, d, tcp = bank["goal"], bank["dxy"], bank["tcp_env"]
    from gm_robotics.analysis.envs_forced import CORNER
    return {"marker_x": (g[:, 0], "r2", ones), "marker_y": (g[:, 1], "r2", ones),
            "bearing_x": (g[:, 0] - tcp[:, 0], "r2", ones), "bearing_y": (g[:, 1] - tcp[:, 1], "r2", ones),
            "marker_near_corner": ((np.linalg.norm(g[:, :2] - CORNER, axis=1) < 0.10).astype(float), "auc", ones),
            "distractor_x": (d[:, 0], "r2", ones), "distractor_y": (d[:, 1], "r2", ones)}


@torch.no_grad()
def features(agent, bank, dev, bs=256):
    out = {s: [] for s in SITES}
    for sl in C.batched(len(bank["state"]), bs):
        t = C.taps(agent, C.to_t(bank["rgb"][sl], dev), C.to_t(bank["state"][sl], dev))
        for s in SITES:
            x = t[s]
            out[s].append((x.mean((2, 3)) if x.ndim == 4 else x).cpu().numpy().astype(np.float32))
    return {s: np.concatenate(v) for s, v in out.items()}


def pixel_blocks(rgb, k=16):
    """Raw-pixel input control as in the paper: k x k x 3 block means."""
    n, h, w, c = rgb.shape
    return rgb.reshape(n, k, h // k, k, w // k, c).mean((2, 4)).reshape(n, -1).astype(np.float32)


class RidgeX:
    """Ridge on standardised features via one SVD of the design (stable when dim > n); the SVD does not depend on y,
    so one fit serves every target and alpha. Same minimiser as gm_geometry's fit_ridge on standardised inputs."""

    def __init__(self, x):
        self.mu, sd = x.mean(0), x.std(0)
        self.keep = sd > 1e-8
        self.sd = sd[self.keep]
        xc = (x[:, self.keep] - self.mu[self.keep]) / self.sd
        self.U, self.s, self.Vt = np.linalg.svd(xc.astype(np.float64), full_matrices=False)

    def predict(self, y, alpha, xnew):
        ymu = y.mean()
        w = self.Vt.T @ (self.s / (self.s ** 2 + alpha) * (self.U.T @ (y - ymu)))
        return ((xnew[:, self.keep] - self.mu[self.keep]) / self.sd) @ w + ymu


def score(y, pred, kind):
    return C.gmg_linear.roc_auc(y, pred) if kind == "auc" else C.gmg_linear.r2_score(y, pred)


def run_feature_set(name_cols, ftr, fte, ttr, tte, folds=5, seed=0):
    """Alpha chosen by k-fold CV on the train bank only; the held-out bank is scored once per target."""
    rows = []
    masks = {}
    for tname, (_, _, mtr) in ttr.items():
        masks.setdefault(mtr.tobytes(), mtr)
    for site, xall in ftr.items():
        for mkey, mtr in masks.items():
            x = xall[mtr]
            fold = np.random.default_rng(seed).integers(folds, size=len(x))
            fold_models = [RidgeX(x[fold != f]) for f in range(folds)]
            full = RidgeX(x)
            for tname, (yall, kind, m2) in ttr.items():
                if m2.tobytes() != mkey:
                    continue
                y = yall[mtr]
                cv = [np.mean([score(y[fold == f], fm.predict(y[fold != f], al, x[fold == f]), kind)
                               for f, fm in enumerate(fold_models)]) for al in ALPHAS]
                best = ALPHAS[int(np.nanargmax(cv))]
                yte, _, mte = tte[tname]
                val = score(yte[mte], full.predict(y, best, fte[site][mte]), kind)
                rows.append(dict(**name_cols, site=site, target=tname, metric=kind, value=val, alpha=best))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "LR", "SZ", "PTM"], required=True)
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    C.ensure_dirs()
    dev = C.device()
    tr, te = C.load_bank(f"{a.env}_probe_train"), C.load_bank(f"{a.env}_probe_test")
    ttr, tte = targets(tr, a.env), targets(te, a.env)
    raw_dir = os.path.join(C.RESULTS, "probe_raw")
    os.makedirs(raw_dir, exist_ok=True)
    out_ctrl = os.path.join(raw_dir, f"{a.env}_input_controls.csv")
    if not os.path.exists(out_ctrl):
        rows = run_feature_set(dict(cohort="pixels", seed=0), {"input": pixel_blocks(tr["rgb"])},
                               {"input": pixel_blocks(te["rgb"])}, ttr, tte)
        rows += run_feature_set(dict(cohort="proprio", seed=0), {"input": tr["state"]}, {"input": te["state"]}, ttr, tte)
        pd.DataFrame(rows).to_csv(out_ctrl, index=False)
    dfs = [pd.read_csv(out_ctrl)]
    for cohort, seed, ck in C.agent_list(a.env, a.seeds_gm, a.seeds_control, a.init_seeds):
        out = os.path.join(raw_dir, f"{a.env}_{cohort}_s{seed}.csv")
        if not os.path.exists(out):
            agent = C.build_agent(ck, init_seed=seed, dev=dev)
            rows = run_feature_set(dict(cohort=cohort, seed=seed), features(agent, tr, dev), features(agent, te, dev),
                                   ttr, tte)
            pd.DataFrame(rows).to_csv(out, index=False)
        df = pd.read_csv(out)
        print(f"{a.env} {cohort} s{seed}: " + ", ".join(
            f"{t}@h_fc {g.loc[g.site == 'h_fc', 'value'].iloc[0]:.3f}" for t, g in df.groupby("target")), flush=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(C.RESULTS, f"probes_{a.env}_per_seed.csv"), index=False)
    rows = []
    for (cohort, site, tname, metric), g in df.groupby(["cohort", "site", "target", "metric"]):
        m, lo, hi = C.ci95(g["value"].to_numpy())
        rows.append(dict(cohort=cohort, site=site, target=tname, metric=metric, n=len(g), mean=m, lo=lo, hi=hi))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(C.RESULTS, f"probes_{a.env}_summary.csv"), index=False)
    piv = summ.assign(v=[C.fmt_ci(r.mean, r.lo, r.hi) for r in summ.itertuples()]).pivot_table(
        index=["target", "site"], columns="cohort", values="v", aggfunc="first")
    with open(os.path.join(C.RESULTS, f"probes_{a.env}_summary.md"), "w") as f:
        f.write(f"# Linear probes, {a.env} (held-out bank; AUC or R2; mean and 95% t-CI across seeds)\n\n")
        f.write("```\n" + piv.to_string() + "\n```\n")
    print(piv.to_string())


if __name__ == "__main__":
    main()
