"""Seeking vs perceiving: paired paths with and without the tower, and each object's along-trajectory pull on the command."""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from gm_robotics.isaac.analysis import common_isaac as C
from gm_robotics.isaac.analysis import geom
from gm_robotics.isaac.analysis.pushpull_isaac import oracle_form

TRACE = os.path.join(C.RESULTS, "trace")
OUT = os.path.join(C.RESULTS, "trace_summary")
NEAR = 1.0   # "near the tower" radius (m), > PROXY_R so passing close without touching counts


def canon(xy, s, u, v):
    """Path in a frame with the start at 0, the flag on +x and the tower on the +y side (mirrored when needed)."""
    d = u - s
    ang = np.arctan2(d[1], d[0])
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    q = (xy - s) @ R.T
    vv = (v - s) @ R.T
    if vv[1] < 0:
        q[..., 1] *= -1
    return q


def path_stats(xy, s, u, v):
    ok = ~np.isnan(xy[:, 0])
    q = xy[ok]
    c = canon(q, s, u, v)
    dv = np.linalg.norm(q - v, axis=1)
    du = np.linalg.norm(q - u, axis=1)
    L = np.linalg.norm(u - s)
    inside = (c[:, 0] > 0) & (c[:, 0] < L)
    return dict(min_d_tower=dv.min(), steps_near_tower=int((dv < NEAR).sum()), dev_max=c[:, 1].max(),
                dev_mean=c[inside, 1].mean() if inside.any() else 0.0, min_d_flag=du.min(), n_steps=len(q),
                path_len=np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def follow_rows(z, run, bank, mu_steps):
    """Along the base trajectory: tower pull = oracle-form S at the command toward the tower, base copy vs no-tower copy
    (same pose); flag pull likewise from the no-flag copy. Positive = the object's presence adds support for heading to it."""
    cop = list(z["copies"])
    sig = z["sigma"].astype(np.float64)
    mu = z["base_mu"].astype(np.float64)
    xy, yw = z["base_xy"], z["base_yaw"]
    u, v = z["lay_base_flag"], z["lay_base_tower"]
    rows = []
    for t in range(mu_steps):
        alive = ~np.isnan(mu[:, 0, t, 0]) & ~np.isnan(xy[:, t, 0])
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        ru = geom.robot_frame(u[idx], xy[idx, t], yw[idx, t])
        rv = geom.robot_frame(v[idx], xy[idx, t], yw[idx, t])
        bu, bv = np.arctan2(ru[:, 1], ru[:, 0]), np.arctan2(rv[:, 1], rv[:, 0])
        m_b = mu[idx, cop.index("base"), t]
        tw = oracle_form(mu[idx, cop.index("no_tower"), t], m_b, sig, bv, bu)
        fl = oracle_form(mu[idx, cop.index("no_flag"), t], m_b, sig, bu, bv)
        dist_v, dist_u = np.linalg.norm(ru, axis=1), np.linalg.norm(rv, axis=1)
        rows.append(pd.DataFrame(dict(run=run, bank=bank, t=t, layout=idx, tower_pull=tw["S_fav"], tower_vs_flag=tw["S_fav"] - tw["S_comp"],
                                      flag_pull=fl["S_fav"], flag_vs_tower=fl["S_fav"] - fl["S_comp"],
                                      tower_in_fov=np.abs(bv) < np.deg2rad(50), flag_in_fov=np.abs(bu) < np.deg2rad(50),
                                      d_tower=np.linalg.norm(rv, axis=1), d_flag=np.linalg.norm(ru, axis=1))))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mu_steps", type=int, default=40)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    prs, frs = [], []
    for f in sorted(glob.glob(os.path.join(TRACE, "*_main_*.npz"))):
        bank, run = os.path.basename(f)[:-4].split("_main_")
        run = "main_" + run
        z = np.load(f, allow_pickle=True)
        s, u, v = z["start"], z["lay_base_flag"], z["lay_base_tower"]
        for i in range(len(s)):
            r = dict(run=run, bank=bank, layout=i)
            for acting in ("base", "no_tower"):
                st = path_stats(z[f"{acting}_xy"][i], s[i], u[i], v[i])
                r.update({f"{acting}_{k}": val for k, val in st.items()})
                r[f"{acting}_outcome"] = str(z[f"{acting}_outcome"][i])
            prs.append(r)
        frs += follow_rows(z, run, bank, a.mu_steps)
        print(f, "drift", float(z["base_drift"].max()), float(z["no_tower_drift"].max()), flush=True)
    pdf = pd.DataFrame(prs)
    pdf["cohort"] = pdf.run.str.split("_").str[1]
    pdf["seed"] = pdf.run.str[-1].astype(int)
    for k in ("min_d_tower", "steps_near_tower", "dev_max", "dev_mean", "path_len", "n_steps"):
        pdf[f"diff_{k}"] = pdf[f"base_{k}"] - pdf[f"no_tower_{k}"]
    pdf["base_goal_first"] = (pdf.base_outcome == "goal").astype(float)
    pdf["base_tower_touch"] = (pdf.base_min_d_tower < 0.6).astype(float)
    pdf["notower_spot_touch"] = (pdf.no_tower_min_d_tower < 0.6).astype(float)
    pdf.to_csv(os.path.join(OUT, "paths_per_layout.csv"), index=False)
    fdf = pd.concat(frs, ignore_index=True)
    fdf["cohort"] = fdf.run.str.split("_").str[1]
    fdf["seed"] = fdf.run.str[-1].astype(int)
    fdf.to_csv(os.path.join(OUT, "follow_per_step.csv.gz"), index=False)
    cols = ["diff_min_d_tower", "diff_steps_near_tower", "diff_dev_max", "diff_dev_mean", "diff_path_len", "diff_n_steps",
            "base_goal_first", "base_tower_touch", "notower_spot_touch", "base_dev_max", "no_tower_dev_max"]
    ps = pdf.groupby(["bank", "cohort", "seed"])[cols].mean().reset_index()
    fdf["tbin"] = pd.cut(fdf.t, [-1, 0, 9, 19, 39], labels=["t0", "t1-9", "t10-19", "t20-39"])
    fps = fdf.groupby(["bank", "cohort", "seed", "tbin"], observed=True)[["tower_pull", "flag_pull", "tower_vs_flag",
                                                                           "flag_vs_tower"]].mean().reset_index()
    fps_all = fdf[fdf.t >= 1].groupby(["bank", "cohort", "seed"])[["tower_pull", "flag_pull"]].mean().reset_index()
    fps_vis = fdf[(fdf.t >= 1) & fdf.tower_in_fov].groupby(["bank", "cohort", "seed"])[["tower_pull"]].mean().rename(
        columns={"tower_pull": "tower_pull_in_fov"}).reset_index()
    fps_fvis = fdf[(fdf.t >= 1) & fdf.flag_in_fov].groupby(["bank", "cohort", "seed"])[["flag_pull"]].mean().rename(
        columns={"flag_pull": "flag_pull_in_fov"}).reset_index()
    ps = ps.merge(fps_all, on=["bank", "cohort", "seed"]).merge(fps_vis, on=["bank", "cohort", "seed"], how="left").merge(
        fps_fvis, on=["bank", "cohort", "seed"], how="left")
    ps.to_csv(os.path.join(OUT, "trace_per_seed.csv"), index=False)
    fps.to_csv(os.path.join(OUT, "follow_per_seed_tbin.csv"), index=False)
    allc = cols + ["tower_pull", "flag_pull", "tower_pull_in_fov", "flag_pull_in_fov"]
    with open(os.path.join(OUT, "trace_summary.md"), "w") as fh:
        fh.write("# Seeking vs perceiving (mean [95% t-CI] across seeds of per-layout means)\n\ndiff_* = acting with the tower "
                 "minus acting on the same layout without it (paired). dev = lateral offset from the start-flag line toward the "
                 "tower's side (m). steps_near = steps within 1 m of the tower spot. tower_pull / flag_pull: oracle-form change "
                 "in log support of heading to the object caused by its presence, at the base trajectory's poses, steps 1-39 "
                 "(t0 excluded); *_in_fov restricted to steps where the object is inside the camera's field of view.\n\n")
        fh.write("| bank | cohort | " + " | ".join(allc) + " |\n|---|---|" + "---|" * len(allc) + "\n")
        for (b, coh), sub in ps.groupby(["bank", "cohort"], sort=False):
            fh.write(f"| {b} | {coh} | " + " | ".join(C.fmt_ci(*C.ci95(sub[c].to_numpy()), 2) for c in allc) + " |\n")
        fh.write("\n## pulls by time bin\n\n| bank | cohort | t | tower_pull | flag_pull | tower_vs_flag |\n|---|---|---|---|---|---|\n")
        for (b, coh, tb), sub in fps.groupby(["bank", "cohort", "tbin"], sort=False, observed=True):
            fh.write(f"| {b} | {coh} | {tb} | " + " | ".join(C.fmt_ci(*C.ci95(sub[c].to_numpy()), 2)
                                                            for c in ("tower_pull", "flag_pull", "tower_vs_flag")) + " |\n")
    print(open(os.path.join(OUT, "trace_summary.md")).read())


if __name__ == "__main__":
    main()
