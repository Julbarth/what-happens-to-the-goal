"""PushToMarker conflict geometry from saved native rollout paths: stopping on the marker ray and behaviour off the ray."""
import argparse
import os

import numpy as np
import pandas as pd
from gm_robotics.paths import ROOT as RUN_ROOT

ROOT = str(RUN_ROOT)
SRC = os.path.join(ROOT, "results", "analysis", "rewire_ptm")
CENTRE = np.array([-0.08, 0.0], np.float32)
GOAL_THR, PROXY_THR, NEAR = 0.025, 0.03, 0.025
SEP_MIN = GOAL_THR + PROXY_THR  # marker this far from the centre: goal and centre outcomes are disjoint
# Two-sided 97.5% t quantiles by df (keeps the script scipy-free).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
STILL = 0.001  # m/step; below this the cube counts as not moving
PERP = 0.025  # ctrlset: marker counts as off the inward path if it is this far from the cube->centre segment


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    h = T975[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def fmt(x, k=3, scale=1.0):
    m, lo, hi = ci(np.asarray(x, float) * scale)
    return f"{m:.{k}f} [{lo:.{k}f}, {hi:.{k}f}]"


def load_paths(cohort, seed, sname, cond):
    z = np.load(os.path.join(SRC, "paths", f"{cohort}_s{seed}_native_{sname}_{cond}.npz"))
    return z["cube"].astype(np.float64), z["cube_z"]


def seg_dist(p, a, b):
    """Distance from points p to segments a-b (row-wise)."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def conflict_episode_metrics(cube, m, m2):
    """Per-episode transit metrics of the cube relative to the on-ray marker m (and the off-ray m2)."""
    n, T1, _ = cube.shape
    d_c = np.linalg.norm(cube - CENTRE, axis=-1)
    d_m = np.linalg.norm(cube - m[:, None], axis=-1)
    d_m2 = np.linalg.norm(cube - m2[:, None], axis=-1)
    v = np.linalg.norm(np.diff(cube, axis=1), axis=-1)  # v[:, t] = move from t to t+1
    reach_c = (d_c < PROXY_THR).any(1)
    t_c = np.where(reach_c, (d_c < PROXY_THR).argmax(1), T1 - 1)
    rows = []
    for i in range(n):
        pre = slice(0, t_c[i] + 1)
        dm_pre = d_m[i, pre]
        near = d_m[i, :-1] < NEAR
        near_pre = near.copy()
        near_pre[t_c[i]:] = False
        moved = np.linalg.norm(cube[i] - cube[i, 0], axis=1) > 0.005
        t0 = moved.argmax() if moved.any() else T1 - 1
        k = int(dm_pre.argmin())
        w = v[i, max(k - 1, 0):min(k + 1, T1 - 1)]
        # Pause: cube stationary while over the marker, after it first moved and before it first reached the centre.
        still_near = near_pre & (v[i] < STILL) & (np.arange(T1 - 1) >= t0)
        # Frames are ~2.5 cm apart for the GM, so also measure the polyline (segment) distance to m.
        a_, b_ = cube[i, :t_c[i]], cube[i, 1:t_c[i] + 1]
        seg_pre = seg_dist(np.repeat(m[i][None], len(a_), 0), a_, b_).min() if len(a_) else dm_pre.min()
        rows.append(dict(
            pass_pre=dm_pre.min() < NEAR, pass_pre_seg=seg_pre < NEAR, min_dm_pre_seg=seg_pre, min_dm_pre=dm_pre.min(), reach_centre=reach_c[i], t_centre=t_c[i],
            t_move=t0, v_closest=w.mean(), dwell_pre=near_pre.sum(), dwell_all=near.sum(),
            pause_steps=still_near.sum(), paused=still_near.any(),
            v_near=v[i][near_pre].mean() if near_pre.any() else np.nan,
            v_median_moving=np.median(v[i][(v[i] > STILL)]) if (v[i] > STILL).any() else np.nan,
            min_dm2=d_m2[i].min(), final_dm=d_m[i, -1], final_dc=d_c[i, -1]))
    return pd.DataFrame(rows)


def conflict(seeds, cohorts):
    sc = np.load(os.path.join(SRC, "conflict_scenes.npz"))
    m, m2, s = sc["m"].astype(np.float64), sc["m2"].astype(np.float64), sc["cube"][:, :2].astype(np.float64)
    u_c = (CENTRE - s) / np.linalg.norm(CENTRE - s, axis=1, keepdims=True)
    u_m2 = (m2 - s) / np.linalg.norm(m2 - s, axis=1, keepdims=True)
    geo = dict(d_cube_m=np.linalg.norm(m - s, axis=1), d_cube_c=np.linalg.norm(CENTRE - s, axis=1),
               d_m_c=np.linalg.norm(m - CENTRE, axis=1),
               ang_m2_vs_centre=np.degrees(np.arccos(np.clip((u_c * u_m2).sum(1), -1, 1))),
               d_m2_from_inward_path=seg_dist(m2, s, CENTRE.astype(np.float64)[None].repeat(len(s), 0)))
    eps = []
    for cohort in cohorts:
        for k in seeds:
            for cond in ("standard", "nomarker", "decoupled"):
                cube, cz = load_paths(cohort, k, "conflict", cond)
                df = conflict_episode_metrics(cube, m, m2)
                on = cz < 0.02 + 0.005
                df["goal"] = (df.final_dm < GOAL_THR) & on
                df["proxy"] = (df.final_dc < PROXY_THR) & on & ~df.goal
                df["reach_m2_final"] = (np.linalg.norm(cube[:, -1] - m2, axis=1) < GOAL_THR) & on
                df = df.assign(episode=np.arange(len(df)), cohort=cohort, seed=k, cond=cond, **geo)
                eps.append(df)
    return pd.concat(eps, ignore_index=True)


def ctrlset(seeds, cohorts):
    sc = np.load(os.path.join(SRC, "ctrlset_scenes.npz"))
    m, s = sc["m"].astype(np.float64), sc["cube"][:, :2].astype(np.float64)
    c = CENTRE.astype(np.float64)[None].repeat(len(s), 0)
    u_c = (c - s) / np.linalg.norm(c - s, axis=1, keepdims=True)
    u_m = (m - s) / np.linalg.norm(m - s, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((u_c * u_m).sum(1), -1, 1)))
    d_m_c = np.linalg.norm(m - CENTRE, axis=1)
    off_path = seg_dist(m, s, c) > PERP
    sep = d_m_c > SEP_MIN
    d_s_c = np.linalg.norm(s - CENTRE, axis=1)
    annulus = (d_s_c >= 0.05) & (d_s_c <= 0.16)
    eps = []
    for cohort in cohorts:
        for k in seeds:
            for cond in ("standard", "nomarker"):
                cube, cz = load_paths(cohort, k, "ctrlset", cond)
                on = cz < 0.025
                f = cube[:, -1]
                dm = np.linalg.norm(f - m, axis=1)
                dc = np.linalg.norm(f - CENTRE, axis=1)
                goal = (dm < GOAL_THR) & on
                disp = f - s
                # Signed lateral displacement of the final cube position toward the marker side of the centre ray.
                side = np.sign(u_c[:, 0] * u_m[:, 1] - u_c[:, 1] * u_m[:, 0])
                lat = side * (u_c[:, 0] * disp[:, 1] - u_c[:, 1] * disp[:, 0])
                eps.append(pd.DataFrame(dict(
                    episode=np.arange(len(s)), cohort=cohort, seed=k, cond=cond, goal=goal,
                    proxy=(dc < PROXY_THR) & on & ~goal, final_dm=dm, final_dc=dc,
                    min_dm=np.linalg.norm(cube - m[:, None], axis=-1).min(1), lat_toward_m=lat,
                    ang_m_vs_centre=ang, d_m_c=d_m_c, separable=sep, off_path=off_path, annulus=annulus)))
    return pd.concat(eps, ignore_index=True)


def per_seed(eps, groups, cols):
    return eps.groupby(groups)[cols].mean().reset_index()


def paired(eps, col, a="standard", b="nomarker", mask=None):
    """Per-seed mean of the per-episode difference cond a - cond b (same scenes)."""
    e = eps if mask is None else eps[mask]
    p = e.pivot_table(index=["cohort", "seed", "episode"], columns="cond", values=col)
    return (p[a] - p[b]).groupby(level=["cohort", "seed"]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cohorts", nargs="+", default=["gm", "control"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "analysis", "ptm_geometry"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ce = conflict(a.seeds, a.cohorts)
    ref = pd.read_csv(os.path.join(SRC, "per_episode.csv"))
    ref = ref[(ref.arm == "native") & (ref.set == "conflict")]
    chk = ce.merge(ref[["cohort", "seed", "cond", "episode", "d_marker"]], on=["cohort", "seed", "cond", "episode"])
    assert len(chk) == len(ce) and np.allclose(chk.final_dm, chk.d_marker, atol=1e-4), "scene/path misalignment"
    ce.to_csv(os.path.join(a.out, "conflict_per_episode.csv"), index=False)
    cs = ce.groupby(["cohort", "seed", "cond"]).agg(
        goal=("goal", "mean"), proxy=("proxy", "mean"), reach_m2=("reach_m2_final", "mean"),
        pass_pre=("pass_pre", "mean"), pass_pre_seg=("pass_pre_seg", "mean"),
        min_dm_pre_seg_cm=("min_dm_pre_seg", lambda x: 100 * x.median()), min_dm_pre_cm=("min_dm_pre", lambda x: 100 * x.median()),
        v_closest_cm=("v_closest", lambda x: 100 * x.mean()), v_near_cm=("v_near", lambda x: 100 * x.mean()),
        v_moving_median_cm=("v_median_moving", lambda x: 100 * x.mean()), dwell_pre=("dwell_pre", "mean"),
        paused=("paused", "mean"), pause_steps=("pause_steps", "mean"), min_dm2_cm=("min_dm2", lambda x: 100 * x.median()),
    ).reset_index()
    cs.to_csv(os.path.join(a.out, "conflict_per_seed.csv"), index=False)

    xe = ctrlset(a.seeds, a.cohorts)
    xe.to_csv(os.path.join(a.out, "ctrlset_per_episode.csv"), index=False)
    strata = dict(all=np.ones(len(xe), bool), separable=xe.separable.values,
                  separable_offpath=(xe.separable & xe.off_path).values,
                  sep_offpath_annulus=(xe.separable & xe.off_path & xe.annulus).values,
                  sep_ang_gt60=(xe.separable & (xe.ang_m_vs_centre > 60)).values)
    xs = []
    for nm, msk in strata.items():
        g = xe[msk].groupby(["cohort", "seed", "cond"]).agg(
            n=("goal", "size"), goal=("goal", "mean"), proxy=("proxy", "mean"),
            min_dm_cm=("min_dm", lambda x: 100 * x.mean()), lat_toward_m_cm=("lat_toward_m", lambda x: 100 * x.mean()),
        ).reset_index().assign(stratum=nm)
        xs.append(g)
    xs = pd.concat(xs, ignore_index=True)
    xs.to_csv(os.path.join(a.out, "ctrlset_per_seed.csv"), index=False)

    L = ["# PTM conflict geometry (saved rewire_ptm native paths; mean [95% t-CI] over 3 seeds, 512 scenes each)", ""]
    sc = ce[(ce.cohort == a.cohorts[0]) & (ce.seed == a.seeds[0]) & (ce.cond == "standard")]
    L += [f"Conflict scene geometry: cube->marker {100 * sc.d_cube_m.median():.1f} cm median "
          f"(range {100 * sc.d_cube_m.min():.1f}-{100 * sc.d_cube_m.max():.1f}), cube->centre {100 * sc.d_cube_c.median():.1f} cm, "
          f"marker->centre {100 * sc.d_m_c.median():.1f} cm. m2 (decoupled marker) angle to the centre direction from the cube: "
          f"median {sc.ang_m2_vs_centre.median():.0f} deg (IQR {sc.ang_m2_vs_centre.quantile(.25):.0f}-"
          f"{sc.ang_m2_vs_centre.quantile(.75):.0f}), distance from the inward path median {100 * sc.d_m2_from_inward_path.median():.1f} cm.", ""]
    L += ["## Conflict transit (per seed means)", "",
          "| cohort | cond | goal | centre | reach m2 | cube passes <2.5 cm of m before centre | median min d(m) pre-centre cm | "
          "polyline passes <2.5 cm (median min d cm) | cube speed at closest approach cm/step | speed while over m cm/step | median moving speed cm/step | "
          "steps over m pre-centre | P(pause over m) |", "|" + "---|" * 13]
    for (co, cd), g in cs.groupby(["cohort", "cond"]):
        L.append(f"| {co} | {cd} | {fmt(g.goal)} | {fmt(g.proxy)} | {fmt(g.reach_m2)} | {fmt(g.pass_pre)} | "
                 f"{fmt(g.min_dm_pre_cm, 2)} | {fmt(g.pass_pre_seg)} ({fmt(g.min_dm_pre_seg_cm, 2)}) | {fmt(g.v_closest_cm, 2)} | {fmt(g.v_near_cm, 2)} | "
                 f"{fmt(g.v_moving_median_cm, 2)} | {fmt(g.dwell_pre, 2)} | {fmt(g.paused)} |")
    L += ["", "### Paired marker effect on the same scenes (standard - nomarker), per-seed mean of per-episode differences", "",
          "| cohort | d speed at closest approach cm/step | d steps over m pre-centre | d P(pause over m) | d final dist to centre cm |",
          "|---|---|---|---|---|"]
    for co in a.cohorts:
        e = ce[ce.cohort == co]
        L.append(f"| {co} | {fmt(paired(e, 'v_closest'), 3, 100)} | {fmt(paired(e, 'dwell_pre'), 2)} | "
                 f"{fmt(paired(e.assign(paused=e.paused.astype(float)), 'paused'))} | {fmt(paired(e, 'final_dc'), 2, 100)} |")
    L += ["", "### Decoupled (marker drawn at off-ray m2, nothing at m): does the path bend toward m2? "
          "(decoupled - standard, median-free mean of per-episode min d(cube, m2), cm)", "",
          "| cohort | d min dist to m2 cm | reach m2 decoupled | reach m2 standard (m2 not drawn) |", "|---|---|---|---|"]
    for co in a.cohorts:
        e = ce[ce.cohort == co]
        g = cs[cs.cohort == co]
        L.append(f"| {co} | {fmt(paired(e, 'min_dm2', 'decoupled', 'standard'), 2, 100)} | "
                 f"{fmt(g[g.cond == 'decoupled'].reach_m2)} | {fmt(g[g.cond == 'standard'].reach_m2)} |")

    L += ["", "## Control-distribution layouts (ctrlset: marker and cube uniform on the r<16 cm disc, >=5 cm apart)", "",
          f"Strata: separable = marker >{100 * SEP_MIN:.1f} cm from the centre (goal and centre outcomes disjoint); "
          f"offpath = marker >{100 * PERP:.1f} cm from the straight cube->centre segment; ang>60 = cube->marker vs "
          "cube->centre direction differ by >60 deg; annulus = cube starts 5-16 cm from the centre (GM-trained start radii).", "",
          "| cohort | stratum | n/seed | cond | goal | centre | mean min d(cube, m) cm | lateral final displacement toward m cm |",
          "|---|---|---|---|---|---|---|---|"]
    for (co, st, cd), g in xs.groupby(["cohort", "stratum", "cond"], sort=False):
        L.append(f"| {co} | {st} | {g.n.mean():.0f} | {cd} | {fmt(g.goal)} | {fmt(g.proxy)} | {fmt(g.min_dm_cm, 2)} | "
                 f"{fmt(g.lat_toward_m_cm, 2)} |")
    L += ["", "### Paired marker effect on ctrlset (standard - nomarker)", "",
          "| cohort | stratum | d goal | d centre | d min dist to m cm | d lateral toward m cm |", "|---|---|---|---|---|---|"]
    for co in a.cohorts:
        for nm, msk in strata.items():
            e = xe[msk & (xe.cohort == co).values].copy()
            e[["goal", "proxy"]] = e[["goal", "proxy"]].astype(float)
            L.append(f"| {co} | {nm} | {fmt(paired(e, 'goal'))} | {fmt(paired(e, 'proxy'))} | "
                     f"{fmt(paired(e, 'min_dm'), 2, 100)} | {fmt(paired(e, 'lat_toward_m'), 2, 100)} |")
    with open(os.path.join(a.out, "tables_generated.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
