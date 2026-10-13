"""CPU analysis of rewire_check rollouts: reproduction, phantom-goal chance, marker-decoupled reach, directionality; per-episode CSV then seed CIs."""
import argparse
import os

import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C
WS_LO, WS_HI = np.array((-0.20, -0.25)), np.array((0.20, 0.25))  # location_reach.py; importing it needs sapien

OUT = os.path.join(C.RESULTS, "rewire_check")
PATHS = os.path.join(OUT, "paths")
THR, GOAL_Z = 0.03, 0.06


def first_hit(path, pts, thr=THR):
    """First step within thr of each point; path [T, 3], pts [K, 3] -> [K] (T if never)."""
    hit = np.linalg.norm(path[None] - pts[:, None], axis=-1) < thr
    return np.where(hit.any(1), hit.argmax(1), path.shape[0])


def phantom_pool(family, seeds, rng, n_box=4000):
    """Candidate phantom goals. 'pool': eval_offpath.py's goal pool (GM pass-1 gripper positions at marker height
    +-1 cm, >= 10 cm from the corner, inside the workspace) pooled over all pair seeds, because one seed's pool holds
    only ~28 positions. 'box': uniform over the bounding box of the actual off-path goals at marker height."""
    pts, own = [], []
    for k in seeds:
        z = np.load(os.path.join(C.ROOT, "results", "onpath", f"offpath_paths_s{k}.npz"))
        p1, corner = z["gm_pass1"], z["corner"]
        q, o = p1.reshape(-1, 3), np.repeat(np.arange(len(p1)), p1.shape[1])
        keep = (np.abs(q[:, 2] - GOAL_Z) <= 0.01) & (np.linalg.norm(q[:, :2] - corner[:2], axis=1) >= 0.10)
        keep &= ((q[:, :2] >= WS_LO) & (q[:, :2] <= WS_HI)).all(1)
        if family == "pool":
            pts.append(q[keep]), own.append(np.stack([np.full(keep.sum(), k), o[keep]], 1))
        else:
            pts.append(z["goal"])
    if family == "pool":
        return np.concatenate(pts), np.concatenate(own), corner
    g = np.concatenate(pts)
    lo, hi = g.min(0), g.max(0)
    b = rng.uniform(lo, hi, (n_box, 3)).astype(np.float32)
    b = b[np.linalg.norm(b[:, :2] - corner[:2], axis=1) >= 0.10]
    return b, np.full((len(b), 2), -1), corner


def sample_phantoms(k, family, seeds, goal, dxy, tcp0, K, rng, tol=(0.02, 0.04, 0.08, np.inf), sep=0.10):
    """K phantom goals per episode under the true goal's eligibility rules (>= 8 cm off this start's own GM pass-1
    path, not from its own pass-1 episode, >= 8 cm from the distractor) and >= sep from the true goal, matched in 3D
    distance from the start TCP and xy distance from the corner; tolerance widened only if < K candidates."""
    pts, own, corner = phantom_pool(family, seeds, rng)
    p1 = np.load(os.path.join(C.ROOT, "results", "onpath", f"offpath_paths_s{k}.npz"))["gm_pass1"]
    dc_pts = np.linalg.norm(pts[:, :2] - corner[:2], axis=1)
    out, used_tol = np.zeros((len(goal), K, 3), np.float32), np.zeros(len(goal))
    for i in range(len(goal)):
        ok = np.linalg.norm(p1[i][None] - pts[:, None], axis=-1).min(1) >= 0.08
        ok &= ~((own[:, 0] == k) & (own[:, 1] == i)) & (np.linalg.norm(pts[:, :2] - dxy[i], axis=1) >= 0.08)
        ok &= np.linalg.norm(pts - goal[i], axis=1) >= sep
        ds_g, dc_g = np.linalg.norm(goal[i] - tcp0[i]), np.linalg.norm(goal[i, :2] - corner[:2])
        ds_p = np.linalg.norm(pts - tcp0[i], axis=1)
        for t in tol:
            m = ok & (np.abs(ds_p - ds_g) <= t) & (np.abs(dc_pts - dc_g) <= t)
            if m.sum() >= K:
                break
        if not m.any():
            # Only a few pooled positions survive the own-path rule for some starts; then drop that rule (tol -1).
            m, t = (np.linalg.norm(pts - goal[i], axis=1) >= sep) & (np.linalg.norm(pts[:, :2] - dxy[i], axis=1) >= 0.08), -1
        idx = np.nonzero(m)[0]
        out[i] = pts[rng.choice(idx, size=K, replace=len(idx) < K)]
        used_tol[i] = t
    return out, used_tol


def md_table(df):
    rows = [list(map(str, df.columns))] + [list(map(str, r)) for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def corr_xy(a, b):
    return np.mean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in (0, 1)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--heading_steps", type=int, default=5)
    a = p.parse_args()
    files = sorted(f for f in os.listdir(PATHS) if f.endswith(".npz"))
    ref = pd.read_csv(os.path.join(C.RESULTS, "rewire_LR", "closed_loop_per_episode.csv"))
    ref = ref[ref.set == "offpath"]
    phantoms = {}
    ep_rows, seed_rows = [], []
    for f in files:
        parts = f[:-4].split("_")
        cohort, sk, cond, arm = parts[0], parts[1], parts[-1], "_".join(parts[2:-1])
        k = int(sk[1:])
        if k not in a.seeds:
            continue
        z = np.load(os.path.join(PATHS, f))
        path, marker, scored, dxy, corner = z["path"], z["marker"], z["scored"], z["dxy"], z["corner"]
        n, T = path.shape[:2]
        tcp0 = path[:, 0]
        if k not in phantoms:
            phantoms[k] = {fam: sample_phantoms(k, fam, a.seeds, scored, dxy, tcp0, a.K, np.random.default_rng(500 + k))
                           for fam in ("pool", "box")}
        ph, ph_tol = phantoms[k]["pool"]
        phb, phb_tol = phantoms[k]["box"]
        tc_tmp = np.array([first_hit(path[i], corner[None])[0] for i in range(n)])
        reach_phb = (np.stack([first_hit(path[i], phb[i]) for i in range(n)]) < tc_tmp[:, None]).mean(1)
        t_c = np.array([first_hit(path[i], corner[None])[0] for i in range(n)])
        t_s = np.array([first_hit(path[i], scored[i][None])[0] for i in range(n)])
        t_m = np.array([first_hit(path[i], marker[i][None])[0] for i in range(n)])
        t_ph = np.stack([first_hit(path[i], ph[i]) for i in range(n)])
        reach_s, reach_m = t_s < t_c, t_m < t_c
        reach_ph = (t_ph < t_c[:, None]).mean(1)
        h = path[:, a.heading_steps, :2] - tcp0[:, :2]
        moved = np.linalg.norm(h, axis=1)
        ep = pd.DataFrame(dict(cohort=cohort, seed=k, arm=arm, cond=cond, episode=np.arange(n),
                               reach_scored=reach_s, reach_marker=reach_m, reach_corner=t_c < np.minimum(t_s, T),
                               t_scored=t_s, t_marker=t_m, t_corner=t_c, p_reach_phantom=reach_ph,
                               phantom_tol=ph_tol, p_reach_phantom_box=reach_phb, phantom_box_tol=phb_tol,
                               min_d_scored=np.linalg.norm(path - scored[:, None], axis=-1).min(1),
                               min_d_marker=np.linalg.norm(path - marker[:, None], axis=-1).min(1),
                               min_d_phantom_med=np.median(np.linalg.norm(path[:, None] - ph[:, :, None], axis=-1).min(2), 1),
                               heading_disp=moved,
                               cos_head_scored=cos(h, scored[:, :2] - tcp0[:, :2]),
                               cos_head_marker=cos(h, marker[:, :2] - tcp0[:, :2]),
                               cos_head_corner=cos(h, corner[None, :2] - tcp0[:, :2]),
                               cos_head_phantom=np.mean([cos(h, ph[:, j, :2] - tcp0[:, :2]) for j in range(a.K)], 0),
                               final_x=path[:, -1, 0], final_y=path[:, -1, 1], final_z=path[:, -1, 2],
                               scored_x=scored[:, 0], scored_y=scored[:, 1], marker_x=marker[:, 0], marker_y=marker[:, 1],
                               path_len=np.linalg.norm(np.diff(path, axis=1), axis=-1).sum(1)))
        if cond == "standard" and arm in ("native", "fc_head", "pre", "remix"):
            r = ref[(ref.cohort == cohort) & (ref.seed == k) & (ref.arm == arm)].sort_values("episode")
            if len(r) == n:
                ep["ref_outcome"] = r.outcome.to_numpy()
        ep_rows.append(ep)
        fin = path[:, -1, :2]
        srow = dict(cohort=cohort, seed=k, arm=arm, cond=cond, reach_scored=reach_s.mean(), reach_marker=reach_m.mean(),
                    reach_corner=ep.reach_corner.mean(), p_phantom=reach_ph.mean(), p_phantom_box=reach_phb.mean(),
                    frac_matched_2cm=np.mean(ph_tol <= 0.02), frac_box_matched_2cm=np.mean(phb_tol <= 0.02),
                    excess_over_phantom=reach_s.mean() - reach_ph.mean(),
                    excess_over_box=reach_s.mean() - reach_phb.mean(),
                    cos_head_scored=np.nanmean(ep.cos_head_scored), cos_head_marker=np.nanmean(ep.cos_head_marker),
                    cos_head_corner=np.nanmean(ep.cos_head_corner), cos_head_phantom=np.nanmean(ep.cos_head_phantom),
                    corr_final_scored=corr_xy(fin, scored[:, :2]),
                    corr_final_marker=corr_xy(fin, marker[:, :2]) if cond != "nomarker" else np.nan,
                    corr_final_tcp0=corr_xy(fin, tcp0[:, :2]),
                    med_min_d_scored=np.median(ep.min_d_scored), med_min_d_phantom=np.median(ep.min_d_phantom_med),
                    path_len=ep.path_len.mean())
        if "ref_outcome" in ep:
            srow["repro_agree"] = float(((ep.ref_outcome == "goal") == ep.reach_scored).mean())
        seed_rows.append(srow)
        print(f"{cohort} s{k} {arm:10s} {cond:9s} reach {srow['reach_scored']:.3f} marker {srow['reach_marker']:.3f} "
              f"phantom {srow['p_phantom']:.3f}/{srow['p_phantom_box']:.3f} cos_head {srow['cos_head_scored']:+.2f}/{srow['cos_head_marker']:+.2f} "
              f"corrF {srow['corr_final_scored']:+.2f}/{srow['corr_final_marker']:+.2f} repro {srow.get('repro_agree', np.nan):.3f}",
              flush=True)
    ep_df = pd.concat(ep_rows)
    ep_df.to_csv(os.path.join(OUT, "per_episode.csv"), index=False)
    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    cols = ["reach_scored", "reach_marker", "p_phantom", "excess_over_phantom", "p_phantom_box", "excess_over_box", "cos_head_scored", "cos_head_marker",
            "cos_head_corner", "corr_final_scored", "corr_final_marker", "corr_final_tcp0", "med_min_d_scored",
            "med_min_d_phantom"]
    rows = []
    for (cohort, arm, cond), g in per_seed.groupby(["cohort", "arm", "cond"]):
        r = dict(cohort=cohort, arm=arm, cond=cond, n=len(g))
        for c in cols:
            m, lo, hi = C.ci95(g.sort_values("seed")[c].to_numpy())
            r[c] = C.fmt_ci(m, lo, hi)
            r[c + "_seeds"] = " / ".join(f"{v:.3f}" for v in g.sort_values("seed")[c])
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("# LR rewire stress test (mean [95% t-CI] over seeds; per-seed in *_seeds)\n\n")
        for cols_sub, title in ((["reach_scored", "reach_marker", "p_phantom", "excess_over_phantom", "p_phantom_box", "excess_over_box"], "Reach"),
                                (["cos_head_scored", "cos_head_marker", "cos_head_corner", "corr_final_scored",
                                  "corr_final_marker", "corr_final_tcp0"], "Directionality"),
                                (["med_min_d_scored", "med_min_d_phantom"], "Closest approach (m)")):
            fh.write(f"## {title}\n\n" + md_table(summ[["cohort", "arm", "cond", "n"] + cols_sub]) + "\n\n")
            fh.write(md_table(summ[["cohort", "arm", "cond"] + [c + "_seeds" for c in cols_sub]]) + "\n\n")


if __name__ == "__main__":
    main()
