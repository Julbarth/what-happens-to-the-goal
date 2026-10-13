"""Per-episode kinematics of the LR and PTM rewiring rollouts: stratified success, stop-point and heading checks, figure."""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from gm_robotics.paths import ROOT as RUN_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = str(RUN_ROOT)
RES = os.path.join(ROOT, "results", "analysis")
OUT = os.path.join(RES, "rewire_behaviour")
FIG = os.path.join(ROOT, "figures", "analysis", "paper_style")
PTM_C = np.array([-0.08, 0.0])
PTM_GOAL_THR, LR_THR = 0.025, 0.03
LR_SEEDS, PTM_SEEDS = (1, 2, 3), (0, 1, 2)
ARMS = [("gm", "native"), ("gm", "pre"), ("control", "pre"), ("init", "pre"), ("gm", "pre_masked"), ("gm", "fc_head"), ("gm", "remix")]
LAB = {("gm", "native"): "GM native", ("gm", "pre"): "GM rewired h_pre", ("control", "pre"): "control rewired h_pre",
       ("init", "pre"): "init-trunk h_pre", ("gm", "pre_masked"): "GM masked retrain", ("gm", "fc_head"): "GM h_fc head",
       ("gm", "remix"): "GM remix"}
STYLE = {("gm", "native"): ("#C45C58", ":"), ("gm", "pre"): ("#7A1520", "-"), ("control", "pre"): ("#3D8B8B", "-"),
         ("init", "pre"): ("#8E8E8E", "--"), ("gm", "pre_masked"): ("#2A2A2A", "-."), }
K_HEAD = 5


def cos(a, b):
    return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)


# ---------------------------------------------------------------- LocationReach (TCP paths, first contact within 50 steps)

def lr_episodes(seeds):
    rows, curves = [], []
    for cohort, arm in ARMS:
        for k in seeds:
            for cond in ("standard", "nomarker", "decoupled"):
                f = os.path.join(RES, "rewire_check", "paths", f"{cohort}_s{k}_{arm}_{cond}.npz")
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                p, goal, marker, corner = z["path"], z["scored"], z["marker"], z["corner"]
                d = np.linalg.norm(p - goal[:, None], axis=-1)
                dm = np.linalg.norm(p - marker[:, None], axis=-1)
                dc = np.linalg.norm(p - corner[None, None], axis=-1)
                T1 = p.shape[1]
                hit_g, hit_c = d < LR_THR, dc < LR_THR
                tg = np.where(hit_g.any(1), hit_g.argmax(1), T1)
                tc = np.where(hit_c.any(1), hit_c.argmax(1), T1)
                reach = (tg < tc)
                # stop accounting at first contact (the trained episode end): truncate the path there
                t_end = np.minimum(np.minimum(tg, tc), T1 - 1)
                step = np.linalg.norm(np.diff(p, axis=1), axis=-1)
                live = np.arange(T1 - 1)[None] < t_end[:, None]
                plen = (step * live).sum(1)
                head = p[:, K_HEAD] - p[:, 0]
                # per-step progress toward the goal, over steps before first contact
                prog = (d[:, :-1] - d[:, 1:])
                frac_toward = np.where(live, prog > 0, False).sum(1) / np.maximum(live.sum(1), 1)
                eff = (d[:, 0] - d.min(1)) / np.maximum(plen, 1e-6)
                for i in range(len(p)):
                    rows.append(dict(task="LR", cohort=cohort, seed=k, arm=arm, cond=cond, episode=i, reach=bool(reach[i]),
                                     corner=bool(tc[i] < tg[i]), t_contact=int(tg[i]), d0=float(d[i, 0]), dmin=float(d[i].min()),
                                     d_final=float(d[i, t_end[i]]), dmin_marker=float(dm[i].min()),
                                     cos_head_goal=float(cos(head[i, :2], goal[i, :2] - p[i, 0, :2])),
                                     cos_head_corner=float(cos(head[i, :2], corner[:2] - p[i, 0, :2])),
                                     head_speed=float(np.linalg.norm(head[i]) / K_HEAD), path_len=float(plen[i]),
                                     mean_speed=float(plen[i] / max(t_end[i], 1)), frac_steps_toward=float(frac_toward[i]),
                                     efficiency=float(eff[i]), goal_x=float(goal[i, 0]), goal_y=float(goal[i, 1]),
                                     goal_corner_dist=float(np.linalg.norm(goal[i, :2] - corner[:2]))))
                if cond == "standard":
                    curves.append(dict(cohort=cohort, seed=k, arm=arm, curve=(d / d[:, :1]).mean(0)))
    return pd.DataFrame(rows), curves


# ---------------------------------------------------------------- PushToMarker (cube + TCP paths, final state at 100 steps)

def ptm_scenes():
    c = np.load(os.path.join(RES, "rewire_ptm", "conflict_scenes.npz"))
    s = np.load(os.path.join(RES, "rewire_ptm", "ctrlset_scenes.npz"))
    return {"conflict": dict(m=c["m"], m2=c["m2"], cube=c["cube"][:, :2]), "ctrlset": dict(m=s["m"], cube=s["cube"][:, :2])}


def ptm_episodes(seeds):
    sc = ptm_scenes()
    rows, curves = [], []
    for cohort, arm in ARMS:
        for k in seeds:
            for sset in ("conflict", "ctrlset"):
                for cond in ("standard", "nomarker", "decoupled"):
                    f = os.path.join(RES, "rewire_ptm", "paths", f"{cohort}_s{k}_{arm}_{sset}_{cond}.npz")
                    if not os.path.exists(f):
                        continue
                    z = np.load(f)
                    cube, tcp = z["cube"], z["tcp"]
                    m = sc[sset]["m"]
                    drawn = sc[sset]["m2"] if cond == "decoupled" else m
                    c0 = cube[:, 0]
                    d = np.linalg.norm(cube - m[:, None], axis=-1)
                    d_dr = np.linalg.norm(cube - drawn[:, None], axis=-1)
                    moved = np.linalg.norm(cube - c0[:, None], axis=-1)
                    T1 = cube.shape[1]
                    mv = moved > 0.005
                    t_contact = np.where(mv.any(1), mv.argmax(1), T1)
                    # push direction = cube displacement once it has moved 1 cm (the first committed push)
                    mv1 = moved > 0.01
                    t1 = np.where(mv1.any(1), mv1.argmax(1), T1 - 1)
                    pdir = cube[np.arange(len(cube)), t1] - c0
                    fin = cube[:, -1]
                    on = z["cube_z"] < 0.025
                    goal = (d[:, -1] < PTM_GOAL_THR) & on
                    # stop point along the cube -> marker line: 1 = on the marker, the centre sits at s_c
                    u = m - c0
                    L = np.linalg.norm(u, axis=1)
                    s_stop = np.sum((fin - c0) * u, 1) / L ** 2
                    s_c = np.sum((PTM_C - c0) * u, 1) / L ** 2
                    lat = np.abs(np.cross(fin - c0, u / L[:, None]))
                    tcp_d0 = np.linalg.norm(tcp[:, 0, :2] - c0, axis=1)
                    for i in range(len(cube)):
                        rows.append(dict(task="PTM", set=sset, cohort=cohort, seed=k, arm=arm, cond=cond, episode=i,
                                         goal=bool(goal[i]), reach_drawn=bool((d_dr[i, -1] < PTM_GOAL_THR) & on[i]),
                                         d0=float(L[i]), d_final=float(d[i, -1]), dmin=float(d[i].min()),
                                         d_final_drawn=float(d_dr[i, -1]), d_centre_final=float(np.linalg.norm(fin[i] - PTM_C)),
                                         cube_moved=float(moved[i, -1]), t_contact=int(t_contact[i]),
                                         cos_push_marker=float(cos(pdir[i], m[i] - c0[i])) if t1[i] < T1 - 1 else np.nan,
                                         cos_push_drawn=float(cos(pdir[i], drawn[i] - c0[i])) if t1[i] < T1 - 1 else np.nan,
                                         cos_push_centre=float(cos(pdir[i], PTM_C - c0[i])) if t1[i] < T1 - 1 else np.nan,
                                         s_stop=float(s_stop[i]), s_centre=float(s_c[i]), lateral=float(lat[i]),
                                         marker_r=float(np.linalg.norm(m[i] - PTM_C)), tcp_cube_d0=float(tcp_d0[i])))
                    if cond == "standard":
                        curves.append(dict(set=sset, cohort=cohort, seed=k, arm=arm, curve=(d / d[:, :1]).mean(0)))
    return pd.DataFrame(rows), curves


# ---------------------------------------------------------------- summaries

def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = x.mean() if len(x) else np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


def seed_table(ep, keys, metrics):
    """Per-seed means of each metric (raw table), then mean [95% t-CI] over seeds."""
    ps = ep.groupby(keys + ["seed"])[metrics].mean().reset_index()
    out = []
    for g, d in ps.groupby(keys):
        r = dict(zip(keys, g if isinstance(g, tuple) else (g,)))
        r["n"] = len(d)
        for mtr in metrics:
            m, lo, hi = tci(d[mtr])
            r[mtr] = f"{m:.3f} [{lo:.2f}, {hi:.2f}]"
        out.append(r)
    return ps, pd.DataFrame(out)


def strata(ep, by, q, keys, value):
    """Per-seed success rate within quantile bins of `by` (bins from the pooled GM h_pre episodes, same for all arms)."""
    edges = np.quantile(ep[by], np.linspace(0, 1, q + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = ep.assign(bin=pd.cut(ep[by], edges, labels=False))
    ps = e.groupby(keys + ["bin", "seed"])[value].mean().reset_index()
    rows = []
    for g, d in ps.groupby(keys + ["bin"]):
        m, lo, hi = tci(d[value])
        rows.append(dict(zip(keys + ["bin"], g), mean=m, lo=lo, hi=hi, seeds=" / ".join(f"{v:.3f}" for v in d[value])))
    return pd.DataFrame(rows), edges


def md_table(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) if not isinstance(v, float) else f"{v:.3f}" for v in r) + " |" for r in df.itertuples(index=False)])


def report(lr, ptm):
    L = ["# Location-task rewiring: what the weak recoveries look like (per-episode kinematics of the saved rollouts)", "",
         "Mean [95% t-CI] over seeds of per-seed means (LR seeds 1-3, 500 off-path episodes each; PTM seeds 0-2, 512 "
         "episodes per set). Raw per-episode metrics in LR_per_episode.csv / PTM_per_episode.csv, per-seed means in "
         "*_per_seed.csv.", ""]
    keep = [a for a in ARMS]
    lrs = lr[lr.cond == "standard"]
    lrs = lrs[[tuple(x) in keep for x in lrs[["cohort", "arm"]].itertuples(index=False)]]
    met = ["reach", "corner", "cos_head_goal", "cos_head_corner", "head_speed", "dmin", "d_final", "path_len", "mean_speed",
           "frac_steps_toward", "efficiency"]
    ps, t = seed_table(lrs, ["cohort", "arm"], met)
    ps.to_csv(os.path.join(OUT, "LR_per_seed.csv"), index=False)
    L += ["## LR, standard condition (first contact, 50 steps; distances in m, metrics before first contact)", "",
          "cos_head_* = cosine of the first 5-step TCP displacement with the direction to the goal / training corner; "
          "frac_steps_toward = share of steps that reduce the TCP-goal distance; efficiency = (d0 - dmin) / path length.", "",
          md_table(t), ""]
    succ = lrs[lrs.reach]
    ps2, t2 = seed_table(succ, ["cohort", "arm"], ["t_contact", "d0", "mean_speed", "efficiency"])
    L += ["Successful episodes only (t_contact = step of first marker contact):", "", md_table(t2), ""]
    st, edges = strata(lrs, "d0", 3, ["cohort", "arm"], "reach")
    st = st[st.arm.isin(["pre", "native"])]
    L += [f"Reach by start-to-goal distance tercile (edges {np.round(edges[1:-1], 3).tolist()} m):", "", md_table(st), ""]
    st2, edges2 = strata(lrs, "goal_corner_dist", 3, ["cohort", "arm"], "reach")
    st2 = st2[st2.arm.isin(["pre"])]
    L += [f"Reach by goal-to-corner distance tercile (edges {np.round(edges2[1:-1], 3).tolist()} m):", "", md_table(st2), ""]
    # distribution of the per-episode GM h_pre outcome: are failures near misses or wanderers
    g = lrs[(lrs.cohort == "gm") & (lrs.arm == "pre")]
    fail = g[~g.reach]
    nm = fail.groupby("seed").apply(lambda d: pd.Series(dict(lt6=np.mean(d.dmin < 0.06), lt10=np.mean(d.dmin < 0.10),
                                                            med=np.median(d.dmin))), include_groups=False)
    L += ["GM h_pre failures, closest approach to the goal (per seed): " +
          "; ".join(f"s{s}: <6 cm {r.lt6:.2f}, <10 cm {r.lt10:.2f}, median {r.med * 100:.1f} cm" for s, r in nm.iterrows()), ""]

    pts = ptm[(ptm.cond == "standard")]
    pts = pts[[tuple(x) in keep for x in pts[["cohort", "arm"]].itertuples(index=False)]]
    metp = ["goal", "d_final", "dmin", "d_centre_final", "cube_moved", "t_contact", "cos_push_marker", "cos_push_centre",
            "s_stop", "lateral"]
    ps3, t3 = seed_table(pts, ["set", "cohort", "arm"], metp)
    ps3.to_csv(os.path.join(OUT, "PTM_per_seed.csv"), index=False)
    L += ["## PTM, standard condition (final cube state at 100 steps)", "",
          "t_contact = first step the cube has moved 5 mm; cos_push_* = cosine of the cube's displacement at 1 cm with the "
          "direction to the marker / centre (in the conflict set both lie on the same inward ray, so only the ctrlset "
          "separates them); s_stop = final cube position projected on the start-cube -> marker line (1 = on the marker; "
          "conflict-set centre sits at s ~ 1.6-2.6); lateral = off-line distance (m).", "", md_table(t3), ""]
    st3, e3 = strata(pts[pts.set == "ctrlset"], "d0", 3, ["cohort", "arm"], "goal")
    st3 = st3[st3.arm.isin(["pre"])]
    L += [f"PTM ctrlset goal by push length (cube-to-marker distance) tercile (edges {np.round(e3[1:-1], 3).tolist()} m):", "",
          md_table(st3), ""]
    st4, e4 = strata(pts[pts.set == "ctrlset"], "marker_r", 3, ["cohort", "arm"], "goal")
    st4 = st4[st4.arm.isin(["pre"])]
    L += [f"PTM ctrlset goal by marker distance from the table centre tercile (edges {np.round(e4[1:-1], 3).tolist()} m):", "",
          md_table(st4), ""]
    # stop-short check: GM h_pre conflict standard vs nomarker vs decoupled
    sub = ptm[(ptm.set == "conflict") & (ptm.cohort.isin(["gm", "control"])) & (ptm.arm.isin(["pre", "fc_head"]))]
    ps5, t5 = seed_table(sub, ["cohort", "arm", "cond"], ["goal", "reach_drawn", "s_stop", "lateral", "d_final_drawn",
                                                           "cos_push_drawn"])
    L += ["Stop-short / marker-blind check on the conflict set (decoupled: marker drawn at m2, off the ray; scored at m; "
          "reach_drawn = cube ends on the drawn marker):", "", md_table(t5), ""]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    return "\n".join(L)


def figure(lr, lrc, ptm, ptc):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0))
    show = [("gm", "native"), ("gm", "pre"), ("control", "pre"), ("init", "pre"), ("gm", "pre_masked")]
    for key in show:
        col, ls = STYLE[key]
        cs = np.stack([c["curve"] for c in lrc if (c["cohort"], c["arm"]) == key])
        ax[0].plot(cs.mean(0), ls, color=col, lw=1.6, label=LAB[key])
        cs = np.stack([c["curve"] for c in ptc if (c["cohort"], c["arm"]) == key and c["set"] == "ctrlset"])
        ax[2].plot(cs.mean(0), ls, color=col, lw=1.6, label=LAB[key])
    ax[0].set(title="LR: TCP-goal distance / start\n(full 50 steps, not stopped at contact)", xlabel="step", ylim=(0, 1.6))
    ax[2].set(title="PTM ctrlset: cube-marker distance / start", xlabel="step", ylim=(0, 1.6))
    lrs = lr[lr.cond == "standard"]
    for j, key in enumerate([("gm", "pre"), ("control", "pre"), ("init", "pre"), ("gm", "pre_masked")]):
        st, edges = strata(lrs, "d0", 3, ["cohort", "arm"], "reach")
        s = st[(st.cohort == key[0]) & (st.arm == key[1])]
        col, _ = STYLE[key]
        ax[1].errorbar(s.bin + (j - 1.5) * 0.1, s["mean"], yerr=[s["mean"] - s.lo, s.hi - s["mean"]], fmt="o-", color=col,
                       ms=5, capsize=2, label=LAB[key])
    ax[1].set_xticks([0, 1, 2], ["near", "mid", "far"])
    ax[1].set(title="LR: reach by start-goal distance tercile", ylabel="goal reach", ylim=(-0.05, 1.05))
    sub = ptm[(ptm.set == "conflict") & (ptm.cohort == "gm") & (ptm.arm == "pre")]
    bins = np.linspace(-0.5, 4.0, 46)
    for cond, col, lab in (("standard", "#7A1520", "marker on ray (standard)"), ("nomarker", "#8E8E8E", "marker hidden"),
                           ("decoupled", "#E5A01A", "marker drawn off-ray")):
        ax[3].hist(sub[sub.cond == cond].s_stop.clip(-0.5, 4.0), bins=bins, histtype="step", lw=1.6, color=col, density=True,
                   label=lab)
    ax[3].axvline(1, color="#2DBE4E", lw=1)
    ax[3].axvline(sub.s_centre.median(), color="#F28E1C", lw=1, ls="--")
    ax[3].set(title="PTM conflict, GM h_pre: cube stop point", xlabel="along cube->marker line (1 = marker, dashed = centre, 4 = beyond)",
              ylabel="density")
    for j, a in enumerate(ax):
        a.legend(fontsize=6, frameon=False, loc="center right" if j == 1 else "best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_rewire_behaviour.{ext}"), dpi=200, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds_lr", type=int, nargs="+", default=list(LR_SEEDS))
    p.add_argument("--seeds_ptm", type=int, nargs="+", default=list(PTM_SEEDS))
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lr, lrc = lr_episodes(a.seeds_lr)
    lr.to_csv(os.path.join(OUT, "LR_per_episode.csv"), index=False)
    ptm, ptc = ptm_episodes(a.seeds_ptm)
    ptm.to_csv(os.path.join(OUT, "PTM_per_episode.csv"), index=False)
    print(report(lr, ptm))
    figure(lr, lrc, ptm, ptc)


if __name__ == "__main__":
    main()
