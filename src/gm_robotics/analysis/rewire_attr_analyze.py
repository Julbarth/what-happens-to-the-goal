"""Summary of rewire_attr rollouts: goal / proxy / neither rates, P(goal | decisive), cue-decoupled contrasts, leakage."""
import argparse
import os

import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C

OUT = os.path.join(C.RESULTS, "rewire_attr")
ARM_ORDER = ["native", "fc_head", "pre", "remix", "untrained", "pre_masked", "rand_walk"]


def per_seed(ep):
    g = ep.groupby(["env", "cohort", "seed", "arm", "cond"])
    ps = g.outcome.agg(P_u=lambda o: np.mean(o == "u"), P_v=lambda o: np.mean(o == "v"),
                       P_neither=lambda o: np.mean(o == "neither"), n=len).reset_index()
    ps["D"] = ps.P_u + ps.P_v
    ps["P_u_dec"] = np.where(ps.D > 0, ps.P_u / ps.D.where(ps.D > 0, 1), np.nan)
    if "env_outcome" in ep:
        agree = ep[ep.env_outcome.notna()].assign(a=lambda d: d.outcome == d.env_outcome)
        ps = ps.merge(agree.groupby(["env", "cohort", "seed", "arm", "cond"]).a.mean().rename("env_agree").reset_index(),
                      how="left")
    return ps


def summarize(ps, cols):
    rows = []
    for key, g in ps.groupby(["env", "cohort", "arm", "cond"]):
        g = g.sort_values("seed")
        r = dict(zip(["env", "cohort", "arm", "cond"], key), n_seeds=len(g))
        for c in cols:
            v = g[c].dropna().to_numpy()
            m, lo, hi = C.ci95(v) if len(v) else (np.nan, np.nan, np.nan)
            r[c], r[c + "_lo"], r[c + "_hi"] = m, lo, hi
            r[c + "_seeds"] = " / ".join("nan" if not np.isfinite(x) else f"{x:.3f}" for x in g[c])
        rows.append(r)
    return pd.DataFrame(rows)


def paired(ps, env, a, b, col_a, col_b=None, pair_on="seed"):
    """Per-seed difference a - b (dicts of filters), mean [95% t-CI] and per-seed values."""
    col_b = col_b or col_a
    sel = lambda f: ps[(ps.env == env) & np.logical_and.reduce([ps[k] == v for k, v in f.items()])].set_index(pair_on)
    x, y = sel(a)[col_a], sel(b)[col_b]
    d = (x - y).dropna().sort_index()
    m, lo, hi = C.ci95(d.to_numpy())
    return dict(contrast_mean=m, lo=lo, hi=hi, per_seed=" / ".join(f"{v:.3f}" for v in d), n=len(d))


def strict_leak(env, ep, tol=(0.01, 0.02)):
    """Nearest BC episode showing the conflict cue layout itself (AT: blue cube + red sphere; SZ: cross on the purple
    pad, crossed pad on the same side), distance = max over candidate locations (and SZ cube xy). Returns per-scene
    distances and the GM / Non-GM h_pre and native reach on scenes with no such BC scene within each tolerance."""
    b = np.load(os.path.join(OUT, f"{env}_bc_cfg.npz"))
    w = C.load_bank(f"{env}_worldedit")
    if env == "AT":
        lay = ~b["cfg_cube_red"] & b["cfg_sphere_red"]
    else:
        lay = ~b["cfg_cross_green"]
    bu, bv = b["cfg_u"][lay], b["cfg_v"][lay]
    d = np.maximum(np.linalg.norm(bu[None] - w["u"][:, None], axis=-1), np.linalg.norm(bv[None] - w["v"][:, None], axis=-1))
    if env == "SZ":
        d = np.maximum(d, np.linalg.norm(b["cfg_cube"][lay][None, :, :2] - w["cube"][:, None, :2], axis=-1))
    nn = d.min(1)
    rows = [dict(env=env, n_bc_same_layout=int(lay.sum()), frac_within_1cm=float(np.mean(nn < 0.01)),
                 frac_within_2cm=float(np.mean(nn < 0.02)), nn_median=float(np.median(nn)))]
    st = ep[(ep.env == env) & (ep.cond == "standard") & ep.arm.isin(["pre", "native", "fc_head"]) & ep.cohort.isin(["gm", "control"])]
    for t in tol:
        keep = np.nonzero(nn >= t)[0]
        sub = st[st.episode.isin(keep)]
        r = sub.groupby(["cohort", "arm", "seed"]).outcome.apply(lambda o: np.mean(o == "u")).groupby(["cohort", "arm"]).agg(list)
        for (coh, arm), v in r.items():
            m, lo, hi = C.ci95(np.array(v))
            rows.append(dict(env=env, tol=t, n_scenes_kept=len(keep), cohort=coh, arm=arm, goal_reach=C.fmt_ci(m, lo, hi),
                             per_seed=" / ".join(f"{x:.3f}" for x in v)))
    return pd.DataFrame(rows), nn


def md(df):
    rows = [list(map(str, df.columns))] + [list(map(str, r)) for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=["AT", "SZ"])
    a = p.parse_args()
    eps = [pd.read_csv(os.path.join(OUT, f"{k}_per_episode.csv")) for k in a.envs
           if os.path.exists(os.path.join(OUT, f"{k}_per_episode.csv"))]
    ep = pd.concat(eps, ignore_index=True)
    ps = per_seed(ep)
    tl = []
    for f in sorted(os.listdir(os.path.join(OUT, "trainlogs"))):
        k, cohort, s, *arm = f[:-4].split("_")
        d = pd.read_csv(os.path.join(OUT, "trainlogs", f))
        tl.append(dict(env=k, cohort=cohort, seed=int(s[1:]), arm="_".join(arm), final_val_loss=d.val_loss.iloc[-1],
                       final_train_loss=d.train_loss.iloc[-1], n_trainable=int(d.n_trainable.iloc[0])))
    tl = pd.DataFrame(tl)
    ps = ps.merge(tl, how="left", on=["env", "cohort", "seed", "arm"])
    ps.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    cols = ["P_u", "P_v", "P_neither", "P_u_dec", "final_val_loss"]
    sm = summarize(ps, cols)
    sm["arm"] = pd.Categorical(sm.arm, ARM_ORDER, ordered=True)
    sm = sm.sort_values(["env", "cond", "arm", "cohort"])
    sm.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    con = []
    for env in a.envs:
        nog = ["nogoal_sph", "nogoal_cube"] if env == "AT" else ["nogoal"]
        nop = ["noproxy_blue", "noproxy_red"] if env == "AT" else ["noproxy_green", "noproxy_purple"]
        for coh in ("gm", "control", "init"):
            base = dict(cohort=coh, arm="pre", cond="standard")
            if coh != "init":
                con.append(dict(env=env, cohort=coh, contrast="pre - pre_masked, goal reach (standard)",
                                **paired(ps, env, base, dict(base, arm="pre_masked"), "P_u")))
            for c in nog + nop:
                con.append(dict(env=env, cohort=coh, contrast=f"P(u|dec) standard - {c} (pre)",
                                **paired(ps, env, base, dict(base, cond=c), "P_u_dec")))
            con.append(dict(env=env, cohort=coh, contrast="pre - fc_head, goal reach (standard)",
                            **paired(ps, env, base, dict(base, arm="fc_head"), "P_u")))
            con.append(dict(env=env, cohort=coh, contrast="pre - remix, goal reach (standard)",
                            **paired(ps, env, base, dict(base, arm="remix"), "P_u")))
        con.append(dict(env=env, cohort="gm", contrast="GM pre - init-trunk pre, goal reach (standard)",
                        **paired(ps, env, dict(cohort="gm", arm="pre", cond="standard"),
                                 dict(cohort="init", arm="pre", cond="standard"), "P_u")))
        con.append(dict(env=env, cohort="gm", contrast="GM pre - init-trunk pre, P(u|dec) (standard)",
                        **paired(ps, env, dict(cohort="gm", arm="pre", cond="standard"),
                                 dict(cohort="init", arm="pre", cond="standard"), "P_u_dec")))
    con = pd.DataFrame(con)
    con.to_csv(os.path.join(OUT, "contrasts.csv"), index=False)

    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("# AttributeTable / SortZones rewiring (rewire_lr protocol), mean [95% t-CI] over seeds\n\n"
                 "u = goal location of the base conflict scene (AT: blue cube, SZ: crossed purple pad), v = proxy location "
                 "(AT: red sphere, SZ: green pad). Cue-decoupled conditions keep u, v and the start; see rewire_attr.CONDS. "
                 "Deterministic actions, first touch (AT, 50 steps) / first pad entry (SZ, 100 steps), 512 held-out scenes.\n\n")
        for env in a.envs:
            s = sm[sm.env == env]
            t = s.assign(**{c: [C.fmt_ci(m, lo, hi) for m, lo, hi in zip(s[c], s[c + "_lo"], s[c + "_hi"])]
                            for c in ["P_u", "P_v", "P_neither", "P_u_dec"]})
            fh.write(f"## {env}\n\n" + md(t[["cond", "arm", "cohort", "n_seeds", "P_u", "P_v", "P_neither", "P_u_dec"]]) + "\n\n")
            fh.write("Per seed (P_u / P_u_dec / final BC val loss):\n\n" +
                     md(s[["cond", "arm", "cohort", "P_u_seeds", "P_u_dec_seeds", "final_val_loss_seeds"]]) + "\n\n")
            c = con[con.env == env]
            fh.write(f"### {env} paired contrasts (per-seed differences)\n\n" + md(c.assign(
                value=[C.fmt_ci(m, lo, hi) for m, lo, hi in zip(c.contrast_mean, c.lo, c.hi)])[
                ["cohort", "contrast", "n", "value", "per_seed"]]) + "\n\n")
            sl, _ = strict_leak(env, ep)
            sl.to_csv(os.path.join(OUT, f"{env}_leakage_strict.csv"), index=False)
            fh.write(f"### {env} same-cue-layout leakage and reach on scenes without a close BC match\n\n" + md(sl.fillna("")) + "\n\n")
            lf = os.path.join(OUT, f"{env}_leakage.csv")
            if os.path.exists(lf):
                lk = pd.read_csv(lf)
                fh.write(f"### {env} leakage\n\nnearest BC scene (both candidate locations, either role): median "
                         f"{lk.nn_loc_dist.median():.4f} m, min {lk.nn_loc_dist.min():.4f} m; eval scenes with a BC scene within "
                         f"1 cm at both locations: {np.mean(lk.nn_loc_dist < 0.01):.4f}; min start-qpos distance to any BC "
                         f"start: median {lk.min_qpos_dist.median():.4f} rad; eval start frames pixel-identical to any BC frame: "
                         f"{lk.start_frame_pixel_identical_to_bank.mean():.4f}\n\n")
    print(open(os.path.join(OUT, "summary.md")).read())


if __name__ == "__main__":
    main()
