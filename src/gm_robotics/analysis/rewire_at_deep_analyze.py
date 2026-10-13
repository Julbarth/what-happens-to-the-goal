"""Summary of rewire_at_deep.py (AttributeTable rewiring): rates, gaps, chance, BC fit, probes, edit sensitivity."""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from gm_robotics.analysis import common as C

OUT = os.path.join(C.RESULTS, "rewire_at_deep")
NEAR = 0.03  # 'near miss': a 'neither' episode whose TCP came within 3 cm of the goal surface (touch = 1.5 cm)


def fmt(v, nd=3):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return "nan"
    m, lo, hi = C.ci95(v)
    return C.fmt_ci(m, lo, hi, nd) if len(v) > 1 else f"{m:.{nd}f}"


def seeds_str(v):
    return " / ".join("nan" if not np.isfinite(x) else f"{x:.3f}" for x in v)


def welch(a, b):
    """Mean difference a - b with a Welch 95% t-interval (independent cohorts, unequal n)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return f"{d:.3f} [{d - h:.3f}, {d + h:.3f}]"


def per_seed(ep):
    rows = []
    for key, g in ep.groupby(["variant", "cohort", "seed", "arm", "cond"]):
        r = dict(zip(["variant", "cohort", "seed", "arm", "cond"], key), n=len(g))
        for T in (50, 100):
            o = g[f"outcome_{T}"]
            pu, pv = np.mean(o == "u"), np.mean(o == "v")
            r[f"P_u_{T}"], r[f"P_v_{T}"], r[f"P_n_{T}"] = pu, pv, 1 - pu - pv
            r[f"P_u_dec_{T}"] = pu / (pu + pv) if pu + pv > 0 else np.nan
            nei = o == "neither"
            r[f"near_miss_{T}"] = np.mean(nei & (g[f"min_d_u_{T}"] < NEAR))
            r[f"near_miss_share_{T}"] = np.mean(g[f"min_d_u_{T}"][nei] < NEAR) if nei.any() else np.nan
        r["late_u"] = r["P_u_100"] - r["P_u_50"]
        rows.append(r)
    return pd.DataFrame(rows)


def md(df):
    rows = [list(map(str, df.columns))] + [list(map(str, r)) for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def get(ps, variant, cohort, arm, cond, col):
    s = ps[(ps.variant == variant) & (ps.cohort == cohort) & (ps.arm == arm) & (ps.cond == cond)].sort_values("seed")
    return s.set_index("seed")[col]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(OUT, "summary.md"))
    a = p.parse_args()
    ep = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "AT_deep_per_episode_*.csv")))],
                   ignore_index=True)
    ps = per_seed(ep)
    ps.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    L = ["# AttributeTable rewiring deep-dive (rewire_at_deep.py), mean [95% t-CI] over seeds (GM n=3, Non-GM n=4, init n=3)\n",
         "512 held-out conflict scenes, deterministic, T = 100 steps; T = 50 is the first 50 steps of the same rollout. "
         f"near miss = 'neither' with closest TCP approach < {NEAR * 100:.0f} cm of the goal surface.\n"]

    # reproduction of rewire_attr at T = 50
    ref_f = os.path.join(C.RESULTS, "rewire_attr", "AT_per_episode.csv")
    if os.path.exists(ref_f):
        ref = pd.read_csv(ref_f)
        m = ep[ep.variant == "base"].merge(ref, on=["cohort", "seed", "arm", "cond", "episode"])
        if len(m):
            L.append(f"Reproduction: base heads, outcome at 50 steps vs rewire_attr AT_per_episode.csv: agreement "
                     f"{np.mean(m.outcome_50 == m.outcome):.4f} over {len(m)} episodes "
                     f"({m.groupby(['cohort', 'seed', 'arm', 'cond']).ngroups} runs).\n")

    # chance
    ch = ps[ps.variant == "chance"]
    if len(ch):
        L.append("## Chance / scripted reference policies (standard conflict unless noted)\n")
        t = ch[["arm", "cond", "P_u_50", "P_v_50", "P_u_100", "P_v_100", "P_n_100"]].round(3)
        L.append(md(t) + "\n")
        cu = ch[(ch.arm == "oracle_u") & (ch.cond == "standard")]
        cv = ch[(ch.arm == "oracle_v") & (ch.cond == "standard")]
        if len(cu) and len(cv):
            for T in (50, 100):
                L.append(f"- random-object commit policy (oracle to u or v w.p. 1/2), T={T}: P(goal) = "
                         f"{0.5 * cu[f'P_u_{T}'].iloc[0] + 0.5 * cv[f'P_u_{T}'].iloc[0]:.3f}, P(proxy) = "
                         f"{0.5 * cu[f'P_v_{T}'].iloc[0] + 0.5 * cv[f'P_v_{T}'].iloc[0]:.3f}")
        L.append("")

    # main table
    L.append("## Goal reach by variant / arm (standard conflict)\n")
    rows = []
    st = ps[(ps.cond == "standard") & (ps.variant != "chance")]
    for (variant, arm, cohort), g in st.groupby(["variant", "arm", "cohort"]):
        g = g.sort_values("seed")
        rows.append(dict(variant=variant, arm=arm, cohort=cohort, n=len(g), goal_50=fmt(g.P_u_50), goal_100=fmt(g.P_u_100),
                         proxy_100=fmt(g.P_v_100), neither_100=fmt(g.P_n_100), P_dec_50=fmt(g.P_u_dec_50),
                         P_dec_100=fmt(g.P_u_dec_100), near_miss_50=fmt(g.near_miss_50), near_miss_100=fmt(g.near_miss_100),
                         per_seed_goal_100=seeds_str(g.P_u_100)))
    L.append(md(pd.DataFrame(rows)) + "\n")

    # gaps
    L.append("## Non-GM minus GM gap, h_pre rewire (Welch 95% t-CI, independent cohorts)\n")
    rows = []
    for variant in sorted(st.variant.unique()):
        for arm in ("pre", "fc_head", "pre_masked"):
            g, c = get(ps, variant, "gm", arm, "standard", "P_u_100"), get(ps, variant, "control", arm, "standard", "P_u_100")
            if len(g) < 2 or len(c) < 2:
                continue
            r = dict(variant=variant, arm=arm)
            for T in (50, 100):
                for col in ("P_u", "P_u_dec"):
                    r[f"gap_{col}_{T}"] = welch(get(ps, variant, "control", arm, "standard", f"{col}_{T}"),
                                                get(ps, variant, "gm", arm, "standard", f"{col}_{T}"))
            rows.append(r)
    L.append(md(pd.DataFrame(rows)) + "\n")

    # paired within-cohort contrasts
    L.append("## Paired per-seed contrasts (same trunk seed), T=100 and T=50\n")
    rows = []
    for cohort in ("gm", "control", "init"):
        for variant in sorted(st.variant.unique()):
            for T in (50, 100):
                col = f"P_u_{T}"
                pre = get(ps, variant, cohort, "pre", "standard", col)
                cands = [("pre - pre_masked", get(ps, variant, cohort, "pre_masked", "standard", col)),
                         ("pre - remix", get(ps, variant, cohort, "remix", "standard", col)),
                         ("pre - fc_head", get(ps, variant, cohort, "fc_head", "standard", col))]
                if variant != "base":
                    cands.append(("pre - base pre", get(ps, "base", cohort, "pre", "standard", col)))
                for name, other in cands:
                    d = (pre - other).dropna()
                    if len(d) >= 2:
                        rows.append(dict(cohort=cohort, variant=variant, T=T, contrast=name, value=fmt(d), per_seed=seeds_str(d)))
                # residual proxy pull: P(u|dec) without the red cue minus standard
                for cond in ("noproxy_blue", "noproxy_red", "nogoal_sph", "nogoal_cube"):
                    x = get(ps, variant, cohort, "pre", cond, f"P_u_dec_{T}")
                    d = (x - get(ps, variant, cohort, "pre", "standard", f"P_u_dec_{T}")).dropna()
                    if len(d) >= 2:
                        rows.append(dict(cohort=cohort, variant=variant, T=T, contrast=f"P(u|dec) {cond} - standard (pre)",
                                         value=fmt(d), per_seed=seeds_str(d)))
    L.append(md(pd.DataFrame(rows)) + "\n")

    # decoupled conditions
    L.append("## Cue-decoupled conditions, pre arm, P(u | decisive) and goal reach at T=100\n")
    rows = []
    dec = ps[(ps.arm.isin(["pre", "native"])) & (ps.variant != "chance")]
    for (variant, arm, cohort, cond), g in dec.groupby(["variant", "arm", "cohort", "cond"]):
        rows.append(dict(variant=variant, arm=arm, cohort=cohort, cond=cond, P_dec_100=fmt(g.P_u_dec_100),
                         goal_100=fmt(g.P_u_100), proxy_100=fmt(g.P_v_100), per_seed_P_dec=seeds_str(g.sort_values("seed").P_u_dec_100)))
    L.append(md(pd.DataFrame(rows)) + "\n")

    # BC fit
    tl = []
    for f in sorted(glob.glob(os.path.join(OUT, "trainlogs", "*.csv"))) + \
            sorted(glob.glob(os.path.join(C.RESULTS, "rewire_attr", "trainlogs", "AT_*.csv"))):
        name = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        if "rewire_attr" in f:
            _, cohort, s, *arm = name.split("_")
            variant = "base"
        else:
            parts = name.split("_")
            i = next(j for j, x in enumerate(parts) if x in ("gm", "control", "init"))
            variant, cohort, s, arm = "_".join(parts[:i]), parts[i], parts[i + 1], parts[i + 2:]
        tl.append(dict(variant=variant, cohort=cohort, seed=int(s[1:]), arm="_".join(arm), final_train=d.train_loss.iloc[-1],
                       final_val=d.val_loss.iloc[-1], min_val=d.val_loss.min(), argmin_epoch=int(d.val_loss.idxmin()),
                       epochs=len(d), n_frames=d.get("n_frames", pd.Series([np.nan])).iloc[0]))
    tl = pd.DataFrame(tl)
    if len(tl):
        tl.to_csv(os.path.join(OUT, "bc_fit_per_seed.csv"), index=False)
        L.append("## BC fit (sigma-weighted MSE to the labels; 80/20 split by episode)\n")
        rows = []
        for (variant, arm, cohort), g in tl.groupby(["variant", "arm", "cohort"]):
            rows.append(dict(variant=variant, arm=arm, cohort=cohort, n=len(g), train=fmt(g.final_train), val=fmt(g.final_val),
                             min_val=fmt(g.min_val), argmin_epoch=seeds_str(g.argmin_epoch.astype(float)), frames=int(g.n_frames.iloc[0])
                             if np.isfinite(g.n_frames.iloc[0]) else ""))
        L.append(md(pd.DataFrame(rows)) + "\n")

    hf = os.path.join(OUT, "AT_heldout_bank2_loss.csv")
    if os.path.exists(hf):
        h = pd.read_csv(hf)
        L.append("## Loss on bank 2 (fully held out, teacher labels) by frame category, bank-1 heads\n")
        rows = []
        for (variant, arm, cat, cohort), g in h.groupby(["variant", "arm", "category", "cohort"]):
            rows.append(dict(variant=variant, arm=arm, category=cat, cohort=cohort, loss=fmt(g.heldout_loss), n_frames=int(g.n.iloc[0])))
        L.append(md(pd.DataFrame(rows)) + "\n")

    tg = os.path.join(OUT, "AT_teacher_label_geometry.csv")
    if os.path.exists(tg):
        t = pd.read_csv(tg)
        t = t[t.bank == 1]
        L.append("## Teacher labels on bank 1: cosine of the teacher's first-order TCP step with the direction to the cube / lure, "
                 "and with the oracle's step (mean over the four teachers)\n")
        g = t.groupby(["source", "scene", "region"]).agg(n=("n", "mean"), frac_of_source=("frac_of_source", "mean"),
                                                          cos_cube=("cos_teacher_to_cube", "mean"), cos_lure=("cos_teacher_to_lure", "mean"),
                                                          cos_oracle=("cos_teacher_oracle", "mean"), teacher_cm=("teacher_speed_cm", "mean"),
                                                          oracle_cm=("oracle_speed_cm", "mean")).reset_index().round(3)
        L.append(md(g) + "\n")

    pf = os.path.join(OUT, "probes_goal_proxy.csv")
    if os.path.exists(pf):
        pr = pd.read_csv(pf)
        L.append("## Probes on BC-bank frames of control-distribution scenes (episode-split 60/20/20; R2 for xy, AUC for left)\n")
        rows = []
        for (env, site, target), g in pr.groupby(["env", "site", "target"]):
            r = dict(env=env, site=site, target=target)
            for kind in ("linear", "mlp"):
                for cohort in ("gm", "control", "init"):
                    r[f"{kind}_{cohort}"] = fmt(g[g.cohort == cohort][kind])
                gi = g[g.cohort == "init"][kind].mean()
                r[f"{kind}_gm_minus_init"] = f"{g[g.cohort == 'gm'][kind].mean() - gi:.3f}"
            rows.append(r)
        L.append(md(pd.DataFrame(rows)) + "\n")

    sf = os.path.join(OUT, "AT_edit_sensitivity.csv")
    if os.path.exists(sf):
        s = pd.read_csv(sf)
        L.append("## Edit sensitivity of the heads on the 512 world-edit start frames (KL with the teacher sigma, median over "
                 "states; follow = first-order TCP step change toward where the edited cue now sits, mm)\n")
        rows = []
        for (variant, arm, cohort), g in s.groupby(["variant", "arm", "cohort"]):
            r = dict(variant=variant, arm=arm, cohort=cohort)
            piv = g.pivot(index="seed", columns="edit", values="kl_median")
            fol = g.pivot(index="seed", columns="edit", values="follow_mm_mean")
            for e in ("shape_swap", "colour_swap", "shape_ablate", "colour_ablate"):
                r[f"KL_{e}"] = fmt(piv[e], 2)
                r[f"follow_{e}"] = fmt(fol[e], 2)
            r["KL colour/shape swap"] = fmt(piv.colour_swap / piv.shape_swap, 2)
            rows.append(r)
        L.append(md(pd.DataFrame(rows)) + "\n")

    open(a.out, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
