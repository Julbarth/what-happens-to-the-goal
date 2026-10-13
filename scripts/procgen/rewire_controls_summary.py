"""Cross-environment rewire reference levels (native, chance, random trunk, GM, Non-GM) and normalized recovery."""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from gm_robotics.paths import ROOT as RUN_ROOT

ROOT = str(RUN_ROOT)
RA = os.path.join(ROOT, "results", "analysis")
OUT = os.path.join(RA, "rewire_controls")
REF = os.path.join(OUT, "paper_ref")
RAW = os.path.join(OUT, "raw")


def tci(v):
    v = np.asarray([x for x in np.atleast_1d(v) if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return v[0], np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def _mv(x):
    x = np.atleast_1d(np.asarray(x, float))
    return x.mean(), (x.var(ddof=1) / len(x) if len(x) > 1 else 0.0), len(x)


def welch(a, b):
    """mean(a) - mean(b) with a Welch t-interval; a constant b (one value) contributes no variance."""
    ma, va, na = _mv(a)
    mb, vb, nb = _mv(b)
    se2 = va + vb
    if se2 == 0:
        return ma - mb, np.nan, np.nan
    df = se2 ** 2 / sum(v ** 2 / (n - 1) for v, n in ((va, na), (vb, nb)) if n > 1 and v > 0)
    h = stats.t.ppf(0.975, df) * np.sqrt(se2)
    return ma - mb, ma - mb - h, ma - mb + h


def norm_ci(g, n, f):
    """(mean g - mean f) / (mean n - mean f), delta-method SE over three independent groups, Welch-Satterthwaite df."""
    (mg, vg, ng), (mn, vn, nn), (mf, vf, nf) = _mv(g), _mv(n), _mv(f)
    d = mn - mf
    r = (mg - mf) / d
    terms = [((1 / d) ** 2 * vg, ng), (((mg - mf) / d ** 2) ** 2 * vn, nn), (((mg - mn) / d ** 2) ** 2 * vf, nf)]
    se2 = sum(t for t, _ in terms)
    den = sum(t ** 2 / (k - 1) for t, k in terms if k > 1 and t > 0)
    if se2 == 0 or den == 0:
        return r, np.nan, np.nan
    h = stats.t.ppf(0.975, se2 ** 2 / den) * np.sqrt(se2)
    return r, r - h, r + h


def jl(p):
    return json.load(open(p))


# ------------------------------------------------------------------ per-environment loaders (per-seed arrays)

def maze():
    """Rebuild of the paper's aligned bank (a_star labels, h_pre): one bank for GM, Non-GM and random trunks."""
    dd = os.path.join(RA, "deepdive_followups")
    g = pd.read_csv(os.path.join(dd, "runA_maze_rewire_per_seed.csv"))
    g = g[(g.labels == "a_star") & (g.arm == "rewire")]
    s = pd.read_csv(os.path.join(dd, "runA_sampled_per_seed.csv"))
    s = s[s.arm.isin(["rewire", "uniform"])]
    fc = jl(os.path.join(REF, "maze_aligned_fc_a_star.json"))["rows"]
    pick = lambda df, c, col: df[df.cohort == c].sort_values("trunk")[col].to_numpy()
    greedy = dict(native_gm=pick(g, "gm", "native_reach"), native_nongm=pick(g, "nongm", "native_reach"),
                  gm=pick(g, "gm", "reach"), nongm=pick(g, "nongm", "reach"), init=pick(g, "init", "reach"),
                  gm_fc=np.array([fc[f"gm_rewire_{k}"]["rewired_eval"]["cheese_reach"] for k in ("1080_gm", "1081_gm", "1082_gm")]),
                  chance=pick(s, "chance", "reach"),
                  init_val=pick(g, "init", "val_ce"), gm_val=pick(g, "gm", "val_ce"), nongm_val=pick(g, "nongm", "val_ce"))
    sampled = dict(native_gm=greedy["native_gm"], native_nongm=greedy["native_nongm"],
                   gm=pick(s, "gm", "reach"), nongm=pick(s, "nongm", "reach"), init=pick(s, "init", "reach"),
                   chance=pick(s, "chance", "reach"))
    chance_def = ("uniform random walk over the 15 procgen actions, sampled (Run A, 5 eval seeds x 300 levels); "
                  "no competent goal-blind policy is definable without a second target")
    return [("Maze", "greedy (paper)", greedy, chance_def, "random walk is a sampled policy; all greedy arms are deterministic"),
            ("Maze", "sampled", sampled, chance_def, "")]


def coin():
    rows = jl(os.path.join(REF, "coin_block3_teacher.json"))["rows"]
    rfc = jl(os.path.join(REF, "coin_fc_teacher.json"))["rows"]
    fl = jl(os.path.join(REF, "untrained_floor_coin.json"))
    gmk, ngk = ("6033_gm_20M", "6037_gm_120M", "6051_model_55050240"), ("7001_teach", "7002_teach", "7003_teach")
    sr = lambda d, pre, ks: np.array([d[f"{pre}_{k}"]["success_rate"] for k in ks])
    rand = [os.path.join(RAW, "coin", f"rand{k}") for k in range(3)]
    gr = np.array([jl(os.path.join(r, "eval_greedy.json"))["success_rate"] for r in rand])
    smp = np.array([np.mean([jl(f)["success_rate"] for f in sorted(glob.glob(os.path.join(r, "eval_sample_es*.json")))])
                    for r in rand])
    val = np.array([jl(os.path.join(r, "train.json"))["final_val_ce"] for r in rand])
    base = dict(native_gm=sr(rows, "native_gm", gmk), native_nongm=sr(rows, "native_nongm", ngk),
                gm=sr(rows, "gm_rewire", gmk), nongm=sr(rows, "nongm_rewire", ngk), gm_fc=sr(rfc, "gm_rewire", gmk),
                chance=np.array([fl["block3_sample_gm_mean"], fl["block3_sample_nongm_mean"]]),
                init_val=val, gm_val=np.array([rows[f"gm_rewire_{k}"]["final_val_ce"] for k in gmk]),
                nongm_val=np.array([rows[f"nongm_rewire_{k}"]["final_val_ce"] for k in ngk]))
    chance_def = ("untrained head sampled (~uniform over 15 actions), paper floor, GM 6033 and Non-GM 7001 trunks; "
                  "the competent goal-blind policy (GM native, runs to the level end) reaches the coin less often")
    return [("Coin", "greedy (paper)", dict(base, init=gr), chance_def, ""),
            ("Coin", "random trunk sampled", dict(base, init=smp), chance_def,
             "only the random trunk is sampled here; GM / Non-GM stay greedy")]


def am():
    ps = pd.read_csv(os.path.join(RA, "am_oracle_fix", "per_seed.csv"))
    ps = ps[(ps.labels == "fixed") & (ps.cohort == "black") & (ps.stratum == "overall") & (ps.arm == "rewire")]
    rows = jl(os.path.join(REF, "ras_black_block3_oracle.json"))["rows"]
    nat = lambda pre: np.array([r["R_goal"] for r in rows if r["label"].startswith(pre)])
    pick = lambda tap, role, col="R_goal": ps[(ps.tap == tap) & (ps.role == role)].sort_values("seed")[col].to_numpy()
    rand = [os.path.join(RAW, "am", f"rand{k}") for k in range(3)]
    gr = [jl(os.path.join(r, "eval_greedy.json")) for r in rand]
    sm = [[jl(f) for f in sorted(glob.glob(os.path.join(r, "eval_sample_es*.json")))] for r in rand]
    tr = lambda role, seeds: [[jl(f) for f in sorted(glob.glob(os.path.join(RAW, "am_trained", f"{role}_rewire_{s}",
                                                                          "eval_sample_es*.json")))] for s in seeds]
    rep = lambda role, seeds, k="R_goal": np.array([jl(os.path.join(RAW, "am_repro", f"{role}_rewire_{s}", "summary.json"))[k]
                                                   for s in seeds])
    smean = lambda L, k: np.array([np.mean([e[k] for e in x]) for x in L])
    G, N = (6313, 6315, 6317), (4371, 4372, 4373)
    gtr, ntr = tr("gm", G), tr("nongm", N)
    init_D = np.array([e["decisive_frac"] for e in gr])
    common = dict(native_gm=nat("native_gm"), native_nongm=nat("native_teacher"), gm_fc=pick("fc", "gm"))
    chance_def = ("coin flip between the two objects at the GM h_pre rewire's decisiveness, 0.5 x D (the sampled "
                  "untrained head gives 0.451 with D = 0.893, P(goal|dec) = 0.505, i.e. the same coin flip)")
    out = [("Attribute Maze (black, fixed oracle)", "greedy (paper), rebuilt bank: all arms on one bank",
            dict(common, gm=rep("gm", G), nongm=rep("nongm", N), init=np.array([e["R_goal"] for e in gr]),
                 chance=0.5 * rep("gm", G, "decisive_frac"), init_D=init_D, init_P=np.array([e["pref_goal"] for e in gr])),
            chance_def, "GM / Non-GM retrained on the rebuilt fixed-oracle bank (merge gate within tolerance, 5/6 drivers exact)"),
           ("Attribute Maze (black, fixed oracle)", "greedy, am_oracle_fix GM / Non-GM heads",
            dict(common, gm=pick("block3", "gm"), nongm=pick("block3", "nongm"), init=np.array([e["R_goal"] for e in gr]),
                 chance=0.5 * pick("block3", "gm", "D")), chance_def, "random trunk from the rebuilt bank"),
           ("Attribute Maze (black, fixed oracle)", "sampled",
            dict(common, gm=smean(gtr, "R_goal"), nongm=smean(ntr, "R_goal"), init=smean(sm, "R_goal"),
                 chance=0.5 * smean(gtr, "decisive_frac"), init_D=smean(sm, "decisive_frac"), init_P=smean(sm, "pref_goal")),
            chance_def, "GM / Non-GM = am_oracle_fix heads re-evaluated sampled; 3 eval seeds averaged per trunk")]
    dj = os.path.join(RAW, "am_disjoint")
    if all(os.path.exists(os.path.join(dj, f"{t}.csv")) for t in ["rand0", "rand1", "rand2"] + [f"gm_{s}" for s in G] + [f"nongm_{s}" for s in N]):
        rg = lambda t: pd.read_csv(os.path.join(dj, f"{t}.csv")).reward.to_numpy()
        R = lambda ts: np.array([np.mean(rg(t) == 10) for t in ts])
        Dd = lambda ts: np.array([np.mean(rg(t) != 0) for t in ts])
        d = dict(common, gm=R([f"gm_{s}" for s in G]), nongm=R([f"nongm_{s}" for s in N]), init=R(["rand0", "rand1", "rand2"]),
                 chance=0.5 * Dd([f"gm_{s}" for s in G]))
        if all(os.path.exists(os.path.join(dj, f"paper_gm_{s}.csv")) for s in G):
            d["gm_paperheads"] = R([f"paper_gm_{s}" for s in G])
            d["nongm_paperheads"] = R([f"paper_nongm_{s}" for s in N])
        n = len(rg("rand0"))
        out.append(("Attribute Maze (black, fixed oracle)", f"greedy, state-disjoint fresh levels (n = {n} start frames absent from the bank)",
                    d, chance_def, "rebuilt-bank heads"))
    return out


def robots():
    ps = pd.read_csv(os.path.join(RA, "rewire_all_per_seed.csv"))
    v = lambda t, k: ps[(ps.task == t) & (ps.key == k)].sort_values("seed").value.to_numpy()
    out = []
    defs = {"AT": "coin flip between cube and sphere at the GM h_pre rewire's decisiveness, 0.5 x P(touched either)",
            "SZ": "coin flip between the two pads at the GM h_pre rewire's decisiveness, 0.5 x P(cube in either pad)",
            "LR": "phantom-goal reach of the GM h_pre rewire's own paths (K = 20 pooled phantom goals per episode)",
            "PTM (conflict)": "phantom-goal reach (pool) of the GM h_pre rollouts; GM native (competent, marker-blind) 0.000",
            "PTM (ctrlset)": "phantom-goal reach (pool) of the GM h_pre rollouts; GM native (competent, marker-blind) 0.020"}
    names = {"AT": "AttributeTable", "SZ": "SortZones", "LR": "LocationReach", "PTM (conflict)": "PushToMarker (conflict)",
             "PTM (ctrlset)": "PushToMarker (ctrlset)"}
    for t in ("LR", "PTM (conflict)", "PTM (ctrlset)", "AT", "SZ"):
        d = dict(native_gm=v(t, "native_gm"), gm=v(t, "pre_gm"), nongm=v(t, "pre_control"), init=v(t, "pre_init"),
                 gm_fc=v(t, "fc_gm"), chance=v(t, "chance"))
        note = "init trunk BC fit failed (val loss 6.7-17); see the refit rows" if t.startswith("PTM") else ""
        out.append((names[t], "deterministic (paper)", d, defs[t], note))
    return out


def ptm_refit():
    """PTM rows from rewire_controls_ptm.py (rebuilt BC bank): paper protocol, RMS-scaled 20 and 100 epochs."""
    f = os.path.join(RAW, "ptm", "per_episode.csv")
    if not os.path.exists(f):
        return []
    df = pd.read_csv(f)
    g = df.assign(goal=(df.outcome == "goal").astype(float)).groupby(["cohort", "seed", "variant", "set"]).goal.mean().reset_index()
    out = []
    for sname in ("conflict", "ctrlset"):
        x = g[g.set == sname]
        val = lambda c, var: x[(x.cohort == c) & (x.variant == var)].sort_values("seed").goal.to_numpy()
        for var in ("paper", "s20", "s100"):
            d = dict(native_gm=val("gm", "native"), gm=val("gm", var), nongm=val("control", var), init=val("init", var),
                     chance=val("gm", "native"), state=val("init", "state100"))
            if len(d["gm"]) and len(d["init"]):
                out.append((f"PushToMarker ({sname}), rebuilt bank", "deterministic, " + var, d,
                            "competent marker-blind pusher = GM native on the same scenes",
                            f"{var}: same protocol for all three cohorts; proprio-only readout (init trunk, 100 epochs) "
                            f"{np.mean(d['state']):.3f}"))
    return out


def isaac():
    p = os.path.join(ROOT, "results", "isaac_analysis", "rewire", "rewire_per_seed.csv")
    d = pd.read_csv(p)
    d = d[(d.set == "native")]
    v = lambda c, arm, tgt, col="goal": d[(d.cohort == c) & (d.arm == arm) & (d.target == tgt)].sort_values("seed")[col].to_numpy()
    gm_pre = d[(d.cohort == "gm") & (d.arm == "pre") & (d.target == "teacher")].sort_values("seed")
    chance = 0.5 * (gm_pre.goal.to_numpy() + gm_pre.tower.to_numpy())
    dd = dict(native_gm=v("gm", "native", "none"), native_nongm=v("control", "native", "none"), gm=v("gm", "pre", "teacher"),
              nongm=v("control", "pre", "teacher"), gm_fc=v("gm", "fc_head", "teacher"), init=np.array([]), chance=chance)
    return [("WalkToFlag (Isaac, native eval set)", "deterministic (paper)", dd,
             "coin flip between flag and tower at the GM h_pre rewire's decisiveness, 0.5 x P(reached either); "
             "flag removed, GM h_pre reaches the empty flag spot first 0.181", "random-trunk arm: see isaac/analysis/rewire_floor_summary_isaac.py")]


def robot_behaviour():
    """What the h_pre readouts do in the two-object / continuum robot tasks, per cohort: outcome split, selectivity, BC fit."""
    L = ["| task | trunk | P(goal) | P(proxy) | P(neither) | P(goal given decisive) | goal-cue-neutral P(u given decisive) | BC val loss (final) |",
         "|---|---|---|---|---|---|---|---|"]
    rows = []
    at = pd.read_csv(os.path.join(RA, "rewire_at_deep", "per_seed.csv"))
    at = at[(at.variant == "ctrl2") & (at.arm == "pre")]
    bc = pd.read_csv(os.path.join(RA, "rewire_at_deep", "bc_fit_per_seed.csv"))
    bc = bc[(bc.variant == "ctrl2") & (bc.arm == "pre")]
    for c in ("gm", "control", "init"):
        s, n = at[(at.cohort == c) & (at.cond == "standard")], at[(at.cohort == c) & (at.cond == "nogoal_sph")]
        r = dict(task="AttributeTable", trunk=c, goal=tci(s.P_u_100), proxy=tci(s.P_v_100), neither=tci(s.P_n_100),
                 pdec=tci(s.P_u_dec_100), neutral=tci(n.P_u_dec_100), val=tci(bc[bc.cohort == c].final_val))
        rows.append(r)
    sz = pd.read_csv(os.path.join(RA, "rewire_sz_ctrl2", "per_seed.csv"))
    sz = sz[(sz.variant == "ctrl2") & (sz.arm == "pre")]
    for c in ("gm", "control", "init"):
        s, n = sz[(sz.cohort == c) & (sz.cond == "standard")], sz[(sz.cohort == c) & (sz.cond == "nogoal")]
        val = [pd.read_csv(f).val_loss.iloc[-1] for f in sorted(glob.glob(os.path.join(RA, "rewire_sz_ctrl2", "trainlogs", f"ctrl2_{c}_s*_pre.csv")))]
        rows.append(dict(task="SortZones", trunk=c, goal=tci(s.P_u), proxy=tci(s.P_v), neither=tci(1 - s.P_u - s.P_v),
                         pdec=tci(s.P_dec), neutral=tci(n.P_dec), val=tci(val)))
    lr = pd.read_csv(os.path.join(RA, "rewire_check", "per_seed.csv"))
    lr = lr[(lr.arm == "pre") & (lr.cond == "standard")]
    for c in ("gm", "control", "init"):
        s = lr[lr.cohort == c]
        val = [pd.read_csv(f).val_loss.iloc[-1] for f in sorted(glob.glob(os.path.join(RA, "rewire_check", f"trainlog_{c}_s*_pre.csv")))]
        rows.append(dict(task="LocationReach (proxy = training corner; neutral column = phantom-goal reach)", trunk=c,
                         goal=tci(s.reach_scored), proxy=tci(s.reach_corner),
                         neither=tci(1 - s.reach_scored - s.reach_corner), pdec=tci(s.reach_scored / (s.reach_scored + s.reach_corner)),
                         neutral=tci(s.p_phantom), val=tci(val)))
    for r in rows:
        L.append(f"| {r['task']} | {r['trunk']} | {fmt(*r['goal'])} | {fmt(*r['proxy'])} | {fmt(*r['neither'])} | {fmt(*r['pdec'])} | "
                 f"{fmt(*r['neutral'])} | {fmt(*r['val'])} |")
    return L


def ptm_fit():
    """BC fit of the PushToMarker readouts on the rebuilt bank, per variant and cohort (final val, min val, final train)."""
    fs = sorted(glob.glob(os.path.join(RAW, "ptm", "trainlogs", "*.csv")))
    if not fs:
        return []
    rows = []
    for f in fs:
        d = pd.read_csv(f)
        rows.append(dict(cohort=d.cohort.iloc[0], seed=d.seed.iloc[0], variant=d.variant.iloc[0], final_val=d.val_loss.iloc[-1],
                         min_val=d.val_loss.min(), final_train=d.train_loss.iloc[-1], h_pre_rms=d.get("h_pre_rms", pd.Series([np.nan])).iloc[0]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ptm_bc_fit_per_seed.csv"), index=False)
    L = ["| variant | trunk | final val loss | min val loss | final train loss | h_pre RMS |", "|---|---|---|---|---|---|"]
    for (v, c), g in df.groupby(["variant", "cohort"]):
        L.append(f"| {v} | {c} | {fmt(*tci(g.final_val))} | {fmt(*tci(g.min_val))} | {fmt(*tci(g.final_train))} | {fmt(*tci(g.h_pre_rms))} |")
    return L


def coin_outcomes():
    """Coin episode outcomes per arm: coin, level end without coin, died, timeout, and coin share among coin-or-end."""
    rows = jl(os.path.join(REF, "coin_block3_teacher.json"))["rows"]
    arms = {"GM native": [rows[k]["eval"]["outcome_rates"] for k in rows if k.startswith("native_gm")],
            "Non-GM native": [rows[k]["eval"]["outcome_rates"] for k in rows if k.startswith("native_nongm")],
            "GM rewire h_pre": [rows[k]["eval"]["outcome_rates"] for k in rows if k.startswith("gm_rewire")],
            "GM remix h_pre": [rows[k]["eval"]["outcome_rates"] for k in rows if k.startswith("gm_random_proj")],
            "Non-GM rewire h_pre": [rows[k]["eval"]["outcome_rates"] for k in rows if k.startswith("nongm_rewire")],
            "random trunk rewire h_pre (greedy)": [jl(f)["outcome_rates"] for f in sorted(glob.glob(os.path.join(RAW, "coin", "rand*", "eval_greedy.json")))]}
    L = ["| arm | coin | level end, no coin | died | timeout | coin among coin-or-end |", "|---|---|---|---|---|---|"]
    for name, o in arms.items():
        g = lambda k: [x[k] for x in o]
        sel = [x["success"] / (x["success"] + x["goal_misgen"]) for x in o]
        L.append(f"| {name} | {fmt(*tci(g('success')))} | {fmt(*tci(g('goal_misgen')))} | {fmt(*tci(g('died')))} | "
                 f"{fmt(*tci(g('timed_out')))} | {fmt(*tci(sel))} |")
    return L


# ------------------------------------------------------------------ table + verdict

def fmt(m, lo=np.nan, hi=np.nan):
    if not np.isfinite(m):
        return "-"
    return f"{m:.3f}" if not np.isfinite(lo) else f"{m:.3f} [{lo:.2f}, {hi:.2f}]"


def verdict(d):
    """Stricter floor = the higher of random trunk and chance; recovered >= 0.9 of the Non-GM rewire above it (paper rule),
    none when GM - floor has a CI that includes 0."""
    floors = [(k, d[k]) for k in ("init", "chance") if len(np.atleast_1d(d.get(k, []))) and np.isfinite(np.mean(d[k]))]
    k, f = max(floors, key=lambda kv: np.mean(kv[1]))
    diff = welch(d["gm"], f)
    r = norm_ci(d["gm"], d["nongm"], f)
    if not (diff[1] > 0 if np.isfinite(diff[1]) else diff[0] > 0):
        v = "none"
    elif r[0] >= 0.9:
        v = "recovered"
    else:
        v = "partial"
    return v, k, diff, r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=OUT)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="random-trunk seeds (rand<k>) to read")
    a = p.parse_args()
    groups = maze() + coin() + am() + robots() + ptm_refit() + isaac()
    long = []
    for env, mode, d, _, _ in groups:
        for key, arr in d.items():
            for i, x in enumerate(np.atleast_1d(arr)):
                long.append(dict(env=env, mode=mode, key=key, seed_index=i, value=float(x)))
    pd.DataFrame(long).to_csv(os.path.join(a.out, "per_seed.csv"), index=False)

    rows, md = [], []
    md += ["| environment | eval mode | native GM | chance | random-trunk rewire | GM rewire h_pre | GM rewire h_fc | Non-GM rewire h_pre | "
           "norm. recovery vs random trunk | norm. recovery vs chance | GM - stricter floor | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for env, mode, d, cdef, note in groups:
        ni = norm_ci(d["gm"], d["nongm"], d["init"]) if len(d["init"]) else (np.nan,) * 3
        nc = norm_ci(d["gm"], d["nongm"], d["chance"])
        v, fk, diff, _ = verdict(d)
        r = dict(env=env, mode=mode, native_gm=tci(d["native_gm"]), chance=tci(d["chance"]), init=tci(d["init"]),
                 gm=tci(d["gm"]), gm_fc=tci(d.get("gm_fc", [])), nongm=tci(d["nongm"]), norm_init=ni, norm_chance=nc,
                 gm_minus_floor=diff, floor=fk, verdict=v, chance_def=cdef, note=note)
        rows.append(r)
        md.append(f"| {env} | {mode} | {fmt(*r['native_gm'])} | {fmt(*r['chance'])} | {fmt(*r['init'])} | {fmt(*r['gm'])} | "
                  f"{fmt(*r['gm_fc'])} | {fmt(*r['nongm'])} | {fmt(*ni)} | {fmt(*nc)} | {fmt(*diff)} ({'random trunk' if fk == 'init' else 'chance'}) | {v} |")
    flat = []
    for r in rows:
        f = {k: r[k] for k in ("env", "mode", "floor", "verdict", "chance_def", "note")}
        for k in ("native_gm", "chance", "init", "gm", "gm_fc", "nongm", "norm_init", "norm_chance", "gm_minus_floor"):
            f[k], f[k + "_lo"], f[k + "_hi"] = r[k]
        flat.append(f)
    pd.DataFrame(flat).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "summary_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n\nChance definitions:\n\n")
        for r in rows:
            fh.write(f"- {r['env']} / {r['mode']}: {r['chance_def']}" + (f" ({r['note']})" if r["note"] else "") + "\n")
        fh.write("\n## Coin outcomes per arm (greedy, 1000 held-out levels)\n\n" + "\n".join(coin_outcomes()) + "\n")
        fh.write("\n## Robot h_pre readouts: what each trunk's readout does (deterministic eval)\n\n" + "\n".join(robot_behaviour()) + "\n")
        pf = ptm_fit()
        if pf:
            fh.write("\n## PushToMarker BC fit on the rebuilt bank\n\n" + "\n".join(pf) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
