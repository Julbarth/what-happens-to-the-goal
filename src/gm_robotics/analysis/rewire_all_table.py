"""One shared rewiring table for AT / SZ / LR / PTM from the per-seed results (raw per-seed CSV, then summary CSV + markdown)."""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from gm_robotics.paths import ROOT as RUN_ROOT

ROOT = str(RUN_ROOT)
RES = os.path.join(ROOT, "results", "analysis")

# (column key, header) in table order
COLS = [("native_gm", "GM native"), ("pre_gm", "GM rewired h_pre"), ("pre_control", "control rewired h_pre"),
        ("pre_init", "random-init trunk h_pre"), ("masked_gm", "GM goal-cue-masked retrain h_pre"),
        ("fc_gm", "GM head on h_fc"), ("remix_gm", "GM random remix"), ("pdec_gm", "GM h_pre P(goal | decisive)"),
        ("nocue_gm", "GM h_pre, goal cue removed at test"), ("chance", "chance reference"),
        ("excess_gm", "GM h_pre minus chance reference"),
        ("excess_masked_gm", "GM h_pre minus GM masked retrain (paired by seed)")]

METRIC = {
    "AT": "first TCP touch of the goal object (1.5 cm of its surface) before the proxy, within 100 steps; 512 held-out "
          "conflict scenes (blue cube u, red sphere v); ctrl2 BC (control-distribution scenes only); GM s2-4, control s2-5, init s2-4",
    "SZ": "first cube entry into the crossed pad before the green pad, within 100 steps; 512 held-out conflict scenes; ctrl2 BC; "
          "seeds 0-2 per cohort",
    "LR": "first TCP contact (3 cm) with the marker before the training corner, within 50 steps (the LR episode length); "
          "500 off-path episodes per seed (~30 distinct goal positions); BC bank = control-distribution scenes; seeds 1-3",
    "PTM (conflict)": "final cube position within 2.5 cm of the marker after 100 steps; 512 held-out conflict scenes "
                      "(marker on the cube's inward ray, so a marker-blind push that stops short can score); BC = control scenes; seeds 0-2",
    "PTM (ctrlset)": "as PTM (conflict) but on 512 control-distribution scenes (marker anywhere), where the stop-short shortcut "
                     "cannot score",
}
CHANCE = {
    "AT": "coin flip at the GM h_pre decisiveness, 0.5 x P(touched u or v); masked retrain in its own column",
    "SZ": "coin flip at the GM h_pre decisiveness, 0.5 x P(cube in either pad); masked retrain in its own column",
    "LR": "phantom-goal reach of the same GM h_pre paths (pool phantoms, K = 20 per episode)",
    "PTM (conflict)": "phantom-goal reach (pool) of the same GM h_pre rollouts; the stop-short check is the 'goal cue removed' column",
    "PTM (ctrlset)": "phantom-goal reach (pool) of the same GM h_pre rollouts",
}
NOCUE = {"AT": "two spheres (blue at u, red at v), scored at u", "SZ": "no cross on either pad, scored at u",
         "LR": "marker hidden, scored at the original goal", "PTM (conflict)": "marker hidden, scored at m",
         "PTM (ctrlset)": "marker hidden, scored at m"}


def per_seed():
    rows = []

    def add(task, key, df, col):
        for _, r in df.iterrows():
            rows.append(dict(task=task, key=key, seed=int(r.seed), value=float(r[col])))

    at = pd.read_csv(os.path.join(RES, "rewire_at_deep", "per_seed.csv"))
    std = at[at.cond == "standard"]
    add("AT", "native_gm", std[(std.variant == "base") & (std.arm == "native") & (std.cohort == "gm")], "P_u_100")
    c2 = std[std.variant == "ctrl2"]
    for key, coh, arm in (("pre_gm", "gm", "pre"), ("pre_control", "control", "pre"), ("pre_init", "init", "pre"),
                          ("masked_gm", "gm", "pre_masked"), ("fc_gm", "gm", "fc_head"), ("remix_gm", "gm", "remix")):
        add("AT", key, c2[(c2.cohort == coh) & (c2.arm == arm)], "P_u_100")
    g = c2[(c2.cohort == "gm") & (c2.arm == "pre")].copy()
    add("AT", "pdec_gm", g, "P_u_dec_100")
    g["coin"] = 0.5 * (g.P_u_100 + g.P_v_100)
    add("AT", "chance", g, "coin")
    g["excess"] = g.P_u_100 - g.coin
    add("AT", "excess_gm", g, "excess")
    nc = at[(at.variant == "ctrl2") & (at.cohort == "gm") & (at.arm == "pre") & (at.cond == "nogoal_sph")]
    add("AT", "nocue_gm", nc, "P_u_100")

    sz = pd.read_csv(os.path.join(RES, "rewire_sz_ctrl2", "per_seed.csv"))
    s = sz[sz.cond == "standard"]
    add("SZ", "native_gm", s[(s.variant == "base") & (s.arm == "native") & (s.cohort == "gm")], "P_u")
    c2 = s[s.variant == "ctrl2"]
    for key, coh, arm in (("pre_gm", "gm", "pre"), ("pre_control", "control", "pre"), ("pre_init", "init", "pre"),
                          ("masked_gm", "gm", "pre_masked"), ("fc_gm", "gm", "fc_head"), ("remix_gm", "gm", "remix")):
        add("SZ", key, c2[(c2.cohort == coh) & (c2.arm == arm)], "P_u")
    g = c2[(c2.cohort == "gm") & (c2.arm == "pre")].copy()
    add("SZ", "pdec_gm", g, "P_dec")
    g["coin"] = 0.5 * (g.P_u + g.P_v)
    add("SZ", "chance", g, "coin")
    g["excess"] = g.P_u - g.coin
    add("SZ", "excess_gm", g, "excess")
    nc = sz[(sz.variant == "ctrl2") & (sz.cohort == "gm") & (sz.arm == "pre") & (sz.cond == "nogoal")]
    add("SZ", "nocue_gm", nc, "P_u")

    lr = pd.read_csv(os.path.join(RES, "rewire_check", "per_seed.csv"))
    s = lr[lr.cond == "standard"]
    for key, coh, arm in (("native_gm", "gm", "native"), ("pre_gm", "gm", "pre"), ("pre_control", "control", "pre"),
                          ("pre_init", "init", "pre"), ("masked_gm", "gm", "pre_masked"), ("fc_gm", "gm", "fc_head"),
                          ("remix_gm", "gm", "remix")):
        add("LR", key, s[(s.cohort == coh) & (s.arm == arm)], "reach_scored")
    g = s[(s.cohort == "gm") & (s.arm == "pre")].copy()
    g["pdec"] = g.reach_scored / (g.reach_scored + g.reach_corner)
    add("LR", "pdec_gm", g, "pdec")
    add("LR", "chance", g, "p_phantom")
    add("LR", "excess_gm", g, "excess_over_phantom")
    add("LR", "nocue_gm", lr[(lr.cohort == "gm") & (lr.arm == "pre") & (lr.cond == "nomarker")], "reach_scored")

    pt = pd.read_csv(os.path.join(RES, "ptm", "rewire_per_seed.csv"))
    for sset in ("conflict", "ctrlset"):
        task = f"PTM ({sset})"
        s = pt[(pt.set == sset) & (pt.cond == "standard")]
        for key, coh, arm in (("native_gm", "gm", "native"), ("pre_gm", "gm", "pre"), ("pre_control", "control", "pre"),
                              ("pre_init", "init", "pre"), ("masked_gm", "gm", "pre_masked"), ("fc_gm", "gm", "fc_head"),
                              ("remix_gm", "gm", "remix")):
            add(task, key, s[(s.cohort == coh) & (s.arm == arm)], "goal")
        g = s[(s.cohort == "gm") & (s.arm == "pre")].copy()
        add(task, "pdec_gm", g, "goal_given_decisive")
        add(task, "chance", g, "phantom_pool")
        g["excess"] = g.goal - g.phantom_pool
        add(task, "excess_gm", g, "excess")
        add(task, "nocue_gm", pt[(pt.set == sset) & (pt.cond == "nomarker") & (pt.cohort == "gm") & (pt.arm == "pre")], "goal")
    df = pd.DataFrame(rows)
    w = df[df.key.isin(["pre_gm", "masked_gm"])].pivot_table(index=["task", "seed"], columns="key", values="value").dropna()
    ex = (w.pre_gm - w.masked_gm).rename("value").reset_index().assign(key="excess_masked_gm")
    return pd.concat([df, ex[["task", "key", "seed", "value"]]], ignore_index=True)


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def summarise(ps):
    out = []
    for (task, key), g in ps.groupby(["task", "key"], sort=False):
        g = g.sort_values("seed")
        m, lo, hi = tci(g.value)
        out.append(dict(task=task, key=key, n=len(g), mean=m, lo=lo, hi=hi,
                        seeds=" / ".join(f"{v:.3f}" for v in g.value), seed_ids=" / ".join(str(s) for s in g.seed)))
    return pd.DataFrame(out)


def markdown(sm):
    tasks = ["AT", "SZ", "LR", "PTM (conflict)", "PTM (ctrlset)"]
    cell = {(r.task, r.key): r for r in sm.itertuples()}

    def fmt(task, key, with_seeds=False):
        r = cell.get((task, key))
        if r is None:
            return "–"
        s = f"{r.mean:.3f} [{r.lo:.2f}, {r.hi:.2f}]" if np.isfinite(r.lo) else f"{r.mean:.3f}"
        return s + (f" ({r.seeds})" if with_seeds else "")

    L = ["# Rewiring, all tasks (mean [95% t-CI] over seeds; per-seed values in parentheses for the GM h_pre arm)", "",
         "Rewiring = new readout BC-trained on a frozen trunk, teacher = a different control seed, BC frames from "
         "control-distribution scenes only (AT / SZ: ctrl2; LR / PTM: their banks never had GM-distribution scenes). "
         "Goal reach is the fraction of evaluation episodes whose outcome is the goal. '–' = arm not run.", ""]
    L.append("| column | " + " | ".join(tasks) + " |")
    L.append("|---|" + "---|" * len(tasks))
    for key, head in COLS:
        L.append(f"| {head} | " + " | ".join(fmt(t, key, key == "pre_gm") for t in tasks) + " |")
    L += ["", "## Per-task metric definitions (horizons and scoring differ; compare within a task, not across)", ""]
    L += [f"- **{t}**: {METRIC[t]}." for t in tasks]
    L += ["", "## Chance reference and goal-cue-removed condition per task", ""]
    L += [f"- **{t}**: chance = {CHANCE[t]}; goal cue removed = {NOCUE[t]}." for t in tasks]
    L += ["", "Notes: AT and SZ have two candidate objects, so a goal-blind but decisive policy scores 0.5 x P(decisive); "
          "LR and PTM have a continuum of goal positions, so their chance is the phantom-goal reach. The masked retrain "
          "is the empirical goal-blind BC head on the same trunk and is the stricter floor in every task. PTM init-trunk "
          "h_pre cannot fit the teacher (BC val loss 6.7-17), so its ~0.01 is a failed fit, not a goal-blind floor."]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=RES)
    a = p.parse_args()
    ps = per_seed()
    ps.to_csv(os.path.join(a.out, "rewire_all_per_seed.csv"), index=False)
    sm = summarise(ps)
    sm.to_csv(os.path.join(a.out, "rewire_all_summary.csv"), index=False)
    md = markdown(sm)
    open(os.path.join(a.out, "rewire_all_summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
