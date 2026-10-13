"""PushToMarker summaries: world edits, steering, rewiring with phantom-goal chance and paired contrasts, per-seed CSVs."""
import argparse
import os

import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C

CENTRE = np.array([-0.08, 0.0])  # push_to_marker.CENTRE (importing the env needs sapien)
GOAL_THR, PROXY_THR, R_OUT = 0.025, 0.03, 0.16
OUT = os.path.join(C.RESULTS, "ptm")
COH = {"gm": "GM", "control": "Non-GM", "init": "init"}


def ci_row(vals):
    m, lo, hi = C.ci95(np.asarray(vals, float))
    return m, lo, hi


def fmt(vals, nd=3):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "n/a"
    return C.fmt_ci(*ci_row(v), nd) if len(v) > 1 else f"{v[0]:.{nd}f}"


# ---------------------------------------------------------------- world edits

WE_COLS = ["KL", "SF", "Q_h_fc", "G_h_fc", "D_F_h_fc", "dnorm2_h_fc", "trF_per_d_h_fc", "Q_h_pre", "G_h_pre", "D_F_h_pre",
           "dnorm2_h_pre", "trF_per_d_h_pre", "calib_t1_h_fc"]
BEH = ("KL", "SF")


def worldedit(tag, bank_npz=None, min_frac_visible=0.75):
    """Per seed: mean (KL, SF) or median (Fisher terms) over states, then mean and 95% t-CI across seeds. For the hover
    bank also on the subset of states whose marker is visible (hidden-marker pixel change >= min_frac_visible x the
    start bank's median), because the hand above the cube can occlude the marker."""
    raw = os.path.join(C.RESULTS, "worldedit_raw")
    fs = sorted(f for f in os.listdir(raw) if f.startswith(tag + "_"))
    df = pd.concat([pd.read_csv(os.path.join(raw, f)) for f in fs], ignore_index=True)
    subsets = {"all": None}
    if bank_npz and os.path.exists(bank_npz):
        z = np.load(bank_npz)
        ref = np.median(np.load(os.path.join(C.BANKS, "PTM_worldedit.npz"))["npx_nomarker"])
        vis = z["npx_nomarker"] >= min_frac_visible * ref
        subsets["marker_visible"] = np.nonzero(vis)[0]
        print(f"{tag}: marker visible in {vis.mean():.3f} of states (>= {min_frac_visible} x {ref:.0f} px)")
    rows, ps_all = [], []
    for sub, keep in subsets.items():
        d = df if keep is None else df[df.state.isin(keep)]
        g = d.groupby(["edit", "cohort", "seed"])
        ps = pd.concat([g[list(BEH)].mean(), g[[c for c in WE_COLS if c not in BEH]].median()], axis=1).reset_index()
        ps["subset"], ps["n_states"] = sub, g.size().to_numpy()
        ps_all.append(ps)
        for (e, coh), s in ps.groupby(["edit", "cohort"]):
            r = dict(bank=tag, subset=sub, edit=e, cohort=coh, n=len(s))
            for c in WE_COLS:
                r[c], r[c + "_lo"], r[c + "_hi"] = ci_row(s[c])
            rows.append(r)
    ps = pd.concat(ps_all, ignore_index=True)
    ps.to_csv(os.path.join(OUT, f"worldedit_{tag}_per_seed.csv"), index=False)
    sm = pd.DataFrame(rows)
    sm.to_csv(os.path.join(OUT, f"worldedit_{tag}_summary.csv"), index=False)
    cols = ["KL", "SF", "Q_h_fc", "G_h_fc", "D_F_h_fc", "dnorm2_h_fc", "trF_per_d_h_fc", "Q_h_pre", "G_h_pre", "D_F_h_pre"]
    lines = [f"## World edits, {tag}", "", "Mean [95% t-CI] over seeds; KL, SF per-seed means, Fisher terms per-seed medians. "
             "SF (marker_move only) = change in P(TCP step has a positive component along p(m2) - p(m)).", "",
             "| subset | edit | cohort | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 4)]
    for _, r in sm.sort_values(["subset", "edit", "cohort"]).iterrows():
        lines.append(f"| {r.subset} | {r.edit} | {COH[r.cohort]} | {r.n} | " +
                     " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi']) for c in cols) + " |")
    return "\n".join(lines) + "\n", ps


# ---------------------------------------------------------------- probes

def probes():
    f = os.path.join(C.RESULTS, "probes_PTM_per_seed.csv")
    if not os.path.exists(f):
        return ""
    df = pd.read_csv(f)
    sites = ["input", "conv1", "conv2", "conv3", "h_pre", "h_fc", "a_hid"]
    lines = ["## Linear probes (held-out bank, R2; mean [95% t-CI] over seeds)", ""]
    for t in sorted(df.target.unique()):
        lines += [f"### {t}", "", "| cohort | " + " | ".join(sites) + " |", "|" + "---|" * (len(sites) + 1)]
        for coh in ("gm", "control", "init", "pixels", "proprio"):
            d = df[(df.target == t) & (df.cohort == coh)]
            if not len(d):
                continue
            cells = [fmt(d[d.site == s].value) if (d.site == s).any() else "" for s in sites]
            lines.append(f"| {COH.get(coh, coh)} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- steering

def steering():
    f = os.path.join(C.RESULTS, "steer_PTM_per_state.csv")
    if not os.path.exists(f):
        return "", None
    df = pd.read_csv(f)
    df["goal"], df["proxy"] = df.outcome.eq("goal"), df.outcome.eq("proxy")
    df["kind"] = np.where(df.arm == "local", np.where(df.beta > 0, "corrective", "anticorrective"), df.arm)
    df["dose"] = df.beta.abs()
    ps = df.groupby(["cohort", "seed", "kind", "dose", "rep"])[["goal", "proxy", "d_marker", "d_centre", "dnorm0"]].mean()
    ps = ps.groupby(["cohort", "seed", "kind", "dose"]).mean().reset_index()  # random: mean over repeats per seed
    ps.to_csv(os.path.join(OUT, "steer_per_seed.csv"), index=False)
    lines = ["## h_fc steering (state-local d = h(marker at m) - h(marker at centre)), 200 conflict states, det",
             "", "| cohort | arm | beta | goal | proxy (centre) | final d_marker | |d(s0)| |", "|---|---|---|---|---|---|---|"]
    for (coh, kind, dose), s in ps.groupby(["cohort", "kind", "dose"]):
        lines.append(f"| {COH[coh]} | {kind} | {dose:g} | {fmt(s.goal)} | {fmt(s.proxy)} | {fmt(s.d_marker)} | {fmt(s.dnorm0, 2)} |")
    drift = df.groupby("cohort").copy_drift.median()
    lines += ["", "copy_drift (largest one-step state divergence before re-sync) median: " +
              ", ".join(f"{COH[k]} {v:.2f}" for k, v in drift.items())]
    return "\n".join(lines) + "\n", ps


# ---------------------------------------------------------------- rewiring

def phantom_points(family, scene_m, cube0, rng, K=20, sep=0.08):
    """K phantom goals per episode: 'pool' = other episodes' markers of the same eval set, 'box' = uniform on the
    marker disc (radius R_OUT around the centre). Eligibility as for real goals: >= sep from the true marker, outside
    the proxy disc plus the goal radius, and >= 5 cm (the shortest trained push) from the start cube."""
    n = len(scene_m)
    out = np.full((n, K, 2), np.nan)
    for i in range(n):
        if family == "pool":
            cand = np.delete(scene_m, i, 0)
        else:
            r = np.sqrt(rng.random(4000)) * R_OUT
            th = rng.random(4000) * 2 * np.pi
            cand = CENTRE + np.stack([r * np.cos(th), r * np.sin(th)], 1)
        ok = (np.linalg.norm(cand - scene_m[i], axis=1) >= sep) & (np.linalg.norm(cand - CENTRE, axis=1) >= PROXY_THR + GOAL_THR) \
            & (np.linalg.norm(cand - cube0[i], axis=1) >= 0.05)
        c = cand[ok]
        out[i] = c[rng.choice(len(c), size=K, replace=len(c) < K)]
    return out


def rewiring():
    d = os.path.join(C.RESULTS, "rewire_ptm")
    f = os.path.join(d, "per_episode.csv")
    if not os.path.exists(f):
        return "", None
    df = pd.read_csv(f)
    scenes = scene_markers(d)
    rng = np.random.default_rng(0)
    ph = {sn: {fam: phantom_points(fam, *scenes[sn], rng) for fam in ("pool", "box")} for sn in scenes
          if sn != "conflict_m2"}
    out = []
    for (coh, k, arm, sname, cond), g in df.groupby(["cohort", "seed", "arm", "set", "cond"]):
        z = np.load(os.path.join(d, "paths", f"{coh}_s{k}_{arm}_{sname}_{cond}.npz"))
        g = g.sort_values("episode")
        ep = g.episode.to_numpy()
        fin, start, on = z["cube"][:, -1], z["cube"][:, 0], z["cube_z"] < 0.025
        m = scenes[sname][0][ep]
        r = dict(cohort=coh, seed=k, arm=arm, set=sname, cond=cond, n=len(g),
                 goal=g.outcome.eq("goal").mean(), proxy=g.outcome.eq("proxy").mean(),
                 neither=g.outcome.eq("neither").mean(), d_marker=g.d_marker.mean(), cube_moved=g.cube_moved.mean())
        dec = r["goal"] + r["proxy"]
        r["goal_given_decisive"] = r["goal"] / dec if dec > 0 else np.nan
        for fam, pts in ph[sname].items():
            hit = (np.linalg.norm(fin[:, None, :] - pts[ep], axis=-1) < GOAL_THR) & on[:, None]
            r[f"phantom_{fam}"] = hit.mean()
        disp = fin - start
        mv = np.linalg.norm(disp, axis=1) > 0.01

        def hcos(target):
            w = target - start
            c = (disp * w).sum(1) / np.maximum(np.linalg.norm(disp, axis=1) * np.linalg.norm(w, axis=1), 1e-9)
            return float(c[mv].mean()) if mv.any() else np.nan
        r["heading_cos_scored"] = hcos(m)
        if cond == "decoupled":
            r["reach_m2"] = g.reach_m2.mean()
            r["heading_cos_drawn"] = hcos(scenes["conflict_m2"][ep])
        out.append(r)
    ps = pd.DataFrame(out)
    ps.to_csv(os.path.join(OUT, "rewire_per_seed.csv"), index=False)
    return rewire_md(ps), ps


def scene_markers(d):
    """Scored marker and start cube per eval scene: the world-edit bank (conflict) and rewire_ptm's saved ctrlset."""
    b = np.load(os.path.join(C.BANKS, "PTM_worldedit.npz"))
    out = {"conflict": (b["m"], b["cube"][:, :2]), "conflict_m2": b["m2"]}
    f = os.path.join(d, "ctrlset_scenes.npz")
    if os.path.exists(f):
        cs = np.load(f)
        out["ctrlset"] = (cs["m"], cs["cube"][:, :2])
    return out


ARMS = ("native", "fc_head", "pre", "remix", "untrained", "pre_masked", "rand_walk")


def rewire_md(ps):
    lines = ["## Rewiring (rewire_lr protocol), final-state outcome, det; mean [95% t-CI] over seeds", ""]
    for sname, cond in (("conflict", "standard"), ("conflict", "decoupled"), ("conflict", "nomarker"),
                        ("ctrlset", "standard"), ("ctrlset", "nomarker")):
        s = ps[(ps.set == sname) & (ps.cond == cond)]
        if not len(s):
            continue
        extra = ["reach_m2", "heading_cos_drawn"] if cond == "decoupled" else []
        cols = ["goal", "proxy", "neither", "goal_given_decisive", "phantom_pool", "phantom_box", "heading_cos_scored"] + extra
        lines += [f"### {sname} / {cond}", "", "| arm | cohort | n | " + " | ".join(cols) + " | per-seed goal |",
                  "|" + "---|" * (len(cols) + 4)]
        for arm in ARMS:
            for coh in ("gm", "control", "init"):
                x = s[(s.arm == arm) & (s.cohort == coh)].sort_values("seed")
                if not len(x):
                    continue
                lines.append(f"| {arm} | {COH[coh]} | {len(x)} | " + " | ".join(fmt(x[c]) for c in cols) + " | " +
                             " / ".join(f"{v:.3f}" for v in x.goal) + " |")
        lines.append("")
    lines += contrasts(ps)
    return "\n".join(lines) + "\n"


def contrasts(ps):
    """Paired per-seed differences for GM (and Non-GM where meaningful)."""
    def get(coh, arm, sname, cond, col="goal"):
        x = ps[(ps.cohort == coh) & (ps.arm == arm) & (ps.set == sname) & (ps.cond == cond)].set_index("seed")[col]
        return x

    pairs = [("GM pre, conflict: standard - nomarker", ("gm", "pre", "conflict", "standard"), ("gm", "pre", "conflict", "nomarker")),
             ("GM pre, conflict: standard - masked retrain", ("gm", "pre", "conflict", "standard"), ("gm", "pre_masked", "conflict", "standard")),
             ("GM pre, conflict: standard - phantom(pool)", ("gm", "pre", "conflict", "standard"), ("gm", "pre", "conflict", "standard", "phantom_pool")),
             ("GM pre, decoupled: reach drawn m2 - reach scored m", ("gm", "pre", "conflict", "decoupled", "reach_m2"), ("gm", "pre", "conflict", "decoupled")),
             ("GM pre, ctrlset: standard - nomarker", ("gm", "pre", "ctrlset", "standard"), ("gm", "pre", "ctrlset", "nomarker")),
             ("GM pre - init pre, ctrlset standard", ("gm", "pre", "ctrlset", "standard"), ("init", "pre", "ctrlset", "standard")),
             ("GM pre - init pre, conflict standard", ("gm", "pre", "conflict", "standard"), ("init", "pre", "conflict", "standard")),
             ("GM pre - GM fc_head, ctrlset standard", ("gm", "pre", "ctrlset", "standard"), ("gm", "fc_head", "ctrlset", "standard")),
             ("GM pre - GM remix, ctrlset standard", ("gm", "pre", "ctrlset", "standard"), ("gm", "remix", "ctrlset", "standard")),
             ("Non-GM pre - GM pre, ctrlset standard", ("control", "pre", "ctrlset", "standard"), ("gm", "pre", "ctrlset", "standard")),
             ("Non-GM pre, conflict: standard - nomarker", ("control", "pre", "conflict", "standard"), ("control", "pre", "conflict", "nomarker")),
             ("Non-GM pre, decoupled: reach m2 - reach m", ("control", "pre", "conflict", "decoupled", "reach_m2"), ("control", "pre", "conflict", "decoupled"))]
    lines = ["### Paired contrasts (per-seed differences, mean [95% t-CI]; per seed)", "", "| contrast | value | per seed |", "|---|---|---|"]
    rows = []
    for name, a, b in pairs:
        xa, xb = get(*a[:4], *(a[4:] or ["goal"])), get(*b[:4], *(b[4:] or ["goal"]))
        dlt = (xa - xb).dropna()
        if not len(dlt):
            continue
        lines.append(f"| {name} | {fmt(dlt)} | " + " / ".join(f"{v:.3f}" for v in dlt) + " |")
        rows.append(dict(contrast=name, **dict(zip(("mean", "lo", "hi"), ci_row(dlt))), per_seed=list(dlt.round(4))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rewire_contrasts.csv"), index=False)
    return lines + [""]


def train_logs():
    d = os.path.join(C.RESULTS, "rewire_ptm", "trainlogs")
    if not os.path.isdir(d):
        return ""
    rows = []
    for f in sorted(os.listdir(d)):
        x = pd.read_csv(os.path.join(d, f))
        coh, s, arm = f[:-4].split("_", 2)
        rows.append(dict(cohort=coh, seed=s, arm=arm, train=x.train_loss.iloc[-1], val=x.val_loss.iloc[-1], params=x.n_trainable.iloc[0]))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "rewire_trainlogs.csv"), index=False)
    lines = ["### BC final losses (sigma-weighted MSE; train / val per seed)", "", "| cohort | arm | params | train | val |", "|---|---|---|---|---|"]
    for (coh, arm), g in t.groupby(["cohort", "arm"]):
        lines.append(f"| {COH.get(coh, coh)} | {arm} | {g.params.iloc[0]} | " + " / ".join(f"{v:.2f}" for v in g.train) +
                     " | " + " / ".join(f"{v:.2f}" for v in g.val) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["worldedit", "probes", "steer", "rewire"])
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    md = ["# PushToMarker analysis summary", "",
          "GM m50 s0-2; Non-GM m50 s0, m100 s1-2 (BUDGET MISMATCH: s1, s2 trained 100M). 95% t-CI over 3 seeds.", ""]
    if "worldedit" in a.parts:
        for tag, bn in (("PTM", None), ("PTMhover", os.path.join(C.BANKS, "PTM_hover.npz"))):
            try:
                t, _ = worldedit(tag, bn)
                md.append(t)
            except (FileNotFoundError, ValueError) as e:
                print(f"skip worldedit {tag}: {e}")
    if "probes" in a.parts:
        md.append(probes())
    if "steer" in a.parts:
        md.append(steering()[0])
    if "rewire" in a.parts:
        md.append(rewiring()[0])
        md.append(train_logs())
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
