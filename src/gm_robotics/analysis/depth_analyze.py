"""Per-layer goal decodability, Fisher edit terms and rewiring recoverability, GM vs Non-GM (and init): table and figures."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from gm_robotics.analysis import common as C

P = C.PAL
RES = os.path.join(C.RESULTS, "depth")
FIG = os.path.join(C.FIGURES, "depth")
ENVS = {"LR": "LocationReach", "PTM": "PushToMarker", "AT": "AttributeTable", "SZ": "SortZones"}
SITES = ("conv1", "conv2", "h_pre", "h_fc", "a_hid")
SITE_LAB = {"conv1": "conv1", "conv2": "conv2", "h_pre": r"conv3 / $h_{pre}$", "h_fc": r"$h_{fc}$", "a_hid": "actor"}
# Goal targets fixed a priori from the task (not from the results): goal position (mean R2 of x and y) and goal side.
GOAL_POS = {"AT": ("cube_x", "cube_y"), "SZ": ("cross_x", "cross_y"), "LR": ("marker_x", "marker_y"),
            "PTM": ("marker_x", "marker_y")}
GOAL_SIDE = {"AT": "cube_left", "SZ": "cross_left", "LR": "marker_left", "PTM": "marker_left"}
GOAL_EDIT = {"AT": "shape_swap", "SZ": "mark_swap", "LR": "goal_move", "PTM": "marker_move"}
PROXY_EDIT = {"AT": "colour_swap", "SZ": "colour_swap", "LR": "distractor_move", "PTM": "cube_yaw"}
# standard condition and the goal-cue-absent condition (chance check) per env, in the per-episode 'cond' naming
# PTM uses the control-distribution set: in its conflict set a marker-blind inward push that stops short scores 'goal'
STD = {"AT": "standard", "SZ": "standard", "LR": "standard", "PTM": "ctrlset:standard"}
NOCUE = {"AT": "nogoal_sph", "SZ": "nogoal", "LR": "nomarker", "PTM": "ctrlset:nomarker"}
REWIRE_ARM = {"conv1": "conv1", "conv2": "conv2", "h_pre": "pre", "h_fc": "fc_head"}


def welch(a, b):
    """Mean difference a - b and Welch 95% t-interval (unpaired: GM seed k and Non-GM seed k are separate runs)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else np.inf
    h = stats.t.ppf(0.975, df) * se if se > 0 else 0.0
    return d, d - h, d + h


def log_ratio(a, b):
    """GM / Non-GM ratio of cohort means with a Welch interval on the log scale (delta method); NaN if a mean <= 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    ma, mb = a.mean(), b.mean()
    if ma <= 0 or mb <= 0:
        return np.nan, np.nan, np.nan
    va, vb = a.var(ddof=1) / len(a) / ma ** 2, b.var(ddof=1) / len(b) / mb ** 2
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else np.inf
    h = stats.t.ppf(0.975, df) * se if se > 0 else 0.0
    r = ma / mb
    return r, r * np.exp(-h), r * np.exp(h)


# ---------------------------------------------------------------- per-seed long table

def probe_rows(env):
    f = os.path.join(C.RESULTS, "probe_audit", f"{env}_per_seed.csv")
    if not os.path.exists(f):
        return []
    d = pd.read_csv(f)
    d = d[(d.subset == "all") & d.site.isin(SITES) & d.cohort.isin(["gm", "control", "init"])]
    rows = []
    for probe, lab in (("ridge_sel", "linear"), ("mlp", "MLP"), ("knn", "kNN")):
        g = d[d.probe == probe]
        pos = g[g.target.isin(GOAL_POS[env])].groupby(["cohort", "seed", "site"]).value.mean().reset_index()
        rows += [dict(env=env, measure=f"decode_pos_{lab}", cohort=r.cohort, seed=r.seed, site=r.site, value=r.value)
                 for r in pos.itertuples()]
        side = g[g.target == GOAL_SIDE[env]]
        # AUC - 0.5 so that 0 = chance and ratios are meaningful
        rows += [dict(env=env, measure=f"decode_side_{lab}", cohort=r.cohort, seed=r.seed, site=r.site, value=r.value - 0.5)
                 for r in side.itertuples()]
    return rows


def fisher_rows(env):
    f = os.path.join(RES, f"fisher_{env}_per_seed.csv")
    if not os.path.exists(f):
        return []
    d = pd.read_csv(f)
    rows = []
    for edit, role in ((GOAL_EDIT[env], "goal"), (PROXY_EDIT[env], "proxy")):
        g = d[d.edit == edit]
        for col in ("rel_edit", "Q", "G", "D_F", "calib"):
            rows += [dict(env=env, measure=f"{col}_{role}", cohort=r.cohort, seed=r.seed, site=r.site, value=getattr(r, col))
                     for r in g.itertuples()]
        rows += [dict(env=env, measure=f"KL_{role}", cohort=r.cohort, seed=r.seed, site=r.site, value=r.KL)
                 for r in g.itertuples()]
    return rows


def _goal(df, env):
    if env == "LR":
        return df.reach_scored.astype(str).str.lower().eq("true")
    return df.outcome.eq("goal" if env == "PTM" else "u")


def rewire_episodes(env, variant):
    """Goal outcome per episode for every (cohort, seed, arm, cond). variant 'orig' = the original BC banks (h_pre /
    h_fc from rewire_attr / rewire_check / rewire_ptm, conv1 / conv2 from depth/rewire); 'ctrl2' = AT/SZ BC data
    from control-distribution scenes only, all four layers from depth/rewire_ctrl2."""
    cols = ["cohort", "seed", "arm", "cond", "episode", "goal"]
    parts = []
    if variant == "orig":
        if env == "LR":
            old = pd.read_csv(os.path.join(C.RESULTS, "rewire_check", "per_episode.csv"), low_memory=False)
        elif env == "PTM":
            old = pd.read_csv(os.path.join(C.RESULTS, "rewire_ptm", "per_episode.csv"))
            old = old.assign(cond=old["set"] + ":" + old["cond"])
        else:
            old = pd.read_csv(os.path.join(C.RESULTS, "rewire_attr", f"{env}_per_episode.csv"))
        old = old[old.arm.isin(["native", "fc_head", "pre", "pre_masked"])]
        parts.append(old.assign(goal=_goal(old, env))[cols])
        f = os.path.join(RES, "rewire", f"{env}_per_episode.csv")
    else:
        f = os.path.join(RES, "rewire_ctrl2", f"{env}_per_episode.csv")
    if os.path.exists(f):
        new = pd.read_csv(f, low_memory=False)
        new = new.assign(arm=new.arm.replace({"h_pre": "pre", "h_pre_masked": "pre_masked", "h_fc": "fc_head",
                                              "h_fc_masked": "fc_head_masked"}))
        parts.append(new.assign(goal=_goal(new, env))[cols])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def rewire_rows(env):
    rows = []
    variants = [("ctrl2", ""), ("orig", "_origbank")] if env in ("AT", "SZ") else [("orig", "")]
    for variant, suf in variants:
        ep = rewire_episodes(env, variant)
        if not len(ep):
            continue
        rate = ep.groupby(["cohort", "seed", "arm", "cond"]).goal.mean().reset_index()
        std, noc = rate[rate.cond == STD[env]], rate[rate.cond == NOCUE[env]]
        for site, arm in REWIRE_ARM.items():
            masked_arm = "pre_masked" if arm == "pre" else f"{arm}_masked"
            for r in std[std.arm == arm].itertuples():
                key = (std.cohort == r.cohort) & (std.seed == r.seed)
                rows.append(dict(env=env, measure="rewire_goal" + suf, cohort=r.cohort, seed=r.seed, site=site, value=r.goal))
                m = std[(std.arm == masked_arm) & key]
                if len(m):
                    rows.append(dict(env=env, measure="rewire_masked" + suf, cohort=r.cohort, seed=r.seed, site=site,
                                     value=m.goal.iloc[0]))
                    rows.append(dict(env=env, measure="rewire_excess" + suf, cohort=r.cohort, seed=r.seed, site=site,
                                     value=r.goal - m.goal.iloc[0]))
                c = noc[(noc.arm == arm) & (noc.cohort == r.cohort) & (noc.seed == r.seed)]
                if len(c):
                    rows.append(dict(env=env, measure="rewire_nocue" + suf, cohort=r.cohort, seed=r.seed, site=site,
                                     value=c.goal.iloc[0]))
        if variant == "orig":
            for r in std[std.arm == "native"].itertuples():
                rows.append(dict(env=env, measure="native_goal", cohort=r.cohort, seed=r.seed, site="a_hid", value=r.goal))
    return rows


def summarize(long):
    rows = []
    for (env, measure, site), g in long.groupby(["env", "measure", "site"]):
        r = dict(env=env, measure=measure, site=site)
        for coh in ("gm", "control", "init"):
            v = g[g.cohort == coh].value.to_numpy()
            if len(v):
                m, lo, hi = C.ci95(v)
                r.update({f"{coh}": m, f"{coh}_lo": lo, f"{coh}_hi": hi, f"{coh}_n": len(v)})
        gm, ct, ini = (g[g.cohort == c].value.to_numpy() for c in ("gm", "control", "init"))
        if len(gm) > 1 and len(ct) > 1:
            r["diff"], r["diff_lo"], r["diff_hi"] = welch(gm, ct)
            r["ratio"], r["ratio_lo"], r["ratio_hi"] = log_ratio(gm, ct)
        if len(ini) > 1 and len(ct) > 1:
            r["init_ratio"], r["init_ratio_lo"], r["init_ratio_hi"] = log_ratio(ini, ct)
        if len(ini) > 1 and len(gm) > 1:
            r["gm_minus_init"], r["gm_minus_init_lo"], r["gm_minus_init_hi"] = welch(gm, ini)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, color=P["ink"])


def fig_compact(summ, envs, name):
    """One panel per env: each measure as a ratio to the env's own Non-GM agents (1 = parity, 0 = goal absent);
    burgundy = GM / Non-GM, grey = init trunk / Non-GM (architecture floor)."""
    lines = [("decode_pos_linear", "goal decodability", "-", "o"), ("rel_edit_goal", "goal-edit size", "--", "s"),
             ("rewire_goal", "rewired goal reach", ":", "^")]
    fig, axes = plt.subplots(1, len(envs), figsize=(2.3 * len(envs), 2.45), sharey=True)
    axes = np.atleast_1d(axes)
    sites = SITES[:4]  # no rewiring from the actor layer, and init's actor layer is mostly the proprio branch
    x = np.arange(len(sites))
    ylim = (-0.05, 1.5)
    for ax, env in zip(axes, envs):
        s = summ[summ.env == env]
        for meas, lab, ls, mk in lines:
            g = s[s.measure == meas].set_index("site").reindex(sites)
            for key, col, filled in (("ratio", P["gm"], True), ("init_ratio", P["init"], False)):
                if key not in g or g[key].notna().sum() == 0:
                    continue
                # values above the axis are drawn on the top edge (init rel_edit in AT is 2.4-3.1 at conv1-2)
                y = np.clip(g[key].to_numpy(float), *ylim)
                lo, hi = (np.clip(g[f"{key}_{b}"].to_numpy(float), *ylim) for b in ("lo", "hi"))
                ax.plot(x, y, ls=ls, marker=mk, color=col, lw=1.2, ms=4, mfc=col if filled else "white")
                ax.fill_between(x, lo, hi, color=col, alpha=0.10, lw=0)
        ax.axhline(1, color=P["mute"], lw=0.6)
        ax.axhline(0, color=P["mute"], lw=0.4, ls=":")
        ax.set_xticks(x, [SITE_LAB[t] for t in sites], rotation=35, ha="right", fontsize=7)
        ax.set_title(ENVS[env], fontsize=9)
        ax.set_ylim(*ylim)
        style(ax)
    axes[0].set_ylabel("ratio to Non-GM", fontsize=8)
    h = [plt.Line2D([], [], color=P["ink"], ls=ls, marker=mk, ms=4, lw=1.2) for _, _, ls, mk in lines]
    h += [plt.Line2D([], [], color=P["gm"], lw=2), plt.Line2D([], [], color=P["init"], lw=2, marker="o", mfc="white")]
    fig.legend(h, [l[1] for l in lines] + ["GM", "init"], loc="lower center", ncol=5, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, name)


def fig_grid(long, envs, name):
    """Rows = measure family, columns = env; absolute values per cohort with 95% t tunnels (GM, Non-GM, init)."""
    rows = [("decode_pos_MLP", r"goal position $R^2$ (MLP)", "decode_pos_linear"),
            ("decode_side_MLP", "goal side AUC - 0.5 (MLP)", "decode_side_linear"),
            ("rel_edit_goal", r"goal-edit $|\Delta|^2$ / scene$^2$", None),
            ("G_goal", "goal-edit G", None),
            ("rewire_goal", "rewired goal reach", "rewire_masked")]
    fig, axes = plt.subplots(len(rows), len(envs), figsize=(2.5 * len(envs), 1.75 * len(rows)), squeeze=False)
    x = np.arange(len(SITES))
    for i, (meas, ylab, second) in enumerate(rows):
        for j, env in enumerate(envs):
            ax = axes[i, j]
            for m, ls, filled in ((meas, "-", True), (second, "--", False)):
                if m is None:
                    continue
                d = long[(long.env == env) & (long.measure == m)]
                for coh, col in (("gm", P["gm"]), ("control", P["nongm"]), ("init", P["init"])):
                    g = d[d.cohort == coh]
                    if not len(g):
                        continue
                    st = [C.ci95(g[g.site == s].value.to_numpy()) if (g.site == s).any() else (np.nan,) * 3 for s in SITES]
                    y, lo, hi = (np.array([t[k] for t in st]) for k in range(3))
                    ax.plot(x, y, ls=ls, marker="o", ms=3, color=col, lw=1.1, mfc=col if filled else "white")
                    ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
            if meas in ("G_goal",):
                ax.set_yscale("log")
            ax.set_xticks(x, [SITE_LAB[t] for t in SITES] if i == len(rows) - 1 else [""] * len(x), rotation=35,
                          ha="right", fontsize=7)
            if i == 0:
                ax.set_title(ENVS[env], fontsize=9)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=7)
            style(ax)
    h = [plt.Line2D([], [], color=c, marker="o", ms=3) for c in (P["gm"], P["nongm"], P["init"])]
    h += [plt.Line2D([], [], color=P["ink"], ls="--", marker="o", ms=3, mfc="white")]
    fig.legend(h, ["GM", "Non-GM", "init", "linear probe / goal cue masked"], loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    save(fig, name)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_md(summ, path):
    keep = ["decode_pos_linear", "decode_pos_MLP", "decode_side_linear", "decode_side_MLP", "rel_edit_goal", "Q_goal",
            "G_goal", "calib_goal", "rel_edit_proxy", "rewire_goal", "rewire_masked", "rewire_excess", "rewire_nocue",
            "rewire_goal_origbank", "rewire_masked_origbank", "rewire_excess_origbank", "rewire_nocue_origbank",
            "native_goal"]
    f = lambda r, c: C.fmt_ci(r[c], r[c + "_lo"], r[c + "_hi"]) if c in r and np.isfinite(r[c]) else ""
    lines = ["# Depth map: mean [95% t-CI] over seeds; diff = GM - Non-GM (Welch), ratio = GM / Non-GM (Welch on log)", ""]
    for env in ENVS:
        s = summ[summ.env == env]
        if not len(s):
            continue
        lines += [f"## {ENVS[env]}", "", "| measure | site | GM | Non-GM | init | GM - Non-GM | GM / Non-GM | GM - init |",
                  "|---|---|---|---|---|---|---|---|"]
        for meas in keep:
            for site in SITES:
                r = s[(s.measure == meas) & (s.site == site)]
                if not len(r):
                    continue
                r = r.iloc[0]
                lines.append(f"| {meas} | {site} | {f(r, 'gm')} | {f(r, 'control')} | {f(r, 'init')} | {f(r, 'diff')} | "
                             f"{f(r, 'ratio')} | {f(r, 'gm_minus_init')} |")
        lines.append("")
    open(path, "w").write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=list(ENVS))
    a = p.parse_args()
    os.makedirs(RES, exist_ok=True)
    rows = []
    a.envs = [e for e in a.envs if os.path.exists(os.path.join(RES, f"fisher_{e}_per_seed.csv"))]
    for env in a.envs:
        rows += probe_rows(env) + fisher_rows(env) + rewire_rows(env)
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(RES, "depth_per_seed.csv"), index=False)
    summ = summarize(long)
    summ.to_csv(os.path.join(RES, "depth_summary.csv"), index=False)
    write_md(summ, os.path.join(RES, "depth_summary.md"))
    fig_compact(summ, a.envs, "fig_depth_map")
    fig_grid(long, a.envs, "fig_depth_grid")


if __name__ == "__main__":
    main()
