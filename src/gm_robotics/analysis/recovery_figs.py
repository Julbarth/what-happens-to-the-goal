"""Steering and rewiring recovery for the ManiSkill agents: chance-control summary and dose, rewiring and chance figures."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C
from gm_robotics.analysis.plots_paper import P, ci

RES = C.RESULTS
OUT_RES = os.path.join(RES, "recovery")
OUT_FIG = os.path.join(C.FIGURES, "recovery")
ENVS = {"AT": "AttributeTable", "SZ": "SortZones", "LR": "LocationReach", "PTM": "PushToMarker"}
COH = (("gm", "GM"), ("control", "Non-GM"))
HUE = {"gm": (P["gm"], P["gm_proxy"]), "control": (P["nongm"], P["nongm_proxy"]), "init": (P["init"], P["init"])}


def save(fig, name):
    os.makedirs(OUT_FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fmt(v):
    m, lo, hi = ci(v)
    return f"{m:.3f} [{m - lo:.2f}, {m + hi:.2f}]"


# ---------------------------------------------------------------- steering chance controls

ARM_LAB = {("baseline", 0.0): "unsteered", ("local", 1.0): "local +1", ("local", -1.0): "local -1 (reversed)",
           ("local", 0.25): "local +0.25", ("random", 1.0): "random, same norm (3 dirs)",
           ("fixed_local", 1.0): "d(s0) held fixed +1", ("shared", 0.25): "sign-gated axis +0.25",
           ("shared", -0.25): "sign-gated axis -0.25", ("unsigned", 0.25): "unsigned axis +0.25"}


def chance_per_seed(env):
    f = os.path.join(OUT_RES, f"steer_chance_{env}_per_state.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d["is_u"], d["is_v"] = (d.outcome == "u").astype(float), (d.outcome == "v").astype(float)
    # random: average the three directions within a seed first, so n = seeds
    g = d.groupby(["env", "cond", "cohort", "seed", "arm", "beta"])
    ps = g.agg(P_u=("is_u", "mean"), P_v=("is_v", "mean"), n=("state", "size"),
               med_min_d_u=("min_d_u", "median"), med_min_d_v=("min_d_v", "median")).reset_index()
    ps["D"] = ps.P_u + ps.P_v
    ps["P_u_dec"] = ps.P_u / ps.D.where(ps.D > 0)
    ps.to_csv(os.path.join(OUT_RES, f"steer_chance_{env}_per_seed.csv"), index=False)
    return ps


def chance_summary(envs):
    lines = ["# Steering chance controls (h_fc; push built from the real conflict's swap copies)", "",
             "u = goal location of the base conflict (where the corrective push points), v = lure location.",
             "standard: goal cue at u, proxy cue at v. phantom: goal cue removed (AT two spheres, blue at u; SZ no cross).",
             "decoupled: goal cue moved to v (AT cube at v, sphere at u, colours unchanged; SZ cross on the green pad).",
             "Mean [95% t-CI] over seeds; random = mean of 3 norm-matched directions per seed. Deterministic actions.", ""]
    rows = []
    for env in envs:
        ps = chance_per_seed(env)
        if ps is None:
            continue
        lines += [f"## {ENVS[env]}", "", "| cohort | arm | standard P(u) / P(v) | phantom P(u) / P(v) | decoupled P(u) / P(v) |",
                  "|---|---|---|---|---|"]
        for coh, clab in COH:
            for (arm, beta), lab in ARM_LAB.items():
                cells = []
                for cond in ("standard", "phantom", "decoupled"):
                    s = ps[(ps.cohort == coh) & (ps.cond == cond) & (ps.arm == arm) & np.isclose(ps.beta, beta)]
                    if not len(s):
                        cells.append("")
                        continue
                    cells.append(f"{fmt(s.P_u)} / {s.P_v.mean():.3f}")
                    rows.append(dict(env=env, cohort=coh, cond=cond, arm=arm, beta=beta, n=len(s),
                                     P_u=s.P_u.mean(), P_u_lo=ci(s.P_u)[0] - ci(s.P_u)[1],
                                     P_u_hi=ci(s.P_u)[0] + ci(s.P_u)[2], P_v=s.P_v.mean()))
                lines.append(f"| {clab} | {lab} | " + " | ".join(cells) + " |")
        lines += ["", "Paired per-seed contrasts, P(u):", ""]
        for coh, clab in COH:
            for arm, beta in (("local", 1.0), ("fixed_local", 1.0), ("shared", 0.25)):
                w = ps[(ps.cohort == coh) & (ps.arm == arm) & np.isclose(ps.beta, beta)].pivot(index="seed", columns="cond",
                                                                                               values="P_u")
                b = ps[(ps.cohort == coh) & (ps.arm == "baseline")].pivot(index="seed", columns="cond", values="P_u")
                if {"standard", "phantom", "decoupled"} - set(w.columns):
                    continue
                lines.append(f"- {clab} {ARM_LAB[(arm, beta)]}: phantom - standard {fmt(w.phantom - w.standard)}; "
                             f"decoupled - standard {fmt(w.decoupled - w.standard)}; steering lift over unsteered: "
                             f"standard {fmt(w.standard - b.standard)}, phantom {fmt(w.phantom - b.phantom)}, "
                             f"decoupled {fmt(w.decoupled - b.decoupled)}")
        lines.append("")
    os.makedirs(OUT_RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_RES, "steer_chance_summary.csv"), index=False)
    open(os.path.join(OUT_RES, "steer_chance_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def fig_chance(envs):
    data = {e: chance_per_seed(e) for e in envs}
    envs = [e for e in envs if data[e] is not None]
    if not envs:
        return
    arms = [("baseline", 0.0), ("local", 1.0), ("local", -1.0), ("random", 1.0), ("fixed_local", 1.0), ("shared", 0.25)]
    xl = ["none", r"local $+1$", r"local $-1$", "random", r"$d(s_0)$ fixed", r"axis $+0.25$"]
    conds = [("standard", "goal cue at u"), ("phantom", "goal cue removed"), ("decoupled", "goal cue at v")]
    fig, axes = plt.subplots(2, len(envs) * 3, figsize=(3.1 * 3 * len(envs), 5.0), sharey=True, squeeze=False)
    for j, e in enumerate(envs):
        ps = data[e]
        for c_i, (cond, clab) in enumerate(conds):
            for i, (coh, rlab) in enumerate(COH):
                ax = axes[i, 3 * j + c_i]
                for a_i, (arm, beta) in enumerate(arms):
                    s = ps[(ps.cohort == coh) & (ps.cond == cond) & (ps.arm == arm) & np.isclose(ps.beta, beta)]
                    if not len(s):
                        continue
                    for v, col, off in ((s.P_u, HUE[coh][0], -0.19), (s.P_v, HUE[coh][1], 0.19)):
                        m, lo, hi = ci(v)
                        ax.bar(a_i + off, m, 0.36, color=col, yerr=[[lo], [hi]], ecolor=P["ink"], capsize=1.5, lw=0)
                        ax.scatter(np.full(len(v), a_i + off), v, s=5, color=P["ink"], zorder=3, lw=0)
                ax.set_xticks(range(len(arms)), xl, rotation=40, ha="right", fontsize=7)
                ax.set_ylim(0, 1.05)
                if i == 0:
                    ax.set_title(f"{ENVS[e]}, {clab}", fontsize=8.5)
                if 3 * j + c_i == 0:
                    ax.set_ylabel(f"{rlab}, P(first contact)")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (P["gm"], P["gm_proxy"], P["nongm"], P["nongm_proxy"])]
    fig.legend(h, ["GM, location u", "GM, location v", "Non-GM, location u", "Non-GM, location v"], loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=8)
    fig.tight_layout()
    save(fig, "fig_recovery_steer_chance")


# ---------------------------------------------------------------- dose curves (paper fig_main_sec4_1_dose layout)

def tunnel(ax, x, ys, color, ls="-", marker="o", filled=True, label=None):
    st = [ci(y) for y in ys]
    m = np.array([s[0] for s in st])
    ax.plot(x, m, ls=ls, color=color, marker=marker, ms=3.5, mfc=color if filled else "white", mec=color, lw=1.4,
            label=label)
    ax.fill_between(x, m - np.array([s[1] for s in st]), m + np.array([s[2] for s in st]), color=color, alpha=0.15, lw=0)


def fig_dose():
    envs = [e for e in ("AT", "SZ", "LR", "PTM") if os.path.exists(os.path.join(RES, f"steer_{e}_per_seed.csv"))]
    data = {e: pd.read_csv(os.path.join(RES, f"steer_{e}_per_seed.csv")) for e in envs}
    betas = sorted(set.intersection(*[set(data[e][(data[e].arm == "local") & (data[e].beta > 0)].beta.round(6))
                                      for e in envs]))
    ncol = len(envs) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(2.7 * ncol, 4.9), squeeze=False)
    for i, (coh, clab) in enumerate(COH):
        for j, e in enumerate(envs):
            ax = axes[i, j]
            d = data[e][data[e].cohort == coh]
            bm, blo, bhi = ci(d[d.arm == "baseline"].reach)
            ax.axhspan(bm - blo, bm + bhi, color=P["init"], alpha=0.18, lw=0)
            ax.axhline(bm, color=P["mute"], ls=":", lw=1)
            g = lambda arm, sgn: [d[(d.arm == arm) & np.isclose(d.beta, sgn * b)].reach for b in betas]
            tunnel(ax, betas, g("random", 1), P["init"], marker="s", label="random")
            tunnel(ax, betas, g("local", -1), HUE[coh][1], ls="--", filled=False, label="reversed")
            tunnel(ax, betas, g("local", 1), HUE[coh][0], label="corrective")
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(f"({chr(97 + j)}) {ENVS[e]}", fontsize=9.5)
            else:
                ax.set_xlabel(r"dose $\beta$")
            if j == 0:
                ax.set_ylabel(f"{clab} goal reach")
            if i == 0 and j == 0:
                hh, ll = ax.get_legend_handles_labels()
                ax.legend(hh[::-1], ll[::-1], fontsize=7, loc="center right")
        ax = axes[i, ncol - 1]
        d = data["AT"][data["AT"].cohort == coh]
        bm, blo, bhi = ci(d[d.arm == "baseline"].reach)
        ax.axhspan(bm - blo, bm + bhi, color=P["init"], alpha=0.18, lw=0)
        ax.axhline(bm, color=P["mute"], ls=":", lw=1)
        for arm, lab, ls, mk, col, fill in (("fixed_local", r"$d(s_0)$ fixed", "-", "o", HUE[coh][0], True),
                                            ("shared", "sign-gated", "--", "s", HUE[coh][1], True),
                                            ("fisher_proj", "Fisher-projected", "-.", "^", HUE[coh][1], False),
                                            ("unsigned", "unsigned", ":", "D", P["init"], False)):
            b = sorted(d[d.arm == arm].beta.unique())
            tunnel(ax, b, [d[(d.arm == arm) & np.isclose(d.beta, x)].reach for x in b], col, ls=ls, marker=mk,
                   filled=fill, label=lab)
        ax.set_ylim(-0.02, 1.02)
        if i == 0:
            ax.set_title(f"({chr(97 + ncol - 1)}) AttributeTable, shared axis", fontsize=9.5)
            ax.legend(fontsize=6.5, loc="upper left")
        else:
            ax.set_xlabel(r"signed dose $\beta$")
    fig.tight_layout()
    save(fig, "fig_recovery_steer_dose")


# ---------------------------------------------------------------- rewiring (paper fig_main_sec4_2 layout)

ARMS_RW = [("native", "native", None), ("untrained", "untrained head", None), ("remix", "remix", "267k"),
           ("fc_head", r"from $h_{fc}$", "267k"), ("pre", r"from $h_{pre}$", "2.6M")]


def rewire_long():
    """Protocol per task, as in rewire_all_table.py: AT ctrl2 (100 steps), SZ ctrl2, LR off-path, PTM control-set."""
    out = []
    at = pd.read_csv(os.path.join(RES, "rewire_at_deep", "per_seed.csv"))
    at = at[at.cond == "standard"]
    for _, r in at[(at.variant == "ctrl2") | ((at.variant == "base") & (at.arm == "native"))].iterrows():
        out.append(dict(env="AT", cohort=r.cohort, seed=r.seed, arm=r.arm, goal=r.P_u_100))
    for _, r in at[(at.variant == "dagger2") & (at.arm == "pre")].iterrows():
        out.append(dict(env="AT", cohort=r.cohort, seed=r.seed, arm="pre_dagger", goal=r.P_u_100))
    sz = pd.read_csv(os.path.join(RES, "rewire_sz_ctrl2", "per_seed.csv"))
    sz = sz[sz.cond == "standard"]
    for _, r in sz[(sz.variant == "ctrl2") | ((sz.variant == "base") & (sz.arm == "native"))].iterrows():
        out.append(dict(env="SZ", cohort=r.cohort, seed=r.seed, arm=r.arm, goal=r.P_u))
    ra = pd.read_csv(os.path.join(RES, "rewire_attr", "per_seed.csv"))
    for _, r in ra[(ra.arm == "untrained") & (ra.cond == "standard")].iterrows():
        out.append(dict(env=r.env, cohort=r.cohort, seed=r.seed, arm="untrained", goal=r.P_u))
    lr = pd.read_csv(os.path.join(RES, "rewire_LR_per_seed.csv"))
    for _, r in lr[lr.set == "offpath"].iterrows():
        out.append(dict(env="LR", cohort=r.cohort, seed=r.seed, arm=r.arm, goal=r.reach))
    lc = pd.read_csv(os.path.join(RES, "rewire_check", "per_seed.csv"))
    for _, r in lc[(lc.cohort == "init") & (lc.cond == "standard")].iterrows():
        out.append(dict(env="LR", cohort="init", seed=r.seed, arm=r.arm, goal=r.reach_scored))
    pt = pd.read_csv(os.path.join(RES, "ptm", "rewire_per_seed.csv"))
    for _, r in pt[(pt.set == "ctrlset") & (pt.cond == "standard")].iterrows():
        out.append(dict(env="PTM", cohort=r.cohort, seed=r.seed, arm=r.arm, goal=r.goal))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT_RES, "rewire_recovery_per_seed.csv"), index=False)
    return df


def fig_rewire():
    df = rewire_long()
    envs = ["AT", "SZ", "LR", "PTM"]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.3), sharey=True)
    lines = ["# Rewiring, final protocols (goal reach, mean [95% t-CI] over seeds)", "",
             "AT ctrl2 (100 steps, first touch), SZ ctrl2 (first pad entry), LR off-path (first contact), PTM control-set "
             "scenes (final cube state). Teacher = control agent k+1 (never the rewired agent).", "",
             "| env | arm | GM | Non-GM | init trunk |", "|---|---|---|---|---|"]
    for j, e in enumerate(envs):
        ax = axes[j]
        arms = ARMS_RW + ([("pre_dagger", r"$h_{pre}$ + DAgger", "2.6M")] if e == "AT" else [])
        for a_i, (arm, lab, npar) in enumerate(arms):
            cells = []
            for k, coh in enumerate(("gm", "control", "init")):
                s = df[(df.env == e) & (df.arm == arm) & (df.cohort == coh)].goal
                if not len(s):
                    cells.append("")
                    continue
                cells.append(fmt(s))
                if coh == "init" and arm == "native":
                    continue
                x = a_i + (k - 1) * 0.28
                m, lo, hi = ci(s)
                ax.errorbar(x, m, yerr=[[lo], [hi]], fmt="o", color=HUE[coh][0], ms=5, capsize=2, lw=1.2)
                ax.scatter(np.full(len(s), x + 0.08), s, s=7, color=HUE[coh][0], alpha=0.55, lw=0)
            lines.append(f"| {e} | {arm} | " + " | ".join(cells) + " |")
            if npar:
                ax.text(a_i, 1.04, npar, ha="center", fontsize=6.5, color=P["mute"])
        ax.set_xticks(range(len(arms)), [a[1] for a in arms], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(f"({chr(97 + j)}) {ENVS[e]}", fontsize=9.5, pad=12)
        ax.set_ylim(-0.03, 1.1)
    axes[0].set_ylabel(r"goal reach $\hat J^\star$")
    h = [plt.Line2D([], [], marker="o", ls="", color=HUE[c][0]) for c in ("gm", "control", "init")]
    fig.legend(h, ["GM", "Non-GM", "init trunk"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout()
    save(fig, "fig_recovery_rewire")
    open(os.path.join(OUT_RES, "rewire_recovery_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["chance", "dose", "rewire"], choices=["chance", "dose", "rewire"])
    p.add_argument("--envs", nargs="+", default=["AT", "SZ"])
    a = p.parse_args()
    os.makedirs(OUT_RES, exist_ok=True)
    if "chance" in a.which:
        chance_summary(a.envs)
        fig_chance(a.envs)
    if "dose" in a.which:
        fig_dose()
    if "rewire" in a.which:
        fig_rewire()


if __name__ == "__main__":
    main()
