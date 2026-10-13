"""Rewiring recovery in all eight environments: rewire arms per site against native GM and Non-GM bands and a chance line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style as S  # noqa: E402
from style import PAL  # noqa: E402
import app_rewire_all_envs as A  # noqa: E402

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES

GR = A.GR
WTF_CSV = sorted((ROBOTICS_ROOT / "results/isaac_analysis/closed_loop_fixed").glob(
    "rewire2_l*/closed_loop.csv"))
ORDER = ["Maze", "LR", "PTM", "Coin", "AM", "AT", "SZ", "WTF"]
TITLES = {"Maze": "Maze", "LR": "LocationReach", "PTM": "PushToMarker", "Coin": "Coin", "AM": "Attribute Maze",
          "AT": "AttributeTable", "SZ": "SortZones", "WTF": "WalkToFlag"}
# (x position, arm key); GM left of Non-GM inside each site, random trunk beside the h_pre pair
SLOTS = [(-0.22, "gm_pre"), (0.0, "nongm_pre"), (0.22, "init"),
         (0.89, "gm_remix"), (1.11, "nongm_remix"),
         (1.89, "gm_fc"), (2.11, "nongm_fc")]
SITES = [(0.0, r"$h_{\rm pre}$"), (1.0, "remix"), (2.0, r"$h_{\rm fc}$")]
AT_NONGM = A.AT_NONGM_SEEDS


def procgen() -> dict[str, dict[str, list[float]]]:
    ps = pd.read_csv(GR / "rewire_controls/per_seed.csv")

    def v(env, mode, key):
        return ps[(ps.env == env) & (ps["mode"] == mode) & (ps.key == key)].sort_values("seed_index").value.tolist()

    ref = GR / "rewire_controls/paper_ref"
    out = {}
    mz = {t: json.load(open(ref / f"maze_aligned_{t}_a_star.json"))["rows"] for t in ("block3", "fc")}

    def mzv(tap, prefix):
        return [r["rewired_eval"]["cheese_reach"] for k, r in mz[tap].items() if k.startswith(prefix)]

    m = "greedy (paper)"
    out["Maze"] = dict(native_gm=v("Maze", m, "native_gm"), native_nongm=v("Maze", m, "native_nongm"),
                       init=v("Maze", m, "init"), chance=v("Maze", m, "chance"),
                       gm_pre=mzv("block3", "gm_rewire"), nongm_pre=mzv("block3", "nongm_rewire"),
                       gm_remix=mzv("block3", "gm_random_proj"), nongm_remix=mzv("block3", "nongm_random_proj"),
                       gm_fc=mzv("fc", "gm_rewire"), nongm_fc=mzv("fc", "nongm_rewire"))
    cn = {t: json.load(open(ref / f"coin_{t}_teacher.json"))["groups"] for t in ("block3", "fc")}
    out["Coin"] = dict(native_gm=v("Coin", m, "native_gm"), native_nongm=v("Coin", m, "native_nongm"),
                       init=v("Coin", m, "init"), chance=v("Coin", m, "chance"),
                       gm_pre=cn["block3"]["gm_rewire"]["per_seed_success_rate"],
                       nongm_pre=cn["block3"]["nongm_rewire"]["per_seed_success_rate"],
                       gm_remix=cn["block3"]["gm_random_proj"]["per_seed_success_rate"],
                       nongm_remix=cn["block3"]["nongm_random_proj"]["per_seed_success_rate"],
                       gm_fc=cn["fc"]["gm_rewire"]["per_seed_success_rate"],
                       nongm_fc=cn["fc"]["nongm_rewire"]["per_seed_success_rate"])
    af = pd.read_csv(GR / "am_oracle_fix/per_seed.csv")
    af = af[(af.labels == "fixed") & (af.cohort == "black") & (af.stratum == "overall")]

    def amv(tap, role, arm):
        return af[(af.tap == tap) & (af.role == role) & (af.arm == arm)].sort_values("seed").R_goal.tolist()

    mm = "greedy, am_oracle_fix GM / Non-GM heads"
    e = "Attribute Maze (black, fixed oracle)"
    out["AM"] = dict(native_gm=v(e, mm, "native_gm"), native_nongm=v(e, mm, "native_nongm"), init=v(e, mm, "init"),
                     chance=v(e, mm, "chance"), gm_pre=amv("block3", "gm", "rewire"),
                     nongm_pre=amv("block3", "nongm", "rewire"), gm_remix=amv("block3", "gm", "random_proj"),
                     nongm_remix=amv("block3", "nongm", "random_proj"), gm_fc=amv("fc", "gm", "rewire"),
                     nongm_fc=amv("fc", "nongm", "rewire"))
    return out


def robots() -> dict[str, dict[str, list[float]]]:
    out = {}
    at = pd.read_csv(GR / "rewire_at_deep/per_seed.csv")
    at = at[at.cond == "standard"]
    c2 = at[at.variant == "ctrl2"]

    def atv(coh, arm):
        x = (at[(at.variant == "base") & (at.arm == "native") & (at.cohort == coh)] if arm == "native"
             else c2[(c2.cohort == coh) & (c2.arm == arm)])
        if coh == "control":
            x = x[x.seed.isin(AT_NONGM)]
        return x.sort_values("seed").P_u_100.tolist()

    g = c2[(c2.cohort == "gm") & (c2.arm == "pre")].sort_values("seed")
    out["AT"] = dict(native_gm=atv("gm", "native"), native_nongm=atv("control", "native"), init=atv("init", "pre"),
                     chance=(0.5 * (g.P_u_100 + g.P_v_100)).tolist(),
                     **{f"{c}_{k}": atv(coh, arm) for c, coh in (("gm", "gm"), ("nongm", "control"))
                        for k, arm in (("pre", "pre"), ("remix", "remix"), ("fc", "fc_head"))})
    sz = pd.read_csv(GR / "rewire_sz_ctrl2/per_seed.csv")
    sz = sz[sz.cond == "standard"]
    s2 = sz[sz.variant == "ctrl2"]

    def szv(coh, arm):
        x = sz[(sz.variant == "base") & (sz.arm == "native") & (sz.cohort == coh)] if arm == "native" else \
            s2[(s2.cohort == coh) & (s2.arm == arm)]
        return x.sort_values("seed").P_u.tolist()

    g = s2[(s2.cohort == "gm") & (s2.arm == "pre")].sort_values("seed")
    out["SZ"] = dict(native_gm=szv("gm", "native"), native_nongm=szv("control", "native"), init=szv("init", "pre"),
                     chance=(0.5 * (g.P_u + g.P_v)).tolist(),
                     **{f"{c}_{k}": szv(coh, arm) for c, coh in (("gm", "gm"), ("nongm", "control"))
                        for k, arm in (("pre", "pre"), ("remix", "remix"), ("fc", "fc_head"))})
    lr = pd.read_csv(GR / "rewire_check/per_seed.csv")
    lr = lr[lr.cond == "standard"]

    def lrv(coh, arm, col="reach_scored"):
        return lr[(lr.cohort == coh) & (lr.arm == arm)].sort_values("seed")[col].tolist()

    out["LR"] = dict(native_gm=lrv("gm", "native"), native_nongm=lrv("control", "native"), init=lrv("init", "pre"),
                     chance=lrv("gm", "pre", "p_phantom"),
                     **{f"{c}_{k}": lrv(coh, arm) for c, coh in (("gm", "gm"), ("nongm", "control"))
                        for k, arm in (("pre", "pre"), ("remix", "remix"), ("fc", "fc_head"))})
    pt = pd.read_csv(GR / "ptm/rewire_per_seed.csv")
    pt = pt[(pt.set == "conflict") & (pt.cond == "standard")]

    def ptv(coh, arm, col="goal"):
        return pt[(pt.cohort == coh) & (pt.arm == arm)].sort_values("seed")[col].tolist()

    out["PTM"] = dict(native_gm=ptv("gm", "native"), native_nongm=ptv("control", "native"), init=ptv("init", "pre"),
                      chance=ptv("gm", "pre", "phantom_pool"),
                      **{f"{c}_{k}": ptv(coh, arm) for c, coh in (("gm", "gm"), ("nongm", "control"))
                         for k, arm in (("pre", "pre"), ("remix", "remix"), ("fc", "fc_head"))})
    if WTF_CSV:
        # reach = first object reached is the flag (u), on the native-start conflict layouts, teacher targets only
        w = pd.concat([pd.read_csv(f) for f in WTF_CSV])
        w = w[w.set == "native"].assign(goal=lambda x: (x["first"] == "u").astype(float),
                                        tower=lambda x: (x["first"] == "v").astype(float))
        w = w.groupby(["cohort", "seed", "arm", "target"])[["goal", "tower"]].mean().reset_index()
        init = [c for c in w.cohort.unique() if c not in ("gm", "control")]

        def wv(coh, arm, tgt="teacher", col="goal"):
            return w[(w.cohort == coh) & (w.arm == arm) & (w.target == tgt)].sort_values("seed")[col].tolist()

        gp = np.array(wv("gm", "pre")) + np.array(wv("gm", "pre", col="tower"))
        out["WTF"] = dict(native_gm=wv("gm", "native", "none"), native_nongm=wv("control", "native", "none"),
                          init=wv(init[0], "pre") if init else [], chance=(0.5 * gp).tolist(),
                          **{f"{c}_{k}": wv(coh, arm) for c, coh in (("gm", "gm"), ("nongm", "control"))
                             for k, arm in (("pre", "pre"), ("remix", "remix"), ("fc", "fc_head"))})
    return out


def band(ax, vals, color, below: bool):
    m = float(np.mean(vals))
    ax.axhline(m, color=color, lw=S.LW_DATA, ls=(0, (4, 2)), zorder=1)
    if below:
        ax.axhspan(-0.03, m, color=color, alpha=0.08, lw=0, zorder=0)
    else:
        ax.axhspan(m, 1.03, color=color, alpha=0.08, lw=0, zorder=0)


def main(a):
    S.style()
    data = {**procgen(), **robots()}
    long = [dict(env=e, arm=k, seed_index=i, value=x) for e, d in data.items() for k, vs in d.items()
            for i, x in enumerate(vs)]
    pd.DataFrame(long).to_csv(TABLES / "fig_app_rewire_recovery_per_seed.csv", index=False)
    summ = [dict(env=e, arm=k, n=len(vs), mean=np.mean(vs), sd=np.std(vs, ddof=1) if len(vs) > 1 else np.nan)
            for e, d in data.items() for k, vs in d.items() if len(vs)]
    pd.DataFrame(summ).to_csv(TABLES / "fig_app_rewire_recovery_summary.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(2, 4, figsize=(S.W_IN, 3.9), sharey=True)
    rng = np.random.default_rng(0)
    col = {"gm": PAL["gm"], "nongm": PAL["nongm"], "init": PAL["init"]}
    for ax, env in zip(axes.flat, ORDER):
        S.panel_title(ax, TITLES[env])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.5, 2.4)
        d = data.get(env, {})
        if d.get("native_gm"):
            band(ax, d["native_gm"], PAL["gm"], below=True)
        if d.get("native_nongm"):
            band(ax, d["native_nongm"], PAL["nongm"], below=False)
        if d.get("chance"):
            ax.axhline(np.mean(d["chance"]), color=PAL["mute"], lw=S.LW_DATA, ls=(0, (1, 1.5)), zorder=1)
        for x, arm in SLOTS:
            vals = d.get(arm, [])
            if not vals:
                ax.text(x, 0.04, "no data", rotation=90, ha="center", va="bottom", fontsize=S.FS_TICK - 1.5,
                        color=PAL["mute"])
                continue
            c = col[arm.split("_")[0]]
            hollow = arm.endswith("remix")
            mk = "s" if arm == "init" else ("s" if hollow else "o")
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(vals)), vals, color=c, zorder=3, **S.SEED)
            mu, h = A.ci(vals)
            ax.plot([x, x], [max(mu - h, 0), min(mu + h, 1)], color=c, lw=S.LW_ERR, zorder=4)
            ax.plot(x, mu, mk, ms=S.MS + 0.8, color=c, mfc="white" if hollow else c, mew=1.0, zorder=5)
        ax.set_xticks([s for s, _ in SITES])
        ax.set_xticklabels([lab for _, lab in SITES])
        S.finish(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"goal reach $\hat J^\star$")
    h = [Line2D([], [], color=PAL["gm"], marker="o", ls="", ms=S.MS + 0.4, label="GM rewire"),
         Line2D([], [], color=PAL["nongm"], marker="o", ls="", ms=S.MS + 0.4, label="Non-GM rewire"),
         Line2D([], [], color=PAL["mute"], marker="s", mfc="white", ls="", ms=S.MS + 0.4, label="random remix"),
         Line2D([], [], color=PAL["init"], marker="s", ls="", ms=S.MS + 0.4, label="random trunk"),
         (Patch(color=PAL["gm"], alpha=0.12, lw=0), Line2D([], [], color=PAL["gm"], ls=(0, (4, 2)))),
         (Patch(color=PAL["nongm"], alpha=0.12, lw=0), Line2D([], [], color=PAL["nongm"], ls=(0, (4, 2)))),
         Line2D([], [], color=PAL["mute"], ls=(0, (1, 1.5)))]
    labels = ["GM rewire", "Non-GM rewire", "random remix", "random trunk", "native GM", "native Non-GM", "chance"]
    fig.legend(h, labels, loc="lower center", ncol=4, frameon=False, fontsize=S.FS_LEG, handletextpad=0.3,
               columnspacing=0.8, handlelength=1.8, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.16, wspace=0.12, hspace=0.62)
    for (r, c0, c1, lab) in ((0, 0, 1, "(i) goal lost in the representation"),
                             (0, 2, 3, "(ii) goal reaches the policy, unused"),
                             (1, 0, 3, "(iii) goal and proxy compete")):
        p0, p1 = axes[r, c0].get_position(), axes[r, c1].get_position()
        y = p0.y1 + 0.075
        fig.text((p0.x0 + p1.x1) / 2, y, lab, ha="center", va="bottom", fontsize=S.FS, fontweight="bold",
                 color=PAL["ink"])
        fig.add_artist(Line2D([p0.x0 + 0.01, p1.x1 - 0.01], [y - 0.008] * 2, color=PAL["ink"], lw=0.5))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"))
    fig.savefig(a.out.with_suffix(".png"), dpi=300)
    print("wrote", a.out.with_suffix(".pdf"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_DIR / "app_rewire/fig_app_rewire_recovery")
    main(p.parse_args())
