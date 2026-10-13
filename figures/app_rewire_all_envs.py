"""Rewiring across the eight environments: native GM, random trunk, GM and Non-GM rewires, chance, per seed with 95% CIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style as S  # noqa: E402
from style import PAL  # noqa: E402

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES

GR = (ROBOTICS_ROOT / "results/analysis")
# two-sided 97.5% t quantiles, tabulated so scipy is not needed
T975 = {1: 12.706, 2: 4.303, 3: 3.182}
ARMS = ["native_gm", "init", "gm_pre", "gm_fc", "nongm_pre", "nongm_fc"]
TITLES = {"Maze": r"$\bf{Maze}$", "LR": r"$\bf{LocationReach}$", "PTM": r"$\bf{PushToMarker}$", "Coin": r"$\bf{Coin}$",
          "AM": r"$\bf{Attribute\ Maze}$", "AT": r"$\bf{AttributeTable}$", "SZ": r"$\bf{SortZones}$",
          "WTF": r"$\bf{WalkToFlag}$"}
ORDER = ["Maze", "LR", "PTM", "Coin", "AM", "AT", "SZ", "WTF"]
AT_NONGM_SEEDS = [3, 4, 5]


def procgen() -> list[dict]:
    rows = []
    ps = pd.read_csv(GR / "rewire_controls/per_seed.csv")

    def v(env, mode, key):
        return ps[(ps.env == env) & (ps["mode"] == mode) & (ps.key == key)].sort_values("seed_index").value.tolist()

    ref = GR / "rewire_controls/paper_ref"
    mz = {t: json.load(open(ref / f"maze_aligned_{t}_a_star.json"))["rows"] for t in ("block3", "fc")}

    def mzv(tap, prefix):
        return [r["rewired_eval"]["cheese_reach"] for k, r in mz[tap].items() if k.startswith(prefix)]

    m = "greedy (paper)"
    d = dict(native_gm=v("Maze", m, "native_gm"), init=v("Maze", m, "init"), gm_pre=mzv("block3", "gm_rewire"),
             gm_fc=mzv("fc", "gm_rewire"), nongm_pre=mzv("block3", "nongm_rewire"), nongm_fc=mzv("fc", "nongm_rewire"),
             chance=v("Maze", m, "chance"))
    rows.append(("Maze", d))
    cn = {t: json.load(open(ref / f"coin_{t}_teacher.json"))["groups"] for t in ("block3", "fc")}
    d = dict(native_gm=v("Coin", m, "native_gm"), init=v("Coin", m, "init"),
             gm_pre=cn["block3"]["gm_rewire"]["per_seed_success_rate"], gm_fc=cn["fc"]["gm_rewire"]["per_seed_success_rate"],
             nongm_pre=cn["block3"]["nongm_rewire"]["per_seed_success_rate"],
             nongm_fc=cn["fc"]["nongm_rewire"]["per_seed_success_rate"], chance=v("Coin", m, "chance"))
    rows.append(("Coin", d))
    af = pd.read_csv(GR / "am_oracle_fix/per_seed.csv")
    af = af[(af.labels == "fixed") & (af.cohort == "black") & (af.stratum == "overall")]

    def amv(tap, role):
        return af[(af.tap == tap) & (af.role == role) & (af.arm == "rewire")].sort_values("seed").R_goal.tolist()

    mm = "greedy, am_oracle_fix GM / Non-GM heads"
    d = dict(native_gm=v("Attribute Maze (black, fixed oracle)", mm, "native_gm"),
             init=v("Attribute Maze (black, fixed oracle)", mm, "init"), gm_pre=amv("block3", "gm"), gm_fc=amv("fc", "gm"),
             nongm_pre=amv("block3", "nongm"), nongm_fc=amv("fc", "nongm"),
             chance=v("Attribute Maze (black, fixed oracle)", mm, "chance"))
    rows.append(("AM", d))
    return rows


def robots() -> list[dict]:
    rows = []
    at = pd.read_csv(GR / "rewire_at_deep/per_seed.csv")
    at = at[at.cond == "standard"]
    c2 = at[at.variant == "ctrl2"]

    def atv(coh, arm, seeds=None):
        x = (at[(at.variant == "base") & (at.arm == "native") & (at.cohort == coh)] if arm == "native"
             else c2[(c2.cohort == coh) & (c2.arm == arm)])
        if seeds is not None:
            x = x[x.seed.isin(seeds)]
        return x.sort_values("seed").P_u_100.tolist()

    g = c2[(c2.cohort == "gm") & (c2.arm == "pre")].sort_values("seed")
    rows.append(("AT", dict(native_gm=atv("gm", "native"), init=atv("init", "pre"), gm_pre=atv("gm", "pre"),
                            gm_fc=atv("gm", "fc_head"), nongm_pre=atv("control", "pre", AT_NONGM_SEEDS),
                            nongm_fc=atv("control", "fc_head", AT_NONGM_SEEDS),
                            native_nongm=atv("control", "native", AT_NONGM_SEEDS),
                            chance=(0.5 * (g.P_u_100 + g.P_v_100)).tolist())))
    sz = pd.read_csv(GR / "rewire_sz_ctrl2/per_seed.csv")
    sz = sz[sz.cond == "standard"]
    s2 = sz[sz.variant == "ctrl2"]

    def szv(coh, arm):
        x = sz[(sz.variant == "base") & (sz.arm == "native") & (sz.cohort == coh)] if arm == "native" else \
            s2[(s2.cohort == coh) & (s2.arm == arm)]
        return x.sort_values("seed").P_u.tolist()

    g = s2[(s2.cohort == "gm") & (s2.arm == "pre")].sort_values("seed")
    rows.append(("SZ", dict(native_gm=szv("gm", "native"), init=szv("init", "pre"), gm_pre=szv("gm", "pre"),
                            gm_fc=szv("gm", "fc_head"), nongm_pre=szv("control", "pre"), nongm_fc=szv("control", "fc_head"),
                            native_nongm=szv("control", "native"), chance=(0.5 * (g.P_u + g.P_v)).tolist())))
    lr = pd.read_csv(GR / "rewire_check/per_seed.csv")
    lr = lr[lr.cond == "standard"]

    def lrv(coh, arm, col="reach_scored"):
        return lr[(lr.cohort == coh) & (lr.arm == arm)].sort_values("seed")[col].tolist()

    rows.append(("LR", dict(native_gm=lrv("gm", "native"), init=lrv("init", "pre"), gm_pre=lrv("gm", "pre"),
                            gm_fc=lrv("gm", "fc_head"), nongm_pre=lrv("control", "pre"), nongm_fc=lrv("control", "fc_head"),
                            native_nongm=lrv("control", "native"), chance=lrv("gm", "pre", "p_phantom"))))
    pt = pd.read_csv(GR / "ptm/rewire_per_seed.csv")
    pt = pt[(pt.set == "conflict") & (pt.cond == "standard")]

    def ptv(coh, arm, col="goal"):
        return pt[(pt.cohort == coh) & (pt.arm == arm)].sort_values("seed")[col].tolist()

    rows.append(("PTM", dict(native_gm=ptv("gm", "native"), init=ptv("init", "pre"), gm_pre=ptv("gm", "pre"),
                             gm_fc=ptv("gm", "fc_head"), nongm_pre=ptv("control", "pre"), nongm_fc=ptv("control", "fc_head"),
                             native_nongm=ptv("control", "native"), chance=ptv("gm", "pre", "phantom_pool"))))
    wtf = sorted((GR.parent / "isaac_analysis/closed_loop_fixed").glob("rewire*/closed_loop.csv"))
    if wtf:
        # reach is the first object reached on the native-start conflict test, flag position u before tower v
        w = pd.concat([pd.read_csv(f) for f in wtf])
        w = w[w.set == "native"].assign(goal=lambda x: (x["first"] == "u").astype(float),
                                        tower=lambda x: (x["first"] == "v").astype(float))
        w = w.groupby(["cohort", "seed", "arm", "target"])[["goal", "tower"]].mean().reset_index()
        init = [c for c in w.cohort.unique() if c not in ("gm", "control")]

        def wv(coh, arm, tgt, col="goal"):
            return w[(w.cohort == coh) & (w.arm == arm) & (w.target == tgt)].sort_values("seed")[col].tolist()

        gp = np.array(wv("gm", "pre", "teacher")) + np.array(wv("gm", "pre", "teacher", "tower"))
        rows.append(("WTF", dict(native_gm=wv("gm", "native", "none"),
                                 init=wv(init[0], "pre", "teacher") if init else [],
                                 gm_pre=wv("gm", "pre", "teacher"), gm_fc=wv("gm", "fc_head", "teacher"),
                                 nongm_pre=wv("control", "pre", "teacher"), nongm_fc=wv("control", "fc_head", "teacher"),
                                 chance=(0.5 * gp).tolist())))
    return rows


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), 0.0
    return x.mean(), T975[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x))


def main(a):
    S.style()
    data = dict(procgen() + robots())
    long = [dict(env=e, arm=k, seed_index=i, value=v) for e, d in data.items() for k, vs in d.items() for i, v in enumerate(vs)]
    pd.DataFrame(long).to_csv(TABLES / "fig_app_rewire_all8_per_seed.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(S.W_IN, 3.8), sharey=True)
    xs = np.arange(len(ARMS))
    sty = {"native_gm": dict(c=PAL["gm"], m="o", open=True), "init": dict(c=PAL["init"], m="o", open=False),
           "gm_pre": dict(c=PAL["gm"], m="s", open=False), "gm_fc": dict(c=PAL["gm"], m="^", open=False),
           "nongm_pre": dict(c=PAL["nongm"], m="s", open=False), "nongm_fc": dict(c=PAL["nongm"], m="^", open=False)}
    rng = np.random.default_rng(0)
    for ax, env in zip(axes.flat, ORDER):
        S.panel_title(ax, TITLES[env])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.axvline(1.5, color=PAL["mute"], lw=0.4, alpha=0.5)
        S.finish(ax)
        d = data.get(env)
        if d is None:
            ax.text(len(ARMS) / 2 - 0.5, 0.5, "no data", ha="center", va="center", color=PAL["mute"], fontsize=S.FS)
            continue
        if d.get("chance"):
            ax.axhline(np.mean(d["chance"]), **S.REF)
        for x, arm in zip(xs, ARMS):
            vals = d.get(arm, [])
            if not vals:
                continue
            st = sty[arm]
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, color=st["c"], **S.SEED)
            m, h = ci(vals)
            ax.plot([x, x], [max(m - h, 0), min(m + h, 1)], color=st["c"], lw=S.LW_ERR)
            ax.plot(x, m, st["m"], ms=S.MS + 0.6, color=st["c"], mfc="white" if st["open"] else st["c"], mew=1.0)
    for ax in axes[:, 0]:
        ax.set_ylabel("goal reach")
    h = [Line2D([], [], color=PAL["gm"], marker="o", mfc="white", ls="", ms=S.MS, label="GM native"),
         Line2D([], [], color=PAL["init"], marker="o", ls="", ms=S.MS, label=r"random trunk, $h_{\rm pre}$"),
         Line2D([], [], color=PAL["gm"], marker="s", ls="", ms=S.MS, label=r"GM rewire, $h_{\rm pre}$"),
         Line2D([], [], color=PAL["gm"], marker="^", ls="", ms=S.MS, label=r"GM rewire, $h_{\rm fc}$"),
         Line2D([], [], color=PAL["nongm"], marker="s", ls="", ms=S.MS, label=r"Non-GM rewire, $h_{\rm pre}$"),
         Line2D([], [], color=PAL["nongm"], marker="^", ls="", ms=S.MS, label=r"Non-GM rewire, $h_{\rm fc}$"),
         Line2D([], [], **S.REF, label="chance")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=S.FS_LEG, handletextpad=0.3,
               columnspacing=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.86, bottom=0.14, wspace=0.12, hspace=0.55)
    for (r, c0, c1, lab) in ((0, 0, 1, "(i) goal lost in the representation"), (0, 2, 3, "(ii) goal reaches the policy, unused"),
                             (1, 0, 3, "(iii) goal and proxy compete")):
        p0, p1 = axes[r, c0].get_position(), axes[r, c1].get_position()
        fig.text((p0.x0 + p1.x1) / 2, p0.y1 + 0.075, lab, ha="center", va="bottom", fontsize=S.FS, fontweight="bold",
                 color=PAL["ink"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"))
    fig.savefig(a.out.with_suffix(".png"), dpi=300)
    print("wrote", a.out.with_suffix(".pdf"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_DIR / "app_rewire/fig_app_rewire_all8")
    main(p.parse_args())
