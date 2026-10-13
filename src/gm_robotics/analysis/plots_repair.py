"""Paper-style figures for h_fc steering (dose for AT / SZ / LR / PTM, compression) and LocationReach rewiring."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C
from gm_robotics.analysis.plots_paper import P, ci, save

ENVS = {"AT": "AttributeTable", "SZ": "SortZones", "LR": "LocationReach", "PTM": "PushToMarker"}
COH = (("gm", "GM"), ("control", "Non-GM"))
HUE = {"gm": (P["gm"], P["gm_proxy"]), "control": (P["nongm"], P["nongm_proxy"])}


def per_seed(name):
    f = os.path.join(C.RESULTS, name)
    return pd.read_csv(f) if os.path.exists(f) else None




def tunnel(ax, x, ys, color, ls="-", marker="o", filled=True, label=None):
    m = np.array([ci(y)[0] for y in ys])
    lo = np.array([ci(y)[1] for y in ys])
    hi = np.array([ci(y)[2] for y in ys])
    ax.plot(x, m, ls=ls, color=color, marker=marker, ms=3.5, mfc=color if filled else "white", mec=color, lw=1.4,
            label=label)
    ax.fill_between(x, m - lo, m + hi, color=color, alpha=0.15, lw=0)


def fig_dose(envs=("AT", "SZ", "LR", "PTM")):
    """Local h_fc steering, goal reach vs dose (corrective, reversed, norm-matched random), GM and Non-GM, 95% t-CI
    over seeds. PTM goal = cube on the marker, proxy = cube at the table centre (final state)."""
    data = {e: per_seed(f"steer_{e}_per_seed.csv") for e in envs}
    envs = [e for e in envs if data[e] is not None]
    if not envs:
        return
    # Plot only the doses every env shares (PTM also has beta = 4).
    betas = sorted(set.intersection(*[set(data[e][(data[e].arm == "local") & (data[e].beta > 0)].beta.round(6))
                                      for e in envs]))
    fig, axes = plt.subplots(2, len(envs), figsize=(3.0 * len(envs), 4.8), squeeze=False)
    for j, e in enumerate(envs):
        df = data[e]
        for i, (coh, clab) in enumerate(COH):
            ax = axes[i, j]
            d = df[df.cohort == coh]
            base = d[d.arm == "baseline"].reach
            bm, blo, bhi = ci(base)
            ax.axhspan(bm - blo, bm + bhi, color=P["init"], alpha=0.18, lw=0)
            ax.axhline(bm, color=P["mute"], ls=":", lw=1)
            g = lambda arm, sgn: [d[(d.arm == arm) & np.isclose(d.beta, sgn * b)].reach for b in betas]
            # drawn back to front so the corrective curve stays visible where all arms sit at 0 (LR / PTM GM)
            tunnel(ax, betas, g("random", 1), P["init"], marker="s", label="random")
            tunnel(ax, betas, g("local", -1), HUE[coh][1], ls="--", filled=False, label="reversed")
            tunnel(ax, betas, g("local", 1), HUE[coh][0], label="corrective")
            shown = d[d.arm.isin(["local", "random"]) & np.isclose(d.beta.abs().to_numpy()[:, None], betas).any(1)]
            top = max(shown.groupby(["arm", "beta"]).reach.mean().max(), bm)
            if top < 0.05:
                ax.text(0.5, 0.5, f"goal reach $\\leq$ {top:.3f}\nin every arm and dose shown", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7.5, color=P["mute"])
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(ENVS[e], fontsize=10)
            if i == 1:
                ax.set_xlabel(r"dose $\beta$")
            if j == 0:
                ax.set_ylabel(f"{clab} goal reach")
            if i == 0 and j == 0:
                h, l = ax.get_legend_handles_labels()
                ax.legend(h[::-1], l[::-1], fontsize=7, loc="best")
    fig.tight_layout()
    save(fig, "fig_steer_local_dose")


def fig_compression(envs=("AT", "SZ")):
    data = {e: per_seed(f"steer_{e}_per_seed.csv") for e in envs}
    envs = [e for e in envs if data[e] is not None and (data[e].arm == "shared").any()]
    if not envs:
        return
    arms = [("fixed_local", "local, fixed", "-", "o"), ("shared", "sign-gated", "--", "s"),
            ("fisher_proj", "Fisher-projected", "-.", "^"), ("unsigned", "unsigned", ":", "D")]
    fig, axes = plt.subplots(2, len(envs), figsize=(3.6 * len(envs), 4.8), squeeze=False)
    for j, e in enumerate(envs):
        df = data[e]
        for i, (coh, clab) in enumerate(COH):
            ax = axes[i, j]
            d = df[df.cohort == coh]
            bm = d[d.arm == "baseline"].reach.mean()
            ax.axhline(bm, color=P["mute"], ls=":", lw=1)
            for arm, lab, ls, mk in arms:
                b = sorted(d[d.arm == arm].beta.unique())
                col = P["init"] if arm == "unsigned" else HUE[coh][0] if arm == "fixed_local" else HUE[coh][1]
                tunnel(ax, b, [d[(d.arm == arm) & np.isclose(d.beta, x)].reach for x in b], col, ls=ls, marker=mk,
                       filled=arm == "fixed_local", label=lab)
            b = sorted(d[d.arm == "local"].beta.unique())
            b = [x for x in b if abs(x) <= 1]
            tunnel(ax, b, [d[(d.arm == "local") & np.isclose(d.beta, x)].reach for x in b], P["ink"], ls="-",
                   marker=".", label="local, rebuilt")
            ax.axvspan(-1.05, -0.25, color=P["init"], alpha=0.08, lw=0)
            ax.axvspan(0.25, 1.05, color=P["init"], alpha=0.08, lw=0)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(ENVS[e], fontsize=10)
            if i == 1:
                ax.set_xlabel(r"signed dose $\beta$")
            if j == 0:
                ax.set_ylabel(f"{clab} goal reach")
            if i == 0 and j == len(envs) - 1:
                ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    save(fig, "fig_steer_compression")


def fig_rewire():
    df = per_seed("rewire_LR_per_seed.csv")
    if df is None:
        return
    arms = [("native", "native"), ("fc_head", r"head on $h_{fc}$"), ("pre", r"from $h_{pre}$"),
            ("remix", "random remix"), ("untrained", "untrained")]
    # in_region is omitted: its goal is the corner itself, so every first contact is a tie.
    sets = [("offpath", "Off-path goal"), ("out_region", "Off-corner goal")]
    fig, axes = plt.subplots(1, len(sets), figsize=(7.4, 3.2), sharey=True)
    for j, (st, lab) in enumerate(sets):
        ax = axes[j]
        for a_i, (arm, _) in enumerate(arms):
            for k, (coh, _) in enumerate(COH):
                s = df[(df.set == st) & (df.arm == arm) & (df.cohort == coh)]
                if not len(s):
                    continue
                x = a_i + (k - 0.5) * 0.4
                for v, c, off in ((s.reach, HUE[coh][0], -0.1), (s.proxy, HUE[coh][1], 0.1)):
                    m, lo, hi = ci(v)
                    ax.bar(x + off, m, 0.18, color=c, yerr=[[lo], [hi]], ecolor=P["ink"], capsize=1.5, lw=0)
        ax.set_xticks(range(len(arms)), [a[1] for a in arms], rotation=30, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("P(first contact)")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (P["gm"], P["gm_proxy"], P["nongm"], P["nongm_proxy"])]
    fig.legend(h, ["GM goal", "GM proxy", "Non-GM goal", "Non-GM proxy"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_rewire_LR")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["dose", "compression", "rewire"],
                   choices=["dose", "compression", "rewire"])
    a = p.parse_args()
    for w in a.which:
        dict(dose=fig_dose, compression=fig_compression, rewire=fig_rewire)[w]()


if __name__ == "__main__":
    main()
