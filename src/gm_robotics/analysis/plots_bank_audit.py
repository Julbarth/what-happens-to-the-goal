"""Paper-style figures for bank_audit.py: geometry per bank variant, bank layouts and robot push-pull."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from gm_robotics.analysis import common as C
from gm_robotics.analysis import plots_paper as PP

P = C.PAL
AUD = os.path.join(C.RESULTS, "bank_audit")
COH = (("gm", P["gm"], "GM"), ("control", P["nongm"], "Non-GM"), ("init", P["init"], "init"))
BANK_LABEL = {"orig": "original\nt=0", "orig_wide": "original\nwide", "strict": "strict\nt=0",
              "decision": "decision\nframes", "decision_gmsrc": "decision\nGM src", "decision_ctrlsrc": "decision\nNon-GM src"}
ENVS = {"AT": "AttributeTable", "SZ": "SortZones"}


def points(ax, df, col, banks, open_marker=False, dx0=0.0):
    for (coh, c, _), dx in zip(COH, (-0.22, 0.0, 0.22)):
        for x, bk in enumerate(banks):
            v = df[(df.bank == bk) & (df.cohort == coh)][col].to_numpy()
            if not len(v):
                continue
            m, lo, hi = PP.ci(v)
            xx = x + dx + dx0
            ax.scatter(np.full(len(v), xx - 0.05), v, s=9, color=c, alpha=0.35, lw=0)
            ax.errorbar(xx, m, yerr=[[lo], [hi]], fmt="o", ms=5.5, color=c, mfc="white" if open_marker else c,
                        mew=1.4, capsize=0, lw=1.5)


FINAL_BANKS = ("strict", "decision", "decision_gmsrc", "decision_ctrlsrc")


def geometry_figure(env, final=False):
    f = os.path.join(AUD, f"geometry_{env}_per_seed.csv")
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    banks = [b for b in BANK_LABEL if b in set(df.bank) and (not final or b in FINAL_BANKS)]
    panels = [("cosF_within", r"$\cos_F(\Delta_\star,\Delta_p)$", (-1.05, 1.05)),
              ("cosF_tcpsub_within", r"$\cos_F$, TCP-moving subspace", (-1.05, 1.05)),
              ("cosF_same", r"$\kappa_F$ same side", (-0.1, 1.05)),
              ("heldout_align", r"held-out $\cos_E(\sigma d_{comp}, u_{comp})$", (-0.1, 1.05)),
              ("pull_ratio", "colour / goal-cue pull", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.2))
    for k, (ax, (col, lab, yl)) in enumerate(zip(axes, panels)):
        points(ax, df, col, banks)
        if col == "pull_ratio":
            sub = df.copy()
            sub["bank"] = sub.bank.str.replace("_pxmatched", "", regex=False).where(sub.bank.str.endswith("_pxmatched"), "x")
            points(ax, sub, col, banks, open_marker=True, dx0=0.08)
            ax.set_yscale("log")
            ax.axhline(1, color=P["mute"], lw=0.8, ls="--")
        else:
            ax.axhline(0, color=P["init"], lw=0.6)
            ax.set_ylim(*yl)
        ax.set_xticks(range(len(banks)), [BANK_LABEL[b] for b in banks], fontsize=7)
        ax.set_xlim(-0.5, len(banks) - 0.5)
        ax.set_ylabel(lab, fontsize=9)
        PP.panel_label(ax, k, "")
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for _, c, l in COH]
    h += [Line2D([], [], marker="o", ls="", color=P["mute"], mfc="white", label="pixel-matched edits")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(ENVS[env], fontsize=10, y=1.0)
    fig.tight_layout()
    PP.save(fig, f"fig_geometry_final_{env}" if final else f"fig_bank_audit_geometry_{env}")


def layout_figure(final=False):
    from gm_robotics.analysis import bank_audit as BA
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for r, env in enumerate(("AT", "SZ")):
        variants = (("strict", "--"), ("decision", ":")) if final else (("orig", "-"), ("strict", "--"), ("decision", ":"))
        for variant, ls in variants:
            f = BA.bank_path(env, variant)
            if not os.path.exists(f):
                continue
            b = C.load_bank(os.path.basename(f)[:-4])
            L = BA.layout(env, b)
            g, p = BA.GOAL_E[env], BA.PROX_E[env]
            lab = {"orig": "original", "strict": "strict", "decision": "decision"}[variant]
            axes[r, 0].hist(L["angle"], bins=np.linspace(0, 180, 37), histtype="step", color=P["ink"], ls=ls, lw=1.4, label=lab, density=True)
            axes[r, 1].hist(L["ddist"] * 100, bins=np.linspace(0, 30, 31), histtype="step", color=P["ink"], ls=ls, lw=1.4, density=True)
            axes[r, 2].scatter(b[f"pxl2_{g}"], b[f"pxl2_{p}"], s=4, alpha=0.4, color=P["ink"] if variant == "orig" else
                               (P["mute"] if variant == "strict" else P["init"]), label=lab, lw=0)
        axes[r, 0].set_xlabel("angle u-ref-v (deg)"), axes[r, 1].set_xlabel("|d(u) - d(v)| (cm)")
        lim = axes[r, 2].get_xlim()[1]
        axes[r, 2].plot([0, lim], [0, lim], "--", color=P["mute"], lw=0.8)
        axes[r, 2].set_xlabel(r"goal-cue edit pixel $\ell_2$"), axes[r, 2].set_ylabel(r"colour edit pixel $\ell_2$")
        axes[r, 0].set_ylabel(ENVS[env])
        axes[r, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    PP.save(fig, "fig_bank_layouts_final" if final else "fig_bank_audit_layouts")


def push_pull_figure():
    """Paper fig_main_sec3_3 (c) analogue: S_fav vs S_comp per swap at start (strict t=0) and decision frames."""
    pp = os.path.join(C.RESULTS, "push_pull")
    envs = [e for e in ("AT", "SZ") if os.path.exists(os.path.join(pp, f"{e}_per_state.csv"))]
    if not envs:
        return
    swap = {"AT": (("shape_swap", "shape swap"), ("colour_swap", "colour swap")),
            "SZ": (("mark_swap", "cross swap"), ("colour_swap", "colour swap"))}
    banks = (("strict", "start"), ("decision", "decision frame"))
    fig, axes = plt.subplots(len(envs), 4, figsize=(13, 3.3 * len(envs)), squeeze=False)
    rng = np.random.default_rng(0)
    for r, env in enumerate(envs):
        st = pd.read_csv(os.path.join(pp, f"{env}_per_state.csv"))
        ps = pd.read_csv(os.path.join(pp, f"{env}_per_seed.csv"))
        lim = np.nanquantile(np.abs(st[st.cohort != "init"][["S_fav", "S_comp"]].to_numpy()), 0.98) * 1.1
        for j, ((bk, btxt), (ed, etxt)) in enumerate([(b, e) for b in banks for e in swap[env]]):
            ax = axes[r, j]
            ax.add_patch(plt.Rectangle((0, -lim), lim, lim, color="#F2F2F2", zorder=0))
            ax.axhline(0, color=P["mute"], ls="--", lw=0.8), ax.axvline(0, color=P["mute"], ls="--", lw=0.8)
            for coh, col in (("control", P["nongm"]), ("gm", P["gm"])):
                sub = st[(st.bank == bk) & (st.edit == ed) & (st.cohort == coh)]
                if len(sub) > 600:
                    sub = sub.iloc[rng.choice(len(sub), 600, replace=False)]
                ax.scatter(sub.S_fav, sub.S_comp, s=5, color=col, alpha=0.15, lw=0)
                sd = ps[(ps.bank == bk) & (ps.edit == ed) & (ps.cohort == coh)]
                ax.scatter(sd.S_fav, sd.S_comp, marker="D", s=40, color=col, ec="white", lw=0.8, zorder=4)
                mf, lf, hf = PP.ci(sd.S_fav)
                mc, lc, hc = PP.ci(sd.S_comp)
                ax.errorbar(mf, mc, xerr=[[lf], [hf]], yerr=[[lc], [hc]], fmt="o", ms=5, color=col, mfc="white",
                            mew=1.4, lw=1.4, zorder=5)
            ax.set_xlim(-lim * 0.45, lim), ax.set_ylim(-lim, lim * 0.45)
            ax.set_title(f"{etxt}, {btxt}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ENVS[env]}\n" + r"$S_{\mathrm{comp}}$")
            if r == len(envs) - 1:
                ax.set_xlabel(r"$S_{\mathrm{fav}}$")
            PP.panel_label(ax, r * 4 + j, ax.get_title())
    h = [Line2D([], [], marker="o", ls="", color=P["gm"], label="GM"),
         Line2D([], [], marker="o", ls="", color=P["nongm"], label="Non-GM"),
         Line2D([], [], marker="o", color=P["mute"], mfc="white", label="mean ± 95% CI")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout()
    PP.save(fig, "fig_push_pull_robot")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["geometry", "layout", "pushpull"])
    p.add_argument("--final", action="store_true", help="only the rebuilt (strict, decision) banks")
    a = p.parse_args()
    if "geometry" in a.which:
        for env in ("AT", "SZ"):
            geometry_figure(env, a.final)
    if "layout" in a.which:
        layout_figure(a.final)
    if "pushpull" in a.which:
        push_pull_figure()


if __name__ == "__main__":
    main()
