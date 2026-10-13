"""Competition geometry and push-pull figures for Attribute Maze next to AttributeTable and SortZones."""
import argparse
import glob
import os

import matplotlib
from gm_robotics.paths import REPO
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from gm_robotics.analysis import common as C
from gm_robotics.analysis import plots_paper as PP

P = C.PAL
AM_DIR = os.path.join(os.environ.get("GM_DATA_ROOT", str(REPO / "data")), "probes")
GF = os.path.join(C.RESULTS, "geometry_final")
AUD = os.path.join(C.RESULTS, "bank_audit")
# Paper primary roster: black background, h_fc; deco = the paper's Non-GM cohort.
AM_SEEDS = {"gm": ("6313", "6315", "6317"), "control": ("4371", "4372", "4373"), "init": ("3000", "3001", "3002")}
AM_TAG = {"gm": "gm", "control": "deco", "init": "init"}
GOAL_CUE = {"AM": "shape", "AT": "shape", "SZ": "cross"}
BANK_TXT = {"strict": "strict t=0", "decision": "decision frames"}
ENV_TXT = {"AM": "Attribute Maze\n(paper)", "AT": "AttributeTable", "SZ": "SortZones"}
COHS = ("gm", "control", "init")


# ---------------------------------------------------------------- data: Attribute Maze (paper)

def am_geometry(d, stat):
    g = os.path.join(d, "canon_geometry_20260924", "swap", "per_seed")
    within = {}
    for coh in COHS:
        fs = [pd.read_csv(os.path.join(g, f"within_state_{AM_TAG[coh]}_{s}_black_h_fc.csv")) for s in AM_SEEDS[coh]]
        within[coh] = [np.array([getattr(f.cos_e, stat)() for f in fs]), np.array([getattr(f.cos_f, stat)() for f in fs])]
    w = pd.read_csv(os.path.join(g, f"within_state_gm_{AM_SEEDS['gm'][0]}_black_h_fc.csv"))
    order = np.lexsort((w.layout.to_numpy(), w.d.to_numpy(), -w.sign.to_numpy()))
    zs = [np.load(os.path.join(g, f"matrices_gm_{s}_black_h_fc.npz")) for s in AM_SEEDS["gm"]]
    assert all((z["state_key"] == w.state_key.to_numpy()).all() for z in zs)
    mats = {k: np.mean([z[k].astype(float) for z in zs], 0)[order][:, order] for k in ("kappa_e", "kappa_f")}
    tug = {}
    for coh in ("gm", "control"):
        fs = [pd.read_csv(os.path.join(d, "am_axis_tug_of_war_20260925", "per_seed",
                                       f"per_state_{AM_TAG[coh]}_{s}_black_h_fc.csv")) for s in AM_SEEDS[coh]]
        fs = [f[(f.axis == "unit_mean") & (f.split_role == "eval")] for f in fs]
        tug[coh] = (np.concatenate([f.aF_star for f in fs]), np.concatenate([f.aF_p for f in fs]),
                    np.array([f.aF_star.mean() for f in fs]), np.array([f.aF_p.mean() for f in fs]))
    return dict(within=within, wlabels=[r"$\cos_E$", r"$\cos_F$"], E=mats["kappa_e"], F=mats["kappa_f"],
                sign=w.sign.to_numpy()[order], sides=("UP", "RIGHT"), tug=tug, cue="shape")


def am_push_pull(d):
    out = {}
    for coh in ("gm", "control"):
        for s in AM_SEEDS[coh]:
            df = pd.read_csv(os.path.join(d, "canon_cue_20260924", "rows_csv_swap", f"rows_swap_black_{AM_TAG[coh]}_{s}.csv"))
            for ed in ("shape", "colour"):
                sub = df[df.edit == ed]
                # Paper Table 10 convention: seed value = equal-weight mean of the two conflict-orientation means.
                sm = sub.groupby("axis_sign")[["S_fav", "S_comp"]].mean().mean()
                e = out.setdefault((ed, coh), dict(st=[], seed=[]))
                e["st"].append(sub[["S_fav", "S_comp"]].to_numpy())
                e["seed"].append(sm.to_numpy())
    return {k: (np.concatenate(v["st"]), np.array(v["seed"])) for k, v in out.items()}


# ---------------------------------------------------------------- data: robots, final banks

def robot_geometry(env, bank, stat, tug_matched):
    st = pd.read_csv(os.path.join(GF, f"{env}_per_state.csv"))
    aud = pd.read_csv(os.path.join(AUD, f"geometry_{env}_per_state.csv"), usecols=["bank", "cohort", "seed", "cos_tcp"])
    within = {}
    for coh in COHS:
        s = st[(st.bank == bank) & (st.cohort == coh)].groupby("seed")
        a = aud[(aud.bank == bank) & (aud.cohort == coh)].groupby("seed").cos_tcp
        within[coh] = [s.cosE_within.agg(stat).to_numpy(), s.cosF_within.agg(stat).to_numpy(), a.agg(stat).to_numpy()]
    fs = sorted(glob.glob(os.path.join(GF, "matrices", f"{env}_{bank}_gm_s*.npz")))
    zs = [np.load(f) for f in fs]
    assert all((z["order"] == zs[0]["order"]).all() for z in zs)
    E, F = (np.mean([z[k].astype(float) for z in zs], 0) for k in ("cosE", "cosF"))
    tb = bank + ("_pxmatched" if tug_matched else "")
    tug = {}
    for coh in ("gm", "control"):
        s = st[(st.bank == tb) & (st.cohort == coh) & st["eval"]]
        # Goal-cue pull reported toward its own candidate, as in the paper: sign * u points to the goal candidate,
        # which the goal-cue swap moves away from.
        g = s.groupby("seed")
        tug[coh] = (-s.alpha_goal.to_numpy(), s.alpha_prox.to_numpy(), -g.alpha_goal.mean().to_numpy(),
                    g.alpha_prox.mean().to_numpy())
    return dict(within=within, wlabels=[r"$\cos_E$", r"$\cos_F$", "hand\nmotion"], E=E, F=F,
                sign=zs[0]["sign_sorted"], sides=("goal left", "goal right"), tug=tug, cue=GOAL_CUE[env])


# ---------------------------------------------------------------- drawing

def draw_matrix(ax, M, sign, sides, ylabels):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    nl = int((sign > 0).sum())
    ax.axhline(nl - 0.5, color="k", lw=1.0), ax.axvline(nl - 0.5, color="k", lw=1.0)
    ax.set_xticks([])
    ax.set_yticks([nl / 2, nl + (len(sign) - nl) / 2], sides if ylabels else ["", ""], rotation=90, va="center")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
    return im


def bar_to_floor(ax, x, y, lo, hi, axis, floor, col):
    """Paper style for a CI that runs below the log-axis floor: the bar ends at the floor in an arrow head."""
    if axis == "y":
        bottom = max(y - lo, floor)
        ax.plot([x, x], [bottom, y + hi], color=col, lw=1.6, zorder=4)
        if y - lo <= floor:
            ax.plot(x, floor, marker="v", color=col, ms=6, zorder=4, clip_on=False)
    else:
        left = max(x - lo, floor)
        ax.plot([left, x + hi], [y, y], color=col, lw=1.6, zorder=4)
        if x - lo <= floor:
            ax.plot(floor, y, marker="<", color=col, ms=6, zorder=4, clip_on=False)


def draw_tug(ax, tug, cue, prox="colour"):
    pos = np.concatenate([np.concatenate([t[0], t[1]]) for t in tug.values()])
    pos = pos[pos > 0]
    lim = (np.quantile(pos, 0.02) / 1.5, pos.max() * 1.5)
    ratios = {}
    for coh, col in (("control", P["nongm"]), ("gm", P["gm"])):
        x, y, xs, ys = tug[coh]
        # States whose pull points the wrong way (<= 0) cannot sit on a log axis; they sit on the floor, as in the paper.
        ax.scatter(np.clip(x, lim[0], None), np.clip(y, lim[0], None), s=6, color=col, alpha=0.35, lw=0, clip_on=False)
        mx, lx, hx = PP.ci(xs)
        my, ly, hy = PP.ci(ys)
        bar_to_floor(ax, max(mx, lim[0]), my, ly, hy, "y", lim[0], col)
        bar_to_floor(ax, mx, max(my, lim[0]), lx, hx, "x", lim[0], col)
        ax.plot(max(mx, lim[0]), max(my, lim[0]), "o", ms=6, color=col, mec="white", mew=0.8, zorder=5, clip_on=False)
        ratios[coh] = np.mean(ys / xs)
    ax.plot(lim, lim, "--", color=P["mute"], lw=1)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel(rf"{cue} pull $\alpha_F(\Delta_\star)$")
    ax.set_ylabel(rf"{prox} pull $\alpha_F(\Delta_p)$")
    ax.text(0.03, 0.97, f"{prox} / {cue} pull\nGM {ratios['gm']:.2f}, Non-GM {ratios['control']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=P["ink"],
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    return ratios


def geometry_combined(a):
    rows = [("AM", None, am_geometry(a.am_dir, a.stat))]
    rows += [(env, bank, robot_geometry(env, bank, a.stat, a.tug_matched)) for env, bank in (("AT", a.at_bank), ("SZ", a.sz_bank))]
    fig = plt.figure(figsize=(13.5, 3.25 * len(rows) + 0.6))
    gs = fig.add_gridspec(len(rows) + 1, 4, height_ratios=[1] * len(rows) + [0.05], width_ratios=[1, 1, 1, 1.05],
                          hspace=0.42, wspace=0.42, bottom=0.07, top=0.97)
    ratios = {}
    for r, (env, bank, d) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        nq = len(d["wlabels"])
        PP.cohort_points(ax, list(range(nq)), d["within"])
        ax.set_xticks(range(nq), d["wlabels"])
        ax.set_xlim(-0.5, 2.5)
        ax.axhline(0, color=P["init"], lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("cosine")
        PP.panel_label(ax, 4 * r, "Within a state")
        row = ENV_TXT[env] + ("" if bank is None else f"\n{BANK_TXT[bank]}")
        ax.text(-0.5, 0.5, row, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=10,
                fontweight="bold")
        for j, k in enumerate(("E", "F")):
            ax = fig.add_subplot(gs[r, 1 + j])
            im = draw_matrix(ax, d[k], d["sign"], d["sides"], ylabels=j == 0)
            PP.panel_label(ax, 4 * r + 1 + j, "Euclidean" if k == "E" else "Fisher")
        ax = fig.add_subplot(gs[r, 3])
        ratios[env] = draw_tug(ax, d["tug"], d["cue"])
        PP.panel_label(ax, 4 * r + 3, "Tug of war" + (" (pixel-matched)" if env != "AM" and a.tug_matched else ""))
    cax = fig.add_subplot(gs[len(rows), 1:3])
    fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    h = [Line2D([], [], marker="o", ls="", color=P["gm"], label="GM"),
         Line2D([], [], marker="o", ls="", color=P["nongm"], label="Non-GM"),
         Line2D([], [], marker="o", ls="", color=P["init"], label="init"),
         Line2D([], [], marker="o", color=P["mute"], label="mean ± 95% CI")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    PP.save(fig, a.out_geometry)
    print("tug ratios (seed mean of per-seed ratio of mean pulls):",
          {e: {c: round(v, 3) for c, v in r.items()} for e, r in ratios.items()})


def push_pull_panel(ax, st, seeds, lim, title, k, sym=False):
    ax.add_patch(plt.Rectangle((0, -lim), lim, lim, color="#F2F2F2", zorder=0))
    ax.axhline(0, color=P["mute"], ls="--", lw=0.8), ax.axvline(0, color=P["mute"], ls="--", lw=0.8)
    for coh, col in (("control", P["nongm"]), ("gm", P["gm"])):
        s, sd = st[coh], seeds[coh]
        ax.scatter(s[:, 0], s[:, 1], s=5, color=col, alpha=0.15, lw=0)
        ax.scatter(sd[:, 0], sd[:, 1], marker="D", s=40, color=col, ec="white", lw=0.8, zorder=4)
        mf, lf, hf = PP.ci(sd[:, 0])
        mc, lc, hc = PP.ci(sd[:, 1])
        ax.errorbar(mf, mc, xerr=[[lf], [hf]], yerr=[[lc], [hc]], fmt="o", ms=5, color=col, mfc="white", mew=1.4,
                    lw=1.4, zorder=5)
    lo = lim if sym else lim * 0.45
    ax.set_xlim(-lo, lim), ax.set_ylim(-lim, lo)
    ax.set_xlabel(r"$S_{\mathrm{fav}}$")
    PP.panel_label(ax, k, title)


def push_pull_combined(a):
    pp = os.path.join(C.RESULTS, "push_pull")
    swap = {"AT": (("shape_swap", "shape swap"), ("colour_swap", "colour swap")),
            "SZ": (("mark_swap", "cross swap"), ("colour_swap", "colour swap"))}
    banks = (("strict", "start"), ("decision", "decision frame"))
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.2))
    rng = np.random.default_rng(0)
    am = am_push_pull(a.am_dir)
    lim = np.nanquantile(np.abs(np.concatenate([v[0] for v in am.values()])), 0.98) * 1.1
    # Maze states are decision points, so the two Attribute Maze swaps sit above the robots' decision-frame columns.
    for j, ed in enumerate(("shape", "colour")):
        push_pull_panel(axes[0, 2 + j], {c: am[(ed, c)][0] for c in ("gm", "control")},
                        {c: am[(ed, c)][1] for c in ("gm", "control")}, lim, f"{ed} swap", j)
    axes[0, 2].set_ylabel("Attribute Maze (paper)\n" + r"$S_{\mathrm{comp}}$")
    for ax in axes[0, :2]:
        ax.axis("off")
    axes[0, 0].text(0.0, 0.62, "Attribute Maze (paper, push-pull panel c)", fontsize=11, fontweight="bold", transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.0, 0.08,
                    "Black background, 196 conflict states, 3 seeds per cohort.\n"
                    "Only the two swaps exist; maze states are decision\n"
                    "points, so they sit above the robots' decision frames.\n"
                    r"$S$ = change in log support relative to neutral actions;" "\n"
                    r"robot rows: change in probit support $\Phi(m/s)$ of each" "\n"
                    "candidate's progress half-space. Axes differ per row.\n"
                    "Seed diamonds: sign-stratified state means (AM),\nstate means (robots).",
                    fontsize=8.5, color=P["ink"], transform=axes[0, 0].transAxes, va="bottom")
    for r, env in enumerate(("AT", "SZ"), start=1):
        st = pd.read_csv(os.path.join(pp, f"{env}_per_state.csv"))
        ps = pd.read_csv(os.path.join(pp, f"{env}_per_seed.csv"))
        st = st[st.bank.isin([b for b, _ in banks])]
        lim = np.nanquantile(np.abs(st[st.cohort != "init"][["S_fav", "S_comp"]].to_numpy()), 0.98) * 1.1
        for j, ((bk, btxt), (ed, etxt)) in enumerate([(b, e) for b in banks for e in swap[env]]):
            s, sd = {}, {}
            for coh in ("gm", "control"):
                sub = st[(st.bank == bk) & (st.edit == ed) & (st.cohort == coh)]
                if len(sub) > 600:
                    sub = sub.iloc[rng.choice(len(sub), 600, replace=False)]
                s[coh] = sub[["S_fav", "S_comp"]].to_numpy()
                sd[coh] = ps[(ps.bank == bk) & (ps.edit == ed) & (ps.cohort == coh)][["S_fav", "S_comp"]].to_numpy()
            push_pull_panel(axes[r, j], s, sd, lim, f"{etxt}, {btxt}", 2 + 4 * (r - 1) + j)
            if j == 0:
                axes[r, j].set_ylabel(f"{C.ENV_IDS[env].split('-')[0]}\n" + r"$S_{\mathrm{comp}}$")
    h = [Line2D([], [], marker="o", ls="", color=P["gm"], label="GM"),
         Line2D([], [], marker="o", ls="", color=P["nongm"], label="Non-GM"),
         Line2D([], [], marker="D", ls="", color=P["mute"], mec="white", label="seed mean"),
         Line2D([], [], marker="o", color=P["mute"], mfc="white", label="mean ± 95% CI")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(h_pad=1.6)
    PP.save(fig, a.out_push_pull)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["geometry", "pushpull"])
    p.add_argument("--am_dir", default=AM_DIR, help="gm-geometry data/probes (read only)")
    p.add_argument("--at_bank", default="strict", choices=["strict", "decision"])
    p.add_argument("--sz_bank", default="decision", choices=["strict", "decision"])
    p.add_argument("--stat", default="mean", choices=["mean", "median"],
                   help="per-seed statistic over states for the within-state panel (paper: mean)")
    p.add_argument("--tug_raw", dest="tug_matched", action="store_false",
                   help="robot tug of war on the raw edits instead of the pixel-matched ones")
    p.add_argument("--out_geometry", default="fig_geometry_combined")
    p.add_argument("--out_push_pull", default="fig_push_pull_combined")
    a = p.parse_args()
    if "geometry" in a.which:
        geometry_combined(a)
    if "pushpull" in a.which:
        push_pull_combined(a)


if __name__ == "__main__":
    main()
