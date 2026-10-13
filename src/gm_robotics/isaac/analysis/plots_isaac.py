"""Isaac quadruped figures: outside/inside tables, G vs D_F, probes by depth, competition, rollouts, summary row."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gm_robotics.isaac.analysis import common_isaac as C

P = C.PAL
R = C.RESULTS
OUT = C.FIGURES
COH = {"gm": "GM", "control": "Non-GM", "init": "init"}
COL = {("gm", "goal"): P["gm"], ("gm", "proxy"): P["gm_proxy"], ("control", "goal"): P["nongm"],
       ("control", "proxy"): P["nongm_proxy"], ("init", "goal"): P["init"], ("init", "proxy"): P["init"]}
MAIN = [("flag_swap", "Flag swap (goal)", "goal"), ("tower_swap", "Tower swap (proxy)", "proxy")]
SUPP = [("no_flag", "Remove flag", "goal"), ("no_tower", "Remove tower", "proxy"), ("box_move", "Move box", "nuisance"),
        ("rerender", "Re-render", "null")]
plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.edgecolor": P["ink"], "axes.labelcolor": P["ink"], "xtick.color": P["ink"],
                     "ytick.color": P["ink"], "text.color": P["ink"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def ci(v):
    m, lo, hi = C.ci95(np.asarray(v, float))
    return m, (m - lo if np.isfinite(lo) else 0.0), (hi - m if np.isfinite(hi) else 0.0)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def panel_label(ax, k, title):
    ax.set_title(title, fontsize=10)
    ax.text(-0.16, 1.06, f"({chr(97 + k)})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


WE = pd.read_csv(os.path.join(R, "worldedit_per_seed.csv"))
TAB = [("KL", "KL"), ("SFD", "SF$_\\Delta$"), ("Q_h_fc", "$Q$"), ("G_h_fc", "$G$"), ("D_F_h_fc", "$D_F$"),
       ("trF_per_d_h_fc", "tr$(F)/d$"), ("dnorm2_h_fc", r"$\|\Delta\|^2$")]


def table(edits, bank, tag):
    rows = []
    for edit, lab, _ in edits:
        for coh in ("gm", "control", "init"):
            sub = WE[(WE.bank == bank) & (WE.edit == edit) & (WE.cohort == coh)]
            if not len(sub):
                continue
            r = dict(edit=lab, cohort=COH[coh], n=len(sub))
            for c, _ in TAB:
                v = sub[c].to_numpy(float) if c in sub else np.array([np.nan])
                m, lo, hi = C.ci95(v)
                r[c] = (m, lo, hi)
            rows.append(r)
    md = [f"# Outside / inside at h_fc, Isaac quadruped, bank {bank} ({tag})", "",
          "Mean and 95% t-CI over seeds of per-seed state means (KL, SF) or medians (Fisher terms).", "",
          "| Edit | Cohort | n | " + " | ".join(c for c, _ in TAB) + " |", "|" + "---|" * (len(TAB) + 3)]
    for r in rows:
        md.append(f"| {r['edit']} | {r['cohort']} | {r['n']} | " + " | ".join(C.fmt_ci(*r[c]) for c, _ in TAB) + " |")
    open(os.path.join(OUT, f"table_outside_inside_{bank}_{tag}.md"), "w").write("\n".join(md) + "\n")
    hdr = ["Edit", "Cohort"] + [lab for _, lab in TAB]
    cells, prev = [], None

    def f(m, lo, hi):
        if not np.isfinite(m):
            return ""
        nd = 3 if abs(m) < 100 else 1
        return f"{m:.{nd}f}" + (f" [{lo:.{nd}f}, {hi:.{nd}f}]" if np.isfinite(lo) else "")

    for r in rows:
        cells.append([r["edit"] if r["edit"] != prev else "", r["cohort"]] + [f(*r[c]) for c, _ in TAB])
        prev = r["edit"]
    fig, ax = plt.subplots(figsize=(14, 0.34 * (len(cells) + 1.5)))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center", edges="horizontal")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.3)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor(P["mute"])
        cell.set_linewidth(0.4 if i else 0.8)
        if i == 0:
            cell.set_text_props(fontweight="bold")
        else:
            c = {"GM": P["gm"], "Non-GM": P["nongm"]}.get(cells[i - 1][1], P["mute"])
            cell.get_text().set_color(c if j >= 1 else P["ink"])
    save(fig, f"table_outside_inside_{bank}_{tag}")


def g_vs_df():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for k, (ax, site) in enumerate(zip(axes, ("h_fc", "h_pre"))):
        allx, ally = [], []
        for edit, _, role in MAIN:
            mk = "^" if role == "goal" else "s"
            ini = WE[(WE.bank == "visible") & (WE.edit == edit) & (WE.cohort == "init")][f"G_{site}"].to_numpy()
            # init networks have near-zero actor output weights (D_F ~1e-9), so only their G is drawn, as a line.
            ax.axhline(np.exp(np.log(ini).mean()), color=P["init"], lw=1.2, ls="--" if role == "goal" else ":")
            for coh in ("control", "gm"):
                sub = WE[(WE.bank == "visible") & (WE.edit == edit) & (WE.cohort == coh)]
                col = COL[(coh, role)]
                x, y = sub[f"D_F_{site}"].to_numpy(), sub[f"G_{site}"].to_numpy()
                allx += list(x)
                ally += list(y)
                face = col if role == "goal" else "white"
                ax.scatter(x, y, s=16, marker=mk, color=face, edgecolor=col, alpha=0.55, lw=0.8, zorder=2)
                ax.scatter(np.exp(np.log(x).mean()), np.exp(np.log(y).mean()), s=70, marker=mk, color=face,
                           edgecolor=col, lw=1.6, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lx = np.log10([min(allx) / 3, max(allx) * 3])
        ax.set_xlim(10 ** lx)
        yl = ax.get_ylim()
        for q in np.arange(np.floor(lx[0] + np.log10(min(ally))) - 2, np.ceil(lx[1] + np.log10(max(ally))) + 2, 2):
            xs = np.logspace(lx[0], lx[1], 50)
            ax.plot(xs, 10 ** q / xs, ":", color=P["init"], lw=0.9, zorder=1)
        ax.set_ylim(min(yl[0], 0.5), yl[1])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"$D_F$")
        ax.set_ylabel(r"$G$")
        panel_label(ax, k, r"$h_{\mathrm{fc}}$" if site == "h_fc" else r"$h_{\mathrm{pre}}$")
    h = [Line2D([], [], marker="o", ls="", color=P["gm"], label="GM"),
         Line2D([], [], marker="o", ls="", color=P["nongm"], label="Non-GM"),
         Line2D([], [], ls="--", color=P["init"], label="init $G$"),
         Line2D([], [], marker="^", ls="", color=P["ink"], label="flag swap (goal)"),
         Line2D([], [], marker="s", ls="", mfc="white", color=P["ink"], label="tower swap (proxy)")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_g_vs_df")


SITES = [("conv1", r"$h^{(1)}$"), ("conv2", r"$h^{(2)}$"), ("conv3", r"$h^{(3)}$"), ("h_pre", r"$h_{\mathrm{pre}}$"),
         ("h_fc", r"$h_{\mathrm{fc}}$"), ("a_hid", r"$h_{\mathrm{act}}$")]
PROBES = [("flag_visible", "flag visible", "AUC"), ("flag_bearing", "flag bearing", "$R^2$"),
          ("flag_dist", "flag distance", "$R^2$"), ("tower_visible", "tower visible", "AUC"),
          ("tower_bearing", "tower bearing", "$R^2$"), ("tower_dist", "tower distance", "$R^2$")]


def probes():
    df = pd.read_csv(os.path.join(R, "probes_per_seed.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    x = np.arange(len(SITES))
    for k, (ax, (tgt, title, metric)) in enumerate(zip(axes.flat, PROBES)):
        for coh, col, ls in (("control", P["nongm"], "-"), ("gm", P["gm"], "-"), ("init", P["init"], "--")):
            mm = [ci(df[(df.cohort == coh) & (df.site == s) & (df.target == tgt)].value) for s, _ in SITES]
            m = np.array([v[0] for v in mm])
            lo, hi = m - np.array([v[1] for v in mm]), m + np.array([v[2] for v in mm])
            ax.fill_between(x, np.clip(lo, -0.5, 1.05), np.clip(hi, -0.5, 1.05), color=col, alpha=0.15, lw=0)
            ax.plot(x, m, ls, color=col, marker="o", ms=4, lw=1.6)
        pix = df[(df.cohort == "pixels") & (df.target == tgt)].value
        if len(pix):
            ax.axhline(float(pix.iloc[0]), ls=":", color=P["null"], lw=1.2)
        ax.set_xticks(x, [s for _, s in SITES], fontsize=8)
        ax.set_ylabel(metric)
        if metric == "AUC":
            ax.set_ylim(0.45, 1.02)
        else:
            ax.set_ylim(min(-0.1, ax.get_ylim()[0]), 1.02)
            ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
        panel_label(ax, k, title)
    h = [Line2D([], [], color=P["gm"], marker="o", label="GM"), Line2D([], [], color=P["nongm"], marker="o", label="Non-GM"),
         Line2D([], [], color=P["init"], ls="--", marker="o", label="init"),
         Line2D([], [], color=P["null"], ls=":", label="pixels")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_probes_depth")


def bars(ax, vals, labels, cols, ylab, hline=None, faces=None):
    for i, (v, c) in enumerate(zip(vals, cols)):
        m, l, h = ci(v)
        fc = c if faces is None else faces[i]
        ax.bar(i, m, 0.7, color=fc, edgecolor=c, lw=1.4)
        ax.errorbar(i, m, yerr=[[l], [h]], color=P["ink"], capsize=3, lw=1)
        ax.scatter(i + np.linspace(-0.12, 0.12, len(v)), v, s=10, color="white", edgecolor=P["ink"], lw=0.7, zorder=3)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(ylab)
    if hline is not None:
        ax.axhline(hline, color=P["mute"], lw=0.8)


def competition():
    ps = pd.read_csv(os.path.join(R, "axis_visible_per_seed.csv"))
    st = pd.read_csv(os.path.join(R, "axis_visible_per_state.csv"))
    cohs = ["gm", "control", "init"]
    cols = [P["gm"], P["nongm"], P["init"]]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    bars(axes[0], [ps[ps.cohort == c].cosF_within.to_numpy() for c in cohs], [COH[c] for c in cohs], cols,
         r"$\cos_F(\Delta_\star, \Delta_p)$", hline=0)
    axes[0].set_ylim(-1, 1)
    panel_label(axes[0], 0, "within state")
    ax = axes[1]
    for j, (key, lab) in enumerate((("cosF_same", "same side"), ("cosF_opp", "opposite side"))):
        for i, c in enumerate(cohs):
            v = ps[ps.cohort == c][key].to_numpy()
            m, l, h = ci(v)
            xpos = j * 3.6 + i
            ax.bar(xpos, m, 0.8, color=cols[i] if j == 0 else "white", edgecolor=cols[i], lw=1.4)
            ax.errorbar(xpos, m, yerr=[[l], [h]], color=P["ink"], capsize=3, lw=1)
    ax.set_xticks([1, 4.6], ["same side", "opposite side"])
    ax.axhline(0, color=P["mute"], lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(r"$\cos_F(d_{\mathrm{comp}}, d'_{\mathrm{comp}})$")
    panel_label(ax, 1, "across states")
    ax = axes[2]
    for i, c in enumerate(cohs):
        sub = ps[ps.cohort == c]
        m, l, h = ci(sub.heldout_align_eval)
        ax.bar(i, m, 0.7, color=cols[i])
        ax.errorbar(i, m, yerr=[[l], [h]], color=P["ink"], capsize=3, lw=1)
        ax.scatter([i], [sub.perm_mean.mean()], marker="_", s=300, color=P["null"], lw=2, zorder=4)
    ax.set_xticks(range(3), [COH[c] for c in cohs])
    ax.set_ylabel("held-out alignment")
    ax.set_ylim(-0.05, 1)
    panel_label(ax, 2, r"$u_{\mathrm{comp}}$")
    ax = axes[3]
    for c, col in (("init", P["init"]), ("control", P["nongm"]), ("gm", P["gm"])):
        s = st[st.cohort == c]
        g, pr = s.alpha_goal.abs().to_numpy(), s.alpha_prox.abs().to_numpy()
        ax.scatter(g, pr, s=5, color=col, alpha=0.3, lw=0)
        ax.scatter([np.median(g)], [np.median(pr)], s=80, color=col, edgecolor="white", lw=1.4, zorder=4)
    lim = (1e-4, 1e2)
    ax.plot(lim, lim, "--", color=P["mute"], lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("flag pull")
    ax.set_ylabel("tower pull")
    panel_label(ax, 3, "pull per state")
    h = [Patch(color=P["gm"], label="GM"), Patch(color=P["nongm"], label="Non-GM"), Patch(color=P["init"], label="init"),
         Line2D([], [], marker="_", ls="", ms=14, mew=2, color=P["null"], label="permutation null")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig_competition")


def rollouts_df():
    import glob
    fs = sorted(glob.glob(os.path.join(R, "rollouts", "*.csv")))
    df = pd.concat([pd.read_csv(f).assign(bank=os.path.basename(f).split("_main_")[0]) for f in fs], ignore_index=True)
    df["cohort"] = df.run.str.extract(r"main_(gm|control)_s")[0]
    df["seed"] = df.run.str.extract(r"_s(\d)$")[0].astype(int)
    return df


def rollouts():
    df = rollouts_df()
    banks = [b for b in ("visible", "conflict", "far") if b in set(df.bank)]
    edits = [("base", "base"), ("no_tower", "remove tower"), ("no_flag", "remove flag"), ("flag_swap", "flag swap"),
             ("tower_swap", "tower swap")]
    fig, axes = plt.subplots(1, len(banks), figsize=(4.6 * len(banks), 3.0), squeeze=False)
    for k, (ax, bank) in enumerate(zip(axes[0], banks)):
        for i, (e, _) in enumerate(edits):
            for j, (coh, col) in enumerate((("gm", P["gm"]), ("control", P["nongm"]))):
                sub = df[(df.bank == bank) & (df.edit == e) & (df.cohort == coh)]
                if not len(sub):
                    continue
                v = sub.groupby("seed").reached.apply(lambda s: (s == "u").mean()).to_numpy()
                m, l, h = ci(v)
                xpos = i * 2.7 + j
                ax.bar(xpos, m, 0.85, color=col)
                ax.errorbar(xpos, m, yerr=[[l], [h]], color=P["ink"], capsize=3, lw=1)
                ax.scatter(xpos + np.linspace(-0.15, 0.15, len(v)), v, s=9, color="white", edgecolor=P["ink"], lw=0.6, zorder=3)
        ax.set_xticks([i * 2.7 + 0.5 for i in range(len(edits))], [lab for _, lab in edits], rotation=20)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("first reached: flag location $u$")
        panel_label(ax, k, {"visible": "flag near", "conflict": "eval separation", "far": "flag far"}[bank])
    fig.legend(handles=[Patch(color=P["gm"], label="GM"), Patch(color=P["nongm"], label="Non-GM")], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_rollouts_first_reached")


def recovery_panel(ax):
    """Native eval layouts: unsteered, steered at h_fc (state-local d_comp, beta=1), rewired from h_pre / h_fc (teacher BC)."""
    st = pd.read_csv(os.path.join(R, "steer", "steer_per_seed.csv"))
    rw = pd.read_csv(os.path.join(R, "rewire", "rewire_per_seed.csv"))
    st = st[(st.bank == "native") & (st.plan.isin(["full", "control"]))]
    rw = rw[rw.set == "native"]
    groups = [("native", lambda c: st[(st.cohort == c) & (st.arm == "baseline")].goal),
              ("steered\nh_fc, β=1", lambda c: st[(st.cohort == c) & (st.arm == "local") & (st.beta == 1.0)].goal),
              ("random\npush, β=1", lambda c: st[(st.cohort == c) & (st.arm == "random") & (st.beta == 1.0)].goal),
              ("rewired\nfrom h_pre", lambda c: rw[(rw.cohort == c) & (rw.arm == "pre") & (rw.target == "teacher")].goal),
              ("rewired\nfrom h_fc", lambda c: rw[(rw.cohort == c) & (rw.arm == "fc_head") & (rw.target == "teacher")].goal)]
    for i, (lab, fn) in enumerate(groups):
        for k, (coh, col) in enumerate((("gm", P["gm"]), ("control", P["nongm"]))):
            v = fn(coh).to_numpy()
            if not len(v):
                continue
            m, l, hh = ci(v)
            xpos = i * 2.6 + k
            ax.bar(xpos, m, 0.8, color=col)
            ax.errorbar(xpos, m, yerr=[[l], [hh]], color=P["ink"], capsize=4, lw=1.3)
            ax.scatter(xpos + np.linspace(-0.15, 0.15, len(v)), v, s=14, facecolor="white", edgecolor=P["ink"], zorder=3, lw=0.8)
    ax.set_xticks([i * 2.6 + 0.5 for i in range(len(groups))], [g for g, _ in groups], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("flag reached first (native eval)")


def summary_row():
    """One row, four columns (representation | readout | decision | recovery)."""
    GM, CT, GMP, CTP, INIT, INK, MUTE = P["gm"], P["nongm"], P["gm_proxy"], P["nongm_proxy"], P["init"], P["ink"], P["mute"]
    pr = pd.read_csv(os.path.join(R, "probes_per_seed.csv"))
    fig = plt.figure(figsize=(24, 4.6), dpi=100)
    x0s, w, y0, h = [0.165, 0.375, 0.585, 0.80], 0.175, 0.2, 0.58
    for j, head in enumerate(["Representation", "Readout", "Decision", "Recovery"]):
        fig.text(x0s[j] + w / 2, 0.88, head, fontsize=20, weight="bold", color=INK, ha="center")
    fig.text(0.02, y0 + h / 2 + 0.05, "Isaac ANYmal-C", fontsize=17, weight="bold", color=INK)
    fig.text(0.02, y0 + h / 2 - 0.04, "flag goal, tower proxy", fontsize=15, color=INK)
    ax = fig.add_axes([x0s[0], y0, w, h])
    sites = ["conv1", "conv2", "conv3", "h_pre", "h_fc", "a_hid"]
    x = np.arange(len(sites))
    for tgt, coh, col, ls, mk, filled in (("flag_bearing", "control", CT, "-", "o", True), ("flag_bearing", "gm", GM, "-", "o", True),
                                          ("flag_bearing", "init", INIT, ":", "o", False),
                                          ("tower_bearing", "control", CTP, "--", "s", False), ("tower_bearing", "gm", GMP, "--", "s", False)):
        d = np.array([ci(pr[(pr.cohort == coh) & (pr.site == s) & (pr.target == tgt)].value)[:2] for s in sites])
        ax.fill_between(x, np.clip(d[:, 0] - d[:, 1], -0.1, 1.05), np.clip(d[:, 0] + d[:, 1], -0.1, 1.05), color=col, alpha=0.08, lw=0)
        ax.plot(x, d[:, 0], ls, color=col, lw=2.2, marker=mk, ms=6, mfc=col if filled else "white", mec=col)
    ax.set_xticks(x, ["conv1", "conv2", "conv3", "flat", "feat", "actor"], fontsize=9)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("bearing R²")
    ax.axhline(0, color="#dddddd", lw=1, zorder=0)
    ax = fig.add_axes([x0s[1], y0, w, h])
    for gi, (edit, cg, cc) in enumerate((("flag_swap", GM, CT), ("tower_swap", GMP, CTP))):
        for k, (coh, col) in enumerate((("gm", cg), ("control", cc))):
            v = WE[(WE.bank == "visible") & (WE.edit == edit) & (WE.cohort == coh)].Q_h_fc.to_numpy()
            m, l, hh = ci(v)
            xpos = gi * 2.6 + k
            ax.bar(xpos, m, 0.8, color=col)
            ax.errorbar(xpos, m, yerr=[[min(l, m * 0.95)], [hh]], color=INK, capsize=4, lw=1.3)
            ax.scatter(xpos + np.linspace(-0.15, 0.15, len(v)), v, s=14, facecolor="white", edgecolor=INK, zorder=3, lw=0.8)
    nul = WE[(WE.bank == "visible") & (WE.edit == "rerender") & (WE.cohort.isin(["gm", "control"]))].Q_h_fc.mean()
    ax.axhline(nul, ls=":", color=MUTE, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e3)
    ax.set_xticks([0.5, 3.1], ["flag swap (goal)", "tower swap (proxy)"])
    ax.set_xlim(-0.7, 4.3)
    ax.set_ylabel("Q")
    ax = fig.add_axes([x0s[2], y0, w, h])
    df = rollouts_df()
    df = df[df.bank == "visible"]
    for i, e in enumerate(("base", "no_tower")):
        for k, (coh, col) in enumerate((("gm", GM), ("control", CT))):
            v = df[(df.edit == e) & (df.cohort == coh)].groupby("seed").reached.apply(lambda s: (s == "u").mean()).to_numpy()
            m, l, hh = ci(v)
            xpos = i * 2.6 + k
            ax.bar(xpos, m, 0.8, color=col)
            ax.errorbar(xpos, m, yerr=[[l], [hh]], color=INK, capsize=4, lw=1.3)
            ax.scatter(xpos + np.linspace(-0.15, 0.15, len(v)), v, s=14, facecolor="white", edgecolor=INK, zorder=3, lw=0.8)
    ax.set_xticks([0.5, 3.1], ["flag + tower", "tower removed"])
    ax.set_xlim(-0.7, 4.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("flag reached first")
    recovery_panel(fig.add_axes([x0s[3], y0, w, h]))
    hs = [Line2D([], [], color=GM, lw=2.2, marker="o", label="goal cue, GM"), Line2D([], [], color=CT, lw=2.2, marker="o", label="goal cue, control"),
          Line2D([], [], color=GMP, lw=2.2, ls="--", marker="s", mfc="white", label="proxy cue, GM"),
          Line2D([], [], color=CTP, lw=2.2, ls="--", marker="s", mfc="white", label="proxy cue, control"),
          Line2D([], [], color=INIT, lw=2, ls=":", marker="o", mfc="white", label="random network")]
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, fontsize=12, bbox_to_anchor=(0.55, -0.02))
    save(fig, "slide_inside_isaac")


def main():
    table(MAIN, "visible", "main")
    table(SUPP, "visible", "supp")
    for b in ("conflict", "far"):
        table(MAIN + SUPP, b, "all")
    g_vs_df()
    if os.path.exists(os.path.join(R, "probes_per_seed.csv")):
        probes()
    competition()
    rollouts()
    if os.path.exists(os.path.join(R, "probes_per_seed.csv")):
        summary_row()


if __name__ == "__main__":
    main()
