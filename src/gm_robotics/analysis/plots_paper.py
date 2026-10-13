"""Paper-style figures and the outside/inside table for AttributeTable, LocationReach and SortZones, from the analysis CSVs."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from gm_robotics.analysis import common as C

P = C.PAL
OUT = os.path.join(C.FIGURES, "paper_style")
ENVS = {"AT": "AttributeTable", "LR": "LocationReach", "SZ": "SortZones"}
# Canonical edits per env: (edit, label, role). LR has no proxy edit (the proxy is a location with no visual token).
MAIN_EDITS = {"AT": [("shape_swap", "Shape (goal)", "goal"), ("colour_swap", "Colour (proxy)", "proxy")],
              "LR": [("goal_move", "Marker (goal)", "goal")],
              "SZ": [("mark_swap", "Cross (goal)", "goal"), ("colour_swap", "Colour (proxy)", "proxy")]}
SUPP_EDITS = {"AT": [("shape_ablate", "Shape ablate", "goal"), ("colour_ablate", "Colour ablate", "proxy")],
              "LR": [("goal_to_corner", "Marker to corner", "goal"), ("distractor_move", "Distractor", "proxy")],
              "SZ": [("mark_ablate", "Cross ablate", "goal"), ("colour_ablate", "Colour ablate", "proxy")]}
COH = {"gm": "GM", "control": "Non-GM", "init": "init"}
COL = {("gm", "goal"): P["gm"], ("gm", "proxy"): P["gm_proxy"], ("control", "goal"): P["nongm"],
       ("control", "proxy"): P["nongm_proxy"], ("init", "goal"): P["init"], ("init", "proxy"): P["init"]}
plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.edgecolor": P["ink"], "axes.labelcolor": P["ink"], "xtick.color": P["ink"],
                     "ytick.color": P["ink"], "text.color": P["ink"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def ci(v):
    """Mean, lower and upper 95% t-interval half-widths (0 when n < 2)."""
    m, lo, hi = C.ci95(np.asarray(v, float))
    return m, (m - lo if np.isfinite(lo) else 0.0), (hi - m if np.isfinite(hi) else 0.0)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, k, title):
    ax.set_title(title, fontsize=10)
    ax.text(-0.14, 1.06, f"({chr(97 + k)})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def load(kind, env):
    f = os.path.join(C.RESULTS, f"{kind}_{env}_per_seed.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


# ---------------------------------------------------------------- (a) outside / inside table

TAB_COLS = [("KL", "KL"), ("Q_h_fc", "$Q$"), ("G_h_fc", "$G$"), ("D_F_h_fc", "$D_F$"),
            ("trF_per_d_h_fc", "tr$(F)/d$"), ("dnorm2_h_fc", r"$\|\Delta\|^2$")]


def table_rows(edits_by_env):
    rows = []
    for env, edits in edits_by_env.items():
        df = load("worldedit", env)
        if df is None:
            continue
        for edit, lab, _ in edits:
            for coh in ("gm", "control"):
                sub = df[(df.edit == edit) & (df.cohort == coh)]
                if not len(sub):
                    continue
                r = dict(env=ENVS[env], edit=lab, cohort=COH[coh], n=len(sub))
                for c, _ in TAB_COLS:
                    v = sub[c].to_numpy(float)
                    m, lo, hi = C.ci95(v)
                    r[c] = (v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, lo, hi)
                rows.append(r)
    return rows


def fmt_sd(m, sd):
    return f"{m:.3f}" if not np.isfinite(sd) else f"{m:.3f}$\\pm${sd:.2f}"


def outside_inside_table():
    for tag, edits in (("main", MAIN_EDITS), ("supp", SUPP_EDITS)):
        rows = table_rows(edits)
        if not rows:
            continue
        # Markdown with 95% t-CI (analysis convention) ...
        md = [f"# Outside / inside at h_fc ({tag} edits)", "",
              "Mean and 95% t-CI over seeds of per-seed state means (KL) or medians (Fisher terms). "
              "tr(F)/d = average Fisher sensitivity per dimension, |Delta|^2 = squared Euclidean edit displacement.", "",
              "| Environment | Edit | Cohort | n | " + " | ".join(c for c, _ in TAB_COLS) + " |",
              "|" + "---|" * (len(TAB_COLS) + 4)]
        for r in rows:
            md.append(f"| {r['env']} | {r['edit']} | {r['cohort']} | {r['n']} | " +
                      " | ".join(C.fmt_ci(r[c][0], r[c][2], r[c][3]) for c, _ in TAB_COLS) + " |")
        open(os.path.join(OUT, f"table_outside_inside_{tag}.md"), "w").write("\n".join(md) + "\n")
        # ... and a LaTeX-style block in the paper's table convention (mean +- SD over seeds).
        tex = [r"\begin{tabular}{lllcccccc}", r"\toprule",
               r"& & & Outside & \multicolumn{5}{c}{Inside ($h_{\mathrm{fc}}$)} \\",
               r"\cmidrule(lr){4-4}\cmidrule(lr){5-9}",
               r"Environment & Edit & Cohort & KL & $Q$ & $G$ & $D_F$ & $\operatorname{tr}(F)/d$ & $\|\Delta\|^2$ \\",
               r"\midrule"]
        prev = None
        for r in rows:
            if prev is not None and r["env"] != prev:
                tex.append(r"\midrule")
            tex.append(f"{r['env'] if r['env'] != prev else ''} & {r['edit']} & {r['cohort']} & " +
                       " & ".join(f"${fmt_sd(r[c][0], r[c][1])}$".replace("$$", "") for c, _ in TAB_COLS) + r" \\")
            prev = r["env"]
        tex += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(OUT, f"table_outside_inside_{tag}.tex"), "w").write("\n".join(tex) + "\n")
        # Rendered figure version of the same table.
        hdr = ["Environment", "Edit", "Cohort"] + [lab for _, lab in TAB_COLS]
        cells, prev_env, prev_edit = [], None, None
        for r in rows:
            cells.append([r["env"] if r["env"] != prev_env else "",
                          r["edit"] if (r["edit"], r["env"]) != (prev_edit, prev_env) else "", r["cohort"]] +
                         [f"{r[c][0]:.3f}" + (f" ± {r[c][1]:.2f}" if np.isfinite(r[c][1]) else "") for c, _ in TAB_COLS])
            prev_env, prev_edit = r["env"], r["edit"]
        fig, ax = plt.subplots(figsize=(12.5, 0.32 * (len(cells) + 1.5)))
        ax.axis("off")
        t = ax.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center", edges="horizontal")
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.25)
        for (i, j), cell in t.get_celld().items():
            cell.set_edgecolor(P["mute"])
            cell.set_linewidth(0.4 if i else 0.8)
            if i == 0:
                cell.set_text_props(fontweight="bold")
            elif cells[i - 1][2] == "GM":
                cell.get_text().set_color(P["gm"] if j >= 2 else P["ink"])
            else:
                cell.get_text().set_color(P["nongm"] if j >= 2 else P["ink"])
        save(fig, f"table_outside_inside_{tag}")


# ---------------------------------------------------------------- (b) G vs D_F

def g_vs_df(site="h_fc"):
    envs = [e for e in ENVS if load("worldedit", e) is not None]
    fig, axes = plt.subplots(1, len(envs), figsize=(3.6 * len(envs), 3.0))
    for k, (ax, env) in enumerate(zip(np.atleast_1d(axes), envs)):
        df = load("worldedit", env)
        allx, ally = [], []
        for edit, _, role in MAIN_EDITS[env]:
            mk = "^" if role == "goal" else "s"
            for coh in ("init", "control", "gm"):
                sub = df[(df.edit == edit) & (df.cohort == coh)]
                if not len(sub):
                    continue
                col = COL[(coh, role)]
                x, y = sub[f"D_F_{site}"].to_numpy(), sub[f"G_{site}"].to_numpy()
                allx += list(x)
                ally += list(y)
                face = col if role == "goal" else "white"
                ax.scatter(x, y, s=16, marker=mk, color=face, edgecolor=col, alpha=0.55, lw=0.8, zorder=2)
                # Geometric mean across seeds: both factors are ratios spread over decades.
                ax.scatter(np.exp(np.log(x).mean()), np.exp(np.log(y).mean()), s=70, marker=mk, color=face,
                           edgecolor=col, lw=1.6, zorder=3)
        ax.set_xscale("log"), ax.set_yscale("log")
        lx = np.log10([min(allx) / 3, max(allx) * 3])
        ax.set_xlim(10 ** lx)
        yl = ax.get_ylim()
        for q in np.arange(np.floor(lx[0] + np.log10(min(ally))) - 2, np.ceil(lx[1] + np.log10(max(ally))) + 2, 2):
            xs = np.logspace(lx[0], lx[1], 50)
            ax.plot(xs, 10 ** q / xs, ":", color=P["init"], lw=0.9, zorder=1)
        ax.set_ylim(yl)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
        ax.set_xlabel(r"$D_F$")
        if k == 0:
            ax.set_ylabel(r"$G$ at $h_{\mathrm{fc}}$")
        panel_label(ax, k, ENVS[env])
    h = [Line2D([], [], marker="o", ls="", color=P["gm"], label="GM"),
         Line2D([], [], marker="o", ls="", color=P["nongm"], label="Non-GM"),
         Line2D([], [], marker="o", ls="", color=P["init"], label="init"),
         Line2D([], [], marker="^", ls="", color=P["ink"], label="goal edit"),
         Line2D([], [], marker="s", ls="", mfc="white", color=P["ink"], label="proxy edit")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig_g_vs_df")


# ---------------------------------------------------------------- (c) probes by depth

SITES = [("conv1", r"$h^{(1)}$"), ("conv2", r"$h^{(2)}$"), ("conv3", r"$h^{(3)}$"), ("h_pre", r"$h_{\mathrm{pre}}$"),
         ("h_fc", r"$h_{\mathrm{fc}}$")]
# (env, title, targets averaged per seed, input-level control cohort)
PROBE_PANELS = [("AT", "cube colour", ["cube_is_red"], "pixels"),
                ("AT", "cube side", ["cube_left"], "pixels"),
                ("SZ", "cross on green", ["cross_is_green"], "pixels"),
                ("SZ", "cross side", ["cross_left"], "pixels"),
                ("LR", "marker position", ["marker_x", "marker_y"], "pixels"),
                ("LR", "marker bearing", ["bearing_x", "bearing_y"], "proprio")]


def probe_series(df, coh, targets, site):
    sub = df[(df.cohort == coh) & (df.site == site) & (df.target.isin(targets))]
    return sub.groupby("seed").value.mean().to_numpy()


def probes_by_depth():
    panels = [p for p in PROBE_PANELS if load("probes", p[0]) is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.55 * len(panels), 2.9))
    x = np.arange(len(SITES))
    for k, (ax, (env, title, tg, ctrl)) in enumerate(zip(np.atleast_1d(axes), panels)):
        df = load("probes", env)
        metric = df[df.target == tg[0]].metric.iloc[0]
        mi = np.array([ci(probe_series(df, "init", tg, s)) for s, _ in SITES])
        ax.fill_between(x, mi[:, 0] - mi[:, 1], mi[:, 0] + mi[:, 2], color=P["init"], alpha=0.18, lw=0)
        ax.plot(x, mi[:, 0], "--", color=P["init"], lw=1.6)
        for coh, col in (("control", P["nongm"]), ("gm", P["gm"])):
            ms = np.array([ci(probe_series(df, coh, tg, s)) for s, _ in SITES])
            ax.fill_between(x, ms[:, 0] - ms[:, 1], ms[:, 0] + ms[:, 2], color=col, alpha=0.18, lw=0)
            ax.plot(x, ms[:, 0], "-o", ms=4.5, color=col, mec="white", mew=0.8, lw=1.8)
        v = probe_series(df, ctrl, tg, "input")
        if len(v):
            ax.axhline(v.mean(), ls=":", color=P["null"], lw=1.4)
        ax.axhline(0.5 if metric == "auc" else 0.0, color=P["init"], lw=0.6, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in SITES], fontsize=8)
        ax.set_ylim((0.4, 1.02) if metric == "auc" else (-0.3, 1.02))
        ax.set_ylabel("AUC" if metric == "auc" else r"$R^2$")
        panel_label(ax, k, f"{ENVS[env]}\n{title}")
    h = [Line2D([], [], color=P["gm"], marker="o", mec="white", label="GM"),
         Line2D([], [], color=P["nongm"], marker="o", mec="white", label="Non-GM"),
         Line2D([], [], color=P["init"], ls="--", label="init"),
         Line2D([], [], color=P["null"], ls=":", label="input")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig_probes_by_depth")


# ---------------------------------------------------------------- (d) competition geometry

def cohort_points(ax, xs, vals_by_coh, offs=(-0.22, 0.0, 0.22)):
    for (coh, col), dx in zip((("gm", P["gm"]), ("control", P["nongm"]), ("init", P["init"])), offs):
        for x, v in zip(xs, vals_by_coh[coh]):
            if not len(v):
                continue
            m, lo, hi = ci(v)
            ax.scatter(np.full(len(v), x + dx - 0.07), v, s=12, color=col, alpha=0.4, lw=0)
            ax.errorbar(x + dx, m, yerr=[[lo], [hi]], fmt="o", ms=6, color=col, capsize=0, lw=1.6)


def competition():
    envs = [e for e in ("AT", "SZ") if load("axis", e) is not None]
    if not envs:
        return
    fig, axes = plt.subplots(len(envs), 4, figsize=(13, 3.0 * len(envs)), squeeze=False)
    for r, env in enumerate(envs):
        ps = load("axis", env)
        st = pd.read_csv(os.path.join(C.RESULTS, f"axis_{env}_per_state.csv"))
        sub = {c: ps[ps.cohort == c] for c in ("gm", "control", "init")}
        cue = "shape" if env == "AT" else "cross"
        ax = axes[r, 0]
        cohort_points(ax, [0, 1], {c: [s.cosE_within.to_numpy(), s.cosF_within.to_numpy()] for c, s in sub.items()})
        ax.set_xticks([0, 1], [r"$\cos_E$", r"$\cos_F$"])
        ax.axhline(0, color=P["init"], lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("cosine")
        panel_label(ax, 4 * r, f"{ENVS[env]}, within a state")
        for j, (name, pre) in enumerate((("Euclidean", "cosE"), ("Fisher", "cosF"))):
            ax = axes[r, 1 + j]
            cohort_points(ax, [0, 1], {c: [s[f"{pre}_same"].to_numpy(), s[f"{pre}_opp"].to_numpy()]
                                       for c, s in sub.items()})
            ax.set_xticks([0, 1], ["same side", "opposite side"])
            ax.axhline(0, color=P["init"], lw=0.6)
            ax.set_ylim(-1.05, 1.05)
            panel_label(ax, 4 * r + 1 + j, name)
        ax = axes[r, 3]
        allv = []
        for coh, col in (("init", P["init"]), ("control", P["nongm"]), ("gm", P["gm"])):
            s = st[st.cohort == coh]
            if not len(s):
                continue
            g, p = s.alpha_goal.abs().to_numpy(), s.alpha_prox.abs().to_numpy()
            allv += [g, p]
            ax.scatter(g, p, s=5, color=col, alpha=0.3, lw=0)
            pg = ps[ps.cohort == coh]
            mg, lg, hg = ci(pg.pull_goal)
            mp, lp, hp = ci(pg.pull_proxy)
            ax.errorbar(mg, mp, xerr=[[min(lg, 0.95 * mg)], [hg]], yerr=[[min(lp, 0.95 * mp)], [hp]], fmt="o", ms=6,
                        color=col, mec="white", mew=0.8, lw=1.6, zorder=4)
        v = np.concatenate(allv)
        v = v[v > 0]
        lim = (np.quantile(v, 0.01) / 2, v.max() * 2)
        ax.plot(lim, lim, "--", color=P["mute"], lw=1)
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_xlabel(rf"{cue} pull $|\alpha_F(\Delta_\star)|$")
        ax.set_ylabel(rf"colour pull $|\alpha_F(\Delta_p)|$")
        panel_label(ax, 4 * r + 3, "Tug of war")
    h = [Line2D([], [], marker="o", ls="", color=P["gm"], label="GM"),
         Line2D([], [], marker="o", ls="", color=P["nongm"], label="Non-GM"),
         Line2D([], [], marker="o", ls="", color=P["init"], label="init"),
         Line2D([], [], marker="o", color=P["mute"], label="mean ± 95% CI")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig_competition")


def competition_matrices(env="SZ"):
    """Supplementary: pairwise d_comp cosines between fit states (one seed per cohort), sorted by goal side."""
    d = os.path.join(C.RESULTS, "axis")
    files = {c: sorted(f for f in os.listdir(d) if f.startswith(f"{env}_{c}_s")) if os.path.isdir(d) else []
             for c in ("gm", "control", "init")}
    files = {c: f[0] for c, f in files.items() if f}
    if not files:
        return
    fig, axes = plt.subplots(2, len(files), figsize=(3.0 * len(files), 6.0), squeeze=False)
    for j, (coh, f) in enumerate(files.items()):
        z = np.load(os.path.join(d, f))
        if "cosE" not in z.files:
            continue
        for i, key in enumerate(("cosE", "cosF")):
            ax = axes[i, j]
            im = ax.imshow(z[key].astype(float), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            nl = int((z["sign_sorted"] > 0).sum())
            ax.axhline(nl - 0.5, color="k", lw=0.8), ax.axvline(nl - 0.5, color="k", lw=0.8)
            ax.set_xticks([]), ax.set_yticks([nl / 2, nl + (len(z["sign_sorted"]) - nl) / 2], ["left", "right"])
            ax.set_title(f"{COH[coh]}, {'Euclidean' if key == 'cosE' else 'Fisher'}", fontsize=9)
    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04)
    save(fig, f"fig_competition_matrices_{env}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["table", "gdf", "probes", "competition", "matrices"])
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fns = dict(table=outside_inside_table, gdf=g_vs_df, probes=probes_by_depth, competition=competition,
               matrices=competition_matrices)
    for w in a.which:
        fns[w]()


if __name__ == "__main__":
    main()
