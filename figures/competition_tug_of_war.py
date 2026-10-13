"""Attribute Maze tug of war on u_comp (black, h_fc): appendix figure, main-text panel 4, and the 4-panel main figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style as fs  # noqa: E402
from palette import PAL  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

DATA = DATA_ROOT / "probes/am_axis_tug_of_war_20260925"
FS_TICK, FS_LABEL, FS_TITLE = 7.0, 7.5, 8.0
LW = 0.7
T975_DF2 = 4.302652729911275
COHORTS = [("gm", "GM", "gm"), ("deco", "Non-GM", "nongm")]
AUC_SCORES = [("auc_roll_aF_star", "goal\npull"), ("auc_roll_neg_aF_p", "proxy\npull"), ("auc_roll_xF_mid", "both")]
LINTHRESH = 1.0
N_BINS = 5


def style() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                         "pdf.fonttype": 42, "mathtext.fontset": "dejavusans", "axes.linewidth": LW,
                         "axes.edgecolor": PAL["ink"], "text.color": PAL["ink"], "xtick.color": PAL["ink"],
                         "ytick.color": PAL["ink"], "axes.labelcolor": PAL["ink"],
                         "figure.facecolor": PAL["paper"], "axes.facecolor": PAL["paper"]})


def finish(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_TICK, length=2, width=LW, pad=1.5)


def load(summary: dict, data: Path = DATA) -> pd.DataFrame:
    """Per-state rows for GM and Non-GM, checked against summary.json so the figure cannot drift from the table."""
    df = pd.concat([pd.read_csv(f) for f in sorted((data / "per_seed").glob("per_state_*_black_h_fc.csv"))])
    df = df[(df.axis == "unit_mean") & (df.cohort.isin(["gm", "deco"]))].copy()
    df["seed"] = df.seed.astype(str)
    for co, _, _ in COHORTS:
        ev = df[(df.cohort == co) & (df.split_role == "eval")]
        for col, key in (("aF_star", "F_mean_a_star"), ("aF_p", "F_mean_a_p"), ("tF", "median_tF")):
            agg = ev.groupby("seed")[col].median() if col == "tF" else ev.groupby("seed")[col].mean()
            ref = summary["eval50"][co][key]["per_seed"]
            assert all(abs(agg[s] - ref[s]) < 1e-9 for s in ref), (co, col)
        assert len(ev) == 150, (co, len(ev))
    return df


def stat(summary: dict, mask: str, co: str, key: str) -> dict:
    v = summary[mask][co][key]
    return {"mean": v["mean"], "lo": v["lo"], "hi": v["hi"], "sd": v["sd"], "per_seed": v["per_seed"]}


def panel_pulls(ax, df: pd.DataFrame, summary: dict, record: dict, compact: bool) -> list:
    ev = df[df.split_role == "eval"]
    # Symlog on both axes keeps the identity line straight while showing GM (~6, ~9) and Non-GM (~17, ~1) at once.
    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=0.5)
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
    lim = (-3, 70)
    ax.plot(np.linspace(*lim, 400), np.linspace(*lim, 400), color=PAL["mute"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    handles = []
    for co, name, col in COHORTS:
        g = ev[ev.cohort == co]
        ax.scatter(g.aF_star, g.aF_p, s=1.6 if compact else 2.2, color=PAL[col], alpha=0.3, lw=0, zorder=2,
                   rasterized=True)
        x, y = stat(summary, "eval50", co, "F_mean_a_star"), stat(summary, "eval50", co, "F_mean_a_p")
        ax.errorbar(x["mean"], y["mean"], xerr=[[x["mean"] - x["lo"]], [x["hi"] - x["mean"]]],
                    yerr=[[y["mean"] - y["lo"]], [y["hi"] - y["mean"]]], fmt="o", ms=3.0, mfc=PAL[col],
                    mec=PAL["paper"], mew=0.4, color=PAL[col], capsize=1.4, capthick=LW, elinewidth=LW, zorder=4)
        record[f"pulls|{name}"] = {"a_star": x, "a_p": y, "n_points": int(len(g))}
        handles.append(Line2D([], [], ls="", marker="o", ms=3.0, mfc=PAL[col], mec=PAL[col], label=name))
    handles.append(Line2D([], [], color=PAL["mute"], lw=0.6, ls=(0, (3, 2)), label="equal pull"))
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ticks = [1, 10]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(r"shape pull $\alpha_F(\Delta_\star)$", fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel(r"colour pull $\alpha_F(\Delta_p)$", fontsize=FS_LABEL, labelpad=1.5)
    finish(ax)
    return handles


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def panel_ordering(ax, df: pd.DataFrame, record: dict) -> list:
    """Goal-outcome fraction per quantile bin of t, GM, all 196 states pooled over seeds, timeouts excluded."""
    g = df[(df.cohort == "gm") & df.native_rollout_outcome.isin(["goal", "proxy"])]
    t = g.tF.to_numpy()
    y = (g.native_rollout_outcome == "goal").to_numpy()
    edges = np.quantile(t, np.linspace(0, 1, N_BINS + 1))
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, N_BINS - 1)
    rows = []
    for b in range(N_BINS):
        m = idx == b
        k, n = int(y[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        rows.append({"bin": b, "n": n, "k_goal": k, "t_mean": float(t[m].mean()), "t_median": float(np.median(t[m])),
                     "t_lo": float(edges[b]), "t_hi": float(edges[b + 1]), "frac_goal": k / n, "wilson_lo": lo,
                     "wilson_hi": hi})
    x = np.array([r["t_mean"] for r in rows])
    f = np.array([r["frac_goal"] for r in rows])
    err = np.array([[r["frac_goal"] - r["wilson_lo"] for r in rows], [r["wilson_hi"] - r["frac_goal"] for r in rows]])
    base = float(y.mean())
    ax.axhline(base, color=PAL["init"], lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.axvline(0.5, color=PAL["mute"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot(x, f, color=PAL["gm"], lw=LW, zorder=3)
    ax.errorbar(x, f, yerr=err, fmt="o", ms=2.8, mfc=PAL["gm"], mec=PAL["gm"], color=PAL["gm"], capsize=1.6,
                capthick=LW, elinewidth=LW, zorder=4)
    record["ordering|GM|all196|pooled"] = {"n_states": int(len(y)), "n_goal": int(y.sum()), "base_rate": base,
                                           "bins": rows}
    ax.set_xlim(0.1, 0.75)
    ax.set_xticks([0.2, 0.4, 0.6])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"base position $t$ (proxy $\rightarrow$ goal)", fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel("fraction goal", fontsize=FS_LABEL, labelpad=1.5)
    finish(ax)
    return [Line2D([], [], ls="-", marker="o", ms=2.8, color=PAL["gm"], lw=LW, label="GM"),
            Line2D([], [], color=PAL["init"], lw=0.6, ls=(0, (1, 1.5)), label="all states"),
            Line2D([], [], color=PAL["mute"], lw=0.6, ls=(0, (3, 2)), label="$t=0.5$")]


def panel_auc(ax, summary: dict, record: dict) -> list:
    styles = [dict(mfc=PAL["gm"], mec=PAL["gm"]), dict(mfc="none", mec=PAL["gm_proxy"]),
              dict(mfc=PAL["gm"], mec=PAL["gm"])]
    markers = ["o", "o", "D"]
    for i, ((key, _), st, mk) in enumerate(zip(AUC_SCORES, styles, markers)):
        v = stat(summary, "all196", "gm", key)
        seeds = list(v["per_seed"].values())
        ax.scatter(np.full(3, i - 0.2), seeds, s=2.0, color=st["mec"], alpha=0.5, lw=0, zorder=3)
        col = st["mec"]
        ax.errorbar(i + 0.08, v["mean"], yerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]], fmt=mk,
                    ms=2.8 if mk == "o" else 2.5, mfc=st["mfc"], mec=col, mew=LW, color=col, capsize=1.6,
                    capthick=LW, elinewidth=LW, zorder=4)
        record[f"auc|{key}|gm|all196"] = v
    ax.axhline(0.5, color=PAL["init"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(AUC_SCORES)), [s[1] for s in AUC_SCORES], fontsize=FS_LABEL)
    ax.set_xlim(-0.6, len(AUC_SCORES) - 0.4)
    ax.set_ylim(-0.08, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("AUC (goal vs proxy)", fontsize=FS_LABEL, labelpad=1.5)
    finish(ax)
    ax.tick_params(axis="x", labelsize=FS_LABEL)
    return [Line2D([], [], color=PAL["init"], lw=0.6, ls=(0, (3, 2)), label="chance")]


def legend_below(ax, handles, ncol, dy=-0.30, **kw) -> None:
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, frameon=False,
              fontsize=FS_TICK, handletextpad=0.25, columnspacing=0.7, borderaxespad=0, handlelength=1.3, **kw)


def apply_appendix_type(ax) -> None:
    """Paper appendix type sizes (fig_style_main), applied after the shared panel helpers, which keep the
    main-text sizes for the main figures."""
    ax.xaxis.label.set_size(fs.FS)
    ax.yaxis.label.set_size(fs.FS)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(fs.FS_TICK)
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(fs.FS_LEG)


def appendix_figure(df, summary, out: Path, record: dict) -> None:
    """One row (a)(b)(c), sized natively for inclusion at 0.9 linewidth (4.95 in)."""
    W, H = 4.95, 1.85

    def box(x0, y0, w, h):
        return (x0 / W, y0 / H, w / W, h / H)

    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes(box(0.42, 0.62, 1.12, 0.95))
    b = fig.add_axes(box(2.08, 0.62, 1.12, 0.95))
    c = fig.add_axes(box(3.72, 0.62, 1.1, 0.95))
    legend_below(a, panel_pulls(a, df, summary, record, False), ncol=2, dy=-0.36)
    legend_below(b, panel_ordering(b, df, record), ncol=2, dy=-0.36)
    legend_below(c, panel_auc(c, summary, record), ncol=1, dy=-0.36)
    for ax, (letter, t) in zip((a, b, c), (("a", "Per-state pulls"), ("b", "Base position"), ("c", "Outcome AUC"))):
        apply_appendix_type(ax)
        ax.set_title(rf"$\bf{{({letter})}}$ {t}", loc="left", fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"), dpi=600)
    plt.close(fig)


def compact_panel(ax, df, summary, record) -> list:
    handles = panel_pulls(ax, df, summary, record, True)
    ax.set_title("Tug of war", fontsize=FS_TITLE, fontweight="bold", pad=3)
    ax.set_xlabel(r"shape $\alpha_F(\Delta_\star)$", fontsize=FS_TICK, labelpad=1)
    ax.set_ylabel(r"colour $\alpha_F(\Delta_p)$", fontsize=FS_TICK, labelpad=1)
    return handles


def panel4_figure(df, summary, out: Path, record: dict) -> None:
    fig = plt.figure(figsize=(1.4, 1.6))
    ax = fig.add_axes((0.3, 0.38, 0.64, 0.5))
    handles = compact_panel(ax, df, summary, record)
    ax.legend(handles=[handles[0], handles[2], handles[1]], loc="upper center", bbox_to_anchor=(0.36, -0.3), ncol=2, frameon=False,
              fontsize=FS_TICK - 0.5, handletextpad=0.2, columnspacing=0.6, borderaxespad=0, handlelength=1.3)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"), dpi=600)
    plt.close(fig)


def four_panel_figure(df, summary, out: Path, record: dict, tap: str) -> None:
    import competition_base as m3

    from canon_geometry_run import BANK_DIR, load_state_table  # noqa: E402
    cmap = plt.get_cmap("RdBu_r").copy()
    # Diagonal (self-similarity) is NaN by construction; grey keeps it distinct from kappa = 0 (white).
    cmap.set_bad("#BDBDBD")
    st = load_state_table(BANK_DIR / "ras_gt_bank_canonical.json")
    fig = plt.figure(figsize=(5.5, 1.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 0.82, 0.82, 1.0], left=0.085, right=0.985, top=0.87,
                          bottom=0.235, wspace=0.22)
    ax0 = fig.add_subplot(gs[0])
    m3.panel_within(ax0, tap, record)
    ax0.set_title("Within a state", fontsize=FS_TITLE, fontweight="bold", pad=3)
    im, _, _ = m3.panel_heat(fig.add_subplot(gs[1]), "kappa_e", tap, st, "Across, $\\kappa_E$", True, cmap)
    ax_f = fig.add_subplot(gs[2])
    m3.panel_heat(ax_f, "kappa_f", tap, st, "Across, $\\kappa_F$", False, cmap)
    ax_f.apply_aspect()
    pos = ax_f.get_position()
    cb = fig.colorbar(im, cax=fig.add_axes((pos.x1 + 0.008, pos.y0, 0.009, pos.height)), ticks=[-1, 0, 1])
    cb.ax.tick_params(labelsize=FS_TICK, length=1.5, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.solids.set_rasterized(True)
    ax3 = fig.add_subplot(gs[3])
    h3 = compact_panel(ax3, df, summary, record)
    p3 = ax3.get_position()
    # Extra left margin keeps the y-label clear of the colourbar, and lifting the bottom makes room for the legend.
    ax3.set_position((p3.x0 + 0.075, p3.y0 + 0.07, p3.width - 0.075, p3.height - 0.07))
    ax3.legend(handles=h3[-1:], loc="upper center", bbox_to_anchor=(0.45, -0.34), frameon=False, fontsize=FS_TICK,
               handletextpad=0.3, borderaxespad=0, handlelength=1.3)
    handles = [Line2D([], [], ls="", marker="o", ms=2.6, mfc=PAL[c], mec=PAL[c], mew=LW, label=name)
               for _, name, c in m3.COHORTS]
    ax0.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
               fontsize=FS_TICK, handletextpad=0.1, columnspacing=0.8, borderaxespad=0, handlelength=1.0)
    fig.savefig(out.with_suffix(".png"), dpi=600)
    fig.savefig(out.with_suffix(".pdf"), dpi=600)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--tap", default="h_fc", choices=["h_fc"])
    ap.add_argument("--fig-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--values-out", type=Path, default=TABLES / "fig_tug_of_war_values.json")
    ap.add_argument("--appendix-only", action="store_true", help="skip the main-text figures")
    args = ap.parse_args()
    style()
    summary = json.loads((args.data / "summary.json").read_text())["summary"]["unit_mean"]
    df = load(summary, args.data)
    record: dict = {"source": "data/probes/am_axis_tug_of_war_20260925/summary.json", "axis": "unit_mean",
                    "tap": args.tap, "background": "black", "states_a_b": "eval50", "states_c": "all196"}
    appendix_figure(df, summary, args.fig_dir / "fig_appendix_attribute_tug_of_war", record)
    if args.appendix_only:
        args.values_out.with_name("fig_tug_of_war_appendix_values.json").write_text(json.dumps(record, indent=1, default=float))
        return
    panel4_figure(df, summary, args.fig_dir / "fig_main_sec3_4_panel4", record)
    four_panel_figure(df, summary, args.fig_dir / "fig_main_sec3_4_4panel", record, args.tap)
    args.values_out.write_text(json.dumps(record, indent=1, default=float))
    print("wrote", args.fig_dir, args.values_out)


if __name__ == "__main__":
    main()
