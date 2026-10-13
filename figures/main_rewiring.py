"""Main-text rewiring figure (Sec. 4.2): closed-loop intended-goal reach after rewiring at h_pre, the fixed remix, and h_fc, GM vs Non-GM, against the recovery-rule floor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from scipy import stats  # noqa: E402

import style as fs  # noqa: E402
from steering_base import polys_overlap, text_poly  # noqa: E402
from palette import PAL  # noqa: E402

# Per-seed cells of the appendix recovery figure (oracle targets in Maze and Attribute Maze, teacher targets elsewhere),
# so the main figure shows exactly the numbers of tab:rewire-main.
CELLS = TABLES / "fig_app_rewire_recovery_bars_mixed.csv"
# Print minimum at \linewidth; the shared style's 6.5pt ticks and legend fall under it.
FS_MIN = 7.0
W_IN, H_IN = fs.W_IN, 1.40
# Panel layout in inches: left margin (y label, tick labels, "params"), right margin, gap between panels,
# axes bottom (legend, params row, tick labels below it) and top margin (title).
L, R, GAP, B, T = 0.47, 0.03, 0.14, 0.48, 0.17
PARAMS_DY = -13.5
COH = {"GM": PAL["gm"], "Non-GM": PAL["nongm"]}
OFFSET = {"GM": -0.13, "Non-GM": 0.13}
# Trainable parameter counts (policy head + 2048->256 map) are in the tick labels because depth and function class change together.
ARMS = [("pre", "$h_{\\mathrm{pre}}$", "528k"), ("remix", "remix", "4k"), ("fc", "$h_{\\mathrm{fc}}$", "4k")]
# DAgger (final round) is shown for Maze at both sites (oracle relabeling) and for PushToMarker from h_pre (teacher
# relabeling), the same PushToMarker rewire as Figure 1 and the appendix overview.
MAZE_ARMS = [("pre", "$h_{\\mathrm{pre}}$", "528k"), ("dagger_pre", "DAgger", "528k"), ("remix", "remix", "4k"),
             ("fc", "$h_{\\mathrm{fc}}$", "4k"), ("dagger_fc", "DAgger", "4k")]
PTM_ARMS = [("pre", "$h_{\\mathrm{pre}}$", "528k"), ("dagger_pre", "DAgger", "528k"), ("remix", "remix", "4k"),
            ("fc", "$h_{\\mathrm{fc}}$", "4k")]
DAGGER_CSV = TABLES / "fig_maze_rewire_overview_raw.csv"
# tab:rewire-dagger PushToMarker rows (site pre, mode dagger, round 3, set conflict, cond standard).
PTM_DAGGER_CSV = ROBOTICS_ROOT / "results/analysis/rewire_ptm_dagger/per_seed.csv"
PTM_COHORT = {"gm": "GM", "control": "Non-GM", "init": "init"}
# The robot network has a two-layer head and a 9216->256 map (tab:rewire-main Baselines paragraph), so its counts differ.
ROBOT_PARAMS = {"pre": "2.6M", "dagger_pre": "2.6M", "remix": "267k", "fc": "267k"}
# Cells drawn hatched because the success criterion rates them partial (tab:rewire-dagger, underlined).
PARTIAL = {("PTM", "GM", "dagger_pre")}
HATCH = "//////"
# Extra room after the second Maze arm so the "DAgger" and "remix" tick labels do not touch.
MAZE_SHIFT = 0.3


def mean_ci(v) -> tuple[float, float, float]:
    """Mean and 95% t-interval across the three checkpoints, clipped to [0, 1] since reach is a proportion."""
    v = np.asarray(v, float)
    m = float(v.mean())
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, max(0.0, m - h), min(1.0, m + h)


def load_env(env: str) -> dict:
    """Panel data for one environment from CELLS; the floor at each site is the higher of random trunk and chance."""
    import csv
    cells = {r["arm"]: [float(x) for x in r["seeds"].split(";")] for r in csv.DictReader(open(CELLS)) if r["env"] == env}
    chance = float(np.mean(cells["chance"]))
    pre, fc = max(np.mean(cells["init"]), chance), max(np.mean(cells["init_fc"]), chance)
    out = {"floor": {"pre": pre, "remix": pre, "fc": fc},
           "native": {"GM": float(np.mean(cells["native_gm"])), "Non-GM": float(np.mean(cells["native_nongm"]))}}
    for c, key in (("GM", "gm"), ("Non-GM", "nongm")):
        out[c] = {a: cells[f"{key}_{a}"] for a in ("pre", "remix", "fc")}
    if env == "Maze":
        for r in csv.DictReader(open(DAGGER_CSV)):
            if r["attempt"] == "dagger":
                out[r["cohort"]].setdefault(f"dagger_{'pre' if r['site'] == 'block3' else 'fc'}", []).append(float(r["cheese_reach"]))
        out["floor"]["dagger_pre"], out["floor"]["dagger_fc"] = pre, fc
    if env == "PTM":
        dag: dict = {}
        for r in csv.DictReader(open(PTM_DAGGER_CSV)):
            if (r["site"], r["mode"], r["round"], r["set"], r["cond"]) == ("pre", "dagger", "3", "conflict", "standard"):
                dag.setdefault(PTM_COHORT[r["cohort"]], []).append(float(r["goal"]))
        out["GM"]["dagger_pre"], out["Non-GM"]["dagger_pre"] = dag["GM"], dag["Non-GM"]
        out["floor"]["dagger_pre"] = max(float(np.mean(dag["init"])), chance)
    out["partial"] = {(c, a) for e, c, a in PARTIAL if e == env}
    return out


def panel(ax, d: dict, rec: dict, robot: bool = False) -> None:
    arms = MAZE_ARMS if "dagger_fc" in d["GM"] else PTM_ARMS if "dagger_pre" in d["GM"] else ARMS
    shifted = arms is not ARMS
    if robot:
        arms = [(a, lab, ROBOT_PARAMS[a]) for a, lab, _ in arms]
    x = np.arange(len(arms)) + (MAZE_SHIFT * (np.arange(len(arms)) >= 2) if shifted else 0.0)
    ax.axhspan(0, d["native"]["GM"], color=COH["GM"], alpha=0.08, lw=0, zorder=0)
    ax.axhspan(d["native"]["Non-GM"], 1.02, color=COH["Non-GM"], alpha=0.08, lw=0, zorder=0)
    for i, (a, _, _) in enumerate(arms):
        ax.hlines(d["floor"][a], x[i] - 0.34, x[i] + 0.34, color=PAL["init"], lw=1.0, ls=(0, (1.2, 1.2)), zorder=2)
    for c, col in COH.items():
        ax.axhline(d["native"][c], color=col, lw=0.8, ls=(0, (4, 2)), zorder=1)
        r = rec.setdefault(c, {"native": round(float(d["native"][c]), 3)})
        for i, (a, _, _) in enumerate(arms):
            v = np.asarray(d[c][a], float)
            m, lo, hi = mean_ci(v)
            r[a] = {"mean": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)], "seeds": [round(s, 3) for s in v]}
            xi = x[i] + OFFSET[c]
            hollow = a == "remix"
            diamond = a.startswith("dagger")
            ax.scatter(np.full(len(v), xi + 0.06), v, color=col, zorder=3, **fs.SEED)
            ax.vlines(xi, lo, hi, color=col, lw=fs.LW_ERR, zorder=4)
            if (c, a) in d["partial"]:
                ax.scatter([xi], [m], marker="D", s=(fs.MS + 1.6) ** 2, facecolor="white", edgecolor=col, hatch=HATCH,
                           lw=0.8, zorder=5)
            else:
                ax.plot(xi, m, "s" if hollow else ("D" if diamond else "o"), color=col, mfc="white" if hollow else col,
                        mec=col if hollow else "white", mew=0.9 if hollow else 0.4, ms=fs.MS + 0.4, zorder=5)
    rec["floor"] = {k: round(float(v), 3) for k, v in d["floor"].items()}
    ax.set_xticks(x, [lab for _, lab, _ in arms])
    for i, (_, _, n) in enumerate(arms):
        ax.annotate(n, xy=(x[i], 0), xycoords=("data", "axes fraction"), xytext=(0, PARAMS_DY), textcoords="offset points",
                    ha="center", va="top", fontsize=FS_MIN, color=PAL["mute"])
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    fs.finish(ax)


def check_layout(fig) -> None:
    """Assert every visible text is >= FS_MIN, inside the figure, and overlaps no other text or the legend."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    leg = fig.legends[0]
    for t in texts:
        assert t.get_fontsize() >= FS_MIN - 1e-9, f"{t.get_text()!r} at {t.get_fontsize()}pt"
    polys = [(t.get_text(), text_poly(t, r), any(t is u for u in leg.get_texts())) for t in texts]
    fb = fig.bbox
    for n, p, _ in polys:
        assert (p[:, 0].min() >= fb.x0 - 0.5 and p[:, 0].max() <= fb.x1 + 0.5
                and p[:, 1].min() >= fb.y0 - 0.5 and p[:, 1].max() <= fb.y1 + 0.5), f"{n!r} clipped"
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys_overlap(polys[i][1], polys[j][1]):
                raise AssertionError(f"overlap: {polys[i][0]!r} vs {polys[j][0]!r}")
    e = leg.get_window_extent(r)
    assert e.x0 >= fb.x0 - 0.5 and e.x1 <= fb.x1 + 0.5, "legend wider than the figure"
    box = np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]])
    for n, p, in_leg in polys:
        assert in_leg or not polys_overlap(p, box), f"{n!r} hits legend"
    print(f"layout ok: {len(polys)} text boxes, all >= {FS_MIN}pt, no overlaps")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DIR / "fig_main_rewiring"))
    ap.add_argument("--values", default=str(TABLES / "main_rewiring_values.json"))
    args = ap.parse_args()

    fs.FS_TICK = fs.FS_LEG = max(fs.FS_TICK, FS_MIN)
    fs.style()
    panels = [("Maze", load_env("Maze")), ("PushToMarker", load_env("PTM")), ("Coin", load_env("Coin")),
              ("Attribute Maze", load_env("AM"))]
    widths = [len(MAZE_ARMS) + MAZE_SHIFT, len(PTM_ARMS) + MAZE_SHIFT, len(ARMS), len(ARMS)]
    fig = plt.figure(figsize=(W_IN, H_IN))
    unit = (W_IN - L - R - (len(panels) - 1) * GAP) / sum(widths)
    x0s = [L + sum(widths[:i]) * unit + i * GAP for i in range(len(panels))]
    axes = []
    for x0, n in zip(x0s, widths):
        axes.append(fig.add_axes([x0 / W_IN, B / H_IN, n * unit / W_IN, (H_IN - B - T) / H_IN],
                                 sharey=axes[0] if axes else None))
    record: dict = {"config": vars(args)}
    for i, (ax, (t, d)) in enumerate(zip(axes, panels)):
        panel(ax, d, record.setdefault(t, {}), robot=t == "PushToMarker")
        fs.panel_title(ax, t)
        fs.panel_letter(ax, "abcd"[i], dx=-24.0 if i == 0 else -12.0)
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("goal reach")
    axes[0].annotate("params", xy=(0, 0), xycoords="axes fraction", xytext=(-4, PARAMS_DY), textcoords="offset points",
                     ha="right", va="top", fontsize=FS_MIN, color=PAL["mute"])
    handles = [fs.cohort_handle("GM"), fs.cohort_handle("Non-GM"),
               Line2D([], [], ls="", marker="s", color=PAL["mute"], mfc="white", ms=fs.MS + 0.4, mew=0.9),
               Line2D([], [], ls="", marker="D", color=PAL["mute"], mec="white", ms=fs.MS, mew=0.4),
               Patch(facecolor="white", edgecolor=PAL["mute"], hatch=HATCH, lw=0.8)]
    handles += [(Patch(color=COH[c], alpha=0.12, lw=0), Line2D([], [], color=COH[c], lw=0.8, ls=(0, (4, 2)))) for c in COH]
    handles += [Line2D([], [], color=PAL["init"], lw=1.0, ls=(0, (1.2, 1.2)))]
    labels = ["GM", "Non-GM", "random remix", "DAgger", "partial", "native GM", "native Non-GM", "floor"]
    fs.legend_row(fig, handles, labels, y_in=0.02, columnspacing=0.6, handlelength=1.4,
                  handletextpad=0.3)
    check_layout(fig)
    # Fixed canvas (no tight bbox) so the file is exactly \linewidth wide and 7pt prints as 7pt.
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    Path(args.values).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
