"""Cross-environment Fisher figures: G vs D_F decomposition (main and appendix) and the signed target-selection raincloud."""
from __future__ import annotations

import os
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
DATA = DATA_ROOT / "probes"
FIGS = ROOT / "paper_figures"

MCFISHER = DATA / "canon_mcfisher_20260924/summary_mcfisher.json"
MAZE_PERSTATE = DATA / "canon_mcfisher_20260924/perstate_signed_derivative_mcfisher.json"
# Coin is read only from the canonical 600-state files below. The Coin rows inside
# canon_mcfisher_20260924 come from a different bank and are deliberately not used.
COIN_QGD = DATA / "coin_fisher_canon_20260924/reaggregated_paper_convention.json"
# Holds the six trained checkpoints (bit-identical to the trained-only scoring) plus the three init draws.
COIN_SIGNED = DATA / "coin_signed_taskdef_init_20260924"
RAS = DATA / "canon_fisher_20260924/summary_canon_swap.json"
RAS_ROWS = DATA / "canon_fisher_20260924/rows_swap_pair"

# The paper's Fisher table reports Attribute Maze on the black background;
# the appendix decomposition adds the texture background as a fourth column.
RAS_BG = "black"

LABEL_FS, TICK_FS, LEG_FS, SUB_FS = 15, 13, 12, 14

# Paper palette.
BURGUNDY = "#7A1520"   # GM
TEAL = "#3D8B8B"       # Non-GM
MUTED = "#8E8E8E"      # init / architectural reference

COHORT_COLOR = {"GM": BURGUNDY, "Non-GM": TEAL, "init": MUTED}
TAPS = [("block3", r"$h_\mathrm{pre}$"), ("fc", r"$h_\mathrm{fc}$")]

provenance: list[str] = []


def note(line: str) -> None:
    provenance.append(line)
    print(line, flush=True)


# ----------------------------------------------------------------- loaders
def maze_points(tap: str) -> list[dict]:
    """Maze G/D_F per seed (corrected Jacobian)."""
    fig = json.loads(MCFISHER.read_text())["figure"]["maze"][tap]
    out = []
    for cohort, key in (("GM", "gm"), ("Non-GM", "nongm"), ("init", "init")):
        g = fig["G"][f"{key}_seed_values"]
        d = fig["D_F"][f"{key}_seed_values"]
        out.append({"cohort": cohort, "G": g, "D_F": d, "marker": "o", "edit": None})
    note(f"Maze/{tap}: G,D_F <- {MCFISHER.name}[figure][maze][{tap}], 3 seeds/cohort")
    return out


def coin_points(tap: str) -> list[dict]:
    """Coin G/D_F per checkpoint, canonical 600-state bank, seed medians."""
    key = "h_fc" if tap == "fc" else "h_pre"
    blob = json.loads(COIN_QGD.read_text())[key]
    out = []
    for cohort in ("GM", "Non-GM", "init"):
        per = blob[cohort]["per_ckpt"]
        g = [per[c]["G_median"] for c in per]
        d = [per[c]["D_F_median"] for c in per]
        out.append({"cohort": cohort, "G": g, "D_F": d, "marker": "o", "edit": None})
    note(f"Coin/{tap}: G,D_F <- {COIN_QGD.name}[{key}] per_ckpt medians, canonical 600-state bank")
    return out


def ras_points(tap: str, bg: str = RAS_BG) -> list[dict]:
    """Attribute Maze G/D_F per seed, both swap edits. Shape = triangle, colour = square."""
    fig = json.loads(RAS.read_text())["figure_ras"][bg][tap]
    out = []
    for edit, marker in (("shape", "^"), ("colour", "s")):
        for cohort, key in (("GM", "gm"), ("Non-GM", "nongm"), ("init", "init")):
            out.append({
                "cohort": cohort,
                "G": fig[f"G_{edit}"][f"{key}_seed_values"],
                "D_F": fig[f"D_F_{edit}"][f"{key}_seed_values"],
                "marker": marker,
                "edit": edit,
            })
    note(f"Attribute Maze/{tap}: G,D_F <- {RAS.name}[figure_ras][{bg}][{tap}], "
         f"shape+colour swaps, 3 seeds/cohort")
    return out


def maze_signed(tap: str) -> dict:
    """Per-state g and per-seed medians, Maze."""
    per = json.loads(MAZE_PERSTATE.read_text())["MAZE"]
    groups = {"GM": ["gm_1080", "gm_1081", "gm_1082"],
              "Non-GM": ["nongm_1090", "nongm_1091", "nongm_1092"],
              "init": [k for k in per if k.startswith("init_")]}
    out = {}
    for cohort, ckpts in groups.items():
        states, seedmeds = [], []
        for c in ckpts:
            if c not in per or tap not in per[c]:
                continue
            v = per[c][tap]["g"]
            states.extend(v)
            seedmeds.append(float(np.median(v)))
        out[cohort] = {"states": states, "seed_medians": seedmeds}
    note(f"Maze/{tap}: per-state g <- {MAZE_PERSTATE.name}[MAZE], "
         f"{len(out['GM']['states'])} GM states")
    return out


def coin_signed(tap: str) -> dict:
    """Per-state task-defined LEFT-minus-RIGHT g for Coin, pooling both sources into the 600-state bank."""
    fp = COIN_SIGNED / ("coin_signed_taskdefined_canon_block3.json" if tap == "block3"
                        else "coin_signed_taskdefined_canon.json")
    blob = json.loads(fp.read_text())
    pf = blob["per_frame"]
    groups = {"GM": ["gm_6033", "gm_6037", "gm_6051"],
              "Non-GM": ["nongm_7001", "nongm_7002", "nongm_7003"],
              "init": [f"init_{i}" for i in range(blob.get("n_init", 3))]}
    out = {}
    for cohort, ckpts in groups.items():
        states, seedmeds = [], []
        for c in ckpts:
            v = [r["g_star_task"] for src in pf for r in pf[src].get(c, [])
                 if "g_star_task" in r]
            states.extend(v)
            seedmeds.append(float(np.median(v)))
        out[cohort] = {"states": states, "seed_medians": seedmeds}
    note(f"Coin/{tap}: per-state task-defined g <- {fp.name} "
         f"(LEFT{blob['LEFT_ACTIONS']} minus RIGHT{blob['RIGHT_ACTIONS']}), "
         f"{len(out['GM']['states'])} GM states, both sources pooled, "
         f"{len(out['init']['states'])} init states")
    return out


# The per-state CSVs label taps h_pre/h_fc, while the summary JSONs key them
# block3/fc. Same two sites, different spelling.
TAP_CSV = {"block3": "h_pre", "fc": "h_fc"}


def ras_signed(tap: str, edit: str) -> dict:
    """Per-state swap g, Attribute Maze."""
    col = f"g_{edit}_swap"
    tap_csv = TAP_CSV[tap]
    groups = {"GM": ("gm", ["6313", "6315", "6317"]),
              "Non-GM": ("deco", ["4371", "4372", "4373"]),
              "init": ("init", ["3000", "3001", "3002"])}
    out = {}
    for cohort, (tag, seeds) in groups.items():
        states, seedmeds = [], []
        for s in seeds:
            fp = RAS_ROWS / f"rows_swap_pair_{RAS_BG}_{tag}_{s}.csv"
            if not fp.exists():
                continue
            vals = []
            with open(fp, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("tap") != tap_csv or not row.get(col):
                        continue
                    try:
                        vals.append(float(row[col]))
                    except ValueError:
                        continue
            if vals:
                states.extend(vals)
                seedmeds.append(float(np.median(vals)))
        out[cohort] = {"states": states, "seed_medians": seedmeds}
    note(f"Attribute Maze/{tap}/{edit}: per-state g <- {RAS_ROWS.name}/*_{RAS_BG}_*.csv "
         f"tap={tap_csv} col={col}, {len(out['GM']['states'])} GM states, "
         f"{len(out['init']['states'])} init states")
    return out


# ----------------------------------------------- figure 1: G vs D_F
def draw_q_contours(ax, xlim, ylim):
    """Constant-Q reference contours. Q = G * D_F, so on log-log these are
    straight lines of slope -1."""
    for q in [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]:
        xs = np.array(xlim)
        ys = q / xs
        ax.plot(xs, ys, ls=":", lw=0.9, color="#BBBBBB", zorder=0)


def fig_decomposition(ras_bgs: tuple[str, ...], stem: str, figsize: tuple[float, float],
                      compact: bool = False, taps: list = TAPS, summary: bool = False) -> None:
    """G vs D_F grid: rows = taps, columns = Maze, Coin, one per Attribute Maze background."""
    fs = dict(label=8, tick=6.5, leg=7, title=8.5, ms=22, lw=0.5, lms=5) if compact else \
        dict(label=LABEL_FS, tick=TICK_FS - 1, leg=LEG_FS + 1, title=SUB_FS + 1, ms=70, lw=0.7, lms=9)
    cols = [("Maze", maze_points), ("Coin", coin_points)]
    for bg in ras_bgs:
        title = "Attribute Maze" if len(ras_bgs) == 1 else f"Attribute Maze\n({bg})"
        cols.append((title, lambda tap, bg=bg: ras_points(tap, bg)))
    fig, axes = plt.subplots(len(taps), len(cols), figsize=figsize, squeeze=False)
    last = len(taps) - 1

    for r, (tap, tap_lab) in enumerate(taps):
        for c, (env, loader) in enumerate(cols):
            ax = axes[r, c]
            for grp in loader(tap):
                xs = np.asarray(grp["D_F"], dtype=float)
                ys = np.asarray(grp["G"], dtype=float)
                keep = (xs > 0) & (ys > 0)
                if not keep.any():
                    continue
                color = COHORT_COLOR[grp["cohort"]]
                if not summary:
                    ax.scatter(xs[keep], ys[keep], s=fs["ms"], color=color, marker=grp["marker"],
                               edgecolor="black", linewidth=fs["lw"], zorder=3,
                               alpha=0.55 if grp["cohort"] == "init" else 0.95)
                    continue
                xs, ys = xs[keep], ys[keep]
                ax.scatter(xs, ys, s=fs["ms"] * 0.35, color=color, marker=grp["marker"],
                           linewidth=0, alpha=0.5, zorder=2)
                # Geometric mean: the axes are log, so the arithmetic mean would sit
                # off-centre of the seeds it summarises.
                gx, gy = np.exp(np.log(xs).mean()), np.exp(np.log(ys).mean())
                ax.scatter([gx], [gy], s=fs["ms"] * 1.9, color=color, marker=grp["marker"],
                           edgecolor="black", linewidth=fs["lw"], zorder=4)
            # Q spans about seven decades across environments; log axes also make constant-Q contours straight.
            ax.set_xscale("log")
            ax.set_yscale("log")
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            draw_q_contours(ax, xlim, ylim)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=fs["tick"])
            if compact:
                ax.tick_params(length=2, pad=1.5)
            # No rectangular grid: the constant-Q contours already supply the
            # reference structure, and a major grid on top of them plus log
            # minor-tick labels reads as three competing grids per panel.
            ax.grid(False)
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            # D_F spans up to nine decades; every-decade labels collide at print width.
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4 if compact else 6))
            if np.log10(ylim[1] / ylim[0]) < 1.5:
                # Maze and Coin G spans under two decades, which leaves a single
                # labelled tick at 10^0; label 1-2-5 steps so the axis is readable.
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            else:
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4 if compact else 6))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(env, fontsize=fs["title"], fontweight="bold",
                             pad=3 if compact else 8)
            if r == last:
                # Short symbol labels so the axis names fit the panels at print width.
                ax.set_xlabel(r"$D_F$",
                              fontsize=fs["label"], labelpad=1 if compact else 4)
            if c == 0:
                ax.set_ylabel(f"{tap_lab}\n" + r"$G$",
                              fontsize=fs["label"], labelpad=1 if compact else 4)

    def h(marker, color, label, **kw):
        return plt.Line2D([], [], marker=marker, ls="", color=color, markersize=fs["lms"],
                          markeredgecolor="black", markeredgewidth=fs["lw"], label=label, **kw)
    handles = [h("o", BURGUNDY, "GM"), h("o", TEAL, "Non-GM"), h("o", MUTED, "init"),
               h("^", "white", "shape swap"), h("s", "white", "colour swap"),
               plt.Line2D([], [], ls=":", color="#BBBBBB", lw=1.4, label="constant $Q$")]
    fig.tight_layout(pad=0.3 if compact else 1.08, h_pad=0.6 if compact else None,
                     w_pad=0.4 if compact else None)
    fig.legend(handles=handles, fontsize=fs["leg"], loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.95,
               handletextpad=0.2 if compact else 0.8, columnspacing=0.9 if compact else 2.0,
               borderpad=0.3 if compact else 0.4)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{stem}.{ext}", dpi=300 if compact else 200, bbox_inches="tight")
    plt.close(fig)
    note(f"wrote {stem}.{{pdf,png}}  (Attribute Maze backgrounds: {', '.join(ras_bgs)})")


# ------------------------------------------- figure 2: signed raincloud
def draw_raincloud(ax, groups: list[tuple[str, str, dict]]) -> None:
    """groups: (x-label, cohort, {states, seed_medians})."""
    for i, (_lab, cohort, data) in enumerate(groups):
        color = COHORT_COLOR[cohort]
        states = np.asarray(data["states"], dtype=float)
        if states.size:
            # density, drawn as a half-violin above the jittered points
            try:
                from scipy.stats import gaussian_kde
                if np.ptp(states) > 0:
                    kde = gaussian_kde(states)
                    ys = np.linspace(states.min(), states.max(), 200)
                    dens = kde(ys)
                    dens = 0.32 * dens / dens.max()
                    ax.fill_betweenx(ys, i + 0.06, i + 0.06 + dens,
                                     color=color, alpha=0.32, linewidth=0, zorder=2)
            except Exception:
                pass
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-0.13, -0.02, size=states.size)
            ax.scatter(i + jitter, states, s=3.5, color=color, alpha=0.30,
                       linewidth=0, zorder=1)
        for m in data["seed_medians"]:
            ax.scatter([i], [m], marker="D", s=95, color=color,
                       edgecolor="black", linewidth=0.9, zorder=4)
    ax.axhline(0.0, color="#777777", lw=1.2, ls="--", zorder=0)


def fig_raincloud() -> None:
    # Printed at \linewidth (5.5 in); 13 in keeps 15 pt labels near 6 pt on the page.
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.8))

    for r, (tap, tap_lab) in enumerate(TAPS):
        maze_d = maze_signed(tap)
        coin_d = coin_signed(tap)
        ras_shape = ras_signed(tap, "shape")
        ras_colour = ras_signed(tap, "colour")

        panels = [
            ("Maze", [("GM", "GM", maze_d["GM"]),
                      ("Non-GM", "Non-GM", maze_d["Non-GM"]),
                      ("init", "init", maze_d["init"])],
             ["GM", "Non-GM", "init"]),
            ("Coin", [("GM", "GM", coin_d["GM"]),
                      ("Non-GM", "Non-GM", coin_d["Non-GM"]),
                      ("init", "init", coin_d["init"])],
             ["GM", "Non-GM", "init"]),
            ("Attribute Maze", [("GM shape", "GM", ras_shape["GM"]),
                                ("Non-GM shape", "Non-GM", ras_shape["Non-GM"]),
                                ("init shape", "init", ras_shape["init"]),
                                ("GM colour", "GM", ras_colour["GM"]),
                                ("Non-GM colour", "Non-GM", ras_colour["Non-GM"]),
                                ("init colour", "init", ras_colour["init"])],
             ["GM\nshape", "Non-GM\nshape", "init\nshape",
              "GM\ncolour", "Non-GM\ncolour", "init\ncolour"]),
        ]

        for c, (env, groups, ticklabels) in enumerate(panels):
            ax = axes[r, c]
            draw_raincloud(ax, groups)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(ticklabels,
                               fontsize=TICK_FS - (3 if len(groups) > 4 else 1))
            ax.tick_params(axis="y", labelsize=TICK_FS)
            ax.grid(axis="y", alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(env, fontsize=SUB_FS + 2, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(f"{tap_lab}\nsigned target-selection effect",
                              fontsize=LABEL_FS)

    handles = [
        plt.Line2D([], [], marker="D", ls="", color=BURGUNDY, markersize=10,
                   markeredgecolor="black", label="GM (seed median)"),
        plt.Line2D([], [], marker="D", ls="", color=TEAL, markersize=10,
                   markeredgecolor="black", label="Non-GM (seed median)"),
        plt.Line2D([], [], marker="D", ls="", color=MUTED, markersize=10,
                   markeredgecolor="black", label="init (seed median)"),
        plt.Line2D([], [], marker="o", ls="", color="#999999", markersize=8,
                   label="evaluation state"),
        plt.Line2D([], [], ls="--", color="#777777", lw=1.4, label="no effect"),
    ]
    fig.tight_layout()
    fig.legend(handles=handles, fontsize=LEG_FS, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.95)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_signed_target_selection_raincloud.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    note("wrote fig_signed_target_selection_raincloud.{pdf,png}")


def main() -> None:
    print("=== provenance ===", flush=True)
    fig_decomposition(("black", "texture"), "fig_fisher_g_vs_df_appendix", (11.5, 6.2))
    fig_decomposition(("black",), "fig_main_fisher_g_vs_df", (5.5, 1.85), compact=True,
                      taps=[TAPS[1]], summary=True)
    fig_raincloud()
    out = FIGS / "fig_final_fisher_cross_env_provenance.txt"
    out.write_text("\n".join(provenance) + "\n")
    print(f"\nprovenance written to {out}")


if __name__ == "__main__":
    main()
