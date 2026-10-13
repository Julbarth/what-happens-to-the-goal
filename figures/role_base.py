"""Goal-role choice planes for Maze/Coin and S_fav/S_comp planes for Attribute Maze; shared style for the role figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats

from palette import PAL  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

MAZE_JSON = DATA_ROOT / "probes/maze_detour_definition_check_20260925/result_summary.json"
COIN_JSON = DATA_ROOT / "probes/coin_choice_mass_canon_20260924/coin_choice_mass_canon.json"
AM_JSON = DATA_ROOT / "probes/canon_cue_20260924/ras_cue_competition_canon_swap_{bg}.json"
COIN_SEEDS = {"GM": ["gm_6033", "gm_6037", "gm_6051"], "Non-GM": ["nongm_7001", "nongm_7002", "nongm_7003"]}
# Same rosters as the appendix figure and Table 10.
AM_SEEDS = {"black": {"GM": ["gm_6313", "gm_6315", "gm_6317"], "Non-GM": ["deco_4371", "deco_4372", "deco_4373"]},
            "texture": {"GM": ["gm_6301", "gm_6302", "gm_6303"], "Non-GM": ["deco_4361", "deco_4362", "deco_4363"]}}
COH = ["GM", "Non-GM"]
COL = {"GM": PAL["gm"], "Non-GM": PAL["nongm"]}
COND = {"maze_all": ("o", "Maze, all conflict"), "maze_detour": ("s", "Maze, detour"), "coin": ("^", "Coin")}
FS = 7.0
W_IN = 5.5
AM_XLIM = (-1.5, 9.5)
AM_YLIM = (-9.5, 1.5)


def ci_half(v) -> float:
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def load_choice(maze_json: Path, coin_json: Path) -> dict:
    """Per-seed (goal choice, proxy choice) for each condition and cohort."""
    s = json.loads(maze_json.read_text())["summary"]
    per = json.loads(coin_json.read_text())["per_ckpt"]
    key = {"GM": "GM", "Non-GM": "NonGM"}
    out = {}
    for c in COH:
        out[("maze_all", c)] = (s["C_text_all"][key[c]]["choice_s_set"]["vals"],
                                s["C_text_all"][key[c]]["choice_p_set"]["vals"])
        out[("maze_detour", c)] = (s["D_text"][key[c]]["cat_cheese_only"]["vals"],
                                   s["D_text"][key[c]]["cat_corner"]["vals"])
        out[("coin", c)] = ([per[l]["choice_star"] for l in COIN_SEEDS[c]],
                            [per[l]["choice_p"] for l in COIN_SEEDS[c]])
    return {k: (np.asarray(x, float), np.asarray(y, float)) for k, (x, y) in out.items()}


def load_am(am_json: Path, bg: str) -> dict:
    """Per-state and per-seed (S_fav, S_comp); seed means average the two axis signs first, as in Table 10."""
    res = json.loads(am_json.read_text())["results"]
    out = {}
    for edit in ("shape", "colour"):
        for c in COH:
            states, seeds = [], []
            for lab in AM_SEEDS[bg][c]:
                rr = [r for r in res[lab] if r["edit"] == edit]
                states += [(r["S_fav"], r["S_comp"]) for r in rr]
                seeds.append([np.mean([np.mean([r[m] for r in rr if r["axis_sign"] == sg]) for sg in (1, -1)])
                              for m in ("S_fav", "S_comp")])
            out[(edit, c)] = (np.asarray(states), np.asarray(seeds))
    return out


def mean_ci(ax, x, y, color, marker, ms, z=5):
    ax.errorbar(x.mean(), y.mean(), xerr=ci_half(x), yerr=ci_half(y), fmt=marker, ms=ms, color=color,
                mec="white", mew=0.5, elinewidth=0.7, capsize=0, zorder=z)


def choice_panel(ax, data: dict):
    ax.plot([0, 1], [1, 0], color=PAL["mute"], lw=0.6, ls=(0, (3, 2)), zorder=1)
    for (cond, c), (x, y) in data.items():
        mk = COND[cond][0]
        ax.scatter(x, y, s=9, marker=mk, color=COL[c], alpha=0.35, lw=0, zorder=3)
        # GM all-conflict and detour means nearly coincide: a smaller square on top of a larger circle keeps both visible.
        mean_ci(ax, x, y, COL[c], mk, {"maze_all": 5.8, "maze_detour": 3.8}.get(cond, 5.0),
                z=6 if cond == "maze_detour" else 5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel(r"goal choice, $\mathrm{Choice}_\star$")
    ax.set_ylabel(r"proxy choice, $\mathrm{Choice}_p$")
    ax.set_aspect("equal")


def am_panel(ax, am: dict, edit: str, rng, ylabel: bool):
    ax.add_patch(Rectangle((0, AM_YLIM[0]), AM_XLIM[1], -AM_YLIM[0], color=PAL["mute"], alpha=0.08, lw=0, zorder=0))
    ax.axhline(0, color=PAL["mute"], lw=0.5, zorder=1)
    ax.axvline(0, color=PAL["mute"], lw=0.5, zorder=1)
    for c in COH:
        states, seeds = am[(edit, c)]
        ax.scatter(states[:, 0], states[:, 1], s=1.0, color=COL[c], alpha=0.12, lw=0, zorder=2, rasterized=True)
        # Seed diamonds sit under a smaller mean marker so a seed that lands on the mean still shows its corners.
        ax.scatter(seeds[:, 0], seeds[:, 1], s=16, marker="D", color=COL[c], edgecolor="white", lw=0.3, zorder=4)
        mean_ci(ax, seeds[:, 0], seeds[:, 1], COL[c], "o", 3.6)
    ax.set_xlim(*AM_XLIM)
    ax.set_ylim(*AM_YLIM)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_yticks([0, -3, -6, -9])
    ax.set_xlabel(r"$S_{\mathrm{fav}}$")
    if ylabel:
        ax.set_ylabel(r"$S_{\mathrm{comp}}$", labelpad=1)
    ax.set_aspect("equal")


def handles(with_choice: bool) -> list:
    g = dict(color="none", markerfacecolor=PAL["mute"], markeredgecolor="none")
    h = [Patch(color=PAL["gm"], label="GM"), Patch(color=PAL["nongm"], label="Non-GM")]
    if with_choice:
        h += [Line2D([0], [0], marker=COND[k][0], ms=4.5, label=COND[k][1], **g) for k in COND]
        h.append(Line2D([0], [0], color=PAL["mute"], lw=0.6, ls=(0, (3, 2)), label=r"$x+y=1$"))
    h += [Line2D([0], [0], marker="o", ms=1.6, alpha=0.5, label="state", **g),
          Line2D([0], [0], marker="D", ms=3.0, label="seed", **g),
          Line2D([0], [0], marker="o", ms=4.0, color=PAL["mute"], lw=0.7, mec="white", mew=0.5,
                 label="mean ± 95% CI"),
          Patch(color=PAL["mute"], alpha=0.16, lw=0, label="push–pull")]
    return h


def row_major(h: list, ncol: int) -> list:
    """Reorder so matplotlib's column-major legend fill reads left to right, row by row."""
    nrow = int(np.ceil(len(h) / ncol))
    return [h[r * ncol + c] for c in range(ncol) for r in range(nrow) if r * ncol + c < len(h)]


def style():
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "legend.fontsize": FS, "axes.titlesize": FS, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": PAL["ink"],
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5,
                         "ytick.major.pad": 1.5, "axes.labelpad": 2, "text.color": PAL["ink"],
                         "axes.labelcolor": PAL["ink"], "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
                         "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans", "font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))


def fig_a(choice: dict, am: dict, bg: str, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, 2.35))
    rng = np.random.default_rng(0)
    choice_panel(axes[0], choice)
    am_panel(axes[1], am, "shape", rng, ylabel=True)
    am_panel(axes[2], am, "colour", rng, ylabel=False)
    axes[0].set_title("Maze & Coin")
    axes[1].set_title("Attribute Maze, shape swap")
    axes[2].set_title("Attribute Maze, colour swap")
    fig.legend(handles=row_major(handles(True), 5), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.4, columnspacing=1.0, handletextpad=0.4,
               labelspacing=0.3)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.25, wspace=0.42)
    save(fig, out)


def fig_c(am: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 2.35))
    rng = np.random.default_rng(0)
    am_panel(axes[0], am, "shape", rng, ylabel=True)
    am_panel(axes[1], am, "colour", rng, ylabel=False)
    axes[0].set_title("Attribute Maze, shape swap")
    axes[1].set_title("Attribute Maze, colour swap")
    fig.legend(handles=handles(False), loc="center left", frameon=False, bbox_to_anchor=(0.76, 0.52),
               handlelength=1.4, handletextpad=0.5, labelspacing=0.6)
    fig.subplots_adjust(left=0.07, right=0.75, top=0.92, bottom=0.14, wspace=0.28)
    save(fig, out)


def write_table(choice: dict, out: Path):
    """Choice table: mean ± SD across seeds (paper table convention)."""
    rows = [("maze_all", r"\textsc{Maze}, all conflict ($n=1{,}500$)"),
            ("maze_detour", r"\textsc{Maze}, detour ($n=168$)"),
            ("coin", r"\textsc{Coin}, conflict ($n=600$)")]
    fmt = lambda v: f"${v.mean():.3f}\\pm{v.std(ddof=1):.2f}$"
    body = "\n".join(f"{name}\n& " + "\n& ".join(fmt(v) for c in COH for v in choice[(k, c)]) + r" \\"
                     for k, name in rows)
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Which target the greedy action serves on conflict states. Fraction of states where the greedy action makes goal progress ($\operatorname{{Choice}}_\star$, cheese or coin) or proxy progress ($\operatorname{{Choice}}_p$, corner or rightward), mean $\pm$ SD across three seeds. On \textsc{{Maze}} detour states every corner-progress action points left or down.}}
\label{{tab:main-sec3-3-choice}}
\small
\begin{{tabular}}{{lcccc}}
\toprule
& \multicolumn{{2}}{{c}}{{GM}} & \multicolumn{{2}}{{c}}{{Non-GM}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
States & $\operatorname{{Choice}}_\star$ & $\operatorname{{Choice}}_p$ & $\operatorname{{Choice}}_\star$ & $\operatorname{{Choice}}_p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maze-json", type=Path, default=MAZE_JSON)
    ap.add_argument("--coin-json", type=Path, default=COIN_JSON)
    ap.add_argument("--am-json", default=str(AM_JSON), help="path template with {bg}")
    ap.add_argument("--am-background", default="black", choices=["black", "texture"])
    ap.add_argument("--out-a", type=Path, default=OUT_DIR / "fig_main_sec3_3_A")
    ap.add_argument("--out-c", type=Path, default=OUT_DIR / "fig_main_sec3_3_C")
    ap.add_argument("--out-table", type=Path, default=OUT_DIR / "tab_main_sec3_3.tex")
    ap.add_argument("--out-values", type=Path, default=TABLES / "role_base_values.json")
    args = ap.parse_args()

    style()
    choice = load_choice(args.maze_json, args.coin_json)
    am = load_am(Path(args.am_json.format(bg=args.am_background)), args.am_background)
    fig_a(choice, am, args.am_background, args.out_a)
    fig_c(am, args.out_c)
    write_table(choice, args.out_table)

    summ = lambda v: {"per_seed": np.round(v, 4).tolist(), "mean": round(float(v.mean()), 4),
                      "sd": round(float(v.std(ddof=1)), 4), "ci95_half": round(ci_half(v), 4)}
    vals = {"choice": {f"{k}|{c}": {"goal": summ(x), "proxy": summ(y)} for (k, c), (x, y) in choice.items()},
            f"am_{args.am_background}": {f"{e}|{c}": {"S_fav": summ(sd[:, 0]), "S_comp": summ(sd[:, 1]),
                                                     "frac_states_pushpull": round(float(np.mean(
                                                         (st[:, 0] > 0) & (st[:, 1] < 0))), 4)}
                                         for (e, c), (st, sd) in am.items()}}
    args.out_values.write_text(json.dumps(vals, indent=1))
    print("wrote", args.out_values)


if __name__ == "__main__":
    main()
