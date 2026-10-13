"""Paper figures and tables for the steering section: local steering across envs, and Attribute-Maze shared-axis compression."""
from __future__ import annotations

import os
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAL = {"gm": "#7A1520", "gm_soft": "#C45C58", "nongm": "#3D8B8B", "nongm_soft": "#5FA8A0",
       "init": "#8E8E8E", "null": "#5A5A5A", "ink": "#2A2A2A"}
ARMS = ("corrective", "anticorrective", "random")
# Native cohorts only: each Attribute-Maze policy is rendered on its own training background.
AM_COHORTS = {("GM", "black"): "GM", ("deco", "black"): "Non-GM", ("GM", "texture"): "GM", ("deco", "texture"): "Non-GM"}
MAZE_AGENTS = {"GM": ["gm_1080", "gm_1081", "gm_1082"], "Non-GM": ["nongm_1090", "nongm_1091", "nongm_1092"]}
COIN_AGENTS = {"GM": ["gm_6033", "gm_6037", "gm_6051"], "Non-GM": ["nongm_7001", "nongm_7002", "nongm_7003"]}


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def load_am(canon: Path, population: str) -> dict:
    """{(bg, cohort): {arm: {beta: [per-seed reach]}}} with baseline under beta 0."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(canon / "local" / "local_per_seed.csv")):
        key = (r["cohort"], r["background"])
        if (r["tap"] != "h_fc" or r["population"] != population or r["sign"] != "all"
                or r["d_stratum"] != "all" or key not in AM_COHORTS):
            continue
        out[(r["background"], AM_COHORTS[key])][r["arm"]][float(r["beta"])].append(float(r["reach"]))
    return out


def _rows_to_reach(rows: list[dict], names: dict, success: str) -> dict:
    """Per-agent reach per (arm, beta). Random pools its directions within an agent."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        b = float(row["beta"])
        acc["baseline"][b].append(row[names["baseline"]] == success)
        acc["corrective"][b].append(row[names["corrective"]] == success)
        acc["anticorrective"][b].append(row[names["anticorrective"]] == success)
        acc["random"][b].extend(o == success for o in row[names["random"]])
    return {arm: {b: float(np.mean(v)) for b, v in d.items()} for arm, d in acc.items()}


def _collect(per_agent: dict[str, dict]) -> dict:
    out: dict = defaultdict(lambda: defaultdict(list))
    for reach in per_agent.values():
        for arm, d in reach.items():
            for b, v in d.items():
                out[arm][b].append(v)
    base = [float(np.mean(list(reach["baseline"].values()))) for reach in per_agent.values()]
    out["baseline"] = {0.0: base}
    return out


def load_maze(fp: Path) -> dict:
    res = json.load(open(fp))["results"]
    names = {"baseline": "baseline_outcome", "corrective": "corrective_outcome",
             "anticorrective": "anticorrective_outcome", "random": "random_outcomes"}
    return {c: _collect({a: _rows_to_reach(res[a]["rollout_rows"], names, "cheese") for a in agents})
            for c, agents in MAZE_AGENTS.items()}


def load_coin(fp: Path) -> dict:
    res = json.load(open(fp))["results"]
    names = {"baseline": "baseline", "corrective": "corrective", "anticorrective": "anticorrective", "random": "random"}
    out = {}
    for c, agents in COIN_AGENTS.items():
        per = {}
        for a in agents:
            rows = [row for src in res for row in res[src].get(a, {}).get("rollout_rows", [])]
            if rows:
                per[a] = _rows_to_reach(rows, names, "success")
        out[c] = _collect(per)
    return out


def _panel(ax, data: dict, cohort: str) -> None:
    hue, soft = (PAL["gm"], PAL["gm_soft"]) if cohort == "GM" else (PAL["nongm"], PAL["nongm_soft"])
    base = data["baseline"][0.0]
    ax.axhspan(np.mean(base) - np.std(base, ddof=1), np.mean(base) + np.std(base, ddof=1), color=PAL["init"], alpha=0.15, lw=0)
    ax.axhline(np.mean(base), color=PAL["init"], lw=1.2, ls=":")
    style = {"corrective": (hue, "-", "o"), "anticorrective": (soft, "--", "s"), "random": (PAL["null"], "-.", "^")}
    for arm in ARMS:
        betas = sorted(data[arm])
        m = np.array([np.mean(data[arm][b]) for b in betas])
        sd = np.array([np.std(data[arm][b], ddof=1) for b in betas])
        c, ls, mk = style[arm]
        ax.fill_between(betas, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), color=c, alpha=0.15, lw=0)
        ax.plot(betas, m, color=c, ls=ls, marker=mk, ms=4, lw=1.6,
                mfc=c if arm != "anticorrective" else "white", mec=c)
    ax.set_ylim(-0.03, 1.03)
    ax.set_facecolor("white")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9, colors=PAL["ink"])


def fig_local(am: dict, maze: dict, coin: dict | None, out: Path) -> None:
    cols = [("Attribute Maze (black)", lambda c: am[("black", c)]),
            ("Attribute Maze (texture)", lambda c: am[("texture", c)]),
            ("Maze", lambda c: maze[c])]
    if coin is not None:
        cols.append(("Coin", lambda c: coin[c]))
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 5.6), sharey=True, facecolor="white")
    for j, (title, get) in enumerate(cols):
        for i, cohort in enumerate(("GM", "Non-GM")):
            ax = axes[i, j]
            _panel(ax, get(cohort), cohort)
            if i == 0:
                ax.set_title(title, fontsize=11, color=PAL["ink"])
            if i == 1:
                ax.set_xlabel(r"dose $\beta$", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{cohort}\ngoal reach", fontsize=10)
    handles = [plt.Line2D([], [], color=PAL["ink"], ls="-", marker="o", ms=4, label="corrective"),
               plt.Line2D([], [], color=PAL["ink"], ls="--", marker="s", ms=4, mfc="white", label="anticorrective"),
               plt.Line2D([], [], color=PAL["null"], ls="-.", marker="^", ms=4, label="random"),
               plt.Line2D([], [], color=PAL["init"], ls=":", label="no push")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


COMP_ARMS = (("full_local", "full local", PAL["gm"], "-", "o"),
             ("full_local_fixed", "local, fixed at $s_0$", PAL["gm"], "--", "o"),
             ("sign_gated", "sign-gated axis", PAL["gm_soft"], "-", "D"),
             ("fisher_projected", "Fisher-projected axis", PAL["gm_soft"], "--", "D"),
             ("ungated", "unsigned axis", PAL["ink"], ":", "v"),
             ("random", "random", PAL["init"], "-.", "^"))


def load_global(canon: Path, population: str, cohort: str = "GM") -> dict:
    """{bg: {arm: {signed beta: [per-seed reach]}}} for one cohort (CSV label; Non-GM is "deco"). Anti arms map to negative doses; baseline rows repeat per beta, so one per seed is kept."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(canon / "global" / "global_per_seed.csv")):
        if r["population"] != population or r["cohort"] != cohort or r["sign"] != "all" or r["d_stratum"] != "all":
            continue
        arm, b = r["arm"], float(r["beta"])
        if arm == "baseline":
            seen = out[r["background"]]["_base_seed"]
            if r["seed"] not in seen:
                seen[r["seed"]] = [float(r["reach"])]
                out[r["background"]]["baseline"][0.0].append(float(r["reach"]))
            continue
        anti = arm.endswith("_anti")
        base_arm = arm[:-5] if anti else arm
        out[r["background"]][base_arm][-b if anti else b].append(float(r["reach"]))
    return out


def fig_compression(globs: dict, out: Path) -> None:
    """Rows: cohorts (GM, Non-GM); columns: backgrounds (black, texture)."""
    cols = {"GM": {PAL["gm"]: PAL["gm"], PAL["gm_soft"]: PAL["gm_soft"]},
            "Non-GM": {PAL["gm"]: PAL["nongm"], PAL["gm_soft"]: PAL["nongm_soft"]}}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharey=True, sharex=True, facecolor="white")
    for row, cohort in zip(axes, ("GM", "Non-GM")):
        for ax, bg in zip(row, ("black", "texture")):
            d = globs[cohort][bg]
            base = np.mean(d["baseline"][0.0])
            ax.axvspan(0.25, 1.05, color=PAL["init"], alpha=0.08, lw=0)
            ax.axvspan(-1.05, -0.25, color=PAL["init"], alpha=0.08, lw=0)
            for arm, lab, c, ls, mk in COMP_ARMS:
                c = cols[cohort].get(c, c)
                doses = sorted(d[arm])
                if arm == "random":
                    doses = sorted(b for b in d[arm] if b > 0)
                xs = [0.0] + doses if arm == "random" else sorted(doses + [0.0])
                ys = np.array([base if x == 0.0 else np.mean(d[arm][x]) for x in xs])
                sd = np.array([np.std(d["baseline"][0.0], ddof=1) if x == 0.0 else np.std(d[arm][x], ddof=1) for x in xs])
                ax.fill_between(xs, np.clip(ys - sd, 0, 1), np.clip(ys + sd, 0, 1), color=c, alpha=0.07, lw=0)
                ax.plot(xs, ys, color=c, ls=ls, marker=mk, ms=4, lw=1.5, label=lab)
            ax.axvline(0, color=PAL["ink"], lw=0.6)
            ax.set_title(f"{cohort} ({bg})", fontsize=11, color=PAL["ink"])
            ax.set_ylim(-0.03, 1.03)
            ax.set_facecolor("white")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9, colors=PAL["ink"])
        row[0].set_ylabel("goal reach", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel(r"signed dose $\beta$", fontsize=10)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _ms(v) -> str:
    return f"${np.mean(v):.3f}\\pm{np.std(v, ddof=1):.2f}$"


def summary_table(am: dict, maze: dict, coin: dict | None, glob: dict, out: Path) -> None:
    lines = ["| env | cohort | baseline | corrective β=1 | anticorrective β=1 | random β=1 |", "|---|---|---|---|---|---|"]
    envs = [("AM black", lambda c: am[("black", c)]), ("AM texture", lambda c: am[("texture", c)]),
            ("Maze", lambda c: maze[c])] + ([("Coin", lambda c: coin[c])] if coin else [])
    for name, get in envs:
        for c in ("GM", "Non-GM"):
            d = get(c)
            lines.append(f"| {name} | {c} | {_ms(d['baseline'][0.0])} | "
                         + " | ".join(_ms(d[a][1.0]) for a in ARMS) + " |")
    lines += ["", "| bg | fixed local β=.25 | sign-gated | Fisher-projected | ungated | ungated anti | full local |",
              "|---|---|---|---|---|---|---|"]
    for bg in ("black", "texture"):
        d = glob[bg]
        lines.append(f"| {bg} | " + " | ".join(_ms(d[a][b]) for a, b in (
            ("full_local_fixed", 0.25), ("sign_gated", 0.25), ("fisher_projected", 0.25),
            ("ungated", 0.25), ("ungated", -0.25), ("full_local", 0.25))) + " |")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canon", type=Path, default=DATA_ROOT / "probes/canon_steer_20260924/swap")
    ap.add_argument("--population", default="eval50", choices=["eval50", "all196"])
    ap.add_argument("--maze", type=Path, default=DATA_ROOT / "probes/maze_local_steering_hfc_20260923/maze_local_steering_hfc.json")
    ap.add_argument("--coin", type=Path, default=DATA_ROOT / "probes/coin_local_steering_canon_20260924/coin_local_steering_canon.json")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "paper_figures")
    args = ap.parse_args()
    am, maze = load_am(args.canon, args.population), load_maze(args.maze)
    coin = load_coin(args.coin) if args.coin.exists() else None
    print("Coin:", "canonical bank" if coin else "not on disk, panel omitted")
    glob = load_global(args.canon, "primary")
    globs = {"GM": glob, "Non-GM": load_global(args.canon, "primary", "deco")}
    fig_local(am, maze, coin, args.fig_dir / "fig_paper_steering_local_dose_response")
    fig_compression(globs, args.fig_dir / "fig_paper_steering_shared_axis_compression")
    summary_table(am, maze, coin, glob, args.canon / "paper_steering_summary_mean_sd.md")


if __name__ == "__main__":
    main()
