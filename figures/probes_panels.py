"""Probe figure, one panel per failure type (Maze, PushToMarker, Coin, Attribute Maze): GM, Non-GM, init, input control."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from palette import PAL  # noqa: E402
import style as fs  # noqa: E402
from role_panels_styled import FS_MIN, check_layout  # noqa: E402
from probes_base import SITES, band, load_am_seeded, load_coin, load_init_file, load_maze  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, ROBOTICS_ROOT, TABLES

# Inputs come from the released data/probes/ archive (see README).
DATA = DATA_ROOT / "probes"
ROBOT_CSV = ROBOTICS_ROOT / "results/analysis/probe_audit/PTM_per_seed.csv"

COH = {"GM": PAL["gm"], "Non-GM": PAL["nongm"]}
INIT_LS = (0, (3.5, 1.8))
CTRL_LS = (0, (1, 1.3))
DODGE = {"GM": -0.11, "init": 0.0, "Non-GM": 0.11}

PROCGEN_X = [0, 1, 2, 3, 4]
PROCGEN_TICKS = [r"$h^{(1)}$", r"$h^{(2)}$", r"$h^{(3)}$", r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$"]
# Robot Nature CNN: conv3 is flattened directly into h_pre, so there is no separate third site. Probes read the
# full flattened conv1/conv2 maps (H^(k)), not their average-pooled vectors, hence the capital H.
ROBOT_SITES = ["conv1", "conv2", "h_pre", "h_fc"]
ROBOT_X = [0, 1, 3, 4]
ROBOT_TICKS = [r"$H^{(1)}$", r"$H^{(2)}$", r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$"]
ROBOT_COH = {"gm": "GM", "control": "Non-GM", "init": "init"}
ROBOT_TARGETS = ("marker_x", "marker_y")
ROBOT_CTRL_SITE = "input_px32"

W_IN = fs.W_IN
H_IN = 1.7


def load_robot(path: Path, probe: str) -> dict:
    """PTM marker-position R² (mean over x and y per seed) -> {cohort: {site: per-seed array}} plus the pixel control."""
    vals: dict = defaultdict(lambda: defaultdict(dict))
    ctrl = []
    for r in csv.DictReader(open(path)):
        if r["subset"] != "all" or r["probe"] != probe or r["target"] not in ROBOT_TARGETS:
            continue
        if r["cohort"] == "input" and r["site"] == ROBOT_CTRL_SITE:
            ctrl.append(float(r["value"]))
        elif r["cohort"] in ROBOT_COH and r["site"] in ROBOT_SITES:
            vals[ROBOT_COH[r["cohort"]]][r["site"]].setdefault(r["seed"], []).append(float(r["value"]))
    out = {}
    for c, d in vals.items():
        out[c] = {}
        for s, by_seed in d.items():
            assert all(len(v) == len(ROBOT_TARGETS) for v in by_seed.values()), (c, s)
            out[c][s] = np.array([np.mean(v) for _, v in sorted(by_seed.items())])
    assert len(ctrl) == len(ROBOT_TARGETS), ctrl
    return {"cohorts": out, "ctrl": float(np.mean(ctrl))}


def procgen_series(p: dict) -> dict:
    """Cohort and init -> list of (mean, 95% t half-width) over the five Procgen sites."""
    s = {c: [band(p[c][k]) for k in SITES] for c in COH}
    s["init"] = [p["init"][k] for k in SITES]
    return s


def build_panels(a) -> list[dict]:
    maze = load_maze(a.data / "maze_probes_grouped_20260924/checkpoint_level.csv")["cheese_bearing"]
    maze_ref = next(float(r["auc"]) for r in csv.DictReader(open(a.data / "maze_probes_grouped_20260924/checkpoint_level.csv"))
                    if r["cohort"] == "reference" and r["site"] == "agent_xy" and r["concept"] == "cheese_bearing")
    coin_dir = a.data / "coinrun_concept_grouped_canonical_20260924"
    coin = load_coin(coin_dir / "cells.csv")["bearing"]
    coin_ref = json.load(open(coin_dir / "agent_position_control_bearing.json"))["agent_xy"]["mean"]
    am_dir = a.data / "goal_identity_loci_v2_rseed_20260925_seeded"
    am = dict(load_am_seeded(am_dir)["goal_is_line"])
    am.update(init_kind="shared", init=load_init_file(am_dir / "summary.json", "goal_is_line"))
    rob = load_robot(a.robot, a.robot_probe)
    rob_s = {c: [band(rob["cohorts"][c][k]) for k in ROBOT_SITES] for c in ("GM", "Non-GM", "init")}
    return [
        dict(env="Maze", sub="cheese bearing", metric="AUC", x=PROCGEN_X, ticks=PROCGEN_TICKS,
             s=procgen_series(maze), ctrl=maze_ref, ctrl_lab="agent position"),
        dict(env="PushToMarker", sub="marker position", metric="R²", x=ROBOT_X, ticks=ROBOT_TICKS,
             s=rob_s, ctrl=rob["ctrl"], ctrl_lab="pixels"),
        dict(env="Coin", sub="coin bearing", metric="R²", x=PROCGEN_X, ticks=PROCGEN_TICKS,
             s=procgen_series(coin), ctrl=coin_ref, ctrl_lab="agent position", hollow={4}),
        dict(env="Attribute Maze", sub="line as goal", metric="AUC", x=PROCGEN_X, ticks=PROCGEN_TICKS,
             s=procgen_series(am), ctrl=None),
    ]


def draw_series(ax, x, ms, col, ls, dx, clip, hollow=frozenset(), init=False) -> list[float]:
    """Mean line with dodged 95% t whiskers; init gets open markers so it reads as a baseline, not a cohort."""
    x = np.asarray(x, float) + dx
    m = np.array([a for a, _ in ms])
    h = np.array([b for _, b in ms])
    lo, hi = m - h, m + h
    if clip:
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
    ax.vlines(x, lo, hi, color=col, lw=fs.LW_ERR, zorder=3)
    ax.plot(x, m, ls=ls, color=col, lw=fs.LW_DATA if not init else 1.0, zorder=4)
    for i in range(len(x)):
        open_ = init or i in hollow
        ax.plot(x[i], m[i], "o", ms=fs.MS, zorder=5, mfc="white" if open_ else col, mec=col if open_ else "white",
                mew=0.9 if open_ else 0.4)
    return list(lo) + list(hi)


def draw_panel(ax, P: dict, letter: str) -> None:
    clip = P["metric"] == "AUC"
    ax.axhline(0.5 if clip else 0.0, ls="-", color="#CFCFCF", lw=0.8, zorder=1)
    edges = draw_series(ax, P["x"], P["s"]["init"], PAL["init"], INIT_LS, DODGE["init"], clip, init=True)
    for c, col in COH.items():
        edges += draw_series(ax, P["x"], P["s"][c], col, "-", DODGE[c], clip, P.get("hollow", frozenset()))
    if P["ctrl"] is not None:
        ax.axhline(P["ctrl"], ls=CTRL_LS, color=PAL["null"], lw=1.0, zorder=2)
        edges.append(P["ctrl"])
    if clip:
        # Floor drops below 0.4 only when a whisker would otherwise be cut (Attribute Maze init, n = 3 draws).
        lo = min(0.4, np.floor(min(edges) * 10) / 10)
        ax.set_ylim(lo, 1.02)
        ax.set_yticks(np.arange(lo, 1.01, 0.2))
    else:
        assert min(edges) >= -0.3, min(edges)
        ax.set_ylim(-0.3, 1.02)
        ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(P["metric"], fontsize=fs.FS, labelpad=1)
    ax.set_xticks(P["x"], P["ticks"])
    ax.set_xlim(-0.4, 4.4)
    fs.finish(ax)
    fs.panel_title(ax, P["env"], P["sub"])
    fs.panel_letter(ax, letter, sub=True, dx=-24)


def control_label(panels: list[dict]) -> str:
    """One legend entry naming each panel's control by letter, e.g. 'agent position (a, c), pixels (b)'."""
    by: dict = defaultdict(list)
    for i, P in enumerate(panels):
        if P["ctrl"] is not None:
            by[P["ctrl_lab"]].append("abcd"[i])
    return "input control: " + ", ".join(f"{k} ({', '.join(v)})" for k, v in by.items())


def render(panels: list[dict], out_stem: Path, h_in: float) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(W_IN, h_in))
    fig.subplots_adjust(left=0.34 / W_IN, right=1 - 0.04 / W_IN, top=1 - 0.33 / h_in, bottom=0.58 / h_in, wspace=0.42)
    for i, (ax, P) in enumerate(zip(axes, panels)):
        draw_panel(ax, P, "abcd"[i])
    g = PAL["mute"]
    handles = [fs.cohort_handle("GM", line=True), fs.cohort_handle("Non-GM", line=True),
               Line2D([], [], color=PAL["init"], lw=1.0, ls=INIT_LS, marker="o", ms=fs.MS, mfc="white",
                      mec=PAL["init"], mew=0.9),
               Line2D([], [], color=PAL["null"], lw=1.0, ls=CTRL_LS),
               Line2D([], [], color=g, lw=fs.LW_ERR, marker="o", ms=fs.MS, mec="white", mew=0.4)]
    labels = ["GM agents", "Non-GM agents", "random-init network", control_label(panels), fs.CI_LABEL]
    fs.legend_row(fig, handles, labels, y_in=0.03, ncol=3, columnspacing=1.4)
    check_layout(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out_stem.with_suffix(".{png,pdf}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA, help="gm-geometry data/probes root (Procgen panels)")
    ap.add_argument("--robot", type=Path, default=ROBOT_CSV, help="robotics code probe_audit PTM_per_seed.csv")
    ap.add_argument("--robot-probe", default="ridge_sel", help="probe_audit probe column (ridge_sel = paper default)")
    ap.add_argument("--height", type=float, default=H_IN)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-name", default="fig_probes_panels")
    ap.add_argument("--values-out", type=Path, default=TABLES / "probes_panels_values.json")
    args = ap.parse_args()
    fs.FS_TICK = fs.FS_LEG = fs.FS_SUB = max(fs.FS_SUB, FS_MIN)
    fs.style()
    panels = build_panels(args)
    render(panels, args.out_dir / args.out_name, args.height)
    rec = {P["env"]: {"metric": P["metric"], "sites": P["ticks"], "control": P["ctrl"],
                      **{c: [[round(m, 4), round(h, 4)] for m, h in P["s"][c]] for c in ("GM", "Non-GM", "init")}}
           for P in panels}
    args.values_out.write_text(json.dumps({"note": "per site [mean, 95% t half-width]", "panels": rec}, indent=1))
    print("wrote", args.values_out)


if __name__ == "__main__":
    main()
