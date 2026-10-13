"""Appendix robot probe figure and table: goal and proxy decodability across depth in the five robot tasks."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import probes_styled as v4  # noqa: E402
from probes_base import band  # noqa: E402
from probes_panels import COH, CTRL_LS, INIT_LS, PAL, fs  # noqa: E402

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES

GR = ROBOTICS_ROOT / "results"
AUDIT = GR / "analysis/probe_audit"
ISAAC = GR / "isaac_analysis/ghost_check/fixed/probes_per_seed.csv"

ARM_SITES = ["conv1", "conv2", "h_pre", "h_fc"]
ARM_X = [0, 1, 3, 4]
ARM_TICKS = [r"$H^{(1)}$", r"$H^{(2)}$", r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$"]
ARM_POOLED = ["conv1_gap", "conv2_gap", "conv3_gap"]
# Isaac conv maps are average-pooled to 4x4 by probes_isaac.py (POOL = 4), so they are neither the unpooled arm maps
# nor the globally pooled Procgen vectors; the bar marks the pooling.
ISAAC_SITES = ["conv1", "conv2", "conv3", "h_pre", "h_fc"]
ISAAC_X = [0, 1, 2, 3, 4]
ISAAC_TICKS = [r"$\bar H^{(1)}$", r"$\bar H^{(2)}$", r"$\bar H^{(3)}$", r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$"]
ARM_COH = {"gm": "GM", "control": "Non-GM", "init": "init"}
ARM_CTRL = "input_px32"
# The probe CSVs cover every capable AttributeTable control seed (2-5), but the qualified Non-GM cohort is seeds 3-5
# (tab:robot-qualification); unlisted (env, cohort) pairs keep every seed in the CSV.
COHORT_SEEDS = {("AT", "control"): {3, 4, 5}}

# (task key, display name, role, concept label, targets averaged per seed, metric)
CONCEPTS = [
    ("PTM", "PushToMarker", "goal", "marker position", ("marker_x", "marker_y"), "R²"),
    ("PTM", "PushToMarker", "goal", "marker bearing from gripper", ("bearing_x", "bearing_y"), "R²"),
    ("LR", "LocationReach", "goal", "ball position", ("marker_x", "marker_y"), "R²"),
    ("LR", "LocationReach", "goal", "ball bearing from gripper", ("bearing_x", "bearing_y"), "R²"),
    ("AT", "AttributeTable", "goal", "cube position", ("cube_x", "cube_y"), "R²"),
    ("AT", "AttributeTable", "proxy", "red-object position", ("red_x", "red_y"), "R²"),
    ("SZ", "SortZones", "goal", "crossed-pad position", ("cross_x", "cross_y"), "R²"),
    ("SZ", "SortZones", "goal", "crossed pad left", ("cross_left",), "AUC"),
    ("SZ", "SortZones", "proxy", "green-pad position", ("green_x", "green_y"), "R²"),
    ("WTF", r"\quadname", "goal", "flag bearing", ("flag_bearing",), "R²"),
    ("WTF", r"\quadname", "goal", "flag visible", ("flag_visible",), "AUC"),
    ("WTF", r"\quadname", "proxy", "tower bearing", ("tower_bearing",), "R²"),
    ("WTF", r"\quadname", "proxy", "tower visible", ("tower_visible",), "AUC"),
]
FIG_PROXY = {"AT": "red-object position", "SZ": "green-pad position", "WTF": "tower bearing"}
# Panel sub-labels; arm panels are xy position, Isaac panels camera bearing (caption says so).
SHORT = {"marker position": "marker", "ball position": "ball", "cube position": "cube", "red-object position": "red object",
         "crossed-pad position": "crossed pad", "green-pad position": "green pad", "flag bearing": "flag bearing",
         "tower bearing": "tower bearing"}
FIG_ORDER = [("PTM", "marker position"), ("LR", "ball position"), ("AT", "cube position"), ("AT", "red-object position"),
             ("SZ", "crossed-pad position"), ("SZ", "green-pad position"), ("WTF", "flag bearing"),
             ("WTF", "tower bearing")]
TITLES = {"PTM": "PushToMarker", "LR": "LocationReach", "AT": "AttributeTable", "SZ": "SortZones", "WTF": "WalkToFlag"}


def load_arm(env: str, targets: tuple[str, ...], probe: str) -> dict:
    """{cohort: {site: per-seed array}} (targets averaged within seed) and {site: pixel/state control}."""
    vals: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ctrl: dict = defaultdict(list)
    for r in csv.DictReader(open(AUDIT / f"{env}_per_seed.csv")):
        if r["subset"] != "all" or r["probe"] != probe or r["target"] not in targets:
            continue
        if r["cohort"] == "input":
            ctrl[r["site"]].append(float(r["value"]))
        elif r["cohort"] in ARM_COH:
            keep = COHORT_SEEDS.get((env, r["cohort"]))
            if keep is not None and int(r["seed"]) not in keep:
                continue
            vals[ARM_COH[r["cohort"]]][r["site"]][int(r["seed"])].append(float(r["value"]))
    out = {}
    for c, d in vals.items():
        out[c] = {}
        for s, by_seed in d.items():
            assert all(len(v) == len(targets) for v in by_seed.values()), (env, c, s)
            out[c][s] = np.array([np.mean(v) for _, v in sorted(by_seed.items())])
    for s, v in ctrl.items():
        assert len(v) == len(targets), (env, s, v)
    return {"cohorts": out, "ctrl": {s: float(np.mean(v)) for s, v in ctrl.items()}}


def load_isaac(targets: tuple[str, ...]) -> dict:
    vals: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ctrl: dict = defaultdict(list)
    for r in csv.DictReader(open(ISAAC)):
        if r["target"] not in targets:
            continue
        if r["cohort"] == "pixels":
            ctrl["input_px16"].append(float(r["value"]))
        elif r["cohort"] in ARM_COH:
            vals[ARM_COH[r["cohort"]]][r["site"]][int(r["seed"])].append(float(r["value"]))
    out = {c: {s: np.array([np.mean(v) for _, v in sorted(b.items())]) for s, b in d.items()} for c, d in vals.items()}
    return {"cohorts": out, "ctrl": {s: float(np.mean(v)) for s, v in ctrl.items()}}


def load(env: str, targets: tuple[str, ...], probe: str) -> dict:
    return load_isaac(targets) if env == "WTF" else load_arm(env, targets, probe)


def panel(env: str, concept: str, probe: str) -> dict:
    tg, metric = next((c[4], c[5]) for c in CONCEPTS if c[0] == env and c[3] == concept)
    d = load(env, tg, probe)
    sites, x, ticks = (ISAAC_SITES, ISAAC_X, ISAAC_TICKS) if env == "WTF" else (ARM_SITES, ARM_X, ARM_TICKS)
    ctrl = d["ctrl"]["input_px16" if env == "WTF" else ARM_CTRL]
    return dict(env=TITLES[env], sub=concept, metric=metric, x=x, ticks=ticks, ctrl=ctrl,
                s={c: [band(d["cohorts"][c][k]) for k in sites] for c in ("GM", "Non-GM", "init")},
                n={c: len(d["cohorts"][c][sites[0]]) for c in ("GM", "Non-GM", "init")})


def draw(ax, P: dict, floor: float) -> None:
    lim = (-np.inf, 1.0)
    ax.axhline(0.0, ls="-", color="#CFCFCF", lw=0.7, zorder=0.5)
    v4.draw_series(ax, P["x"], P["s"]["init"], PAL["init"], INIT_LS, lim, init=True, band_alpha=0.14)
    for c, col in COH.items():
        v4.draw_series(ax, P["x"], P["s"][c], col, "-", lim)
    ax.axhline(P["ctrl"], ls=CTRL_LS, color=PAL["null"], lw=0.9, zorder=2.5)
    # No plotted mean is negative (lowest 0.063, LocationReach GM h_fc), so the floor sits just below 0 and only
    # the lower edge of a few 95% bands is clipped.
    ax.set_ylim(floor, v4.R2_LIM[1])
    ax.set_yticks(v4.R2_MAJOR)
    ax.yaxis.set_major_formatter(v4.FuncFormatter(v4.short_num))
    ax.set_ylabel("R²", fontsize=v4.FS_LAB, labelpad=1)
    ax.set_xticks(P["x"], P["ticks"])
    ax.set_xlim(-0.35, 4.35)
    ax.tick_params(labelsize=v4.FS_TICK, length=2, width=fs.LW_AX, pad=1.2)
    ax.tick_params(axis="x", pad=0.8)


def label(ax, P: dict, letter: str) -> None:
    ax.annotate(P["sub"], xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 2.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=v4.FS_SUB, color=PAL["ink"])
    ax.annotate(f"$\\mathbf{{({letter})}}$ {P['env']}", xy=(0.5, 1), xycoords="axes fraction",
                xytext=(0, 2.5 + v4.FS_SUB * 1.25), textcoords="offset points", ha="center", va="bottom",
                fontsize=v4.FS_TITLE, color=PAL["ink"])


def render(panels: list[dict], out_stem: Path, h_in: float, floor: float) -> None:
    """2 x 4 grid, goal panels before the proxy panel of the same task; legend in one row underneath."""
    W = fs.W_IN
    fig, axes = plt.subplots(2, 4, figsize=(W, h_in))
    k = v4.FS_LAB / v4.BASE_FS
    fig.subplots_adjust(left=0.27 * k / W, right=1 - 0.03 / W, top=1 - 0.27 * k / h_in, bottom=0.40 * k / h_in,
                        wspace=0.30 * k, hspace=0.55 * k)
    for i, (ax, P) in enumerate(zip(axes.flat, panels)):
        draw(ax, P, floor)
        label(ax, P, "abcdefgh"[i])
    g = PAL["mute"]
    handles = [Line2D([], [], color=COH["GM"], lw=fs.LW_DATA, marker="o", ms=v4.MS, mec="white", mew=0.35),
               Line2D([], [], color=COH["Non-GM"], lw=fs.LW_DATA, marker="o", ms=v4.MS, mec="white", mew=0.35),
               Line2D([], [], color=PAL["init"], lw=v4.LW_INIT, ls=INIT_LS, marker="o", ms=v4.MS_INIT, mfc="white",
                      mec=PAL["init"], mew=0.7),
               Line2D([], [], color=PAL["null"], lw=0.9, ls=CTRL_LS),
               (Patch(color=g, alpha=0.3, lw=0), Line2D([], [], color=g, lw=fs.LW_DATA, marker="o", ms=v4.MS,
                                                        mec="white", mew=0.35))]
    labels = ["GM", "Non-GM", "random init", "pixels", fs.CI_LABEL]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02 / h_in), ncol=len(handles),
               frameon=False, fontsize=v4.FS_LEG, handlelength=2.0, handleheight=0.6, handletextpad=0.35,
               columnspacing=1.1, borderaxespad=0, borderpad=0)
    v4.check_layout(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out_stem.with_suffix(".{png,pdf}"))


def table_rows(probe: str) -> list[dict]:
    """Mean and SD across seeds at h_pre and h_fc, the pixel control, and the pooled-conv range for every concept."""
    rows = []
    for env, name, role, concept, tg, metric in CONCEPTS:
        d = load(env, tg, probe)
        r = dict(env=env, task=name, role=role, concept=concept, metric=metric, targets=list(tg))
        for site in ("h_pre", "h_fc", "conv1", "conv2", "conv3", *ARM_POOLED):
            for c in ("GM", "Non-GM", "init"):
                v = d["cohorts"][c].get(site)
                if v is not None:
                    r[f"{c}@{site}"] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=len(v),
                                            ci95=band(v)[1], seeds=[float(x) for x in v])
        r["pixels"] = d["ctrl"].get("input_px16" if env == "WTF" else ARM_CTRL)
        r["pixels_px16"] = d["ctrl"].get("input_px16")
        r["arm_state"] = d["ctrl"].get("input_state")
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", default="ridge_sel", help="probe_audit probe column (ridge_sel = paper default)")
    ap.add_argument("--font-size", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=3.3)
    ap.add_argument("--r2-floor", type=float, default=-0.05)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "app_probes")
    ap.add_argument("--out-name", default="fig_app_robot_probes")
    ap.add_argument("--values-out", type=Path, default=TABLES / "plotted_values_app_robot.json")
    ap.add_argument("--table-out", type=Path, default=TABLES / "table_values_app_robot.json")
    args = ap.parse_args()
    fs.style()
    v4.set_font_size(args.font_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for e, c in FIG_ORDER:
        P = panel(e, c, args.probe)
        P["role"] = "proxy" if c in FIG_PROXY.values() else "goal"
        P["sub"] = f"{P['role']}: {SHORT[c]}"
        panels.append(P)
    render(panels, args.out_dir / args.out_name, args.height, args.r2_floor)
    rec = {f"{P['env']} | {P['sub']}": {"metric": P["metric"], "sites": P["ticks"], "pixel_control": P["ctrl"],
                                             "n": P["n"],
                                             **{c: [[round(m, 4), round(h, 4)] for m, h in P["s"][c]]
                                                for c in ("GM", "Non-GM", "init")}}
           for P in panels}
    args.values_out.write_text(json.dumps({"note": "per site [mean, 95% t half-width] over seeds",
                                           "sources": {"arm": str(AUDIT / "<ENV>_per_seed.csv"), "isaac": str(ISAAC)},
                                           "probe": args.probe, "panels": rec}, indent=1))
    args.table_out.write_text(json.dumps({"probe": args.probe, "rows": table_rows(args.probe)}, indent=1))
    print("wrote", args.values_out, args.table_out)


if __name__ == "__main__":
    main()
