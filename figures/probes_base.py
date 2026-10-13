"""Main-text probe summary figure (Sec. 3.1): absolute held-out probe score per site with init and input-level baselines."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from palette import PAL  # noqa: E402
import style as fs  # noqa: E402
from role_panels_styled import FS_MIN, check_layout  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

D = DATA_ROOT / "probes"
MAZE_CSV = D / "maze_probes_grouped_20260924/checkpoint_level.csv"
MAZE_INDEP = D / "maze_probes_grouped_independent_check_20260924/summary.json"
COIN_DIR = D / "coinrun_concept_grouped_canonical_20260924"
AM_SEEDED = D / "goal_identity_loci_v2_rseed_20260925_seeded"

# Panel order (one row). R² panels should go last: their y-axis is drawn on the right.
PANELS = ["maze:cheese_present", "maze:cheese_bearing", "am_black:goal_is_line", "coin:bearing"]

SITES = ["h1", "h2", "h3", "h_pre", "h_fc"]
SITE_LABEL = [r"$h^{(1)}$", r"$h^{(2)}$", r"$h^{(3)}$", r"$h_{\mathrm{pre}}$", r"$h_{\mathrm{fc}}$"]
SITE_MAP = {"block1_gap": "h1", "block2_gap": "h2", "block3_gap": "h3", "pre_fc": "h_pre", "fc": "h_fc",
            "H(1)": "h1", "H(2)": "h2", "H(3)": "h3", "h_pre": "h_pre", "h_fc": "h_fc"}
COH = {"GM": PAL["gm"], "Non-GM": PAL["nongm"]}


def _maze_ref(concept: str):
    return lambda rows: next(float(r["auc"]) for r in rows if r["cohort"] == "reference"
                             and r["site"] == "agent_xy" and r["concept"] == concept)


# Input-level controls: label (also the legend entry and line style key), file path, and a getter applied to the
# loaded file (JSON -> dict, CSV -> list of row dicts).
CTRL = {
    "maze_cheese_px": dict(label="cheese-cell pixels", path=MAZE_INDEP,
                           get=lambda d: d["edit_audit"]["cheese_cell_window_probe_auc_mean"]),
    "maze_agent_pos": dict(label="agent position", path=MAZE_CSV, get=_maze_ref("cheese_bearing")),
    "maze_agent_pos_corner": dict(label="agent position", path=MAZE_CSV, get=_maze_ref("corner_bearing")),
    "coin_px_presence": dict(label="raw pixels", path=COIN_DIR / "summary.json",
                             get=lambda d: d["pixel_baseline"]["all|presence"]["mean"]),
    "coin_agent_pos_bearing": dict(label="agent position", path=COIN_DIR / "agent_position_control_bearing.json",
                                   get=lambda d: d["agent_xy"]["mean"]),
}
# Controls are named in the caption, not the legend, so they share one dotted style (one control per panel).
CTRL_LS = (0, (1, 1.3))

# ctrl: key into CTRL, or None when no input-level control exists or it is off-scale (Coin raw-pixel R² = -0.552).
# hollow: sites drawn with open markers because the appendix does not interpret them.
# init_file (Attribute Maze): CSV with columns site,auc (one row per init draw per site, site names as in SITE_MAP),
#   or a seeded-run summary.json (controls.<site>.<concept>.init_draw_means). When set it replaces the per-cohort
#   init replayed on checkpoint frames with one line pooled over both cohorts' frames; band = 95% t across draws.
PANEL_SPECS = {
    "maze:cheese_present": dict(title="Maze:\ncheese presence", metric="AUC", ctrl="maze_cheese_px"),
    "maze:cheese_bearing": dict(title="Maze:\ncheese bearing", metric="AUC", ctrl="maze_agent_pos"),
    "maze:corner_bearing": dict(title="Maze:\ncorner bearing", metric="AUC", ctrl="maze_agent_pos_corner"),
    "coin:presence": dict(title="Coin:\ncoin presence", metric="AUC", ctrl="coin_px_presence"),
    # h_fc cohort ordering flips with the frame population (appendix), so it is de-emphasised.
    "coin:bearing": dict(title="Coin:\ncoin bearing", metric="R²", ctrl="coin_agent_pos_bearing", hollow=("h_fc",)),
}
for _bg in ("black", "texture"):
    for _k, _t in (("line_present", "line presence"), ("goal_is_line", "line as goal"),
                   ("yellow_present", "yellow presence"), ("goal_is_yellow", "yellow as goal")):
        PANEL_SPECS[f"am_{_bg}:{_k}"] = dict(title=f"Attribute Maze:\n{_t}", metric="AUC", ctrl=None,
                                             init_file=AM_SEEDED / ("" if _bg == "black" else _bg) / "summary.json")

W_IN = fs.W_IN
H_IN = 1.56
INIT_LS = (0, (3.5, 1.8))
# Side margins in inches: just the y tick labels and axis label (left) or the right-hand R² axis (right).
LEFT_IN, RIGHT_IN = 0.36, 0.38


def tci(n: int) -> float:
    return float(stats.t.ppf(0.975, n - 1))


def band(v: np.ndarray) -> tuple[float, float]:
    """Mean and 95% t half-width across units (rows of v are units)."""
    v = np.asarray(v, float)
    return float(v.mean()), tci(len(v)) * float(v.std(ddof=1)) / np.sqrt(len(v))


# ---------- loaders: every panel becomes {"GM": {site: [3]}, "Non-GM": ..., "init": ..., "delta": ...} ----------

def load_maze(path: Path) -> dict:
    """Init is 3 shared draws (not matched per checkpoint); delta = checkpoint AUC minus the init mean at that site."""
    raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(path)):
        if r["site"] in SITE_MAP and r["cohort"] in ("GM", "Non-GM", "init"):
            raw[r["concept"]][r["cohort"]][SITE_MAP[r["site"]]].append(float(r["auc"]))
    out = {}
    for concept, d in raw.items():
        init = {s: np.array(d["init"][s]) for s in SITES}
        p = {"init_kind": "shared", "init": {s: band(init[s]) for s in SITES}, "init_n": 3}
        for c in COH:
            p[c] = {s: np.array(d[c][s]) for s in SITES}
            p["delta_" + c] = {s: p[c][s] - init[s].mean() for s in SITES}
        out[concept] = p
    return out


def load_coin(path: Path) -> dict:
    """Init is 8 shared networks; the table keeps only their mean/SD, so the init band is rebuilt from those."""
    out: dict = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["subset"] != "all" or r["paper_site"] not in SITE_MAP or r["cohort"] not in COH:
            continue
        s, c, k = SITE_MAP[r["paper_site"]], r["cohort"], r["concept"]
        tr = np.array(ast.literal_eval(r["trained_ckpts"]))
        de = np.array(ast.literal_eval(r["delta_ckpts"]))
        assert abs(tr.mean() - float(r["trained_mean"])) < 1e-9 and np.allclose(tr - float(r["init_mean"]), de), r
        p = out[k]
        p.setdefault("init_kind", "shared")
        p.setdefault("init_n", 8)
        p.setdefault(c, {})[s] = tr
        p.setdefault("delta_" + c, {})[s] = de
        m, sd = float(r["init_mean"]), float(r["init_sd"])
        prev = p.setdefault("init", {}).get(s)
        assert prev is None or abs(prev[0] - m) < 1e-12, "coin init should be cohort-independent (shared frames)"
        p["init"][s] = (m, tci(8) * sd / np.sqrt(8))
    return dict(out)


def load_am_seeded(bg_dir: Path) -> dict:
    """Seeded per_checkpoint.csv; the per-cohort init (3 draws replayed on that cohort's frames) is overridden by init_file."""
    raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(Path(bg_dir) / "per_checkpoint.csv")):
        if r["site"] in SITE_MAP and r["cohort"] in COH:
            ra, ia, de = float(r["trained_auc"]), float(r["init_pooled_auc"]), float(r["delta_init_pooled"])
            assert abs(ra - ia - de) < 1e-6, r
            raw[r["concept"]][r["cohort"]][SITE_MAP[r["site"]]].append((ra, ia))
    out = {}
    for concept, d in raw.items():
        p = {"init_kind": "paired", "init_n": 3, "init": {}}
        for c in COH:
            arr = {s: np.array(d[c][s]) for s in SITES}
            p[c] = {s: arr[s][:, 0] for s in SITES}
            p["init"][c] = {s: band(arr[s][:, 1]) for s in SITES}
            p["delta_" + c] = {s: arr[s][:, 0] - arr[s][:, 1] for s in SITES}
        out[concept] = p
    return out


def read_file(path: Path):
    path = Path(path)
    return json.load(open(path)) if path.suffix == ".json" else list(csv.DictReader(open(path)))


def control_value(key: str) -> tuple[str, float]:
    c = CTRL[key]
    return c["label"], float(c["get"](read_file(c["path"])))


def load_init_file(path: Path, concept: str) -> dict:
    """Override init with independent draws (CSV or seeded summary.json); returns {site: (mean, 95% t half-width)}."""
    d = defaultdict(list)
    if Path(path).suffix == ".json":
        for site, cells in read_file(path)["controls"].items():
            if site in SITE_MAP:
                d[SITE_MAP[site]] = [float(v) for v in cells[concept]["init_draw_means"].values()]
        assert all(len(d[s]) >= 2 for s in SITES), {s: len(d[s]) for s in SITES}
        return {s: band(np.array(d[s])) for s in SITES}
    for r in read_file(path):
        if r["site"] in SITE_MAP:
            d[SITE_MAP[r["site"]]].append(float(r["auc"]))
    assert all(len(d[s]) >= 2 for s in SITES), {s: len(d[s]) for s in SITES}
    return {s: band(np.array(d[s])) for s in SITES}


# ---------- drawing ----------

def style_axes(ax, title: str, letter: str) -> None:
    ax.set_xticks(range(len(SITES)), SITE_LABEL)
    ax.set_xlim(-0.3, len(SITES) - 0.7)
    fs.finish(ax)
    env, sub = title.split(":\n")
    fs.panel_title(ax, env, sub)
    fs.panel_letter(ax, letter, sub=True)


def cohort_lines(ax, p: dict, clip: bool, hollow: tuple = ()) -> list[float]:
    """Cohort mean + 95% t band across the 3 checkpoints (clipped to [0, 1] for AUC); returns band edges."""
    x = np.arange(len(SITES))
    edges = []
    for c, col in COH.items():
        mh = [band(p[c][s]) for s in SITES]
        m, h = np.array([a for a, _ in mh]), np.array([b for _, b in mh])
        lo, hi = (np.clip(m - h, 0, 1), np.clip(m + h, 0, 1)) if clip else (m - h, m + h)
        ax.fill_between(x, lo, hi, color=col, alpha=fs.CI_ALPHA, lw=0, zorder=2)
        ax.plot(x, m, "-", color=col, lw=fs.LW_DATA, zorder=4)
        solid = np.array([s not in hollow for s in SITES])
        ax.plot(x[solid], m[solid], "o", color=col, ms=fs.MS, mec="white", mew=0.4, zorder=5)
        ax.plot(x[~solid], m[~solid], "o", mfc="white", mec=col, mew=0.9, ms=fs.MS, zorder=5)
        edges += list(lo) + list(hi)
    return edges


def init_lines(ax, p: dict, clip: bool) -> list[float]:
    """Grey dashed init at every site; a paired init (one per cohort frame set) is drawn as two lines, one legend entry."""
    x = np.arange(len(SITES))
    edges = []
    series = [p["init"]] if p["init_kind"] == "shared" else list(p["init"].values())
    for ini in series:
        m = np.array([ini[s][0] for s in SITES])
        h = np.array([ini[s][1] for s in SITES])
        lo, hi = (np.clip(m - h, 0, 1), np.clip(m + h, 0, 1)) if clip else (m - h, m + h)
        ax.fill_between(x, lo, hi, color=PAL["init"], alpha=0.16 / len(series), lw=0, zorder=1)
        ax.plot(x, m, ls=INIT_LS, color=PAL["init"], lw=fs.LW_DATA, zorder=3)
        edges += list(lo) + list(hi)
    return edges


def make_panel(ax, p: dict, spec: dict, ctrl: tuple[str, float] | None) -> list[float]:
    clip = spec["metric"] == "AUC"
    edges = cohort_lines(ax, p, clip, spec.get("hollow", ()))
    edges += init_lines(ax, p, clip)
    ax.axhline(0.5 if clip else 0.0, ls="-", color="#CFCFCF", lw=0.8, zorder=1.5)
    if ctrl:
        ax.axhline(ctrl[1], ls=CTRL_LS, color=PAL["null"], lw=1.0, zorder=1.5)
        edges.append(ctrl[1])
    return edges


def render(panels: list[tuple[dict, dict, tuple | None]], out_stem: Path, h_in: float = H_IN) -> dict:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(W_IN, h_in), squeeze=False)
    axes = axes[0]
    right_axis = panels[-1][1]["metric"] != panels[0][1]["metric"]
    fig.subplots_adjust(left=LEFT_IN / W_IN, right=1 - (RIGHT_IN if right_axis else 0.03) / W_IN,
                        top=1 - 0.30 / h_in, bottom=0.36 / h_in, wspace=0.20)
    plotted, by_metric = {}, defaultdict(list)
    for i, (p, spec, ctrl) in enumerate(panels):
        ax = axes[i]
        by_metric[spec["metric"]].append((ax, make_panel(ax, p, spec, ctrl)))
        style_axes(ax, spec["title"], "abcdefgh"[i])
        if right_axis and i == n - 1:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(False)
            ax.set_ylabel(spec["metric"], fontsize=fs.FS, labelpad=7, rotation=270)
        elif i == 0:
            ax.set_ylabel(spec["metric"], fontsize=fs.FS, labelpad=1)
        else:
            ax.tick_params(axis="y", labelleft=False)
        plotted[spec["title"].replace("\n", " ")] = {c: {s: band(p[c][s])[0] for s in SITES} for c in COH}
    for metric, items in by_metric.items():
        lo = min(min(e) for _, e in items)
        hi = max(max(e) for _, e in items)
        for ax, _ in items:
            if metric == "AUC":
                ax.set_ylim(max(0.0, np.floor(lo * 20) / 20), 1.02)
                ax.set_yticks(np.arange(0.4, 1.01, 0.2) if lo > 0.2 else np.arange(0.0, 1.01, 0.2))
            else:
                ax.set_ylim(np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10 + 0.02)
                ax.set_yticks(np.arange(np.ceil(lo * 5) / 5, hi + 1e-9, 0.2))
    handles = [fs.cohort_handle("GM", line=True), fs.cohort_handle("Non-GM", line=True), fs.init_handle(INIT_LS),
               fs.ci_handle(tunnel=True)]
    fs.legend_row(fig, handles, ["GM", "Non-GM", "init", fs.CI_LABEL])
    check_layout(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out_stem.with_suffix(".{png,pdf}"))
    return plotted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maze", type=Path, default=MAZE_CSV)
    ap.add_argument("--coin", type=Path, default=COIN_DIR / "cells.csv")
    ap.add_argument("--am", type=Path, default=AM_SEEDED, help="seeded-run dir (black, with texture/ inside)")
    ap.add_argument("--panels", nargs="+", default=PANELS)
    ap.add_argument("--height", type=float, default=H_IN, help="figure height in inches (width is \\textwidth)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-name", default="fig_main_probes_abs")
    ap.add_argument("--values-out", type=Path, default=TABLES / "plotted_values.json")
    args = ap.parse_args()
    fs.FS_TICK = fs.FS_LEG = fs.FS_SUB = max(fs.FS_SUB, FS_MIN)
    fs.style()
    data = {"maze": load_maze(args.maze), "coin": load_coin(args.coin), "am_black": load_am_seeded(args.am),
            "am_texture": load_am_seeded(args.am / "texture")}
    panels, record = [], {}
    for key in args.panels:
        env, concept = key.split(":")
        p, spec = dict(data[env][concept]), PANEL_SPECS[key]
        if spec.get("init_file"):
            p.update(init_kind="shared", init=load_init_file(spec["init_file"], concept))
        ctrl = control_value(spec["ctrl"]) if spec["ctrl"] else None
        panels.append((p, spec, ctrl))
        init = ({s: p["init"][s][0] for s in SITES} if p["init_kind"] == "shared"
                else {c: {s: p["init"][c][s][0] for s in SITES} for c in COH})
        record[key] = {"init_means": init, "control": ctrl, "init_file": str(spec.get("init_file"))}
    means = render(panels, args.out_dir / args.out_name, args.height)
    args.values_out.write_text(json.dumps({"cohort_means": means, "panels": record}, indent=1))
    print("wrote", args.values_out)


if __name__ == "__main__":
    main()
