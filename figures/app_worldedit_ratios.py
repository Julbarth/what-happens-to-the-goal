"""Within-environment anchored world-edit ratios GM / Non-GM for KL and SF: ratio of cohort means and per-GM-seed ratios."""
# CPU only, reads existing per-seed results. Run from the procgen/ directory:
#   python figures/app_worldedit_ratios.py --data-root <data/probes copy>
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style as fs  # noqa: E402
from style import PAL  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, ROBOTICS_ROOT, TABLES

GR = ROBOTICS_ROOT
SF_CSV = TABLES / "robot_step_follow_per_seed.csv"
MLAT = GR / "results/analysis/ptm_mlat/raw"
# AttributeTable Non-GM: the table drops seed 2 (default_rng(0).choice([2,3,4,5], 3, replace=False) = [4, 5, 3]).
SEEDS = {"AT": {"gm": [2, 3, 4], "control": [3, 4, 5], "init": [0, 1, 2]}}
ROLE_COL = {"goal": PAL["gm_goal"], "proxy": PAL["gm_proxy"]}
LOC, COMP = "Location targets", "Attribute and object competition"
INERT = ("Maze, cheese moved", "LocationReach, ball moved")

# (group, label, env key, edit key, role); Procgen env keys come from procgen_values, robot keys are (task, edit).
ROWS = [
    (LOC, "Maze, cheese moved", "maze", "cheese", "goal"),
    (LOC, "Maze, top-right opened", "maze", "corner", "proxy"),
    (LOC, "LocationReach, ball moved", "LR", "goal_move", "goal"),
    ("PushToMarker", "PushToMarker, marker moved", "PTM", "marker_move", "goal"),
    ("Coin", "Coin, coin moved", "coin", "coin", "goal"),
    (COMP, "WalkToFlag, flag moved", "WTF", "flag_swap", "goal"),
    (COMP, "WalkToFlag, tower moved", "WTF", "tower_swap", "proxy"),
    (COMP, "Attribute Maze black, shape swap", "am_black", "shape", "goal"),
    (COMP, "Attribute Maze black, colour swap", "am_black", "colour", "proxy"),
    (COMP, "Attribute Maze textured, shape swap", "am_texture", "shape", "goal"),
    (COMP, "Attribute Maze textured, colour swap", "am_texture", "colour", "proxy"),
    (COMP, "AttributeTable, shape swap", "AT", "shape_swap", "goal"),
    (COMP, "AttributeTable, colour swap", "AT", "colour_swap", "proxy"),
    (COMP, "SortZones, cross swap", "SZ", "mark_swap", "goal"),
    (COMP, "SortZones, colour swap", "SZ", "colour_swap", "proxy"),
]
# Null edits, CSV only (not every edit has one): (row env, row edit, name, metric, source edit). Value = mean(GM on the null edit) /
# mean(Non-GM on the row's edit): where the GM would sit if it reacted to the edit only as much as to one that
# should not matter.
FLOORS = [
    ("PTM", "marker_move", "cube yaw", "KL", "cube_yaw"),
    ("LR", "goal_move", "distractor move", "KL", "distractor_move"),
    ("LR", "goal_move", "distractor move", "SF", "distractor_move"),
    ("WTF", "flag_swap", "re-render", "KL", "rerender"),
    ("WTF", "flag_swap", "box move", "KL", "box_move"),
    ("WTF", "tower_swap", "re-render", "KL", "rerender"),
    ("WTF", "tower_swap", "box move", "KL", "box_move"),
]


def procgen_values(root: Path) -> dict:
    """Per-seed Procgen KL / SF behind tab:world-edit-dependence (each reproduces the table's mean ± SD).

    Maze uses the shared 128-state conflict sample, Coin the 600-state canonical bank, Attribute Maze the native
    background (subset all). No Procgen bank has an init cohort or a nuisance edit."""
    import json
    out = {}
    mz = pd.read_csv(TABLES / "maze_shared128_per_seed.csv").sort_values("ckpt")
    for _, r in mz.iterrows():
        coh = "gm" if r.cohort == "GM" else "control"
        for edit in ("cheese", "corner"):
            out.setdefault(("maze", edit, "KL", coh), []).append(r[f"kl_{edit}"])
            out.setdefault(("maze", edit, "SF", coh), []).append(r[f"sf_{edit}"])
    c = json.loads((root / "coin_worldedit_canon_20260924/coin_worldedit_canon.json").read_text())
    for k, coh in (("GM", "gm"), ("NonGM", "control")):
        for met in ("kl", "sf"):
            out[("coin", "coin", met.upper(), coh)] = c["summary"]["cohort_full_bank"][k][met]["per_seed"]
    d = pd.read_csv(root / "canon_behav_20260924/analysis/per_seed.csv")
    d = d[(d.subset == "all") & (d.native_bg == d.background) & d.edit.isin(["shape_swap", "colour_swap"])]
    for (bg, edit, coh), g in d.groupby(["background", "edit", "cohort"]):
        for met in ("kl", "sf"):
            out[(f"am_{bg}", edit.split("_")[0], met.upper(), "gm" if coh == "GM" else "control")] = \
                g.sort_values("seed")[met].tolist()
    return {k: np.asarray(v, float) for k, v in out.items()}


def robot_kl() -> pd.DataFrame:
    """Per-seed KL from the scored world-edit CSVs (WalkToFlag: ghost-free visible bank), as in the table."""
    parts = []
    for t in ("PTM", "LR", "AT", "SZ"):
        d = pd.read_csv(GR / f"results/analysis/worldedit_{t}_per_seed.csv")[["edit", "cohort", "seed", "KL"]]
        parts.append(d.assign(env=t))
    w = pd.read_csv(GR / "results/isaac_analysis/ghost_check/fixed/worldedit_per_seed.csv")
    parts.append(w[w.bank == "visible"][["edit", "cohort", "seed", "KL"]].assign(env="WTF"))
    return pd.concat(parts)


def robot_values() -> dict:
    """{(env, edit, metric, cohort): per-seed array}. Scored KL CSVs round init KL to 0, so init KL comes from the
    recomputed KL in robot_step_follow_per_seed.csv (matches the scored KL to < 0.06% for trained agents)."""
    kl, sf = robot_kl(), pd.read_csv(SF_CSV)
    sf = sf.rename(columns={"task": "env"})
    out = {}
    for (env, edit, coh), g in kl.groupby(["env", "edit", "cohort"]):
        seeds = SEEDS.get(env, {}).get(coh)
        g = g[g.seed.isin(seeds)] if seeds else g
        if coh != "init":
            out[(env, edit, "KL", coh)] = g.sort_values("seed").KL.to_numpy(float)
    for (env, edit, coh), g in sf.groupby(["env", "edit", "cohort"]):
        seeds = SEEDS.get(env, {}).get(coh)
        g = g[g.seed.isin(seeds)] if seeds else g
        out[(env, edit, "SF", coh)] = g.sort_values("seed").step_follow_mm.to_numpy(float)
        if coh == "init":
            out[(env, edit, "KL", coh)] = g.sort_values("seed").KL.to_numpy(float)
    # PushToMarker presence test: a green (non-goal) disc vs the white marker drawn at the same place m.
    m = pd.concat([pd.read_csv(f) for f in sorted(MLAT.glob("ptm_mlat_*_s*.csv"))])
    m = m[(m.bank == "PTM_worldedit") & (m["loc"] == "m")]
    for (col, coh, seed), g in m.groupby(["colour", "cohort", "seed"]):
        out.setdefault(("PTM", f"presence_{col}", "KL", coh), []).append(g.KL.mean())
    return {k: np.asarray(v, float) for k, v in out.items()}


def build_table(vals: dict) -> pd.DataFrame:
    """One row per (edit, metric, kind). A ratio is defined only when every Non-GM seed lies above 0, so the anchor
    is a response the Non-GM reliably shows; otherwise the row is kept with defined = False."""
    rows = []
    ratio = lambda num, den: (float(num.mean() / den.mean()), (num / den.mean()).tolist())
    for group, label, env, edit, role in ROWS:
        for met in ("KL", "SF"):
            for kind, coh in (("edit", "gm"), ("init", "init")):
                num, den = vals.get((env, edit, met, coh)), vals[(env, edit, met, "control")]
                if num is None:
                    continue
                r, per = ratio(num, den)
                rows.append(dict(group=group, row=label, role=role, env=env, edit=edit, metric=met, kind=kind,
                                 defined=bool((den > 0).all()), R=r, seed_ratios=np.round(per, 6).tolist(),
                                 log10R=np.log10(r) if met == "KL" and r > 0 else np.nan, num_mean=num.mean(),
                                 nongm_mean=den.mean(), num_seeds=np.round(num, 6).tolist(),
                                 nongm_seeds=np.round(den, 6).tolist()))
    floors = [(e, ed, n, m, ("PTM" if e == "PTM" else e, src)) for e, ed, n, m, src in FLOORS]
    # PushToMarker presence test, CSV only: green non-goal disc vs the Non-GM response to the white marker at m.
    floors.append(("PTM", "marker_move", "green non-goal disc (presence, vs Non-GM white marker presence)", "KL",
                   ("PTM", "presence_green")))
    for env, edit, name, met, (senv, sedit) in floors:
        group, label, *_ = next(r for r in ROWS if (r[2], r[3]) == (env, edit))
        num = vals[(senv, sedit, met, "gm")]
        den = vals[(senv, "presence_white", met, "control")] if sedit.startswith("presence") else vals[(env, edit, met, "control")]
        r, per = ratio(num, den)
        rows.append(dict(group=group, row=label, env=env, edit=edit, metric=met, kind=f"floor: {name}", defined=True,
                         R=r, seed_ratios=np.round(per, 6).tolist(), log10R=np.log10(r) if met == "KL" else np.nan,
                         num_mean=num.mean(), nongm_mean=den.mean(), num_seeds=np.round(num, 6).tolist(),
                         nongm_seeds=np.round(den, 6).tolist(), plotted=not sedit.startswith("presence")))
    return pd.DataFrame(rows)


def layout() -> tuple[dict, dict, float]:
    """Row y positions (top = 0, downward), group header positions, and the lowest y used."""
    groups = [g for g, *_ in ROWS]
    multi = {g for g in groups if groups.count(g) > 1}
    y, ys, heads, prev = 0.0, {}, {}, None
    for g, (_, lab, *_) in zip(groups, ROWS):
        if g != prev:
            y += 0.5 if prev is not None else 0.0
            if g in multi:
                heads[g] = -y
                y += 0.95
        ys[lab] = -y
        y += 1.0
        prev = g
    return ys, heads, -y + 1.0


def inert_gap(t: pd.DataFrame) -> tuple[float, float, float]:
    """KL ratio gap between the inert location targets and every other edit: (largest inert mean, smallest other
    mean, geometric midpoint), on the ratio of cohort means."""
    e = t[(t.metric == "KL") & (t.kind == "edit")]
    inert = e[e.row.isin(INERT)].R
    rest = e[~e.row.isin(INERT)].R
    return float(inert.max()), float(rest.min()), float(np.sqrt(inert.max() * rest.min()))


def draw(ax, t: pd.DataFrame, met: str, ys: dict, clip: tuple[float, float], undefined_mark: bool) -> None:
    lo, hi = clip
    for _, r in t[(t.metric == met) & (t.kind == "edit")].iterrows():
        y, c, goal = ys[r.row], ROLE_COL[r.role], r.role == "goal"
        if not r.defined:
            if undefined_mark:
                ax.annotate("n/a", xy=(0.0, y), xycoords="data", xytext=(-4, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=7, color=PAL["init"])
            continue
        per = np.asarray(r.seed_ratios)
        inside = (per >= lo) & (per <= hi)
        ax.scatter(per[inside], np.full(inside.sum(), y), s=7, color=c, alpha=0.4, lw=0, zorder=4)
        if lo <= r.R <= hi:
            ax.plot([r.R], [y], "o", ms=fs.MS + 1.0, mfc=c if goal else "white", mec=c, mew=0.4 if goal else 1.1,
                    zorder=5)


def plot(t: pd.DataFrame, out: Path, kl_lim: tuple[float, float], sf_lims: tuple[float, float, float],
         kl_ref: float) -> None:
    fs.FS_TICK = fs.FS_LEG = fs.FS_SUB = 7.0
    fs.style()
    ys, heads, ybot = layout()
    h = 0.78 + 0.155 * (1.0 - ybot)
    left, w_kl, gap, w_sf, w_sf2, brk = 2.02, 1.42, 0.30, 1.05, 0.48, 0.08
    y0, top = 0.62, 0.16
    ah = (h - y0 - top) / h
    X = lambda x: x / fs.W_IN
    fig = plt.figure(figsize=(fs.W_IN, h))
    axk = fig.add_axes([X(left), y0 / h, X(w_kl), ah])
    axs = fig.add_axes([X(left + w_kl + gap), y0 / h, X(w_sf), ah], sharey=axk)
    axs2 = fig.add_axes([X(left + w_kl + gap + w_sf + brk), y0 / h, X(w_sf2), ah], sharey=axk)
    axs2.patch.set_visible(False)
    s_lo, s_mid, s_hi = sf_lims
    draw(axk, t, "KL", ys, kl_lim, False)
    draw(axs, t, "SF", ys, (s_lo, s_mid), True)
    draw(axs2, t, "SF", ys, (s_mid, s_hi), False)
    light = dict(color=PAL["init"], lw=fs.LW_REF, ls=(0, (2, 2)), zorder=1)
    axk.axvline(kl_ref, **light)
    axk.axvline(1.0, **fs.REF, zorder=1)
    axs.axvline(0.0, color=PAL["mute"], lw=fs.LW_REF, zorder=1)
    axs.axvline(1.0, **fs.REF, zorder=1)
    axk.set_xscale("log")
    axk.set_xlim(*kl_lim)
    axk.set_xticks([1e-2, kl_ref, 1e-1, 1, 10, 100])
    axk.set_xticklabels(["0.01", "", "0.1", "1", "10", "100"])
    axk.annotate(f"{kl_ref:.2g}", xy=(kl_ref, 1), xycoords=("data", "axes fraction"), xytext=(0, 1),
                 textcoords="offset points", ha="center", va="bottom", fontsize=7, color=PAL["init"])
    axs.set_xlim(s_lo, s_mid)
    axs.set_xticks([-0.5, 0, 0.5, 1, 1.5])
    axs.set_xticklabels(["−0.5", "0", "0.5", "1", "1.5"])
    axs2.set_xlim(s_mid, s_hi)
    axs2.set_xticks([t for t in (10, 20) if s_mid < t <= s_hi])
    axk.set_yticks([ys[r[1]] for r in ROWS])
    axk.set_yticklabels([r[1] for r in ROWS])
    for ax in (axk, axs, axs2):
        ax.spines["left"].set_visible(False)
        fs.finish(ax)
        ax.tick_params(axis="y", length=0)
    for ax in (axs, axs2):
        ax.tick_params(axis="y", labelleft=False)
    for ax, xa, sgn in ((axs, 1.0, 1), (axs2, 0.0, -1)):
        d = 0.025 * (w_sf / w_sf2 if xa == 0.0 else 1)
        ax.plot([xa - d, xa + d], [-0.012, 0.012], transform=ax.transAxes, color=PAL["ink"], lw=fs.LW_AX,
                clip_on=False)
    axk.set_xlabel("KL, GM ÷ Non-GM")
    axs.set_xlabel("SF, GM ÷ Non-GM")
    axs.xaxis.label.set_x((w_sf + brk + w_sf2) / 2 / w_sf)
    for g, yh in heads.items():
        axk.annotate(g, xy=(0, yh), xycoords=("axes fraction", "data"), xytext=(-left * 72 + 4, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=7, fontweight="bold",
                     color=PAL["ink"], annotation_clip=False)
    axk.set_ylim(ybot - 0.7, 0.7)
    c = PAL["gm"]
    hl = [Line2D([], [], ls="", marker="o", ms=fs.MS + 1, color=c, mec=c, mew=0.4),
          Line2D([], [], ls="", marker="o", ms=fs.MS + 1, mfc="white", mec=PAL["gm_proxy"], mew=1.1),
          Line2D([], [], ls="", marker="o", ms=2.4, color=c, alpha=0.4, mec="none")]
    fig.legend(hl, ["goal edit", "proxy edit", "single GM seed"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.4,
               borderaxespad=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", out.with_suffix(".png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT / "probes",
                    help="data/probes copy")
    ap.add_argument("--kl-lim", type=float, nargs=2, default=[1e-3, 400.0])
    ap.add_argument("--kl-ref", type=float, default=None,
                    help="light KL reference; default: geometric midpoint of the inert / rest gap, rounded")
    ap.add_argument("--sf-lims", type=float, nargs=3, default=[-0.6, 1.6, 26.0],
                    help="SF axis: left segment lo..mid, compressed right segment mid..hi")
    ap.add_argument("--csv", type=Path, default=TABLES / "worldedit_ratios.csv")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "app_worldedit/fig_app_worldedit_ratios")
    args = ap.parse_args()
    vals = robot_values()
    vals.update(procgen_values(args.data_root))
    t = build_table(vals)
    t.to_csv(args.csv, index=False)
    print("wrote", args.csv)
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 60):
        print(t[["row", "metric", "kind", "defined", "R", "seed_ratios"]].round(4).to_string(index=False))
    hi_inert, lo_rest, mid = inert_gap(t)
    print(f"KL gap: inert max {hi_inert:.4f}, others min {lo_rest:.4f}, geometric midpoint {mid:.4f}")
    plot(t, args.out, tuple(args.kl_lim), tuple(args.sf_lims), args.kl_ref if args.kl_ref else round(mid, 2))


if __name__ == "__main__":
    main()
