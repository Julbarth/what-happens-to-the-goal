"""Rewiring from every layer of the frozen robot (conv1, conv2, h_pre, h_fc) and Procgen (h1, h2, h_pre, h_fc) trunks, with the goal probe overlaid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style as S  # noqa: E402
from style import PAL  # noqa: E402
import app_rewire_all_envs as A  # noqa: E402
from app_rewire_recovery import band  # noqa: E402

import app_robot_probes as P  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, TABLES

GR = A.GR
SITES = ["conv1", "conv2", "h_pre", "h_fc"]
XLAB = [r"conv$_1$", r"conv$_2$", r"$h_{\rm pre}$", r"$h_{\rm fc}$"]
ARM = {"conv1": "conv1", "conv2": "conv2", "h_pre": "pre", "h_fc": "fc_head", "native": "native"}
TASKS = [("LR", "LocationReach"), ("PTM", "PushToMarker"), ("AT", "AttributeTable"), ("SZ", "SortZones")]
# goal-probe targets per task, as in the robot probe figure (ridge_sel, R^2 averaged over x and y)
PROBE = {"LR": ("marker_x", "marker_y"), "PTM": ("marker_x", "marker_y"), "AT": ("cube_x", "cube_y"),
         "SZ": ("cross_x", "cross_y")}
EXTRA = DATA_ROOT / "probes/rewire_extra_20261012"
PG_SITES = ["h1", "h2", "h_pre", "h_fc"]
PG_XLAB = [r"$h^{(1)}$", r"$h^{(2)}$", r"$h_{\rm pre}$", r"$h_{\rm fc}$"]
# (key, title, cloning targets of the sweep, native key in the recovery-bar CSV)
PROCGEN = [("maze", "Maze", "oracle", "Maze"), ("coin", "Coin", "teacher", "Coin"), ("am", "Attribute Maze", "oracle", "AM")]
# goal probe per Procgen env, as cited in the rewiring text: Maze cheese bearing (AUC), Coin coin bearing
# residualized on progress (R^2), Attribute Maze black line presence (AUC); probe sites h1, h2, h3, h_pre, h_fc
PG_PROBE_IDX = [0, 1, 3, 4]
AT_NONGM = A.AT_NONGM_SEEDS
# the random trunk stays in the CSVs but is not drawn; it is compared in tab:rewire-main
DRAWN = ("gm", "control")


def episodes(env: str) -> pd.DataFrame:
    """One goal indicator per episode, standard condition, one protocol per task across all four sites."""
    if env in ("AT", "SZ"):
        d = pd.read_csv(GR / f"depth/rewire_ctrl2/{env}_per_episode.csv", low_memory=False)
        d = d[d.cond == "standard"].assign(goal=lambda x: x.outcome.eq("u").astype(float))
        d = d.assign(arm=d.arm.replace({"h_pre": "pre", "h_pre_masked": "pre_masked", "h_fc": "fc_head",
                                        "h_fc_masked": "fc_head_masked"}))
        if env == "AT":
            # native reach at the sweep's 50-step horizon, from the paper-protocol file
            at = pd.read_csv(GR / "rewire_at_deep/per_seed.csv")
            at = at[(at.variant == "base") & (at.arm == "native") & (at.cond == "standard")]
            nat = at[["cohort", "seed"]].assign(arm="native", goal=at.P_u_50.values)
        else:
            sz = pd.read_csv(GR / "rewire_sz_ctrl2/per_seed.csv")
            sz = sz[(sz.variant == "base") & (sz.arm == "native") & (sz.cond == "standard")]
            nat = sz[["cohort", "seed"]].assign(arm="native", goal=sz.P_u.values)
        rates = d.groupby(["cohort", "seed", "arm"]).goal.mean().reset_index()
        return pd.concat([rates, nat], ignore_index=True)
    if env == "LR":
        old = pd.read_csv(GR / "rewire_check/per_episode.csv", low_memory=False)
        new = pd.read_csv(GR / "depth/rewire/LR_per_episode.csv", low_memory=False)
        parts = [x[x.cond == "standard"].assign(goal=lambda y: y.reach_scored.astype(str).str.lower().eq("true")
                                                .astype(float)) for x in (old, new)]
    else:
        old = pd.read_csv(GR / "rewire_ptm/per_episode.csv")
        old = old[(old.set == "conflict") & (old.cond == "standard")]
        new = pd.read_csv(GR / "depth/rewire/PTM_per_episode.csv")
        new = new[new.cond == "conflict:standard"]
        parts = [x.assign(goal=x.outcome.eq("goal").astype(float)) for x in (old, new)]
    d = pd.concat([p[["cohort", "seed", "arm", "goal"]] for p in parts], ignore_index=True)
    return d.groupby(["cohort", "seed", "arm"]).goal.mean().reset_index()


def values(r: pd.DataFrame, env: str, coh: str, arm: str) -> list[float]:
    x = r[(r.cohort == coh) & (r.arm == arm)]
    if env == "AT" and coh == "control":
        x = x[x.seed.isin(AT_NONGM)]
    return x.sort_values("seed").goal.tolist()


def procgen_probes() -> dict:
    fp = TABLES
    pb = json.loads((fp / "plotted_values_procgen_baselines.json").read_text())
    am = json.loads((fp / "plotted_values_attribute_seeded.json").read_text())["means"]["Black|Line presence"]
    return {"maze": pb["maze"]["Cheese bearing"], "coin": pb["coin"]["Coin bearing"], "am": am}


def procgen_panels(axes, sty, pco):
    """Procgen sweep from data/probes/rewire_extra_20261012 (standard condition only; goal-removed not drawn)."""
    nat = pd.read_csv(TABLES / "fig_app_rewire_recovery_bars_teacher.csv")
    probes = procgen_probes()
    rows, prow = [], []
    xs = np.arange(len(PG_SITES))
    coh_csv = {"gm": "gm", "control": "nongm", "init": "init"}
    for ax, (env, title, lab, nkey) in zip(axes, PROCGEN):
        d = pd.read_csv(EXTRA / f"per_seed_{env}_{lab}_standard.csv")
        S.panel_title(ax, f"{title} ({lab})")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.5, len(PG_SITES) - 0.5)
        for coh, below in (("gm", True), ("nongm", False)):
            v = nat[(nat.env == nkey) & (nat.arm == f"native_{coh}")].seeds.iloc[0]
            band(ax, [float(x) for x in v.split(";")], PAL["gm" if coh == "gm" else "nongm"], below=below)
        for coh, (c, m, off) in sty.items():
            pv = probes[env][pco[coh]]
            pm = [float(pv[i]) for i in PG_PROBE_IDX]
            prow += [dict(env=env, cohort=coh, site=s, probe=v) for s, v in zip(PG_SITES, pm)]
            if coh in DRAWN:
                ax.plot(xs + off, pm, color=c, lw=0.8, ls=(0, (3, 1.5)), marker="D", ms=S.MS - 0.8, mfc="white",
                        mec=c, mew=0.8, zorder=2, alpha=0.9)
            for x, s in zip(xs, PG_SITES):
                v = d[(d.cohort == coh_csv[coh]) & (d.tap == s)].sort_values("trunk").goal.to_numpy()
                rows += [dict(env=env, labels=lab, cohort=coh, site=s, seed_index=i, value=float(u))
                         for i, u in enumerate(v)]
                if not len(v) or coh not in DRAWN:
                    continue
                mu, h = A.ci(v)
                ax.plot([x + off] * 2, [max(mu - h, 0), min(mu + h, 1)], color=c, lw=S.LW_ERR, zorder=4)
                ax.plot(x + off, mu, m, ms=S.MS + 0.6, color=c, zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(PG_XLAB)
        S.finish(ax)
    return rows, prow


def main(a):
    S.style()
    rows = []
    for env, _ in TASKS:
        r = episodes(env)
        for coh in ("gm", "control", "init"):
            for site in SITES + ["native"]:
                for suf in ("", "_masked"):
                    arm = ARM[site] + suf if site != "native" or not suf else None
                    if arm is None:
                        continue
                    for i, v in enumerate(values(r, env, coh, arm)):
                        rows.append(dict(env=env, cohort=coh, site=site, masked=bool(suf), seed_index=i, value=v))
    long = pd.DataFrame(rows)
    long.to_csv(TABLES / "fig_app_rewire_layers_per_seed.csv", index=False)
    summ = long.groupby(["env", "cohort", "site", "masked"]).value.agg(["count", "mean", lambda v: v.std(ddof=1)])
    summ.columns = ["n", "mean", "sd"]
    summ.to_csv(TABLES / "fig_app_rewire_layers_summary.csv", float_format="%.4f")

    probe_rows = []
    fig, axes2 = plt.subplots(2, 4, figsize=(S.W_IN, 4.2), sharey=True)
    axes = axes2[0]
    xs = np.arange(len(SITES))
    sty = {"gm": (PAL["gm"], "o", -0.09), "control": (PAL["nongm"], "o", 0.09), "init": (PAL["init"], "s", 0.0)}
    pco = {"gm": "GM", "control": "Non-GM", "init": "init"}
    for ax, (env, title) in zip(axes, TASKS):
        e = long[long.env == env]
        S.panel_title(ax, title)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.5, len(SITES) - 0.5)
        band(ax, e[(e.cohort == "gm") & (e.site == "native")].value, PAL["gm"], below=True)
        band(ax, e[(e.cohort == "control") & (e.site == "native")].value, PAL["nongm"], below=False)
        pr = P.load_arm(env, PROBE[env], "ridge_sel")["cohorts"]
        for coh, (c, m, off) in sty.items():
            pm = [float(np.mean(pr[pco[coh]][s])) for s in SITES]
            probe_rows += [dict(env=env, cohort=coh, site=s, probe_r2=v) for s, v in zip(SITES, pm)]
            if coh not in DRAWN:
                continue
            ax.plot(xs + off, pm, color=c, lw=0.8, ls=(0, (3, 1.5)), marker="D", ms=S.MS - 0.8, mfc="white",
                    mec=c, mew=0.8, zorder=2, alpha=0.9)
            for x, s in zip(xs, SITES):
                v = e[(e.cohort == coh) & (e.site == s) & ~e.masked].value.to_numpy()
                if not len(v):
                    continue
                mu, h = A.ci(v)
                ax.plot([x + off] * 2, [max(mu - h, 0), min(mu + h, 1)], color=c, lw=S.LW_ERR, zorder=4)
                ax.plot(x + off, mu, m, ms=S.MS + 0.6, color=c, zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(XLAB)
        S.finish(ax)
    pd.DataFrame(probe_rows).to_csv(TABLES / "fig_app_rewire_layers_probe_overlay.csv", index=False, float_format="%.4f")
    pg_rows, pg_probe = procgen_panels(axes2[1], sty, pco)
    pd.DataFrame(pg_rows).to_csv(TABLES / "fig_app_rewire_layers_procgen_per_seed.csv", index=False)
    pd.DataFrame(pg_probe).to_csv(TABLES / "fig_app_rewire_layers_procgen_probe_overlay.csv", index=False,
                                  float_format="%.4f")
    axes2[1, 3].axis("off")
    axes[0].set_ylabel(r"$\hat J^\star$ or probe $R^2$")
    axes2[1, 0].set_ylabel(r"$\hat J^\star$ or probe")
    h = [Line2D([], [], color=PAL["gm"], marker="o", ls="", ms=S.MS + 0.4),
         Line2D([], [], color=PAL["nongm"], marker="o", ls="", ms=S.MS + 0.4),
         Line2D([], [], color=PAL["mute"], ls=(0, (3, 1.5)), lw=0.8, marker="D", mfc="white", ms=S.MS - 0.8),
         (Patch(color=PAL["gm"], alpha=0.12, lw=0), Line2D([], [], color=PAL["gm"], ls=(0, (4, 2)))),
         (Patch(color=PAL["nongm"], alpha=0.12, lw=0), Line2D([], [], color=PAL["nongm"], ls=(0, (4, 2))))]
    labels = ["GM rewire", "Non-GM rewire", "goal probe", "native GM",
              "native Non-GM"]
    fig.legend(h, labels, loc="center", ncol=1, frameon=False, fontsize=S.FS_LEG, handletextpad=0.4,
               handlelength=2.2, bbox_to_anchor=(0.875, 0.27))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.07, wspace=0.12, hspace=0.45)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"))
    fig.savefig(a.out.with_suffix(".png"), dpi=300)
    print("wrote", a.out.with_suffix(".pdf"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_DIR / "app_rewire/fig_app_rewire_layers")
    main(p.parse_args())
