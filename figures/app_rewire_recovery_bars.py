"""Rewiring recovery as bars by depth group (oracle targets in Maze and Attribute Maze, teacher elsewhere), against chance and the random trunk."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import style as S  # noqa: E402
from style import PAL  # noqa: E402
import app_rewire_all_envs as A  # noqa: E402
import app_rewire_recovery as R  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, ROBOTICS_ROOT, TABLES

MATRIX = DATA_ROOT / "experiments/Cross-env/rewire_appendix_matrix.md"
MAZE_RAW = TABLES / "fig_maze_rewire_overview_raw.csv"
# Mean decisiveness of the GM teacher rewire from h_pre in Attribute Maze, as reported in the rewiring appendix.
AM_TEACHER_D = 0.690
EXTRA = DATA_ROOT / "probes/rewire_extra_20261012"
ROBOT_EXTRA = (ROBOTICS_ROOT / "results/analysis/rewire_extra/robots_rewire_per_seed.csv")
BAR_W = 0.2
# (offset within a site, arm prefix); the random trunk sits right of the cohort pair
OFFS = [(-BAR_W, "gm"), (0.0, "nongm"), (BAR_W, "init")]
SITES = [(0.0, "pre", r"$h_{\rm pre}$"), (1.0, "remix", "remix"), (2.0, "fc", r"$h_{\rm fc}$")]


def matrix(env: str, bg: str) -> dict[str, tuple[float, float]]:
    """{arm_key: (mean, 95% half-width)} for the teacher rows of the rewire matrix."""
    out = {}
    for line in MATRIX.read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 8 or c[0] != env or c[1] != bg or c[2] != "teacher" or c[4] == "native":
            continue
        lo, hi = map(float, re.findall(r"-?\d+\.\d+", c[7]))
        arm = {"rewire": c[3].replace("h_", ""), "random_proj": "remix"}[c[4]]
        out[f"{c[5]}_{arm}"] = (float(c[6]), (hi - lo) / 2)
    return out


def teacher_procgen(oracle: dict) -> dict:
    """Teacher-target Maze and Attribute Maze; natives and the target-free random walk come from the oracle loader."""
    raw = pd.read_csv(MAZE_RAW)
    raw = raw[raw.attempt == "bc_teacher"]

    def mz(site, coh):
        return raw[(raw.site == site) & (raw.cohort == coh)].sort_values("checkpoint").cheese_reach.tolist()

    mm = matrix("Maze", "-")
    maze = dict(native_gm=oracle["Maze"]["native_gm"], native_nongm=oracle["Maze"]["native_nongm"],
                chance=oracle["Maze"]["chance"], gm_pre=mz("block3", "GM"), nongm_pre=mz("block3", "Non-GM"),
                gm_fc=mz("fc", "GM"), nongm_fc=mz("fc", "Non-GM"), gm_remix=mm["gm_remix"],
                nongm_remix=mm["nongm_remix"])
    am = matrix("Attribute Maze", "Black")
    attr = dict(native_gm=oracle["AM"]["native_gm"], native_nongm=oracle["AM"]["native_nongm"],
                chance=[0.5 * AM_TEACHER_D], **{k: am[k] for k in ("gm_pre", "nongm_pre", "gm_remix", "nongm_remix",
                                                                    "gm_fc", "nongm_fc")})
    for env, key in (("Maze", "maze"), ("AM", "am"), ("Coin", "coin")):
        ps = pd.read_csv(EXTRA / f"per_seed_{key}_teacher_standard.csv")
        ps = ps[ps.cohort == "init"].sort_values("trunk")
        target = {"Maze": maze, "AM": attr}.get(env, oracle["Coin"])
        if env != "Coin":
            target["init"] = ps[ps.tap == "h_pre"].goal.tolist()
        target["init_fc"] = ps[ps.tap == "h_fc"].goal.tolist()
    return {**oracle, "Maze": maze, "AM": attr}


def robot_fc_floors(data: dict) -> None:
    """Random trunk rewired from h_fc in the five robot tasks, teacher targets."""
    rs = pd.read_csv(ROBOT_EXTRA)
    for env, task in (("LR", "LR"), ("PTM", "PTM (conflict)"), ("AT", "AT"), ("SZ", "SZ")):
        data[env]["init_fc"] = rs[(rs.task == task) & (rs.cohort == "init") & (rs.tap == "h_fc")].sort_values(
            "seed").goal.tolist()
    w = pd.concat([pd.read_csv(f) for f in R.WTF_CSV])
    w = w[(w.cohort == "init") & (w.arm == "fc_head") & (w.target == "teacher") & (w.set == "native")]
    data["WTF"]["init_fc"] = (w["first"] == "u").groupby(w.seed).mean().sort_index().tolist()


def stat(v) -> tuple[float, float, list[float]]:
    if isinstance(v, tuple):
        return v[0], v[1], []
    mu, h = A.ci(v)
    return mu, h, list(v)


def oracle_fc_floors(oracle: dict) -> None:
    """Random trunk rewired from h_fc with oracle targets in Maze and Attribute Maze."""
    for env, key in (("Maze", "maze"), ("AM", "am")):
        ps = pd.read_csv(EXTRA / f"per_seed_{key}_oracle_standard.csv")
        ps = ps[(ps.cohort == "init") & (ps.tap == "h_fc")].sort_values("trunk")
        oracle[env]["init_fc"] = ps.goal.tolist()


def main(a):
    S.style()
    oracle = R.procgen()
    if a.targets == "teacher":
        procgen = teacher_procgen(oracle)
    else:
        oracle_fc_floors(oracle)
        procgen = oracle if a.targets == "oracle" else {**teacher_procgen(dict(oracle)), "Maze": oracle["Maze"],
                                                        "AM": oracle["AM"]}
    data = {**procgen, **R.robots()}
    robot_fc_floors(data)
    rows = []
    for e, d in data.items():
        for k, v in d.items():
            mu, h, seeds = stat(v)
            rows.append(dict(env=e, arm=k, n=len(seeds) if seeds else 3, mean=mu, ci95_half=h,
                             seeds=";".join(f"{x:.4f}" for x in seeds)))
    pd.DataFrame(rows).to_csv(TABLES / f"fig_app_rewire_recovery_bars_{a.targets}.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(2, 4, figsize=(S.W_IN, 3.9), sharey=True)
    rng = np.random.default_rng(0)
    col = {"gm": PAL["gm"], "nongm": PAL["nongm"], "init": PAL["init"]}
    for ax, env in zip(axes.flat, R.ORDER):
        S.panel_title(ax, R.TITLES[env])
        ax.set_ylim(0, 1.03)
        ax.set_xlim(-0.45, 2.45)
        d = data.get(env, {})
        R.band(ax, d["native_gm"], PAL["gm"], below=True)
        R.band(ax, d["native_nongm"], PAL["nongm"], below=False)
        ax.axhline(np.mean(d["chance"]), color=PAL["ink"], lw=S.LW_REF + 0.2, ls=(0, (1, 1.5)), zorder=6)
        for xs, site, _ in SITES:
            for off, coh in OFFS:
                x = xs + off
                if coh == "init" and site == "remix":
                    continue
                key = "init" if (coh, site) == ("init", "pre") else f"{coh}_{site}"
                if key not in d or (isinstance(d[key], list) and not d[key]):
                    continue
                mu, h, seeds = stat(d[key])
                c = col[coh]
                hollow = site == "remix"
                ax.add_patch(Rectangle((x - BAR_W / 2 + 0.01, 0), BAR_W - 0.02, mu, facecolor="white" if hollow else c,
                                       edgecolor=c, lw=0.9, alpha=1.0 if hollow else 0.85, zorder=3))
                ax.plot([x, x], [max(mu - h, 0), min(mu + h, 1.03)], color=PAL["ink"], lw=S.LW_ERR, zorder=5)
                if seeds:
                    ax.scatter(x + rng.uniform(-0.04, 0.04, len(seeds)), seeds, color=PAL["ink"], zorder=4, s=3,
                               alpha=0.45, lw=0)
        ax.set_xticks([s for s, _, _ in SITES])
        ax.set_xticklabels([lab for _, _, lab in SITES])
        S.finish(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"goal reach $\hat J^\star$")
    h = [Patch(facecolor=PAL["gm"], edgecolor=PAL["gm"], alpha=0.85), Patch(facecolor=PAL["nongm"], alpha=0.85),
         Patch(facecolor=PAL["init"], alpha=0.85), Patch(facecolor="white", edgecolor=PAL["mute"]),
         (Patch(color=PAL["gm"], alpha=0.12, lw=0), Line2D([], [], color=PAL["gm"], ls=(0, (4, 2)))),
         (Patch(color=PAL["nongm"], alpha=0.12, lw=0), Line2D([], [], color=PAL["nongm"], ls=(0, (4, 2)))),
         Line2D([], [], color=PAL["ink"], ls=(0, (1, 1.5)))]
    labels = ["GM rewire", "Non-GM rewire", "random trunk", "fixed random remix", "native GM", "native Non-GM",
              "chance"]
    fig.legend(h, labels, loc="lower center", ncol=4, frameon=False, fontsize=S.FS_LEG, handletextpad=0.3,
               columnspacing=0.8, handlelength=1.8, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.16, wspace=0.12, hspace=0.62)
    for (r, c0, c1, lab) in ((0, 0, 1, "(i) goal lost in the representation"),
                             (0, 2, 3, "(ii) goal reaches the policy, unused"),
                             (1, 0, 3, "(iii) goal and proxy compete")):
        p0, p1 = axes[r, c0].get_position(), axes[r, c1].get_position()
        y = p0.y1 + 0.075
        fig.text((p0.x0 + p1.x1) / 2, y, lab, ha="center", va="bottom", fontsize=S.FS, fontweight="bold",
                 color=PAL["ink"])
        fig.add_artist(Line2D([p0.x0 + 0.01, p1.x1 - 0.01], [y - 0.008] * 2, color=PAL["ink"], lw=0.5))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"))
    fig.savefig(a.out.with_suffix(".png"), dpi=300)
    print("wrote", a.out.with_suffix(".pdf"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", choices=["mixed", "teacher", "oracle"], default="mixed",
                   help="Maze and Attribute Maze cloning targets (mixed = oracle); Coin and the robots are teacher-only")
    p.add_argument("--out", type=Path, default=OUT_DIR / "app_rewire/fig_app_rewire_recovery_bars")
    main(p.parse_args())
