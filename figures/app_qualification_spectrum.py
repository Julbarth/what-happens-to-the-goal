"""Every scored checkpoint of the eight environments in the (decisiveness, conditional goal preference) plane."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from paths import OUT_DIR, TABLES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import style as S  # noqa: E402

PAL = S.PAL
GREY = "#A8A8A8"
PANELS = [("maze", "Maze"), ("coin", "Coin"), ("attr", "Attribute Maze"), ("PushToMarker", "PushToMarker"),
          ("LocationReach", "LocationReach"), ("AttributeTable", "AttributeTable"), ("SortZones", "SortZones"),
          ("WalkToFlag", "WalkToFlag")]
ROSTER = ("gm_roster", "nongm_roster", "published_pass", "published_fail")
BIG = 10**9  # CAP/arm encodes "never reached" as t = 1e9
MAZE = {"gm": (1080, 1081, 1082), "nongm": (1090, 1091, 1092)}
COIN = {"gm": ("6033_gm_20M", "6037_gm_120M", "6051_gm_55M"), "nongm": ("7001_teach", "7002_teach", "7003_teach")}
ATTR = {"gm": ("6313_black", "6315_black", "6317_black", "6301_texture", "6302_texture", "6303_texture"),
        "nongm": ("4371_black", "4372_black", "4373_black", "4361_texture", "4362_texture", "4363_texture")}
# (CAP/arm file stem, task, in-distribution set per cohort, conflict set). The CAP/arm PushToMarker CSVs have no
# end-state proxy column, so its points come from --robots (GR/results/ptm50, end-state proxy and goal).
ARM = [("AT", "AttributeTable", dict(gm="aligned", control="control"), "conflict"),
       ("LR", "LocationReach", dict(gm="in_region", control="control"), "out_region"),
       ("SZ", "SortZones", dict(gm="aligned", control="control"), "conflict")]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def robot_points(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path)
    cls = np.where(r.cls == "roster", np.where(r.cohort == "gm", "gm_roster", "nongm_roster"),
                   np.where(r.passes, np.where(r.cohort == "gm", "pass", "pass_nongm"), "fail"))
    return pd.DataFrame(dict(env=r.task, ckpt=r.cohort + "_s" + r.seed.astype(str) + "@" + r.steps, n=r.n,
                             n_goal=r.n_goal, n_proxy=r.n_proxy, D=r.D, P=r.P, cls=cls))


def _row(env, ckpt, n, n_goal, n_proxy, cap, cls, src):
    dec = n_goal + n_proxy
    return dict(env=env, ckpt=ckpt, n=n, n_goal=n_goal, n_proxy=n_proxy, D=dec / n,
                P=n_goal / dec if dec else np.nan, capability=cap, cls=cls, src=src)


def cap_points(cap: Path) -> pd.DataFrame:
    """Cohort checkpoints from the N = 500 evaluations. Proxy = proxy reached and goal never (episodes end at the goal)."""
    import json
    rows = []
    for coh, seeds in MAZE.items():
        for s in seeds:
            e = pd.read_csv(cap / f"sampled/maze_ood_greedy_{s}_episodes.csv")
            gf = e.goal.astype(bool)
            pf = (e.t_proxy >= 0) & ~gf & (e.cheese_in_corner == 0)
            c = json.loads((cap / f"sampled/maze_train_sample_{s}_summary.json").read_text())["goal_rate"]
            rows.append(_row("maze", f"{s}@200M", len(e), int(gf.sum()), int(pf.sum()), c, f"{coh}_roster",
                             f"CAP/sampled/maze_ood_greedy_{s}_episodes.csv"))
    for coh, ids in COIN.items():
        for k in ids:
            o = json.loads((cap / f"n500/coin/ood_{k}.json").read_text())
            cfile = f"sampled/coin_indist_{k}_summary.json" if coh == "gm" else f"sampled/coin_aisc_{k}.json"
            c = json.loads((cap / cfile).read_text())["success_rate"]
            oc = o["outcome_counts"]
            rows.append(_row("coin", k, o["n_levels"], oc["success"], oc["goal_misgen"], c, f"{coh}_roster",
                             f"CAP/n500/coin/ood_{k}.json"))
    for coh, ids in ATTR.items():
        for k in ids:
            o = json.loads((cap / f"n500/ras/ood_{k}.json").read_text())
            c = json.loads((cap / f"sampled/ras_indist_{k}.json").read_text())["capability"]
            rows.append(_row("attr", k.replace("_", " "), o["n_episodes"], o["n_goal_hit"], o["n_proxy_hit"], c,
                             f"{coh}_roster", f"CAP/n500/ras/ood_{k}.json"))
    for stem, task, in_set, ood_set in ARM:
        for coh in ("gm", "control"):
            d = pd.read_csv(cap / f"arm/{stem}_{coh}.csv")
            d = d[d["mode"] == "det"]
            d["seed"] = d.ckpt.str.extract(r"_s(\d+)/")[0].astype(int)
            for s, g in d.groupby("seed"):
                ood, tr = g[g.set == ood_set], g[g.set == in_set[coh]]
                gf = ood.goal.astype(bool)
                pf = (ood.t_proxy < BIG) & ~gf
                rows.append(_row(task, f"{coh}_s{s}", len(ood), int(gf.sum()), int(pf.sum()), tr.goal.mean(),
                                 "gm_roster" if coh == "gm" else "nongm_roster", f"CAP/arm/{stem}_{coh}.csv"))
    return pd.DataFrame(rows)


def style(cls: str, texture: bool) -> dict:
    gm, ng = PAL["gm"], PAL["nongm"]
    st = {
        "gm_roster": dict(marker="o", s=16, facecolor=gm, edgecolor="white", lw=0.4, zorder=6),
        "nongm_roster": dict(marker="o", s=16, facecolor=ng, edgecolor="white", lw=0.4, zorder=6),
        "pass": dict(marker="o", s=12, facecolor="white", edgecolor=gm, lw=0.8, zorder=5),
        "pass_nongm": dict(marker="o", s=12, facecolor="white", edgecolor=ng, lw=0.8, zorder=5),
        "fail": dict(marker="o", s=6, facecolor=GREY, edgecolor="none", zorder=3),
        "published_pass": dict(marker="D", s=10, facecolor=gm, edgecolor="white", lw=0.4, alpha=0.55, zorder=5),
        "published_fail": dict(marker="D", s=10, facecolor=GREY, edgecolor="white", lw=0.4, zorder=5),
    }[cls]
    if texture:
        st["marker"] = "s"
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procgen", type=Path, default=TABLES / "spectrum_points.csv")
    ap.add_argument("--robots", type=Path, default=TABLES / "robot_qual_per_seed.csv")
    ap.add_argument("--cap", type=Path, required=True,
                    help="local copy of the released cap_eval_20261011/ results")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "app_qual/fig_app_qual_spectrum")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    S.style()
    cols = ["env", "ckpt", "n", "n_goal", "n_proxy", "D", "P", "cls"]
    old = pd.concat([pd.read_csv(a.procgen)[cols], robot_points(a.robots)], ignore_index=True)
    new = cap_points(a.cap)
    # Cohort checkpoints come from the N = 500 evaluations; non-cohort points and PushToMarker (end-state rule) use the
    # standard evaluations, and WalkToFlag comes from --robots.
    keep = ~old.cls.isin(["gm_roster", "nongm_roster", "nongm_roster_sampled"]) | old.env.isin(["PushToMarker", "WalkToFlag"])
    old = old[keep].assign(src="earlier evaluation")
    old.loc[old.env == "WalkToFlag", "src"] = "closed_loop_fixed/eval (via --robots)"
    pts = pd.concat([new, old], ignore_index=True)
    pts.to_csv(TABLES / "spectrum_points_all8.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(S.W_IN, 3.05), sharex=True, sharey=True)
    for ax, (env, title) in zip(axes.flat, PANELS):
        for _, r in pts[pts.env == env].iterrows():
            st = style(r.cls, "texture" in r.ckpt)
            col = st["facecolor"] if st["facecolor"] != "white" else st["edgecolor"]
            if r.cls in ROSTER:
                xlo, xhi = wilson(r.n_goal + r.n_proxy, r.n)
                ylo, yhi = wilson(r.n_goal, r.n_goal + r.n_proxy)
                ax.plot([xlo, xhi], [r.P] * 2, color=col, lw=S.LW_ERR * 0.7, alpha=0.6, zorder=2)
                ax.plot([r.D] * 2, [ylo, yhi], color=col, lw=S.LW_ERR * 0.7, alpha=0.6, zorder=2)
            ax.scatter(r.D, r.P, **st)
        ax.axhline(0.5, **S.REF, zorder=1)
        ax.set_xlim(0, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        S.panel_title(ax, title)
        S.finish(ax)
    for ax in axes[1]:
        ax.set_xlabel(r"decisiveness $D$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Pr(\mathrm{goal}\mid\mathrm{dec})$")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.215, wspace=0.12, hspace=0.42)

    def h(cls, texture=False):
        st = style(cls, texture)
        return Line2D([], [], ls="", marker=st["marker"], ms=3.6, mfc=st["facecolor"], mec=st["edgecolor"],
                      mew=0.8 if st["facecolor"] == "white" else 0.3, alpha=st.get("alpha", 1))
    handles = [h("gm_roster"), h("nongm_roster"), (h("pass"), h("pass_nongm")),
               h("fail"), h("published_pass"), h("published_fail"),
               Line2D([], [], ls="", marker="s", ms=3.6, mfc="white", mec=PAL["ink"], mew=0.6)]
    labels = ["GM", "Non-GM", "qualifies, not carried", "does not qualify",
              "published, qualifies", "published, does not qualify", "texture background"]
    S.legend_row(fig, handles, labels, y_in=0.02, ncol=4, handlelength=1.6, columnspacing=1.4,
                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
    fig.savefig(a.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out.with_suffix(".png"), dpi=a.dpi, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
