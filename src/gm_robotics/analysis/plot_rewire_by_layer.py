"""Per-layer rewiring figure and table: goal reach for GM / control / random-init trunks from conv1, conv2, h_pre, h_fc."""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from gm_robotics.paths import ROOT as RUN_ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = str(RUN_ROOT)
RES = os.path.join(ROOT, "results", "analysis")
FIG = os.path.join(ROOT, "figures", "analysis", "paper_style")
SITES = ["conv1", "conv2", "h_pre", "h_fc"]
TASKS = [("LR", "LocationReach"), ("PTM", "PushToMarker"), ("AT", "AttributeTable"), ("SZ", "SortZones")]
# Paper cohort colours; marker shape and line style carry identity as well, since the grey / teal pair is close in CVD.
COH = {"gm": ("GM", "#7A1520", "o", "-"), "control": ("control", "#3D8B8B", "s", "-"), "init": ("random-init trunk", "#8E8E8E", "^", "--")}
NOTE = {"LR": "first contact, 50 steps, off-path set", "PTM": "final cube on marker, 100 steps, control-distribution set",
        "AT": "first touch, 50 steps, ctrl2 BC", "SZ": "first pad entry, 100 steps, ctrl2 BC"}


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - h, m + h


def load():
    d = pd.read_csv(os.path.join(RES, "depth", "depth_per_seed.csv"))
    return d[d.measure.isin(["rewire_goal", "rewire_masked"]) & d.site.isin(SITES)]


def table(d):
    rows = []
    for (env, meas, coh, site), g in d.groupby(["env", "measure", "cohort", "site"]):
        m, lo, hi = tci(g.value)
        rows.append(dict(task=env, measure=meas, cohort=coh, site=site, n=len(g), mean=m, lo=lo, hi=hi,
                         seeds=" / ".join(f"{v:.3f}" for v in g.sort_values("seed").value)))
    return pd.DataFrame(rows)


def markdown(t):
    L = ["# Rewired goal reach by layer (mean [95% t-CI] over seeds)", "",
         "Source: results/analysis/depth/depth_per_seed.csv (one protocol per task across all four layers). "
         "Masked = GM goal-cue-masked retrain at that layer (the empirical goal-blind floor).", "",
         "| task | layer | GM | control | random-init | GM masked |", "|---|---|---|---|---|---|"]
    c = {(r.task, r.measure, r.cohort, r.site): r for r in t.itertuples()}

    def f(k):
        r = c.get(k)
        return "–" if r is None else f"{r.mean:.2f} [{r.lo:.2f}, {r.hi:.2f}]"
    for env, _ in TASKS:
        for s in SITES:
            L.append(f"| {env} | {s} | " + " | ".join(f((env, "rewire_goal", coh, s)) for coh in COH) + f" | {f((env, 'rewire_masked', 'gm', s))} |")
    L += ["", "Scoring per task: " + "; ".join(f"{e} {NOTE[e]}" for e, _ in TASKS) + ". AT here uses separately trained "
          "ctrl2 heads scored at 50 steps (GM h_pre 0.73 vs 0.76 in rewire_at_deep at 100 steps; that run shows 50 vs 100 "
          "steps changes nothing, so the gap is retraining variance). PTM is scored on the control-distribution set, where "
          "a marker-blind stop-short push cannot score (conflict-set h_pre 0.314, same value)."]
    return "\n".join(L) + "\n"


def plot(d, out):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9), sharey=True)
    rng = np.random.default_rng(0)
    x = np.arange(len(SITES))
    for ax, (env, title) in zip(axes, TASKS):
        e = d[d.env == env]
        mk = e[(e.measure == "rewire_masked") & (e.cohort == "gm")]
        ms = [mk[mk.site == s].value.mean() if (mk.site == s).any() else np.nan for s in SITES]
        ax.plot(x, ms, ":", color="#2A2A2A", lw=1.2, label="GM goal-cue-masked retrain (floor)", zorder=1)
        for j, (coh, (lab, col, m, ls)) in enumerate(COH.items()):
            g = e[(e.measure == "rewire_goal") & (e.cohort == coh)]
            st = [tci(g[g.site == s].value) for s in SITES]
            off = (j - 1) * 0.12
            mu = np.array([s[0] for s in st])
            ax.plot(x + off, mu, ls, color=col, lw=1.6, zorder=3)
            ax.errorbar(x + off, mu, yerr=[mu - np.array([s[1] for s in st]), np.array([s[2] for s in st]) - mu],
                        fmt=m, color=col, ms=6, capsize=2.5, lw=1.2, mec="white", mew=0.8, zorder=4, label=lab)
            for i, s in enumerate(SITES):
                v = g[g.site == s].value.to_numpy()
                ax.scatter(np.full(len(v), i + off) + rng.uniform(-0.03, 0.03, len(v)), v, s=9, color=col, alpha=0.45,
                           lw=0, zorder=2)
        ax.set_xticks(x, SITES)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=9, pad=14)
        ax.text(0.5, 1.02, NOTE[env], transform=ax.transAxes, fontsize=6.5, color="#6E6E6E", ha="center")
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    axes[0].set_ylabel("rewired goal reach")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_rewire_by_layer.{ext}"), dpi=200, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=FIG)
    a = p.parse_args()
    d = load()
    t = table(d)
    t.to_csv(os.path.join(RES, "rewire_by_layer_summary.csv"), index=False)
    md = markdown(t)
    open(os.path.join(RES, "rewire_by_layer_summary.md"), "w").write(md)
    plot(d, a.out)
    print(md)


if __name__ == "__main__":
    main()
