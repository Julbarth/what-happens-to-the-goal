"""Normalized rewire recovery (GM - floor) / (Non-GM - floor), floor = max(random trunk, chance), per environment and site."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import style as S  # noqa: E402
from style import PAL  # noqa: E402

from paths import DATA_ROOT, OUT_DIR, ROBOTICS_ROOT, TABLES

BARS = TABLES / "fig_app_rewire_recovery_bars_teacher.csv"
MAZE_DAGGER = DATA_ROOT / "experiments/Maze 1/data/maze_dagger_raw_20260920.json"
PTM_DAGGER = (ROBOTICS_ROOT / "results/analysis/rewire_ptm_dagger/per_seed.csv")
ROLES = [("Inert", ["Maze", "LR"]), ("Weakly wired", ["PTM", "Coin"]), ("Competition", ["AM", "AT", "SZ", "WTF"])]
NAMES = dict(Maze="Maze", LR="LocationReach", PTM="PushToMarker", Coin="Coin", AM="Attribute Maze",
             AT="AttributeTable", SZ="SortZones", WTF="WalkToFlag")
T_CRIT_N3 = stats.t.ppf(0.975, 2)


def arm(df, env, key):
    """(mean, SE, n) of one arm; per-seed values where stored, else SE from the stored 95% half-width (n = 3, t)."""
    r = df[(df.env == env) & (df.arm == key)]
    if r.empty:
        return None
    r = r.iloc[0]
    if isinstance(r.seeds, str) and r.seeds:
        v = np.array([float(x) for x in r.seeds.split(";")])
        if len(v) > 1:
            return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)
        return v[0], 0.0, 1
    return r["mean"], r.ci95_half / T_CRIT_N3, int(r.n)


def from_seeds(v):
    v = np.asarray(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)


def ratio(g, nn, rt, ch):
    """Delta-method ratio with a Welch-Satterthwaite df over the three variance components; the floor is whichever of
    random trunk and chance has the larger mean, with that arm's SE."""
    f = rt if rt[0] >= ch[0] else ch
    num, den = g[0] - f[0], nn[0] - f[0]
    if den <= 0:
        return dict(floor=f[0], floor_src="random trunk" if f is rt else "chance", gm=g[0], nongm=nn[0], gm_excess=num,
                    nongm_excess=den, recovery=np.nan, ci_lo=np.nan, ci_hi=np.nan, df=np.nan,
                    denominator_ci_includes_0=True)
    r = num / den
    grads = [1 / den, -num / den ** 2, (g[0] - nn[0]) / den ** 2]
    comps = [(gr * x[1]) ** 2 for gr, x in zip(grads, (g, nn, f))]
    var = sum(comps)
    dfw = var ** 2 / sum(c ** 2 / max(x[2] - 1, 1) for c, x in zip(comps, (g, nn, f)) if c > 0) if var > 0 else np.inf
    half = stats.t.ppf(0.975, dfw) * np.sqrt(var)
    # The ratio is not interpretable once the Non-GM excess over the floor is not clearly positive.
    den_se = np.sqrt(nn[1] ** 2 + f[1] ** 2)
    unstable = den - stats.t.ppf(0.975, max(nn[2] - 1, 1)) * den_se <= 0
    return dict(floor=f[0], floor_src="random trunk" if f is rt else "chance", gm=g[0], nongm=nn[0],
                gm_excess=num, nongm_excess=den, recovery=r, ci_lo=r - half, ci_hi=r + half, df=dfw,
                denominator_ci_includes_0=bool(unstable))


def maze_dagger():
    d = json.loads(MAZE_DAGGER.read_text())["rows"]
    out = {}
    for f, v in d.items():
        _, tap, coh, *mode = f.replace(".json", "").split("_")
        out[(tap, coh, "_".join(mode))] = [s["history"][-1]["cheese_reach"] for s in v.values()]
    return out


def build():
    df = pd.read_csv(BARS)
    rows = []
    for _, envs in ROLES:
        for env in envs:
            ch = arm(df, env, "chance")
            for site, sfx, rk in (("h_pre", "pre", "init"), ("h_fc", "fc", "init_fc")):
                g, nn, rt = arm(df, env, f"gm_{sfx}"), arm(df, env, f"nongm_{sfx}"), arm(df, env, rk)
                rows.append(dict(env=env, site=site, variant="BC (teacher)", **ratio(g, nn, rt, ch)))
    bars = df.set_index(["env", "arm"])
    md = maze_dagger()
    ch = arm(df, "Maze", "chance")
    # Maze DAgger runs with oracle supervision; its floor is the oracle random trunk (Table rewire-main, oracle block).
    rt_oracle = {"h_pre": (0.018, 0.0, 3), "h_fc": (0.009, 0.0, 3)}
    for site, tap in (("h_pre", "block3"), ("h_fc", "fc")):
        for mode, lab in (("onpolicy", "DAgger (oracle)"), ("base_only", "Matched (oracle)")):
            g, nn = from_seeds(md[(tap, "gm", mode)]), from_seeds(md[(tap, "nongm", mode)])
            rows.append(dict(env="Maze", site=site, variant=lab, **ratio(g, nn, rt_oracle[site], ch)))
    if PTM_DAGGER.exists():
        p = pd.read_csv(PTM_DAGGER)
        p = p[(p.set == "conflict") & (p["round"] == p["round"].max())]
        for site in sorted(p.site.unique()):
            for mode, lab in (("dagger", "DAgger (teacher)"), ("matched", "Matched (teacher)")):
                q = p[(p.site == site) & (p["mode"] == mode)]
                if not {"gm", "control"} <= set(q.cohort):
                    continue
                g = from_seeds(q[q.cohort == "gm"].goal)
                nn = from_seeds(q[q.cohort == "control"].goal)
                rt = from_seeds(q[q.cohort == "init"].goal) if (q.cohort == "init").sum() > 1 else \
                    arm(df, "PTM", "init" if site == "pre" else "init_fc")
                rows.append(dict(env="PTM", site="h_" + site, variant=lab, **ratio(g, nn, rt, arm(df, "PTM", "chance"))))
    del bars
    return pd.DataFrame(rows)


def plot(res, out):
    S.style()
    fig, ax = plt.subplots(figsize=(S.W_IN, 2.1))
    x, ticks, labels, groups = 0.0, [], [], []
    col = {"h_pre": PAL["gm"], "h_fc": PAL["nongm"]}
    lo_lim, hi_lim = -1.5, 2.0
    for role, envs in ROLES:
        x0 = x
        for env in envs:
            for j, site in enumerate(("h_pre", "h_fc")):
                sub = res[(res.env == env) & (res.site == site)]
                for k, (_, r) in enumerate(sub.iterrows()):
                    xi = x + (j - 0.5) * 0.36 + k * 0.09
                    dag = r.variant.startswith("DAgger")
                    if r.variant.startswith("Matched"):
                        continue
                    if np.isnan(r.recovery):
                        ax.text(xi, 0.5, "n.d.", rotation=90, ha="center", va="center", fontsize=S.FS_TICK - 1,
                                color=col[site])
                        continue
                    lo, hi = max(r.ci_lo, lo_lim), min(r.ci_hi, hi_lim)
                    ax.plot([xi, xi], [lo, hi], color=col[site], lw=S.LW_ERR, alpha=0.5 if r.denominator_ci_includes_0 else 1)
                    for v, lim, mk in ((r.ci_lo, lo_lim, "v"), (r.ci_hi, hi_lim, "^")):
                        if (v < lim) if mk == "v" else (v > lim):
                            ax.plot(xi, lim, mk, color=col[site], ms=2.5)
                    y = min(max(r.recovery, lo_lim), hi_lim)
                    if y != r.recovery:
                        ax.text(xi + 0.07, y, f"{r.recovery:.1f}", fontsize=S.FS_TICK - 1, color=col[site],
                                ha="left", va="center")
                    face = "white" if (dag or r.denominator_ci_includes_0) else col[site]
                    ax.plot(xi, y, "D" if dag else "o", ms=S.MS, mfc=face, mec=col[site], mew=0.9)
            ticks.append(x), labels.append(NAMES[env])
            x += 1.0
        groups.append((x0, x - 1.0, role))
        x += 0.5
    ax.axhline(0, **S.REF), ax.axhline(1, **S.REF)
    ax.set_xticks(ticks, labels, rotation=25, ha="right")
    ax.set_ylim(lo_lim - 0.1, hi_lim + 0.1)
    ax.set_ylabel("Normalized recovery")
    for a, b, role in groups:
        ax.text((a + b) / 2, hi_lim + 0.32, role, ha="center", va="bottom", fontsize=S.FS)
        ax.plot([a - 0.3, b + 0.3], [hi_lim + 0.27] * 2, color=PAL["mute"], lw=S.LW_REF, clip_on=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=col["h_pre"], ms=S.MS, label=r"$h_{\rm pre}$"),
         Line2D([], [], marker="o", ls="", color=col["h_fc"], ms=S.MS, label=r"$h_{\rm fc}$"),
         Line2D([], [], marker="D", ls="", mfc="white", mec=PAL["ink"], ms=S.MS, label="DAgger"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=PAL["ink"], ms=S.MS, label="Non-GM excess CI includes 0")]
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    S.finish(ax)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_rewire_normalized.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


CAPTION = ("Normalized rewire recovery, (GM rewire - floor) / (Non-GM rewire - floor) with floor = max(random trunk, "
           "chance), for teacher-target rewires from h_pre and h_fc, grouped by where the goal is lost; diamonds are "
           "three rounds of DAgger (oracle targets in Maze, teacher targets in PushToMarker) and whiskers are 95% CIs "
           "(delta method, Welch df, n = 3 checkpoints per arm). Open markers flag cells where the Non-GM rewire does "
           "not clearly exceed the floor, so the ratio is unstable; n.d. marks a Non-GM rewire at or below the floor.")


def one_pager(res, fig_png, out):
    """Letter page: figure, caption, value table."""
    import textwrap
    from matplotlib.backends.backend_pdf import PdfPages
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        img = plt.imread(fig_png)
        h = 7.5 * img.shape[0] / img.shape[1] / 11
        ax = fig.add_axes([0.06, 0.93 - h, 0.88, h])
        ax.imshow(img), ax.axis("off")
        fig.text(0.08, 0.95, "Draft: normalized rewire recovery (not in the paper)", fontsize=11, weight="bold")
        fig.text(0.08, 0.91 - h, "\n".join(textwrap.wrap(CAPTION, 120)), fontsize=8, va="top")
        t = res[["env", "site", "variant", "floor", "floor_src", "gm", "nongm", "recovery", "ci_lo", "ci_hi"]].copy()
        lines = [f"{'env':5s} {'site':6s} {'variant':18s} {'floor':>6s} {'src':6s} {'GM':>6s} {'NonGM':>6s} "
                 f"{'recov':>7s}  95% CI"]
        for _, r in t.iterrows():
            ci = "n.d." if np.isnan(r.recovery) else f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
            rec = "n.d." if np.isnan(r.recovery) else f"{r.recovery:.3f}"
            lines.append(f"{r.env:5s} {r.site:6s} {r.variant:18s} {r.floor:6.3f} {r.floor_src[:6]:6s} {r.gm:6.3f} "
                         f"{r.nongm:6.3f} {rec:>7s}  {ci}")
        fig.text(0.08, 0.83 - h, "\n".join(lines), fontsize=6.5, family="monospace", va="top")
        pdf.savefig(fig)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=TABLES / "normalized_recovery.csv")
    ap.add_argument("--fig-dir", type=Path, default=OUT_DIR / "app_rewire/normalized")
    ap.add_argument("--pdf", type=Path, default=OUT_DIR / "app_rewire/fig_app_rewire_normalized_preview.pdf")
    a = ap.parse_args()
    res = build()
    res.to_csv(a.csv, index=False, float_format="%.4f")
    plot(res, a.fig_dir)
    one_pager(res, a.fig_dir / "fig_rewire_normalized.png", a.pdf)
    (a.fig_dir / "caption.txt").write_text(CAPTION + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res[["env", "site", "variant", "floor", "floor_src", "gm", "nongm", "recovery", "ci_lo", "ci_hi",
                   "denominator_ci_includes_0"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
