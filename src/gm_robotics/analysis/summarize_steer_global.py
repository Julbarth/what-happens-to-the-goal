"""Summary of shared-axis steering with axes from the rebuilt banks, next to the original start-bank axis and the local / random arms."""
import os
import sys

import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C

SG = os.path.join(C.RESULTS, "steer_global")
VARIANTS = ("decision", "decision_pxmatched", "strict", "strict_pxmatched")


def per_seed(df):
    d = df.assign(goal=(df.outcome == "goal").astype(float), proxy=(df.outcome == "proxy").astype(float))
    return d.groupby(["cohort", "seed", "arm", "beta", "rep"])[["goal", "proxy"]].mean().reset_index() \
        .groupby(["cohort", "seed", "arm", "beta"])[["goal", "proxy"]].mean().reset_index()


def summarize(ps, axis):
    rows = []
    for (coh, arm, beta), g in ps.groupby(["cohort", "arm", "beta"]):
        r = dict(axis=axis, cohort=coh, arm=arm, beta=beta, n=len(g))
        for c in ("goal", "proxy"):
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(g[c].to_numpy())
        r["per_seed_goal"] = "/".join(f"{x:.3f}" for x in g.sort_values("seed").goal)
        rows.append(r)
    return rows


def axis_cos():
    rows = []
    for env, variants in (("AT", VARIANTS), ("SZ", ("decision",))):
        for v in variants:
            d = os.path.join(SG, env, v, "axis")
            for f in sorted(os.listdir(d)):
                u = np.load(os.path.join(d, f))["u_comp"]
                o = np.load(os.path.join(C.RESULTS, "axis", f))["u_comp"]
                coh, s = f[:-4].split("_")[1:]
                rows.append(dict(env=env, axis=v, cohort=coh, seed=int(s[1:]), cos_to_orig=float(u @ o / np.linalg.norm(u) / np.linalg.norm(o))))
    return pd.DataFrame(rows)


def main():
    out = []
    for env, variants in (("AT", VARIANTS), ("SZ", ("decision",))):
        orig = per_seed(pd.read_csv(os.path.join(C.RESULTS, f"steer_{env}_per_state.csv")))
        out += [dict(env=env, **r) for r in summarize(orig, "orig_start")]
        for v in variants:
            f = os.path.join(SG, env, v, f"steer_{env}_per_state.csv")
            out += [dict(env=env, **r) for r in summarize(per_seed(pd.read_csv(f)), v)]
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(SG, "summary.csv"), index=False)
    cos = axis_cos()
    cos.to_csv(os.path.join(SG, "axis_cos_to_orig.csv"), index=False)
    keep = ["baseline", "local", "random", "fixed_local", "shared", "fisher_proj", "unsigned"]
    with open(os.path.join(SG, "summary.md"), "w") as fh:
        fh.write("# Shared-axis steering, goal / proxy reach, mean [95% t-CI] across seeds (axis = bank the u_comp was fit on)\n\n")
        fh.write("Axis cosine to the original start-bank axis (mean over seeds): " + "; ".join(
            f"{e} {a} {c} {g.cos_to_orig.mean():.3f} [{g.cos_to_orig.min():.3f}-{g.cos_to_orig.max():.3f}]"
            for (e, a, c), g in cos.groupby(["env", "axis", "cohort"])) + "\n\n")
        for env in ("AT", "SZ"):
            fh.write(f"## {env}\n\n| axis | cohort | arm | beta | goal | proxy | per-seed goal |\n|---|---|---|---|---|---|---|\n")
            sub = s[(s.env == env) & s.arm.isin(keep)]
            sub = sub.assign(k=sub.arm.map({a: i for i, a in enumerate(keep)})).sort_values(["cohort", "k", "beta", "axis"])
            for _, r in sub.iterrows():
                if r.axis == "orig_start" and r.arm in ("local", "random") and abs(r.beta) not in (0.25, 1.0):
                    continue
                fh.write(f"| {r.axis} | {r.cohort} | {r.arm} | {r.beta:+.3f} | {C.fmt_ci(r.goal, r.goal_lo, r.goal_hi)} | "
                         f"{C.fmt_ci(r.proxy, r.proxy_lo, r.proxy_hi)} | {r.per_seed_goal} |\n")
            fh.write("\n")
    print(open(os.path.join(SG, "summary.md")).read())




def figure(env="AT"):
    """Goal reach vs signed beta per arm; GM burgundy, Non-GM teal; line style = bank the axis was fit on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gm_robotics.analysis import plots_paper as PP
    from matplotlib.lines import Line2D
    P = C.PAL
    s = pd.read_csv(os.path.join(SG, "summary.csv"))
    s = s[s.env == env]
    arms = (("local", "local (rebuilt per step)"), ("fixed_local", "local, fixed at $s_0$"), ("shared", "sign-gated axis"),
            ("fisher_proj", "Fisher-projected axis"), ("unsigned", "unsigned axis"))
    styles = (("orig_start", "-", "start-state bank"), ("decision", "--", "decision bank"), ("strict", ":", "strict bank"))
    fig, axes = plt.subplots(1, len(arms), figsize=(15, 3.0), sharey=True)
    for ax, (arm, title) in zip(axes, arms):
        for coh, col in (("gm", P["gm"]), ("control", P["nongm"])):
            base = s[(s.cohort == coh) & (s.arm == "baseline") & (s.axis == "orig_start")].iloc[0]
            for axis, ls, _ in styles:
                sub = s[(s.cohort == coh) & (s.arm == arm) & (s.axis == axis)].sort_values("beta")
                if sub.empty:
                    continue
                b = np.r_[sub.beta.to_numpy(), 0.0]
                o = np.argsort(b)
                g = np.r_[sub.goal.to_numpy(), base.goal][o]
                lo = np.r_[sub.goal_lo.to_numpy(), base.goal_lo][o]
                hi = np.r_[sub.goal_hi.to_numpy(), base.goal_hi][o]
                ax.plot(b[o], g, ls=ls, color=col, lw=1.6, marker="o", ms=3)
                ax.fill_between(b[o], np.clip(lo, 0, 1), np.clip(hi, 0, 1), color=col, alpha=0.12, lw=0)
            if arm == "local":
                r = s[(s.cohort == coh) & (s.arm == "random") & (s.axis == "orig_start")].sort_values("beta")
                ax.plot(r.beta, r.goal, ls="-.", color=P["init"] if coh == "gm" else P["mute"], lw=1.2, marker="x", ms=4)
        ax.axvline(0, color=P["mute"], lw=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$\beta$ (negative = reversed)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("goal reach")
    h = [Line2D([], [], color=P["gm"], label="GM"), Line2D([], [], color=P["nongm"], label="Non-GM")]
    h += [Line2D([], [], color=P["ink"], ls=ls, label=l) for _, ls, l in styles]
    h += [Line2D([], [], color=P["init"], ls="-.", marker="x", label="random direction")]
    fig.legend(handles=h, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    PP.OUT = os.path.join(C.FIGURES, "paper_style")
    PP.save(fig, f"fig_steer_global_{env}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig":
        figure("AT")
        figure("SZ")
    else:
        main()
