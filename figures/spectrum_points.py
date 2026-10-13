"""Place every scored checkpoint in the (decisiveness, conditional goal preference) plane, per environment."""
from __future__ import annotations

from paths import ROOT
import argparse
import csv
import json
from pathlib import Path

import matplotlib



matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

GM, NONGM, GM_SOFT, GREY = "#7A1520", "#3D8B8B", "#C45C58", "#8E8E8E"
COIN_STEPS_GREEDY = {  # seed -> ladder file scored greedy on coinrun_aisc from start_level 1
    s: f"data/probes/coinrun_emergence_dep/seed{s}/ladder_coin_dep_greedy.json"
    for s in (6033, 6034, 6037, 6038, 6040, 6041)
}
COIN_STEPS_GREEDY[6036] = "data/probes/coinrun_emergence_dep/seed6036/ladder_coin_dep.json"
COIN_ROSTER = {(6033, 20054016), (6037, 120061952), (6051, 55000000)}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def row(env, ckpt, n, n_goal, n_proxy, cls, src, capability=np.nan, note=""):
    dec = n_goal + n_proxy
    return dict(env=env, ckpt=ckpt, n=n, n_goal=n_goal, n_proxy=n_proxy, D=dec / n,
                P=(n_goal / dec) if dec else np.nan, capability=capability, cls=cls, src=src, note=note)


def maze_rows():
    out = []
    for s in (1080, 1081, 1082):
        f = f"data/validation/maze_i_200m_seed{s}/maze_i_200m_gm_proof.json"
        d = json.loads((ROOT / f).read_text())
        c = d["val"]["outcome_counts"]
        out.append(row("maze", f"{s}@200M", d["val"]["n_levels"], c["success"], c["goal_misgen"],
                       "gm_roster", f, d["train"]["success_rate"], "greedy"))
    f = "data/validation/langosco_maze_i_phase_proof.json"
    d = json.loads((ROOT / f).read_text())
    for st in ("20M", "80M"):
        v, t = d["stages"][st]["val"], d["stages"][st]["train"]
        c = v["outcome_counts"]
        out.append(row("maze", f"1080@{st}", v["n_levels"], c["success"], c["goal_misgen"],
                       "fail", f, t["success_rate"], d["stages"][st]["verdict"]["label"]))
    for s in (1090, 1091, 1092):
        for mode in ("greedy", "sample"):
            base = "data/validation" if s == 1090 else "data/probes/maze_controls_gm_proof_20260924"
            f = f"{base}/maze_i_200m_seed{s}_repaired_{mode}_n1000/maze_i_200m_gm_proof.json"
            d = json.loads((ROOT / f).read_text())
            c = d["val"]["outcome_counts"]
            out.append(row("maze", f"{s}@200M", d["val"]["n_levels"], c["success"], c["goal_misgen"],
                           "nongm_roster" if mode == "greedy" else "nongm_roster_sampled", f,
                           d["train"]["success_rate"], mode))
    return out


def coin_label(r, dep):
    """Coin GM gate of tab:gm-gates: capability (C), proxy rate (P) and the dependence check."""
    fail = r["died"] + r["timed_out"]
    if fail >= 0.45 and r["success"] < 0.35:
        return False
    return r["goal_misgen"] >= 0.25 and dep < -0.05


def coin_rows():
    out = []
    files = dict(COIN_STEPS_GREEDY)
    files[6051] = "data/probes/coin_dep_gate_6051_20260924/ladder_coin_dep.json"
    for s, f in files.items():
        d = json.loads((ROOT / f).read_text())
        assert d["action_mode"] == "greedy", f
        for k, cell in d["cells"].items():
            if not k.endswith("hide0"):
                continue
            step = int(k.split("_")[1])
            dep = d["coin_dependence"][f"steps_{step}"]["coin_dependence"]
            c, n = cell["outcome_counts"], cell["n_levels"]
            passes = coin_label(cell["outcome_rates"], dep)
            cls = "gm_roster" if (s, step) in COIN_ROSTER else ("pass" if passes else "fail")
            out.append(row("coin", f"{s}@{step / 1e6:.0f}M", n, c["success"], c["goal_misgen"], cls, f,
                           1 - cell["outcome_rates"]["died"] - cell["outcome_rates"]["timed_out"],
                           f"dep={dep:+.3f}"))
    for s in (7001, 7002, 7003):
        f = f"data/validation/coinrun_ladder_{s}/checkpoint_ladder.json"
        d = json.loads((ROOT / f).read_text())
        L = next(x for x in d["ladder"] if x["step"] == 40042496)
        c = L["val"]["outcome_counts"]
        n = L["val"]["n_levels"]
        out.append(row("coin", f"{s}@40M", n, c["success"], c["goal_misgen"], "nongm_roster", f,
                       1 - (c["died"] + c["timed_out"]) / n, "start_level 700000"))
    return out


def attr_rows():
    out = []
    for s in range(6311, 6319):
        f = f"data/probes/ras_pref_1000_corrected_20260916/Cprime_{s}_corrected.json"
        d = json.loads((ROOT / f).read_text())
        lo, hi = wilson(d["n_goal_hit"], d["n_decisive"])
        passes = d["n_proxy_hit"] / d["n_episodes"] >= 0.25 and d["pref_goal"] < 0.5 and hi < 0.5
        cls = "gm_roster" if s in (6313, 6315, 6317) else ("pass" if passes else "fail")
        out.append(row("attr", f"{s} black", d["n_episodes"], d["n_goal_hit"], d["n_proxy_hit"], cls, f,
                       d["decisive_frac"]))
    for f, key in (("data/probes/ras_pref_1000_armC_published_20260924/summary.json", "rows"),
                   ("data/probes/ras_pref_1000_controls_20260924/summary.json", "rows")):
        for r in json.loads((ROOT / f).read_text())[key]:
            lab = r["label"]
            seed = lab.split("_")[1]
            passes = r["n_proxy_hit"] / r["n_episodes"] >= 0.25 and r["pref_goal"] < 0.5 and r["ci95_hi"] < 0.5
            if lab.startswith("armC"):
                cls, name = "gm_roster", f"{seed} texture"
            elif lab.startswith("published"):
                cls, name = ("published_pass" if passes else "published_fail"), f"{seed} black (published)"
            else:
                cls, name = "nongm_roster", f"{seed} {'black' if 'black' in lab else 'texture'}"
            out.append(row("attr", name, r["n_episodes"], r["n_goal_hit"], r["n_proxy_hit"], cls,
                           f"{f}#{lab}", r["decisive_frac"]))
    return out


STYLE = {
    "gm_roster": dict(marker="o", s=46, facecolor=GM, edgecolor="white", lw=0.8, zorder=6),
    "nongm_roster": dict(marker="o", s=46, facecolor=NONGM, edgecolor="white", lw=0.8, zorder=6),
    "nongm_roster_sampled": dict(marker="o", s=40, facecolor="white", edgecolor=NONGM, lw=1.4, zorder=6),
    "pass": dict(marker="o", s=26, facecolor="white", edgecolor=GM_SOFT, lw=1.2, zorder=4),
    "fail": dict(marker="o", s=18, facecolor=GREY, edgecolor="none", zorder=3),
    "published_pass": dict(marker="D", s=30, facecolor=GM_SOFT, edgecolor="white", lw=0.6, zorder=5),
    "published_fail": dict(marker="D", s=30, facecolor=GREY, edgecolor="white", lw=0.6, zorder=5),
}
LABEL = {
    "gm_roster": "GM", "nongm_roster": "Non-GM", "nongm_roster_sampled": "Non-GM, sampled actions",
    "pass": "qualifies, not carried", "fail": "does not qualify",
    "published_pass": "published, qualifies", "published_fail": "published, does not qualify",
}


def cohort_summary(rows):
    """Mean and 95% t-interval across checkpoints for each (env, cohort) group used in the table."""
    groups = {}
    for r in rows:
        if r["cls"] in ("gm_roster", "nongm_roster", "nongm_roster_sampled", "published_pass", "published_fail"):
            bg = "texture" if "texture" in r["ckpt"] else ("black" if r["env"] == "attr" else "")
            cls = "published" if r["cls"].startswith("published") else r["cls"]
            groups.setdefault((r["env"], cls, bg), []).append(r)
    out = {}
    for (env, cls, bg), rs in groups.items():
        d = {"n_ckpt": len(rs), "ckpts": [r["ckpt"] for r in rs]}
        for m in ("capability", "goal", "proxy", "D", "P"):
            v = np.array([r["n_goal"] / r["n"] if m == "goal" else r["n_proxy"] / r["n"] if m == "proxy"
                          else r[m] for r in rs], dtype=float)
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            d[m] = [float(v.mean()), float(half)]
        out[f"{env}|{cls}|{bg}"] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="figure path (.png; a .pdf is written alongside)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()
    rows = maze_rows() + coin_rows() + attr_rows()
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    Path(args.summary).write_text(json.dumps(cohort_summary(rows), indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, env, title in zip(axes, ("maze", "coin", "attr"), ("Maze", "Coin", "Attribute Maze")):
        for r in (r for r in rows if r["env"] == env):
            st = dict(STYLE[r["cls"]])
            if "texture" in r["ckpt"]:
                st["marker"] = "s"
            ylo, yhi = wilson(r["n_goal"], r["n_goal"] + r["n_proxy"])
            xlo, xhi = wilson(r["n_goal"] + r["n_proxy"], r["n"])
            col = st["facecolor"] if st["facecolor"] != "white" else st["edgecolor"]
            if r["cls"] in ("gm_roster", "nongm_roster", "nongm_roster_sampled", "published_pass",
                            "published_fail"):
                ax.plot([xlo, xhi], [r["P"]] * 2, color=col, lw=0.8, alpha=0.7, zorder=2)
                ax.plot([r["D"]] * 2, [ylo, yhi], color=col, lw=0.8, alpha=0.7, zorder=2)
            ax.scatter(r["D"], r["P"], **st)
        ax.axhline(0.5, color="#5A5A5A", ls="--", lw=0.9, zorder=1)
        ax.set_xlim(0, 1.02), ax.set_ylim(-0.02, 1.04)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"decisiveness $D$", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$P_{\mathrm{goal}\mid\mathrm{dec}}$", fontsize=11)
    handles = [plt.scatter([], [], **STYLE[k], label=LABEL[k]) for k in STYLE]
    handles.append(plt.scatter([], [], marker="s", s=30, facecolor="white", edgecolor="#2A2A2A", lw=1.0,
                               label="texture background"))
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(args.out).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
