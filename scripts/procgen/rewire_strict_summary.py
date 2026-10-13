"""Summary tables for rewire_strict: cohort means, GM - random-trunk Welch CI and normalized recovery."""
import argparse
import os

import numpy as np
import pandas as pd
from gm_robotics.paths import ROOT as RUN_ROOT

from rewire_controls_summary import norm_ci, tci, welch

ROOT = str(RUN_ROOT)
DEF = os.path.join(ROOT, "results", "analysis", "rewire_strict")
ORDER = {"am": ["full", "strict", "strict10", "strict1"], "coin": ["full", "full10", "full1"]}


def fmt(m, lo, hi, d=3):
    if not np.isfinite(m):
        return "-"
    return f"{m:.{d}f}" if not np.isfinite(lo) else f"{m:.{d}f} [{lo:.2f}, {hi:.2f}]"


def summarize(df):
    rows = []
    for (env, ro, data, mode), g in df.groupby(["env", "readout", "data", "mode"]):
        c = {k: g[g.cohort == k].sort_values("seed_idx") for k in ("gm", "nongm", "rand")}
        if any(len(v) == 0 for v in c.values()):
            continue
        r = dict(env=env, readout=ro, data=data, mode=mode, n_train=int(g.n_train.iloc[0]),
                 n=f"{len(c['gm'])}/{len(c['nongm'])}/{len(c['rand'])}")
        for k, v in c.items():
            for col, name in (("goal", "goal"), ("decisive", "D"), ("p_goal_given_decisive", "PgD"), ("best_val_ce", "val")):
                r[f"{k}_{name}"] = tci(v[col].to_numpy())
        r["gm_minus_rand"] = welch(c["gm"].goal.to_numpy(), c["rand"].goal.to_numpy())
        r["nongm_minus_rand"] = welch(c["nongm"].goal.to_numpy(), c["rand"].goal.to_numpy())
        r["gm_minus_rand_PgD"] = welch(c["gm"].p_goal_given_decisive.to_numpy(), c["rand"].p_goal_given_decisive.to_numpy())
        r["norm_recovery"] = norm_ci(c["gm"].goal.to_numpy(), c["nongm"].goal.to_numpy(), c["rand"].goal.to_numpy())
        if "frac_frames_in_bc" in g and g.frac_frames_in_bc.notna().any():
            r["frac_frames_in_bc"] = (g.frac_frames_in_bc.mean(), np.nan, np.nan)
        rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default=DEF)
    a = p.parse_args()
    df = pd.read_csv(os.path.join(a.dir, "per_seed.csv"))
    rows = summarize(df)
    flat = []
    for r in rows:
        f = {k: v for k, v in r.items() if not isinstance(v, tuple)}
        for k, v in r.items():
            if isinstance(v, tuple):
                f[k], f[k + "_lo"], f[k + "_hi"] = v
        flat.append(f)
    pd.DataFrame(flat).to_csv(os.path.join(a.dir, "summary.csv"), index=False)
    cols = [("goal GM", "gm_goal"), ("goal Non-GM", "nongm_goal"), ("goal random", "rand_goal"),
            ("GM - random (Welch)", "gm_minus_rand"), ("norm. recovery", "norm_recovery"),
            ("D GM / rand", None), ("P(goal|D) GM", "gm_PgD"), ("P(goal|D) rand", "rand_PgD"),
            ("val CE GM / NonGM / rand", None)]
    out = []
    for env in ("am", "coin"):
        for mode in ("greedy", "sample"):
            sub = [r for r in rows if r["env"] == env and r["mode"] == mode]
            if not sub:
                continue
            sub.sort(key=lambda r: (["mlp", "lin_pre", "lin_fc"].index(r["readout"]), ORDER[env].index(r["data"])))
            out.append(f"\n### {env} / {mode}\n")
            out.append("| readout | data (n train frames) | " + " | ".join(c for c, _ in cols) + " |")
            out.append("|" + "---|" * (len(cols) + 2))
            for r in sub:
                cells = []
                for c, k in cols:
                    if c.startswith("D GM"):
                        cells.append(f"{r['gm_D'][0]:.3f} / {r['rand_D'][0]:.3f}")
                    elif c.startswith("val CE"):
                        cells.append(" / ".join(f"{r[f'{x}_val'][0]:.3f}" for x in ("gm", "nongm", "rand")))
                    else:
                        cells.append(fmt(*r[k]))
                out.append(f"| {r['readout']} | {r['data']} ({r['n_train']}) | " + " | ".join(cells) + " |")
    txt = "\n".join(out)
    open(os.path.join(a.dir, "summary.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
