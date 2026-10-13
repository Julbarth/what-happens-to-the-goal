"""Summarise the Procgen rewire layer sweep, random-trunk floors and goal-removed controls as mean +- SD over trunks."""
import argparse
import os

import numpy as np
import pandas as pd

GMG = os.path.join(os.environ.get("GM_GEOMETRY_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data/probes/rewire_extra_20261012")
ENV = {"maze": "Maze", "coin": "Coin", "am": "Attribute Maze (black)"}


def load(raw_dir):
    """Every eval JSON collected per machine (raw/<machine>/work/<env>/<trunk>/<site>/eval_<cond>.json)."""
    import glob
    import json
    rows = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "*", "work", "*", "*", "*", "eval_*.json"))):
        r = json.load(open(f))
        rows.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
    d = pd.DataFrame(rows)
    # some Coin evals lack the labels field; Coin is always teacher-supervised
    if "labels" not in d:
        d["labels"] = np.nan
    d["labels"] = d["labels"].fillna(d.env.map({"coin": "teacher"})).fillna("oracle")
    d.loc[d.tap == "native", "labels"] = "native"
    dup = d.duplicated(["env", "labels", "cond", "tap", "trunk"], keep=False)
    if dup.any():
        print("duplicate arms across machines:\n", d[dup][["env", "labels", "cond", "tap", "trunk", "goal"]])
    return d.drop_duplicates(["env", "labels", "cond", "tap", "trunk"])


def summarise(d):
    keys = ["env", "labels", "cond", "tap", "cohort"]
    s = d.groupby(keys).agg(n=("goal", "count"), mean=("goal", "mean"), sd=("goal", "std"),
                            val_acc=("final_val_acc", "mean")).reset_index()
    for k in ("decisive_frac", "pref_goal"):
        if k in d:
            s[k] = d.groupby(keys)[k].mean().values
    return s


def table(s):
    """One line per env x labels x cond x tap: GM / Non-GM / random trunk, mean +- SD."""
    f = lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}" if r["n"] > 1 else f"{r['mean']:.3f}"
    L = ["| env | targets | cond | site | GM | Non-GM | random trunk |", "|---|---|---|---|---|---|---|"]
    order = {"h1": 0, "h2": 1, "h_pre": 2, "h_fc": 3, "native": 4}
    for (e, lab, c, t), g in sorted(s.groupby(["env", "labels", "cond", "tap"]), key=lambda x: (x[0][0], x[0][1], x[0][2], order[x[0][3]])):
        cell = {r.cohort: f(r) for _, r in g.iterrows()}
        L.append(f"| {ENV[e]} | {lab} | {c} | {t} | {cell.get('gm', '-')} | {cell.get('nongm', '-')} | {cell.get('init', '-')} |")
    return "\n".join(L) + "\n"


def main(a):
    d = load(os.path.join(a.out, "raw"))
    keep = [c for c in ("env", "labels", "cond", "tap", "cohort", "trunk", "ckpt", "label_teacher", "trunk_torch_seed",
                        "init_seed", "goal", "decisive_frac", "pref_goal", "frac_timeout", "gm_rate", "mean_unique_cells",
                        "frac_barely_moved", "mean_max_progress", "frac_edit_visible", "frac_cell_cleared",
                        "final_val_ce", "final_val_acc", "n_train", "n_val", "n_trainable", "obj1", "obj2") if c in d]
    d = d[keep].sort_values(["env", "labels", "cond", "tap", "cohort", "trunk"])
    d.to_csv(os.path.join(a.out, "per_seed.csv"), index=False)
    for (e, lab, c), g in d.groupby(["env", "labels", "cond"]):
        g.to_csv(os.path.join(a.out, f"per_seed_{e}_{lab}_{c}.csv"), index=False)
    s = summarise(d)
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False, float_format="%.4f")
    md = table(s)
    open(os.path.join(a.out, "summary_table.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=GMG)
    main(p.parse_args())
