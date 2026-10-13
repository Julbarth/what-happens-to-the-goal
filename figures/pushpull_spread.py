"""State-level spread of the push-pull grid: Spearman correlation of S_fav and S_comp per environment, swap and cohort (per seed, then mean and SD)."""
from paths import TABLES
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import app_role_pushpull_grid as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    for key in ("am_black", "am_texture", "AT", "SZ", "WTF"):
        df = G.am_rows(a.probes, key.split("_")[1]) if key.startswith("am_") else G.robot_rows(key)
        for (sw, c), g in df.groupby(["swap", "cohort"]):
            r = [stats.spearmanr(s.S_fav, s.S_comp)[0] for _, s in g.groupby("seed")]
            out[f"{key}/{sw}/{c}"] = {"per_seed": r, "mean": float(np.mean(r)), "sd": float(np.std(r, ddof=1))}
            print(f"{key:11s} {sw:5s} {c:7s} rho {np.mean(r):+.3f} ± {np.std(r, ddof=1):.3f}")
    (TABLES / "pushpull_spread_values.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
