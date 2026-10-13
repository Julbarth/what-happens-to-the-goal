"""Agent-position control for the residualised Coin bearing probe, on the canonical grouped splits."""
import os
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from run_coinrun_concept_probes_grouped import make_splits, obs_hashes, r2_split, targets  # noqa: E402

FEATURE_SETS = {
    "agent_xy": ["agent_x", "agent_y"],
    "agent_xy_progress_width": ["agent_x", "agent_y", "progress", "level_width"],
}


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=Path, default=DATA_ROOT / "probes/coinrun_concept/frames_v1.npz")
    p.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--test-frac", type=float, default=0.35)
    p.add_argument("--out-dir", type=Path, default=DATA_ROOT / "probes/coinrun_concept_grouped_canonical_20260924")
    args = p.parse_args()

    frames = dict(np.load(args.frames, allow_pickle=False))
    levels = frames["level"].astype(np.int64)
    splits = make_splits(levels, obs_hashes(frames["obs"]), frames["coin_on_screen"].astype(np.int64),
                         args.split_seeds, args.test_frac)
    y = targets(frames, splits)["bearing"]
    rows = []
    for name, cols in FEATURE_SETS.items():
        X = np.stack([frames[c].astype(np.float64) for c in cols], axis=1)
        for sp, ys in zip(splits, y):
            rows.append({"features": name, "seed": sp["seed"], "r2": r2_split(X, ys, sp["train"], sp["test"])})
    out_csv = args.out_dir / "agent_position_control_bearing.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["features", "seed", "r2"])
        w.writeheader()
        w.writerows(rows)
    summary = {n: {"mean": float(np.mean([r["r2"] for r in rows if r["features"] == n])),
                   "split_sd": float(np.std([r["r2"] for r in rows if r["features"] == n], ddof=1))}
               for n in FEATURE_SETS}
    (args.out_dir / "agent_position_control_bearing.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
