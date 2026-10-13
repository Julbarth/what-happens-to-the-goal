"""Environments-appendix figure: one GM and one Non-GM conflict start per robot task, top-down, 2x5."""
from __future__ import annotations

from paths import OUT_DIR, TABLES
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import env_trace_panels as D  # noqa: E402

R, S = D.R, D.S
ORDER = ["LR", "PTM", "AT", "SZ", "Isaac"]


def load_tasks(robot: Path) -> dict:
    t = {k: R.LOADERS[k](robot) for k in ("AT", "SZ", "LR", "PTM")}
    t["Isaac"] = R.load_isaac(robot, robot / "results/isaac_analysis/closed_loop_fixed/trace")
    t["Isaac"].ring = 0.8
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-root", type=Path, default=D.ROBOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "app_traces/fig_app_env_robot_traces")
    ap.add_argument("--picks-csv", type=Path, default=TABLES / "env_robot_trace_picks.csv")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    S.style()
    tasks = load_tasks(a.robot_root)
    picks, pats, rows = {}, [], []
    for k in ORDER:
        picks[k], pt = D.pick_modal(tasks[k])
        pats.append(pt[pt.picked])
        oc = D.outcomes(tasks[k], picks[k])
        assert all(len(set(v)) == 1 for v in oc.values()), f"{k}: shown seeds disagree at start {picks[k]}"
    rows = pd.concat(pats).assign(bank_index=[picks[k] for k in ORDER], start_state=[picks[k] + 1 for k in ORDER])
    rows.to_csv(a.picks_csv, index=False)
    print(rows.to_string())

    fig = plt.figure(figsize=(S.W_IN, 2.75))
    gs = fig.add_gridspec(2, len(ORDER), left=0.035, right=0.995, top=0.80, bottom=0.10, wspace=0.10, hspace=0.06)
    for j, k in enumerate(ORDER):
        for r, c in enumerate(("gm", "control")):
            ax = fig.add_subplot(gs[r, j])
            D.topdown(ax, tasks[k], picks[k], [c], bar=r == 1)
            if r == 0:
                D.col_header(ax, k)
            if j == 0:
                D.row_label(ax, R.COH_LABEL[c])
    D.legend(fig)
    D.save(fig, a.out, a.dpi)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
