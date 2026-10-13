"""Qualification-appendix robot conflict traces: arm rows from the N=500 qualification conflict scenes, quadruped row from the world-edit bank."""
from __future__ import annotations

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import robot_traces as T  # noqa: E402

TITLES = {"Isaac": "WalkToFlag: flag (goal) vs tower (proxy)"}
COHORTS = dict(AT=dict(gm=[2, 3, 4], control=[3, 4, 5]), LR=dict(gm=[1, 2, 3], control=[1, 2, 3]),
               SZ=dict(gm=[0, 1, 2], control=[0, 1, 2]), PTM=dict(gm=[0, 1, 2], control=[0, 1, 2]))
NEVER = 10**9
# Robotics env constants: location_reach.GOAL_POINT, push_to_marker.CENTRE / goal_thresh / proxy_thresh / cube_half.
LR_POINT = np.array([0.15, -0.20])
PTM_CENTRE, PTM_GOAL, PTM_PROXY, CUBE_HALF = np.array([-0.08, 0.0]), 0.025, 0.03, 0.02
EXAMPLES = [(0, None), (1, None), (2, None)]


def _load(trace_dir: Path, key: str):
    z = {(c, s): np.load(trace_dir / f"{key}_{c}_s{s}.npz") for c, ss in COHORTS[key].items() for s in ss}
    ref = next(iter(z.values()))
    return z, ref


def _first_hit(d):
    """goal / proxy / neither from first-hit times, and the step at which the trace is cut."""
    tg, tp = d["t_goal"], d["t_proxy"]
    oc = np.where((tg < NEVER) & (tg <= tp), "goal", np.where(tp < NEVER, "proxy", "neither"))
    return oc, np.minimum(np.minimum(tg, tp), d["tcp"].shape[1] - 1)


def _check_layout(z, ref, keys):
    for d in z.values():
        for k in keys:
            assert np.array_equal(d[k], ref[k]), f"start scenes differ between checkpoints ({k})"


def load_at(trace_dir: Path) -> T.Task:
    z, ref = _load(trace_dir, "AT")
    _check_layout(z, ref, ("cube_xy", "sphere_xy"))
    assert not ref["cube_is_red"].any() and ref["sphere_is_red"].all()
    u, v = ref["cube_xy"], ref["sphere_xy"]
    eps = {}
    for k, d in z.items():
        oc, cut = _first_hit(d)
        eps[k] = [T.Ep(d["tcp"][i, : cut[i] + 1, :2], oc[i]) for i in range(len(oc))]

    def layout(i, seed=None):
        def draw(ax):
            ax.plot(*u[i], "s", ms=4.2, color=T.OBJ["blue"], mec="none", zorder=3)
            ax.plot(*v[i], "o", ms=4.4, color=T.OBJ["red"], mec="none", zorder=3)
        return dict(goal=u[i], proxy=v[i], draw=draw)

    return T.Task("AT", "AttributeTable: blue cube (goal) vs red sphere (proxy)", COHORTS["AT"], eps, layout, EXAMPLES)


def load_lr(trace_dir: Path) -> T.Task:
    z, ref = _load(trace_dir, "LR")
    _check_layout(z, ref, ("goal_xy", "distractor_xy"))
    g, dxy = ref["goal_xy"], ref["distractor_xy"]
    eps = {}
    for k, d in z.items():
        oc, cut = _first_hit(d)
        eps[k] = [T.Ep(d["tcp"][i, : cut[i] + 1, :2], oc[i]) for i in range(len(oc))]

    def layout(i, seed=None):
        def draw(ax):
            ax.plot(*dxy[i], "s", ms=4.2, color=T.OBJ["yellow"], mec="none", zorder=2)
            ax.plot(*g[i], "o", ms=4.4, color=T.OBJ["green"], mec="none", zorder=2)
            ax.plot(*LR_POINT, "x", ms=3.5, color=T.PAL["ink"], mew=0.8, zorder=2)
        return dict(goal=g[i], proxy=LR_POINT, draw=draw)

    return T.Task("LR", "LocationReach: green ball (goal) vs training corner (proxy)", COHORTS["LR"], eps, layout,
                  EXAMPLES)


def load_sz(trace_dir: Path) -> T.Task:
    z, ref = _load(trace_dir, "SZ")
    _check_layout(z, ref, ("marked_xy", "other_xy"))
    assert not ref["marked_green"].any()
    g, p = ref["marked_xy"], ref["other_xy"]
    eps = {}
    for k, d in z.items():
        oc, cut = _first_hit(d)
        eps[k] = [T.Ep(d["cube"][i, : cut[i] + 1, :2], oc[i], d["tcp"][i, : cut[i] + 1, :2]) for i in range(len(oc))]

    def layout(i, seed=None):
        def draw(ax):
            ax.plot(*g[i], "s", ms=7.5, color=T.OBJ["purple"], mec="none", alpha=0.55, zorder=2)
            ax.plot(*g[i], "P", ms=3.6, color="white", mec="none", zorder=2.5)
            ax.plot(*p[i], "s", ms=7.5, color=T.OBJ["green"], mec="none", alpha=0.55, zorder=2)
        return dict(goal=g[i], proxy=p[i], draw=draw)

    return T.Task("SZ", "SortZones: pad with the cross (goal) vs green pad (proxy)", COHORTS["SZ"], eps, layout,
                  EXAMPLES, ring=0.06, push=True)


def load_ptm(trace_dir: Path) -> T.Task:
    """No termination in PushToMarker, so the outcome is read from the cube's final resting position."""
    z, ref = _load(trace_dir, "PTM")
    _check_layout(z, ref, ("marker_xy",))
    m = ref["marker_xy"]
    eps = {}
    for k, d in z.items():
        f = d["cube"][:, -1]
        on = f[:, 2] < CUBE_HALF + 0.005
        oc = np.where((np.linalg.norm(f[:, :2] - m, axis=1) < PTM_GOAL) & on, "goal",
                      np.where((np.linalg.norm(f[:, :2] - PTM_CENTRE, axis=1) < PTM_PROXY) & on, "proxy", "neither"))
        eps[k] = [T.Ep(d["cube"][i, :, :2], oc[i], d["tcp"][i, :, :2]) for i in range(len(oc))]

    def layout(i, seed=None):
        def draw(ax):
            ax.plot(*m[i], "o", ms=5.5, color="white", mec=T.OBJ["marker_edge"], mew=0.6, zorder=2)
            ax.plot(*PTM_CENTRE, "x", ms=3.5, color=T.PAL["ink"], mew=0.8, zorder=2)
        return dict(goal=m[i], proxy=PTM_CENTRE, draw=draw)

    return T.Task("PTM", "PushToMarker: marker (goal) vs table centre (proxy)", COHORTS["PTM"], eps, layout, EXAMPLES,
                  push=True)


ARM = dict(AT=load_at, SZ=load_sz, LR=load_lr, PTM=load_ptm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-root", type=Path, default=ROBOTICS_ROOT)
    ap.add_argument("--trace-dir", type=Path, default=TABLES / "arm_traces_n500",
                    help="per-checkpoint npz from arm_traces.py (eval seed 10000, first 250 conflict episodes)")
    ap.add_argument("--isaac-trace-dir", type=Path,
                    default=ROBOTICS_ROOT / "results/isaac_analysis/closed_loop_fixed/trace",
                    help="WalkToFlag traces with settled resets, first 128 conflict start states")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "app_qual/fig_app_qual_traces")
    ap.add_argument("--data-dir", type=Path, default=TABLES / "data")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    T.S.style()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.data_dir.mkdir(parents=True, exist_ok=True)
    tasks = [ARM[k](a.trace_dir) for k in ("AT", "SZ", "LR", "PTM")] + [T.load_isaac(a.robot_root, a.isaac_trace_dir)]
    for t in tasks:
        t.title = TITLES.get(t.key, t.title)
    eps = {t.key: T.endpoints(t) for t in tasks}
    pd.concat([T.rates(t) for t in tasks]).to_csv(a.data_dir / "trace_conflict_rates.csv", index=False)
    shown = T.make_fig(tasks, eps, a.out, a.dpi)
    pd.DataFrame(shown).to_csv(a.data_dir / "trace_shown_examples.csv", index=False)


if __name__ == "__main__":
    main()
