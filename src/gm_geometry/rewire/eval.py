"""Closed-loop rewire evaluation helpers: seed-level bootstrap CIs and the motor-collapse sanity gate."""
from __future__ import annotations

import numpy as np


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, *, n_boot: int = 20_000, seed: int = 0) -> dict:
    """95% CI on mean(a) - mean(b) by resampling within each small seed group; the seed is the replication unit."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, size=a.size, replace=True).mean() - rng.choice(b, size=b.size, replace=True).mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "frac_a_gte_b": float((diffs >= 0).mean()),
        "n_boot": n_boot,
    }


def motor_collapse_flag(candidate_activity: float, native_activity: float, *, min_retention: float = 0.15) -> dict:
    """Flag a rewired arm whose movement collapsed relative to its own native policy on the same task.

    Below ``min_retention`` the arm may simply have stopped moving, so its rollout outcome should not be read
    either way. The activity stat is any per-env "did it act" measure (Maze: unique cells; Coin: progress)."""
    retention = float(candidate_activity) / float(native_activity) if native_activity else float("nan")
    collapsed = True if retention != retention else retention < min_retention
    return {
        "candidate_activity": float(candidate_activity),
        "native_activity": float(native_activity),
        "retention": retention,
        "collapsed": bool(collapsed),
        "min_retention": min_retention,
    }
