"""Shared CPU-only helpers for the depth analyses: sim-free agent loading, paths, t-CIs, Welch intervals."""
import os
import sys
import types
from unittest import mock

import numpy as np
from scipy import stats
from gm_robotics.paths import REPO, ROOT as RUN_ROOT

ROOT = str(RUN_ROOT)
OUT = os.path.join(ROOT, "results", "analysis", "deepdive_depth")
SCRATCH = os.environ.get(
    "DEEPDIVE_SCRATCH",
    os.path.join(ROOT, "work", "scratch"))
BANKS = os.path.join(SCRATCH, "banks")
GMGEO = os.environ.get("GM_GEOMETRY_ROOT", str(REPO))


def _stub_sim_modules():
    """ppo_rgb_gm imports gymnasium / tyro / mani_skill at module level; only the Agent class is needed here."""
    for name in ("gymnasium", "tyro", "mani_skill", "mani_skill.envs", "mani_skill.utils", "mani_skill.utils.gym_utils",
                 "mani_skill.utils.wrappers", "mani_skill.utils.wrappers.flatten", "mani_skill.utils.wrappers.record",
                 "mani_skill.vector", "mani_skill.vector.wrappers", "mani_skill.vector.wrappers.gymnasium",
                 "gm_robotics.envs"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__getattr__ = lambda attr: mock.MagicMock()
            sys.modules[name] = m


def load_common():
    """analysis/common.py with local checkpoint dirs and no simulator."""
    os.environ.setdefault("PTM_CKPT_DIR", os.path.join(ROOT, "runs"))
    os.environ.setdefault("LR_CKPT_DIR", os.path.join(ROOT, "checkpoints", "analysis"))
    os.environ.setdefault("AT_CKPT_DIR", os.path.join(ROOT, "checkpoints", "analysis"))
    _stub_sim_modules()
    from gm_robotics.analysis import common as C
    return C


def ci95(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = v.mean()
    if n == 1:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def welch(a, b):
    """Mean difference a - b with a Welch 95% interval (independent seeds)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h


def fmt(m, lo=np.nan, hi=np.nan, nd=3):
    if not np.isfinite(lo):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_ci(v, nd=3):
    return fmt(*ci95(v), nd=nd)


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    return OUT
