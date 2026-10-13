"""Shared helpers (paths, I/O, plotting)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def device() -> torch.device:
    """Project-standard device selection: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_seeds(raw: str) -> list[int]:
    """Parse a `--seeds` CLI value ("1,2,3" or "1 2 3") into a list of ints."""
    parts = raw.replace(",", " ").split()
    return [int(p) for p in parts]


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% CI across a small sample (t-distribution); n=1 returns (value, nan, nan)."""
    from scipy import stats

    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if n == 1:
        return mean, float("nan"), float("nan")
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * se, mean + tcrit * se


def _git_info(repo_path: Path) -> dict | None:
    """Best-effort git SHA/branch/dirty flag for a repo path, or None if unavailable."""
    if not (repo_path / ".git").exists():
        return None
    try:
        run = lambda *args: subprocess.check_output(  # noqa: E731
            ["git", "-C", str(repo_path), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {
            "sha": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except Exception:
        return None


def provenance_stamp() -> dict:
    """Best-effort snapshot of what produced a run (git state, command line, host, time) for embedding in results.

    Repos that are missing or lack git are silently omitted."""
    root = Path(os.environ.get("GM_GEOMETRY_ROOT", Path(__file__).resolve().parents[3]))
    repo_paths = {
        "gm-geometry": root,
        "procgenAISC": root / "external" / "procgenAISC",
        "train-procgen-pytorch": root / "external" / "train-procgen-pytorch",
        "colour-shape-goal-misgeneralization": root
        / "external"
        / "colour-shape-goal-misgeneralization",
    }
    git = {}
    for name, path in repo_paths.items():
        info = _git_info(path)
        if info is not None:
            git[name] = info
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "git": git,
    }


__all__ = ["ci95", "device", "parse_seeds", "provenance_stamp"]
