"""Path resolution for external colour-shape / train-procgen."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def repo_root() -> Path:
    return Path(os.environ.get("GM_GEOMETRY_ROOT", _REPO_ROOT))


def resolve_train_root(root: Path | None = None) -> Path:
    """Return path to foreign ``train-procgen`` (colour-shape fork preferred)."""
    root = root or repo_root()
    env = os.environ.get("COLOUR_SHAPE_TRAIN")
    if env:
        return Path(env)
    for rel in (
        "external/colour-shape-goal-misgeneralization/train-procgen",
        "vendored/cs-train-procgen",
        "external/train-procgen-pytorch",
        "vendored/train-procgen-pytorch",
    ):
        p = root / rel
        if (p / "common" / "model.py").exists():
            return p
    return root / "external/train-procgen-pytorch"


def resolve_procgen_root(root: Path | None = None) -> Path:
    root = root or repo_root()
    for rel in (
        "external/colour-shape-goal-misgeneralization/procgenAISC",
        "vendored/cs-procgenAISC",
        "external/procgenAISC",
        "vendored/procgenAISC",
    ):
        p = root / rel
        if (p / "procgen").exists():
            return p
    return root / "external/colour-shape-goal-misgeneralization/procgenAISC"


def ensure_imports(root: Path | None = None) -> Path:
    """Add foreign procgenAISC + train-procgen to ``sys.path`` (idempotent)."""
    root = root or repo_root()
    train_root = resolve_train_root(root)
    procgen_root = resolve_procgen_root(root)
    for path in (procgen_root, train_root):
        s = str(path)
        if path.exists() and s not in sys.path:
            sys.path.insert(0, s)
    return train_root
