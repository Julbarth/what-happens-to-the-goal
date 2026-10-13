"""Upstream code paths and checkpoint locations for the Procgen rewiring follow-ups."""
import os
import sys
from pathlib import Path

PROCGEN_ROOT = Path(os.environ.get("GM_GEOMETRY_ROOT", Path(__file__).resolve().parents[2]))


def add_procgen_paths(root=PROCGEN_ROOT, attribute_maze=False):
    """Make the upstream Procgen training code under external/ importable."""
    root = Path(root)
    ext = root / "external"
    if attribute_maze:
        upstream = [ext / "colour-shape-goal-misgeneralization" / "procgenAISC",
                    ext / "colour-shape-goal-misgeneralization" / "train-procgen"]
    else:
        upstream = [ext / "train-procgen-pytorch", ext / "procgenAISC"]
    for p in upstream:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def ckpt_dir(root, env):
    """Released checkpoints, laid out as $GM_DATA_ROOT/checkpoints/{maze,coin,ras}/<seed>_<gm|teach>[_<step>].pth."""
    return Path(os.environ.get("GM_DATA_ROOT", Path(root) / "data")) / "checkpoints" / env
