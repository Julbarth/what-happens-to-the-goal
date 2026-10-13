"""Run directory for the robotics experiments: checkpoints/, results/ and figures/ live under it."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("GM_ROBOTICS_ROOT", REPO / "data" / "robotics"))
