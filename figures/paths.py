"""Locations shared by the figure scripts: code, released data, robot results and rendered output."""
import os
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
ROOT = FIG_DIR.parent
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", ROOT / "data"))
ROBOTICS_ROOT = Path(os.environ.get("GM_ROBOTICS_ROOT", DATA_ROOT / "robotics"))
# Small per-figure tables (CSV/JSON) written by one figure script and read by another.
TABLES = DATA_ROOT / "figure_tables"
OUT_DIR = Path(os.environ.get("GM_FIGURE_DIR", DATA_ROOT / "paper_figures"))
