"""Task-phase / decision-aligned representation analysis."""

from gm_geometry.phase.coinrun import (
    analyze_coinrun_phase,
    assign_anchors,
    collect_coinrun_episodes,
    commitment_step,
    yellow_coin_mask,
    y_coding_contingency,
)

__all__ = [
    "analyze_coinrun_phase",
    "assign_anchors",
    "collect_coinrun_episodes",
    "commitment_step",
    "yellow_coin_mask",
    "y_coding_contingency",
]
