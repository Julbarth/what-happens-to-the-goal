"""Frozen-trunk rewiring: shared tap/freeze/reinit and closed-loop eval utilities."""

from gm_geometry.rewire.eval import bootstrap_diff_ci, motor_collapse_flag
from gm_geometry.rewire.taps import (
    TAPS,
    freeze_fc,
    freeze_through,
    freeze_trunk,
    prepare_tap,
    reinit_fc,
    reinit_heads,
    trainable_parameters,
)

__all__ = [
    "TAPS",
    "freeze_fc",
    "freeze_through",
    "freeze_trunk",
    "prepare_tap",
    "reinit_fc",
    "reinit_heads",
    "trainable_parameters",
    "bootstrap_diff_ci",
    "motor_collapse_flag",
]
