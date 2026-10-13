"""Attribute Maze object-pair configurations for the GM (confounded) and Non-GM (decorrelated) trainers."""

from __future__ import annotations

# Non-GM training set: corr(line present, yellow present) = 0 across the eight groups, and no pair
# overlaps the conflict-test objects. 8 groups x 32 envs = 256.
DUEL_CONFIGS_B2: list[tuple[str, str]] = [
    ("yellow_line_diag", "blue_gem"),    # dual-cue       (line, yellow) = (1, 1)
    ("yellow_line_diag", "yellow_gem"),  # shape-forcing  (1, 1)
    ("green_line_diag", "blue_gem"),     # dual-cue       (1, 0)
    ("red_line_diag", "red_gem"),        # shape-forcing  (1, 0)
    ("yellow_gem", "blue_gem"),          # colour-forcing (0, 1)
    ("yellow_gem", "green_gem"),         # colour-forcing (0, 1)
    ("red_gem", "green_gem"),            # colour-forcing (0, 0)
    ("purple_gem", "red_gem"),           # colour-forcing (0, 0)
]

DUEL_GROUP_TYPES_B2 = [
    "dual-cue",
    "shape-forcing",
    "dual-cue",
    "shape-forcing",
    "colour-forcing",
    "colour-forcing",
    "colour-forcing",
    "colour-forcing",
]

# GM training: the yellow line is always the rewarded object, so colour and shape are confounded.
FIXED_CONFOUNDED_PAIR = ("yellow_line_diag", "blue_gem")

CONFIG_SETS = {
    "b2": (DUEL_CONFIGS_B2, DUEL_GROUP_TYPES_B2),
}
