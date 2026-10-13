"""Canonical Attribute Maze counterfactual edits: attribute swaps (canonical) and ablations (supplementary)."""
# shape_swap puts the yellow line at v, colour_swap puts it at u; D_COMP = h(colour_swap) - h(shape_swap)
# points toward the state with the yellow line at u, the line-shaped intended-goal candidate.
from __future__ import annotations

OBJ_LINE = "red_line_diag"     # (line, red), obj1, GOAL1, reward +10
OBJ_GEM = "yellow_gem"         # (gem, yellow), obj2, GOAL2, reward -10

CANONICAL_EDITS: dict[str, tuple[str, str]] = {
    "shape_swap": ("red_gem", "yellow_line_diag"),
    "colour_swap": ("yellow_line_diag", "red_gem"),
}
SUPPLEMENTARY_EDITS: dict[str, tuple[str, str]] = {
    "shape_ablate": ("red_gem", OBJ_GEM),
    "colour_ablate": (OBJ_LINE, "green_gem"),
}
D_COMP_PLUS, D_COMP_MINUS = "colour_swap", "shape_swap"


def edit_objects(edit: str) -> tuple[str, str]:
    """(obj1, obj2) for make_env; obj1 stays in slot GOAL1 (cell u), obj2 in GOAL2 (cell v)."""
    if edit == "base":
        return OBJ_LINE, OBJ_GEM
    table = CANONICAL_EDITS | SUPPLEMENTARY_EDITS
    if edit not in table:
        raise KeyError(f"unknown edit {edit!r}; canonical: {sorted(CANONICAL_EDITS)}")
    return table[edit]
