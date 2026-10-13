"""Covariates for the diversified-start Attribute-Maze bank: action groups, straight-line bearings, bearing actions."""
from __future__ import annotations

import math

import numpy as np

from gm_geometry.probes.ras_gt_bank import ENGINE_DELTAS

GROUP_DIR = {"LEFT": (-1, 0), "DOWN": (0, -1), "UP": (0, 1), "RIGHT": (1, 0)}
GROUP_ORDER = ("LEFT", "DOWN", "UP", "RIGHT")
EXPECTED_GROUPS = {"LEFT": (0, 1, 2), "DOWN": (3,), "UP": (5,), "RIGHT": (6, 7, 8)}
OCT8 = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
QUAD4 = ("E", "N", "W", "S")


def action_groups() -> dict[str, tuple[int, ...]]:
    """Group engine action ids by their measured cell delta; asserts the LEFT/DOWN/UP/RIGHT mapping."""
    by_dir: dict[tuple[int, int], list[int]] = {}
    for a, d in ENGINE_DELTAS.items():
        if d is not None:
            by_dir.setdefault(d, []).append(a)
    groups = {g: tuple(sorted(by_dir[d])) for g, d in GROUP_DIR.items()}
    if groups != EXPECTED_GROUPS or sum(len(v) for v in by_dir.values()) != 8:
        raise AssertionError(f"engine action groups {groups} differ from {EXPECTED_GROUPS}")
    return groups


GROUPS = action_groups()
GROUP_OF = {a: g for g, ids in GROUPS.items() for a in ids}


def groups_of(actions: list[int]) -> list[str]:
    return [g for g in GROUP_ORDER if any(GROUP_OF.get(a) == g for a in actions)]


def bearing(agent, target) -> dict:
    """Straight-line bearing of `target` from `agent` in world coordinates (y grows with UP)."""
    dx, dy = target[0] - agent[0], target[1] - agent[1]
    n = math.hypot(dx, dy)
    ang = math.degrees(math.atan2(dy, dx))
    dots = {g: (dx * v[0] + dy * v[1]) / n for g, v in GROUP_DIR.items()}
    best = max(dots.values())
    b_groups = [g for g in GROUP_ORDER if dots[g] >= best - 1e-9]
    return {"vec": [dx, dy], "unit": [dx / n, dy / n], "angle_deg": ang,
            "oct8": OCT8[int(round(ang / 45.0)) % 8],
            "quad4": QUAD4[int(math.floor((ang + 45.0) / 90.0)) % 4],
            "bearing_groups": b_groups, "bearing_unique": len(b_groups) == 1}


def contrast_vector(groups_pos: list[str], groups_neg: list[str], space: str = "logit") -> np.ndarray:
    """mean_{a in pos} e_a - mean_{a in neg} e_a, over the 15 action ids ('logit') or the 4 groups ('group')."""
    if space == "group":
        w = np.zeros(4)
        for g in groups_pos:
            w[GROUP_ORDER.index(g)] += 1.0 / len(groups_pos)
        for g in groups_neg:
            w[GROUP_ORDER.index(g)] -= 1.0 / len(groups_neg)
        return w
    pos = [a for g in groups_pos for a in GROUPS[g]]
    neg = [a for g in groups_neg for a in GROUPS[g]]
    w = np.zeros(15)
    w[pos] += 1.0 / len(pos)
    w[neg] -= 1.0 / len(neg)
    return w


def covariates(agent, line, gem, geo: dict) -> dict:
    """Per-state action-group, bearing and bearing-disagreement covariates from a ras_gt_bank.geometry() record."""
    gl, gg = groups_of(geo["A_line"]), groups_of(geo["A_gem"])
    bl, bg = bearing(agent, line), bearing(agent, gem)
    dis_l = not (set(bl["bearing_groups"]) & set(gl))
    dis_g = not (set(bg["bearing_groups"]) & set(gg))
    defined = bl["bearing_unique"] and bg["bearing_unique"] and bl["bearing_groups"] != bg["bearing_groups"]
    out = {
        "groups_line": gl, "groups_gem": gg,
        "contrast": "+".join(gl) + ">" + "+".join(gg),
        "conflict_pair": "|".join(sorted(["+".join(gl), "+".join(gg)])),
        "bearing_line": bl, "bearing_gem": bg,
        "bearing_action_line": bl["bearing_groups"], "bearing_action_gem": bg["bearing_groups"],
        "bearing_action_disagrees_line": dis_l, "bearing_action_disagrees_gem": dis_g,
        "bearing_action_disagrees": dis_l or dis_g,
        "bearing_contrast_defined": defined,
        "disagree_decisive": bool(defined and (dis_l or dis_g)),
        "reversed": bool(defined and set(bl["bearing_groups"]) <= set(gg) and set(bg["bearing_groups"]) <= set(gl)),
        "quad_pair": "|".join(sorted([bl["quad4"], bg["quad4"]])),
    }
    return out
