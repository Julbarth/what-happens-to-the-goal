"""Pixel-derived concept labels for colour vs shape probes."""

from __future__ import annotations

import numpy as np


def _yellow_pixel_score(image_hwc: np.ndarray) -> float:
    r, g, b = image_hwc[..., 0], image_hwc[..., 1], image_hwc[..., 2]
    mask = (r > 0.45) & (g > 0.45) & (b < 0.45) & ((r + g) > 0.95)
    return float(mask.mean())


def _red_pixel_score(image_hwc: np.ndarray) -> float:
    r, g, b = image_hwc[..., 0], image_hwc[..., 1], image_hwc[..., 2]
    mask = (r > 0.45) & (g < 0.35) & (b < 0.35) & (r > g + 0.1)
    return float(mask.mean())


def _diagonal_energy(image_hwc: np.ndarray) -> float:
    """Proxy for line/diag sprites and maze wall edges."""
    gray = image_hwc.mean(axis=2)
    d1 = np.abs(gray[1:, 1:] - gray[:-1, :-1])
    d2 = np.abs(gray[1:, :-1] - gray[:-1, 1:])
    return float(max(d1.mean(), d2.mean()))


def _compact_color_blob(image_hwc: np.ndarray) -> float:
    """Proxy for saturated gem-like sprites."""
    r, g, b = image_hwc[..., 0], image_hwc[..., 1], image_hwc[..., 2]
    sat = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    vivid = sat > 0.15
    return float(vivid.mean())


def concept_scores_from_obs(obs_chw: np.ndarray) -> dict[str, float]:
    """Return continuous proxy/true-goal concept scores for one frame (3, H, W)."""
    image = np.transpose(obs_chw, (1, 2, 0)).astype(np.float32)
    yellow = _yellow_pixel_score(image)
    red = _red_pixel_score(image)
    line = _diagonal_energy(image)
    gem = _compact_color_blob(image)

    return {
        "yellow": yellow,
        "line": line,
        "gem": gem,
        "line_over_gem": line - gem,
        "red": red,
    }


def concept_scores_batch(obs_nchw: np.ndarray) -> dict[str, np.ndarray]:
    scores = [concept_scores_from_obs(obs_nchw[i]) for i in range(len(obs_nchw))]
    keys = scores[0].keys()
    return {k: np.array([s[k] for s in scores], dtype=np.float32) for k in keys}


# ---- Ground-truth (state/config-derived) labels — no pixels, no shared terms ----

YELLOW_COLOURS = {"yellow"}

# Colours with pure_* assets in maze_colorsobjects_duel (excl. invisible).
PRESENCE_COLOURS = (
    "black",
    "blue",
    "cyan",
    "green",
    "purple",
    "red",
    "white",
    "yellow",
)


def _parse_object(name: str) -> dict[str, object]:
    """Parse a procgen object name like 'red_line_diag' or 'yellow_gem'."""
    is_line = "line" in name
    colour = name.split("_")[0]
    return {"is_line": is_line, "colour": colour}


def ground_truth_presence(obj1: str, obj2: str) -> dict[str, float]:
    """Ground-truth object-presence and goal-role labels from the level config.

    ``obj1`` is the +10 goal and ``obj2`` the −10 lure. Presence keys (``line_present``, ``yellow_present``, …)
    fire for either slot; role keys (``goal_is_line``, …) only for ``obj1``. Includes ``{colour}_present``
    for every registered colour."""
    a = _parse_object(obj1)
    b = _parse_object(obj2)
    colours = {a["colour"], b["colour"]}
    line_present = 1.0 if (a["is_line"] or b["is_line"]) else 0.0
    gem_present = 1.0 if ((not a["is_line"]) or (not b["is_line"])) else 0.0
    out: dict[str, float] = {
        "line_present": line_present,
        "gem_present": gem_present,
        "yellow_present": 1.0 if ("yellow" in colours) else 0.0,
        # Role / goal-identity (obj1 = +10). Sharper than presence when the
        # grid puts the same feature in the lure slot.
        "goal_is_line": 1.0 if a["is_line"] else 0.0,
        "lure_is_line": 1.0 if b["is_line"] else 0.0,
        "goal_is_yellow": 1.0 if a["colour"] == "yellow" else 0.0,
        "lure_is_yellow": 1.0 if b["colour"] == "yellow" else 0.0,
    }
    for c in PRESENCE_COLOURS:
        out[f"{c}_present"] = 1.0 if c in colours else 0.0
    return out
