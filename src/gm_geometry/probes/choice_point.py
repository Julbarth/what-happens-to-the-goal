"""Matched-geometry choice-point readout plus bearing and target labels for colour-shape duel frames."""
# Black backgrounds give clean colour masks; textured backgrounds fall back to frame-diff agent tracking.

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Procgen discrete combos (see procgen.env.ProcgenGymEnvWrapper.get_combos):
# index → (dx_image_right, dy_image_down) for pure cardinal moves.
CARDINAL_ACTIONS: dict[int, np.ndarray] = {
    1: np.array([-1.0, 0.0]),  # LEFT
    3: np.array([0.0, 1.0]),  # DOWN
    5: np.array([0.0, -1.0]),  # UP
    7: np.array([1.0, 0.0]),  # RIGHT
}


def _centroid(mask: np.ndarray, *, min_px: int = 3) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) < min_px:
        return None
    return float(xs.mean()), float(ys.mean())


def _connected_components(mask: np.ndarray, min_px: int = 6) -> list[dict[str, Any]]:
    """4-connected components with centroid + compactness (area / bbox)."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    ccs: list[dict[str, Any]] = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < min_px:
                continue
            ys = np.array([c[0] for c in cells])
            xs = np.array([c[1] for c in cells])
            bb = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            ccs.append({
                "xy": (float(xs.mean()), float(ys.mean())),
                "n_px": int(len(cells)),
                "compactness": float(len(cells) / max(bb, 1)),
            })
    return ccs


def _line_gem_from_same_colour(
    colour_mask: np.ndarray,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Split one colour into line vs gem via CC compactness (line = elongated)."""
    ccs = _connected_components(colour_mask, min_px=6)
    if len(ccs) < 2:
        return None, None
    ccs = sorted(ccs, key=lambda c: c["n_px"], reverse=True)[:4]
    a, b = ccs[0], ccs[1]
    if a["compactness"] <= b["compactness"]:
        return a["xy"], b["xy"]
    return b["xy"], a["xy"]


def yellow_mask(img_hwc: np.ndarray) -> np.ndarray:
    r, g, b = img_hwc[..., 0], img_hwc[..., 1], img_hwc[..., 2]
    return (r > 0.55) & (g > 0.55) & (b < 0.40) & (r > b + 0.2) & (g > b + 0.2)


def red_mask(img_hwc: np.ndarray) -> np.ndarray:
    r, g, b = img_hwc[..., 0], img_hwc[..., 1], img_hwc[..., 2]
    return (r > 0.55) & (g < 0.35) & (b < 0.35) & (r > g + 0.2)


def line_shape_mask(img_hwc: np.ndarray, colour_mask: np.ndarray | None = None) -> np.ndarray:
    """Diagonal-energy blobs (line/diag sprites), optionally restricted to a colour."""
    gray = img_hwc.mean(axis=2)
    d1 = np.zeros_like(gray)
    d2 = np.zeros_like(gray)
    d1[1:, 1:] = np.abs(gray[1:, 1:] - gray[:-1, :-1])
    d2[1:, :-1] = np.abs(gray[1:, :-1] - gray[:-1, 1:])
    energy = np.maximum(d1, d2)
    thr = float(np.quantile(energy[energy > 0], 0.92)) if (energy > 0).any() else 1.0
    m = energy >= thr
    if colour_mask is not None:
        m = m & colour_mask
    return m


def gem_shape_mask(img_hwc: np.ndarray, colour_mask: np.ndarray | None = None) -> np.ndarray:
    """Compact high-saturation blobs (gem sprites)."""
    r, g, b = img_hwc[..., 0], img_hwc[..., 1], img_hwc[..., 2]
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    m = sat > 0.25
    if colour_mask is not None:
        m = m & colour_mask
    return m


def sand_mask(img_hwc: np.ndarray) -> np.ndarray:
    """Kenney sandCenter walls (beige), exclude yellow gems."""
    r, g, b = img_hwc[..., 0], img_hwc[..., 1], img_hwc[..., 2]
    yel = yellow_mask(img_hwc)
    return (r > 0.45) & (g > 0.35) & (b > 0.20) & (b < 0.55) & (r > b) & (g > b) & ~yel


def mouse_mask_black_bg(img_hwc: np.ndarray) -> np.ndarray:
    yel, red, sand = yellow_mask(img_hwc), red_mask(img_hwc), sand_mask(img_hwc)
    floor = img_hwc.sum(axis=-1) < 0.25
    return ~(yel | red | sand | floor) & (img_hwc.sum(axis=-1) > 0.25)


def localize_dual_objects(
    obs_chw: np.ndarray,
    *,
    obj1: str,
    obj2: str,
    prev_hwc: np.ndarray | None = None,
    use_backgrounds: bool = False,
) -> dict[str, Any]:
    """Localize two duel objects by colour and/or shape, plus the agent.

    Returns colour-indexed and shape-indexed centroids when recoverable:
      yellow_xy / red_xy / line_xy / gem_xy / agent_xy
    plus distances from agent to each.
    """
    img = np.transpose(np.asarray(obs_chw, dtype=np.float32), (1, 2, 0))
    yel, red = yellow_mask(img), red_mask(img)
    # shape within colour when both objects share a colour
    line_m = line_shape_mask(img, None)
    gem_m = gem_shape_mask(img, None)

    # Prefer colour centroids when colours differ; else shape within shared colour
    colours = {obj1.split("_")[0], obj2.split("_")[0]}
    shapes = {("line" if "line" in o else "gem") for o in (obj1, obj2)}

    yellow_xy = _centroid(yel)
    red_xy = _centroid(red)

    if colours == {"red"}:
        line_xy, gem_xy = _line_gem_from_same_colour(red)
    elif colours == {"yellow"}:
        line_xy, gem_xy = _line_gem_from_same_colour(yel)
    elif shapes == {"gem"}:
        # colour-only duel (e.g. yellow_gem vs red_gem): no line centroid
        line_xy = None
        gem_xy = None
    elif shapes == {"line"}:
        line_xy = None
        gem_xy = None
    else:
        # colours and shapes both differ: colour separates; map from config names
        line_obj = obj1 if "line" in obj1 else obj2
        gem_obj = obj1 if "line" not in obj1 else obj2
        line_xy = yellow_xy if line_obj.startswith("yellow") else red_xy
        gem_xy = yellow_xy if gem_obj.startswith("yellow") else red_xy

    agent = None
    agent_method = "none"
    if not use_backgrounds:
        agent = _centroid(mouse_mask_black_bg(img), min_px=3)
        agent_method = "residual" if agent is not None else "none"
    if agent is None and prev_hwc is not None:
        diff = np.abs(img - prev_hwc).sum(axis=-1)
        mot = (diff > 0.12) & ~yel & ~red
        agent = _centroid(mot, min_px=5)
        agent_method = "motion" if agent is not None else agent_method

    def _dist(xy):
        if agent is None or xy is None:
            return float("nan")
        return float(np.hypot(xy[0] - agent[0], xy[1] - agent[1]))

    return {
        "img_hwc": img,
        "agent_xy": agent,
        "agent_method": agent_method,
        "yellow_xy": yellow_xy,
        "red_xy": red_xy,
        "line_xy": line_xy,
        "gem_xy": gem_xy,
        "dist_yellow": _dist(yellow_xy),
        "dist_red": _dist(red_xy),
        "dist_line": _dist(line_xy),
        "dist_gem": _dist(gem_xy),
        "n_yellow_px": int(yel.sum()),
        "n_red_px": int(red.sum()),
        "obj1": obj1,
        "obj2": obj2,
        "colours": sorted(colours),
        "shapes": sorted(shapes),
    }


def localize_frame(
    obs_chw: np.ndarray,
    *,
    prev_hwc: np.ndarray | None = None,
    use_backgrounds: bool = False,
    obj1: str = "red_line_diag",
    obj2: str = "yellow_gem",
) -> dict[str, Any]:
    """Localization wrapper for the classic trap configuration (line=red, gem=yellow)."""
    loc = localize_dual_objects(
        obs_chw, obj1=obj1, obj2=obj2, prev_hwc=prev_hwc, use_backgrounds=use_backgrounds,
    )
    return loc


def matched_geometry(
    loc: dict[str, Any],
    *,
    max_rel_gap: float = 0.35,
    min_dist_px: float = 8.0,
) -> bool:
    """Both objects visible, agent localized, distances within relative gap."""
    dg, dl = loc.get("dist_gem"), loc.get("dist_line")
    if not (np.isfinite(dg) and np.isfinite(dl)):
        return False
    if min(dg, dl) < min_dist_px:
        return False
    return abs(dg - dl) / max(dg, dl, 1e-6) <= max_rel_gap


def action_toward(agent_xy: tuple[float, float], target_xy: tuple[float, float]) -> int:
    """Cardinal action whose unit vector best aligns with agent→target."""
    v = np.array([target_xy[0] - agent_xy[0], target_xy[1] - agent_xy[1]], dtype=np.float64)
    return int(max(CARDINAL_ACTIONS, key=lambda a: float(CARDINAL_ACTIONS[a] @ v)))


def logit_contrast_line_minus_gem(logits: np.ndarray, loc: dict[str, Any]) -> float:
    """Log-prob contrast ℓ_line − ℓ_gem (Ladder passes log-softmax as ``logits``)."""
    agent, gem, line = loc["agent_xy"], loc["gem_xy"], loc["line_xy"]
    if agent is None or gem is None or line is None:
        return float("nan")
    a_line = action_toward(agent, line)
    a_gem = action_toward(agent, gem)
    if a_line == a_gem:
        return float("nan")  # ambiguous geometry (same cardinal)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    return float(logits[a_line] - logits[a_gem])


def prob_contrast_line_minus_gem(probs: np.ndarray, loc: dict[str, Any]) -> float:
    """Bounded probability contrast p(a_line) − p(a_gem) at choice frames, in [−1, 1]."""
    agent, gem, line = loc["agent_xy"], loc["gem_xy"], loc["line_xy"]
    if agent is None or gem is None or line is None:
        return float("nan")
    a_line = action_toward(agent, line)
    a_gem = action_toward(agent, gem)
    if a_line == a_gem:
        return float("nan")
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    return float(probs[a_line] - probs[a_gem])


def gem_prefer_at_choice(probs: np.ndarray, loc: dict[str, Any]) -> float:
    """1.0 if greedy action points at gem vs line (paired with ``line_prefer_at_choice``)."""
    lp = line_prefer_at_choice(probs, loc)
    if not np.isfinite(lp):
        return float("nan")
    return 1.0 - lp


def goal_lure_xy(loc: dict[str, Any], obj1: str, obj2: str) -> tuple[Any, Any]:
    """Map procgen obj1 (+10 goal) / obj2 (−10 lure) to localized centroids."""
    o1_line, o2_line = "line" in obj1, "line" in obj2
    if o1_line and not o2_line:
        return loc.get("line_xy"), loc.get("gem_xy")
    if o2_line and not o1_line:
        return loc.get("gem_xy"), loc.get("line_xy")
    return None, None


def goal_prefer_at_choice(
    probs: np.ndarray, loc: dict[str, Any], *, obj1: str, obj2: str,
) -> float:
    """Fraction of frames where greedy action points at obj1 (+10 goal), not obj2 lure."""
    agent = loc.get("agent_xy")
    goal_xy, lure_xy = goal_lure_xy(loc, obj1, obj2)
    if agent is None or goal_xy is None or lure_xy is None:
        return float("nan")
    a_goal = action_toward(agent, goal_xy)
    a_lure = action_toward(agent, lure_xy)
    if a_goal == a_lure:
        return float("nan")
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    act = int(probs.argmax())
    return 1.0 if act == a_goal else 0.0


def line_prefer_at_choice(probs: np.ndarray, loc: dict[str, Any]) -> float:
    """1.0 if greedy action points at line vs gem, else 0.0 (Maze ``cheese_prefer`` analogue)."""
    agent, gem, line = loc["agent_xy"], loc["gem_xy"], loc["line_xy"]
    if agent is None or gem is None or line is None:
        return float("nan")
    a_line = action_toward(agent, line)
    a_gem = action_toward(agent, gem)
    if a_line == a_gem:
        return float("nan")
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    act = int(probs.argmax())
    return 1.0 if act == a_line else 0.0


def bearing_features(loc: dict[str, Any]) -> dict[str, float]:
    """Signed egocentric geometry for gem vs line and colour-indexed twins."""
    agent = loc.get("agent_xy")
    out = {
        "dist_gem": float(loc.get("dist_gem", float("nan"))),
        "dist_line": float(loc.get("dist_line", float("nan"))),
        "dist_yellow": float(loc.get("dist_yellow", float("nan"))),
        "dist_red": float(loc.get("dist_red", float("nan"))),
        "dist_line_minus_gem": float("nan"),
        "dist_yellow_minus_red": float("nan"),
        "dist_yellow_minus_other": float("nan"),
        "dist_line_minus_other": float("nan"),
        "bearing_gem_x": float("nan"),
        "bearing_gem_y": float("nan"),
        "bearing_line_x": float("nan"),
        "bearing_line_y": float("nan"),
        "closer_is_line": float("nan"),
        "rel_gap": float("nan"),
    }
    gem, line = loc.get("gem_xy"), loc.get("line_xy")
    if agent is None:
        return out
    if gem is not None:
        out["bearing_gem_x"] = float(gem[0] - agent[0])
        out["bearing_gem_y"] = float(gem[1] - agent[1])
    if line is not None:
        out["bearing_line_x"] = float(line[0] - agent[0])
        out["bearing_line_y"] = float(line[1] - agent[1])
    if np.isfinite(out["dist_gem"]) and np.isfinite(out["dist_line"]):
        out["dist_line_minus_gem"] = out["dist_line"] - out["dist_gem"]
        out["closer_is_line"] = 1.0 if out["dist_line"] < out["dist_gem"] else 0.0
        mx = max(out["dist_gem"], out["dist_line"], 1e-6)
        out["rel_gap"] = abs(out["dist_gem"] - out["dist_line"]) / mx
        out["dist_line_minus_other"] = out["dist_line_minus_gem"]
    if np.isfinite(out["dist_yellow"]) and np.isfinite(out["dist_red"]):
        out["dist_yellow_minus_red"] = out["dist_yellow"] - out["dist_red"]
        out["dist_yellow_minus_other"] = out["dist_yellow_minus_red"]
    return out


def heading_target_label(
    prev_loc: dict[str, Any] | None,
    loc: dict[str, Any],
    *,
    min_delta: float = 0.5,
) -> float:
    """1 if agent closed distance to line more than to gem; 0 opposite; NaN if unclear."""
    if prev_loc is None:
        return float("nan")
    dg0, dl0 = prev_loc.get("dist_gem"), prev_loc.get("dist_line")
    dg1, dl1 = loc.get("dist_gem"), loc.get("dist_line")
    if not all(np.isfinite(x) for x in (dg0, dl0, dg1, dl1)):
        return float("nan")
    d_gem = float(dg0) - float(dg1)  # positive = approaching gem
    d_line = float(dl0) - float(dl1)
    if abs(d_line - d_gem) < min_delta:
        return float("nan")
    return 1.0 if d_line > d_gem else 0.0


def register_layer_hooks(policy, layer_names: list[str]) -> tuple[dict[str, list], list]:
    """Register forward hooks on Impala submodules and return (store, handles).

    ImpalaModel.forward is block1 -> block2 -> block3 -> ReLU -> flatten -> fc -> ReLU, and both ReLUs are
    unhooked functional calls, so post-ReLU sites apply ReLU inside the hook:

      ``fc``         embedder.fc output (256), post-ReLU
      ``fc_raw``     embedder.fc output, pre-ReLU
      ``pre_fc``     flatten(ReLU(block3)) (2048), the literal input to fc
      ``block3``     last Impala block, post-ReLU, kept spatial (B, C, H, W)
      ``block3_raw`` block3 residual-sum output before the ReLU, flattened (2048)
      ``block2``     second Impala block, spatial
      ``block1``     first Impala block, spatial"""
    store: dict[str, list] = {n: [] for n in layer_names}
    handles = []
    embedder = policy.embedder

    def _keep(name):
        def _hook(_m, _i, out):
            store[name].append(out.detach())

        return _hook

    def _keep_relu(name):
        def _hook(_m, _i, out):
            store[name].append(torch.relu(out).detach())

        return _hook

    for name in layer_names:
        if name == "fc":
            handles.append(embedder.fc.register_forward_hook(_keep_relu("fc")))
        elif name == "fc_raw":
            handles.append(embedder.fc.register_forward_hook(_keep("fc_raw")))
        elif name == "block3":
            handles.append(embedder.block3.register_forward_hook(_keep_relu("block3")))
        elif name == "block2":
            handles.append(embedder.block2.register_forward_hook(_keep("block2")))
        elif name == "block1":
            handles.append(embedder.block1.register_forward_hook(_keep("block1")))
        elif name == "pre_fc":
            # Literal post-ReLU input to fc (flatten happens in tensor_to_feature).
            handles.append(embedder.block3.register_forward_hook(_keep_relu("pre_fc")))
        elif name == "block3_raw":
            handles.append(embedder.block3.register_forward_hook(_keep("block3_raw")))
        else:
            raise ValueError(f"unknown layer {name!r}")
    return store, handles


def tensor_to_feature(name: str, t: torch.Tensor) -> np.ndarray:
    """Convert hooked activation to a 1-d (or flat) numpy feature for probing."""
    x = t.detach().cpu().numpy()
    if x.ndim == 4 and name in ("pre_fc", "block3_raw"):
        # (B, C, H, W) → flatten spatial+channel
        return x.reshape(x.shape[0], -1).astype(np.float32)
    if x.ndim == 4:
        # keep spatial for spatial probes: return (B, C, H, W)
        return x.astype(np.float32)
    if x.ndim == 2:
        return x.astype(np.float32)
    return x.reshape(x.shape[0], -1).astype(np.float32)


def spatial_probe_auc(
    acts_bchw: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Per-location ridge on channel vectors; return AUC map and peak cell."""
    from gm_geometry.probes.linear import fit_ridge, roc_auc

    y = np.asarray(y).reshape(-1)
    b, c, h, w = acts_bchw.shape
    auc_map = np.full((h, w), np.nan, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            x = acts_bchw[:, :, i, j]
            if len(np.unique(y[~np.isnan(y.astype(float))])) < 2:
                continue
            ok = np.isfinite(y.astype(float))
            if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
                continue
            w_vec, bias = fit_ridge(x[ok], y[ok], alpha=alpha)
            scores = x[ok] @ w_vec + bias
            auc_map[i, j] = roc_auc(y[ok], scores)
    finite = np.isfinite(auc_map)
    peak = None
    if finite.any():
        flat = np.nanargmax(auc_map)
        peak = (int(flat // w), int(flat % w), float(auc_map.flat[flat]))
    return {
        "auc_map": auc_map,
        "peak_ij_auc": peak,
        "mean_auc": float(np.nanmean(auc_map)) if finite.any() else float("nan"),
        "max_auc": float(np.nanmax(auc_map)) if finite.any() else float("nan"),
    }
