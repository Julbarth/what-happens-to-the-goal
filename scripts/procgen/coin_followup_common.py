"""Shared path setup and labels for Coin follow-up probes/evals."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Egocentric place grid: coin relative to agent (world cells), binned for navigation.
# rel_x = coin_x - agent_x; rel_y = coin_y - agent_y (y grows up in world).
EGO_X_EDGES = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float64)
EGO_Y_EDGES = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
EGO_N_X = len(EGO_X_EDGES) + 1
EGO_N_Y = len(EGO_Y_EDGES) + 1
EGO_N_CELLS = EGO_N_X * EGO_N_Y
FC_DIM = 256


def setup_coin_paths() -> Path:
    """Insert repo + procgen paths (safe to call repeatedly)."""
    for sub in ("src", "pkg"):
        p = ROOT / sub
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    scripts = ROOT / "scripts" / "procgen"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    train = ROOT / "external" / "train-procgen-pytorch"
    if not train.exists():
        train = ROOT / "vendored" / "train-procgen-pytorch"
    proc = ROOT / "external" / "procgenAISC"
    if not proc.exists():
        proc = ROOT / "vendored" / "procgenAISC"
    for p in (train, proc):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return ROOT


def egocentric_place_cell(rel_x: float, rel_y: float) -> int:
    """Bin relative coin position into a single egocentric cell index (0..N-1)."""
    if not (np.isfinite(rel_x) and np.isfinite(rel_y)):
        return -1
    bx = int(np.digitize(rel_x, EGO_X_EDGES))
    by = int(np.digitize(rel_y, EGO_Y_EDGES))
    bx = int(np.clip(bx, 0, EGO_N_X - 1))
    by = int(np.clip(by, 0, EGO_N_Y - 1))
    return bx * EGO_N_Y + by


def labels_from_info_dict(info: dict) -> dict | None:
    from gm_geometry.probes.coin_location import labels_from_info, world_to_feat  # noqa: E402

    labels = labels_from_info(info)
    if labels is None:
        return None
    w = int(labels["level_width"])
    ax = float(labels.get("agent_x", 0.0))
    ay = float(labels.get("agent_y", 0.0))
    cx = float(labels["coin_x"])
    cy = float(labels["coin_y"])
    coin_dx = cx - ax
    side = "neutral"
    if coin_dx > 0.5:
        side = "right"
    elif coin_dx < -0.5:
        side = "left"
    labels["coin_side"] = side
    labels["rel_x"] = coin_dx
    labels["rel_y"] = cy - ay
    labels["bearing_right"] = 1.0 if side == "right" else 0.0 if side == "left" else float("nan")

    # Absolute map cell is for reference only: it is confounded with level layout and progress.
    if labels.get("coin_present", 0) == 1:
        fx, fy = world_to_feat(labels["coin_x"], labels["coin_y"], level_width=w)
        labels["place_cell_absolute"] = int(fy * 8 + fx)
        labels["place_ego_cell"] = egocentric_place_cell(coin_dx, labels["rel_y"])
    else:
        labels["place_cell_absolute"] = -1
        labels["place_ego_cell"] = -1
    # Primary probe label: egocentric binned place.
    labels["place_cell"] = labels["place_ego_cell"]
    return labels


def episode_coin_strata(info: dict) -> dict[str, bool]:
    """Level-level strata from first observed env info in an episode."""
    labels = labels_from_info_dict(info)
    if labels is None:
        return {}
    jump = int(info.get("coin_requires_jump", labels.get("coin_requires_jump", 0)))
    side = labels["coin_side"]
    w = float(labels.get("level_width", 64) or 64)
    cx = float(labels["coin_x"])
    # Layout strata (val levels): agent spawn is left-heavy so ego coin_side is often all-right.
    layout_left = cx < w * 0.4
    layout_right = cx > w * 0.6
    return {
        "coin_left": layout_left,
        "coin_right": layout_right,
        "coin_left_ego": side == "left",
        "coin_right_ego": side == "right",
        "flat_ground": jump == 0,
        "jump_required": jump == 1,
        "coin_left_flat": layout_left and jump == 0,
        "coin_right_flat": layout_right and jump == 0,
        "on_crate": int(info.get("coin_on_crate", 0)) == 1,
    }


def place_class_counts(y: np.ndarray) -> dict[str, Any]:
    """Per-class counts for place labels (y >= 0)."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    mask = y >= 0
    yv = y[mask]
    if len(yv) == 0:
        return {"n_valid": 0, "n_classes": 0, "counts": {}, "min_count": 0, "median_count": 0.0}
    classes, counts = np.unique(yv, return_counts=True)
    count_map = {int(c): int(n) for c, n in zip(classes, counts)}
    return {
        "n_valid": int(len(yv)),
        "n_classes": int(len(classes)),
        "counts": count_map,
        "min_count": int(counts.min()),
        "median_count": float(np.median(counts)),
        "max_count": int(counts.max()),
    }


def fit_pca(H: np.ndarray, k: int, *, seed: int = 0, fit_frac: float = 0.8) -> dict[str, np.ndarray]:
    """PCA on a random subset; returns mean [D] and components [k, D]."""
    H = np.asarray(H, dtype=np.float64)
    k = min(k, H.shape[1], H.shape[0] - 1)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(H))
    n_fit = max(k + 1, int(len(H) * fit_frac))
    fit = H[idx[:n_fit]]
    mu = fit.mean(axis=0, keepdims=True)
    Hc = fit - mu
    _u, _s, vt = np.linalg.svd(Hc, full_matrices=False)
    components = vt[:k].astype(np.float64)
    return {"mean": mu.reshape(-1), "components": components}


def apply_pca(H: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    mu = pca["mean"]
    comp = pca["components"]
    return (H - mu) @ comp.T


def ridge_binary_auc(
    H: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    val_frac: float = 0.2,
    seed: int = 0,
    min_per_class: int = 25,
    shuffle_labels: bool = False,
) -> dict:
    """Held-out ridge AUC for binary labels."""
    from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc  # noqa: E402

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y)
    H, y = H[mask], y[mask]
    if shuffle_labels:
        rng = np.random.RandomState(seed + 999)
        y = rng.permutation(y)
    if len(y) < 2 * min_per_class:
        return {"auc": float("nan"), "n": int(len(y)), "ok": False, "shuffled": shuffle_labels}
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y)), "ok": False, "shuffled": shuffle_labels}
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_val = max(min_per_class, int(len(y) * val_frac))
    va, tr = idx[:n_val], idx[n_val:]
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
        return {"auc": float("nan"), "n": int(len(y)), "ok": False, "shuffled": shuffle_labels}
    w, b = fit_ridge(H[tr], y[tr], alpha=alpha)
    scores = predict_ridge(H[va], w, b)
    return {
        "auc": float(roc_auc(y[va], scores)),
        "n": int(len(y)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "ok": True,
        "shuffled": shuffle_labels,
    }


def ridge_multiclass_auc(
    H: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    val_frac: float = 0.2,
    seed: int = 0,
    min_per_class: int = 30,
    shuffle_labels: bool = False,
) -> dict:
    """Macro one-vs-rest AUC; only classes with >= min_per_class total frames."""
    from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc  # noqa: E402

    y = np.asarray(y, dtype=np.int64).reshape(-1)
    mask = y >= 0
    H, y = H[mask], y[mask]
    if shuffle_labels:
        rng = np.random.RandomState(seed + 999)
        y = rng.permutation(y)

    class_counts = place_class_counts(y)
    eligible = {c for c, n in class_counts.get("counts", {}).items() if n >= min_per_class}
    if len(eligible) < 2:
        return {
            "auc_macro": float("nan"),
            "n": int(len(y)),
            "n_classes": class_counts["n_classes"],
            "n_classes_eligible": len(eligible),
            "ok": False,
            "shuffled": shuffle_labels,
            "class_counts": class_counts,
        }

    y_mask = np.array([yi in eligible for yi in y], dtype=bool)
    H, y = H[y_mask], y[y_mask]
    classes = np.array(sorted(eligible), dtype=np.int64)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_val = max(min_per_class, int(len(y) * val_frac))
    va, tr = idx[:n_val], idx[n_val:]
    aucs: list[float] = []
    scored_classes: list[int] = []
    for c in classes:
        yb = (y == c).astype(np.float64)
        if yb[tr].sum() < min_per_class // 2 or yb[va].sum() < 5:
            continue
        if len(np.unique(yb[tr])) < 2 or len(np.unique(yb[va])) < 2:
            continue
        w, b = fit_ridge(H[tr], yb[tr], alpha=alpha)
        scores = predict_ridge(H[va], w, b)
        aucs.append(float(roc_auc(yb[va], scores)))
        scored_classes.append(int(c))
    if not aucs:
        return {
            "auc_macro": float("nan"),
            "n": int(len(y)),
            "n_classes": int(len(classes)),
            "n_classes_scored": 0,
            "ok": False,
            "shuffled": shuffle_labels,
            "class_counts": class_counts,
        }
    return {
        "auc_macro": float(np.mean(aucs)),
        "n": int(len(y)),
        "n_classes": int(len(classes)),
        "n_classes_scored": int(len(aucs)),
        "scored_classes": scored_classes,
        "ok": True,
        "shuffled": shuffle_labels,
        "class_counts": class_counts,
    }


def probe_place_suite(
    H_fc: np.ndarray,
    H_b3: np.ndarray,
    y_place: np.ndarray,
    *,
    alpha: float = 1.0,
    seed: int = 0,
    min_per_class: int = 30,
    match_dim: int = FC_DIM,
) -> dict:
    """Place probes with shuffled null and PCA-matched block3 dimensionality."""
    pca = fit_pca(H_b3, match_dim, seed=seed)
    H_b3_pca = apply_pca(H_b3, pca)

    def place_row(H: np.ndarray, tag: str) -> dict:
        real = ridge_multiclass_auc(
            H, y_place, alpha=alpha, seed=seed, min_per_class=min_per_class
        )
        null = ridge_multiclass_auc(
            H, y_place, alpha=alpha, seed=seed, min_per_class=min_per_class, shuffle_labels=True
        )
        auc = real.get("auc_macro", np.nan)
        null_auc = null.get("auc_macro", np.nan)
        delta = auc - null_auc if np.isfinite(auc) and np.isfinite(null_auc) else float("nan")
        return {
            "tag": tag,
            "dim": int(H.shape[1]),
            "real": real,
            "shuffled_null": null,
            "auc_macro": auc,
            "null_auc_macro": null_auc,
            "delta_above_null": delta,
        }

    return {
        "place_fc": place_row(H_fc, "fc"),
        "place_block3_full": place_row(H_b3, "block3_full"),
        "place_block3_pca256": place_row(H_b3_pca, "block3_pca256"),
        "pca": {"k": match_dim, "seed": seed},
        "class_counts": place_class_counts(y_place),
    }
