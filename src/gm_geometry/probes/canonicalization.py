"""Canonicalization (Canon) and transport-matrix metrics for state probes across appearance partitions."""
# Contexts are appearance partitions (unions of duel configs sharing how a feature is realized) with shared
# negatives, since presence labels are constant within a single config.

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from gm_geometry.probes.configs import STATE_PROBE_TRAIN_CONFIGS
from gm_geometry.probes.labels import ground_truth_presence
from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc


def _unit(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(w)) + 1e-12
    return w / n


def pairwise_mean_cosine(weights: list[np.ndarray]) -> float:
    units = [_unit(w) for w in weights]
    if len(units) < 2:
        return float("nan")
    acc = [float(units[i] @ units[j]) for i in range(len(units)) for j in range(i + 1, len(units))]
    return float(np.mean(acc))


def top_singular_alignment(weights: list[np.ndarray]) -> float:
    if not weights:
        return float("nan")
    P = np.stack([_unit(w) for w in weights], axis=0)
    s = np.linalg.svd(P / np.sqrt(P.shape[0]), compute_uv=False)
    return float(s[0])


def _config_label(gi: int, concept: str) -> float:
    obj1, obj2 = STATE_PROBE_TRAIN_CONFIGS[gi]
    return float(ground_truth_presence(obj1, obj2)[concept])


def appearance_partitions_yellow(n_configs: int | None = None) -> tuple[list[list[int]], list[str]]:
    """Yellow-on-line vs yellow-on-gem (shared negatives)."""
    n = n_configs or len(STATE_PROBE_TRAIN_CONFIGS)
    negs = [i for i in range(n) if _config_label(i, "yellow_present") == 0.0]
    on_line = []
    on_gem = []
    for i in range(n):
        if _config_label(i, "yellow_present") != 1.0:
            continue
        obj1, obj2 = STATE_PROBE_TRAIN_CONFIGS[i]
        objs = {obj1, obj2}
        if any(o.endswith("_line_diag") and o.startswith("yellow") for o in objs):
            on_line.append(i)
        elif any(o == "yellow_gem" for o in objs):
            on_gem.append(i)
    parts = []
    names = []
    if on_line and negs:
        parts.append(sorted(set(on_line + negs)))
        names.append("yellow_on_line")
    if on_gem and negs:
        parts.append(sorted(set(on_gem + negs)))
        names.append("yellow_on_gem")
    return parts, names


def appearance_partitions_line(n_configs: int | None = None) -> tuple[list[list[int]], list[str]]:
    """Line coloured yellow / red / green (shared negatives); tests paint-colour invariance, not partner context."""
    n = n_configs or len(STATE_PROBE_TRAIN_CONFIGS)
    negs = [i for i in range(n) if _config_label(i, "line_present") == 0.0]
    buckets: dict[str, list[int]] = {"yellow": [], "red": [], "green": [], "other": []}
    for i in range(n):
        if _config_label(i, "line_present") != 1.0:
            continue
        obj1, obj2 = STATE_PROBE_TRAIN_CONFIGS[i]
        line = next((o for o in (obj1, obj2) if o.endswith("_line_diag")), None)
        if line is None:
            continue
        colour = line.replace("_line_diag", "")
        buckets.get(colour, buckets["other"]).append(i)
    parts, names = [], []
    for colour in ("yellow", "red", "green"):
        if buckets[colour] and negs:
            parts.append(sorted(set(buckets[colour] + negs)))
            names.append(f"line_colour_{colour}")
    return parts, names


def appearance_partitions_line_by_partner(
    n_configs: int | None = None,
) -> tuple[list[list[int]], list[str]]:
    """Line present, bucketed by partner object (the non-line collectible), with shared negatives.

    On the 8-config train grid only green_line has two partners; yellow_line and red_line buckets are
    single-positive-config."""
    n = n_configs or len(STATE_PROBE_TRAIN_CONFIGS)
    negs = [i for i in range(n) if _config_label(i, "line_present") == 0.0]
    buckets: dict[str, list[int]] = {}
    for i in range(n):
        if _config_label(i, "line_present") != 1.0:
            continue
        obj1, obj2 = STATE_PROBE_TRAIN_CONFIGS[i]
        partner = next((o for o in (obj1, obj2) if not o.endswith("_line_diag")), None)
        if partner is None:
            continue
        buckets.setdefault(partner, []).append(i)
    parts, names = [], []
    for partner in sorted(buckets.keys()):
        if buckets[partner] and negs:
            parts.append(sorted(set(buckets[partner] + negs)))
            names.append(f"line_partner_{partner}")
    return parts, names


def appearance_partitions_line_by_partner_fixed_paint(
    n_configs: int | None = None,
    line_paint: str = "green",
) -> tuple[list[list[int]], list[str]]:
    """Fix the line's own paint colour and vary only the partner (``line_paint='green'`` gives two contexts)."""
    n = n_configs or len(STATE_PROBE_TRAIN_CONFIGS)
    negs = [i for i in range(n) if _config_label(i, "line_present") == 0.0]
    target = f"{line_paint}_line_diag"
    buckets: dict[str, list[int]] = {}
    for i in range(n):
        if _config_label(i, "line_present") != 1.0:
            continue
        obj1, obj2 = STATE_PROBE_TRAIN_CONFIGS[i]
        objs = {obj1, obj2}
        if target not in objs:
            continue
        partner = next(o for o in objs if o != target)
        buckets.setdefault(partner, []).append(i)
    parts, names = [], []
    for partner in sorted(buckets.keys()):
        if buckets[partner] and negs:
            parts.append(sorted(set(buckets[partner] + negs)))
            names.append(f"line_{line_paint}_partner_{partner}")
    return parts, names


# Default line axis is the partner object, not the line's paint colour.
APPEARANCE_PARTITIONERS: dict[str, Callable[..., tuple[list[list[int]], list[str]]]] = {
    "yellow_present": appearance_partitions_yellow,
    "line_present": appearance_partitions_line_by_partner,
}

# Alternative line context axes.
LINE_PARTITIONERS: dict[str, Callable[..., tuple[list[list[int]], list[str]]]] = {
    "by_partner": appearance_partitions_line_by_partner,
    "by_paint_colour": appearance_partitions_line,
    "by_partner_fixed_green": appearance_partitions_line_by_partner_fixed_paint,
}


def fit_partition_probes(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    partitions: list[list[int]],
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    activations = np.asarray(activations)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups, dtype=np.int32).reshape(-1)
    weights, biases, kept, skipped = [], [], [], []
    for part in partitions:
        mask = np.isin(groups, np.asarray(part, dtype=np.int32))
        y = labels[mask]
        if y.size == 0 or y.min() == y.max():
            skipped.append(part)
            continue
        w, b = fit_ridge(activations[mask], y, alpha=alpha)
        weights.append(w)
        biases.append(b)
        kept.append(part)
    return {"weights": weights, "biases": biases, "partitions": kept, "skipped": skipped}


def split_half_ceiling(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    partition: list[int],
    *,
    alpha: float = 1.0,
    rng: np.random.RandomState | None = None,
    n_tries: int = 8,
) -> float:
    if rng is None:
        rng = np.random.RandomState(0)
    mask = np.isin(groups, np.asarray(partition, dtype=np.int32))
    idx = np.where(mask)[0]
    if idx.size < 64:
        return float("nan")
    for _ in range(n_tries):
        perm = idx.copy()
        rng.shuffle(perm)
        mid = perm.size // 2
        a, b = perm[:mid], perm[mid:]
        if labels[a].min() == labels[a].max() or labels[b].min() == labels[b].max():
            continue
        wa, _ = fit_ridge(activations[a], labels[a], alpha=alpha)
        wb, _ = fit_ridge(activations[b], labels[b], alpha=alpha)
        return float(_unit(wa) @ _unit(wb))
    return float("nan")


def transport_matrix(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    partitions: list[list[int]],
    weights: list[np.ndarray],
    biases: list[float],
) -> np.ndarray:
    n = len(weights)
    T = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j, part in enumerate(partitions):
            mask = np.isin(groups, np.asarray(part, dtype=np.int32))
            y = labels[mask]
            if y.min() == y.max():
                continue
            T[i, j] = roc_auc(y, predict_ridge(activations[mask], weights[i], biases[i]))
    return T


def transport_flatness(T: np.ndarray) -> dict[str, float]:
    T = np.asarray(T, dtype=np.float64)
    n = T.shape[0]
    if n < 2:
        return {"diag_mean": float("nan"), "offdiag_mean": float("nan"), "ratio": float("nan")}
    diag = np.diag(T)
    off = T[~np.eye(n, dtype=bool)]
    diag = diag[np.isfinite(diag)]
    off = off[np.isfinite(off)]
    d = float(np.mean(diag)) if diag.size else float("nan")
    o = float(np.mean(off)) if off.size else float("nan")
    return {
        "diag_mean": d,
        "offdiag_mean": o,
        "ratio": (o / d) if d and np.isfinite(d) and d != 0 else float("nan"),
    }


def rho_from_transport(T: np.ndarray) -> float:
    """Cross-context coherence rho in [0, 1] from a feature's transport matrix.

    Uses AUC excess over chance (AUC - 0.5) because in-context AUC saturates near 1 for both canonical and
    starved features; rho = 0 means chance transport, rho = 1 means transport equals in-context performance."""
    T = np.asarray(T, dtype=np.float64)
    n = T.shape[0]
    if n < 2:
        return float("nan")
    diag = np.diag(T)
    off = T[~np.eye(n, dtype=bool)]
    diag = diag[np.isfinite(diag)]
    off = off[np.isfinite(off)]
    if diag.size == 0 or off.size == 0:
        return float("nan")
    diag_excess = float(np.mean(diag)) - 0.5
    off_excess = float(np.mean(off)) - 0.5
    if diag_excess <= 0:
        return float("nan")
    return float(np.clip(off_excess / diag_excess, 0.0, 1.0))


def predicted_colour_shape_ratio(n_contexts: int, rho: float) -> float:
    """Closed-form colour:shape effective-gain ratio C / (C*rho + 1 - rho); rho=0 gives C:1, rho=1 gives 1:1."""
    if not np.isfinite(rho):
        return float("nan")
    denom = n_contexts * rho + (1.0 - rho)
    return float(n_contexts / denom) if denom != 0 else float("nan")


def canon_report(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    concept: str,
    *,
    alpha: float = 1.0,
    rng: np.random.RandomState | None = None,
    line_partitioner: str = "by_partner",
) -> dict[str, Any]:
    """Canon and transport for one concept; ``line_partitioner`` picks the context axis for ``line_present``."""
    if rng is None:
        rng = np.random.RandomState(0)
    if concept == "line_present":
        if line_partitioner not in LINE_PARTITIONERS:
            return {"error": f"unknown line_partitioner {line_partitioner!r}"}
        partitions, names = LINE_PARTITIONERS[line_partitioner]()
    elif concept in APPEARANCE_PARTITIONERS:
        partitions, names = APPEARANCE_PARTITIONERS[concept]()
    else:
        return {"error": f"no appearance partitioner for {concept}"}
    fitted = fit_partition_probes(activations, labels, groups, partitions, alpha=alpha)
    weights, biases, kept = fitted["weights"], fitted["biases"], fitted["partitions"]
    kept_names = []
    for part in kept:
        for p, name in zip(partitions, names):
            if p == part:
                kept_names.append(name)
                break
        else:
            kept_names.append(str(part))
    if len(weights) < 2:
        return {
            "concept": concept,
            "n_partitions": len(weights),
            "partition_names": kept_names,
            "line_partitioner": line_partitioner if concept == "line_present" else None,
            "error": "fewer than 2 balanced appearance partitions",
            "skipped": fitted["skipped"],
        }
    ceilings = [
        split_half_ceiling(activations, labels, groups, part, alpha=alpha, rng=rng)
        for part in kept
    ]
    T = transport_matrix(activations, labels, groups, kept, weights, biases)
    flat = transport_flatness(T)
    mean_cos = pairwise_mean_cosine(weights)
    top_s = top_singular_alignment(weights)
    ceil = float(np.nanmean(ceilings))
    rho = rho_from_transport(T)
    return {
        "concept": concept,
        "n_partitions": len(weights),
        "partition_names": kept_names,
        "line_partitioner": line_partitioner if concept == "line_present" else None,
        "effective_C": len(weights),
        "canon_mean_cosine": mean_cos,
        "canon_top_singular": top_s,
        "split_half_ceiling_mean": ceil,
        "canon_mean_cosine_rel_ceiling": (
            mean_cos / ceil if ceil and np.isfinite(ceil) and ceil != 0 else float("nan")
        ),
        "transport": flat,
        "transport_matrix": T.tolist(),
        "rho_hat": rho,
        "split_half_ceilings": ceilings,
    }
