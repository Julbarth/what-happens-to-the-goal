"""Compress a family of state-local d_comp(s) vectors to one shared axis via a two-bucket mean difference."""
from __future__ import annotations

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def fit_u_comp(d_comp_by_bucket: dict[object, list[np.ndarray]]) -> np.ndarray:
    """u_comp = normalize(mean(bucket_A) - mean(bucket_B)) for a two-bucket split.

    A mean difference rather than SVD: SVD finds the pooled max-variance direction, which can differ from the
    bucket contrast. Requires exactly 2 buckets with at least one vector each."""
    keys = list(d_comp_by_bucket.keys())
    if len(keys) != 2:
        raise ValueError(f"fit_u_comp expects exactly 2 buckets, got {len(keys)}: {keys}")
    a, b = keys
    mean_a = np.mean(np.stack(d_comp_by_bucket[a], axis=0), axis=0)
    mean_b = np.mean(np.stack(d_comp_by_bucket[b], axis=0), axis=0)
    return unit(mean_a - mean_b)


def held_out_alignment(
    u_comp: np.ndarray,
    held_out_d_comp: list[np.ndarray],
    held_out_sign: list[float],
) -> dict:
    """cos(u_comp, sign(s) * d_comp(s)) on held-out states not used to fit u_comp."""
    cos = []
    for d, sgn in zip(held_out_d_comp, held_out_sign):
        du = unit(np.asarray(d, dtype=np.float64))
        cos.append(float(sgn) * float(u_comp @ du))
    arr = np.asarray(cos, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "frac_pos": float((arr > 0).mean()) if arr.size else float("nan"),
        "per_state_cos": arr.tolist(),
    }


def bucket_permutation_null(
    fit_d_comp: list[np.ndarray],
    fit_bucket_labels: list[object],
    held_out_d_comp: list[np.ndarray],
    held_out_true_bucket: list[object],
    *,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """Permutation null for the bucket partition: refit u_comp on ``n_perm`` same-sized random relabelings.

    Each refit is scored on the held-out bank with the true buckets; reports p = (1 + n_exceed) / (1 + n_perm)."""
    rng = np.random.default_rng(seed)
    fit_d = [np.asarray(d, dtype=np.float64) for d in fit_d_comp]
    labels = np.asarray(fit_bucket_labels, dtype=object)
    keys = sorted(set(labels.tolist()), key=str)
    if len(keys) != 2:
        raise ValueError(f"expects exactly 2 buckets in fit_bucket_labels, got {keys}")
    a_key, b_key = keys
    n = len(fit_d)
    sizes = {a_key: int((labels == a_key).sum()), b_key: int((labels == b_key).sum())}

    def _score(bucket_of_fit_idx: dict[int, object]) -> float:
        d_by_bucket: dict[object, list[np.ndarray]] = {a_key: [], b_key: []}
        for i, d in enumerate(fit_d):
            d_by_bucket[bucket_of_fit_idx[i]].append(d)
        u = fit_u_comp(d_by_bucket)
        sign = [1.0 if b == a_key else -1.0 for b in held_out_true_bucket]
        res = held_out_alignment(u, held_out_d_comp, sign)
        return res["mean"]

    true_bucket_of_idx = {i: labels[i] for i in range(n)}
    true_score = _score(true_bucket_of_idx)

    perm_scores = np.empty(n_perm, dtype=np.float64)
    idx_pool = np.arange(n)
    for p in range(n_perm):
        shuffled = rng.permutation(idx_pool)
        a_idx = set(shuffled[: sizes[a_key]].tolist())
        perm_bucket_of_idx = {i: (a_key if i in a_idx else b_key) for i in range(n)}
        perm_scores[p] = _score(perm_bucket_of_idx)

    n_exceed = int((perm_scores >= true_score).sum())
    return {
        "true_score": float(true_score),
        "n_perm": n_perm,
        "n_exceed": n_exceed,
        "p_upper_bound": float((1 + n_exceed) / (1 + n_perm)),
        "perm_score_mean": float(perm_scores.mean()),
        "perm_score_std": float(perm_scores.std()),
    }


def fisher_projection_alpha(u_comp: np.ndarray, f_state: np.ndarray, d_comp_state: np.ndarray) -> float:
    """alpha_F(s) = (u_comp^T F_s d_comp(s)) / (u_comp^T F_s u_comp)."""
    denom = float(u_comp @ f_state @ u_comp)
    if denom < 1e-18:
        return 0.0
    return float(u_comp @ f_state @ d_comp_state) / denom


def matched_norm_random_direction(norm: float, dim: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.normal(size=dim)
    r = r / (float(np.linalg.norm(r)) + 1e-18)
    return r * norm


def bucket_outcome_distribution(outcomes: list[str], categories: tuple[str, ...]) -> dict[str, float]:
    n = len(outcomes)
    return {c: float(np.mean([o == c for o in outcomes])) for c in categories} if n else {}
