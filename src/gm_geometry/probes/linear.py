"""Ridge linear probes (numpy only — no sklearn dependency)."""

from __future__ import annotations

import numpy as np


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = x.mean(axis=0, keepdims=True)
    x_c = x - mu
    y_mu = float(y.mean())
    y_c = y - y_mu

    xt_x = x_c.T @ x_c
    reg = alpha * np.eye(x.shape[1], dtype=np.float64)
    w = np.linalg.solve(xt_x + reg, x_c.T @ y_c)
    b = y_mu - float((mu @ w.reshape(-1, 1)).squeeze())
    return w.astype(np.float32), b


def predict_ridge(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) @ w.reshape(-1)) + b


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = float(np.linalg.norm(y_true) * np.linalg.norm(y_pred)) + 1e-12
    return float((y_true @ y_pred) / denom)


def evaluate_probe(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> dict[str, float]:
    pred = predict_ridge(x, w, b)
    return {"r2": r2_score(y, pred), "pearson_r": pearson_r(y, pred)}


def roc_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC. Robust to bimodal/clustered targets."""
    y = np.asarray(y_binary).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(y_binary: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y = np.asarray(y_binary).reshape(-1)
    pred = (np.asarray(scores).reshape(-1) >= threshold).astype(int)
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    p = float((y == 1).sum())
    n = float((y == 0).sum())
    tpr = tp / p if p else float("nan")
    tnr = tn / n if n else float("nan")
    return 0.5 * (tpr + tnr)


def shuffle_labels_by_groups(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Permute constant-within-group labels across groups (correct exchangeability unit)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups, dtype=np.int32).reshape(-1)
    unique = np.unique(groups)
    group_vals = np.array([float(y[groups == g][0]) for g in unique])
    perm = group_vals.copy()
    rng.shuffle(perm)
    out = np.empty_like(y)
    for g, val in zip(unique, perm):
        out[groups == g] = val
    return out


def shuffle_null_ood_auc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float = 1.0,
    n_shuffles: int = 256,
    rng: np.random.RandomState | None = None,
) -> dict[str, float]:
    """OOD AUC null from group-level label permutations on the train grid."""
    if rng is None:
        rng = np.random.RandomState(0)
    null = []
    for _ in range(n_shuffles):
        y_sh = shuffle_labels_by_groups(y_train, groups, rng)
        w_s, b_s = fit_ridge(x_train, y_sh, alpha=alpha)
        null.append(roc_auc(y_ood, predict_ridge(x_ood, w_s, b_s)))
    null_arr = np.asarray(null, dtype=np.float64)
    real = roc_auc(y_ood, predict_ridge(x_ood, *fit_ridge(x_train, y_train, alpha=alpha)))
    return {
        "ood_auc": real,
        "shuffled_ood_auc": float(np.median(null_arr)),
        "shuffled_ood_auc_mean": float(null_arr.mean()),
        "shuffled_ood_auc_std": float(null_arr.std()),
        "shuffled_ood_auc_median": float(np.median(null_arr)),
        "shuffled_ood_auc_min": float(null_arr.min()),
        "shuffled_ood_auc_max": float(null_arr.max()),
        "shuffle_null_percentile": float((null_arr <= real).mean()),
        "n_shuffles": int(n_shuffles),
        "shuffle_unit": "group",
    }
