"""PCA, participation-ratio, and Huh effective-rank metrics on activation matrices."""

from __future__ import annotations

import numpy as np


def participation_ratio(singular_values: np.ndarray) -> float:
    """Classic PR on singular values of a centered activation matrix (λ = σ²)."""
    eig = singular_values ** 2
    return float((eig.sum() ** 2) / max((eig ** 2).sum(), 1e-12))


def participation_ratio_from_activations(x: np.ndarray) -> float:
    x_centered = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(x_centered, full_matrices=False)
    return participation_ratio(s)


def effective_rank_spectral_entropy(singular_values: np.ndarray) -> float:
    """Huh et al. Def. 2.1 (Roy & Vetterli): ρ(A) = −∑ σ̄ᵢ log(σ̄ᵢ) with σ̄ᵢ = σᵢ / ∑ⱼ σⱼ (no exp)."""
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    s = s[s > 0]
    if s.size == 0:
        return float("nan")
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())


def cosine_gram(x: np.ndarray) -> np.ndarray:
    """Cosine-kernel Gram on rows of x (Huh §2.3 embedding map)."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    xn = x / norms
    return xn @ xn.T


def huh_effective_rank_from_activations(
    x: np.ndarray,
    *,
    kernel: str = "cosine",
    max_samples: int | None = 512,
    seed: int = 0,
) -> dict:
    """Effective rank of the penultimate embedding, commensurable with Huh et al. 2023.

    ρ is the spectral entropy of a cosine Gram on (subsampled) rows; also reports PR on centered
    activations and ρ on the centered feature matrix itself."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2d activations, got {x.shape}")
    n = x.shape[0]
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        x = x[idx]
        n = max_samples

    x_c = x - x.mean(axis=0, keepdims=True)
    _u, s_feat, _vt = np.linalg.svd(x_c, full_matrices=False)
    pr = participation_ratio(s_feat)
    rho_feat = effective_rank_spectral_entropy(s_feat)

    if kernel == "cosine":
        K = cosine_gram(x)
    elif kernel == "linear":
        K = x_c @ x_c.T / max(x_c.shape[1], 1)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    s_gram = np.linalg.svd(K, compute_uv=False)
    rho_gram = effective_rank_spectral_entropy(s_gram)

    return {
        "n_samples": int(n),
        "dim": int(x.shape[1]),
        "kernel": kernel,
        "huh_effective_rank_gram": rho_gram,
        "huh_effective_rank_features": rho_feat,
        "participation_ratio": pr,
        "pc1_var_frac": float((s_feat[0] ** 2) / max((s_feat ** 2).sum(), 1e-12)),
    }


def difference_of_means_direction(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Unit class-conditional mean difference normalize(μ₁ − μ₀), a probe-free concept direction."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).ravel().astype(np.float64)
    pos = y > 0.5
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"need both classes; got n_pos={n_pos} n_neg={n_neg}")
    mu_pos = x[pos].mean(axis=0)
    mu_neg = x[neg].mean(axis=0)
    d = mu_pos - mu_neg
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        raise ValueError("class means coincide")
    return {
        "direction": d / nrm,
        "norm": nrm,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "mean_pos": mu_pos,
        "mean_neg": mu_neg,
    }


def whitened_difference_of_means_direction(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
) -> dict:
    """Unit LDA-style direction normalize((Σ + αI)^{-1} Δμ): the mean shift, variance-adjusted."""
    dom = difference_of_means_direction(x, y)
    h = np.asarray(x, dtype=np.float64)
    hc = h - h.mean(axis=0, keepdims=True)
    cov = (hc.T @ hc) / max(len(h) - 1, 1)
    delta = dom["mean_pos"] - dom["mean_neg"]
    w = np.linalg.solve(cov + alpha * np.eye(cov.shape[0]), delta)
    nrm = float(np.linalg.norm(w))
    if nrm < 1e-12:
        raise ValueError("whitened Dom vanished")
    return {
        "direction": w / nrm,
        "norm": nrm,
        "alpha": float(alpha),
        "n_pos": dom["n_pos"],
        "n_neg": dom["n_neg"],
        "mean_pos": dom["mean_pos"],
        "mean_neg": dom["mean_neg"],
    }


def fit_dom_directions(
    activations: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Raw Dom unit directions keyed by concept name."""
    out: dict[str, np.ndarray] = {}
    h = np.asarray(activations, dtype=np.float64)
    for concept, y in labels.items():
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.any(y > 0.5) or not np.any(y <= 0.5):
            out[concept] = np.zeros(h.shape[1], dtype=np.float64)
            continue
        out[concept] = difference_of_means_direction(h, y)["direction"]
    return out


def fit_lda_directions(
    activations: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """Whitened Dom / LDA-ish unit directions keyed by concept name."""
    out: dict[str, np.ndarray] = {}
    h = np.asarray(activations, dtype=np.float64)
    for concept, y in labels.items():
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.any(y > 0.5) or not np.any(y <= 0.5):
            out[concept] = np.zeros(h.shape[1], dtype=np.float64)
            continue
        out[concept] = whitened_difference_of_means_direction(h, y, alpha=alpha)["direction"]
    return out


def probe_pc_alignment(x: np.ndarray, probe_dirs: dict[str, np.ndarray], n_pcs: int = 10) -> dict:
    """|cos| of each probe direction with the top-k PCs of the centered activations.

    High alignment means the concept is prominent in variance; it does not imply the readout uses that PC."""
    x_c = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x_c, full_matrices=False)
    pcs = vt[:n_pcs]
    explained = (s[:n_pcs] ** 2) / max((s ** 2).sum(), 1e-12)
    out: dict = {"pc_explained_frac": explained.tolist(), "n_pcs": n_pcs}
    for name, w in probe_dirs.items():
        w = np.asarray(w, dtype=np.float64).ravel()
        w = w / max(np.linalg.norm(w), 1e-12)
        cos = np.abs(pcs @ w)
        out[name] = {
            "abs_cos_per_pc": cos.tolist(),
            "abs_cos_pc1": float(cos[0]),
            "energy_in_top_k": float((cos ** 2).sum()),  # since pcs orthonormal and w unit
            "argmax_pc": int(np.argmax(cos)) + 1,
        }
    return out


def pca_two_components(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_centered = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:2]
    projected = x_centered @ components.T
    explained = (s ** 2) / max(len(x) - 1, 1)
    explained_frac = explained / max(explained.sum(), 1e-12)
    return projected, explained_frac, s
