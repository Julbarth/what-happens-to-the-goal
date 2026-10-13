"""Project linear probe directions onto policy / value readout weights."""

from __future__ import annotations

import numpy as np

from gm_geometry.pca.metrics import fit_dom_directions, fit_lda_directions
from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc, shuffle_labels_by_groups


def unit_vector(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return w / (float(np.linalg.norm(w)) + 1e-12)


def probe_norm(w: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(w, dtype=np.float64).reshape(-1)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit_vector(a) @ unit_vector(b))


def residualize(w: np.ndarray, w_ref: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    r = unit_vector(w_ref)
    return w - (w @ r) * r


def weight_loading(W: np.ndarray, w: np.ndarray) -> float:
    """``||W û||`` where ``û`` is the unit probe direction (comparable to isotropic null)."""
    u = unit_vector(w)
    return float(np.linalg.norm(W @ u))


def weight_loading_raw(W: np.ndarray, w: np.ndarray) -> float:
    """``||W w||`` using the raw ridge weight vector (natural probe scale)."""
    return float(np.linalg.norm(W @ np.asarray(w, dtype=np.float64).reshape(-1)))


def activation_swing_variance(h: np.ndarray, w: np.ndarray, W: np.ndarray) -> float:
    """Var_i( (h_i·û) * ||W û|| ) — unit-direction readout swing."""
    u = unit_vector(w)
    proj = h @ u
    scale = weight_loading(W, w)
    return float(np.var(proj * scale))


def activation_swing_variance_raw(h: np.ndarray, w: np.ndarray, W: np.ndarray) -> float:
    """Var_i( (h_i·w) * ||W w|| ) — natural-scale readout swing."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    proj = h @ w
    scale = weight_loading_raw(W, w)
    return float(np.var(proj * scale))


def value_swing_variance(h: np.ndarray, w: np.ndarray, w_value: np.ndarray) -> float:
    u = unit_vector(w)
    w_value = np.asarray(w_value, dtype=np.float64).reshape(-1)
    coef = float(w_value @ u)
    return float(np.var((h @ u) * coef))


def value_swing_variance_raw(h: np.ndarray, w: np.ndarray, w_value: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w_value = np.asarray(w_value, dtype=np.float64).reshape(-1)
    return float(np.var((h @ w) * float(w_value @ w)))


def percentile_against_null(score: float, null_scores: np.ndarray) -> float:
    null_scores = np.asarray(null_scores, dtype=np.float64)
    if null_scores.size == 0:
        return float("nan")
    return float((null_scores <= score).mean())


def isotropic_null_loadings(W: np.ndarray, *, n_dirs: int, rng: np.random.RandomState) -> np.ndarray:
    dim = W.shape[1]
    out = np.empty(n_dirs, dtype=np.float64)
    for i in range(n_dirs):
        v = rng.randn(dim)
        v /= np.linalg.norm(v) + 1e-12
        out[i] = weight_loading(W, v)
    return out


def shuffle_probe_weights(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.RandomState,
    *,
    alpha: float = 1.0,
    n_shuffles: int = 32,
    groups: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Null probe directions from shuffled labels.

    With ``groups`` set, labels are permuted across config groups (the exchangeability unit when labels are
    constant within groups); otherwise frames are shuffled."""
    weights = []
    y_base = np.asarray(y, dtype=np.float64).reshape(-1)
    for _ in range(n_shuffles):
        if groups is not None:
            y_sh = shuffle_labels_by_groups(y_base, groups, rng)
        else:
            y_sh = y_base.copy()
            rng.shuffle(y_sh)
        w_sh, _ = fit_ridge(x, y_sh, alpha=alpha)
        weights.append(w_sh)
    return weights


def fit_object_salience_probe(
    train_act: np.ndarray,
    train_pixel: dict[str, np.ndarray],
    *,
    alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    salience = np.maximum(train_pixel["yellow"], train_pixel["gem"]).astype(np.float64)
    return fit_ridge(train_act, salience, alpha=alpha)


def _loading_block(
    w: np.ndarray,
    W_policy: np.ndarray,
    W_value: np.ndarray,
    h_train: np.ndarray,
    h_ood: np.ndarray,
) -> dict:
    wv = W_value.reshape(-1)
    return {
        "probe_norm": probe_norm(w),
        "policy_weight_loading": weight_loading(W_policy, w),
        "value_weight_loading": weight_loading(W_value.reshape(1, -1), w),
        "policy_weight_loading_raw": weight_loading_raw(W_policy, w),
        "value_weight_loading_raw": weight_loading_raw(W_value.reshape(1, -1), w),
        "policy_swing_var_train": activation_swing_variance(h_train, w, W_policy),
        "policy_swing_var_ood": activation_swing_variance(h_ood, w, W_policy),
        "value_swing_var_train": value_swing_variance(h_train, w, wv),
        "value_swing_var_ood": value_swing_variance(h_ood, w, wv),
        "policy_swing_var_train_raw": activation_swing_variance_raw(h_train, w, W_policy),
        "policy_swing_var_ood_raw": activation_swing_variance_raw(h_ood, w, W_policy),
        "value_swing_var_train_raw": value_swing_variance_raw(h_train, w, wv),
        "value_swing_var_ood_raw": value_swing_variance_raw(h_ood, w, wv),
    }


def score_probe_on_head(
    w: np.ndarray,
    W_policy: np.ndarray,
    W_value: np.ndarray,
    h_train: np.ndarray,
    h_ood: np.ndarray,
    *,
    shuffle_weights: list[np.ndarray] | None = None,
    iso_policy: np.ndarray | None = None,
    iso_value: np.ndarray | None = None,
) -> dict:
    block = _loading_block(w, W_policy, W_value, h_train, h_ood)
    if iso_policy is not None:
        block["policy_iso_null_percentile"] = percentile_against_null(
            block["policy_weight_loading"], iso_policy,
        )
    if iso_value is not None:
        block["value_iso_null_percentile"] = percentile_against_null(
            block["value_weight_loading"], iso_value,
        )
    if shuffle_weights:
        pol_sh = np.asarray([weight_loading(W_policy, ws) for ws in shuffle_weights])
        val_sh = np.asarray([weight_loading(W_value.reshape(1, -1), ws) for ws in shuffle_weights])
        block["policy_shuffle_percentile"] = percentile_against_null(
            block["policy_weight_loading"], pol_sh,
        )
        block["value_shuffle_percentile"] = percentile_against_null(
            block["value_weight_loading"], val_sh,
        )
        block["policy_shuffle_mean"] = float(pol_sh.mean())
        block["value_shuffle_mean"] = float(val_sh.mean())
    return block


def fused_and_difference(w_yellow: np.ndarray, w_line: np.ndarray) -> dict[str, np.ndarray]:
    """Unit fused ``ŝ = û_y+û_l`` and difference ``d̂ = û_y−û_l`` axes."""
    u_y, u_l = unit_vector(w_yellow), unit_vector(w_line)
    return {
        "yellow": u_y,
        "line": u_l,
        "fused_s": unit_vector(u_y + u_l),
        "diff_d": unit_vector(u_y - u_l),
    }


def readout_row_basis(W_policy: np.ndarray, W_value: np.ndarray) -> np.ndarray:
    """Orthonormal basis for row-span of ``W_π ∪ {w_v}`` (readout subspace R)."""
    Wv = np.asarray(W_value, dtype=np.float64).reshape(1, -1)
    stacked = np.vstack([np.asarray(W_policy, dtype=np.float64), Wv])
    _u, s, vt = np.linalg.svd(stacked, full_matrices=False)
    rank = int(np.sum(s > 1e-8 * s[0]))
    return vt[:rank]


def project_onto_basis(w: np.ndarray, basis: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return basis.T @ (basis @ w)


def fraction_in_readout(w: np.ndarray, basis_R: np.ndarray) -> float:
    """‖P_R w‖² / ‖w‖² — how much of the probe lives in the policy/value row-span."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    proj = project_onto_basis(w, basis_R)
    return float(np.dot(proj, proj) / (np.dot(w, w) + 1e-12))


def occupied_pc_basis(h: np.ndarray, *, energy: float = 0.95) -> tuple[np.ndarray, dict]:
    """Orthonormal PC basis spanning the energy-$energy$ subspace of centered activations.

    Returns ``(U, meta)`` with ``U`` shape ``(m, d)`` (rows = PCs) and meta for logging.
    """
    h = np.asarray(h, dtype=np.float64)
    h_c = h - h.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(h_c, full_matrices=False)
    e = s ** 2
    cum = np.cumsum(e) / max(e.sum(), 1e-12)
    m = int(np.searchsorted(cum, energy) + 1)
    m = max(m, 1)
    return vt[:m], {
        "energy": float(energy),
        "occupied_m": m,
        "pc_explained_cum_at_m": float(cum[m - 1]),
        "dim": int(h.shape[1]),
        "n_samples": int(h.shape[0]),
    }


def chance_frac_in_subspace(basis_R: np.ndarray, U: np.ndarray) -> float:
    """E[‖P_R v‖²] for unit v uniform in span(U). = ‖basis_R Uᵀ‖_F² / m."""
    m = U.shape[0]
    M = np.asarray(basis_R, dtype=np.float64) @ U.T
    return float(np.sum(M ** 2) / max(m, 1))


def frac_R_within_subspace(w: np.ndarray, basis_R: np.ndarray, U: np.ndarray) -> float:
    """frac_R of the unit direction of w after projection into S = span(U)."""
    w = np.asarray(w, dtype=np.float64).ravel()
    w_S = U.T @ (U @ w)
    n = np.linalg.norm(w_S)
    if n < 1e-12:
        return float("nan")
    return fraction_in_readout(w_S / n, basis_R)


def frac_R_effective_null_report(
    probes: dict[str, np.ndarray],
    basis_R: np.ndarray,
    h_train: np.ndarray,
    *,
    energy: float = 0.95,
    n_null: int = 2048,
    rng: np.random.RandomState | None = None,
    names: tuple[str, ...] = ("yellow", "line", "fused_s", "diff_d"),
) -> dict:
    """Ambient frac_R plus the S_energy null (analytic chance and Monte Carlo percentile).

    Compare obs_in_S to chance_analytic_S; ambient below chance does not by itself indicate avoidance."""
    rng = rng or np.random.RandomState(0)
    U, meta = occupied_pc_basis(h_train, energy=energy)
    ambient_chance = float(basis_R.shape[0] / meta["dim"])
    chance_S = chance_frac_in_subspace(basis_R, U)
    m = U.shape[0]

    # Monte Carlo null dirs in S
    A = rng.normal(size=(n_null, m))
    A /= np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)
    V = A @ U
    null_fracs = np.array([fraction_in_readout(v, basis_R) for v in V])

    per_probe = {}
    for name in names:
        if name not in probes:
            continue
        w = probes[name]
        ambient = fraction_in_readout(w, basis_R)
        obs_S = frac_R_within_subspace(w, basis_R, U)
        per_probe[name] = {
            "frac_R_ambient": ambient,
            "delta_vs_ambient_chance": float(ambient - ambient_chance),
            "frac_R_in_S": obs_S,
            "chance_analytic_S": chance_S,
            "delta_vs_S_chance": float(obs_S - chance_S) if np.isfinite(obs_S) else float("nan"),
            "percentile_in_S": float(np.mean(null_fracs <= obs_S)) if np.isfinite(obs_S) else float("nan"),
            "null_S_mean": float(null_fracs.mean()),
            "null_S_std": float(null_fracs.std()),
        }

    n_axes = len(per_probe)
    below_amb = sum(1 for v in per_probe.values() if v["frac_R_ambient"] < ambient_chance)
    below_S = sum(
        1 for v in per_probe.values()
        if np.isfinite(v["frac_R_in_S"]) and v["frac_R_in_S"] < v["chance_analytic_S"]
    )
    return {
        "null": "S_energy PC subspace (effective-dim), not ambient R^d",
        "energy": float(energy),
        "occupied_m": meta["occupied_m"],
        "pc_explained_cum_at_m": meta["pc_explained_cum_at_m"],
        "rank_R": int(basis_R.shape[0]),
        "ambient_chance": ambient_chance,
        "chance_analytic_S": chance_S,
        "n_null": int(n_null),
        "probes": per_probe,
        "n_axes_below_ambient_chance": below_amb,
        "n_axes_below_S_chance": below_S,
        "n_axes": n_axes,
        "interpretation": (
            "Primary: frac_R_in_S vs chance_analytic_S / percentile_in_S. "
            "Ambient below-chance is continuity only — not avoidance."
        ),
    }


def fit_probe_bundle(
    h_train: np.ndarray,
    h_ood: np.ndarray,
    y_train: dict[str, np.ndarray],
    y_ood: dict[str, np.ndarray],
    train_pixel: dict[str, np.ndarray],
    *,
    alpha: float = 1.0,
) -> dict:
    w_y, b_y = fit_ridge(h_train, y_train["yellow_present"], alpha=alpha)
    w_l, b_l = fit_ridge(h_train, y_train["line_present"], alpha=alpha)
    w_sal, _ = fit_object_salience_probe(h_train, train_pixel, alpha=alpha)
    salience = np.maximum(train_pixel["yellow"], train_pixel["gem"]).astype(np.float64)
    axes = fused_and_difference(w_y, w_l)
    return {
        "yellow": w_y,
        "line": w_l,
        "line_residualized": residualize(w_l, w_y),
        "yellow_residualized": residualize(w_y, w_l),
        "fused_s": axes["fused_s"],
        "diff_d": axes["diff_d"],
        "object_salience": w_sal,
        "salience_target": salience,
        "ood_auc": {
            "yellow": roc_auc(y_ood["yellow_present"], predict_ridge(h_ood, w_y, b_y)),
            "line": roc_auc(y_ood["line_present"], predict_ridge(h_ood, w_l, b_l)),
        },
        "cos_yellow_line": cosine_similarity(w_y, w_l),
        "probe_norm_yellow": probe_norm(w_y),
        "probe_norm_line": probe_norm(w_l),
    }


def _ratio(yellow_val: float, line_val: float) -> float:
    return float(yellow_val / (line_val + 1e-12))


# ---------------------------------------------------------------------------
# Environment-agnostic generic path (Coin / Maze I / colour-shape)
# ---------------------------------------------------------------------------
# ``projection_report`` above is the colour-shape reference. The generic functions below reproduce it
# exactly for (proxy, goal) = (yellow, line) while also supporting any proxy/goal names and single-concept
# envs, for which the fused_s / diff_d axes and proxy-over-goal ratios are omitted.


def fit_probe_bundle_generic(
    h_train: np.ndarray,
    h_ood: np.ndarray,
    concepts_train: dict[str, np.ndarray],
    concepts_ood: dict[str, np.ndarray],
    *,
    proxy_goal: tuple[str, str] | None,
    salience_target: np.ndarray | None = None,
    alpha: float = 1.0,
) -> dict:
    """Ridge probe bundle for arbitrary concept names.

    Two-concept (proxy, goal) envs additionally get residualized axes plus the
    fused ``ŝ`` / difference ``d̂`` pair. Single-concept envs get only the ridge
    direction, its OOD AUC, and norm.
    """
    biases: dict[str, float] = {}
    weights: dict[str, np.ndarray] = {}
    for name, y in concepts_train.items():
        w, b = fit_ridge(h_train, y, alpha=alpha)
        weights[name] = w
        biases[name] = b

    out: dict[str, object] = dict(weights)
    ood_auc = {
        name: roc_auc(concepts_ood[name], predict_ridge(h_ood, weights[name], biases[name]))
        for name in weights
    }
    out["ood_auc"] = ood_auc
    for name, w in weights.items():
        out[f"probe_norm_{name}"] = probe_norm(w)

    if proxy_goal is not None:
        p, g = proxy_goal
        axes = fused_and_difference(weights[p], weights[g])
        out[f"{g}_residualized"] = residualize(weights[g], weights[p])
        out[f"{p}_residualized"] = residualize(weights[p], weights[g])
        out["fused_s"] = axes["fused_s"]
        out["diff_d"] = axes["diff_d"]
        out[f"cos_{p}_{g}"] = cosine_similarity(weights[p], weights[g])

    if salience_target is not None:
        w_sal, _ = fit_ridge(h_train, np.asarray(salience_target, dtype=np.float64), alpha=alpha)
        out["object_salience"] = w_sal
        out["salience_target"] = np.asarray(salience_target, dtype=np.float64)

    return out


def projection_report_generic(
    h_train: np.ndarray,
    h_ood: np.ndarray,
    concepts_train: dict[str, np.ndarray],
    concepts_ood: dict[str, np.ndarray],
    W_policy: np.ndarray,
    W_value: np.ndarray,
    rng: np.random.RandomState,
    *,
    proxy_goal: tuple[str, str] | None,
    salience_target: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    alpha: float = 1.0,
    n_isotropic: int = 512,
    n_shuffle: int = 256,
) -> dict:
    """Env-agnostic head-projection report; matches :func:`projection_report` key-for-key for a (proxy, goal) pair.

    ``proxy_goal=None`` yields the single-concept report."""
    probes = fit_probe_bundle_generic(
        h_train, h_ood, concepts_train, concepts_ood,
        proxy_goal=proxy_goal, salience_target=salience_target, alpha=alpha,
    )
    concept_order = list(concepts_train.keys())

    # rng draw order matches the colour-shape reference:
    #   iso_pi, iso_val, shuffle(proxy), shuffle(goal), shuffle(salience),
    #   paired (proxy,goal) s/d shuffles, then frac_R MC null.
    iso_pi = isotropic_null_loadings(W_policy, n_dirs=n_isotropic, rng=rng)
    iso_val = isotropic_null_loadings(W_value.reshape(1, -1), n_dirs=n_isotropic, rng=rng)

    shuffle_for: dict[str, list[np.ndarray]] = {}
    per_concept_shuffles: dict[str, list[np.ndarray]] = {}
    for name in concept_order:
        per_concept_shuffles[name] = shuffle_probe_weights(
            h_train, concepts_train[name], rng, alpha=alpha, n_shuffles=n_shuffle, groups=groups,
        )
        shuffle_for[name] = per_concept_shuffles[name]

    shuf_sal: list[np.ndarray] | None = None
    if salience_target is not None:
        shuf_sal = shuffle_probe_weights(
            h_train, probes["salience_target"], rng, alpha=alpha, n_shuffles=n_shuffle, groups=None,
        )
        shuffle_for["object_salience"] = shuf_sal

    scored_names = list(concept_order)
    if proxy_goal is not None:
        p, g = proxy_goal
        shuffle_for[f"{g}_residualized"] = per_concept_shuffles[g]
        shuffle_for[f"{p}_residualized"] = per_concept_shuffles[p]
        shuf_s: list[np.ndarray] = []
        shuf_d: list[np.ndarray] = []
        p_base = np.asarray(concepts_train[p], dtype=np.float64).reshape(-1)
        g_base = np.asarray(concepts_train[g], dtype=np.float64).reshape(-1)
        for _ in range(n_shuffle):
            if groups is not None:
                p_sh = shuffle_labels_by_groups(p_base, groups, rng)
                g_sh = shuffle_labels_by_groups(g_base, groups, rng)
            else:
                p_sh, g_sh = p_base.copy(), g_base.copy()
                rng.shuffle(p_sh)
                rng.shuffle(g_sh)
            w_ps, _ = fit_ridge(h_train, p_sh, alpha=alpha)
            w_gs, _ = fit_ridge(h_train, g_sh, alpha=alpha)
            axes_sh = fused_and_difference(w_ps, w_gs)
            shuf_s.append(axes_sh["fused_s"])
            shuf_d.append(axes_sh["diff_d"])
        shuffle_for["fused_s"] = shuf_s
        shuffle_for["diff_d"] = shuf_d
        scored_names += [f"{g}_residualized", f"{p}_residualized", "fused_s", "diff_d"]
    if salience_target is not None:
        scored_names.append("object_salience")

    concepts: dict[str, dict] = {}
    for name in scored_names:
        concepts[name] = score_probe_on_head(
            probes[name], W_policy, W_value, h_train, h_ood,
            shuffle_weights=shuffle_for.get(name),
            iso_policy=iso_pi,
            iso_value=iso_val,
        )

    basis_R = readout_row_basis(W_policy, W_value)
    frac_names = list(concept_order)
    if proxy_goal is not None:
        frac_names += ["fused_s", "diff_d"]
    readout_frac = {name: fraction_in_readout(probes[name], basis_R) for name in frac_names}
    frac_R_effective = frac_R_effective_null_report(
        {name: probes[name] for name in frac_names},
        basis_R,
        h_train,
        energy=0.95,
        n_null=2048,
        rng=rng,
        names=tuple(frac_names),
    )

    dim = W_policy.shape[1]
    report: dict[str, object] = {
        "n_shuffle": n_shuffle,
        "shuffle_unit": "group" if groups is not None else "frame",
        "concepts": concepts,
        "readout_fraction": readout_frac,
        "rank_R": int(basis_R.shape[0]),
        "expected_random_frac_R": float(basis_R.shape[0] / dim),
        "readout_fraction_delta_vs_random": {
            name: float(frac - basis_R.shape[0] / dim) for name, frac in readout_frac.items()
        },
        "frac_R_effective": frac_R_effective,
        "isotropic_policy_null_mean": float(iso_pi.mean()),
        "isotropic_policy_null_std": float(iso_pi.std()),
        "concept_names": concept_order,
        "proxy_goal": list(proxy_goal) if proxy_goal is not None else None,
    }
    for name in concept_order:
        report[f"ood_auc_{name}"] = probes["ood_auc"][name]
        report[f"probe_norm_{name}"] = probes[f"probe_norm_{name}"]

    if proxy_goal is not None:
        p, g = proxy_goal
        report[f"cos_{p}_{g}"] = probes[f"cos_{p}_{g}"]
        report["residualization_near_noop"] = abs(probes[f"cos_{p}_{g}"]) < 0.15
        report["shuffle_unit_" + f"{p}_{g}"] = "group" if groups is not None else "frame"
        report[f"ratio_value_{p}_over_{g}"] = _ratio(
            concepts[p]["value_weight_loading"], concepts[g]["value_weight_loading"],
        )
        report[f"ratio_policy_{p}_over_{g}"] = _ratio(
            concepts[p]["policy_weight_loading"], concepts[g]["policy_weight_loading"],
        )
        report[f"ratio_value_{p}_over_{g}_residualized"] = _ratio(
            concepts[p]["value_weight_loading"], concepts[f"{g}_residualized"]["value_weight_loading"],
        )
        report[f"ratio_policy_{p}_over_{g}_residualized"] = _ratio(
            concepts[p]["policy_weight_loading"], concepts[f"{g}_residualized"]["policy_weight_loading"],
        )
        report[f"ratio_value_{p}_over_{g}_raw"] = _ratio(
            concepts[p]["value_weight_loading_raw"], concepts[g]["value_weight_loading_raw"],
        )
        report[f"ratio_policy_{p}_over_{g}_raw"] = _ratio(
            concepts[p]["policy_weight_loading_raw"], concepts[g]["policy_weight_loading_raw"],
        )
        report["ratio_policy_s_over_d"] = _ratio(
            concepts["fused_s"]["policy_weight_loading"], concepts["diff_d"]["policy_weight_loading"],
        )
        report["ratio_value_s_over_d"] = _ratio(
            concepts["fused_s"]["value_weight_loading"], concepts["diff_d"]["value_weight_loading"],
        )
        fused_value_iso = concepts["fused_s"]["value_iso_null_percentile"]
        report["value_head_positive_control_fused_s_iso_percentile"] = fused_value_iso
        report["value_head_positive_control_fused_s_ok"] = fused_value_iso >= 0.5

    if salience_target is not None:
        sal_value_iso = concepts["object_salience"]["value_iso_null_percentile"]
        report["value_head_positive_control_salience_iso_percentile"] = sal_value_iso
        report["value_head_positive_control_salience_ok"] = sal_value_iso >= 0.5

    # Dom / LDA run after all ridge RNG draws so earlier numbers are unchanged.
    attach_dom_lda_projection(
        report, h_train, concepts_train, W_policy, W_value, rng,
        h_ood=h_ood, concepts_ood=concepts_ood, lda_alpha=alpha,
    )
    return report


def attach_dom_lda_projection(
    report: dict,
    h_train: np.ndarray,
    concepts_train: dict[str, np.ndarray],
    W_policy: np.ndarray,
    W_value: np.ndarray,
    rng: np.random.RandomState,
    *,
    h_ood: np.ndarray | None = None,
    concepts_ood: dict[str, np.ndarray] | None = None,
    lda_alpha: float = 1.0,
    n_null: int = 2048,
) -> dict:
    """Append raw Dom and whitened Dom (LDA) axes, with ambient and S_95 ``frac_R`` and AUCs, to a report.

    Runs after the ridge/shuffle/isotropic draws so their RNG stream is unchanged."""
    dom = fit_dom_directions(h_train, concepts_train)
    lda = fit_lda_directions(h_train, concepts_train, alpha=lda_alpha)
    named: dict[str, np.ndarray] = {}
    for c, u in dom.items():
        named[f"dom_{c}"] = u
    for c, u in lda.items():
        named[f"lda_{c}"] = u

    basis_R = readout_row_basis(W_policy, W_value)
    readout_frac = {name: fraction_in_readout(u, basis_R) for name, u in named.items()}
    frac_eff = frac_R_effective_null_report(
        named, basis_R, h_train, energy=0.95, n_null=n_null, rng=rng, names=tuple(named.keys()),
    )

    aucs: dict[str, dict[str, float]] = {}
    for c, y_tr in concepts_train.items():
        y_tr = np.asarray(y_tr, dtype=np.float64).ravel()
        scores_dom = h_train @ named[f"dom_{c}"]
        scores_lda = h_train @ named[f"lda_{c}"]
        row = {
            "dom_train_auc": float(roc_auc(y_tr, scores_dom)),
            "lda_train_auc": float(roc_auc(y_tr, scores_lda)),
        }
        if h_ood is not None and concepts_ood is not None and c in concepts_ood:
            y_od = np.asarray(concepts_ood[c], dtype=np.float64).ravel()
            row["dom_ood_auc"] = float(roc_auc(y_od, h_ood @ named[f"dom_{c}"]))
            row["lda_ood_auc"] = float(roc_auc(y_od, h_ood @ named[f"lda_{c}"]))
        aucs[c] = row

    # Merge into existing frac_R tables without renaming ridge keys.
    report.setdefault("readout_fraction", {}).update(readout_frac)
    fe = report.setdefault("frac_R_effective", {})
    fe.setdefault("probes", {}).update(frac_eff.get("probes", {}))
    report["dom_lda"] = {
        "estimators": ["dom", "lda"],
        "axes": list(named.keys()),
        "readout_fraction": readout_frac,
        "frac_R_effective": frac_eff,
        "auc": aucs,
        "note": (
            "Raw Dom = loud / weak colour decoder (illusion-control axis). "
            "Whitened Dom (LDA) = quiet discriminative twin of ridge. "
            "Attached after ridge path; does not disturb ridge RNG draws."
        ),
    }
    return report


def projection_report(
    h_train: np.ndarray,
    h_ood: np.ndarray,
    y_train: dict[str, np.ndarray],
    y_ood: dict[str, np.ndarray],
    train_pixel: dict[str, np.ndarray],
    W_policy: np.ndarray,
    W_value: np.ndarray,
    rng: np.random.RandomState,
    *,
    groups: np.ndarray | None = None,
    alpha: float = 1.0,
    n_isotropic: int = 512,
    n_shuffle: int = 256,
) -> dict:
    probes = fit_probe_bundle(h_train, h_ood, y_train, y_ood, train_pixel, alpha=alpha)
    iso_pi = isotropic_null_loadings(W_policy, n_dirs=n_isotropic, rng=rng)
    iso_val = isotropic_null_loadings(W_value.reshape(1, -1), n_dirs=n_isotropic, rng=rng)

    shuf_y = shuffle_probe_weights(
        h_train, y_train["yellow_present"], rng, alpha=alpha, n_shuffles=n_shuffle, groups=groups,
    )
    shuf_l = shuffle_probe_weights(
        h_train, y_train["line_present"], rng, alpha=alpha, n_shuffles=n_shuffle, groups=groups,
    )
    shuf_sal = shuffle_probe_weights(
        h_train, probes["salience_target"], rng, alpha=alpha, n_shuffles=n_shuffle, groups=None,
    )
    # Paired group-shuffles of (yellow, line) give null ŝ / d̂ axes with the same algebraic
    # geometry, a stronger null than isotropic directions.
    shuf_s: list[np.ndarray] = []
    shuf_d: list[np.ndarray] = []
    y_base = np.asarray(y_train["yellow_present"], dtype=np.float64).reshape(-1)
    l_base = np.asarray(y_train["line_present"], dtype=np.float64).reshape(-1)
    for _ in range(n_shuffle):
        if groups is not None:
            y_sh = shuffle_labels_by_groups(y_base, groups, rng)
            l_sh = shuffle_labels_by_groups(l_base, groups, rng)
        else:
            y_sh, l_sh = y_base.copy(), l_base.copy()
            rng.shuffle(y_sh)
            rng.shuffle(l_sh)
        w_ys, _ = fit_ridge(h_train, y_sh, alpha=alpha)
        w_ls, _ = fit_ridge(h_train, l_sh, alpha=alpha)
        axes_sh = fused_and_difference(w_ys, w_ls)
        shuf_s.append(axes_sh["fused_s"])
        shuf_d.append(axes_sh["diff_d"])

    shuffle_for = {
        "yellow": shuf_y,
        "line": shuf_l,
        "line_residualized": shuf_l,
        "yellow_residualized": shuf_y,
        "fused_s": shuf_s,
        "diff_d": shuf_d,
        "object_salience": shuf_sal,
    }

    concepts = {}
    for name, w in (
        ("yellow", probes["yellow"]),
        ("line", probes["line"]),
        ("line_residualized", probes["line_residualized"]),
        ("yellow_residualized", probes["yellow_residualized"]),
        ("fused_s", probes["fused_s"]),
        ("diff_d", probes["diff_d"]),
        ("object_salience", probes["object_salience"]),
    ):
        concepts[name] = score_probe_on_head(
            w, W_policy, W_value, h_train, h_ood,
            shuffle_weights=shuffle_for.get(name),
            iso_policy=iso_pi,
            iso_value=iso_val,
        )

    basis_R = readout_row_basis(W_policy, W_value)
    readout_frac = {
        "yellow": fraction_in_readout(probes["yellow"], basis_R),
        "line": fraction_in_readout(probes["line"], basis_R),
        "fused_s": fraction_in_readout(probes["fused_s"], basis_R),
        "diff_d": fraction_in_readout(probes["diff_d"], basis_R),
    }
    frac_R_effective = frac_R_effective_null_report(
        {
            "yellow": probes["yellow"],
            "line": probes["line"],
            "fused_s": probes["fused_s"],
            "diff_d": probes["diff_d"],
        },
        basis_R,
        h_train,
        energy=0.95,
        n_null=2048,
        rng=rng,
    )

    # Value gates are kept as separate keys:
    # - salience: both duel objects are always present, so this gate often fails.
    # - fused_s: gate on the fused axis.
    fused_value_iso = concepts["fused_s"]["value_iso_null_percentile"]
    sal_value_iso = concepts["object_salience"]["value_iso_null_percentile"]

    ratio_block = {
        "ratio_value_yellow_over_line": _ratio(
            concepts["yellow"]["value_weight_loading"],
            concepts["line"]["value_weight_loading"],
        ),
        "ratio_policy_yellow_over_line": _ratio(
            concepts["yellow"]["policy_weight_loading"],
            concepts["line"]["policy_weight_loading"],
        ),
        "ratio_value_yellow_over_line_residualized": _ratio(
            concepts["yellow"]["value_weight_loading"],
            concepts["line_residualized"]["value_weight_loading"],
        ),
        "ratio_policy_yellow_over_line_residualized": _ratio(
            concepts["yellow"]["policy_weight_loading"],
            concepts["line_residualized"]["policy_weight_loading"],
        ),
        "ratio_value_yellow_over_line_raw": _ratio(
            concepts["yellow"]["value_weight_loading_raw"],
            concepts["line"]["value_weight_loading_raw"],
        ),
        "ratio_policy_yellow_over_line_raw": _ratio(
            concepts["yellow"]["policy_weight_loading_raw"],
            concepts["line"]["policy_weight_loading_raw"],
        ),
        "ratio_policy_s_over_d": _ratio(
            concepts["fused_s"]["policy_weight_loading"],
            concepts["diff_d"]["policy_weight_loading"],
        ),
        "ratio_value_s_over_d": _ratio(
            concepts["fused_s"]["value_weight_loading"],
            concepts["diff_d"]["value_weight_loading"],
        ),
    }

    report = {
        "cos_yellow_line": probes["cos_yellow_line"],
        "residualization_near_noop": abs(probes["cos_yellow_line"]) < 0.15,
        "n_shuffle": n_shuffle,
        "shuffle_unit_yellow_line": "group" if groups is not None else "frame",
        "probe_norm_yellow": probes["probe_norm_yellow"],
        "probe_norm_line": probes["probe_norm_line"],
        "ood_auc_yellow": probes["ood_auc"]["yellow"],
        "ood_auc_line": probes["ood_auc"]["line"],
        "concepts": concepts,
        "readout_fraction": readout_frac,
        "rank_R": int(basis_R.shape[0]),
        # E[‖P_R v‖²] for random unit v = rank(R) / d (ambient null).
        "expected_random_frac_R": float(basis_R.shape[0] / W_policy.shape[1]),
        "readout_fraction_delta_vs_random": {
            name: float(frac - basis_R.shape[0] / W_policy.shape[1])
            for name, frac in readout_frac.items()
        },
        # Primary null: energy subspace S_95 (see frac_R_effective).
        "frac_R_effective": frac_R_effective,
        "value_head_positive_control_fused_s_iso_percentile": fused_value_iso,
        "value_head_positive_control_fused_s_ok": fused_value_iso >= 0.5,
        "value_head_positive_control_salience_iso_percentile": sal_value_iso,
        "value_head_positive_control_salience_ok": sal_value_iso >= 0.5,
        "isotropic_policy_null_mean": float(iso_pi.mean()),
        "isotropic_policy_null_std": float(iso_pi.std()),
        **ratio_block,
    }
    concepts_train = {
        "yellow": y_train["yellow_present"],
        "line": y_train["line_present"],
    }
    concepts_ood = {
        "yellow": y_ood["yellow_present"],
        "line": y_ood["line_present"],
    }
    attach_dom_lda_projection(
        report, h_train, concepts_train, W_policy, W_value, rng,
        h_ood=h_ood, concepts_ood=concepts_ood, lda_alpha=alpha,
    )
    return report


def head_null_comparison(
    probe_weights: dict,
    W_policy: np.ndarray,
    W_value: np.ndarray,
    rng: np.random.RandomState,
    *,
    n_isotropic: int = 512,
    head_label: str = "alternate",
) -> dict:
    """Project trained probe directions onto a readout head; percentiles vs that head's null."""
    skip = {
        "ood_auc", "cos_yellow_line", "salience_target",
        "probe_norm_yellow", "probe_norm_line",
    }
    iso_pi = isotropic_null_loadings(W_policy, n_dirs=n_isotropic, rng=rng)
    iso_val = isotropic_null_loadings(W_value.reshape(1, -1), n_dirs=n_isotropic, rng=rng)

    out = {"head_label": head_label}
    for name, w in probe_weights.items():
        if name in skip or not isinstance(w, np.ndarray):
            continue
        pol = weight_loading(W_policy, w)
        val = weight_loading(W_value.reshape(1, -1), w)
        out[name] = {
            "policy_weight_loading": pol,
            "value_weight_loading": val,
            "policy_iso_null_percentile": percentile_against_null(pol, iso_pi),
            "value_iso_null_percentile": percentile_against_null(val, iso_val),
        }
    return out


def compare_trained_vs_init_heads(
    probe_weights: dict,
    W_policy_trained: np.ndarray,
    W_value_trained: np.ndarray,
    W_policy_init: np.ndarray,
    W_value_init: np.ndarray,
    rng: np.random.RandomState,
    *,
    n_isotropic: int = 512,
) -> dict:
    """Side-by-side isotropic percentiles: each loading vs its own head's null.

    Do **not** compare raw loadings across trained vs init matrices — scales differ.

    Confound: ``W_*_init`` is typically from a fully fresh network (trunk + heads),
    not heads re-initialised on the trained trunk. Deltas mix trunk and readout change.
    """
    trained = head_null_comparison(
        probe_weights, W_policy_trained, W_value_trained, rng,
        n_isotropic=n_isotropic, head_label="trained",
    )
    init = head_null_comparison(
        probe_weights, W_policy_init, W_value_init, rng,
        n_isotropic=n_isotropic, head_label="random_init",
    )
    keys = ("yellow", "line", "object_salience")
    paired = {}
    for key in keys:
        if key not in trained or key not in init:
            continue
        paired[key] = {
            "trained_policy_iso_percentile": trained[key]["policy_iso_null_percentile"],
            "init_policy_iso_percentile": init[key]["policy_iso_null_percentile"],
            "trained_value_iso_percentile": trained[key]["value_iso_null_percentile"],
            "init_value_iso_percentile": init[key]["value_iso_null_percentile"],
        }
    return {"trained_head": trained, "random_init_head": init, "paired_percentiles": paired}
