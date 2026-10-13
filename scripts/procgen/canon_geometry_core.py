"""Pure-numpy cue/competition geometry for the canonical Attribute-Maze bank (Tables 11-13, Figs 11-14)."""
from __future__ import annotations

import numpy as np
from scipy import stats

EPS = 1e-12


def m_matrix(p: np.ndarray) -> np.ndarray:
    return np.diag(p) - np.outer(p, p)


def fisher_bilinear(J: np.ndarray, p: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """u^T J^T (diag p - p p^T) J v without materialising the D x D Fisher. u, v: [D] or [n, D]."""
    gu = u @ J.T
    gv = v @ J.T
    return (gu * p * gv).sum(-1) - (gu @ p) * (gv @ p)


def cos_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean cosine; NaN where either vector is ~0."""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    out = (a * b).sum(-1) / np.maximum(na * nb, EPS)
    return np.where((na < EPS) | (nb < EPS), np.nan, out)


def _broadcast_J(J: np.ndarray, n: int) -> np.ndarray:
    return np.broadcast_to(J, (n,) + J.shape) if J.ndim == 2 else J


def within_state(ds: np.ndarray, dc: np.ndarray, J: np.ndarray, p: np.ndarray) -> dict:
    """cos_E and cos_F(Delta_shape, Delta_colour) per state, each under that state's own Fisher."""
    n = ds.shape[0]
    Jb = _broadcast_J(J, n)
    cf = np.full(n, np.nan)
    qs = np.empty(n)
    qc = np.empty(n)
    for i in range(n):
        qs[i] = fisher_bilinear(Jb[i], p[i], ds[i], ds[i])
        qc[i] = fisher_bilinear(Jb[i], p[i], dc[i], dc[i])
        if qs[i] > EPS and qc[i] > EPS:
            cf[i] = fisher_bilinear(Jb[i], p[i], ds[i], dc[i]) / np.sqrt(qs[i] * qc[i])
    return {"cos_e": cos_e(ds, dc), "cos_f": cf, "q_shape": qs, "q_colour": qc}


def cross_state(d: np.ndarray, J: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """kappa_E and kappa_F matrices [n, n] (diagonal NaN); kappa_F uses the pair metric F_ij = (F_i + F_j)/2."""
    n, D = d.shape
    Jb = _broadcast_J(J, n)
    A = Jb.shape[1]
    # G[k, i, :] = J_k d_i: logit displacement of state i's direction read through state k's Jacobian
    G = (Jb.reshape(n * A, D) @ d.T).reshape(n, A, n).transpose(0, 2, 1)
    pg = np.einsum("ka,kia->ki", p, G)
    Dq = np.einsum("ka,kia->ki", p, G * G) - pg ** 2
    idx = np.arange(n)
    Gself = G[idx, idx]
    R = np.einsum("ka,ka,kja->kj", p, Gself, G) - pg[idx, idx][:, None] * pg
    num = 0.5 * (R + R.T)
    qi = 0.5 * (np.diag(Dq)[:, None] + Dq.T)
    qj = qi.T
    with np.errstate(invalid="ignore", divide="ignore"):
        kf = num / np.sqrt(qi * qj)
    kf[(qi <= EPS) | (qj <= EPS)] = np.nan
    un = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), EPS)
    ke = un @ un.T
    np.fill_diagonal(ke, np.nan)
    np.fill_diagonal(kf, np.nan)
    return ke, kf


def bearing_relation(b_i, b_j) -> str:
    """Relation between the line-candidate first-step bearings (dx, dy) of two states."""
    bi, bj = np.asarray(b_i), np.asarray(b_j)
    if np.array_equal(bi, bj):
        return "same"
    if np.array_equal(bi, -bj):
        return "opposite"
    return "perpendicular"


def relation_matrix(line_bearings: np.ndarray) -> np.ndarray:
    n = len(line_bearings)
    out = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            out[i, j] = bearing_relation(line_bearings[i], line_bearings[j])
    return out


def upper_pairs(n: int, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangle pair indices; with a state mask, only pairs where both states are in the mask."""
    iu, ju = np.triu_indices(n, k=1)
    if mask is not None:
        keep = mask[iu] & mask[ju]
        iu, ju = iu[keep], ju[keep]
    return iu, ju


def block_means(K: np.ndarray, rel: np.ndarray, mask: np.ndarray | None = None) -> dict:
    iu, ju = upper_pairs(K.shape[0], mask)
    v = K[iu, ju]
    r = rel[iu, ju]
    out = {}
    for name in ("same", "opposite", "perpendicular"):
        m = (r == name) & np.isfinite(v)
        out[name] = float(v[m].mean()) if m.any() else float("nan")
        out[f"n_{name}"] = int(m.sum())
    return out


def logit_contrast_fraction(J: np.ndarray, p: np.ndarray, d: np.ndarray, a_line: list[list[int]],
                            a_gem: list[list[int]]) -> np.ndarray:
    """Fraction of d's Fisher energy lying along the target-selection logit contrast w = 1[A_line] - 1[A_gem].

    In logit space with metric M(p), this is cos_M(J d, w)^2. A value near 1 means d moves the policy only by
    trading line-group against gem-group probability, i.e. along the state's single UP-vs-RIGHT axis.
    """
    n = d.shape[0]
    Jb = _broadcast_J(J, n)
    A = Jb.shape[1]
    out = np.full(n, np.nan)
    for i in range(n):
        w = np.zeros(A)
        w[list(a_line[i])] += 1.0
        w[list(a_gem[i])] -= 1.0
        g = Jb[i] @ d[i]
        M = m_matrix(p[i])
        gg, ww, gw = g @ M @ g, w @ M @ w, g @ M @ w
        if gg > EPS and ww > EPS:
            out[i] = gw ** 2 / (gg * ww)
    return out


def signed_contrast(J: np.ndarray, p: np.ndarray, d: np.ndarray, a_line, a_gem) -> np.ndarray:
    """Directional derivative of P(A_line) - P(A_gem) along d (sign says which target d favours)."""
    n = d.shape[0]
    Jb = _broadcast_J(J, n)
    A = Jb.shape[1]
    out = np.empty(n)
    for i in range(n):
        w = np.zeros(A)
        w[list(a_line[i])] += 1.0
        w[list(a_gem[i])] -= 1.0
        out[i] = w @ m_matrix(p[i]) @ (Jb[i] @ d[i])
    return out


def spatial_descriptor(line_cells: np.ndarray, gem_cells: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Ordered spatial configuration (line x,y, gem x,y, d), z-scored across the states given."""
    X = np.column_stack([line_cells, gem_cells, d]).astype(float)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - X.mean(0)) / sd


def spatial_similarity(Z: np.ndarray, role_free: bool = False) -> np.ndarray:
    """-||z_i - z_j||. role_free also tries j with line/gem exchanged, so a state and its swap mirror (same two
    cells, roles exchanged) count as identical; the ordered version separates signs by construction."""
    S = -np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    if role_free:
        Zs = Z[:, [2, 3, 0, 1, 4]]
        S = np.maximum(S, -np.linalg.norm(Z[:, None, :] - Zs[None, :, :], axis=-1))
    return S


def pair_features(sign: np.ndarray, d: np.ndarray, layout: np.ndarray, spatial_sim: np.ndarray,
                  partner: np.ndarray, iu: np.ndarray, ju: np.ndarray, spatial_rf: np.ndarray | None = None) -> dict:
    """State-pair covariates. `partner[i]` = index of i's line/gem swap mirror (or -1)."""
    return {
        "same_sign": (sign[iu] == sign[ju]).astype(float),
        "sign_prod": (sign[iu] * sign[ju]).astype(float),
        "abs_dd": np.abs(d[iu] - d[ju]).astype(float),
        "same_layout": (layout[iu] == layout[ju]).astype(float),
        "spatial_sim": spatial_sim[iu, ju],
        "swap_partner": (partner[iu] == ju).astype(float),
        "spatial_sim_rf": (spatial_rf if spatial_rf is not None else spatial_sim)[iu, ju],
    }


FINE = ("abs_dd", "same_layout", "spatial_sim", "spatial_sim_rf", "swap_partner")


def design(feats: dict, model: str) -> np.ndarray:
    """'sign' = two-block (same vs cross sign) model; 'full' adds fine spatial variables, each with its main
    effect and its interaction with the sign product, because the d_comp-as-signed-axis picture predicts
    finer effects enter as s_i s_j * g(features)."""
    n = len(feats["same_sign"])
    cols = [np.ones(n), feats["same_sign"]]
    if model == "full":
        for f in FINE:
            x = feats[f]
            cols += [x, x * feats["sign_prod"]]
    return np.column_stack(cols)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def fit_predict(X_fit, y_fit, X_eval) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    return X_eval @ beta


def within_sign_perm(sign, in_mask, rng, partner=None) -> np.ndarray:
    """State relabelling that permutes within each sign class. With `partner` (line/gem mirror map, mirrors of
    opposite sign), mirror pairs move as one unit: permute the +1 class and carry the same relabelling to the
    mirrors, because on black d_comp(mirror(s)) = -d_comp(s) exactly, so untied permutations would treat a
    duplicated state as two independent ones and make the null too narrow."""
    perm = np.arange(len(sign))
    tied = (partner is not None and (partner >= 0).all() and (sign[partner] == -sign).all()
            and (in_mask[partner] == in_mask).all())
    if tied:
        idx = np.flatnonzero((sign == 1) & in_mask)
        new = rng.permutation(idx)
        perm[idx] = new
        perm[partner[idx]] = partner[new]
        return perm
    for s in np.unique(sign):
        idx = np.flatnonzero((sign == s) & in_mask)
        perm[idx] = rng.permutation(idx)
    return perm


def variance_decomposition(K: np.ndarray, sign, d, layout, spatial_sim, partner, mask=None,
                           n_perm: int = 200, seed: int = 0, spatial_rf=None) -> dict:
    """In-sample R^2 of pairwise similarity: sign-only vs sign+fine, plus a within-sign permutation null for
    the fine increment (state labels permuted within each sign class, so the sign block structure is kept)."""
    iu, ju = upper_pairs(K.shape[0], mask)
    y = K[iu, ju]
    ok = np.isfinite(y)
    iu, ju, y = iu[ok], ju[ok], y[ok]

    def _r2s(perm):
        f = pair_features(sign, d[perm], layout[perm], spatial_sim[np.ix_(perm, perm)],
                          _perm_partner(partner, perm), iu, ju,
                          None if spatial_rf is None else spatial_rf[np.ix_(perm, perm)])
        r_s = r2(y, fit_predict(design(f, "sign"), y, design(f, "sign")))
        r_f = r2(y, fit_predict(design(f, "full"), y, design(f, "full")))
        return r_s, r_f

    ident = np.arange(K.shape[0])
    r_sign, r_full = _r2s(ident)
    rng = np.random.default_rng(seed)
    in_mask = np.ones(K.shape[0], bool) if mask is None else mask
    null = []
    for _ in range(n_perm):
        null.append(_r2s(within_sign_perm(sign, in_mask, rng, partner))[1] - r_sign)
    null = np.array(null)
    inc = r_full - r_sign
    return {"n_pairs": int(len(y)), "r2_sign": r_sign, "r2_full": r_full, "r2_fine_increment": inc,
            "perm_null_increment_mean": float(null.mean()) if n_perm else float("nan"),
            "perm_null_increment_q95": float(np.quantile(null, 0.95)) if n_perm else float("nan"),
            "perm_p": float((1 + (null >= inc).sum()) / (1 + n_perm)) if n_perm else float("nan")}


def _perm_partner(partner: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Partner map after relabelling state i's features with those of perm[i]."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    out = np.full_like(partner, -1)
    for i in range(len(perm)):
        q = partner[perm[i]]
        out[i] = inv[q] if q >= 0 else -1
    return out


def sign_gap(K: np.ndarray, sign: np.ndarray, mask=None) -> dict:
    """Within-sign vs between-sign similarity: the whole 'bearing organization' if the sign is all there is."""
    iu, ju = upper_pairs(K.shape[0], mask)
    v = K[iu, ju]
    ss = sign[iu] == sign[ju]
    ok = np.isfinite(v)
    same, cross = v[ss & ok], v[~ss & ok]
    folded = (sign[iu] * sign[ju] * v)[ok]
    return {"same_sign_mean": float(same.mean()), "cross_sign_mean": float(cross.mean()),
            "gap": float(same.mean() - cross.mean()),
            "asymmetry": float(same.mean() + cross.mean()),
            "folded_mean": float(folded.mean()), "folded_sd": float(folded.std()),
            "folded_frac_pos": float((folded > 0).mean()),
            "same_sign_sd": float(same.std()), "cross_sign_sd": float(cross.std())}


def rank1_offdiag_r2(K: np.ndarray, mask=None, n_iter: int = 50) -> dict:
    """Best symmetric rank-1 fit a a^T to the off-diagonal of K (diagonal imputed iteratively).
    Tests the one-shared-signed-axis model kappa_ij ~ a_i a_j; returns the leading loading vector too."""
    idx = np.arange(K.shape[0]) if mask is None else np.flatnonzero(mask)
    S = K[np.ix_(idx, idx)].copy()
    S = np.where(np.isfinite(S), S, 0.0)
    off = ~np.eye(len(idx), dtype=bool)
    fill = S.copy()
    np.fill_diagonal(fill, 1.0)
    for _ in range(n_iter):
        w, V = np.linalg.eigh(fill)
        a = V[:, -1] * np.sqrt(max(w[-1], 0.0))
        fill = S.copy()
        np.fill_diagonal(fill, a ** 2)
    iu, ju = np.triu_indices(len(idx), k=1)
    y = S[iu, ju]
    return {"r2_rank1": r2(y, a[iu] * a[ju]), "loadings": a, "states": idx}


def shared_axis(d: np.ndarray, sign: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    """Unit sign-folded mean competition direction, estimated on fit states only."""
    un = d[fit_idx] / np.maximum(np.linalg.norm(d[fit_idx], axis=1, keepdims=True), EPS)
    u = (sign[fit_idx, None] * un).mean(0)
    return u / max(np.linalg.norm(u), EPS)


def axis_loadings(d: np.ndarray, u: np.ndarray, J: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean and state-local-Fisher cosine of each d with the axis u."""
    n = d.shape[0]
    Jb = _broadcast_J(J, n)
    ae = cos_e(d, np.broadcast_to(u, d.shape))
    af = np.full(n, np.nan)
    for i in range(n):
        qd = fisher_bilinear(Jb[i], p[i], d[i], d[i])
        qu = fisher_bilinear(Jb[i], p[i], u, u)
        if qd > EPS and qu > EPS:
            af[i] = fisher_bilinear(Jb[i], p[i], d[i], u) / np.sqrt(qd * qu)
    return ae, af


def heldout_eval(K: np.ndarray, sign, d, layout, spatial_sim, partner, fit_mask, eval_mask,
                 loadings: np.ndarray | None = None, spatial_rf=None, mask=None) -> dict:
    """Predictive R^2 on pairs touching >= 1 eval state, with coefficients fit on fit-fit pairs only.
    `loadings` (per-state cosine with a fit-only axis) gives the rank-1 shared-axis prediction a_i a_j."""
    iu, ju = upper_pairs(K.shape[0], mask)
    y = K[iu, ju]
    ok = np.isfinite(y)
    iu, ju, y = iu[ok], ju[ok], y[ok]
    f = pair_features(sign, d, layout, spatial_sim, partner, iu, ju, spatial_rf)
    ff = fit_mask[iu] & fit_mask[ju]
    ev = eval_mask[iu] | eval_mask[ju]
    ee = eval_mask[iu] & eval_mask[ju]
    out = {"n_fit_pairs": int(ff.sum()), "n_eval_pairs": int(ev.sum()), "n_eval_eval_pairs": int(ee.sum())}
    for model in ("sign", "full"):
        X = design(f, model)
        yhat = fit_predict(X[ff], y[ff], X)
        out[f"r2_{model}_eval"] = r2(y[ev], yhat[ev])
        out[f"r2_{model}_evaleval"] = r2(y[ee], yhat[ee]) if ee.sum() > 2 else float("nan")
    if loadings is not None:
        yhat = loadings[iu] * loadings[ju]
        # rank-1 prediction has a free scale; calibrate it on fit-fit pairs only
        c = float((yhat[ff] @ y[ff]) / max(yhat[ff] @ yhat[ff], EPS))
        out["axis_scale_fit"] = c
        out["r2_axis_eval"] = r2(y[ev], c * yhat[ev])
        out["r2_axis_evaleval"] = r2(y[ee], c * yhat[ee]) if ee.sum() > 2 else float("nan")
    return out


def lowrank_null(d: np.ndarray, J: np.ndarray, p: np.ndarray, n_draws: int, seed: int, mask=None) -> dict:
    """Table 13: real |kappa_F| vs isotropic Gaussian directions under each pair's own rank<=A-1 metric."""
    _, kf = cross_state(d, J, p)
    iu, ju = upper_pairs(d.shape[0], mask)
    real = np.abs(kf[iu, ju])
    real = real[np.isfinite(real)]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_draws):
        _, kn = cross_state(rng.standard_normal(d.shape), J, p)
        v = np.abs(kn[iu, ju])
        null.append(v[np.isfinite(v)])
    null = np.concatenate(null)
    return {"real_abs": float(real.mean()), "null_abs": float(null.mean()),
            "real_gt_0.9": float((real > 0.9).mean()), "null_gt_0.9": float((null > 0.9).mean()),
            "real_gt_0.99": float((real > 0.99).mean()), "null_gt_0.99": float((null > 0.99).mean()),
            "n_real": int(real.size), "n_null": int(null.size)}


def within_relation_corr(K: np.ndarray, S: np.ndarray, rel: np.ndarray, mask=None) -> dict:
    """Fig 14: Pearson r between spatial similarity and kappa, within each bearing relation."""
    iu, ju = upper_pairs(K.shape[0], mask)
    k, s, r = K[iu, ju], S[iu, ju], rel[iu, ju]
    out = {}
    for name in ("same", "opposite", "perpendicular"):
        m = (r == name) & np.isfinite(k) & np.isfinite(s)
        if m.sum() > 3 and k[m].std() > 1e-12 and s[m].std() > 1e-12:
            out[name] = float(np.corrcoef(s[m], k[m])[0, 1])
        else:
            out[name] = float("nan")
    return out


def t_ci(values) -> dict:
    """Across-seed mean, SD (ddof=1; the paper's ± column) and t-distribution 95% CI (n = number of finite values)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m = float(v.mean())
    if n == 1:
        return {"mean": m, "sd": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 1}
    sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"mean": m, "sd": sd, "lo": m - h, "hi": m + h, "n": n}


def mirror_antisymmetry(K: np.ndarray, partner: np.ndarray) -> dict:
    """How exactly kappa(i, j) = -kappa(i, m(j)) holds. For swap edits d_comp(m(s)) = -d_comp(s) by construction
    (the two swapped observations are the same images with roles exchanged), so kappa_E satisfies it exactly and the
    whole cross-sign block is a negated copy of the same-sign block; only same-sign structure is informative."""
    A, B = K, -K[:, partner]
    ok = np.isfinite(A) & np.isfinite(B)
    n = K.shape[0]
    return {"mirror_pair_mean": float(np.nanmean(K[np.arange(n), partner])),
            "max_abs_dev": float(np.abs(A - B)[ok].max()), "median_abs_dev": float(np.median(np.abs(A - B)[ok]))}


FINE_WITHIN = ("abs_dd", "same_layout", "spatial_sim")


def _within_sign_pairs(K, sign, mask):
    iu, ju = upper_pairs(K.shape[0], mask)
    keep = (sign[iu] == sign[ju]) & np.isfinite(K[iu, ju])
    return iu[keep], ju[keep]


def _within_X(d, layout, S, iu, ju):
    return np.column_stack([np.ones(len(iu)), np.abs(d[iu] - d[ju]), (layout[iu] == layout[ju]).astype(float),
                            S[iu, ju]])


def within_sign_fine(K: np.ndarray, sign, d, layout, S, mask=None, n_perm: int = 200, seed: int = 0,
                     fit_mask=None, eval_mask=None, partner=None) -> dict:
    """R^2 of same-sign kappa on |d_i - d_j|, same layout and spatial similarity, with a within-sign permutation null.

    With fit/eval masks, also the held-out R^2 on same-sign pairs touching an eval state, fit on fit-fit pairs."""
    iu, ju = _within_sign_pairs(K, sign, mask)
    y = K[iu, ju]
    X = _within_X(d, layout, S, iu, ju)
    r_obs = r2(y, fit_predict(X, y, X))
    rng = np.random.default_rng(seed)
    in_mask = np.ones(K.shape[0], bool) if mask is None else mask
    null = []
    for _ in range(n_perm):
        perm = within_sign_perm(sign, in_mask, rng, partner)
        Xp = _within_X(d[perm], layout[perm], S[np.ix_(perm, perm)], iu, ju)
        null.append(r2(y, fit_predict(Xp, y, Xp)))
    null = np.array(null) if n_perm else np.array([np.nan])
    out = {"n_pairs": int(len(y)), "mean": float(y.mean()), "sd": float(y.std()), "r2_fine": r_obs,
           "perm_null_mean": float(np.nanmean(null)), "perm_null_q95": float(np.nanquantile(null, 0.95)),
           "perm_p": float((1 + (null >= r_obs).sum()) / (1 + n_perm)) if n_perm else float("nan")}
    if fit_mask is not None:
        ff = fit_mask[iu] & fit_mask[ju]
        ev = eval_mask[iu] | eval_mask[ju]
        out["r2_fine_heldout"] = r2(y[ev], fit_predict(X[ff], y[ff], X[ev])) if ev.sum() > 2 else float("nan")
    return out


def rank1_within_sign(K: np.ndarray, sign, mask=None) -> float:
    """Mean over the two sign blocks of the rank-1 (one shared axis, per-state loading) R^2 of the same-sign block."""
    m = np.ones(K.shape[0], bool) if mask is None else mask
    return float(np.mean([rank1_offdiag_r2(K, m & (sign == s))["r2_rank1"] for s in np.unique(sign)]))
