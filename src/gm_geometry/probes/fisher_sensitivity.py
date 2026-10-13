"""KL-metric (Fisher) sensitivity F = J^T (diag(π) − ππ^T) J at intermediate Impala activations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from gm_geometry.probes.coin_location import BLOCK3_HW
from gm_geometry.utils import ci95 as _ci95


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def fisher_from_logit_jacobian(
    jacobian: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Full Fisher F = J^T (diag(π) − ππ^T) J from the logit Jacobian [A, D] and probs [A]."""
    j = np.asarray(jacobian, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if j.ndim != 2:
        raise ValueError(f"jacobian must be [A,D], got {j.shape}")
    if p.shape[0] != j.shape[0]:
        raise ValueError(f"probs {p.shape} vs jacobian {j.shape}")
    w = np.diag(p) - np.outer(p, p)
    return j.T @ w @ j


def fisher_diag_from_logit_jacobian(
    jacobian: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Diagonal of Fisher without forming the full D×D matrix.

    For each unit i: F_ii = Σ_a π(a) s_a(i)^2 with
    s_a = ∇logit_a − Σ_b π(b)∇logit_b.
    """
    j = np.asarray(jacobian, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    mean = p @ j
    centered = j - mean[None, :]
    return (p[:, None] * (centered ** 2)).sum(axis=0)


def spatial_fisher_map_from_block3_grads(
    grads_achw: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """8×8 map M(y,x) = Σ_c F_diag[c,y,x] from ∂logit/∂block3 grads [A, C, H, W]."""
    g = np.asarray(grads_achw, dtype=np.float64)
    a, c, h, w = g.shape
    j = g.reshape(a, c * h * w)
    diag = fisher_diag_from_logit_jacobian(j, probs)
    return diag.reshape(c, h, w).sum(axis=0)


def kl_categorical(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for discrete distributions."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def quadratic_form(f_mat: np.ndarray, v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    f_mat = np.asarray(f_mat, dtype=np.float64)
    return float(v @ f_mat @ v)


def bilinear_form_via_jacobian(
    jacobian: np.ndarray, probs: np.ndarray, u: np.ndarray, v: np.ndarray,
) -> float:
    """u^T F v = (Ju)^T W (Jv) with W = diag(p) - pp^T, exact and without forming the [D, D] matrix F."""
    j = np.asarray(jacobian, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    ju = j @ u
    jv = j @ v
    w = np.diag(p) - np.outer(p, p)
    return float(ju @ w @ jv)


def quadratic_form_via_jacobian(jacobian: np.ndarray, probs: np.ndarray, v: np.ndarray) -> float:
    """v^T F v = (Jv)^T W (Jv) with W = diag(p) - pp^T, exact and without forming the [D, D] matrix F."""
    return bilinear_form_via_jacobian(jacobian, probs, v, v)


def trace_via_jacobian(jacobian: np.ndarray, probs: np.ndarray) -> float:
    """tr(F) = tr(W J J^T), without forming F."""
    j = np.asarray(jacobian, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    w = np.diag(p) - np.outer(p, p)
    jjt = j @ j.T
    return float(np.trace(w @ jjt))


def sensitivity_along(f_mat: np.ndarray, v: np.ndarray) -> float:
    """S = (v^T F v) / ||v||^2."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n2 = float(v @ v)
    if n2 < 1e-18:
        return float("nan")
    return quadratic_form(f_mat, v) / n2


def cell_mass(map_hw: np.ndarray, y: int, x: int, radius: int = 1) -> float:
    h, w = map_hw.shape
    total = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = int(y) + dy, int(x) + dx
            if 0 <= yy < h and 0 <= xx < w:
                total += float(map_hw[yy, xx])
    return total


def path_band_cells(
    *,
    agent_y: float,
    agent_x: float,
    hw: int = BLOCK3_HW,
    band_half_height: float = 1.25,
) -> set[tuple[int, int]]:
    """Feature cells in the rightward proxy band (same rule as path_band_mass)."""
    ay = float(agent_y)
    ax = float(agent_x)
    cells: set[tuple[int, int]] = set()
    for y in range(hw):
        if abs(y - ay) > band_half_height:
            continue
        for x in range(hw):
            if x + 0.5 >= ax:
                cells.add((y, x))
    return cells


def coin_neighborhood_cells(
    coin_yx: tuple[int, int],
    *,
    radius: int = 1,
    hw: int = BLOCK3_HW,
) -> set[tuple[int, int]]:
    cy, cx = int(coin_yx[0]), int(coin_yx[1])
    cells: set[tuple[int, int]] = set()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < hw and 0 <= xx < hw:
                cells.add((yy, xx))
    return cells


def path_band_mass(
    map_hw: np.ndarray,
    *,
    agent_y: float,
    agent_x: float,
    band_half_height: float = 1.25,
) -> float:
    """Mass on cells to the right of the agent in a horizontal band (proxy path)."""
    cells = path_band_cells(
        agent_y=agent_y, agent_x=agent_x, band_half_height=band_half_height
    )
    return float(sum(float(map_hw[y, x]) for y, x in cells))


def coin_path_separation(
    coin_yx: tuple[int, int],
    agent_yx: tuple[float, float],
    *,
    radius: int = 1,
) -> dict[str, float]:
    """Feature-cell overlap / distance between coin neighbourhood and path band."""
    coin_cells = coin_neighborhood_cells(coin_yx, radius=radius)
    path_cells = path_band_cells(agent_y=agent_yx[0], agent_x=agent_yx[1])
    inter = coin_cells & path_cells
    # Min L1 distance from coin centre to any path cell (inf if path empty).
    cy, cx = int(coin_yx[0]), int(coin_yx[1])
    if path_cells:
        min_l1 = min(abs(cy - y) + abs(cx - x) for y, x in path_cells)
    else:
        min_l1 = float("nan")
    return {
        "n_coin_cells": float(len(coin_cells)),
        "n_path_cells": float(len(path_cells)),
        "n_overlap_cells": float(len(inter)),
        "overlap": float(1.0 if inter else 0.0),
        "min_l1_coin_to_path": float(min_l1),
    }


def block3_fisher_pack(
    policy,
    obs_nchw: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (grads[A,C,H,W], probs[A], acts[C,H,W], M[H,W]) for one frame, with respect to h_pre = ReLU(H^3)."""
    from gm_geometry.probes.coin_location import block3_logit_gradients

    grads, acts = block3_logit_gradients(policy, obs_nchw, device, apply_relu=True)
    x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    b = x.shape[0]
    h0 = torch.zeros(b, policy.embedder.output_dim, device=device)
    m = torch.zeros(b, device=device)
    with torch.no_grad():
        dist, _v, _nh = policy(x, h0, m)
        probs = dist.probs.detach()[0].cpu().numpy()
    m_map = spatial_fisher_map_from_block3_grads(grads, probs)
    return grads, probs, acts, m_map


def block3_fisher_map(
    policy,
    obs_nchw: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (M[H,W], probs[A], acts[C,H,W]) for one frame."""
    _grads, probs, acts, m_map = block3_fisher_pack(policy, obs_nchw, device)
    return m_map, probs, acts


def block3_logit_jacobian_and_act(
    policy,
    obs_nchw: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block3 analogue of fc_logit_jacobian_and_act with respect to h_pre = ReLU(H^3), flattened.

    Shapes [A, D'], [A], [D'] with D' = C*H*W; returns the Jacobian, never the [D', D'] Fisher."""
    from gm_geometry.probes.coin_location import block3_logit_gradients

    grads, acts = block3_logit_gradients(
        policy, obs_nchw, device, apply_relu=True
    )  # [A,C,H,W], [C,H,W]
    a = grads.shape[0]
    j = grads.reshape(a, -1).astype(np.float64)
    h = acts.reshape(-1).astype(np.float64)

    x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    b = x.shape[0]
    h0 = torch.zeros(b, policy.embedder.output_dim, device=device)
    m = torch.zeros(b, device=device)
    with torch.no_grad():
        dist, _v, _nh = policy(x, h0, m)
        probs = dist.probs.detach()[0].cpu().numpy()
    return j, probs, h


def fisher_trace_in_topk_pcs(
    f_mat: np.ndarray,
    activations: np.ndarray,
    *,
    k: int = 16,
) -> dict[str, float]:
    """Fraction of tr(F) inside the top-k PC subspace of ``activations`` [N, D]."""
    f_mat = np.asarray(f_mat, dtype=np.float64)
    h = np.asarray(activations, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"activations must be [N,D], got {h.shape}")
    d = f_mat.shape[0]
    if h.shape[1] != d:
        raise ValueError(f"act dim {h.shape[1]} != F dim {d}")
    tr = float(np.trace(f_mat))
    if tr < 1e-18 or h.shape[0] < 2:
        return {
            "k": float(k),
            "tr_F": tr,
            "tr_F_in_topk": float("nan"),
            "frac_topk": float("nan"),
        }
    h0 = h - h.mean(axis=0, keepdims=True)
    # Economy SVD on covariance via thin SVD of centred matrix.
    _u, _s, vt = np.linalg.svd(h0, full_matrices=False)
    kk = int(min(k, vt.shape[0], d))
    u_pcs = vt[:kk].T  # [D, k]
    tr_sub = float(np.trace(u_pcs.T @ f_mat @ u_pcs))
    return {
        "k": float(kk),
        "tr_F": tr,
        "tr_F_in_topk": tr_sub,
        "frac_topk": tr_sub / (tr + 1e-12),
    }


def fc_logit_jacobian_and_act(
    policy,
    obs_nchw: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """∂logit_a / ∂h_fc, probs and h_fc. Shapes [A, D], [A], [D].

    The Jacobian is taken at the output of the whole `policy.embedder` (post-ReLU h_fc): a hook on
    `embedder.fc` would sit before ImpalaModel's own ReLU, whose zero gradient wipes inactive units' columns."""
    store: dict[str, torch.Tensor] = {}

    def _hook(_m, _i, out):
        store["h"] = out

    handle = policy.embedder.register_forward_hook(_hook)
    try:
        x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise ValueError("fc_logit_jacobian expects batch size 1")
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        m = torch.zeros(1, device=device)
        dist, _value, _nh = policy(x, h0, m)
        h = store["h"]
        logits = dist.logits
        n_act = int(logits.shape[1])
        grads = []
        for a in range(n_act):
            g_a = torch.autograd.grad(
                logits[0, a],
                h,
                retain_graph=(a < n_act - 1),
                allow_unused=False,
            )[0]
            grads.append(g_a.detach()[0].cpu().numpy())
        j = np.stack(grads, axis=0)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        return j, probs, h.detach()[0].cpu().numpy()
    finally:
        handle.remove()


def fc_fisher_and_act(
    policy,
    obs_nchw: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full F_h [D,D], probs [A], h [D] at embedder.fc."""
    j, probs, h = fc_logit_jacobian_and_act(policy, obs_nchw, device)
    f_mat = fisher_from_logit_jacobian(j, probs)
    return f_mat, probs, h


def fc_logit_jacobian_and_act_at_h_pre(
    policy,
    obs_nchw: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """∂logit_a / ∂h_pre, probs and h_pre, captured by a forward pre-hook on embedder.fc. Shapes [A, D], [A], [D]."""
    store: dict[str, torch.Tensor] = {}

    def _hook(_m, args):
        store["h"] = args[0]

    handle = policy.embedder.fc.register_forward_pre_hook(_hook)
    try:
        x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise ValueError("fc_logit_jacobian expects batch size 1")
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        m = torch.zeros(1, device=device)
        dist, _value, _nh = policy(x, h0, m)
        h = store["h"]
        logits = dist.logits
        n_act = int(logits.shape[1])
        grads = []
        for a in range(n_act):
            g_a = torch.autograd.grad(
                logits[0, a],
                h,
                retain_graph=(a < n_act - 1),
                allow_unused=False,
            )[0]
            grads.append(g_a.detach()[0].cpu().numpy())
        j = np.stack(grads, axis=0)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        return j, probs, h.detach()[0].cpu().numpy()
    finally:
        handle.remove()


def summarize_map_vs_landmarks(
    map_hw: np.ndarray,
    *,
    coin_yx: tuple[int, int] | None,
    agent_yx: tuple[float, float] | None,
    radius: int = 1,
) -> dict[str, float]:
    """Coin-cell mass, path-band mass, their ratio, and feature-cell overlap."""
    total = float(np.asarray(map_hw, dtype=np.float64).sum()) + 1e-12
    out: dict[str, float] = {"fisher_trace_spatial": total}
    cm = float("nan")
    if coin_yx is not None:
        cm = cell_mass(map_hw, coin_yx[0], coin_yx[1], radius=radius)
        out["coin_mass"] = cm
        out["coin_frac"] = cm / total
    else:
        out["coin_mass"] = float("nan")
        out["coin_frac"] = float("nan")
    if agent_yx is not None:
        pm = path_band_mass(map_hw, agent_y=agent_yx[0], agent_x=agent_yx[1])
        out["path_mass"] = pm
        out["path_frac"] = pm / total
        out["coin_minus_path_frac"] = float(out["coin_frac"] - out["path_frac"])
        # Ratio coin / path; the path floor avoids division by zero.
        if coin_yx is not None and np.isfinite(cm):
            out["coin_over_path"] = float(cm / (pm + 1e-12))
            out.update(coin_path_separation(coin_yx, agent_yx, radius=radius))
        else:
            out["coin_over_path"] = float("nan")
    else:
        out["path_mass"] = float("nan")
        out["path_frac"] = float("nan")
        out["coin_minus_path_frac"] = float("nan")
        out["coin_over_path"] = float("nan")
    return out


def mean_map(maps: list[np.ndarray]) -> np.ndarray:
    if not maps:
        return np.zeros((BLOCK3_HW, BLOCK3_HW), dtype=np.float64)
    return np.mean(np.stack(maps, axis=0), axis=0)


def pack_scalar_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std/median and t-based 95% CI (ci95_lo/hi) for numeric keys across per-frame dicts."""
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.floating))]
    out: dict[str, Any] = {"n": len(rows)}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        _mean, _lo, _hi = _ci95(vals)
        out[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "median": float(np.nanmedian(vals)),
            "ci95_lo": _lo,
            "ci95_hi": _hi,
        }
    return out
