#!/usr/bin/env python3
"""Maze Fisher decision-fork dual-source harvest and scoring (Q, gain, signed target-selection derivative)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gm_geometry.probes.coin_location import pixel_diff_feat_cell
from gm_geometry.probes.fisher_sensitivity import (
    block3_fisher_pack,
    cell_mass,
    fc_fisher_and_act,
    fc_logit_jacobian_and_act,
    fisher_from_logit_jacobian,
    kl_categorical,
    mean_map,
    quadratic_form,
    sensitivity_along,
    spatial_fisher_map_from_block3_grads,
)
from run_maze_i_goal_proxy_ladder import (
    _iscalar,
    _reachable,
    _reachable_extreme,
    _read_grid,
    first_step_action,
    get_agent_cell,
    get_info_maze,
    load_policy,
    make_env,
    obs_after_edit,
    set_cheese,
)

PIXEL_DIFF_THR = 0.05
# The signed derivative is one-sided: the corner is a region, not a sprite, so there is no second editable cue
# and no Fisher cross-term.
PAL = {
    "gm": "#7A1520",
    "nongm": "#3D8B8B",
    "init": "#8E8E8E",
    "ink": "#2A2A2A",
    "paper": "#FFFFFF",
}


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _chw01(env) -> np.ndarray:
    x = np.asarray(obs_after_edit(env), dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    if x.max() > 1.5:
        x = x / 255.0
    return x


# On stock maze/maze_aisc, present=0 clears GOAL and the reward cache
# (cheese_x/y = -1). That is fine for RGB Delta h. There is no restore after
# present=1 at bl because the env is closed immediately after this call.


def _present_gone_relocated(env, cx: int, cy: int, grid: np.ndarray, agent_cell: tuple[int, int]):
    """Present/absent obs at the cheese cell plus the cheese relocated to the reachable 'bl' extreme.

    Returns (obs_p, obs_g, obs_reloc, cheese_feat, reloc_xy), or None on a degenerate or failed step.
    """
    bl = _reachable_extreme(grid, agent_cell, "bl")
    if bl is None or bl == (cx, cy) or bl == agent_cell:
        return None
    if set_cheese(env, cx, cy, 1) != 0:
        return None
    obs_p = _chw01(env)
    if set_cheese(env, 0, 0, 0) != 0:
        return None
    obs_g = _chw01(env)
    info = get_info_maze(env)
    if _iscalar(info, "cheese_x") != -1 or _iscalar(info, "cheese_y") != -1:
        raise RuntimeError(
            "set_cheese(present=0) left cheese_x/y nonzero; reward cache not cleared"
        )
    if float(np.max(np.abs(obs_p - obs_g))) < PIXEL_DIFF_THR:
        return None
    cell = pixel_diff_feat_cell(obs_p, obs_g, thr=PIXEL_DIFF_THR)
    if cell is None:
        return None
    blx, bly = bl
    if set_cheese(env, blx, bly, 1) != 0:
        return None
    obs_reloc = _chw01(env)
    info2 = get_info_maze(env)
    if _iscalar(info2, "cheese_x") != blx or _iscalar(info2, "cheese_y") != bly:
        raise RuntimeError("set_cheese(present=1) at bl did not register the relocated position")
    if float(np.max(np.abs(obs_reloc - obs_p))) < PIXEL_DIFF_THR:
        return None
    return obs_p, obs_g, obs_reloc, (int(cell[0]), int(cell[1])), (int(blx), int(bly))


def signed_derivative_scalars(policy, obs_base, obs_reloc, a_cheese: int, a_corner: int, device: str) -> dict:
    """g_star = grad_h[P(a_cheese) - P(a_corner)]^T Delta_star with Delta_star = h(cheese relocated) - h(base).

    Also reports Q_star, S_star, trFd_star and S_over_trFd_star on the same Delta_star (Q = G * D_F)."""
    f_base, probs_base, h_base = fc_fisher_and_act(policy, obs_base, device)
    j_base, _, _ = fc_logit_jacobian_and_act(policy, obs_base, device)
    _, _, h_reloc = fc_logit_jacobian_and_act(policy, obs_reloc, device)
    d_star = (h_reloc - h_base).astype(np.float64)
    n_star = float(np.linalg.norm(d_star)) + 1e-18
    mean_grad = (probs_base[:, None] * j_base).sum(axis=0)
    grad_r = (
        probs_base[a_cheese] * (j_base[a_cheese] - mean_grad)
        - probs_base[a_corner] * (j_base[a_corner] - mean_grad)
    )
    g_star = float(np.dot(grad_r, d_star))

    q_star = quadratic_form(f_base, d_star)
    s_star = sensitivity_along(f_base, d_star)
    d_fc = f_base.shape[0]
    trFd_star = float(np.trace(f_base)) / d_fc
    s_over_trFd_star = s_star / (trFd_star + 1e-18)

    return {
        "g_star": g_star,
        "ghat_star": g_star / n_star,
        "Q_star": q_star,
        "S_star": s_star,
        "trFd_star": trFd_star,
        "S_over_trFd_star": s_over_trFd_star,
        "D_F_star": n_star ** 2 * trFd_star,
        "r_base": float(probs_base[a_cheese] - probs_base[a_corner]),
        "norm_delta_star": n_star,
    }


def probs_from_overridden_h(policy, h_override: np.ndarray, device: str) -> np.ndarray:
    """Real forward pass with fc's output replaced; used only for the finite-difference check."""
    import torch.nn.functional as tnf

    override = torch.as_tensor(h_override, dtype=torch.float32, device=device).unsqueeze(0)

    def _hook(_m, _i, _out):
        return override

    handle = policy.embedder.fc.register_forward_hook(_hook)
    try:
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        m = torch.zeros(1, device=device)
        with torch.no_grad():
            dist, _value, _nh = policy(dummy, h0, m)
            probs = tnf.softmax(dist.logits, dim=-1)[0].cpu().numpy()
        return probs
    finally:
        handle.remove()


def _classify_step(env) -> tuple[str, dict | None]:
    """Return (status, geo). status in decision|unreachable_cheese|not_diverged|no_info."""
    info = get_info_maze(env)
    wd = _iscalar(info, "maze_world_dim")
    cx, cy = _iscalar(info, "cheese_x"), _iscalar(info, "cheese_y")
    ax, ay = get_agent_cell(env)
    if ax < 0 or cx < 0 or wd <= 0:
        return "no_info", None
    grid = _read_grid(env, wd)
    tr = _reachable_extreme(grid, (ax, ay), "tr")
    if tr is None:
        return "no_info", None
    if (cx, cy) not in _reachable(grid, (ax, ay)):
        return "unreachable_cheese", None
    a_cheese, path_len_cheese = first_step_action(grid, (ax, ay), (cx, cy))
    a_corner, path_len_tr = first_step_action(grid, (ax, ay), tr)
    if a_cheese is None or a_corner is None or a_cheese == a_corner:
        return "not_diverged", None
    d_eucl = float(np.hypot(cx - tr[0], cy - tr[1]))
    return "decision", {
        "agent_cell": (int(ax), int(ay)),
        "cheese_xy": (int(cx), int(cy)),
        "tr_cell": (int(tr[0]), int(tr[1])),
        "a_cheese": int(a_cheese),
        "a_corner": int(a_corner),
        "path_len_cheese": int(path_len_cheese),
        "path_len_tr": int(path_len_tr),
        "d_cheese_tr": d_eucl,
        "grid": grid,
    }


@torch.no_grad()
def _greedy_act(policy, obs_chw, device: str, hidden: np.ndarray):
    x = torch.as_tensor(obs_chw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    h = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    m = torch.zeros(1, device=device)
    dist, _v, nh = policy(x, h, m)
    act = int(dist.probs.argmax(-1).item())
    return act, nh.cpu().numpy()


def _first_ds_snapshot(policy, *, device: str, start_level: int, max_steps: int) -> dict | None:
    env = make_env("maze", start_level=start_level)
    obs = env.reset()
    hidden = np.zeros((1, policy.embedder.output_dim), dtype=np.float32)
    found = None
    for t in range(max_steps):
        chw = np.asarray(obs[0], dtype=np.float32)
        if chw.max() > 1.5:
            chw = chw / 255.0
        status, geo = _classify_step(env)
        if status == "decision" and geo is not None:
            found = {
                "t": t,
                "agent_cell": list(geo["agent_cell"]),
                "cheese_xy": list(geo["cheese_xy"]),
                "tr_cell": list(geo["tr_cell"]),
                "d_cheese_tr": geo["d_cheese_tr"],
            }
            break
        act, hidden = _greedy_act(policy, chw, device, hidden)
        obs, _r, done, _i = env.step(np.asarray([act]))
        if done[0]:
            break
    env.close()
    return found


def determinism_check(
    policy,
    *,
    device: str,
    start_level: int,
    max_steps: int = 200,
    label: str = "",
    scan_levels: int = 64,
) -> dict:
    """Find a level with a first DS, then re-roll that level twice and compare geometry.

    Fixed single-level checks are too brittle: Non-GM may never hit a fork on the
    harvest start_level within max_steps even when later levels harvest fine.
    """
    keys = ("t", "agent_cell", "cheese_xy", "tr_cell")
    probe_level = None
    probe_snap = None
    for offset in range(scan_levels):
        lvl = start_level + offset
        snap = _first_ds_snapshot(policy, device=device, start_level=lvl, max_steps=max_steps)
        if snap is not None:
            probe_level = lvl
            probe_snap = snap
            break
    if probe_level is None:
        return {
            "ok": False,
            "label": label,
            "rep0": None,
            "rep1": None,
            "start_level": start_level,
            "probe_level": None,
            "scan_levels": scan_levels,
            "compared_keys": list(keys),
            "reason": "no_decision_square_in_scan",
        }
    reps = [
        _first_ds_snapshot(policy, device=device, start_level=probe_level, max_steps=max_steps)
        for _ in range(2)
    ]
    ok = reps[0] is not None and reps[1] is not None and all(
        reps[0][k] == reps[1][k] for k in keys
    )
    return {
        "ok": bool(ok),
        "label": label,
        "rep0": reps[0],
        "rep1": reps[1],
        "start_level": start_level,
        "probe_level": probe_level,
        "probe_first": probe_snap,
        "scan_levels": scan_levels,
        "compared_keys": list(keys),
    }


def harvest_decision_fork(
    warm_policy,
    *,
    device: str,
    n_frames: int,
    start_level: int,
    max_levels: int,
    max_steps: int,
    source_name: str,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    level = start_level
    used = 0
    rejects = {
        "timeout": 0,
        "done_before_ds": 0,
        "pixel_gate": 0,
        "set_fail": 0,
        "unreachable_cheese": 0,
    }
    while len(rows) < n_frames and used < max_levels:
        env = make_env("maze", start_level=level)
        obs = env.reset()
        hidden = np.zeros((1, warm_policy.embedder.output_dim), dtype=np.float32)
        kept = False
        saw_unreachable = False
        for t in range(max_steps):
            chw = np.asarray(obs[0], dtype=np.float32)
            if chw.max() > 1.5:
                chw = chw / 255.0
            status, geo = _classify_step(env)
            if status == "unreachable_cheese":
                saw_unreachable = True
            elif status == "decision" and geo is not None:
                cx, cy = geo["cheese_xy"]
                triple = _present_gone_relocated(env, cx, cy, geo["grid"], geo["agent_cell"])
                if triple is None:
                    rejects["pixel_gate"] += 1
                else:
                    obs_p, obs_g, obs_reloc, cheese_feat, reloc_xy = triple
                    rows.append(
                        {
                            "source": source_name,
                            "level": int(level),
                            "t": int(t),
                            "obs_a": obs_p.astype(np.float32),
                            "obs_gone": obs_g.astype(np.float32),
                            "obs_reloc": obs_reloc.astype(np.float32),
                            "cheese_feat": cheese_feat,
                            "reloc_xy": reloc_xy,
                            "agent_cell": geo["agent_cell"],
                            "cheese_xy": geo["cheese_xy"],
                            "tr_cell": geo["tr_cell"],
                            "a_cheese": geo["a_cheese"],
                            "a_corner": geo["a_corner"],
                            "path_len_cheese": geo["path_len_cheese"],
                            "path_len_tr": geo["path_len_tr"],
                            "d_cheese_tr": geo["d_cheese_tr"],
                        }
                    )
                    kept = True
                break
            act, hidden = _greedy_act(warm_policy, chw, device, hidden)
            obs, _r, done, _i = env.step(np.asarray([act]))
            if done[0]:
                rejects["done_before_ds"] += 1
                break
        else:
            rejects["timeout"] += 1
        if not kept and saw_unreachable:
            rejects["unreachable_cheese"] += 1
        env.close()
        used += 1
        level += 1
        if kept and len(rows) % 8 == 0:
            print(
                f"  [{source_name}] harvested {len(rows)}/{n_frames} used={used} rejects={rejects}",
                flush=True,
            )
    meta = {
        "source": source_name,
        "n": len(rows),
        "levels_used": used,
        "start_level": start_level,
        "rejects": rejects,
        "mean_d_cheese_tr": float(np.mean([r["d_cheese_tr"] for r in rows])) if rows else float("nan"),
        "median_d_cheese_tr": float(np.median([r["d_cheese_tr"] for r in rows])) if rows else float("nan"),
        "frac_path_len_cheese_ge2": float(np.mean([r["path_len_cheese"] >= 2 for r in rows]))
        if rows
        else float("nan"),
    }
    print(f"harvest [{source_name}] done {meta}", flush=True)
    return rows, meta


def score_scalars(policy, obs_a, obs_g, device: str) -> dict:
    grads_a, probs_a, acts_a, _m = block3_fisher_pack(policy, obs_a, device)
    _g, _p, acts_g, _ = block3_fisher_pack(policy, obs_g, device)
    f_fc, p_a, h_a = fc_fisher_and_act(policy, obs_a, device)
    _f, p_g, h_g = fc_fisher_and_act(policy, obs_g, device)
    a, c, h, w = grads_a.shape
    f_b3 = fisher_from_logit_jacobian(grads_a.reshape(a, c * h * w), probs_a)
    dh_b3 = (acts_a - acts_g).reshape(-1).astype(np.float64)
    dh_fc = (h_a - h_g).reshape(-1).astype(np.float64)
    d_b3, d_fc = 2048, 256
    hb = float(np.linalg.norm(acts_a.reshape(-1)))
    hf = float(np.linalg.norm(h_a.reshape(-1)))
    nb = float(np.linalg.norm(dh_b3))
    nf = float(np.linalg.norm(dh_fc))
    q_b3 = quadratic_form(f_b3, dh_b3)
    q_fc = quadratic_form(f_fc, dh_fc)
    s_b3 = sensitivity_along(f_b3, dh_b3)
    s_fc = sensitivity_along(f_fc, dh_fc)
    trFd_b3 = float(np.trace(f_b3)) / d_b3
    trFd_fc = float(np.trace(f_fc)) / d_fc
    return {
        "Q_b3": q_b3,
        "Q_fc": q_fc,
        "S_b3": s_b3,
        "S_fc": s_fc,
        "trFd_b3": trFd_b3,
        "trFd_fc": trFd_fc,
        "S_over_trFd_b3": s_b3 / (trFd_b3 + 1e-18),
        "S_over_trFd_fc": s_fc / (trFd_fc + 1e-18),
        "dh_b3": nb,
        "dh_fc": nf,
        "h_b3": hb,
        "h_fc": hf,
        "rel_b3": nb / (hb + 1e-18),
        "rel_fc": nf / (hf + 1e-18),
        "T": nf / (nb + 1e-18),
        "two_kl": 2.0 * kl_categorical(p_a, p_g),
        "entropy": _entropy(p_a),
        "max_prob": float(np.max(p_a)),
        "_grads_a": grads_a,
        "_probs_a": probs_a,
        "_m_present": _m,
    }


def _map_cosine(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-18 or nb < 1e-18:
        return float("nan")
    return float(np.dot(va, vb) / (na * nb))


def score_agent_on_frames(
    policy, frames: list[dict], *, device: str, label: str, n_findiff: int = 0,
    findiff_eps_frac: float = 0.02,
) -> dict:
    scalar_rows = []
    map_rows = []
    maps_diff = []
    maps_shuf = []
    maps_present = []
    grads_list = []
    probs_list = []
    signed_rows = []

    for i, fr in enumerate(frames):
        sc = score_scalars(policy, fr["obs_a"], fr["obs_gone"], device)
        grads_a = sc.pop("_grads_a")
        probs_a = sc.pop("_probs_a")
        m_p = sc.pop("_m_present")
        _g, _p, _a, m_g = block3_fisher_pack(policy, fr["obs_gone"], device)
        m_d = m_p - m_g
        maps_present.append(m_p)
        maps_diff.append(m_d)
        grads_list.append(grads_a)
        probs_list.append(probs_a)
        scalar_rows.append(sc)

        cy, cx = fr["cheese_feat"]
        abs_d = np.abs(m_d)
        tot = float(abs_d.sum()) + 1e-12
        cheese_frac = cell_mass(abs_d, cy, cx, radius=0) / tot
        map_rows.append({"cheese_frac_absdiff": float(cheese_frac)})

        signed = signed_derivative_scalars(
            policy, fr["obs_a"], fr["obs_reloc"], fr["a_cheese"], fr["a_corner"], device,
        )
        if i < n_findiff:
            j_base, probs_base, h_base = fc_logit_jacobian_and_act(policy, fr["obs_a"], device)
            _, _, h_reloc = fc_logit_jacobian_and_act(policy, fr["obs_reloc"], device)
            d_star = (h_reloc - h_base).astype(np.float64)
            n_star = float(np.linalg.norm(d_star)) + 1e-18
            eps = findiff_eps_frac * n_star
            u = d_star / n_star

            def r_cheese_corner(probs, a_cheese=fr["a_cheese"], a_corner=fr["a_corner"]):
                return float(probs[a_cheese] - probs[a_corner])

            r_plus = r_cheese_corner(probs_from_overridden_h(policy, h_base + eps * u, device))
            r_minus = r_cheese_corner(probs_from_overridden_h(policy, h_base - eps * u, device))
            fd_ghat = (r_plus - r_minus) / (2 * eps)
            signed["findiff_check"] = {
                "eps": eps,
                "fd_ghat_star": float(fd_ghat),
                "analytic_ghat_star": signed["ghat_star"],
                "rel_err": float(abs(fd_ghat - signed["ghat_star"]) / (abs(signed["ghat_star"]) + 1e-9)),
            }
        signed_rows.append(signed)

    pi_bar = np.mean(np.stack(probs_list, axis=0), axis=0)
    cosines = []
    for i, grads_a in enumerate(grads_list):
        m_shuf = spatial_fisher_map_from_block3_grads(grads_a, pi_bar)
        maps_shuf.append(m_shuf)
        cos = _map_cosine(maps_present[i], m_shuf)
        cosines.append(cos)
        map_rows[i]["cosine_present_shuf"] = cos
        cy, cx = frames[i]["cheese_feat"]
        abs_s = np.abs(m_shuf)
        map_rows[i]["cheese_frac_shuf"] = float(
            cell_mass(abs_s, cy, cx, radius=0) / (float(abs_s.sum()) + 1e-12)
        )

    def med(key, rows):
        return float(np.nanmedian([r[key] for r in rows]))

    scalar_keys = [
        "Q_b3",
        "Q_fc",
        "S_b3",
        "S_fc",
        "trFd_b3",
        "trFd_fc",
        "S_over_trFd_b3",
        "S_over_trFd_fc",
        "dh_b3",
        "dh_fc",
        "h_b3",
        "h_fc",
        "rel_b3",
        "rel_fc",
        "T",
        "two_kl",
        "entropy",
        "max_prob",
    ]
    summary = {f"{k}_median": med(k, scalar_rows) for k in scalar_keys}
    summary["cheese_frac_absdiff_median"] = med("cheese_frac_absdiff", map_rows)
    summary["cosine_present_shuf_median"] = med("cosine_present_shuf", map_rows)

    g_star_vals = [r["g_star"] for r in signed_rows]
    ghat_star_vals = [r["ghat_star"] for r in signed_rows]

    def _mm(key: str) -> tuple[float, float]:
        vals = np.asarray([r[key] for r in signed_rows], dtype=np.float64)
        return float(np.mean(vals)), float(np.median(vals))

    q_mean, q_median = _mm("Q_star")
    s_mean, s_median = _mm("S_star")
    trfd_mean, trfd_median = _mm("trFd_star")
    gain_mean, gain_median = _mm("S_over_trFd_star")
    df_mean, df_median = _mm("D_F_star")
    signed_summary = {
        "g_star_mean": float(np.mean(g_star_vals)),
        "g_star_median": float(np.median(g_star_vals)),
        "ghat_star_mean": float(np.mean(ghat_star_vals)),
        "ghat_star_median": float(np.median(ghat_star_vals)),
        "Q_star_mean": q_mean, "Q_star_median": q_median,
        "S_star_mean": s_mean, "S_star_median": s_median,
        "trFd_star_mean": trfd_mean, "trFd_star_median": trfd_median,
        "S_over_trFd_star_mean": gain_mean, "S_over_trFd_star_median": gain_median,
        "D_F_star_mean": df_mean, "D_F_star_median": df_median,
        "r_base_mean": float(np.mean([r["r_base"] for r in signed_rows])),
        "n_g_star_positive": int(sum(1 for v in g_star_vals if v > 0)),
        "n": len(signed_rows),
    }
    findiff_rows = [r["findiff_check"] for r in signed_rows if "findiff_check" in r]
    if findiff_rows:
        signed_summary["findiff_rel_err_median"] = float(
            np.median([f["rel_err"] for f in findiff_rows])
        )

    return {
        "label": label,
        "summary": summary,
        "per_frame_scalars": [{k: float(r[k]) for k in scalar_keys} for r in scalar_rows],
        "per_frame_maps": map_rows,
        "mean_map_diff": mean_map(maps_diff).tolist(),
        "mean_map_shuf": mean_map(maps_shuf).tolist(),
        "mean_map_present": mean_map(maps_present).tolist(),
        "signed_derivative": signed_summary,
        "signed_derivative_per_frame": signed_rows,
    }


def recipe_stats(agents: dict[str, dict]) -> dict:
    """Recipe aggregates = mean of per-seed medians (same as Coin multi-seed tables)."""
    ng = [agents[k]["summary"] for k in agents if k.startswith("nongm_")]
    gm = [agents[k]["summary"] for k in agents if k.startswith("gm_")]

    def mean_key(rows, key):
        return float(np.mean([r[key] for r in rows]))

    out = {
        "aggregator": "mean_of_seed_medians",
        "n_nongm_seeds": len(ng),
        "n_gm_seeds": len(gm),
    }
    for key in (
        "Q_fc_median",
        "Q_b3_median",
        "S_over_trFd_b3_median",
        "S_over_trFd_fc_median",
        "dh_fc_median",
        "dh_b3_median",
        "T_median",
        "entropy_median",
        "cheese_frac_absdiff_median",
        "cosine_present_shuf_median",
    ):
        ng_m = mean_key(ng, key)
        gm_m = mean_key(gm, key)
        out[f"ng_{key}"] = ng_m
        out[f"gm_{key}"] = gm_m
        out[f"ratio_{key}"] = ng_m / (gm_m + 1e-18)

    ng_cf = [agents[k]["summary"]["cheese_frac_absdiff_median"] for k in agents if k.startswith("nongm_")]
    gm_cf = [agents[k]["summary"]["cheese_frac_absdiff_median"] for k in agents if k.startswith("gm_")]
    out["cheese_frac_abs_diff"] = float(abs(np.mean(ng_cf) - np.mean(gm_cf)))
    out["complete_sep_ng_gt_gm_cheese_frac"] = bool(min(ng_cf) > max(gm_cf))
    out["complete_sep_gm_gt_ng_cheese_frac"] = bool(min(gm_cf) > max(ng_cf))
    out["complete_sep_ng_gt_gm_Q_fc"] = bool(
        min(agents[k]["summary"]["Q_fc_median"] for k in agents if k.startswith("nongm_"))
        > max(agents[k]["summary"]["Q_fc_median"] for k in agents if k.startswith("gm_"))
    )
    return out


def save_maps_figure(by_source: dict, out_png: Path) -> None:
    """Shared colour scale across Non-GM and GM panels within each column."""
    panels = []
    for src, agents in by_source.items():
        for lab in ("nongm_1090", "gm_1080", "init_0"):
            if lab in agents:
                panels.append((src, lab, agents[lab]))
            elif lab == "init_0" and "init" in agents:
                panels.append((src, "init", agents["init"]))
    if not panels:
        return
    # Shared vmax per column over trained agents only (exclude init from scale).
    trained = [(s, lab, res) for s, lab, res in panels if not str(lab).startswith("init")]
    vmax_diff = max(
        float(np.abs(np.asarray(res["mean_map_diff"], dtype=np.float64)).max())
        for _, _, res in trained
    ) if trained else 1.0
    vmax_shuf = max(
        float(np.asarray(res["mean_map_shuf"], dtype=np.float64).max())
        for _, _, res in trained
    ) if trained else 1.0
    fig, axes = plt.subplots(len(panels), 2, figsize=(5.6, 2.4 * len(panels)), squeeze=False)
    for i, (src, lab, res) in enumerate(panels):
        for j, (key, title, vmax) in enumerate(
            (
                ("mean_map_diff", "|pres−abs|", vmax_diff),
                ("mean_map_shuf", "shuf-π", vmax_shuf),
            )
        ):
            ax = axes[i][j]
            m = np.asarray(res[key], dtype=np.float64)
            if "diff" in key:
                m = np.abs(m)
            im = ax.imshow(m, origin="upper", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-12))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, color=PAL["ink"], fontsize=10)
            if j == 0:
                color = PAL["gm"] if "gm" in lab else PAL["nongm"] if "nongm" in lab else PAL["init"]
                ax.set_ylabel(f"{src}\n{lab}", color=color, fontsize=9)
            if i == len(panels) - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=PAL["paper"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames-per-source", type=int, default=48)
    ap.add_argument("--start-level-gm", type=int, default=1_000_000)
    ap.add_argument("--start-level-nongm", type=int, default=1_010_000)
    ap.add_argument("--max-levels", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    ck = args.ckpt_dir

    paths = {
        "gm_1080": ck / "1080_gm.pth",
        "gm_1081": ck / "1081_gm.pth",
        "gm_1082": ck / "1082_gm.pth",
        "nongm_1090": ck / "1090_teach.pth",
        "nongm_1091": ck / "1091_teach.pth",
        "nongm_1092": ck / "1092_teach.pth",
    }
    for p in paths.values():
        if not p.is_file():
            raise SystemExit(f"missing ckpt {p}")

    print("=== load warm policies ===", flush=True)
    warm_gm = load_policy(paths["gm_1080"], device)
    warm_ng = load_policy(paths["nongm_1090"], device)

    det = {"skipped": True}
    if not args.skip_determinism:
        print("=== determinism check both warm policies ===", flush=True)
        det_gm = determinism_check(
            warm_gm,
            device=device,
            start_level=args.start_level_gm,
            max_steps=args.max_steps,
            label="gm_1080",
        )
        det_ng = determinism_check(
            warm_ng,
            device=device,
            start_level=args.start_level_nongm,
            max_steps=args.max_steps,
            label="nongm_1090",
        )
        det = {"gm_1080": det_gm, "nongm_1090": det_ng, "ok": bool(det_gm["ok"] and det_ng["ok"])}
        print(json.dumps(det), flush=True)
        if not det["ok"]:
            raise SystemExit("determinism check failed on one or both warm policies")

    print("=== harvest GM source ===", flush=True)
    frames_gm, meta_gm = harvest_decision_fork(
        warm_gm,
        device=device,
        n_frames=args.n_frames_per_source,
        start_level=args.start_level_gm,
        max_levels=args.max_levels,
        max_steps=args.max_steps,
        source_name="gm_rollouts",
    )
    print("=== harvest Non-GM source ===", flush=True)
    frames_ng, meta_ng = harvest_decision_fork(
        warm_ng,
        device=device,
        n_frames=args.n_frames_per_source,
        start_level=args.start_level_nongm,
        max_levels=args.max_levels,
        max_steps=args.max_steps,
        source_name="nongm_rollouts",
    )
    if len(frames_gm) < args.n_frames_per_source or len(frames_ng) < args.n_frames_per_source:
        raise SystemExit(
            f"insufficient frames: gm={len(frames_gm)} ng={len(frames_ng)} want={args.n_frames_per_source}"
        )

    # Mean cheese-to-corner distance of the full GM decision-square pool (all squares, not only the first per level).
    PREREG_D_CHEESE_TR = 11.66
    GEOM_TOL = 2.5
    separation_check = {}
    for name, meta in (("gm_rollouts", meta_gm), ("nongm_rollouts", meta_ng)):
        mean_d = float(meta["mean_d_cheese_tr"])
        delta = abs(mean_d - PREREG_D_CHEESE_TR)
        ok = delta <= GEOM_TOL
        separation_check[name] = {
            "mean_d_cheese_tr": mean_d,
            "prereg_recollect_mean": PREREG_D_CHEESE_TR,
            "abs_delta": delta,
            "tol": GEOM_TOL,
            "matches_prereg_pool": ok,
            "note": (
                "This harvest takes the first decision square from step 0. "
                "Gate A 11.7 was the full recollect JSONL (later DS on GM paths). "
                "A miss is a sampling mismatch, not automatic claim failure."
            ),
        }
        if not ok:
            print(
                f"WARNING geometry mismatch [{name}]: mean_d_cheese_tr={mean_d:.2f} "
                f"vs prereg {PREREG_D_CHEESE_TR} (|delta|={delta:.2f} > {GEOM_TOL}). "
                f"Do not treat Gate A 11.7 as met.",
                flush=True,
            )
        else:
            print(
                f"geometry ok [{name}]: mean_d_cheese_tr={mean_d:.2f} ≈ {PREREG_D_CHEESE_TR}",
                flush=True,
            )

    print("=== load all agents ===", flush=True)
    agents_pol = {lab: load_policy(p, device) for lab, p in paths.items()}
    for i, seed in enumerate((0, 1, 2)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        agents_pol[f"init_{i}"] = load_policy(paths["nongm_1090"], device, random_init=True)

    by_source = {}
    for src_name, frames, meta in (
        ("gm_rollouts", frames_gm, meta_gm),
        ("nongm_rollouts", frames_ng, meta_ng),
    ):
        print(f"=== score all agents on {src_name} ===", flush=True)
        scored = {}
        for lab, pol in agents_pol.items():
            print(f"  score {lab}", flush=True)
            n_findiff = 8 if lab == "gm_1080" else 0
            scored[lab] = score_agent_on_frames(
                pol, frames, device=device, label=lab, n_findiff=n_findiff,
            )
        by_source[src_name] = {
            "harvest": meta,
            "agents": {k: {kk: vv for kk, vv in v.items() if kk != "per_frame_scalars"} for k, v in scored.items()},
            "agents_full": scored,
            "recipe": recipe_stats(scored),
            "frame_meta": [
                {
                    "level": r["level"],
                    "t": r["t"],
                    "d_cheese_tr": r["d_cheese_tr"],
                    "path_len_cheese": r["path_len_cheese"],
                    "agent_cell": list(r["agent_cell"]),
                    "cheese_xy": list(r["cheese_xy"]),
                    "cheese_feat": list(r["cheese_feat"]),
                }
                for r in frames
            ],
        }

    # Slim agents for JSON (drop mean maps into separate npz)
    slim = {}
    npz_payload = {}
    for src, block in by_source.items():
        slim[src] = {
            "harvest": block["harvest"],
            "recipe": block["recipe"],
            "frame_meta": block["frame_meta"],
            "agents": {},
        }
        for lab, res in block["agents_full"].items():
            slim[src]["agents"][lab] = {
                "summary": res["summary"],
                "per_frame_maps": res["per_frame_maps"],
                "signed_derivative": res["signed_derivative"],
                "signed_derivative_per_frame": res["signed_derivative_per_frame"],
            }
            for mk in ("mean_map_diff", "mean_map_shuf", "mean_map_present"):
                npz_payload[f"{src}__{lab}__{mk}"] = np.asarray(res[mk], dtype=np.float32)

    save_maps_figure(
        {s: {k: by_source[s]["agents_full"][k] for k in by_source[s]["agents_full"]} for s in by_source},
        out / "mean_absdiff_maps.png",
    )
    np.savez_compressed(out / "mean_maps.npz", **npz_payload)

    summary = {
        "prereg": None,
        "protocol": "maze decision-fork dual-source cheese present/absent Fisher",
        "determinism": det,
        "separation_check": separation_check,
        "sources": slim,
        "gate_a_claim1": {
            src: {
                "cheese_frac_abs_diff": slim[src]["recipe"]["cheese_frac_abs_diff"],
                "null_abs_diff_le_0p02": slim[src]["recipe"]["cheese_frac_abs_diff"] <= 0.02,
                "complete_sep_ng_gt_gm": slim[src]["recipe"]["complete_sep_ng_gt_gm_cheese_frac"],
                "complete_sep_gm_gt_ng": slim[src]["recipe"]["complete_sep_gm_gt_ng_cheese_frac"],
                "null_no_complete_sep": (
                    not slim[src]["recipe"]["complete_sep_ng_gt_gm_cheese_frac"]
                    and not slim[src]["recipe"]["complete_sep_gm_gt_ng_cheese_frac"]
                ),
                "mean_d_cheese_tr_used": slim[src]["harvest"]["mean_d_cheese_tr"],
                "geometry_matches_prereg_pool": separation_check[src]["matches_prereg_pool"],
                "cosine_shuf_report_only": {
                    k: slim[src]["agents"][k]["summary"]["cosine_present_shuf_median"]
                    for k in slim[src]["agents"]
                },
            }
            for src in slim
        },
        "gate_a_claim2": {
            src: {
                "complete_sep_Q_fc": slim[src]["recipe"]["complete_sep_ng_gt_gm_Q_fc"],
                "ratio_Q_fc": slim[src]["recipe"]["ratio_Q_fc_median"],
                "ratio_dh_fc": slim[src]["recipe"]["ratio_dh_fc_median"],
            }
            for src in slim
        },
        "signed_derivative_summary": {
            src: {
                lab: slim[src]["agents"][lab]["signed_derivative"]
                for lab in slim[src]["agents"]
            }
            for src in slim
        },
    }
    (out / "decision_fork_dual_source_v1.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "separation_check": summary["separation_check"],
                "gate_a_claim1": summary["gate_a_claim1"],
                "gate_a_claim2": summary["gate_a_claim2"],
                "determinism": summary["determinism"],
                "signed_derivative_summary": summary["signed_derivative_summary"],
            },
            indent=2,
        )
    )
    print("wrote", out / "decision_fork_dual_source_v1.json", flush=True)


if __name__ == "__main__":
    main()
