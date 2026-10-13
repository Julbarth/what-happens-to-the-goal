#!/usr/bin/env python3
"""Maze signed-derivative Fisher at fc and block3 on one matched frame bank (Q/G/D_F, g_star, ghat_star)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gm_geometry.probes.fisher_sensitivity import (
    block3_fisher_pack,
    fisher_from_logit_jacobian,
    quadratic_form,
    sensitivity_along,
)
from run_maze_fisher_decision_fork import (
    determinism_check,
    fc_fisher_and_act,
    fc_logit_jacobian_and_act,
    harvest_decision_fork,
    signed_derivative_scalars as signed_derivative_scalars_fc,
)
from run_maze_i_goal_proxy_ladder import load_policy

# At both sites Delta_e = h(cheese relocated) - h(base) with F at the base state. Cohort summaries
# average per-seed medians, so the seed (n=3 per cohort) stays the unit.


def block3_fisher_and_act(policy, obs_chw: np.ndarray, device: str):
    """(F [D',D'], probs [A], h [D'], J [A,D']) at block3, D'=C*H*W."""
    grads, probs, acts, _m = block3_fisher_pack(policy, obs_chw, device)
    a, c, h, w = grads.shape
    j = grads.reshape(a, c * h * w).astype(np.float64)
    f_mat = fisher_from_logit_jacobian(j, probs)
    h_flat = acts.reshape(-1).astype(np.float64)
    return f_mat, probs, h_flat, j


def signed_derivative_scalars_block3(policy, obs_base, obs_reloc, a_cheese: int, a_corner: int, device: str) -> dict:
    """Block3-hooked mirror of signed_derivative_scalars (fc version) -- identical
    math, only the representation site changes."""
    f_base, probs_base, h_base, j_base = block3_fisher_and_act(policy, obs_base, device)
    _, _, h_reloc, _ = block3_fisher_and_act(policy, obs_reloc, device)
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
    d_dim = f_base.shape[0]
    trFd_star = float(np.trace(f_base)) / d_dim
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


def _verify_qform_shortcut_matches_full(policy, obs_base: np.ndarray, device: str) -> dict:
    """Check (J v)^T W (J v) == v^T F v at fc on a real frame for a random probe direction v."""
    f_full, probs, h = fc_fisher_and_act(policy, obs_base, device)
    j, _, _ = fc_logit_jacobian_and_act(policy, obs_base, device)
    rng = np.random.RandomState(0)
    v = rng.randn(h.shape[0]).astype(np.float64)
    q_full = float(v @ f_full @ v)
    jv = j.astype(np.float64) @ v
    p = probs.astype(np.float64)
    w_mat = np.diag(p) - np.outer(p, p)
    q_shortcut = float(jv @ w_mat @ jv)
    rel_err = abs(q_full - q_shortcut) / (abs(q_full) + 1e-18)
    return {"q_full": q_full, "q_shortcut": q_shortcut, "rel_err": rel_err, "ok": rel_err < 1e-6}


def _mm(rows: list[dict], key: str) -> tuple[float, float]:
    vals = np.asarray([r[key] for r in rows], dtype=np.float64)
    return float(np.mean(vals)), float(np.median(vals))


def _summarize(rows: list[dict]) -> dict:
    q_mean, q_median = _mm(rows, "Q_star")
    s_mean, s_median = _mm(rows, "S_star")
    trfd_mean, trfd_median = _mm(rows, "trFd_star")
    gain_mean, gain_median = _mm(rows, "S_over_trFd_star")
    df_mean, df_median = _mm(rows, "D_F_star")
    g_mean, g_median = _mm(rows, "g_star")
    ghat_mean, ghat_median = _mm(rows, "ghat_star")
    return {
        "g_star_mean": g_mean, "g_star_median": g_median,
        "ghat_star_mean": ghat_mean, "ghat_star_median": ghat_median,
        "Q_star_mean": q_mean, "Q_star_median": q_median,
        "S_star_mean": s_mean, "S_star_median": s_median,
        "trFd_star_mean": trfd_mean, "trFd_star_median": trfd_median,
        "S_over_trFd_star_mean": gain_mean, "S_over_trFd_star_median": gain_median,
        "D_F_star_mean": df_mean, "D_F_star_median": df_median,
        "n": len(rows),
    }


def score_signed_both_sites(policy, frames: list[dict], device: str) -> dict:
    """Score fc and block3 on the SAME frame list, per frame -- guarantees the two
    hooks are compared on an identical matched bank, not resampled independently."""
    rows_fc, rows_b3 = [], []
    for fr in frames:
        rows_fc.append(
            signed_derivative_scalars_fc(policy, fr["obs_a"], fr["obs_reloc"], fr["a_cheese"], fr["a_corner"], device)
        )
        rows_b3.append(
            signed_derivative_scalars_block3(policy, fr["obs_a"], fr["obs_reloc"], fr["a_cheese"], fr["a_corner"], device)
        )
    return {
        "fc": {**_summarize(rows_fc), "per_frame": rows_fc},
        "block3": {**_summarize(rows_b3), "per_frame": rows_b3},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=128)
    ap.add_argument("--start-level-gm", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    ck = args.ckpt_dir

    paths = {
        "gm_1080": ck / "1080_gm.pth", "gm_1081": ck / "1081_gm.pth", "gm_1082": ck / "1082_gm.pth",
        "nongm_1090": ck / "1090_teach.pth", "nongm_1091": ck / "1091_teach.pth", "nongm_1092": ck / "1092_teach.pth",
    }
    for p in paths.values():
        if not p.is_file():
            raise SystemExit(f"missing ckpt {p}")

    print(f"=== load warm gm policy + harvest gm_rollouts, n={args.n_frames} (matched fc+block3 bank) ===", flush=True)
    warm_gm = load_policy(paths["gm_1080"], device)

    det = {"skipped": True}
    if not args.skip_determinism:
        det_gm = determinism_check(warm_gm, device=device, start_level=args.start_level_gm,
                                    max_steps=args.max_steps, label="gm_1080")
        det = {"gm_1080": det_gm, "ok": bool(det_gm["ok"])}
        print(json.dumps(det), flush=True)
        if not det["ok"]:
            raise SystemExit("determinism check failed")

    frames, meta = harvest_decision_fork(
        warm_gm, device=device, n_frames=args.n_frames, start_level=args.start_level_gm,
        max_levels=args.max_levels, max_steps=args.max_steps, source_name="gm_rollouts",
    )
    if len(frames) < args.n_frames:
        raise SystemExit(f"insufficient frames: {len(frames)} < {args.n_frames}")
    n_distinct_levels = len({fr["level"] for fr in frames})
    if n_distinct_levels != len(frames):
        raise SystemExit(f"frames are not all distinct levels: {n_distinct_levels} distinct of {len(frames)}")

    print("=== verify JΔ-shortcut identity against full-matrix Fisher (fc, cheap) on frame 0 ===", flush=True)
    qform_check = _verify_qform_shortcut_matches_full(warm_gm, frames[0]["obs_a"], device)
    print(json.dumps(qform_check), flush=True)
    if not qform_check["ok"]:
        raise SystemExit(f"Q-form identity check failed: {qform_check}")

    print("=== load all agents (6 seeds + 3 init draws) ===", flush=True)
    agents_pol = {lab: load_policy(p, device) for lab, p in paths.items()}
    for i, seed in enumerate((0, 1, 2)):
        torch.manual_seed(seed)
        np.random.seed(seed)
        agents_pol[f"init_{i}"] = load_policy(paths["nongm_1090"], device, random_init=True)

    results = {}
    for lab, pol in agents_pol.items():
        print(f"  scoring fc+block3 signed-derivative: {lab}", flush=True)
        results[lab] = score_signed_both_sites(pol, frames, device)

    def cohort_mean(site: str, prefix: str, key: str) -> float:
        vals = [results[k][site][key] for k in results if k.startswith(prefix)]
        return float(np.mean(vals))

    cohort_summary = {}
    for site in ("fc", "block3"):
        cohort_summary[site] = {}
        for key in ("Q_star_median", "S_over_trFd_star_median", "D_F_star_median",
                    "g_star_median", "ghat_star_median"):
            cohort_summary[site][f"gm_{key}"] = cohort_mean(site, "gm_", key)
            cohort_summary[site][f"nongm_{key}"] = cohort_mean(site, "nongm_", key)
            cohort_summary[site][f"init_{key}"] = cohort_mean(site, "init_", key)

    out_payload = {
        "protocol": f"maze fc+block3 signed-derivative Fisher, gm_rollouts source, "
                    f"n={args.n_frames}/seed (matched bank, both hooks scored on identical frames), "
                    "same cheese-relocation-to-bl-extreme counterfactual, harvest/filter rules unchanged "
                    "from the original 48-level pipeline",
        "n_frames": args.n_frames,
        "n_distinct_levels": n_distinct_levels,
        "determinism": det,
        "harvest_meta": meta,
        "qform_shortcut_check": qform_check,
        "per_seed": {
            lab: {
                site: {k: v for k, v in results[lab][site].items() if k != "per_frame"}
                for site in ("fc", "block3")
            }
            for lab in results
        },
        "cohort_summary": cohort_summary,
    }
    (out / "maze_fc_block3_signed_derivative.json").write_text(json.dumps(out_payload, indent=2))
    per_frame_payload = {
        lab: {site: results[lab][site]["per_frame"] for site in ("fc", "block3")}
        for lab in results
    }
    (out / "maze_fc_block3_signed_derivative_per_frame.json").write_text(json.dumps(per_frame_payload, indent=2))
    print("wrote", out / "maze_fc_block3_signed_derivative.json", flush=True)
    print(json.dumps(cohort_summary, indent=2))


if __name__ == "__main__":
    main()
