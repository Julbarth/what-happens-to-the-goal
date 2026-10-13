#!/usr/bin/env python3
"""Coin signed target-selection derivative and Fisher terms (Q, S) on dual-source decision windows at h_fc."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gm_geometry.probes.coin_location import RIGHT_ACTIONS
from gm_geometry.probes.fisher_sensitivity import (
    fc_fisher_and_act,
    fc_logit_jacobian_and_act,
    quadratic_form,
    sensitivity_along,
)
from run_coin_fisher_decision_window import harvest_decision_windows
from run_coin_fisher_sensitivity import load_init_policy, load_policy

# States are harvested under a GM and a Non-GM policy and every agent is scored on both sources. Coin has one
# editable goal object and a behavioural proxy (go right), so the signed derivative is one-sided (no cross-term).


def probs_from_overridden_h(policy, h_override: np.ndarray, device: str) -> np.ndarray:
    """Action probabilities from a real forward pass with embedder.fc's output replaced (finite-difference check)."""
    override = torch.as_tensor(h_override, dtype=torch.float32, device=device).unsqueeze(0)
    handle = policy.embedder.fc.register_forward_hook(lambda _m, _i, _o: override)
    try:
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        with torch.no_grad():
            dist, _value, _nh = policy(dummy, h0, torch.zeros(1, device=device))
            return torch.softmax(dist.logits, dim=-1)[0].cpu().numpy()
    finally:
        handle.remove()

def signed_and_fisher_scalars(
    policy, obs_base: np.ndarray, obs_reloc: np.ndarray, a_coin: int, device: str,
) -> dict:
    """Signed gradient g_star and Fisher terms Q_star, S_star, S_over_trFd_star on Delta_star = h(reloc) - h(base)."""
    f_base, probs_base, h_base = fc_fisher_and_act(policy, obs_base, device)
    j_base, _, _ = fc_logit_jacobian_and_act(policy, obs_base, device)
    _, _, h_reloc = fc_logit_jacobian_and_act(policy, obs_reloc, device)

    d_star = (h_reloc - h_base).astype(np.float64)
    n_star = float(np.linalg.norm(d_star)) + 1e-18

    mean_grad = (probs_base[:, None] * j_base).sum(axis=0)
    p_right = float(probs_base[list(RIGHT_ACTIONS)].sum())
    grad_right = np.zeros_like(mean_grad)
    for a in RIGHT_ACTIONS:
        grad_right += probs_base[a] * (j_base[a] - mean_grad)
    grad_r = probs_base[a_coin] * (j_base[a_coin] - mean_grad) - grad_right
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
        "norm_delta_star": n_star,
        "r_base": float(probs_base[a_coin] - p_right),
    }


def score_agent_on_frames(
    policy, frames: list[dict], *, device: str, label: str,
    n_findiff: int = 0, findiff_eps_frac: float = 0.02,
) -> dict:
    rows = []
    for i, fr in enumerate(frames):
        scal = signed_and_fisher_scalars(policy, fr["obs_a"], fr["obs_reloc"], fr["a_coin"], device)
        if i < n_findiff:
            _, _, h_base = fc_logit_jacobian_and_act(policy, fr["obs_a"], device)
            _, _, h_reloc = fc_logit_jacobian_and_act(policy, fr["obs_reloc"], device)
            d_star = (h_reloc - h_base).astype(np.float64)
            n_star = float(np.linalg.norm(d_star)) + 1e-18
            eps = findiff_eps_frac * n_star
            u = d_star / n_star

            def r_coin_right(probs, a_coin=fr["a_coin"]):
                return float(probs[a_coin] - probs[list(RIGHT_ACTIONS)].sum())

            r_plus = r_coin_right(probs_from_overridden_h(policy, h_base + eps * u, device))
            r_minus = r_coin_right(probs_from_overridden_h(policy, h_base - eps * u, device))
            fd_ghat = (r_plus - r_minus) / (2 * eps)
            scal["findiff_check"] = {
                "eps": eps,
                "fd_ghat_star": float(fd_ghat),
                "analytic_ghat_star": scal["ghat_star"],
                "rel_err": float(abs(fd_ghat - scal["ghat_star"]) / (abs(scal["ghat_star"]) + 1e-9)),
            }
        scal["level"] = int(fr["level"])
        rows.append(scal)
        if (i + 1) % 8 == 0:
            print(f"  [{label}] {i + 1}/{len(frames)}", flush=True)

    def _agg(key):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals))}

    g_vals = [r["g_star"] for r in rows]
    findiff_rows = [r["findiff_check"] for r in rows if "findiff_check" in r]
    summary = {
        "g_star_mean": float(np.mean(g_vals)),
        "g_star_median": float(np.median(g_vals)),
        "ghat_star_mean": _agg("ghat_star")["mean"],
        "ghat_star_median": _agg("ghat_star")["median"],
        "Q_star_mean": _agg("Q_star")["mean"],
        "Q_star_median": _agg("Q_star")["median"],
        "S_star_mean": _agg("S_star")["mean"],
        "S_star_median": _agg("S_star")["median"],
        "trFd_star_mean": _agg("trFd_star")["mean"],
        "trFd_star_median": _agg("trFd_star")["median"],
        "S_over_trFd_star_mean": _agg("S_over_trFd_star")["mean"],
        "S_over_trFd_star_median": _agg("S_over_trFd_star")["median"],
        "D_F_star_mean": _agg("D_F_star")["mean"],
        "D_F_star_median": _agg("D_F_star")["median"],
        "r_base_mean": float(np.mean([r["r_base"] for r in rows])),
        "n_g_star_positive": int(sum(1 for v in g_vals if v > 0)),
        "n": len(rows),
    }
    if findiff_rows:
        summary["findiff_rel_err_median"] = float(np.median([f["rel_err"] for f in findiff_rows]))
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--start-level", type=int, default=1_100_000)
    ap.add_argument("--max-levels", type=int, default=8000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-findiff", type=int, default=8)
    ap.add_argument("--findiff-eps-frac", type=float, default=0.02)
    ap.add_argument("--skip-init", action="store_true")
    ap.add_argument("--n-init", type=int, default=1,
                     help="Number of independent random-init draws. 1 = a single unseeded draw labelled "
                          "'init'; >1 labels draws init_0..init_{n-1}, each with an explicit seed.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    ck_nongm = [args.ckpt_dir / f"{s}_teach.pth" for s in (7001, 7002, 7003)]
    ck_gm = [args.ckpt_dir / "6033_gm_20M.pth", args.ckpt_dir / "6037_gm_120M.pth"]
    ck_gm.append(args.ckpt_dir / "6051_gm_55M.pth")
    for c in ck_nongm + ck_gm:
        if not c.is_file():
            raise SystemExit(f"missing ckpt {c}")

    schedule = [
        ("nongm_7001", ck_nongm[0]), ("nongm_7002", ck_nongm[1]), ("nongm_7003", ck_nongm[2]),
        ("gm_6033", ck_gm[0]), ("gm_6037", ck_gm[1]), ("gm_6051", ck_gm[2]),
    ]

    print("=== harvest gm_rollouts (warm+gate policy = gm_6033) ===", flush=True)
    gm_warm = load_policy(ck_gm[0], device)
    frames_gm_src = harvest_decision_windows(
        n_frames=args.n_frames, start_level=args.start_level,
        max_levels=args.max_levels, warm_policy=gm_warm, device=device,
    )
    if len(frames_gm_src) < 16:
        raise SystemExit(f"too few gm_rollouts decision frames: {len(frames_gm_src)}")

    print("=== harvest nongm_rollouts (warm+gate policy = nongm_7001) ===", flush=True)
    nongm_warm = load_policy(ck_nongm[0], device)
    frames_nongm_src = harvest_decision_windows(
        n_frames=args.n_frames, start_level=args.start_level + 20_000,
        max_levels=args.max_levels, warm_policy=nongm_warm, device=device,
    )
    if len(frames_nongm_src) < 16:
        raise SystemExit(f"too few nongm_rollouts decision frames: {len(frames_nongm_src)}")

    # obs_reloc must be computed at harvest time: a later env reset would lose the agent position.
    assert "obs_reloc" in frames_gm_src[0], "harvest_decision_windows must provide obs_reloc"

    state_sources = {"gm_rollouts": frames_gm_src, "nongm_rollouts": frames_nongm_src}
    results: dict = {}
    per_frame: dict = {}
    for src_name, frames in state_sources.items():
        print(f"\n=== score all agents on {src_name} ===", flush=True)
        results[src_name] = {}
        per_frame[src_name] = {}
        for lab, path in schedule:
            pol = load_policy(path, device)
            n_findiff = args.n_findiff if (src_name == "gm_rollouts" and lab == "gm_6033") else 0
            out = score_agent_on_frames(
                pol, frames, device=device, label=lab,
                n_findiff=n_findiff, findiff_eps_frac=args.findiff_eps_frac,
            )
            results[src_name][lab] = out["summary"]
            per_frame[src_name][lab] = out["rows"]
            print(f"{src_name}/{lab}: {json.dumps(out['summary'])}", flush=True)
        if not args.skip_init:
            if args.n_init == 1:
                init_p = load_init_policy(ck_nongm[0], device)
                out = score_agent_on_frames(init_p, frames, device=device, label="init")
                results[src_name]["init"] = out["summary"]
                per_frame[src_name]["init"] = out["rows"]
                print(f"{src_name}/init: {json.dumps(out['summary'])}", flush=True)
            else:
                for i in range(args.n_init):
                    init_p = load_init_policy(ck_nongm[0], device, init_seed=4000 + i)
                    lab = f"init_{i}"
                    out = score_agent_on_frames(init_p, frames, device=device, label=lab)
                    results[src_name][lab] = out["summary"]
                    per_frame[src_name][lab] = out["rows"]
                    print(f"{src_name}/{lab}: {json.dumps(out['summary'])}", flush=True)

    payload = {
        "job_note": "policy-valid dual-source Coin signed-derivative harvest, corrects job 24479's single-policy bank",
        "gm_seeds_used_for_gm_rollouts_warm": "gm_6033",
        "nongm_seeds_used_for_nongm_rollouts_warm": "nongm_7001",
        "n_frames_requested": args.n_frames,
        "n_frames_gm_rollouts": len(frames_gm_src),
        "n_frames_nongm_rollouts": len(frames_nongm_src),
        "results": results,
    }
    (args.out / "coin_conflict_signed_derivative.json").write_text(json.dumps(payload, indent=2))
    (args.out / "coin_conflict_signed_derivative_per_frame.json").write_text(json.dumps(per_frame, indent=2))
    print("wrote", args.out / "coin_conflict_signed_derivative.json", flush=True)
    print("wrote", args.out / "coin_conflict_signed_derivative_per_frame.json", flush=True)


if __name__ == "__main__":
    main()
