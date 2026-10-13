#!/usr/bin/env python3
"""Attribute Maze counterfactual competition steering hooked at the post-ReLU h_fc tap (the policy.embedder output)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tnf

ROOT = Path(__file__).resolve().parents[2]
_CS_ROOT = ROOT / "external" / "colour-shape-goal-misgeneralization"
sys.path.insert(0, str(_CS_ROOT / "procgenAISC"))
sys.path.insert(0, str(_CS_ROOT / "train-procgen"))

import run_ras_conflict_signed_derivative as base  # noqa: E402
from gm_geometry.probes.h_fc_steering import (  # noqa: E402
    extract_h_fc, make_h_fc_push_hook, verify_h_fc_hook,
)

_DEBUG_MISMATCH_COUNT = 0


def assert_h_fc_hook_correct(policy, device: str) -> None:
    """Non-negotiable startup gate: refuse to run if the hook doesn't reproduce
    the unhooked h_fc exactly, or if h_fc isn't actually all-nonnegative
    (which would mean we're not really past the ReLU)."""
    dummy = torch.zeros(1, 3, 64, 64, device=device)
    report = verify_h_fc_hook(policy, dummy)
    print(f"[h_fc hook verification] {report}", flush=True)
    if not report["matches"]:
        raise SystemExit(f"REFUSING TO RUN: h_fc hook does not reproduce unhooked forward pass: {report}")
    if not report["all_nonneg_confirms_post_relu"]:
        raise SystemExit(f"REFUSING TO RUN: h_fc has negative entries, not actually post-ReLU: {report}")


def fc_logit_jacobian_and_act_hfc(policy, obs_nchw: np.ndarray, device: str):
    """d(logit_a)/d(h_fc), probs, h_fc. Shapes [A,D], [A], [D]. D=256.

    J = fc_policy.weight is constant because h_fc feeds fc_policy through a pure linear map (non-recurrent policy).
    """
    x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
    m = torch.zeros(1, device=device)
    with torch.no_grad():
        dist, _value, _nh = policy(x, h0, m)
        h = extract_h_fc(policy, x)
        logits = dist.logits
        probs = tnf.softmax(logits, dim=-1)[0].cpu().numpy()
    j = policy.fc_policy.weight.detach().cpu().numpy().astype(np.float64)  # [A, D], constant
    return j, probs.astype(np.float64), h.detach()[0].cpu().numpy().astype(np.float64)


def compute_d_comp0_hfc(policy, res, distribution_mode, use_backgrounds, device):
    """h_fc analog of compute_d_comp0."""
    level = res["level"]
    obs_base = res["obs0"]
    env_shape = base.make_env(base.SHAPE_EDIT_OBJ, base.OBJ_GEM, level, rand_seed=0,
                               distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    env_colour = base.make_env(base.OBJ_LINE, base.COLOUR_EDIT_OBJ, level, rand_seed=0,
                                distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    try:
        obs_shape = env_shape.reset()
        obs_colour = env_colour.reset()
        sb_shape = base.state_bytes(env_shape)
        sb_colour = base.state_bytes(env_colour)
    finally:
        env_shape.close()
        env_colour.close()
    if sb_shape != res["state_bytes"] or sb_colour != res["state_bytes"]:
        global _DEBUG_MISMATCH_COUNT
        if _DEBUG_MISMATCH_COUNT < 5:
            _DEBUG_MISMATCH_COUNT += 1
            print(f"[DEBUG mismatch level={level}] len(res_sb)={len(res['state_bytes'])} "
                  f"len(sb_shape)={len(sb_shape)} len(sb_colour)={len(sb_colour)} "
                  f"sb_shape==res={sb_shape == res['state_bytes']} sb_colour==res={sb_colour == res['state_bytes']} "
                  f"res_sb[:24]={res['state_bytes'][:24]!r} sb_shape[:24]={sb_shape[:24]!r}", flush=True)
        return None
    del obs_base
    _, _, h_shape = fc_logit_jacobian_and_act_hfc(policy, base.to_nchw(obs_shape), device)
    _, _, h_colour = fc_logit_jacobian_and_act_hfc(policy, base.to_nchw(obs_colour), device)
    return (h_colour - h_shape).astype(np.float64)


def compute_d_comp0_and_fisher_hfc(policy, res, distribution_mode, use_backgrounds, device):
    """d_comp at the spawn state plus the policy Fisher, both at h_fc."""
    level = res["level"]
    env_shape = base.make_env(base.SHAPE_EDIT_OBJ, base.OBJ_GEM, level, rand_seed=0,
                               distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    env_colour = base.make_env(base.OBJ_LINE, base.COLOUR_EDIT_OBJ, level, rand_seed=0,
                                distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    try:
        obs_shape = env_shape.reset()
        obs_colour = env_colour.reset()
        sb_shape = base.state_bytes(env_shape)
        sb_colour = base.state_bytes(env_colour)
    finally:
        env_shape.close()
        env_colour.close()
    if sb_shape != res["state_bytes"] or sb_colour != res["state_bytes"]:
        global _DEBUG_MISMATCH_COUNT
        if _DEBUG_MISMATCH_COUNT < 5:
            _DEBUG_MISMATCH_COUNT += 1
            print(f"[DEBUG mismatch(fisher) level={level}] len(res_sb)={len(res['state_bytes'])} "
                  f"len(sb_shape)={len(sb_shape)} len(sb_colour)={len(sb_colour)} "
                  f"sb_shape==res={sb_shape == res['state_bytes']} sb_colour==res={sb_colour == res['state_bytes']} "
                  f"res_sb[:24]={res['state_bytes'][:24]!r} sb_shape[:24]={sb_shape[:24]!r}", flush=True)
        return None, None
    j_base, probs_base, _h_base = fc_logit_jacobian_and_act_hfc(policy, base.to_nchw(res["obs0"]), device)
    _, _, h_shape = fc_logit_jacobian_and_act_hfc(policy, base.to_nchw(obs_shape), device)
    _, _, h_colour = fc_logit_jacobian_and_act_hfc(policy, base.to_nchw(obs_colour), device)
    f_base = base.fisher_from_logit_jacobian(j_base, probs_base)
    d_comp0 = (h_colour - h_shape).astype(np.float64)
    return d_comp0, f_base


import run_ras_counterfactual_competition_steering as cc_steer  # noqa: E402


def rollout_with_dcomp_hfc(policy, level, distribution_mode, use_backgrounds, device,
                            beta, sign, max_steps, replay_stride, expect_state_bytes):
    """h_fc analog of cc_steer.rollout_with_dcomp; the outcome is read from reward, not action groups."""
    def _shape_env_factory(lvl):
        return base.make_env(base.SHAPE_EDIT_OBJ, base.OBJ_GEM, lvl, rand_seed=0,
                              distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)

    def _colour_env_factory(lvl):
        return base.make_env(base.OBJ_LINE, base.COLOUR_EDIT_OBJ, lvl, rand_seed=0,
                              distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)

    env = base.make_env(base.OBJ_LINE, base.OBJ_GEM, level, rand_seed=0,
                         distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    obs = env.reset()
    sb = base.state_bytes(env)
    if sb != expect_state_bytes:
        env.close()
        raise RuntimeError(f"level {level}: base replay state_bytes mismatch")

    action_history: list[int] = []
    d_comp_cached = None
    outcome, n_steps = "timeout", max_steps
    handle = None
    try:
        for t in range(max_steps):
            x = base.to_nchw(obs)
            if handle is not None:
                handle.remove()
                handle = None

            need_refresh = (d_comp_cached is None) or (t % replay_stride == 0)
            if need_refresh:
                obs_shape_t, done_shape = cc_steer.replay_h_at_step(
                    _shape_env_factory, level, distribution_mode, use_backgrounds, action_history, device)
                obs_colour_t, done_colour = cc_steer.replay_h_at_step(
                    _colour_env_factory, level, distribution_mode, use_backgrounds, action_history, device)
                if not (done_shape or done_colour):
                    _, _, h_shape_t = fc_logit_jacobian_and_act_hfc(policy, base.to_nchw(obs_shape_t), device)
                    _, _, h_colour_t = fc_logit_jacobian_and_act_hfc(policy, base.to_nchw(obs_colour_t), device)
                    d_comp_cached = (h_colour_t - h_shape_t).astype(np.float64)

            if d_comp_cached is not None:
                delta_vec = sign * beta * d_comp_cached
                delta_t = torch.as_tensor(delta_vec, dtype=torch.float32, device=device)

                def _delta_fn(_out, d=delta_t):
                    return d
                handle = policy.embedder.register_forward_hook(make_h_fc_push_hook(_delta_fn))

            x_t = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
            m = torch.zeros(1, device=device)
            with torch.no_grad():
                dist, _v, _nh = policy(x_t, h0, m)
                a = int(dist.probs.argmax(-1).item())

            action_history.append(a)
            obs, rew, done, _info = env.step(np.array([a]))
            r = float(np.asarray(rew).reshape(-1)[0])
            d = bool(np.asarray(done).reshape(-1)[0])
            if d:
                outcome = "line" if r > 5.0 else ("gem" if r < -5.0 else "other_terminal")
                n_steps = t + 1
                break
    finally:
        if handle is not None:
            handle.remove()
        env.close()
    return outcome, n_steps


def rollout_with_fixed_push_hfc(policy, level, distribution_mode, use_backgrounds, device,
                                 delta_vec, max_steps, expect_state_bytes):
    """h_fc analog of cc_steer.rollout_with_fixed_push -- a FIXED delta held constant
    throughout, hooked on policy.embedder (whole module), not .fc."""
    env = base.make_env(base.OBJ_LINE, base.OBJ_GEM, level, rand_seed=0,
                         distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    obs = env.reset()
    if base.state_bytes(env) != expect_state_bytes:
        env.close()
        raise RuntimeError(f"level {level}: base replay state_bytes mismatch")
    delta_t = torch.as_tensor(delta_vec, dtype=torch.float32, device=device)

    def _delta_fn(_out):
        return delta_t
    handle = policy.embedder.register_forward_hook(make_h_fc_push_hook(_delta_fn))
    outcome, n_steps = "timeout", max_steps
    try:
        for t in range(max_steps):
            x = torch.as_tensor(base.to_nchw(obs), dtype=torch.float32, device=device).unsqueeze(0)
            h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
            m = torch.zeros(1, device=device)
            with torch.no_grad():
                dist, _v, _nh = policy(x, h0, m)
                a = int(dist.probs.argmax(-1).item())
            obs, rew, done, _info = env.step(np.array([a]))
            r = float(np.asarray(rew).reshape(-1)[0])
            d = bool(np.asarray(done).reshape(-1)[0])
            if d:
                outcome = "line" if r > 5.0 else ("gem" if r < -5.0 else "other_terminal")
                n_steps = t + 1
                break
    finally:
        handle.remove()
        env.close()
    return outcome, n_steps


def determinism_check_hfc(policy, level, distribution_mode, use_backgrounds, device,
                           beta, sign, max_steps, replay_stride, expect_state_bytes):
    o1, n1 = rollout_with_dcomp_hfc(policy, level, distribution_mode, use_backgrounds, device,
                                     beta, sign, max_steps, replay_stride, expect_state_bytes)
    o2, n2 = rollout_with_dcomp_hfc(policy, level, distribution_mode, use_backgrounds, device,
                                     beta, sign, max_steps, replay_stride, expect_state_bytes)
    return (o1 == o2 and n1 == n2), (o1, n1), (o2, n2)
