#!/usr/bin/env python3
"""Attribute Maze counterfactual competition steering along d_comp(s) = Delta_colour(s) - Delta_shape(s): sign, causal ablation, rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_CS_ROOT = ROOT / "external" / "colour-shape-goal-misgeneralization"
sys.path.insert(0, str(_CS_ROOT / "procgenAISC"))
sys.path.insert(0, str(_CS_ROOT / "train-procgen"))

import run_ras_conflict_signed_derivative as base  # noqa: E402
from gm_geometry.integrations.colour_shape.policy import (  # noqa: E402
    load_impala_policy,
    load_recurrent_flag,
)


def pushed_probs(policy, obs_nchw, device, delta_vec):
    delta_t = torch.as_tensor(delta_vec, dtype=torch.float32, device=device)

    def _hook(_m, _i, out):
        return out + delta_t

    handle = policy.embedder.fc.register_forward_hook(_hook)
    try:
        x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device).unsqueeze(0)
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        m = torch.zeros(1, device=device)
        with torch.no_grad():
            dist, _v, _nh = policy(x, h0, m)
            return dist.probs[0].cpu().numpy()
    finally:
        handle.remove()


def step1_and_step2(policy, matched, groups, distribution_mode, use_backgrounds, device):
    """Step 1 (sign story, from already-linear scalars) + Step 2 (3-way causal
    ablation via real hooked forward passes on the spawn conflict state)."""
    rows = []
    for res in matched:
        level = res["level"]
        ids_line = groups[res["first_group_line"]]
        ids_gem = groups[res["first_group_gem"]]
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
            continue

        j_base, probs_base, h_base = base.fc_logit_jacobian_and_act(policy, base.to_nchw(obs_base), device)
        _, _, h_shape = base.fc_logit_jacobian_and_act(policy, base.to_nchw(obs_shape), device)
        _, _, h_colour = base.fc_logit_jacobian_and_act(policy, base.to_nchw(obs_colour), device)

        d_s = (h_shape - h_base).astype(np.float64)
        d_c = (h_colour - h_base).astype(np.float64)
        d_comp = d_c - d_s

        grad_r = base.dP_group_dh(j_base, probs_base, ids_line) - base.dP_group_dh(j_base, probs_base, ids_gem)
        g_shape = float(np.dot(grad_r, d_s))       # analytic partial_{Delta_shape} C
        g_colour = float(np.dot(grad_r, d_c))      # analytic partial_{Delta_colour} C
        g_comp_analytic = float(np.dot(grad_r, d_comp))

        # Step 2: hooked-forward-pass causal ablation, not just the analytic dot product
        r_base = float(probs_base[ids_line].sum() - probs_base[ids_gem].sum())

        def r_of(probs):
            return float(probs[ids_line].sum() - probs[ids_gem].sum())

        d_shape_push = -d_s
        d_colour_push = d_c
        d_joint_push = d_comp

        probs_push_shape = pushed_probs(policy, base.to_nchw(obs_base), device, d_shape_push)
        probs_push_colour = pushed_probs(policy, base.to_nchw(obs_base), device, d_colour_push)
        probs_push_joint = pushed_probs(policy, base.to_nchw(obs_base), device, d_joint_push)

        dC_shape_real = r_of(probs_push_shape) - r_base
        dC_colour_real = r_of(probs_push_colour) - r_base
        dC_joint_real = r_of(probs_push_joint) - r_base

        # matched-norm random-direction control (state-wise, same convention as elsewhere)
        rng = np.random.default_rng(abs(hash((level, "rand"))) % (2**32))
        rand_dir = rng.normal(size=d_comp.shape)
        rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-18) * np.linalg.norm(d_comp)
        probs_push_rand = pushed_probs(policy, base.to_nchw(obs_base), device, rand_dir)
        dC_rand_real = r_of(probs_push_rand) - r_base

        rows.append({
            "level": int(level),
            "g_shape_analytic": g_shape, "g_colour_analytic": g_colour,
            "g_comp_analytic": g_comp_analytic,
            "g_comp_additivity_check": g_comp_analytic - (g_colour - g_shape),  # should be ~0 exactly
            "dC_shape_real": dC_shape_real, "dC_colour_real": dC_colour_real,
            "dC_joint_real": dC_joint_real, "dC_rand_real": dC_rand_real,
            "joint_additivity_real_check": dC_joint_real - (dC_shape_real + dC_colour_real),  # NOT exact (nonlinear real fwd pass)
            "r_base": r_base,
        })
    return rows


def compute_d_comp0(policy, res, distribution_mode, use_backgrounds, device):
    """Recompute d_comp at the spawn state; its norm sets the matched-norm random control's per-beta dose."""
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
        return None
    _, _, h_base = base.fc_logit_jacobian_and_act(policy, base.to_nchw(obs_base), device)
    _, _, h_shape = base.fc_logit_jacobian_and_act(policy, base.to_nchw(obs_shape), device)
    _, _, h_colour = base.fc_logit_jacobian_and_act(policy, base.to_nchw(obs_colour), device)
    del h_base
    return (h_colour - h_shape).astype(np.float64)


def replay_h_at_step(env_factory, level, distribution_mode, use_backgrounds, action_seq, device):
    """Deterministic replay: fresh env at (level, rand_seed=0), take action_seq in
    order, return the fc-layer h AFTER those actions (i.e. h of the observation
    the live env would show at step len(action_seq)), plus whether it terminated
    early (in which case there is no valid counterfactual h to use)."""
    env = env_factory(level)
    obs = env.reset()
    done_early = False
    try:
        for a in action_seq:
            obs, _rew, done, _info = env.step(np.array([a]))
            if bool(np.asarray(done).reshape(-1)[0]):
                done_early = True
                break
    finally:
        env.close()
    return obs, done_early


def rollout_with_dcomp(policy, level, distribution_mode, use_backgrounds, device,
                        ids_line, ids_gem, beta, sign, max_steps, replay_stride,
                        expect_state_bytes):
    """Persistent rollout with d_comp(s_t) reconstructed by deterministic replay.

    Edited envs cannot be forked from a live snapshot via set_state, so fresh shape-edit and colour-edit envs replay the
    live action sequence, which reproduces the agent position because movement does not depend on sprite identity.
    Replay costs O(T^2) actions per rollout; --replay-stride refreshes d_comp only every k steps.
    """
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
                obs_shape_t, done_shape = replay_h_at_step(
                    _shape_env_factory, level, distribution_mode, use_backgrounds, action_history, device)
                obs_colour_t, done_colour = replay_h_at_step(
                    _colour_env_factory, level, distribution_mode, use_backgrounds, action_history, device)
                if done_shape or done_colour:
                    # counterfactual twin already terminated (e.g. reached its own
                    # object early) -- cannot construct a valid d_comp here, hold
                    # the last cached value (or push nothing if none yet).
                    pass
                else:
                    _, _, h_shape_t = base.fc_logit_jacobian_and_act(policy, base.to_nchw(obs_shape_t), device)
                    _, _, h_colour_t = base.fc_logit_jacobian_and_act(policy, base.to_nchw(obs_colour_t), device)
                    d_comp_cached = (h_colour_t - h_shape_t).astype(np.float64)

            if d_comp_cached is not None:
                delta_vec = sign * beta * d_comp_cached
                delta_t = torch.as_tensor(delta_vec, dtype=torch.float32, device=device)

                def _hook(_m, _i, out, d=delta_t):
                    return out + d
                handle = policy.embedder.fc.register_forward_hook(_hook)

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


def rollout_with_fixed_push(policy, level, distribution_mode, use_backgrounds, device,
                             delta_vec, max_steps, expect_state_bytes):
    """Persistent-hook rollout with a fixed delta, for the matched-norm random-direction control (dose matched to
    d_comp's t=0 norm)."""
    env = base.make_env(base.OBJ_LINE, base.OBJ_GEM, level, rand_seed=0,
                         distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    obs = env.reset()
    if base.state_bytes(env) != expect_state_bytes:
        env.close()
        raise RuntimeError(f"level {level}: base replay state_bytes mismatch")
    delta_t = torch.as_tensor(delta_vec, dtype=torch.float32, device=device)

    def _hook(_m, _i, out):
        return out + delta_t
    handle = policy.embedder.fc.register_forward_hook(_hook)
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


def determinism_check(policy, level, distribution_mode, use_backgrounds, device,
                       ids_line, ids_gem, beta, sign, max_steps, replay_stride, expect_state_bytes):
    o1, n1 = rollout_with_dcomp(policy, level, distribution_mode, use_backgrounds, device,
                                 ids_line, ids_gem, beta, sign, max_steps, replay_stride, expect_state_bytes)
    o2, n2 = rollout_with_dcomp(policy, level, distribution_mode, use_backgrounds, device,
                                 ids_line, ids_gem, beta, sign, max_steps, replay_stride, expect_state_bytes)
    return (o1 == o2 and n1 == n2), (o1, n1), (o2, n2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, action="append", required=True, help="seed:path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-level", type=int, default=2000)
    ap.add_argument("--n-levels-to-try", type=int, default=800)
    ap.add_argument("--n-target", type=int, default=24)
    ap.add_argument("--distribution-mode", type=str, default="easy")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--betas", type=str, default="0.125,0.25,0.5,1.0")
    ap.add_argument("--n-rollout-states", type=int, default=24,
                     help="how many of the n-target conflict states to run Step 3 on")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--replay-stride", type=int, default=1,
                     help="refresh d_comp every k steps (1 = every step); >1 trades fidelity for speed")
    ap.add_argument("--skip-rollout", action="store_true", help="Steps 1-2 only")
    ap.add_argument("--n-random-dirs", type=int, default=3,
                     help="independent matched-norm random directions per state/beta")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpts = {}
    for entry in args.ckpt:
        seed_str, path_str = entry.split(":", 1)
        ckpts[int(seed_str)] = Path(path_str)
    betas = [float(b) for b in args.betas.split(",")]

    print(f"[setup] seeds={list(ckpts)} betas={betas} n_rollout_states={args.n_rollout_states} "
          f"replay_stride={args.replay_stride} max_steps={args.max_steps}", flush=True)

    probe_env = base.make_env(base.OBJ_LINE, base.OBJ_GEM, 0, 0, args.distribution_mode, False)
    n_actions = int(probe_env.action_space.n)
    probe_env.close()

    groups, matched = base.find_conflict_levels(
        start_level=args.start_level, n_levels_to_try=args.n_levels_to_try, n_target=args.n_target,
        distribution_mode=args.distribution_mode, use_backgrounds=False, n_actions=n_actions,
    )
    if len(matched) < 5:
        raise SystemExit(f"too few conflict levels matched ({len(matched)})")
    print(f"[harvest] matched={len(matched)} conflict levels", flush=True)

    recurrent = load_recurrent_flag()
    results = {}
    for seed, ckpt in ckpts.items():
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint for seed {seed}: {ckpt}")
        print(f"[seed {seed}] loading policy", flush=True)
        policy = load_impala_policy(ckpt, in_channels=3, action_size=n_actions, recurrent=recurrent, device=args.device)

        print(f"[seed {seed}] Step 1+2 (spawn-state sign story + 3-way causal ablation)", flush=True)
        step12_rows = step1_and_step2(policy, matched, groups, args.distribution_mode, False, args.device)
        n12 = len(step12_rows)
        frac_dcomp_pos = float(np.mean([r["g_comp_analytic"] > 0 for r in step12_rows])) if n12 else float("nan")
        max_add_err = float(np.max(np.abs([r["g_comp_additivity_check"] for r in step12_rows]))) if n12 else float("nan")
        max_joint_nonlin = float(np.max(np.abs([r["joint_additivity_real_check"] for r in step12_rows]))) if n12 else float("nan")
        print(f"[seed {seed}] Step1: frac(analytic d_comp > 0)={frac_dcomp_pos:.3f} "
              f"max|analytic additivity err|={max_add_err:.2e} "
              f"max|real joint-vs-sum nonlinearity|={max_joint_nonlin:.4f} (n={n12})", flush=True)

        seed_out = {"step1_2_rows": step12_rows}

        if not args.skip_rollout:
            print(f"[seed {seed}] determinism check before trusting any rollout number", flush=True)
            ids_line = groups[matched[0]["first_group_line"]]
            ids_gem = groups[matched[0]["first_group_gem"]]
            ok, r1, r2 = determinism_check(
                policy, matched[0]["level"], args.distribution_mode, False, args.device,
                ids_line, ids_gem, beta=betas[0], sign=1.0, max_steps=args.max_steps,
                replay_stride=args.replay_stride, expect_state_bytes=matched[0]["state_bytes"],
            )
            print(f"[seed {seed}] determinism check: ok={ok} run1={r1} run2={r2}", flush=True)
            if not ok:
                raise SystemExit(f"seed {seed}: determinism check FAILED, refusing to trust rollout numbers")

            rollout_rows = []
            n_states = min(args.n_rollout_states, len(matched))
            n_rand_dirs = max(1, args.n_random_dirs)
            for si, res in enumerate(matched[:n_states]):
                level = res["level"]
                ids_line = groups[res["first_group_line"]]
                ids_gem = groups[res["first_group_gem"]]
                d_comp0 = compute_d_comp0(policy, res, args.distribution_mode, False, args.device)
                norm0 = float(np.linalg.norm(d_comp0)) if d_comp0 is not None else float("nan")
                rand_rng = np.random.default_rng(abs(hash((seed, level, "rand"))) % (2**32))
                for beta in betas:
                    o_base, n_base = rollout_with_dcomp(
                        policy, level, args.distribution_mode, False, args.device,
                        ids_line, ids_gem, beta=0.0, sign=1.0, max_steps=args.max_steps,
                        replay_stride=10**9, expect_state_bytes=res["state_bytes"])
                    o_corr, n_corr = rollout_with_dcomp(
                        policy, level, args.distribution_mode, False, args.device,
                        ids_line, ids_gem, beta=beta, sign=1.0, max_steps=args.max_steps,
                        replay_stride=args.replay_stride, expect_state_bytes=res["state_bytes"])
                    o_anti, n_anti = rollout_with_dcomp(
                        policy, level, args.distribution_mode, False, args.device,
                        ids_line, ids_gem, beta=beta, sign=-1.0, max_steps=args.max_steps,
                        replay_stride=args.replay_stride, expect_state_bytes=res["state_bytes"])
                    rand_outcomes = []
                    if d_comp0 is not None:
                        for _ in range(n_rand_dirs):
                            rdir = rand_rng.normal(size=d_comp0.shape)
                            rdir = rdir / (np.linalg.norm(rdir) + 1e-18) * (beta * norm0)
                            o_rand, _n_rand = rollout_with_fixed_push(
                                policy, level, args.distribution_mode, False, args.device,
                                rdir, max_steps=args.max_steps, expect_state_bytes=res["state_bytes"])
                            rand_outcomes.append(o_rand)
                    rollout_rows.append({
                        "level": int(level), "beta": beta,
                        "baseline_outcome": o_base, "corrective_outcome": o_corr, "anticorrective_outcome": o_anti,
                        "baseline_steps": n_base, "corrective_steps": n_corr, "anticorrective_steps": n_anti,
                        "random_outcomes": rand_outcomes,
                    })
                print(f"[seed {seed}] rollout progress {si + 1}/{n_states} states done", flush=True)
            seed_out["rollout_rows"] = rollout_rows

            def _dist(outs):
                n = len(outs)
                if n == 0:
                    return {}
                return {k: float(np.mean([o == k for o in outs])) for k in ("line", "gem", "other_terminal", "timeout")}

            for beta in betas:
                sub = [r for r in rollout_rows if r["beta"] == beta]
                base_outs = [r["baseline_outcome"] for r in sub]
                corr_outs = [r["corrective_outcome"] for r in sub]
                anti_outs = [r["anticorrective_outcome"] for r in sub]
                rand_outs = [o for r in sub for o in r["random_outcomes"]]
                print(f"[seed {seed}] beta={beta}: baseline(line/gem/other/timeout)={_dist(base_outs)} "
                      f"corrective={_dist(corr_outs)} anticorrective={_dist(anti_outs)} "
                      f"random={_dist(rand_outs)} (n_states={len(sub)}, n_rand={len(rand_outs)})", flush=True)

        results[f"seed_{seed}"] = seed_out
        (args.out / "counterfactual_competition_steering.json").write_text(json.dumps(results, indent=2))
        print(f"[seed {seed}] done, wrote partial results", flush=True)

    print("all seeds done, wrote", args.out / "counterfactual_competition_steering.json", flush=True)


if __name__ == "__main__":
    main()
