"""Coin harvest and live-rollout harness with the steering hook site passed in as ``push_hook_factory``."""
from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import torch

from gm_geometry.probes.coin_location import (
    RIGHT_ACTIONS,
    find_floor_cell,
    get_info,
    labels_from_info,
    obs_nchw_after_set_coin,
    pixel_diff_feat_cell,
    set_coin,
    unwrap_base_env,
)
from run_coin_fisher_sensitivity import PIXEL_DIFF_THR, load_policy  # noqa: F401  (re-exported for callers)
from validate_langosco_coinrun_200m import classify_done, make_env

PushHookFactory = Callable[[np.ndarray, str], "torch.utils.hooks.RemovableHandle"]


def _argmax_action(policy, obs_nchw: np.ndarray, device: str) -> int:
    with torch.no_grad():
        o = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
        if o.ndim == 3:
            o = o.unsqueeze(0)
        hidden = torch.zeros((1, policy.embedder.output_dim), dtype=torch.float32, device=device)
        dist, _v, _h = policy(o, hidden, torch.zeros(1, device=device))
        return int(dist.probs.argmax(-1).item())


def harvest_with_state_snapshot(
    *, n_frames: int, start_level: int, max_levels: int, warm_policy, device: str,
    min_agent_x: float = 8.0, warm_cap: int = 80, min_dx: int = 1,
    skip_policy_gate: bool = False,
) -> list[dict]:
    """Same selection gates as harvest_decision_windows; additionally snapshots
    get_state() at the decision point so a real rollout can resume from it."""
    rows: list[dict] = []
    level = start_level
    used = 0
    rejects: Counter = Counter()
    while len(rows) < n_frames and used < max_levels:
        env = make_env("coinrun_aisc", num_levels=1, start_level=level)
        obs = env.reset()
        hidden = np.zeros((1, warm_policy.embedder.output_dim), dtype=np.float32)
        lab0 = labels_from_info(get_info(env))
        for _ in range(warm_cap):
            with torch.no_grad():
                dist, _, nh = warm_policy(
                    torch.as_tensor(obs, dtype=torch.float32, device=device),
                    torch.as_tensor(hidden, dtype=torch.float32, device=device),
                    torch.zeros(1, device=device),
                )
                act = dist.probs.argmax(-1).cpu().numpy()
                hidden = nh.cpu().numpy()
            obs, _r, done, _i = env.step(act)
            if done[0]:
                rejects["done_warm"] += 1
                break
            lab0 = labels_from_info(get_info(env))
            if lab0 is not None and float(lab0["agent_x"]) >= min_agent_x:
                break
        else:
            lab0 = labels_from_info(get_info(env))

        used += 1
        if lab0 is None or float(lab0["agent_x"]) < min_agent_x - 0.5:
            rejects["ax_low"] += 1
            env.close()
            level += 1
            continue

        obs_m, _r, done_m, _i = env.step(np.array([4], dtype=np.int64))
        if done_m[0]:
            rejects["done_motion"] += 1
            env.close()
            level += 1
            continue

        placed = None
        for md in (min_dx, min_dx + 1, min_dx + 2, 1, 2, 3):
            xy = find_floor_cell(env, prefer_left_of_agent=False, min_dx=md, vertical_first=True)
            if xy is None:
                continue
            if set_coin(env, xy[0], xy[1], 1) != 0:
                rejects["set_fail"] += 1
                continue
            obs_a = obs_nchw_after_set_coin(env)
            if set_coin(env, xy[0], xy[1], 0) != 0:
                rejects["set_fail"] += 1
                continue
            obs_g = obs_nchw_after_set_coin(env)
            set_coin(env, xy[0], xy[1], 1)
            if obs_a.ndim == 4:
                obs_a = obs_a[0]
            if obs_g.ndim == 4:
                obs_g = obs_g[0]
            if float(np.max(np.abs(obs_a - obs_g))) < PIXEL_DIFF_THR:
                rejects["pixel_gate"] += 1
                continue
            cell = pixel_diff_feat_cell(obs_a, obs_g, thr=PIXEL_DIFF_THR)
            if cell is None:
                rejects["pixel_gate"] += 1
                continue
            a_coin = _argmax_action(warm_policy, obs_a, device)
            # This gate drops states by one warm policy's argmax, so a bank built with it is filtered
            # for that checkpoint; skip_policy_gate=True keeps the pool policy-blind.
            if not skip_policy_gate and a_coin in RIGHT_ACTIONS:
                rejects["a_coin_right"] += 1
                continue

            reloc_xy = None
            for rmd in (min_dx, min_dx + 1, min_dx + 2, 1, 2, 3):
                reloc_xy = find_floor_cell(env, prefer_left_of_agent=True, min_dx=rmd, vertical_first=True)
                if reloc_xy is not None:
                    break
            if reloc_xy is None or set_coin(env, reloc_xy[0], reloc_xy[1], 1) != 0:
                rejects["reloc_fail"] += 1
                continue
            obs_reloc = obs_nchw_after_set_coin(env)
            if obs_reloc.ndim == 4:
                obs_reloc = obs_reloc[0]

            set_coin(env, reloc_xy[0], reloc_xy[1], 0)
            set_coin(env, xy[0], xy[1], 1)
            state_reloc = unwrap_base_env(env).get_state()

            placed = {
                "level": int(level),
                "obs_a": obs_a.astype(np.float32),
                "obs_reloc": obs_reloc.astype(np.float32),
                "reloc_xy": (int(reloc_xy[0]), int(reloc_xy[1])),
                "xy_a": (int(xy[0]), int(xy[1])),
                "a_coin": int(a_coin),
                "state_reloc": state_reloc,
            }
            break
        env.close()
        if placed is None:
            rejects["no_xy"] += 1
        else:
            rows.append(placed)
            if len(rows) % 8 == 0:
                print(f"  harvested {len(rows)}/{n_frames} rejects={dict(rejects)}", flush=True)
        level += 1
    print(f"harvest+snapshot done n={len(rows)} used={used} rejects={dict(rejects)}", flush=True)
    return rows


def validate_snapshot_roundtrip(frames: list[dict]) -> dict:
    max_abs_diffs = []
    for fr in frames:
        env = make_env("coinrun_aisc", num_levels=1, start_level=0)
        env.reset()
        unwrap_base_env(env).set_state(fr["state_reloc"])
        obs_check = obs_nchw_after_set_coin(env)
        if obs_check.ndim == 4:
            obs_check = obs_check[0]
        max_abs_diffs.append(float(np.max(np.abs(obs_check - fr["obs_a"]))))
        env.close()
    return {
        "n": len(frames),
        "max_abs_diff_over_all": float(max(max_abs_diffs)) if max_abs_diffs else float("nan"),
        "all_exact": all(d < 1e-6 for d in max_abs_diffs),
    }


def rollout_from_state(
    policy, state_bytes, *, delta_fn, device: str, push_hook_factory: PushHookFactory,
    max_steps: int = 150, short_horizon: int = 30, probe_only_zero_push: bool = False,
    refresh_stride: int = 1,
) -> dict:
    """Resume from a saved snapshot, optionally pushing a delta recomputed from the live state.

    If ``delta_fn`` is given it is called every ``refresh_stride`` steps as delta_fn(env, policy, device) ->
    (delta, ok) and returns the full, already beta-scaled delta; ``push_hook_factory(delta_vec, device)``
    decides where it is injected."""
    env = make_env("coinrun_aisc", num_levels=1, start_level=0)
    env.reset()
    unwrap_base_env(env).set_state(state_bytes)
    obs = obs_nchw_after_set_coin(env)
    if obs.ndim == 3:
        obs = obs[None]
    hidden = np.zeros((1, policy.embedder.output_dim), dtype=np.float32)
    total_rew = 0.0
    outcome = "timed_out"
    last_info: dict = {}
    lab0 = labels_from_info(get_info(env))
    agent_x0 = float(lab0["agent_x"]) if lab0 is not None else float("nan")
    agent_x_short = agent_x0
    n_refresh_fail = 0
    handle = None
    t = -1
    try:
        for t in range(max_steps):
            if delta_fn is not None and (t % refresh_stride == 0):
                if handle is not None:
                    handle.remove()
                    handle = None
                delta_vec, ok = delta_fn(env, policy, device)
                if not ok:
                    n_refresh_fail += 1
                if delta_vec is not None and not probe_only_zero_push:
                    handle = push_hook_factory(delta_vec, device)
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                h = torch.as_tensor(hidden, dtype=torch.float32, device=device)
                dist, _v, nh = policy(x, h, torch.zeros(1, device=device))
                act = dist.probs.argmax(-1).cpu().numpy()
                hidden = nh.cpu().numpy()
            obs, rew, done, info = env.step(act)
            total_rew += float(np.asarray(rew)[0])
            last_info = info[0]
            if t + 1 == short_horizon or done[0]:
                lab_t = labels_from_info(get_info(env))
                if lab_t is not None:
                    agent_x_short = float(lab_t["agent_x"])
            if last_info.get("invisible_coin_collected") and not done[0]:
                outcome = "goal_misgen"
                break
            if done[0]:
                outcome = classify_done(last_info, max_steps)
                break
        else:
            outcome = "timed_out"
    finally:
        if handle is not None:
            handle.remove()
        env.close()
    return {
        "reward": total_rew, "steps": int(t) + 1, "outcome": outcome,
        "coin_reached": outcome == "success",
        "agent_dx_short": agent_x_short - agent_x0,
        "n_refresh_fail": n_refresh_fail,
    }


def validate_probe_noninterference(
    policy, state_bytes, device: str, max_steps: int, *, push_hook_factory: PushHookFactory,
    zero_delta_dim: int,
) -> dict:
    """Check that calling ``delta_fn`` every step with zero push leaves the rollout outcome unchanged."""
    out_clean = rollout_from_state(
        policy, state_bytes, delta_fn=None, device=device,
        push_hook_factory=push_hook_factory, max_steps=max_steps,
    )
    out_probed = rollout_from_state(
        policy, state_bytes,
        delta_fn=lambda env, pol, dev: (np.zeros(zero_delta_dim, dtype=np.float64), True),
        device=device, push_hook_factory=push_hook_factory, max_steps=max_steps,
        probe_only_zero_push=True,
    )
    return {
        "outcome_match": out_clean["outcome"] == out_probed["outcome"],
        "steps_match": out_clean["steps"] == out_probed["steps"],
        "clean": {"outcome": out_clean["outcome"], "steps": out_clean["steps"]},
        "probed": {"outcome": out_probed["outcome"], "steps": out_probed["steps"]},
    }
