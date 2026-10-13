"""Greedy evaluation rollouts: the paper's single-level trap protocol and a vectorized progress check."""

import numpy as np
import torch

from envs import make_progress_env, make_single_level_env


def _raw_rewards(rew, info):
    # VecNormalize rescales rewards; info['env_reward'] is the raw +-10 the outcome classification needs.
    return np.array([i.get("env_reward", r) for i, r in zip(info, rew)], dtype=np.float64)


@torch.no_grad()
def rollout_level(policy, obj1, obj2, level, max_steps=256, device="cuda", sample_seed=None):
    """One episode on one level, stopped at done or max_steps.

    Greedy (argmax) unless sample_seed is given, in which case actions are sampled from a generator seeded per episode.
    """
    gen = None
    if sample_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(sample_seed))
    env = make_single_level_env(obj1, obj2, level)
    try:
        obs = env.reset()
        state = policy.initial_state(1, device)
        start = torch.ones(1, device=device)
        total, steps = 0.0, 0
        for _ in range(max_steps):
            dist, _v, state = policy(torch.as_tensor(obs, device=device), state, start)
            if gen is None:
                act = dist.probs.argmax(-1)
            else:
                act = torch.multinomial(dist.probs, 1, generator=gen)[:, 0]
            obs, rew, done, info = env.step(act.cpu().numpy())
            total += float(_raw_rewards(rew, info)[0])
            steps += 1
            if bool(done[0]):
                break
            start = torch.zeros(1, device=device)
    finally:
        env.close()
    return total, steps


def outcome(total):
    return "goal" if total == 10.0 else "proxy" if total == -10.0 else "neither"


@torch.no_grad()
def progress_eval(policy, pairs, n_per_pair, start_level, seed, max_steps=256, device="cuda"):
    """First-episode greedy outcome of n_per_pair envs per pair on held-out levels. Returns (goal, proxy, n)."""
    goal = proxy = n = 0
    for k, (o1, o2) in enumerate(pairs):
        env = make_progress_env(o1, o2, n_per_pair, start_level, 100_000, rand_seed=seed * 1000 + k)
        obs = env.reset()
        state = policy.initial_state(n_per_pair, device)
        start = torch.ones(n_per_pair, device=device)
        res = np.zeros(n_per_pair)
        finished = np.zeros(n_per_pair, dtype=bool)
        for _ in range(max_steps):
            dist, _v, state = policy(torch.as_tensor(obs, device=device), state, start)
            obs, rew, done, info = env.step(dist.probs.argmax(-1).cpu().numpy())
            raw = _raw_rewards(rew, info)
            newly = done & ~finished
            res[newly] = raw[newly]
            finished |= done
            start = torch.as_tensor(done.astype(np.float32), device=device)
            if finished.all():
                break
        env.close()
        goal += int((res == 10.0).sum())
        proxy += int((res == -10.0).sum())
        n += n_per_pair
    return goal, proxy, n
