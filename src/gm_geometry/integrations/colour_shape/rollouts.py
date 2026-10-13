"""Activation collection during rollouts on foreign train-procgen policies."""

from __future__ import annotations

import numpy as np
import torch


def collect_activations(policy, env, n_steps: int) -> dict:
    """Collect penultimate-layer activations and per-step rewards."""
    activations = []
    rewards = []
    module = policy.embedder.fc

    def _hook(_mod, _inp, out):
        activations.append(out.detach().cpu().numpy())

    handle = module.register_forward_hook(_hook)
    obs = env.reset()
    hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)

    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            act_dist, _value, hidden_t = policy(
                obs_t,
                torch.as_tensor(hidden),
                torch.zeros(env.num_envs),
            )
        hidden = hidden_t.numpy()
        obs, rew, _done, _info = env.step(act_dist.sample().numpy())
        rewards.append(np.asarray(rew))

    handle.remove()
    return {
        "activations": np.concatenate(activations, axis=0),
        "rewards": np.stack(rewards),
    }
