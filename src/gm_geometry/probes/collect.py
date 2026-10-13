"""Collect activations and labels for state-based probes."""

from __future__ import annotations

import numpy as np
import torch

from gm_geometry.integrations.colour_shape.envs import make_duel_env
from gm_geometry.probes.labels import concept_scores_batch, ground_truth_presence


def collect_probe_config(
    policy,
    num_envs: int,
    num_levels: int,
    start_level: int,
    n_steps: int,
    num_threads: int,
    obj1: str,
    obj2: str,
    *,
    use_backgrounds: bool = False,
    episode_id_base: int = 0,
) -> dict:
    env = make_duel_env(
        num_envs, num_levels, num_threads, obj1, obj2,
        start_level=start_level, use_backgrounds=use_backgrounds,
    )
    activations, observations, episode_ids = [], [], []
    module = policy.embedder.fc
    device = next(policy.parameters()).device

    def _hook(_m, _i, out):
        activations.append(out.detach().cpu().numpy())

    handle = module.register_forward_hook(_hook)
    obs = env.reset()
    hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)
    # Per-env episode counter so split-half and null tests never split within an episode.
    ep_count = np.zeros(env.num_envs, dtype=np.int32)
    for _ in range(n_steps):
        observations.append(obs.copy())
        episode_ids.append(
            episode_id_base + np.arange(env.num_envs, dtype=np.int64) * 100_000 + ep_count.astype(np.int64)
        )
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=device)
        masks = torch.zeros(env.num_envs, device=device)
        with torch.no_grad():
            act_dist, _v, hidden_t = policy(obs_t, hidden_t, masks)
        hidden = hidden_t.detach().cpu().numpy()
        obs, _r, done, _i = env.step(act_dist.sample().detach().cpu().numpy())
        done = np.asarray(done, dtype=np.bool_)
        ep_count = ep_count + done.astype(np.int32)
    handle.remove()

    act = np.concatenate(activations, axis=0)
    obs_all = np.concatenate(observations, axis=0)
    ep_all = np.concatenate(episode_ids, axis=0)
    gt = ground_truth_presence(obj1, obj2)
    n = len(act)
    labels = {k: np.full(n, v, dtype=np.float32) for k, v in gt.items()}
    pixel = concept_scores_batch(obs_all)
    return {
        "activations": act,
        "labels": labels,
        "pixel": pixel,
        "n": n,
        "observations": obs_all,
        "episode_ids": ep_all,
    }


def activations_from_observations(
    policy,
    observations: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Forward fixed observations through ``policy.embedder.fc`` (no env rollouts)."""
    obs = np.asarray(observations, dtype=np.float32)
    acts = []
    module = policy.embedder.fc
    device = next(policy.parameters()).device
    buf: list[np.ndarray] = []

    def _hook(_m, _i, out):
        buf.append(out.detach().cpu().numpy())

    handle = module.register_forward_hook(_hook)
    hidden_dim = policy.embedder.output_dim
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            batch = obs[start : start + batch_size]
            b = len(batch)
            obs_t = torch.as_tensor(batch, dtype=torch.float32, device=device)
            hidden = torch.zeros(b, hidden_dim, device=device)
            masks = torch.zeros(b, device=device)
            policy(obs_t, hidden, masks)
            acts.append(buf.pop())
    handle.remove()
    return np.concatenate(acts, axis=0)


def collect_probe_grid(
    policy,
    configs: list[tuple[str, str]],
    num_envs: int,
    num_levels: int,
    start_level: int,
    n_steps: int,
    num_threads: int,
    *,
    use_backgrounds: bool = False,
    return_observations: bool = False,
    label_keys: list[str] | None = None,
) -> dict:
    acts, groups, episode_ids = [], [], []
    obs_chunks: list[np.ndarray] = []
    keys = list(label_keys) if label_keys is not None else [
        "line_present", "gem_present", "yellow_present",
    ]
    labels: dict[str, list] = {k: [] for k in keys}
    pixel = {k: [] for k in ["yellow", "line_over_gem", "gem"]}
    for gi, (obj1, obj2) in enumerate(configs):
        batch = collect_probe_config(
            policy, num_envs, num_levels, start_level, n_steps, num_threads,
            obj1, obj2, use_backgrounds=use_backgrounds,
            episode_id_base=gi * 10_000_000,
        )
        acts.append(batch["activations"])
        groups.append(np.full(batch["n"], gi, dtype=np.int32))
        episode_ids.append(batch["episode_ids"])
        if return_observations:
            obs_chunks.append(batch["observations"])
        for k in keys:
            if k not in batch["labels"]:
                raise KeyError(f"label {k!r} missing for config {(obj1, obj2)}")
            labels[k].append(batch["labels"][k])
        for k in pixel:
            pixel[k].append(batch["pixel"][k])
    result = {
        "activations": np.concatenate(acts, axis=0),
        "groups": np.concatenate(groups, axis=0),
        "episode_ids": np.concatenate(episode_ids, axis=0),
        "labels": {k: np.concatenate(v) for k, v in labels.items()},
        "pixel": {k: np.concatenate(v) for k, v in pixel.items()},
    }
    if return_observations:
        result["observations"] = np.concatenate(obs_chunks, axis=0)
    return result
