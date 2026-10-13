"""Causal steering and ablation at ``embedder.fc`` with per-episode ±10 preference scoring."""

from __future__ import annotations

from typing import Callable

try:
    from typing import Literal
except ImportError:  # Python < 3.8 (procgen env)
    from typing_extensions import Literal

import numpy as np
import torch

from gm_geometry.integrations.colour_shape.envs import make_duel_env
from gm_geometry.probes.head_projection import unit_vector

SteerMode = Literal["add", "project_out"]


def projection_std(h: np.ndarray, w: np.ndarray) -> float:
    """Per-frame std of scalar projections ``h·û``."""
    u = unit_vector(w)
    return float(np.std(h @ u))


def make_steer_hook(
    direction: np.ndarray,
    *,
    alpha: float,
    mode: SteerMode = "add",
) -> Callable:
    """Return a forward hook that steers ``embedder.fc`` outputs.

    Modes
    -----
    ``add``
        ``h' = h + α û`` (constant additive injection).
    ``project_out``
        ``h' = h - α (h·û) û`` (α=1 fully removes the axis; α∈(0,1) soft ablation).
    """
    u = unit_vector(direction)
    u_t = torch.as_tensor(u, dtype=torch.float32)

    if mode == "add":
        if alpha == 0.0:
            def _noop(_m, _i, out):
                return out
            return _noop

        def _add(_m, _i, out):
            return out + alpha * u_t.to(device=out.device, dtype=out.dtype).view(1, -1)

        return _add

    if mode == "project_out":
        def _proj(_m, _i, out):
            u_d = u_t.to(device=out.device, dtype=out.dtype).view(1, -1)
            coef = (out * u_d).sum(dim=-1, keepdim=True)
            return out - alpha * coef * u_d

        return _proj

    raise ValueError(f"unknown steer mode: {mode!r}")


def rollout_episode_reward(
    policy,
    env,
    *,
    max_steps: int,
    greedy: bool,
    steer_dir: np.ndarray | None = None,
    alpha: float = 0.0,
    mode: SteerMode = "add",
) -> tuple[float, int]:
    """Single episode until ``done`` (or ``max_steps``). Returns (total_reward, steps)."""
    module = policy.embedder.fc
    handle = None
    if steer_dir is not None and not (mode == "add" and alpha == 0.0):
        handle = module.register_forward_hook(
            make_steer_hook(steer_dir, alpha=alpha, mode=mode),
        )

    obs = env.reset()
    hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)
    total = 0.0
    steps = 0
    device = next(policy.parameters()).device
    try:
        with torch.no_grad():
            for _ in range(max_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                dist, _v, hidden_t = policy(
                    obs_t,
                    torch.as_tensor(hidden, dtype=torch.float32, device=device),
                    torch.zeros(env.num_envs, device=device),
                )
                hidden = hidden_t.detach().cpu().numpy()
                act = dist.probs.argmax(-1) if greedy else dist.sample()
                obs, rew, done, info = env.step(act.detach().cpu().numpy())
                # VecNormalize return-scales `rew`, so a terminal ±10 is not exactly ±10 here;
                # info['env_reward'] carries the raw reward needed for exact-value classification.
                info0 = info[0] if isinstance(info, (list, tuple)) else info
                raw = info0.get("env_reward") if isinstance(info0, dict) else None
                total += float(raw) if raw is not None else float(np.asarray(rew)[0])
                steps += 1
                if bool(np.asarray(done)[0]):
                    break
    finally:
        if handle is not None:
            handle.remove()
    return total, steps


def preference_from_rewards(rewards: np.ndarray) -> dict:
    """Decisive ±10 preference — same definition as behavioural GM evals."""
    arr = np.asarray(rewards, dtype=np.float64)
    n_goal = int((arr == 10.0).sum())
    n_proxy = int((arr == -10.0).sum())
    n_zero = int((arr == 0.0).sum())
    decisive = n_goal + n_proxy
    return {
        "mean_reward": float(arr.mean()) if arr.size else float("nan"),
        "std_reward": float(arr.std()) if arr.size else float("nan"),
        "n_goal_hit": n_goal,
        "n_proxy_hit": n_proxy,
        "n_zero": n_zero,
        "n_decisive": decisive,
        "pref_goal": float(n_goal / decisive) if decisive else float("nan"),
        # Goal-hit rate over all episodes, not only decisive ones.
        "frac_goal_reward": float((arr == 10.0).mean()) if arr.size else float("nan"),
        "episode_rewards": arr.tolist(),
        "n_episodes": int(arr.size),
    }


def collection_rate_preference(rewards: np.ndarray, *, goal_value: float = 10.0) -> dict:
    """Single-goal collect rate (CoinRun / Maze): fraction of episodes that reach the goal reward.

    Scalar reward cannot separate goal-misgeneralizing episodes from capability timeouts; that needs per-step info bins."""
    arr = np.asarray(rewards, dtype=np.float64)
    n_goal = int((arr >= goal_value).sum())
    n = int(arr.size)
    return {
        "mean_reward": float(arr.mean()) if n else float("nan"),
        "std_reward": float(arr.std()) if n else float("nan"),
        "n_goal_hit": n_goal,
        "n_episodes": n,
        "pref_goal": float(n_goal / n) if n else float("nan"),
        "collect_rate": float(n_goal / n) if n else float("nan"),
        "episode_rewards": arr.tolist(),
        "outcome_classifier": "collection_rate_scalar_reward",
    }


def run_preference_episodes_env(
    policy,
    env_factory: Callable[[int], object],
    *,
    n_episodes: int,
    preference_fn: Callable[[np.ndarray], dict] = preference_from_rewards,
    max_steps: int = 256,
    start_level: int = 100_000,
    greedy: bool = True,
    steer_dir: np.ndarray | None = None,
    alpha: float = 0.0,
    mode: SteerMode = "add",
) -> dict:
    """Env-agnostic steering loop.

    ``env_factory(level)`` returns a fresh single-env venv and ``preference_fn(rewards)`` maps the per-episode
    rewards to a behavioural summary; the steering hooks are shared across envs."""
    policy.eval()
    rewards: list[float] = []
    steps_list: list[int] = []
    for i in range(n_episodes):
        env = env_factory(start_level + i)
        try:
            rew, steps = rollout_episode_reward(
                policy, env,
                max_steps=max_steps,
                greedy=greedy,
                steer_dir=steer_dir,
                alpha=alpha,
                mode=mode,
            )
        finally:
            if hasattr(env, "close"):
                env.close()
        rewards.append(rew)
        steps_list.append(steps)

    out = preference_fn(np.asarray(rewards, dtype=np.float64))
    out["mean_steps"] = float(np.mean(steps_list)) if steps_list else float("nan")
    out["greedy"] = greedy
    out["mode"] = mode
    out["alpha"] = float(alpha)
    return out


def run_preference_episodes(
    policy,
    obj1: str,
    obj2: str,
    *,
    n_episodes: int,
    max_steps: int = 256,
    start_level: int = 100_000,
    use_backgrounds: bool = False,
    greedy: bool = True,
    steer_dir: np.ndarray | None = None,
    alpha: float = 0.0,
    mode: SteerMode = "add",
    num_threads: int = 1,
) -> dict:
    """Colour-shape duel steering, a thin wrapper over the env-agnostic loop."""
    def _duel_factory(level: int):
        return make_duel_env(
            1, 1, num_threads, obj1, obj2,
            start_level=level, use_backgrounds=use_backgrounds,
        )

    return run_preference_episodes_env(
        policy, _duel_factory,
        n_episodes=n_episodes,
        preference_fn=preference_from_rewards,
        max_steps=max_steps,
        start_level=start_level,
        greedy=greedy,
        steer_dir=steer_dir,
        alpha=alpha,
        mode=mode,
    )


def collect_episode_activations(
    policy,
    obj1: str,
    obj2: str,
    *,
    n_episodes: int,
    max_steps: int = 256,
    start_level: int = 100_000,
    use_backgrounds: bool = False,
    greedy: bool = True,
    num_threads: int = 1,
) -> dict:
    """Unsteered trap rollouts with per-episode mean ``embedder.fc`` activation.

    Used to build a behavioural difference-in-means direction
    ``normalize(E[h|goal] − E[h|lure])``.
    """
    policy.eval()
    module = policy.embedder.fc
    ep_means: list[np.ndarray] = []
    ep_rewards: list[float] = []
    ep_early: list[np.ndarray] = []

    for i in range(n_episodes):
        env = make_duel_env(
            1, 1, num_threads, obj1, obj2,
            start_level=start_level + i,
            use_backgrounds=use_backgrounds,
        )
        frames: list[np.ndarray] = []

        def _hook(_m, _i, out):
            frames.append(out.detach().cpu().numpy().reshape(-1).astype(np.float64))

        handle = module.register_forward_hook(_hook)
        try:
            rew, _steps = rollout_episode_reward(
                policy, env, max_steps=max_steps, greedy=greedy,
            )
        finally:
            handle.remove()
            env.close()

        if not frames:
            continue
        stacked = np.stack(frames, axis=0)
        ep_means.append(stacked.mean(axis=0))
        ep_early.append(stacked[: min(8, len(stacked))].mean(axis=0))
        ep_rewards.append(rew)

    rewards = np.asarray(ep_rewards, dtype=np.float64)
    means = np.stack(ep_means, axis=0) if ep_means else np.zeros((0, 0))
    early = np.stack(ep_early, axis=0) if ep_early else np.zeros((0, 0))
    goal_mask = rewards == 10.0
    lure_mask = rewards == -10.0
    return {
        "episode_mean_activations": means,
        "episode_early_activations": early,
        "episode_rewards": rewards.tolist(),
        "n_goal": int(goal_mask.sum()),
        "n_lure": int(lure_mask.sum()),
        "goal_mask": goal_mask,
        "lure_mask": lure_mask,
    }


def behavioural_contrast_direction(
    episode_activations: np.ndarray,
    goal_mask: np.ndarray,
    lure_mask: np.ndarray,
) -> np.ndarray | None:
    """Unit ``E[h|goal] − E[h|lure]``, or ``None`` if either class is empty."""
    if episode_activations.size == 0:
        return None
    if not np.any(goal_mask) or not np.any(lure_mask):
        return None
    delta = episode_activations[goal_mask].mean(axis=0) - episode_activations[lure_mask].mean(axis=0)
    return unit_vector(delta)


def bootstrap_pref_ci(
    episode_rewards: list[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for ``pref_goal`` among decisive (±10) episodes."""
    arr = np.asarray(episode_rewards, dtype=np.float64)
    decisive = arr[(arr == 10.0) | (arr == -10.0)]
    if decisive.size == 0:
        return {"pref_goal": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_decisive": 0}
    rng = np.random.default_rng(seed)
    prefs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(decisive, size=decisive.size, replace=True)
        prefs[i] = float((sample == 10.0).mean())
    lo, hi = np.quantile(prefs, [alpha / 2, 1 - alpha / 2])
    return {
        "pref_goal": float((decisive == 10.0).mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_decisive": int(decisive.size),
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Direction triangulation (ridge vs contrastive vs readout)
# ---------------------------------------------------------------------------


def fit_ridge_directions(
    activations: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    ridge_alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    from gm_geometry.probes.linear import fit_ridge

    out: dict[str, np.ndarray] = {}
    for concept, y in labels.items():
        w, _ = fit_ridge(activations, y, alpha=ridge_alpha)
        out[concept] = unit_vector(w)
    return out


def contrastive_directions(
    activations: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Unit ``mean(h|y=1) − mean(h|y=0)`` per concept — alias of raw Dom."""
    from gm_geometry.pca.metrics import fit_dom_directions

    return fit_dom_directions(activations, labels)


def fit_dom_directions(
    activations: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Raw Dom: unit ``normalize(μ₁ − μ₀)`` per concept (probe-free)."""
    from gm_geometry.pca.metrics import fit_dom_directions as _fit

    return _fit(activations, labels)


def fit_lda_directions(
    activations: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    """Whitened Dom / LDA-ish: unit ``normalize((Σ+αI)^{-1} Δμ)`` per concept."""
    from gm_geometry.pca.metrics import fit_lda_directions as _fit

    return _fit(activations, labels, alpha=alpha)


def value_head_direction(policy) -> np.ndarray:
    w = policy.fc_value.weight.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return unit_vector(w)


def policy_subspace_direction(policy, span: list[np.ndarray]) -> np.ndarray:
    """Dominant direction of ``W_π`` inside the span of the given unit vectors."""
    W = policy.fc_policy.weight.detach().cpu().numpy().astype(np.float64)
    cols = [unit_vector(v) for v in span if float(np.linalg.norm(v)) > 1e-8]
    if not cols:
        return np.zeros(W.shape[1], dtype=np.float64)
    Q, _ = np.linalg.qr(np.column_stack(cols))
    _, _, vt = np.linalg.svd(W @ Q, full_matrices=False)
    return unit_vector(Q @ vt[0])


def mean_policy_row_direction(
    policy,
    observations: np.ndarray,
    *,
    greedy: bool = True,
    max_frames: int = 256,
) -> np.ndarray:
    """Mean ``W_π[a_t]`` under the policy's actions — cheap ``∇_h log π(a|h)`` proxy."""
    W = policy.fc_policy.weight.detach().cpu().numpy().astype(np.float64)
    obs = np.asarray(observations, dtype=np.float32)
    if obs.shape[0] > max_frames:
        idx = np.linspace(0, obs.shape[0] - 1, max_frames).astype(int)
        obs = obs[idx]
    hidden_dim = policy.embedder.output_dim
    rows = []
    policy.eval()
    with torch.no_grad():
        for i in range(obs.shape[0]):
            obs_t = torch.as_tensor(obs[i : i + 1], dtype=torch.float32)
            dist, _v, _hx = policy(obs_t, torch.zeros(1, hidden_dim), torch.zeros(1))
            a = int(dist.probs.argmax(-1).item() if greedy else dist.sample().item())
            rows.append(W[a])
    if not rows:
        return np.zeros(W.shape[1], dtype=np.float64)
    return unit_vector(np.mean(np.stack(rows, axis=0), axis=0))


def direction_triangulation(
    *,
    ridge: dict[str, np.ndarray],
    contrastive: dict[str, np.ndarray],
    value_dir: np.ndarray,
    policy_span_dir: np.ndarray,
    policy_action_dir: np.ndarray,
    behavioural_dir: np.ndarray | None = None,
    concepts: tuple[str, ...] = ("yellow", "line"),
) -> dict:
    """Pairwise cosines between candidate directions plus the steer gate."""
    from gm_geometry.probes.head_projection import cosine_similarity

    named: dict[str, np.ndarray] = {
        "value_head": value_dir,
        "policy_span_yl": policy_span_dir,
        "policy_action_mean": policy_action_dir,
    }
    if behavioural_dir is not None:
        named["behavioural_goal_minus_lure"] = behavioural_dir
    for c in concepts:
        named[f"ridge_{c}"] = ridge[c]
        named[f"contrastive_{c}"] = contrastive[c]

    cosines: dict[str, float] = {}
    keys = list(named.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            cosines[f"cos({a},{b})"] = cosine_similarity(named[a], named[b])

    gate = {
        "cos_ridge_line_vs_policy_span": cosine_similarity(ridge["line"], policy_span_dir),
        "cos_ridge_yellow_vs_policy_span": cosine_similarity(ridge["yellow"], policy_span_dir),
        "cos_ridge_line_vs_policy_action": cosine_similarity(ridge["line"], policy_action_dir),
        "cos_ridge_yellow_vs_policy_action": cosine_similarity(ridge["yellow"], policy_action_dir),
        "cos_ridge_line_vs_contrastive_line": cosine_similarity(ridge["line"], contrastive["line"]),
        "cos_ridge_yellow_vs_contrastive_yellow": cosine_similarity(
            ridge["yellow"], contrastive["yellow"]
        ),
    }
    if behavioural_dir is not None:
        gate["cos_ridge_line_vs_behavioural"] = cosine_similarity(ridge["line"], behavioural_dir)
        gate["cos_ridge_yellow_vs_behavioural"] = cosine_similarity(ridge["yellow"], behavioural_dir)

    return {
        "directions_unit_first5": {k: v[:5].tolist() for k, v in named.items()},
        "pairwise_cosines": cosines,
        "steer_gate": gate,
        "note": (
            "Steering along ridge counts as causal evidence only if "
            "cos(ridge, policy_*) is high; disagreement → probe≠used."
        ),
    }
