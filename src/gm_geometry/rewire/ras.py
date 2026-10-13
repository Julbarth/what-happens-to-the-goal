"""Attribute Maze frozen-trunk rewiring: h_pre/h_fc taps, teacher-BC and closed-loop held-out evaluation."""
# Env access (make_ras_env, load_ras_policy, trap-preference rollouts) is imported lazily from the
# head_rewire helper scripts, which must be on sys.path.
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gm_geometry.rewire import bootstrap_diff_ci, prepare_tap, trainable_parameters

COHORTS: dict[str, dict] = {
    "black": {
        "gm_seeds": (6313, 6315, 6317),
        "teacher_seeds": (4371, 4372, 4373),
        "use_backgrounds": False,
    },
    "texture": {
        "gm_seeds": (6301, 6302, 6303),
        "teacher_seeds": (4361, 4362, 4363),
        "use_backgrounds": True,
    },
}


def _bundle_imports():
    """Lazy import of the head_rewire helper scripts (must be on sys.path)."""
    import head_rewire_layer_features as hrlf  # noqa
    import head_rewire_ras_maze as hrm  # noqa

    return hrm, hrlf


def collect_block3_teacher_bc(
    trunk_policy,
    teacher_policy,
    *,
    use_backgrounds: bool,
    target_steps: int,
    num_envs: int,
    device: str,
    seed: int = 0,
):
    """Roll teacher_policy through the background-matched env and cache raw (pre-ReLU) block3 flats from
    trunk_policy plus the teacher's action per frame; ReLU is applied at train/eval time."""
    hrm, hrlf = _bundle_imports()
    env = hrm.make_ras_env(num_envs=num_envs, num_levels=100_000, start_level=0, use_backgrounds=use_backgrounds)
    rng = np.random.RandomState(seed)
    obs = env.reset()
    feats: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    steps = 0
    with torch.no_grad():
        while steps < target_steps:
            h = hrlf.block3_flat_from_policy(trunk_policy, obs, device)
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            hid = torch.zeros(x.shape[0], teacher_policy.embedder.output_dim, device=device)
            dist, _v, _ = teacher_policy(x, hid, torch.zeros(x.shape[0], device=device))
            act = dist.probs.argmax(-1).cpu().numpy().astype(np.int64)
            feats.append(h)
            acts.append(act)
            obs, _r, _done, _info = env.step(act)
            steps += obs.shape[0]
    if hasattr(env, "close"):
        env.close()
    return np.concatenate(feats, axis=0), np.concatenate(acts, axis=0)


def train_block3_bc(
    policy,
    block3_features: np.ndarray,
    actions: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    prepare: bool = True,
    epoch_cb=None,
) -> list[dict]:
    """BC on cached block3 flats: prepare_tap(policy, "block3") freezes block1-3 and this trains the fresh suffix.

    `prepare=False` is required for the random-projection control, whose seeded fixed fc a second
    prepare_tap would silently redraw."""
    if prepare:
        prepare_tap(policy, "block3")
    # requires_grad filter: with a frozen random fc the tap's module list still contains fc,
    # and passing frozen params to Adam would misreport trainable capacity.
    params = [p for p in trainable_parameters(policy, "block3") if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.RandomState(0)
    n_val = max(1, len(block3_features) // 5)
    idx = rng.permutation(len(block3_features))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    fc, head = policy.embedder.fc, policy.fc_policy
    log: list[dict] = []
    grad_steps = 0
    for epoch in range(epochs):
        perm = rng.permutation(tr_idx)
        total = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            xb = torch.as_tensor(block3_features[b], device=device).clamp_min(0.0)
            yb = torch.as_tensor(actions[b], device=device, dtype=torch.long)
            opt.zero_grad()
            logits = head(F.relu(fc(xb)))
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            grad_steps += len(b)
            total += float(loss.detach()) * len(b)
        with torch.no_grad():
            xv = torch.as_tensor(block3_features[val_idx], device=device).clamp_min(0.0)
            yv = torch.as_tensor(actions[val_idx], device=device, dtype=torch.long)
            logits_v = head(F.relu(fc(xv)))
            vloss = float(F.cross_entropy(logits_v, yv).cpu())
            vacc = float((logits_v.argmax(-1) == yv).float().mean().cpu())
        row = {"epoch": epoch, "step": grad_steps, "train_ce": total / max(len(tr_idx), 1), "val_ce": vloss, "val_acc": vacc}
        log.append(row)
        print(f"  block3 bc epoch {epoch}: train_ce={row['train_ce']:.4f} val_ce={vloss:.4f} val_acc={vacc:.4f}", flush=True)
        if epoch_cb is not None:
            epoch_cb(epoch, row)
    return log


def collect_fc_teacher_bc(
    trunk_policy,
    teacher_policy,
    *,
    use_backgrounds: bool,
    target_steps: int,
    num_envs: int,
    device: str,
    seed: int = 0,
):
    """Like collect_block3_teacher_bc but caches post-ReLU fc embeddings, for the arm that keeps the teacher's fc frozen."""
    hrm, _hrlf = _bundle_imports()
    env = hrm.make_ras_env(num_envs=num_envs, num_levels=100_000, start_level=0, use_backgrounds=use_backgrounds)
    obs = env.reset()
    feats: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    steps = 0
    with torch.no_grad():
        while steps < target_steps:
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)
            h = trunk_policy.embedder(x).detach().cpu().numpy()
            hid = torch.zeros(x.shape[0], teacher_policy.embedder.output_dim, device=device)
            dist, _v, _ = teacher_policy(x, hid, torch.zeros(x.shape[0], device=device))
            act = dist.probs.argmax(-1).cpu().numpy().astype(np.int64)
            feats.append(h)
            acts.append(act)
            obs, _r, _done, _info = env.step(act)
            steps += obs.shape[0]
    if hasattr(env, "close"):
        env.close()
    return np.concatenate(feats, axis=0), np.concatenate(acts, axis=0)


def train_fc_bc(
    policy,
    fc_features: np.ndarray,
    actions: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    prepare: bool = True,
    epoch_cb=None,
) -> list[dict]:
    """h_fc tap: prepare_tap(policy, "fc") freezes the (already-trained, not
    reinitialized) block1-3+fc and reinitializes only the heads; trains those."""
    if prepare:
        prepare_tap(policy, "fc")
    params = [p for p in trainable_parameters(policy, "fc") if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.RandomState(0)
    n_val = max(1, len(fc_features) // 5)
    idx = rng.permutation(len(fc_features))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    head = policy.fc_policy
    log: list[dict] = []
    grad_steps = 0
    for epoch in range(epochs):
        perm = rng.permutation(tr_idx)
        total = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            xb = torch.as_tensor(fc_features[b], device=device)
            yb = torch.as_tensor(actions[b], device=device, dtype=torch.long)
            opt.zero_grad()
            loss = F.cross_entropy(head(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            grad_steps += len(b)
            total += float(loss.detach()) * len(b)
        with torch.no_grad():
            xv = torch.as_tensor(fc_features[val_idx], device=device)
            yv = torch.as_tensor(actions[val_idx], device=device, dtype=torch.long)
            logits_v = head(xv)
            vloss = float(F.cross_entropy(logits_v, yv).cpu())
            vacc = float((logits_v.argmax(-1) == yv).float().mean().cpu())
        row = {"epoch": epoch, "step": grad_steps, "train_ce": total / max(len(tr_idx), 1), "val_ce": vloss, "val_acc": vacc}
        log.append(row)
        print(f"  fc bc epoch {epoch}: train_ce={row['train_ce']:.4f} val_ce={vloss:.4f} val_acc={vacc:.4f}", flush=True)
        if epoch_cb is not None:
            epoch_cb(epoch, row)
    return log


def eval_trap_preference(policy, device: str, *, use_backgrounds: bool, n_episodes: int, max_steps: int = 256, start_level: int = 100_000) -> dict:
    """Greedy closed-loop rollout to termination, classified by the raw terminal reward (info['env_reward'], ±10)."""
    from gm_geometry.probes.steering import run_preference_episodes

    return run_preference_episodes(
        policy, "red_line_diag", "yellow_gem",
        n_episodes=n_episodes, max_steps=max_steps, start_level=start_level,
        use_backgrounds=use_backgrounds, greedy=True,
    )


def random_frozen_fc(policy, *, seed: int = 0) -> None:
    """random_fc control arm: block1-3 frozen (native GM), fc replaced by a fixed
    random (untrained, never-updated) projection, only heads trained."""
    prepare_tap(policy, "block3")
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # Generate on CPU (matches the default, device-less Generator above) then
        # let copy_ move it to whatever device the real weight lives on -- xavier_uniform_
        # requires the generator and tensor to be on the same device.
        w = torch.empty(policy.embedder.fc.weight.shape)
        torch.nn.init.xavier_uniform_(w, generator=g)
        policy.embedder.fc.weight.copy_(w)
        policy.embedder.fc.bias.zero_()
    for p in policy.embedder.fc.parameters():
        p.requires_grad = False
    policy.embedder.fc.eval()


# BC bank and closed-loop eval draw from disjoint level ranges: the bank rolls
# levels [0, 100_000) and eval uses [100_000, 100_000 + n_episodes). procgen ties
# no difficulty to level id, so this is a clean held-out split.
BC_START_LEVEL = 0
BC_NUM_LEVELS = 100_000
EVAL_START_LEVEL = 100_000

OBS_SHAPE = (3, 64, 64)


def collect_balanced_obs_bank(
    driver_policies,
    *,
    use_backgrounds: bool,
    frames_per_driver: int,
    num_envs: int,
    device: str,
    rand_seed: int,
    out_obs,
    out_driver,
) -> dict:
    """Fill `out_obs` with a driver-balanced observation bank (one block per driver).

    Rolling every GM and teacher checkpoint for an equal share keeps the bank equally on-distribution for both
    populations. Observations are stored as uint8; the wrapper stack makes the float obs exactly uint8/255."""
    # Only the env glue is needed here; head_rewire_layer_features drags in the coin path setup.
    import head_rewire_ras_maze as hrm

    filled = 0
    per_driver: list[int] = []
    for d_idx, driver in enumerate(driver_policies):
        # Distinct level stream per driver: with one shared rand_seed the bank
        # would be six near-copies of the same level sequence, which would make
        # the pooled bank far less diverse than its frame count suggests.
        env = hrm.make_ras_env(
            num_envs=num_envs,
            num_levels=BC_NUM_LEVELS,
            start_level=BC_START_LEVEL,
            use_backgrounds=use_backgrounds,
            rand_seed=rand_seed + d_idx,
        )
        obs = env.reset()
        if tuple(obs.shape[1:]) != OBS_SHAPE:
            raise ValueError(f"expected obs {OBS_SHAPE} from the wrapper stack, got {tuple(obs.shape[1:])}")
        got = 0
        with torch.no_grad():
            while got < frames_per_driver:
                take = min(obs.shape[0], frames_per_driver - got)
                out_obs[filled : filled + take] = np.rint(obs[:take] * 255.0).astype(np.uint8)
                out_driver[filled : filled + take] = d_idx
                filled += take
                got += take
                x = torch.as_tensor(obs, dtype=torch.float32, device=device)
                hid = torch.zeros(x.shape[0], driver.embedder.output_dim, device=device)
                dist, _v, _h = driver(x, hid, torch.zeros(x.shape[0], device=device))
                act = dist.probs.argmax(-1).cpu().numpy().astype(np.int64)
                obs, _r, _done, _info = env.step(act)
        if hasattr(env, "close"):
            env.close()
        per_driver.append(got)
        print(f"  bank: driver {d_idx} contributed {got} frames (total {filled})", flush=True)
    return {"n_frames": filled, "per_driver": per_driver}


@torch.no_grad()
def label_obs_bank_teacher_greedy(labeler_policy, obs_u8, *, device: str, batch_size: int = 512) -> np.ndarray:
    """Greedy argmax of the teacher's action distribution on every banked frame.

    Batched labelling equals labelling during the rollout because these policies are non-recurrent."""
    n = len(obs_u8)
    out = np.empty(n, dtype=np.int64)
    for s in range(0, n, batch_size):
        chunk = np.asarray(obs_u8[s : s + batch_size], dtype=np.float32) / 255.0
        x = torch.as_tensor(chunk, device=device)
        hid = torch.zeros(x.shape[0], labeler_policy.embedder.output_dim, device=device)
        dist, _v, _h = labeler_policy(x, hid, torch.zeros(x.shape[0], device=device))
        out[s : s + batch_size] = dist.probs.argmax(-1).cpu().numpy().astype(np.int64)
    return out


def _oracle_labels(*_args, **_kwargs) -> np.ndarray:
    raise NotImplementedError(
        "oracle labels cannot be produced post-hoc from a stored observation: the graph-exact "
        "BFS optimal-action SET needs the walkable grid, the agent cell and both object cells, "
        "none of which survive into a 64x64 uint8 frame. The oracle labels ONLINE, during the "
        "rollout that produces the frame -- see "
        "gm_geometry.rewire.ras_oracle.collect_balanced_oracle_obs_bank, which the aligned "
        "runner dispatches to via COLLECTOR_LABEL_SOURCES."
    )


# Post-hoc labelers: labeler(policy, obs_u8, *, device, batch_size) -> (N,) int64 hard labels.
# Swapping one of these is the only thing such a variant changes; the state bank, the taps,
# the arms and the eval are all label-agnostic.
LABEL_SOURCES = {
    "teacher_greedy": label_obs_bank_teacher_greedy,
    "oracle": _oracle_labels,
}

# Label sources that cannot be expressed as a post-hoc labeler and instead replace the
# COLLECTOR, emitting (N, n_actions) soft targets alongside the observations. They keep the
# same on-disk bank layout and the same driver-balance guarantee; only the label array's
# shape, dtype and arity (one shared array, not one per teacher) differ.
COLLECTOR_LABEL_SOURCES = frozenset({"oracle"})


@torch.no_grad()
def tap_features_from_obs(policy, obs_u8, tap: str, *, device: str, batch_size: int = 512) -> np.ndarray:
    """Cache the frozen-trunk representation each tap's fresh suffix consumes: h_pre (2048-d) or h_fc (256-d).

    Both are recomputed rather than hooked, so the cached tensor is the same function the rollout applies."""
    emb = policy.embedder
    if not hasattr(emb, "block1"):
        raise TypeError("tap_features_from_obs expects a plain ImpalaModel embedder")
    if tap == "block3":
        dim = int(emb.fc.in_features)
    elif tap == "fc":
        dim = int(emb.output_dim)
    else:
        raise ValueError(f"unsupported tap {tap!r} for Ras rewire (expected 'block3' or 'fc')")
    policy.eval()
    n = len(obs_u8)
    out = np.empty((n, dim), dtype=np.float32)
    for s in range(0, n, batch_size):
        chunk = np.asarray(obs_u8[s : s + batch_size], dtype=np.float32) / 255.0
        x = torch.as_tensor(chunk, device=device)
        if tap == "block3":
            h = F.relu(emb.block3(emb.block2(emb.block1(x)))).flatten(1)
        else:
            h = emb(x)
        out[s : s + batch_size] = h.cpu().numpy()
    return out


def train_tap_bc(policy, features: np.ndarray, actions: np.ndarray, tap: str, *, groups=None, **kwargs) -> list[dict]:
    """Train a tap's suffix, dispatching on the label's shape rather than a flag.

    (N,) int64 labels use hard cross-entropy and argmax accuracy; (N, A) float labels use soft-target
    cross-entropy over the tied optimal set and set-aware accuracy."""
    if np.asarray(actions).ndim == 2:
        from gm_geometry.rewire.ras_oracle import train_soft_bc

        return train_soft_bc(policy, features, actions, tap, groups=groups, **kwargs)
    if groups is not None:
        raise ValueError("`groups` is only meaningful for soft (set) labels; hard-label BC uses the i.i.d. frame split")
    if tap == "block3":
        return train_block3_bc(policy, features, actions, **kwargs)
    if tap == "fc":
        return train_fc_bc(policy, features, actions, **kwargs)
    raise ValueError(f"unsupported tap {tap!r}")


def trainable_param_count(policy, tap: str) -> int:
    return int(sum(p.numel() for p in trainable_parameters(policy, tap) if p.requires_grad))


def outcome_metrics(rewards: np.ndarray, steps: np.ndarray, *, max_steps: int) -> dict:
    """Unconditional goal rate R_goal = n_goal/n_episodes plus its factorization R_goal = D * Pref_goal.

    `pref_goal` alone is uninterpretable when an arm rarely reaches either object, so decisiveness D and
    conditional preference are reported separately."""
    from fractions import Fraction

    from gm_geometry.probes.steering import preference_from_rewards

    rewards = np.asarray(rewards, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.int64)
    base = preference_from_rewards(rewards)
    n = int(rewards.size)
    n_goal, n_proxy = int(base["n_goal_hit"]), int(base["n_proxy_hit"])
    n_decisive = n_goal + n_proxy
    if base["n_decisive"] != n_decisive or n_goal + n_proxy + int(base["n_zero"]) != n:
        raise AssertionError(
            f"episode bookkeeping inconsistent: goal={n_goal} proxy={n_proxy} "
            f"zero={base['n_zero']} decisive={base['n_decisive']} n={n}"
        )

    r_goal = n_goal / n if n else float("nan")
    decisive_frac = n_decisive / n if n else float("nan")
    pref_goal = n_goal / n_decisive if n_decisive else float("nan")
    if n_decisive:
        # Exact over the counts (Fraction, no rounding); the float check is
        # abs_tol rather than `==` only because D * Pref rounds the same rational
        # twice -- e.g. g=8, p=4, n=1000 gives 0.008000000000000002 vs 0.008.
        if Fraction(n_goal, n) != Fraction(n_decisive, n) * Fraction(n_goal, n_decisive):
            raise AssertionError(f"R_goal != D * Pref_goal over counts: {n_goal}/{n}, {n_decisive}, {n_goal}")
        if abs(r_goal - decisive_frac * pref_goal) > 1e-12:
            raise AssertionError(
                f"R_goal != D * Pref_goal in floats: {r_goal!r} vs {decisive_frac * pref_goal!r} "
                f"(goal={n_goal} proxy={n_proxy} n={n})"
            )

    timeout = steps >= max_steps
    goal_mask, proxy_mask = rewards == 10.0, rewards == -10.0
    neither = ~(goal_mask | proxy_mask)

    def _mean(a):
        return float(a.mean()) if a.size else float("nan")

    return {
        "n_episodes": n,
        "n_goal_hit": n_goal,
        "n_proxy_hit": n_proxy,
        "n_zero": int(base["n_zero"]),
        "n_decisive": n_decisive,
        "R_goal": r_goal,
        "decisive_frac": decisive_frac,
        "pref_goal": pref_goal,
        "mean_reward": float(base["mean_reward"]),
        "mean_steps": _mean(steps.astype(np.float64)),
        "median_steps": float(np.median(steps)) if n else float("nan"),
        "p10_steps": float(np.quantile(steps, 0.10)) if n else float("nan"),
        "p90_steps": float(np.quantile(steps, 0.90)) if n else float("nan"),
        "frac_timeout": _mean(timeout.astype(np.float64)),
        "mean_steps_goal": _mean(steps[goal_mask].astype(np.float64)),
        "mean_steps_proxy": _mean(steps[proxy_mask].astype(np.float64)),
        "mean_steps_neither": _mean(steps[neither].astype(np.float64)),
    }


def _rollout_episode_with_actions(policy, env, *, max_steps: int, greedy: bool = True):
    """One greedy episode returning (summed raw reward, steps, action trace), as probes.steering.rollout_episode_reward."""
    obs = env.reset()
    hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)
    total = 0.0
    acts: list[int] = []
    device = next(policy.parameters()).device
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
            act_np = act.detach().cpu().numpy()
            obs, rew, done, info = env.step(act_np)
            acts.append(int(act_np[0]))
            info0 = info[0] if isinstance(info, (list, tuple)) else info
            raw = info0.get("env_reward") if isinstance(info0, dict) else None
            total += float(raw) if raw is not None else float(np.asarray(rew)[0])
            if bool(np.asarray(done)[0]):
                break
    return total, len(acts), np.asarray(acts, dtype=np.int64)


def eval_closed_loop(
    policy,
    device: str,
    *,
    use_backgrounds: bool,
    n_episodes: int,
    max_steps: int = 256,
    start_level: int = EVAL_START_LEVEL,
    n_actions: int | None = None,
    greedy: bool = True,   # False => sample the logits; used only for the chance floor
   
) -> dict:
    """Held-out closed-loop rollout: unconditional R_goal plus validity diagnostics.

    Level i is start_level + i with num_levels=1, so every arm is scored on identical levels."""
    from gm_geometry.integrations.colour_shape import make_duel_env

    policy.eval()
    n_actions = int(n_actions or policy.fc_policy.out_features)
    rewards, steps, entropies, distinct = [], [], [], []
    for i in range(n_episodes):
        env = make_duel_env(
            1, 1, 1, "red_line_diag", "yellow_gem",
            start_level=start_level + i, use_backgrounds=use_backgrounds,
        )
        try:
            rew, n_steps, acts = _rollout_episode_with_actions(policy, env, max_steps=max_steps,
                                                                  greedy=greedy)
        finally:
            if hasattr(env, "close"):
                env.close()
        counts = np.bincount(acts, minlength=n_actions).astype(np.float64)
        p = counts / max(counts.sum(), 1.0)
        nz = p[p > 0]
        rewards.append(rew)
        steps.append(n_steps)
        entropies.append(float(max(0.0, -(nz * np.log(nz)).sum())))
        distinct.append(int((counts > 0).sum()))

    out = outcome_metrics(np.asarray(rewards), np.asarray(steps), max_steps=max_steps)
    # Degeneracy screen, not a competence measure: a suffix collapsed onto one action has
    # entropy 0. Low-but-nonzero entropy is normal for a policy running down a corridor.
    out["mean_action_entropy"] = float(np.mean(entropies)) if entropies else float("nan")
    out["mean_distinct_actions"] = float(np.mean(distinct)) if distinct else float("nan")
    out["episode_rewards"] = [float(r) for r in rewards]
    out["episode_steps"] = [int(s) for s in steps]
    out["episode_action_entropy"] = entropies
    out["episode_distinct_actions"] = distinct
    out["start_level"] = start_level
    out["max_steps"] = max_steps
    return out
