"""Graph-exact BFS optimal-action-set oracle labels for Attribute Maze rollout frames."""
# Object cells come from get_object_pos and the walkable grid from set_object return codes. The agent cell is
# not exposed by this procgen fork, so it is dead-reckoned (one cell per move, wall moves are no-ops) and checked
# with an exact probe (GOAL1 placed on a candidate pays REWARD1 on no-op iff it is the agent's cell); any
# disagreement drops the whole level. Labels are tied optimal SETS: LEFT and RIGHT each have three synonyms.
from __future__ import annotations

import time
from collections import deque

import numpy as np

N_ACTIONS = 15
NOOP_ACTION = 4
WALKABLE_RC = 0
GOAL1_WHICH = 1  # obj1 = red_line_diag, +10: the line-defined INTENDED goal
GOAL2_WHICH = 2  # obj2 = yellow_gem, -10: the proxy
SPAWN_CELL = (0, 0)


def action_grid_deltas() -> dict[int, tuple[int, int]]:
    """Each action index's exact (dx, dy) world-grid displacement, transcribed from the game source.

    {0,1,2} are LEFT, {6,7,8} RIGHT, 3 DOWN, 5 UP, and 4 plus 9..14 are no-ops (this game never moves diagonally)."""
    out: dict[int, tuple[int, int]] = {}
    for a in range(N_ACTIONS):
        move = 4 if a >= 9 else a % 9
        vx, vy = move // 3 - 1, move % 3 - 1
        out[a] = (vx, 0) if vx != 0 else (0, vy)
    return out


ACTION_DELTAS = action_grid_deltas()


# --------------------------------------------------------------------------------------
# pure graph logic (no env, unit-testable)
# --------------------------------------------------------------------------------------

def bfs_distances(walkable: set[tuple[int, int]], target: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Single-source BFS distance from `target` over the 4-connected walkable set."""
    if target not in walkable:
        return {}
    dist = {target: 0}
    q = deque([target])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if n in walkable and n not in dist:
                dist[n] = dist[(x, y)] + 1
                q.append(n)
    return dist


def successor_cell(cell: tuple[int, int], action: int, walkable: set[tuple[int, int]]) -> tuple[int, int]:
    """Where action `a` actually lands the agent -- a wall-blocked move is a full no-op."""
    dx, dy = ACTION_DELTAS[action]
    nxt = (cell[0] + dx, cell[1] + dy)
    return nxt if nxt in walkable else cell


def optimal_action_set(
    dist_from_target: dict[tuple[int, int], int],
    cell: tuple[int, int],
    walkable: set[tuple[int, int]],
) -> list[int]:
    """Every action whose realized successor strictly decreases BFS distance to the target.

    Returns [] when the target is unreachable from `cell`, when `cell` IS the target, or
    when nothing makes progress. No-ops and wall-blocked moves are excluded automatically
    because their successor is `cell` itself, whose distance is not strictly smaller.
    """
    base = dist_from_target.get(cell)
    if base is None or base == 0:
        return []
    best, per_action = None, {}
    for a in range(N_ACTIONS):
        d = dist_from_target.get(successor_cell(cell, a, walkable))
        if d is None:
            continue
        per_action[a] = d
        if best is None or d < best:
            best = d
    if best is None or best >= base:
        return []
    return sorted(a for a, d in per_action.items() if d == best)


def make_soft_targets(target_sets, n_actions: int = N_ACTIONS) -> np.ndarray:
    """Uniform mass over each frame's full tied optimal set -- the label the BC head sees."""
    targets = np.zeros((len(target_sets), n_actions), dtype=np.float32)
    for i, s in enumerate(target_sets):
        if not len(s):
            raise ValueError(f"empty target set at row {i}; unresolvable frames must be skipped, not kept")
        for a in s:
            targets[i, a] = 1.0 / len(s)
    return targets


def soft_target_cross_entropy(logits, targets):
    """CE against a distribution, not a class index: -sum_a t_a log pi_a, averaged."""
    import torch.nn.functional as F

    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def set_accuracy(logits, targets):
    """1[argmax_a pi(a) in A*(s)], averaged; argmax equality against one set member would penalize LEFT/RIGHT synonyms."""
    import torch

    pred = logits.argmax(dim=1)
    return (targets[torch.arange(len(pred), device=targets.device), pred] > 0).float().mean()


# --------------------------------------------------------------------------------------
# env access
# --------------------------------------------------------------------------------------

def _unwrap(env):
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "set_object"):
            return cur
        cur = getattr(cur, "venv", None) or getattr(cur, "env", None)
    return None


def _find_vecnormalize(env):
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "ret_rms"):
            return cur
        cur = getattr(cur, "venv", None) or getattr(cur, "env", None)
    return None


def read_object_cells(base) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """(intended GOAL1 cell, proxy GOAL2 cell); None where the object is absent."""
    g1 = base.get_object_pos([GOAL1_WHICH])[0]
    g2 = base.get_object_pos([GOAL2_WHICH])[0]
    return (tuple(g1) if g1 is not None else None, tuple(g2) if g2 is not None else None)


def probe_walkable_cells(base, world_dim: int, margin: int = 2) -> set[tuple[int, int]]:
    """Walkable (SPACE) cell set from set_object's return code; destructive, so the caller restores a snapshot.

    Probes `margin` cells beyond world_dim as a tripwire for an env built at a larger world_dim."""
    base.set_object([GOAL1_WHICH], [0], [0], [0])
    base.set_object([GOAL2_WHICH], [0], [0], [0])
    walkable = set()
    for x in range(-margin, world_dim + margin):
        for y in range(-margin, world_dim + margin):
            if base.set_object([GOAL1_WHICH], [x], [y], [1])[0] == WALKABLE_RC:
                walkable.add((x, y))
    return walkable


def probe_agent_cell(env, base, candidates, snapshot=None) -> tuple[int, int] | None:
    """Ground-truth agent cell: place GOAL1 on a candidate cell and take the no-op action.

    A positive raw reward (info['env_reward'], since VecNormalize scales returns) identifies the agent's cell.
    Every probe step is undone by set_state, which also restores both RNGs."""
    snap = snapshot if snapshot is not None else base.get_state()
    vecnorm = _find_vecnormalize(env)
    saved_ret = None if vecnorm is None else np.array(vecnorm.ret, copy=True)
    hit = None
    for x, y in candidates:
        base.set_state(snap)
        if base.set_object([GOAL1_WHICH], [x], [y], [1])[0] != WALKABLE_RC:
            continue  # cell holds the OTHER goal; the agent cannot be standing there
        _obs, _rew, _done, info = env.step(np.array([NOOP_ACTION]))
        if float(info[0].get("env_reward", 0.0)) > 0.0:
            hit = (x, y)
            break
    base.set_state(snap)
    # ret_rms itself is left perturbed: it only scales rewards, which nothing here reads.
    if vecnorm is not None and saved_ret is not None:
        vecnorm.ret = saved_ret
    return hit


def _greedy_action(policy, obs, device: str, greedy: bool, rng) -> int:
    import torch

    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    with torch.no_grad():
        hid = torch.zeros(x.shape[0], policy.embedder.output_dim, device=device)
        dist, _v, _h = policy(x, hid, torch.zeros(x.shape[0], device=device))
        probs = dist.probs[0].detach().cpu().numpy().astype(np.float64)
    if greedy:
        return int(probs.argmax())
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))



def _new_oracle_stats() -> dict:
    return {
        "n_levels_tried": 0,
        "n_levels_used": 0,
        "n_levels_dropped_geometry": 0,
        "n_levels_dropped_desync": 0,
        "n_levels_dropped_terminal_mismatch": 0,
        "n_frames_unresolved": 0,
        "n_terminal_checks_passed": 0,
        "action_deltas": {str(k): list(v) for k, v in ACTION_DELTAS.items()},
    }


def _summarize_oracle_stats(stats: dict, target_sets, proxy_sets) -> None:
    """Fill the frame-level summary fields from the surviving frames, so dropped levels cannot leak in."""
    hist: dict[str, int] = {}
    for s in target_sets:
        hist[str(len(s))] = hist.get(str(len(s)), 0) + 1
    stats["tie_size_hist"] = hist
    stats["n_frames"] = len(target_sets)
    stats["mean_tie_size"] = float(np.mean([len(s) for s in target_sets])) if target_sets else float("nan")
    stats["n_frames_conflict"] = sum(
        1 for a, p in zip(target_sets, proxy_sets) if p and not set(a) & set(p)
    )
    stats["frac_conflict"] = stats["n_frames_conflict"] / len(target_sets) if target_sets else float("nan")


def oracle_level_frames(
    env,
    base,
    rollout_policy,
    *,
    device: str,
    greedy: bool,
    rng,
    max_frames: int,
    max_steps_per_level: int,
    world_dim: int,
    verify_every: int,
    stats: dict,
):
    """One level's oracle-labelled frames and a drop reason; a non-None reason means discard every frame.

    Each frame is (obs float32 (1, 3, 64, 64), a_star, a_proxy, (ax, ay)). `stats` is updated in place."""
    obs = env.reset()
    snap0 = base.get_state()
    goal_intended, goal_proxy = read_object_cells(base)
    walkable = probe_walkable_cells(base, world_dim)
    base.set_state(snap0)

    if any(max(abs(c[0]), abs(c[1])) >= world_dim for c in walkable):
        raise RuntimeError(
            f"walkable cell outside world_dim={world_dim}; env was built at a "
            "different world_dim than this call assumes"
        )
    if goal_intended is None or goal_intended not in walkable or SPAWN_CELL not in walkable:
        return [], "n_levels_dropped_geometry"

    dist_intended = bfs_distances(walkable, goal_intended)
    dist_proxy = bfs_distances(walkable, goal_proxy) if goal_proxy in walkable else {}
    candidates = sorted(walkable)

    agent = probe_agent_cell(env, base, [SPAWN_CELL], snapshot=snap0)
    if agent is None:
        agent = probe_agent_cell(env, base, candidates, snapshot=snap0)
    if agent is None:
        return [], "n_levels_dropped_geometry"

    frames: list[tuple] = []
    for t in range(max_steps_per_level):
        if t > 0 and verify_every > 0 and t % verify_every == 0:
            if probe_agent_cell(env, base, candidates) != agent:
                return frames, "n_levels_dropped_desync"

        a_star = optimal_action_set(dist_intended, agent, walkable)
        if not a_star:
            stats["n_frames_unresolved"] += 1
        else:
            a_proxy = optimal_action_set(dist_proxy, agent, walkable) if dist_proxy else []
            frames.append((np.asarray(obs, dtype=np.float32), a_star, a_proxy, agent))
            if len(frames) >= max_frames:
                return frames, None

        act = _greedy_action(rollout_policy, obs, device, greedy, rng)
        obs, _rew, done, info = env.step(np.array([act]))
        agent = successor_cell(agent, act, walkable)
        if bool(done[0]):
            env_reward = float(info[0].get("env_reward", 0.0))
            if env_reward > 0 and agent != goal_intended:
                return frames, "n_levels_dropped_terminal_mismatch"
            if env_reward < 0 and agent != goal_proxy:
                return frames, "n_levels_dropped_terminal_mismatch"
            if env_reward != 0:
                stats["n_terminal_checks_passed"] += 1
            break
    return frames, None


def collect_broad_oracle_bc_ras(
    trunk_policy,
    rollout_policy,
    *,
    use_backgrounds: bool,
    target_steps: int,
    num_envs: int = 1,
    device: str,
    seed: int = 0,
    start_level: int = 0,
    n_levels_to_try: int = 20_000,
    max_steps_per_level: int = 64,
    world_dim: int = 5,
    verify_every: int = 8,
    greedy: bool = True,
    verbose: bool = True,
):
    """Roll `rollout_policy` through broad levels, labelling every frame with the optimal first-action SET to the line.

    Drop-in for gm_geometry.rewire.ras.collect_block3_teacher_bc, except the second policy only drives the walk.
    Returns (features (N, D) block3 flats, target_sets, meta). ``verify_every=1`` probes the agent cell every frame;
    larger values rely on dead reckoning in between. Unresolvable or inconsistent levels are dropped and counted."""
    if num_envs != 1:
        raise ValueError(
            "collect_broad_oracle_bc_ras requires num_envs=1: agent-cell tracking and the "
            f"state-snapshot probe are per-env and single-env only (got {num_envs})"
        )
    import head_rewire_layer_features as hrlf  # noqa: E402
    import head_rewire_ras_maze as hrm  # noqa: E402

    rng = np.random.RandomState(seed)
    feats: list[np.ndarray] = []
    target_sets: list[list[int]] = []
    proxy_sets: list[list[int]] = []
    levels: list[int] = []
    agent_cells: list[list[int]] = []
    stats = _new_oracle_stats()

    cur_level = start_level
    t0 = time.time()
    while len(target_sets) < target_steps and stats["n_levels_tried"] < n_levels_to_try:
        stats["n_levels_tried"] += 1
        level, cur_level = cur_level, cur_level + 1
        env = hrm.make_ras_env(num_envs=1, num_levels=1, start_level=level, use_backgrounds=use_backgrounds)
        try:
            base = _unwrap(env)
            if base is None:
                raise RuntimeError("no procgen layer exposing set_object under this env wrapper stack")
            frames, dropped = oracle_level_frames(
                env, base, rollout_policy,
                device=device, greedy=greedy, rng=rng,
                max_frames=target_steps - len(target_sets),
                max_steps_per_level=max_steps_per_level,
                world_dim=world_dim, verify_every=verify_every, stats=stats,
            )
        finally:
            if hasattr(env, "close"):
                env.close()

        if dropped is not None:
            stats[dropped] += 1
        elif frames:
            for obs_f, a_star, a_proxy, agent in frames:
                feats.append(hrlf.block3_flat_from_policy(trunk_policy, obs_f, device)[0].astype(np.float32))
                target_sets.append(a_star)
                proxy_sets.append(a_proxy)
                levels.append(level)
                agent_cells.append([agent[0], agent[1]])
            stats["n_levels_used"] += 1

        if verbose and stats["n_levels_tried"] % 50 == 0:
            print(
                f"  oracle-bc: {len(target_sets)}/{target_steps} frames, "
                f"{stats['n_levels_tried']} levels ({time.time() - t0:.0f}s)",
                flush=True,
            )

    if not target_sets:
        raise RuntimeError(f"collect_broad_oracle_bc_ras: zero usable frames (stats={stats})")
    _summarize_oracle_stats(stats, target_sets, proxy_sets)
    meta = {"stats": stats, "proxy_sets": proxy_sets, "levels": levels, "agent_cells": agent_cells}
    return np.stack(feats), target_sets, meta


def collect_balanced_oracle_obs_bank(
    driver_policies,
    *,
    use_backgrounds: bool,
    frames_per_driver: int,
    device: str,
    rand_seed: int,
    out_obs,
    out_driver,
    out_soft,
    out_level,
    bc_start_level: int,
    bc_num_levels: int,
    obs_shape=(3, 64, 64),
    n_actions: int = N_ACTIONS,
    max_steps_per_level: int = 64,
    world_dim: int = 5,
    verify_every: int = 8,
    greedy: bool = True,
    log_every_levels: int = 250,
) -> dict:
    """Driver-balanced uint8 observation bank with graph-exact soft targets over tied optimal sets.

    Oracle counterpart of gm_geometry.rewire.ras.collect_balanced_obs_bank: exactly `frames_per_driver` frames
    per driver (raises otherwise) and one shared (N, n_actions) label array. Each driver walks its own disjoint,
    contiguous slice of the BC level range, one env per level."""
    import head_rewire_ras_maze as hrm  # noqa: E402

    n_drivers = len(driver_policies)
    span = bc_num_levels // n_drivers
    if span <= 0:
        raise ValueError(f"bc_num_levels={bc_num_levels} cannot be split across {n_drivers} drivers")

    stats = _new_oracle_stats()
    per_driver: list[int] = []
    per_driver_levels: list[int] = []
    all_sets: list[list[int]] = []
    all_proxy: list[list[int]] = []
    filled = 0
    t0 = time.time()

    for d_idx, driver in enumerate(driver_policies):
        rng = np.random.RandomState(rand_seed + d_idx)
        lo = bc_start_level + d_idx * span
        level = lo
        got = 0
        levels_tried = 0
        while got < frames_per_driver:
            if level >= lo + span:
                raise RuntimeError(
                    f"driver {d_idx} exhausted its level slice [{lo}, {lo + span}) with only "
                    f"{got}/{frames_per_driver} frames; the bank would be driver-imbalanced"
                )
            stats["n_levels_tried"] += 1
            levels_tried += 1
            env = hrm.make_ras_env(num_envs=1, num_levels=1, start_level=level, use_backgrounds=use_backgrounds)
            try:
                base = _unwrap(env)
                if base is None:
                    raise RuntimeError("no procgen layer exposing set_object under this env wrapper stack")
                frames, dropped = oracle_level_frames(
                    env, base, driver,
                    device=device, greedy=greedy, rng=rng,
                    max_frames=frames_per_driver - got,
                    max_steps_per_level=max_steps_per_level,
                    world_dim=world_dim, verify_every=verify_every, stats=stats,
                )
            finally:
                if hasattr(env, "close"):
                    env.close()

            if dropped is not None:
                stats[dropped] += 1
            elif frames:
                for obs_f, a_star, a_proxy, _agent in frames:
                    frame = np.asarray(obs_f)[0]
                    if tuple(frame.shape) != tuple(obs_shape):
                        raise ValueError(f"expected obs {tuple(obs_shape)} from the wrapper stack, got {tuple(frame.shape)}")
                    out_obs[filled] = np.rint(frame * 255.0).astype(np.uint8)
                    out_driver[filled] = d_idx
                    out_level[filled] = level
                    out_soft[filled] = 0.0
                    for a in a_star:
                        out_soft[filled, a] = 1.0 / len(a_star)
                    all_sets.append(a_star)
                    all_proxy.append(a_proxy)
                    filled += 1
                    got += 1
                stats["n_levels_used"] += 1

            level += 1
            if log_every_levels and levels_tried % log_every_levels == 0:
                print(
                    f"  oracle bank: driver {d_idx} {got}/{frames_per_driver} frames, "
                    f"{levels_tried} levels ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        per_driver.append(got)
        per_driver_levels.append(levels_tried)
        print(f"  oracle bank: driver {d_idx} contributed {got} frames over {levels_tried} levels (total {filled})", flush=True)

    if len(set(per_driver)) != 1:
        raise AssertionError(f"driver-balance broken: per-driver frame counts {per_driver}")
    _summarize_oracle_stats(stats, all_sets, all_proxy)
    stats["n_actions"] = int(n_actions)
    stats["levels_tried_per_driver"] = per_driver_levels
    stats["collect_seconds"] = float(time.time() - t0)
    return {"n_frames": filled, "per_driver": per_driver, "oracle_stats": stats}


# --------------------------------------------------------------------------------------
# soft-target BC (labels are sets, so loss and accuracy differ from the hard-label trainers)
# --------------------------------------------------------------------------------------

def _soft_val_split(n: int, groups, rng):
    """Held-out index split, grouped by `groups` (level ids) when given, since consecutive frames are near-duplicates."""
    if groups is None:
        print("[warn] soft-target BC: no `groups` -- frames from one level will land on "
              "both sides of the split; pass the bank's level ids", flush=True)
        idx = rng.permutation(n)
        n_val = max(1, n // 5)
        return idx[:n_val], idx[n_val:]
    g = np.asarray(groups)
    if len(g) != n:
        raise ValueError(f"groups has {len(g)} entries for {n} frames")
    uniq = np.unique(g)
    held = set(rng.permutation(uniq)[: max(1, len(uniq) // 5)].tolist())
    mask = np.array([x in held for x in g])
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def train_soft_bc(
    policy,
    features: np.ndarray,
    soft_targets: np.ndarray,
    tap: str,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    prepare: bool = True,
    epoch_cb=None,
    groups=None,
    seed: int = 0,
) -> list[dict]:
    """Soft-target BC at either tap, the oracle-label counterpart of ras.train_tap_bc.

    Freeze/reinit, ReLU placement and batching match the teacher path; only the loss and set-aware accuracy differ.
    `prepare=False` is required for the random-projection control, as in ras.train_block3_bc."""
    import torch
    import torch.nn.functional as F

    from gm_geometry.rewire import prepare_tap, trainable_parameters

    if tap not in ("block3", "fc"):
        raise ValueError(f"unsupported tap {tap!r} for soft-target BC (expected 'block3' or 'fc')")
    if soft_targets.ndim != 2:
        raise ValueError(f"soft_targets must be (N, n_actions), got shape {soft_targets.shape}")
    if len(soft_targets) != len(features):
        raise ValueError(f"{len(features)} features vs {len(soft_targets)} soft targets")
    if prepare:
        prepare_tap(policy, tap)
    params = [p for p in trainable_parameters(policy, tap) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    rng = np.random.RandomState(seed)
    val_idx, tr_idx = _soft_val_split(len(features), groups, rng)
    fc, head = policy.embedder.fc, policy.fc_policy

    def _logits(batch_idx):
        xb = torch.as_tensor(features[batch_idx], dtype=torch.float32, device=device)
        if tap == "block3":
            return head(F.relu(fc(xb.clamp_min(0.0))))
        return head(xb)

    log: list[dict] = []
    grad_steps = 0
    for epoch in range(epochs):
        perm = rng.permutation(tr_idx)
        total = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            yb = torch.as_tensor(soft_targets[b], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = soft_target_cross_entropy(_logits(b), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            grad_steps += len(b)
            total += float(loss.detach()) * len(b)
        with torch.no_grad():
            vloss_sum, vacc_sum = 0.0, 0.0
            for s in range(0, len(val_idx), batch_size):
                b = val_idx[s : s + batch_size]
                yv = torch.as_tensor(soft_targets[b], dtype=torch.float32, device=device)
                logits_v = _logits(b)
                vloss_sum += float(soft_target_cross_entropy(logits_v, yv)) * len(b)
                vacc_sum += float(set_accuracy(logits_v, yv)) * len(b)
            n_val = max(len(val_idx), 1)
            vloss, vacc = vloss_sum / n_val, vacc_sum / n_val
        # val_acc carries the set-aware number so the shared curve CSV stays populated;
        # val_set_acc names it, and is what tells a reader which definition was used.
        row = {"epoch": epoch, "step": grad_steps, "train_ce": total / max(len(tr_idx), 1),
               "val_ce": vloss, "val_acc": vacc, "val_set_acc": vacc}
        log.append(row)
        print(f"  {tap} oracle-bc epoch {epoch}: train_ce={row['train_ce']:.4f} "
              f"val_ce={vloss:.4f} val_set_acc={vacc:.4f}", flush=True)
        if epoch_cb is not None:
            epoch_cb(epoch, row)
    return log


def train_block3_oracle_bc(
    policy,
    block3_features: np.ndarray,
    target_sets,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    groups=None,
    seed: int = 0,
) -> list[dict]:
    """Set-label wrapper over train_soft_bc at the block3 tap, taking the collector's tied-optimal action lists."""
    return train_soft_bc(
        policy, block3_features, make_soft_targets(target_sets), "block3",
        device=device, epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, groups=groups, seed=seed,
    )
