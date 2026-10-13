"""Attribute Maze oracle labels whose target BFS excludes the proxy (gem) cell, emitted alongside the original labels."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

N_ACTIONS = 15
NOOP_ACTION = 4
WALKABLE_RC = 0
GOAL1_WHICH = 1  # obj1 = red_line_diag, +10: intended goal
GOAL2_WHICH = 2  # obj2 = yellow_gem, -10: proxy; touching it ends the episode
SPAWN_CELL = (0, 0)


def action_grid_deltas() -> dict[int, tuple[int, int]]:
    """Exact (dx, dy) per action, matching ras_oracle.action_grid_deltas."""
    out: dict[int, tuple[int, int]] = {}
    for a in range(N_ACTIONS):
        move = 4 if a >= 9 else a % 9
        vx, vy = move // 3 - 1, move % 3 - 1
        out[a] = (vx, 0) if vx != 0 else (0, vy)
    return out


ACTION_DELTAS = action_grid_deltas()


def bfs_distances(walkable, target):
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


def successor_cell(cell, action, walkable):
    """Physical successor: the gem is not a wall, so the agent CAN step onto it (and the episode ends there)."""
    dx, dy = ACTION_DELTAS[action]
    nxt = (cell[0] + dx, cell[1] + dy)
    return nxt if nxt in walkable else cell


def optimal_action_set(dist_from_target, cell, walkable):
    """Actions whose realized successor strictly decreases distance-to-target.

    With the gem-excluded distance map the gem cell has no entry, so a move onto it never counts as progress.
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


def intended_distances_buggy(walkable, goal_intended, goal_proxy):
    """Original map: BFS to the line over a walkable set that still contains the gem."""
    return bfs_distances(walkable, goal_intended)


def intended_distances_fixed(walkable, goal_intended, goal_proxy):
    """Distance to the line over walkable cells minus the gem, so no shortest path passes over the -10 terminal."""
    w = set(walkable)
    if goal_proxy is not None:
        w.discard(tuple(goal_proxy))
    return bfs_distances(w, goal_intended)


# ------------------------------------------------------------------ env access

def _find_vecnormalize(env):
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "ret_rms"):
            return cur
        cur = getattr(cur, "venv", None) or getattr(cur, "env", None)
    return None


def unwrap(env):
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "set_object"):
            return cur
        cur = getattr(cur, "venv", None) or getattr(cur, "env", None)
    return None


def read_object_cells(base):
    g1 = base.get_object_pos([GOAL1_WHICH])[0]
    g2 = base.get_object_pos([GOAL2_WHICH])[0]
    return (tuple(g1) if g1 is not None else None, tuple(g2) if g2 is not None else None)


def probe_walkable_cells(base, world_dim, margin=2):
    """Walkable set incl. both object cells (objects cleared first); destructive, caller restores a snapshot."""
    base.set_object([GOAL1_WHICH], [0], [0], [0])
    base.set_object([GOAL2_WHICH], [0], [0], [0])
    walkable = set()
    for x in range(-margin, world_dim + margin):
        for y in range(-margin, world_dim + margin):
            if base.set_object([GOAL1_WHICH], [x], [y], [1])[0] == WALKABLE_RC:
                walkable.add((x, y))
    return walkable


def probe_agent_cell(env, base, candidates, snapshot=None):
    snap = snapshot if snapshot is not None else base.get_state()
    vecnorm = _find_vecnormalize(env)
    saved_ret = None if vecnorm is None else np.array(vecnorm.ret, copy=True)
    hit = None
    for x, y in candidates:
        base.set_state(snap)
        if base.set_object([GOAL1_WHICH], [x], [y], [1])[0] != WALKABLE_RC:
            continue
        _obs, _rew, _done, info = env.step(np.array([NOOP_ACTION]))
        if float(info[0].get("env_reward", 0.0)) > 0.0:
            hit = (x, y)
            break
    base.set_state(snap)
    if vecnorm is not None and saved_ret is not None:
        vecnorm.ret = saved_ret
    return hit


def _greedy_action(policy, obs, device, greedy, rng):
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


def level_geometry(env, base, world_dim):
    """(line, gem, walkable) for the level the env was just reset into; restores the start snapshot."""
    env.reset()
    snap0 = base.get_state()
    goal_intended, goal_proxy = read_object_cells(base)
    walkable = probe_walkable_cells(base, world_dim)
    base.set_state(snap0)
    return goal_intended, goal_proxy, walkable


def oracle_level_frames_dual(env, base, rollout_policy, *, device, greedy, rng, max_frames,
                             max_steps_per_level, world_dim, verify_every, stats):
    """ras_oracle.oracle_level_frames with the gem-excluded label added to every frame.

    Frame inclusion, rollout, probes and drop rules are keyed on the original label, so the frame stream
    matches the original bank. Each frame is (obs, a_star_buggy, a_star_fixed, a_proxy, agent).
    """
    obs = env.reset()
    snap0 = base.get_state()
    goal_intended, goal_proxy = read_object_cells(base)
    walkable = probe_walkable_cells(base, world_dim)
    base.set_state(snap0)
    info_lv = {"line": goal_intended, "gem": goal_proxy}

    if any(max(abs(c[0]), abs(c[1])) >= world_dim for c in walkable):
        raise RuntimeError(f"walkable cell outside world_dim={world_dim}")
    if goal_intended is None or goal_intended not in walkable or SPAWN_CELL not in walkable:
        return [], "n_levels_dropped_geometry", info_lv

    dist_buggy = intended_distances_buggy(walkable, goal_intended, goal_proxy)
    dist_fixed = intended_distances_fixed(walkable, goal_intended, goal_proxy)
    dist_proxy = bfs_distances(walkable, goal_proxy) if goal_proxy in walkable else {}
    info_lv["spawn_line_reachable_gem_free"] = SPAWN_CELL in dist_fixed
    info_lv["d_line_buggy"] = dist_buggy.get(SPAWN_CELL, -1)
    info_lv["d_line_fixed"] = dist_fixed.get(SPAWN_CELL, -1)
    candidates = sorted(walkable)

    agent = probe_agent_cell(env, base, [SPAWN_CELL], snapshot=snap0)
    if agent is None:
        agent = probe_agent_cell(env, base, candidates, snapshot=snap0)
    if agent is None:
        return [], "n_levels_dropped_geometry", info_lv

    frames = []
    for t in range(max_steps_per_level):
        if t > 0 and verify_every > 0 and t % verify_every == 0:
            if probe_agent_cell(env, base, candidates) != agent:
                return frames, "n_levels_dropped_desync", info_lv

        a_star = optimal_action_set(dist_buggy, agent, walkable)
        if not a_star:
            stats["n_frames_unresolved"] += 1
        else:
            a_fixed = optimal_action_set(dist_fixed, agent, walkable)
            a_proxy = optimal_action_set(dist_proxy, agent, walkable) if dist_proxy else []
            frames.append((np.asarray(obs, dtype=np.float32), a_star, a_fixed, a_proxy, agent))
            if len(frames) >= max_frames:
                return frames, None, info_lv

        act = _greedy_action(rollout_policy, obs, device, greedy, rng)
        obs, _rew, done, info = env.step(np.array([act]))
        agent = successor_cell(agent, act, walkable)
        if bool(done[0]):
            env_reward = float(info[0].get("env_reward", 0.0))
            if env_reward > 0 and agent != goal_intended:
                return frames, "n_levels_dropped_terminal_mismatch", info_lv
            if env_reward < 0 and agent != goal_proxy:
                return frames, "n_levels_dropped_terminal_mismatch", info_lv
            if env_reward != 0:
                stats["n_terminal_checks_passed"] += 1
            break
    return frames, None, info_lv


def new_stats():
    return {"n_levels_tried": 0, "n_levels_used": 0, "n_levels_dropped_geometry": 0,
            "n_levels_dropped_desync": 0, "n_levels_dropped_terminal_mismatch": 0,
            "n_frames_unresolved": 0, "n_terminal_checks_passed": 0}


def collect_driver_dual(driver, d_idx, *, n_drivers, use_backgrounds, frames_per_driver, device, rand_seed,
                        out_obs, out_soft_buggy, out_soft_fixed, out_level, out_agent,
                        bc_start_level=0, bc_num_levels=100_000, obs_shape=(3, 64, 64), max_steps_per_level=64,
                        world_dim=5, verify_every=8, greedy=True, log_every_levels=250):
    """One driver's block of ras_oracle.collect_balanced_oracle_obs_bank, with both label sets.

    Level slice, rng seed (rand_seed + d_idx), quota and one-env-per-level loop match the sequential
    collector, so the six drivers can run in separate processes and still yield the same bank.
    """
    import head_rewire_ras_maze as hrm  # noqa: E402

    span = bc_num_levels // n_drivers
    rng = np.random.RandomState(rand_seed + d_idx)
    lo = bc_start_level + d_idx * span
    level, got = lo, 0
    stats = new_stats()
    levels = []
    t0 = time.time()
    while got < frames_per_driver:
        if level >= lo + span:
            raise RuntimeError(f"driver {d_idx} exhausted its level slice with {got}/{frames_per_driver}")
        stats["n_levels_tried"] += 1
        env = hrm.make_ras_env(num_envs=1, num_levels=1, start_level=level, use_backgrounds=use_backgrounds)
        try:
            base = unwrap(env)
            frames, dropped, info_lv = oracle_level_frames_dual(
                env, base, driver, device=device, greedy=greedy, rng=rng,
                max_frames=frames_per_driver - got, max_steps_per_level=max_steps_per_level,
                world_dim=world_dim, verify_every=verify_every, stats=stats)
        finally:
            if hasattr(env, "close"):
                env.close()
        rec = {"level": level, "driver": d_idx, "dropped": dropped, "n_frames": 0,
               **{k: (list(v) if isinstance(v, tuple) else v) for k, v in info_lv.items()}}
        if dropped is not None:
            stats[dropped] += 1
        elif frames:
            for obs_f, a_star, a_fixed, _a_proxy, agent in frames:
                frame = np.asarray(obs_f)[0]
                if tuple(frame.shape) != tuple(obs_shape):
                    raise ValueError(f"unexpected obs shape {frame.shape}")
                out_obs[got] = np.rint(frame * 255.0).astype(np.uint8)
                out_level[got] = level
                out_agent[got] = agent
                out_soft_buggy[got] = 0.0
                for a in a_star:
                    out_soft_buggy[got, a] = 1.0 / len(a_star)
                out_soft_fixed[got] = 0.0
                for a in a_fixed:
                    out_soft_fixed[got, a] = 1.0 / len(a_fixed)
                got += 1
            rec["n_frames"] = len(frames)
            stats["n_levels_used"] += 1
        levels.append(rec)
        level += 1
        if log_every_levels and stats["n_levels_tried"] % log_every_levels == 0:
            print(f"  driver {d_idx}: {got}/{frames_per_driver} frames, {stats['n_levels_tried']} levels "
                  f"({time.time() - t0:.0f}s)", flush=True)
    stats["collect_seconds"] = float(time.time() - t0)
    return stats, levels
