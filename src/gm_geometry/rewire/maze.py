"""Maze rewiring: broad graph-exact cheese-BFS behaviour cloning across block2/block3/fc taps."""
# Every collected frame gets a BFS label toward the cheese, not just conflict states; narrow training
# distributions caused motor collapse (see gm_geometry.rewire.eval.motor_collapse_flag).
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gm_geometry.rewire.eval import motor_collapse_flag
from gm_geometry.rewire.taps import freeze_fc, prepare_tap, trainable_parameters

TAPS = ("block2", "block3", "fc")


def add_procgen_scripts(root: Path) -> None:
    p = str(Path(root) / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


def _ladder():
    import run_maze_i_goal_proxy_ladder as m

    return m


def _feature_fn(policy: nn.Module, tap: str):
    """Frozen-part-only feature extractor -- cheap to cache once, reused every epoch."""
    emb = policy.embedder

    @torch.no_grad()
    def fc_tap(x: torch.Tensor) -> torch.Tensor:
        return emb(x)  # full frozen embedder (block1-3 + fc), matches trunk_features

    @torch.no_grad()
    def block3_tap(x: torch.Tensor) -> torch.Tensor:
        h = emb.block1(x)
        h = emb.block2(h)
        h = emb.block3(h)
        return F.relu(h).flatten(1)

    @torch.no_grad()
    def block2_tap(x: torch.Tensor) -> torch.Tensor:
        return emb.block1(x)  # only block1 is frozen; cache its output, nothing else

    return {"fc": fc_tap, "block3": block3_tap, "block2": block2_tap}[tap]


def collect_broad_cheese_bc(
    policy: nn.Module,
    *,
    tap: str,
    device: str,
    target_steps: int,
    start_level: int,
    n_levels_to_try: int,
    max_steps_per_level: int = 200,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll `policy` through training levels, labelling every visited frame with the graph-exact BFS action toward cheese.

    run_arm calls this after prepare_tap, so the driver is the reinitialized policy rather than the native agent;
    the DAgger arm addresses the resulting covariate shift. Levels must be disjoint from the eval pool."""
    m = _ladder()
    feat_fn = _feature_fn(policy, tap)
    feats: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    level = start_level
    tried = 0
    t0 = time.time()
    while len(acts) < target_steps and tried < n_levels_to_try:
        env = m.make_env("maze_aisc", start_level=level, extra={"rand_region": 25})
        obs = env.reset()
        for _t in range(max_steps_per_level):
            chw = np.asarray(obs[0], dtype=np.float32)
            info = m.get_info_maze(env)
            wd = m._iscalar(info, "maze_world_dim")
            ax, ay = m.get_agent_cell(env)
            cx, cy = m._iscalar(info, "cheese_x"), m._iscalar(info, "cheese_y")
            x = torch.as_tensor(chw, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                h = torch.zeros(1, policy.embedder.output_dim, device=device)
                dist, _v, _ = policy(x, h, torch.zeros(1, device=device))
                step_act = int(dist.probs.argmax(-1).item())
            if ax >= 0 and cx >= 0:
                grid = m._read_grid(env, wd)
                a_cheese, _ = m.first_step_action(grid, (ax, ay), (cx, cy))
                if a_cheese is not None:
                    f = feat_fn(x).cpu().numpy()[0]
                    feats.append(f.astype(np.float32))
                    acts.append(np.int64(a_cheese))
                    if len(acts) >= target_steps:
                        break
            obs, _r, done, _i = env.step(np.array([step_act]))
            if bool(done[0]):
                break
        if hasattr(env, "close"):
            env.close()
        level += 1
        tried += 1
        if verbose and tried % 25 == 0:
            print(f"  collected {len(acts)}/{target_steps} frames, {tried} levels tried "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not acts:
        raise RuntimeError("collect_broad_cheese_bc: zero usable frames (BFS to cheese never resolved)")
    return np.stack(feats), np.asarray(acts, dtype=np.int64)


def _bc_epoch_loop(
    forward_fn,
    params: list[nn.Parameter],
    features: np.ndarray,
    actions: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> list[dict]:
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.RandomState(0)
    n_val = max(1, len(features) // 5)
    idx = rng.permutation(len(features))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    log: list[dict] = []
    grad_steps = 0
    for epoch in range(epochs):
        perm = rng.permutation(tr_idx)
        total = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            xb = torch.as_tensor(features[b], device=device)
            yb = torch.as_tensor(actions[b], device=device, dtype=torch.long)
            opt.zero_grad()
            logits = forward_fn(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            grad_steps += len(b)
            total += float(loss) * len(b)
        with torch.no_grad():
            xv = torch.as_tensor(features[val_idx], device=device)
            yv = torch.as_tensor(actions[val_idx], device=device, dtype=torch.long)
            logits_v = forward_fn(xv)
            vloss = float(F.cross_entropy(logits_v, yv).cpu())
            vacc = float((logits_v.argmax(-1) == yv).float().mean().cpu())
        row = {"epoch": epoch, "step": grad_steps,
               "train_ce": total / max(len(tr_idx), 1), "val_ce": vloss, "val_acc": vacc}
        log.append(row)
        print(f"  bc epoch {epoch}: train_ce={row['train_ce']:.4f} val_ce={vloss:.4f} val_acc={vacc:.4f}", flush=True)
    return log


def train_tap(
    policy: nn.Module,
    tap: str,
    features: np.ndarray,
    actions: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    mode: str = "rewire",
) -> list[dict]:
    """Train the fresh downstream suffix on cached frozen-part features for `tap` (fc, block3 or block2).

    mode="random_proj" (block3 only) freezes the fresh fc at its Xavier draw, so a fixed random 2048->256 map
    replaces the learned compression and only the heads train."""
    if mode not in ("rewire", "random_proj"):
        raise ValueError(f"unknown mode {mode!r}, expected 'rewire' or 'random_proj'")
    emb = policy.embedder
    if mode == "random_proj":
        # Only definable here. At fc the analogous square random map is a.s. invertible
        # (a basis scramble, not an information bottleneck); at block2 there is no single
        # learned map to swap out.
        if tap != "block3":
            raise ValueError(f"random_proj control is only defined at tap='block3', got {tap!r}")
        freeze_fc(policy)
        params = list(trainable_parameters(policy, "fc"))
    else:
        params = list(trainable_parameters(policy, tap))
    if tap == "fc":
        def forward_fn(xb):
            return policy.fc_policy(xb)
    elif tap == "block3":
        def forward_fn(xb):
            return policy.fc_policy(F.relu(emb.fc(xb)))
    else:  # "block2": xb is frozen block1 output, forward through the trainable rest
        def forward_fn(xb):
            h = emb.block2(xb)
            h = emb.block3(h)
            h = F.relu(h).flatten(1)
            h = F.relu(emb.fc(h))
            return policy.fc_policy(h)
    return _bc_epoch_loop(forward_fn, params, features, actions,
                           device=device, epochs=epochs, batch_size=batch_size, lr=lr)


def _bfs_distances(grid: np.ndarray, goal: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Flood-fill distance from `goal` to every walkable cell it can reach."""
    from collections import deque

    m = _ladder()
    wd = grid.shape[0]
    gx, gy = goal
    if not (0 <= gx < wd and 0 <= gy < wd):
        return {}
    dist = {goal: 0}
    q = deque([goal])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < wd and 0 <= ny < wd) or (nx, ny) in dist:
                continue
            if m._is_walk(int(grid[nx, ny])):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def cheese_progress_action_set(grid: np.ndarray, agent: tuple[int, int], cheese: tuple[int, int]) -> list[int]:
    """Every cardinal action that strictly reduces BFS distance to cheese from `agent`."""
    m = _ladder()
    dist = _bfs_distances(grid, cheese)
    ax, ay = agent
    if (ax, ay) not in dist:
        return []
    d0 = dist[(ax, ay)]
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = ax + dx, ay + dy
        if (nx, ny) in dist and dist[(nx, ny)] < d0:
            out.append(m.action_from_grid_delta(dx, dy))
    return out


def collect_policy_independent_probe_bank(
    *, device: str, n_target: int, start_level: int, n_levels_to_try: int,
    max_steps_per_level: int = 200, seed: int = 0, verbose: bool = True,
) -> list[dict]:
    """States sampled by a uniform random walk on levels disjoint from BC training, with BFS labels.

    Returns raw frames (float32 CHW) and labels; features are extracted per checkpoint by the caller."""
    m = _ladder()
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    level = start_level
    tried = 0
    t0 = time.time()
    while len(rows) < n_target and tried < n_levels_to_try:
        env = m.make_env("maze_aisc", start_level=level, extra={"rand_region": 25})
        obs = env.reset()
        n_act = env.action_space.n
        for _t in range(max_steps_per_level):
            chw = np.asarray(obs[0], dtype=np.float32)
            info = m.get_info_maze(env)
            wd = m._iscalar(info, "maze_world_dim")
            ax, ay = m.get_agent_cell(env)
            cx, cy = m._iscalar(info, "cheese_x"), m._iscalar(info, "cheese_y")
            if ax >= 0 and cx >= 0:
                grid = m._read_grid(env, wd)
                a_cheese, _ = m.first_step_action(grid, (ax, ay), (cx, cy))
                if a_cheese is not None:
                    action_set = cheese_progress_action_set(grid, (ax, ay), (cx, cy))
                    # Same construction toward the top-right corner proxy, so a wrong
                    # prediction can be classified as corner-progress vs neither.
                    # _reachable_extreme (not _extreme_cell) follows the decision-square convention and
                    # guarantees BFS distances exist.
                    corner = m._reachable_extreme(grid, (ax, ay), "tr")
                    corner_set = (
                        cheese_progress_action_set(grid, (ax, ay), corner) if corner else []
                    )
                    rows.append({
                        "obs": chw, "level": int(level), "a_cheese": int(a_cheese),
                        "action_set": action_set, "corner_set": corner_set,
                    })
                    if len(rows) >= n_target:
                        break
            act = int(rng.randint(0, n_act))  # uniform random -- no policy involved
            obs, _r, done, _i = env.step(np.array([act]))
            if bool(done[0]):
                break
        if hasattr(env, "close"):
            env.close()
        level += 1
        tried += 1
        if verbose and tried % 25 == 0:
            print(f"  probe bank: {len(rows)}/{n_target} states, {tried} levels tried "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not rows:
        raise RuntimeError("collect_policy_independent_probe_bank: zero usable states")
    return rows


def collect_leveled_probe_banks(
    *, device: str, n_levels: int, start_level: int, max_states_per_level: int = 10,
    max_conflict_per_level: int = 5, max_steps_per_level: int = 200, seed: int = 0,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Two banks from one uniform-random walk per level, with the level as the unit.

    Bank A (accuracy): up to `max_states_per_level` states evenly spaced along the walk, the natural mix.
    Bank B (taxonomy): conflict states only (cheese- and corner-progress action sets disjoint), capped per level."""
    m = _ladder()
    rng = np.random.RandomState(seed)
    bank_a: list[dict] = []
    bank_b: list[dict] = []
    level = start_level
    levels_used = 0
    t0 = time.time()
    while levels_used < n_levels:
        env = m.make_env("maze_aisc", start_level=level, extra={"rand_region": 25})
        obs = env.reset()
        n_act = env.action_space.n
        per_level: list[dict] = []
        for _t in range(max_steps_per_level):
            chw = np.asarray(obs[0], dtype=np.float32)
            info = m.get_info_maze(env)
            wd = m._iscalar(info, "maze_world_dim")
            ax, ay = m.get_agent_cell(env)
            cx, cy = m._iscalar(info, "cheese_x"), m._iscalar(info, "cheese_y")
            if ax >= 0 and cx >= 0:
                grid = m._read_grid(env, wd)
                a_cheese, _ = m.first_step_action(grid, (ax, ay), (cx, cy))
                if a_cheese is not None:
                    cheese_set = cheese_progress_action_set(grid, (ax, ay), (cx, cy))
                    corner = m._reachable_extreme(grid, (ax, ay), "tr")
                    corner_set = cheese_progress_action_set(grid, (ax, ay), corner) if corner else []
                    per_level.append({
                        "obs": chw, "level": int(level), "agent": [int(ax), int(ay)],
                        "a_cheese": int(a_cheese), "action_set": cheese_set,
                        "corner_set": corner_set,
                        "is_conflict": bool(cheese_set and corner_set
                                            and not (set(cheese_set) & set(corner_set))),
                    })
            act = int(rng.randint(0, n_act))  # uniform random -- no policy involved
            obs, _r, done, _i = env.step(np.array([act]))
            if bool(done[0]):
                break
        if hasattr(env, "close"):
            env.close()
        if per_level:
            stride = max(1, len(per_level) // max_states_per_level)
            bank_a.extend(per_level[::stride][:max_states_per_level])
            bank_b.extend([r for r in per_level if r["is_conflict"]][:max_conflict_per_level])
            levels_used += 1
        level += 1
        if verbose and levels_used and levels_used % 50 == 0:
            print(f"  banks: {levels_used}/{n_levels} levels, A={len(bank_a)} B={len(bank_b)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not bank_a:
        raise RuntimeError("collect_leveled_probe_banks: zero usable states")
    return bank_a, bank_b


@torch.no_grad()
def _predict(policy: nn.Module, bank: list[dict], device: str) -> list[int]:
    preds = []
    for row in bank:
        x = torch.as_tensor(row["obs"], device=device, dtype=torch.float32).unsqueeze(0)
        h = torch.zeros(1, policy.embedder.output_dim, device=device)
        dist, _v, _ = policy(x, h, torch.zeros(1, device=device))
        preds.append(int(dist.probs.argmax(-1).item()))
    return preds


def _bootstrap_mean_over_levels(per_level: dict[int, float], *, n_boot: int = 10_000,
                                 seed: int = 0) -> dict:
    """CI on the mean of a per-level statistic by resampling LEVELS, not states --
    the level is the independent unit; states within a level are one random walk."""
    vals = np.array([v for v in per_level.values() if v == v], dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n_levels": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(vals, size=vals.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)], "n_levels": int(vals.size)}


def score_leveled(policy: nn.Module, bank_a: list[dict], bank_b: list[dict], device: str,
                   *, keep_records: bool = True) -> dict:
    """Level-aggregated accuracy, error taxonomy and a degeneracy diagnostic.

    Acc_l and B_l = P(pred in corner set | pred not in cheese set) are averaged over levels (bootstrap over
    levels); the dominant-action fraction shows when a corner-error fraction is a property of bank geometry."""
    preds_a = _predict(policy, bank_a, device)
    preds_b = _predict(policy, bank_b, device)

    acc_by_level: dict[int, list[int]] = {}
    for row, pred in zip(bank_a, preds_a):
        acc_by_level.setdefault(row["level"], []).append(int(pred in row["action_set"]))
    acc_l = {lv: float(np.mean(v)) for lv, v in acc_by_level.items()}

    err_by_level: dict[int, list[int]] = {}
    records: list[dict] = []
    for row, pred in zip(bank_b, preds_b):
        correct = pred in row["action_set"]
        if not correct:
            err_by_level.setdefault(row["level"], []).append(int(pred in row["corner_set"]))
        if keep_records:
            records.append({
                "level": row["level"], "agent": row["agent"], "a_cheese": row["a_cheese"],
                "action_set": row["action_set"], "corner_set": row["corner_set"],
                "pred": int(pred), "correct": bool(correct),
                "err_class": None if correct else ("corner" if pred in row["corner_set"] else "neither"),
            })
    b_l = {lv: float(np.mean(v)) for lv, v in err_by_level.items() if v}

    counts = np.bincount(np.asarray(preds_b, dtype=np.int64), minlength=1) if preds_b else np.array([1])
    p = counts / max(counts.sum(), 1)
    nz = p[p > 0]
    return {
        "n_states_a": len(bank_a), "n_levels_a": len(acc_l),
        "n_states_b": len(bank_b), "n_levels_b_with_errors": len(b_l),
        "acc_level_aggregated": _bootstrap_mean_over_levels(acc_l),
        "corner_error_frac_level_aggregated": _bootstrap_mean_over_levels(b_l),
        "dominant_action_frac_conflict": float(p.max()) if preds_b else float("nan"),
        "pred_entropy_conflict": float(-(nz * np.log(nz)).sum()) if preds_b else float("nan"),
        "acc_by_level": acc_l,
        "corner_error_frac_by_level": b_l,
        "records": records,
    }


def score_action_accuracy(policy: nn.Module, tap: str, bank: list[dict], device: str) -> dict:
    """One-step exact-action and set accuracy of the whole `policy` against the policy-independent probe bank.

    `tap` is unused and kept for call-site symmetry."""
    del tap
    exact_hits = 0
    set_hits = 0
    # Error taxonomy: of the predictions that are NOT cheese-progress, how many are
    # corner-progress vs neither? Restricted to states where the two sets are disjoint,
    # since "wrong toward the corner" is undefined where they overlap.
    n_conflict = 0
    conflict_errors = 0
    err_corner = 0
    err_neither = 0
    with torch.no_grad():
        for row in bank:
            x = torch.as_tensor(row["obs"], device=device, dtype=torch.float32).unsqueeze(0)
            h = torch.zeros(1, policy.embedder.output_dim, device=device)
            dist, _v, _ = policy(x, h, torch.zeros(1, device=device))
            pred = int(dist.probs.argmax(-1).item())
            if pred == row["a_cheese"]:
                exact_hits += 1
            cheese_set = row["action_set"]
            if pred in cheese_set:
                set_hits += 1
            corner_set = row.get("corner_set", [])
            if cheese_set and corner_set and not (set(cheese_set) & set(corner_set)):
                n_conflict += 1
                if pred not in cheese_set:
                    conflict_errors += 1
                    if pred in corner_set:
                        err_corner += 1
                    else:
                        err_neither += 1
    n = len(bank)
    return {
        "n": n,
        "exact_acc": exact_hits / n,
        "set_acc": set_hits / n,
        "n_conflict_states": n_conflict,
        "conflict_set_acc": (n_conflict - conflict_errors) / n_conflict if n_conflict else float("nan"),
        "n_conflict_errors": conflict_errors,
        "err_corner_frac": err_corner / conflict_errors if conflict_errors else float("nan"),
        "err_neither_frac": err_neither / conflict_errors if conflict_errors else float("nan"),
    }


def eval_cheese_reach(policy: nn.Module, device: str, *, n_levels: int, start_level: int = 1_000_000) -> dict:
    """Closed-loop held-out cheese-reach plus the movement diagnostics needed for the motor-collapse gate."""
    from validate_langosco_maze_i_200m import eval_split

    val = eval_split(
        policy, env_name="maze", extra={}, n_levels=n_levels, start_level=start_level,
        max_steps=1000, device=device, track_corner=True, action_mode="greedy",
    )
    return {
        "cheese_reach": val["success_rate"],
        "gm_rate": val["gm_rate"],
        "mean_unique_cells": val["mean_unique_cells"],
        "mean_path_length": val["mean_path_length"],
        "mean_frac_no_move": val["mean_frac_no_move"],
        "mean_start_end_disp": val["mean_start_end_disp"],
        "frac_barely_moved": val["frac_barely_moved"],
        "n_levels": n_levels,
        "start_level": start_level,
    }


def sanity_gate(rewired_eval: dict, native_eval: dict, *, min_retention: float = 0.15) -> dict:
    return motor_collapse_flag(
        rewired_eval["mean_unique_cells"], native_eval["mean_unique_cells"], min_retention=min_retention
    )


def collect_onpolicy_cheese_bc(
    policy: nn.Module, *, tap: str, device: str, target_steps: int, start_level: int,
    n_levels_to_try: int, max_steps_per_level: int = 200, verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """DAgger collection: roll out the current controller and label visited states with the BFS cheese action.

    Must be called with a training level range, never the held-out probe/eval pool."""
    m = _ladder()
    feat_fn = _feature_fn(policy, tap)
    feats: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    level = start_level
    tried = 0
    t0 = time.time()
    while len(acts) < target_steps and tried < n_levels_to_try:
        env = m.make_env("maze_aisc", start_level=level, extra={"rand_region": 25})
        obs = env.reset()
        for _t in range(max_steps_per_level):
            chw = np.asarray(obs[0], dtype=np.float32)
            info = m.get_info_maze(env)
            wd = m._iscalar(info, "maze_world_dim")
            ax, ay = m.get_agent_cell(env)
            cx, cy = m._iscalar(info, "cheese_x"), m._iscalar(info, "cheese_y")
            x = torch.as_tensor(chw, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                h = torch.zeros(1, policy.embedder.output_dim, device=device)
                dist, _v, _ = policy(x, h, torch.zeros(1, device=device))
                step_act = int(dist.probs.argmax(-1).item())  # the CURRENT controller drives
            if ax >= 0 and cx >= 0:
                grid = m._read_grid(env, wd)
                a_cheese, _ = m.first_step_action(grid, (ax, ay), (cx, cy))
                if a_cheese is not None:
                    feats.append(feat_fn(x).cpu().numpy()[0].astype(np.float32))
                    acts.append(np.int64(a_cheese))
                    if len(acts) >= target_steps:
                        break
            obs, _r, done, _i = env.step(np.array([step_act]))
            if bool(done[0]):
                break
        if hasattr(env, "close"):
            env.close()
        level += 1
        tried += 1
        if verbose and tried % 50 == 0:
            print(f"    on-policy: {len(acts)}/{target_steps} frames, {tried} levels "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not acts:
        raise RuntimeError("collect_onpolicy_cheese_bc: zero usable frames")
    return np.stack(feats), np.asarray(acts, dtype=np.int64)


def run_dagger_arm(
    *,
    ckpt: Path,
    rewired_ckpt: Path,
    tap: str,
    procgen_root: Path,
    device: str,
    bank_a: list[dict],
    bank_b: list[dict],
    rounds: int,
    base_collect_steps: int,
    onpolicy_collect_steps: int,
    train_start_level: int,
    n_levels_to_try: int,
    bc_epochs: int,
    batch_size: int,
    lr: float,
    eval_n: int,
    eval_start_level: int,
    out_dir: Path,
    control_mode: str = "onpolicy",
) -> dict[str, Any]:
    """Warm-started DAgger arm: from the BC-rewired controller, iteratively aggregate on-policy BFS-labelled data.

    Each round rolls out the controller on training levels, aggregates, fine-tunes the suffix (trunk frozen)
    and evaluates; round 0 is the untouched BC controller. control_mode="base_only" matches dataset size,
    epochs and optimizer steps but resamples the original BC set, isolating the effect of on-policy data."""
    add_procgen_scripts(procgen_root)
    m = _ladder()
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = m.load_policy(ckpt, device)
    prepare_tap(policy, tap)
    policy.load_state_dict(torch.load(rewired_ckpt, map_location=device)["model_state_dict"])
    policy.to(device)

    def _evaluate(rnd: int) -> dict:
        probe = score_leveled(policy, bank_a, bank_b, device, keep_records=False)
        closed = eval_cheese_reach(policy, device, n_levels=eval_n, start_level=eval_start_level)
        row = {
            "round": rnd,
            "acc": probe["acc_level_aggregated"],
            "corner_err": probe["corner_error_frac_level_aggregated"],
            "dominant_action_frac": probe["dominant_action_frac_conflict"],
            "pred_entropy": probe["pred_entropy_conflict"],
            "cheese_reach": closed["cheese_reach"],
            "gm_rate": closed["gm_rate"],
            "mean_unique_cells": closed["mean_unique_cells"],
        }
        print(f"  [round {rnd}] acc={row['acc']['mean']:.3f} "
              f"P(corner|err)={row['corner_err']['mean']:.3f} "
              f"dom={row['dominant_action_frac']:.3f} "
              f"cheese_reach={row['cheese_reach']:.3f} gm_rate={row['gm_rate']:.3f}", flush=True)
        return row

    history = [_evaluate(0)]

    # Base BC set, re-collected at the original recipe so aggregation starts from the
    # same distribution the existing controller was trained on (deterministic: the
    # native policy is greedy and the level range is fixed).
    native = m.load_policy(ckpt, device)
    feats, acts = collect_broad_cheese_bc(
        native, tap=tap, device=device, target_steps=base_collect_steps,
        start_level=train_start_level, n_levels_to_try=n_levels_to_try, verbose=False,
    )
    del native
    print(f"  base BC set: {len(acts)} frames", flush=True)

    n_base = len(acts)
    rng_ctrl = np.random.RandomState(0)
    for rnd in range(1, rounds + 1):
        if control_mode == "onpolicy":
            # On-policy levels are offset per round so rounds don't re-walk identical
            # levels, while staying inside the training range.
            f_new, a_new = collect_onpolicy_cheese_bc(
                policy, tap=tap, device=device, target_steps=onpolicy_collect_steps,
                start_level=train_start_level + 100_000 * rnd, n_levels_to_try=n_levels_to_try,
                verbose=False,
            )
        else:  # "base_only": same dataset growth, same optimizer steps, no new states
            idx = rng_ctrl.randint(0, n_base, size=onpolicy_collect_steps)
            f_new, a_new = feats[:n_base][idx], acts[:n_base][idx]
        feats = np.concatenate([feats, f_new], axis=0)
        acts = np.concatenate([acts, a_new], axis=0)
        onpolicy_frac = 0.0 if control_mode != "onpolicy" else (len(acts) - n_base) / len(acts)
        print(f"  [round {rnd}] +{len(a_new)} frames ({control_mode}) -> {len(acts)} total, "
              f"on-policy fraction={onpolicy_frac:.3f}", flush=True)
        train_tap(policy, tap, feats, acts, device=device, epochs=bc_epochs,
                   batch_size=batch_size, lr=lr)
        row = _evaluate(rnd)
        row["n_train_frames"] = int(len(acts))
        row["onpolicy_frac"] = float(onpolicy_frac)
        row["control_mode"] = control_mode
        history.append(row)
        torch.save({"model_state_dict": policy.state_dict(), "tap": tap, "round": rnd},
                    out_dir / f"{control_mode}_round{rnd}.pth")

    summary = {"tap": tap, "ckpt": str(ckpt), "rewired_ckpt": str(rewired_ckpt),
               "rounds": rounds, "control_mode": control_mode, "n_base_frames": int(n_base),
               "history": history, "elapsed_s": time.time() - t0}
    (out_dir / "dagger_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_arm(
    *,
    ckpt: Path,
    tap: str,
    procgen_root: Path,
    device: str,
    collect_steps: int,
    train_start_level: int,
    n_levels_to_try: int,
    bc_epochs: int,
    batch_size: int,
    lr: float,
    eval_n: int,
    eval_start_level: int,
    out_dir: Path,
    mode: str = "rewire",
    init_seed: int = 0,
) -> dict[str, Any]:
    """One (checkpoint, tap, mode) arm end to end: prepare_tap, collect, train, eval rewired and native, gate.

    `init_seed` is drawn before prepare_tap so "rewire" and "random_proj" arms on one checkpoint share
    fresh weights and therefore an identical BC bank."""
    add_procgen_scripts(procgen_root)
    m = _ladder()
    t0 = time.time()

    native_policy = m.load_policy(ckpt, device)
    native_eval = eval_cheese_reach(native_policy, device, n_levels=eval_n, start_level=eval_start_level)
    del native_policy

    policy = m.load_policy(ckpt, device)
    torch.manual_seed(init_seed)
    prepare_tap(policy, tap)
    policy.to(device)

    features, actions = collect_broad_cheese_bc(
        policy, tap=tap, device=device, target_steps=collect_steps,
        start_level=train_start_level, n_levels_to_try=n_levels_to_try,
    )
    bc_log = train_tap(policy, tap, features, actions, device=device,
                        epochs=bc_epochs, batch_size=batch_size, lr=lr, mode=mode)

    rewired_eval = eval_cheese_reach(policy, device, n_levels=eval_n, start_level=eval_start_level)
    gate = sanity_gate(rewired_eval, native_eval)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": policy.state_dict(), "tap": tap, "mode": mode, "ckpt": str(ckpt)},
               out_dir / "rewire_final.pth")
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    summary = {
        "tap": tap, "mode": mode, "init_seed": init_seed, "ckpt": str(ckpt),
        "n_frames": int(len(actions)), "n_trainable_params": int(n_trainable),
        "bc_log": bc_log, "native_eval": native_eval, "rewired_eval": rewired_eval,
        "sanity_gate": gate, "elapsed_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tap}/{mode}] {ckpt.name}: cheese_reach {native_eval['cheese_reach']:.3f} -> "
          f"{rewired_eval['cheese_reach']:.3f} | gate retention={gate['retention']:.3f} "
          f"collapsed={gate['collapsed']}", flush=True)
    return summary


# ── aligned protocol ──────────────────────────────────────────────────────────
# One driver-balanced bank of raw observations is shared by every arm and both taps, and records
# both label sources (teacher and A*) for the same states, so the label is the only variable.

ALIGNED_MAZE_MARKER = "ALIGNED_MAZE_V1_shared_balanced_bank_dual_label"


def collect_balanced_obs_bank_maze(
    drivers: list[tuple[str, nn.Module]],
    teacher: nn.Module,
    *,
    device: str,
    frames_per_driver: int,
    start_level: int,
    n_levels_to_try: int,
    max_steps_per_level: int = 200,
) -> dict[str, Any]:
    """One bank of raw uint8 observations, equal frames from every driver.

    Each retained frame carries the teacher's greedy action AND the graph-exact optimal
    action set, both computed against the same env state, so the two supervision sources
    differ only in the label. Frames whose BFS does not resolve are dropped so that the
    two label sources see an identical state distribution.
    """
    m = _ladder()
    obs_out: list[np.ndarray] = []
    a_teacher: list[int] = []
    a_star: list[np.ndarray] = []
    driver_id: list[int] = []
    n_act = int(teacher.fc_policy.out_features)

    for d_idx, (d_name, driver) in enumerate(drivers):
        got, level, tried, t0 = 0, start_level + d_idx * n_levels_to_try, 0, time.time()
        while got < frames_per_driver and tried < n_levels_to_try:
            env = m.make_env("maze_aisc", start_level=level, extra={"rand_region": 25})
            obs = env.reset()
            for _t in range(max_steps_per_level):
                chw = np.asarray(obs[0], dtype=np.float32)
                info = m.get_info_maze(env)
                wd = m._iscalar(info, "maze_world_dim")
                ax, ay = m.get_agent_cell(env)
                cx, cy = m._iscalar(info, "cheese_x"), m._iscalar(info, "cheese_y")
                x = torch.as_tensor(chw, device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    h = torch.zeros(1, driver.embedder.output_dim, device=device)
                    zero = torch.zeros(1, device=device)
                    step_act = int(driver(x, h, zero)[0].probs.argmax(-1).item())
                    t_act = int(teacher(x, h, zero)[0].probs.argmax(-1).item())
                if ax >= 0 and cx >= 0:
                    grid = m._read_grid(env, wd)
                    a_canon, _ = m.first_step_action(grid, (ax, ay), (cx, cy))
                    if a_canon is not None:
                        aset = np.zeros(n_act, dtype=np.float32)
                        for a in cheese_progress_action_set(grid, (ax, ay), (cx, cy)) or [a_canon]:
                            aset[a] = 1.0
                        # uint8 round-trip is lossless: the wrapper stack scales by 1/255.
                        obs_out.append((chw * 255.0).round().astype(np.uint8))
                        a_teacher.append(t_act)
                        a_star.append(aset)
                        driver_id.append(d_idx)
                        got += 1
                        if got >= frames_per_driver:
                            break
                obs, _r, done, _i = env.step(np.array([step_act]))
                if bool(done[0]):
                    break
            if hasattr(env, "close"):
                env.close()
            level += 1
            tried += 1
            if tried % 50 == 0:
                print(f"  [{d_name}] {got}/{frames_per_driver} frames, {tried} levels "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if got < frames_per_driver:
            raise RuntimeError(f"driver {d_name}: only {got}/{frames_per_driver} frames in "
                               f"{tried} levels -- raise --n-levels-to-try")

    return {
        "marker": ALIGNED_MAZE_MARKER,
        "obs_u8": np.stack(obs_out),
        "a_teacher": np.asarray(a_teacher, dtype=np.int64),
        "a_star": np.stack(a_star),
        "driver_id": np.asarray(driver_id, dtype=np.int64),
        "meta": {
            "marker": ALIGNED_MAZE_MARKER,
            "n_frames": len(a_teacher),
            "frames_per_driver": frames_per_driver,
            "drivers": [d for d, _ in drivers],
            "start_level": start_level,
            "n_actions": n_act,
            "label_sources": ["teacher_greedy", "a_star_set"],
            "state_filter": "BFS-to-cheese resolves (identical for both label sources)",
        },
    }


def features_from_obs(policy: nn.Module, obs_u8: np.ndarray, tap: str, *, device: str,
                      batch: int = 512) -> np.ndarray:
    """Frozen-part features for `tap`, computed from the shared uint8 observation bank."""
    fn = _feature_fn(policy, tap)
    out = []
    for s in range(0, obs_u8.shape[0], batch):
        x = torch.as_tensor(obs_u8[s : s + batch], device=device, dtype=torch.float32) / 255.0
        out.append(fn(x).cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def soft_set_cross_entropy(logits: torch.Tensor, target_set: torch.Tensor) -> torch.Tensor:
    """CE against the uniform distribution over an optimal-action SET (A* labels)."""
    p = target_set / target_set.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(p * torch.log_softmax(logits, dim=-1)).sum(dim=1).mean()
