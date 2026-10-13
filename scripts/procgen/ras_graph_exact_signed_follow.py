#!/usr/bin/env python3
"""Graph-exact first-action sets (wall-respecting BFS) for the Attribute-Maze world-edit signed-follow metric."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from collect_ras_matched_geometry_bank import (  # noqa: E402
    bfs_distances,
    calibrate_pixel_to_grid,
    first_action_set,
    probe_wall_grid,
)

# The toward-set is every first action that reduces wall-respecting BFS distance to the target (more than one
# on a tie). A straight-line bearing guess is not used: in this walled maze it disagrees with the policy's greedy action.

WORLD_DIM = 5
MARGIN = 3  # same wall-probe bound as collect_ras_matched_geometry_bank.py


def _unwrap(env):
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "set_object"):
            return cur
        cur = getattr(cur, "venv", None) or getattr(cur, "env", None)
    return None


_ACTION_DELTAS_CACHE: dict[tuple, dict] = {}
NEEDED_DIRECTIONS = {(1, 0), (-1, 0), (0, 1), (0, -1)}


def _robust_action_deltas(base_api, env, obs0, calib, max_hops: int = 8) -> dict:
    """Action -> grid-direction map, walking along known directions and recalibrating until all four are found.

    `calibrate_action_deltas` makes a single hop from a corner spawn, which can leave directions unresolved."""
    from collect_ras_matched_geometry_bank import calibrate_action_deltas

    deltas = calibrate_action_deltas(base_api, env, obs0, calib)
    snapshot = base_api.get_state()
    hops = 0
    while hops < max_hops and NEEDED_DIRECTIONS - {d for d in deltas.values() if d is not None}:
        known_action = next((a for a, d in deltas.items() if d is not None), None)
        if known_action is None:
            break
        base_api.set_state(snapshot)
        moved_obs = None
        for _ in range(3):
            moved_obs = env.step(np.array([known_action]))[0]
        moved_snapshot = base_api.get_state()
        refined = calibrate_action_deltas(base_api, env, moved_obs, calib)
        for a, d in refined.items():
            if deltas.get(a) is None and d is not None:
                deltas[a] = d
        snapshot = moved_snapshot
        hops += 1
    base_api.set_state(snapshot)
    return deltas


def _get_action_deltas(base_api, env, obs0, calib, cache_key: tuple):
    if cache_key in _ACTION_DELTAS_CACHE:
        return _ACTION_DELTAS_CACHE[cache_key]
    deltas = _robust_action_deltas(base_api, env, obs0, calib)
    found = {d for d in deltas.values() if d is not None}
    if NEEDED_DIRECTIONS <= found:
        # Action semantics are engine-level and constant across levels, so one complete map is reused.
        _ACTION_DELTAS_CACHE[cache_key] = deltas
    return deltas


def make_ras_env(level: int, obj1: str, obj2: str, use_backgrounds: bool):
    """Fresh env at the given level/config, built from the colour-shape procgen fork in external/."""
    for mod in list(sys.modules):
        if mod == "procgen" or mod.startswith("procgen."):
            del sys.modules[mod]
    cs_root = ROOT / "external" / "colour-shape-goal-misgeneralization"
    for sub in ("procgenAISC", "train-procgen"):
        p = str(cs_root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from procgen import ProcgenEnv
    from common.env.procgen_wrappers import VecExtractDictObs

    venv = ProcgenEnv(
        num_envs=1, env_name="maze_colorsobjects_duel", num_levels=1,
        start_level=int(level), distribution_mode="easy",
        world_dim=WORLD_DIM, obj1=obj1, obj2=obj2, use_backgrounds=use_backgrounds,
    )
    return VecExtractDictObs(venv, "rgb")


def graph_exact_first_action_set(
    *, level: int, obj1: str, obj2: str, use_backgrounds: bool,
    agent_xy: tuple[float, float], target_xy: tuple[float, float],
) -> set[int] | None:
    """First actions that reduce wall-respecting BFS distance from the agent's cell to the target's cell, given pixel centroids.

    Returns None when wall probing, pixel-to-grid calibration or cell resolution fails; such frames should be skipped."""
    env = make_ras_env(level, obj1, obj2, use_backgrounds)
    try:
        obs0 = env.reset()
        base_api = _unwrap(env)
        if base_api is None:
            return None
        coord_range = range(-MARGIN, WORLD_DIM + MARGIN)
        walkable = probe_wall_grid(base_api, coord_range, coord_range)
        if len(walkable) < 3:
            return None
        calib, _last_obs = calibrate_pixel_to_grid(base_api, env, walkable, obj1, obj2)
        if calib is None:
            return None
        action_deltas = _get_action_deltas(base_api, env, obs0, calib, (obj1, obj2, use_backgrounds))
        if not (NEEDED_DIRECTIONS <= {d for d in action_deltas.values() if d is not None}):
            # Rare: the level never exposed all four directions; an incomplete map would give a wrong set.
            return None

        def px_to_grid(xy):
            aug = np.array([xy[0], xy[1], 1.0])
            est = aug @ calib
            return (int(round(est[0])), int(round(est[1])))

        agent_cell = px_to_grid(agent_xy)
        target_cell = px_to_grid(target_xy)
        if agent_cell not in walkable or target_cell not in walkable:
            return None
        dist = bfs_distances(walkable, target_cell)
        return first_action_set(dist, agent_cell, action_deltas)
    finally:
        if hasattr(env, "close"):
            env.close()
