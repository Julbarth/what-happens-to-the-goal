#!/usr/bin/env python3
"""Maze I (Langosco cheese) goal/proxy ladder: location-GM counterpart of the Attribute Maze rung 1-5 ladder, library and CLI."""

from __future__ import annotations

# The cheese (goal) is a relocatable object while the corner (proxy) is a positional prior, not an
# object, so the ladder is asymmetric. Causal rungs 4/5 (cheese relocation, corner wall-geometry lever)
# run only with --causal.

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

def _first_existing(*cands: Path) -> Path:
    for p in cands:
        if p.exists():
            return p
    return cands[-1]


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = _first_existing(
    ROOT / "vendored" / "train-procgen-pytorch", ROOT / "external" / "train-procgen-pytorch"
)
PROCGEN_AISC = _first_existing(
    ROOT / "vendored" / "procgenAISC", ROOT / "external" / "procgenAISC"
)
for p in (
    ROOT / "src",
    PROCGEN_AISC,
    TRAIN_ROOT,
    ROOT / "scripts" / "procgen",
):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)

from common.env.procgen_wrappers import (  # noqa: E402
    ScaledFloatFrame,
    TransposeFrame,
    VecExtractDictObs,
    VecNormalize,
)
from procgen import ProcgenEnv  # noqa: E402

from validate_langosco_maze_i_200m import (  # noqa: E402
    _chw_to_hwc01,
    in_top_right,
    motion_agent_xy,
)
from gm_geometry.integrations.colour_shape import load_impala_policy_from_env  # noqa: E402
from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc  # noqa: E402
from gm_geometry.procgen import unwrap_base as _unwrap_base  # noqa: E402
from gm_geometry.utils import ci95 as _ci95, device as _default_device  # noqa: E402


def make_env(env_name, *, start_level, extra=None, num_threads=2):
    venv = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=1,
                      start_level=start_level, distribution_mode="hard",
                      num_threads=num_threads, **(extra or {}))
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False, ret=False)
    venv = TransposeFrame(venv)
    return ScaledFloatFrame(venv)


def load_policy(checkpoint, device, *, random_init=False):
    return load_impala_policy_from_env(
        checkpoint, "maze", device, random_init=random_init, train_root=TRAIN_ROOT
    )


def cheese_xy_bright(frame_hwc):
    """Cheese as a small bright orange blob (sand walls confuse naive masks)."""
    rgb = frame_hwc.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 0.85) & (g > 0.45) & (b < 0.55) & ((r + g + b) > 2.15)
    if mask.sum() < 1:
        s = r + 0.8 * g - 1.5 * b
        flat = np.argpartition(s.ravel(), -5)[-5:]
        ys, xs = np.unravel_index(flat, s.shape)
        keep = s[ys, xs] > 1.0
        if keep.sum() == 0:
            return None
        xs, ys = xs[keep], ys[keep]
        return float(xs.mean()), float(ys.mean())
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())

LOCI = ["block2", "block3", "fc"]
TRUNK = ["block2", "block3"]
# procgen cardinal action indices: LEFT=1, DOWN=3, UP=5, RIGHT=7 (image y grows downward)
A_LEFT, A_DOWN, A_UP, A_RIGHT = 1, 3, 5, 7


def action_toward(agent_xy, target_xy):
    """Cardinal action index that reduces image-space distance to target."""
    if agent_xy is None or target_xy is None:
        return None
    dx = target_xy[0] - agent_xy[0]
    dy = target_xy[1] - agent_xy[1]  # image: +dy is downward
    if abs(dx) >= abs(dy):
        return A_RIGHT if dx > 0 else A_LEFT
    return A_DOWN if dy > 0 else A_UP


def action_from_grid_delta(dx, dy):
    """Map a maze-grid step to a discrete action. World +y is up (→ A_UP)."""
    if abs(dx) >= abs(dy):
        return A_RIGHT if dx > 0 else A_LEFT
    return A_UP if dy > 0 else A_DOWN


def _reachable_extreme(grid, start, which="tr"):
    """Extreme cell among cells reachable from start (walkable only)."""
    reach = _reachable(grid, start)
    if not reach:
        return None
    if which == "tr":
        return max(reach, key=lambda c: (c[0] + c[1], c[0], c[1]))
    if which == "bl":
        return min(reach, key=lambda c: (c[0] + c[1], c[0], c[1]))
    raise ValueError(which)


def first_step_action(grid, start, goal):
    """First action on the shortest grid path from start to goal, or None if unreachable."""
    path = _bfs_path(grid, start, goal, allow_walls=False)
    if path is None or len(path) < 2:
        return None, None
    x0, y0 = path[0]
    x1, y1 = path[1]
    return action_from_grid_delta(x1 - x0, y1 - y0), len(path) - 1


# ----------------------------- env set_cheese helpers (causal rungs) -----------------------------
def _unwrap(env, attr):
    return _unwrap_base(env, attr, on_missing="none")


def set_cheese(env, x, y, present):
    return int(_unwrap(env, "set_cheese").set_cheese([x], [y], [present])[0])


def get_state(env):
    return _unwrap(env, "get_state").get_state()


def set_state(env, s):
    _unwrap(env, "set_state").set_state(s)


def get_info_maze(env):
    base = _unwrap(env, "get_info")
    infos = base.get_info()
    return infos[0] if isinstance(infos, (list, tuple)) else infos


def _iscalar(info, key):
    return int(np.asarray(info[key]).reshape(-1)[0])


def obs_after_edit(env):
    base = _unwrap(env, "observe")
    _r, o, _f = base.observe()
    rgb = o["rgb"] if isinstance(o, dict) else o
    rgb = np.asarray(rgb)
    rgb = rgb[0] if rgb.ndim == 4 else rgb
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    if chw.max() > 1.5:
        chw = chw / 255.0
    return chw


# Maze grid edit API (wall-geometry lever). Object ids from procgen object-ids.h.
WALL_OBJ, SPACE_OBJ, GOAL_OBJ = 51, 100, 2


def get_maze_obj(env, x, y):
    return int(_unwrap(env, "get_maze_obj").get_maze_obj([x], [y])[0])


def set_maze_obj(env, x, y, obj):
    return int(_unwrap(env, "set_maze_obj").set_maze_obj([x], [y], [obj])[0])


def get_agent_cell(env):
    return tuple(_unwrap(env, "get_agent_cell").get_agent_cell()[0])


def _read_grid(env, wd):
    return np.array([[get_maze_obj(env, x, y) for y in range(wd)] for x in range(wd)], dtype=np.int32)


def _is_walk(t):
    return t == SPACE_OBJ or t == GOAL_OBJ


def _bfs_path(grid, start, goal, allow_walls=False):
    """Shortest path. If allow_walls, WALL cells are traversable (for planning a carve)."""
    wd = grid.shape[0]
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < wd and 0 <= sy < wd and 0 <= gx < wd and 0 <= gy < wd):
        return None
    from collections import deque
    q = deque([(sx, sy)])
    prev = {(sx, sy): None}
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < wd and 0 <= ny < wd) or (nx, ny) in prev:
                continue
            t = int(grid[nx, ny])
            if _is_walk(t) or (allow_walls and t == WALL_OBJ):
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))
    if (gx, gy) not in prev:
        return None
    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _reachable(grid, start):
    from collections import deque
    wd = grid.shape[0]
    q = deque([start])
    seen = {start}
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < wd and 0 <= ny < wd) or (nx, ny) in seen:
                continue
            if _is_walk(int(grid[nx, ny])):
                seen.add((nx, ny))
                q.append((nx, ny))
    return seen


def _extreme_cell(grid, mode="tr"):
    """Extreme corner among interior cells (skip outer border). tr = max x+y, bl = min x+y."""
    wd = grid.shape[0]
    best, score = None, None
    for x in range(1, wd - 1):
        for y in range(1, wd - 1):
            t = int(grid[x, y])
            if t not in (WALL_OBJ, SPACE_OBJ, GOAL_OBJ):
                continue
            s = x + y
            if mode == "tr":
                if best is None or s > score or (s == score and (x, y) > best):
                    best, score = (x, y), s
            else:
                if best is None or s < score or (s == score and (x, y) < best):
                    best, score = (x, y), s
    return best


def carve_path_to(env, target, *, max_walls=40):
    """Open WALL cells along a shortest path from the agent to target. Returns #walls opened."""
    info = get_info_maze(env)
    wd = _iscalar(info, "maze_world_dim")
    ax, ay = get_agent_cell(env)
    if ax < 0:
        return 0, []
    grid = _read_grid(env, wd)
    # If target is a wall, keep it as the destination (will be opened).
    path = _bfs_path(grid, (ax, ay), target, allow_walls=True)
    if path is None:
        return 0, []
    opened = []
    for (x, y) in path:
        if int(grid[x, y]) == WALL_OBJ:
            if set_maze_obj(env, x, y, SPACE_OBJ) == 0:
                opened.append((x, y))
                grid[x, y] = SPACE_OBJ
            if len(opened) >= max_walls:
                break
    return len(opened), opened


def _local_open_count(grid, start, r=3):
    """Walkable cells in a Chebyshev ball around start (disruption diagnostic)."""
    wd = grid.shape[0]
    x0, y0 = start
    n = 0
    for x in range(max(1, x0 - r), min(wd - 1, x0 + r + 1)):
        for y in range(max(1, y0 - r), min(wd - 1, y0 + r + 1)):
            if _is_walk(int(grid[x, y])):
                n += 1
    return n


def _start_degree(grid, start):
    wd = grid.shape[0]
    x, y = start
    d = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < wd and 0 <= ny < wd and _is_walk(int(grid[nx, ny])):
            d += 1
    return d


def capture_corner_geometry(env, lad, a_corner):
    """Corner geometry lever: carve walls so the extreme top-right cell becomes reachable, vs a null carve.

    The null opens the same number of walls toward the bottom-left. Restores state. a_corner refers to the
    reachable TR on the pre-edit maze, so the post-edit first step and local disruption stats are stored too.
    """
    st = get_state(env)
    info = get_info_maze(env)
    wd = _iscalar(info, "maze_world_dim")
    ax, ay = get_agent_cell(env)
    grid0 = _read_grid(env, wd)
    start = (ax, ay)
    reach0 = _reachable(grid0, start) if ax >= 0 else set()
    tr = _extreme_cell(grid0, "tr")
    bl = _extreme_cell(grid0, "bl")
    if tr is None or bl is None or ax < 0:
        set_state(env, st)
        return None

    loc0 = _local_open_count(grid0, start)
    deg0 = _start_degree(grid0, start)

    # --- lever: open path to extreme TR ---
    n_tr, opened_tr = carve_path_to(env, tr)
    grid_tr = _read_grid(env, wd)
    reach_tr = _reachable(grid_tr, start)
    tr_newly = tr in reach_tr and tr not in reach0
    a_corner_post, _ = first_step_action(grid_tr, start, tr)
    ob_tr = obs_after_edit(env)
    run_tr = lad.run(ob_tr, capture=LOCI)
    set_state(env, st)

    # --- null: open same # walls toward BL ---
    n_bl, opened_bl = carve_path_to(env, bl, max_walls=max(1, n_tr))
    # If BL path opened fewer walls, open additional BL-adjacent walls not on TR geodesic
    if n_bl < n_tr:
        grid_bl = _read_grid(env, wd)
        extras = []
        for x in range(1, wd - 1):
            for y in range(1, wd - 1):
                if int(grid_bl[x, y]) != WALL_OBJ:
                    continue
                if (x, y) in opened_tr:
                    continue
                # prefer low x+y (BL side)
                extras.append((x + y, x, y))
        extras.sort()
        for _, x, y in extras:
            if n_bl >= n_tr:
                break
            if set_maze_obj(env, x, y, SPACE_OBJ) == 0:
                opened_bl.append((x, y))
                n_bl += 1
    grid_bl = _read_grid(env, wd)
    reach_bl = _reachable(grid_bl, start)
    ob_bl = obs_after_edit(env)
    run_bl = lad.run(ob_bl, capture=LOCI)
    set_state(env, st)

    return {
        "n_walls_tr": int(n_tr),
        "n_walls_null": int(n_bl),
        "tr_target": [int(tr[0]), int(tr[1])],
        "bl_target": [int(bl[0]), int(bl[1])],
        "tr_newly_reachable": bool(tr_newly),
        "reach0": int(len(reach0)),
        "reach_tr": int(len(reach_tr)),
        "reach_bl": int(len(reach_bl)),
        "d_reach_tr": int(len(reach_tr) - len(reach0)),
        "d_reach_bl": int(len(reach_bl) - len(reach0)),
        "d_local_tr": int(_local_open_count(grid_tr, start) - loc0),
        "d_local_bl": int(_local_open_count(grid_bl, start) - loc0),
        "d_degree_tr": int(_start_degree(grid_tr, start) - deg0),
        "d_degree_bl": int(_start_degree(grid_bl, start) - deg0),
        "opened_caps": {k: run_tr["caps"][k][0] for k in LOCI},
        "null_caps": {k: run_bl["caps"][k][0] for k in LOCI},
        "opened_probs": run_tr["probs"][0].astype(np.float32),
        "null_probs": run_bl["probs"][0].astype(np.float32),
        "opened_value": float(run_tr["value"][0]),
        "null_value": float(run_bl["value"][0]),
        "a_corner": int(a_corner),  # pre-edit (reachable-TR) action used at the decision square
        "a_corner_post": None if a_corner_post is None else int(a_corner_post),
        "a_corner_flipped": bool(a_corner_post is not None and int(a_corner_post) != int(a_corner)),
    }


# ----------------------------- policy wrapper w/ hooks -----------------------------
class Ladder:
    def __init__(self, policy, device):
        self.policy = policy.to(device).eval()
        self.device = device
        self.emb = policy.embedder
        self.mods = {"block2": self.emb.block2, "block3": self.emb.block3, "fc": self.emb.fc}
        self.cap: dict[str, torch.Tensor] = {}
        self.patch = None  # {locus, donor(BxC..), channels(np.ndarray|None), scale}
        self.handles = [m.register_forward_hook(self._hook(n)) for n, m in self.mods.items()]

    def _hook(self, name):
        def fn(_m, _i, out):
            self.cap[name] = out
            p = self.patch
            if p is not None and p["locus"] == name:
                donor = p["donor"].to(out.device)
                if donor.shape[0] == 1 and out.shape[0] > 1:
                    donor = donor.expand_as(out)
                ch = p["channels"]
                s = p["scale"]
                new = out.clone()
                sl = slice(None) if ch is None else torch.as_tensor(ch, device=out.device, dtype=torch.long)
                cur = new[:, sl]
                new[:, sl] = cur + s * (donor[:, sl] - cur)
                return new
            return None
        return fn

    def close(self):
        for h in self.handles:
            h.remove()

    @torch.no_grad()
    def run(self, obs, capture=None, patch=None):
        self.patch = patch
        self.cap = {}
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        h = torch.zeros(x.shape[0], self.emb.output_dim, device=self.device)
        m = torch.zeros(x.shape[0], device=self.device)
        dist, v, _ = self.policy(x, h, m)
        out = {
            "probs": dist.probs.detach().cpu().numpy(),
            "value": v.detach().cpu().numpy().reshape(-1),
            "caps": {k: self.cap[k].detach().cpu().numpy() for k in (capture or [])},
        }
        self.patch = None
        return out

    def sensitivity(self, obs, locus, a_goal, a_prox):
        """Per-channel |grad| of (logp[a_goal]-logp[a_prox]) and of V wrt locus acts."""
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        h = torch.zeros(B, self.emb.output_dim, device=self.device)
        m = torch.zeros(B, device=self.device)
        idx = torch.arange(B, device=self.device)
        ag = torch.as_tensor(a_goal, device=self.device, dtype=torch.long)
        ap = torch.as_tensor(a_prox, device=self.device, dtype=torch.long)

        def _grad(scalar_fn):
            self.cap = {}
            with torch.enable_grad():
                dist, v, _ = self.policy(x, h, m)
                act = self.cap[locus]
                act.retain_grad()
                scalar = scalar_fn(dist, v)
                self.policy.zero_grad(set_to_none=True)
                scalar.backward()
                g = act.grad.detach()
            if g.ndim == 4:
                return g.abs().sum(dim=(2, 3)).mean(dim=0).cpu().numpy()
            return g.abs().mean(dim=0).cpu().numpy()

        pol = _grad(lambda d, v: (d.logits[idx, ag] - d.logits[idx, ap]).sum())
        val = _grad(lambda d, v: v.reshape(-1).sum())
        return pol, val

    def signed_grad_pf(self, obs, locus, a_goal, a_prox):
        """Signed per-channel mean grad of per-frame (logit[a_goal]-logit[a_prox]) and V."""
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        h = torch.zeros(B, self.emb.output_dim, device=self.device)
        m = torch.zeros(B, device=self.device)
        idx = torch.arange(B, device=self.device)
        ag = torch.as_tensor(a_goal, device=self.device, dtype=torch.long)
        ap = torch.as_tensor(a_prox, device=self.device, dtype=torch.long)

        def _grad(scalar_fn):
            self.cap = {}
            with torch.enable_grad():
                dist, v, _ = self.policy(x, h, m)
                act = self.cap[locus]
                act.retain_grad()
                self.policy.zero_grad(set_to_none=True)
                scalar_fn(dist, v).backward()
                g = act.grad.detach()
            if g.ndim == 4:
                return g.sum(dim=(2, 3)).mean(dim=0).cpu().numpy()
            return g.mean(dim=0).cpu().numpy()

        g_pol = _grad(lambda d, v: (d.logits[idx, ag] - d.logits[idx, ap]).sum())
        g_val = _grad(lambda d, v: v.reshape(-1).sum())
        return g_pol, g_val


# ----------------------------- collection -----------------------------
def _hwc01(chw):
    return np.transpose(chw, (1, 2, 0))


def capture_cheese_donors(env, lad, agent_xy, agent_cell, orig_cell, wd, base_obs):
    """Relocate the cheese with set_cheese and return moved/same/removed donor activations. Restores state.

    The moved cell flips the cheese to the other horizontal side of the agent (max |dx| tiebreak).
    a_cheese_post is the BFS first step toward the moved cell, since pixel-bearing actions ignore walls.
    grid_agent_matched confirms the walls and agent cell are unchanged after the edit.
    """
    st = get_state(env)
    ocx, ocy = orig_cell
    grid0 = _read_grid(env, wd)
    orig_cpx = cheese_xy_bright(_hwc01(base_obs))
    orig_side = np.sign(orig_cpx[0] - agent_xy[0]) if orig_cpx is not None else 0.0
    best_moved, best_xy, best_gcell, best_score = None, None, None, -1.0
    for gx in range(wd):
        for gy in range(wd):
            if set_cheese(env, gx, gy, 1) != 0:
                continue
            ob = obs_after_edit(env)
            cpx = cheese_xy_bright(_hwc01(ob))
            if cpx is None:
                continue
            if float(np.abs(ob - base_obs).max()) < 0.15:
                continue
            new_side = np.sign(cpx[0] - agent_xy[0])
            # prefer moving the cheese to the opposite horizontal side of the agent
            flip = 1.0 if (orig_side != 0 and new_side == -orig_side) else 0.0
            dist_move = abs(cpx[0] - (orig_cpx[0] if orig_cpx is not None else agent_xy[0]))
            score = 100.0 * flip + dist_move
            if score > best_score:
                best_moved, best_xy, best_gcell, best_score = ob, cpx, (gx, gy), score
    moved = lad.run(best_moved, capture=LOCI) if best_moved is not None else None
    a_cheese_post, path_len_post, grid_agent_matched = None, None, None
    if best_gcell is not None and agent_cell is not None and agent_cell[0] >= 0:
        set_cheese(env, best_gcell[0], best_gcell[1], 1)
        ax2, ay2 = get_agent_cell(env)
        grid_after = _read_grid(env, wd)
        # get_maze_obj marks the cheese cell GOAL_OBJ, so moving the cheese flips exactly the old and new
        # cheese cells. The invariant is that nothing else changed and both cells stay walkable.
        diff_cells = {tuple(c) for c in np.argwhere(grid_after != grid0)}
        expected_cells = {tuple(orig_cell), tuple(best_gcell)}
        walls_unchanged = diff_cells <= expected_cells and all(
            _is_walk(int(grid0[c])) and _is_walk(int(grid_after[c])) for c in diff_cells
        )
        grid_agent_matched = bool((ax2, ay2) == tuple(agent_cell) and walls_unchanged)
        if grid_agent_matched:
            a_cheese_post, path_len_post = first_step_action(grid0, tuple(agent_cell), best_gcell)
    # same-location control
    set_cheese(env, ocx, ocy, 1)
    same_ob = obs_after_edit(env)
    same = lad.run(same_ob, capture=LOCI)
    # removed
    set_cheese(env, 0, 0, 0)
    rm_ob = obs_after_edit(env)
    rm = lad.run(rm_ob, capture=LOCI)
    set_state(env, st)
    return {
        "moved_xy": [float(best_xy[0]), float(best_xy[1])] if best_xy is not None else None,
        "moved_cell": [int(best_gcell[0]), int(best_gcell[1])] if best_gcell is not None else None,
        "agent_cell": [int(agent_cell[0]), int(agent_cell[1])] if agent_cell is not None else None,
        "grid_agent_matched": grid_agent_matched,
        "a_cheese_post": None if a_cheese_post is None else int(a_cheese_post),
        "path_len_cheese_post": path_len_post,
        "moved_caps": {k: moved["caps"][k][0] for k in LOCI} if moved else None,
        "moved_probs": moved["probs"][0].astype(np.float32) if moved else None,
        "moved_value": float(moved["value"][0]) if moved else None,
        "same_caps": {k: same["caps"][k][0] for k in LOCI},
        "rm_caps": {k: rm["caps"][k][0] for k in LOCI},
        "rm_probs": rm["probs"][0].astype(np.float32),
        "rm_value": float(rm["value"][0]),
    }


def collect(lad, *, n_target, start_level, max_levels, max_steps, greedy, verbose,
            max_per_level=3, capture_causal=False, graph_exact=True):
    """Collect graph-exact decision squares (first BFS step to cheese ≠ first step to the reachable extreme TR).

    Falls back to image bearing when a path is missing.
    """
    frames = []
    level = start_level
    t0 = time.time()
    n_graph, n_bearing = 0, 0
    while len(frames) < n_target and (level - start_level) < max_levels:
        env = make_env("maze", start_level=level)
        obs = env.reset()
        prev = _chw_to_hwc01(obs)
        h, w = prev.shape[:2]
        tr_corner = (float(w - 1), 0.0)  # image top-right
        last_xy = None
        kept = 0
        for t in range(max_steps):
            chw = np.asarray(obs[0], dtype=np.float32)
            out = lad.run(chw, capture=LOCI)
            probs = out["probs"][0]
            cur = _chw_to_hwc01(obs)
            cheese_xy = cheese_xy_bright(cur)
            agent_xy = motion_agent_xy(prev, cur)
            if agent_xy is not None:
                last_xy = agent_xy
            a = last_xy
            info = get_info_maze(env)
            wd = _iscalar(info, "maze_world_dim")
            cx, cy = _iscalar(info, "cheese_x"), _iscalar(info, "cheese_y")
            ax_g, ay_g = get_agent_cell(env)
            a_cheese = a_corner = None
            path_len_cheese = path_len_tr = None
            mode = "bearing"
            if graph_exact and ax_g >= 0 and cx >= 0:
                grid = _read_grid(env, wd)
                # TR target is the extreme among reachable cells; the global max x+y cell is often walled off.
                tr = _reachable_extreme(grid, (ax_g, ay_g), "tr")
                if tr is not None and (cx, cy) in _reachable(grid, (ax_g, ay_g)):
                    a_cheese, path_len_cheese = first_step_action(grid, (ax_g, ay_g), (cx, cy))
                    a_corner, path_len_tr = first_step_action(grid, (ax_g, ay_g), tr)
                    if a_cheese is not None and a_corner is not None:
                        mode = "graph"
            if mode == "bearing" and a is not None and cheese_xy is not None:
                a_cheese = action_toward(a, cheese_xy)
                a_corner = action_toward(a, tr_corner)
            keep = (
                kept < max_per_level
                and a is not None
                and cheese_xy is not None
                and a_cheese is not None
                and a_corner is not None
                and a_cheese != a_corner
            )
            if keep:
                if mode == "graph":
                    n_graph += 1
                else:
                    n_bearing += 1
                # Path-length gap is the main distance confound; image distances are kept for plots.
                dist_tr_img = float(np.hypot(tr_corner[0] - a[0], tr_corner[1] - a[1]))
                dist_cheese_img = float(np.hypot(cheese_xy[0] - a[0], cheese_xy[1] - a[1]))
                dist_tr = float(path_len_tr) if path_len_tr is not None else dist_tr_img
                dist_cheese = float(path_len_cheese) if path_len_cheese is not None else dist_cheese_img
                rec = {
                    "level": level, "t": t,
                    "agent_xy": [float(a[0]), float(a[1])],
                    "cheese_xy": [float(cheese_xy[0]), float(cheese_xy[1])],
                    "tr_corner": [tr_corner[0], tr_corner[1]],
                    "h": int(h), "w": int(w),
                    "agent_in_tr": int(in_top_right(a, h, w)),
                    "dist_cheese": dist_cheese,
                    "dist_tr": dist_tr,
                    "dist_cheese_img": dist_cheese_img,
                    "dist_tr_img": dist_tr_img,
                    "choice_mode": mode,
                    "bearing_cheese_x": float(np.sign(cheese_xy[0] - a[0])),
                    "a_cheese": int(a_cheese),
                    "a_corner": int(a_corner),
                    "base_probs": probs.astype(np.float32),
                    "base_value": float(out["value"][0]),
                    "caps": {k: out["caps"][k][0] for k in LOCI},
                    "obs": chw,
                }
                if capture_causal:
                    rec["cheese_causal"] = capture_cheese_donors(
                        env, lad, a, (ax_g, ay_g), (cx, cy), wd, chw)
                    rec["corner_geometry"] = capture_corner_geometry(env, lad, rec["a_corner"])
                frames.append(rec)
                kept += 1
                if verbose and len(frames) % 20 == 0:
                    print(f"  frames={len(frames)}/{n_target} level={level} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            act = int(probs.argmax()) if greedy else int(
                np.random.choice(len(probs), p=probs / probs.sum()))
            prev = cur
            obs, _r, done, _i = env.step(np.array([act]))
            if done[0]:
                break
        if hasattr(env, "close"):
            env.close()
        level += 1
    if verbose:
        print(f"  choice_mode graph={n_graph} bearing_fallback={n_bearing}", flush=True)
    return frames


def _group_stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


def _ks_approx(a, b):
    """Two-sample KS statistic without scipy (ECDF max gap)."""
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    xs = np.unique(np.concatenate([a, b]))
    fa = np.searchsorted(a, xs, side="right") / len(a)
    fb = np.searchsorted(b, xs, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def tr_path_balance(frames):
    """Check whether dist_TR is matched across cheese-vs-corner conditions.

    Reports dist_TR (and path-length gap) for frames that prefer cheese vs prefer corner
    in π, and for frames where cheese is path-closer vs TR path-closer. Large KS / mean
    gaps flag agent-position leakage into the corner decode."""
    dtr = np.array([f["dist_tr"] for f in frames], float)
    dch = np.array([f["dist_cheese"] for f in frames], float)
    net = np.array([float(f["base_probs"][f["a_cheese"]] - f["base_probs"][f["a_corner"]])
                    for f in frames])
    gap = dtr - dch  # >0 → cheese closer on the path metric
    pref_ch = net > 0
    pref_co = net < 0
    closer_ch = gap > 0
    closer_co = gap < 0
    modes = [f.get("choice_mode", "bearing") for f in frames]
    out = {
        "n": len(frames),
        "frac_graph_exact": float(np.mean([m == "graph" for m in modes])),
        "dist_tr_all": _group_stats(dtr),
        "dist_tr_prefer_cheese": _group_stats(dtr[pref_ch]),
        "dist_tr_prefer_corner": _group_stats(dtr[pref_co]),
        "dist_tr_ks_prefer": _ks_approx(dtr[pref_ch], dtr[pref_co]),
        "mean_gap_prefer_cheese": float(np.nanmean(gap[pref_ch])) if pref_ch.any() else float("nan"),
        "mean_gap_prefer_corner": float(np.nanmean(gap[pref_co])) if pref_co.any() else float("nan"),
        "dist_tr_cheese_closer": _group_stats(dtr[closer_ch]),
        "dist_tr_tr_closer": _group_stats(dtr[closer_co]),
        "dist_tr_ks_closer": _ks_approx(dtr[closer_ch], dtr[closer_co]),
        "frac_prefer_cheese": float(np.mean(pref_ch)),
        "frac_cheese_path_closer": float(np.mean(closer_ch)),
        "corr_net_vs_neg_dist_tr": float(np.corrcoef(net, -dtr)[0, 1]) if len(net) > 2 else float("nan"),
        "corr_agent_in_tr_vs_dist_tr": float(
            np.corrcoef(np.array([f["agent_in_tr"] for f in frames], float), dtr)[0, 1]
        ) if len(frames) > 2 else float("nan"),
    }
    # crude balance flag: only meaningful when both prefer-groups are non-empty
    n_ch = out["dist_tr_prefer_cheese"]["n"]
    n_co = out["dist_tr_prefer_corner"]["n"]
    m1, m2 = out["dist_tr_prefer_cheese"]["mean"], out["dist_tr_prefer_corner"]["mean"]
    sd = float(np.nanstd(dtr)) + 1e-9
    if n_ch < 10 or n_co < 10:
        out["balanced_prefer_split"] = None  # not evaluable (policy almost always picks one side)
        out["balance_note"] = (
            f"prefer-split unevaluable (n_cheese_pref={n_ch}, n_corner_pref={n_co}); "
            "policy collapses to one side at decision squares"
        )
    else:
        out["balanced_prefer_split"] = bool(
            np.isfinite(out["dist_tr_ks_prefer"])
            and out["dist_tr_ks_prefer"] < 0.25
            and abs(m1 - m2) < 0.25 * sd
        )
        out["balance_note"] = "prefer-split KS and mean gap within thresholds"
    # closer-split is the geometry confound check (always defined if both sides nonempty)
    n_cch = out["dist_tr_cheese_closer"]["n"]
    n_ctr = out["dist_tr_tr_closer"]["n"]
    if n_cch >= 10 and n_ctr >= 10 and np.isfinite(out["dist_tr_ks_closer"]):
        mc1, mc2 = out["dist_tr_cheese_closer"]["mean"], out["dist_tr_tr_closer"]["mean"]
        out["balanced_closer_split"] = bool(
            out["dist_tr_ks_closer"] < 0.25 and abs(mc1 - mc2) < 0.25 * sd
        )
    else:
        out["balanced_closer_split"] = None
    return out


# ----------------------------- probes (mirror Attribute Maze) -----------------------------
def _split(levels):
    uniq = np.sort(np.unique(levels))
    mid = uniq[len(uniq) // 2]
    return levels < mid, levels >= mid


def _fc_present(X, y, levels, n_shuffle, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y.astype(float))
    X, y, levels = X[ok], y[ok], levels[ok]
    if len(y) < 16 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y))}
    tr, te = _split(levels)
    if tr.sum() < 8 or te.sum() < 8 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        w, b = fit_ridge(X, y)
        return {"auc": float(roc_auc(y, predict_ridge(X, w, b))), "heldout": False,
                "n": int(len(y)), "C_F_topk": np.argsort(-np.abs(w))[:12].tolist()}
    w, b = fit_ridge(X[tr], y[tr])
    auc = float(roc_auc(y[te], predict_ridge(X[te], w, b)))
    null = []
    for _ in range(n_shuffle):
        ys = y[tr].copy(); rng.shuffle(ys)
        ws, bs = fit_ridge(X[tr], ys)
        null.append(float(roc_auc(y[te], predict_ridge(X[te], ws, bs))))
    p95 = float(np.quantile(null, 0.95)) if null else float("nan")
    return {"auc": auc, "null_p95": p95, "beats_null": bool(null and auc > p95),
            "heldout": True, "n": int(len(y)),
            "C_F_topk": np.argsort(-np.abs(w))[:12].tolist()}


def _spatial_present(X, y, levels, n_shuffle):
    ok = np.isfinite(y.astype(float))
    X, y, levels = X[ok], y[ok], levels[ok]
    if len(y) < 16 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y))}
    N, C, H, W = X.shape
    tr, te = _split(levels)
    heldout = not (tr.sum() < 8 or te.sum() < 8 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2)
    best = None
    for i in range(H):
        for j in range(W):
            xc = X[:, :, i, j]
            if heldout:
                w, b = fit_ridge(xc[tr], y[tr])
                tr_auc = roc_auc(y[tr], predict_ridge(xc[tr], w, b))
            else:
                w, b = fit_ridge(xc, y)
                tr_auc = roc_auc(y, predict_ridge(xc, w, b))
            score = abs(tr_auc - 0.5)
            if best is None or score > best["score"]:
                best = {"score": score, "ij": (i, j), "w": w, "b": b}
    i, j = best["ij"]
    xc = X[:, :, i, j]
    if heldout:
        auc = float(roc_auc(y[te], predict_ridge(xc[te], best["w"], best["b"])))
        rng = np.random.default_rng(0)
        null = []
        for _ in range(n_shuffle):
            ys = y[tr].copy(); rng.shuffle(ys)
            ws, bs = fit_ridge(xc[tr], ys)
            null.append(float(roc_auc(y[te], predict_ridge(xc[te], ws, bs))))
        p95 = float(np.quantile(null, 0.95)) if null else float("nan")
    else:
        auc = float(roc_auc(y, predict_ridge(xc, best["w"], best["b"])))
        p95 = float("nan")
    return {"auc": auc, "null_p95": p95, "beats_null": bool(np.isfinite(p95) and auc > p95),
            "peak_cell": [int(i), int(j)], "heldout": heldout, "n": int(len(y)),
            "C_F_topk": np.argsort(-np.abs(best["w"]))[:12].tolist()}


def _present_at(X, y, levels, n_shuffle):
    return _spatial_present(X, y, levels, n_shuffle) if X.ndim == 4 else _fc_present(X, y, levels, n_shuffle)


def _pool_pixels(obs):
    x = obs.reshape(3, 8, 8, 8, 8).mean(axis=(2, 4))
    return x.reshape(-1)


def rung1_present(frames, init_lad, n_shuffle=50):
    levels = np.array([f["level"] for f in frames])
    # corner-priority label: agent near the top-right (median split of distance to TR),
    # a positional variable with guaranteed variance. Expected trained ~ init (geometric readout).
    dtr = np.array([f["dist_tr"] for f in frames])
    near_tr = (dtr <= np.median(dtr)).astype(float)
    labels = {
        "cheese": np.array([1.0 if f["bearing_cheese_x"] > 0 else 0.0 for f in frames]),
        "corner": near_tr,
    }
    pix = np.stack([_pool_pixels(f["obs"]) for f in frames])
    trained_caps = {k: np.stack([f["caps"][k] for f in frames]) for k in LOCI}
    init_caps = {k: [] for k in LOCI}
    for f in frames:
        r = init_lad.run(f["obs"], capture=LOCI)
        for k in LOCI:
            init_caps[k].append(r["caps"][k][0])
    init_caps = {k: np.stack(v) for k, v in init_caps.items()}
    out = {}
    for feat, y in labels.items():
        out[feat] = {}
        for locus in LOCI:
            trained = _present_at(trained_caps[locus], y, levels, n_shuffle)
            init = _present_at(init_caps[locus], y, levels, 0)
            ta, ia = trained.get("auc", np.nan), init.get("auc", np.nan)
            out[feat][locus] = {
                "auc": ta, "beats_shuffle": trained.get("beats_null"),
                "peak_cell": trained.get("peak_cell"), "init_auc": ia,
                "beats_init": bool(np.isfinite(ta) and np.isfinite(ia) and ta > ia + 0.02),
                "n": trained.get("n"), "C_F_topk": trained.get("C_F_topk", []),
            }
        out[feat]["pixel_floor_auc"] = _fc_present(pix, y, levels, 0).get("auc")
        best = max(LOCI, key=lambda L: abs(out[feat][L]["auc"] - 0.5)
                   if np.isfinite(out[feat][L]["auc"]) else -1)
        out[feat]["best_locus"] = best
        out[feat]["C_F"] = out[feat][best].get("C_F_topk", [])
    return out


def rung23_wiring(lad, frames, present):
    obs = np.stack([f["obs"] for f in frames])
    a_goal = np.array([f["a_cheese"] for f in frames])
    a_prox = np.array([f["a_corner"] for f in frames])
    pol_val = {L: lad.sensitivity(obs, L, a_goal, a_prox) for L in LOCI}
    # per-activation-element normaliser so trunk (conv, summed over space) is comparable to fc
    SPATIAL = {"block2": 16 * 16, "block3": 8 * 8, "fc": 1}
    rng = np.random.default_rng(0)
    out = {}
    for feat in ("cheese", "corner"):
        locus = present[feat]["best_locus"]
        C_F = present[feat]["C_F"]
        pol, val = pol_val[locus]
        C = len(pol)
        if not C_F:
            C_F = list(range(min(12, C)))
        rand = rng.choice([c for c in range(C) if c not in set(C_F)],
                          size=min(len(C_F), max(1, C - len(C_F))), replace=False)
        # trunk vs readout: mean contrast-sensitivity per activation element, trunk loci vs fc
        trunk_load = float(np.mean([np.mean(pol_val[L][0]) / SPATIAL[L] for L in TRUNK]))
        fc_load = float(np.mean(pol_val["fc"][0]) / SPATIAL["fc"])
        out[feat] = {
            "locus": locus,
            "policy_load_CF": float(np.mean(pol[C_F])),
            "policy_load_null": float(np.mean(pol[rand])),
            "policy_ratio": float(np.mean(pol[C_F]) / (np.mean(pol[rand]) + 1e-9)),
            "value_load_CF": float(np.mean(val[C_F])),
            "value_load_null": float(np.mean(val[rand])),
            "value_ratio": float(np.mean(val[C_F]) / (np.mean(val[rand]) + 1e-9)),
            "trunk_load": trunk_load, "fc_load": fc_load,
            "trunk_vs_readout": float(trunk_load / (fc_load + 1e-9)),
        }
    return out


def value_at_corner(frames):
    """Rung 3: does V rise as the agent nears the top-right, on cheese-far frames?

    Correlation of base_value with -dist_tr, restricted to frames where the cheese is
    NOT toward the top-right (so high V near TR is not just 'cheese is there').
    """
    v = np.array([f["base_value"] for f in frames])
    dtr = np.array([f["dist_tr"] for f in frames])
    in_tr = np.array([f["agent_in_tr"] for f in frames])
    if len(v) < 8 or np.std(dtr) < 1e-6:
        return {"n": int(len(v)), "corr_V_negdistTR": float("nan")}
    corr = float(np.corrcoef(v, -dtr)[0, 1])
    return {
        "n": int(len(v)),
        "corr_V_negdistTR": corr,
        "V_in_tr_mean": float(v[in_tr == 1].mean()) if (in_tr == 1).any() else float("nan"),
        "V_out_tr_mean": float(v[in_tr == 0].mean()) if (in_tr == 0).any() else float("nan"),
    }


def _tensor(caps, locus):
    return torch.as_tensor(caps[locus][None], dtype=torch.float32)


def rung45_cheese_causal(lad, frames, present, scales=(1.0,), competence_min=0.15):
    """Cheese relocate+patch: patch C_cheese(m') into m and read retargeting toward the cheese's new cell.

    Controls: same location (~0), random channels, whole block, and the env-level move.
    """
    locus = present["cheese"]["best_locus"]
    C_F = present["cheese"]["C_F"]
    # Frames without a graph-exact a_cheese_post (moved cell BFS-unreachable or grid_agent_matched failed)
    # are excluded rather than scored with a pixel-bearing action.
    all_captured = [f for f in frames if f.get("cheese_causal", {}).get("moved_caps") is not None]
    usable = [f for f in all_captured if f["cheese_causal"].get("a_cheese_post") is not None]
    n_no_graph_exact_action = len(all_captured) - len(usable)
    if not usable:
        return {"locus": locus, "n": 0, "note": "no relocations captured"}
    Ctot = usable[0]["caps"][locus].shape[0]
    if not C_F:
        C_F = list(range(min(12, Ctot)))
    rng = np.random.default_rng(0)
    rand_ch = rng.choice([c for c in range(Ctot) if c not in set(C_F)],
                         size=min(len(C_F), Ctot - len(C_F)), replace=False)
    res = {
        "locus": locus, "n": len(usable), "n_no_graph_exact_action": n_no_graph_exact_action,
        "k": len(C_F), "C_total": int(Ctot), "scales": {},
    }
    for s in scales:
        rt_cf, rt_rand, rt_block, rt_env, rt_same, dv_cf, dv_env, comp = ([] for _ in range(8))
        for f in usable:
            cc = f["cheese_causal"]
            a_new = cc["a_cheese_post"]
            base = f["base_probs"][a_new]
            donor = _tensor(cc["moved_caps"], locus)
            r = lad.run(f["obs"], patch={"locus": locus, "donor": donor, "channels": np.array(C_F), "scale": s})
            p = r["probs"][0]
            rt_cf.append(float(p[a_new] - base)); dv_cf.append(float(r["value"][0] - f["base_value"]))
            comp.append(float(p.max()))
            rr = lad.run(f["obs"], patch={"locus": locus, "donor": donor, "channels": rand_ch, "scale": s})
            rt_rand.append(float(rr["probs"][0][a_new] - base))
            rb = lad.run(f["obs"], patch={"locus": locus, "donor": donor, "channels": None, "scale": s})
            rt_block.append(float(rb["probs"][0][a_new] - base))
            sd = _tensor(cc["same_caps"], locus)
            rs = lad.run(f["obs"], patch={"locus": locus, "donor": sd, "channels": np.array(C_F), "scale": s})
            rt_same.append(float(rs["probs"][0][a_new] - base))
            rt_env.append(float(cc["moved_probs"][a_new] - base))
            dv_env.append(float(cc["moved_value"] - f["base_value"]))

        def m(a):
            a = np.asarray(a, float)
            finite = a[np.isfinite(a)]
            out = {"mean": float(np.nanmean(a)), "se": float(np.nanstd(a) / max(1, np.sqrt(len(a))))}
            if len(finite) > 1:
                _, lo, hi = _ci95(finite)
                out["ci95_lo"], out["ci95_hi"] = lo, hi
            else:
                out["ci95_lo"], out["ci95_hi"] = None, None
            return out
        k_frac = len(C_F) / Ctot
        uniform_share = k_frac * float(np.nanmean(rt_env))
        res["scales"][f"{s:.2f}"] = {
            "retarget_CF": m(rt_cf), "retarget_random": m(rt_rand), "retarget_block": m(rt_block),
            "retarget_same": m(rt_same), "env_move_ceiling": m(rt_env),
            "frac_env_captured": float(np.nanmean(rt_cf) / (np.nanmean(rt_env) + 1e-9)),
            "k_frac": float(k_frac), "excess_over_share": float(np.nanmean(rt_cf) - uniform_share),
            "excess_over_random": float(np.nanmean(rt_cf) - np.nanmean(rt_rand)),
            "dV_CF": m(dv_cf), "dV_env": m(dv_env),
            "competence_in_band": float(np.mean(np.asarray(comp) >= competence_min)),
        }
    return res


def rung45_corner_lever(lad, frames, present, scales=(1.0,)):
    """Corner geometry lever: patch activations from the open-TR maze into the original vs a BL-carve null.

    env_open_TR_ceiling is Δπ on the pre-edit a_corner (negative when the first step flips after the carve);
    env_open_TR_ceiling_post is Δπ on the post-edit first step toward the opened TR.
    """
    locus = present["corner"]["best_locus"]
    C_F = present["corner"]["C_F"]
    usable = [f for f in frames if f.get("corner_geometry") and f["corner_geometry"].get("opened_caps")]
    if not usable:
        return {"locus": locus, "n": 0, "note": "no geometry donors captured"}
    Ctot = usable[0]["caps"][locus].shape[0]
    if not C_F:
        C_F = list(range(min(12, Ctot)))
    n_tr = float(np.mean([f["corner_geometry"]["n_walls_tr"] for f in usable]))
    n_null = float(np.mean([f["corner_geometry"]["n_walls_null"] for f in usable]))
    frac_new = float(np.mean([f["corner_geometry"]["tr_newly_reachable"] for f in usable]))
    flips = [f["corner_geometry"].get("a_corner_flipped") for f in usable]
    frac_flip = float(np.mean([bool(x) for x in flips if x is not None])) if any(x is not None for x in flips) else float("nan")

    def _mean_key(key):
        vals = [f["corner_geometry"].get(key) for f in usable if f["corner_geometry"].get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    res = {"locus": locus, "n": len(usable), "k": len(C_F), "C_total": int(Ctot),
           "mean_walls_tr": n_tr, "mean_walls_null": n_null,
           "frac_tr_newly_reachable": frac_new,
           "frac_a_corner_flipped": frac_flip,
           "mean_d_reach_tr": _mean_key("d_reach_tr"),
           "mean_d_reach_bl": _mean_key("d_reach_bl"),
           "mean_d_local_tr": _mean_key("d_local_tr"),
           "mean_d_local_bl": _mean_key("d_local_bl"),
           "mean_d_degree_tr": _mean_key("d_degree_tr"),
           "mean_d_degree_bl": _mean_key("d_degree_bl"),
           "scales": {}}
    for s in scales:
        rt, rt_null, rt_env, rt_env_post, dv, dv_env, comp = [], [], [], [], [], [], []
        for f in usable:
            cg = f["corner_geometry"]
            a_pre = cg["a_corner"]
            a_post = cg.get("a_corner_post")
            base_pre = f["base_probs"][a_pre]
            donor = _tensor(cg["opened_caps"], locus)
            r = lad.run(f["obs"], patch={"locus": locus, "donor": donor, "channels": np.array(C_F), "scale": s})
            rt.append(float(r["probs"][0][a_pre] - base_pre))
            dv.append(float(r["value"][0] - f["base_value"]))
            comp.append(float(r["probs"][0].max()))
            dn = _tensor(cg["null_caps"], locus)
            rn = lad.run(f["obs"], patch={"locus": locus, "donor": dn, "channels": np.array(C_F), "scale": s})
            rt_null.append(float(rn["probs"][0][a_pre] - base_pre))
            # pre-edit action index on post-edit probs
            rt_env.append(float(cg["opened_probs"][a_pre] - base_pre))
            # matched ceiling: post-edit first-step action, Δ vs base on that same index
            if a_post is not None:
                rt_env_post.append(float(cg["opened_probs"][a_post] - f["base_probs"][a_post]))
            else:
                rt_env_post.append(float("nan"))
            dv_env.append(float(cg["opened_value"] - f["base_value"]))

        def m(a):
            a = np.asarray(a, float)
            finite = a[np.isfinite(a)]
            out = {"mean": float(np.nanmean(a)), "se": float(np.nanstd(a) / max(1, np.sqrt(len(a))))}
            if len(finite) > 1:
                _, lo, hi = _ci95(finite)
                out["ci95_lo"], out["ci95_hi"] = lo, hi
            else:
                out["ci95_lo"], out["ci95_hi"] = None, None
            return out
        res["scales"][f"{s:.2f}"] = {
            "retarget_TR": m(rt), "retarget_null": m(rt_null),
            "excess_over_null": float(np.nanmean(rt) - np.nanmean(rt_null)),
            "env_open_TR_ceiling": m(rt_env),
            "env_open_TR_ceiling_post": m(rt_env_post),
            "dV": m(dv), "dV_env": m(dv_env),
            "competence_in_band": float(np.mean(np.asarray(comp) >= 0.15)),
        }
    return res


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _cos(a, b):
    return float(np.dot(_unit(a), _unit(b)))


def head_divergence(lad, frames, n_null=4000, seed=0):
    """Do policy and value read the same fc subspace? cos(g_pol,g_val) vs random null, and which
    head reads each feature axis. g_pol from per-frame (a_cheese vs a_corner), g_val from V."""
    obs = np.stack([f["obs"] for f in frames])
    a_goal = np.array([f["a_cheese"] for f in frames])
    a_prox = np.array([f["a_corner"] for f in frames])
    g_pol, g_val = lad.signed_grad_pf(obs, "fc", a_goal, a_prox)
    overlap = _cos(g_pol, g_val)
    rng = np.random.default_rng(seed)
    C = g_pol.shape[0]
    R = rng.standard_normal((n_null, C))
    nulls = np.abs((R @ _unit(g_pol)) / (np.linalg.norm(R, axis=1) + 1e-12))
    null_p95 = float(np.quantile(nulls, 0.95))
    fc = np.stack([f["caps"]["fc"] for f in frames])
    y_cheese = np.array([1.0 if f["bearing_cheese_x"] > 0 else 0.0 for f in frames])
    dtr = np.array([f["dist_tr"] for f in frames])
    y_corner = (dtr <= np.median(dtr)).astype(float)
    w_cheese, _ = fit_ridge(fc, y_cheese)
    w_corner, _ = fit_ridge(fc, y_corner)
    W_pi = lad.policy.fc_policy.weight.detach().cpu().numpy().astype(np.float64)
    w_v = lad.policy.fc_value.weight.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return {
        "overlap_gpol_gval": overlap, "overlap_null_p95": null_p95,
        "heads_share": bool(abs(overlap) > null_p95),
        "cheese_read_by_policy": abs(_cos(w_cheese, g_pol)),
        "cheese_read_by_value": abs(_cos(w_cheese, g_val)),
        "corner_read_by_policy": abs(_cos(w_corner, g_pol)),
        "corner_read_by_value": abs(_cos(w_corner, g_val)),
        "weight_cos_pol_val": _cos(W_pi[A_LEFT] - W_pi[A_RIGHT], w_v),
        "n": len(frames),
    }


def retarget_gradient(frames, n_bins=5):
    """Base preference gradient: net-prob(toward cheese) - net-prob(toward corner) vs dist_tr."""
    dtr = np.array([f["dist_tr"] for f in frames])
    net = np.array([float(f["base_probs"][f["a_cheese"]] - f["base_probs"][f["a_corner"]])
                    for f in frames])
    if len(dtr) < n_bins:
        return {"bins": [], "net": []}
    edges = np.quantile(dtr, np.linspace(0, 1, n_bins + 1))
    centers, vals = [], []
    for k in range(n_bins):
        m = (dtr >= edges[k]) & (dtr <= edges[k + 1])
        if m.any():
            centers.append(float(dtr[m].mean()))
            vals.append(float(net[m].mean()))
    return {"dist_tr": centers, "net_cheese_minus_corner": vals}


# ----------------------------- driver -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed-tag", default="")
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--start-level", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=6000)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-per-level", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--causal", action="store_true", help="also run rungs 4/5 (needs set_cheese)")
    ap.add_argument("--scales", default="0.5,1.0,1.5")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scales = tuple(float(x) for x in args.scales.split(","))
    if args.quick:
        args.n_frames, args.max_levels = 16, 300
    device = args.device or str(_default_device())

    print(f"[load] {args.ckpt} device={device}", flush=True)
    policy = load_policy(args.ckpt, device)
    lad = Ladder(policy, device)
    init_policy = load_policy(args.ckpt, device, random_init=True)
    init_lad = Ladder(init_policy, device)

    t0 = time.time()
    print("[stageA] collecting choice points (maze OOD)...", flush=True)
    frames = collect(lad, n_target=args.n_frames, start_level=args.start_level,
                     max_levels=args.max_levels, max_steps=args.max_steps,
                     greedy=True, verbose=True, max_per_level=args.max_per_level,
                     capture_causal=args.causal)
    print(f"[stageA] {len(frames)} frames in {time.time()-t0:.0f}s", flush=True)
    tag = ("_" + args.seed_tag) if args.seed_tag else ""
    if len(frames) < 10:
        (args.out_dir / f"ladder{tag}.json").write_text(
            json.dumps({"error": "too few choice points", "n": len(frames)}, indent=2))
        return

    print("[rung1] present + floors...", flush=True)
    present = rung1_present(frames, init_lad)
    print("[rung2/3] wiring...", flush=True)
    wiring = rung23_wiring(lad, frames, present)
    value_corner = value_at_corner(frames)
    heads = head_divergence(lad, frames)
    grad = retarget_gradient(frames)
    balance = tr_path_balance(frames)

    result = {
        "ckpt": str(args.ckpt), "seed_tag": args.seed_tag, "env": "maze",
        "n_frames": len(frames), "device": device, "loci": LOCI,
        "gm_type": "location", "goal": "cheese", "proxy": "corner_top_right",
        "rung1_present": present,
        "rung23_wiring": wiring,
        "rung3_value_at_corner": value_corner,
        "head_divergence": heads,
        "retarget_gradient_base": grad,
        "tr_path_balance": balance,
        "runtime_s": time.time() - t0,
    }
    if args.causal:
        print("[rung4/5] cheese relocate+patch...", flush=True)
        result["rung45_cheese_causal"] = rung45_cheese_causal(lad, frames, present, scales=scales)
        print("[rung4/5] corner geometry lever (open-TR vs BL-null)...", flush=True)
        result["rung45_corner_lever"] = rung45_corner_lever(lad, frames, present, scales=scales)
    else:
        result["causal_note"] = ("rung4/5 not run (pass --causal). Cheese uses set_cheese relocate+patch; "
                                 "corner uses open-TR wall-geometry lever vs BL-matched null.")
    out = args.out_dir / f"ladder{tag}.json"
    out.write_text(json.dumps(result, indent=2,
                              default=lambda o: float(o) if isinstance(o, np.floating) else o))
    print(f"[done] wrote {out}", flush=True)
    for feat in ("cheese", "corner"):
        p = present[feat]
        bl = p["best_locus"]
        print(f"  {feat}: best={bl} auc={p[bl]['auc']:.3f} init={p[bl]['init_auc']:.3f} "
              f"beats_init={p[bl]['beats_init']} pol_ratio={wiring[feat]['policy_ratio']:.2f} "
              f"trunk/readout={wiring[feat]['trunk_vs_readout']:.2f}", flush=True)
    print(f"  V~-dist_TR corr={value_corner['corr_V_negdistTR']}", flush=True)
    print(f"  head overlap cos(g_pol,g_val)={heads['overlap_gpol_gval']:+.3f} "
          f"(null_p95={heads['overlap_null_p95']:.3f} share={heads['heads_share']}) | "
          f"cheese pol/val={heads['cheese_read_by_policy']:.3f}/{heads['cheese_read_by_value']:.3f} "
          f"corner pol/val={heads['corner_read_by_policy']:.3f}/{heads['corner_read_by_value']:.3f}",
          flush=True)
    print(f"  TR-path balance: graph_frac={balance['frac_graph_exact']:.2f} "
          f"KS_prefer={balance['dist_tr_ks_prefer']:.3f} "
          f"balanced={balance['balanced_prefer_split']} "
          f"mean_dtr cheese-pref={balance['dist_tr_prefer_cheese']['mean']:.2f} "
          f"corner-pref={balance['dist_tr_prefer_corner']['mean']:.2f}", flush=True)
    if args.causal:
        ch = result["rung45_cheese_causal"].get("scales", {}).get("1.00", {})
        co = result["rung45_corner_lever"].get("scales", {}).get("1.00", {})
        if ch:
            print(f"  cheese causal: retarget_CF={ch['retarget_CF']['mean']:+.3f} "
                  f"same={ch['retarget_same']['mean']:+.3f} env_ceiling={ch['env_move_ceiling']['mean']:+.3f} "
                  f"comp={ch['competence_in_band']:.2f}", flush=True)
        if co:
            post = co.get("env_open_TR_ceiling_post", {})
            post_m = post.get("mean", float("nan")) if isinstance(post, dict) else float("nan")
            lever = result["rung45_corner_lever"]
            print(f"  corner geometry: retarget_TR={co['retarget_TR']['mean']:+.3f} "
                  f"null={co['retarget_null']['mean']:+.3f} excess={co['excess_over_null']:+.3f} "
                  f"env_open_TR(legacy)={co['env_open_TR_ceiling']['mean']:+.3f} "
                  f"env_open_TR(post)={post_m:+.3f} "
                  f"flip_frac={lever.get('frac_a_corner_flipped', float('nan')):.2f}", flush=True)


if __name__ == "__main__":
    main()
