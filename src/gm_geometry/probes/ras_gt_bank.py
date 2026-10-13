"""Attribute Maze conflict-state geometry computed from the serialized procgen state only (no pixels)."""
# get_state() layout: ... entities (x, y, vx, vy, rx, ry float32, ...) ... grid.w, grid.h, n, grid.data[n] (int32,
# index y*w + x), maze_dim, world_dim. Object ids: WALL=51, SPACE=100, GOAL1=2 (line), GOAL2=3 (gem).
# Movement is grid_step, one cell per step; diagonal actions move horizontally.
from __future__ import annotations

import hashlib
import json
from collections import deque

import numpy as np

WALL_OBJ, SPACE_OBJ, GOAL_LINE, GOAL_GEM = 51, 100, 2, 3
N_ACTIONS = 15
# procgen combo order: LEFT_DOWN, LEFT, LEFT_UP, DOWN, NOOP, UP, RIGHT_DOWN, RIGHT, RIGHT_UP, D, A, W, S, Q, E
ENGINE_DELTAS: dict[int, tuple[int, int] | None] = {
    0: (-1, 0), 1: (-1, 0), 2: (-1, 0), 3: (0, -1), 4: None, 5: (0, 1),
    6: (1, 0), 7: (1, 0), 8: (1, 0), 9: None, 10: None, 11: None, 12: None, 13: None, 14: None,
}
PATH_ACTION = {(-1, 0): 1, (1, 0): 7, (0, -1): 3, (0, 1): 5}
LINE_N_GEM_E = ((0, 1), (1, 0))
LINE_E_GEM_N = ((1, 0), (0, 1))

Cell = tuple[int, int]


VALID_CELL_IDS = {WALL_OBJ, SPACE_OBJ, GOAL_LINE, GOAL_GEM}


def parse_grid(raw: bytes, world_dim: int = 5) -> tuple[int, int, np.ndarray]:
    """(w, h, data[h, w]) located by Grid::serialize's own signature: int32 (w, h, w*h)
    followed by w*h cell ids from {wall, space, line, gem}. Must be unique in the buffer."""
    n = world_dim * world_dim
    header = np.array([world_dim, world_dim, n], dtype="<i4").tobytes()
    hits = []
    i = raw.find(header)
    while i != -1:
        end = i + 12 + 4 * n
        if end <= len(raw):
            data = np.frombuffer(raw[i + 12:end], dtype="<i4")
            if set(int(v) for v in data) <= VALID_CELL_IDS:
                hits.append(data)
        i = raw.find(header, i + 1)
    if len(hits) != 1:
        raise ValueError(f"expected exactly one grid block for world_dim={world_dim}, found {len(hits)}")
    return world_dim, world_dim, hits[0].reshape(world_dim, world_dim).copy()


def parse_agent(raw: bytes) -> Cell:
    """The unique entity record with half-integer (x, y) and radius 0.5 (the player)."""
    hits = []
    for align in range(4):
        n = (len(raw) - align) // 4
        f = np.frombuffer(raw[align:align + 4 * n], dtype="<f4")
        if len(f) < 6:
            continue
        x, y, rx, ry = f[:-5], f[1:-4], f[4:-1], f[5:]
        with np.errstate(invalid="ignore"):
            ok = ((rx == 0.5) & (ry == 0.5) & (x >= 0) & (x < 64) & (y >= 0) & (y < 64)
                  & (np.mod(x, 1) == 0.5) & (np.mod(y, 1) == 0.5))
        hits += [(align + 4 * int(i), int(x[i]), int(y[i])) for i in np.nonzero(ok)[0]]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one player entity, found {sorted(hits)}")
    (_, ax, ay), = hits
    return ax, ay


def cells_of(grid: np.ndarray, obj: int) -> list[Cell]:
    ys, xs = np.nonzero(grid == obj)
    return sorted((int(x), int(y)) for x, y in zip(xs, ys))


def walkable_cells(grid: np.ndarray) -> set[Cell]:
    ys, xs = np.nonzero(grid != WALL_OBJ)
    return {(int(x), int(y)) for x, y in zip(xs, ys)}


def successor(cell: Cell, action: int, walkable: set[Cell]) -> Cell:
    d = ENGINE_DELTAS[action]
    if d is None:
        return cell
    nxt = (cell[0] + d[0], cell[1] + d[1])
    return nxt if nxt in walkable else cell


def bfs(walkable: set[Cell], target: Cell) -> dict[Cell, int]:
    dist = {target: 0}
    q = deque([target])
    while q:
        c = q.popleft()
        for d in PATH_ACTION:
            n = (c[0] + d[0], c[1] + d[1])
            if n in walkable and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def progress_actions(dist: dict[Cell, int], cell: Cell, walkable: set[Cell]) -> list[int]:
    """All engine actions (diagonal aliases included) whose successor is one step closer."""
    d0 = dist.get(cell)
    if d0 is None or d0 == 0:
        return []
    return [a for a in range(N_ACTIONS) if dist.get(successor(cell, a, walkable)) == d0 - 1]


def directions(actions: list[int]) -> list[Cell]:
    return sorted({d for a in actions if (d := ENGINE_DELTAS[a]) is not None})


def shortest_path(dist: dict[Cell, int], cell: Cell, walkable: set[Cell]) -> list[int]:
    path, cur = [], cell
    while dist[cur] > 0:
        a = next(PATH_ACTION[d] for d in sorted(PATH_ACTION)
                 if dist.get(successor(cur, PATH_ACTION[d], walkable)) == dist[cur] - 1)
        path.append(a)
        cur = successor(cur, a, walkable)
    return path


def geometry(agent: Cell, line: Cell, gem: Cell, walkable: set[Cell]) -> dict:
    """Distances, complete progress-action sets, directions, bearing and one-axis sign, and
    the two conflict criteria stored separately: equidistant (both reachable, d_line ==
    d_gem > 0) and disjoint (A_line and A_gem non-empty with no common action). A strict
    conflict ("conflict") requires both; ties (equidistant, overlapping) are excluded."""
    # Reaching either target ends the episode, so neither target's cell may be passed through
    # on the way to the other.
    dl, dg = bfs(walkable - {gem}, line), bfs(walkable - {line}, gem)
    a_line, a_gem = progress_actions(dl, agent, walkable), progress_actions(dg, agent, walkable)
    dirs_line, dirs_gem = directions(a_line), directions(a_gem)
    d_line, d_gem = dl.get(agent), dg.get(agent)
    equidistant = bool(d_line is not None and d_gem is not None and d_line == d_gem and d_line > 0)
    disjoint = bool(a_line and a_gem and not set(a_line) & set(a_gem))
    conflict = equidistant and disjoint
    bearing = None
    if len(dirs_line) == 1 and len(dirs_gem) == 1:
        bearing = list(dirs_line[0]) + list(dirs_gem[0])
    pair = (tuple(dirs_line[0]), tuple(dirs_gem[0])) if bearing else None
    sign = 1 if pair == LINE_N_GEM_E else (-1 if pair == LINE_E_GEM_N else None)
    return {
        "d_line": d_line, "d_gem": d_gem, "A_line": a_line, "A_gem": a_gem,
        "dirs_line": [list(d) for d in dirs_line], "dirs_gem": [list(d) for d in dirs_gem],
        "equidistant": equidistant, "disjoint": disjoint, "conflict": conflict,
        "bearing": bearing, "axis_sign_lineN_gemE": sign,
        "path_line": shortest_path(dl, agent, walkable) if d_line else [],
        "path_gem": shortest_path(dg, agent, walkable) if d_gem else [],
    }


def state_record(raw: bytes, level: int, background: str) -> dict:
    """Full record for one start state, from the serialized buffer alone."""
    w, h, grid = parse_grid(raw)
    agent = parse_agent(raw)
    lines, gems = cells_of(grid, GOAL_LINE), cells_of(grid, GOAL_GEM)
    if len(lines) != 1 or len(gems) != 1:
        raise ValueError(f"level {level}: expected one line and one gem, got {lines} {gems}")
    walkable = walkable_cells(grid)
    rec = {"level": int(level), "background": background, "agent": list(agent),
           "line": list(lines[0]), "gem": list(gems[0]), "grid_w": w, "grid_h": h,
           "walls": sorted([list(c) for c in {(x, y) for x in range(w) for y in range(h)} - walkable]),
           "state_sha256": hashlib.sha256(raw).hexdigest()}
    rec["state_key"] = state_key(rec)
    rec.update(geometry(agent, lines[0], gems[0], walkable))
    return rec


def state_key(rec: dict) -> str:
    """Collision-safe identity of a start state: sha256 over (agent, walls, line, gem,
    grid size). Duplicate levels share it; splits must be made on it, never on level ids."""
    payload = [rec["agent"], sorted(rec["walls"]), rec["line"], rec["gem"], rec["grid_w"], rec["grid_h"]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def walls_key(rec: dict) -> str:
    return hashlib.sha256(json.dumps(sorted(rec["walls"]), separators=(",", ":")).encode()).hexdigest()


LAYOUT_KEYS = ("level", "state_key", "agent", "line", "gem", "grid_w", "grid_h", "walls", "d_line", "d_gem",
               "A_line", "A_gem", "dirs_line", "dirs_gem", "equidistant", "disjoint", "conflict", "bearing",
               "axis_sign_lineN_gemE", "path_line", "path_gem")


def canonical_hash(records: list[dict], keys=None) -> str:
    rows = [{k: r[k] for k in (keys or sorted(r))} for r in sorted(records, key=lambda r: r["level"])]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
