"""Maze corner-open edit: does the policy head to the NEW top-right extreme or the OLD reachable one?"""
from __future__ import annotations

import os
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "external" / "procgenAISC", ROOT / "external" / "train-procgen-pytorch",
          ROOT / "scripts" / "procgen"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
CKPT_DIR = DATA_ROOT / "checkpoints/head_rewire/maze"
LABELS = ["1080_gm", "1081_gm", "1082_gm", "1090_teach", "1091_teach", "1092_teach"]
SAVED = DATA_ROOT / "probes/maze_envmove_graph_exact_20260914"
T975 = {2: 4.302652729911275}


def use_prebuilt_procgen() -> None:
    """Reuse the existing arm64 libenv; procgen's cmake step would otherwise try to rebuild it."""
    import procgen.env as pe
    lib = Path(pe.__file__).parent / ".build" / "relwithdebinfo"
    if (lib / "libenv.dylib").exists() or (lib / "libenv.so").exists():
        pe.build = lambda debug=False, package=False: str(lib)


def bfs_dist(grid, start, is_walk):
    from collections import deque
    wd = grid.shape[0]
    d = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if 0 <= n[0] < wd and 0 <= n[1] < wd and n not in d and is_walk(int(grid[n])):
                d[n] = d[(x, y)] + 1
                q.append(n)
    return d


SCREEN_CORNER = (23, 23)
NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def insulated_corridor(grid, target, higher_is_tr, forbid, is_walk, wall, max_len=12):
    """Shortest wall corridor ending at `target` that touches the open maze only at its entry cell.

    Every corridor cell except the entry has no open 4-neighbour, so after carving `target` is reachable
    only through the corridor. All corridor cells stay strictly less extreme than `target` (x+y below it
    for TR, above it for BL), so `target` is the unique new extreme. The entry wall must touch exactly one
    open cell, and that cell must not be in `forbid` (the old extreme: otherwise the path runs through it).
    Returns (cells entry->target, entry open cell) or (None, None).
    """
    from collections import deque
    wd = grid.shape[0]
    s_t = target[0] + target[1]

    def inside(c):
        return 1 <= c[0] <= wd - 2 and 1 <= c[1] <= wd - 2

    def less_extreme(c):
        return c[0] + c[1] < s_t if higher_is_tr else c[0] + c[1] > s_t

    def open_nbrs(c):
        return [(c[0] + dx, c[1] + dy) for dx, dy in NBRS
                if 0 <= c[0] + dx < wd and 0 <= c[1] + dy < wd and is_walk(int(grid[c[0] + dx, c[1] + dy]))]

    if not inside(target) or int(grid[target]) != wall or open_nbrs(target):
        return None, None
    prev = {target: None}
    q = deque([(target, 1)])
    while q:
        c, n = q.popleft()
        if n >= max_len:
            continue
        for dx, dy in NBRS:
            nb = (c[0] + dx, c[1] + dy)
            if nb in prev or not inside(nb) or int(grid[nb]) != wall or not less_extreme(nb):
                continue
            prev[nb] = c
            on = open_nbrs(nb)
            if on:
                if len(on) == 1 and on[0] not in forbid:
                    cells, cur = [], nb
                    while cur is not None:
                        cells.append(cur)
                        cur = prev[cur]
                    return cells, on[0]
                continue
            q.append((nb, n + 1))
    return None, None


def build_extension(grid, agent, anchor, tr, is_walk, wall, bfs_path, reach_extreme, min_sc, match_len=None):
    """Carve plan for a 2-cell extension beyond `anchor` (right/up for TR, left/down for BL).

    Candidates are anchor+(2,0) and anchor+(0,2) (mirrored for BL); anchor+(1,1) and one-cell steps are
    adjacent to the anchor's own neighbours and cannot be insulated from it. Picks the shortest valid
    corridor (or, with match_len, the one closest to that length). Valid = new cell becomes the unique reachable extreme, the agent's shortest path to it does
    not pass through the anchor, and (TR only) every carved cell is >= min_sc Chebyshev from (23, 23).
    Returns a dict (with 'reason' on failure).
    """
    sgn = 1 if tr else -1
    cands = [("right" if tr else "left", (anchor[0] + 2 * sgn, anchor[1])),
             ("up" if tr else "down", (anchor[0], anchor[1] + 2 * sgn))]
    best, reasons = None, []
    for kind, tgt in cands:
        forbid = {anchor}
        for _ in range(4):
            cells, entry = insulated_corridor(grid, tgt, tr, forbid, is_walk, wall)
            if cells is None:
                reasons.append(f"{kind}:no_corridor")
                break
            sc = min(max(SCREEN_CORNER[0] - x, SCREEN_CORNER[1] - y) for x, y in cells)
            g1 = grid.copy()
            for c in cells:
                g1[c] = 100
            path = bfs_path(g1, agent, tgt)
            if path is None:
                reasons.append(f"{kind}:unreachable")
                break
            if tuple(anchor) in map(tuple, path):
                forbid = forbid | {entry}
                reasons.append(f"{kind}:via_old")
                continue
            if reach_extreme(g1, agent, "tr" if tr else "bl") != tgt:
                reasons.append(f"{kind}:not_extreme")
                break
            if tr and sc < min_sc:
                reasons.append(f"{kind}:near_screen_corner")
                break
            plan = {"kind": kind, "target": tgt, "cells": cells, "entry": entry, "sc_cheb": sc, "grid": g1}
            key = (abs(len(cells) - match_len) if match_len else 0, len(cells))
            if best is None or key < best["key"]:
                best = dict(plan, key=key)
            break
    return best if best is not None else {"reason": ";".join(reasons)}


def run_one(label: str, args) -> None:
    import torch
    use_prebuilt_procgen()
    import run_maze_i_goal_proxy_ladder as L
    from validate_langosco_maze_i_200m import _chw_to_hwc01, motion_agent_xy

    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    # Tiny batch-1 forwards: CPU matches the saved CPU/CUDA runs more closely and is not slower.
    device = "cpu" if args.cpu else device
    policy = L.load_policy(CKPT_DIR / f"{label}.pth", device)
    lad = L.Ladder(policy, device)

    def probs_of(chw):
        return lad.run(chw)["probs"][0].astype(np.float64)

    def greedy_rollout(env, grid, targets, steps):
        """Greedy rollout in the current env state; per-step BFS distance to each target cell."""
        dist_maps = {k: bfs_dist(grid, t, L._is_walk) for k, t in targets.items()}
        cell = L.get_agent_cell(env)
        traj = [cell]
        obs = L.obs_after_edit(env)
        done_at = None
        for t in range(steps):
            a = int(probs_of(obs).argmax())
            o, _r, done, _i = env.step(np.array([a]))
            if done[0]:
                done_at = t + 1
                break
            obs = np.asarray(o[0], dtype=np.float32)
            cell = L.get_agent_cell(env)
            traj.append(cell)
        d = {k: [dm.get(c, np.nan) for c in traj] for k, dm in dist_maps.items()}
        return traj, d, done_at

    def first_hit(arr):
        hit = np.where(np.asarray(arr, float) == 0)[0]
        return int(hit[0]) if len(hit) else -1

    def apply_plan(env, plan):
        for x, y in plan["cells"]:
            if L.set_maze_obj(env, x, y, L.SPACE_OBJ) != 0:
                raise RuntimeError(f"set_maze_obj failed at {(x, y)}")

    def edit_arm(env, st, grid0, ag, old, probs, plan, prefix):
        """Carve plan, score first step + greedy rollout toward plan target vs the old TR extreme, restore."""
        out = {f"{prefix}_ok": int("target" in plan)}
        if "target" not in plan:
            out[f"{prefix}_reason"] = plan["reason"]
            return out
        tgt = plan["target"]
        apply_plan(env, plan)
        wd = grid0.shape[0]
        grid1 = L._read_grid(env, wd)
        assert all(grid1[c] == L.SPACE_OBJ for c in plan["cells"])
        a_new, len_new = L.first_step_action(grid1, ag, tgt)
        a_old, len_old = L.first_step_action(grid1, ag, old)
        p1 = probs_of(L.obs_after_edit(env))
        traj, dist, done_at = greedy_rollout(env, grid1, {"new": tgt, "old": old}, args.rollout_steps)
        L.set_state(env, st)
        dn, do = np.asarray(dist["new"], float), np.asarray(dist["old"], float)
        step1 = (dn[1] - dn[0], do[1] - do[0]) if len(dn) > 1 else (np.nan, np.nan)
        cells = set(plan["cells"])
        out.update({
            f"{prefix}_kind": plan["kind"], f"{prefix}_x": tgt[0], f"{prefix}_y": tgt[1],
            f"{prefix}_len_corridor": len(plan["cells"]), f"{prefix}_sc_cheb": plan["sc_cheb"],
            f"{prefix}_tgt_sc_cheb": max(SCREEN_CORNER[0] - tgt[0], SCREEN_CORNER[1] - tgt[1]),
            f"{prefix}_len_new": len_new, f"{prefix}_len_old": len_old,
            f"{prefix}_a_new": a_new, f"{prefix}_a_old": a_old,
            f"{prefix}_diagnostic": int(a_new is not None and a_old is not None and a_new != a_old),
            f"{prefix}_argmax_post": int(p1.argmax()),
            f"{prefix}_p_new_post": float(p1[a_new]) if a_new is not None else np.nan,
            f"{prefix}_p_old_post": float(p1[a_old]) if a_old is not None else np.nan,
            f"{prefix}_p_new_pre": float(probs[a_new]) if a_new is not None else np.nan,
            f"{prefix}_p_old_pre": float(probs[a_old]) if a_old is not None else np.nan,
            f"{prefix}_step1_dnew": step1[0], f"{prefix}_step1_dold": step1[1],
            f"{prefix}_d_new0": dn[0], f"{prefix}_d_old0": do[0],
            f"{prefix}_min_d_new": float(np.nanmin(dn)), f"{prefix}_min_d_old": float(np.nanmin(do)),
            f"{prefix}_hit_new_step": first_hit(dn), f"{prefix}_hit_old_step": first_hit(do),
            f"{prefix}_enter_corridor": int(bool(cells & set(map(tuple, traj)))),
            f"{prefix}_done_at": -1 if done_at is None else done_at,
        })
        return out

    def offscreen_row(env, grid0, ag, old, probs, level, t, cx, cy):
        """Small TR extension off the screen corner, plus a length-matched BL-direction null, same state."""
        st = L.get_state(env)
        common = dict(is_walk=L._is_walk, wall=L.WALL_OBJ, bfs_path=L._bfs_path,
                      reach_extreme=L._reachable_extreme, min_sc=args.min_screen_cheb)
        nz = np.argwhere(grid0 != L.WALL_OBJ)
        row = {"label": label, "level": level, "t": t, "agent_x": ag[0], "agent_y": ag[1],
               "cheese_x": cx, "cheese_y": cy, "old_x": old[0], "old_y": old[1],
               "maze_hi_x": int(nz.max(0)[0]), "maze_hi_y": int(nz.max(0)[1]),
               "maze_lo_x": int(nz.min(0)[0]), "maze_lo_y": int(nz.min(0)[1]),
               "old_sc_cheb": max(SCREEN_CORNER[0] - old[0], SCREEN_CORNER[1] - old[1]),
               "argmax_pre": int(probs.argmax())}
        _tb, dist_b, _db = greedy_rollout(env, grid0, {"old": old}, args.rollout_steps)
        L.set_state(env, st)
        row["base_hit_old_step"] = first_hit(dist_b["old"])
        plan_tr = build_extension(grid0, ag, old, True, **common)
        row.update(edit_arm(env, st, grid0, ag, old, probs, plan_tr, "tr"))
        bl = L._reachable_extreme(grid0, ag, "bl")
        row["bl_anchor_x"], row["bl_anchor_y"] = bl
        plan_bl = build_extension(grid0, ag, bl, False, **common,
                                  match_len=len(plan_tr["cells"]) if "cells" in plan_tr else None)
        row.update(edit_arm(env, st, grid0, ag, old, probs, plan_bl, "null"))
        return row

    rows = []
    level = args.start_level
    while len(rows) < args.n_frames and level - args.start_level < args.max_levels:
        env = L.make_env("maze", start_level=level)
        obs = env.reset()
        prev = _chw_to_hwc01(obs)
        last_xy, kept = None, 0
        for t in range(args.max_steps):
            chw = np.asarray(obs[0], dtype=np.float32)
            probs = probs_of(chw)
            cur = _chw_to_hwc01(obs)
            cheese_xy = L.cheese_xy_bright(cur)
            axy = motion_agent_xy(prev, cur)
            if axy is not None:
                last_xy = axy
            info = L.get_info_maze(env)
            wd = L._iscalar(info, "maze_world_dim")
            cx, cy = L._iscalar(info, "cheese_x"), L._iscalar(info, "cheese_y")
            ag = L.get_agent_cell(env)
            a_ch = a_co = None
            if ag[0] >= 0 and cx >= 0:
                grid0 = L._read_grid(env, wd)
                old = L._reachable_extreme(grid0, ag, "tr")
                if old is not None and (cx, cy) in L._reachable(grid0, ag):
                    a_ch, _ = L.first_step_action(grid0, ag, (cx, cy))
                    a_co, _ = L.first_step_action(grid0, ag, old)
            # Same admission rule as collect() in graph mode; bearing-fallback frames are skipped
            # (they are ~0 of the saved bank and have no graph-defined old extreme).
            keep = (kept < 3 and last_xy is not None and cheese_xy is not None
                    and a_ch is not None and a_co is not None and a_ch != a_co)
            if keep and args.mode == "offscreen":
                rows.append(offscreen_row(env, grid0, ag, old, probs, level, t, cx, cy))
                kept += 1
            elif keep:
                st = L.get_state(env)
                new = L._extreme_cell(grid0, "tr")
                n_open, opened = L.carve_path_to(env, new)
                grid1 = L._read_grid(env, wd)
                reach1 = L._reachable(grid1, ag)
                new_reach = L._reachable_extreme(grid1, ag, "tr")
                a_new, len_new = L.first_step_action(grid1, ag, new)
                a_old, len_old = L.first_step_action(grid1, ag, old)
                ob1 = L.obs_after_edit(env)
                p1 = probs_of(ob1)
                traj, dist, done_at = greedy_rollout(env, grid1, {"new": new, "old": old}, args.rollout_steps)
                L.set_state(env, st)
                # Baseline: same state, unedited maze.
                _tb, dist_b, done_b = greedy_rollout(env, grid0, {"old": old}, args.rollout_steps)
                L.set_state(env, st)
                # Geometry-matched null (as in capture_corner_geometry): open as many walls toward
                # the bottom-left extreme. Separates "follow newly opened space" from "go top-right".
                bl = L._extreme_cell(grid0, "bl")
                n_bl, opened_bl = L.carve_path_to(env, bl, max_walls=max(1, n_open))
                grid_bl = L._read_grid(env, wd)
                traj_bl, _d, done_nb = greedy_rollout(env, grid_bl, {"bl": bl}, args.rollout_steps)
                L.set_state(env, st)

                def corridor_stats(tr_, cells):
                    cells = set(map(tuple, cells))
                    if not cells:
                        return 0, 0
                    far = max(cells, key=lambda c: abs(c[0] - ag[0]) + abs(c[1] - ag[1]))
                    seen = set(map(tuple, tr_))
                    return int(bool(seen & cells)), int(far in seen)
                ent_tr, deep_tr = corridor_stats(traj, opened)
                ent_bl, deep_bl = corridor_stats(traj_bl, opened_bl)
                nz = np.argwhere(grid0 != L.WALL_OBJ)
                maze_hi = nz.max(0)
                dn, do = np.asarray(dist["new"], float), np.asarray(dist["old"], float)
                step1 = (dn[1] - dn[0], do[1] - do[0]) if len(dn) > 1 else (np.nan, np.nan)

                def first_hit(arr):
                    hit = np.where(arr == 0)[0]
                    return int(hit[0]) if len(hit) else -1
                hit_new, hit_old = first_hit(dn), first_hit(do)
                rows.append({
                    "label": label, "level": level, "t": t,
                    "agent_x": ag[0], "agent_y": ag[1], "cheese_x": cx, "cheese_y": cy,
                    "old_x": old[0], "old_y": old[1], "new_x": new[0], "new_y": new[1],
                    "new_reach_x": new_reach[0], "new_reach_y": new_reach[1],
                    "maze_hi_x": int(maze_hi[0]), "maze_hi_y": int(maze_hi[1]),
                    "old_is_maze_corner": int(tuple(old) == (int(maze_hi[0]), int(maze_hi[1]))),
                    "new_outside_maze": int(new[0] > maze_hi[0] or new[1] > maze_hi[1]),
                    "extreme_moved": int(tuple(new) != tuple(old)),
                    "old_reachable_pre_new": int(tuple(new) in L._reachable(grid0, ag)),
                    "n_walls_opened": n_open, "d_reach": len(reach1) - len(L._reachable(grid0, ag)),
                    "len_new": len_new, "len_old": len_old,
                    "a_new": a_new, "a_old": a_old, "a_corner_pre": a_co, "a_cheese": a_ch,
                    "diagnostic": int(a_new is not None and a_old is not None and a_new != a_old),
                    "argmax_post": int(p1.argmax()), "argmax_pre": int(probs.argmax()),
                    "p_new_post": float(p1[a_new]) if a_new is not None else np.nan,
                    "p_old_post": float(p1[a_old]) if a_old is not None else np.nan,
                    "p_new_pre": float(probs[a_new]) if a_new is not None else np.nan,
                    "p_old_pre": float(probs[a_old]) if a_old is not None else np.nan,
                    "signed_corner_post": float(p1[a_new] - probs[a_new]) if a_new is not None else np.nan,
                    "step1_dnew": step1[0], "step1_dold": step1[1],
                    "d_new0": dn[0], "d_old0": do[0],
                    "min_d_new": float(np.nanmin(dn)), "min_d_old": float(np.nanmin(do)),
                    "end_d_new": dn[-1], "end_d_old": do[-1],
                    "hit_new_step": hit_new, "hit_old_step": hit_old,
                    "done_at": -1 if done_at is None else done_at, "n_steps": len(traj) - 1,
                    "base_hit_old_step": first_hit(np.asarray(dist_b["old"], float)),
                    "base_min_d_old": float(np.nanmin(np.asarray(dist_b["old"], float))),
                    "base_done_at": -1 if done_b is None else done_b,
                    "n_walls_null": n_bl, "enter_tr_corridor": ent_tr, "deep_tr_corridor": deep_tr,
                    "enter_bl_corridor": ent_bl, "deep_bl_corridor": deep_bl,
                })
                kept += 1
            act = int(probs.argmax())
            prev = cur
            obs, _r, done, _i = env.step(np.array([act]))
            if done[0]:
                break
        env.close()
        level += 1
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"raw_{label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(f"{label}: {len(rows)} states, levels {rows[0]['level']}..{rows[-1]['level']}", flush=True)


def seed_stats(vals):
    a = np.asarray(vals, float)
    m, sd = float(a.mean()), float(a.std(ddof=1))
    h = T975[len(a) - 1] * sd / np.sqrt(len(a))
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "per_seed": a.tolist()}


def summarize(args) -> None:
    import pandas as pd
    per = {}
    for lab in LABELS:
        df = pd.read_csv(args.out / f"raw_{lab}.csv")
        saved = json.load(open(SAVED / f"maze_{lab}.json"))["features"]["corner_open_TR"]["per_frame"]
        # State identity against the saved paper run: same level and same signed Δπ on a_corner_post.
        by_level = {}
        for r in saved:
            by_level.setdefault(int(r["level"]), []).append(float(r["signed"]))
        ident = float(np.mean([any(abs(v - s) < 1e-3 for s in by_level.get(int(lv), []))
                               for lv, v in zip(df.level, df.signed_corner_post)]))
        dg = df[df.diagnostic == 1]
        ok = dg[(dg.step1_dnew.notna())]
        s1_new = ((ok.step1_dnew < 0) & (ok.step1_dold >= 0)).mean()
        s1_old = ((ok.step1_dold < 0) & (ok.step1_dnew >= 0)).mean()
        hn, ho = dg.hit_new_step.values, dg.hit_old_step.values
        first_new = np.mean((hn >= 0) & ((ho < 0) | (hn < ho)))
        first_old = np.mean((ho >= 0) & ((hn < 0) | (ho < hn)))
        prog_new = np.mean((dg.d_new0 - dg.min_d_new) / dg.d_new0)
        prog_old = np.mean((dg.d_old0 - dg.min_d_old) / dg.d_old0)
        per[lab] = {
            "n_states": int(len(df)), "state_identity_vs_saved": ident,
            "frac_extreme_moved": float(df.extreme_moved.mean()),
            "frac_new_outside_original_maze": float(df.new_outside_maze.mean()),
            "frac_old_is_maze_corner": float(df.old_is_maze_corner.mean()),
            "frac_diagnostic": float(df.diagnostic.mean()), "n_diagnostic": int(len(dg)),
            "first_argmax_is_a_new": float((dg.argmax_post == dg.a_new).mean()),
            "first_argmax_is_a_old": float((dg.argmax_post == dg.a_old).mean()),
            "pre_argmax_is_a_old": float((dg.argmax_pre == dg.a_old).mean()),
            "p_new_post": float(dg.p_new_post.mean()), "p_old_post": float(dg.p_old_post.mean()),
            "p_new_pre": float(dg.p_new_pre.mean()), "p_old_pre": float(dg.p_old_pre.mean()),
            "step1_toward_new_only": float(s1_new), "step1_toward_old_only": float(s1_old),
            "reach_new_first": float(first_new), "reach_old_first": float(first_old),
            "reach_neither": float(1 - first_new - first_old),
            "progress_new": float(prog_new), "progress_old": float(prog_old),
            "base_reach_old": float((dg.base_hit_old_step >= 0).mean()),
            "episode_ended_in_rollout": float((dg.done_at >= 0).mean()),
            "moved_n": int(df.extreme_moved.sum()),
            "moved_reach_new": float((df[df.extreme_moved == 1].hit_new_step >= 0).mean()),
            "moved_reach_old": float((df[df.extreme_moved == 1].hit_old_step >= 0).mean()),
            "moved_reach_new_first": float(np.mean(((mv := df[df.extreme_moved == 1]).hit_new_step >= 0)
                                                   & ((mv.hit_old_step < 0) | (mv.hit_new_step < mv.hit_old_step)))),
            "moved_base_reach_old": float((mv.base_hit_old_step >= 0).mean()),
            "moved_enter_tr_corridor": float(mv.enter_tr_corridor.mean()),
            "moved_deep_tr_corridor": float(mv.deep_tr_corridor.mean()),
            "moved_enter_bl_corridor": float(mv[mv.n_walls_null > 0].enter_bl_corridor.mean()),
            "moved_deep_bl_corridor": float(mv[mv.n_walls_null > 0].deep_bl_corridor.mean()),
            "mean_d_new0": float(dg.d_new0.mean()), "mean_d_old0": float(dg.d_old0.mean()),
        }
    keys = [k for k in per[LABELS[0]]]
    coh = {c: {k: seed_stats([per[l][k] for l in labs]) for k in keys}
           for c, labs in (("GM", LABELS[:3]), ("NonGM", LABELS[3:]))}
    out = {"per_ckpt": per, "cohort": coh,
           "definitions": {
               "old": "reachable top-right extreme (max x+y among cells reachable from the agent) on the pre-edit maze",
               "new": "global interior extreme _extreme_cell(grid,'tr') that capture_corner_geometry carves to",
               "diagnostic": "a_new != a_old, BFS first steps on the edited grid",
               "step1": "actual greedy env transition; toward X = BFS distance to X decreases",
               "reach_first": "greedy rollout <= rollout_steps on the edited maze, which extreme is hit first",
               "progress": "(d0 - min_t d_t) / d0 over the rollout"}}
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    for k in ("frac_extreme_moved", "frac_new_outside_original_maze", "frac_diagnostic",
              "first_argmax_is_a_new", "first_argmax_is_a_old", "step1_toward_new_only",
              "step1_toward_old_only", "reach_new_first", "reach_old_first", "progress_new",
              "progress_old", "base_reach_old", "state_identity_vs_saved", "moved_reach_new",
              "moved_reach_old", "moved_reach_new_first", "moved_base_reach_old", "moved_enter_tr_corridor",
              "moved_deep_tr_corridor", "moved_enter_bl_corridor", "moved_deep_bl_corridor"):
        g, n = coh["GM"][k], coh["NonGM"][k]
        print(f"{k:34s} GM {g['mean']:.3f}±{g['sd']:.3f} [{g['ci95'][0]:.3f},{g['ci95'][1]:.3f}]  "
              f"NonGM {n['mean']:.3f}±{n['sd']:.3f} [{n['ci95'][0]:.3f},{n['ci95'][1]:.3f}]")


def summarize_offscreen(args) -> None:
    import pandas as pd
    per = {}
    for lab in LABELS:
        df = pd.read_csv(args.out / f"raw_{lab}.csv")
        r = {"n_states": int(len(df)), "frac_usable_tr": float(df.tr_ok.mean()),
             "frac_usable_both": float((df.tr_ok.astype(bool) & df.null_ok.astype(bool)).mean())}
        for arm in ("tr", "null"):
            # Both arms on the same states: those with a valid TR extension and a valid null.
            ok = df[(df.tr_ok == 1) & (df.null_ok == 1)]
            dg = ok[ok[f"{arm}_diagnostic"] == 1]
            hn, ho = ok[f"{arm}_hit_new_step"].values, ok[f"{arm}_hit_old_step"].values
            dhn, dho = dg[f"{arm}_hit_new_step"].values, dg[f"{arm}_hit_old_step"].values

            def firsts(a, b):
                return (float(np.mean((a >= 0) & ((b < 0) | (a < b)))) if len(a) else np.nan,
                        float(np.mean((b >= 0) & ((a < 0) | (b < a)))) if len(a) else np.nan)
            fn, fo = firsts(hn, ho)
            dfn, dfo = firsts(dhn, dho)
            r.update({
                f"{arm}_n_usable": int(len(ok)), f"{arm}_n_diagnostic": int(len(dg)),
                f"{arm}_frac_diagnostic": float(len(dg) / max(1, len(ok))),
                f"{arm}_step1_toward_new_only": np.nan if dg.empty else float(((dg[f"{arm}_step1_dnew"] < 0) & (dg[f"{arm}_step1_dold"] >= 0)).mean()),
                f"{arm}_step1_toward_old_only": np.nan if dg.empty else float(((dg[f"{arm}_step1_dold"] < 0) & (dg[f"{arm}_step1_dnew"] >= 0)).mean()),
                f"{arm}_diag_reach_new_first": dfn, f"{arm}_diag_reach_old_first": dfo,
                f"{arm}_all_reach_new_first": fn, f"{arm}_all_reach_old_first": fo,
                f"{arm}_all_reach_new_any": float((hn >= 0).mean()), f"{arm}_all_reach_old_any": float((ho >= 0).mean()),
                f"{arm}_enter_corridor": float(ok[f"{arm}_enter_corridor"].mean()),
                f"{arm}_p_new_pre": float(dg[f"{arm}_p_new_pre"].mean()), f"{arm}_p_new_post": float(dg[f"{arm}_p_new_post"].mean()),
                f"{arm}_p_old_pre": float(dg[f"{arm}_p_old_pre"].mean()), f"{arm}_p_old_post": float(dg[f"{arm}_p_old_post"].mean()),
                f"{arm}_len_corridor": float(ok[f"{arm}_len_corridor"].mean()),
                f"{arm}_d_new0": float(ok[f"{arm}_d_new0"].mean()), f"{arm}_d_old0": float(ok[f"{arm}_d_old0"].mean()),
                f"{arm}_base_reach_old": float((ok.base_hit_old_step >= 0).mean()),
            })
        ok = df[df.tr_ok == 1]
        r["tr_sc_cheb_min"] = float(ok.tr_sc_cheb.min()) if len(ok) else np.nan
        r["tr_tgt_sc_cheb_mean"] = float(ok.tr_tgt_sc_cheb.mean()) if len(ok) else np.nan
        both = df[(df.tr_ok == 1) & (df.null_ok == 1)]
        r["null_len_matched"] = float((both.null_len_corridor == both.tr_len_corridor).mean())
        per[lab] = r
    keys = list(per[LABELS[0]])
    coh = {c: {k: seed_stats([per[l][k] for l in labs]) for k in keys}
           for c, labs in (("GM", LABELS[:3]), ("NonGM", LABELS[3:]))}
    reasons = {}
    for lab in LABELS:
        df = pd.read_csv(args.out / f"raw_{lab}.csv")
        for v in df.loc[df.tr_ok == 0, "tr_reason"].fillna("?"):
            reasons[v] = reasons.get(v, 0) + 1
    out = {"per_ckpt": per, "cohort": coh, "tr_unusable_reasons_all_ckpts": reasons,
           "definitions": {
               "old": "reachable top-right extreme (max x+y) on the pre-edit maze",
               "tr target": "old+(2,0) or old+(0,2); insulated wall corridor that joins the maze at one open cell "
                            "that is not old; target is the unique new reachable TR extreme; agent's shortest path "
                            "to it avoids old; every carved cell >= min_screen_cheb Chebyshev from (23,23)",
               "null target": "same construction at the reachable BL extreme, old_bl-(2,0) or old_bl-(0,2), "
                              "corridor length matched to the TR corridor when possible; 'new' = null end, 'old' = TR old",
               "diagnostic": "a_new != a_old, BFS first steps on the edited grid",
               "step1": "actual greedy env transition; toward X = BFS distance to X decreases",
               "reach_first": "greedy rollout <= rollout_steps on the edited maze, which target is hit first"},
           "min_screen_cheb": args.min_screen_cheb}
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    for k in keys:
        g, n = coh["GM"][k], coh["NonGM"][k]
        print(f"{k:30s} GM {g['mean']:.3f}±{g['sd']:.3f} [{g['ci95'][0]:.3f},{g['ci95'][1]:.3f}]  "
              f"NonGM {n['mean']:.3f}±{n['sd']:.3f} [{n['ci95'][0]:.3f},{n['ci95'][1]:.3f}]")
    print("TR unusable reasons:", reasons)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mode", choices=["global", "offscreen"], default="global",
                    help="global: carve to world cell (23,23); offscreen: 2-cell TR extension off the screen corner")
    ap.add_argument("--min-screen-cheb", type=int, default=3,
                    help="offscreen: every carved cell must be at least this Chebyshev distance from (23,23)")
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--one", default=None, help="internal: run a single checkpoint in this process")
    ap.add_argument("--n-frames", type=int, default=140)
    ap.add_argument("--start-level", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--rollout-steps", type=int, default=100)
    ap.add_argument("--cpu", action="store_true", default=True)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = ROOT / ("data/probes/maze_corner_offscreen_20260925" if args.mode == "offscreen"
                           else "data/probes/maze_corner_new_vs_old_20260925")
    if args.one:
        run_one(args.one, args)
        return
    if not args.summarize_only:
        # One fresh process per checkpoint: looping procgen envs in one process leaks memory.
        for lab in args.labels.split(","):
            subprocess.run([sys.executable, __file__, "--one", lab, "--out", str(args.out),
                            "--n-frames", str(args.n_frames), "--start-level", str(args.start_level),
                            "--max-steps", str(args.max_steps), "--rollout-steps", str(args.rollout_steps),
                            "--mode", args.mode, "--min-screen-cheb", str(args.min_screen_cheb)],
                           check=True)
    summarize_offscreen(args) if args.mode == "offscreen" else summarize(args)


if __name__ == "__main__":
    main()
