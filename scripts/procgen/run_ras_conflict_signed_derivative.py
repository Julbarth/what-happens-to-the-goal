#!/usr/bin/env python3
"""Attribute Maze Fisher Q, cross-term and signed target-selection derivative at graph-exact line-vs-yellow conflict states (black backgrounds)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tnf
import yaml
from procgen import ProcgenEnv

ROOT = Path(__file__).resolve().parents[2]
for p in (
    ROOT / "src",
    ROOT / "external" / "colour-shape-goal-misgeneralization" / "train-procgen",
    ROOT / "external" / "colour-shape-goal-misgeneralization" / "procgenAISC",
):
    if p.exists():
        sys.path.insert(0, str(p))

OBJ_LINE = "red_line_diag"
OBJ_GEM = "yellow_gem"
SHAPE_EDIT_OBJ = "red_gem"      # ablate line's shape, keep its colour (red)
COLOUR_EDIT_OBJ = "green_gem"   # ablate gem's colour, keep its shape (gem)
WORLD_DIM = 5
MARGIN = 3


from gm_geometry.probes.choice_point import localize_dual_objects  # noqa: E402
from gm_geometry.probes.fisher import make_env, rgb_hwc, to_nchw  # noqa: E402,F401
from gm_geometry.probes.ras_agent_localization import (  # noqa: E402
    bfs_distances,
    calibrate_action_deltas,
    calibrate_pixel_to_grid,
    first_action_set,
    locate_agent_cell,
    probe_wall_grid,
)
from gm_geometry.probes import ras_bank_io  # noqa: E402



def unwrap_base(env):
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "get_state") and callable(cur.get_state):
            if hasattr(cur, "set_object") or not hasattr(cur, "env"):
                return cur
        cur = getattr(cur, "env", getattr(cur, "venv", getattr(cur, "_env", None)))
    raise RuntimeError("unwrap failed")


def state_api(env):
    return unwrap_base(env)


def state_bytes(env) -> bytes:
    """Layout and pose fingerprint, used to confirm the shape/colour edits landed on the same maze layout as the base scene."""
    st = unwrap_base(env).get_state()
    if isinstance(st, (list, tuple)):
        parts = []
        for s in st:
            if isinstance(s, (bytes, bytearray)):
                parts.append(bytes(s))
            else:
                parts.append(np.asarray(s).tobytes())
        return b"||".join(parts)
    if isinstance(st, (bytes, bytearray)):
        return bytes(st)
    return np.asarray(st).tobytes()


def find_conflict_state_at_level(level: int, distribution_mode: str, use_backgrounds: bool):
    """Ground-truth-calibrated conflict-state search at one level; None unless line and gem are graph-equidistant with diverging first moves."""
    env = make_env(OBJ_LINE, OBJ_GEM, level, rand_seed=0,
                    distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    try:
        obs0 = env.reset()
        base_sbytes = state_bytes(env)
        api = state_api(env)

        # Captured before wall-probing, which clears and moves object 1, so this is the level's real spawned position.
        loc0 = localize_dual_objects(to_nchw(obs0), obj1=OBJ_LINE, obj2=OBJ_GEM,
                                      use_backgrounds=use_backgrounds)
        line_xy0, gem_xy0 = loc0.get("line_xy"), loc0.get("gem_xy")
        if line_xy0 is None or gem_xy0 is None:
            return None

        coord_range = range(-MARGIN, WORLD_DIM + MARGIN)
        walkable = probe_wall_grid(api, coord_range, coord_range)
        if len(walkable) < 4:
            return None
        calib, _ = calibrate_pixel_to_grid(api, env, walkable, OBJ_LINE, OBJ_GEM)
        if calib is None:
            return None
        action_deltas = calibrate_action_deltas(api, env, obs0, calib)
        found = {d for d in action_deltas.values() if d is not None}
        if not any((-d[0], -d[1]) in found for d in found):
            return None  # locate_agent_cell needs a real opposite pair
        agent_cell = locate_agent_cell(api, env, obs0, action_deltas, calib, walkable)
        if agent_cell is None:
            return None

        def px_to_grid(xy):
            aug = np.array([xy[0], xy[1], 1.0])
            est = aug @ calib
            return (int(round(est[0])), int(round(est[1])))

        line_cell, gem_cell = px_to_grid(line_xy0), px_to_grid(gem_xy0)
        if line_cell not in walkable or gem_cell not in walkable:
            return None
        if line_cell == agent_cell or gem_cell == agent_cell:
            return None

        dist_line_map = bfs_distances(walkable, line_cell)
        dist_gem_map = bfs_distances(walkable, gem_cell)
        d_line, d_gem = dist_line_map.get(agent_cell), dist_gem_map.get(agent_cell)
        if d_line is None or d_gem is None or d_line != d_gem or d_line == 0:
            return None

        aa = first_action_set(dist_line_map, agent_cell, action_deltas)
        ab = first_action_set(dist_gem_map, agent_cell, action_deltas)
        if not aa or not ab or not aa.isdisjoint(ab):
            return None

        return {
            "level": int(level),
            "obs0": obs0,
            "state_bytes": base_sbytes,
            "dist_line": int(d_line),
            "dist_gem": int(d_gem),
            "first_actions_line": sorted(aa),
            "first_actions_gem": sorted(ab),
            # Grid-direction tuples (not action ids) give callers a small, clusterable bearing descriptor;
            # the lowest action id per side determines the tuple.
            "first_group_line": action_deltas[min(aa)],
            "first_group_gem": action_deltas[min(ab)],
            "n_cells_explored": len(walkable),
            # Per-level geometry for callers that need the bearing at later rollout steps.
            # Not JSON-serializable (set / ndarray).
            "geometry": {
                "walkable": walkable, "calib": calib, "action_deltas": action_deltas,
                "agent_cell": agent_cell, "line_cell": line_cell, "gem_cell": gem_cell,
            },
        }
    finally:
        env.close()


def action_group_probs(probs: np.ndarray, ids: list[int]) -> float:
    return float(np.sum(probs[ids])) if ids else float("nan")


def dP_group_dh(grad_cell: np.ndarray, probs: np.ndarray, group_ids: list[int]) -> np.ndarray:
    """Analytic dP(group)/dh at fc; grad_cell is the constant [A,D] logit-Jacobian, so no finite differencing is needed."""
    mean_grad = (probs[:, None] * grad_cell).sum(axis=0)
    contrib = np.zeros_like(mean_grad)
    for a in group_ids:
        contrib += probs[a] * (grad_cell[a] - mean_grad)
    return contrib


def fc_logit_jacobian_and_act(policy, obs_nchw: np.ndarray, device: str):
    """d(logit_a)/d(fc), probs, fc activation h. Shapes [A,D], [A], [D]."""
    store: dict[str, torch.Tensor] = {}

    def _hook(_m, _i, out):
        store["h"] = out
        return out

    handle = policy.embedder.fc.register_forward_hook(_hook)
    try:
        x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        m = torch.zeros(1, device=device)
        dist, _value, _nh = policy(x, h0, m)
        h = store["h"]
        logits = dist.logits
        n_act = int(logits.shape[1])
        grads = []
        for a in range(n_act):
            g_a = torch.autograd.grad(logits[0, a], h, retain_graph=(a < n_act - 1), allow_unused=False)[0]
            grads.append(g_a.detach()[0].cpu().numpy())
        j = np.stack(grads, axis=0)
        with torch.no_grad():
            probs = tnf.softmax(logits, dim=-1)[0].cpu().numpy()
        return j, probs, h.detach()[0].cpu().numpy()
    finally:
        handle.remove()


def probs_from_overridden_h(policy, h_override: np.ndarray, device: str) -> np.ndarray:
    """Forward pass with fc's output replaced by h_override, used only for the finite-difference check."""
    override = torch.as_tensor(h_override, dtype=torch.float32, device=device).unsqueeze(0)

    def _hook(_m, _i, _out):
        return override

    handle = policy.embedder.fc.register_forward_hook(_hook)
    try:
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        m = torch.zeros(1, device=device)
        with torch.no_grad():
            dist, _value, _nh = policy(dummy, h0, m)
            probs = tnf.softmax(dist.logits, dim=-1)[0].cpu().numpy()
        return probs
    finally:
        handle.remove()


def fisher_from_logit_jacobian(jacobian: np.ndarray, probs: np.ndarray) -> np.ndarray:
    j = np.asarray(jacobian, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    w = np.diag(p) - np.outer(p, p)
    return j.T @ w @ j


def quadratic_form(f_mat: np.ndarray, v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(v @ f_mat @ v)


def bilinear_form(f_mat: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(u @ f_mat @ v)


def load_policy(ckpt: Path, device: str, *, root: Path, random_init: bool = False, init_seed: int = 0):
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy

    train_root = None
    for cand in (root / "external" / "colour-shape-goal-misgeneralization" / "train-procgen",
                 root / "external" / "train-procgen-pytorch"):
        if (cand / "hyperparams" / "procgen" / "config.yml").exists():
            train_root = cand
            break
    if train_root is None:
        raise SystemExit("missing train-procgen hyperparams")
    with open(train_root / "hyperparams/procgen/config.yml") as f:
        recurrent = yaml.safe_load(f)["hard-500"]["recurrent"]
    probe = make_env(OBJ_LINE, OBJ_GEM, 0, 0)
    try:
        n_act = int(probe.action_space.n)
    finally:
        probe.close()
    if random_init:
        torch.manual_seed(int(init_seed))
        np.random.seed(int(init_seed))
    model = ImpalaModel(in_channels=3)
    policy = CategoricalPolicy(model, recurrent, n_act)
    if not random_init:
        blob = torch.load(ckpt, map_location="cpu")
        policy.load_state_dict(blob["model_state_dict"])
    policy.to(device).eval()
    return policy, n_act


def find_conflict_levels(
    *,
    start_level: int,
    n_levels_to_try: int,
    n_target: int,
    distribution_mode: str,
    use_backgrounds: bool,
) -> list[dict]:
    """Policy-free conflict-level harvest shared across every scored seed, calibrated per level."""
    matched = []
    stats = {"tried": 0, "rejected": 0, "matched": 0}
    level = start_level
    while level < start_level + n_levels_to_try and len(matched) < n_target:
        stats["tried"] += 1
        res = find_conflict_state_at_level(level, distribution_mode, use_backgrounds)
        level += 1
        if res is None:
            stats["rejected"] += 1
            continue
        matched.append(res)
        stats["matched"] += 1
        if len(matched) % 5 == 0:
            print(f"  matched {len(matched)}/{n_target} (tried={stats['tried']}) stats={stats}", flush=True)
    print(f"conflict-level search done: matched={len(matched)} stats={stats}", flush=True)
    return matched


def score_seed(policy, matched: list[dict], distribution_mode: str, use_backgrounds: bool,
               device: str, *, n_findiff: int, findiff_eps_frac: float) -> list[dict]:
    rows = []
    for i, res in enumerate(matched):
        level = res["level"]
        ids_line = res["first_actions_line"]
        ids_gem = res["first_actions_gem"]

        obs_base = res["obs0"]
        env_shape = make_env(SHAPE_EDIT_OBJ, OBJ_GEM, level, rand_seed=0,
                              distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
        env_colour = make_env(OBJ_LINE, COLOUR_EDIT_OBJ, level, rand_seed=0,
                               distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
        try:
            obs_shape = env_shape.reset()
            obs_colour = env_colour.reset()
            sb_shape = state_bytes(env_shape)
            sb_colour = state_bytes(env_colour)
        finally:
            env_shape.close()
            env_colour.close()

        if sb_shape != res["state_bytes"] or sb_colour != res["state_bytes"]:
            print(f"  [layout mismatch] level={level} skipping -- shape/colour edit did not "
                  f"land on the same maze layout as the base scene", flush=True)
            continue

        j_base, probs_base, h_base = fc_logit_jacobian_and_act(policy, to_nchw(obs_base), device)
        _, _, h_shape = fc_logit_jacobian_and_act(policy, to_nchw(obs_shape), device)
        _, _, h_colour = fc_logit_jacobian_and_act(policy, to_nchw(obs_colour), device)

        f_base = fisher_from_logit_jacobian(j_base, probs_base)
        d_s = (h_shape - h_base).astype(np.float64)
        d_c = (h_colour - h_base).astype(np.float64)
        n_s = float(np.linalg.norm(d_s)) + 1e-18
        n_c = float(np.linalg.norm(d_c)) + 1e-18

        q_s = quadratic_form(f_base, d_s)
        q_c = quadratic_form(f_base, d_c)
        c_sc = bilinear_form(f_base, d_s, d_c)
        cos_f = c_sc / (np.sqrt(max(q_s, 0.0) * max(q_c, 0.0)) + 1e-18)

        # S = Q/||Delta||^2 is the sensitivity_along definition, inlined.
        trfd = float(np.trace(f_base)) / f_base.shape[0]
        s_s = q_s / (n_s ** 2)
        s_c = q_c / (n_c ** 2)
        g_s_gain = s_s / (trfd + 1e-18)
        g_c_gain = s_c / (trfd + 1e-18)
        d_f_s = n_s ** 2 * trfd
        d_f_c = n_c ** 2 * trfd

        g_r_vec = dP_group_dh(j_base, probs_base, ids_line) - dP_group_dh(j_base, probs_base, ids_gem)
        g_s = float(np.dot(g_r_vec, d_s))
        g_c = float(np.dot(g_r_vec, d_c))
        ghat_s = g_s / n_s
        ghat_c = g_c / n_c
        r_base = action_group_probs(probs_base, ids_line) - action_group_probs(probs_base, ids_gem)

        row = {
            "level": int(level),
            "dist": res["dist_line"],
            "n_cells_explored": res["n_cells_explored"],
            "Q_shape": q_s, "Q_colour": q_c,
            "C_sc": c_sc, "cos_F": cos_f,
            "S_shape": s_s, "S_colour": s_c,
            "trFd": trfd,
            "G_shape": g_s_gain, "G_colour": g_c_gain,
            "D_F_shape": d_f_s, "D_F_colour": d_f_c,
            "g_shape": g_s, "g_colour": g_c,
            "ghat_shape": ghat_s, "ghat_colour": ghat_c,
            "norm_delta_shape": n_s, "norm_delta_colour": n_c,
            "r_base": r_base,
        }

        if i < n_findiff:
            eps_s = findiff_eps_frac * n_s
            eps_c = findiff_eps_frac * n_c
            u_s, u_c = d_s / n_s, d_c / n_c
            r_line_gem = lambda probs: (
                action_group_probs(probs, ids_line) - action_group_probs(probs, ids_gem)
            )
            r_plus_s = r_line_gem(probs_from_overridden_h(policy, h_base + eps_s * u_s, device))
            r_minus_s = r_line_gem(probs_from_overridden_h(policy, h_base - eps_s * u_s, device))
            r_plus_c = r_line_gem(probs_from_overridden_h(policy, h_base + eps_c * u_c, device))
            r_minus_c = r_line_gem(probs_from_overridden_h(policy, h_base - eps_c * u_c, device))
            fd_ghat_s = (r_plus_s - r_minus_s) / (2 * eps_s)
            fd_ghat_c = (r_plus_c - r_minus_c) / (2 * eps_c)
            row["findiff_check"] = {
                "eps_s": eps_s, "eps_c": eps_c,
                "fd_ghat_shape": float(fd_ghat_s), "fd_ghat_colour": float(fd_ghat_c),
                "analytic_ghat_shape": ghat_s, "analytic_ghat_colour": ghat_c,
                "rel_err_shape": float(abs(fd_ghat_s - ghat_s) / (abs(ghat_s) + 1e-9)),
                "rel_err_colour": float(abs(fd_ghat_c - ghat_c) / (abs(ghat_c) + 1e-9)),
            }
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-level", type=int, default=2000)
    ap.add_argument("--n-levels-to-try", type=int, default=800)
    ap.add_argument("--n-target", type=int, default=24)
    ap.add_argument("--n-findiff", type=int, default=8,
                     help="Finite-difference validation on this many levels, first seed only.")
    ap.add_argument("--findiff-eps-frac", type=float, default=0.02,
                     help="Perturbation size as a fraction of ||Delta|| (unit-direction finite diff).")
    ap.add_argument("--distribution-mode", type=str, default="easy")
    ap.add_argument("--use-backgrounds", action="store_true", default=False)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--skip-init", action="store_true")
    ap.add_argument("--gm-seeds", type=str, required=True, help="Comma list, e.g. 6313,6315,6317")
    ap.add_argument("--deco-seeds", type=str, required=True, help="Comma list, e.g. 4371,4372,4373")
    ras_bank_io.add_bank_args(ap)
    args = ap.parse_args()
    use_bank = ras_bank_io.check_bank_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    gm_seeds = [int(s) for s in args.gm_seeds.replace(",", " ").split()]
    deco_seeds = [int(s) for s in args.deco_seeds.replace(",", " ").split()]

    if args.use_backgrounds and not use_bank:
        raise SystemExit(
            "This script's agent localization is calibrated against a colour-based "
            "object localizer (choice_point.localize_dual_objects) that doesn't work "
            "under textured backgrounds. Use run_ras_conflict_signed_derivative_texture.py "
            "for the texture cohort instead."
        )

    bank_prov = None
    if use_bank:
        # The bank replaces localization entirely, so the texture restriction above does not apply.
        matched, bank_prov, _ = ras_bank_io.bank_res_list(
            args, args.split_role, state_bytes=state_bytes, use_backgrounds=args.use_backgrounds,
            make_base_env=lambda lvl: make_env(OBJ_LINE, OBJ_GEM, lvl, 0, args.distribution_mode, args.use_backgrounds))
    else:
        matched = find_conflict_levels(
            start_level=args.start_level,
            n_levels_to_try=args.n_levels_to_try,
            n_target=args.n_target,
            distribution_mode=args.distribution_mode,
            use_backgrounds=args.use_backgrounds,
        )
    if len(matched) < 5 and not use_bank:
        raise SystemExit(f"too few conflict levels matched ({len(matched)}) -- widen --n-levels-to-try")

    results: dict[str, list[dict]] = {}
    for s in gm_seeds:
        ck = args.ckpt_dir / f"{s}_gm.pth"
        if not ck.is_file():
            raise SystemExit(f"missing {ck}")
        policy, _ = load_policy(ck, args.device, root=ROOT)
        rows = score_seed(policy, matched, args.distribution_mode, args.use_backgrounds,
                           args.device, n_findiff=(args.n_findiff if s == gm_seeds[0] else 0),
                           findiff_eps_frac=args.findiff_eps_frac)
        results[f"gm_{s}"] = rows
        print(f"scored gm_{s}: n={len(rows)} "
              f"Q_shape_median={np.median([r['Q_shape'] for r in rows]):.4g} "
              f"Q_colour_median={np.median([r['Q_colour'] for r in rows]):.4g} "
              f"g_shape_mean={np.mean([r['g_shape'] for r in rows]):.4g} "
              f"g_colour_mean={np.mean([r['g_colour'] for r in rows]):.4g}", flush=True)

    for s in deco_seeds:
        ck = args.ckpt_dir / f"{s}_teach.pth"
        if not ck.is_file():
            raise SystemExit(f"missing {ck}")
        policy, _ = load_policy(ck, args.device, root=ROOT)
        rows = score_seed(policy, matched, args.distribution_mode, args.use_backgrounds,
                           args.device, n_findiff=0, findiff_eps_frac=args.findiff_eps_frac)
        results[f"deco_{s}"] = rows
        print(f"scored deco_{s}: n={len(rows)} "
              f"Q_shape_median={np.median([r['Q_shape'] for r in rows]):.4g} "
              f"Q_colour_median={np.median([r['Q_colour'] for r in rows]):.4g} "
              f"g_shape_mean={np.mean([r['g_shape'] for r in rows]):.4g} "
              f"g_colour_mean={np.mean([r['g_colour'] for r in rows]):.4g}", flush=True)

    if not args.skip_init:
        for i in range(args.n_init):
            policy, _ = load_policy(
                args.ckpt_dir / f"{gm_seeds[0]}_gm.pth", args.device,
                root=ROOT, random_init=True, init_seed=3000 + i,
            )
            rows = score_seed(policy, matched, args.distribution_mode, args.use_backgrounds,
                               args.device, n_findiff=0, findiff_eps_frac=args.findiff_eps_frac)
            results[f"init_{i}"] = rows

    out = {
        "obj_line": OBJ_LINE, "obj_gem": OBJ_GEM,
        "shape_edit_obj": SHAPE_EDIT_OBJ, "colour_edit_obj": COLOUR_EDIT_OBJ,
        "distribution_mode": args.distribution_mode, "use_backgrounds": args.use_backgrounds,
        "gm_seeds": gm_seeds, "deco_seeds": deco_seeds,
        "n_conflict_levels": len(matched),
        "agent_localization": "ground_truth_calibrated_v2_20260923",
        "conflict_levels": [
            {"level": m["level"], "dist": m["dist_line"],
             "first_actions_line": list(m["first_actions_line"]),
             "first_actions_gem": list(m["first_actions_gem"])}
            for m in matched
        ],
        "results": results,
    }
    if use_bank:
        for rows in results.values():
            ras_bank_io.annotate_rows(rows, matched)
        ras_bank_io.annotate_rows(out["conflict_levels"], matched)
        out["bank_provenance"] = bank_prov
        out["agent_localization"] = "canonical_bank"
    else:
        out["LEGACY_HARVEST_WARNING"] = ras_bank_io.LEGACY_HARVEST_WARNING
    (args.out / "conflict_signed_derivative.json").write_text(json.dumps(out, indent=2))
    print("wrote", args.out / "conflict_signed_derivative.json", flush=True)


if __name__ == "__main__":
    main()
