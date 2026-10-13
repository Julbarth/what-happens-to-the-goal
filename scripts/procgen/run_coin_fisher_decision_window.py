#!/usr/bin/env python3
"""Harvest Coin decision-window frames and score Fisher sensitivity across GM, Non-GM and init policies."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from gm_geometry.probes.coin_location import (
    RIGHT_ACTIONS,
    agent_feat_yx_camera,
    agent_feat_yx_from_motion,
    find_floor_cell,
    get_info,
    labels_from_info,
    obs_nchw_after_set_coin,
    pixel_diff_feat_cell,
    set_coin,
)
from run_coin_fisher_ahead_ordinary import score_frame, summarize
from run_coin_fisher_sensitivity import PIXEL_DIFF_THR, load_init_policy, load_policy
from validate_langosco_coinrun_200m import make_env


def _argmax_action(policy, obs_nchw, device: str) -> int:
    with torch.no_grad():
        o = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
        if o.ndim == 3:
            o = o.unsqueeze(0)
        hidden = torch.zeros(
            (1, policy.embedder.output_dim), dtype=torch.float32, device=device
        )
        dist, _v, _h = policy(o, hidden, torch.zeros(1, device=device))
        return int(dist.probs.argmax(-1).item())


def harvest_decision_windows(
    *,
    n_frames: int,
    start_level: int,
    max_levels: int,
    warm_policy,
    device: str,
    min_agent_x: float = 8.0,
    warm_cap: int = 80,
    min_dx: int = 1,
) -> list[dict]:
    """Ahead placements (vertical-first floor search) where the warm policy's argmax a_coin is not a RIGHT action.

    No on_path_band gate: the band starts at column 4, so ahead coins are in-band by construction.
    """
    rows: list[dict] = []
    level = start_level
    used = 0
    rejects = Counter()
    while len(rows) < n_frames and used < max_levels:
        env = make_env("coinrun_aisc", num_levels=1, start_level=level)
        obs = env.reset()
        hidden = np.zeros((1, warm_policy.embedder.output_dim), dtype=np.float32)
        lab0 = labels_from_info(get_info(env))
        for _ in range(warm_cap):
            with torch.no_grad():
                dist, _, nh = warm_policy(
                    torch.as_tensor(obs, dtype=torch.float32, device=device),
                    torch.as_tensor(hidden, dtype=torch.float32, device=device),
                    torch.zeros(1, device=device),
                )
                act = dist.probs.argmax(-1).cpu().numpy()
                hidden = nh.cpu().numpy()
            obs, _r, done, _i = env.step(act)
            if done[0]:
                rejects["done_warm"] += 1
                break
            lab0 = labels_from_info(get_info(env))
            if lab0 is not None and float(lab0["agent_x"]) >= min_agent_x:
                break
        else:
            lab0 = labels_from_info(get_info(env))

        used += 1
        if lab0 is None or float(lab0["agent_x"]) < min_agent_x - 0.5:
            rejects["ax_low"] += 1
            env.close()
            level += 1
            continue

        # Motion residual for per-frame agent feat (camera-lag check).
        o_pre = obs[0] if getattr(obs, "ndim", 0) == 4 else obs
        if o_pre.shape[-1] in (3, 4) and o_pre.shape[0] not in (3, 4):
            o_pre = np.transpose(o_pre, (2, 0, 1))
        obs_m, _r, done_m, _i = env.step(np.array([4], dtype=np.int64))  # jump: no scroll
        if done_m[0]:
            rejects["done_motion"] += 1
            env.close()
            level += 1
            continue
        o_post = obs_m[0] if obs_m.ndim == 4 else obs_m
        if o_post.shape[-1] in (3, 4) and o_post.shape[0] not in (3, 4):
            o_post = np.transpose(o_post, (2, 0, 1))
        agent_motion = agent_feat_yx_from_motion(o_pre[:3], o_post[:3])
        lab0 = labels_from_info(get_info(env)) or lab0

        placed = None
        for md in (min_dx, min_dx + 1, min_dx + 2, 1, 2, 3):
            xy = find_floor_cell(
                env,
                prefer_left_of_agent=False,
                min_dx=md,
                vertical_first=True,
            )
            if xy is None:
                continue
            if set_coin(env, xy[0], xy[1], 1) != 0:
                rejects["set_fail"] += 1
                continue
            obs_a = obs_nchw_after_set_coin(env)
            if set_coin(env, xy[0], xy[1], 0) != 0:
                rejects["set_fail"] += 1
                continue
            obs_g = obs_nchw_after_set_coin(env)
            set_coin(env, xy[0], xy[1], 1)
            if obs_a.ndim == 4:
                obs_a = obs_a[0]
            if obs_g.ndim == 4:
                obs_g = obs_g[0]
            if float(np.max(np.abs(obs_a - obs_g))) < PIXEL_DIFF_THR:
                rejects["pixel_gate"] += 1
                continue
            cell = pixel_diff_feat_cell(obs_a, obs_g, thr=PIXEL_DIFF_THR)
            if cell is None:
                rejects["pixel_gate"] += 1
                continue
            a_coin = _argmax_action(warm_policy, obs_a, device)
            if a_coin in RIGHT_ACTIONS:
                rejects["a_coin_right"] += 1
                continue

            # Relocation counterfactual: same level and layout, coin moved to the mirror-side floor cell.
            reloc_xy = None
            for rmd in (min_dx, min_dx + 1, min_dx + 2, 1, 2, 3):
                reloc_xy = find_floor_cell(
                    env, prefer_left_of_agent=True, min_dx=rmd, vertical_first=True,
                )
                if reloc_xy is not None:
                    break
            if reloc_xy is None or set_coin(env, reloc_xy[0], reloc_xy[1], 1) != 0:
                rejects["reloc_fail"] += 1
                continue
            obs_reloc = obs_nchw_after_set_coin(env)
            if obs_reloc.ndim == 4:
                obs_reloc = obs_reloc[0]
            set_coin(env, reloc_xy[0], reloc_xy[1], 0)
            set_coin(env, xy[0], xy[1], 1)

            lab_a = labels_from_info(get_info(env)) or lab0
            placed = {
                "level": int(level),
                "obs_a": obs_a.astype(np.float32),
                "obs_gone": obs_g.astype(np.float32),
                "obs_reloc": obs_reloc.astype(np.float32),
                "reloc_xy": (int(reloc_xy[0]), int(reloc_xy[1])),
                "coin_cell_a": (int(cell[0]), int(cell[1])),
                "xy_a": (int(xy[0]), int(xy[1])),
                "agent_xy": (float(lab_a["agent_x"]), float(lab_a["agent_y"])),
                "coin_xy": (float(lab_a["coin_x"]), float(lab_a["coin_y"])),
                "world_dx": float(lab_a["coin_x"] - lab_a["agent_x"]),
                "world_dy": float(lab_a["coin_y"] - lab_a["agent_y"]),
                "a_coin": int(a_coin),
                "agent_feat_camera": agent_feat_yx_camera(),
                "agent_feat_motion": agent_motion,
            }
            break
        env.close()
        if placed is None:
            rejects["no_xy"] += 1
        else:
            rows.append(placed)
            if len(rows) % 8 == 0:
                print(
                    f"  harvested {len(rows)}/{n_frames} rejects={dict(rejects)}",
                    flush=True,
                )
        level += 1
    print(f"harvest done n={len(rows)} used={used} rejects={dict(rejects)}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=48)
    p.add_argument("--start-level", type=int, default=1_100_000)
    p.add_argument("--max-levels", type=int, default=8000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-init", action="store_true")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    ck_nongm = [
        args.ckpt_dir / "7001_teach.pth",
        args.ckpt_dir / "7002_teach.pth",
        args.ckpt_dir / "7003_teach.pth",
    ]
    ck_gm = [
        args.ckpt_dir / "6033_gm.pth",
        args.ckpt_dir / "6034_gm.pth",
        args.ckpt_dir / "6035_gm.pth",
    ]
    for c in ck_nongm + ck_gm:
        if not c.is_file():
            raise SystemExit(f"missing ckpt {c}")

    warm = load_policy(ck_nongm[0], device)
    print("=== harvest decision-windows (no on_path_band) ===", flush=True)
    frames = harvest_decision_windows(
        n_frames=args.n_frames,
        start_level=args.start_level,
        max_levels=args.max_levels,
        warm_policy=warm,
        device=device,
    )
    if len(frames) < 16:
        raise SystemExit(f"too few decision frames: {len(frames)}")

    meta = {
        "n": len(frames),
        "frac_dx_positive": float(np.mean([f["world_dx"] > 0 for f in frames])),
        "mean_abs_dy": float(np.mean([abs(f["world_dy"]) for f in frames])),
        "frac_abs_dy_ge2": float(np.mean([abs(f["world_dy"]) >= 2 for f in frames])),
        "a_coin_counts": dict(Counter(int(f["a_coin"]) for f in frames)),
        "on_path_band_gate": False,
        "vertical_first": True,
        "divergence_gate": "a_coin not in RIGHT_ACTIONS",
    }
    # Camera-lag check on kept frames
    mot_errs = []
    for f in frames:
        m = f.get("agent_feat_motion")
        if m is not None:
            mot_errs.append(abs(float(m[1]) - float(f["agent_feat_camera"][1])))
    meta["motion_vs_camera_abs_err_x_mean"] = (
        float(np.mean(mot_errs)) if mot_errs else None
    )
    meta["motion_vs_camera_abs_err_x_max"] = (
        float(np.max(mot_errs)) if mot_errs else None
    )
    print("meta", meta, flush=True)

    agents = {}
    for lab, path in [
        ("nongm_7001", ck_nongm[0]),
        ("nongm_7002", ck_nongm[1]),
        ("nongm_7003", ck_nongm[2]),
        ("gm_6033", ck_gm[0]),
        ("gm_6034", ck_gm[1]),
        ("gm_6035", ck_gm[2]),
    ]:
        pol = load_policy(path, device)
        scored = [
            score_frame(pol, f["obs_a"], f["obs_gone"], device) for f in frames
        ]
        agents[lab] = summarize(scored)
        print(lab, agents[lab], flush=True)

    if not args.skip_init:
        init_p = load_init_policy(ck_nongm[0], device)
        agents["init"] = summarize(
            [score_frame(init_p, f["obs_a"], f["obs_gone"], device) for f in frames]
        )
        print("init", agents["init"], flush=True)

    payload = {
        "window": "decision",
        "prereg": None,
        "meta": meta,
        "agents": agents,
        "frame_meta": [
            {
                "level": f["level"],
                "coin_cell_a": f["coin_cell_a"],
                "xy_a": f["xy_a"],
                "world_dx": f["world_dx"],
                "world_dy": f["world_dy"],
                "a_coin": f["a_coin"],
                "agent_feat_camera": list(f["agent_feat_camera"]),
                "agent_feat_motion": list(f["agent_feat_motion"])
                if f["agent_feat_motion"] is not None
                else None,
            }
            for f in frames
        ],
    }
    (out / "decision_window_v1.json").write_text(json.dumps(payload, indent=2))
    print("BEGIN_SUMMARY", flush=True)
    print(json.dumps({"meta": meta, "agents": agents}), flush=True)
    print("END_SUMMARY", flush=True)


if __name__ == "__main__":
    main()
