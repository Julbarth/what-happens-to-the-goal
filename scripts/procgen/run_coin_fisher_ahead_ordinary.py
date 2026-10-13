#!/usr/bin/env python3
"""Ahead-ordinary Coin Fisher: headline Q/S/dh/T at block3+fc, in-distribution present/absent, multi-seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gm_geometry.probes.coin_location import (
    find_floor_cell,
    get_info,
    labels_from_info,
    obs_nchw_after_set_coin,
    pixel_diff_feat_cell,
    set_coin,
)
from gm_geometry.probes.fisher import dual_layer_scalars, summarize_scalar_rows
from gm_geometry.utils import ci95
from gm_geometry.utils import device as _default_device
from run_coin_fisher_sensitivity import PIXEL_DIFF_THR, load_init_policy, load_policy
from validate_langosco_coinrun_200m import make_env

# No off-path or path-band gate: every agent is scored on the same frame set.


def harvest_ahead_ordinary(
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
    rows: list[dict] = []
    level = start_level
    used = 0
    rejects = {
        "done_warm": 0,
        "ax_low": 0,
        "no_xy": 0,
        "set_fail": 0,
        "pixel_gate": 0,
    }
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

        placed = None
        for md in (min_dx, min_dx + 1, min_dx + 2, 1, 2, 3):
            xy = find_floor_cell(env, prefer_left_of_agent=False, min_dx=md)
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
            lab_a = labels_from_info(get_info(env)) or lab0
            placed = {
                "level": int(level),
                "obs_a": obs_a.astype(np.float32),
                "obs_gone": obs_g.astype(np.float32),
                "coin_cell_a": (int(cell[0]), int(cell[1])),
                "xy_a": (int(xy[0]), int(xy[1])),
                "agent_xy": (float(lab_a["agent_x"]), float(lab_a["agent_y"])),
                "coin_xy": (float(lab_a["coin_x"]), float(lab_a["coin_y"])),
                "world_dx": float(lab_a["coin_x"] - lab_a["agent_x"]),
                "world_dy": float(lab_a["coin_y"] - lab_a["agent_y"]),
            }
            break
        env.close()
        if placed is None:
            rejects["no_xy"] += 1
        else:
            rows.append(placed)
            if len(rows) % 8 == 0:
                print(f"  harvested {len(rows)}/{n_frames} rejects={rejects}", flush=True)
        level += 1
    print(f"harvest done n={len(rows)} used={used} rejects={rejects}", flush=True)
    return rows


# Re-exported for run_coin_fisher_decision_window.py, which imports score_frame and summarize from here.
score_frame = dual_layer_scalars

# The summary omits h_b3/h_fc, unlike the Maze Fisher scripts' full SCALAR_KEYS.
_SUMMARY_KEYS = [
    "Q_b3",
    "Q_fc",
    "S_b3",
    "S_fc",
    "trFd_b3",
    "trFd_fc",
    "S_over_trFd_b3",
    "S_over_trFd_fc",
    "dh_b3",
    "dh_fc",
    "rel_b3",
    "rel_fc",
    "T",
    "two_kl",
    "entropy",
    "max_prob",
]


def summarize(rows: list[dict]) -> dict:
    return summarize_scalar_rows(rows, keys=_SUMMARY_KEYS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=48)
    p.add_argument("--start-level", type=int, default=1_000_000)
    p.add_argument("--max-levels", type=int, default=4000)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-init", action="store_true")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or str(_default_device())

    ck_nongm = [
        args.ckpt_dir / "7001_teach.pth",
        args.ckpt_dir / "7002_teach.pth",
        args.ckpt_dir / "7003_teach.pth",
    ]
    # Only these checkpoints are GM-dominant in or near the 20-40M window (harvest_coin_dep_ladder phase check).
    ck_gm = [
        (args.ckpt_dir / "6033_gm_20M.pth", "gm_6033_20M"),
        (args.ckpt_dir / "6035_gm_40M.pth", "gm_6035_40M"),
        (args.ckpt_dir / "6037_gm_40M.pth", "gm_6037_40M"),
    ]
    for c in ck_nongm + [p for p, _ in ck_gm]:
        if not c.is_file():
            raise SystemExit(f"missing ckpt {c}")

    warm = load_policy(ck_nongm[0], device)
    print("=== harvest ahead-ordinary ===", flush=True)
    frames = harvest_ahead_ordinary(
        n_frames=args.n_frames,
        start_level=args.start_level,
        max_levels=args.max_levels,
        warm_policy=warm,
        device=device,
    )
    if len(frames) < 24:
        raise SystemExit(f"too few ahead frames: {len(frames)}")
    meta = {
        "n_frames": len(frames),
        "mean_world_dx": float(np.mean([f["world_dx"] for f in frames])),
        "mean_world_dy": float(np.mean([f["world_dy"] for f in frames])),
        "frac_dx_positive": float(np.mean([f["world_dx"] > 0 for f in frames])),
        "gm_seed_set": "6033@20M, 6035@40M, 6037@40M (dep-ladder phase check)",
        "frame_meta": [
            {
                "level": f["level"],
                "xy_a": f["xy_a"],
                "coin_cell_a": f["coin_cell_a"],
                "world_dx": f["world_dx"],
                "world_dy": f["world_dy"],
            }
            for f in frames
        ],
    }
    print(
        f"harvest meta mean_dx={meta['mean_world_dx']:.2f} "
        f"mean_dy={meta['mean_world_dy']:.2f} "
        f"frac_ahead_x={meta['frac_dx_positive']:.3f}",
        flush=True,
    )

    agents: dict = {}
    for ck in ck_nongm:
        agents[f"nongm_{ck.stem.split('_')[0]}"] = load_policy(ck, device)
    for ck, lab in ck_gm:
        agents[lab] = load_policy(ck, device)
    if not args.skip_init:
        agents["init"] = load_init_policy(ck_nongm[0], device)

    results = {}
    per_frame = {}
    for lab, pol in agents.items():
        print(f"=== score {lab} ===", flush=True)
        rows = []
        for i, fr in enumerate(frames):
            rows.append(score_frame(pol, fr["obs_a"], fr["obs_gone"], device))
            if (i + 1) % 8 == 0:
                print(f"  [{lab}] {i+1}/{len(frames)}", flush=True)
        per_frame[lab] = rows
        results[lab] = summarize(rows)
        print(f"{lab}: {results[lab]}", flush=True)

    # Recipe means then ratios (primary DV on seed-averaged medians)
    def recipe_mean(prefix: str, key: str) -> float:
        labs = [k for k in results if k.startswith(prefix)]
        return float(np.mean([results[k][key] for k in labs]))

    paired = {
        "n_nongm_seeds": 3,
        "n_gm_seeds": 3,
        "ratio_Q_b3": recipe_mean("nongm_", "Q_b3_median")
        / (recipe_mean("gm_", "Q_b3_median") + 1e-18),
        "ratio_Q_fc": recipe_mean("nongm_", "Q_fc_median")
        / (recipe_mean("gm_", "Q_fc_median") + 1e-18),
        "ratio_S_over_trFd_b3": recipe_mean("nongm_", "S_over_trFd_b3_median")
        / (recipe_mean("gm_", "S_over_trFd_b3_median") + 1e-18),
        "ratio_S_over_trFd_fc": recipe_mean("nongm_", "S_over_trFd_fc_median")
        / (recipe_mean("gm_", "S_over_trFd_fc_median") + 1e-18),
        "ratio_dh_b3": recipe_mean("nongm_", "dh_b3_median")
        / (recipe_mean("gm_", "dh_b3_median") + 1e-18),
        "ratio_dh_fc": recipe_mean("nongm_", "dh_fc_median")
        / (recipe_mean("gm_", "dh_fc_median") + 1e-18),
        "ratio_rel_b3": recipe_mean("nongm_", "rel_b3_median")
        / (recipe_mean("gm_", "rel_b3_median") + 1e-18),
        "ratio_rel_fc": recipe_mean("nongm_", "rel_fc_median")
        / (recipe_mean("gm_", "rel_fc_median") + 1e-18),
        "ratio_T": recipe_mean("nongm_", "T_median")
        / (recipe_mean("gm_", "T_median") + 1e-18),
        "ratio_trFd_fc": recipe_mean("nongm_", "trFd_fc_median")
        / (recipe_mean("gm_", "trFd_fc_median") + 1e-18),
        "ratio_entropy": recipe_mean("nongm_", "entropy_median")
        / (recipe_mean("gm_", "entropy_median") + 1e-18),
    }
    paired["Q_fc_ratio_over_Q_b3_ratio"] = paired["ratio_Q_fc"] / (
        paired["ratio_Q_b3"] + 1e-18
    )

    # Per-seed Q_fc for bootstrap-ready dump
    seed_q = {
        lab: results[lab]["Q_fc_median"]
        for lab in results
        if lab.startswith(("nongm_", "gm_"))
    }
    # Bootstrap over seed pairs: all 3x3 ratios
    ng_labs = [k for k in results if k.startswith("nongm_")]
    gm_labs = [k for k in results if k.startswith("gm_")]
    boot_q = []
    for ng in ng_labs:
        for gm in gm_labs:
            boot_q.append(
                results[ng]["Q_fc_median"] / (results[gm]["Q_fc_median"] + 1e-18)
            )
    paired["bootstrap_Q_fc_ratios_3x3"] = boot_q
    paired["bootstrap_Q_fc_ratio_median"] = float(np.median(boot_q))
    paired["bootstrap_Q_fc_ratio_p10"] = float(np.percentile(boot_q, 10))
    paired["bootstrap_Q_fc_ratio_p90"] = float(np.percentile(boot_q, 90))
    # 95% CI over the 3x3 seed-pair ratios, reported alongside p10/p90.
    _bq_mean, _bq_lo, _bq_hi = ci95(np.asarray(boot_q, dtype=np.float64))
    paired["bootstrap_Q_fc_ratio_mean"] = _bq_mean
    paired["bootstrap_Q_fc_ratio_ci95_lo"] = _bq_lo
    paired["bootstrap_Q_fc_ratio_ci95_hi"] = _bq_hi

    summary = {
        "window": "ahead_ordinary",
        "prereg": None,
        "gm_set": ["6033_20M", "6035_40M", "6037_40M"],
        "gm_set_note": (
            "Honest GM set from harvest_coin_dep_ladder phase check: "
            "only seeds GM-dominant in/near 20–40M. Replaces v1's "
            "{6033,6034,6035} (6034 never GM; 6037 missing)."
        ),
        "harvest": meta,
        "agents": results,
        "seed_Q_fc": seed_q,
        "paired_recipe_mean": paired,
    }
    print("paired_recipe_mean", paired, flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "per_frame.json").write_text(json.dumps(per_frame, indent=2, default=float))
    print(f"wrote {out}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
