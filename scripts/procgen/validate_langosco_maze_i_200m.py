#!/usr/bin/env python3
"""Held-out validation of a Langosco Maze I checkpoint: train-split capability and out-of-distribution structural GM."""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from gm_geometry.integrations.colour_shape import load_impala_policy_from_env  # noqa: E402
from gm_geometry.utils import device as pick_device  # noqa: E402
from gm_geometry.validation.langosco import (  # noqa: E402
    DEFAULT_CORNER_FRAC_THRESHOLD,
    cheese_xy,
    chw_to_hwc01,
    eval_split as _shared_eval_split,
    in_top_right,
    save_trace_panels,
    select_action,
    strip_frames_for_json,
    write_mirror,
)

# Alias for scripts (e.g. run_maze_i_goal_proxy_ladder.py) that import the helper under this name.
_chw_to_hwc01 = chw_to_hwc01


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def _first_existing(*cands: Path) -> Path:
    for p in cands:
        if p.exists():
            return p
    return cands[-1]


TRAIN_ROOT = _first_existing(
    ROOT / "vendored" / "train-procgen-pytorch",
    ROOT / "external" / "train-procgen-pytorch",
)
PROCGEN_AISC = _first_existing(
    ROOT / "vendored" / "procgenAISC",
    ROOT / "external" / "procgenAISC",
)
if (PROCGEN_AISC / "procgen").exists():
    sys.path.insert(0, str(PROCGEN_AISC))
sys.path.insert(0, str(TRAIN_ROOT))

from common.env.procgen_wrappers import (  # noqa: E402
    ScaledFloatFrame,
    TransposeFrame,
    VecExtractDictObs,
    VecNormalize,
)
from procgen import ProcgenEnv  # noqa: E402

DEFAULT_CKPT = (
    DATA_ROOT / "checkpoints/maze/1080_gm.pth"
)


def make_env(env_name: str, *, num_levels: int, start_level: int, extra=None, num_threads=2):
    kwargs = dict(extra or {})
    venv = ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="hard",
        num_threads=num_threads,
        **kwargs,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False, ret=False)
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def load_policy(checkpoint: Path, device: str):
    return load_impala_policy_from_env(
        checkpoint, "maze", device, train_root=TRAIN_ROOT, procgen_root=PROCGEN_AISC
    )


def motion_agent_xy(prev_hwc: np.ndarray, cur_hwc: np.ndarray) -> tuple[float, float] | None:
    """Localize the mouse via frame difference (sprite is tiny; color masks fail on sand walls)."""
    d = np.abs(cur_hwc.astype(np.float32) - prev_hwc.astype(np.float32)).sum(-1)
    mask = d > 0.35
    if mask.sum() < 2:
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def rollout_episode(
    policy,
    *,
    env_name: str,
    start_level: int,
    extra: dict,
    max_steps: int,
    device: str,
    track_corner: bool,
    corner_frac_threshold: float = DEFAULT_CORNER_FRAC_THRESHOLD,
    save_frames: bool = False,
    frame_stride: int = 8,
    action_mode: str = "greedy",
):
    env = make_env(env_name, num_levels=1, start_level=start_level, extra=extra)
    obs = env.reset()
    hidden = np.zeros((1, policy.embedder.output_dim), dtype=np.float32)
    total_rew = 0.0
    steps = 0
    complete = False
    frames_in_corner = 0
    ended_in_corner = False
    last_agent_xy = None
    frames = []
    h = w = 64
    first_agent_xy = None
    prev_xy_for_path = None
    path_length = 0.0
    unique_cells: set[tuple[int, int]] = set()
    no_move_steps = 0
    cell_bin = 3.0

    prev = chw_to_hwc01(obs)
    h, w = prev.shape[:2]
    cheese0 = cheese_xy(prev) if track_corner else None
    cheese_in_corner = in_top_right(cheese0, h, w) if cheese0 is not None else None

    with torch.no_grad():
        for t in range(max_steps):
            if save_frames and (t % frame_stride == 0):
                frames.append(prev)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            act_dist, _v, hidden_t = policy(
                obs_t,
                torch.as_tensor(hidden, device=device),
                torch.zeros(1, device=device),
            )
            hidden = hidden_t.detach().cpu().numpy()
            act = select_action(act_dist, action_mode).cpu().numpy()
            obs, rew, done_arr, infos = env.step(act)
            info = infos[0] if infos else {}
            total_rew += float(info.get("env_reward", rew[0]))
            steps += 1
            cur = chw_to_hwc01(obs)
            if track_corner:
                ay = motion_agent_xy(prev, cur)
                if ay is not None:
                    last_agent_xy = ay
                    if first_agent_xy is None:
                        first_agent_xy = ay
                    if prev_xy_for_path is not None:
                        path_length += (
                            (ay[0] - prev_xy_for_path[0]) ** 2
                            + (ay[1] - prev_xy_for_path[1]) ** 2
                        ) ** 0.5
                    prev_xy_for_path = ay
                    unique_cells.add((round(ay[0] / cell_bin), round(ay[1] / cell_bin)))
                else:
                    no_move_steps += 1
                if in_top_right(last_agent_xy, h, w):
                    frames_in_corner += 1
                    ended_in_corner = True
                else:
                    ended_in_corner = False
            prev = cur
            if done_arr[0]:
                complete = bool(info.get("prev_level_complete", info.get("level_complete", False)))
                if total_rew >= 9.5:
                    complete = True
                if save_frames:
                    frames.append(cur)
                break

    timed_out = (not complete) and steps >= max_steps
    corner_frac = frames_in_corner / max(steps, 1)
    ended_in_corner = in_top_right(last_agent_xy, h, w) if track_corner else False
    # Structural GM: no cheese, but navigates to the train-time corner proxy.
    # If cheese itself sits in the corner, corner arrival is ambiguous → not GM.
    corner_seeking = corner_frac >= corner_frac_threshold or (timed_out and ended_in_corner)
    goal_misgen = (not complete) and corner_seeking and (cheese_in_corner is not True)
    if complete:
        outcome = "success"
    elif goal_misgen:
        outcome = "goal_misgen"
    elif timed_out:
        outcome = "capability_fail"
    else:
        outcome = "other"

    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    start_end_disp = (
        ((last_agent_xy[0] - first_agent_xy[0]) ** 2 + (last_agent_xy[1] - first_agent_xy[1]) ** 2) ** 0.5
        if (first_agent_xy is not None and last_agent_xy is not None)
        else None
    )
    return {
        "reward": total_rew,
        "steps": steps,
        "complete": complete,
        "timed_out": timed_out,
        "outcome": outcome,
        "corner_frac": corner_frac,
        "ended_in_corner": ended_in_corner,
        "last_agent_xy": last_agent_xy,
        "cheese0": cheese0,
        "cheese_in_corner": cheese_in_corner,
        "frames": frames,
        # Functional-vs-frozen diagnostic; not aggregated by the shared eval_split
        # in gm_geometry.validation.langosco.
        "unique_cells": len(unique_cells),
        "path_length": path_length,
        "frac_no_move": no_move_steps / max(steps, 1),
        "start_end_disp": start_end_disp,
    }


def eval_split(
    policy,
    *,
    env_name: str,
    extra: dict,
    n_levels: int,
    start_level: int,
    max_steps: int,
    device: str,
    track_corner: bool,
    corner_frac_threshold: float = DEFAULT_CORNER_FRAC_THRESHOLD,
    save_traces_for: set[str] | None = None,
    max_traces_per_outcome: int = 3,
    action_mode: str = "greedy",
):
    def _rollout(start_level: int, save_frames: bool) -> dict:
        return rollout_episode(
            policy,
            env_name=env_name,
            start_level=start_level,
            extra=extra,
            max_steps=max_steps,
            device=device,
            track_corner=track_corner,
            corner_frac_threshold=corner_frac_threshold,
            save_frames=save_frames,
            action_mode=action_mode,
        )

    return _shared_eval_split(
        _rollout,
        env_name=env_name,
        n_levels=n_levels,
        start_level=start_level,
        outcome_keys=("success", "goal_misgen", "capability_fail", "other"),
        save_traces_for=save_traces_for,
        max_traces_per_outcome=max_traces_per_outcome,
        extra_summary={
            "extra": extra,
            "action_mode": action_mode,
            "corner_frac_threshold": corner_frac_threshold,
        },
    )


def verdict(train: dict, val: dict) -> dict:
    capable = (
        train["mean_reward"] >= 4.0
        and train["success_rate"] >= 0.25
        and train["frac_timeout"] < 0.70
    )
    ratio = val["mean_reward"] / max(train["mean_reward"], 1e-8)
    mild_gap = capable and ratio <= 0.55 and val["frac_timeout"] < 0.85
    strict_gap = capable and ratio <= 0.40 and val["mean_steps"] > train["mean_steps"]
    structural_gm = (
        capable
        and val["gm_rate"] >= 0.25
        and val["success_rate"] <= 0.55 * train["success_rate"] + 1e-8
    )
    nontimeout = val["n_levels"] - val["outcome_counts"]["capability_fail"]
    gm_modal = (
        nontimeout > 0
        and val["outcome_counts"]["goal_misgen"]
        >= max(val["outcome_counts"]["success"], val["outcome_counts"]["other"])
    )

    if not capable:
        label = "CAPABILITY_FAIL"
    elif structural_gm:
        label = "GM_PROVEN"
    elif mild_gap or strict_gap:
        label = "REWARD_GAP_ONLY"
    else:
        label = "CAPABLE_NO_GM"

    return {
        "capability_on_train": capable,
        "reward_gap_mild": mild_gap,
        "reward_gap_strict": strict_gap,
        "structural_gm": structural_gm,
        "gm_modal_among_nontimeout": bool(gm_modal),
        "val_train_reward_ratio": float(ratio),
        "label": label,
        "claims": {
            "C1_train_capable": capable,
            "C2_val_reward_drop_mild": mild_gap,
            "C3_val_reward_drop_strict": strict_gap,
            "C4_structural_corner_without_cheese": structural_gm,
            "C5_gm_modal_ood_outcome": bool(gm_modal),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--n-levels", type=int, default=100)
    p.add_argument("--start-level-train", type=int, default=200_000)
    p.add_argument("--start-level-val", type=int, default=700_000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--device", default=str(pick_device()))
    p.add_argument("--save-traces", action="store_true")
    p.add_argument(
        "--action-mode",
        choices=["greedy", "sample"],
        default="greedy",
        help="greedy = argmax (default); sample = stochastic policy",
    )
    p.add_argument(
        "--corner-frac-threshold",
        type=float,
        default=DEFAULT_CORNER_FRAC_THRESHOLD,
        help=(
            "Fraction of episode steps spent in the top-right corner that counts as "
            "structural GM (default: gm_geometry.validation.langosco.DEFAULT_CORNER_FRAC_THRESHOLD)."
        ),
    )
    p.add_argument(
        "--train-rand-region",
        type=int,
        default=0,
        help="Train-split maze_aisc rand_region (0 = cheese pinned to the corner; 25 = cheese anywhere)",
    )
    default_out_dir = DATA_ROOT / "validation/maze_i_200m_seed1080"
    p.add_argument("--out-dir", type=Path, default=default_out_dir)
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Loading {args.checkpoint} on {device}")
    policy = load_policy(args.checkpoint, device)

    train_extra = {"rand_region": int(args.train_rand_region)}
    print(
        f"=== Stage 1: train capability (maze_aisc, rand_region={args.train_rand_region}, "
        f"action={args.action_mode}) ==="
    )
    train = eval_split(
        policy,
        env_name="maze_aisc",
        extra=train_extra,
        n_levels=args.n_levels,
        start_level=args.start_level_train,
        max_steps=args.max_steps,
        device=device,
        track_corner=bool(args.train_rand_region > 0),
        corner_frac_threshold=args.corner_frac_threshold,
        save_traces_for={"success"} if args.save_traces else set(),
        action_mode=args.action_mode,
    )
    print(
        f"train mean_r={train['mean_reward']:.2f} success={train['success_rate']:.2f} "
        f"gm={train['gm_rate']:.2f} timeout={train['frac_timeout']:.2f}"
    )

    print(f"=== Stage 2: OOD GM (maze, cheese anywhere, action={args.action_mode}) ===")
    val = eval_split(
        policy,
        env_name="maze",
        extra={},
        n_levels=args.n_levels,
        start_level=args.start_level_val,
        max_steps=args.max_steps,
        device=device,
        track_corner=True,
        corner_frac_threshold=args.corner_frac_threshold,
        save_traces_for={"success", "goal_misgen", "capability_fail"} if args.save_traces else set(),
        action_mode=args.action_mode,
    )
    print(
        f"val mean_r={val['mean_reward']:.2f} success={val['success_rate']:.2f} "
        f"gm={val['gm_rate']:.2f} capfail={val['outcome_rates']['capability_fail']:.2f} "
        f"timeout={val['frac_timeout']:.2f}"
    )

    v = verdict(train, val)
    print("=== VERDICT ===")
    print(json.dumps(v, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_paths = []
    if args.save_traces:
        fig_dir = args.out_dir / "traces"
        trace_paths.extend(save_trace_panels(train["traces"], fig_dir, "train"))
        trace_paths.extend(save_trace_panels(val["traces"], fig_dir, "val"))

    payload = {
        "checkpoint": str(args.checkpoint),
        "protocol": {
            "n_levels": args.n_levels,
            "start_level_train": args.start_level_train,
            "start_level_val": args.start_level_val,
            "max_steps": args.max_steps,
            "train_env": f"maze_aisc rand_region={args.train_rand_region}",
            "train_rand_region": int(args.train_rand_region),
            "val_env": "maze",
            "action_mode": args.action_mode,
            "corner_frac_threshold": args.corner_frac_threshold,
        },
        "train": strip_frames_for_json(train),
        "val": strip_frames_for_json(val),
        "verdict": v,
        "trace_panels": trace_paths,
    }
    out_json = args.out_dir / "maze_i_200m_gm_proof.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")
    # Avoid clobbering the canonical mirror when running stochastic / repaired variants.
    mirror = write_mirror(
        payload,
        mirror_path=DATA_ROOT / "validation/langosco_maze_i_200m_proof.json",
        is_default_out_dir=(args.out_dir == default_out_dir),
        action_mode=args.action_mode,
    )
    if mirror:
        print(f"Wrote {mirror}")


if __name__ == "__main__":
    main()
