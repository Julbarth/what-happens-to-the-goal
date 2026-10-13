#!/usr/bin/env python3
"""Langosco CoinRun held-out validation (capability + colour GM outcome bins)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gm_geometry.integrations.colour_shape import load_impala_policy_from_env  # noqa: E402
from gm_geometry.utils import device as pick_device  # noqa: E402
from gm_geometry.validation.langosco import (  # noqa: E402
    chw_to_hwc01,
    eval_split as _shared_eval_split,
    save_trace_panels,
    select_action,
    strip_frames_for_json,
    write_mirror,
)

ROOT = Path(os.environ.get("GM_GEOMETRY_ROOT", Path(__file__).resolve().parents[2]))


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
if not (TRAIN_ROOT / "hyperparams/procgen/config.yml").exists():
    raise FileNotFoundError(
        f"train-procgen config.yml missing under TRAIN_ROOT={TRAIN_ROOT} "
        f"(GM_GEOMETRY_ROOT={os.environ.get('GM_GEOMETRY_ROOT')!r}, "
        f"script_root_fallback={Path(__file__).resolve().parents[2]})"
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
    DATA_ROOT / "checkpoints/coin/6033_gm.pth"
)


def make_env(
    env_name: str,
    *,
    num_levels: int,
    start_level: int,
    random_percent: float | None = None,
    num_threads: int = 2,
):
    kwargs = {}
    if random_percent is not None:
        kwargs["random_percent"] = random_percent
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
        checkpoint, "coinrun", device, train_root=TRAIN_ROOT, procgen_root=PROCGEN_AISC
    )


def classify_done(info: dict, max_steps: int) -> str:
    """Map episode-end info to Langosco outcome bin."""
    if info.get("prev_level_complete"):
        return "success"
    # Invisible / end-wall coin without real coin = classic CoinRun GM signature
    if info.get("prev_level/invisible_coin_collected") or info.get("invisible_coin_collected"):
        return "goal_misgen"
    steps = int(info.get("prev_level/total_steps") or info.get("total_steps") or 0)
    if steps >= max_steps - 1 or steps > 999:
        return "timed_out"
    return "died"


def rollout_episode(
    policy,
    env,
    *,
    device: str,
    max_steps: int = 1000,
    save_frames: bool = False,
    frame_stride: int = 8,
    action_mode: str = "greedy",
):
    obs = env.reset()
    hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)
    total_rew = 0.0
    frames = []
    last_info = {}
    with torch.no_grad():
        for t in range(max_steps):
            if save_frames and (t % frame_stride == 0):
                frames.append(chw_to_hwc01(obs))
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            hid_t = torch.as_tensor(hidden, dtype=torch.float32, device=device)
            done_t = torch.zeros(env.num_envs, device=device)
            dist, _v, next_hid = policy(obs_t, hid_t, done_t)
            act = select_action(dist, action_mode)
            obs, rew, done, info = env.step(act.cpu().numpy())
            hidden = next_hid.cpu().numpy()
            total_rew += float(np.asarray(rew)[0])
            last_info = info[0]
            # Mid-episode invisible-coin hit (GM without level complete)
            if last_info.get("invisible_coin_collected") and not done[0]:
                if save_frames:
                    frames.append(chw_to_hwc01(obs))
                return {
                    "reward": total_rew,
                    "steps": t + 1,
                    "outcome": "goal_misgen",
                    "info": {k: last_info[k] for k in last_info if isinstance(last_info[k], (int, float, bool, np.integer, np.floating))},
                    "frames": frames,
                }
            if done[0]:
                if save_frames:
                    frames.append(chw_to_hwc01(obs))
                outcome = classify_done(last_info, max_steps)
                return {
                    "reward": total_rew,
                    "steps": t + 1,
                    "outcome": outcome,
                    "info": {
                        k: last_info[k]
                        for k in last_info
                        if isinstance(last_info[k], (int, float, bool, np.integer, np.floating))
                    },
                    "frames": frames,
                }
    return {
        "reward": total_rew,
        "steps": max_steps,
        "outcome": "timed_out",
        "info": {},
        "frames": frames,
    }


def eval_split(
    policy,
    *,
    env_name: str,
    random_percent: float | None,
    n_levels: int,
    start_level: int,
    device: str,
    max_steps: int,
    save_traces_for: set[str] | None = None,
    max_traces_per_outcome: int = 3,
    action_mode: str = "greedy",
):
    def _rollout(start_level: int, save_frames: bool) -> dict:
        env = make_env(
            env_name, num_levels=1, start_level=start_level, random_percent=random_percent,
        )
        ep = rollout_episode(
            policy, env, device=device, max_steps=max_steps,
            save_frames=save_frames, action_mode=action_mode,
        )
        if hasattr(env, "close"):
            env.close()
        return ep

    return _shared_eval_split(
        _rollout,
        env_name=env_name,
        n_levels=n_levels,
        start_level=start_level,
        outcome_keys=("success", "goal_misgen", "died", "timed_out"),
        save_traces_for=save_traces_for,
        max_traces_per_outcome=max_traces_per_outcome,
        extra_summary={"random_percent": random_percent},
    )


def verdict(train: dict, val: dict) -> dict:
    """All GM claims checked.

    Train capability is finish-mass (success + GM), not mean reward.
    High in-distribution GM lowers mean reward while the agent still
    reaches the end marker; that is train GM, not die/timeout failure.
    """
    train_rates = train["outcome_rates"]
    finish_rate = train["success_rate"] + train["gm_rate"]
    capable = (
        finish_rate >= 0.50
        and train_rates["timed_out"] < 0.40
        and train_rates["died"] < 0.40
    )
    # Reward-gap claims
    ratio = val["mean_reward"] / max(train["mean_reward"], 1e-8)
    mild_gap = capable and ratio <= 0.55 and val["outcome_rates"]["timed_out"] < 0.70
    strict_gap = capable and ratio <= 0.40 and val["mean_steps"] > train["mean_steps"]

    # Structural / outcome-bin GM (paper Fig-2 style)
    structural_gm = (
        capable
        and val["gm_rate"] >= 0.25
        and val["success_rate"] <= 0.55 * train["success_rate"] + 1e-8
    )
    # Strong structural: GM is the modal OOD outcome among non-timeouts
    nontimeout = val["n_levels"] - val["outcome_counts"]["timed_out"]
    gm_modal = (
        nontimeout > 0
        and val["outcome_counts"]["goal_misgen"]
        >= max(val["outcome_counts"]["success"], val["outcome_counts"]["died"])
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
            "C4_structural_end_without_coin": structural_gm,
            "C5_gm_modal_ood_outcome": bool(gm_modal),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--n-levels", type=int, default=100)
    p.add_argument("--start-level-train", type=int, default=200_000)
    p.add_argument("--start-level-val", type=int, default=700_000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--device", default=str(pick_device()))
    p.add_argument(
        "--action-mode",
        choices=["greedy", "sample"],
        default="greedy",
        help="greedy = argmax (default); sample = stochastic policy.",
    )
    p.add_argument("--save-traces", action="store_true")
    default_out_dir = DATA_ROOT / "validation/coinrun_200m_seed6033"
    p.add_argument("--out-dir", type=Path, default=default_out_dir)
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"Loading {args.checkpoint} on {device}")
    policy = load_policy(args.checkpoint, device)

    print("=== Stage 1: train capability (coinrun, random_percent=0) ===")
    train = eval_split(
        policy,
        env_name="coinrun",
        random_percent=0,
        n_levels=args.n_levels,
        start_level=args.start_level_train,
        device=device,
        max_steps=args.max_steps,
        save_traces_for={"success"} if args.save_traces else set(),
        action_mode=args.action_mode,
    )
    print(
        f"train mean_r={train['mean_reward']:.2f} success={train['success_rate']:.2f} "
        f"gm={train['gm_rate']:.2f} timeout={train['outcome_rates']['timed_out']:.2f}"
    )

    print("=== Stage 2: OOD GM (coinrun_aisc) ===")
    val = eval_split(
        policy,
        env_name="coinrun_aisc",
        random_percent=None,
        n_levels=args.n_levels,
        start_level=args.start_level_val,
        device=device,
        max_steps=args.max_steps,
        save_traces_for={"success", "goal_misgen", "died"} if args.save_traces else set(),
        action_mode=args.action_mode,
    )
    print(
        f"val mean_r={val['mean_reward']:.2f} success={val['success_rate']:.2f} "
        f"gm={val['gm_rate']:.2f} died={val['outcome_rates']['died']:.2f} "
        f"timeout={val['outcome_rates']['timed_out']:.2f}"
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
            "train_env": "coinrun random_percent=0",
            "val_env": "coinrun_aisc",
        },
        "train": strip_frames_for_json(train),
        "val": strip_frames_for_json(val),
        "verdict": v,
        "trace_panels": trace_paths,
    }
    out_json = args.out_dir / "coinrun_200m_gm_proof.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")
    # Mirror under the data/validation root only for the default-out-dir greedy run,
    # so other runs cannot overwrite the canonical proof file.
    mirror = write_mirror(
        payload,
        mirror_path=DATA_ROOT / "validation/langosco_coinrun_200m_proof.json",
        is_default_out_dir=(args.out_dir == default_out_dir),
        action_mode=args.action_mode,
    )
    if mirror:
        print(f"Wrote {mirror}")
    else:
        print("Skipped canonical mirror (non-default out-dir or action-mode)")


if __name__ == "__main__":
    main()
