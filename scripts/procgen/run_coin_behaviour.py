#!/usr/bin/env python3
"""Coin behaviour suite: checkpoint crossover, ledge and hidden-coin splits, graded directedness, stochastic decode."""

from __future__ import annotations

import os
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from gm_geometry.integrations.colour_shape import load_impala_policy_from_env  # noqa: E402
from gm_geometry.utils import device as _default_device  # noqa: E402
from gm_geometry.utils import parse_seeds  # noqa: E402


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def _existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing path (tried: {[str(p) for p in paths]})")


TRAIN_ROOT = _existing(
    ROOT / "external" / "train-procgen-pytorch",
    ROOT / "vendored" / "train-procgen-pytorch",
)
PROCGEN_AISC = _existing(
    ROOT / "external" / "procgenAISC",
    ROOT / "vendored" / "procgenAISC",
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
from common.policy import CategoricalPolicy  # noqa: E402
from procgen import ProcgenEnv  # noqa: E402

CKPT_ROOT = (
    DATA_ROOT / "checkpoints/train_logs/coinrun/coinrun-200m-langosco"
)

SEED_DIRS = {
    6033: CKPT_ROOT / "2026-07-15__23-37-27__seed_6033",
    6034: CKPT_ROOT / "2026-07-29__23-53-54__seed_6034",
    6035: CKPT_ROOT / "2026-07-29__23-54-20__seed_6035",
}

# Anchor checkpoints per seed (early GM / late collect).
ANCHORS = {
    6033: {"early": 20_054_016, "late": 200_015_872},
    6034: {"early": 20_054_016, "late": 200_015_872},
    6035: {"early": 40_042_496, "late": 200_015_872},
}

VAL_START = 1_000_000
DEFAULT_OUT = DATA_ROOT / "probes/coinrun_sprint_A"


def ckpt_path(seed: int, steps: int) -> Path:
    p = SEED_DIRS[seed] / f"model_{steps}.pth"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def make_env(
    env_name: str,
    *,
    num_levels: int,
    start_level: int,
    random_percent: float | None = None,
    hide_visible_coin: bool = False,
    num_threads: int = 2,
):
    kwargs = {}
    if random_percent is not None:
        kwargs["random_percent"] = random_percent
    if hide_visible_coin:
        kwargs["hide_visible_coin"] = True
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


def load_policy(checkpoint: Path, device: str) -> CategoricalPolicy:
    return load_impala_policy_from_env(
        checkpoint, "coinrun", device, train_root=TRAIN_ROOT, procgen_root=PROCGEN_AISC
    )


def load_init_policy(device: str) -> CategoricalPolicy:
    """Fresh random-init Impala + heads (same architecture as trained agents)."""
    return load_impala_policy_from_env(
        None, "coinrun", device, random_init=True, train_root=TRAIN_ROOT, procgen_root=PROCGEN_AISC
    )


def crossover_policy(
    trunk_ckpt: Optional[Path],
    head_ckpt: Path,
    device: str,
    *,
    init_trunk: bool = False,
    head_parts: str = "both",
    cut: str = "post_fc",
) -> CategoricalPolicy:
    """Compose trunk and head across a cut.

    ``post_fc`` (default): the whole Impala ``embedder`` (blocks + fc) is the trunk; only ``fc_policy`` and
    ``fc_value`` come from the head ckpt. ``pre_fc``: the conv stack is the trunk; ``embedder.fc`` and the
    linear heads come from the head ckpt.
    """
    if init_trunk:
        trunk = load_init_policy(device)
    else:
        assert trunk_ckpt is not None
        trunk = load_policy(trunk_ckpt, device)
    head = load_policy(head_ckpt, device)
    if cut == "pre_fc":
        # Keep trunk.embedder.block*; take fc (+ ReLU path lives in forward) from head.
        trunk.embedder.fc.load_state_dict(head.embedder.fc.state_dict())
        trunk.fc_policy.load_state_dict(head.fc_policy.state_dict())
        trunk.fc_value.load_state_dict(head.fc_value.state_dict())
    else:
        if head_parts in ("both", "actor"):
            trunk.fc_policy.load_state_dict(head.fc_policy.state_dict())
        if head_parts in ("both", "critic"):
            trunk.fc_value.load_state_dict(head.fc_value.state_dict())
    del head
    return trunk


def _nearest_ckpt(seed: int, target_steps: int) -> tuple[int, Path]:
    """Closest model_*.pth in SEED_DIRS[seed] to target_steps."""
    run_dir = SEED_DIRS[seed]
    best_s, best_p = None, None
    for p in run_dir.glob("model_*.pth"):
        m = re.search(r"model_(\d+)\.pth$", p.name)
        if not m:
            continue
        s = int(m.group(1))
        if best_s is None or abs(s - target_steps) < abs(best_s - target_steps):
            best_s, best_p = s, p
    if best_p is None:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    return best_s, best_p


def _eval_cell(
    name: str,
    trunk_p: Optional[Path],
    head_p: Optional[Path],
    *,
    device: str,
    n_levels: int,
    max_steps: int,
    init_trunk: bool = False,
    head_parts: str = "both",
    cut: str = "post_fc",
) -> dict:
    assert head_p is not None, "head_p required (use load_init_policy path for pure init)"
    if (
        trunk_p == head_p
        and not init_trunk
        and head_parts == "both"
        and cut == "post_fc"
    ):
        policy = load_policy(trunk_p, device)
    else:
        policy = crossover_policy(
            trunk_p,
            head_p,
            device,
            init_trunk=init_trunk,
            head_parts=head_parts,
            cut=cut,
        )
    summary = eval_policy_val(
        policy,
        n_levels=n_levels,
        start_level=VAL_START,
        device=device,
        max_steps=max_steps,
        action_mode="greedy",
        hide_visible_coin=False,
    )
    summary["trunk_ckpt"] = "INIT" if init_trunk else str(trunk_p)
    summary["head_ckpt"] = str(head_p) if head_p is not None else "INIT"
    summary["head_parts"] = head_parts
    summary["cut"] = cut
    summary["cell"] = name
    print(
        f"  success={summary['success_rate']:.3f} "
        f"gm={summary['gm_rate']:.3f} "
        f"died={summary['outcome_rates']['died']:.3f} "
        f"timeout={summary['outcome_rates']['timed_out']:.3f} "
        f"cut={cut}"
    )
    del policy
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


def classify_done(info: dict, max_steps: int) -> str:
    if info.get("prev_level_complete"):
        return "success"
    if info.get("prev_level/invisible_coin_collected") or info.get("invisible_coin_collected"):
        return "goal_misgen"
    steps = int(info.get("prev_level/total_steps") or info.get("total_steps") or 0)
    if steps >= max_steps - 1 or steps > 999:
        return "timed_out"
    return "died"


def action_vx(action: int) -> int:
    """Horizontal intent from Procgen combo index (0–8). Specials (≥9) → 0."""
    a = int(action)
    if a >= 9:
        return 0
    return a // 3 - 1


def _scalar_info(info: dict) -> dict:
    out = {}
    for k, v in info.items():
        if isinstance(v, (int, float, bool, np.integer, np.floating)):
            out[k] = int(v) if isinstance(v, (np.integer,)) else (
                float(v) if isinstance(v, (np.floating,)) else v
            )
    return out


def rollout_episode(
    policy,
    env,
    *,
    device: str,
    max_steps: int = 1000,
    action_mode: str = "greedy",
    track_directedness: bool = False,
):
    obs = env.reset()
    hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)
    total_rew = 0.0
    last_info = {}
    coin_hits = end_hits = directed_n = 0
    ledge0 = None
    crate0 = None
    max_progress = 0.0

    with torch.no_grad():
        for t in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            hid_t = torch.as_tensor(hidden, dtype=torch.float32, device=device)
            done_t = torch.zeros(env.num_envs, device=device)
            dist, _v, next_hid = policy(obs_t, hid_t, done_t)
            act = dist.probs.argmax(-1) if action_mode == "greedy" else dist.sample()
            act_i = int(act.cpu().numpy()[0])

            # Info is read after the step: the wrappers do not report reset info consistently.
            obs, rew, done, info = env.step(act.cpu().numpy())
            hidden = next_hid.cpu().numpy()
            total_rew += float(np.asarray(rew)[0])
            last_info = info[0]
            # End-reach proxy: fractional progress along level_width.
            try:
                ax = float(last_info.get("agent_x_fp", 0)) / 100.0
                width = float(last_info.get("level_width", 0) or 0)
                if width > 0:
                    max_progress = max(max_progress, ax / width)
            except Exception:
                pass
            if ledge0 is None and "coin_requires_jump" in last_info:
                ledge0 = int(last_info["coin_requires_jump"])
                crate0 = int(last_info.get("coin_on_crate", 0))

            if track_directedness and last_info.get("coin_present", 1):
                ax = float(last_info.get("agent_x_fp", 0)) / 100.0
                cx = float(last_info.get("coin_x", 0))
                ix = float(last_info.get("inv_coin_x", 0))
                vx = action_vx(act_i)
                coin_dx = cx - ax
                end_dx = ix - ax
                if abs(coin_dx) > 0.5:
                    directed_n += 1
                    if np.sign(vx) == np.sign(coin_dx) and vx != 0:
                        coin_hits += 1
                if abs(end_dx) > 0.5:
                    if abs(coin_dx) <= 0.5:
                        directed_n += 0
                    if np.sign(vx) == np.sign(end_dx) and vx != 0:
                        end_hits += 1

            if last_info.get("invisible_coin_collected") and not done[0]:
                return _ep_result(
                    total_rew, t + 1, "goal_misgen", last_info,
                    coin_hits, end_hits, directed_n, ledge0, crate0, track_directedness,
                    max_progress=max_progress,
                )
            if done[0]:
                return _ep_result(
                    total_rew, t + 1, classify_done(last_info, max_steps), last_info,
                    coin_hits, end_hits, directed_n, ledge0, crate0, track_directedness,
                    max_progress=max_progress,
                )

    return _ep_result(
        total_rew, max_steps, "timed_out", last_info,
        coin_hits, end_hits, directed_n, ledge0, crate0, track_directedness,
        max_progress=max_progress,
    )


def _ep_result(
    reward, steps, outcome, info,
    coin_hits, end_hits, directed_n, ledge0, crate0, track_directedness,
    *,
    max_progress: float = 0.0,
):
    out = {
        "reward": reward,
        "steps": steps,
        "outcome": outcome,
        "info": _scalar_info(info),
        "coin_requires_jump": ledge0,
        "coin_on_crate": crate0,
        "max_progress": float(max_progress),
        # Finish at end wall (collect or invisible) OR progress deep into the level.
        "reached_end_region": bool(
            outcome in ("success", "goal_misgen") or max_progress >= 0.85
        ),
    }
    if track_directedness:
        out["coin_directed_hits"] = coin_hits
        out["end_directed_hits"] = end_hits
        out["directed_steps"] = directed_n
    return out


def rollout_graded(
    policy,
    env,
    *,
    device: str,
    max_steps: int = 1000,
    action_mode: str = "greedy",
):
    """Per-timestep coin vs end directedness with separate denominators."""
    obs = env.reset()
    hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)
    total_rew = 0.0
    last_info = {}
    coin_match = coin_n = end_match = end_n = 0
    # Disagreement steps: coin and end pull opposite horizontal signs.
    # Agreement steps are non-disambiguating (both rightward on typical aisc).
    dis_coin_match = dis_end_match = dis_n = 0
    ledge0 = crate0 = None

    with torch.no_grad():
        for t in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            hid_t = torch.as_tensor(hidden, dtype=torch.float32, device=device)
            done_t = torch.zeros(env.num_envs, device=device)
            dist, _v, next_hid = policy(obs_t, hid_t, done_t)
            act = dist.probs.argmax(-1) if action_mode == "greedy" else dist.sample()
            act_i = int(act.cpu().numpy()[0])
            obs, rew, done, info = env.step(act.cpu().numpy())
            hidden = next_hid.cpu().numpy()
            total_rew += float(np.asarray(rew)[0])
            last_info = info[0]

            if ledge0 is None and "coin_requires_jump" in last_info:
                ledge0 = int(last_info["coin_requires_jump"])
                crate0 = int(last_info.get("coin_on_crate", 0))

            ax = float(last_info.get("agent_x_fp", 0)) / 100.0
            cx = float(last_info.get("coin_x", 0))
            ix = float(last_info.get("inv_coin_x", 0))
            present = int(last_info.get("coin_present", 1))
            vx = action_vx(act_i)
            coin_dx = (cx - ax) if present else None
            end_dx = ix - ax
            if present and coin_dx is not None and abs(coin_dx) > 0.5:
                coin_n += 1
                if vx != 0 and np.sign(vx) == np.sign(coin_dx):
                    coin_match += 1
            if abs(end_dx) > 0.5:
                end_n += 1
                if vx != 0 and np.sign(vx) == np.sign(end_dx):
                    end_match += 1
            if (
                present
                and coin_dx is not None
                and abs(coin_dx) > 0.5
                and abs(end_dx) > 0.5
                and np.sign(coin_dx) != np.sign(end_dx)
            ):
                dis_n += 1
                if vx != 0 and np.sign(vx) == np.sign(coin_dx):
                    dis_coin_match += 1
                if vx != 0 and np.sign(vx) == np.sign(end_dx):
                    dis_end_match += 1

            if last_info.get("invisible_coin_collected") and not done[0]:
                outcome = "goal_misgen"
                break
            if done[0]:
                outcome = classify_done(last_info, max_steps)
                break
        else:
            outcome = "timed_out"

    dis_coin_rate = (dis_coin_match / dis_n) if dis_n else float("nan")
    dis_end_rate = (dis_end_match / dis_n) if dis_n else float("nan")
    return {
        "reward": total_rew,
        "steps": t + 1,
        "outcome": outcome,
        "info": _scalar_info(last_info),
        "coin_requires_jump": ledge0,
        "coin_on_crate": crate0,
        "coin_directed_rate": (coin_match / coin_n) if coin_n else float("nan"),
        "end_directed_rate": (end_match / end_n) if end_n else float("nan"),
        "coin_minus_end": (
            (coin_match / coin_n) - (end_match / end_n)
            if coin_n and end_n
            else float("nan")
        ),
        "coin_n": coin_n,
        "end_n": end_n,
        "disagree_n": dis_n,
        "disagree_coin_directed_rate": dis_coin_rate,
        "disagree_end_directed_rate": dis_end_rate,
        "disagree_coin_minus_end": (
            dis_coin_rate - dis_end_rate if dis_n else float("nan")
        ),
    }


def rate_summary(outcomes: list[str]) -> dict:
    keys = ("success", "goal_misgen", "died", "timed_out")
    n = len(outcomes)
    counts = {k: int(sum(o == k for o in outcomes)) for k in keys}
    return {
        "n_levels": n,
        "outcome_counts": counts,
        "outcome_rates": {k: counts[k] / n for k in keys},
        "success_rate": counts["success"] / n,
        "gm_rate": counts["goal_misgen"] / n,
    }


def bootstrap_mean(xs: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> dict:
    xs = np.asarray(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(xs, size=xs.size, replace=True).mean()
    return {
        "mean": float(xs.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": int(xs.size),
    }


def eval_policy_val(
    policy,
    *,
    n_levels: int,
    start_level: int,
    device: str,
    max_steps: int,
    action_mode: str = "greedy",
    hide_visible_coin: bool = False,
    track_ledge: bool = True,
    keep_episodes: bool = False,
    env_name: str = "coinrun_aisc",
    random_percent: float | None = None,
):
    outcomes = []
    ledges = []
    crates = []
    episodes = []
    for i in range(n_levels):
        env = make_env(
            env_name,
            num_levels=1,
            start_level=start_level + i,
            hide_visible_coin=hide_visible_coin,
            random_percent=random_percent,
        )
        ep = rollout_episode(
            policy,
            env,
            device=device,
            max_steps=max_steps,
            action_mode=action_mode,
        )
        if hasattr(env, "close"):
            env.close()
        outcomes.append(ep["outcome"])
        if keep_episodes:
            episodes.append(
                {
                    "level": start_level + i,
                    "outcome": ep["outcome"],
                    "max_progress": ep.get("max_progress", 0.0),
                    "reached_end_region": ep.get("reached_end_region", False),
                }
            )
        if track_ledge:
            ledges.append(ep.get("coin_requires_jump"))
            crates.append(ep.get("coin_on_crate"))
        if (i + 1) % 50 == 0:
            print(f"  … {i+1}/{n_levels} success={sum(o=='success' for o in outcomes)/len(outcomes):.3f}")

    summary = rate_summary(outcomes)
    summary["hide_visible_coin"] = hide_visible_coin
    summary["action_mode"] = action_mode
    summary["start_level"] = start_level
    summary["env_name"] = env_name
    if keep_episodes:
        summary["episodes"] = episodes
        summary["end_reach_rate"] = float(
            np.mean([bool(e["reached_end_region"]) for e in episodes])
        )
        summary["mean_max_progress"] = float(
            np.mean([float(e["max_progress"]) for e in episodes])
        )

    if track_ledge:
        strata = {}
        for name, mask_fn in (
            ("jump_required", lambda L, C: L == 1),
            ("flat_ground", lambda L, C: L == 0),
            ("on_crate", lambda L, C: C == 1),
            ("not_crate", lambda L, C: C == 0),
        ):
            idx = [
                j
                for j, (L, C) in enumerate(zip(ledges, crates))
                if L is not None and C is not None and mask_fn(L, C)
            ]
            sub = [outcomes[j] for j in idx]
            strata[name] = rate_summary(sub) if sub else {"n_levels": 0}
            strata[name]["n_levels"] = len(sub)
        summary["strata"] = strata
        summary["ledge_counts"] = {
            "jump_required": int(sum(L == 1 for L in ledges if L is not None)),
            "flat_ground": int(sum(L == 0 for L in ledges if L is not None)),
            "on_crate": int(sum(C == 1 for C in crates if C is not None)),
            "unknown": int(sum(L is None for L in ledges)),
        }
    return summary


# ── modes ────────────────────────────────────────────────────────────────────


def run_crossover(args):
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed else (args.seeds_list or [6033, 6035])
    all_results = {}
    init_only = bool(getattr(args, "init_trunk_only", False))

    for seed in seeds:
        early_s = ANCHORS[seed]["early"]
        late_s = ANCHORS[seed]["late"]
        early_p = ckpt_path(seed, early_s)
        late_p = ckpt_path(seed, late_s)
        # (name, trunk_path_or_None, head_path, init_trunk)
        cells = []
        if not init_only:
            cells.extend(
                [
                    ("early_trunk_early_head", early_p, early_p, False),
                    ("late_trunk_late_head", late_p, late_p, False),
                    ("early_trunk_late_head", early_p, late_p, False),
                    ("late_trunk_early_head", late_p, early_p, False),
                ]
            )
        if args.include_init_trunk or init_only:
            cells.extend(
                [
                    ("init_trunk_late_head", None, late_p, True),
                    ("init_trunk_early_head", None, early_p, True),
                    ("init_trunk_init_head", None, None, True),
                ]
            )
        seed_res = {}
        for name, trunk_p, head_p, init_trunk in cells:
            print(f"\n=== crossover seed={seed} {name} n={args.n_levels} ===")
            if name == "init_trunk_init_head":
                policy = load_init_policy(device)
                summary = eval_policy_val(
                    policy,
                    n_levels=args.n_levels,
                    start_level=VAL_START,
                    device=device,
                    max_steps=args.max_steps,
                    action_mode="greedy",
                    hide_visible_coin=False,
                )
                summary["trunk_ckpt"] = "INIT"
                summary["head_ckpt"] = "INIT"
                summary["head_parts"] = "both"
                summary["cell"] = name
                print(
                    f"  success={summary['success_rate']:.3f} "
                    f"gm={summary['gm_rate']:.3f}"
                )
                del policy
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                summary = _eval_cell(
                    name,
                    trunk_p,
                    head_p,
                    device=device,
                    n_levels=args.n_levels,
                    max_steps=args.max_steps,
                    init_trunk=init_trunk,
                )
            summary["cut"] = (
                "trunk=Impala embedder (incl fc→256); "
                "head=fc_policy+fc_value"
            )
            seed_res[name] = summary
        all_results[str(seed)] = seed_res
        tag = "init_trunk" if init_only else "crossover"
        path = out_dir / f"{tag}_seed{seed}_n{args.n_levels}.json"
        path.write_text(json.dumps(seed_res, indent=2))
        print(f"wrote {path}")

    out_name = (
        f"init_trunk_all_n{args.n_levels}.json"
        if init_only
        else f"crossover_all_n{args.n_levels}.json"
    )
    (out_dir / out_name).write_text(json.dumps(all_results, indent=2))
    return all_results


def run_crossover_ext(args):
    """Crossover extensions: mid-ladder, adjacent, cross-seed and actor/critic-only head swaps."""
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.n_levels
    results: dict = {"n_levels": n, "cells": {}}

    # Mid-ladder + adjacent (per seed). Targets ≈20/40/80/180/200M; snap to disk.
    mid_specs = {
        6033: [
            ("20x80", 20_000_000, 80_000_000),
            ("80x200", 80_000_000, 200_000_000),
            ("180x200", 180_000_000, 200_000_000),
            ("adjacent_180t_200h", 180_000_000, 200_000_000),
            ("within_gm_20x80", 20_000_000, 80_000_000),
        ],
        6035: [
            ("40x80", 40_000_000, 80_000_000),
            ("80x200", 80_000_000, 200_000_000),
            ("180x200", 180_000_000, 200_000_000),
            ("adjacent_180t_200h", 180_000_000, 200_000_000),
            ("within_gm_40x80", 40_000_000, 80_000_000),
        ],
    }
    seeds = [args.seed] if args.seed else (args.seeds_list or [6033, 6035])
    for seed in seeds:
        for name, t_steps, h_steps in mid_specs[seed]:
            ts, tp = _nearest_ckpt(seed, t_steps)
            hs, hp = _nearest_ckpt(seed, h_steps)
            key = f"seed{seed}_{name}_t{ts}_h{hs}"
            print(f"\n=== ext mid {key} n={n} ===")
            results["cells"][key] = _eval_cell(
                key, tp, hp, device=device, n_levels=n, max_steps=args.max_steps
            )

    # Cross-seed negative (early×early and late×late trunks/heads swapped)
    for t_seed, h_seed in ((6033, 6035), (6035, 6033)):
        for tag in ("early", "late"):
            ts = ANCHORS[t_seed][tag]
            hs = ANCHORS[h_seed][tag]
            tp = ckpt_path(t_seed, ts)
            hp = ckpt_path(h_seed, hs)
            key = f"cross_t{t_seed}{tag}_h{h_seed}{tag}"
            print(f"\n=== ext cross {key} n={n} ===")
            results["cells"][key] = _eval_cell(
                key, tp, hp, device=device, n_levels=n, max_steps=args.max_steps
            )

    # Actor-only vs critic-only vs both on the early-trunk x late-head cell
    for seed in seeds:
        early_p = ckpt_path(seed, ANCHORS[seed]["early"])
        late_p = ckpt_path(seed, ANCHORS[seed]["late"])
        for parts in ("both", "actor", "critic"):
            key = f"seed{seed}_early_t_late_h_parts_{parts}"
            print(f"\n=== ext head_parts {key} n={n} ===")
            results["cells"][key] = _eval_cell(
                key,
                early_p,
                late_p,
                device=device,
                n_levels=n,
                max_steps=args.max_steps,
                head_parts=parts,
            )

    path = out_dir / f"crossover_ext_n{n}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return results


def run_ledge_invis(args):
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    action_mode = getattr(args, "action_mode", None) or "greedy"

    seeds = [args.seed] if args.seed else (args.seeds_list or list(ANCHORS))
    tag_filter = None
    if getattr(args, "tags", None):
        tag_filter = {t.strip() for t in args.tags.split(",") if t.strip()}

    for seed in seeds:
        if seed not in ANCHORS:
            raise SystemExit(f"no anchors for seed {seed}; known={sorted(ANCHORS)}")
        anchors = ANCHORS[seed]
        for tag, steps in anchors.items():
            if tag_filter and tag not in tag_filter:
                continue
            ckpt = ckpt_path(seed, steps)
            for hide in (False, True):
                key = f"seed{seed}_{tag}_{steps}_{action_mode}_hide{int(hide)}"
                print(f"\n=== ledge_invis {key} n={args.n_levels} ===")
                policy = load_policy(ckpt, device)
                summary = eval_policy_val(
                    policy,
                    n_levels=args.n_levels,
                    start_level=VAL_START,
                    device=device,
                    max_steps=args.max_steps,
                    action_mode=action_mode,
                    hide_visible_coin=hide,
                )
                summary["seed"] = seed
                summary["steps"] = steps
                summary["tag"] = tag
                summary["checkpoint"] = str(ckpt)
                results[key] = summary
                print(
                    f"  success={summary['success_rate']:.3f} gm={summary['gm_rate']:.3f} "
                    f"jump_n={summary.get('ledge_counts', {}).get('jump_required')} "
                    f"flat_n={summary.get('ledge_counts', {}).get('flat_ground')}"
                )
                if "strata" in summary:
                    for sname, s in summary["strata"].items():
                        if s.get("n_levels", 0) > 0:
                            print(
                                f"    [{sname}] n={s['n_levels']} "
                                f"success={s['success_rate']:.3f} gm={s['gm_rate']:.3f}"
                            )
                del policy
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

    # Coin-dependence deltas for each seed/tag/mode
    deps = {}
    for seed in seeds:
        for tag in (tag_filter or ANCHORS[seed].keys()):
            steps = ANCHORS[seed][tag]
            k0 = f"seed{seed}_{tag}_{steps}_{action_mode}_hide0"
            k1 = f"seed{seed}_{tag}_{steps}_{action_mode}_hide1"
            if k0 in results and k1 in results:
                deps[f"seed{seed}_{tag}_{action_mode}"] = {
                    "visible": results[k0]["success_rate"],
                    "hidden": results[k1]["success_rate"],
                    "coin_dependence": results[k0]["success_rate"] - results[k1]["success_rate"],
                }
    results["_coin_dependence"] = deps

    seed_suffix = f"_seed{args.seed}" if args.seed else ""
    tag_suffix = f"_{args.tags.replace(',', '-')}" if getattr(args, "tags", None) else ""
    mode_suffix = f"_{action_mode}" if action_mode != "greedy" else ""
    path = out_dir / f"ledge_invis{seed_suffix}{tag_suffix}{mode_suffix}_n{args.n_levels}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    for k, v in deps.items():
        print(
            f"  dep {k}: vis={v['visible']:.3f} hid={v['hidden']:.3f} "
            f"Δ={v['coin_dependence']:+.3f}"
        )
    return results


def _level_bootstrap_rate_diff(
    outcomes_a: list[str],
    outcomes_b: list[str],
    *,
    target: str = "goal_misgen",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Level-clustered bootstrap on rate(target)_a - rate(target)_b (paired levels)."""
    assert len(outcomes_a) == len(outcomes_b)
    n = len(outcomes_a)
    ya = np.array([o == target for o in outcomes_a], dtype=np.float64)
    yb = np.array([o == target for o in outcomes_b], dtype=np.float64)
    delta = float(ya.mean() - yb.mean())
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    idx = np.arange(n)
    for i in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        boots[i] = ya[b].mean() - yb[b].mean()
    return {
        "delta": delta,
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
        "n": n,
        "target": target,
    }


def run_head_bootstrap(args):
    """Level-clustered CIs on GM-rate head contrasts (same trunk, swapped head); ``--cut pre_fc`` puts embedder.fc with the head."""
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = getattr(args, "cut", "post_fc") or "post_fc"
    seeds = [args.seed] if args.seed else (args.seeds_list or [6033, 6035])
    results = {
        "meta": {"n_levels": args.n_levels, "n_boot": 2000, "cut": cut},
        "seeds": {},
    }

    for seed in seeds:
        early_p = ckpt_path(seed, ANCHORS[seed]["early"])
        late_p = ckpt_path(seed, ANCHORS[seed]["late"])
        cells = {
            "early_trunk_early_head": (early_p, early_p),
            "early_trunk_late_head": (early_p, late_p),
            "late_trunk_early_head": (late_p, early_p),
            "late_trunk_late_head": (late_p, late_p),
        }
        seed_out = {}
        ep_cache = {}
        for name, (tp, hp) in cells.items():
            print(f"\n=== head_bootstrap cut={cut} {seed} {name} n={args.n_levels} ===")
            policy = crossover_policy(tp, hp, device, cut=cut)
            summary = eval_policy_val(
                policy,
                n_levels=args.n_levels,
                start_level=VAL_START,
                device=device,
                max_steps=args.max_steps,
                action_mode="greedy",
                keep_episodes=True,
                track_ledge=False,
            )
            outs = [e["outcome"] for e in summary["episodes"]]
            ep_cache[name] = outs
            seed_out[name] = {
                "success_rate": summary["success_rate"],
                "gm_rate": summary["gm_rate"],
                "outcome_rates": summary["outcome_rates"],
                "trunk": str(tp),
                "head": str(hp),
            }
            print(f"  succ={summary['success_rate']:.3f} gm={summary['gm_rate']:.3f}")
            del policy
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        contrasts = {
            "head_on_early_trunk": _level_bootstrap_rate_diff(
                ep_cache["early_trunk_late_head"], ep_cache["early_trunk_early_head"]
            ),
            "head_on_late_trunk": _level_bootstrap_rate_diff(
                ep_cache["late_trunk_late_head"], ep_cache["late_trunk_early_head"]
            ),
            "trunk_main_early_heads": _level_bootstrap_rate_diff(
                ep_cache["early_trunk_early_head"], ep_cache["late_trunk_early_head"]
            ),
            "trunk_main_late_heads": _level_bootstrap_rate_diff(
                ep_cache["early_trunk_late_head"], ep_cache["late_trunk_late_head"]
            ),
        }
        # Also success contrasts for competence axis
        succ = {
            k: _level_bootstrap_rate_diff(ep_cache[a], ep_cache[b], target="success")
            for k, (a, b) in {
                "head_on_early_trunk": ("early_trunk_late_head", "early_trunk_early_head"),
                "head_on_late_trunk": ("late_trunk_late_head", "late_trunk_early_head"),
            }.items()
        }
        seed_out["gm_contrasts"] = contrasts
        seed_out["success_contrasts"] = succ
        results["seeds"][str(seed)] = seed_out
        for k, v in contrasts.items():
            print(f"  GM {k}: Δ={v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]")

    tag = "prefec_" if cut == "pre_fc" else ""
    path = out_dir / f"head_bootstrap_{tag}n{args.n_levels}.json"
    if args.seed:
        path = out_dir / f"head_bootstrap_{tag}seed{args.seed}_n{args.n_levels}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return results


def run_init_trunk_reach(args):
    """Whether a random-init trunk can reach the end region at all."""
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed or 6033
    late_p = ckpt_path(seed, ANCHORS[seed]["late"])
    early_p = ckpt_path(seed, ANCHORS[seed]["early"])
    cells = {
        "init_trunk_late_head": (None, late_p, True),
        "init_trunk_early_head": (None, early_p, True),
        "init_trunk_init_head": (None, None, True),
        "early_trunk_early_head": (early_p, early_p, False),
    }
    results = {}
    for name, (tp, hp, init_trunk) in cells.items():
        print(f"\n=== init_reach {seed} {name} n={args.n_levels} ===")
        if name == "init_trunk_init_head":
            policy = load_init_policy(device)
        else:
            policy = crossover_policy(tp, hp, device, init_trunk=init_trunk)
        # Val aisc (preference) + train rp=0 (proxy path)
        for env_name, rp, split, start in (
            ("coinrun_aisc", None, "val", VAL_START),
            ("coinrun", 0.0, "train", 0),
        ):
            summary = eval_policy_val(
                policy,
                n_levels=args.n_levels,
                start_level=start,
                device=device,
                max_steps=args.max_steps,
                action_mode="greedy",
                keep_episodes=True,
                track_ledge=False,
                env_name=env_name,
                random_percent=rp,
            )
            # Drop heavy episode list from disk after aggregating
            key = f"{name}_{split}"
            results[key] = {
                "success_rate": summary["success_rate"],
                "gm_rate": summary["gm_rate"],
                "outcome_rates": summary["outcome_rates"],
                "end_reach_rate": summary["end_reach_rate"],
                "mean_max_progress": summary["mean_max_progress"],
                "env_name": env_name,
            }
            print(
                f"  [{split}] succ={summary['success_rate']:.3f} gm={summary['gm_rate']:.3f} "
                f"end_reach={summary['end_reach_rate']:.3f} "
                f"mean_max_prog={summary['mean_max_progress']:.3f}"
            )
        del policy
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    path = out_dir / f"init_trunk_reach_seed{seed}_n{args.n_levels}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return results


def run_train_gm_traj(args):
    """In-distribution train GM under greedy decoding at the early/late anchors (is rising train GM shared across seeds?)."""
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed else (args.seeds_list or [6033, 6034, 6035])
    results = {}
    for seed in seeds:
        for tag, steps in ANCHORS[seed].items():
            ckpt = ckpt_path(seed, steps)
            key = f"seed{seed}_{tag}_{steps}_train_greedy"
            print(f"\n=== train_gm_traj {key} n={args.n_levels} ===")
            policy = load_policy(ckpt, device)
            summary = eval_policy_val(
                policy,
                n_levels=args.n_levels,
                start_level=0,
                device=device,
                max_steps=args.max_steps,
                action_mode="greedy",
                hide_visible_coin=False,
                track_ledge=False,
                env_name="coinrun",
                random_percent=0.0,
            )
            summary.update(
                {
                    "seed": seed,
                    "steps": steps,
                    "tag": tag,
                    "split": "train",
                    "checkpoint": str(ckpt),
                }
            )
            results[key] = summary
            print(
                f"  train succ={summary['success_rate']:.3f} "
                f"gm={summary['gm_rate']:.3f} "
                f"finish={summary['success_rate']+summary['gm_rate']:.3f}"
            )
            del policy
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    seed_suffix = f"_seed{args.seed}" if args.seed else ""
    path = out_dir / f"train_gm_traj{seed_suffix}_n{args.n_levels}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return results


def run_graded(args):
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed else (args.seeds_list or [6033, 6035])
    action_mode = getattr(args, "action_mode", None) or "greedy"
    step_allow = None
    if getattr(args, "steps", None):
        step_allow = {int(s.strip()) for s in args.steps.split(",") if s.strip()}
    all_results = {}

    for seed in seeds:
        run_dir = SEED_DIRS[seed]
        ckpts = sorted(run_dir.glob("model_*.pth"))
        seed_res = []
        for ckpt in ckpts:
            steps = int(ckpt.stem.split("_")[1])
            if step_allow is not None and steps not in step_allow:
                continue
            print(
                f"\n=== graded seed={seed} steps={steps} "
                f"n={args.n_levels} mode={action_mode} ==="
            )
            policy = load_policy(ckpt, device)
            per_ep = []
            for i in range(args.n_levels):
                env = make_env(
                    "coinrun_aisc",
                    num_levels=1,
                    start_level=VAL_START + i,
                )
                ep = rollout_graded(
                    policy,
                    env,
                    device=device,
                    max_steps=args.max_steps,
                    action_mode=action_mode,
                )
                if hasattr(env, "close"):
                    env.close()
                per_ep.append(ep)
                if (i + 1) % 50 == 0:
                    cds = [e["coin_minus_end"] for e in per_ep]
                    dds = [e["disagree_coin_minus_end"] for e in per_ep]
                    print(
                        f"  … {i+1}/{args.n_levels} "
                        f"coin−end={np.nanmean(cds):.3f} "
                        f"disagree coin−end={np.nanmean(dds):.3f}"
                    )
            outcomes = [e["outcome"] for e in per_ep]
            summary = rate_summary(outcomes)
            summary["steps"] = steps
            summary["checkpoint"] = str(ckpt)
            summary["action_mode"] = action_mode
            summary["coin_directed"] = bootstrap_mean(
                np.array([e["coin_directed_rate"] for e in per_ep])
            )
            summary["end_directed"] = bootstrap_mean(
                np.array([e["end_directed_rate"] for e in per_ep])
            )
            summary["coin_minus_end"] = bootstrap_mean(
                np.array([e["coin_minus_end"] for e in per_ep])
            )
            summary["disagree_coin_minus_end"] = bootstrap_mean(
                np.array([e["disagree_coin_minus_end"] for e in per_ep])
            )
            summary["disagree_frac_steps"] = float(
                np.nansum([e["disagree_n"] for e in per_ep])
                / max(1, np.nansum([e["coin_n"] for e in per_ep]))
            )
            # Keep rates + disagreement subset; no per-step arrays.
            summary["per_level"] = [
                {
                    "outcome": e["outcome"],
                    "coin_directed_rate": e["coin_directed_rate"],
                    "end_directed_rate": e["end_directed_rate"],
                    "coin_minus_end": e["coin_minus_end"],
                    "disagree_n": e["disagree_n"],
                    "disagree_coin_minus_end": e["disagree_coin_minus_end"],
                    "coin_requires_jump": e["coin_requires_jump"],
                }
                for e in per_ep
            ]
            seed_res.append(summary)
            dsum = summary["disagree_coin_minus_end"]
            print(
                f"  success={summary['success_rate']:.3f} "
                f"coin−end={summary['coin_minus_end']['mean']:.3f} "
                f"[{summary['coin_minus_end']['lo']:.3f},{summary['coin_minus_end']['hi']:.3f}] "
                f"disagree coin−end={dsum['mean']:.3f} "
                f"[{dsum['lo']:.3f},{dsum['hi']:.3f}] "
                f"(disagree_frac≈{summary['disagree_frac_steps']:.2f})"
            )
            del policy
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        all_results[str(seed)] = seed_res
        suffix = "_disagree" if step_allow is not None else ""
        mode_suffix = f"_{action_mode}" if action_mode != "greedy" else ""
        path = out_dir / f"graded_seed{seed}_n{args.n_levels}{suffix}{mode_suffix}.json"
        path.write_text(json.dumps(seed_res, indent=2))
        print(f"wrote {path}")

    return all_results


def rollout_entropy(
    policy,
    env,
    *,
    device: str,
    max_steps: int = 1000,
    action_mode: str = "sample",
):
    """Collect per-step policy entropy along one episode.

    Entropy is of π(·|s) at visited states. Trajectory decode (`action_mode`)
    affects which states are visited; the entropy contrast uses sampling.
    """
    obs = env.reset()
    hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)
    ents: list[float] = []
    maxps: list[float] = []
    last_info = {}

    with torch.no_grad():
        for t in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            hid_t = torch.as_tensor(hidden, dtype=torch.float32, device=device)
            done_t = torch.zeros(env.num_envs, device=device)
            dist, _v, next_hid = policy(obs_t, hid_t, done_t)
            ents.append(float(dist.entropy().cpu().numpy()[0]))
            maxps.append(float(dist.probs.max(-1).values.cpu().numpy()[0]))
            act = dist.probs.argmax(-1) if action_mode == "greedy" else dist.sample()
            obs, _rew, done, info = env.step(act.cpu().numpy())
            hidden = next_hid.cpu().numpy()
            last_info = info[0]
            if last_info.get("invisible_coin_collected") and not done[0]:
                break
            if done[0]:
                break

    return {
        "n_steps": len(ents),
        "mean_entropy": float(np.mean(ents)) if ents else float("nan"),
        "mean_max_prob": float(np.mean(maxps)) if maxps else float("nan"),
        "frac_max_prob_lt_0.5": float(np.mean(np.asarray(maxps) < 0.5)) if maxps else float("nan"),
        "outcome": classify_done(last_info, max_steps) if last_info else "unknown",
    }


def run_policy_entropy(args):
    """Mean policy entropy at the early/late anchors, a direct test for a bimodal policy.

    Higher late entropy for 6034/6035 than for 6033 indicates a proxy mode mixed in.
    """
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    action_mode = getattr(args, "action_mode", None) or "sample"
    seeds = [args.seed] if args.seed else (args.seeds_list or [6033, 6034, 6035])
    tag_filter = None
    if args.tags:
        tag_filter = {t.strip() for t in args.tags.split(",") if t.strip()}

    results: dict = {}
    for seed in seeds:
        for tag, steps in ANCHORS[seed].items():
            if tag_filter is not None and tag not in tag_filter:
                continue
            ckpt = ckpt_path(seed, steps)
            key = f"seed{seed}_{tag}_{steps}_{action_mode}"
            print(f"\n=== policy_entropy {key} n={args.n_levels} ===")
            policy = load_policy(ckpt, device)
            ep_ents = []
            ep_maxps = []
            ep_fracs = []
            outcomes = []
            for i in range(args.n_levels):
                env = make_env(
                    "coinrun_aisc",
                    num_levels=1,
                    start_level=VAL_START + i,
                )
                ep = rollout_entropy(
                    policy,
                    env,
                    device=device,
                    max_steps=args.max_steps,
                    action_mode=action_mode,
                )
                if hasattr(env, "close"):
                    env.close()
                ep_ents.append(ep["mean_entropy"])
                ep_maxps.append(ep["mean_max_prob"])
                ep_fracs.append(ep["frac_max_prob_lt_0.5"])
                outcomes.append(ep["outcome"])
                if (i + 1) % 50 == 0 or i == 0:
                    print(
                        f"  [{i+1}/{args.n_levels}] "
                        f"H={np.mean(ep_ents):.3f} maxp={np.mean(ep_maxps):.3f}"
                    )

            summary = {
                "seed": seed,
                "tag": tag,
                "steps": steps,
                "checkpoint": str(ckpt),
                "action_mode": action_mode,
                "n_levels": args.n_levels,
                "env_name": "coinrun_aisc",
                "start_level": VAL_START,
                "mean_entropy": bootstrap_mean(np.asarray(ep_ents, dtype=np.float64)),
                "mean_max_prob": bootstrap_mean(np.asarray(ep_maxps, dtype=np.float64)),
                "frac_max_prob_lt_0.5": bootstrap_mean(np.asarray(ep_fracs, dtype=np.float64)),
                "outcome_rates": rate_summary(outcomes),
            }
            results[key] = summary
            h = summary["mean_entropy"]
            print(
                f"  H={h['mean']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}]  "
                f"maxp={summary['mean_max_prob']['mean']:.3f}  "
                f"frac_maxp<0.5={summary['frac_max_prob_lt_0.5']['mean']:.3f}"
            )
            del policy
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # Late-only contrast table for the prediction.
    late = {}
    for seed in (6033, 6034, 6035):
        steps = ANCHORS[seed]["late"]
        key = f"seed{seed}_late_{steps}_{action_mode}"
        if key in results:
            late[seed] = results[key]["mean_entropy"]["mean"]
    if len(late) >= 2:
        results["_late_contrast"] = {
            "prediction": "H(6034), H(6035) > H(6033) at late under sample traj",
            "late_mean_entropy": late,
            "6034_minus_6033": late.get(6034, float("nan")) - late.get(6033, float("nan")),
            "6035_minus_6033": late.get(6035, float("nan")) - late.get(6033, float("nan")),
        }
        print("\n=== late contrast ===")
        print(json.dumps(results["_late_contrast"], indent=2))

    path = out_dir / f"policy_entropy_n{args.n_levels}_{action_mode}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return results


def run_stochastic(args):
    """Greedy vs sample on 6035 anchors (and optional other seed)."""
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed or 6035
    results = {}
    for tag, steps in ANCHORS[seed].items():
        ckpt = ckpt_path(seed, steps)
        for mode in ("greedy", "sample"):
            key = f"seed{seed}_{tag}_{steps}_{mode}"
            print(f"\n=== stochastic {key} n={args.n_levels} ===")
            policy = load_policy(ckpt, device)
            # Train env (rp=0) + val aisc
            for env_name, rp, split in (
                ("coinrun", 0, "train"),
                ("coinrun_aisc", None, "val"),
            ):
                outcomes = []
                for i in range(args.n_levels):
                    env = make_env(
                        env_name,
                        num_levels=1,
                        start_level=(0 if split == "train" else VAL_START) + i,
                        random_percent=rp,
                    )
                    ep = rollout_episode(
                        policy,
                        env,
                        device=device,
                        max_steps=args.max_steps,
                        action_mode=mode,
                        track_directedness=False,
                    )
                    if hasattr(env, "close"):
                        env.close()
                    outcomes.append(ep["outcome"])
                summary = rate_summary(outcomes)
                summary.update(
                    {
                        "seed": seed,
                        "steps": steps,
                        "tag": tag,
                        "split": split,
                        "env_name": env_name,
                        "action_mode": mode,
                        "checkpoint": str(ckpt),
                    }
                )
                results[f"{key}_{split}"] = summary
                print(
                    f"  [{split}/{mode}] success={summary['success_rate']:.3f} "
                    f"gm={summary['gm_rate']:.3f}"
                )
            del policy
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    path = out_dir / f"stochastic_seed{seed}_n{args.n_levels}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return results


def run_crossover_prefec(args):
    """Four-cell factorial with cut before embedder.fc (conv stack vs fc+heads)."""
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    seeds = [args.seed] if args.seed else (args.seeds_list or [6033, 6035])
    for seed in seeds:
        early_s = ANCHORS[seed]["early"]
        late_s = ANCHORS[seed]["late"]
        early_p = ckpt_path(seed, early_s)
        late_p = ckpt_path(seed, late_s)
        cells = [
            ("early_trunk_early_head", early_p, early_p),
            ("early_trunk_late_head", early_p, late_p),
            ("late_trunk_early_head", late_p, early_p),
            ("late_trunk_late_head", late_p, late_p),
        ]
        for name, tp, hp in cells:
            key = f"seed{seed}_{name}_prefec"
            print(f"\n=== crossover_prefec {key} n={args.n_levels} ===")
            results[key] = _eval_cell(
                key,
                tp,
                hp,
                device=device,
                n_levels=args.n_levels,
                max_steps=args.max_steps,
                cut="pre_fc",
            )
    path = out_dir / f"crossover_prefec_n{args.n_levels}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    return results


def run_coin_dep(args):
    """Coin dependence (visible minus hidden-coin success) on aisc OOD levels or the training distribution (rp=0)."""
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_name = getattr(args, "env_name", "coinrun_aisc")
    rp = getattr(args, "random_percent", None)
    if env_name == "coinrun" and rp is None:
        rp = 0.0
    start = 0 if env_name == "coinrun" else VAL_START
    results = {}
    seeds = [args.seed] if args.seed else (args.seeds_list or [6033, 6035])
    tag_filter = None
    if getattr(args, "tags", None):
        tag_filter = {t.strip() for t in args.tags.split(",") if t.strip()}
    for seed in seeds:
        for tag, steps in ANCHORS[seed].items():
            if tag_filter and tag not in tag_filter:
                continue
            ckpt = ckpt_path(seed, steps)
            for hide in (False, True):
                key = (
                    f"seed{seed}_{tag}_{steps}_{env_name}_"
                    f"hide{int(hide)}_{args.action_mode}"
                )
                print(f"\n=== coin_dep {key} n={args.n_levels} ===")
                policy = load_policy(ckpt, device)
                summary = eval_policy_val(
                    policy,
                    n_levels=args.n_levels,
                    start_level=start,
                    device=device,
                    max_steps=args.max_steps,
                    action_mode=args.action_mode,
                    hide_visible_coin=hide,
                    env_name=env_name,
                    random_percent=rp,
                )
                summary["seed"] = seed
                summary["steps"] = steps
                summary["tag"] = tag
                summary["checkpoint"] = str(ckpt)
                results[key] = summary
                print(
                    f"  success={summary['success_rate']:.3f} "
                    f"gm={summary['gm_rate']:.3f} hide={hide}"
                )
                del policy
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    # attach coin-dependence deltas per seed/tag
    deps = {}
    for seed in seeds:
        for tag, steps in ANCHORS[seed].items():
            if tag_filter and tag not in tag_filter:
                continue
            k0 = f"seed{seed}_{tag}_{steps}_{env_name}_hide0_{args.action_mode}"
            k1 = f"seed{seed}_{tag}_{steps}_{env_name}_hide1_{args.action_mode}"
            if k0 in results and k1 in results:
                deps[f"seed{seed}_{tag}"] = {
                    "visible": results[k0]["success_rate"],
                    "hidden": results[k1]["success_rate"],
                    "coin_dependence": results[k0]["success_rate"]
                    - results[k1]["success_rate"],
                    "env_name": env_name,
                    "random_percent": rp,
                }
    out = {"cells": results, "coin_dependence": deps, "env_name": env_name}
    path = out_dir / f"coin_dep_{env_name}_n{args.n_levels}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    print("coin_dependence:", json.dumps(deps, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=(
            "crossover",
            "crossover_ext",
            "crossover_prefec",
            "ledge_invis",
            "coin_dep",
            "graded",
            "stochastic",
            "policy_entropy",
            "head_bootstrap",
            "init_trunk_reach",
            "train_gm_traj",
            "all_A",
        ),
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma/space list of seeds, overriding this mode's default "
            "multi-seed list (e.g. [6033, 6035]) when --seed is not given. "
            "Seeds must already be keys of SEED_DIRS/ANCHORS."
        ),
    )
    p.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma list of anchor tags (e.g. early). Default: all.",
    )
    p.add_argument(
        "--action-mode",
        type=str,
        default="greedy",
        choices=("greedy", "sample"),
        help="Decode for ledge_invis / coin_dep / graded (default greedy).",
    )
    p.add_argument(
        "--env-name",
        type=str,
        default="coinrun_aisc",
        help="Env for coin_dep (coinrun_aisc OOD vs coinrun train).",
    )
    p.add_argument(
        "--random-percent",
        type=float,
        default=None,
        help="Override random_percent (coin_dep train uses 0).",
    )
    p.add_argument("--n-levels", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma list of checkpoint timesteps for graded.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--include-init-trunk", action="store_true")
    p.add_argument("--init-trunk-only", action="store_true")
    p.add_argument(
        "--cut",
        type=str,
        default="post_fc",
        choices=("post_fc", "pre_fc"),
        help="Crossover cut for head_bootstrap (pre_fc moves embedder.fc to the head).",
    )
    args = p.parse_args()
    if args.device is None:
        args.device = str(_default_device())
    args.seeds_list = parse_seeds(args.seeds) if args.seeds else None
    if args.seeds_list is not None:
        unknown = [s for s in args.seeds_list if s not in SEED_DIRS]
        if unknown:
            raise SystemExit(f"--seeds has unknown seed(s) {unknown}; known={sorted(SEED_DIRS)}")

    if args.mode == "crossover":
        run_crossover(args)
    elif args.mode == "crossover_ext":
        run_crossover_ext(args)
    elif args.mode == "crossover_prefec":
        run_crossover_prefec(args)
    elif args.mode == "ledge_invis":
        run_ledge_invis(args)
    elif args.mode == "coin_dep":
        run_coin_dep(args)
    elif args.mode == "graded":
        run_graded(args)
    elif args.mode == "stochastic":
        run_stochastic(args)
    elif args.mode == "policy_entropy":
        run_policy_entropy(args)
    elif args.mode == "head_bootstrap":
        run_head_bootstrap(args)
    elif args.mode == "init_trunk_reach":
        run_init_trunk_reach(args)
    elif args.mode == "train_gm_traj":
        run_train_gm_traj(args)
    elif args.mode == "all_A":
        run_crossover(args)
        run_ledge_invis(args)
        run_graded(args)
        run_stochastic(args)


if __name__ == "__main__":
    main()
