"""Shared rollout/eval-split/trace-panel harness for the validate_langosco_* scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from gm_geometry.utils import ci95

ACTION_MODES = ("greedy", "sample")

# Structural-GM corner-dwelling threshold; override per run with --corner-frac-threshold.
DEFAULT_CORNER_FRAC_THRESHOLD = 0.12


def chw_to_hwc01(obs: np.ndarray) -> np.ndarray:
    """CHW (optionally batched) float obs -> single HWC frame clipped to [0,1]."""
    x = np.asarray(obs)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] in (3, 4):
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x[..., :3], 0, 1)


def in_top_right(xy: tuple[float, float] | None, h: int, w: int, frac: float = 0.28) -> bool:
    if xy is None:
        return False
    x, y = xy
    return x >= (1.0 - frac) * w and y <= frac * h


def cheese_xy(frame_hwc: np.ndarray) -> tuple[float, float] | None:
    """Locate the maze cheese sprite as a bright orange-yellow blob.

    A green mask is wrong here: it fires on the green/olive wall-tile theme procgen assigns to some levels."""
    rgb = frame_hwc.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 0.85) & (g > 0.45) & (g < 0.95) & (b < 0.45) & (r > g) & ((r + g) / 2 - b > 0.35)
    if mask.sum() < 3:
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def select_action(dist, mode: str):
    """Greedy (argmax) or stochastic (sample) action from a policy distribution."""
    if mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {mode!r}")
    return dist.probs.argmax(-1) if mode == "greedy" else dist.sample()


def summarize_episodes(episodes: list[dict], outcome_keys: tuple[str, ...]) -> dict:
    """Aggregate per-episode dicts (reward/steps/outcome/[timed_out]) into a reward/steps/outcome-rate block with std and 95% CI."""
    rewards = np.asarray([e["reward"] for e in episodes], dtype=np.float64)
    steps = np.asarray([e["steps"] for e in episodes], dtype=np.float64)
    outcomes = [e["outcome"] for e in episodes]
    n = len(episodes)
    counts = {k: int(sum(o == k for o in outcomes)) for k in outcome_keys}
    mean_r, lo_r, hi_r = ci95(rewards)
    out = {
        "n_levels": n,
        "mean_reward": float(rewards.mean()),
        "median_reward": float(np.median(rewards)),
        "std_reward": float(rewards.std()),
        "ci95_reward": [lo_r, hi_r],
        "mean_steps": float(steps.mean()),
        "outcome_counts": counts,
        "outcome_rates": {k: counts[k] / n for k in counts},
        "success_rate": counts.get("success", 0) / n,
        "gm_rate": counts.get("goal_misgen", 0) / n,
    }
    if any("timed_out" in e for e in episodes):
        out["frac_timeout"] = float(np.mean([bool(e.get("timed_out", False)) for e in episodes]))
    return out


def eval_split(
    rollout_fn: Callable[..., dict],
    *,
    env_name: str,
    n_levels: int,
    start_level: int,
    outcome_keys: tuple[str, ...],
    save_traces_for: set[str] | None = None,
    max_traces_per_outcome: int = 3,
    extra_summary: dict | None = None,
) -> dict:
    """Run n_levels episodes via rollout_fn(start_level, save_frames), collect
    outcome traces, and return the common summary block (see summarize_episodes)."""
    save_traces_for = save_traces_for or set()
    episodes: list[dict] = []
    traces: dict[str, list] = {k: [] for k in outcome_keys}
    for i in range(n_levels):
        want_frames = any(len(traces[o]) < max_traces_per_outcome for o in save_traces_for)
        ep = rollout_fn(start_level=start_level + i, save_frames=want_frames)
        episodes.append(ep)
        if ep["outcome"] in save_traces_for and len(traces[ep["outcome"]]) < max_traces_per_outcome:
            item = {
                "level": start_level + i,
                "reward": ep["reward"],
                "steps": ep["steps"],
                "frames": ep.get("frames", []),
            }
            if "corner_frac" in ep:
                item["corner_frac"] = ep["corner_frac"]
            traces[ep["outcome"]].append(item)
        if (i + 1) % 20 == 0:
            mean_r = float(np.mean([e["reward"] for e in episodes]))
            print(f"  [{env_name}] {i+1}/{n_levels} mean_r={mean_r:.2f}")

    summary = summarize_episodes(episodes, outcome_keys)
    summary["env_name"] = env_name
    summary["start_level"] = start_level
    summary["traces"] = traces
    if extra_summary:
        summary.update(extra_summary)
    return summary


def strip_frames_for_json(block: dict) -> dict:
    out = dict(block)
    traces = {}
    for k, items in block.get("traces", {}).items():
        traces[k] = [{kk: vv for kk, vv in it.items() if kk != "frames"} for it in items]
    out["traces"] = traces
    return out


def save_trace_panels(traces: dict, out_dir: Path, prefix: str) -> list[str]:
    """Save horizontal frame-strip PNGs (+ first/mid/last stills) per traced episode."""
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for outcome, items in traces.items():
        for j, item in enumerate(items):
            frames = item.get("frames") or []
            if not frames:
                continue
            n = len(frames)
            fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 1.8))
            if n == 1:
                axes = [axes]
            for ax, fr in zip(axes, frames):
                ax.imshow(fr)
                ax.axis("off")
            fig.suptitle(
                f"{prefix} | {outcome} | level={item['level']} "
                f"r={item['reward']:.1f} steps={item['steps']}",
                fontsize=9,
            )
            fig.tight_layout()
            path = out_dir / f"{prefix}_{outcome}_{j}_level{item['level']}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path))
            for tag, fr in (("first", frames[0]), ("mid", frames[len(frames) // 2]), ("last", frames[-1])):
                arr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
                PILImage.fromarray(arr).save(out_dir / f"{prefix}_{outcome}_{j}_{tag}.png")
    return paths


def write_mirror(
    payload: dict,
    *,
    mirror_path: Path,
    is_default_out_dir: bool,
    action_mode: str,
) -> Path | None:
    """Refresh the canonical mirror JSON only for the default out-dir with greedy actions, so other runs cannot overwrite it."""
    if is_default_out_dir and action_mode == "greedy":
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_path.write_text(json.dumps(payload, indent=2))
        return mirror_path
    return None
