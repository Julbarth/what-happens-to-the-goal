#!/usr/bin/env python3
"""Random-init (and optional fixed-frame) linear probe baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from gm_geometry.integrations.colour_shape import (  # noqa: E402
    load_impala_policy,
    load_recurrent_flag,
    make_duel_env,
    repo_root,
)
from gm_geometry.probes import (  # noqa: E402
    STATE_PROBE_OOD_CONFIGS,
    STATE_PROBE_TRAIN_CONFIGS,
    activations_from_observations,
    collect_probe_grid,
    fit_ridge,
    predict_ridge,
    roc_auc,
)


def load_untrained_policy(in_channels: int, action_size: int, recurrent: bool):
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy

    model = ImpalaModel(in_channels=in_channels)
    policy = CategoricalPolicy(model, recurrent, action_size)
    policy.eval()
    return policy


def run_probe_metrics(h_train, h_ood, y_train, y_ood, *, alpha: float) -> dict:
    out = {}
    for concept in ("yellow_present", "line_present"):
        w, b = fit_ridge(h_train, y_train[concept], alpha=alpha)
        out[concept] = {
            "ood_auc": roc_auc(y_ood[concept], predict_ridge(h_ood, w, b)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Random-init probe baseline")
    parser.add_argument("--label", default="random_init")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-levels", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--fixed-frames-checkpoint",
        default=None,
        help="If set, collect observations from this trained checkpoint, then forward "
        "through random-init embedder (isolates encoding from visitation).",
    )
    parser.add_argument("--use-backgrounds", action="store_true")
    args = parser.parse_args()

    use_backgrounds = args.use_backgrounds
    out_dir = Path(args.out_dir) if args.out_dir else repo_root() / f"data/probes/baselines/{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    recurrent = load_recurrent_flag()
    probe_env = make_duel_env(1, 1, 1, "yellow_line_diag", "red_gem", use_backgrounds=use_backgrounds)
    random_policy = load_untrained_policy(
        probe_env.observation_space.shape[0], probe_env.action_space.n, recurrent,
    )

    mode = "rollout"
    if args.fixed_frames_checkpoint:
        mode = "fixed_frames"
        ckpt = Path(args.fixed_frames_checkpoint)
        frame_policy = load_impala_policy(
            ckpt, probe_env.observation_space.shape[0], probe_env.action_space.n, recurrent,
        )
        print(f"Collecting canonical frames from {ckpt.name}…")
        frame_data = collect_probe_grid(
            frame_policy, STATE_PROBE_TRAIN_CONFIGS, args.num_envs, args.num_levels, 0,
            args.n_steps, args.num_threads, use_backgrounds=use_backgrounds, return_observations=True,
        )
        frame_data_ood = collect_probe_grid(
            frame_policy, STATE_PROBE_OOD_CONFIGS, args.num_envs, args.num_levels, 10_000,
            args.n_steps, args.num_threads, use_backgrounds=use_backgrounds, return_observations=True,
        )
        print("Forwarding fixed frames through random-init embedder…")
        h_train = activations_from_observations(random_policy, frame_data["observations"])
        h_ood = activations_from_observations(random_policy, frame_data_ood["observations"])
        train_labels, ood_labels = frame_data["labels"], frame_data_ood["labels"]
    else:
        print("Rolling out random-init policy (visitation confound possible)…")
        train_data = collect_probe_grid(
            random_policy, STATE_PROBE_TRAIN_CONFIGS, args.num_envs, args.num_levels, 0,
            args.n_steps, args.num_threads, use_backgrounds=use_backgrounds,
        )
        ood_data = collect_probe_grid(
            random_policy, STATE_PROBE_OOD_CONFIGS, args.num_envs, args.num_levels, 10_000,
            args.n_steps, args.num_threads, use_backgrounds=use_backgrounds,
        )
        h_train, h_ood = train_data["activations"], ood_data["activations"]
        train_labels, ood_labels = train_data["labels"], ood_data["labels"]

    metrics = run_probe_metrics(
        h_train, h_ood, train_labels, ood_labels, alpha=args.ridge_alpha,
    )
    results = {
        "label": args.label,
        "mode": mode,
        "use_backgrounds": use_backgrounds,
        "fixed_frames_checkpoint": args.fixed_frames_checkpoint,
        "metrics": metrics,
    }
    out_path = out_dir / "random_init_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
