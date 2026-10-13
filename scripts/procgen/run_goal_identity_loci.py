#!/usr/bin/env python3
"""Goal-identity loci: layer sweep (fc/pre_fc/block1-3 GAP) + nonlinear (MLP) probe at fc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from gm_geometry.integrations.colour_shape import (  # noqa: E402
    load_impala_policy,
    load_recurrent_flag,
    make_duel_env,
    repo_root,
)
from run_random_init_probes import load_untrained_policy  # noqa: E402
from gm_geometry.probes.choice_point import register_layer_hooks, tensor_to_feature  # noqa: E402
from gm_geometry.probes.configs import (  # noqa: E402
    GOAL_IDENTITY_CONCEPTS,
    GOAL_IDENTITY_GRID_TAG,
    GOAL_IDENTITY_OOD_CONFIGS,
    GOAL_IDENTITY_TRAIN_CONFIGS,
    audit_goal_identity_grids,
)
from gm_geometry.probes.labels import ground_truth_presence  # noqa: E402
from gm_geometry.probes.linear import (  # noqa: E402
    fit_ridge,
    predict_ridge,
    roc_auc,
    shuffle_labels_by_groups,
    shuffle_null_ood_auc,
)
from gm_geometry.utils import device as _default_device  # noqa: E402

# Same four-way discriminating grid and group-shuffle null as run_goal_identity_vs_presence.py,
# so results are comparable across the two scripts.
CONCEPTS = GOAL_IDENTITY_CONCEPTS
VECTOR_LAYERS = ("fc", "pre_fc", "block3_gap", "block2_gap", "block1_gap")
HOOK_LAYERS = ("fc", "pre_fc", "block3", "block2", "block1")


def _gap(x: np.ndarray) -> np.ndarray:
    """Global average pool (B,C,H,W) → (B,C)."""
    if x.ndim == 4:
        return x.mean(axis=(2, 3)).astype(np.float32)
    return x.astype(np.float32)


def _to_vector(name: str, t: torch.Tensor) -> np.ndarray:
    if name.endswith("_gap") or name in ("block1", "block2", "block3"):
        raw = tensor_to_feature(name.replace("_gap", ""), t)
        return _gap(raw) if raw.ndim == 4 else raw.astype(np.float32)
    feat = tensor_to_feature(name, t)
    if feat.ndim == 4:
        return feat.reshape(feat.shape[0], -1).astype(np.float32)
    return feat.astype(np.float32)


def collect_multilayer_grid(
    policy,
    configs: list[tuple[str, str]],
    *,
    num_envs: int,
    num_levels: int,
    start_level: int,
    n_steps: int,
    num_threads: int,
    use_backgrounds: bool,
    device: str = "cpu",
    return_observations: bool = False,
    rand_seed: int | None = None,
) -> dict:
    """Roll out configs and return per-layer activations, labels and groups.

    return_observations keeps the raw frames so an untrained control can be probed on the same frames (no
    visitation confound). With rand_seed=None procgen draws a random level-sampling seed and the rollout is not
    reproducible.
    """
    stores = {L: [] for L in VECTOR_LAYERS}
    labels_acc: dict[str, list] = {c: [] for c in CONCEPTS}
    groups = []
    obs_store: list[np.ndarray] = []
    group_id = 0

    for obj1, obj2 in configs:
        env = make_duel_env(
            num_envs, num_levels, num_threads, obj1, obj2,
            start_level=start_level, use_backgrounds=use_backgrounds,
            rand_seed=rand_seed,
        )
        hook_store, handles = register_layer_hooks(policy, list(HOOK_LAYERS))
        obs = env.reset()
        hidden = np.zeros((env.num_envs, policy.embedder.output_dim), dtype=np.float32)
        for _ in range(n_steps):
            for k in HOOK_LAYERS:
                hook_store[k].clear()
            if return_observations:
                obs_store.append(obs.copy())
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                act_dist, _v, hidden_t = policy(
                    obs_t,
                    torch.as_tensor(hidden, device=device),
                    torch.zeros(env.num_envs, device=device),
                )
            hidden = hidden_t.detach().cpu().numpy()
            stores["fc"].append(_to_vector("fc", hook_store["fc"][-1]))
            stores["pre_fc"].append(_to_vector("pre_fc", hook_store["pre_fc"][-1]))
            stores["block3_gap"].append(_to_vector("block3", hook_store["block3"][-1]))
            stores["block2_gap"].append(_to_vector("block2", hook_store["block2"][-1]))
            stores["block1_gap"].append(_to_vector("block1", hook_store["block1"][-1]))
            obs, _r, done, _i = env.step(act_dist.sample().detach().cpu().numpy())
        for h in handles:
            h.remove()
        env.close()

        gt = ground_truth_presence(obj1, obj2)
        n = num_envs * n_steps
        for c in CONCEPTS:
            labels_acc[c].append(np.full(n, gt[c], dtype=np.float32))
        groups.append(np.full(n, group_id, dtype=np.int32))
        group_id += 1

    out = {
        "activations": {L: np.concatenate(stores[L], axis=0) for L in VECTOR_LAYERS},
        "labels": {c: np.concatenate(labels_acc[c], axis=0) for c in CONCEPTS},
        "groups": np.concatenate(groups, axis=0),
    }
    if return_observations:
        out["observations"] = np.concatenate(obs_store, axis=0)
    return out


def replay_multilayer_activations(
    policy,
    observations: np.ndarray,
    *,
    batch_size: int = 256,
    device: str = "cpu",
) -> dict:
    """Forward fixed frames through ``policy`` (any weights) at all ``VECTOR_LAYERS``.

    Used for the init-floor control: same frames as the trained-policy rollout,
    swapped-in random-init weights, so the comparison isolates representation
    from visitation.
    """
    policy.eval()
    hook_store, handles = register_layer_hooks(policy, list(HOOK_LAYERS))
    stores = {L: [] for L in VECTOR_LAYERS}
    obs = np.asarray(observations, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            batch = obs[start : start + batch_size]
            b = len(batch)
            for k in HOOK_LAYERS:
                hook_store[k].clear()
            obs_t = torch.as_tensor(batch, dtype=torch.float32, device=device)
            hidden = torch.zeros(b, policy.embedder.output_dim, device=device)
            masks = torch.zeros(b, device=device)
            policy(obs_t, hidden, masks)
            stores["fc"].append(_to_vector("fc", hook_store["fc"][-1]))
            stores["pre_fc"].append(_to_vector("pre_fc", hook_store["pre_fc"][-1]))
            stores["block3_gap"].append(_to_vector("block3", hook_store["block3"][-1]))
            stores["block2_gap"].append(_to_vector("block2", hook_store["block2"][-1]))
            stores["block1_gap"].append(_to_vector("block1", hook_store["block1"][-1]))
    for h in handles:
        h.remove()
    return {L: np.concatenate(stores[L], axis=0) for L in VECTOR_LAYERS}


def _linear_null(train_x, train_y, ood_x, ood_y, groups, *, alpha, n_shuffles, rng) -> dict:
    if len(np.unique(train_y)) < 2 or len(np.unique(ood_y)) < 2:
        return {"ood_auc": float("nan"), "error": "degenerate_label"}
    return shuffle_null_ood_auc(
        train_x, train_y, ood_x, ood_y, groups,
        alpha=alpha, n_shuffles=n_shuffles, rng=rng,
    )


def _mlp_ood_auc(
    train_x, train_y, ood_x, ood_y, groups, *, n_shuffles: int, rng: np.random.RandomState, seed: int,
) -> dict:
    """sklearn MLPClassifier at fc; group-shuffle null on train labels."""
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        return {"ood_auc": float("nan"), "error": f"sklearn_missing:{e}"}

    if len(np.unique(train_y)) < 2 or len(np.unique(ood_y)) < 2:
        return {"ood_auc": float("nan"), "error": "degenerate_label"}

    def fit_score(y_tr: np.ndarray) -> float:
        sc = StandardScaler()
        x_tr = sc.fit_transform(train_x)
        x_te = sc.transform(ood_x)
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
        clf.fit(x_tr, y_tr.astype(int))
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(x_te)[:, 1]
        else:
            scores = clf.decision_function(x_te)
        return float(roc_auc(ood_y, scores))

    real = fit_score(train_y)
    null = []
    for _ in range(n_shuffles):
        y_sh = shuffle_labels_by_groups(train_y, groups, rng)
        if len(np.unique(y_sh)) < 2:
            null.append(0.5)
            continue
        null.append(fit_score(y_sh))
    null_arr = np.asarray(null, dtype=np.float64)
    return {
        "ood_auc": real,
        "shuffled_ood_auc_mean": float(null_arr.mean()),
        "shuffled_ood_auc_std": float(null_arr.std()),
        "shuffled_ood_auc_p95": float(np.quantile(null_arr, 0.95)),
        "shuffle_null_percentile": float((null_arr <= real).mean()),
        "beats_null_p95": bool(real > float(np.quantile(null_arr, 0.95))),
        "n_shuffles": int(n_shuffles),
        "probe": "mlp_128_64",
        "layer": "fc",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--arm", default="")
    p.add_argument("--use-backgrounds", action="store_true")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-levels", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--n-shuffles", type=int, default=64, help="null draws (MLP is expensive)")
    p.add_argument("--n-shuffles-linear", type=int, default=256)
    p.add_argument(
        "--seed", type=int, default=0,
        help="Label-permutation seed for the shuffle nulls (linear and MLP). The ridge fit is closed-form and "
        "train-vs-ood is a fixed config split, so this is the only linear-probe randomness.",
    )
    p.add_argument(
        "--mlp-fit-seed", type=int, default=0,
        help="MLPClassifier random_state (weight init and early-stopping split) for the MLP@fc probe; "
        "independent of --seed so probe-fitting variance is separable from permutation variance.",
    )
    p.add_argument("--skip-mlp", action="store_true")
    p.add_argument(
        "--skip-init-floor", action="store_true",
        help="Skip the untrained-network control (same frames replayed through a random-init policy at every "
        "layer). Report trained-minus-init: bright objects are linearly decodable even from an untrained conv net.",
    )
    p.add_argument("--init-seed", type=int, default=0, help="torch seed for the untrained policy")
    p.add_argument(
        "--rand-seed", type=int, default=None,
        help="procgen level-sampling seed. Unset (default), procgen draws a random seed and the rollout is "
        "not reproducible run-to-run; pass a value for a reproducible run.",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = args.device or str(_default_device())

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    recurrent = load_recurrent_flag()
    probe_env = make_duel_env(
        1, 1, 1, "yellow_line_diag", "blue_gem", use_backgrounds=args.use_backgrounds,
    )
    policy = load_impala_policy(
        ckpt, probe_env.observation_space.shape[0], probe_env.action_space.n, recurrent,
        device=device,
    )
    probe_env.close()
    policy.eval()

    audit_goal_identity_grids(assert_gates=True)
    need_frames = not args.skip_init_floor
    print(f"collect train ({len(GOAL_IDENTITY_TRAIN_CONFIGS)} configs)…", flush=True)
    train = collect_multilayer_grid(
        policy, list(GOAL_IDENTITY_TRAIN_CONFIGS),
        num_envs=args.num_envs, num_levels=args.num_levels, start_level=0,
        n_steps=args.n_steps, num_threads=args.num_threads,
        use_backgrounds=args.use_backgrounds, device=device,
        return_observations=need_frames, rand_seed=args.rand_seed,
    )
    print(f"collect ood ({len(GOAL_IDENTITY_OOD_CONFIGS)} configs)…", flush=True)
    ood = collect_multilayer_grid(
        policy, list(GOAL_IDENTITY_OOD_CONFIGS),
        num_envs=args.num_envs, num_levels=args.num_levels, start_level=10_000,
        n_steps=args.n_steps, num_threads=args.num_threads,
        use_backgrounds=args.use_backgrounds, device=device,
        return_observations=need_frames, rand_seed=args.rand_seed,
    )

    rng = np.random.RandomState(args.seed)
    layers_out: dict = {}
    for layer in VECTOR_LAYERS:
        print(f"linear @ {layer}…", flush=True)
        concepts = {}
        for concept in CONCEPTS:
            block = _linear_null(
                train["activations"][layer], train["labels"][concept],
                ood["activations"][layer], ood["labels"][concept],
                train["groups"],
                alpha=args.ridge_alpha,
                n_shuffles=args.n_shuffles_linear,
                rng=rng,
            )
            concepts[concept] = block
            pct = block.get("shuffle_null_percentile", float("nan"))
            print(f"  {concept}: auc={block.get('ood_auc', float('nan')):.3f} null%={pct}", flush=True)
        layers_out[layer] = {"probe": "ridge", "concepts": concepts}

    init_layers_out: dict | None = None
    trained_minus_init: dict | None = None
    if need_frames:
        print("init-floor: replaying same frames through untrained policy…", flush=True)
        torch.manual_seed(args.init_seed)
        # Rebuild a probe env only for its obs/action shapes (cheap, closed immediately).
        shape_env = make_duel_env(1, 1, 1, "yellow_line_diag", "blue_gem", use_backgrounds=args.use_backgrounds)
        init_policy = load_untrained_policy(
            shape_env.observation_space.shape[0], shape_env.action_space.n, recurrent,
        )
        shape_env.close()
        init_policy = init_policy.to(device)
        init_train_acts = replay_multilayer_activations(init_policy, train["observations"], device=device)
        init_ood_acts = replay_multilayer_activations(init_policy, ood["observations"], device=device)

        init_layers_out = {}
        trained_minus_init = {}
        for layer in VECTOR_LAYERS:
            print(f"linear (init-floor) @ {layer}…", flush=True)
            concepts = {}
            deltas = {}
            for concept in CONCEPTS:
                block = _linear_null(
                    init_train_acts[layer], train["labels"][concept],
                    init_ood_acts[layer], ood["labels"][concept],
                    train["groups"],
                    alpha=args.ridge_alpha,
                    n_shuffles=args.n_shuffles_linear,
                    rng=rng,
                )
                concepts[concept] = block
                trained_auc = layers_out[layer]["concepts"][concept].get("ood_auc", float("nan"))
                init_auc = block.get("ood_auc", float("nan"))
                deltas[concept] = trained_auc - init_auc
                print(
                    f"  {concept}: init_auc={init_auc:.3f} trained_auc={trained_auc:.3f} "
                    f"delta={deltas[concept]:.3f}", flush=True,
                )
            init_layers_out[layer] = {"probe": "ridge", "concepts": concepts}
            trained_minus_init[layer] = deltas

    mlp_out = None
    if not args.skip_mlp:
        print("MLP @ fc…", flush=True)
        mlp_out = {}
        for concept in CONCEPTS:
            # fewer shuffles for MLP — expensive
            block = _mlp_ood_auc(
                train["activations"]["fc"], train["labels"][concept],
                ood["activations"]["fc"], ood["labels"][concept],
                train["groups"],
                n_shuffles=args.n_shuffles,
                rng=rng,
                seed=args.mlp_fit_seed,
            )
            mlp_out[concept] = block
            print(
                f"  {concept}: auc={block.get('ood_auc', float('nan')):.3f} "
                f"null%={block.get('shuffle_null_percentile', float('nan'))} "
                f"beats95={block.get('beats_null_p95')}",
                flush=True,
            )

    out_root = Path(args.out_dir) if args.out_dir else repo_root() / "data/probes/goal_identity_loci_v2"
    out_dir = out_root / args.label.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "arm": args.arm,
        "checkpoint": str(ckpt),
        "use_backgrounds": bool(args.use_backgrounds),
        "grid": GOAL_IDENTITY_GRID_TAG,
        "n_train_configs": len(GOAL_IDENTITY_TRAIN_CONFIGS),
        "n_ood_configs": len(GOAL_IDENTITY_OOD_CONFIGS),
        "protocol": {
            "num_envs": args.num_envs,
            "num_levels": args.num_levels,
            "n_steps": args.n_steps,
            "num_threads": args.num_threads,
            "ridge_alpha": args.ridge_alpha,
            "n_shuffles": args.n_shuffles,
            "n_shuffles_linear": args.n_shuffles_linear,
            "seed": args.seed,
            "mlp_fit_seed": args.mlp_fit_seed,
            "rand_seed": args.rand_seed,
            "init_seed": args.init_seed,
            "note": (
                "Added 2026-09-22: none of these were previously recorded, so no "
                "prior goal_identity_loci_v2 run's exact rollout-size/seed args "
                "could be verified after the fact. rand_seed=null means the rollout was not "
                "reproducible (procgen drew a system-random level-sampling seed)."
            ),
        },
        "layers_linear": layers_out,
        "layers_linear_init_floor": init_layers_out,
        "trained_minus_init": trained_minus_init,
        "init_seed": args.init_seed if init_layers_out is not None else None,
        "mlp_fc": mlp_out,
        "note": (
            "Linear at each layer + MLP at fc. Same four-way discriminating grid as goal-identity. "
            "If goal_is_line fails linear everywhere AND MLP@fc, readout lead strengthens; "
            "if it appears at an earlier layer, story is early-rep / late-loss. "
            "layers_linear_init_floor replays the identical train/ood frames through an "
            "untrained policy of the same architecture (paired, not a separate rollout) -- "
            "report trained_minus_init, not raw trained ood_auc, as the learned-representation claim."
        ),
    }
    out_path = out_dir / "goal_identity_loci.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
