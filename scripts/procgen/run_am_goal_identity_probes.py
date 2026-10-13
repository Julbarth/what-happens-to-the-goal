#!/usr/bin/env python3
"""Seeded, frame-saving Attribute Maze goal-identity probes with 4 init draws and a pixel-block control."""

from __future__ import annotations

import os
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
from gm_geometry.integrations.colour_shape import (  # noqa: E402
    load_impala_policy,
    load_recurrent_flag,
    make_duel_env,
)
from gm_geometry.probes.configs import (  # noqa: E402
    GOAL_IDENTITY_OOD_CONFIGS,
    GOAL_IDENTITY_TRAIN_CONFIGS,
    audit_goal_identity_grids,
)
from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc, shuffle_null_ood_auc  # noqa: E402
from run_goal_identity_loci import (  # noqa: E402
    CONCEPTS,
    VECTOR_LAYERS,
    collect_multilayer_grid,
    replay_multilayer_activations,
)
from run_random_init_probes import load_untrained_policy  # noqa: E402

BLACK_CHECKPOINTS = (
    "Cprime_seed_6313:GM:6313_gm.pth",
    "Cprime_seed_6315:GM:6315_gm.pth",
    "Cprime_seed_6317:GM:6317_gm.pth",
    "B2prime_seed_4371:Non-GM:4371_teach.pth",
    "B2prime_seed_4372:Non-GM:4372_teach.pth",
    "B2prime_seed_4373:Non-GM:4373_teach.pth",
)
TEXTURE_CHECKPOINTS = (
    "C_seed_6301:GM:6301_gm.pth",
    "C_seed_6302:GM:6302_gm.pth",
    "C_seed_6303:GM:6303_gm.pth",
    "B2_seed_4361:Non-GM:4361_teach.pth",
    "B2_seed_4362:Non-GM:4362_teach.pth",
    "B2_seed_4363:Non-GM:4363_teach.pth",
)
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
DEFAULT_OUT = DATA_ROOT / "probes/goal_identity_loci_v2_rseed_20260925_seeded"
T_975_DF2 = 4.302652729911275
# Split offsets keep train and held-out action-sampling streams disjoint for every config index.
SPLIT_SEED_OFFSET = {"train": 0, "ood": 5000}


def pixel_features(obs: np.ndarray, size: int = 16) -> np.ndarray:
    """Block-averaged raw RGB (size x size per channel); identical to run_coinrun_concept_probes_grouped.pixel_features."""
    n, c, h, w = obs.shape
    hs, ws = h // size, w // size
    x = obs[:, :, : hs * size, : ws * size].reshape(n, c, size, hs, size, ws).mean(axis=(3, 5))
    return x.reshape(n, -1).astype(np.float32)


def action_seed(rollout_seed: int, split: str, config_idx: int) -> int:
    return 10_000 * rollout_seed + SPLIT_SEED_OFFSET[split] + config_idx


def collect_split_seeded(policy, configs, split: str, *, rollout_seed: int, start_level: int, args) -> dict:
    """collect_multilayer_grid one config at a time, with torch's action-sampling RNG seeded per config.

    Pinning procgen's rand_seed alone does not fix trajectories after step 0; the action sampler must be seeded too.
    """
    parts = []
    for i, cfg in enumerate(configs):
        torch.manual_seed(action_seed(rollout_seed, split, i))
        out = collect_multilayer_grid(
            policy, [cfg],
            num_envs=args.num_envs, num_levels=args.num_levels, start_level=start_level,
            n_steps=args.n_steps, num_threads=args.num_threads,
            use_backgrounds=args.background == "texture", device=args.device,
            return_observations=True, rand_seed=rollout_seed,
        )
        out["groups"] = np.full_like(out["groups"], i)
        parts.append(out)
    return {
        "activations": {L: np.concatenate([p["activations"][L] for p in parts]) for L in VECTOR_LAYERS},
        "labels": {c: np.concatenate([p["labels"][c] for p in parts]) for c in CONCEPTS},
        "groups": np.concatenate([p["groups"] for p in parts]),
        "observations": np.concatenate([p["observations"] for p in parts]),
    }


def to_uint8_exact(obs: np.ndarray) -> np.ndarray:
    u8 = np.rint(np.asarray(obs, dtype=np.float64) * 255.0).astype(np.uint8)
    if not np.array_equal(from_uint8(u8), np.asarray(obs, dtype=np.float32)):
        raise RuntimeError("frames are not exactly representable as uint8/255")
    return u8


def from_uint8(u8: np.ndarray) -> np.ndarray:
    return (u8.astype(np.float64) / 255.0).astype(np.float32)


def shape_policy_args(use_backgrounds: bool):
    env = make_duel_env(1, 1, 1, "yellow_line_diag", "blue_gem", use_backgrounds=use_backgrounds)
    shape = (env.observation_space.shape[0], env.action_space.n)
    env.close()
    return shape


def ood_auc(xtr, ytr, xte, yte, alpha: float) -> float:
    w, b = fit_ridge(xtr, ytr, alpha=alpha)
    return roc_auc(yte, predict_ridge(xte, w, b))


def run_one(args) -> None:
    """Collect seeded frames for one checkpoint x rollout seed, save them, then probe every network on the saved frames."""
    label, cohort, fname = args.checkpoint_spec.split(":")
    rs = args.rollout_seed
    torch.set_num_threads(args.torch_threads)
    out_root = Path(args.out_root)
    frame_path = out_root / "frames" / label / f"rseed{rs}.npz"
    raw_path = out_root / "raw" / label / f"rseed{rs}.csv"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    in_ch, n_act = shape_policy_args(args.background == "texture")
    recurrent = load_recurrent_flag()
    if recurrent:
        raise SystemExit("recurrent policy: replay with zero hidden state would not equal the rollout activations")
    ckpt = Path(args.ckpt_dir) / fname
    policy = load_impala_policy(ckpt, in_ch, n_act, recurrent, device=args.device)
    policy.eval()
    audit_goal_identity_grids(assert_gates=True)

    splits = {
        "train": collect_split_seeded(policy, list(GOAL_IDENTITY_TRAIN_CONFIGS), "train", rollout_seed=rs, start_level=0, args=args),
        "ood": collect_split_seeded(policy, list(GOAL_IDENTITY_OOD_CONFIGS), "ood", rollout_seed=rs, start_level=10_000, args=args),
    }
    save = {}
    for s, d in splits.items():
        save[f"{s}_obs_u8"] = to_uint8_exact(d["observations"])
        save[f"{s}_groups"] = d["groups"]
        for c in CONCEPTS:
            save[f"{s}_label_{c}"] = d["labels"][c]
    np.savez_compressed(frame_path, **save)
    meta = {
        "label": label, "cohort": cohort, "checkpoint": fname, "rollout_seed": rs,
        "procgen_rand_seed": rs, "action_seed_rule": "10000*rollout_seed + {train:0, ood:5000} + config_idx",
        "train_configs": [list(c) for c in GOAL_IDENTITY_TRAIN_CONFIGS],
        "ood_configs": [list(c) for c in GOAL_IDENTITY_OOD_CONFIGS],
        "num_envs": args.num_envs, "num_levels": args.num_levels, "n_steps": args.n_steps,
        "num_threads": args.num_threads, "torch_threads": args.torch_threads, "device": args.device,
        "start_level": {"train": 0, "ood": 10_000}, "use_backgrounds": args.background == "texture",
    }
    frame_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    # Everything below reads only the saved frames, so any later reprobe reproduces these numbers exactly.
    z = np.load(frame_path)
    obs = {s: from_uint8(z[f"{s}_obs_u8"]) for s in ("train", "ood")}
    lab = {s: {c: z[f"{s}_label_{c}"] for c in CONCEPTS} for s in ("train", "ood")}
    groups = z["train_groups"]

    nets = {"trained": policy}
    for seed in args.init_seeds:
        torch.manual_seed(seed)
        nets[f"init{seed}"] = load_untrained_policy(in_ch, n_act, recurrent).to(args.device)

    rows = []
    rng = np.random.RandomState(args.perm_seed)
    for name, net in nets.items():
        acts = {s: replay_multilayer_activations(net, obs[s], device=args.device) for s in ("train", "ood")}
        if name == "trained":
            for s in ("train", "ood"):
                for L in VECTOR_LAYERS:
                    err = float(np.max(np.abs(acts[s][L] - splits[s]["activations"][L])))
                    if err > 1e-3:
                        raise RuntimeError(f"replay != rollout activations at {s}/{L}: max|d|={err}")
        for L in VECTOR_LAYERS:
            for c in CONCEPTS:
                row = dict(label=label, cohort=cohort, rseed=rs, network=name, site=L, concept=c)
                if name == "trained" and args.n_shuffles_linear > 0:
                    blk = shuffle_null_ood_auc(
                        acts["train"][L], lab["train"][c], acts["ood"][L], lab["ood"][c], groups,
                        alpha=args.ridge_alpha, n_shuffles=args.n_shuffles_linear, rng=rng,
                    )
                    row.update(auc=blk["ood_auc"], shuffle_pct=blk["shuffle_null_percentile"])
                else:
                    row.update(auc=ood_auc(acts["train"][L], lab["train"][c], acts["ood"][L], lab["ood"][c], args.ridge_alpha),
                               shuffle_pct="")
                rows.append(row)
    px = {s: pixel_features(obs[s]) for s in ("train", "ood")}
    for c in CONCEPTS:
        rows.append(dict(label=label, cohort=cohort, rseed=rs, network="pixel", site="pixels16", concept=c,
                         auc=ood_auc(px["train"], lab["train"][c], px["ood"], lab["ood"][c], args.ridge_alpha),
                         shuffle_pct=""))
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{label} rseed{rs}: {len(rows)} rows, {time.time() - t0:.0f}s -> {raw_path}", flush=True)


def launch_all(args) -> None:
    """One fresh subprocess per checkpoint x rollout seed (procgen/torch state leaks across long-lived loops)."""
    jobs = [(spec, rs) for spec in args.checkpoints for rs in args.rollout_seeds]
    todo = [(s, r) for s, r in jobs
            if args.overwrite or not (Path(args.out_root) / "raw" / s.split(":")[0] / f"rseed{r}.csv").exists()]
    procs: list[subprocess.Popen] = []
    base = [sys.executable, "-W", "ignore", __file__, "one", "--out-root", str(args.out_root),
            "--ckpt-dir", str(args.ckpt_dir), "--background", args.background, "--device", args.device, "--torch-threads", str(args.torch_threads),
            "--init-seeds", *map(str, args.init_seeds), "--n-shuffles-linear", str(args.n_shuffles_linear)]
    while todo or procs:
        while todo and len(procs) < args.workers:
            spec, rs = todo.pop(0)
            procs.append(subprocess.Popen(base + ["--checkpoint-spec", spec, "--rollout-seed", str(rs)]))
        time.sleep(2)
        for p in list(procs):
            if p.poll() is not None:
                procs.remove(p)
                if p.returncode != 0:
                    raise SystemExit(f"subprocess failed ({p.returncode}): {p.args}")


def mean_sd_ci(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    half = T_975_DF2 * sd / np.sqrt(len(x)) if len(x) == 3 else float("nan")
    return {"mean": float(x.mean()), "sd": sd, "ci95": [float(x.mean() - half), float(x.mean() + half)], "n": len(x)}


def aggregate(args) -> None:
    """Checkpoint = unit: average rollout seeds (and init draws) within a checkpoint, then mean/SD/t-CI (df=2) across 3."""
    out_root = Path(args.out_root)
    rows = []
    for p in sorted((out_root / "raw").glob("*/rseed*.csv")):
        with p.open() as f:
            rows += list(csv.DictReader(f))
    for r in rows:
        r["auc"] = float(r["auc"])
        r["rseed"] = int(r["rseed"])
    with (out_root / "per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    labels = sorted({r["label"] for r in rows})
    cohort_of = {r["label"]: r["cohort"] for r in rows}
    init_draws = [f"init{s}" for s in args.init_seeds if s != 0]
    sites = list(VECTOR_LAYERS)

    def cell(label, network, site, concept):
        v = [r["auc"] for r in rows if r["label"] == label and r["network"] == network
             and r["site"] == site and r["concept"] == concept]
        if len(v) != len(args.rollout_seeds):
            raise RuntimeError(f"{label}/{network}/{site}/{concept}: {len(v)} rollout seeds")
        return float(np.mean(v))

    per_ckpt = []
    for lab in labels:
        for c in CONCEPTS:
            for L in sites:
                tr = cell(lab, "trained", L, c)
                i0 = cell(lab, "init0", L, c)
                draws = [cell(lab, d, L, c) for d in init_draws]
                shuf = [float(r["shuffle_pct"]) for r in rows if r["label"] == lab and r["network"] == "trained"
                        and r["site"] == L and r["concept"] == c and r["shuffle_pct"] != ""]
                per_ckpt.append(dict(
                    label=lab, cohort=cohort_of[lab], checkpoint=lab.split("_")[-1], site=L, concept=c,
                    trained_auc=tr, init0_auc=i0, delta_init0=tr - i0,
                    **{f"{d}_auc": v for d, v in zip(init_draws, draws)},
                    init_pooled_auc=float(np.mean(draws)), delta_init_pooled=tr - float(np.mean(draws)),
                    pixel_auc=cell(lab, "pixel", "pixels16", c),
                    shuffle_pct=float(np.mean(shuf)) if shuf else float("nan"),
                ))
    with (out_root / "per_checkpoint.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ckpt[0].keys()))
        w.writeheader()
        w.writerows(per_ckpt)

    summary = {"unit": "checkpoint (mean over rollout seeds; init_pooled also over init draws)",
               "ci": "95% t-interval across 3 checkpoints, df=2",
               "rollout_seeds": args.rollout_seeds, "init_draws_pooled": init_draws, "init_original": "init0",
               "cohorts": {}, "controls": {}}
    for coh in ("GM", "Non-GM"):
        summary["cohorts"][coh] = {}
        for L in sites:
            summary["cohorts"][coh][L] = {}
            for c in CONCEPTS:
                sel = [r for r in per_ckpt if r["cohort"] == coh and r["site"] == L and r["concept"] == c]
                summary["cohorts"][coh][L][c] = {k: mean_sd_ci([r[k] for r in sel]) for k in (
                    "trained_auc", "init0_auc", "delta_init0", "init_pooled_auc", "delta_init_pooled", "pixel_auc")}
    # Pooled over all six checkpoints' frame sets, as the paper's Init column. Two SDs: across the 6 frame
    # sets (paper convention for AM), and across the 3 init draws (Maze/Coin convention: across networks).
    for L in sites:
        summary["controls"][L] = {}
        for c in CONCEPTS:
            sel = [r for r in per_ckpt if r["site"] == L and r["concept"] == c]
            draw_means = [np.mean([r[f"{d}_auc"] for r in sel]) for d in init_draws]
            summary["controls"][L][c] = {
                "init_mean": float(np.mean([r["init_pooled_auc"] for r in sel])),
                "init_sd": float(np.std([r["init_pooled_auc"] for r in sel], ddof=1)),
                "init_sd_across_draws": float(np.std(draw_means, ddof=1)),
                "init_draw_means": {d: float(m) for d, m in zip(init_draws, draw_means)},
                "init0_mean": float(np.mean([r["init0_auc"] for r in sel])),
                "init0_sd": float(np.std([r["init0_auc"] for r in sel], ddof=1)),
                "pixel_mean": float(np.mean([r["pixel_auc"] for r in sel])),
                "pixel_sd": float(np.std([r["pixel_auc"] for r in sel], ddof=1)),
                "n_checkpoints": len(sel),
            }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_root / 'per_checkpoint.csv'} and {out_root / 'summary.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("all", "one", "aggregate"), nargs="?", default="all")
    p.add_argument("--background", choices=("black", "texture"), default="black")
    p.add_argument("--checkpoints", nargs="+", default=None, help="label:cohort:file (default: the --background cohort)")
    p.add_argument("--checkpoint-spec", help="one label:cohort:file (mode=one)")
    p.add_argument("--rollout-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rollout-seed", type=int, default=0, help="mode=one")
    p.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1000, 1001, 1002],
                   help="0 is the canonical single init; the rest are pooled as the multi-draw init")
    p.add_argument("--ckpt-dir", default=str(DATA_ROOT / "checkpoints/ras"))
    p.add_argument("--out-root", default=None, help="default: DEFAULT_OUT, or DEFAULT_OUT/texture")
    # CPU, not the project mps>cuda>cpu default: action sampling and float reduction order differ by device,
    # and the determinism guarantee only holds on a fixed device and thread count.
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-levels", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--n-shuffles-linear", type=int, default=256, help="group-shuffle null, trained network only")
    p.add_argument("--perm-seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.checkpoints is None:
        args.checkpoints = list(TEXTURE_CHECKPOINTS if args.background == "texture" else BLACK_CHECKPOINTS)
    if args.out_root is None:
        args.out_root = str(DEFAULT_OUT / "texture" if args.background == "texture" else DEFAULT_OUT)
    if args.mode == "one":
        run_one(args)
    elif args.mode == "all":
        launch_all(args)
        aggregate(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
