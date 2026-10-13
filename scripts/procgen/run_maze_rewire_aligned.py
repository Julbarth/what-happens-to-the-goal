#!/usr/bin/env python3
"""Maze frozen-trunk rewiring on one shared driver-balanced BC bank, per tap and label source (teacher or A*)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", ROOT / "data"))
for p in (ROOT / "src", ROOT / "scripts" / "procgen", ROOT / "external" / "train-procgen-pytorch", ROOT / "external" / "procgenAISC"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gm_geometry.rewire.maze import (  # noqa: E402
    ALIGNED_MAZE_MARKER, collect_balanced_obs_bank_maze,
    eval_cheese_reach, features_from_obs, sanity_gate, soft_set_cross_entropy, _ladder,
)
from gm_geometry.rewire.taps import freeze_fc, prepare_tap, trainable_parameters  # noqa: E402

GM_SEEDS = ["1080_gm", "1081_gm", "1082_gm"]
NONGM_SEEDS = ["1090_teach", "1091_teach", "1092_teach"]
LABEL_TEACHER = "1090_teach"


def ckpt_path(label: str, ckpt_root: Path) -> Path:
    return ckpt_root / f"{label}.pth"


def _load_published(bank_dir: Path) -> dict:
    meta = json.loads((bank_dir / "meta.json").read_text())
    if meta.get("marker") != ALIGNED_MAZE_MARKER:
        raise SystemExit(f"STOP: bank at {bank_dir} has marker {meta.get('marker')!r}")
    return {
        "obs_u8": np.load(bank_dir / "obs_u8.npy", mmap_mode="r"),
        "a_teacher": np.load(bank_dir / "a_teacher.npy"),
        "a_star": np.load(bank_dir / "a_star.npy"),
        "driver_id": np.load(bank_dir / "driver_id.npy"),
        "meta": meta,
    }


def build_or_load_bank(bank_dir: Path, args, ckpt_root: Path) -> dict:
    if (bank_dir / "DONE").exists():
        print(f"=== reusing published bank at {bank_dir} ===", flush=True)
        return _load_published(bank_dir)
    m = _ladder()
    drivers = [(s, m.load_policy(ckpt_path(s, ckpt_root), args.device))
               for s in GM_SEEDS + NONGM_SEEDS]
    teacher = m.load_policy(ckpt_path(LABEL_TEACHER, ckpt_root), args.device)
    per = int(np.ceil(args.collect_steps / len(drivers)))
    print(f"=== building bank: {len(drivers)} drivers x {per} frames ===", flush=True)
    bank = collect_balanced_obs_bank_maze(
        drivers, teacher, device=args.device, frames_per_driver=per,
        start_level=args.train_start_level, n_levels_to_try=args.n_levels_to_try,
    )
    # Per-process temp dir so concurrent builders never write into each other's files, then
    # one atomic publish. If another builder published first the rename fails; its bank
    # is equivalent (same drivers, same levels, same filter), so reload it rather than die.
    tmp = bank_dir.with_name(f"{bank_dir.name}.tmp.{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    for k in ("obs_u8", "a_teacher", "a_star", "driver_id"):
        np.save(tmp / f"{k}.npy", bank[k])
    (tmp / "meta.json").write_text(json.dumps(bank["meta"], indent=2))
    (tmp / "DONE").write_text(ALIGNED_MAZE_MARKER)
    try:
        tmp.rename(bank_dir)
        print(f"=== published bank to {bank_dir} ===", flush=True)
        return bank
    except OSError:
        if not (bank_dir / "DONE").exists():
            raise
        print(f"=== lost the publish race; reloading {bank_dir} ===", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return _load_published(bank_dir)


def build_arm(policy, tap: str, arm: str):
    """Returns (forward_fn, params, description). Mirrors the Coin and Attribute Maze aligned runners."""
    emb = policy.embedder
    if tap == "block3":
        if arm == "random_proj":
            freeze_fc(policy)
            params = list(trainable_parameters(policy, "fc"))
            desc = {"frozen": ["block1-3"], "fixed_random": ["embedder.fc (xavier, frozen)"],
                    "fresh": ["fc_policy", "fc_value"]}
        else:
            params = list(trainable_parameters(policy, "block3"))
            desc = {"frozen": ["block1-3"], "fixed_random": [],
                    "fresh": ["embedder.fc", "fc_policy", "fc_value"]}
        return (lambda xb: policy.fc_policy(F.relu(emb.fc(xb)))), params, desc
    params = list(trainable_parameters(policy, "fc"))
    return (lambda xb: policy.fc_policy(xb)), params, {
        "frozen": ["block1-3", "embedder.fc (learned)"], "fixed_random": [],
        "fresh": ["fc_policy", "fc_value"]}


def train(forward_fn, params, feats, labels, *, label_source, device, epochs, batch_size, lr):
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.RandomState(0)  # fixed split, identical for every arm
    n = feats.shape[0]
    idx = rng.permutation(n)
    val_idx, tr_idx = idx[: max(1, n // 5)], idx[max(1, n // 5):]
    log = []
    for ep in range(epochs):
        perm = rng.permutation(tr_idx)
        for s in range(0, len(perm), batch_size):
            b = np.sort(perm[s : s + batch_size])  # sorted: feats may be a memmap view
            xb = torch.as_tensor(feats[b], device=device)
            yb = torch.as_tensor(labels[b], device=device)
            opt.zero_grad()
            loss = (F.cross_entropy(forward_fn(xb), yb.long()) if label_source == "teacher"
                    else soft_set_cross_entropy(forward_fn(xb), yb.float()))
            loss.backward()
            # Same clipping (0.5) as the other rewiring runners, so optimizers match across environments.
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
        with torch.no_grad():
            xb = torch.as_tensor(feats[val_idx], device=device)
            yb = torch.as_tensor(labels[val_idx], device=device)
            lg = forward_fn(xb)
            if label_source == "teacher":
                vce = float(F.cross_entropy(lg, yb.long()))
                vacc = float((lg.argmax(-1) == yb.long()).float().mean())
            else:
                vce = float(soft_set_cross_entropy(lg, yb.float()))
                vacc = float(yb[torch.arange(len(yb)), lg.argmax(-1)].gt(0).float().mean())
        log.append({"epoch": ep, "val_ce": vce, "val_acc": vacc})
        print(f"    epoch {ep}: val_ce={vce:.4f} val_acc={vacc:.4f}", flush=True)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap", required=True, choices=("block3", "fc"))
    ap.add_argument("--label-source", required=True, choices=("teacher", "a_star"))
    ap.add_argument("--ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/maze")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--collect-steps", type=int, default=200_000)
    ap.add_argument("--train-start-level", type=int, default=2_000_000)
    ap.add_argument("--n-levels-to-try", type=int, default=20_000)
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--eval-start-level", type=int, default=1_000_000)
    ap.add_argument("--init-seed", type=int, default=0)
    args = ap.parse_args()

    m = _ladder()
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_root = args.ckpt_dir

    bank = build_or_load_bank(args.bank_dir, args, ckpt_root)
    labels = bank["a_teacher"] if args.label_source == "teacher" else bank["a_star"]
    obs = bank["obs_u8"]
    print(f"=== bank {bank['meta']['n_frames']} frames | labels={args.label_source} "
          f"| tap={args.tap} ===", flush=True)

    # Natives are tap- and label-independent; cache them beside the bank.
    native_dir = args.bank_dir / "natives"
    native_dir.mkdir(parents=True, exist_ok=True)
    natives = {}
    for s in GM_SEEDS + NONGM_SEEDS:
        cache = native_dir / f"{s}.json"
        if cache.exists():
            natives[s] = json.loads(cache.read_text())
        else:
            pol = m.load_policy(ckpt_path(s, ckpt_root), args.device)
            natives[s] = eval_cheese_reach(pol, args.device, n_levels=args.eval_n,
                                           start_level=args.eval_start_level)
            cache.write_text(json.dumps(natives[s], indent=2))
            del pol
        print(f"  native {s}: cheese_reach={natives[s]['cheese_reach']:.4f}", flush=True)

    arms = [("rewire", s, "gm") for s in GM_SEEDS] + [("rewire", s, "nongm") for s in NONGM_SEEDS]
    if args.tap == "block3":
        arms += [("random_proj", s, "gm") for s in GM_SEEDS]
        arms += [("random_proj", s, "nongm") for s in NONGM_SEEDS]

    results = {}
    for arm, seed, cohort in arms:
        name = f"{cohort}_{arm}_{seed}"
        t0 = time.time()
        print(f"\n=== {name} tap={args.tap} labels={args.label_source} ===", flush=True)
        pol = m.load_policy(ckpt_path(seed, ckpt_root), args.device)
        torch.manual_seed(args.init_seed)
        prepare_tap(pol, args.tap)
        pol.to(args.device)
        feats = features_from_obs(pol, obs, args.tap, device=args.device)
        fwd, params, desc = build_arm(pol, args.tap, arm)
        n_params = sum(p.numel() for p in params)
        print(f"    training ({n_params} params, {feats.shape[0]} frames)", flush=True)
        log = train(fwd, params, feats, labels, label_source=args.label_source,
                    device=args.device, epochs=args.bc_epochs,
                    batch_size=args.batch_size, lr=args.lr)
        ev = eval_cheese_reach(pol, args.device, n_levels=args.eval_n,
                               start_level=args.eval_start_level)
        gate = sanity_gate(ev, natives[seed])
        results[name] = {
            "arm": arm, "cohort": cohort, "seed": seed, "tap": args.tap,
            "label_source": args.label_source, "n_trainable_params": n_params,
            **desc, "native_eval": natives[seed], "rewired_eval": ev, "sanity_gate": gate,
            "bc_log": log, "elapsed_s": time.time() - t0,
        }
        print(f"  [{name}] {natives[seed]['cheese_reach']:.3f} -> {ev['cheese_reach']:.3f} "
              f"| retention={gate['retention']:.3f} collapsed={gate['collapsed']}", flush=True)
        del pol, feats

    summary = {
        "marker": ALIGNED_MAZE_MARKER, "tap": args.tap,
        "tap_alias": "h_pre(2048-d)" if args.tap == "block3" else "h_fc(256-d)",
        "label_source": args.label_source, "bank_meta": bank["meta"],
        "config": {k: str(v) for k, v in vars(args).items()},
        "natives": natives, "rows": results,
    }
    out = args.out / f"summary_{args.tap}_{args.label_source}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {out}", flush=True)


if __name__ == "__main__":
    main()
