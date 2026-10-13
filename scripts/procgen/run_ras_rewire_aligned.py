#!/usr/bin/env python3
"""Attribute Maze frozen-trunk rewiring on one shared, driver-balanced BC bank at h_pre or h_fc, scored by unconditional R_goal."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", ROOT / "data"))
for p in (ROOT / "src", ROOT / "scripts" / "procgen"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gm_geometry.rewire import bootstrap_diff_ci, motor_collapse_flag, prepare_tap  # noqa: E402
from gm_geometry.rewire.ras import (  # noqa: E402
    BC_NUM_LEVELS,
    BC_START_LEVEL,
    COHORTS,
    COLLECTOR_LABEL_SOURCES,
    EVAL_START_LEVEL,
    LABEL_SOURCES,
    OBS_SHAPE,
    collect_balanced_obs_bank,
    eval_closed_loop,
    random_frozen_fc,
    tap_features_from_obs,
    train_tap_bc,
    trainable_param_count,
)

# Oracle labels need the walkable grid and object cells, which a stored frame lacks, so they are collected online
# on their own states: GM vs Non-GM within one label source is matched, comparisons across sources are not.

# Written into every summary so results can be matched to the runner version that produced them.
RUNNER_MARKER = "ras-rewire-aligned-v1-Rgoal-unconditional"
ORACLE_MARKER = "ras-rewire-aligned-oracle-v1-softset-online"


_HB_PATH: Path | None = None


def _hb(msg: str) -> None:
    """Append-only progress log; one open/write/close per line so it survives a SIGKILL."""
    if _HB_PATH is None:
        return
    with open(_HB_PATH, "a") as f:
        f.write(f"{time.time():.1f} {msg}\n")


def _ckpt(ckpt_dir: Path, seed: int, kind: str) -> Path:
    return ckpt_dir / f"{seed}_{'gm' if kind == 'gm' else 'teach'}.pth"


def _load(ckpt: Path, device: str):
    import head_rewire_ras_maze as hrm

    return hrm.load_ras_policy(ckpt, device)


def _boot_mean_ci(vals, *, n_boot: int = 20_000, seed: int = 0) -> dict:
    """Percentile bootstrap on the across-seed mean.

    With 3 seeds there are only 10 distinct resampling multisets, so this is an interval across seeds, not a
    precise confidence statement.
    """
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(arr.mean()), "ci95": [float(lo), float(hi)], "n": int(arr.size), "values": arr.tolist()}


# ---------------------------------------------------------------- BC state bank


def _bank_paths(bank: Path, teacher_seeds) -> dict:
    """On-disk layout of one BC bank: obs_u8.npy, driver_id.npy, meta.json and DONE for both label sources.

    teacher_greedy adds one (N,) int64 label array per teacher; the oracle adds one (N, n_actions) float32
    soft-target array plus the level id of every frame (its train/val split is grouped by level).
    """
    return {
        "obs": bank / "obs_u8.npy",
        "driver": bank / "driver_id.npy",
        "labels": {s: bank / f"labels_teacher_{s}.npy" for s in teacher_seeds},
        "soft": bank / "labels_oracle_soft.npy",
        "level": bank / "level_id.npy",
        "meta": bank / "meta.json",
        "done": bank / "DONE",
    }


def _default_bank_dir(shared: Path, label_source: str) -> Path:
    """Oracle banks live beside, never on top of, teacher banks: they are different states,
    so sharing a directory name across label sources could only ever mean one silently
    reusing the other's frames."""
    return shared / ("bc_bank" if label_source == "teacher_greedy" else f"bc_bank_{label_source}")


def _read_labels(paths: dict, teacher_seeds, *, oracle: bool) -> dict:
    """Labels keyed by the teacher seed each (gm, teacher) pair is trained against.

    Under the policy-independent oracle every key maps to the same array.
    """
    if oracle:
        soft = np.load(paths["soft"])
        return {s: soft for s in teacher_seeds}
    return {s: np.load(paths["labels"][s]) for s in teacher_seeds}


def build_or_load_bank(shared: Path, cohort: dict, args) -> tuple[np.ndarray, dict, dict, np.ndarray | None]:
    """One balanced observation bank per cohort, reused by every arm and both taps.

    Observations rather than features are cached because features are trunk-specific:
    six trunks x two taps would be six banks, and no two arms would then be trained
    on provably the same states.
    """
    bank = Path(args.bank_dir) if args.bank_dir else _default_bank_dir(shared, args.label_source)
    teacher_seeds = list(cohort["teacher_seeds"])
    oracle = args.label_source in COLLECTOR_LABEL_SOURCES
    paths = _bank_paths(bank, teacher_seeds)
    if paths["done"].exists():
        print(f"=== BC bank: reusing {bank} ===", flush=True)
        meta = json.loads(paths["meta"].read_text())
        if meta.get("label_source") != args.label_source:
            raise SystemExit(
                f"bank at {bank} was labelled with {meta.get('label_source')!r} but this job asked "
                f"for {args.label_source!r}; point --bank-dir somewhere else rather than mixing them"
            )
        obs = np.load(paths["obs"], mmap_mode="r")
        labels = _read_labels(paths, teacher_seeds, oracle=oracle)
        return obs, labels, meta, (np.load(paths["level"]) if oracle else None)

    gm_ckpts = [_ckpt(args.ckpt_dir, s, "gm") for s in cohort["gm_seeds"]]
    teach_ckpts = [_ckpt(args.ckpt_dir, s, "teach") for s in teacher_seeds]
    drivers_ckpts = gm_ckpts + teach_ckpts
    frames_per_driver = args.collect_steps // len(drivers_ckpts)
    n_frames = frames_per_driver * len(drivers_ckpts)

    tmp = bank.parent / f"{bank.name}.tmp-{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    tpaths = _bank_paths(tmp, teacher_seeds)
    print(
        f"=== BC bank: collecting {n_frames} frames ({frames_per_driver}/driver x "
        f"{len(drivers_ckpts)} drivers) into {tmp} ===",
        flush=True,
    )
    _hb(f"bank: building {n_frames} frames into {tmp}")

    obs_mm = np.lib.format.open_memmap(
        tpaths["obs"], mode="w+", dtype=np.uint8, shape=(n_frames, *OBS_SHAPE)
    )
    driver_mm = np.lib.format.open_memmap(tpaths["driver"], mode="w+", dtype=np.int8, shape=(n_frames,))

    drivers = [_load(c, args.device) for c in drivers_ckpts]
    n_act = int(drivers[0].fc_policy.out_features)

    if oracle:
        from gm_geometry.rewire.ras_oracle import N_ACTIONS, collect_balanced_oracle_obs_bank

        if n_act != N_ACTIONS:
            raise SystemExit(
                f"oracle action table is {N_ACTIONS}-wide but this checkpoint's head emits "
                f"{n_act} logits; the transcribed action deltas would not line up"
            )
        soft_mm = np.lib.format.open_memmap(
            tpaths["soft"], mode="w+", dtype=np.float32, shape=(n_frames, n_act)
        )
        level_mm = np.lib.format.open_memmap(tpaths["level"], mode="w+", dtype=np.int64, shape=(n_frames,))
        stats = collect_balanced_oracle_obs_bank(
            drivers,
            use_backgrounds=cohort["use_backgrounds"],
            frames_per_driver=frames_per_driver,
            device=args.device,
            rand_seed=args.seed,
            out_obs=obs_mm,
            out_driver=driver_mm,
            out_soft=soft_mm,
            out_level=level_mm,
            bc_start_level=BC_START_LEVEL,
            bc_num_levels=BC_NUM_LEVELS,
            obs_shape=OBS_SHAPE,
            n_actions=n_act,
        )
        soft_mm.flush()
        level_mm.flush()
        # Mass per action rather than a count: a tied frame contributes 1/|A*| to each
        # member, so the column sums total n_frames exactly and stay comparable to the
        # teacher path's histograms.
        label_counts = {"oracle": soft_mm.sum(axis=0).tolist()}
        print(f"  bank: oracle soft-target mass per action {label_counts['oracle']}", flush=True)
        del soft_mm, level_mm
    else:
        stats = collect_balanced_obs_bank(
            drivers,
            use_backgrounds=cohort["use_backgrounds"],
            frames_per_driver=frames_per_driver,
            num_envs=args.num_envs,
            device=args.device,
            rand_seed=args.seed,
            out_obs=obs_mm,
            out_driver=driver_mm,
        )
    for d in drivers:
        del d
    obs_mm.flush()
    driver_mm.flush()

    if not oracle:
        labeler = LABEL_SOURCES[args.label_source]
        label_counts = {}
        for s in teacher_seeds:
            _hb(f"bank: labelling with teacher {s} ({args.label_source})")
            teacher = _load(_ckpt(args.ckpt_dir, s, "teach"), args.device)
            n_act = int(teacher.fc_policy.out_features)
            lab = labeler(teacher, obs_mm, device=args.device, batch_size=args.feature_batch)
            del teacher
            np.save(tpaths["labels"][s], lab)
            label_counts[str(s)] = np.bincount(lab, minlength=n_act).tolist()
            print(f"  bank: labels from teacher {s} -- action histogram {label_counts[str(s)]}", flush=True)

    meta = {
        "cohort_use_backgrounds": cohort["use_backgrounds"],
        "gm_seeds": list(cohort["gm_seeds"]),
        "teacher_seeds": teacher_seeds,
        "driver_order": [c.name for c in drivers_ckpts],
        "frames_per_driver": frames_per_driver,
        "n_frames": n_frames,
        "collect_steps_requested": args.collect_steps,
        "num_envs": 1 if oracle else args.num_envs,
        "rand_seed": args.seed,
        "bc_start_level": BC_START_LEVEL,
        "bc_num_levels": BC_NUM_LEVELS,
        "label_source": args.label_source,
        "label_kind": "soft_set" if oracle else "hard_argmax",
        "n_actions": n_act,
        "label_action_histograms": label_counts,
        **stats,
    }
    tpaths["meta"].write_text(json.dumps(meta, indent=2))
    tpaths["done"].write_text(RUNNER_MARKER)
    del obs_mm, driver_mm

    try:
        os.rename(tmp, bank)
    except OSError:
        # Another process published its own bank first; both
        # are valid, but every arm must read the same one.
        if not paths["done"].exists():
            raise
        print(f"=== BC bank: another job published {bank} first, using it ===", flush=True)

    obs = np.load(paths["obs"], mmap_mode="r")
    labels = _read_labels(paths, teacher_seeds, oracle=oracle)
    return obs, labels, json.loads(paths["meta"].read_text()), (np.load(paths["level"]) if oracle else None)


# ---------------------------------------------------------------------- arms


def _preflight(label: str, policy, *, tap: str, arm: str, feats_shape, n_frames: int, label_kind: str = "hard_argmax") -> dict:
    if tap == "block3":
        frozen = ["embedder.block1", "embedder.block2", "embedder.block3"]
        if arm == "random_proj":
            fixed_random = ["embedder.fc (xavier, seeded, never updated)"]
            fresh = ["fc_policy", "fc_value"]
        else:
            fixed_random = []
            fresh = ["embedder.fc", "fc_policy", "fc_value"]
        count_tap = "block3"
    else:
        frozen = ["embedder.block1", "embedder.block2", "embedder.block3", "embedder.fc"]
        fixed_random = []
        fresh = ["fc_policy", "fc_value"]
        count_tap = "fc"
    n_params = trainable_param_count(policy, count_tap)
    info = {
        "tap": tap,
        "arm": arm,
        "frozen_modules": frozen,
        "fixed_random_modules": fixed_random,
        "fresh_trainable_modules": fresh,
        "trainable_params": n_params,
        "feature_shape": list(feats_shape),
        "n_bank_frames": n_frames,
        "label_kind": label_kind,
        "bc_loss": "soft_target_cross_entropy" if label_kind == "soft_set" else "cross_entropy",
        "val_accuracy": "set_accuracy_1[argmax_pi_in_Astar]" if label_kind == "soft_set" else "argmax_equality",
    }
    print(
        f"--- PREFLIGHT {label}\n"
        f"    tap={tap} ({'h_pre 2048-d' if tap == 'block3' else 'h_fc 256-d'})  arm={arm}\n"
        f"    frozen:          {', '.join(frozen)}\n"
        f"    fixed random:    {', '.join(fixed_random) or '(none)'}\n"
        f"    fresh trainable: {', '.join(fresh)}\n"
        f"    trainable params: {n_params}\n"
        f"    labels:          {label_kind}  (loss={info['bc_loss']}, val_acc={info['val_accuracy']})\n"
        f"    cached features: {tuple(feats_shape)} float32  ({feats_shape[0] * feats_shape[1] * 4 / 1e9:.2f} GB)",
        flush=True,
    )
    return info


def _write_episodes_csv(path: Path, res: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "level", "reward", "steps", "timed_out", "action_entropy", "distinct_actions"])
        for i, (r, s, e, d) in enumerate(
            zip(res["episode_rewards"], res["episode_steps"], res["episode_action_entropy"], res["episode_distinct_actions"])
        ):
            w.writerow([i, res["start_level"] + i, r, s, int(s >= res["max_steps"]), f"{e:.6f}", d])


def _write_curve_csv(path: Path, bc_log: list[dict]) -> None:
    # val_acc_kind is not decoration: "set" is 1[argmax pi in A*(s)] over a tied optimal set,
    # "argmax" is equality with a single teacher action, and the two are not comparable.
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "train_ce", "val_ce", "val_acc", "val_acc_kind"])
        for row in bc_log:
            kind = "set" if "val_set_acc" in row else "argmax"
            w.writerow([row["epoch"], row["step"], row["train_ce"], row["val_ce"], row["val_acc"], kind])


def _strip_episodes(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("episode_")}


def _finish(sub: Path, summary: dict, res: dict) -> dict:
    _write_episodes_csv(sub / "episodes.csv", res)
    (sub / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  {summary['label']}: R_goal={summary['R_goal']:.4f}  D={summary['decisive_frac']:.4f}  "
        f"Pref={summary['pref_goal']:.4f}  timeout={summary['frac_timeout']:.3f}  "
        f"mean_steps={summary['mean_steps']:.1f}",
        flush=True,
    )
    return summary


def run_native(label: str, ckpt: Path, *, cohort: dict, args, out: Path) -> dict:
    sub = out / label
    sub.mkdir(parents=True, exist_ok=True)
    if (sub / "summary.json").exists():
        print(f"SKIP {label} (already done)", flush=True)
        return json.loads((sub / "summary.json").read_text())
    _hb(f"{label}: native eval")
    t0 = time.time()
    policy = _load(ckpt, args.device)
    res = eval_closed_loop(
        policy, args.device,
        use_backgrounds=cohort["use_backgrounds"],
        n_episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        start_level=EVAL_START_LEVEL,
    )
    summary = {"label": label, "mode": "native", "tap": None, "ckpt": str(ckpt), "elapsed_s": time.time() - t0, **res}
    return _finish(sub, summary, res)


def run_rewire_arm(
    label: str,
    *,
    ckpt: Path,
    arm: str,
    tap: str,
    feats: np.ndarray,
    labels: np.ndarray,
    cohort: dict,
    args,
    out: Path,
    proj_seed: int,
    native_entropy: float | None,
    groups=None,
) -> dict:
    sub = out / label
    sub.mkdir(parents=True, exist_ok=True)
    if (sub / "summary.json").exists():
        print(f"SKIP {label} (already done)", flush=True)
        return json.loads((sub / "summary.json").read_text())

    t0 = time.time()
    policy = _load(ckpt, args.device)
    if arm == "random_proj":
        random_frozen_fc(policy, seed=proj_seed)
    else:
        prepare_tap(policy, tap)
    pre = _preflight(
        label, policy, tap=tap, arm=arm, feats_shape=feats.shape, n_frames=len(feats),
        label_kind="soft_set" if np.asarray(labels).ndim == 2 else "hard_argmax",
    )

    def _epoch_cb(epoch, _row):
        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0:
            torch.save({"model_state_dict": policy.state_dict(), "label": label, "epoch": epoch}, sub / f"epoch{epoch:03d}.pth")

    _hb(f"{label}: training ({pre['trainable_params']} params, {len(feats)} frames)")
    bc_log = train_tap_bc(
        policy, feats, labels, "block3" if arm == "random_proj" else tap,
        device=args.device, epochs=args.bc_epochs, batch_size=args.batch_size, lr=args.lr,
        prepare=False, epoch_cb=_epoch_cb, groups=groups,
    )
    _write_curve_csv(sub / "bc_curve.csv", bc_log)

    _hb(f"{label}: closed-loop eval n={args.eval_episodes}")
    res = eval_closed_loop(
        policy, args.device,
        use_backgrounds=cohort["use_backgrounds"],
        n_episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        start_level=EVAL_START_LEVEL,
    )
    torch.save({"model_state_dict": policy.state_dict(), "label": label, "arm": arm, "tap": tap}, sub / "final.pth")

    summary = {
        "label": label, "mode": arm, "tap": tap, "ckpt": str(ckpt),
        "preflight": pre, "bc_log": bc_log,
        "proj_seed": proj_seed if arm == "random_proj" else None,
        "elapsed_s": time.time() - t0, **res,
    }
    if native_entropy is not None:
        summary["motor_collapse"] = motor_collapse_flag(res["mean_action_entropy"], native_entropy)
    return _finish(sub, summary, res)


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=tuple(COHORTS), required=True)
    ap.add_argument("--tap", choices=("block3", "fc"), required=True,
                    help="block3 = h_pre (2048-d, fresh fc+heads); fc = h_fc (256-d, fresh heads)")
    ap.add_argument("--out-dir", type=Path, required=True, help="per-tap results directory")
    ap.add_argument("--shared-dir", type=Path, default=None,
                    help="cohort-level directory holding the BC bank and the tap-independent native evals "
                         "(default: parent of --out-dir)")
    ap.add_argument("--bank-dir", type=Path, default=None, help="default: <shared-dir>/bc_bank")
    ap.add_argument("--ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/ras")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gm-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--teacher-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--collect-steps", type=int, default=500_000)
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--feature-batch", type=int, default=512)
    ap.add_argument("--eval-episodes", type=int, default=1000)
    ap.add_argument("--eval-max-steps", type=int, default=256)
    ap.add_argument("--ckpt-every", type=int, default=5, help="epochs between intermediate checkpoints (0 = off)")
    ap.add_argument("--label-source", choices=tuple(LABEL_SOURCES), default="teacher_greedy")
    ap.add_argument("--random-proj-arm", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--no-nongm-random-proj", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cohort = dict(COHORTS[args.cohort])
    if args.gm_seeds:
        cohort["gm_seeds"] = tuple(args.gm_seeds)
    if args.teacher_seeds:
        cohort["teacher_seeds"] = tuple(args.teacher_seeds)
    if len(cohort["gm_seeds"]) != len(cohort["teacher_seeds"]):
        raise SystemExit("gm_seeds and teacher_seeds must pair 1:1")
    oracle = args.label_source in COLLECTOR_LABEL_SOURCES
    if oracle and args.num_envs != 1:
        # Hard error rather than a silent override: the oracle's agent-cell reckoning and its
        # get_state/set_object probe are single-env and single-level.
        raise SystemExit(
            f"--label-source {args.label_source} labels online and requires --num-envs 1 "
            f"(got {args.num_envs}); the oracle collector builds one env per level"
        )

    out = args.out_dir
    shared = args.shared_dir or out.parent
    out.mkdir(parents=True, exist_ok=True)
    shared.mkdir(parents=True, exist_ok=True)
    native_dir = shared / "native"
    native_dir.mkdir(parents=True, exist_ok=True)

    global _HB_PATH
    _HB_PATH = shared / f"heartbeat_{args.tap}.log"
    _hb(f"main() started tap={args.tap} cohort={args.cohort} cuda={torch.cuda.is_available()}")
    print(
        f"=== Ras aligned rewire [{RUNNER_MARKER}"
        f"{'; ' + ORACLE_MARKER if oracle else ''}]: cohort={args.cohort} tap={args.tap} "
        f"label_source={args.label_source} "
        f"use_backgrounds={cohort['use_backgrounds']} gm={cohort['gm_seeds']} teacher={cohort['teacher_seeds']} ===",
        flush=True,
    )
    (out / "config.json").write_text(json.dumps({"cohort": args.cohort, **cohort, **vars(args)}, indent=2, default=str))

    for s in cohort["gm_seeds"]:
        assert _ckpt(args.ckpt_dir, s, "gm").exists(), f"missing {_ckpt(args.ckpt_dir, s, 'gm')}"
    for s in cohort["teacher_seeds"]:
        assert _ckpt(args.ckpt_dir, s, "teach").exists(), f"missing {_ckpt(args.ckpt_dir, s, 'teach')}"

    obs_bank, labels_by_teacher, bank_meta, bank_levels = build_or_load_bank(shared, cohort, args)
    print(
        f"=== BC bank ready: {obs_bank.shape} uint8, label source={bank_meta['label_source']} "
        f"({bank_meta.get('label_kind', 'hard_argmax')}) ===",
        flush=True,
    )

    # At h_fc a random 256->256 projection is invertible a.s., so a linear head behind it is as expressive as
    # reading h_fc directly and would not be a control. The capacity-matched control is the block3-tap
    # random_frozen_fc arm (heads only on a fixed random projection of h_pre), so it is only run at tap=block3.
    run_random_proj = args.random_proj_arm == "on" or (args.random_proj_arm == "auto" and args.tap == "block3")
    if run_random_proj and args.tap != "block3":
        raise SystemExit("--random-proj-arm on needs h_pre features, i.e. --tap block3")
    if not run_random_proj:
        (out / "random_proj_SKIPPED.json").write_text(json.dumps({
            "skipped": True,
            "reason": "at tap=fc the capacity-matched random-projection control is identical by "
                      "construction to the tap=block3 random_proj arm (heads-only training on a fixed "
                      "random projection of h_pre); a fresh 256->256 map at h_fc is a.s. invertible and "
                      "would leave a linear head exactly as expressive, i.e. not a control",
            "see": "the sibling tap=block3 run's gm_random_proj_* / nongm_random_proj_* arms",
        }, indent=2))
        print("=== random-projection arm SKIPPED at tap=fc (see random_proj_SKIPPED.json) ===", flush=True)

    rows: list[dict] = []
    natives: dict[int, dict] = {}
    for gm_seed, teach_seed in zip(cohort["gm_seeds"], cohort["teacher_seeds"]):
        gm_ckpt = _ckpt(args.ckpt_dir, gm_seed, "gm")
        teach_ckpt = _ckpt(args.ckpt_dir, teach_seed, "teach")
        # Native evals are tap-independent; they live in the shared dir so the
        # second tap's run reuses the first's instead of re-rolling 1000 episodes.
        natives[gm_seed] = run_native(f"native_gm_{gm_seed}", gm_ckpt, cohort=cohort, args=args, out=native_dir)
        natives[teach_seed] = run_native(f"native_teacher_{teach_seed}", teach_ckpt, cohort=cohort, args=args, out=native_dir)

        labels = labels_by_teacher[teach_seed]
        for role, ckpt, roster_seed in (("gm", gm_ckpt, gm_seed), ("nongm", teach_ckpt, teach_seed)):
            wanted = [f"{role}_rewire_{roster_seed}"]
            if run_random_proj and not (role == "nongm" and args.no_nongm_random_proj):
                wanted.append(f"{role}_random_proj_{roster_seed}")
            if all((out / w / "summary.json").exists() for w in wanted):
                # Feature extraction is ~500k forwards; do not pay for it on a
                # resume where both of this trunk's arms already finished.
                feats = np.zeros((0, 0), dtype=np.float32)
            else:
                _hb(f"{role}_{roster_seed}: extracting {args.tap} features over {len(obs_bank)} frames")
                feat_policy = _load(ckpt, args.device)
                feats = tap_features_from_obs(feat_policy, obs_bank, args.tap, device=args.device, batch_size=args.feature_batch)
                del feat_policy
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            rows.append(run_rewire_arm(
                f"{role}_rewire_{roster_seed}", ckpt=ckpt, arm="rewire", tap=args.tap,
                feats=feats, labels=labels, cohort=cohort, args=args, out=out,
                proj_seed=args.seed, native_entropy=natives[roster_seed]["mean_action_entropy"],
                groups=bank_levels,
            ))
            if run_random_proj and not (role == "nongm" and args.no_nongm_random_proj):
                rows.append(run_rewire_arm(
                    f"{role}_random_proj_{roster_seed}", ckpt=ckpt, arm="random_proj", tap=args.tap,
                    feats=feats, labels=labels, cohort=cohort, args=args, out=out,
                    proj_seed=args.seed + roster_seed, native_entropy=natives[roster_seed]["mean_action_entropy"],
                    groups=bank_levels,
                ))
            del feats

    def _pick(prefix: str, key: str = "R_goal") -> list[float]:
        return [r[key] for r in rows if r["label"].startswith(prefix)]

    groups = {
        "gm_rewire": _pick("gm_rewire_"),
        "nongm_rewire": _pick("nongm_rewire_"),
        "gm_random_proj": _pick("gm_random_proj_"),
        "nongm_random_proj": _pick("nongm_random_proj_"),
        "native_gm": [natives[s]["R_goal"] for s in cohort["gm_seeds"]],
        "native_teacher": [natives[s]["R_goal"] for s in cohort["teacher_seeds"]],
    }
    contrasts = {}
    for a, b in (("gm_rewire", "nongm_rewire"), ("gm_rewire", "gm_random_proj"), ("gm_rewire", "native_gm")):
        if groups[a] and groups[b]:
            contrasts[f"{a}_minus_{b}"] = bootstrap_diff_ci(np.array(groups[a]), np.array(groups[b]))

    caveats = [
        "3 seeds per group: the bootstrap CI over the across-seed mean has only 10 distinct "
        "resampling multisets and cannot be narrower than the seed spread.",
        "R_goal, D and Pref_goal come from the same 1000 held-out episodes per arm, so the "
        "three are not independent measurements.",
    ]
    if oracle:
        caveats.append(
            "ORACLE vs TEACHER LABELS ARE NOT STATE-MATCHED. The graph oracle needs the walkable "
            "grid, the agent cell and both object cells, none of which survive into a stored "
            "64x64 frame, so it must label online, during the rollout. The oracle bank is "
            "therefore a DIFFERENT set of states from the teacher bank: one env per level with "
            "num_envs=1 and a 64-step cap, versus 128 parallel envs rolling continuously -- and the "
            "two banks need not even hold the same NUMBER of frames (see bank_meta.n_frames). "
            "GM vs Non-GM WITHIN this run is clean -- every arm reads this one driver-balanced "
            "bank. Any oracle-vs-teacher_greedy comparison is confounded by state distribution "
            "and must not be read as a label effect."
        )
        caveats.append(
            "Validation numbers are not comparable across label sources either: the oracle path "
            "reports set accuracy 1[argmax pi in A*(s)] on a split held out at the LEVEL "
            "boundary (consecutive oracle frames are near-duplicates), while the teacher path "
            "reports argmax equality on an i.i.d. frame split."
        )
        caveats.append(
            "Levels whose geometry or agent-cell reckoning could not be verified contribute no "
            "frames at all; see bank_meta['oracle_stats'] for the drop counts. The retained "
            "frames are therefore conditioned on resolvable geometry, a mild selection on "
            "level topology."
        )
    summary = {
        "runner_marker": RUNNER_MARKER,
        "oracle_marker": ORACLE_MARKER if oracle else None,
        "cohort": args.cohort, **cohort, "tap": args.tap,
        "label_source": args.label_source,
        "primary_metric": "R_goal = n_goal_hit / n_episodes (unconditional)",
        "caveats": caveats,
        "bank_meta": bank_meta,
        # Per-episode arrays stay in each arm's own summary.json / episodes.csv to keep the aggregate small.
        "rows": [_strip_episodes(r) for r in rows + [natives[s] for s in list(cohort["gm_seeds"]) + list(cohort["teacher_seeds"])]],
        "R_goal_by_group": groups,
        "R_goal_group_ci95": {k: _boot_mean_ci(v) for k, v in groups.items()},
        "decisive_frac_by_group": {
            "gm_rewire": _pick("gm_rewire_", "decisive_frac"),
            "nongm_rewire": _pick("nongm_rewire_", "decisive_frac"),
            "gm_random_proj": _pick("gm_random_proj_", "decisive_frac"),
            "nongm_random_proj": _pick("nongm_random_proj_", "decisive_frac"),
        },
        "pref_goal_by_group": {
            "gm_rewire": _pick("gm_rewire_", "pref_goal"),
            "nongm_rewire": _pick("nongm_rewire_", "pref_goal"),
            "gm_random_proj": _pick("gm_random_proj_", "pref_goal"),
            "nongm_random_proj": _pick("nongm_random_proj_", "pref_goal"),
        },
        "bootstrap_contrasts_R_goal": contrasts,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "bank_meta")}, indent=2), flush=True)
    for r in summary["rows"]:
        print(
            f"  {r['label']}: R_goal={r['R_goal']:.4f} D={r['decisive_frac']:.4f} "
            f"Pref={r['pref_goal']:.4f} timeout={r['frac_timeout']:.3f} H={r['mean_action_entropy']:.3f}",
            flush=True,
        )


def _out_dir_from_argv() -> Path | None:
    if "--out-dir" in sys.argv:
        i = sys.argv.index("--out-dir")
        if i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    return None


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        out_dir = _out_dir_from_argv()
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "CRASH.log").write_text(traceback.format_exc())
        raise
