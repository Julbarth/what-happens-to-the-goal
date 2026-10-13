#!/usr/bin/env python3
"""Coin frozen-trunk rewiring (h_pre or h_fc tap) on one shared, behaviour-balanced BC bank labelled by a fixed Non-GM teacher."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALIGNED_REWIRE_MARKER = "ALIGNED_REWIRE_V1_shared_balanced_bank"

ROOT = Path(__file__).resolve().parents[2]
for _p in (
    ROOT / "src",
    ROOT / "scripts" / "procgen",
    ROOT / "external" / "train-procgen-pytorch",
    ROOT / "external" / "procgenAISC",
):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)

from gm_geometry.rewire.eval import bootstrap_diff_ci, motor_collapse_flag  # noqa: E402
from gm_geometry.rewire.taps import prepare_tap  # noqa: E402
from train_frozen_trunk_head_rewire import eval_coin_dependence, sample_actions  # noqa: E402
from run_coin_behaviour import (  # noqa: E402
    VAL_START,
    bootstrap_mean,
    eval_policy_val,
    load_policy,
)
from common.env.procgen_wrappers import (  # noqa: E402
    ScaledFloatFrame,
    TransposeFrame,
    VecExtractDictObs,
    VecNormalize,
)
from procgen import ProcgenEnv  # noqa: E402

# The Non-GM arm whose trunk is the label teacher is self-labelled; every arm records
# `label_source_is_own_trunk` and the summary reports the Non-GM contrast with and without it.

def ckpt_label(p: Path) -> str:
    """Readable arm suffix; takes the seed from the directory name when the file stem (model_<steps>) lacks it."""
    m = re.search(r"seed[_-](\d+)", str(p))
    if m and not p.stem.startswith(m.group(1)):
        return f"{m.group(1)}_{p.stem}"
    return p.stem


BANK_FILES = ("obs_u8.npy", "actions.npy", "source.npy", "meta.json")
TAPS = ("block3", "fc")
TAP_ALIASES = {"block3": "h_pre(2048-d)", "fc": "h_fc(256-d)"}


# ── shared BC state bank ─────────────────────────────────────────────────────


def make_bank_env(env_name: str, *, num_envs: int, start_level: int, num_levels: int, rand_seed: int):
    """Same wrapper stack as train_frozen_trunk_head_rewire.make_coin_env, plus an explicit procgen rand_seed
    so an independent recollection of the bank reproduces the same level sequence."""
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="hard",
        num_threads=max(2, num_envs // 2),
        rand_seed=int(rand_seed),
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False, ret=False)
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def bank_meta_path(bank_dir: Path) -> Path:
    return bank_dir / "meta.json"


def bank_is_complete(bank_dir: Path) -> bool:
    return all((bank_dir / f).exists() for f in BANK_FILES)


def collect_bank(bank_dir: Path, *, sources: list[tuple[str, Path]], label_ckpt: Path, device: str, args) -> dict:
    """Balanced mixture bank: every behaviour policy contributes the same number of
    frames, each labelled by the label teacher's greedy argmax at the visited state.

    The env is stepped with the BEHAVIOUR policy's action and the label is recorded
    for the pre-step observation, so GM-rolled frames stay on the GM state
    distribution while still carrying a competent label.
    """
    n_src = len(sources)
    per_env_steps = int(np.ceil(args.collect_steps / n_src / args.num_envs))
    n_per = per_env_steps * args.num_envs
    total = n_per * n_src
    print(f"=== collecting shared BC bank: {n_src} sources x {n_per} frames = {total} ===", flush=True)

    label_policy = load_policy(label_ckpt, device)
    obs_buf: np.ndarray | None = None
    acts = np.empty(total, dtype=np.int64)
    srcs = np.empty(total, dtype=np.uint8)
    w = 0
    t0 = time.time()
    for si, (sname, spath) in enumerate(sources):
        same_as_label = Path(spath).resolve() == Path(label_ckpt).resolve()
        behav = label_policy if same_as_label else load_policy(spath, device)
        env = make_bank_env(
            args.train_env,
            num_envs=args.num_envs,
            start_level=0,
            num_levels=args.num_levels,
            # Distinct stream per source so the six segments are not the same levels
            # in the same order; fixed offset keeps the whole bank reproducible.
            rand_seed=args.bank_seed + 1009 * si,
        )
        obs = env.reset()
        got = 0
        while got < n_per:
            arr = np.asarray(obs, dtype=np.float64)
            u8 = np.rint(arr * 255.0).astype(np.uint8)
            if obs_buf is None:
                # ScaledFloatFrame returns raw_uint8/255, so uint8 storage is exact;
                # verify once rather than assume (6 GB of obs vs 25 GB of float32).
                err = float(np.max(np.abs(arr * 255.0 - u8)))
                assert err < 1e-3, f"uint8 obs round-trip is lossy (max err {err})"
                obs_buf = np.empty((total,) + arr.shape[1:], dtype=np.uint8)
            a_lab = sample_actions(label_policy, obs, device)
            a_beh = a_lab if same_as_label else sample_actions(behav, obs, device)
            k = arr.shape[0]
            obs_buf[w : w + k] = u8
            acts[w : w + k] = a_lab
            srcs[w : w + k] = si
            w += k
            got += k
            obs, _rew, _done, _infos = env.step(a_beh)
        if hasattr(env, "close"):
            env.close()
        if not same_as_label:
            del behav
        print(f"  source {si} {sname}: {got} frames ({time.time() - t0:.0f}s)", flush=True)

    assert obs_buf is not None and w == total, f"bank fill mismatch {w} != {total}"
    meta = {
        "marker": ALIGNED_REWIRE_MARKER,
        "n_frames": int(total),
        "n_per_source": int(n_per),
        "sources": [{"id": i, "name": n, "ckpt": str(p)} for i, (n, p) in enumerate(sources)],
        "label_ckpt": str(label_ckpt),
        "collect_steps": int(args.collect_steps),
        "num_envs": int(args.num_envs),
        "num_levels": int(args.num_levels),
        "bank_seed": int(args.bank_seed),
        "train_env": args.train_env,
        "obs_shape": list(obs_buf.shape[1:]),
        "label_mode": "teacher_greedy_argmax",
        "behaviour_mode": "greedy_argmax_per_source",
    }

    tmp = bank_dir / f".tmp-{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "obs_u8.npy", obs_buf)
    np.save(tmp / "actions.npy", acts)
    np.save(tmp / "source.npy", srcs)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    bank_dir.mkdir(parents=True, exist_ok=True)
    # meta.json last: it is the completion marker a concurrent or restarted run checks.
    for fn in ("obs_u8.npy", "actions.npy", "source.npy", "meta.json"):
        os.replace(tmp / fn, bank_dir / fn)
    tmp.rmdir()
    print(f"=== bank written to {bank_dir} ({total} frames, {time.time() - t0:.0f}s) ===", flush=True)
    return meta


def load_or_build_bank(bank_dir: Path, *, sources, label_ckpt: Path, device: str, args):
    if bank_is_complete(bank_dir):
        meta = json.loads(bank_meta_path(bank_dir).read_text())
        want = {
            "collect_steps": int(args.collect_steps),
            "num_envs": int(args.num_envs),
            "num_levels": int(args.num_levels),
            "bank_seed": int(args.bank_seed),
            "train_env": args.train_env,
            "label_ckpt": str(label_ckpt),
        }
        bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        src_now = [str(p) for _n, p in sources]
        src_old = [s["ckpt"] for s in meta.get("sources", [])]
        if src_now != src_old:
            bad["sources"] = (src_old, src_now)
        if bad:
            raise SystemExit(
                f"STOP: cached bank at {bank_dir} was built with different settings {bad}. "
                "Point --bank-dir somewhere new or delete the cache; silently reusing it "
                "would break the 'identical bank across taps' guarantee."
            )
        print(f"=== reusing cached BC bank {bank_dir} ({meta['n_frames']} frames) ===", flush=True)
    else:
        meta = collect_bank(bank_dir, sources=sources, label_ckpt=label_ckpt, device=device, args=args)
    obs = np.load(bank_dir / "obs_u8.npy", mmap_mode="r")
    acts = np.load(bank_dir / "actions.npy")
    srcs = np.load(bank_dir / "source.npy")
    return obs, acts, srcs, meta


# ── frozen-trunk feature caching ─────────────────────────────────────────────


@torch.no_grad()
def block3_flat_batch(policy, obs_f32: np.ndarray, device: str) -> torch.Tensor:
    """h_pre = vec(ReLU(H^3)) for one obs batch, exactly as ImpalaModel.forward does
    it before embedder.fc."""
    emb = policy.embedder
    x = torch.as_tensor(obs_f32, dtype=torch.float32, device=device)
    h = emb.block3(emb.block2(emb.block1(x)))
    return F.relu(h).flatten(1)


@torch.no_grad()
def cache_features(policy, obs_u8, device: str, *, batch: int, apply_fc: bool, check_ref: bool = False) -> np.ndarray:
    n = int(obs_u8.shape[0])
    dim = 256 if apply_fc else int(policy.embedder.fc.in_features)
    out = np.empty((n, dim), dtype=np.float32)
    for s in range(0, n, batch):
        o = np.asarray(obs_u8[s : s + batch], dtype=np.float32) / 255.0
        h = block3_flat_batch(policy, o, device)
        if s == 0 and check_ref:
            _cross_check_block3(policy, o, h, device)
        if apply_fc:
            h = F.relu(policy.embedder.fc(h))
        out[s : s + h.shape[0]] = h.detach().cpu().numpy()
    return out


def _cross_check_block3(policy, obs_f32, h, device: str) -> None:
    """One-batch agreement check against head_rewire_layer_features.block3_flat_from_policy.

    A broken diagnostic import must not abort the run, but a genuine numeric
    mismatch must: the whole experiment rests on this tap being h_pre.
    """
    try:
        from head_rewire_layer_features import block3_flat_from_policy

        ref = block3_flat_from_policy(policy, obs_f32, device)
    except Exception as e:  # noqa: BLE001 - diagnostic import only
        print(f"  [block3 tap cross-check skipped: {type(e).__name__}: {e}]", flush=True)
        return
    delta = float(np.max(np.abs(ref - h.detach().cpu().numpy())))
    status = "OK" if delta < 1e-4 else "MISMATCH"
    print(f"  [block3 tap cross-check vs block3_flat_from_policy: {status} max|delta|={delta:.2e}]", flush=True)
    assert delta < 1e-4, f"block3 tap disagrees with block3_flat_from_policy (max delta {delta})"


# ── arms ─────────────────────────────────────────────────────────────────────


class Arm:
    def __init__(self, *, name: str, cohort: str, mode: str, tap: str, ckpt: Path, policy, forward,
                 train_params, state_modules: dict, frozen: list[str], fixed_random: list[str],
                 fresh_trainable: list[str], feature_kind: str, feature_dim: int):
        self.name = name
        self.cohort = cohort
        self.mode = mode
        self.tap = tap
        self.ckpt = ckpt
        self.policy = policy
        self.forward = forward
        self.train_params = train_params
        self.state_modules = state_modules
        self.frozen = frozen
        self.fixed_random = fixed_random
        self.fresh_trainable = fresh_trainable
        self.feature_kind = feature_kind
        self.feature_dim = feature_dim

    @property
    def n_trainable(self) -> int:
        return int(sum(p.numel() for p in self.train_params))


def build_arm(*, name: str, cohort: str, mode: str, tap: str, ckpt: Path, device: str, init_seed: int) -> Arm:
    """Compose one rewired policy.

    `init_seed` is the SAME for every arm on purpose: the fresh suffix (embedder.fc
    draw, head draws, random projection) is then bit-identical across cohorts and
    taps, so the only thing that differs between arms is the frozen trunk and which
    downstream modules are allowed to learn. Seed-to-seed variation in the reported
    CIs is checkpoint-seed variation, which is the replication unit.
    """
    torch.manual_seed(init_seed)
    policy = load_policy(ckpt, device)
    prepare_tap(policy, tap)
    policy.to(device)
    emb = policy.embedder
    n_act = int(policy.fc_policy.out_features)

    if tap == "block3":
        if mode == "rewire":
            fc, head = emb.fc, policy.fc_policy
            forward = lambda xb: head(F.relu(fc(xb.clamp_min(0.0))))  # noqa: E731
            params = list(fc.parameters()) + list(head.parameters())
            return Arm(
                name=name, cohort=cohort, mode=mode, tap=tap, ckpt=ckpt, policy=policy, forward=forward,
                train_params=params, state_modules={"embedder.fc": fc, "fc_policy": head},
                frozen=["embedder.block1", "embedder.block2", "embedder.block3"],
                fixed_random=[],
                fresh_trainable=["embedder.fc (2048->256, xavier)", f"fc_policy (256->{n_act}, orth 0.01)"],
                feature_kind="h_pre", feature_dim=int(emb.fc.in_features),
            )
        # Tap-matched random-projection control at h_pre: prepare_tap has already
        # redrawn embedder.fc from the same Xavier init a from-scratch model uses;
        # freezing it leaves a FIXED random 2048->256 (+ReLU) compression, and only
        # the linear head learns.
        for p in emb.fc.parameters():
            p.requires_grad = False
        emb.fc.eval()
        head = policy.fc_policy
        forward = lambda xb: head(xb)  # noqa: E731
        return Arm(
            name=name, cohort=cohort, mode=mode, tap=tap, ckpt=ckpt, policy=policy, forward=forward,
            train_params=list(head.parameters()), state_modules={"fc_policy": head},
            frozen=["embedder.block1", "embedder.block2", "embedder.block3"],
            fixed_random=["embedder.fc (2048->256, xavier, FROZEN random)"],
            fresh_trainable=[f"fc_policy (256->{n_act}, orth 0.01)"],
            feature_kind="relu_randfc_h_pre", feature_dim=256,
        )

    # tap == "fc": embedder (blocks + learned fc) is frozen; heads are fresh.
    if mode == "rewire":
        head = policy.fc_policy
        forward = lambda xb: head(xb)  # noqa: E731
        return Arm(
            name=name, cohort=cohort, mode=mode, tap=tap, ckpt=ckpt, policy=policy, forward=forward,
            train_params=list(head.parameters()), state_modules={"fc_policy": head},
            frozen=["embedder.block1-3", "embedder.fc (learned)"],
            fixed_random=[],
            fresh_trainable=[f"fc_policy (256->{n_act}, orth 0.01)"],
            feature_kind="h_fc", feature_dim=256,
        )

    # Tap-matched random-projection control at h_fc: a FIXED random 256->256 Xavier layer + ReLU
    # between the frozen h_fc and a fresh head, training the head only (same trainable capacity as
    # the h_fc rewire arm). Rolling the trunk's fc back to random would just duplicate the h_pre
    # random_proj arm. The square map is almost surely invertible, so this tests whether the head
    # needs h_fc in its learned basis; it scrambles the basis rather than bottlenecking information
    # and is a weaker control than its h_pre counterpart.
    torch.manual_seed(init_seed + 1)
    proj = nn.Linear(256, 256)
    nn.init.xavier_uniform_(proj.weight)
    nn.init.zeros_(proj.bias)
    for p in proj.parameters():
        p.requires_grad = False
    head = nn.Linear(256, n_act)
    nn.init.orthogonal_(head.weight, gain=0.01)
    nn.init.zeros_(head.bias)
    policy.fc_policy = nn.Sequential(proj, nn.ReLU(), head).to(device)
    proj.eval()
    forward = lambda xb: head(xb)  # noqa: E731
    return Arm(
        name=name, cohort=cohort, mode=mode, tap=tap, ckpt=ckpt, policy=policy, forward=forward,
        train_params=list(head.parameters()), state_modules={"fc_policy.head": head},
        frozen=["embedder.block1-3", "embedder.fc (learned)"],
        fixed_random=["random_proj (256->256, xavier, FROZEN) + ReLU"],
        fresh_trainable=[f"fc_policy head (256->{n_act}, orth 0.01)"],
        feature_kind="relu_randproj_h_fc", feature_dim=256,
    )


def arm_features(arm: Arm, h_tap: np.ndarray, device: str, *, batch: int) -> np.ndarray:
    """Map the trunk's cached tap features to what this arm's trainable head sees.
    Must stay exactly in step with `arm.forward` and with the rollout graph."""
    if arm.feature_kind in ("h_pre", "h_fc"):
        return h_tap
    if arm.feature_kind == "relu_randfc_h_pre":
        fc = arm.policy.embedder.fc
        out = np.empty((h_tap.shape[0], 256), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, h_tap.shape[0], batch):
                x = torch.as_tensor(h_tap[s : s + batch], device=device).clamp_min(0.0)
                out[s : s + x.shape[0]] = F.relu(fc(x)).cpu().numpy()
        return out
    if arm.feature_kind == "relu_randproj_h_fc":
        proj = arm.policy.fc_policy[0]
        out = np.empty((h_tap.shape[0], 256), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, h_tap.shape[0], batch):
                x = torch.as_tensor(h_tap[s : s + batch], device=device)
                out[s : s + x.shape[0]] = F.relu(proj(x)).cpu().numpy()
        return out
    raise ValueError(f"unknown feature kind {arm.feature_kind}")


# ── BC training ──────────────────────────────────────────────────────────────


def train_bc(arm: Arm, features: np.ndarray, actions: np.ndarray, *, device: str, epochs: int,
             batch_size: int, lr: float, ckpt_dir: Path, ckpt_every: int, val_chunk: int = 8192) -> list[dict]:
    opt = torch.optim.Adam(arm.train_params, lr=lr)
    rng = np.random.RandomState(0)  # fixed 80/20 split, identical for every arm
    n = int(features.shape[0])
    n_val = max(1, n // 5)
    idx = rng.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    curve: list[dict] = []
    grad_frames = 0
    for epoch in range(epochs):
        perm = rng.permutation(tr_idx)
        total = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            xb = torch.as_tensor(features[b], device=device)
            yb = torch.as_tensor(actions[b], device=device, dtype=torch.long)
            opt.zero_grad()
            loss = F.cross_entropy(arm.forward(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(arm.train_params, 0.5)
            opt.step()
            grad_frames += len(b)
            total += float(loss.detach()) * len(b)
        with torch.no_grad():
            vloss_sum, vcorrect = 0.0, 0
            for s in range(0, n_val, val_chunk):
                vb = val_idx[s : s + val_chunk]
                xv = torch.as_tensor(features[vb], device=device)
                yv = torch.as_tensor(actions[vb], device=device, dtype=torch.long)
                logits = arm.forward(xv)
                vloss_sum += float(F.cross_entropy(logits, yv, reduction="sum"))
                vcorrect += int((logits.argmax(-1) == yv).sum())
        row = {
            "epoch": epoch,
            "frames": grad_frames,
            "train_ce": total / max(len(tr_idx), 1),
            "val_ce": vloss_sum / max(n_val, 1),
            "val_acc": vcorrect / max(n_val, 1),
        }
        curve.append(row)
        print(f"  [{arm.name}] epoch {epoch}: train_ce={row['train_ce']:.4f} "
              f"val_ce={row['val_ce']:.4f} val_acc={row['val_acc']:.4f}", flush=True)
        if (epoch + 1) % ckpt_every == 0 or epoch == epochs - 1:
            torch.save(
                {"arm": arm.name, "tap": arm.tap, "epoch": epoch, "curve_row": row,
                 "trainable_state": {k: m.state_dict() for k, m in arm.state_modules.items()}},
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )
    return curve


# ── evaluation ───────────────────────────────────────────────────────────────


def eval_closed_loop(policy, *, device: str, n_levels: int, start_level: int, max_steps: int = 1000) -> dict:
    policy.eval()
    return eval_policy_val(
        policy, n_levels=n_levels, start_level=start_level, device=device,
        max_steps=max_steps, keep_episodes=True,
    )


def write_episodes_csv(path: Path, episodes: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["level", "outcome", "max_progress", "reached_end_region"])
        w.writeheader()
        for e in episodes:
            w.writerow({k: e.get(k) for k in w.fieldnames})


def eval_summary_row(ev: dict) -> dict:
    return {
        "success_rate": ev["success_rate"],
        "gm_rate": ev["gm_rate"],
        "outcome_rates": ev["outcome_rates"],
        "outcome_counts": ev["outcome_counts"],
        "n_levels": ev["n_levels"],
        "mean_max_progress": ev.get("mean_max_progress"),
        "end_reach_rate": ev.get("end_reach_rate"),
        "strata": ev.get("strata"),
    }


# ── orchestration ────────────────────────────────────────────────────────────


def preflight(arms: list[Arm], n_frames: int, natives: list[tuple[str, Path]], args) -> list[dict]:
    print("\n" + "=" * 96, flush=True)
    print(f"PRE-FLIGHT  tap={args.tap} ({TAP_ALIASES[args.tap]})  bank_frames={n_frames}  "
          f"epochs={args.bc_epochs} batch={args.batch_size} lr={args.lr}", flush=True)
    print("=" * 96, flush=True)
    rows = []
    for arm in arms:
        row = {
            "arm": arm.name, "cohort": arm.cohort, "mode": arm.mode, "tap": arm.tap,
            "tap_alias": TAP_ALIASES[arm.tap], "ckpt": str(arm.ckpt),
            "frozen_modules": arm.frozen, "fixed_random_modules": arm.fixed_random,
            "fresh_trainable_modules": arm.fresh_trainable,
            "n_trainable_params": arm.n_trainable,
            "feature_kind": arm.feature_kind,
            "feature_shape": [n_frames, arm.feature_dim],
        }
        rows.append(row)
        print(f"\n  ARM {arm.name}  [{arm.cohort}/{arm.mode}]", flush=True)
        print(f"    tap              : {arm.tap} = {TAP_ALIASES[arm.tap]}", flush=True)
        print(f"    trunk ckpt       : {arm.ckpt}", flush=True)
        print(f"    frozen           : {', '.join(arm.frozen)}", flush=True)
        print(f"    fixed random     : {', '.join(arm.fixed_random) or '(none)'}", flush=True)
        print(f"    fresh trainable  : {', '.join(arm.fresh_trainable)}", flush=True)
        print(f"    trainable params : {arm.n_trainable}", flush=True)
        print(f"    cached features  : {arm.feature_kind} float32 [{n_frames}, {arm.feature_dim}]", flush=True)
    for name, ckpt, cohort in natives:
        rows.append({"arm": name, "cohort": cohort, "mode": "native", "tap": "none",
                     "ckpt": str(ckpt), "frozen_modules": ["entire policy"], "fixed_random_modules": [],
                     "fresh_trainable_modules": [], "n_trainable_params": 0,
                     "feature_kind": "none", "feature_shape": None})
        print(f"\n  ARM {name}  [native, no retraining]\n    trunk ckpt       : {ckpt}\n"
              f"    frozen           : entire policy\n    trainable params : 0", flush=True)
    print("\n" + "=" * 96 + "\n", flush=True)
    # fc_value is reinitialized by prepare_tap but never optimized: BC is
    # cross-entropy on actions only, and rollouts read the action distribution.
    print("NOTE: fc_value is reinitialized but never trained (BC is CE on actions; "
          "rollouts use dist.probs.argmax) -- it is excluded from the counts above.\n", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank-dir", type=Path, default=None,
                    help="shared BC bank dir; default <out>/../bank so both tap jobs reuse one bank")
    ap.add_argument("--tap", required=True, choices=TAPS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gm-ckpt-dir", type=Path, required=True)
    ap.add_argument("--gm-ckpts", required=True, help="comma list of filenames under --gm-ckpt-dir (abs paths ok)")
    ap.add_argument("--teacher-ckpt-dir", type=Path, required=True)
    ap.add_argument("--teacher-ckpts", default="7001_teach.pth,7002_teach.pth,7003_teach.pth")
    ap.add_argument("--label-teacher", default="", help="label policy; default = first --teacher-ckpts entry")
    ap.add_argument("--train-env", default="coinrun_aisc")
    ap.add_argument("--collect-steps", type=int, default=500_000)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-levels", type=int, default=100_000)
    ap.add_argument("--bank-seed", type=int, default=0)
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--feature-batch", type=int, default=512)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--eval-start-level", type=int, default=VAL_START)
    ap.add_argument("--eval-hidden", action="store_true",
                    help="also run the hidden-coin eval (doubles eval cost; coin_dependence)")
    ap.add_argument("--skip-native", action="store_true")
    ap.add_argument("--only-arms", default="", help="comma list of arm names to run")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank_dir = args.bank_dir or (args.out.parent / "bank")
    bank_dir.mkdir(parents=True, exist_ok=True)
    only = set(a for a in args.only_arms.split(",") if a) or None

    def _resolve(f: str, base: Path) -> Path:
        return Path(f) if f.startswith("/") else base / f

    gm_ckpts = [_resolve(f, args.gm_ckpt_dir) for f in args.gm_ckpts.split(",")]
    teacher_ckpts = [_resolve(f, args.teacher_ckpt_dir) for f in args.teacher_ckpts.split(",")]
    label_ckpt = _resolve(args.label_teacher, args.teacher_ckpt_dir) if args.label_teacher else teacher_ckpts[0]
    for p in gm_ckpts + teacher_ckpts + [label_ckpt]:
        assert p.exists(), f"missing checkpoint {p}"
    if args.eval_start_level < args.num_levels:
        raise SystemExit(f"STOP: eval levels [{args.eval_start_level}, ...) overlap the BC level range "
                         f"[0, {args.num_levels}) -- eval would not be held out")

    # Behaviour mixture: every checkpoint under test drives an equal share of the
    # bank, so ~50% GM-visited and ~50% teacher-visited states, and no single seed
    # (including the labeller) dominates the state distribution.
    sources = ([(f"gm_{ckpt_label(p)}", p) for p in gm_ckpts]
               + [(f"nongm_{ckpt_label(p)}", p) for p in teacher_ckpts])
    obs_u8, bank_actions, bank_src, bank_meta = load_or_build_bank(
        bank_dir, sources=sources, label_ckpt=label_ckpt, device=args.device, args=args)
    n_frames = int(obs_u8.shape[0])
    hist = np.bincount(bank_src, minlength=len(sources)).tolist()
    print("bank source mixture: " + ", ".join(
        f"{sources[i][0]}={hist[i]}" for i in range(len(sources))), flush=True)
    gm_frac = sum(hist[: len(gm_ckpts)]) / max(n_frames, 1)
    print(f"bank GM-rolled fraction = {gm_frac:.3f} (target 0.5)", flush=True)

    specs = []
    for p in gm_ckpts:
        specs.append((f"gm_rewire_{ckpt_label(p)}", "gm", "rewire", p))
        specs.append((f"gm_random_proj_{ckpt_label(p)}", "gm", "random_proj", p))
    for p in teacher_ckpts:
        specs.append((f"nongm_rewire_{ckpt_label(p)}", "nongm", "rewire", p))
        specs.append((f"nongm_random_proj_{ckpt_label(p)}", "nongm", "random_proj", p))
    if only is not None:
        specs = [s for s in specs if s[0] in only]
    # Arm names key the per-arm JSON cache; a collision (two checkpoints with the same
    # filename stem, e.g. two absolute .../model_55050240.pth paths) would silently
    # overwrite one arm's result with another's.
    dupes = {n for n, *_ in specs if sum(1 for m, *_ in specs if m == n) > 1}
    if dupes:
        raise SystemExit(f"STOP: duplicate arm names from checkpoint stems: {sorted(dupes)}")

    arms = [build_arm(name=n, cohort=c, mode=m, tap=args.tap, ckpt=p, device=args.device,
                      init_seed=args.init_seed) for (n, c, m, p) in specs]
    natives = ([] if args.skip_native else
               [(f"native_gm_{ckpt_label(p)}", p, "gm") for p in gm_ckpts]
               + [(f"native_nongm_{ckpt_label(p)}", p, "nongm") for p in teacher_ckpts])
    if only is not None:
        natives = [nv for nv in natives if nv[0] in only]
    pre = preflight(arms, n_frames, natives, args)
    (args.out / "preflight.json").write_text(json.dumps(
        {"tap": args.tap, "bank": bank_meta, "source_frame_counts": hist,
         "gm_rolled_fraction": gm_frac, "arms": pre}, indent=2))

    rows: dict[str, dict] = {}

    def _cached(path: Path, name: str) -> dict | None:
        # A restarted run reuses completed arms instead of recomputing them.
        if path.exists():
            print(f"=== {name}: cached result found, skipping ===", flush=True)
            return json.loads(path.read_text())
        return None

    # Native arms are tap-independent and deterministic, so they are cached beside the
    # shared bank and reused by the other tap.
    native_dir = bank_dir / "natives"
    native_dir.mkdir(parents=True, exist_ok=True)
    for name, ckpt, cohort in natives:
        cache = native_dir / f"{name}.json"
        r = _cached(cache, name)
        if r is None:
            t0 = time.time()
            print(f"=== {name} (native, no retraining) ===", flush=True)
            policy = load_policy(ckpt, args.device)
            ev = eval_closed_loop(policy, device=args.device, n_levels=args.eval_n,
                                  start_level=args.eval_start_level)
            write_episodes_csv(native_dir / f"{name}_episodes.csv", ev.get("episodes", []))
            r = {"arm": name, "cohort": cohort, "mode": "native",
                 "tap": "none", "ckpt": str(ckpt), "n_trainable_params": 0,
                 "eval": eval_summary_row(ev), "success_rate": ev["success_rate"],
                 "gm_rate": ev["gm_rate"], "mean_max_progress": ev.get("mean_max_progress"),
                 "elapsed_s": time.time() - t0}
            if args.eval_hidden:
                r["coin_dependence"] = eval_coin_dependence(
                    policy, device=args.device, n_levels=args.eval_n, start_level=args.eval_start_level)
            cache.write_text(json.dumps(r, indent=2))
            del policy
        rows[name] = r
        print(f"  -> {name}: success_rate={r['success_rate']:.3f} gm_rate={r['gm_rate']:.3f}", flush=True)

    by_ckpt: dict[str, list[Arm]] = {}
    for arm in arms:
        by_ckpt.setdefault(str(arm.ckpt), []).append(arm)

    for ckpt_str, group in by_ckpt.items():
        pending = [a for a in group if not (args.out / f"{a.name}.json").exists()]
        for a in group:
            if a not in pending:
                rows[a.name] = json.loads((args.out / f"{a.name}.json").read_text())
                print(f"=== {a.name}: cached result found, skipping ===", flush=True)
        if not pending:
            continue
        t0 = time.time()
        print(f"=== caching {TAP_ALIASES[args.tap]} features for {Path(ckpt_str).name} ===", flush=True)
        src_policy = load_policy(Path(ckpt_str), args.device)
        h_tap = cache_features(src_policy, obs_u8, args.device, batch=args.feature_batch,
                               apply_fc=(args.tap == "fc"), check_ref=(args.tap == "block3"))
        del src_policy
        print(f"  cached {h_tap.shape} in {time.time() - t0:.0f}s", flush=True)

        for arm in pending:
            t1 = time.time()
            print(f"=== {arm.name} tap={arm.tap} mode={arm.mode} trunk={arm.ckpt.name} "
                  f"label={label_ckpt.name} trainable={arm.n_trainable} ===", flush=True)
            feats = arm_features(arm, h_tap, args.device, batch=args.feature_batch)
            curve = train_bc(arm, feats, bank_actions, device=args.device, epochs=args.bc_epochs,
                             batch_size=args.batch_size, lr=args.lr,
                             ckpt_dir=args.out / "ckpts" / arm.name, ckpt_every=args.ckpt_every)
            if feats is not h_tap:
                del feats
            ev = eval_closed_loop(arm.policy, device=args.device, n_levels=args.eval_n,
                                  start_level=args.eval_start_level)
            write_episodes_csv(args.out / f"{arm.name}_episodes.csv", ev.get("episodes", []))
            r = {
                "arm": arm.name, "cohort": arm.cohort, "mode": arm.mode, "tap": arm.tap,
                "tap_alias": TAP_ALIASES[arm.tap], "ckpt": str(arm.ckpt),
                "label_ckpt": str(label_ckpt),
                "label_source_is_own_trunk": Path(arm.ckpt).resolve() == Path(label_ckpt).resolve(),
                "n_trainable_params": arm.n_trainable,
                "frozen_modules": arm.frozen, "fixed_random_modules": arm.fixed_random,
                "fresh_trainable_modules": arm.fresh_trainable,
                "feature_kind": arm.feature_kind, "feature_shape": [n_frames, arm.feature_dim],
                "n_frames": n_frames,
                "bc_curve": curve,
                "final_val_ce": curve[-1]["val_ce"], "final_val_acc": curve[-1]["val_acc"],
                "eval": eval_summary_row(ev),
                "success_rate": ev["success_rate"], "gm_rate": ev["gm_rate"],
                "mean_max_progress": ev.get("mean_max_progress"),
                "elapsed_s": time.time() - t1,
            }
            if args.eval_hidden:
                r["coin_dependence"] = eval_coin_dependence(
                    arm.policy, device=args.device, n_levels=args.eval_n, start_level=args.eval_start_level)
            (args.out / f"{arm.name}.json").write_text(json.dumps(r, indent=2))
            rows[arm.name] = r
            print(f"  -> {arm.name}: success_rate={r['success_rate']:.3f} gm_rate={r['gm_rate']:.3f} "
                  f"mean_max_progress={r['mean_max_progress']:.3f} val_acc={r['final_val_acc']:.3f} "
                  f"({r['elapsed_s']:.0f}s)", flush=True)
            arm.policy.to("cpu")
        del h_tap

    args.gm_rolled_fraction = gm_frac
    summary = summarize(rows, tap=args.tap, bank_meta=bank_meta, label_ckpt=label_ckpt, args=args)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _sr(rows: dict, cohort: str, mode: str, exclude_self_labelled: bool = False) -> np.ndarray:
    vals = [r["success_rate"] for r in rows.values()
            if r.get("cohort") == cohort and r.get("mode") == mode
            and not (exclude_self_labelled and r.get("label_source_is_own_trunk"))]
    return np.asarray(vals, dtype=np.float64)


def _prog(rows: dict, cohort: str, mode: str) -> float:
    vals = [r["mean_max_progress"] for r in rows.values()
            if r.get("cohort") == cohort and r.get("mode") == mode and r.get("mean_max_progress") is not None]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(rows: dict, *, tap: str, bank_meta: dict, label_ckpt: Path, args) -> dict:
    groups = {}
    for cohort in ("gm", "nongm"):
        for mode in ("rewire", "random_proj", "native"):
            v = _sr(rows, cohort, mode)
            if v.size:
                groups[f"{cohort}_{mode}"] = {
                    "n_seeds": int(v.size), "per_seed_success_rate": [float(x) for x in v],
                    "mean_success_rate": float(v.mean()),
                    "ci95_across_seeds": bootstrap_mean(v),
                    "mean_max_progress": _prog(rows, cohort, mode),
                }
    contrasts = {}

    def _add(name: str, a: np.ndarray, b: np.ndarray):
        if a.size and b.size:
            contrasts[name] = bootstrap_diff_ci(a, b)

    gm_re, ng_re = _sr(rows, "gm", "rewire"), _sr(rows, "nongm", "rewire")
    _add("gm_rewire_minus_nongm_rewire", gm_re, ng_re)
    _add("gm_rewire_minus_gm_random_proj", gm_re, _sr(rows, "gm", "random_proj"))
    _add("nongm_rewire_minus_nongm_random_proj", ng_re, _sr(rows, "nongm", "random_proj"))
    ng_re_x = _sr(rows, "nongm", "rewire", exclude_self_labelled=True)
    _add("gm_rewire_minus_nongm_rewire_excl_self_labelled", gm_re, ng_re_x)

    gates = {}
    for cohort in ("gm", "nongm"):
        nat = _prog(rows, cohort, "native")
        for mode in ("rewire", "random_proj"):
            cand = _prog(rows, cohort, mode)
            if cand == cand and nat == nat:
                gates[f"{cohort}_{mode}_vs_{cohort}_native"] = motor_collapse_flag(cand, nat)

    norm = {}
    if gm_re.size and ng_re.size and ng_re.mean() > 0:
        norm["r_norm_gm_rewire_over_nongm_rewire"] = float(gm_re.mean() / ng_re.mean())
    nat_ng = _sr(rows, "nongm", "native")
    if gm_re.size and nat_ng.size and nat_ng.mean() > 0:
        norm["gm_rewire_over_native_nongm"] = float(gm_re.mean() / nat_ng.mean())

    return {
        "marker": ALIGNED_REWIRE_MARKER,
        "tap": tap, "tap_alias": TAP_ALIASES[tap],
        "config": {"collect_steps": args.collect_steps, "bc_epochs": args.bc_epochs,
                   "batch_size": args.batch_size, "lr": args.lr, "eval_n": args.eval_n,
                   "eval_start_level": args.eval_start_level, "bc_num_levels": args.num_levels,
                   "init_seed": args.init_seed, "label_ckpt": str(label_ckpt),
                   "eval_hidden": bool(args.eval_hidden)},
        "bank": bank_meta,
        "bank_gm_rolled_fraction": args.gm_rolled_fraction,
        "groups": groups,
        "contrasts": contrasts,
        "motor_collapse_gates": gates,
        "normalized": norm,
        "rows": {k: {kk: vv for kk, vv in v.items() if kk != "bc_curve"} for k, v in rows.items()},
        "bc_curves": {k: v.get("bc_curve") for k, v in rows.items() if v.get("bc_curve")},
        "caveats": [
            "n=3 checkpoint seeds per cohort: the across-seed bootstrap CI is coarse and "
            "cannot resolve small effects; treat as moderate confidence at best.",
            "Only three Coin teacher checkpoints exist, so the nongm arm whose trunk is the "
            "label teacher is self-labelled (label_source_is_own_trunk=true). The "
            "*_excl_self_labelled contrast drops it; if the two disagree, the Non-GM cohort "
            "mean is carrying a label-match advantage.",
            "Behaviour policies in the bank act greedily (argmax), so state coverage comes "
            "from procgen level diversity rather than action noise.",
            "At the fc tap the random-projection control is a fixed square 256->256 map + "
            "ReLU: capacity-matched but almost surely invertible, so it scrambles the basis "
            "rather than bottlenecking information. It is a weaker control than its h_pre "
            "counterpart and must not be read as the same test.",
            "BC labels are teacher greedy argmax, so every rewired arm inherits the "
            "teacher's own failure modes; absolute success rates are bounded by the "
            "teacher's competence on these levels (see native_nongm arms).",
        ],
    }


if __name__ == "__main__":
    main()
