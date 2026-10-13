#!/usr/bin/env python3
"""Coin local steering at the post-ReLU h_fc tap on the frozen canonical 600-state conflict bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gm_geometry.probes.coin_canonical_bank import load_bank, select
from gm_geometry.probes.coin_location import (
    reconstruct_competing_target_h_fc,
)
from gm_geometry.probes.coin_rollout import (
    harvest_with_state_snapshot,
    load_policy,
    rollout_from_state,
    validate_probe_noninterference,
    validate_snapshot_roundtrip,
)
from gm_geometry.probes.h_fc_steering import push_hook_from_numpy, verify_h_fc_hook

GM_CKPTS = ["6033_gm_20M.pth", "6037_gm_120M.pth", "6051_gm_55M.pth"]
GM_LABELS = ["gm_6033", "gm_6037", "gm_6051"]
NONGM_CKPTS = ["7001_teach.pth", "7002_teach.pth", "7003_teach.pth"]
NONGM_LABELS = ["nongm_7001", "nongm_7002", "nongm_7003"]

H_FC_DIM = 256
SOURCE_START = {"gm_rollouts": 1_100_000, "nongm_rollouts": 1_120_000}
SOURCE_WARM = {"gm_rollouts": 0, "nongm_rollouts": 0}  # index into that cohort's ckpt list


def make_push_hook_factory(policy):
    def _factory(delta_vec: np.ndarray, device: str):
        def _delta_fn_np(_h_fc_np: np.ndarray) -> np.ndarray:
            return delta_vec
        hook_fn = push_hook_from_numpy(_delta_fn_np)
        return policy.embedder.register_forward_hook(hook_fn)
    return _factory


def make_delta_fn(policy, device: str, beta: float, sign: float):
    def _fn(env, pol, dev):
        d_comp, ok = reconstruct_competing_target_h_fc(env, pol, dev)
        if not ok or d_comp is None:
            return None, False
        return sign * beta * d_comp, True
    return _fn


def make_random_delta_fn(rng, device: str, beta: float):
    """Norm-matched random push: same magnitude as the corrective push at this beta, random direction."""
    def _fn(env, pol, dev):
        d_comp, ok = reconstruct_competing_target_h_fc(env, pol, dev)
        if not ok or d_comp is None:
            return None, False
        norm = float(np.linalg.norm(d_comp)) * beta
        r = rng.normal(size=d_comp.shape[0])
        r = r / (np.linalg.norm(r) + 1e-18)
        return r * norm, True
    return _fn


def run_h_fc_verification(policy, device: str) -> dict:
    obs = torch.zeros(1, 3, 64, 64, dtype=torch.float32, device=device)
    report = verify_h_fc_hook(policy, obs)
    print("h_fc hook verification:", report, flush=True)
    if not report["matches"]:
        raise SystemExit(f"REFUSING TO PROCEED: h_fc hook verification failed: {report}")
    if report["dim"] != H_FC_DIM:
        raise SystemExit(f"REFUSING TO PROCEED: h_fc dim {report['dim']} != expected {H_FC_DIM}")
    return report


def recover_state_bytes(bank_states: list[dict], source: str, warm_policy, device: str,
                        max_levels: int) -> list[dict]:
    """Replay the deterministic harvest and attach state bytes to bank states.

    The bank stores geometry only. skip_policy_gate=True reproduces the bank's candidate pool, and any
    bank level missing from the replay aborts the run.
    """
    want = {int(s["level"]) for s in bank_states}
    n_needed = len(want)
    frames = harvest_with_state_snapshot(
        n_frames=n_needed, start_level=SOURCE_START[source], max_levels=max_levels,
        warm_policy=warm_policy, device=device, skip_policy_gate=True,
    )
    by_level = {int(f["level"]): f for f in frames}
    missing = sorted(want - set(by_level))
    print(f"[{source}] bank wants {n_needed} levels; harvest returned {len(frames)}; "
          f"missing {len(missing)}", flush=True)
    if missing:
        raise SystemExit(
            f"REFUSING TO PROCEED: harvest replay for {source} missed {len(missing)} bank "
            f"levels (first few: {missing[:5]}) -- population would differ from the frozen bank")
    out = []
    for s in bank_states:
        f = by_level[int(s["level"])]
        # Keep the whole harvest frame: validate_snapshot_roundtrip needs obs_a alongside state_reloc.
        rec = dict(f)
        rec["bank"] = s
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank-json", type=Path, default=None)
    ap.add_argument("--expect-sha256", type=str, default=None)
    ap.add_argument("--max-states", type=int, default=0,
                    help="cap states per source for smoke runs; 0 = all")
    ap.add_argument("--max-levels", type=int, default=40_000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-random-dirs", type=int, default=3)
    ap.add_argument("--rollout-max-steps", type=int, default=150)
    ap.add_argument("--refresh-stride", type=int, default=1)
    ap.add_argument("--betas", type=str, default="0.125,0.25,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    rng = np.random.default_rng(args.seed)
    betas = [float(x) for x in args.betas.split(",")]

    bank = load_bank(args.bank_json, expect_sha256=args.expect_sha256)
    print(f"canonical bank loaded: freeze_sha256={bank['freeze_sha256']} "
          f"n_states={bank['summary']['n_states']}", flush=True)

    ck_gm = [args.ckpt_dir / f for f in GM_CKPTS]
    ck_nongm = [args.ckpt_dir / f for f in NONGM_CKPTS]
    for c in ck_gm + ck_nongm:
        if not c.is_file():
            raise SystemExit(f"missing ckpt {c}")
    print(f"locked checkpoints verified: {[c.name for c in ck_gm + ck_nongm]}", flush=True)

    print("=== h_fc hook verification (gm_6033) ===", flush=True)
    verify_policy = load_policy(ck_gm[0], device)
    run_h_fc_verification(verify_policy, device)
    del verify_policy

    warm_for = {"gm_rollouts": load_policy(ck_gm[0], device),
                "nongm_rollouts": load_policy(ck_nongm[0], device)}

    state_sources: dict = {}
    for src in ("gm_rollouts", "nongm_rollouts"):
        st = select(bank, source=src)
        st = sorted(st, key=lambda s: int(s["level"]))
        if args.max_states:
            st = st[: args.max_states]
        print(f"=== recovering state bytes for {src} ({len(st)} bank states) ===", flush=True)
        state_sources[src] = recover_state_bytes(
            st, src, warm_for[src], device, args.max_levels)

    print("=== snapshot round-trip determinism check ===", flush=True)
    for src, frames in state_sources.items():
        chk = validate_snapshot_roundtrip(frames[: min(6, len(frames))])
        print(f"roundtrip {src}:", chk, flush=True)
        if not chk["all_exact"]:
            raise SystemExit("snapshot round-trip NOT pixel-exact -- refusing to trust rollout numbers")

    push_factory_gm = make_push_hook_factory(warm_for["gm_rollouts"])
    print("=== probe non-interference determinism check (gm_6033, 3 states) ===", flush=True)
    noninterf = []
    for fr in state_sources["gm_rollouts"][:3]:
        r = validate_probe_noninterference(
            warm_for["gm_rollouts"], fr["state_reloc"], device, args.rollout_max_steps,
            push_hook_factory=push_factory_gm, zero_delta_dim=H_FC_DIM,
        )
        noninterf.append(r)
        print(" ", r, flush=True)
    if not all(r["outcome_match"] and r["steps_match"] for r in noninterf):
        raise SystemExit("probe (zero-push) rollout diverges from clean rollout -- refusing to trust results")

    schedule = list(zip(NONGM_LABELS, ck_nongm)) + list(zip(GM_LABELS, ck_gm))
    results: dict = {}
    for src_name, frames in state_sources.items():
        results[src_name] = {}
        for lab, path in schedule:
            print(f"=== {src_name} / {lab} ===", flush=True)
            pol = load_policy(path, device)
            push_factory = make_push_hook_factory(pol)
            beta_summary: dict = {}
            rollout_rows = []
            for beta in betas:
                base_outs, corr_outs, anti_outs, rand_outs = [], [], [], []
                for fr in frames:
                    r_base = rollout_from_state(
                        pol, fr["state_reloc"], delta_fn=None, device=device,
                        push_hook_factory=push_factory, max_steps=args.rollout_max_steps,
                    )
                    r_corr = rollout_from_state(
                        pol, fr["state_reloc"], delta_fn=make_delta_fn(pol, device, beta, +1.0),
                        device=device, push_hook_factory=push_factory,
                        max_steps=args.rollout_max_steps, refresh_stride=args.refresh_stride,
                    )
                    r_anti = rollout_from_state(
                        pol, fr["state_reloc"], delta_fn=make_delta_fn(pol, device, beta, -1.0),
                        device=device, push_hook_factory=push_factory,
                        max_steps=args.rollout_max_steps, refresh_stride=args.refresh_stride,
                    )
                    r_rand = []
                    for _ in range(args.n_random_dirs):
                        rr = rollout_from_state(
                            pol, fr["state_reloc"], delta_fn=make_random_delta_fn(rng, device, beta),
                            device=device, push_hook_factory=push_factory,
                            max_steps=args.rollout_max_steps, refresh_stride=args.refresh_stride,
                        )
                        r_rand.append(rr["outcome"])
                    base_outs.append(r_base["outcome"])
                    corr_outs.append(r_corr["outcome"])
                    anti_outs.append(r_anti["outcome"])
                    rand_outs.extend(r_rand)
                    # policy_actions_differ is recorded per checkpoint and state for downstream subsetting, not used as a filter.
                    rollout_rows.append({
                        "level": fr["level"], "beta": beta, "baseline": r_base["outcome"],
                        "corrective": r_corr["outcome"], "anticorrective": r_anti["outcome"],
                        "random": r_rand,
                        "policy_actions_differ": bool(
                            fr["bank"]["per_ckpt"][lab]["policy_actions_differ"]),
                    })

                def _rate(outs, key="success"):
                    return float(np.mean([o == key for o in outs])) if outs else float("nan")

                beta_summary[beta] = {
                    "baseline": _rate(base_outs), "corrective": _rate(corr_outs),
                    "anticorrective": _rate(anti_outs), "random": _rate(rand_outs),
                }
                bs = beta_summary[beta]
                print(f"  [{src_name}/{lab}] beta={beta}: baseline={bs['baseline']:.3f} "
                      f"corrective={bs['corrective']:.3f} anticorrective={bs['anticorrective']:.3f} "
                      f"random={bs['random']:.3f} corr-rand={bs['corrective']-bs['random']:+.3f}",
                      flush=True)

            results[src_name][lab] = {"beta_summary": beta_summary, "rollout_rows": rollout_rows}
            del pol
            out = {"tap": "h_fc_canonical_postrelu",
                   "bank": {"freeze_sha256": bank["freeze_sha256"],
                            "n_states_total": bank["summary"]["n_states"],
                            "n_states_used_per_source": {k: len(v) for k, v in state_sources.items()}},
                   "betas": betas, "refresh_stride": args.refresh_stride,
                   "rollout_max_steps": args.rollout_max_steps,
                   "h_fc_verification": {"dim": H_FC_DIM}, "results": results}
            (args.out / "coin_local_steering_canon.json").write_text(json.dumps(out, indent=2))
            print(f"  [{src_name}/{lab}] done, wrote partial results", flush=True)

    print("all done, wrote", args.out / "coin_local_steering_canon.json", flush=True)


if __name__ == "__main__":
    main()
