#!/usr/bin/env python3
"""Coin signed derivative and Fisher Q/G/D_F on the frozen canonical bank, replayed from its deterministic harvest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gm_geometry.probes.coin_canonical_bank import load_bank
from gm_geometry.probes.coin_rollout import harvest_with_state_snapshot

# Metric code is imported unchanged: fc scores h_fc, block3 scores h_pre = vec H^(3).
# The reference action a_coin is the warm policy's argmax at the harvested state, so it is
# policy-derived even though the canonical bank guarantees the goal coin is strictly LEFT.
from run_coin_conflict_signed_derivative import score_agent_on_frames as _score_fc
from run_coin_conflict_signed_derivative_block3 import score_agent_on_frames as _score_block3
from run_coin_fisher_sensitivity import load_init_policy, load_policy


def materialise(bank: dict, *, ckpt_gm0: Path, ckpt_nongm0: Path, device: str,
                max_levels: int, cap: int | None) -> dict[str, list[dict]]:
    """Replay the bank's own harvest and keep only its banked (source, level)."""
    want = {}
    for src in ("gm_rollouts", "nongm_rollouts"):
        want[src] = {int(s["level"]) for s in bank["states"] if s["source"] == src}
    if cap is not None:
        want = {k: set(sorted(v)[:cap]) for k, v in want.items()}

    warm = {
        "gm_rollouts": (load_policy(ckpt_gm0, device), 1_100_000),
        "nongm_rollouts": (load_policy(ckpt_nongm0, device), 1_120_000),
    }
    out: dict[str, list[dict]] = {}
    for src, (pol, start) in warm.items():
        target = want[src]
        # Harvest enough candidates to span the banked level range.
        need = max(target) - start + 2
        print(f"=== replay {src}: seeking {len(target)} banked levels "
              f"(range {min(target)}..{max(target)}) ===", flush=True)
        frames = harvest_with_state_snapshot(
            n_frames=len(target) * 3, start_level=start,
            max_levels=min(max_levels, need + 50), warm_policy=pol, device=device,
            skip_policy_gate=True,
        )
        keep = [f for f in frames if int(f["level"]) in target]
        got = {int(f["level"]) for f in keep}
        missing = target - got
        print(f"{src}: replayed {len(frames)} frames, matched {len(keep)}/{len(target)}", flush=True)
        if missing:
            raise SystemExit(
                f"{src}: replay missed {len(missing)} banked levels "
                f"(e.g. {sorted(missing)[:5]}) -- refusing to score a subset")
        out[src] = sorted(keep, key=lambda f: int(f["level"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--bank-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-levels", type=int, default=40_000)
    ap.add_argument("--cap", type=int, default=None,
                    help="smoke only: keep the first N banked levels per source")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tap", choices=("fc", "block3"), default="fc",
                    help="fc = h_fc (default); "
                         "block3 = h_pre. The block3 scorer takes no finite-difference "
                         "check, so --n-findiff is ignored there.")
    ap.add_argument("--n-findiff", type=int, default=8)
    ap.add_argument("--findiff-eps-frac", type=float, default=0.02)
    ap.add_argument("--n-init", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    bank = load_bank(args.bank_json)          # verifies freeze sha256, raises on drift
    print(f"bank freeze_sha256 {bank['freeze_sha256']}", flush=True)
    print(f"bank n_states {bank['summary']['n_states']}", flush=True)

    ck_nongm = [args.ckpt_dir / f"{s}_teach.pth" for s in (7001, 7002, 7003)]
    ck_gm = [args.ckpt_dir / "6033_gm_20M.pth", args.ckpt_dir / "6037_gm_120M.pth"]
    ck_gm.append(args.ckpt_dir / "6051_gm_55M.pth")
    for c in ck_nongm + ck_gm:
        if not c.is_file():
            raise SystemExit(f"missing ckpt {c}")

    schedule = [
        ("nongm_7001", ck_nongm[0]), ("nongm_7002", ck_nongm[1]), ("nongm_7003", ck_nongm[2]),
        ("gm_6033", ck_gm[0]), ("gm_6037", ck_gm[1]), ("gm_6051", ck_gm[2]),
    ]

    sources = materialise(bank, ckpt_gm0=ck_gm[0], ckpt_nongm0=ck_nongm[0],
                          device=device, max_levels=args.max_levels, cap=args.cap)

    # Per-checkpoint responsiveness, carried from the bank so the subset split is
    # reproducible from the frozen artifact rather than recomputed here.
    resolves = {
        (s["source"], int(s["level"])): {
            l: bool(v["policy_actions_differ"]) for l, v in s["per_ckpt"].items()
        } for s in bank["states"]
    }

    results: dict = {}
    per_frame: dict = {}
    for src_name, frames in sources.items():
        print(f"\n=== score all agents on {src_name} (n={len(frames)}) ===", flush=True)
        results[src_name] = {}
        per_frame[src_name] = {}
        for lab, path in schedule:
            pol = load_policy(path, device)
            if args.tap == "block3":
                out = _score_block3(pol, frames, device=device, label=lab)
            else:
                n_findiff = args.n_findiff if (src_name == "gm_rollouts" and lab == "gm_6033") else 0
                out = _score_fc(
                    pol, frames, device=device, label=lab,
                    n_findiff=n_findiff, findiff_eps_frac=args.findiff_eps_frac,
                )
            for r in out["rows"]:
                r["policy_actions_differ"] = resolves[(src_name, int(r["level"]))][lab]
            results[src_name][lab] = out["summary"]
            per_frame[src_name][lab] = out["rows"]
            print(f"{src_name}/{lab}: {json.dumps(out['summary'])}", flush=True)
        for i in range(args.n_init):
            # init_seed=4000+i is shared with run_coin_conflict_signed_derivative, so init floors are comparable.
            init_p = load_init_policy(ck_nongm[0], device, init_seed=4000 + i)
            lab = f"init_{i}"
            out = (_score_block3 if args.tap == "block3" else _score_fc)(
                init_p, frames, device=device, label=lab)
            results[src_name][lab] = out["summary"]
            per_frame[src_name][lab] = out["rows"]

    blob = {
        "tap": args.tap,
        "bank_freeze_sha256": bank["freeze_sha256"],
        "bank_n_states": bank["summary"]["n_states"],
        "cap": args.cap,
        "n_scored": {k: len(v) for k, v in sources.items()},
        "results": results,
        "per_frame": per_frame,
    }
    suffix = "" if args.tap == "fc" else f"_{args.tap}"
    fp = args.out / f"coin_conflict_signed_derivative_canon{suffix}.json"
    fp.write_text(json.dumps(blob, indent=1))
    print(f"\nwrote {fp}", flush=True)


if __name__ == "__main__":
    main()
