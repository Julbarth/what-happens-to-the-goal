#!/usr/bin/env python3
"""Harvest coin-dependence across checkpoints in a run dir (emergence curve)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from run_coin_behaviour import eval_policy_val, load_policy  # noqa: E402
from gm_geometry.utils import device as default_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-levels", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--action-mode", default="greedy", choices=["greedy", "sample"])
    ap.add_argument(
        "--max-steps-filter",
        type=int,
        default=20_000_000,
        help="Only eval ckpts with steps <= this (develop window).",
    )
    ap.add_argument(
        "--only-steps",
        type=int,
        nargs="*",
        default=None,
        help="If set, eval only these exact step counts (ignores max-steps-filter).",
    )
    ap.add_argument("--env-name", default="coinrun_aisc")
    args = ap.parse_args()
    args.device = args.device or str(default_device())

    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = []
    only = set(args.only_steps) if args.only_steps else None
    for p in sorted(run_dir.glob("model_*.pth")):
        m = re.search(r"model_(\d+)\.pth$", p.name)
        if not m:
            continue
        steps = int(m.group(1))
        if only is not None:
            if steps in only:
                ckpts.append((steps, p))
        elif steps <= args.max_steps_filter:
            ckpts.append((steps, p))
    ckpts.sort()
    if not ckpts:
        raise SystemExit(
            f"no matching ckpts in {run_dir} "
            f"(only_steps={args.only_steps}, max_filter={args.max_steps_filter})"
        )

    cells = {}
    deps = {}
    for steps, ckpt in ckpts:
        tag = f"steps_{steps}"
        rates = {}
        for hide in (False, True):
            key = f"{tag}_hide{int(hide)}"
            print(f"=== {run_dir.name} {key} n={args.n_levels} ===", flush=True)
            policy = load_policy(ckpt, args.device)
            summary = eval_policy_val(
                policy,
                n_levels=args.n_levels,
                start_level=1,
                device=args.device,
                max_steps=args.max_steps,
                action_mode=args.action_mode,
                hide_visible_coin=hide,
                env_name=args.env_name,
            )
            summary["steps"] = steps
            summary["checkpoint"] = str(ckpt)
            summary["hide"] = hide
            cells[key] = summary
            rates[hide] = summary["success_rate"]
            print(
                f"  success={summary['success_rate']:.3f} gm={summary['gm_rate']:.3f}",
                flush=True,
            )
            del policy
        deps[tag] = {
            "steps": steps,
            "visible": rates[False],
            "hidden": rates[True],
            "coin_dependence": rates[False] - rates[True],
        }
        # Rewritten after every checkpoint so partial results survive an interruption.
        payload = {
            "run_dir": str(run_dir),
            "n_levels": args.n_levels,
            "action_mode": args.action_mode,
            "cells": cells,
            "coin_dependence": deps,
        }
        (out_dir / "ladder_coin_dep.json").write_text(json.dumps(payload, indent=2))

    print(f"wrote {out_dir / 'ladder_coin_dep.json'}")
    for k, v in deps.items():
        print(f"  {k}: dep={v['coin_dependence']:+.3f} vis={v['visible']:.3f} hid={v['hidden']:.3f}")


if __name__ == "__main__":
    main()
