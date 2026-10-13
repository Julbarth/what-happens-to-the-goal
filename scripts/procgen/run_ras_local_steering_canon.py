#!/usr/bin/env python3
"""Attribute-Maze state-local d_comp(s_t) steering on the frozen canonical bank, at h_fc or h_pre, resumable per state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_CS_ROOT = ROOT / "external" / "colour-shape-goal-misgeneralization"
sys.path.insert(0, str(_CS_ROOT / "procgenAISC"))
sys.path.insert(0, str(_CS_ROOT / "train-procgen"))

from gm_geometry.utils.seeding import stable_seed  # noqa: E402
from gm_geometry.integrations.colour_shape.policy import load_impala_policy, load_recurrent_flag  # noqa: E402
from gm_geometry.integrations.colour_shape import counterfactual_hfc as cf_fc  # noqa: E402
from gm_geometry.integrations.colour_shape import counterfactual_hpre_local_steering as cf_pre  # noqa: E402
from gm_geometry.probes import ras_bank_io  # noqa: E402
from gm_geometry.probes import h_fc_steering, h_pre_steering  # noqa: E402
from gm_geometry.utils import seeding  # noqa: E402
from gm_geometry.probes import ras_counterfactuals  # noqa: E402
from gm_geometry.integrations.colour_shape import policy as policy_mod  # noqa: E402

TAPS = {
    "h_fc": dict(verify=cf_fc.assert_hfc_hook_verified, d0=cf_fc.compute_d_comp0_hfc,
                 jac=cf_fc.fc_logit_jacobian_and_act_hfc,
                 dcomp=cf_fc.rollout_with_dcomp_hfc, fixed=cf_fc.rollout_with_fixed_push_hfc,
                 module=cf_fc.__file__),
    "h_pre": dict(verify=cf_pre.assert_hpre_hook_verified, d0=cf_pre.compute_d_comp0_hpre,
                  jac=cf_pre.fc_logit_jacobian_and_act_hpre,
                  dcomp=cf_pre.rollout_with_dcomp_hpre, fixed=cf_pre.rollout_with_fixed_push_hpre,
                  module=cf_pre.__file__),
}


def file_sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_base_module(use_backgrounds: bool):
    if use_backgrounds:
        import run_ras_conflict_signed_derivative_texture as base  # noqa: E402
    else:
        import run_ras_conflict_signed_derivative as base  # noqa: E402
    return base


def read_done(path: Path) -> set[str]:
    """State keys whose full row block was committed (the state_done marker is written last)."""
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "state_done":
            done.add(r["state_key"])
    return done


def append_block(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True, help="seed:path")
    ap.add_argument("--tap", choices=sorted(TAPS), default="h_fc")
    ap.add_argument("--edit", choices=sorted(cf_fc.EditWorlds.FAMILIES), default="swap",
                    help="counterfactual family for d_comp: swap = canonical (paper A.2.2), ablation = supplementary")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--distribution-mode", type=str, default="easy")
    ap.add_argument("--use-backgrounds", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--betas", type=str, default="0.125,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--replay-stride", type=int, default=1)
    ap.add_argument("--n-random-dirs", type=int, default=3)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ras_bank_io.add_bank_args(ap)
    args = ap.parse_args()
    if not ras_bank_io.check_bank_args(args):
        raise SystemExit("canonical steering runs only on the frozen bank")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    seed_str, ckpt_str = args.ckpt.split(":", 1)
    seed, ckpt = int(seed_str), Path(ckpt_str)
    betas = [float(b) for b in args.betas.split(",")]
    ub = args.use_backgrounds
    bg = "texture" if ub else "black"
    tap = TAPS[args.tap]
    base = cf_fc.EditWorlds(load_base_module(ub), args.edit)

    tag = f"{'swap_' if args.edit == 'swap' else ''}{args.tap}_{bg}_{seed}_{args.split_role}{'_exd1' if args.exclude_d1 else ''}_shard{args.shard}of{args.n_shards}"
    rows_fp = args.out / f"rows_{tag}.jsonl"
    meta_fp = args.out / f"meta_{tag}.json"

    probe_env = base.make_env(base.OBJ_LINE, base.OBJ_GEM, 0, 0, args.distribution_mode, ub)
    n_actions = int(probe_env.action_space.n)
    probe_env.close()

    matched, prov, _ = ras_bank_io.bank_res_list(
        args, args.split_role, use_backgrounds=ub, state_bytes=base.state_bytes,
        make_base_env=lambda lvl: base.make_env(base.OBJ_LINE, base.OBJ_GEM, lvl, 0, args.distribution_mode, ub))
    mine = matched[args.shard::args.n_shards]
    done = read_done(rows_fp)
    todo = [r for r in mine if r["state_key"] not in done]
    print(f"[setup] tap={args.tap} bg={bg} seed={seed} role={args.split_role} shard={args.shard}/{args.n_shards} "
          f"states={len(mine)} done={len(mine) - len(todo)} betas={betas} stride={args.replay_stride} "
          f"max_steps={args.max_steps}", flush=True)

    policy = load_impala_policy(ckpt, in_channels=3, action_size=n_actions, recurrent=load_recurrent_flag(),
                                device=args.device)
    verify = tap["verify"](policy, base.to_nchw(matched[0]["obs0"]), args.device)

    r0 = matched[0]
    runs = [tap["dcomp"](policy, r0["level"], args.distribution_mode, ub, args.device,
                         r0["first_actions_line"], r0["first_actions_gem"], beta=betas[0], sign=1.0,
                         max_steps=args.max_steps, replay_stride=args.replay_stride,
                         expect_state_bytes=r0["state_bytes"], base_module=base) for _ in range(2)]
    if runs[0] != runs[1]:
        raise SystemExit(f"determinism check failed: {runs}")

    src = [Path(__file__), Path(tap["module"]), Path(cf_fc.__file__), Path(ras_counterfactuals.__file__), Path(ras_bank_io.__file__), Path(base.__file__),
           Path(seeding.__file__), Path(h_fc_steering.__file__), Path(h_pre_steering.__file__),
           Path(policy_mod.__file__)]
    meta = {
        "experiment": "ras_local_steering_canon", "tap": args.tap, "edit": args.edit,
        "edit_worlds": {"d_comp_plus": base.edit_names["plus"], "d_comp_minus": base.edit_names["minus"],
                        "objects": {k: list(v) for k, v in base._objs.items()},
                        "d_comp": "h(plus world) - h(minus world), toward the line candidate u"}, "background": bg, "seed": seed,
        "checkpoint": str(ckpt), "checkpoint_sha256": file_sha(ckpt),
        "canonical_bank_sha256": prov.get("canonical_bank_sha256"), "split_sha256": prov["split_sha256"],
        "bank_provenance": prov, "betas": betas, "max_steps": args.max_steps,
        "replay_stride": args.replay_stride, "n_random_dirs": args.n_random_dirs,
        "shard": args.shard, "n_shards": args.n_shards, "hook_verification": verify,
        "determinism_check": {"ok": True, "run": list(runs[0])},
        "source_sha256": {p.name: file_sha(p) for p in src},
        "rand_seed_rule": "stable_seed(seed, state_key, 'rand', tap, beta, k); unit Gaussian scaled to beta*||d_comp(s0)||",
    }
    meta_fp.write_text(json.dumps(meta, indent=2))

    common = dict(tap=args.tap, edit=args.edit, background=bg, seed=seed, replay_stride=args.replay_stride,
                  max_steps=args.max_steps, split_role=args.split_role, checkpoint=str(ckpt),
                  checkpoint_sha256=meta["checkpoint_sha256"], canonical_bank_sha256=prov.get("canonical_bank_sha256"), split_sha256=prov["split_sha256"])
    for i, res in enumerate(todo):
        t0 = time.time()
        level = res["level"]
        ids_l, ids_g = res["first_actions_line"], res["first_actions_gem"]
        st = dict(state_key=res["state_key"], state_hash=res["state_hash"], level=int(level), d=res["d"],
                  is_d1=res["is_d1"], axis_sign=res["axis_sign_lineN_gemE"], n_dup_levels=len(res["dup_levels"]))
        for world in (base.SHAPE_EDIT_OBJ, base.COLOUR_EDIT_OBJ):
            env_w = (base.make_env(world, base.OBJ_GEM, level, 0, args.distribution_mode, ub) if world == base.SHAPE_EDIT_OBJ
                     else base.make_env(base.OBJ_LINE, world, level, 0, args.distribution_mode, ub))
            try:
                env_w.reset()
                raw_sha = hashlib.sha256(ras_bank_io.env_raw_state(env_w)).hexdigest()
            finally:
                env_w.close()
            if raw_sha != res["state_sha256"]:
                raise SystemExit(f"STOP: {args.edit} world {world} at level {level} does not reproduce the bank "
                                 f"start state (agent/walls/targets): {raw_sha} != {res['state_sha256']}")
        d0 = tap["d0"](policy, res, args.distribution_mode, ub, args.device, base_module=base)
        if d0 is None:
            raise SystemExit(f"STOP: counterfactual worlds disagree with bank start state at level {level}")
        norm0 = float(np.linalg.norm(d0))
        _, probs0, _ = tap["jac"](policy, base.to_nchw(res["obs0"]), args.device)
        p_line0, p_gem0 = float(probs0[ids_l].sum()), float(probs0[ids_g].sum())

        def dcomp(beta, sign):
            return tap["dcomp"](policy, level, args.distribution_mode, ub, args.device, ids_l, ids_g,
                                beta=beta, sign=sign, max_steps=args.max_steps,
                                replay_stride=(10**9 if beta == 0.0 else args.replay_stride),
                                expect_state_bytes=res["state_bytes"], base_module=base)

        rows = []
        o, n = dcomp(0.0, 1.0)
        rows.append({**common, **st, "kind": "rollout", "arm": "baseline", "beta": 0.0, "k": 0,
                     "outcome": o, "n_steps": n})
        for beta in betas:
            for arm, sign in (("corrective", 1.0), ("anticorrective", -1.0)):
                o, n = dcomp(beta, sign)
                rows.append({**common, **st, "kind": "rollout", "arm": arm, "beta": beta, "k": 0,
                             "outcome": o, "n_steps": n})
            for k in range(args.n_random_dirs):
                rng = np.random.default_rng(stable_seed(seed, res["state_key"], "rand", args.tap, beta, k))
                rdir = rng.normal(size=d0.shape)
                rdir = rdir / (np.linalg.norm(rdir) + 1e-18) * (beta * norm0)
                o, n = tap["fixed"](policy, level, args.distribution_mode, ub, args.device, rdir,
                                    max_steps=args.max_steps, expect_state_bytes=res["state_bytes"],
                                    base_module=base)
                rows.append({**common, **st, "kind": "rollout", "arm": "random", "beta": beta, "k": k,
                             "outcome": o, "n_steps": n})
        rows.append({**common, **st, "kind": "state_done", "norm_dcomp0": norm0, "p_line0": p_line0,
                     "p_gem0": p_gem0, "seconds": time.time() - t0})
        append_block(rows_fp, rows)
        print(f"[progress] {tag} {i + 1}/{len(todo)} level={level} d={res['d']} sign={st['axis_sign']:+d} "
              f"base={rows[0]['outcome']} {time.time() - t0:.1f}s", flush=True)
    print(f"[done] {tag}", flush=True)


if __name__ == "__main__":
    main()
