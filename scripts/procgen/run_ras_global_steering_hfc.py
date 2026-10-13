#!/usr/bin/env python3
"""Attribute Maze shared-axis (u_comp) steering at the post-ReLU h_fc tap: axis fit, held-out alignment, permutation null, dose grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_CS_ROOT = ROOT / "external" / "colour-shape-goal-misgeneralization"
sys.path.insert(0, str(_CS_ROOT / "procgenAISC"))
sys.path.insert(0, str(_CS_ROOT / "train-procgen"))

import run_ras_conflict_signed_derivative as base  # noqa: E402
import run_ras_conflict_signed_derivative_texture as texture_base  # noqa: E402
import run_ras_counterfactual_competition_steering_hfc as hfc  # noqa: E402
from gm_geometry.utils.seeding import stable_seed  # noqa: E402
from gm_geometry.integrations.colour_shape.policy import (  # noqa: E402
    load_impala_policy, load_recurrent_flag,
)
from gm_geometry.probes.compressed_axis_geometry import (  # noqa: E402
    bucket_outcome_distribution, bucket_permutation_null, fisher_projection_alpha,
    fit_u_comp, held_out_alignment, matched_norm_random_direction,
)
from gm_geometry.probes import ras_bank_io  # noqa: E402
from gm_geometry.integrations.colour_shape.counterfactual_hfc import pushed_probs_hfc  # noqa: E402
from gm_geometry.probes import compressed_axis_geometry, h_fc_steering  # noqa: E402
from gm_geometry.utils import seeding  # noqa: E402

OUTCOME_CATS = ("line", "gem", "other_terminal", "timeout")


def _file_sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _read_state_blocks(path: Path) -> dict[str, list[dict]]:
    """Committed per-state row blocks (a block counts only once its state_done marker is present)."""
    rows: dict[str, list[dict]] = {}
    done: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "state_done":
                done.add(r["state_key"])
            elif r.get("kind") == "row":
                rows.setdefault(r["state_key"], []).append(r)
    return {k: v for k, v in rows.items() if k in done}


def bearing_of(res: dict) -> tuple:
    return tuple(res["first_group_line"]) + tuple(res["first_group_gem"])


def harvest_bank(policy, distribution_mode, use_backgrounds, start_level, n_levels_to_try,
                  n_target, n_actions, device, *, need_fisher: bool, matched=None):
    """Returns (matched, d_comp_by_level, fisher_by_level or None, bearing_by_level, groups).

    Textured backgrounds use texture_base.harvest_conflict_levels, since the black localizer's colour-based object
    calibration fails under texture.
    """
    matched_given = matched is not None
    if matched is not None:
        groups = None
    elif use_backgrounds:
        groups, matched = texture_base.harvest_conflict_levels(
            start_level=start_level, n_levels_to_try=n_levels_to_try, n_target=n_target,
            distribution_mode=distribution_mode, use_backgrounds=use_backgrounds,
        )
    else:
        matched = base.find_conflict_levels(
            start_level=start_level, n_levels_to_try=n_levels_to_try, n_target=n_target,
            distribution_mode=distribution_mode, use_backgrounds=use_backgrounds,
        )
        groups = None
    d_comp_by_level: dict[int, np.ndarray] = {}
    fisher_by_level: dict[int, np.ndarray] = {}
    bearing_by_level: dict[int, tuple] = {}
    kept = []
    for res in matched:
        if need_fisher:
            d0, f0 = hfc.compute_d_comp0_and_fisher_hfc(policy, res, distribution_mode, use_backgrounds, device)
        else:
            d0, f0 = hfc.compute_d_comp0_hfc(policy, res, distribution_mode, use_backgrounds, device), None
        if d0 is None:
            if matched_given:
                raise SystemExit(f"STOP: edit worlds do not reproduce bank start state at level {res['level']}")
            continue
        lvl = res["level"]
        d_comp_by_level[lvl] = d0
        if f0 is not None:
            fisher_by_level[lvl] = f0
        bearing_by_level[lvl] = bearing_of(res)
        kept.append(res)
    return kept, d_comp_by_level, (fisher_by_level if need_fisher else None), bearing_by_level, groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, action="append", required=True, help="seed:path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-start-level", type=int, default=5000,
                     help="disjoint from the steering bank's 2000+ range")
    ap.add_argument("--fit-n-levels-to-try", type=int, default=1500)
    ap.add_argument("--fit-n-target", type=int, default=100)
    ap.add_argument("--fit-holdout-frac", type=float, default=0.2)
    ap.add_argument("--steer-start-level", type=int, default=2000)
    ap.add_argument("--steer-n-levels-to-try", type=int, default=800)
    ap.add_argument("--steer-n-target", type=int, default=24)
    ap.add_argument("--distribution-mode", type=str, default="easy")
    ap.add_argument("--use-backgrounds", action="store_true", help="textured-background cohort")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--betas", type=str, default="0.125,0.25,0.5,1.0")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--replay-stride", type=int, default=1,
                    help="d_comp(s_t) refresh stride for full_local; 1 = every step, as in local steering")
    ap.add_argument("--skip-full-local", action="store_true",
                    help="omit the replayed full_local arms (fold-independent; taken from the local-steering run)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--edit", choices=["swap", "ablation"], default="swap",
                    help="counterfactual family for d_comp: swap = canonical (paper A.2.2), ablation = supplementary")
    ap.add_argument("--n-random-dirs", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=2000)
    ras_bank_io.add_bank_args(ap, default_role="eval")
    args = ap.parse_args()
    from gm_geometry.integrations.colour_shape.counterfactual_hfc import EditWorlds
    edit_worlds = EditWorlds(base, args.edit)
    hfc.base = edit_worlds  # routes every counterfactual world built inside the hfc helpers
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    # Bank mode: fit bank = "fit" role (all of it), steering bank = --split-role (must be "eval").
    use_bank = ras_bank_io.check_bank_args(args)
    if use_bank and args.split_role != "eval":
        raise SystemExit("global steering: --split-role is the steering role and must be eval (fit bank = fit role)")
    args.out.mkdir(parents=True, exist_ok=True)
    betas = [float(b) for b in args.betas.split(",")]

    ckpts = {}
    for entry in args.ckpt:
        seed_str, path_str = entry.split(":", 1)
        ckpts[int(seed_str)] = Path(path_str)

    probe_env = base.make_env(base.OBJ_LINE, base.OBJ_GEM, 0, 0, args.distribution_mode, args.use_backgrounds)
    n_actions = int(probe_env.action_space.n)
    probe_env.close()

    bank_fit_res = bank_steer_res = None
    if use_bank:
        _mk = lambda lvl: base.make_env(base.OBJ_LINE, base.OBJ_GEM, lvl, 0, args.distribution_mode, args.use_backgrounds)
        bank_fit_res, bank_fit_prov, _ = ras_bank_io.bank_res_list(
            args, "fit", make_base_env=_mk, state_bytes=base.state_bytes,
            use_backgrounds=args.use_backgrounds, max_states=0)
        bank_steer_res, bank_steer_prov, _ = ras_bank_io.bank_res_list(
            args, "eval", make_base_env=_mk, state_bytes=base.state_bytes, use_backgrounds=args.use_backgrounds)
        f_keys = {r["state_hash"] for r in bank_fit_res}
        s_keys = {r["state_hash"] for r in bank_steer_res}
        assert not f_keys & s_keys, f"{len(f_keys & s_keys)} states in both fit and steering banks"
        f_lv = {lv for r in bank_fit_res for lv in r["dup_levels"]}
        s_lv = {lv for r in bank_steer_res for lv in r["dup_levels"]}
        assert not f_lv & s_lv, "a level is in both fit and steering banks"
        for r in bank_fit_res + bank_steer_res:
            assert ras_bank_io.bank_bucket_sign(bearing_of(r)) == r["axis_sign_lineN_gemE"], r["level"]

    recurrent = load_recurrent_flag()
    results: dict = {}
    if use_bank:
        results["bank_provenance"] = {"fit": bank_fit_prov, "steer": bank_steer_prov}
    else:
        results["LEGACY_HARVEST_WARNING"] = ras_bank_io.LEGACY_HARVEST_WARNING
    for seed, ckpt in ckpts.items():
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint for seed {seed}: {ckpt}")
        print(f"[seed {seed}] loading policy", flush=True)
        policy = load_impala_policy(ckpt, in_channels=3, action_size=n_actions, recurrent=recurrent, device=args.device)
        hfc.assert_h_fc_hook_correct(policy, args.device)
        bg_name = "texture" if args.use_backgrounds else "black"
        split_tag = Path(args.bank_split).stem if use_bank else "legacy"
        edit_tag = "swap_" if args.edit == "swap" else ""
        blocks_fp = args.out / f"rows_{edit_tag}{bg_name}_{seed}_{split_tag}{'_exd1' if args.exclude_d1 else ''}.jsonl"
        done_blocks = _read_state_blocks(blocks_fp) if use_bank else {}
        ckpt_sha = _file_sha(ckpt)

        print(f"[seed {seed}] harvesting FIT bank (start_level={args.fit_start_level}, target={args.fit_n_target})", flush=True)
        fit_matched, fit_d, fit_f, fit_bearing, _fg = harvest_bank(
            policy, args.distribution_mode, args.use_backgrounds, args.fit_start_level,
            args.fit_n_levels_to_try, args.fit_n_target, n_actions, args.device, need_fisher=False,
            matched=(list(bank_fit_res) if use_bank else None))
        print(f"[seed {seed}] fit bank: {len(fit_matched)} usable states", flush=True)

        print(f"[seed {seed}] harvesting STEERING bank (start_level={args.steer_start_level}, target={args.steer_n_target})", flush=True)
        steer_matched, steer_d, steer_f, steer_bearing, _sg = harvest_bank(
            policy, args.distribution_mode, args.use_backgrounds, args.steer_start_level,
            args.steer_n_levels_to_try, args.steer_n_target, n_actions, args.device, need_fisher=True,
            matched=(list(bank_steer_res) if use_bank else None))
        print(f"[seed {seed}] steering bank: {len(steer_matched)} usable states", flush=True)

        fit_levels = set(fit_d.keys())
        steer_levels = set(steer_d.keys())
        overlap = fit_levels & steer_levels
        if use_bank:
            assert not overlap, f"bank fit/steer level overlap {sorted(overlap)}"
        if overlap:
            print(f"[seed {seed}] WARNING: {len(overlap)} levels overlap between fit and steering banks, excluding from fit bank: {sorted(overlap)}", flush=True)
            for lvl in overlap:
                fit_d.pop(lvl, None)
                fit_bearing.pop(lvl, None)

        # Bucket signs are discovered per seed from the observed bearings rather than hardcoded, since cohorts use
        # different bearing-tuple conventions. Each tuple's exact negation is also present (canonical axes x 2 signs),
        # so each tuple is paired with its negation and assigned +-1 per pair.
        all_bearings = sorted(set(fit_bearing.values()) | set(steer_bearing.values()))
        remaining = set(all_bearings)
        bucket_sign: dict[tuple, float] = {}
        if use_bank:
            # Bank bearings pair by line/gem SWAP, not negation (agent at (0,0): only N/E); see bank_bucket_sign.
            bucket_sign = {b: ras_bank_io.bank_bucket_sign(b) for b in all_bearings}
            if None in bucket_sign.values() or len(set(bucket_sign.values())) != 2:
                raise SystemExit(f"seed {seed}: bank bearings {all_bearings} do not form the two swap buckets")
            remaining = set()
        while remaining:
            b = min(remaining)
            neg_b = tuple(-x for x in b)
            if neg_b not in remaining:
                raise SystemExit(
                    f"seed {seed}: bearing {b} has no exact negation among observed bearings "
                    f"{sorted(all_bearings)} -- cannot pair into a signed axis")
            bucket_sign[b] = 1.0
            bucket_sign[neg_b] = -1.0
            remaining -= {b, neg_b}
        print(f"[seed {seed}] discovered {len(bucket_sign) // 2} canonical axis pair(s) from "
              f"{len(all_bearings)} raw bearings: {bucket_sign}", flush=True)
        print(f"[seed {seed}] discovered bucket_sign (data-driven, not black's hardcoded values): {bucket_sign}", flush=True)

        # Holdout is stratified by bearing bucket: buckets are skewed (~75/25), and a plain random split can empty
        # the minority bucket, which breaks fit_u_comp.
        by_bucket_levels: dict[str, list[int]] = {"pos": [], "neg": []}
        for lvl, bearing in fit_bearing.items():
            sgn = bucket_sign.get(bearing)
            if sgn is not None:
                by_bucket_levels["pos" if sgn > 0 else "neg"].append(lvl)
        # hash() of a str-containing tuple is salted per process; bank mode needs a reproducible holdout.
        rng = (np.random.default_rng([int(seed), 7919]) if use_bank
               else np.random.default_rng(stable_seed("split", seed)))
        holdout_levels: set[int] = set()
        for key, levels in by_bucket_levels.items():
            levels = sorted(levels)
            rng.shuffle(levels)
            n_ho = max(1, int(len(levels) * args.fit_holdout_frac)) if len(levels) >= 2 else 0
            holdout_levels.update(levels[:n_ho])
            print(f"[seed {seed}] bucket {key}: {len(levels)} total, {n_ho} held out", flush=True)
        axis_levels = [lvl for lvl in fit_d.keys() if lvl not in holdout_levels]
        if len(axis_levels) < 4:
            raise SystemExit(f"seed {seed}: too few fit-bank states after split ({len(axis_levels)})")
        for key, levels in by_bucket_levels.items():
            n_in_axis = sum(1 for lvl in levels if lvl not in holdout_levels)
            if n_in_axis == 0:
                raise SystemExit(f"seed {seed}: bucket {key} has 0 states in the axis-fit split ({len(levels)} total) -- stratification failed")

        d_by_bucket: dict[str, list[np.ndarray]] = {"pos": [], "neg": []}
        for lvl in axis_levels:
            sgn = bucket_sign.get(fit_bearing[lvl])
            if sgn is None:
                continue
            d_by_bucket["pos" if sgn > 0 else "neg"].append(fit_d[lvl])
        u_comp = fit_u_comp(d_by_bucket)

        ho_d = [fit_d[lvl] for lvl in holdout_levels if fit_bearing[lvl] in bucket_sign]
        ho_sign = [1.0 if bucket_sign[fit_bearing[lvl]] > 0 else -1.0
                   for lvl in holdout_levels if fit_bearing[lvl] in bucket_sign]
        gate = held_out_alignment(u_comp, ho_d, ho_sign)
        print(f"[seed {seed}] held-out gate: {gate['mean']=:.3f} {gate['frac_pos']=:.3f} n={gate['n']}", flush=True)

        axis_bucket_labels = ["pos" if bucket_sign[fit_bearing[lvl]] > 0 else "neg" for lvl in axis_levels
                               if fit_bearing[lvl] in bucket_sign]
        axis_d = [fit_d[lvl] for lvl in axis_levels if fit_bearing[lvl] in bucket_sign]
        perm = bucket_permutation_null(axis_d, axis_bucket_labels, ho_d,
                                        ["pos" if s > 0 else "neg" for s in ho_sign],
                                        n_perm=args.n_perm, seed=stable_seed("perm", seed))
        print(f"[seed {seed}] permutation null: p<={perm['p_upper_bound']:.4g} (n_exceed={perm['n_exceed']}/{perm['n_perm']})", flush=True)

        print(f"[seed {seed}] determinism check (full_local arm) before trusting any rollout number", flush=True)
        level0 = steer_matched[0]["level"]
        ok, r1, r2 = (True, None, None) if args.skip_full_local else hfc.determinism_check_hfc(
            policy, level0, args.distribution_mode, args.use_backgrounds, args.device,
            beta=betas[-1], sign=1.0, max_steps=args.max_steps, replay_stride=args.replay_stride,
            expect_state_bytes=steer_matched[0]["state_bytes"])
        print(f"[seed {seed}] determinism check: ok={ok} run1={r1} run2={r2}", flush=True)
        if not ok:
            raise SystemExit(f"seed {seed}: determinism check FAILED, refusing to trust rollout numbers")

        # sigma*u must point toward the line in BOTH buckets: geometric (cos with d_comp) and behavioural
        # (one-step shift in P(A_line) - P(A_gem) at s0 under a push of 0.5*||d_comp|| along sigma*u).
        eval_alignment = []
        for res in steer_matched:
            sg = bucket_sign.get(bearing_of(res))
            d0 = steer_d[res["level"]]
            push = sg * 0.5 * float(np.linalg.norm(d0)) * u_comp
            x0 = hfc.base.to_nchw(res["obs0"])
            p0 = pushed_probs_hfc(policy, x0, args.device, np.zeros_like(u_comp))
            p1 = pushed_probs_hfc(policy, x0, args.device, push)
            il, ig = res["first_actions_line"], res["first_actions_gem"]
            eval_alignment.append({
                "state_key": res["state_key"], "level": int(res["level"]), "sigma": sg,
                "cos_sigma_u_dcomp": float(sg * (u_comp @ d0) / (np.linalg.norm(d0) + 1e-18)),
                "delta_margin_line_minus_gem": float((p1[il].sum() - p1[ig].sum()) - (p0[il].sum() - p0[ig].sum())),
            })
        align_by_bucket = {}
        for sg in (1.0, -1.0):
            sub = [a for a in eval_alignment if a["sigma"] == sg]
            align_by_bucket[f"{sg:+.0f}"] = {
                "n": len(sub),
                "mean_cos": float(np.mean([a["cos_sigma_u_dcomp"] for a in sub])) if sub else float("nan"),
                "frac_cos_pos": float(np.mean([a["cos_sigma_u_dcomp"] > 0 for a in sub])) if sub else float("nan"),
                "mean_delta_margin": float(np.mean([a["delta_margin_line_minus_gem"] for a in sub])) if sub else float("nan"),
                "frac_delta_margin_pos": float(np.mean([a["delta_margin_line_minus_gem"] > 0 for a in sub])) if sub else float("nan"),
            }
        print(f"[seed {seed}] eval sigma*u alignment by bucket: {align_by_bucket}", flush=True)

        rows = []
        for si, res in enumerate(steer_matched):
            level = res["level"]
            bearing = bearing_of(res)
            sigma_bucket = bucket_sign.get(bearing)
            if sigma_bucket is None:
                print(f"[seed {seed}] level={level}: unrecognized bearing {bearing}, skipping", flush=True)
                continue
            if use_bank and res["state_key"] in done_blocks:
                rows.extend(done_blocks[res["state_key"]])
                continue
            state_rows_start = len(rows)
            d_comp0 = steer_d[level]
            f_base = steer_f[level]
            norm0 = float(np.linalg.norm(d_comp0))
            alpha_f = fisher_projection_alpha(u_comp, f_base, d_comp0)

            for beta in betas:
                o_base, _ = hfc.rollout_with_dcomp_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    beta=0.0, sign=1.0, max_steps=args.max_steps, replay_stride=10**9,
                    expect_state_bytes=res["state_bytes"])
                skip = args.skip_full_local
                o_full = None if skip else hfc.rollout_with_dcomp_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    beta=beta, sign=1.0, max_steps=args.max_steps, replay_stride=args.replay_stride,
                    expect_state_bytes=res["state_bytes"])[0]
                # full_local anticorrective, reconstructed here so its replay stride matches this harness.
                o_full_anti = None if skip else hfc.rollout_with_dcomp_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    beta=beta, sign=-1.0, max_steps=args.max_steps, replay_stride=args.replay_stride,
                    expect_state_bytes=res["state_bytes"])[0]

                # full_local_fixed holds beta*d_comp(s0) constant for the whole rollout, like every compressed
                # arm; comparing it with full_local separates "per-step reconstruction" from "compression".
                o_full_fixed, _ = hfc.rollout_with_fixed_push_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    beta * d_comp0, args.max_steps, res["state_bytes"])
                o_full_fixed_anti, _ = hfc.rollout_with_fixed_push_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    -beta * d_comp0, args.max_steps, res["state_bytes"])

                delta_sign_gated = sigma_bucket * beta * norm0 * u_comp
                o_sign_gated, _ = hfc.rollout_with_fixed_push_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    delta_sign_gated, args.max_steps, res["state_bytes"])

                # Sign-reversed sign-gated arm: tests that u_comp is a signed axis, not merely a vector correlated with
                # the corrective direction. Same state/level/beta/bank/hook as sign_gated, negated.
                delta_sign_gated_anti = -sigma_bucket * beta * norm0 * u_comp
                o_sign_gated_anti, _ = hfc.rollout_with_fixed_push_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    delta_sign_gated_anti, args.max_steps, res["state_bytes"])

                delta_fisher = beta * alpha_f * u_comp
                o_fisher, _ = hfc.rollout_with_fixed_push_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    delta_fisher, args.max_steps, res["state_bytes"])

                # Sign-reversed Fisher-projected arm -- same bidirectional-axis
                # rationale as sign_gated_anti above.
                delta_fisher_anti = -beta * alpha_f * u_comp
                o_fisher_anti, _ = hfc.rollout_with_fixed_push_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    delta_fisher_anti, args.max_steps, res["state_bytes"])

                delta_ungated = beta * norm0 * u_comp
                o_ungated, _ = hfc.rollout_with_fixed_push_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    delta_ungated, args.max_steps, res["state_bytes"])

                # Sign-reversed ungated arm -- same bidirectional-axis rationale.
                delta_ungated_anti = -beta * norm0 * u_comp
                o_ungated_anti, _ = hfc.rollout_with_fixed_push_hfc(
                    policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                    delta_ungated_anti, args.max_steps, res["state_bytes"])

                rand_rng = np.random.default_rng(stable_seed(seed, level, beta, "rand"))
                rand_outcomes = []
                for _ in range(args.n_random_dirs):
                    rdir = matched_norm_random_direction(beta * norm0, d_comp0.shape[0], rand_rng)
                    o_rand, _ = hfc.rollout_with_fixed_push_hfc(
                        policy, level, args.distribution_mode, args.use_backgrounds, args.device,
                        rdir, args.max_steps, res["state_bytes"])
                    rand_outcomes.append(o_rand)

                rows.append({
                    "level": int(level), "beta": beta, "bearing": list(bearing), "sigma_bucket": sigma_bucket,
                    "norm0": norm0, "alpha_F": alpha_f,
                    "baseline": o_base, "full_local": o_full, "full_local_anti": o_full_anti,
                    "full_local_fixed": o_full_fixed, "full_local_fixed_anti": o_full_fixed_anti,
                    "sign_gated": o_sign_gated, "sign_gated_anti": o_sign_gated_anti,
                    "fisher_projected": o_fisher, "fisher_projected_anti": o_fisher_anti,
                    "ungated": o_ungated, "ungated_anti": o_ungated_anti,
                    "random": rand_outcomes,
                    "kind": "row", "state_key": res.get("state_key"), "state_hash": res.get("state_hash"),
                    "d": res.get("d"), "is_d1": res.get("is_d1"), "axis_sign": res.get("axis_sign_lineN_gemE"),
                    "tap": "h_fc", "edit": args.edit, "replay_stride": args.replay_stride, "seed": int(seed),
                    "background": bg_name,
                    "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha, "max_steps": args.max_steps,
                })
                print(f"[seed {seed}] {si + 1}/{len(steer_matched)}  level={level} beta={beta} sigma={sigma_bucket:+.0f} "
                      f"alpha_F={alpha_f:+.3f}  baseline={o_base} full_local={o_full} full_local_anti={o_full_anti} "
                      f"sign_gated={o_sign_gated} sign_gated_anti={o_sign_gated_anti} "
                      f"fisher_projected={o_fisher} fisher_projected_anti={o_fisher_anti} "
                      f"ungated={o_ungated} ungated_anti={o_ungated_anti} random={rand_outcomes}", flush=True)
            if use_bank:
                with open(blocks_fp, "a") as f:
                    for r in rows[state_rows_start:]:
                        f.write(json.dumps(r) + "\n")
                    f.write(json.dumps({"kind": "state_done", "state_key": res["state_key"]}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        arms = ("baseline", "full_local", "full_local_anti", "full_local_fixed", "full_local_fixed_anti", "sign_gated", "sign_gated_anti",
                "fisher_projected", "fisher_projected_anti", "ungated", "ungated_anti")
        summary = {}
        for beta in betas:
            sub = [r for r in rows if r["beta"] == beta]
            beta_summary = {a: bucket_outcome_distribution([r[a] for r in sub if r.get(a) is not None], OUTCOME_CATS)
                            for a in arms}
            beta_summary["random"] = bucket_outcome_distribution([o for r in sub for o in r["random"]], OUTCOME_CATS)
            summary[str(beta)] = beta_summary
            print(f"[seed {seed}] beta={beta} SUMMARY: {json.dumps(beta_summary)}", flush=True)

        results[f"seed_{seed}"] = {
            "rows": rows, "summary": summary, "gate": gate, "permutation_null": perm,
            "n_fit_bank": len(axis_levels), "n_holdout": len(holdout_levels),
            "n_steering_bank": len(steer_matched), "n_fit_steer_overlap_excluded": len(overlap),
            "tap": "h_fc", "edit": args.edit, "edit_worlds": {
                "d_comp_plus": edit_worlds.edit_names["plus"], "d_comp_minus": edit_worlds.edit_names["minus"],
                "objects": {k: list(v) for k, v in edit_worlds._objs.items()},
                "d_comp": "h(plus world) - h(minus world), toward the line candidate u"},
            "replay_stride": args.replay_stride, "skip_full_local": args.skip_full_local,
            "max_steps": args.max_steps, "background": bg_name, "checkpoint": str(ckpt),
            "checkpoint_sha256": ckpt_sha, "u_comp": u_comp.tolist(),
            "eval_alignment": eval_alignment, "eval_alignment_by_bucket": align_by_bucket,
            "source_sha256": {Path(m.__file__).name: _file_sha(m.__file__) for m in
                              (sys.modules[__name__], hfc, hfc.base, ras_bank_io, compressed_axis_geometry,
                               sys.modules["gm_geometry.probes.ras_counterfactuals"],
                               sys.modules["gm_geometry.integrations.colour_shape.counterfactual_hfc"],
                               h_fc_steering, seeding)},
        }
        if use_bank:
            ras_bank_io.annotate_rows(rows, bank_steer_res)
            by_lv = {r["level"]: r["state_hash"] for r in bank_fit_res}
            results[f"seed_{seed}"].update({
                "bucket_sign": {str(list(k)): v for k, v in bucket_sign.items()},
                "axis_fit_states": [{"level": lvl, "state_hash": by_lv[lvl]} for lvl in sorted(axis_levels)],
                "holdout_states": [{"level": lvl, "state_hash": by_lv[lvl]} for lvl in sorted(holdout_levels)],
            })
        if use_bank:
            results["canonical_bank_sha256"] = bank_steer_prov.get("canonical_bank_sha256")
            results["split_sha256"] = bank_steer_prov["split_sha256"]
        seeds_tag = "-".join(str(k) for k in ckpts)
        out_name = (f"ras_global_steering_hfc_{edit_tag}{bg_name}_{seeds_tag}_{split_tag}{'_exd1' if args.exclude_d1 else ''}.json"
                    if use_bank else "ras_global_steering_hfc.json")
        (args.out / out_name).write_text(json.dumps(results, indent=2))
        del policy
        torch.cuda.empty_cache()
        print(f"[seed {seed}] done, wrote partial results", flush=True)

    print("all seeds done, wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
