"""Coin world-edit behavioural dependence (KL, SF, Reach, paired retargeting) on the frozen canonical bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gm_geometry.probes.coin_canonical_bank import ALL_LABELS, GM_LABELS, NONGM_LABELS, load_bank
from gm_geometry.probes.coin_location import (
    LEFT_ACTIONS,
    RIGHT_ACTIONS,
    get_info,
    labels_from_info,
    obs_nchw_after_set_coin,
    set_coin,
    unwrap_base_env,
)
from gm_geometry.probes.coin_rollout import harvest_with_state_snapshot, load_policy, make_env
from gm_geometry.probes.fisher_sensitivity import kl_categorical

GM_CKPTS = ["6033_gm_20M.pth", "6037_gm_120M.pth", "6051_gm_55M.pth"]
NONGM_CKPTS = ["7001_teach.pth", "7002_teach.pth", "7003_teach.pth"]
# SF uses LEFT_ACTIONS as the progress set because the bank places the relocated coin strictly left of the agent
# and the shortcut coin strictly right. Retargeting = closest_approach(A) - closest_approach(B) to the relocated coin.


def probs(policy, obs_chw, device: str) -> np.ndarray:
    x = torch.as_tensor(obs_chw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    h = torch.zeros(1, policy.embedder.output_dim, device=device)
    with torch.no_grad():
        dist, _v, _h = policy(x, h, torch.zeros(1, device=device))
    return dist.probs.detach().cpu().numpy().reshape(-1)


def mass(p: np.ndarray, actions) -> float:
    return float(np.asarray(p, dtype=np.float64).reshape(-1)[list(actions)].sum())


def greedy(policy, obs_chw, device: str) -> int:
    return int(np.argmax(probs(policy, obs_chw, device)))


def rollout_branch(policy, device, env, obs, max_steps: int) -> dict:
    """Greedy rollout recording agent positions, coin reach and the fraction of rightward actions."""
    positions, actions = [], []
    coin_reach = False
    for _t in range(max_steps):
        a = greedy(policy, obs, device)
        actions.append(a)
        lab = labels_from_info(get_info(env))
        if lab is not None:
            positions.append([lab["agent_x"], lab["agent_y"]])
        obs_raw, rew, done, _i = env.step(np.array([a]))
        obs = np.asarray(obs_raw[0], dtype=np.float32)
        if float(np.asarray(rew).reshape(-1)[0]) > 0:
            coin_reach = True
        if done[0]:
            break
    right_frac = float(np.mean([1.0 if a in RIGHT_ACTIONS else 0.0 for a in actions])) if actions else None
    return {"coin_reach": coin_reach, "positions": positions,
            "n_steps": len(actions), "right_frac": right_frac}


def closest_approach(positions: list, target_xy) -> float | None:
    if not positions:
        return None
    pos = np.asarray(positions, dtype=np.float64)
    t = np.asarray(target_xy, dtype=np.float64)
    return float(np.min(np.linalg.norm(pos - t, axis=1)))


def score_state(policy, state_bytes: bytes, goal_xy, shortcut_xy, device: str,
                max_steps: int) -> dict:
    """KL/SF from the two placements, then paired rollouts for Reach/retargeting."""
    out: dict = {"ok": False}
    env = make_env("coinrun_aisc", num_levels=1, start_level=0)
    try:
        env.reset()
        base = unwrap_base_env(env)
        base.set_state(state_bytes)
        snap = base.get_state()

        # pi(.|s): coin at the ORIGINAL right-hand cell.
        if set_coin(env, shortcut_xy[0], shortcut_xy[1], 1) != 0:
            return out
        obs_a = obs_nchw_after_set_coin(env)
        if obs_a.ndim == 4:
            obs_a = obs_a[0]
        pa = probs(policy, obs_a, device)
        base.set_state(snap)

        # pi(.|s_e): coin RELOCATED to the left-hand cell.
        if set_coin(env, goal_xy[0], goal_xy[1], 1) != 0:
            return out
        obs_b = obs_nchw_after_set_coin(env)
        if obs_b.ndim == 4:
            obs_b = obs_b[0]
        pb = probs(policy, obs_b, device)
        base.set_state(snap)

        kl = float(kl_categorical(pa, pb))
        sf = mass(pb, LEFT_ACTIONS) - mass(pa, LEFT_ACTIONS)
        a_goal, a_sc = int(np.argmax(pb)), int(np.argmax(pa))

        # Branch A: unperturbed (coin left where the harvest put it originally).
        base.set_state(snap)
        set_coin(env, shortcut_xy[0], shortcut_xy[1], 1)
        obs_a0 = obs_nchw_after_set_coin(env)
        if obs_a0.ndim == 4:
            obs_a0 = obs_a0[0]
        br_a = rollout_branch(policy, device, env, obs_a0, max_steps)

        # Branch B: coin relocated. Fresh env so A's steps cannot leak into B.
        env.close()
        env = make_env("coinrun_aisc", num_levels=1, start_level=0)
        env.reset()
        base = unwrap_base_env(env)
        base.set_state(state_bytes)
        set_coin(env, goal_xy[0], goal_xy[1], 1)
        obs_b0 = obs_nchw_after_set_coin(env)
        if obs_b0.ndim == 4:
            obs_b0 = obs_b0[0]
        br_b = rollout_branch(policy, device, env, obs_b0, max_steps)

        ca = closest_approach(br_a["positions"], goal_xy)
        cb = closest_approach(br_b["positions"], goal_xy)
        out = {
            "ok": True,
            "kl": kl,
            "sf": sf,
            "reach": bool(br_b["coin_reach"]),
            "reach_unperturbed": bool(br_a["coin_reach"]),
            "retargeting": (ca - cb) if (ca is not None and cb is not None) else None,
            "closest_a": ca,
            "closest_b": cb,
            "policy_actions_differ": a_goal != a_sc,
            "right_frac_b": br_b["right_frac"],
        }
    finally:
        env.close()
    return out


def agg(vals: list) -> dict:
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "n": 0}
    a = np.asarray(v, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--max-levels", type=int, default=40_000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap states per source")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    bank = load_bank(args.bank)
    print(f"bank verified: {len(bank['states'])} states, freeze {bank['freeze_sha256'][:16]}", flush=True)

    paths = {l: args.ckpt_dir / f for l, f in zip(ALL_LABELS, GM_CKPTS + NONGM_CKPTS)}
    for p in paths.values():
        if not p.is_file():
            raise SystemExit(f"missing ckpt {p}")
    policies = {l: load_policy(p, device) for l, p in paths.items()}

    by_key = {(s["source"], s["level"]): s for s in bank["states"]}
    sources = {"gm_rollouts": (GM_LABELS[0], 1_100_000), "nongm_rollouts": (NONGM_LABELS[0], 1_120_000)}

    # The bank stores geometry but not env state bytes, so states are recovered by replaying the same
    # deterministic harvest and matching on (source, level).
    recovered: dict[tuple[str, int], bytes] = {}
    geom_mismatch = []
    for src, (warm_lbl, start) in sources.items():
        want = sum(1 for k in by_key if k[0] == src)
        print(f"=== re-harvest {src} (warm={warm_lbl}) to recover {want} states ===", flush=True)
        frames = harvest_with_state_snapshot(
            n_frames=want, start_level=start, max_levels=args.max_levels,
            warm_policy=policies[warm_lbl], device=device, skip_policy_gate=True)
        for fr in frames:
            key = (src, int(fr["level"]))
            if key in by_key:
                recovered[key] = fr["state_reloc"]
    print(f"recovered {len(recovered)}/{len(by_key)} bank states", flush=True)

    rows = []
    keys = sorted(recovered)
    if args.limit:
        keys = [k for k in keys if k[0] == "gm_rollouts"][:args.limit] + \
               [k for k in keys if k[0] == "nongm_rollouts"][:args.limit]
    for n, key in enumerate(keys):
        st = by_key[key]
        for lbl, pol in policies.items():
            r = score_state(pol, recovered[key], st["goal_xy"], st["shortcut_xy"],
                            device, args.max_steps)
            if not r["ok"]:
                continue
            r.update({"source": key[0], "level": key[1], "ckpt": lbl,
                      "bank_policy_actions_differ": st["per_ckpt"][lbl]["policy_actions_differ"]})
            rows.append(r)
        if (n + 1) % 25 == 0:
            print(f"  scored {n+1}/{len(keys)} states", flush=True)

    # Per-checkpoint, full bank and policy-resolving subset.
    per_ckpt = {}
    for lbl in ALL_LABELS:
        mine = [r for r in rows if r["ckpt"] == lbl]
        sub = [r for r in mine if r["policy_actions_differ"]]
        per_ckpt[lbl] = {
            "n_states": len(mine),
            "n_resolving": len(sub),
            "full": {"kl": agg([r["kl"] for r in mine]), "sf": agg([r["sf"] for r in mine]),
                     "reach": agg([1.0 if r["reach"] else 0.0 for r in mine]),
                     "retargeting": agg([r["retargeting"] for r in mine])},
            "resolving": {"kl": agg([r["kl"] for r in sub]), "sf": agg([r["sf"] for r in sub]),
                          "reach": agg([1.0 if r["reach"] else 0.0 for r in sub]),
                          "retargeting": agg([r["retargeting"] for r in sub])},
        }

    # Cohort rows: mean +- sample SD across the three seeds.
    def cohort(labels, subset):
        out = {}
        for m in ("kl", "sf", "reach", "retargeting"):
            seed_means = [per_ckpt[l][subset][m]["mean"] for l in labels
                          if per_ckpt[l][subset][m]["mean"] is not None]
            a = np.asarray(seed_means, dtype=np.float64)
            out[m] = {"mean": float(a.mean()) if len(a) else None,
                      "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
                      "n_seeds": int(len(a)), "per_seed": seed_means}
        return out

    summary = {
        "bank": {"path": str(args.bank), "freeze_sha256": bank["freeze_sha256"],
                 "n_states": len(bank["states"])},
        "n_rows": len(rows),
        "per_ckpt": per_ckpt,
        "cohort_full_bank": {"GM": cohort(GM_LABELS, "full"), "NonGM": cohort(NONGM_LABELS, "full")},
        "cohort_resolving": {"GM": cohort(GM_LABELS, "resolving"),
                             "NonGM": cohort(NONGM_LABELS, "resolving")},
        "geom_mismatch": geom_mismatch,
    }
    (args.out / "coin_worldedit_canon.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary["cohort_full_bank"], indent=1), flush=True)
    print("\n=== policy-resolving subset ===", flush=True)
    print(json.dumps(summary["cohort_resolving"], indent=1), flush=True)


if __name__ == "__main__":
    main()
