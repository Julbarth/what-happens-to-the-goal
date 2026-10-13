#!/usr/bin/env python3
"""Attribute Maze cue competition (S_fav / S_comp / dC_log) under swap or ablation edits on the canonical conflict bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tnf
import yaml

ROOT = Path(__file__).resolve().parents[2]
CS_TRAIN = ROOT / "external" / "colour-shape-goal-misgeneralization" / "train-procgen"
for p in (
    ROOT / "src",
    CS_TRAIN,
    ROOT / "external" / "colour-shape-goal-misgeneralization" / "procgenAISC",
):
    if p.exists():
        sys.path.insert(0, str(p))

from gm_geometry.probes import ras_bank_io  # noqa: E402

# S_g = log P(A_g|s~e)/P(A_0|s~e) - log P(A_g|s)/P(A_0|s) for g in {fav, comp}, dC_log = S_fav - S_comp, with group
# probabilities summed over member actions and A_0 the remaining actions, so any state-wise logit shift cancels.
# The agent spawns at (0, 0), so every conflict is UP (5) vs RIGHT (6, 7, 8); "line"/"gem" name the base cells u/v.

OBJ_LINE = "red_line_diag"
OBJ_GEM = "yellow_gem"
EDIT_NAMES = {"swap": {"shape": "shape_swap", "colour": "colour_swap"},
              "ablate": {"shape": "shape_ablate", "colour": "colour_ablate"}}
# Favoured cell per edit ("line" = base line cell u / obj1 slot, "gem" = base gem cell v / obj2 slot).
# swap:   shape_swap puts the yellow line at v (both cues agree on v); colour_swap puts it at u.
# ablate: shape_ablate removes u's line shape (v keeps yellow); colour_ablate removes v's yellow.
# Both sets therefore favour v under the shape edit and u under the colour edit.
EDIT_FAVOURS = {"shape": ("gem", "line"), "colour": ("line", "gem")}
TARGET_ROLE = {"line": "intended", "gem": "proxy"}
N_ACTIONS = 15
UP_IDS, RIGHT_IDS = [5], [6, 7, 8]
EPS = 1e-12
ASSAY_VERSION = "ras_cue_competition_canon_v2_swap_20260924"


def edit_objs(edit_set: str) -> dict[str, tuple[str, str]]:
    """{"shape": (obj1, obj2), "colour": (obj1, obj2)} from the canonical counterfactual module."""
    from gm_geometry.probes import ras_counterfactuals as rcf
    assert (rcf.OBJ_LINE, rcf.OBJ_GEM) == (OBJ_LINE, OBJ_GEM)
    return {e: rcf.edit_objects(name) for e, name in EDIT_NAMES[edit_set].items()}


def favoured_slot_swap(objs: tuple[str, str]) -> str:
    """For a swap world, the slot holding the yellow line (both cues agree there)."""
    return {0: "line", 1: "gem"}[objs.index("yellow_line_diag")]


def make_env(obj1: str, obj2: str, level: int, *, rand_seed: int = 0,
             distribution_mode: str = "easy", use_backgrounds: bool = False, world_dim: int = 5):
    from procgen import ProcgenEnv
    return ProcgenEnv(
        num_envs=1, env_name="maze_colorsobjects_duel", num_levels=1,
        start_level=int(level), rand_seed=int(rand_seed),
        distribution_mode=distribution_mode, world_dim=world_dim,
        obj1=obj1, obj2=obj2, use_backgrounds=use_backgrounds,
    )


def to_nchw(obs) -> np.ndarray:
    rgb = obs["rgb"][0] if isinstance(obs, dict) else obs[0]
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def action_sets_from_record(rec: dict) -> tuple[list[int], list[int]]:
    """(ids_line, ids_gem) from a canonical bank record's A_line / A_gem, with the spawn-geometry
    invariant enforced: sign +1 (line N, gem E) -> ([5], [6,7,8]); sign -1 -> ([6,7,8], [5])."""
    ids_line, ids_gem = sorted(int(a) for a in rec["A_line"]), sorted(int(a) for a in rec["A_gem"])
    if not ids_line or not ids_gem or set(ids_line) & set(ids_gem):
        raise ValueError(f"state {rec.get('state_key')}: A_line {ids_line} / A_gem {ids_gem} not a strict conflict")
    sign = rec["axis_sign_lineN_gemE"]
    want = {1: (UP_IDS, RIGHT_IDS), -1: (RIGHT_IDS, UP_IDS)}.get(sign)
    if want is None or (ids_line, ids_gem) != want:
        raise ValueError(f"state {rec.get('state_key')}: sign {sign} but A_line {ids_line} / A_gem {ids_gem} "
                         f"-- bank geometry and assay disagree, stop")
    return ids_line, ids_gem


def edit_action_sets(ids_line: list[int], ids_gem: list[int], edit: str) -> dict:
    """Favoured / competitor / neutral action ids for one edit (disjoint partition of all 15 actions)."""
    fav_t, comp_t = EDIT_FAVOURS[edit]
    by_target = {"line": list(ids_line), "gem": list(ids_gem)}
    fav, comp = by_target[fav_t], by_target[comp_t]
    neutral = [a for a in range(N_ACTIONS) if a not in fav and a not in comp]
    return {"fav_target": fav_t, "comp_target": comp_t, "ids_fav": fav, "ids_comp": comp, "ids_neutral": neutral}


def logit_stats_for_state(probs_base: np.ndarray, probs_edit: np.ndarray, sets: dict) -> dict:
    """Group masses, log-group deltas and S_fav / S_comp / dC_log for one state and one edit."""
    def mass(p, ids):
        return float(np.sum(np.asarray(p, dtype=np.float64)[ids]))

    out = {}
    for tag, p in (("before", probs_base), ("after", probs_edit)):
        for g in ("fav", "comp", "neutral"):
            out[f"P_{g}_{tag}"] = mass(p, sets[f"ids_{g}"])
    dL = {g: np.log(max(out[f"P_{g}_after"], EPS)) - np.log(max(out[f"P_{g}_before"], EPS))
          for g in ("fav", "comp", "neutral")}
    out.update({f"dL_{g}": float(v) for g, v in dL.items()})
    out["S_fav"] = float(dL["fav"] - dL["neutral"])
    out["S_comp"] = float(dL["comp"] - dL["neutral"])
    out["dC_log"] = float(dL["fav"] - dL["comp"])
    out["dP_fav"] = out["P_fav_after"] - out["P_fav_before"]
    out["dP_comp"] = out["P_comp_after"] - out["P_comp_before"]
    out["min_group_mass"] = min(out[k] for k in out if k.startswith("P_"))
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(ckpt: Path, device: str):
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy

    cfg = CS_TRAIN / "hyperparams" / "procgen" / "config.yml"
    if not cfg.exists():
        raise SystemExit(f"missing {cfg}; run scripts/setup_external.py first")
    recurrent = yaml.safe_load(cfg.read_text())["hard-500"]["recurrent"]
    policy = CategoricalPolicy(ImpalaModel(in_channels=3), recurrent, N_ACTIONS)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    return policy.to(device).eval()


@torch.no_grad()
def policy_probs_logits(policy, obs_batch: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    x = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    h0 = torch.zeros(x.shape[0], policy.embedder.output_dim, device=device)
    m = torch.zeros(x.shape[0], device=device)
    dist, _v, _nh = policy(x, h0, m)
    logits = dist.logits.double()
    return tnf.softmax(logits, dim=-1).cpu().numpy(), logits.cpu().numpy()


def render_edit_worlds(u: dict, m: dict, args) -> dict:
    """Base obs (bank-verified) plus shape/colour edit obs for one state, with a runtime G6 check:
    the edit world must have the record's agent, line cell, gem cell and walls."""
    from gm_geometry.probes.ras_gt_bank import state_record

    rec, level = u["record"], int(u["level"])
    bg = "texture" if args.use_backgrounds else "black"
    obs = {"base": to_nchw(m["obs0"])}
    diag = {}
    for edit, (o1, o2) in edit_objs(args.edit_set).items():
        env = make_env(o1, o2, level, distribution_mode=args.distribution_mode, use_backgrounds=args.use_backgrounds)
        try:
            ob = env.reset()
            er = state_record(ras_bank_io.env_raw_state(env), level, bg)
        finally:
            env.close()
        got = (list(er["agent"]), list(er["line"]), list(er["gem"]), sorted(map(tuple, er["walls"])))
        want = (list(rec["agent"]), list(rec["line"]), list(rec["gem"]), sorted(map(tuple, rec["walls"])))
        if got != want:
            raise RuntimeError(f"G6 level {level} {edit}: edit world (agent, line, gem, walls) {got} != bank {want}")
        obs[edit] = to_nchw(ob)
        diff = np.abs(obs[edit] - obs["base"]).sum(axis=0) > 1e-6
        diag[f"n_px_changed_{edit}"] = int(diff.sum())
    return {"obs": obs, "diag": diag}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--distribution-mode", default="easy")
    ap.add_argument("--use-backgrounds", action="store_true", default=False)
    ap.add_argument("--gm-seeds", type=str, required=True)
    ap.add_argument("--deco-seeds", type=str, default="")
    ap.add_argument("--gm-suffix", default="_gm.pth")
    ap.add_argument("--deco-suffix", default="_teach.pth")
    ap.add_argument("--all-roles", action="store_true",
                    help="score fit + eval (the whole bank); rows carry split_role. This assay fits nothing.")
    ap.add_argument("--max-px-changed", type=int, default=400,
                    help="abort if an edit changes more pixels than this (edit must only repaint one object)")
    ap.add_argument("--edit-set", choices=["swap", "ablate"], default="swap",
                    help="swap = canonical A.2.2 counterfactuals (primary); ablate = supplementary deletion assay")
    ap.add_argument("--device", default="cpu")
    ras_bank_io.add_bank_args(ap)
    args = ap.parse_args()
    if args.allow_legacy_harvest:
        ap.error("the legacy harvest / --fresh-calib path was removed (y-up vs image flip scored North with DOWN)")
    ras_bank_io.check_bank_args(args)
    if args.edit_set == "swap":
        for e, objs in edit_objs("swap").items():
            assert favoured_slot_swap(objs) == EDIT_FAVOURS[e][0], f"{e}: favoured slot convention broken"
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bg = "texture" if args.use_backgrounds else "black"

    roles = ["fit", "eval"] if args.all_roles else [args.split_role]
    states, provs = [], {}
    for role in roles:
        res, prov, us = ras_bank_io.bank_res_list(
            args, role, state_bytes=ras_bank_io.env_raw_state, use_backgrounds=args.use_backgrounds,
            make_base_env=lambda lvl: make_env(OBJ_LINE, OBJ_GEM, lvl, distribution_mode=args.distribution_mode,
                                               use_backgrounds=args.use_backgrounds))
        provs[role] = prov
        for u, m in zip(us, res):
            assert u["state_key"] == m["state_key"] and int(u["level"]) == int(m["level"])
            ids_line, ids_gem = action_sets_from_record(u["record"])
            if (ids_line, ids_gem) != (sorted(m["first_actions_line"]), sorted(m["first_actions_gem"])):
                raise RuntimeError(f"state {u['state_hash']}: materialized action sets differ from record")
            rend = render_edit_worlds(u, m, args)
            for edit in EDIT_FAVOURS:
                if rend["diag"][f"n_px_changed_{edit}"] > args.max_px_changed:
                    raise RuntimeError(f"state {u['state_hash']} {edit}: {rend['diag']} pixels changed "
                                       f"(> {args.max_px_changed}); edit world is not a one-object repaint")
                if rend["diag"][f"n_px_changed_{edit}"] == 0:
                    raise RuntimeError(f"state {u['state_hash']} {edit}: edit changed no pixels")
            states.append({"u": u, "m": m, "role": role, "ids_line": ids_line, "ids_gem": ids_gem, **rend})
    keys = [s["u"]["state_key"] for s in states]
    assert len(set(keys)) == len(keys), "duplicate state_key across roles"
    print(f"[canon-cue] bg={bg} states={len(states)} roles={roles} "
          f"px_changed shape max={max(s['diag']['n_px_changed_shape'] for s in states)} "
          f"colour max={max(s['diag']['n_px_changed_colour'] for s in states)} ({time.time() - t0:.0f}s)", flush=True)

    obs_all = {k: np.stack([s["obs"][k] for s in states]) for k in ("base", "shape", "colour")}
    obs_sha = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in obs_all.items()}

    policies = [("gm", int(s), args.ckpt_dir / f"{int(s)}{args.gm_suffix}") for s in args.gm_seeds.replace(",", " ").split()]
    policies += [("deco", int(s), args.ckpt_dir / f"{int(s)}{args.deco_suffix}")
                 for s in args.deco_seeds.replace(",", " ").split()]
    results, ckpts, gauge = {}, {}, {}
    for cohort, seed, ck in policies:
        if not ck.is_file():
            raise SystemExit(f"missing {ck}")
        label = f"{cohort}_{seed}"
        ckpts[label] = {"path": str(ck), "sha256": sha256_file(ck)}
        policy = load_policy(ck, args.device)
        probs, logits = {}, {}
        for k, v in obs_all.items():
            probs[k], logits[k] = policy_probs_logits(policy, v, args.device)
        # gauge: grouping raw logits by logsumexp must equal log of summed softmax mass
        lse = np.logaddexp.reduce(logits["base"][:, RIGHT_IDS], axis=1) - np.logaddexp.reduce(logits["base"], axis=1)
        gauge[label] = float(np.max(np.abs(lse - np.log(probs["base"][:, RIGHT_IDS].sum(axis=1)))))
        rows = []
        for i, s in enumerate(states):
            u, m = s["u"], s["m"]
            for edit in EDIT_FAVOURS:
                sets = edit_action_sets(s["ids_line"], s["ids_gem"], edit)
                st = logit_stats_for_state(probs["base"][i], probs[edit][i], sets)
                rows.append({
                    "label": label, "cohort": cohort, "seed": seed, "background": bg, "edit": edit,
                    "edit_set": args.edit_set, "edit_name": EDIT_NAMES[args.edit_set][edit],
                    "state_key": u["state_key"], "state_hash": u["state_hash"], "level": int(u["level"]),
                    "n_dup_levels": len(u["levels"]), "d": int(m["d"]), "is_d1": bool(m["is_d1"]),
                    "axis_sign": int(u["record"]["axis_sign_lineN_gemE"]), "split_role": s["role"],
                    "state_sha256": m["state_sha256"],
                    "fav_target": sets["fav_target"], "comp_target": sets["comp_target"],
                    "fav_role": TARGET_ROLE[sets["fav_target"]], "comp_role": TARGET_ROLE[sets["comp_target"]],
                    "ids_fav": sets["ids_fav"], "ids_comp": sets["ids_comp"], "ids_neutral": sets["ids_neutral"],
                    **st, **s["diag"],
                    "probs_base": [float(x) for x in probs["base"][i]],
                    "probs_edit": [float(x) for x in probs[edit][i]],
                })
        results[label] = rows
        csv_path = args.out / f"rows_{args.edit_set}_{bg}_{label}.csv"
        flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
        msg = " ".join(f"{e}:S_fav={np.mean([r['S_fav'] for r in rows if r['edit'] == e]):+.3f},"
                       f"S_comp={np.mean([r['S_comp'] for r in rows if r['edit'] == e]):+.3f}" for e in EDIT_FAVOURS)
        print(f"[canon-cue] scored {label} bg={bg} n_rows={len(rows)} {msg} gauge_maxdiff={gauge[label]:.2e}", flush=True)

    src_files = [Path(__file__).resolve(), Path(ras_bank_io.__file__).resolve()]
    from gm_geometry.probes import ras_gt_bank
    src_files.append(Path(ras_gt_bank.__file__).resolve())
    from gm_geometry.probes import ras_counterfactuals
    src_files.append(Path(ras_counterfactuals.__file__).resolve())
    out = {
        "assay_version": ASSAY_VERSION, "background": bg, "use_backgrounds": args.use_backgrounds,
        "distribution_mode": args.distribution_mode, "roles": roles, "n_states": len(states),
        "canonical_bank_sha256": provs[roles[0]].get("canonical_bank_sha256"),
        "bank_provenance": provs, "checkpoints": ckpts, "obs_batch_sha256": obs_sha,
        "gauge_check_max_abs_diff": gauge,
        "source_files": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256_file(p)
                         for p in src_files},
        "edit_set": args.edit_set, "edit_names": EDIT_NAMES[args.edit_set],
        "edit_objects": {e: {"obj1": o[0], "obj2": o[1]} for e, o in edit_objs(args.edit_set).items()},
        "favoured_convention": {e: {"favoured": f, "competitor": c} for e, (f, c) in EDIT_FAVOURS.items()},
        "target_roles": TARGET_ROLE,
        "argv": sys.argv, "elapsed_s": time.time() - t0,
        "results": results,
    }
    path = args.out / f"ras_cue_competition_canon_{args.edit_set}_{bg}.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path, f"sha256={sha256_file(path)}", flush=True)


if __name__ == "__main__":
    main()
