"""Coin Choice/Mass on A_star=LEFT vs A_p=RIGHT for every checkpoint on the frozen canonical 600-state conflict bank."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from build_coin_canonical_bank import state_geometry
from gm_geometry.probes.coin_canonical_bank import ALL_LABELS, GM_LABELS, NONGM_LABELS, load_bank
from gm_geometry.probes.coin_location import LEFT_ACTIONS, RIGHT_ACTIONS
from gm_geometry.probes.coin_rollout import harvest_with_state_snapshot, load_policy

CKPT_FILE = {
    "gm_6033": "6033_gm_20M.pth", "gm_6037": "6037_gm_120M.pth", "gm_6051": "6051_gm_55M.pth",
    "nongm_7001": "7001_teach.pth", "nongm_7002": "7002_teach.pth", "nongm_7003": "7003_teach.pth",
}
WARM = {"gm_rollouts": "gm_6033", "nongm_rollouts": "nongm_7001"}
LEFT_IDS = sorted(LEFT_ACTIONS)
RIGHT_IDS = sorted(RIGHT_ACTIONS)
T_CRIT_DF2 = 4.302652729911275


@torch.no_grad()
def probs_of(policy, obs_chw: np.ndarray, device: str) -> np.ndarray:
    x = torch.as_tensor(obs_chw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    h = torch.zeros(1, policy.embedder.output_dim, device=device)
    dist, _v, _h = policy(x, h, torch.zeros(1, device=device))
    return dist.probs[0].detach().cpu().numpy().astype(np.float64)


def set_stats(p: np.ndarray) -> dict:
    a = int(np.argmax(p))
    return {
        "argmax": a,
        "choice_star": float(a in LEFT_IDS),
        "choice_p": float(a in RIGHT_IDS),
        "mass_star": float(p[LEFT_IDS].sum()),
        "mass_p": float(p[RIGHT_IDS].sum()),
    }


def cohort_summary(per_ckpt: dict, labels: list[str], key: str) -> dict:
    v = np.asarray([per_ckpt[l][key] for l in labels])
    sd = float(v.std(ddof=1))
    half = T_CRIT_DF2 * sd / np.sqrt(len(v))
    return {"mean": float(v.mean()), "sd": sd, "ci95_lo": float(v.mean() - half),
            "ci95_hi": float(v.mean() + half), "per_seed": v.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=0, help="states per source, 0 = all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    bank = load_bank(args.bank)
    states = bank["states"]
    print(f"bank freeze {bank['freeze_sha256'][:16]} n_states={len(states)}", flush=True)
    pols = {}
    for lab in ALL_LABELS:
        p = args.ckpt_dir / CKPT_FILE[lab]
        if not p.is_file():
            raise SystemExit(f"missing checkpoint {lab}: {p}")
        pols[lab] = load_policy(p, device)

    # Frames are rebuilt by replaying the deterministic harvest; each state is checked against the bank's
    # goal/shortcut cells and each argmax against the frozen a_goal/a_shortcut, so a drifted replay is caught.
    rows: list[dict] = []
    n_geo_mismatch = 0
    for src, wlab in WARM.items():
        sel = [s for s in states if s["source"] == src]
        if args.cap:
            sel = sel[: args.cap]
        want = {s["level"] for s in sel}
        frames = harvest_with_state_snapshot(
            n_frames=len(want), start_level=min(want), max_levels=60_000,
            warm_policy=pols[wlab], device=device, skip_policy_gate=True)
        by_level = {int(f["level"]): f for f in frames if int(f["level"]) in want}
        missing = want - set(by_level)
        print(f"{src}: recovered {len(by_level)}/{len(want)}", flush=True)
        if missing:
            raise SystemExit(f"{src}: harvest replay missing {len(missing)} levels")

        for s in sel:
            geo = state_geometry(by_level[s["level"]]["state_reloc"])
            if not geo["ok"] or geo["goal_xy"] != s["goal_xy"] or geo["shortcut_xy"] != s["shortcut_xy"]:
                n_geo_mismatch += 1
                print(f"  GEOMETRY MISMATCH {src}/{s['level']}: {geo.get('reject')}", flush=True)
                continue
            for lab, pol in pols.items():
                pg = probs_of(pol, geo["obs_goal"], device)
                ps = probs_of(pol, geo["obs_shortcut"], device)
                g, sc = set_stats(pg), set_stats(ps)
                frozen = s["per_ckpt"][lab]
                rows.append({
                    "source": src, "level": s["level"], "ckpt": lab,
                    **{f"goal_{k}": v for k, v in g.items()},
                    **{f"shortcut_{k}": v for k, v in sc.items()},
                    "frozen_a_goal": frozen["a_goal"], "frozen_a_shortcut": frozen["a_shortcut"],
                    "argmax_matches_frozen": g["argmax"] == frozen["a_goal"] and sc["argmax"] == frozen["a_shortcut"],
                    "policy_actions_differ": frozen["policy_actions_differ"],
                    "probs_goal": pg.tolist(), "probs_shortcut": ps.tolist(),
                })

    for lab in ALL_LABELS:
        fp = args.out / f"choice_mass_{lab}.csv"
        lab_rows = [r for r in rows if r["ckpt"] == lab]
        with fp.open("w", newline="") as f:
            cols = [k for k in lab_rows[0] if not k.startswith("probs_")]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(lab_rows)

    keys = ["choice_star", "choice_p", "mass_star", "mass_p"]
    per_ckpt = {}
    for lab in ALL_LABELS:
        lr = [r for r in rows if r["ckpt"] == lab]
        per_ckpt[lab] = {
            "n": len(lr),
            **{k: float(np.mean([r[f"goal_{k}"] for r in lr])) for k in keys},
            **{f"shortcut_frame_{k}": float(np.mean([r[f"shortcut_{k}"] for r in lr])) for k in keys},
            "argmax_match_frac": float(np.mean([r["argmax_matches_frozen"] for r in lr])),
            "response_rate": float(np.mean([r["policy_actions_differ"] for r in lr])),
        }
    cohorts = {
        name: {k: cohort_summary(per_ckpt, labs, k)
               for k in keys + [f"shortcut_frame_{k}" for k in keys]}
        for name, labs in (("gm", GM_LABELS), ("nongm", NONGM_LABELS))
    }
    payload = {
        "metric": "choice_mass_left_vs_right_at_obs_goal",
        "A_star": LEFT_IDS, "A_p": RIGHT_IDS,
        "bank_freeze_sha256": bank["freeze_sha256"],
        "n_geometry_mismatch": n_geo_mismatch,
        "cap": args.cap or None,
        "cohort_aggregation": "per-checkpoint mean over states, then mean/SD(ddof=1)/t95(df=2) over 3 checkpoints",
        "per_ckpt": per_ckpt,
        "cohorts": cohorts,
        "rows": rows,
    }
    fp = args.out / "coin_choice_mass_canon.json"
    fp.write_text(json.dumps(payload))
    print(json.dumps({"per_ckpt": per_ckpt, "cohorts": {c: {k: {kk: round(vv, 4) for kk, vv in d.items() if kk != 'per_seed'} for k, d in v.items()} for c, v in cohorts.items()}, "n_geometry_mismatch": n_geo_mismatch}, indent=1), flush=True)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
