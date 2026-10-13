"""Dump obs, logit Jacobians, probs and activations at h_fc and h_pre for the first K production frames (Maze or Coin) for independent recomputation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from gm_geometry.probes.fisher_sensitivity import block3_logit_jacobian_and_act, fc_logit_jacobian_and_act


def agent_record(policy, obs_a, obs_reloc, device) -> dict:
    j_fc, p, h_fc = fc_logit_jacobian_and_act(policy, obs_a, device)
    _, _, h_fc_r = fc_logit_jacobian_and_act(policy, obs_reloc, device)
    j_pre, _, h_pre = block3_logit_jacobian_and_act(policy, obs_a, device)
    _, _, h_pre_r = block3_logit_jacobian_and_act(policy, obs_reloc, device)
    return {"J_fc": j_fc, "p": p, "h_fc": h_fc, "h_fc_reloc": h_fc_r,
            "J_pre": j_pre.astype(np.float32), "h_pre": h_pre, "h_pre_reloc": h_pre_r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["maze", "coin"], required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    ck = args.ckpt_dir

    if args.env == "maze":
        from run_maze_fisher_decision_fork import harvest_decision_fork
        from run_maze_i_goal_proxy_ladder import load_policy

        paths = {"gm_1080": ck / "1080_gm.pth", "gm_1081": ck / "1081_gm.pth", "gm_1082": ck / "1082_gm.pth",
                 "nongm_1090": ck / "1090_teach.pth", "nongm_1091": ck / "1091_teach.pth",
                 "nongm_1092": ck / "1092_teach.pth"}
        frames, _ = harvest_decision_fork(load_policy(paths["gm_1080"], device), device=device, n_frames=args.k,
                                          start_level=1_000_000, max_levels=8000, max_steps=200,
                                          source_name="gm_rollouts")
        agents = {lab: load_policy(p, device) for lab, p in paths.items()}
        for i, seed in enumerate((0, 1, 2)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            agents[f"init_{i}"] = load_policy(paths["nongm_1090"], device, random_init=True)
        act_keys = ("a_cheese", "a_corner")
    else:
        from run_coin_fisher_decision_window import harvest_decision_windows
        from run_coin_fisher_sensitivity import load_init_policy, load_policy

        ck_nongm = [ck / f"{s}_teach.pth" for s in (7001, 7002, 7003)]
        ck_gm = [ck / "6033_gm_20M.pth", ck / "6037_gm_120M.pth", ck / "6051_gm_55M.pth"]
        frames = harvest_decision_windows(n_frames=args.k, start_level=1_100_000, max_levels=8000,
                                          warm_policy=load_policy(ck_gm[0], device), device=device)
        agents = {"nongm_7001": load_policy(ck_nongm[0], device), "nongm_7002": load_policy(ck_nongm[1], device),
                  "nongm_7003": load_policy(ck_nongm[2], device), "gm_6033": load_policy(ck_gm[0], device),
                  "gm_6037": load_policy(ck_gm[1], device), "gm_6051": load_policy(ck_gm[2], device)}
        for i in range(3):
            agents[f"init_{i}"] = load_init_policy(ck_nongm[0], device, init_seed=4000 + i)
        act_keys = ("a_coin",)

    args.out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for fi, fr in enumerate(frames[: args.k]):
        arrays[f"f{fi}/obs_a"] = np.asarray(fr["obs_a"], dtype=np.float32)
        arrays[f"f{fi}/obs_reloc"] = np.asarray(fr["obs_reloc"], dtype=np.float32)
        arrays[f"f{fi}/level"] = np.asarray(int(fr["level"]))
        for k in act_keys:
            arrays[f"f{fi}/{k}"] = np.asarray(int(fr[k]))
        for lab, pol in agents.items():
            for name, v in agent_record(pol, fr["obs_a"], fr["obs_reloc"], device).items():
                arrays[f"f{fi}/{lab}/{name}"] = np.asarray(v)
    dst = args.out / f"geom_{args.env}_first{args.k}.npz"
    np.savez_compressed(dst, **arrays)
    print("wrote", dst, os.path.getsize(dst), "bytes", flush=True)


if __name__ == "__main__":
    main()
