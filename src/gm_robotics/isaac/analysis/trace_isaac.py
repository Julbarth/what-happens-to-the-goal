"""Closed-loop paths with and without the tower, plus per-step policy means from synchronised edit copies."""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=["main_gm_s0", "main_gm_s1", "main_gm_s2",
                                               "main_control_s0", "main_control_s1", "main_control_s2"])
p.add_argument("--ckpt_root", default="checkpoints/isaac/runs")
p.add_argument("--banks", nargs="+", default=["conflict", "native", "visible"])
p.add_argument("--n_layouts", type=int, default=128)
p.add_argument("--chunk", type=int, default=64)
p.add_argument("--num_envs", type=int, default=336)
p.add_argument("--actings", nargs="+", default=["base", "no_tower"])
p.add_argument("--mu_steps", type=int, default=40)
p.add_argument("--out", default="results/isaac_analysis/trace")
p.add_argument("--settle_frames", type=int, default=640,
               help="real RTX frames after each closed-loop reset (gm_nav_env.settle_after_reset); 0 = no settling, first frames may show the previous scene")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.analysis import rollout_core as R  # noqa: E402

COPIES = ["base", "no_tower", "no_flag", "tower_swap", "flag_swap"]


def main():
    os.makedirs(args.out, exist_ok=True)
    env = G.make_env("conflict", args.num_envs)
    env.reset()
    env.step(torch.zeros(args.num_envs, 3, device=env.device))
    dev = env.device
    for bank in args.banks:
        base = R.bank_layouts(bank, args.n_layouts)
        lays = R.swap_layouts(base)
        for run in args.runs:
            path = f"{args.out}/{bank}_{run}.npz"
            if os.path.exists(path):
                continue
            agent = R.load_agent(f"{args.ckpt_root}/{run}/final_ckpt.pt", dev)
            out = {}
            for acting in args.actings:
                runner = R.CopyRunner(env, args.chunk, COPIES, acting, settle_frames=args.settle_frames)
                pol = R.Policy(agent, lambda bc, t, st, a=acting: bc[a]["mu"])
                parts = []
                for c0 in range(0, args.n_layouts, args.chunk):
                    sub = {e: {k: v[c0:c0 + args.chunk] for k, v in l.items()} for e, l in lays.items()}
                    parts.append(runner.run(sub, pol, record_mu_steps=args.mu_steps))
                out[f"{acting}_outcome"] = np.concatenate([q["outcome"] for q in parts]).astype(str)
                out[f"{acting}_xy"] = np.concatenate([q["xy"] for q in parts])
                out[f"{acting}_yaw"] = np.concatenate([q["yaw"] for q in parts])
                out[f"{acting}_mu"] = np.concatenate([q["mu"] for q in parts])
                out[f"{acting}_drift"] = np.array([q["drift"] for q in parts])
                oc = out[f"{acting}_outcome"]
                print(f"{bank} {run} acting={acting}: goal {np.mean(oc == 'goal'):.3f} proxy {np.mean(oc == 'proxy'):.3f} "
                      f"timeout {np.mean(oc == 'timeout'):.3f} fall {np.mean(oc == 'fall'):.3f} "
                      f"drift {out[f'{acting}_drift'].max():.2e}", flush=True)
            sig = torch.exp(agent.actor_logstd).reshape(-1).cpu().numpy()
            np.savez_compressed(path, copies=np.array(COPIES), sigma=sig,
                                **{f"lay_{e}_{k}": v for e, l in lays.items() for k, v in l.items() if k in ("flag", "tower")},
                                start=base["start"], yaw0=base["yaw"], **out)
    open(f"{args.out}/DONE", "w").write("ok\n")
    env.close()


main()
app.close()
