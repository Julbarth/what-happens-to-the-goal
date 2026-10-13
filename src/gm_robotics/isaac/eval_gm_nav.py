"""Evaluates Isaac GM nav agents on fixed layout banks (deterministic and sampled), one per-episode CSV per run."""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True, help="run dir names under --runs_root")
p.add_argument("--ckpt", default="final_ckpt.pt")
p.add_argument("--runs_root", default="checkpoints/isaac/runs")
p.add_argument("--variants", nargs="+", default=["gm", "control", "conflict"])
p.add_argument("--modes", nargs="+", default=["det", "sampled"])
p.add_argument("--num_envs", type=int, default=100)
p.add_argument("--episodes_per_env", type=int, default=5)
p.add_argument("--bank_seed", type=int, default=10000)
p.add_argument("--res", type=int, default=96)
p.add_argument("--episode_s", type=float, default=20.0)
p.add_argument("--tag", default="eval")
p.add_argument("--settle_frames", type=int, default=640,
               help="real RTX frames rendered after synchronous per-wave resets (gm_nav_env.run_episode_waves); 0 = "
                    "asynchronous run_episodes, whose first frames after a reset carry the previous layout")
p.add_argument("--out", default="results/isaac")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import csv  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.agent import Agent  # noqa: E402


def main():
    os.makedirs(args.out, exist_ok=True)
    env = G.make_env("gm", args.num_envs, res=args.res, episode_s=args.episode_s)
    obs, _ = env.reset()
    torch.manual_seed(args.bank_seed)
    banks = {v: G.make_bank(v, args.num_envs * (args.episodes_per_env + 1), args.bank_seed + i, env.device)
             for i, v in enumerate(args.variants)}
    for run in args.runs:
        agent = Agent(tuple(obs["rgb"].shape[1:]), obs["state"].shape[1], 3).to(env.device)
        agent.load_state_dict(torch.load(f"{args.runs_root}/{run}/{args.ckpt}", map_location=env.device))
        agent.eval()
        path = f"{args.out}/{args.tag}_{run}_{os.path.splitext(args.ckpt)[0]}.csv"
        rows = []
        for v in args.variants:
            for mode in args.modes:
                torch.manual_seed(args.bank_seed + 1)
                G.set_variant(env, v, banks[v])
                act = lambda o: agent.get_action(o, deterministic=mode == "det")
                if args.settle_frames > 0:
                    res = G.run_episode_waves(env, act, args.episodes_per_env, args.settle_frames)
                else:
                    res = G.run_episodes(env, act, args.episodes_per_env)
                for r in res:
                    rows.append(dict(run=run, ckpt=args.ckpt, variant=v, mode=mode, **r))
                oc = np.array([r["outcome"] for r in res])
                print(f"{run} {v} {mode} n={len(res)} goal={np.mean(oc == 'goal'):.3f} proxy={np.mean(oc == 'proxy'):.3f} "
                      f"timeout={np.mean(oc == 'timeout'):.3f} fall={np.mean(oc == 'fall'):.3f} "
                      f"tower_visit={np.mean([r['tower_visit_step'] >= 0 for r in res]):.3f}", flush=True)
        with open(path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("wrote", path)
    env.close()


main()
app.close()
