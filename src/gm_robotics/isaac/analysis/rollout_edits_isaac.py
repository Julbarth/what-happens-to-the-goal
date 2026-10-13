"""Trajectory consequence of the world edits: closed-loop rollouts from edit-bank layouts, first object reached per edit."""
import argparse
import csv
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--ckpt_root", default="checkpoints/isaac/runs")
p.add_argument("--bank", default="visible")
p.add_argument("--bank_seed", type=int, default=1, help="geom.sample_edit_bank seed (build_banks_isaac.py: seed + 1 = visible)")
p.add_argument("--bank_n", type=int, default=512, help="size the bank was built with in build_banks_isaac.py")
p.add_argument("--n_layouts", type=int, default=256)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--edits", nargs="+", default=["base", "flag_swap", "tower_swap", "no_flag", "no_tower"])
p.add_argument("--out", default="results/isaac_analysis/rollouts")
p.add_argument("--settle_frames", type=int, default=640, help="0 = asynchronous resets without settling (first frames may show the previous scene)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.analysis import geom  # noqa: E402
from gm_robotics.isaac.agent import Agent  # noqa: E402

KEYS = ("start", "yaw", "flag", "tower", "boxes", "flag_yaw", "box_yaw")
# Which candidate location (u = flag's base spot, v = tower's base spot) each env outcome means under each edit.
WHERE = {"base": {"goal": "u", "proxy": "v"}, "flag_swap": {"goal": "v", "proxy": "v"},
         "tower_swap": {"goal": "u", "proxy": "u"}, "no_flag": {"proxy": "v"}, "no_tower": {"goal": "u"}}


def main():
    os.makedirs(args.out, exist_ok=True)
    assert args.n_layouts % args.num_envs == 0
    env = G.make_env("conflict", args.num_envs)
    env.reset()
    env.step(torch.zeros(args.num_envs, 3, device=env.device))   # first physics step syncs the renderer
    b = geom.sample_edit_bank(args.bank_n, args.bank_seed, args.bank)
    lays = geom.edit_layouts({k: v[: args.n_layouts] for k, v in b.items()})
    for run in args.runs:
        path = f"{args.out}/{args.bank}_{run}.csv"
        if os.path.exists(path):
            continue
        agent = Agent((96, 96, 3), 12, 3).to(env.device)
        agent.load_state_dict(torch.load(f"{args.ckpt_root}/{run}/final_ckpt.pt", map_location=env.device))
        agent.eval()
        rows = []
        for e in args.edits:
            bank = {k: torch.as_tensor(lays[e][k], device=env.device, dtype=torch.float32) for k in KEYS}
            G.set_variant(env, "conflict", bank)
            torch.manual_seed(0)
            act = lambda o: agent.get_action(o, deterministic=True)
            if args.settle_frames > 0:
                res = G.run_episode_waves(env, act, args.n_layouts // args.num_envs, args.settle_frames)
            else:
                res = G.run_episodes(env, act, args.n_layouts // args.num_envs)
            for r in res:
                r.update(run=run, edit=e, reached=WHERE[e].get(r["outcome"], "none"))
                rows.append(r)
            oc = np.array([r["reached"] for r in res])
            print(f"{run} {e} n={len(res)} u={np.mean(oc == 'u'):.3f} v={np.mean(oc == 'v'):.3f} none={np.mean(oc == 'none'):.3f}",
                  flush=True)
        with open(path, "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    open(f"{args.out}/DONE_{args.bank}", "w").write("ok\n")
    env.close()


main()
app.close()
