"""Scripted end-effector oracle: checks reachability of the spawn workspace and the success/touch logic."""
import argparse

import gymnasium as gym
import numpy as np
import torch

import gm_robotics.envs  # noqa: F401


def run(env_id, variant, n, steps, seed, target_fn, **kw):
    env = gym.make(env_id, num_envs=n, obs_mode="state_dict", sim_backend="physx_cuda", variant=variant,
                   control_mode="pd_ee_delta_pos", max_episode_steps=steps, **kw)
    u = env.unwrapped
    env.reset(seed=seed)
    done = torch.zeros(n, dtype=torch.bool, device=u.device)
    outcome = torch.zeros(n, dtype=torch.long, device=u.device)
    for t in range(steps):
        tgt = target_fn(u)
        delta = torch.clamp((tgt - u.agent.tcp.pose.p) / 0.1, -1, 1)
        act = torch.cat([delta, torch.ones(n, 1, device=u.device)], 1)
        _, _, term, trunc, info = env.step(act)
        new = term & ~done
        outcome[new & info["success"]] = 1
        if "fail" in info:
            outcome[new & info["fail"]] = 2
        done |= term
    return u, outcome.cpu().numpy(), info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_attribute", action="store_true")
    a = p.parse_args()
    for v in ["control", "out_region", "in_region", "gm_large"]:
        above = lambda u: u.goal.pose.p
        u, out, info = run("LocationReach-v1", v, a.n, a.steps, a.seed, above, start_split="all")
        g = u.goal.pose.p.cpu().numpy(); d = info["d_goal"].cpu().numpy()
        miss = out != 1
        print(f"LocationReach {v}: oracle success {np.mean(out == 1):.3f}; misses at goal xy:",
              np.round(g[miss][:8, :2], 2).tolist(), "final d", np.round(d[miss][:8], 3).tolist())
    if a.skip_attribute:
        return
    for v, tgt in [("control", "cube"), ("conflict", "cube"), ("conflict", "sphere")]:
        fn = (lambda u: u.cube_pos + torch.tensor([0, 0, u.half_size], device=u.device)) if tgt == "cube" else \
             (lambda u: u.sphere_pos + torch.tensor([0, 0, u.half_size], device=u.device))
        u, out, info = run("AttributeTable-v1", v, a.n, a.steps, a.seed, fn)
        print(f"AttributeTable {v} oracle->{tgt}: cube {np.mean(out == 1):.3f} sphere {np.mean(out == 2):.3f} "
              f"none {np.mean(out == 0):.3f}; touched_red {info['touched_red'].float().mean().item():.3f} (last step)")


if __name__ == "__main__":
    main()
