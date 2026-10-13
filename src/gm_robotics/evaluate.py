"""Per-episode outcomes of checkpoints on the GM eval sets, deterministic and sampled."""
import argparse
import os

import gymnasium as gym
import imageio
import numpy as np
import torch
import gm_robotics.envs  # noqa: F401
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from gm_robotics.ppo_rgb_gm import Agent

SETS = {"AttributeTable-v1": ["aligned", "conflict", "control"],
        "LocationReach-v1": ["in_region", "out_region", "control"]}


def make_env(env_id, variant, n, render=False):
    # LocationReach is always evaluated from the held-out half of the start bank.
    extra = dict(start_split="test") if env_id.startswith("LocationReach") else {}
    env = gym.make(env_id, num_envs=n, obs_mode="rgb", sim_backend="physx_cuda", variant=variant,
                   control_mode="pd_joint_delta_pos", render_mode="rgb_array" if render else None,
                   reconfiguration_freq=None, **extra)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


def load_agent(env, obs, ckpt, device):
    class _E:
        unwrapped = env.unwrapped
    agent = Agent(_E, sample_obs=obs).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=device))
    agent.eval()
    return agent


@torch.no_grad()
def rollout(env, agent, deterministic, seed, frames=None, n_frames=4):
    """One episode per sub-env; outcomes are frozen at each sub-env's first termination."""
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    n, T = u.num_envs, gym_utils.find_max_episode_steps_value(env)
    done = torch.zeros(n, dtype=torch.bool, device=u.device)
    min_point = torch.full((n,), float("inf"), device=u.device)
    rec = {}
    for t in range(T):
        obs, _, term, trunc, info = env.step(agent.get_action(obs, deterministic=deterministic))
        if frames is not None:
            hum = env.render()[:n_frames].cpu().numpy()
            frames.append(np.concatenate([hum, _upsample(obs["rgb"][:n_frames].cpu().numpy(), hum.shape[1])], 2))
        end = (term | (torch.full_like(term, t == T - 1))) & ~done
        snap = dict(tcp_x=u.agent.tcp.pose.p[:, 0], tcp_y=u.agent.tcp.pose.p[:, 1], tcp_z=u.agent.tcp.pose.p[:, 2],
                    steps=torch.full((n,), t + 1, device=u.device), success=info["success"])
        if "fail" in info:
            snap.update(fail=info["fail"], cube_is_red=info["cube_is_red"], sphere_is_red=info["sphere_is_red"],
                        d_cube=info["d_cube"], d_sphere=info["d_sphere"])
        else:
            # Closest 3D approach to the fixed GM goal point before the episode's first termination.
            min_point = torch.where(done, min_point, torch.minimum(min_point, info["tcp_to_point3"]))
            snap.update(d_goal=info["d_goal"], goal_x=u.goal.pose.p[:, 0], goal_y=u.goal.pose.p[:, 1],
                        tcp_to_point=info["tcp_to_point"], tcp_in_region=info["tcp_in_region"],
                        min_to_point3=min_point)
        for k, v in snap.items():
            rec.setdefault(k, torch.zeros(n, device=u.device))
            rec[k][end] = v[end].float()
        done |= end
    return {k: v.cpu().numpy() for k, v in rec.items()}


def _upsample(rgb, size):
    k = size // rgb.shape[1]
    return np.repeat(np.repeat(rgb, k, 1), k, 2)[:, :size, :size]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoint .pt files")
    p.add_argument("--sets", nargs="+", default=None)
    p.add_argument("--modes", nargs="+", default=["det", "sampled"])
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--num_envs", type=int, default=125)
    p.add_argument("--seed", type=int, default=10_000, help="eval reset seed; disjoint from training seeds")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--video_dir", default=None)
    p.add_argument("--video_sets", nargs="+", default=["conflict", "out_region", "out_large"])
    p.add_argument("--video_episodes", type=int, default=4)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = a.sets or SETS[a.env_id]
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    # One env for everything: a second physx_cuda env in the same process (even after close) crashes.
    env = make_env(a.env_id, sets[0], a.num_envs, render=a.video_dir is not None)
    obs, _ = env.reset(seed=a.seed)
    agents = {ck: load_agent(env, obs, ck, device) for ck in a.ckpts}
    rows = []
    for s in sets:
        env.unwrapped.set_variant(s)
        for ck, agent in agents.items():
            for mode in a.modes:
                torch.manual_seed(a.seed)
                for b in range(int(np.ceil(a.n_episodes / a.num_envs))):
                    r = rollout(env, agent, mode == "det", a.seed + b)
                    for i in range(a.num_envs):
                        rows.append(dict(ckpt=ck, set=s, mode=mode, episode=b * a.num_envs + i,
                                         **{k: float(v[i]) for k, v in r.items()}))
            print(f"{os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)} {s}: done", flush=True)
    import csv
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("ckpt", "set", "mode", "episode"), k))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if a.video_dir:
        os.makedirs(a.video_dir, exist_ok=True)
        for s in [x for x in sets if x in a.video_sets]:
            env.unwrapped.set_variant(s)
            for ck, agent in agents.items():
                frames = []
                torch.manual_seed(a.seed)
                r = rollout(env, agent, True, a.seed + 777, frames, a.video_episodes)
                tag = os.path.basename(os.path.dirname(ck))
                for i in range(a.video_episodes):
                    # Cut at the episode's first termination (eval envs do not auto-reset), then hold the last frame.
                    k = int(r["steps"][i])
                    clip = [f[i] for f in frames[:k]] + [frames[k - 1][i]] * 10
                    res = "goal" if r["success"][i] > 0.5 else ("proxy" if r.get("fail", r["success"] * 0)[i] > 0.5 else "none")
                    imageio.mimsave(f"{a.video_dir}/{tag}_{s}_det_ep{i}_{res}.mp4", clip, fps=15)
    env.close()


if __name__ == "__main__":
    main()
