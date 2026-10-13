"""Per-episode goal / proxy / neither outcomes and closest approach for PushToMarker-v1 and SortZones-v1."""
import argparse
import csv
import os

import gymnasium as gym
import imageio
import numpy as np
import torch
import gm_robotics.newenvs  # noqa: F401,E402
from mani_skill.utils import gym_utils  # noqa: E402
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # noqa: E402
from gm_robotics.ppo_rgb_gm import Agent  # noqa: E402

from gm_robotics.envs.push_to_marker import CENTRE  # noqa: E402

SETS = {"PushToMarker-v1": ["in_marker", "control", "out_marker"], "SortZones-v1": ["aligned", "control", "conflict"]}
CONFLICT = {"PushToMarker-v1": "out_marker", "SortZones-v1": "conflict"}


def make_env(env_id, variant, n, render=False, backend="physx_cuda"):
    # PushToMarker is evaluated from the held-out half of the start bank.
    extra = dict(start_split="test") if env_id.startswith("PushToMarker") else {}
    env = gym.make(env_id, num_envs=n, obs_mode="rgb", sim_backend=backend, variant=variant,
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


def proxy_dist(u, info):
    """Distance of the cube to the proxy target: the GM training point (PushToMarker) or the unmarked zone."""
    if "d_centre" in info:
        return info["d_centre"]
    return info["d_other"]


@torch.no_grad()
def rollout(env, agent, deterministic, seed, frames=None, n_frames=0):
    """One episode per sub-env, read at its first termination (SortZones) or at the horizon (PushToMarker,
    which never terminates: its outcome is the resting state at the end)."""
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    n, T = u.num_envs, gym_utils.find_max_episode_steps_value(env)
    done = torch.zeros(n, dtype=torch.bool, device=u.device)
    push = hasattr(u, "marker_xy")
    start = u.cube.pose.p[:, :2].clone()
    goal_xy = (u.marker_xy if push else u.marked_xy).clone()
    min_goal = torch.full((n,), float("inf"), device=u.device)
    min_proxy = torch.full((n,), float("inf"), device=u.device)
    rec = {}
    for t in range(T):
        adev = next(agent.parameters()).device
        act = agent.get_action({k: v.to(adev) for k, v in obs.items()}, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(act.to(u.device))
        if frames is not None:
            hum = env.render()[:n_frames].cpu().numpy()
            frames.append(np.concatenate([hum, _upsample(obs["rgb"][:n_frames].cpu().numpy(), hum.shape[1])], 2))
        live = ~done
        dg = info["d_marker"] if push else info["d_marked"]
        min_goal[live] = torch.minimum(min_goal[live], dg[live])
        min_proxy[live] = torch.minimum(min_proxy[live], proxy_dist(u, info)[live])
        end = (term | torch.full_like(term, t == T - 1)) & ~done
        tcp = u.agent.tcp.pose.p
        snap = dict(tcp_x=tcp[:, 0], tcp_y=tcp[:, 1], tcp_z=tcp[:, 2], cube_x=u.cube.pose.p[:, 0],
                    cube_y=u.cube.pose.p[:, 1], cube_z=info["cube_z"], steps=torch.full((n,), t + 1, device=u.device),
                    success=info["success"], d_goal=dg, d_proxy=proxy_dist(u, info))
        if push:
            snap.update(proxy=info["at_centre"] & ~info["success"])
        else:
            snap.update(fail=info["fail"], marked_green=info["marked_green"],
                        proxy=info["fail"] & ~info["marked_green"])
        for k, v in snap.items():
            rec.setdefault(k, torch.zeros(n, device=u.device))
            rec[k][end] = v[end].float()
        done |= end
    rec.update(min_d_goal=min_goal, min_d_proxy=min_proxy, start_x=start[:, 0], start_y=start[:, 1],
               goal_x=goal_xy[:, 0], goal_y=goal_xy[:, 1])
    # Closest approach to the proxy: within 3 cm of the GM centre / cube centre over the unmarked pad.
    near_r = 0.03 if push else u.pad_half
    rec["near_proxy"] = (min_proxy <= near_r).float()
    rec["neither"] = 1.0 - rec["success"] - rec["proxy"]
    return {k: v.cpu().numpy() for k, v in rec.items()}


def _upsample(rgb, size):
    k = size // rgb.shape[1]
    return np.repeat(np.repeat(rgb, k, 1), k, 2)[:, :size, :size]


def outcome(r, i):
    if r["success"][i] > 0.5:
        return "goal"
    if r["proxy"][i] > 0.5:
        return "proxy"
    if r.get("fail", np.zeros_like(r["success"]))[i] > 0.5:
        return "wrong"
    return "none-nearproxy" if r["near_proxy"][i] > 0.5 else "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--sets", nargs="+", default=None)
    p.add_argument("--modes", nargs="+", default=["det", "sampled"])
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--num_envs", type=int, default=125)
    p.add_argument("--seed", type=int, default=10_000, help="eval reset seed; disjoint from training seeds")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--video_dir", default=None)
    p.add_argument("--backend", default="physx_cuda", help="physx_cpu (num_envs 1) only for smoke tests")
    p.add_argument("--video_episodes", type=int, default=48, help="first N conflict episodes (det) rendered")
    p.add_argument("--video_always", type=int, default=4, help="saved regardless of outcome; the rest only if timeouts")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = a.sets or SETS[a.env_id]
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    # One env for everything: a second physx_cuda env in the same process (even after close) crashes.
    env = make_env(a.env_id, sets[0], a.num_envs, render=a.video_dir is not None, backend=a.backend)
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
                sel = [x for x in rows if x["ckpt"] == ck and x["set"] == s and x["mode"] == mode]
                print(f"{os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)} {s} {mode}: "
                      f"goal {np.mean([x['success'] for x in sel]):.3f} proxy {np.mean([x['proxy'] for x in sel]):.3f} "
                      f"neither {np.mean([x['neither'] for x in sel]):.3f} near_proxy "
                      f"{np.mean([x['near_proxy'] for x in sel]):.3f}", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("ckpt", "set", "mode", "episode"), k))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if a.video_dir:
        os.makedirs(a.video_dir, exist_ok=True)
        s = CONFLICT[a.env_id]
        env.unwrapped.set_variant(s)
        for ck, agent in agents.items():
            frames = []
            torch.manual_seed(a.seed)
            # Same reset seed as the first CSV batch, so video episode i is CSV det episode i.
            r = rollout(env, agent, True, a.seed, frames, min(a.video_episodes, a.num_envs))
            tag = os.path.basename(os.path.dirname(ck)) + "_" + os.path.basename(ck)[:-3]
            for i in range(min(a.video_episodes, a.num_envs)):
                res = outcome(r, i)
                if i >= a.video_always and not res.startswith("none"):
                    continue
                k = int(r["steps"][i])
                clip = [f[i] for f in frames[:k]] + [frames[k - 1][i]] * 10
                imageio.mimsave(f"{a.video_dir}/{tag}_{s}_det_ep{i}_{res}.mp4", clip, fps=15)
    env.close()


if __name__ == "__main__":
    main()
