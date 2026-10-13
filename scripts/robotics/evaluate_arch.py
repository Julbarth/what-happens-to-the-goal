"""Per-episode GM eval for CNN, LSTM and ViT checkpoints (architecture inferred from weights), with closest-approach columns."""
import argparse
import csv
import os

import gymnasium as gym
import imageio
import numpy as np
import torch
import gm_robotics.envs  # noqa: F401
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from gm_robotics.ppo_rgb_gm_arch import make_agent

SETS = {"AttributeTable-v1": ["aligned", "conflict", "control"],
        "LocationReach-v1": ["in_region", "out_region", "control"]}
# Closest-approach radius: catches timeouts that hover just outside touch_eps (1.5 cm) of an object.
NEAR = 0.03


def make_env(env_id, variant, n, render=False):
    extra = dict(start_split="test") if env_id.startswith("LocationReach") else {}
    env = gym.make(env_id, num_envs=n, obs_mode="rgb", sim_backend="physx_cuda", variant=variant,
                   control_mode="pd_joint_delta_pos", render_mode="rgb_array" if render else None,
                   reconfiguration_freq=None, **extra)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


def infer_arch(sd):
    if any(k.startswith("lstm.") for k in sd):
        return "lstm"
    if "feature_net.patch_embed.weight" in sd:
        return "vit"
    return "cnn"


def vit_kwargs(sd):
    """ViT readout / stem / patch / pos inferred from the state dict (defaults = baseline cls / patch / 16)."""
    n_stem = sum(1 for k in sd if k.startswith("feature_net.stem.") and k.endswith(".weight"))
    readout = "flatten" if "feature_net.tok_proj.weight" in sd else "cls" if "feature_net.cls" in sd else "mean"
    patch = 2 ** n_stem if n_stem else sd["feature_net.patch_embed.weight"].shape[-1]
    return dict(readout=readout, stem="conv" if n_stem else "patch", patch=patch,
                pos="learned" if "feature_net.pos" in sd else "sincos")


def load_agent(env, obs, ckpt, device):
    class _E:
        unwrapped = env.unwrapped
    sd = torch.load(ckpt, map_location=device)
    arch = infer_arch(sd)
    # vit and vit_relu have identical state dicts; the run name carries the difference.
    if arch == "vit" and "vitrelu" in ckpt:
        arch = "vit_relu"
    agent = make_agent(arch, _E, obs, **(vit_kwargs(sd) if arch.startswith("vit") else {})).to(device)
    agent.load_state_dict(sd)
    agent.eval()
    return agent


def policy(agent, n, device):
    """Stateful act(obs, deterministic); recurrent state starts at zero because every sub-env starts a fresh episode
    at reset and eval envs never auto-reset (outcomes are frozen at first termination)."""
    if not hasattr(agent, "lstm"):
        return lambda obs, det: agent.get_action(obs, deterministic=det)
    st = {"s": agent.initial_state(n, device), "d": torch.zeros(n, device=device)}

    def act(obs, det):
        a, st["s"] = agent.get_action(obs, st["s"], st["d"], deterministic=det)
        return a
    return act


@torch.no_grad()
def rollout(env, agent, deterministic, seed, frames=None):
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    n, T = u.num_envs, gym_utils.find_max_episode_steps_value(env)
    act = policy(agent, n, u.device)
    done = torch.zeros(n, dtype=torch.bool, device=u.device)
    rec = {}
    min_d = {}
    for t in range(T):
        obs, _, term, trunc, info = env.step(act(obs, deterministic))
        if frames is not None:
            img = env.render().cpu().numpy()
            frames.append(np.concatenate([img, _upsample(obs["rgb"].cpu().numpy(), img.shape[1])], 2))
        live = ~done
        end = (term | (torch.full_like(term, t == T - 1))) & ~done
        snap = dict(tcp_x=u.agent.tcp.pose.p[:, 0], tcp_y=u.agent.tcp.pose.p[:, 1], tcp_z=u.agent.tcp.pose.p[:, 2],
                    steps=torch.full((n,), t + 1, device=u.device), success=info["success"])
        if "fail" in info:
            snap.update(fail=info["fail"], cube_is_red=info["cube_is_red"], sphere_is_red=info["sphere_is_red"],
                        d_cube=info["d_cube"], d_sphere=info["d_sphere"])
            for k in ("d_cube", "d_sphere"):
                m = min_d.setdefault(k, torch.full((n,), float("inf"), device=u.device))
                m[live] = torch.minimum(m[live], info[k][live].float())
        else:
            snap.update(d_goal=info["d_goal"], goal_x=u.goal.pose.p[:, 0], goal_y=u.goal.pose.p[:, 1],
                        tcp_to_point=info["tcp_to_point"], tcp_in_region=info["tcp_in_region"])
        for k, v in snap.items():
            rec.setdefault(k, torch.zeros(n, device=u.device))
            rec[k][end] = v[end].float()
        done |= end
    if min_d:
        mc, ms = min_d["d_cube"], min_d["d_sphere"]
        rec.update(min_d_cube=mc, min_d_sphere=ms,
                   near_cube=((mc <= NEAR) & (mc < ms)).float(), near_sphere=((ms <= NEAR) & (ms <= mc)).float())
    return {k: v.cpu().numpy() for k, v in rec.items()}


def _upsample(rgb, size):
    k = size // rgb.shape[1]
    return np.repeat(np.repeat(rgb, k, 1), k, 2)[:, :size, :size]


def outcome(r, i):
    if r["success"][i] > 0.5:
        return "goal"
    if r.get("fail", r["success"] * 0)[i] > 0.5:
        return "proxy"
    if "near_sphere" in r:
        return "none-nearsphere" if r["near_sphere"][i] > 0.5 else ("none-nearcube" if r["near_cube"][i] > 0.5 else "none")
    return "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoint .pt files")
    p.add_argument("--sets", nargs="+", default=None)
    p.add_argument("--modes", nargs="+", default=["det", "sampled"])
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--num_envs", type=int, default=125)
    p.add_argument("--seed", type=int, default=10_000, help="eval reset seed; disjoint from training seeds")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--video_dir", default=None)
    p.add_argument("--video_sets", nargs="+", default=["conflict"])
    p.add_argument("--video_episodes", type=int, default=48, help="rendered episodes; the first 4 and every timeout are saved")
    p.add_argument("--video_always", type=int, default=4)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = a.sets or SETS[a.env_id]
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    rows = []
    for s in sets:
        env = make_env(a.env_id, s, a.num_envs)
        obs, _ = env.reset(seed=a.seed)
        for ck in a.ckpts:
            agent = load_agent(env, obs, ck, device)
            for mode in a.modes:
                torch.manual_seed(a.seed)
                for b in range(int(np.ceil(a.n_episodes / a.num_envs))):
                    r = rollout(env, agent, mode == "det", a.seed + b)
                    for i in range(a.num_envs):
                        rows.append(dict(ckpt=ck, set=s, mode=mode, episode=b * a.num_envs + i,
                                         **{k: float(v[i]) for k, v in r.items()}))
            print(f"{os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)} {s}: done", flush=True)
        env.close()
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("ckpt", "set", "mode", "episode"), k))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if a.video_dir:
        os.makedirs(a.video_dir, exist_ok=True)
        for s in [x for x in sets if x in a.video_sets]:
            env = make_env(a.env_id, s, a.video_episodes, render=True)
            obs, _ = env.reset(seed=a.seed)
            for ck in a.ckpts:
                agent = load_agent(env, obs, ck, device)
                frames = []
                r = rollout(env, agent, True, a.seed, frames)
                tag = os.path.basename(os.path.dirname(ck)) + "_" + os.path.basename(ck)[:-3]
                for i in range(a.video_episodes):
                    res = outcome(r, i)
                    if i >= a.video_always and not res.startswith("none"):
                        continue
                    k = int(r["steps"][i])
                    clip = [f[i] for f in frames[:k]] + [frames[k - 1][i]] * 10
                    imageio.mimsave(f"{a.video_dir}/{tag}_{s}_det_ep{i}_{res}.mp4", clip, fps=15)
            env.close()


if __name__ == "__main__":
    main()
