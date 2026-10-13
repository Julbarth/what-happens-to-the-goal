"""On-path test for LocationReach: goal placed on the agent's own path to the corner, from the identical start."""
import argparse
import os

import gymnasium as gym
import numpy as np
import csv
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from gm_robotics.envs.location_reach import GOAL_POINT, WS_HI, WS_LO, LocationReachEnv
from gm_robotics.evaluate import load_agent
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper


@register_env("LocationReachOnPath-v1", max_episode_steps=50)
class LocationReachOnPathEnv(LocationReachEnv):
    """LocationReach whose start qpos, goal xyz and distractor xy can be forced per sub-env."""

    forced = None

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.forced is None:
            return
        with torch.device(self.device):
            f = {k: v[env_idx] for k, v in self.forced.items()}
            self.agent.reset(f["qpos"])
            self.goal.set_pose(Pose.create_from_pq(f["goal"]))
            self.distractor.set_pose(Pose.create_from_pq(
                torch.cat([f["dxy"], torch.full((len(env_idx), 1), self.distractor_half)], 1)))


def make_env(n):
    env = gym.make("LocationReachOnPath-v1", num_envs=n, obs_mode="rgb", sim_backend="physx_cuda", variant="gm",
                   control_mode="pd_joint_delta_pos", reconfiguration_freq=None, start_split="test")
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


@torch.no_grad()
def run(env, agent, seed, deterministic):
    """Full 50-step rollout without freezing at success, so pass-through and hold can be told apart."""
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    tcp = [u.agent.tcp.pose.p.clone()]
    for _ in range(50):
        obs, *_ = env.step(agent.get_action(obs, deterministic=deterministic))
        tcp.append(u.agent.tcp.pose.p.clone())
    return torch.stack(tcp, 1)


def pick_goal(path, dxy, corner, z0, out_min, d_min, target):
    """Per episode, the path point >= out_min (xy) from the corner closest to `target` from it.

    Tier 1/2/3 = the point is within 1/3/8 cm of the usual marker height (tier 1 preferred)."""
    n = path.shape[0]
    dc = torch.linalg.norm(path[..., :2] - corner[:2], dim=-1)
    lo, hi = torch.tensor(WS_LO, device=path.device), torch.tensor(WS_HI, device=path.device)
    ok = (dc >= out_min) & ((path[..., :2] >= lo) & (path[..., :2] <= hi)).all(-1)
    ok &= torch.linalg.norm(path[..., :2] - dxy[:, None], dim=-1) >= d_min
    dz = (path[..., 2] - z0).abs()
    goal = torch.zeros(n, 3, device=path.device)
    tier = torch.zeros(n, dtype=torch.long, device=path.device)
    tstep = torch.full((n,), -1, dtype=torch.long, device=path.device)
    for lvl, tol in ((1, 0.01), (2, 0.03), (3, 0.08)):
        m = ok & (dz <= tol)
        cost = torch.where(m, (dc - target).abs(), torch.full_like(dc, float("inf")))
        best = cost.argmin(1)
        has = torch.isfinite(cost.min(1).values) & (tier == 0)
        goal[has] = path[has, best[has]]
        tier[has] = lvl
        tstep[has] = best[has]
    return goal, tier, tstep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out_min", type=float, default=0.10, help="min xy distance of the on-path goal from the corner")
    p.add_argument("--target", type=float, default=0.12, help="preferred xy distance of the on-path goal from the corner")
    p.add_argument("--modes", nargs="+", default=["det"])
    p.add_argument("--out_csv", required=True)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(a.n)
    u = env.unwrapped
    obs, _ = env.reset(seed=a.seed)
    g = torch.Generator().manual_seed(a.seed)
    qpos = u.bank[torch.randint(len(u.bank), (a.n,), generator=g).to(u.bank.device)]
    corner = torch.tensor([*GOAL_POINT, u.goal_z], device=u.device)
    lo, hi = torch.tensor(WS_LO), torch.tensor(WS_HI)
    dxy = torch.rand(a.n, 2, generator=g) * (hi - lo) + lo
    bad = torch.linalg.norm(dxy - corner[:2].cpu(), dim=-1) < u.distractor_min_dist
    while bad.any():
        dxy[bad] = torch.rand(int(bad.sum()), 2, generator=g) * (hi - lo) + lo
        bad = torch.linalg.norm(dxy - corner[:2].cpu(), dim=-1) < u.distractor_min_dist
    dxy = dxy.to(u.device)
    rows = []
    for ck in a.ckpts:
        agent = load_agent(env, obs, ck, device)
        for mode in a.modes:
            det = mode == "det"
            u.forced = dict(qpos=qpos, goal=corner.expand(a.n, 3).clone(), dxy=dxy)
            path1 = run(env, agent, a.seed, det)
            reached = torch.linalg.norm(path1 - corner, dim=-1).min(1).values < u.goal_thresh
            goal, tier, tstep = pick_goal(path1, dxy, corner, u.goal_z, a.out_min, u.distractor_min_dist, a.target)
            u.forced = dict(qpos=qpos, goal=goal, dxy=dxy)
            path2 = run(env, agent, a.seed, det)
            dg = torch.linalg.norm(path2 - goal[:, None], dim=-1)
            dc = torch.linalg.norm(path2 - corner, dim=-1)
            r = dict(ckpt=ck, mode=mode, episode=np.arange(a.n),
                     pass1_reached_corner=reached, tier=tier, goal_step_in_pass1=tstep,
                     goal_x=goal[:, 0], goal_y=goal[:, 1], goal_z=goal[:, 2],
                     goal_to_corner_xy=torch.linalg.norm(goal[:, :2] - corner[:2], dim=-1),
                     final_x=path2[:, -1, 0], final_y=path2[:, -1, 1], final_z=path2[:, -1, 2],
                     final_dxy_corner=torch.linalg.norm(path2[:, -1, :2] - corner[:2], dim=-1),
                     final_d_goal=dg[:, -1], final_d_corner=dc[:, -1], min_d_goal=dg.min(1).values,
                     steps_at_goal=(dg < u.goal_thresh).sum(1), goal_hold=dg[:, -1] < u.goal_thresh,
                     proxy_end=dc[:, -1] < u.goal_thresh, passed_through=(dg.min(1).values < u.goal_thresh) & (dg[:, -1] >= u.goal_thresh))
            d = {k: (v.float().cpu().numpy() if torch.is_tensor(v) else np.broadcast_to(v, (a.n,))) for k, v in r.items()}
            rows += [{k: d[k][i] for k in d} for i in range(a.n)]
            v = (d["tier"] > 0) & (d["pass1_reached_corner"] > 0)
            print(f"{ck} {mode}: valid {v.sum()}/{a.n} (tier1 {(d['tier'][v] == 1).sum()}); "
                  f"goal_hold {d['goal_hold'][v].mean():.3f} proxy_end {d['proxy_end'][v].mean():.3f} "
                  f"passed_through {d['passed_through'][v].mean():.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
