"""Off-path test for LocationReach: goal at a gripper position the GM agent reaches from other starts, off its own path."""
import argparse
import csv
import os

import imageio
import numpy as np
import torch
from gm_robotics.eval_onpath import make_env as _make_env_norender
from gm_robotics.evaluate import load_agent
from gm_robotics.envs.location_reach import GOAL_POINT, WS_HI, WS_LO
from gm_robotics import eval_onpath  # noqa: F401  (registers LocationReachOnPath-v1)
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper


def make_env(n, render):
    if not render:
        return _make_env_norender(n)
    env = gym.make("LocationReachOnPath-v1", num_envs=n, obs_mode="rgb", sim_backend="physx_cuda", variant="gm",
                   control_mode="pd_joint_delta_pos", reconfiguration_freq=None, start_split="test",
                   render_mode="rgb_array")
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


@torch.no_grad()
def run(env, agent, seed, render):
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    tcp, frames = [u.agent.tcp.pose.p.clone()], []
    if render:
        frames.append(env.render().cpu().numpy())
    for _ in range(50):
        obs, *_ = env.step(agent.get_action(obs, deterministic=True))
        tcp.append(u.agent.tcp.pose.p.clone())
        if render:
            frames.append(env.render().cpu().numpy())
    return torch.stack(tcp, 1), frames


def first_event(path, goal, corner, thr):
    """Training ends an episode at the first 3 cm contact, so only the first contact is a trained outcome."""
    T = path.shape[1]
    tg = torch.where((torch.linalg.norm(path - goal[:, None], dim=-1) < thr).any(1),
                     (torch.linalg.norm(path - goal[:, None], dim=-1) < thr).float().argmax(1), torch.full((len(path),), T, device=path.device))
    tc = torch.where((torch.linalg.norm(path - corner, dim=-1) < thr).any(1),
                     (torch.linalg.norm(path - corner, dim=-1) < thr).float().argmax(1), torch.full((len(path),), T, device=path.device))
    return tg, tc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gm_ckpt", required=True, help="GM agent whose visited region defines the goal positions")
    p.add_argument("--ctrl_ckpt", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out_min", type=float, default=0.10, help="min xy distance of the goal from the corner")
    p.add_argument("--off_min", type=float, default=0.08, help="min 3D distance of the goal from this start's own GM path")
    p.add_argument("--z_tol", type=float, default=0.01)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--video", default=None, help="mp4 prefix; renders all n episodes (use small n with --load_setup)")
    p.add_argument("--save_setup", default=None, help="npz with starts, distractors and chosen goals")
    p.add_argument("--save_paths", default=None, help="npz with every gripper path (pass 1 GM, pass 2 GM and control)")
    p.add_argument("--load_setup", default=None, help="replay the first n episodes of a saved setup")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render = a.video is not None
    env = make_env(a.n, render)
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
    if a.load_setup:
        z = np.load(a.load_setup)
        qpos, dxy, goal, valid = (torch.tensor(z[k][:a.n], device=u.device) for k in ("qpos", "dxy", "goal", "valid"))
    agents = {"gm": load_agent(env, obs, a.gm_ckpt, device), "control": load_agent(env, obs, a.ctrl_ckpt, device)}

    # Pass 1: the GM agent with the goal at the corner (its training situation), from every start.
    u.forced = dict(qpos=qpos, goal=corner.expand(a.n, 3).clone(), dxy=dxy)
    path1, frames1 = run(env, agents["gm"], a.seed, render)
    if not a.load_setup:
        # Visited region: gripper positions at marker height, away from the corner, pooled over all starts.
        pts = path1.reshape(-1, 3)
        owner = torch.arange(a.n, device=u.device).repeat_interleave(path1.shape[1])
        keep = ((pts[:, 2] - u.goal_z).abs() <= a.z_tol) & (torch.linalg.norm(pts[:, :2] - corner[:2], dim=-1) >= a.out_min)
        keep &= ((pts[:, :2] >= lo.to(u.device)) & (pts[:, :2] <= hi.to(u.device))).all(-1)
        pts, owner = pts[keep], owner[keep]
        goal = torch.zeros(a.n, 3, device=u.device)
        valid = torch.zeros(a.n, dtype=torch.bool, device=u.device)
        for i in range(a.n):
            off = torch.linalg.norm(path1[i][None] - pts[:, None], dim=-1).min(1).values >= a.off_min
            ok = off & (owner != i) & (torch.linalg.norm(pts[:, :2] - dxy[i], dim=-1) >= u.distractor_min_dist)
            idx = torch.nonzero(ok).flatten()
            if len(idx):
                j = idx[torch.randint(len(idx), (1,), generator=g).item()]
                goal[i], valid[i] = pts[j], True
        goal[~valid] = corner
    if a.save_setup:
        np.savez(a.save_setup, qpos=qpos.cpu().numpy(), dxy=dxy.cpu().numpy(), goal=goal.cpu().numpy(), valid=valid.cpu().numpy())

    # Pass 2: both agents, identical start and distractor, goal at the chosen visited-but-off-path position.
    u.forced = dict(qpos=qpos, goal=goal, dxy=dxy)
    rows, vids, paths = [], {}, dict(gm_pass1=path1.cpu().numpy())
    for name, ag in agents.items():
        path2, frames = run(env, ag, a.seed, render)
        tg, tc = first_event(path2, goal, corner, u.goal_thresh)
        paths[name] = path2.cpu().numpy()
        T = path2.shape[1]
        out = np.where((tg < tc).cpu(), "goal", np.where((tc < tg).cpu(), "corner", "neither"))
        for i in range(a.n):
            rows.append(dict(cohort=name, gm_ckpt=a.gm_ckpt, ctrl_ckpt=a.ctrl_ckpt, episode=i, valid=int(valid[i]),
                             outcome=out[i], t_goal=int(tg[i]), t_corner=int(tc[i]),
                             goal_x=float(goal[i, 0]), goal_y=float(goal[i, 1]), goal_z=float(goal[i, 2]),
                             goal_to_corner_xy=float(torch.linalg.norm(goal[i, :2] - corner[:2])),
                             min_d_goal=float(torch.linalg.norm(path2[i] - goal[i], dim=-1).min()),
                             min_d_corner=float(torch.linalg.norm(path2[i] - corner, dim=-1).min())))
        if render:
            vids[name] = (frames, tg.cpu().numpy(), tc.cpu().numpy(), out)
        v = valid.cpu().numpy()
        print(f"{name}: valid {v.sum()}/{a.n}  goal {np.mean(out[v] == 'goal'):.3f}  corner {np.mean(out[v] == 'corner'):.3f}  "
              f"neither {np.mean(out[v] == 'neither'):.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.save_paths:
        np.savez(a.save_paths, goal=goal.cpu().numpy(), dxy=dxy.cpu().numpy(), valid=valid.cpu().numpy(),
                 corner=corner.cpu().numpy(), **paths)
    if render:
        # Episode clips cut at the first contact (the trained episode end), last frame held.
        np.savez(a.video + "_meta.npz", valid=valid.cpu().numpy(), goal=goal.cpu().numpy(),
                 **{f"{k}_{m}": v for k, (_, tg, tc, out) in vids.items() for m, v in (("tg", tg), ("tc", tc), ("out", out))})
        tc1 = first_event(path1, corner.expand(a.n, 3), corner, u.goal_thresh)[1].cpu().numpy()
        vids["gm_pass1"] = (frames1, np.full(a.n, 99), tc1, np.array(["corner"] * a.n))
        for name, (frames, tg, tc, out) in vids.items():
            for i in range(a.n):
                k = min(int(min(tg[i], tc[i])), 50) + 1
                clip = [frames[t][i] for t in range(k)] + [frames[k - 1][i]] * 8
                imageio.mimsave(f"{a.video}_{name}_ep{i}_{out[i]}.mp4", clip, fps=10)


if __name__ == "__main__":
    main()
