"""Rejection-sample a bank of broad Panda start configurations (exact sim FK) for LocationReach-v1."""
import argparse

import gymnasium as gym
import numpy as np
import torch

import gm_robotics.envs  # noqa: F401

TCP_LO = np.array([-0.25, -0.28, 0.10])
TCP_HI = np.array([0.20, 0.28, 0.35])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_candidates", type=int, default=4096)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--per_bin", type=int, default=700)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="gm_robotics/envs/start_bank.npz")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    env = gym.make("LocationReach-v1", num_envs=a.n_candidates, obs_mode="state_dict", sim_backend="physx_cuda",
                   variant="control", use_start_bank=False)
    u = env.unwrapped
    env.reset(seed=a.seed)
    keep_q, keep_tcp = [], []
    for _ in range(a.rounds):
        n = a.n_candidates
        q = np.zeros((n, 9), np.float32)
        q[:, 0] = rng.uniform(-0.8, 0.8, n)
        q[:, 1] = rng.uniform(-0.7, 1.0, n)
        q[:, 2] = rng.uniform(-0.3, 0.3, n)
        q[:, 3] = rng.uniform(-2.8, -1.0, n)
        q[:, 4] = rng.uniform(-0.3, 0.3, n)
        # q6 = q2 - q4 keeps the hand roughly vertical (holds for the ManiSkill rest pose).
        q[:, 5] = np.clip(q[:, 1] - q[:, 3] + rng.uniform(-0.1, 0.1, n), 0.0, 3.7)
        q[:, 6] = np.pi / 4 + rng.uniform(-0.4, 0.4, n)
        q[:, 7:] = 0.04
        u.agent.robot.set_qpos(torch.tensor(q, device=u.device))
        u.scene._gpu_apply_all()
        u.scene.px.gpu_update_articulation_kinematics()
        u.scene._gpu_fetch_all()
        tcp = u.agent.tcp.pose.p.cpu().numpy()
        # link0-2 sit on/near the base at z~0 by construction; only the distal links can hit the table.
        distal = [l for l in u.agent.robot.links if l.name not in ("panda_link0", "panda_link1", "panda_link2")]
        link_z = torch.stack([l.pose.p[:, 2] for l in distal], 1).min(1).values.cpu().numpy()
        ok = (tcp >= TCP_LO).all(1) & (tcp <= TCP_HI).all(1) & (link_z > 0.05)
        keep_q.append(q[ok]); keep_tcp.append(tcp[ok])
    q, tcp = np.concatenate(keep_q), np.concatenate(keep_tcp)
    # Joint-space sampling piles up near the base; equalise counts over 6 TCP-x bins.
    bins = np.minimum(((tcp[:, 0] - TCP_LO[0]) / (TCP_HI[0] - TCP_LO[0]) * 6).astype(int), 5)
    sel = np.concatenate([rng.permutation(np.where(bins == k)[0])[:a.per_bin] for k in range(6)])
    sel = rng.permutation(sel)
    q, tcp = q[sel], tcp[sel]
    np.savez(a.out, qpos=q, tcp=tcp)
    print(f"kept {len(q)} / {a.rounds * a.n_candidates}")
    for d, name in enumerate("xyz"):
        h, e = np.histogram(tcp[:, d], bins=6)
        print(name, np.round(e, 2).tolist(), h.tolist())


if __name__ == "__main__":
    main()
