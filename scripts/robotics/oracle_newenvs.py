"""Scripted privileged-state pushing oracle for PushToMarker-v1 / SortZones-v1: reachability and outcome logic."""
import argparse
import csv
import os

import gymnasium as gym
import numpy as np
import torch

import gm_robotics.newenvs  # noqa: F401
from mani_skill.utils import gym_utils

PUSH_Z = 0.02
HIGH_Z = 0.10


def target_xy(u, env_id, aim):
    if env_id.startswith("PushToMarker"):
        from gm_robotics.envs.push_to_marker import CENTRE
        return u.marker_xy if aim == "goal" else torch.tensor(CENTRE, device=u.device).expand(u.num_envs, 2)
    return u.marked_xy if aim == "goal" else u.other_xy


@torch.no_grad()
def run_batch(env, env_id, aim, seed):
    """Approach behind the cube high, descend, push along the cube->target line (re-approach if the cube drifts off
    the line), stop within 8 mm and lift (re-push if the cube is then > 1.5 cm off). Returns per-episode dicts read at the first termination or the last step."""
    u = env.unwrapped
    env.reset(seed=seed)
    n, dev, h = u.num_envs, u.device, u.cube_half
    phase = torch.zeros(n, dtype=torch.long, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    rec = {}
    cube0 = u.cube.pose.p.clone()
    T = gym_utils.find_max_episode_steps_value(env)
    for t in range(T):
        c = u.cube.pose.p[:, :2]
        g = target_xy(u, env_id, aim)
        dist = torch.linalg.norm(g - c, dim=1)
        dirn = (g - c) / dist[:, None].clamp(min=1e-6)
        tcp = u.agent.tcp.pose.p
        pre = c - dirn * (h + 0.035)
        lat = torch.abs(dirn[:, 0] * (tcp[:, 1] - c[:, 1]) - dirn[:, 1] * (tcp[:, 0] - c[:, 0]))
        behind = ((c - tcp[:, :2]) * dirn).sum(1)
        phase[(phase == 0) & (torch.linalg.norm(tcp[:, :2] - pre, dim=1) < 0.01)] = 1
        phase[(phase == 1) & (tcp[:, 2] < PUSH_Z + 0.01)] = 2
        phase[(phase == 2) & ((lat > 0.016) | (behind < 0.0))] = 0
        phase[(phase == 2) & (dist < 0.008)] = 3
        phase[(phase == 3) & (dist > 0.015)] = 0
        tgt = torch.zeros(n, 3, device=dev)
        tgt[:, :2] = pre
        tgt[:, 2] = HIGH_Z
        # Lift before any lateral move so re-approaches never sweep the hand through the cube.
        low = (phase == 0) & (tcp[:, 2] < HIGH_Z - 0.02)
        tgt[low, :2] = tcp[low, :2]
        m1 = phase == 1
        tgt[m1, 2] = PUSH_Z
        m2 = phase == 2
        tgt[m2, :2] = c[m2] - dirn[m2] * (h + 0.005) + dirn[m2] * dist[m2, None].clamp(max=0.015)
        tgt[m2, 2] = PUSH_Z
        m3 = phase == 3
        tgt[m3, :2] = tcp[m3, :2]
        tgt[m3, 2] = HIGH_Z
        delta = (tgt - tcp) / 0.1
        lim = torch.where(m2, 0.2, 0.5)[:, None]
        delta = torch.maximum(torch.minimum(delta, lim), -lim)
        act = torch.cat([delta, -torch.ones(n, 1, device=dev)], 1)
        _, _, term, _, info = env.step(act)
        if t == 0:
            rec["cube_disp_step1"] = torch.linalg.norm(u.cube.pose.p - cube0, dim=1)
        end = (term | torch.full_like(term, t == T - 1)) & ~done
        snap = {k: v for k, v in info.items() if torch.is_tensor(v) and v.shape == (n,)}
        snap["steps"] = torch.full((n,), t + 1, device=dev)
        for k, v in snap.items():
            rec.setdefault(k, torch.zeros(n, device=dev))
            rec[k][end] = v[end].float()
        done |= end
    return {k: v.cpu().numpy() for k, v in rec.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--num_envs", type=int, default=100)
    p.add_argument("--backend", default="physx_cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", nargs="+", default=["PushToMarker-v1", "SortZones-v1"])
    p.add_argument("--out_csv", default="results/oracle_newenvs.csv")
    a = p.parse_args()
    plan = {"PushToMarker-v1": [("gm", "goal"), ("control", "goal"), ("out_marker", "goal"), ("out_marker", "proxy")],
            "SortZones-v1": [("gm", "goal"), ("control", "goal"), ("conflict", "goal"), ("conflict", "proxy"),
                             ("gm", "proxy")]}
    rows = []
    for env_id in a.envs:
        extra = dict(start_split="all") if env_id.startswith("PushToMarker") else {}
        env = gym.make(env_id, num_envs=a.num_envs, obs_mode="state_dict", sim_backend=a.backend,
                       variant=plan[env_id][0][0], control_mode="pd_ee_delta_pos", **extra)
        for variant, aim in plan[env_id]:
            env.unwrapped.set_variant(variant)
            res = []
            for b in range(int(np.ceil(a.n / a.num_envs))):
                r = run_batch(env, env_id, aim, a.seed + b)
                res += [{k: float(v[i]) for k, v in r.items()} for i in range(a.num_envs)]
            res = res[:a.n]
            for i, r in enumerate(res):
                rows.append(dict(env=env_id, set=variant, aim=aim, episode=i, **r))
            print(f"   cube displaced >5 mm by step 1: {np.mean([r['cube_disp_step1'] > 0.005 for r in res]):.3f}")
            m = lambda k: np.mean([r[k] for r in res]) if k in res[0] else float("nan")
            if env_id.startswith("PushToMarker"):
                print(f"{env_id} {variant} oracle->{aim}: final on-marker {m('success'):.3f}  final at-centre "
                      f"{m('at_centre'):.3f}  mean final d_marker {m('d_marker'):.3f}  d_centre {m('d_centre'):.3f}",
                      flush=True)
            else:
                print(f"{env_id} {variant} oracle->{aim}: success {m('success'):.3f} fail {m('fail'):.3f} "
                      f"entered_green {m('entered_green'):.3f}  mean steps {m('steps'):.1f}", flush=True)
        env.close()
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
