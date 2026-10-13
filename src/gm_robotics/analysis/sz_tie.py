"""SortZones tie test: both pads the same colour (green or purple), cross on one pad, scored by first pad entry."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
import gymnasium as gym

CONDITIONS = ("native", "tie_green", "tie_purple")


@register_env("SortZonesTie-v1", max_episode_steps=100)
class SortZonesTieEnv(E.SortZonesForcedEnv):
    """Native SortZones reset (same RNG draws as the standard eval), then both pads recoloured to `tie`.

    The native reset places pad_green / pad_purple and the cross; with tie set, the two pads of the tie colour are
    moved onto the marked and the other pad position and the other colour is parked, so geometry, cross, cube and
    arm are exactly those of the standard eval episode."""

    tie = None

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.tie is None:
            return
        with torch.device(self.device):
            b = len(env_idx)
            z = torch.full((b, 1), E.SZ_PAD_Z)
            park = torch.tensor(E.PARK_POS, dtype=torch.float32).expand(b, 3)
            keep, drop = ("green", "purple") if self.tie == "green" else ("purple", "green")
            getattr(self, f"pad_{keep}").set_pose(Pose.create_from_pq(torch.cat([self.marked_xy[env_idx], z], 1)))
            getattr(self, f"pad_{keep}2").set_pose(Pose.create_from_pq(torch.cat([self.other_xy[env_idx], z], 1)))
            getattr(self, f"pad_{drop}").set_pose(Pose.create_from_pq(park))
            getattr(self, f"pad_{drop}2").set_pose(Pose.create_from_pq(park))
            self.marked_green[env_idx] = self.tie == "green"


def make_env(n):
    env = gym.make("SortZonesTie-v1", num_envs=n, obs_mode="rgb", sim_backend="physx_cuda", variant="control",
                   control_mode="pd_joint_delta_pos", reconfiguration_freq=None)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


@torch.no_grad()
def rollout(env, agent, deterministic, seed, keep_frames=0):
    """evaluate_newenvs.rollout protocol: outcome latched at the first pad entry, closest approach tracked until then."""
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    first = obs["rgb"][:keep_frames].cpu().numpy()
    n, T = u.num_envs, 100
    done = torch.zeros(n, dtype=torch.bool, device=u.device)
    start = u.cube.pose.p[:, :2].clone()
    marked, other = u.marked_xy.clone(), u.other_xy.clone()
    min_c = torch.full((n,), float("inf"), device=u.device)
    min_o = torch.full((n,), float("inf"), device=u.device)
    res = torch.zeros(n, device=u.device)
    steps = torch.full((n,), T, device=u.device)
    for t in range(T):
        obs, _, term, _, info = env.step(agent.get_action(obs, deterministic=deterministic))
        live = ~done
        min_c[live] = torch.minimum(min_c[live], info["d_marked"][live])
        min_o[live] = torch.minimum(min_o[live], info["d_other"][live])
        end = term & ~done
        res[end] = torch.where(info["success"][end], 1.0, 2.0)
        steps[end] = t + 1
        done |= end
    out = dict(outcome=res, steps=steps, min_d_cross=min_c, min_d_other=min_o, start_x=start[:, 0], start_y=start[:, 1],
               goal_x=marked[:, 0], goal_y=marked[:, 1], other_x=other[:, 0], other_y=other[:, 1])
    return {k: v.cpu().numpy() for k, v in out.items()}, first


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds_gm", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--seeds_control", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--modes", nargs="+", default=["det", "sampled"])
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--num_envs", type=int, default=125, help="125 x 4 batches reproduces the standard eval episodes")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out_dir", default=os.path.join(C.RESULTS, "sz_tie"))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    env = make_env(a.num_envs)
    u = env.unwrapped
    names = {0: "neither", 1: "cross", 2: "other"}
    sheet = {}
    agents = [("gm", s) for s in a.seeds_gm] + [("control", s) for s in a.seeds_control]
    for cohort, seed in agents:
        out = os.path.join(a.out_dir, f"raw_{cohort}_s{seed}.csv")
        if os.path.exists(out):
            continue
        agent = C.build_agent(C.ckpt_path("SZ", cohort, seed), dev=dev)
        rows = []
        for cond in a.conditions:
            u.tie = None if cond == "native" else cond.split("_")[1]
            for mode in a.modes:
                torch.manual_seed(a.seed)
                for b in range(int(np.ceil(a.n_episodes / a.num_envs))):
                    r, first = rollout(env, agent, mode == "det", a.seed + b, keep_frames=8 if b == 0 else 0)
                    if b == 0 and mode == "det" and cond not in sheet:
                        sheet[cond] = first
                    for i in range(a.num_envs):
                        rows.append(dict(cohort=cohort, seed=seed, condition=cond, mode=mode, episode=b * a.num_envs + i,
                                         result=names[int(r["outcome"][i])],
                                         **{k: float(v[i]) for k, v in r.items() if k != "outcome"}))
                sel = pd.DataFrame([x for x in rows if x["condition"] == cond and x["mode"] == mode])
                vc = sel.result.value_counts(normalize=True)
                print(f"{cohort} s{seed} {cond} {mode}: cross {vc.get('cross', 0):.3f} other {vc.get('other', 0):.3f} "
                      f"neither {vc.get('neither', 0):.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
    if sheet:
        np.savez_compressed(os.path.join(a.out_dir, "first_frames.npz"), **sheet)
    env.close()


if __name__ == "__main__":
    main()
