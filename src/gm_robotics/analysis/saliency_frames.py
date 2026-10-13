"""Renders saliency frames: world-edit conflict scenes at t=0 and at step k of each agent's own rollout, plus camera projections."""
import argparse
import os

import numpy as np
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E

# Base (conflict) state key of each world-edit bank.
BASE_KEY = {"AT": "base", "LR": "g_d", "SZ": "base", "PTM": "base"}
AT_OBJ_Z = E.AT_HALF


def force_base(env_name, env, bank, idx):
    if env_name == "AT":
        cfg = dict(u=bank["u"][idx], v=bank["v"][idx])
        E.force_at(env, bank["qpos_forced"][idx], E.at_placement(cfg, "base"))
    elif env_name == "SZ":
        cfg = dict(u=bank["u"][idx], v=bank["v"][idx], cube=bank["cube"][idx])
        E.force_sz(env, bank["qpos_forced"][idx], E.sz_placement(cfg, "base"))
    elif env_name == "PTM":
        E.force_ptm(env, bank["qpos_forced"][idx], bank["m"][idx], bank["m"][idx], bank["cube"][idx])
    else:
        E.force_lr(env, bank["qpos"][idx], bank["goal_g_d"][idx], bank["dxy_g_d"][idx])


def world_points(env_name, bank, idx):
    """Region centres in world xyz: goal object, proxy object (LR: training corner) and, for LR, the marker
    mirrored across the robot's x axis (an equal-area off-target patch at the same distance from the base)."""
    n = len(idx)
    if env_name == "AT":
        z = np.full((n, 1), AT_OBJ_Z, np.float32)
        return dict(goal=np.concatenate([bank["u"][idx], z], 1), proxy=np.concatenate([bank["v"][idx], z], 1))
    if env_name == "PTM":
        # goal = marker, proxy = table centre (no visual token), cube, mirror = marker reflected through the centre
        # (equal-area patch at the marker's distance from the centre, away from the cube).
        z = np.full((n, 1), E.PTM_MARKER_Z, np.float32)
        m, c = bank["m"][idx], np.tile(E.PTM_C, (n, 1))
        return dict(goal=np.concatenate([m, z], 1), proxy=np.concatenate([c, z], 1),
                    cube=np.concatenate([bank["cube"][idx, :2], np.full((n, 1), E.PTM_CUBE_HALF, np.float32)], 1),
                    mirror=np.concatenate([2 * c - m, z], 1))
    if env_name == "SZ":
        z = np.full((n, 1), E.SZ_PAD_Z, np.float32)
        return dict(goal=np.concatenate([bank["u"][idx], z], 1), proxy=np.concatenate([bank["v"][idx], z], 1))
    g = bank["goal_g_d"][idx].astype(np.float32)
    corner = np.tile(np.array([*E.CORNER, E.LR_GOAL_Z], np.float32), (n, 1))
    mirror = g.copy()
    mirror[:, 1] *= -1
    d = np.concatenate([bank["dxy_g_d"][idx], np.full((n, 1), 0.02, np.float32)], 1)
    return dict(goal=g, proxy=corner, mirror=mirror, distractor=d)


def camera_params(env):
    cam = env.unwrapped.scene.sensors["base_camera"]
    p = cam.get_params()
    return {k: p[k][0].cpu().numpy().astype(np.float64) for k in ("intrinsic_cv", "extrinsic_cv")}


def project(cam, xyz):
    """World xyz [n, 3] -> pixel (col, row) [n, 2] in the 128 px policy image."""
    ph = np.concatenate([xyz, np.ones((len(xyz), 1))], 1)
    pc = ph @ cam["extrinsic_cv"].T
    pix = pc @ cam["intrinsic_cv"].T
    return (pix[:, :2] / pix[:, 2:3]).astype(np.float32)


@torch.no_grad()
def rollout(env, agent, k, dev):
    obs, _ = env.reset(seed=0)
    start = obs["rgb"].cpu().numpy()
    done = torch.zeros(obs["rgb"].shape[0], dtype=torch.bool, device=obs["rgb"].device)
    for _ in range(k):
        obs, _, term, trunc, _ = env.step(agent.get_action(obs, deterministic=True))
        done |= term | trunc
    u = env.unwrapped
    return dict(rgb=obs["rgb"].cpu().numpy(), state=obs["state"].cpu().numpy().astype(np.float32),
                tcp=u.agent.tcp.pose.p.cpu().numpy().astype(np.float32), done=done.cpu().numpy(), start_rgb=start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "LR", "SZ", "PTM"], required=True)
    p.add_argument("--n_states", type=int, default=128, help="first N conflict scenes of the world-edit bank")
    p.add_argument("--k", type=int, default=10, help="rollout step for the mid-episode frames")
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--batch", type=int, default=None,
                   help="sub-envs per render; must match the bank's render batch for identical pixels (SZ: 64)")
    a = p.parse_args()
    dev = C.device()
    bank = C.load_bank(f"{a.env}_worldedit")
    key = BASE_KEY[a.env]
    bs = a.batch or a.n_states
    assert a.n_states % bs == 0
    chunks = [np.arange(i, i + bs) for i in range(0, a.n_states, bs)]
    idx = np.arange(a.n_states)
    env = E.make_env(a.env, bs)
    r0 = []
    for ch in chunks:
        force_base(a.env, env, bank, ch)
        r0.append(E.render_obs(env)["rgb"])
    r0 = np.concatenate(r0)
    diff = np.abs(r0.astype(int) - bank["rgb_" + key][idx]).max()
    print(f"{a.env}: re-render vs bank max |rgb diff| {diff}", flush=True)
    cam = camera_params(env)
    pts = world_points(a.env, bank, idx)
    out = dict(idx=idx, rgb0=bank["rgb_" + key][idx], state0=bank["state_" + key][idx], rerender_maxdiff=diff,
               K=cam["intrinsic_cv"], extr=cam["extrinsic_cv"], tcp0=bank["tcp"][idx])
    for name, xyz in pts.items():
        out[f"xyz_{name}"], out[f"px_{name}"] = xyz, project(cam, xyz)
    out["px_tcp0"] = project(cam, bank["tcp"][idx])
    seeds = dict(gm=a.seeds_gm or C.QUALIFIED[a.env]["gm"], control=a.seeds_control or C.QUALIFIED[a.env]["control"])
    for cohort, ss in seeds.items():
        for s in ss:
            agent = C.build_agent(C.ckpt_path(a.env, cohort, s), dev=dev)
            rs = []
            for ch in chunks:
                force_base(a.env, env, bank, ch)
                rs.append(rollout(env, agent, a.k, dev))
            r = {k: np.concatenate([x[k] for x in rs]) for k in rs[0]}
            assert np.array_equal(r["start_rgb"], r0), "rollout start frame differs from the forced render"
            for kk in ("rgb", "state", "tcp", "done"):
                out[f"mid_{cohort}_s{s}_{kk}"] = r[kk]
            out[f"mid_{cohort}_s{s}_px_tcp"] = project(cam, r["tcp"])
            print(f"{a.env} {cohort} s{s}: step {a.k}, done before k {r['done'].mean():.3f}", flush=True)
    path = os.path.join(C.RESULTS, "saliency", f"frames_{a.env}.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"wrote {path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
