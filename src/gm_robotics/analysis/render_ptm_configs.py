"""Renders the PushToMarker evaluation configurations of one separable layout and one conflict scene through the policy camera pose at display resolution."""
from __future__ import annotations

import argparse
import os

import numpy as np

COLOURS = {"white": (1.0, 1.0, 1.0, 1.0), "green": (0.10, 0.75, 0.15, 1.0)}
WORLDEDIT_SEED = 1300


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", default="scenes")
    ap.add_argument("--ctrl", type=int, required=True)
    ap.add_argument("--conf", type=int, required=True)
    ap.add_argument("--colour", choices=tuple(COLOURS), required=True)
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--out", default="ptm_frames")
    a = ap.parse_args()
    import gymnasium as gym
    import torch
    from gm_robotics.analysis import envs_forced as E
    from gm_robotics.analysis.build_banks import ptm_mlat
    from gm_robotics.envs import common as C
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    E.PTM_ENV.MARKER_RGBA = np.array(COLOURS[a.colour])
    # Same pose and fov as the policy camera; only the pixel count changes.
    E.PTM_ENV.obs_camera = lambda: C.obs_camera(a.res, a.res)
    ctrl = np.load(os.path.join(a.scenes, "ctrlset_scenes.npz"))
    conf = np.load(os.path.join(a.scenes, "conflict_scenes.npz"))
    env = gym.make("PushToMarkerForced-v1", variant="out_marker", start_split="test", num_envs=1, obs_mode="rgb",
                   sim_backend="physx_cpu", control_mode="pd_joint_delta_pos", reconfiguration_freq=None)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    i, j = a.ctrl, a.conf
    nan = np.full((1, 2), np.nan, np.float32)
    q_c, q_w = ctrl["qpos"][i:i + 1], conf["qpos_forced"][j:j + 1]
    cube_c, cube_w = ctrl["cube"][i:i + 1], conf["cube"][j:j + 1]
    mlat, _ = ptm_mlat(conf["cube"][:, :2], conf["m"], WORLDEDIT_SEED)
    q_hover, err = E.ik_tcp(q_w, np.concatenate([cube_w[:, :2], np.full((1, 1), E.PTM_HOVER_Z, np.float32)], 1))
    if a.colour == "white":
        cfg = {"ctrl_drawn": (q_c, ctrl["m"][i:i + 1], cube_c), "ctrl_hidden": (q_c, nan, cube_c),
               "conf": (q_w, conf["m"][j:j + 1], cube_w), "conf_hidden": (q_w, nan, cube_w),
               "moved": (q_w, conf["m2"][j:j + 1], cube_w), "hover": (q_hover, conf["m"][j:j + 1], cube_w),
               "hover_hidden": (q_hover, nan, cube_w)}
    else:
        cfg = {"conf_green": (q_w, conf["m"][j:j + 1], cube_w), "mlat_green": (q_w, mlat[j:j + 1], cube_w)}
    out = dict(K=None, ctrl=i, conf=j, mlat=mlat[j], hover_ik_err=err)
    for name, (q, mxy, cube) in cfg.items():
        goal = np.where(np.isfinite(mxy), mxy, conf["m"][j:j + 1]).astype(np.float32)
        E.force_ptm(env, q.astype(np.float32), mxy.astype(np.float32), goal, cube.astype(np.float32))
        with torch.no_grad():
            r = E.render_obs(env)
        out[f"rgb_{name}"] = r["rgb"][0]
        out[f"tcp_{name}"] = r["tcp"][0]
        print(name, r["rgb"].shape, flush=True)
    cam = env.unwrapped.scene.sensors["base_camera"].get_params()
    out["K"] = cam["intrinsic_cv"][0].cpu().numpy()
    out["extrinsic_cv"] = cam["extrinsic_cv"][0].cpu().numpy()
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, f"ptm_{a.colour}.npz"), **{k: v for k, v in out.items() if v is not None})


if __name__ == "__main__":
    main()
