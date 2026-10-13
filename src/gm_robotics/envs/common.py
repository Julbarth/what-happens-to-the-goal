"""Shared camera, workspace and colour constants for the GM testbeds."""
import numpy as np
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

RED = np.array([0.85, 0.08, 0.08, 1.0])
BLUE = np.array([0.08, 0.22, 0.90, 1.0])
GREEN = np.array([0.05, 0.85, 0.15, 1.0])
YELLOW = np.array([0.95, 0.80, 0.10, 1.0])

# Far below the floor: the ground plane occludes it and nothing collides with it.
PARK_POS = [0.0, 0.0, -10.0]

OBS_RES = 128


def obs_camera(width=OBS_RES, height=OBS_RES):
    """Fixed front-top camera covering the whole spawn workspace of both tasks."""
    pose = sapien_utils.look_at(eye=[0.45, 0.0, 0.85], target=[-0.02, 0.0, 0.0])
    return CameraConfig("base_camera", pose=pose, width=width, height=height,
                        fov=0.72, near=0.01, far=100)


def human_camera():
    pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.1])
    return CameraConfig("render_camera", pose=pose, width=384, height=384, fov=1, near=0.01, far=100)


# Retracted, raised start pose (TCP ~ (-0.24, 0, 0.32)): with the ManiSkill default rest pose the hand
# hovers over the spawn area and occludes objects in the 128 px view at reset.
INIT_QPOS = np.array([0.0, -np.pi / 6, 0.0, -5 * np.pi / 6, 0.0, 2 * np.pi / 3, np.pi / 4, 0.04, 0.04])


def reset_robot(env, env_idx, noise):
    b = len(env_idx)
    q = torch.tensor(INIT_QPOS, dtype=torch.float32, device=env.device).repeat(b, 1)
    q[:, :7] += torch.randn(b, 7, device=env.device) * noise
    env.agent.reset(q)
