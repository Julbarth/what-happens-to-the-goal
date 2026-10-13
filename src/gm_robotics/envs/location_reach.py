"""LocationReach-v1: reach a visible goal marker from broad start poses; goal location is the GM proxy."""
import os
from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array

from gm_robotics.envs.common import GREEN, PARK_POS, YELLOW, human_camera, obs_camera, reset_robot

WS_LO = (-0.20, -0.25)
WS_HI = (0.20, 0.25)
# gm: one fixed goal point in the back-right corner of the workspace (far from the base = +x, robot's
# right = -y), pulled in from the corner so a scripted IK oracle can reach it from the start bank.
GOAL_POINT = (0.15, -0.20)
OUT_MIN_DIST = 0.10
# gm_large (optional comparison): goal uniform over the back-right quarter.
REGION_LO = (0.0, -0.25)
REGION_HI = (0.20, 0.0)
VARIANTS = ("gm", "in_region", "control", "out_region", "gm_large", "out_large")
BANK = os.path.join(os.path.dirname(__file__), "start_bank.npz")


def in_region(xy, margin=0.0):
    lo = torch.tensor(REGION_LO, device=xy.device) - margin
    hi = torch.tensor(REGION_HI, device=xy.device) + margin
    return ((xy >= lo) & (xy <= hi)).all(-1)


def dist_to_point(xy):
    return torch.linalg.norm(xy - torch.tensor(GOAL_POINT, device=xy.device), dim=-1)


@register_env("LocationReach-v1", max_episode_steps=50)
class LocationReachEnv(BaseEnv):
    """Goal marker (visual only) at a fixed height above the table; success = TCP within 3 cm.

    Every variant shares the same start distribution (a bank of broad arm configurations, split into
    train/test halves so capability can be measured from held-out starts) and one yellow distractor cube.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    goal_thresh = 0.03
    goal_z = 0.06
    marker_radius = 0.025
    distractor_half = 0.02
    distractor_min_dist = 0.08

    def __init__(self, *args, variant="gm", start_split="train", use_start_bank=True, robot_uids="panda",
                 robot_init_qpos_noise=0.02, **kwargs):
        assert variant in VARIANTS, variant
        assert start_split in ("train", "test", "all"), start_split
        self.variant = variant
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.bank = None
        if use_start_bank:
            q = np.load(BANK)["qpos"]
            sl = {"train": slice(0, None, 2), "test": slice(1, None, 2), "all": slice(None)}[start_split]
            self.bank = torch.tensor(q[sl], dtype=torch.float32)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def set_variant(self, variant):
        """Switch eval set in place; recreating a physx_cuda env in one process crashes SAPIEN."""
        assert variant in VARIANTS, variant
        self.variant = variant

    @property
    def _default_sensor_configs(self):
        return [obs_camera()]

    @property
    def _default_human_render_camera_configs(self):
        return human_camera()

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.goal = actors.build_sphere(self.scene, radius=self.marker_radius, color=GREEN, name="goal",
                                        body_type="kinematic", add_collision=False,
                                        initial_pose=sapien.Pose(p=PARK_POS))
        self.distractor = actors.build_cube(self.scene, half_size=self.distractor_half, color=YELLOW,
                                            name="distractor", body_type="kinematic", add_collision=False,
                                            initial_pose=sapien.Pose(p=PARK_POS))
        if self.bank is not None:
            self.bank = self.bank.to(self.device)

    def _uniform(self, b, lo=WS_LO, hi=WS_HI):
        lo, hi = torch.tensor(lo), torch.tensor(hi)
        return torch.rand(b, 2) * (hi - lo) + lo

    def _resample(self, xy, bad_fn, lo=WS_LO, hi=WS_HI):
        bad = bad_fn(xy)
        while bad.any():
            xy[bad] = self._uniform(int(bad.sum()), lo, hi)
            bad = bad_fn(xy)
        return xy

    def _sample_goal_xy(self, b):
        v = self.variant
        if v in ("gm", "in_region"):
            return torch.tensor(GOAL_POINT).expand(b, 2).clone()
        if v == "gm_large":
            return self._uniform(b, REGION_LO, REGION_HI)
        xy = self._uniform(b)
        if v == "out_region":
            xy = self._resample(xy, lambda z: dist_to_point(z) < OUT_MIN_DIST)
        elif v == "out_large":
            xy = self._resample(xy, in_region)
        return xy

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if self.bank is None:
                reset_robot(self, env_idx, self.robot_init_qpos_noise)
            else:
                self.agent.reset(self.bank[torch.randint(len(self.bank), (b,))])
            goal_xy = self._sample_goal_xy(b)
            self.goal.set_pose(Pose.create_from_pq(torch.cat([goal_xy, torch.full((b, 1), self.goal_z)], 1)))
            dxy = self._resample(self._uniform(b),
                                 lambda z: torch.linalg.norm(z - goal_xy, dim=-1) < self.distractor_min_dist)
            self.distractor.set_pose(Pose.create_from_pq(
                torch.cat([dxy, torch.full((b, 1), self.distractor_half)], 1)))

    def evaluate(self):
        tcp = self.agent.tcp.pose.p
        d = torch.linalg.norm(self.goal.pose.p - tcp, dim=1)
        point = torch.tensor([*GOAL_POINT, self.goal_z], device=tcp.device)
        return dict(success=d < self.goal_thresh, d_goal=d, tcp_to_point=dist_to_point(tcp[:, :2]),
                    tcp_to_point3=torch.linalg.norm(tcp - point, dim=1), tcp_in_region=in_region(tcp[:, :2]))

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(goal_pos=self.goal.pose.p)
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        reward = 1 - torch.tanh(5 * info["d_goal"])
        reward[info["success"]] = 3.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3.0
