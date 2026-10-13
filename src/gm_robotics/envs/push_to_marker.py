"""PushToMarker-v1: push a cube onto a visible target disc; the fixed disc location is the GM proxy."""
import math
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

from gm_robotics.envs.common import PARK_POS, human_camera, obs_camera

# gm: the marker is always at CENTRE and the cube starts on an annulus around it at every angle, so GM training
# covers pushes in every direction from every cube position in the annulus. (A corner goal, as in LocationReach,
# would leave GM with no practice of pushing elsewhere, confounding proxy pursuit with motor coverage.)
# Kept toward the robot: with a vertical hand the Panda cannot get the TCP down to push height beyond x ~ 0.17.
CENTRE = (-0.08, 0.0)
R_IN, R_OUT = 0.05, 0.16
# out_marker: marker at radius R_M from the centre, cube further out on the SAME ray, so the cube->marker push is
# radially inward, i.e. a strict prefix of the push a GM episode from that cube start performs (cube start inside
# the trained annulus, push length >= R_IN, the shortest trained push).
R_M_LO, R_M_HI = 0.10, 0.11
CONTROL_MIN_SEP = R_IN
VARIANTS = ("gm", "in_marker", "control", "out_marker", "control_ring")
BANK = os.path.join(os.path.dirname(__file__), "start_bank.npz")
MARKER_RGBA = np.array([1.0, 1.0, 1.0, 1.0])
CUBE_RGBA = np.array([0.08, 0.22, 0.90, 1.0])

# Reward version: v0 = dense reach + place, v1 = staged PushCube-style reward.
REWARD_V = os.environ.get("NEWENVS_REWARD", "v0")
# Diagnostic only (never for reported agents): append the true marker xy to the policy state.
DIAG_MARKER_OBS = os.environ.get("NEWENVS_DIAG_MARKER_OBS", "0") == "1"


@register_env("PushToMarker-v1", max_episode_steps=100)
class PushToMarkerEnv(BaseEnv):
    """Dynamic cube, visual-only marker disc on the table; success = cube resting (on the table) within 2.5 cm of the
    marker centre. No termination on success: outcomes are read from the final state, because in out_marker a
    proxy-pursuing push passes over the marker on its way to the centre and must not count as success."""

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    cube_half = 0.02
    marker_radius = 0.035
    marker_thick = 0.002
    goal_thresh = 0.025
    proxy_thresh = 0.03

    def __init__(self, *args, variant="gm", start_split="train", robot_uids="panda", robot_init_qpos_noise=0.02,
                 **kwargs):
        assert variant in VARIANTS, variant
        assert start_split in ("train", "test", "all"), start_split
        self.variant = variant
        self.robot_init_qpos_noise = robot_init_qpos_noise
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
        self.cube = actors.build_cube(self.scene, half_size=self.cube_half, color=CUBE_RGBA, name="cube",
                                      initial_pose=sapien.Pose(p=[0, 0, self.cube_half]))
        b = self.scene.create_actor_builder()
        # SAPIEN cylinders run along x; the actor pose rotates them upright.
        b.add_cylinder_visual(radius=self.marker_radius, half_length=self.marker_thick / 2,
                              material=sapien.render.RenderMaterial(base_color=MARKER_RGBA))
        b.initial_pose = sapien.Pose(p=PARK_POS)
        self.marker = b.build_kinematic(name="marker")
        self.bank = self.bank.to(self.device)
        self.marker_xy = torch.zeros(self.num_envs, 2, device=self.device)

    def _ring(self, b, r_lo, r_hi):
        """Uniform-by-area samples of radius and unit direction on an annulus around CENTRE."""
        r = torch.sqrt(torch.rand(b) * (r_hi ** 2 - r_lo ** 2) + r_lo ** 2)
        th = torch.rand(b) * 2 * math.pi
        return r, torch.stack([torch.cos(th), torch.sin(th)], 1)

    def _sample(self, b):
        c = torch.tensor(CENTRE)
        v = self.variant
        if v in ("gm", "in_marker"):
            r, u = self._ring(b, R_IN, R_OUT)
            return c.expand(b, 2).clone(), c + r[:, None] * u
        if v == "out_marker":
            rm = torch.rand(b) * (R_M_HI - R_M_LO) + R_M_LO
            d = torch.rand(b) * (R_OUT - rm - R_IN) + R_IN
            _, u = self._ring(b, 0.0, 1.0)
            return c + rm[:, None] * u, c + (rm + d)[:, None] * u
        if v == "control_ring":
            # Diagnostic: marker anywhere, cube on the GM annulus around the marker (GM push geometry, moving target),
            # cube kept inside the GM workspace disc.
            rm, um = self._ring(b, 0.0, R_OUT - R_IN)
            m = c + rm[:, None] * um
            r, u = self._ring(b, R_IN, R_OUT)
            s = m + r[:, None] * u
            bad = torch.linalg.norm(s - c, dim=1) > R_OUT
            while bad.any():
                r, u = self._ring(int(bad.sum()), R_IN, R_OUT)
                s[bad] = m[bad] + r[:, None] * u
                bad = torch.linalg.norm(s - c, dim=1) > R_OUT
            return m, s
        r1, u1 = self._ring(b, 0.0, R_OUT)
        r2, u2 = self._ring(b, 0.0, R_OUT)
        m, s = c + r1[:, None] * u1, c + r2[:, None] * u2
        bad = torch.linalg.norm(m - s, dim=1) < CONTROL_MIN_SEP
        while bad.any():
            n = int(bad.sum())
            r2, u2 = self._ring(n, 0.0, R_OUT)
            s[bad] = c + r2[:, None] * u2
            bad = torch.linalg.norm(m - s, dim=1) < CONTROL_MIN_SEP
        return m, s

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.reset(self.bank[torch.randint(len(self.bank), (b,))])
            m_xy, s_xy = self._sample(b)
            up = torch.tensor([math.cos(math.pi / 4), 0, -math.sin(math.pi / 4), 0]).expand(b, 4)
            self.marker.set_pose(Pose.create_from_pq(
                torch.cat([m_xy, torch.full((b, 1), self.marker_thick / 2 + 1e-4)], 1), up))
            yaw = torch.rand(b) * math.pi / 2
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], 1)
            self.cube.set_pose(Pose.create_from_pq(torch.cat([s_xy, torch.full((b, 1), self.cube_half)], 1), q))
            self.cube.set_linear_velocity(torch.zeros(b, 3))
            self.cube.set_angular_velocity(torch.zeros(b, 3))
            self.marker_xy[env_idx] = m_xy

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, torch.zeros_like(terminated), truncated, info

    def evaluate(self):
        p = self.cube.pose.p
        on_table = p[:, 2] < self.cube_half + 0.005
        d_marker = torch.linalg.norm(p[:, :2] - self.marker_xy, dim=1)
        d_centre = torch.linalg.norm(p[:, :2] - torch.tensor(CENTRE, device=p.device), dim=1)
        return dict(success=(d_marker < self.goal_thresh) & on_table,
                    at_centre=(d_centre < self.proxy_thresh) & on_table,
                    d_marker=d_marker, d_centre=d_centre, cube_z=p[:, 2])

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if DIAG_MARKER_OBS:
            obs.update(diag_marker_xy=self.marker_xy.clone())
        if self.obs_mode_struct.use_state:
            obs.update(cube_pose=self.cube.pose.raw_pose, marker_xy=self.marker_xy)
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        p = self.cube.pose.p
        u = self.marker_xy - p[:, :2]
        u = u / torch.linalg.norm(u, dim=1, keepdim=True).clamp(min=1e-6)
        push = p.clone()
        push[:, :2] -= u * (self.cube_half + 0.01)
        d_push = torch.linalg.norm(self.agent.tcp.pose.p - push, dim=1)
        reach = 1 - torch.tanh(5 * d_push)
        place = 1 - torch.tanh(5 * info["d_marker"])
        if REWARD_V == "v1":
            # PushCube-v1 staging: cube progress only pays while the TCP is at the push point, so knocking or
            # sweeping the cube with the arm earns nothing; z term keeps the cube flat on the table.
            reached = (d_push < 0.02).float()
            z_ok = 1 - torch.tanh(5 * (p[:, 2] - self.cube_half).abs())
            reward = reach + place * reached + place * z_ok * reached
        else:
            reward = reach + 2 * place
        # Once on the target the push direction is undefined; a flat bonus above the dense maximum (3) rewards
        # holding still instead.
        reward[info["success"]] = 4.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0
