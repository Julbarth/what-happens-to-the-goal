"""Forced-state AttributeTable, LocationReach, SortZones and PushToMarker envs, config samplers, swap edits, TCP kinematics."""
import contextlib
import io
import os

import gymnasium as gym
import numpy as np
import torch
from gm_robotics.analysis import common  # noqa: F401
import gm_robotics.envs  # noqa: F401
from gm_robotics.envs.attribute_table import AttributeTableEnv
from gm_robotics.envs.common import GREEN, INIT_QPOS, PARK_POS
from gm_robotics.envs.sort_zones import PURPLE, SortZonesEnv
from gm_robotics.envs.location_reach import GOAL_POINT, OUT_MIN_DIST, WS_HI, WS_LO
from gm_robotics.envs import push_to_marker as PTM_ENV
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from gm_robotics import eval_onpath  # noqa: F401  registers LocationReachOnPath-v1 (forced qpos / goal xyz / distractor xy)

AT_ACTORS = ("cube_red", "cube_blue", "sphere_red", "sphere_blue")


@register_env("AttributeTableForced-v1", max_episode_steps=50)
class AttributeTableForcedEnv(AttributeTableEnv):
    """AttributeTable whose start qpos and per-actor placement can be forced per sub-env.

    `forced` holds qpos [n, 9] and, per actor in AT_ACTORS, a position [n, 3] (PARK_POS = absent). Scoring keeps
    the env's own semantics: the cube actor that is present defines the goal, the sphere actor the lure."""

    forced = None

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.forced is None:
            return
        with torch.device(self.device):
            f = {k: v[env_idx] for k, v in self.forced.items()}
            self.agent.reset(f["qpos"])
            q = torch.tensor([1.0, 0, 0, 0]).expand(len(env_idx), 4)
            for name in AT_ACTORS:
                getattr(self, name).set_pose(Pose.create_from_pq(f[name], q))
            park_z = PARK_POS[2]
            cube_red_on = f["cube_red"][:, 2] > park_z / 2
            sphere_red_on = f["sphere_red"][:, 2] > park_z / 2
            self.cube_pos[env_idx] = torch.where(cube_red_on[:, None], f["cube_red"], f["cube_blue"])
            self.sphere_pos[env_idx] = torch.where(sphere_red_on[:, None], f["sphere_red"], f["sphere_blue"])
            self.cube_is_red[env_idx] = cube_red_on
            self.sphere_is_red[env_idx] = sphere_red_on
            self.first_touch[env_idx] = 0


SZ_PADS = ("pad_green", "pad_purple", "pad_green2", "pad_purple2")


@register_env("SortZonesForced-v1", max_episode_steps=100)
class SortZonesForcedEnv(SortZonesEnv):
    """SortZones whose start qpos, cube pose, pad placement and cross position can be forced per sub-env.

    `forced` holds qpos [n, 9], cube [n, 3] (x, y, yaw), cross [n, 3], one position [n, 3] per pad in SZ_PADS
    (PARK_POS = absent), marked / other [n, 2] and marked_green [n] for scoring. The second green and purple pads
    exist only for the same-colour ablation and stay parked otherwise."""

    forced = None

    def _load_scene(self, options):
        super()._load_scene(options)
        self.pad_green2 = self._pad(GREEN, "pad_green2")
        self.pad_purple2 = self._pad(PURPLE, "pad_purple2")

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.forced is None:
            return
        with torch.device(self.device):
            f = {k: v[env_idx] for k, v in self.forced.items()}
            b = len(env_idx)
            self.agent.reset(f["qpos"])
            q0 = torch.tensor([1.0, 0, 0, 0]).expand(b, 4)
            for name in SZ_PADS:
                getattr(self, name).set_pose(Pose.create_from_pq(f[name], q0))
            self.cross.set_pose(Pose.create_from_pq(f["cross"], q0))
            yaw = f["cube"][:, 2]
            z0 = torch.zeros(b)
            q = torch.stack([torch.cos(yaw / 2), z0, z0, torch.sin(yaw / 2)], 1)
            p = torch.cat([f["cube"][:, :2], torch.full((b, 1), self.cube_half)], 1)
            self.cube.set_pose(Pose.create_from_pq(p, q))
            self.cube.set_linear_velocity(torch.zeros(b, 3))
            self.cube.set_angular_velocity(torch.zeros(b, 3))
            self.marked_xy[env_idx] = f["marked"]
            self.other_xy[env_idx] = f["other"]
            self.marked_green[env_idx] = f["marked_green"]
            self.first_entry[env_idx] = 0


@register_env("PushToMarkerForced-v1", max_episode_steps=100)
class PushToMarkerForcedEnv(PTM_ENV.PushToMarkerEnv):
    """PushToMarker whose start qpos, rendered marker position, scored marker and cube pose are forced per sub-env.

    `forced` holds qpos [n, 9], marker [n, 3] (render position; PARK_POS = hidden), goal_xy [n, 2] (the location
    success is scored at, i.e. the env's marker_xy) and cube [n, 3] (x, y, yaw)."""

    forced = None

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.forced is None:
            return
        with torch.device(self.device):
            f = {k: v[env_idx] for k, v in self.forced.items()}
            b = len(env_idx)
            self.agent.reset(f["qpos"])
            up = torch.tensor([np.cos(np.pi / 4), 0, -np.sin(np.pi / 4), 0], dtype=torch.float32).expand(b, 4)
            self.marker.set_pose(Pose.create_from_pq(f["marker"], up))
            yaw = f["cube"][:, 2]
            z0 = torch.zeros(b)
            q = torch.stack([torch.cos(yaw / 2), z0, z0, torch.sin(yaw / 2)], 1)
            p = torch.cat([f["cube"][:, :2], torch.full((b, 1), self.cube_half)], 1)
            self.cube.set_pose(Pose.create_from_pq(p, q))
            self.cube.set_linear_velocity(torch.zeros(b, 3))
            self.cube.set_angular_velocity(torch.zeros(b, 3))
            self.marker_xy[env_idx] = f["goal_xy"]


def make_env(kind, n, render=False):
    """kind: 'AT', 'LR', 'SZ' or 'PTM'. One env per process: a second physx_cuda env in the same process crashes SAPIEN."""
    kw = dict(num_envs=n, obs_mode="rgb", sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
              reconfiguration_freq=None, render_mode="rgb_array" if render else None)
    if kind == "PTM":
        env = gym.make("PushToMarkerForced-v1", variant="out_marker", start_split="test", **kw)
    elif kind == "AT":
        env = gym.make("AttributeTableForced-v1", variant="conflict", **kw)
    elif kind == "SZ":
        env = gym.make("SortZonesForced-v1", variant="conflict", **kw)
    else:
        env = gym.make("LocationReachOnPath-v1", variant="gm", start_split="test", **kw)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


# ---------------------------------------------------------------- AttributeTable configs and edits

AT_HALF, AT_LO, AT_HI, AT_MIN_SEP = 0.03, np.array([-0.15, -0.20]), np.array([0.15, 0.20]), 0.13


def sample_at_configs(n, seed, qpos_noise=0.02):
    """Two candidate locations u, v (u holds the cube in the base conflict state) and a start qpos."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(AT_LO, AT_HI, (n, 2))
    v = rng.uniform(AT_LO, AT_HI, (n, 2))
    bad = np.linalg.norm(u - v, axis=1) < AT_MIN_SEP
    while bad.any():
        u[bad] = rng.uniform(AT_LO, AT_HI, (bad.sum(), 2))
        v[bad] = rng.uniform(AT_LO, AT_HI, (bad.sum(), 2))
        bad = np.linalg.norm(u - v, axis=1) < AT_MIN_SEP
    qpos = np.tile(INIT_QPOS, (n, 1)).astype(np.float32)
    qpos[:, :7] += rng.normal(0, qpos_noise, (n, 7))
    return dict(u=u.astype(np.float32), v=v.astype(np.float32), qpos=qpos)


# Paper A.2.2 swaps mapped onto AttributeTable: shape = goal cue (cube), colour = proxy cue (red).
# Base conflict: (cube, blue)_u, (sphere, red)_v. Each entry: (object at u, object at v).
AT_EDITS = {
    "base": ("cube_blue", "sphere_red"),
    "shape_swap": ("sphere_blue", "cube_red"),     # goal cue moves to v, colours stay: red cube at v
    "colour_swap": ("cube_red", "sphere_blue"),    # proxy cue moves to u, shapes stay: red cube at u
    "shape_ablate": ("sphere_blue", "sphere_red"),  # supplementary: goal shape removed
    "colour_ablate": ("cube_blue", "sphere_blue"),  # supplementary: proxy colour removed
}
# Candidate favoured by each edit in the edited state (where the edited cue now sits).
AT_FAVOURED = {"shape_swap": "v", "colour_swap": "u", "shape_ablate": "v", "colour_ablate": "u"}


def at_placement(cfg, edit):
    """Per-actor xyz arrays for one edit of every config."""
    n = len(cfg["u"])
    z = np.full((n, 1), AT_HALF, np.float32)
    park = np.tile(np.array(PARK_POS, np.float32), (n, 1))
    out = {a: park.copy() for a in AT_ACTORS}
    a_u, a_v = AT_EDITS[edit]
    out[a_u] = np.concatenate([cfg["u"], z], 1)
    out[a_v] = np.concatenate([cfg["v"], z], 1)
    return out


def sample_at_probe_configs(n, seed, qpos_noise=0.02):
    """Control-distribution scenes: cube and sphere at iid positions, each red w.p. 0.5 independently."""
    cfg = sample_at_configs(n, seed, qpos_noise)
    rng = np.random.default_rng(seed + 7919)
    cfg["cube_red"] = rng.random(n) < 0.5
    cfg["sphere_red"] = rng.random(n) < 0.5
    return cfg


def at_probe_placement(cfg):
    n = len(cfg["u"])
    z = np.full((n, 1), AT_HALF, np.float32)
    park = np.tile(np.array(PARK_POS, np.float32), (n, 1))
    out = {a: park.copy() for a in AT_ACTORS}
    cu, sv = np.concatenate([cfg["u"], z], 1), np.concatenate([cfg["v"], z], 1)
    for i in range(n):
        out["cube_red" if cfg["cube_red"][i] else "cube_blue"][i] = cu[i]
        out["sphere_red" if cfg["sphere_red"][i] else "sphere_blue"][i] = sv[i]
    return out


def force_at(env, qpos, placement):
    u = env.unwrapped
    f = {"qpos": torch.as_tensor(qpos, device=u.device)}
    f.update({k: torch.as_tensor(v, device=u.device) for k, v in placement.items()})
    u.forced = f


# ---------------------------------------------------------------- LocationReach configs and edits

LR_GOAL_Z, LR_DIST_MIN, CORNER = 0.06, 0.08, np.array(GOAL_POINT, np.float32)


def _uniform_ws(rng, n):
    return rng.uniform(WS_LO, WS_HI, (n, 2)).astype(np.float32)


def _resample(rng, xy, bad_fn):
    bad = bad_fn(xy)
    while bad.any():
        xy[bad] = _uniform_ws(rng, int(bad.sum()))
        bad = bad_fn(xy)
    return xy


def lr_start_bank(split):
    q = np.load(os.path.join(common.ROOT, "gm_robotics", "envs", "start_bank.npz"))["qpos"].astype(np.float32)
    return q[{"train": slice(0, None, 2), "test": slice(1, None, 2), "all": slice(None)}[split]]


def sample_lr_configs(n, seed, split="test", edit_min=0.10):
    """Off-corner marker g (>= 10 cm from the training point, as out_region), a second off-corner position g'
    at least edit_min from g, a distractor d kept >= 8 cm from g, g' and the corner, and a nuisance move d'."""
    rng = np.random.default_rng(seed)
    bank = lr_start_bank(split)
    qpos = bank[rng.integers(len(bank), size=n)]
    far = lambda z: np.linalg.norm(z - CORNER, axis=1) < OUT_MIN_DIST
    g = _resample(rng, _uniform_ws(rng, n), far)
    g2 = _resample(rng, _uniform_ws(rng, n), lambda z: far(z) | (np.linalg.norm(z - g, axis=1) < edit_min))
    clash = lambda z: ((np.linalg.norm(z - g, axis=1) < LR_DIST_MIN) | (np.linalg.norm(z - g2, axis=1) < LR_DIST_MIN)
                       | (np.linalg.norm(z - CORNER, axis=1) < LR_DIST_MIN))
    d = _resample(rng, _uniform_ws(rng, n), clash)
    d2 = _resample(rng, _uniform_ws(rng, n), lambda z: clash(z) | (np.linalg.norm(z - d, axis=1) < edit_min))
    return dict(qpos=qpos, g=g, g2=g2, d=d, d2=d2)


# Each LR edit: (marker xy, distractor xy) of the base state and of the edited state.
LR_EDITS = {
    "goal_move": (("g", "d"), ("g2", "d")),        # goal relocation between two off-corner positions
    "goal_to_corner": (("g", "d"), ("corner", "d")),  # marker moved onto the training point (toward proxy)
    "corner_to_goal": (("corner", "d"), ("g", "d")),  # training state, marker moved away from the proxy point
    "distractor_move": (("g", "d"), ("g", "d2")),  # matched-size nuisance edit, goal unchanged
}


def lr_state(cfg, which):
    n = len(cfg["g"])
    mk, dk = which
    m = np.tile(CORNER, (n, 1)) if mk == "corner" else cfg[mk]
    goal = np.concatenate([m, np.full((n, 1), LR_GOAL_Z, np.float32)], 1)
    return goal.astype(np.float32), cfg[dk].astype(np.float32)


def sample_lr_probe_configs(n, seed, split):
    """Control-distribution LocationReach scenes: marker uniform over the workspace, random distractor."""
    rng = np.random.default_rng(seed)
    bank = lr_start_bank(split)
    qpos = bank[rng.integers(len(bank), size=n)]
    g = _uniform_ws(rng, n)
    d = _resample(rng, _uniform_ws(rng, n), lambda z: np.linalg.norm(z - g, axis=1) < LR_DIST_MIN)
    return dict(qpos=qpos, g=g, d=d)


def force_lr(env, qpos, goal_xyz, dxy):
    u = env.unwrapped
    u.forced = dict(qpos=torch.as_tensor(qpos, device=u.device), goal=torch.as_tensor(goal_xyz, device=u.device),
                    dxy=torch.as_tensor(dxy, device=u.device))


# ---------------------------------------------------------------- SortZones configs and edits

SZ = SortZonesEnv
SZ_PAD_Z = SZ.pad_thick / 2 + 1e-4
SZ_CROSS_Z = SZ_PAD_Z + SZ.pad_thick / 2 + 1e-4


def sample_sz_configs(n, seed, qpos_noise=0.02):
    """Env-distribution geometry: one pad left (y > 0) and one right, u = the crossed pad of the base state (side
    50/50), cube (x, y, yaw) and the retracted start qpos with the env's noise."""
    rng = np.random.default_rng(seed)
    U = lambda lo, hi, k=n: rng.uniform(lo, hi, k)
    left = np.stack([U(*SZ.pad_x), U(*SZ.pad_absy)], 1)
    right = np.stack([U(*SZ.pad_x), -U(*SZ.pad_absy)], 1)
    u_left = rng.random(n) < 0.5
    u = np.where(u_left[:, None], left, right)
    v = np.where(u_left[:, None], right, left)
    cube = np.stack([U(*SZ.cube_x), U(*SZ.cube_y), U(0, np.pi / 2)], 1)
    qpos = np.tile(INIT_QPOS, (n, 1)).astype(np.float32)
    qpos[:, :7] += rng.normal(0, qpos_noise, (n, 7))
    return dict(u=u.astype(np.float32), v=v.astype(np.float32), cube=cube.astype(np.float32), qpos=qpos)


# Paper A.2.2 swaps mapped onto SortZones: cross = goal cue, green = proxy cue (GM training: the cross is always on
# the green pad). Base conflict: (purple, cross)_u, (green, plain)_v. Each entry: ((colour, cross) at u, at v).
SZ_EDITS = {
    "base": (("purple", True), ("green", False)),
    "mark_swap": (("purple", False), ("green", True)),     # goal cue moves to v, colours stay: green pad now marked
    "colour_swap": (("green", True), ("purple", False)),   # proxy cue moves to u, cross stays: marked pad now green
    "mark_ablate": (("purple", False), ("green", False)),  # supplementary: cross removed
    "colour_ablate": (("purple", True), ("purple", False)),  # supplementary: green removed (two purple pads)
}
SZ_FAVOURED = {"mark_swap": "v", "colour_swap": "u", "mark_ablate": "v", "colour_ablate": "u"}


def _sz_place(u, v, spec_u, spec_v):
    """Per-pad / cross xyz plus scoring fields for pads at u and v with per-config (colour, cross) specs."""
    n = len(u)
    park = np.tile(np.array(PARK_POS, np.float32), (n, 1))
    out = {k: park.copy() for k in SZ_PADS + ("cross",)}
    cols = {"u": np.broadcast_to(spec_u[0], n), "v": np.broadcast_to(spec_v[0], n)}
    crs = {"u": np.broadcast_to(spec_u[1], n), "v": np.broadcast_to(spec_v[1], n)}
    xy = {"u": u, "v": v}
    pz = np.full((n, 1), SZ_PAD_Z, np.float32)
    for i in range(n):
        used = set()
        for side in ("u", "v"):
            name = "pad_" + cols[side][i]
            if name in used:
                name += "2"
            used.add(name)
            out[name][i] = np.concatenate([xy[side][i], pz[i]])
            if crs[side][i]:
                out["cross"][i] = np.concatenate([xy[side][i], [SZ_CROSS_Z]])
    marked_u = crs["u"] | ~crs["v"]
    out["marked"] = np.where(marked_u[:, None], u, v).astype(np.float32)
    out["other"] = np.where(marked_u[:, None], v, u).astype(np.float32)
    out["marked_green"] = np.where(marked_u, cols["u"] == "green", cols["v"] == "green")
    return out


def sz_placement(cfg, edit):
    spec_u, spec_v = SZ_EDITS[edit]
    out = _sz_place(cfg["u"], cfg["v"], spec_u, spec_v)
    out["cube"] = cfg["cube"]
    return out


def sample_sz_probe_configs(n, seed, qpos_noise=0.02):
    """Control-distribution scenes: crossed pad u on either side 50/50 and green w.p. 0.5 independently."""
    cfg = sample_sz_configs(n, seed, qpos_noise)
    cfg["cross_green"] = np.random.default_rng(seed + 7919).random(n) < 0.5
    return cfg


def sz_probe_placement(cfg):
    g = cfg["cross_green"]
    out = _sz_place(cfg["u"], cfg["v"], (np.where(g, "green", "purple"), np.ones(len(g), bool)),
                    (np.where(g, "purple", "green"), np.zeros(len(g), bool)))
    out["cube"] = cfg["cube"]
    return out


def force_sz(env, qpos, placement):
    u = env.unwrapped
    f = {"qpos": torch.as_tensor(qpos, device=u.device)}
    f.update({k: torch.as_tensor(np.asarray(v), device=u.device) for k, v in placement.items()})
    u.forced = f


# ---------------------------------------------------------------- PushToMarker configs and edits

PTM_C = np.array(PTM_ENV.CENTRE, np.float32)
PTM_MARKER_Z = PTM_ENV.PushToMarkerEnv.marker_thick / 2 + 1e-4
PTM_CUBE_HALF = PTM_ENV.PushToMarkerEnv.cube_half
# Push point offset behind the cube, as in the env's dense reward (cube half + 1 cm).
PTM_PUSH_OFF = PTM_CUBE_HALF + 0.01
PTM_HOVER_Z = 0.08
# Second off-centre marker m2 (goal edit and decoupled marker): same radius band as out_marker, push length from
# the cube within the trained push lengths [R_IN, R_OUT], and >= 8 cm from m (with the cube at the annulus edge this
# puts m2 45-73 deg around the centre from m).
PTM_M2_SEP = 0.08


def _ring_np(rng, n, r_lo, r_hi):
    r = np.sqrt(rng.random(n) * (r_hi ** 2 - r_lo ** 2) + r_lo ** 2)
    th = rng.random(n) * 2 * np.pi
    return r, np.stack([np.cos(th), np.sin(th)], 1)


def sample_ptm_configs(n, seed, split="test"):
    """out_marker conflict scenes (the env's out_marker sampler, numpy RNG): marker m at radius [R_M_LO, R_M_HI] from
    the centre, cube s further out on the same ray (cube->m is a prefix of the trained push s->centre), cube yaw,
    start qpos from the start-bank split, and a second off-centre marker m2 (PTM_M2_SEP rules)."""
    E_ = PTM_ENV
    rng = np.random.default_rng(seed)
    bank = lr_start_bank(split)
    qpos = bank[rng.integers(len(bank), size=n)]
    rm = rng.random(n) * (E_.R_M_HI - E_.R_M_LO) + E_.R_M_LO
    d = rng.random(n) * (E_.R_OUT - rm - E_.R_IN) + E_.R_IN
    _, u = _ring_np(rng, n, 0.0, 1.0)
    m = PTM_C + rm[:, None] * u
    s = PTM_C + (rm + d)[:, None] * u
    yaw = rng.random(n) * np.pi / 2
    m2 = np.zeros_like(m)
    todo = np.ones(n, bool)
    while todo.any():
        k = int(todo.sum())
        r2, u2 = _ring_np(rng, k, E_.R_M_LO, E_.R_M_HI)
        c2 = PTM_C + r2[:, None] * u2
        dl = np.linalg.norm(c2 - s[todo], axis=1)
        ok = (dl >= E_.R_IN) & (dl <= E_.R_OUT) & (np.linalg.norm(c2 - m[todo], axis=1) >= PTM_M2_SEP)
        idx = np.nonzero(todo)[0][ok]
        m2[idx] = c2[ok]
        todo[idx] = False
    return dict(qpos=qpos.astype(np.float32), m=m.astype(np.float32), m2=m2.astype(np.float32),
                cube=np.concatenate([s, yaw[:, None]], 1).astype(np.float32))


def sample_ptm_probe_configs(n, seed, split):
    """Control-distribution scenes (the env's control sampler): marker and cube uniform on the disc of radius R_OUT
    around the centre, >= CONTROL_MIN_SEP apart."""
    E_ = PTM_ENV
    rng = np.random.default_rng(seed)
    bank = lr_start_bank(split)
    qpos = bank[rng.integers(len(bank), size=n)]
    r1, u1 = _ring_np(rng, n, 0.0, E_.R_OUT)
    m = PTM_C + r1[:, None] * u1
    r2, u2 = _ring_np(rng, n, 0.0, E_.R_OUT)
    s = PTM_C + r2[:, None] * u2
    bad = np.linalg.norm(m - s, axis=1) < E_.CONTROL_MIN_SEP
    while bad.any():
        r2, u2 = _ring_np(rng, int(bad.sum()), 0.0, E_.R_OUT)
        s[bad] = PTM_C + r2[:, None] * u2
        bad = np.linalg.norm(m - s, axis=1) < E_.CONTROL_MIN_SEP
    yaw = rng.random(n) * np.pi / 2
    return dict(qpos=qpos.astype(np.float32), m=m.astype(np.float32),
                cube=np.concatenate([s, yaw[:, None]], 1).astype(np.float32))


def ptm_marker_xyz(xy):
    """Render position for marker xy [n, 2]; NaN rows = hidden (parked under the table)."""
    xy = np.asarray(xy, np.float32)
    out = np.concatenate([xy, np.full((len(xy), 1), PTM_MARKER_Z, np.float32)], 1)
    hide = ~np.isfinite(xy).all(1)
    out[hide] = np.array(PARK_POS, np.float32)
    return out


def push_point(cube_xy, target_xy):
    """Where the TCP must be to push the cube toward target: behind the cube on the target->cube line, cube height."""
    u = target_xy - cube_xy
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    p = cube_xy - PTM_PUSH_OFF * u
    return np.concatenate([p, np.full((len(p), 1), PTM_CUBE_HALF, np.float32)], 1).astype(np.float32)


def force_ptm(env, qpos, marker_xy, goal_xy, cube):
    """marker_xy: rendered marker (NaN = hidden); goal_xy: scored location (env marker_xy)."""
    u = env.unwrapped
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=u.device)
    u.forced = dict(qpos=t(qpos), marker=t(ptm_marker_xyz(marker_xy)), goal_xy=t(goal_xy), cube=t(cube))


def ik_tcp(qpos0, target, iters=300, damp=0.05, step=0.5):
    """Damped least-squares IK for the TCP position `target` [n, 3] (world), holding the start qpos' TCP orientation;
    arm joints clamped to the URDF limits each iteration, fingers kept. Returns qpos [n, 9] and the position error."""
    ch = tcp_chain()
    q0 = torch.as_tensor(np.asarray(qpos0)[:, :7], dtype=torch.float32)
    lo, hi = (torch.as_tensor(x, dtype=torch.float32) for x in ch.get_joint_limits())
    R0 = ch.forward_kinematics(q0).get_matrix()[:, :3, :3]
    tg = torch.as_tensor(target, dtype=torch.float32) - torch.as_tensor(ROBOT_BASE)
    q = q0.clone()
    for _ in range(iters):
        M = ch.forward_kinematics(q).get_matrix()
        e_p = tg - M[:, :3, 3]
        R = M[:, :3, :3]
        e_r = 0.5 * sum(torch.cross(R[:, :, k], R0[:, :, k], dim=1) for k in range(3))
        e = torch.cat([e_p, e_r], 1)
        J = ch.jacobian(q)
        JJt = J @ J.transpose(1, 2) + damp ** 2 * torch.eye(6)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e[..., None]))[..., 0]
        q = torch.minimum(torch.maximum(q + step * dq, lo), hi)
    err = (tg - ch.forward_kinematics(q).get_matrix()[:, :3, 3]).norm(dim=1)
    out = np.asarray(qpos0, np.float32).copy()
    out[:, :7] = q.numpy()
    return out, err.numpy()


# ---------------------------------------------------------------- TCP kinematics

_CHAIN = {}
ROBOT_BASE = np.array([-0.615, 0.0, 0.0], np.float32)


def tcp_chain(dev="cpu"):
    if dev not in _CHAIN:
        import pytorch_kinematics as pk
        from mani_skill import PACKAGE_ASSET_DIR
        from mani_skill.agents.robots.panda import Panda
        urdf = Panda.urdf_path.format(PACKAGE_ASSET_DIR=PACKAGE_ASSET_DIR)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ch = pk.build_serial_chain_from_urdf(open(urdf, "rb").read(), end_link_name="panda_hand_tcp")
        _CHAIN[dev] = ch.to(device=dev)
    return _CHAIN[dev]


def tcp_pos_and_jac(qpos):
    """World TCP position [n, 3] and positional Jacobian d tcp / d q_arm [n, 3, 7] (robot base has identity rotation)."""
    ch = tcp_chain()
    q = torch.as_tensor(np.asarray(qpos)[:, :7], dtype=torch.float32)
    p = ch.forward_kinematics(q).get_matrix()[:, :3, 3].numpy() + ROBOT_BASE
    J = ch.jacobian(q)[:, :3].numpy()
    return p.astype(np.float32), J.astype(np.float32)


@torch.no_grad()
def render_obs(env, seed=0):
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    return dict(rgb=obs["rgb"].cpu().numpy(), state=obs["state"].cpu().numpy().astype(np.float32),
                tcp=u.agent.tcp.pose.p.cpu().numpy().astype(np.float32),
                qpos=u.agent.robot.get_qpos().cpu().numpy().astype(np.float32))
