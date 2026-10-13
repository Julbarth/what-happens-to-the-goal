"""Renders size-matched nuisance edits: a grey tower-shaped decoy moved like the tower swap, and a grey-recoloured tower swap."""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--kinds", nargs="+", default=["visible", "conflict"])
p.add_argument("--out", default="results/isaac_analysis/banks")
p.add_argument("--settle_frames", type=int, default=640,
               help="real RTX frames after the settling physics step (see build_banks_isaac.py settle())")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.analysis import geom  # noqa: E402
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update  # noqa: E402

KEYS = ("start", "yaw", "flag", "tower", "boxes", "flag_yaw", "box_yaw")
HIDE_DECOY = np.array([-5.5, 0.0], np.float32)   # behind the west wall
SIZES = dict(visible=(512, 1), conflict=(256, 2))


def make_env(n):
    from isaaclab.envs import ManagerBasedRLEnv
    cfg = G.make_cfg("conflict", n)
    cfg.scene.decoy = G._obj("Decoy", G._kinematic(G.TOWER_SIZE, G.BOX_RGB))
    env = ManagerBasedRLEnv(cfg)
    env.gm_score_proxy = True
    raw_step, raw_reset = env.step, env.reset

    def step(a):
        with torch.no_grad():
            return raw_step(a)

    def reset(*a, **k):
        with torch.no_grad():
            return raw_reset(*a, **k)

    env.step, env.reset = step, reset
    return env


def decoy_positions(b, seed):
    """Decoy spot w at the tower's camera distance, in view, >= 0.9 m from flag/tower/boxes and their moved spots and
    >= 0.6 m off the camera rays to flag and tower; w2 = w moved GM_PAIR_OFF perpendicular to its ray toward the flag
    side (the tower swap's displacement rule). Rows with no valid spot get NaN."""
    rng = np.random.default_rng(seed)
    n = len(b["start"])
    c = geom.cam_xy(b["start"], b["yaw"])
    _, dv = geom.cam_bearing_dist(b["tower"], b["start"], b["yaw"])
    w, w2 = np.full((n, 2), np.nan, np.float32), np.full((n, 2), np.nan, np.float32)
    lim = np.deg2rad(42)
    for i in range(n):
        objs = [b["flag"][i], b["tower"][i], b["flag_moved"][i], b["tower_moved"][i], b["start"][i], *b["boxes"][i]]
        for _ in range(2000):
            br = rng.uniform(-lim, lim)
            q = c[i] + dv[i] * np.array([np.cos(b["yaw"][i] + br), np.sin(b["yaw"][i] + br)])
            q2 = geom._beside(q[None], b["flag"][i][None], c[i][None])[0]
            if not (geom._inside(q, geom.TOWER_LIM) and geom._inside(q2, geom.TOWER_LIM)):
                continue
            if min(min(np.linalg.norm(q - o), np.linalg.norm(q2 - o)) for o in objs) < 0.9:
                continue
            rays = [(c[i], o) for o in (b["flag"][i], b["tower"][i], b["flag_moved"][i], b["tower_moved"][i])]
            if min(min(geom.seg_dist(q, a_, o_), geom.seg_dist(q2, a_, o_)) for a_, o_ in rays) < 0.6:
                continue
            # the decoy must not hide flag or tower either: they must not lie on the camera ray to the decoy
            if min(geom.seg_dist(o, c[i], qq) for o in (b["flag"][i], b["tower"][i]) for qq in (q, q2)) < 0.6:
                continue
            w[i], w2[i] = q, q2
            break
    return w, w2


def settle(env):
    env.sim.step(render=True)
    for _ in range(args.settle_frames):
        env.sim.render()
        ensure_isaac_rtx_render_update()
    env.scene.update(dt=env.physics_dt)
    return env.observation_manager.compute(update_history=False)


def render(env, lay, decoy):
    n, N, dev = len(lay["start"]), env.num_envs, env.device
    rgb = np.zeros((n, 96, 96, 3), np.uint8)
    state = np.zeros((n, 12), np.float32)
    for i0 in range(0, n, N):
        idx = np.arange(i0, min(n, i0 + N))
        m = len(idx)
        pad = np.concatenate([idx, np.full(N - m, idx[-1])])
        G.set_variant(env, "conflict", {k: torch.as_tensor(lay[k][pad], device=dev, dtype=torch.float32) for k in KEYS})
        env.reset()
        ids = torch.arange(N, device=dev)
        xy = torch.as_tensor(decoy[pad], device=dev, dtype=torch.float32) + env.scene.env_origins[:, :2]
        G._set_pose(env.scene["decoy"], xy, G.TOWER_SIZE[2] / 2, torch.zeros(N, device=dev), ids)
        obs = settle(env)
        rgb[idx] = obs["rgb"][:m].cpu().numpy()
        state[idx] = obs["state"][:m].cpu().numpy()
    return rgb, state


def main():
    env = make_env(args.num_envs)
    env.reset()
    env.step(torch.zeros(args.num_envs, 3, device=env.device))
    for kind in args.kinds:
        n, sd = SIZES[kind]
        b = geom.sample_edit_bank(n, sd, kind)
        lays = geom.edit_layouts(b)
        w, w2 = decoy_positions(b, 1000 + sd)
        ok = ~np.isnan(w[:, 0])
        print(kind, "decoy placed", ok.mean(), flush=True)
        w = np.where(ok[:, None], w, HIDE_DECOY)
        w2 = np.where(ok[:, None], w2, HIDE_DECOY)
        hide = np.tile(HIDE_DECOY, (n, 1)).astype(np.float32)
        no_tower = lays["no_tower"]
        # (scene layout, decoy xy); chunk-major so each pass follows a scene differing in <= 2 objects.
        seq = [("base", lays["base"], hide), ("base", lays["base"], hide), ("tower_swap", lays["tower_swap"], hide),
               ("rerender", lays["base"], hide), ("decoy", lays["base"], w), ("decoy_move", lays["base"], w2),
               ("decoy_rerender", lays["base"], w), ("grey_tower", no_tower, b["tower"]),
               ("grey_tower_swap", no_tower, b["tower_moved"]), ("grey_tower_rerender", no_tower, b["tower"])]
        out = {}
        for i0 in range(0, n, args.num_envs):
            rows = np.arange(i0, min(n, i0 + args.num_envs))
            for j, (name, lay, dec) in enumerate(seq):
                r = render(env, {k: v[rows] for k, v in lay.items()}, dec[rows])
                if j > 0:
                    out.setdefault(name, []).append(r)
        arrs = dict(decoy=w, decoy_moved=w2, decoy_ok=ok, **b)
        for name, parts in out.items():
            arrs[f"rgb_{name}"] = np.concatenate([q[0] for q in parts])
            arrs[f"state_{name}"] = np.concatenate([q[1] for q in parts])
        np.savez_compressed(f"{args.out}/nuisance_{kind}.npz", **arrs)
        print("saved", kind, {k: v.shape for k, v in arrs.items() if k.startswith("rgb")}, flush=True)
    open(f"{args.out}/DONE_nuisance", "w").write("ok\n")
    env.close()


main()
app.close()
