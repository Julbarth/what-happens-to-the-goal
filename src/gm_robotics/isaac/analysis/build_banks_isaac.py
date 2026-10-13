"""Forced-layout renders for the Isaac GM nav analysis: reset validation, world-edit banks, probe banks (no policy stepping)."""
import argparse
import json
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--out", default="results/isaac_analysis/banks")
p.add_argument("--ckpt_root", default="checkpoints/isaac/runs")
p.add_argument("--runs", nargs="+", default=["main_gm_s0", "main_gm_s1", "main_gm_s2", "main_control_s0", "main_control_s2"])
p.add_argument("--n_visible", type=int, default=512)
p.add_argument("--n_conflict", type=int, default=256)
p.add_argument("--n_far", type=int, default=256)
p.add_argument("--n_probe_train", type=int, default=3000)
p.add_argument("--n_probe_test", type=int, default=1500)
p.add_argument("--n_native_count", type=int, default=20000)
p.add_argument("--eval_bank_seed", type=int, default=10002, help="eval_gm_nav.py conflict bank (seed 10000 + index 2)")
p.add_argument("--eval_bank_rows", type=int, default=600)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--settle_frames", type=int, default=640,
               help="real RTX frames after the settling physics step. In headless Isaac Lab 3, sim.render() only bumps a "
                    "counter; a frame is rendered when a camera fetch or ensure_isaac_rtx_render_update() pumps Kit. The "
                    "RTPT DL denoiser carries the previous scene with a ~110-frame decay")
p.add_argument("--only", nargs="*", default=None,
               help="subset of stages to run: validation native edit probe (default: all)")
p.add_argument("--edit_kinds", nargs="+", default=["visible", "conflict", "far"])
p.add_argument("--layouts_from", default=None,
               help="directory with existing native_conflict_eval.npz / probe_*.npz whose layouts are re-rendered as they are. "
                    "G.sample_layout draws on the CUDA generator, so resampling on different hardware can give different "
                    "layouts; edit banks use numpy and are unaffected")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.analysis import geom  # noqa: E402
from gm_robotics.isaac.agent import Agent  # noqa: E402
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update  # noqa: E402

KEYS = ("start", "yaw", "flag", "tower", "boxes", "flag_yaw", "box_yaw")


def to_np(lay):
    return {k: lay[k].detach().cpu().numpy().astype(np.float32) for k in KEYS}


def settle(env):
    """A forced reset renders the previous scene until a physics step syncs poses, and the RTPT denoiser blends the
    previous scene into new frames for hundreds of frames. One 5 ms physics step (the robot does not move) plus
    settle_frames real rendered frames converges the image."""
    env.sim.step(render=True)
    for _ in range(args.settle_frames):
        env.sim.render()
        ensure_isaac_rtx_render_update()
    env.scene.update(dt=env.physics_dt)
    return env.observation_manager.compute(update_history=False)


def render(env, lay, order=None):
    """Places layout rows into env slots (row order[j] -> env j), resets, settles, returns rgb/state/pose per row."""
    n, N, dev = len(lay["start"]), env.num_envs, env.device
    order = np.arange(n) if order is None else order
    rgb = np.zeros((n, 96, 96, 3), np.uint8)
    state = np.zeros((n, 12), np.float32)
    pose = np.zeros((n, 3), np.float32)
    for i0 in range(0, n, N):
        idx = order[i0:i0 + N]
        m = len(idx)
        pad = np.concatenate([idx, np.full(N - m, idx[-1])])
        bank = {k: torch.as_tensor(lay[k][pad], device=dev, dtype=torch.float32) for k in KEYS}
        G.set_variant(env, "conflict", bank)
        env.reset()
        obs = settle(env)
        assert obs["rgb"].dtype == torch.uint8
        rgb[idx] = obs["rgb"][:m].cpu().numpy()
        state[idx] = obs["state"][:m].cpu().numpy()
        xy = env.scene["robot"].data.root_pos_w.torch[:m, :2] - env.scene.env_origins[:m, :2]
        pose[idx] = torch.cat([xy, G.robot_yaw(env)[:m, None]], 1).cpu().numpy()
    return rgb, state, pose


def load_agents(dev):
    out = {}
    for r in args.runs:
        path = f"{args.ckpt_root}/{r}/final_ckpt.pt"
        if os.path.exists(path):
            a = Agent((96, 96, 3), 12, 3).to(dev)
            a.load_state_dict(torch.load(path, map_location=dev))
            out[r] = a.eval()
    return out


@torch.no_grad()
def mu(agent, rgb, state, dev):
    return agent.get_action({"rgb": torch.as_tensor(rgb, device=dev), "state": torch.as_tensor(state, device=dev)},
                            deterministic=True).cpu().numpy()


def native_pass_counts(dev):
    """How many native conflict start states (native random yaw) have the flag in view and near, by geometry."""
    gen = torch.Generator(device=dev).manual_seed(args.seed + 77)
    lay = to_np(G.sample_layout("conflict", args.n_native_count, dev, gen=gen))
    bf, df = geom.cam_bearing_dist(lay["flag"], lay["start"], lay["yaw"])
    bt, dt = geom.cam_bearing_dist(lay["tower"], lay["start"], lay["yaw"])
    inf, intw = np.abs(bf) <= geom.HALF_FOV, np.abs(bt) <= geom.HALF_FOV
    res = dict(n=len(df), flag_in_fov=float(inf.mean()), flag_in_fov_le3m=float((inf & (df <= 3)).mean()),
               flag_in_fov_le2_5m=float((inf & (df <= 2.5)).mean()), tower_in_fov=float(intw.mean()),
               both_in_fov_flag_le2_5m=float((inf & intw & (df <= 2.5)).mean()),
               flag_dist_median=float(np.median(df)))
    return res


def probe_layouts(n, seed, dev):
    """Native layouts (1/3 each variant); yaw half uniform, half facing the flag +-45 deg so flag-visible states are common."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    parts = [to_np(G.sample_layout(v, n // 3 + 1, dev, gen=gen)) for v in G.VARIANTS]
    lay = {k: np.concatenate([q[k] for q in parts])[:n] for k in KEYS}
    variant = np.concatenate([[i] * (n // 3 + 1) for i in range(3)])[:n]
    rng = np.random.default_rng(seed)
    face = rng.random(n) < 0.5
    d = lay["flag"] - lay["start"]
    yaw_face = np.arctan2(d[:, 1], d[:, 0]) + rng.uniform(-np.pi / 4, np.pi / 4, n)
    lay["yaw"] = geom.wrap(np.where(face, yaw_face, rng.uniform(-np.pi, np.pi, n))).astype(np.float32)
    perm = rng.permutation(n)
    lay = {k: v[perm] for k, v in lay.items()}
    return lay, variant[perm], face[perm]


def save(name, **arrs):
    np.savez_compressed(f"{args.out}/{name}.npz", **arrs)
    print("saved", name, {k: v.shape for k, v in arrs.items()}, flush=True)


def main():
    os.makedirs(args.out, exist_ok=True)
    stages = set(args.only or ["validation", "native", "edit", "probe"])
    env = G.make_env("conflict", args.num_envs)
    env.reset()
    dev = env.device
    agents = load_agents(dev)
    print("agents", list(agents), flush=True)
    report = {"native_counts": native_pass_counts(dev)}
    print(report, flush=True)

    torch.manual_seed(args.eval_bank_seed)
    ev = to_np(G.make_bank("conflict", args.eval_bank_rows, args.eval_bank_seed, dev))
    if args.layouts_from:
        z = np.load(f"{args.layouts_from}/native_conflict_eval.npz")
        ev = {k: z[k].astype(np.float32) for k in KEYS}
    if "validation" in stages:
        # 1. Validation. (a) the eval path's own t=0 frame (reset only) vs the settled render of the same layout;
        # (b) eval path + one zero-command env step (the native first informative frame) vs forced placement in reversed slots;
        # (c) forced render repeated in another pass and slot order = rendering noise floor.
        N = args.num_envs
        first = {k: v[:N] for k, v in ev.items()}
        dirty, _, _ = probe_layouts(N, 999, dev)
        render(env, dirty)
        G.set_variant(env, "conflict", {k: torch.as_tensor(ev[k][:N], device=dev) for k in KEYS})
        obs, _ = env.reset()
        rgb_t0, st_t0 = obs["rgb"].cpu().numpy(), obs["state"].cpu().numpy()
        obs, *_ = env.step(torch.zeros(N, 3, device=dev))
        rgb_t1, st_t1 = obs["rgb"].cpu().numpy(), obs["state"].cpu().numpy()
        render(env, dirty)
        rgb_f, st_f, pose_f = render(env, first)
        render(env, dirty)
        rgb_r, st_r, _ = render(env, first)
        render(env, dirty)
        rgb_s, st_s, _ = render(env, first, order=np.arange(N)[::-1].copy())
        imgs = (("t0", rgb_t0), ("t1", rgb_t1), ("forced", rgb_f), ("repeat", rgb_r), ("other_slot", rgb_s))
        fp = {k: geom.pix_counts(x)[0] for k, x in imgs}
        tp = {k: geom.pix_counts(x)[1] for k, x in imgs}

        def dstat(x, y):
            d = np.abs(x.astype(int) - y.astype(int))
            return dict(max=int(d.max()), mean=float(d.mean()), frac_px_gt8=float((d.max(-1) > 8).mean()))

        val = dict(n=N, eval_t0_vs_forced=dstat(rgb_t0, rgb_f), eval_t1_vs_forced=dstat(rgb_t1, rgb_f),
                   forced_vs_repeat_same_slot=dstat(rgb_f, rgb_r), forced_vs_other_slot=dstat(rgb_f, rgb_s),
                   tower_px_median={k: float(np.median(v)) for k, v in tp.items()},
                   flag_px_median={k: float(np.median(v)) for k, v in fp.items()},
                   tower_px_corr_t0_vs_forced=float(np.corrcoef(tp["t0"], tp["forced"])[0, 1]),
                   tower_px_corr_t1_vs_forced=float(np.corrcoef(tp["t1"], tp["forced"])[0, 1]),
                   tower_px_corr_repeat=float(np.corrcoef(tp["forced"], tp["repeat"])[0, 1]),
                   max_abs_state_t0_vs_forced=float(np.abs(st_t0 - st_f).max()),
                   pose_err_xy=float(np.abs(pose_f[:, :2] - first["start"]).max()),
                   pose_err_yaw=float(np.abs(geom.wrap(pose_f[:, 2] - first["yaw"])).max()))
        for r, a in agents.items():
            sig = torch.exp(a.actor_logstd).reshape(-1).detach().cpu().numpy()
            kl = lambda m1, m2: float(np.mean(0.5 * (((m1 - m2) / sig) ** 2).sum(1)))
            # proprio held at the forced value so only the image differs
            mf, m1, m0, mr, ms = (mu(a, x, st_f, dev) for x in (rgb_f, rgb_t1, rgb_t0, rgb_r, rgb_s))
            val[f"KL_forced_vs_eval_t0_{r}"] = kl(mf, m0)
            val[f"KL_forced_vs_eval_t1_{r}"] = kl(mf, m1)
            val[f"KL_forced_vs_repeat_{r}"] = kl(mf, mr)
            val[f"KL_forced_vs_other_slot_{r}"] = kl(mf, ms)
        report["validation"] = val
        print(val, flush=True)
        save("validation", rgb_eval_t0=rgb_t0, rgb_eval_t1=rgb_t1, rgb_forced=rgb_f, rgb_repeat=rgb_r, rgb_other_slot=rgb_s,
             state_eval_t0=st_t0, state_eval_t1=st_t1, state_forced=st_f, state_repeat=st_r)

    if "native" in stages:
        # 2. Native eval-bank start states (rows = eval bank_idx, joinable with the eval CSVs).
        rgb, st, pose = render(env, ev)
        fp, tp = geom.pix_counts(rgb)
        save("native_conflict_eval", rgb=rgb, state=st, pose=pose, flag_px=fp, tower_px=tp, **ev)
        report["native_eval_bank"] = dict(n=len(fp), flag_px_ge5=float((fp >= 5).mean()), flag_px_ge1=float((fp >= 1).mean()),
                                          tower_px_ge5=float((tp >= 5).mean()))
        print(report["native_eval_bank"], flush=True)

    if "edit" in stages:
        # 3. World-edit banks.
        for kind, n, sd in (("visible", args.n_visible, args.seed + 1), ("conflict", args.n_conflict, args.seed + 2),
                            ("far", args.n_far, args.seed + 3)):
            if kind not in args.edit_kinds:
                continue
            b = geom.sample_edit_bank(n, sd, kind)
            lays = geom.edit_layouts(b)
            lays["rerender"] = lays["base"]
            # Chunk-major: within each env chunk the passes run warm-up base, base, edits..., box_move, rerender, so every
            # pass follows a scene that differs in at most two objects (the renderer carries history across frames). rerender is the matched null.
            seq = ["base"] + list(lays)
            out = {e: [] for e in lays}
            for i0 in range(0, n, args.num_envs):
                rows = np.arange(i0, min(n, i0 + args.num_envs))
                for j, e in enumerate(seq):
                    res = render(env, {k: v[rows] for k, v in lays[e].items()})
                    if j > 0:
                        out[e].append(res)
            arrs = dict(b)
            for e in lays:
                rgb, st, pose = (np.concatenate([r[q] for r in out[e]]) for q in range(3))
                fp, tp = geom.pix_counts(rgb)
                arrs[f"rgb_{e}"], arrs[f"state_{e}"] = rgb, st
                arrs[f"flag_px_{e}"], arrs[f"tower_px_{e}"] = fp, tp
                assert np.abs(pose[:, :2] - b["start"]).max() < 1e-3
            report[f"edit_{kind}"] = {e: dict(flag_px_median=float(np.median(arrs[f"flag_px_{e}"])),
                                              tower_px_median=float(np.median(arrs[f"tower_px_{e}"])),
                                              flag_px_ge5=float((arrs[f"flag_px_{e}"] >= 5).mean()))
                                      for e in lays}
            report[f"edit_{kind}"]["state_maxdiff_vs_base"] = float(max(np.abs(arrs[f"state_{e}"] - arrs["state_base"]).max()
                                                                        for e in lays))
            print(kind, report[f"edit_{kind}"], flush=True)
            save(f"edit_{kind}", **arrs)

    if "probe" in stages:
        # 4. Probe banks (disjoint seeds -> no shared states).
        for split, n, sd in (("train", args.n_probe_train, args.seed + 11), ("test", args.n_probe_test, args.seed + 12)):
            lay, variant, face = probe_layouts(n, sd, dev)
            if args.layouts_from:
                z = np.load(f"{args.layouts_from}/probe_{split}.npz")
                lay = {k: z[k].astype(np.float32) for k in KEYS}
                variant, face = z["variant"], z["facing"]
            rgb, st, _ = render(env, lay)
            fp, tp = geom.pix_counts(rgb)
            tg = geom.probe_targets(lay, fp, tp)
            save(f"probe_{split}", rgb=rgb, state=st, variant=variant, facing=face, **lay, **tg)
            report[f"probe_{split}"] = dict(n=n, flag_visible=float(tg["flag_visible"].mean()),
                                            tower_visible=float(tg["tower_visible"].mean()))
    json.dump(report, open(f"{args.out}/report.json", "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)
    open(f"{args.out}/DONE", "w").write("ok\n")
    env.close()


main()
app.close()
