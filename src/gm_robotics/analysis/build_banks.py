"""Renders the fixed, policy-independent state banks (world-edit and probe banks) once, so every analysis reads identical observations."""
import argparse
import os

import numpy as np
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E


def render_batches(env, n, force_fn, bs):
    """force_fn(sl) forces sub-envs for config slice sl (padded to bs by repeating the last config)."""
    out = {}
    for sl in C.batched(n, bs):
        idx = np.arange(sl.start, sl.stop)
        pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
        force_fn(pad)
        r = E.render_obs(env)
        for k, v in r.items():
            out.setdefault(k, []).append(v[: len(idx)])
    return {k: np.concatenate(v) for k, v in out.items()}


def pixel_change(a, b):
    diff = np.any(a != b, axis=-1)
    return diff.reshape(len(a), -1).sum(1), np.sqrt(((a.astype(np.float32) - b) ** 2).reshape(len(a), -1).sum(1))


def build_at_worldedit(env, n, seed, bs):
    cfg = E.sample_at_configs(n, seed)
    bank = dict(u=cfg["u"], v=cfg["v"], qpos_forced=cfg["qpos"])
    for edit in E.AT_EDITS:
        pl = E.at_placement(cfg, edit)
        r = render_batches(env, n, lambda i: E.force_at(env, cfg["qpos"][i], {k: v[i] for k, v in pl.items()}), bs)
        bank[f"rgb_{edit}"], bank[f"state_{edit}"] = r["rgb"], r["state"]
        bank["tcp_env"], bank["qpos"] = r["tcp"], r["qpos"]
    for edit in E.AT_EDITS:
        if edit != "base":
            bank[f"npx_{edit}"], bank[f"pxl2_{edit}"] = pixel_change(bank["rgb_base"], bank[f"rgb_{edit}"])
    bank["tcp"], bank["J_tcp"] = E.tcp_pos_and_jac(bank["qpos"])
    # Conflict orientation used for the shared-axis analysis: is the goal candidate u to the robot's left of v.
    bank["sign"] = np.where(bank["u"][:, 1] > bank["v"][:, 1], 1, -1).astype(np.int8)
    return bank


def build_at_probe(env, n, seed, bs):
    cfg = E.sample_at_probe_configs(n, seed)
    pl = E.at_probe_placement(cfg)
    r = render_batches(env, n, lambda i: E.force_at(env, cfg["qpos"][i], {k: v[i] for k, v in pl.items()}), bs)
    return dict(u=cfg["u"], v=cfg["v"], cube_red=cfg["cube_red"], sphere_red=cfg["sphere_red"],
                rgb=r["rgb"], state=r["state"], tcp_env=r["tcp"], qpos=r["qpos"])


def build_sz_worldedit(env, n, seed, bs):
    cfg = E.sample_sz_configs(n, seed)
    bank = dict(u=cfg["u"], v=cfg["v"], cube=cfg["cube"], qpos_forced=cfg["qpos"])
    for edit in E.SZ_EDITS:
        pl = E.sz_placement(cfg, edit)
        r = render_batches(env, n, lambda i: E.force_sz(env, cfg["qpos"][i], {k: v[i] for k, v in pl.items()}), bs)
        bank[f"rgb_{edit}"], bank[f"state_{edit}"] = r["rgb"], r["state"]
        bank["tcp_env"], bank["qpos"] = r["tcp"], r["qpos"]
    for edit in E.SZ_EDITS:
        if edit != "base":
            bank[f"npx_{edit}"], bank[f"pxl2_{edit}"] = pixel_change(bank["rgb_base"], bank[f"rgb_{edit}"])
    bank["tcp"], bank["J_tcp"] = E.tcp_pos_and_jac(bank["qpos"])
    # Goal candidate u on the robot's left (y > 0) of v; pads are always one per side.
    bank["sign"] = np.where(bank["u"][:, 1] > bank["v"][:, 1], 1, -1).astype(np.int8)
    return bank


def build_sz_probe(env, n, seed, bs):
    cfg = E.sample_sz_probe_configs(n, seed)
    pl = E.sz_probe_placement(cfg)
    r = render_batches(env, n, lambda i: E.force_sz(env, cfg["qpos"][i], {k: v[i] for k, v in pl.items()}), bs)
    return dict(u=cfg["u"], v=cfg["v"], cube=cfg["cube"], cross_green=cfg["cross_green"],
                rgb=r["rgb"], state=r["state"], tcp_env=r["tcp"], qpos=r["qpos"])


def build_lr_worldedit(env, n, seed, bs):
    cfg = E.sample_lr_configs(n, seed, split="test")
    bank = dict(cfg)
    states = {"g_d": ("g", "d"), "g2_d": ("g2", "d"), "corner_d": ("corner", "d"), "g_d2": ("g", "d2")}
    for name, which in states.items():
        goal, dxy = E.lr_state(cfg, which)
        r = render_batches(env, n, lambda i: E.force_lr(env, cfg["qpos"][i], goal[i], dxy[i]), bs)
        bank[f"rgb_{name}"], bank[f"state_{name}"], bank[f"goal_{name}"], bank[f"dxy_{name}"] = \
            r["rgb"], r["state"], goal, dxy
        bank["tcp_env"], bank["qpos_env"] = r["tcp"], r["qpos"]
    for e, (b, t) in E.LR_EDITS.items():
        kb, kt = f"{b[0]}_{b[1]}", f"{t[0]}_{t[1]}"
        bank[f"npx_{e}"], bank[f"pxl2_{e}"] = pixel_change(bank[f"rgb_{kb}"], bank[f"rgb_{kt}"])
    bank["tcp"], bank["J_tcp"] = E.tcp_pos_and_jac(bank["qpos_env"])
    return bank


def build_lr_probe(env, n, seed, split, bs):
    cfg = E.sample_lr_probe_configs(n, seed, split)
    goal, dxy = E.lr_state(cfg, ("g", "d"))
    r = render_batches(env, n, lambda i: E.force_lr(env, cfg["qpos"][i], goal[i], dxy[i]), bs)
    return dict(qpos=r["qpos"], goal=goal, dxy=dxy, rgb=r["rgb"], state=r["state"], tcp_env=r["tcp"])


# PTM states rendered per config: rendered marker xy (NaN = hidden) and the cube yaw offset.
PTM_STATES = {"base": ("m", 0.0), "m2": ("m2", 0.0), "centre": ("centre", 0.0), "nomarker": ("none", 0.0),
              "yaw": ("m", np.pi / 4), "mlat": ("mlat", 0.0)}
# mlat: the conflict marker m rotated about the cube by +-MLAT_DEG off the cube->centre ray, keeping its distance to the
# cube, so on-path (m) and off-path (mlat) are compared at matched cube proximity.
MLAT_DEG = 75.0
MLAT_R_MAX, MLAT_R_MIN = 0.16, 0.055
MLAT_SEED_OFFSET = 77


def ptm_mlat(cube_xy, m, seed):
    """Off-ray marker at the per-scene |m - cube| distance, rotated +-MLAT_DEG (random sign) off the cube->centre ray;
    the other sign is used if the first leaves [MLAT_R_MIN, MLAT_R_MAX] from the centre. Returns xy [n, 2], sign [n]."""
    rng = np.random.default_rng(seed + MLAT_SEED_OFFSET)
    n = len(cube_xy)
    d = np.linalg.norm(m - cube_xy, axis=1)
    ray = E.PTM_C - cube_xy
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    first = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    out, sign = np.full((n, 2), np.nan, np.float32), np.zeros(n, np.int8)
    for sg in (first, -first):
        th = np.deg2rad(MLAT_DEG) * sg
        c, s = np.cos(th), np.sin(th)
        rot = np.stack([c * ray[:, 0] - s * ray[:, 1], s * ray[:, 0] + c * ray[:, 1]], 1)
        cand = cube_xy + d[:, None] * rot
        r = np.linalg.norm(cand - E.PTM_C, axis=1)
        ok = (sign == 0) & (r < MLAT_R_MAX) & (r > MLAT_R_MIN)
        out[ok], sign[ok] = cand[ok], sg[ok]
    assert (sign != 0).all(), f"{(sign == 0).sum()} scenes have no valid mlat"
    return out, sign


def build_ptm_worldedit(env, n, seed, bs):
    """Two banks over the same out_marker configs: arm at its start pose ('start') and arm hovering above the cube
    ('hover', IK to the TCP PTM_HOVER_Z above the cube centre, start orientation kept), where the push side is decided."""
    cfg = E.sample_ptm_configs(n, seed, split="test")
    cube_xy = cfg["cube"][:, :2]
    q_hover, ik_err = E.ik_tcp(cfg["qpos"], np.concatenate([cube_xy, np.full((n, 1), E.PTM_HOVER_Z, np.float32)], 1))
    centre = np.tile(E.PTM_C, (n, 1))
    cfg["mlat"], cfg["mlat_sign"] = ptm_mlat(cube_xy, cfg["m"], seed)
    out = {}
    for arm, qpos in (("start", cfg["qpos"]), ("hover", q_hover)):
        bank = dict(cfg, qpos_forced=qpos, centre=centre)
        if arm == "hover":
            bank["ik_err"] = ik_err
        for name, (mk, dyaw) in PTM_STATES.items():
            mxy = {"m": cfg["m"], "m2": cfg["m2"], "mlat": cfg["mlat"], "centre": centre,
                   "none": np.full((n, 2), np.nan, np.float32)}[mk]
            goal = np.where(np.isfinite(mxy), mxy, cfg["m"]).astype(np.float32)
            cube = cfg["cube"].copy()
            cube[:, 2] = np.mod(cube[:, 2] + dyaw, np.pi / 2)
            r = render_batches(env, n, lambda i: E.force_ptm(env, qpos[i], mxy[i], goal[i], cube[i]), bs)
            bank[f"rgb_{name}"], bank[f"state_{name}"] = r["rgb"], r["state"]
            bank["tcp_env"], bank["qpos"] = r["tcp"], r["qpos"]
        for name in PTM_STATES:
            if name != "base":
                bank[f"npx_{name}"], bank[f"pxl2_{name}"] = pixel_change(bank["rgb_base"], bank[f"rgb_{name}"])
        bank["tcp"], bank["J_tcp"] = E.tcp_pos_and_jac(bank["qpos"])
        # Signed-follow target of the marker move: a far point along the shift of the push point, so world_edit's
        # P(step toward target) becomes P(TCP step has a positive component along p(m2) - p(m)).
        p_m, p_m2 = E.push_point(cube_xy, cfg["m"]), E.push_point(cube_xy, cfg["m2"])
        sh = p_m2 - p_m
        bank["push_m"], bank["push_m2"] = p_m, p_m2
        bank["push_mlat"] = E.push_point(cube_xy, cfg["mlat"])
        bank["shift_m2"] = (bank["tcp"] + 1e3 * sh / np.linalg.norm(sh, axis=1, keepdims=True)).astype(np.float32)
        bank["nan3"] = np.full((n, 3), np.nan, np.float32)
        out[arm] = bank
    return out


def build_ptm_probe(env, n, seed, split, bs):
    cfg = E.sample_ptm_probe_configs(n, seed, split)
    r = render_batches(env, n, lambda i: E.force_ptm(env, cfg["qpos"][i], cfg["m"][i], cfg["m"][i], cfg["cube"][i]), bs)
    return dict(qpos_forced=cfg["qpos"], marker=cfg["m"], cube=cfg["cube"], rgb=r["rgb"], state=r["state"],
                tcp_env=r["tcp"], qpos=r["qpos"])


def check_bank(bank, name):
    """Forced qpos must be what the env holds, and the kinematic TCP must match the simulator's TCP."""
    q_env = bank.get("qpos", bank.get("qpos_env"))
    q_ref = bank.get("qpos_forced", bank.get("qpos"))
    tcp_fk, _ = E.tcp_pos_and_jac(q_env)
    msg = f"{name}: max |qpos env - forced| {np.abs(q_env - q_ref).max():.2e}, max |tcp FK - env| {np.abs(tcp_fk - bank['tcp_env']).max():.2e}"
    print(msg, flush=True)
    return msg


def example_sheet(bank, keys, path, k=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(k, len(keys), figsize=(2.0 * len(keys), 2.0 * k))
    for i in range(k):
        for j, key in enumerate(keys):
            ax[i, j].imshow(bank[key][i])
            ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(key.replace("rgb_", ""), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "LR", "SZ", "PTM"], required=True)
    p.add_argument("--kinds", nargs="+", default=["worldedit", "probe_train", "probe_test"])
    p.add_argument("--n_worldedit", type=int, default=512)
    p.add_argument("--n_probe_train", type=int, default=2048)
    p.add_argument("--n_probe_test", type=int, default=1024)
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()
    C.ensure_dirs()
    env = E.make_env(a.env, a.batch)
    seeds = {"AT": dict(worldedit=1000, probe_train=2000, probe_test=3000),
             "LR": dict(worldedit=1100, probe_train=2100, probe_test=3100),
             "SZ": dict(worldedit=1200, probe_train=2200, probe_test=3200),
             "PTM": dict(worldedit=1300, probe_train=2300, probe_test=3300)}[a.env]
    for kind in a.kinds:
        out = os.path.join(C.BANKS, f"{a.env}_{kind}.npz")
        if os.path.exists(out):
            print(f"{out} exists, skipping", flush=True)
            continue
        if a.env == "PTM":
            if kind == "worldedit":
                for arm, b in build_ptm_worldedit(env, a.n_worldedit, seeds[kind], a.batch).items():
                    nm = "PTM_worldedit" if arm == "start" else "PTM_hover"
                    b["check"] = np.array(check_bank(b, nm))
                    fn = os.path.join(C.BANKS, nm + ".npz")
                    np.savez_compressed(fn + ".tmp.npz", **b)
                    os.replace(fn + ".tmp.npz", fn)
                    example_sheet(b, [k for k in b if k.startswith("rgb_")], os.path.join(C.FIGURES, f"{nm}_examples.png"))
                    print(f"wrote {fn}", flush=True)
                continue
            b = build_ptm_probe(env, getattr(a, "n_" + kind), seeds[kind], "train" if kind == "probe_train" else "test",
                                a.batch)
        elif kind == "worldedit":
            b = dict(AT=build_at_worldedit, LR=build_lr_worldedit, SZ=build_sz_worldedit)[a.env](
                env, a.n_worldedit, seeds[kind], a.batch)
        elif a.env == "SZ":
            b = build_sz_probe(env, getattr(a, "n_" + kind), seeds[kind], a.batch)
        elif a.env == "AT":
            b = build_at_probe(env, getattr(a, "n_" + kind), seeds[kind], a.batch)
        else:
            b = build_lr_probe(env, getattr(a, "n_" + kind), seeds[kind], "train" if kind == "probe_train" else "test",
                               a.batch)
        b["check"] = np.array(check_bank(b, f"{a.env}_{kind}"))
        np.savez_compressed(out + ".tmp.npz", **b)
        os.replace(out + ".tmp.npz", out)
        if kind == "worldedit":
            keys = [k for k in b if k.startswith("rgb_")]
            example_sheet(b, keys, os.path.join(C.FIGURES, f"{a.env}_worldedit_examples.png"))
        print(f"wrote {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
