"""Stress test of the LR h_pre rewiring recovery: masked-marker BC bank, marker-decoupled / no-marker rollouts, videos."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis.rewire_lr import Rewired, bank_features, offpath_setup, teacher_seed, train_bc
from gm_robotics.analysis.rollouts import T_MAX, lr_first_contact

OUT = os.path.join(C.RESULTS, "rewire_check")
HEADS = os.path.join(OUT, "heads")
PATHS = os.path.join(OUT, "paths")
FIG = os.path.join(C.FIGURES, "rewire_check")
TRAIN_ARMS = ("fc_head", "pre", "remix", "pre_masked")
ROLL_ARMS = ("native", "fc_head", "pre", "remix", "pre_masked", "rand_walk")
CONDS = ("standard", "decoupled", "nomarker")
PARK = np.array([0.0, 0.0, -10.0], np.float32)
BC_SEED, BC_NEP, BC_BS = 4100, 192, 128


def bc_configs(n_sources=6):
    """The exact scene configs build_bc_bank drew (one block of BC_NEP per source agent)."""
    return E.sample_lr_probe_configs(BC_NEP * n_sources, BC_SEED, "train")


@torch.no_grad()
def render_static(env, qpos, goal, dxy):
    E.force_lr(env, qpos, goal, dxy)
    obs, _ = env.reset(seed=0)
    return obs["rgb"].cpu().numpy()


def mask_bank(a):
    """Re-render every BC frame from its stored qpos with the marker parked under the table.

    Each frame is rendered in the sub-env slot it was collected in, because rendering depends on the slot; a
    marker-visible re-render of the same frames measures how exactly the reset reproduces the stored pixels."""
    bank = C.load_bank("LR_bc")
    cfg = bc_configs()
    env = E.make_env("LR", BC_BS)
    ep, st = bank["episode"], bank["state"]
    slot = (ep % BC_NEP) % BC_BS
    order = [np.nonzero(slot == s)[0] for s in range(BC_BS)]
    rounds = max(len(o) for o in order)
    masked = np.zeros_like(bank["rgb"])
    rng = np.random.default_rng(0)
    check = set(rng.choice(rounds, size=min(a.check_rounds, rounds), replace=False).tolist())
    diffs = []
    for r in range(rounds):
        fr = np.array([o[r] if r < len(o) else o[-1] for o in order])
        q = st[fr, :9]
        g = np.concatenate([cfg["g"][ep[fr]], np.full((BC_BS, 1), E.LR_GOAL_Z, np.float32)], 1)
        d = cfg["d"][ep[fr]]
        live = np.array([r < len(o) for o in order])
        img = render_static(env, q, np.tile(PARK, (BC_BS, 1)), d)
        masked[fr[live]] = img[live]
        if r in check:
            vis = render_static(env, q, g, d)
            dd = np.abs(vis[live].astype(int) - bank["rgb"][fr[live]].astype(int)).max(-1)
            dm = np.abs(img[live].astype(int) - bank["rgb"][fr[live]].astype(int)).max(-1)
            diffs += [dict(round=r, frame=int(f), vis_frac_px_diff=float((x > 0).mean()), vis_max=int(x.max()),
                           masked_frac_px_diff=float((y > 0).mean())) for f, x, y in zip(fr[live], dd, dm)]
        if r % 50 == 0:
            print(f"mask_bank round {r}/{rounds}", flush=True)
    env.close()
    pd.DataFrame(diffs).to_csv(os.path.join(OUT, "mask_bank_fidelity.csv"), index=False)
    np.savez(os.path.join(C.BANKS, "LR_bc_nomarker.npz.tmp.npz"), rgb=masked)
    os.replace(os.path.join(C.BANKS, "LR_bc_nomarker.npz.tmp.npz"), os.path.join(C.BANKS, "LR_bc_nomarker.npz"))
    df = pd.DataFrame(diffs)
    print("fidelity (marker visible re-render vs stored): mean frac px differing "
          f"{df.vis_frac_px_diff.mean():.5f}, max {df.vis_max.max()}; masked vs stored {df.masked_frac_px_diff.mean():.5f}")


def base_agent(cohort, k, dev):
    """Frozen trunk; cohort 'init' is a randomly initialised network (training init scheme, init seed k)."""
    if cohort == "init":
        return C.build_agent(None, init_seed=k, dev=dev)
    return C.build_agent(C.ckpt_path("LR", cohort, k), dev=dev)


def head_path(cohort, k, arm):
    return os.path.join(HEADS, f"{cohort}_s{k}_{arm}.pt")


def train(a):
    """rewire_lr.py's exact protocol (teacher, init seed 100+k, split seed k), with the heads saved."""
    dev = torch.device("cuda")
    os.makedirs(HEADS, exist_ok=True)
    bank = C.load_bank("LR_bc")
    masked = None
    if "pre_masked" in a.arms:
        masked = dict(bank, rgb=C.load_bank("LR_bc_nomarker")["rgb"])
    teachers = {}
    for k in a.seeds:
        tk = teacher_seed(k, a.seeds)
        if tk not in teachers:
            ta = C.build_agent(C.ckpt_path("LR", "control", tk), dev=dev)
            teachers[tk] = (bank_features(ta, bank, dev)["mu"], C.sigma(ta))
        target, sig_t = teachers[tk]
        for cohort in a.cohorts:
            base = base_agent(cohort, k, dev)
            todo = [arm for arm in a.arms if not os.path.exists(head_path(cohort, k, arm))]
            if not todo:
                continue
            feats = bank_features(base, bank, dev)
            feats_m = bank_features(base, masked, dev) if masked is not None and "pre_masked" in todo else None
            for arm in todo:
                model = Rewired(base, "pre" if arm == "pre_masked" else arm, init_seed=100 + k).to(dev)
                f = feats_m if arm == "pre_masked" else feats
                log, n_par = train_bc(model, f, target, sig_t, bank["episode"], a.epochs, a.lr, a.batch, a.clip, seed=k)
                sd = {n: p.detach().cpu() for n, p in model.state_dict().items() if not n.startswith("base.")}
                torch.save(dict(state_dict=sd, arm=arm, cohort=cohort, seed=k, teacher=tk, n_trainable=n_par), head_path(cohort, k, arm))
                pd.DataFrame(log).to_csv(os.path.join(OUT, f"trainlog_{cohort}_s{k}_{arm}.csv"), index=False)
                print(f"trained {cohort} s{k} {arm}: val {log[-1]['val_loss']:.3f} train {log[-1]['train_loss']:.3f}", flush=True)
            del feats, feats_m
            torch.cuda.empty_cache()


def load_policy(cohort, k, arm, dev, rw_seed):
    base = base_agent(cohort, k, dev)
    if arm == "native":
        return lambda rgb, st: C.taps(base, rgb, st)["mu"]
    if arm == "rand_walk":
        sig = C.sigma(C.build_agent(C.ckpt_path("LR", "control", teacher_seed(k, [1, 2, 3])), dev=dev))
        g = torch.Generator(device=dev).manual_seed(rw_seed)
        return lambda rgb, st: sig * torch.randn(len(st), C.N_ACT, generator=g, device=dev)
    ck = torch.load(head_path(cohort, k, arm), map_location="cpu")
    model = Rewired(base, "pre" if arm == "pre_masked" else arm, init_seed=100 + k).to(dev)
    model.load_state_dict({**{n: p for n, p in model.state_dict().items() if n.startswith("base.")}, **ck["state_dict"]})
    model.eval()
    return model.act


def decoupled_markers(goal, dxy, seed_k, min_sep=0.10):
    """Marker A for each episode = another episode's off-path goal, >= min_sep from the scored goal B and
    >= the distractor clearance from this episode's distractor."""
    rng = np.random.default_rng(7000 + seed_k)
    n = len(goal)
    A = np.zeros_like(goal)
    for i in range(n):
        ok = (np.linalg.norm(goal[:, :2] - goal[i, :2], axis=1) >= min_sep) & \
             (np.linalg.norm(goal[:, :2] - dxy[i], axis=1) >= E.LR_DIST_MIN)
        ok[i] = False
        A[i] = goal[rng.choice(np.nonzero(ok)[0])]
    return A


def condition(off, cond, k):
    """(marker xyz rendered, scored xyz) for one condition."""
    g = off["goal"]
    if cond == "standard":
        return g, g
    if cond == "decoupled":
        return decoupled_markers(g, off["dxy"], k), g
    return np.tile(PARK, (len(g), 1)), g


@torch.no_grad()
def run_paths(env, act, qpos, marker, dxy, bs, keep_frames=0):
    paths, frames = [], None
    for sl in C.batched(len(qpos), bs):
        idx = np.arange(sl.start, sl.stop)
        pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
        E.force_lr(env, qpos[pad], marker[pad], dxy[pad])
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        path = [u.agent.tcp.pose.p.cpu().numpy()]
        if keep_frames and sl.start == 0:
            frames = dict(obs=[obs["rgb"][:keep_frames].cpu().numpy()], human=[env.render()[:keep_frames].cpu().numpy()])
        for _ in range(T_MAX):
            obs, *_ = env.step(act(obs["rgb"], obs["state"]))
            path.append(u.agent.tcp.pose.p.cpu().numpy())
            if keep_frames and sl.start == 0:
                frames["obs"].append(obs["rgb"][:keep_frames].cpu().numpy())
                frames["human"].append(env.render()[:keep_frames].cpu().numpy())
        paths.append(np.stack(path, 1)[: len(idx)])
        if keep_frames:
            break
    return np.concatenate(paths), frames


def rollout(a):
    dev = torch.device("cuda")
    os.makedirs(PATHS, exist_ok=True)
    env = E.make_env("LR", a.env_batch)
    corner = np.array([*E.CORNER, E.LR_GOAL_Z], np.float32)
    rows = []
    for k in a.seeds:
        off = offpath_setup(k)
        for cohort in a.cohorts:
            for arm in a.arms:
                act = None
                for cond in a.conds:
                    fn = os.path.join(PATHS, f"{cohort}_s{k}_{arm}_{cond}.npz")
                    if os.path.exists(fn):
                        continue
                    act = act or load_policy(cohort, k, arm, dev, rw_seed=1000 * k)
                    marker, scored = condition(off, cond, k)
                    path, _ = run_paths(env, act, off["qpos"], marker, off["dxy"], a.env_batch)
                    np.savez(fn, path=path.astype(np.float32), marker=marker, scored=scored, dxy=off["dxy"], corner=corner)
                    o_s = lr_first_contact(path, scored, corner)[0]
                    o_m = lr_first_contact(path, marker, corner)[0] if cond == "decoupled" else o_s
                    print(f"{cohort} s{k} {arm} {cond}: reach scored {np.mean(o_s == 'goal'):.3f} marker "
                          f"{np.mean(o_m == 'goal'):.3f} corner {np.mean(o_s == 'proxy'):.3f}", flush=True)
    env.close()


def topdown(ax, path_t, goal, marker, dxy, corner, scored_label):
    import matplotlib.patches as mp
    lo, hi = E.WS_LO, E.WS_HI
    ax.add_patch(mp.Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fill=False, ls=":", ec="#999"))
    if marker[2] > -1:
        ax.add_patch(mp.Circle(marker[:2], 0.025, color="#1BAA2E", alpha=0.8))
    ax.add_patch(mp.Circle(goal[:2], 0.03, fill=False, ec="#1BAA2E" if np.allclose(goal, marker) else "#7A1520", lw=1.5, ls="--"))
    ax.add_patch(mp.Circle(corner[:2], 0.03, fill=False, ec="#C45C58", lw=1.5))
    ax.plot(*corner[:2], "x", color="#C45C58")
    ax.add_patch(mp.Rectangle(dxy - 0.02, 0.04, 0.04, color="#E5C21A"))
    ax.plot(path_t[:, 0], path_t[:, 1], "-", color="#2A2A2A", lw=1)
    ax.plot(*path_t[0, :2], "o", color="#3D8B8B", ms=4)
    ax.plot(*path_t[-1, :2], "o", color="#2A2A2A", ms=5)
    ax.set_xlim(-0.30, 0.25), ax.set_ylim(-0.30, 0.30), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(scored_label, fontsize=8)


def video(a):
    """First n_vid off-path episodes (fixed rule) per (cohort, seed, arm, cond): human camera | agent camera | live map."""
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = torch.device("cuda")
    os.makedirs(FIG, exist_ok=True)
    env = E.make_env("LR", a.env_batch, render=True)
    corner = np.array([*E.CORNER, E.LR_GOAL_Z], np.float32)
    rows = []
    for k in a.seeds:
        off = offpath_setup(k)
        for cohort, arm, cond in a.vid_specs:
            marker, scored = condition(off, cond, k)
            act = load_policy(cohort, k, arm, dev, rw_seed=1000 * k)
            path, fr = run_paths(env, act, off["qpos"], marker, off["dxy"], a.env_batch, keep_frames=a.n_vid)
            path = path[: a.n_vid]
            out, tg, tc = lr_first_contact(path, scored[: a.n_vid], corner)
            out_m, tm, _ = lr_first_contact(path, marker[: a.n_vid], corner)
            ref = np.load(os.path.join(PATHS, f"{cohort}_s{k}_{arm}_{cond}.npz"))["path"][: a.n_vid]
            for i in range(a.n_vid):
                rows.append(dict(cohort=cohort, seed=k, arm=arm, cond=cond, episode=i, outcome_scored=out[i], t_scored=int(tg[i]),
                                 outcome_marker=out_m[i], t_marker=int(tm[i]), t_corner=int(tc[i]),
                                 max_path_diff_vs_rollout=float(np.abs(path[i] - ref[i]).max())))
                lab = {"goal": "REACHED", "proxy": "CORNER", "neither": "neither"}[out[i]]
                if cond == "decoupled":
                    lab += f" (marker A: {out_m[i]})"
                clip = []
                for t in range(T_MAX + 1):
                    fig = plt.figure(figsize=(7.68, 2.56), dpi=100)
                    ax0, ax1, ax2 = fig.add_axes([0, 0, 1 / 3, 1]), fig.add_axes([1 / 3, 0, 1 / 3, 1]), fig.add_axes([2 / 3 + .01, .02, 1 / 3 - .02, .86])
                    ax0.imshow(fr["human"][t][i]), ax1.imshow(fr["obs"][t][i])
                    for ax in (ax0, ax1):
                        ax.axis("off")
                    ax0.text(4, 16, f"{cohort} s{k} {arm} {cond} ep{i} t={t}", fontsize=7, color="w", backgroundcolor="k")
                    topdown(ax2, path[i, : t + 1], scored[i], marker[i], off["dxy"][i], corner, lab)
                    fig.canvas.draw()
                    clip.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
                    plt.close(fig)
                base = os.path.join(FIG, f"vid_{cohort}_s{k}_{arm}_{cond}_ep{i}_{out[i]}")
                imageio.mimsave(base + ".mp4", clip, fps=8)
                imageio.imwrite(base + "_strip.png", np.concatenate([clip[t] for t in (0, 10, 20, 30, 50)], 0)[::2, ::2])
            print(f"video {cohort} s{k} {arm} {cond}: {list(out)}", flush=True)
    env.close()
    df = pd.DataFrame(rows)
    fn = os.path.join(OUT, "video_episodes.csv")
    df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["mask_bank", "train", "rollout", "video"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--cohorts", nargs="+", default=["gm", "control"])
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--conds", nargs="+", default=list(CONDS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--env_batch", type=int, default=128)
    p.add_argument("--check_rounds", type=int, default=20)
    p.add_argument("--n_vid", type=int, default=8)
    p.add_argument("--vid_specs", nargs="+", default=["gm:native:standard", "gm:pre:standard", "gm:remix:standard",
                                                       "control:pre:standard", "gm:pre:decoupled", "gm:pre_masked:standard"],
                   help="cohort:arm:cond")
    a = p.parse_args()
    a.vid_specs = [tuple(s.split(":")) for s in a.vid_specs]
    a.arms = a.arms or list(TRAIN_ARMS if a.step == "train" else ROLL_ARMS)
    os.makedirs(OUT, exist_ok=True)
    dict(mask_bank=mask_bank, train=train, rollout=rollout, video=video)[a.step](a)


if __name__ == "__main__":
    main()
