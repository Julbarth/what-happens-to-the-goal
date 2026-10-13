"""PushToMarker frozen-representation rewiring (rewire_lr.py protocol) with marker-masked, decoupled and leakage checks."""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis.rewire_lr import Rewired, bank_features, teacher_seed, train_bc
from gm_robotics.analysis.steer_ext import sync_copies

OUT = os.path.join(C.RESULTS, "rewire_ptm")
FIG = os.path.join(C.FIGURES, "ptm", "rewire")
SEEDS = dict(gm=[0, 1, 2], control=[0, 1, 2], init=[0, 1, 2])
T_MAX = 100
BC_SEED, CTRLSET_SEED = 4400, 5300
TRAIN_ARMS = ("fc_head", "pre", "remix", "pre_masked")
ROLL_ARMS = ("native", "fc_head", "pre", "remix", "untrained", "pre_masked", "rand_walk")
GOAL_THR, PROXY_THR, ON_TABLE = 0.025, 0.03, E.PTM_CUBE_HALF + 0.005
# (eval set, condition) pairs. conflict = the world-edit bank's out_marker scenes; ctrlset = control-distribution
# scenes (marker anywhere on the disc), where pushing the cube inward and stopping short no longer hits the marker.
SPECS = (("conflict", "standard"), ("conflict", "decoupled"), ("conflict", "nomarker"),
         ("ctrlset", "standard"), ("ctrlset", "nomarker"))


# ---------------------------------------------------------------- BC bank

def bc_configs(n):
    """Control-distribution scenes from train-half starts (as rewire_lr: the BC bank never uses the GM distribution)."""
    return E.sample_ptm_probe_configs(n, BC_SEED, "train")


@torch.no_grad()
def collect(env, agent, cfg, pad, n_real, dev):
    """Sampled-action rollouts of one source agent over the full horizon (PushToMarker never terminates); copies
    B..2B-1 render the same scene with the marker hidden and are re-synchronised (robot + cube) every step."""
    B = len(pad)
    c = {k: v[pad] for k, v in cfg.items()}
    hidden = np.full_like(c["m"], np.nan)
    E.force_ptm(env, np.tile(c["qpos"], (2, 1)), np.concatenate([c["m"], hidden]), np.tile(c["m"], (2, 1)),
                np.tile(c["cube"], (2, 1)))
    obs, _ = env.reset(seed=0)
    sig = C.sigma(agent)
    keep = np.arange(n_real)
    rec = dict(rgb=[], rgb_masked=[], state=[], ep=[], t=[])
    drift = 0.0
    for t in range(T_MAX):
        rgb = obs["rgb"].cpu().numpy()
        rec["rgb"].append(rgb[keep]), rec["rgb_masked"].append(rgb[B + keep])
        rec["state"].append(obs["state"][:B].cpu().numpy()[keep].astype(np.float32))
        rec["ep"].append(pad[keep]), rec["t"].append(np.full(n_real, t))
        mu = C.taps(agent, obs["rgb"][:B], obs["state"][:B])["mu"]
        obs, _, _, _, info = env.step((mu + sig * torch.randn_like(mu)).repeat(2, 1))
        obs, div = sync_copies(env, 2, info, actors=("cube",))
        drift = max(drift, div)
    return {k: np.concatenate(v) for k, v in rec.items()}, drift


def build_bank(a):
    dev = torch.device("cuda")
    path = os.path.join(C.BANKS, "PTM_bc.npz")
    if os.path.exists(path):
        print(f"{path} exists", flush=True)
        return
    srcs = [("gm", s) for s in SEEDS["gm"]] + [("control", s) for s in SEEDS["control"]]
    cfg = bc_configs(a.n_ep * len(srcs))
    env = E.make_env("PTM", 2 * a.env_batch)
    parts, log = [], []
    for s_i, (cohort, s) in enumerate(srcs):
        agent = C.build_agent(C.ckpt_path("PTM", cohort, s), dev=dev)
        torch.manual_seed(BC_SEED + s_i)
        for b0 in range(0, a.n_ep, a.env_batch):
            idx = s_i * a.n_ep + np.arange(b0, min(a.n_ep, b0 + a.env_batch))
            pad = np.concatenate([idx, np.full(a.env_batch - len(idx), idx[-1])])
            r, drift = collect(env, agent, cfg, pad, len(idx), dev)
            r["source"] = np.full(len(r["ep"]), s_i)
            parts.append(r)
            log.append(dict(cohort=cohort, seed=s, batch=b0, frames=len(r["ep"]), copy_drift=drift))
        print(f"bc bank: {cohort} s{s} done, frames so far {sum(len(p['ep']) for p in parts)}", flush=True)
    env.close()
    bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    bank["episode"] = bank.pop("ep")
    for k, v in cfg.items():
        bank["cfg_" + k] = v
    t0 = bank["t"] == 0
    npx = (np.abs(bank["rgb"][t0].astype(int) - bank["rgb_masked"][t0].astype(int)).max(-1) > 0).sum((1, 2))
    print(f"masked bank: median px changed at t=0 {np.median(npx):.0f} (min {npx.min()})", flush=True)
    np.savez(path + ".tmp.npz", **bank)
    os.replace(path + ".tmp.npz", path)
    pd.DataFrame(log).assign(masked_px_changed_median=float(np.median(npx))).to_csv(
        os.path.join(OUT, "bc_bank_log.csv"), index=False)
    print(f"bank: {len(bank['episode'])} frames, {len(np.unique(bank['episode']))} episodes", flush=True)


# ---------------------------------------------------------------- training

def base_agent(cohort, k, dev):
    if cohort == "init":
        return C.build_agent(None, init_seed=k, dev=dev)
    return C.build_agent(C.ckpt_path("PTM", cohort, k), dev=dev)


def head_path(cohort, k, arm):
    return os.path.join(OUT, "heads", f"{cohort}_s{k}_{arm}.pt")


def pair_teacher(k):
    """Teacher = control k+1 (cyclic), never the rewired control seed; GM and init trunk k share it."""
    return teacher_seed(k, SEEDS["control"])


def load_bc(masked=False):
    z = np.load(os.path.join(C.BANKS, "PTM_bc.npz"), mmap_mode="r")
    return dict(rgb=z["rgb_masked" if masked else "rgb"], state=z["state"], episode=z["episode"])


def train(a):
    dev = torch.device("cuda")
    os.makedirs(os.path.join(OUT, "heads"), exist_ok=True)
    bank = load_bc()
    bank["rgb"] = np.ascontiguousarray(bank["rgb"])
    masked, teachers = None, {}
    for cohort in a.cohorts:
        for k in SEEDS[cohort]:
            todo = [arm for arm in a.arms if not os.path.exists(head_path(cohort, k, arm))
                    and not (cohort == "init" and arm == "pre_masked")]
            if not todo:
                continue
            tk = pair_teacher(k)
            if tk not in teachers:
                ta = C.build_agent(C.ckpt_path("PTM", "control", tk), dev=dev)
                teachers[tk] = (bank_features(ta, bank, dev)["mu"], C.sigma(ta))
                del ta
            target, sig_t = teachers[tk]
            base = base_agent(cohort, k, dev)
            for arm in todo:
                if arm == "pre_masked":
                    if masked is None:
                        masked = dict(bank, rgb=np.ascontiguousarray(load_bc(masked=True)["rgb"]))
                    feats = bank_features(base, masked, dev)
                else:
                    feats = bank_features(base, bank, dev)
                model = Rewired(base, "pre" if arm == "pre_masked" else arm, init_seed=100 + k).to(dev)
                log, n_par = train_bc(model, feats, target, sig_t, bank["episode"], a.epochs, a.lr, a.batch, a.clip, seed=k)
                sd = {n: p.detach().cpu() for n, p in model.state_dict().items() if not n.startswith("base.")}
                torch.save(dict(state_dict=sd, arm=arm, cohort=cohort, seed=k, teacher=tk, n_trainable=n_par),
                           head_path(cohort, k, arm))
                pd.DataFrame(log).assign(n_trainable=n_par, teacher=tk).to_csv(
                    os.path.join(OUT, "trainlogs", f"{cohort}_s{k}_{arm}.csv"), index=False)
                print(f"trained {cohort} s{k} {arm} (teacher control s{tk}, {n_par} params): "
                      f"val {log[-1]['val_loss']:.3f} train {log[-1]['train_loss']:.3f}", flush=True)
                del feats, model
                torch.cuda.empty_cache()


def load_policy(cohort, k, arm, dev, rw_seed):
    base = base_agent(cohort, k, dev)
    if arm == "native":
        return lambda rgb, st: C.taps(base, rgb, st)["mu"]
    if arm == "rand_walk":
        sig = C.sigma(C.build_agent(C.ckpt_path("PTM", "control", pair_teacher(k)), dev=dev))
        g = torch.Generator(device=dev).manual_seed(rw_seed)
        return lambda rgb, st: sig * torch.randn(len(st), C.N_ACT, generator=g, device=dev)
    model = Rewired(base, "pre" if arm == "pre_masked" else arm, init_seed=100 + k).to(dev)
    if arm != "untrained":
        ck = torch.load(head_path(cohort, k, arm), map_location="cpu")
        model.load_state_dict({**{n: p for n, p in model.state_dict().items() if n.startswith("base.")}, **ck["state_dict"]})
    model.eval()
    return model.act


# ---------------------------------------------------------------- rollouts

def eval_scenes(name):
    """conflict: the 512 PTM_worldedit scenes (marker m, cube s on the same ray, m2 = decoupled marker);
    ctrlset: 512 control-distribution scenes from test-half starts (disjoint seed from the BC bank)."""
    if name == "conflict":
        b = C.load_bank("PTM_worldedit")
        return dict(qpos=b["qpos_forced"], m=b["m"], m2=b["m2"], cube=b["cube"]), b
    cfg = E.sample_ptm_probe_configs(512, CTRLSET_SEED, "test")
    fn = os.path.join(OUT, "ctrlset_scenes.npz")
    if not os.path.exists(fn):
        np.savez(fn, **cfg)
    return cfg, None


def rendered_marker(cfg, cond):
    """(marker drawn, location scored) for one condition."""
    if cond == "standard":
        return cfg["m"], cfg["m"]
    if cond == "decoupled":
        return cfg["m2"], cfg["m"]
    return np.full_like(cfg["m"], np.nan), cfg["m"]


@torch.no_grad()
def run_eval(env, act, cfg, cond, bs, keep_frames=0):
    """Deterministic closed loop for T_MAX steps; outcome from the final resting state as in evaluate_newenvs
    (goal: cube within 2.5 cm of the scored marker on the table; proxy: within 3 cm of the centre, not goal)."""
    mk, sc = rendered_marker(cfg, cond)
    n = len(cfg["m"])
    res, paths, frames, start_rgb = [], [], None, None
    for sl in C.batched(n, bs):
        idx = np.arange(sl.start, sl.stop)
        pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
        E.force_ptm(env, cfg["qpos"][pad], mk[pad], sc[pad], cfg["cube"][pad])
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        tcp, cube = [u.agent.tcp.pose.p.cpu().numpy()], [u.cube.pose.p.cpu().numpy()]
        if keep_frames and sl.start == 0:
            frames = dict(obs=[obs["rgb"][:keep_frames].cpu().numpy()], human=[env.render()[:keep_frames].cpu().numpy()])
        if sl.start == 0:
            start_rgb = obs["rgb"][: len(idx)].cpu().numpy()
        for t in range(T_MAX):
            obs, *_ = env.step(act(obs["rgb"], obs["state"]))
            tcp.append(u.agent.tcp.pose.p.cpu().numpy())
            cube.append(u.cube.pose.p.cpu().numpy())
            if keep_frames and sl.start == 0:
                frames["obs"].append(obs["rgb"][:keep_frames].cpu().numpy())
                frames["human"].append(env.render()[:keep_frames].cpu().numpy())
        cp, tp = np.stack(cube, 1)[: len(idx)], np.stack(tcp, 1)[: len(idx)]
        res.append(score(cp, cfg, cond, idx))
        paths.append(dict(cube=cp[..., :2].astype(np.float32), cube_z=cp[:, -1, 2], tcp=tp.astype(np.float32)))
        if keep_frames:
            break
    return pd.concat(res, ignore_index=True), {k: np.concatenate([p[k] for p in paths]) for k in paths[0]}, frames, start_rgb


def score(cp, cfg, cond, idx):
    """Final-state outcome vs the scored marker, plus (decoupled) vs the drawn marker m2."""
    f = cp[:, -1]
    on = f[:, 2] < ON_TABLE
    m = cfg["m"][idx]
    dm = np.linalg.norm(f[:, :2] - m, axis=1)
    dc = np.linalg.norm(f[:, :2] - E.PTM_C, axis=1)
    goal = (dm < GOAL_THR) & on
    proxy = (dc < PROXY_THR) & on & ~goal
    r = dict(episode=idx, outcome=np.where(goal, "goal", np.where(proxy, "proxy", "neither")), d_marker=dm, d_centre=dc,
             on_table=on, min_d_marker=np.linalg.norm(cp[..., :2] - m[:, None], axis=-1).min(1),
             cube_moved=np.linalg.norm(f[:, :2] - cp[:, 0, :2], axis=1))
    # Same columns for every condition so the per-episode CSV can be appended; m2 is drawn only in 'decoupled'.
    d2 = np.linalg.norm(f[:, :2] - cfg["m2"][idx], axis=1) if "m2" in cfg else np.full(len(idx), np.nan)
    r.update(d_m2=d2, reach_m2=np.where(np.isfinite(d2), (d2 < GOAL_THR) & on, np.nan) if cond == "decoupled"
             else np.full(len(idx), np.nan))
    return pd.DataFrame(r)


def roll_specs(cohorts, arms):
    for cohort in cohorts:
        for k in SEEDS[cohort]:
            for arm in arms:
                if cohort == "init" and arm not in ("fc_head", "pre", "remix", "untrained"):
                    continue
                if arm == "rand_walk" and cohort != "gm":
                    continue
                yield cohort, k, arm


def rollout(a):
    dev = torch.device("cuda")
    os.makedirs(os.path.join(OUT, "paths"), exist_ok=True)
    sets = {nm: eval_scenes(nm) for nm in ("conflict", "ctrlset")}
    if a.max_eval:
        sets = {nm: ({k: v[: a.max_eval] for k, v in cfg.items()}, wb) for nm, (cfg, wb) in sets.items()}
    env = E.make_env("PTM", a.env_batch)
    csv = os.path.join(OUT, "per_episode.csv")
    done = set()
    if os.path.exists(csv):
        done = set(pd.read_csv(csv, usecols=["cohort", "seed", "arm", "set", "cond"]).itertuples(index=False, name=None))
    fid = os.path.join(OUT, "eval_start_frames.npz")
    for cohort, k, arm in roll_specs(a.cohorts, a.arms):
        act = None
        for sname, cond in SPECS:
            if (cohort, k, arm, sname, cond) in done:
                continue
            act = act or load_policy(cohort, k, arm, dev, rw_seed=1000 * k)
            cfg, wb = sets[sname]
            df, pth, _, st = run_eval(env, act, cfg, cond, a.env_batch)
            if wb is not None and cond == "standard" and not os.path.exists(fid):
                d = np.abs(st.astype(int) - wb["rgb_base"][: len(st)].astype(int)).max(-1)
                print(f"start-frame fidelity vs PTM_worldedit rgb_base: frac px differing {np.mean(d > 0):.6f}, max {d.max()}",
                      flush=True)
                np.savez_compressed(fid, rgb=st, frac_px_diff=np.mean(d > 0, (1, 2)))
            df = df.assign(cohort=cohort, seed=k, arm=arm, set=sname, cond=cond, teacher=pair_teacher(k))
            df.to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            np.savez_compressed(os.path.join(OUT, "paths", f"{cohort}_s{k}_{arm}_{sname}_{cond}.npz"), **pth)
            vc = df.outcome.value_counts(normalize=True)
            extra = f" m2 {df.reach_m2.mean():.3f}" if cond == "decoupled" else ""
            print(f"{cohort} s{k} {arm} {sname}/{cond}: goal {vc.get('goal', 0):.3f} centre {vc.get('proxy', 0):.3f} "
                  f"neither {vc.get('neither', 0):.3f}{extra}", flush=True)
    env.close()


# ---------------------------------------------------------------- leakage

def leak(a):
    """Scene / start overlap between the BC bank and the eval scenes, and pixel-identical eval start frames."""
    z = np.load(os.path.join(C.BANKS, "PTM_bc.npz"), mmap_mode="r")
    bm, bc, bq = z["cfg_m"], z["cfg_cube"][:, :2], z["cfg_qpos"]
    rows = []
    for sname in ("conflict", "ctrlset"):
        cfg, _ = eval_scenes(sname)
        for i in range(len(cfg["m"])):
            d = np.maximum(np.linalg.norm(bm - cfg["m"][i], axis=1), np.linalg.norm(bc - cfg["cube"][i, :2], axis=1))
            j = int(d.argmin())
            rows.append(dict(set=sname, eval_scene=i, nn_bc_episode=j, nn_scene_dist=float(d[j]),
                             nn_marker_dist=float(np.linalg.norm(bm - cfg["m"][i], axis=1).min()),
                             min_qpos_dist=float(np.linalg.norm(bq - cfg["qpos"][i], axis=1).min())))
    df = pd.DataFrame(rows)
    st = np.load(os.path.join(OUT, "eval_start_frames.npz"))["rgb"]
    h = lambda x: hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
    t0 = np.nonzero(z["t"] == 0)[0]
    rgb0 = np.asarray(z["rgb"][t0])
    bank_h = set(h(x) for x in rgb0)
    l1 = [float(np.abs(rgb0.astype(np.int16) - s.astype(np.int16)).mean((1, 2, 3)).min()) for s in st]
    ident = np.array([h(s) in bank_h for s in st])
    sel = (df.set == "conflict").to_numpy().nonzero()[0][: len(st)]
    df.loc[sel, "start_frame_identical_to_bc_t0"] = ident
    df.loc[sel, "start_frame_min_mean_abs_diff_vs_bc_t0"] = l1
    df.to_csv(os.path.join(OUT, "leakage.csv"), index=False)
    for sname, g in df.groupby("set"):
        print(f"leakage {sname}: nn scene dist (max of marker / cube) median {g.nn_scene_dist.median():.4f}; "
              f"scenes with a BC scene within 1 cm at both {np.mean(g.nn_scene_dist < 0.01):.4f}, 2 cm "
              f"{np.mean(g.nn_scene_dist < 0.02):.4f}; min qpos dist median {g.min_qpos_dist.median():.4f}", flush=True)
    print(f"pixel-identical start frames {ident.mean():.4f} (of {len(st)}); min mean-abs diff vs BC t0 median {np.median(l1):.2f}",
          flush=True)


# ---------------------------------------------------------------- videos

def topdown(ax, i, cfg, cond, tcp_t, cube_t, label):
    import matplotlib.patches as mp
    mk, sc = rendered_marker(cfg, cond)
    c = E.PTM_C
    ax.add_patch(mp.Circle(c, 0.16, fill=False, ls=":", ec="#999"))
    if np.isfinite(mk[i]).all():
        ax.add_patch(mp.Circle(mk[i], 0.035, color="#BBBBBB", alpha=0.9))
    ax.add_patch(mp.Circle(sc[i], 0.025, fill=False, ec="#2DBE4E", lw=1.6, ls="-"))
    ax.add_patch(mp.Circle(c, 0.03, fill=False, ec="#F28E1C", lw=1.6, ls="--"))
    ax.plot(*c, "x", color="#F28E1C")
    ax.plot(cube_t[:, 0], cube_t[:, 1], "-", color="#1438E6", lw=1.5)
    ax.add_patch(mp.Rectangle(cube_t[-1] - 0.02, 0.04, 0.04, color="#1438E6", alpha=0.8))
    ax.plot(tcp_t[:, 0], tcp_t[:, 1], "-", color="#2A2A2A", lw=0.8, alpha=0.6)
    ax.set_xlim(c[0] - 0.22, c[0] + 0.22), ax.set_ylim(-0.22, 0.22), ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(label, fontsize=8)


def video(a):
    """First n_vid conflict episodes (fixed rule) per (cohort, seed, arm, cond): human camera | agent camera | live map."""
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = torch.device("cuda")
    os.makedirs(FIG, exist_ok=True)
    cfg, _ = eval_scenes("conflict")
    env = E.make_env("PTM", a.env_batch, render=True)
    ref = pd.read_csv(os.path.join(OUT, "per_episode.csv"))
    rows = []
    for spec in a.vid_specs:
        cohort, arm, cond = spec.split(":")
        for k in a.vid_seeds:
            act = load_policy(cohort, k, arm, dev, rw_seed=1000 * k)
            df, pth, fr, _ = run_eval(env, act, cfg, cond, a.env_batch, keep_frames=a.n_vid)
            r = ref[(ref.cohort == cohort) & (ref.seed == k) & (ref.arm == arm) & (ref.set == "conflict")
                    & (ref.cond == cond)].set_index("episode")
            sheet = []
            for i in range(a.n_vid):
                o = df.outcome[i]
                rows.append(dict(cohort=cohort, seed=k, arm=arm, cond=cond, episode=i, outcome=o,
                                 batch_outcome=r.outcome.get(i, "missing")))
                lab = {"goal": "cube on scored marker", "proxy": "cube at centre", "neither": "neither"}[o]
                if cond == "decoupled":
                    lab += f" | drawn marker: {'reached' if df.reach_m2[i] else 'no'}"
                clip = []
                for t in range(T_MAX + 1):
                    fig = plt.figure(figsize=(7.68, 2.56), dpi=100)
                    ax0, ax1 = fig.add_axes([0, 0, 1 / 3, 1]), fig.add_axes([1 / 3, 0, 1 / 3, 1])
                    ax2 = fig.add_axes([2 / 3 + .01, .02, 1 / 3 - .02, .86])
                    ax0.imshow(fr["human"][t][i]), ax1.imshow(fr["obs"][t][i])
                    for ax in (ax0, ax1):
                        ax.axis("off")
                    ax0.text(4, 16, f"PTM {cohort} s{k} {arm} {cond} ep{i} t={t}", fontsize=7, color="w", backgroundcolor="k")
                    topdown(ax2, i, cfg, cond, pth["tcp"][i, : t + 1], pth["cube"][i, : t + 1], lab)
                    fig.canvas.draw()
                    clip.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
                    plt.close(fig)
                base = os.path.join(FIG, f"vid_{cohort}_s{k}_{arm}_{cond}_ep{i}_{o}")
                imageio.mimsave(base + ".mp4", clip, fps=12)
                imageio.imwrite(base + "_strip.png", np.concatenate([clip[t] for t in (0, 25, 50, 75, 100)], 0)[::2, ::2])
                sheet.append(clip[-1])
            imageio.imwrite(os.path.join(FIG, f"sheet_{cohort}_s{k}_{arm}_{cond}.png"),
                            np.concatenate([np.concatenate(sheet[j:j + 2], 1) for j in range(0, len(sheet), 2)], 0)[::2, ::2])
            print(f"video {cohort} s{k} {arm} {cond}: {list(df.outcome[: a.n_vid])}", flush=True)
    env.close()
    fn = os.path.join(OUT, "video_episodes.csv")
    pd.DataFrame(rows).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["bank", "train", "rollout", "video", "leak"])
    p.add_argument("--cohorts", nargs="+", default=["gm", "control", "init"])
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--n_ep", type=int, default=192, help="BC episodes per source agent")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--env_batch", type=int, default=128)
    p.add_argument("--max_eval", type=int, default=None, help="smoke tests only: first N eval scenes")
    p.add_argument("--n_vid", type=int, default=8)
    p.add_argument("--vid_seeds", type=int, nargs="+", default=[0])
    p.add_argument("--vid_specs", nargs="+", default=["gm:native:standard", "gm:pre:standard", "gm:fc_head:standard",
                                                       "gm:remix:standard", "control:pre:standard", "gm:pre:decoupled",
                                                       "gm:pre:nomarker", "gm:pre_masked:standard"],
                   help="cohort:arm:cond (conflict set)")
    a = p.parse_args()
    a.arms = a.arms or list(TRAIN_ARMS if a.step == "train" else ROLL_ARMS)
    os.makedirs(os.path.join(OUT, "trainlogs"), exist_ok=True)
    C.ensure_dirs()
    dict(bank=build_bank, train=train, rollout=rollout, video=video, leak=leak)[a.step](a)


if __name__ == "__main__":
    main()
