"""SortZones tie-test outcomes and videos, and h_fc steering rollout videos (AT / SZ / LR / PTM): camera plus live top-down map."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E

VID = os.path.join(C.FIGURES, "videos")
OUT = os.path.join(C.RESULTS, "videos")
N_VID = 4
COL = dict(red="#D91414", blue="#1438E6", green="#0DD926", purple="#8C1ACC", cube="#E5C21A")
GOAL_RING, PROXY_RING = C.PAL["gm"], C.PAL["gm_proxy"]


# ---------------------------------------------------------------- shared drawing

def frame_figure(cam, draw_map, title, sub):
    """One video frame: human camera (left) | top-down map (right)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6.4, 3.2), dpi=100)
    ax0 = fig.add_axes([0, 0, 0.5, 0.9])
    ax0.imshow(cam), ax0.axis("off")
    ax1 = fig.add_axes([0.53, 0.03, 0.45, 0.84])
    draw_map(ax1)
    fig.text(0.01, 0.955, title, fontsize=8, color=C.PAL["ink"], va="center")
    fig.text(0.99, 0.955, sub, fontsize=8, color=C.PAL["ink"], va="center", ha="right", fontweight="bold")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def P(xy):
    """Robot-frame (x, y) to map coordinates (y, x): the robot sits at the top and its left (+y) is on the right,
    as in both cameras, which look at the table from the far side."""
    xy = np.asarray(xy)
    return xy[..., ::-1][..., :2] if xy.shape[-1] == 2 else np.stack([xy[..., 1], xy[..., 0]], -1)


def finish(ax, xlim, ylim):
    ax.set_xlim(*ylim), ax.set_ylim(xlim[1], xlim[0])
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(C.PAL["mute"])


def rings(ax, goal, proxy, r_goal, r_proxy=None):
    import matplotlib.patches as mp
    if goal is not None:
        ax.add_patch(mp.Circle(P(goal), r_goal, fill=False, ec=GOAL_RING, lw=2.2, ls="-"))
    for p in proxy:
        ax.add_patch(mp.Circle(P(p), r_proxy or r_goal, fill=False, ec=PROXY_RING, lw=2.2, ls="--"))


def path(ax, tcp, col=None):
    q = P(tcp[:, :2])
    ax.plot(q[:, 0], q[:, 1], "-", color=col or C.PAL["ink"], lw=1.2)
    ax.plot(*q[-1], "o", color=col or C.PAL["ink"], ms=5)


def sz_map(ax, u, v, col_u, col_v, cross_u, tcp, cube, goal_xy, proxy_xy):
    """Pads at u / v with their colours, cross, goal ring (cross pad) and proxy ring (each green pad), cube path."""
    import matplotlib.patches as mp
    for xy, c, cr in ((u, col_u, cross_u), (v, col_v, not cross_u)):
        ax.add_patch(mp.Rectangle(P(xy) - 0.05, 0.10, 0.10, color=COL[c], alpha=0.85))
        if cr:
            ax.plot(*P(xy), marker="x", color="w", ms=10, mew=3)
    rings(ax, goal_xy, proxy_xy, 0.075, 0.088)
    q = P(cube[:, :2])
    ax.plot(q[:, 0], q[:, 1], "-", color=COL["cube"], lw=2.0)
    ax.add_patch(mp.Rectangle(q[-1] - 0.02, 0.04, 0.04, color=COL["cube"], ec="k", lw=0.5))
    qt = P(tcp[:, :2])
    ax.plot(qt[:, 0], qt[:, 1], "-", color=C.PAL["ink"], lw=0.8, alpha=0.6)
    ax.plot(*qt[-1], "o", color=C.PAL["ink"], ms=4)
    finish(ax, (-0.30, 0.15), (-0.24, 0.24))


def at_map(ax, u, v, tcp):
    """Blue cube at u (goal, solid ring), red sphere at v (proxy, dashed ring), TCP path."""
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle(P(u) - 0.03, 0.06, 0.06, color=COL["blue"]))
    ax.add_patch(mp.Circle(P(v), 0.03, color=COL["red"]))
    rings(ax, u, [v], 0.06)
    path(ax, tcp)
    finish(ax, (-0.30, 0.22), (-0.26, 0.26))


def lr_map(ax, goal, corner, dxy, tcp):
    """Marker (goal, solid ring), training corner (proxy, dashed ring), yellow distractor, TCP path."""
    import matplotlib.patches as mp
    ax.add_patch(mp.Circle(P(goal[:2]), 0.02, color=COL["green"]))
    ax.add_patch(mp.Circle(P(dxy), 0.02, color="#E5C21A"))
    rings(ax, goal[:2], [corner[:2]], 0.045)
    path(ax, tcp)
    finish(ax, (-0.35, 0.25), (-0.32, 0.32))


def write_clip(frames, base, fps, gif_scale=1):
    """mp4 at full rate, GIF at every second frame (and gif_scale downsampling) holding the last frame ~1 s."""
    import imageio
    os.makedirs(os.path.dirname(base), exist_ok=True)
    imageio.mimsave(base + ".mp4", frames, fps=fps, macro_block_size=1)
    s = gif_scale
    small = [f[::s, ::s] for f in frames[::2]] + [frames[-1][::s, ::s]] * max(1, fps // 2)
    imageio.mimsave(base + ".gif", small, duration=1000 * 2 / fps, loop=0)


def grid_clip(clips, labels_rows, base, fps):
    """Rows = agents / arms, columns = episodes; shorter clips hold their last frame."""
    T = max(len(c) for row in clips for c in row)
    rows = []
    for row in clips:
        cells = [np.stack(c + [c[-1]] * (T - len(c))) for c in row]
        rows.append(np.concatenate(cells, 2))
    vid = np.concatenate(rows, 1)
    write_clip([vid[t] for t in range(T)], base, fps, gif_scale=2)


# ---------------------------------------------------------------- SortZones tie test

TIE_CONDS = ("conflict", "tie_purple", "tie_green")
TIE_LABEL = dict(conflict="standard conflict", tie_purple="both purple", tie_green="both green")


def make_tie_env(n, render):
    import gymnasium as gym
    from gm_robotics.analysis import sz_tie  # noqa: F401  registers SortZonesTie-v1
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs import Pose
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper

    if "SortZonesTieConflict-v1" not in gym.registry:
        @register_env("SortZonesTieConflict-v1", max_episode_steps=100)
        class SortZonesTieConflictEnv(sz_tie.SortZonesTieEnv):
            """sz_tie's native reset; tie == 'conflict' recolours the marked pad purple and the other green, so the
            standard conflict uses exactly the tie episodes' geometry."""

            def _initialize_episode(self, env_idx, options):
                conflict = self.tie == "conflict"
                if conflict:
                    self.tie = None
                super()._initialize_episode(env_idx, options)
                if not conflict:
                    return
                self.tie = "conflict"
                with torch.device(self.device):
                    z = torch.full((len(env_idx), 1), E.SZ_PAD_Z)
                    self.pad_purple.set_pose(Pose.create_from_pq(torch.cat([self.marked_xy[env_idx], z], 1)))
                    self.pad_green.set_pose(Pose.create_from_pq(torch.cat([self.other_xy[env_idx], z], 1)))
                    self.marked_green[env_idx] = False

    env = gym.make("SortZonesTieConflict-v1", num_envs=n, obs_mode="rgb", sim_backend="physx_cuda", variant="control",
                   control_mode="pd_joint_delta_pos", reconfiguration_freq=None,
                   render_mode="rgb_array" if render else None)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


def set_tie(u, cond):
    u.tie = {"native": None, "conflict": "conflict", "tie_purple": "purple", "tie_green": "green"}[cond]


def tie_eval(a):
    """Standard conflict on the tie episodes' geometry (sz_tie protocol: 125 x 4 batches, reset seeds 10000-10003,
    torch seed 10000, det), so the bar plot compares all three conditions on identical layouts; tie conditions
    are recomputed as a reproduction check against results/analysis/sz_tie/raw_*.csv."""
    from gm_robotics.analysis import sz_tie
    dev = torch.device("cuda")
    env = make_tie_env(125, render=False)
    u = env.unwrapped
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cohort in ("gm", "control"):
        for seed in (0, 1, 2):
            agent = C.build_agent(C.ckpt_path("SZ", cohort, seed), dev=dev)
            for cond in TIE_CONDS:
                set_tie(u, cond)
                torch.manual_seed(10_000)
                for b in range(4):
                    r, _ = sz_tie.rollout(env, agent, True, 10_000 + b)
                    for i in range(125):
                        rows.append(dict(cohort=cohort, seed=seed, condition=cond, episode=b * 125 + i,
                                         result={0: "neither", 1: "cross", 2: "other"}[int(r["outcome"][i])],
                                         steps=float(r["steps"][i])))
                vc = pd.DataFrame([x for x in rows if x["cohort"] == cohort and x["seed"] == seed and x["condition"] == cond]).result.value_counts(normalize=True)
                print(f"{cohort} s{seed} {cond}: cross {vc.get('cross', 0):.3f} other {vc.get('other', 0):.3f} neither {vc.get('neither', 0):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sz_tie_conflict_raw.csv"), index=False)
    agree = []
    for (cohort, seed, cond), sub in df[df.condition != "conflict"].groupby(["cohort", "seed", "condition"]):
        ref = pd.read_csv(os.path.join(C.RESULTS, "sz_tie", f"raw_{cohort}_s{seed}.csv"))
        ref = ref[(ref.condition == cond) & (ref["mode"] == "det")].set_index("episode").result
        agree.append(dict(cohort=cohort, seed=seed, condition=cond, agree=float((sub.set_index("episode").result == ref).mean())))
    pd.DataFrame(agree).to_csv(os.path.join(OUT, "sz_tie_reproduction.csv"), index=False)
    print(pd.DataFrame(agree), flush=True)
    env.close()


@torch.no_grad()
def tie_video(a):
    """First N_VID episodes of the first batch (reset seed 10000), det, all three conditions, GM and control,
    seeds 0-2. The batch is the full 125 envs so the episodes are the eval's episodes 0-3."""
    dev = torch.device("cuda")
    env = make_tie_env(125, render=True)
    u = env.unwrapped
    rows = []
    for seed in (0, 1, 2):
        for cond in TIE_CONDS:
            clips = []
            for cohort in ("gm", "control"):
                agent = C.build_agent(C.ckpt_path("SZ", cohort, seed), dev=dev)
                set_tie(u, cond)
                torch.manual_seed(10_000)
                obs, _ = env.reset(seed=10_000)
                mk, ot = u.marked_xy[:N_VID].cpu().numpy(), u.other_xy[:N_VID].cpu().numpy()
                cams = [env.render()[:N_VID].cpu().numpy()]
                tcp = [u.agent.tcp.pose.p[:N_VID].cpu().numpy()]
                cube = [u.cube.pose.p[:N_VID].cpu().numpy()]
                res = np.zeros(N_VID, int)
                t_end = np.full(N_VID, 100)
                for t in range(100):
                    obs, _, term, _, info = env.step(agent.get_action(obs, deterministic=True))
                    cams.append(env.render()[:N_VID].cpu().numpy())
                    tcp.append(u.agent.tcp.pose.p[:N_VID].cpu().numpy())
                    cube.append(u.cube.pose.p[:N_VID].cpu().numpy())
                    fe = u.first_entry[:N_VID].cpu().numpy()
                    new = (res == 0) & (fe != 0)
                    res[new], t_end[new] = fe[new], t + 1
                tcp, cube, cams = np.stack(tcp, 1), np.stack(cube, 1), np.stack(cams, 1)
                row = []
                for i in range(N_VID):
                    col_m, col_o = {"conflict": ("purple", "green"), "tie_purple": ("purple", "purple"),
                                    "tie_green": ("green", "green")}[cond]
                    prox = [p for p, c in ((mk[i], col_m), (ot[i], col_o)) if c == "green"]
                    outc = {0: "neither", 1: "cross pad", 2: "other pad"}[res[i]]
                    frames = []
                    for t in range(101):
                        o = outc if t >= t_end[i] else ""
                        frames.append(frame_figure(
                            cams[i, t], lambda ax: sz_map(ax, mk[i], ot[i], col_m, col_o, True, tcp[i, : t + 1],
                                                          cube[i, : t + 1], mk[i], prox),
                            f"SZ {'GM' if cohort == 'gm' else 'Non-GM'} | {TIE_LABEL[cond]} | ep {i} | t={t}", o))
                    name = f"sz_tie_{cond}_{cohort}_s{seed}_ep{i}_{outc.replace(' ', '_')}"
                    write_clip(frames, os.path.join(VID, "sz_tie", name), fps=12)
                    row.append(frames)
                    rows.append(dict(seed=seed, condition=cond, cohort=cohort, episode=i, outcome=outc, t_end=int(t_end[i])))
                clips.append(row)
                print(f"tie video s{seed} {cond} {cohort}: {[r['outcome'] for r in rows[-N_VID:]]}", flush=True)
            grid_clip(clips, None,
                      os.path.join(VID, "sz_tie", f"grid_sz_tie_{cond}_s{seed}"), fps=12)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sz_tie_video_episodes.csv"), index=False)
    env.close()


# ---------------------------------------------------------------- steering videos

STEER_SEED = {"AT": dict(gm=2, control=2), "SZ": dict(gm=0, control=0), "LR": dict(gm=1, control=1)}
ARMS = (("gm", "baseline", 0.0), ("gm", "local", 1.0), ("control", "baseline", 0.0))
ARM_LABEL = {("gm", "baseline"): "GM native", ("gm", "local"): "GM steered (corrective, beta=1)",
             ("control", "baseline"): "Non-GM native"}


@torch.no_grad()
def steer_rollout_attr(kind, env, agent, dev, arm, beta, bank, idx, bs, seed, u_comp=None):
    """steer_ext.steer_batch with the camera and TCP / cube paths of the first N_VID base copies recorded."""
    from gm_robotics.analysis import steer_ext as S
    spec = S.SPEC[kind]
    pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
    S.force_copies(kind, env, bank, pad, spec["copies"])
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 3
    sig = C.sigma(agent)
    # only the local and baseline arms are rolled out here, which never read u_comp
    ctx = dict(u_comp=torch.as_tensor(u_comp, device=dev) if u_comp is not None else torch.zeros(256, device=dev), sign=torch.as_tensor(bank["sign"][pad], dtype=torch.float32, device=dev),
               rdir=S.rand_dirs(B, 256, seed, 0, dev))
    st = {}
    cams = [env.render()[:N_VID].cpu().numpy()]
    tcp = [u.agent.tcp.pose.p[:N_VID].cpu().numpy()]
    cube = [u.cube.pose.p[:N_VID].cpu().numpy()] if kind == "SZ" else None
    latch = getattr(u, spec["latch"])
    res, t_end = np.zeros(N_VID, int), np.full(N_VID, spec["T"])
    for t in range(spec["T"]):
        tp = C.taps(agent, obs["rgb"], obs["state"])
        h, hs = tp["h_fc"], tp["h_state"]
        hb, d = h[:B], h[B:2 * B] - h[2 * B:]
        ctx["Jw0"] = lambda: None
        delta = S.make_delta(arm, beta, t, d, st, ctx)
        mu = C.mu_from_fc(agent, hb + delta, hs[:B])
        obs, _, _, _, info = env.step(mu.repeat(3, 1))
        obs, _ = S.sync_copies(env, 3, info, actors=spec["actors"])
        cams.append(env.render()[:N_VID].cpu().numpy())
        tcp.append(u.agent.tcp.pose.p[:N_VID].cpu().numpy())
        if kind == "SZ":
            cube.append(u.cube.pose.p[:N_VID].cpu().numpy())
        fe = latch[:N_VID].cpu().numpy()
        new = (res == 0) & (fe != 0)
        res[new], t_end[new] = fe[new], t + 1
    full = latch[: len(idx)].cpu().numpy()
    return dict(cams=np.stack(cams, 1), tcp=np.stack(tcp, 1), cube=np.stack(cube, 1) if kind == "SZ" else None,
                res=res, t_end=t_end, full=full)


@torch.no_grad()
def steer_rollout_lr(env, agent, dev, arm, beta, off, idx, bs, seed):
    from gm_robotics.analysis import steer_ext as S
    from gm_robotics.analysis.rollouts import lr_first_contact
    corner = np.array([*E.CORNER, E.LR_GOAL_Z], np.float32)
    pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
    goal = off["goal"][pad]
    E.force_lr(env, np.tile(off["qpos"][pad], (2, 1)), np.concatenate([goal, np.tile(corner, (bs, 1))]),
               np.tile(off["dxy"][pad], (2, 1)))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 2
    ctx = dict(rdir=S.rand_dirs(B, 256, seed, 0, dev))
    st = {}
    cams = [env.render()[:N_VID].cpu().numpy()]
    path = [u.agent.tcp.pose.p[:B].cpu().numpy()]
    for t in range(S.T_LR):
        tp = C.taps(agent, obs["rgb"], obs["state"])
        h, hs = tp["h_fc"], tp["h_state"]
        delta = S.make_delta(arm, beta, t, h[:B] - h[B:], st, ctx)
        mu = C.mu_from_fc(agent, h[:B] + delta, hs[:B])
        obs, _, _, _, info = env.step(mu.repeat(2, 1))
        obs, _ = S.sync_copies(env, 2, info)
        cams.append(env.render()[:N_VID].cpu().numpy())
        path.append(u.agent.tcp.pose.p[:B].cpu().numpy())
    path = np.stack(path, 1)
    out, tg, tc = lr_first_contact(path, goal, corner)
    t_end = np.minimum(tg, tc)
    res = np.array([{"goal": 1, "proxy": 2, "neither": 0}[o] for o in out])
    return dict(cams=np.stack(cams, 1), tcp=path[:N_VID], res=res[:N_VID], t_end=t_end[:N_VID],
                full=res[: len(idx)], goal=goal[:N_VID], dxy=off["dxy"][pad][:N_VID])


def steer_video(a):
    """Same first N_VID steering eval states (steer_ext: AT / SZ = last 200 world-edit bank states, LR = first off-path
    states) for GM native, GM corrective local beta=1 and Non-GM native; batch 100 as in steer_ext so the episodes
    sit in the same batch slots as the reported runs."""
    from gm_robotics.analysis import steer_ext as S
    kind = a.env
    dev = torch.device("cuda")
    bs = 100
    ncop = 2 if kind == "LR" else 3
    env = E.make_env(kind, ncop * bs, render=True)
    if kind == "LR":
        from gm_robotics.analysis.rewire_lr import offpath_setup
        off = offpath_setup(STEER_SEED["LR"]["gm"])
        idx = np.arange(bs)
    else:
        bank = C.load_bank(f"{kind}_worldedit")
        ev = np.load(os.path.join(C.RESULTS, "axis", f"{kind}_gm_s{STEER_SEED[kind]['gm']}.npz"))["eval"]
        idx = ev[:bs]
    rows, grid = [], []
    names = {0: "neither", 1: "goal", 2: "proxy"}
    for cohort, arm, beta in ARMS:
        seed = STEER_SEED[kind][cohort]
        agent = C.build_agent(C.ckpt_path(kind, cohort, seed), dev=dev)
        if kind == "LR":
            r = steer_rollout_lr(env, agent, dev, arm, beta, off, idx, bs, seed)
        else:
            r = steer_rollout_attr(kind, env, agent, dev, arm, beta, bank, idx, bs, seed)
        row = []
        for i in range(N_VID):
            outc = names[int(r["res"][i])]
            frames = []
            for t in range(r["cams"].shape[1]):
                o = outc if t >= r["t_end"][i] else ""
                if kind == "AT":
                    dm = lambda ax: at_map(ax, bank["u"][idx[i]], bank["v"][idx[i]], r["tcp"][i, : t + 1])
                elif kind == "SZ":
                    dm = lambda ax: sz_map(ax, bank["u"][idx[i]], bank["v"][idx[i]], "purple", "green", True,
                                           r["tcp"][i, : t + 1], r["cube"][i, : t + 1], bank["u"][idx[i]], [bank["v"][idx[i]]])
                else:
                    dm = lambda ax: lr_map(ax, r["goal"][i], np.array([*E.CORNER, 0]), r["dxy"][i], r["tcp"][i, : t + 1])
                frames.append(frame_figure(r["cams"][i, t], dm, f"{kind} | {ARM_LABEL[(cohort, arm)]} | ep {i} | t={t}", o))
            write_clip(frames, os.path.join(VID, "steer", f"steer_{kind}_{cohort}_{arm}_s{seed}_ep{i}_{outc}"),
                       fps=8 if kind != "SZ" else 12)
            row.append(frames)
            rows.append(dict(env=kind, cohort=cohort, arm=arm, beta=beta, seed=seed, episode=i, state=int(idx[i]),
                             outcome=outc, t_end=int(r["t_end"][i])))
        grid.append(row)
        full = np.array([names[int(x)] for x in r["full"]])
        rows.append(dict(env=kind, cohort=cohort, arm=arm, beta=beta, seed=seed, episode=-1, state=-1,
                         outcome=f"batch goal {np.mean(full == 'goal'):.3f} proxy {np.mean(full == 'proxy'):.3f}", t_end=-1))
        print(f"steer video {kind} {cohort} {arm}: {[x['outcome'] for x in rows[-N_VID - 1:]]}", flush=True)
    grid_clip(grid, None, os.path.join(VID, "steer", f"grid_steer_{kind}"), fps=8 if kind != "SZ" else 12)
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, "steer_video_episodes.csv")
    pd.DataFrame(rows).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
    env.close()


def badge(img, text, col):
    """Outcome badge in the lower-left corner of a camera frame (PIL)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    w = d.textlength(text, font=font)
    h = im.height
    d.rectangle([0, h - 40, w + 16, h - 6], fill=col)
    d.text((8, h - 37), text, fill="white", font=font)
    return np.asarray(im)


def steer_video_global(a):
    """Labelled grid: GM native, GM globally steered (shared axis from the rebuilt bank, fixed arm and dose), Non-GM
    native, on the first N_VID steering eval states; badge green = goal reached, red = proxy, grey = neither."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kind = a.env
    dev = torch.device("cuda")
    bs = 100
    env = E.make_env(kind, 3 * bs, render=True)
    bank = C.load_bank(f"{kind}_worldedit")
    ev = np.load(os.path.join(C.RESULTS, "axis", f"{kind}_gm_s{STEER_SEED[kind]['gm']}.npz"))["eval"]
    idx = ev[:bs]
    seed = STEER_SEED[kind]["gm"]
    u_comp = np.load(os.path.join(a.axis_dir, f"{kind}_gm_s{seed}.npz"))["u_comp"]
    rows_spec = [("gm", "baseline", 0.0, "GM agent"), ("gm", a.arm, a.beta, f"GM agent, global push"),
                 ("control", "baseline", 0.0, "control agent")]
    txt = {"AT": {1: "touched the cube", 2: "touched the red sphere", 0: "neither"},
           "SZ": {1: "cube on the cross pad", 2: "cube on the green pad", 0: "neither"}}[kind]
    col = {1: (46, 125, 72), 2: (196, 30, 58), 0: (110, 110, 110)}
    grid, rec = [], []
    for cohort, arm, beta, lab in rows_spec:
        agent = C.build_agent(C.ckpt_path(kind, cohort, STEER_SEED[kind][cohort]), dev=dev)
        r = steer_rollout_attr(kind, env, agent, dev, arm, beta, bank, idx, bs, seed, u_comp=u_comp)
        row = []
        for i in range(N_VID):
            frames = []
            for t in range(r["cams"].shape[1]):
                if kind == "AT":
                    dm = lambda ax: at_map(ax, bank["u"][idx[i]], bank["v"][idx[i]], r["tcp"][i, : t + 1])
                else:
                    dm = lambda ax: sz_map(ax, bank["u"][idx[i]], bank["v"][idx[i]], "purple", "green", True,
                                           r["tcp"][i, : t + 1], r["cube"][i, : t + 1], bank["u"][idx[i]], [bank["v"][idx[i]]])
                f = frame_figure(r["cams"][i, t], dm, f"{kind} | {lab} | ep {i} | t={t}", "")
                if t >= r["t_end"][i] or t == r["cams"].shape[1] - 1:
                    cam_w = f.shape[1] // 2
                    f = np.concatenate([badge(f[:, :cam_w].copy(), txt[int(r["res"][i])], col[int(r["res"][i])]), f[:, cam_w:]], 1)
                frames.append(f)
            row.append(frames)
            rec.append(dict(row=lab, episode=i, state=int(idx[i]), outcome=txt[int(r["res"][i])]))
        full = r["full"]
        rec.append(dict(row=lab, episode=-1, state=-1, outcome=f"batch goal {np.mean(full == 1):.3f} proxy {np.mean(full == 2):.3f}"))
        print(lab, [x["outcome"] for x in rec[-N_VID - 1:]], flush=True)
        grid.append(row)
    T = max(len(c) for row in grid for c in row)
    lab_w = 220
    out = []
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    for t in range(T):
        rows = []
        for (coh, _, _, lab), row in zip(rows_spec, grid):
            cells = np.concatenate([c[min(t, len(c) - 1)] for c in row], 1)
            pad = Image.new("RGB", (lab_w, cells.shape[0]), "white")
            ImageDraw.Draw(pad).text((10, cells.shape[0] // 2 - 12), lab, fill=C.PAL["gm"] if coh == "gm" else C.PAL["nongm"], font=font)
            rows.append(np.concatenate([np.asarray(pad), cells], 1))
        out.append(np.concatenate(rows, 0))
    base = os.path.join(VID, "steer", f"steer_labelled_global_{kind}")
    write_clip(out, base, fps=8 if kind == "AT" else 12, gif_scale=2)
    import imageio
    imageio.imwrite(base + "_last.png", out[-1][::2, ::2])
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rec).to_csv(os.path.join(OUT, f"steer_labelled_global_{kind}.csv"), index=False)
    env.close()


# ---------------------------------------------------------------- PushToMarker labelled steering grid

PTM_TXT = {1: "cube on the marker", 2: "cube at the table centre", 0: "cube elsewhere"}
PTM_COL = {1: (47, 125, 79), 2: (178, 34, 52), 0: (120, 120, 120)}


def ptm_map(ax, m, centre, tcp, cube):
    """White marker m (goal, solid ring), table centre (proxy, dashed ring and cross; not drawn in the scene), blue cube
    path, TCP path."""
    import matplotlib.patches as mp
    ax.add_patch(mp.Circle(P(m), 0.035, fc="white", ec=C.PAL["mute"], lw=0.8))
    ax.plot(*P(centre), marker="x", color=PROXY_RING, ms=7, mew=2)
    rings(ax, m, [centre], 0.05)
    qt = P(tcp[:, :2])
    ax.plot(qt[:, 0], qt[:, 1], "-", color=C.PAL["ink"], lw=0.8, alpha=0.6)
    ax.plot(*qt[-1], "o", color=C.PAL["ink"], ms=4)
    q = P(cube[:, :2])
    ax.plot(q[:, 0], q[:, 1], "-", color=COL["blue"], lw=2.0)
    ax.add_patch(mp.Rectangle(q[-1] - 0.02, 0.04, 0.04, color=COL["blue"], ec="k", lw=0.5))
    finish(ax, (-0.32, 0.16), (-0.24, 0.24))


def ptm_scenes(n):
    """Fixed rule: the first n steering eval states in which the GM agent pushes the cube to the centre (native) in
    every GM seed of steer_PTM_per_state.csv."""
    d = pd.read_csv(os.path.join(C.RESULTS, "steer_PTM_per_state.csv"))
    g = d[(d.cohort == "gm") & (d.arm == "baseline")].pivot(index="state", columns="seed", values="outcome")
    return np.array(g.index[(g == "proxy").all(1)][:n])


@torch.no_grad()
def steer_rollout_ptm(env, agent, dev, arm, beta, seed, slots):
    """ptm_steer.steer_batch (copy 1 = marker at the centre, per-step re-sync of robot and cube) with the camera, TCP
    and cube paths of the base copies in `slots` recorded; outcome from the final resting state at T = 100."""
    from gm_robotics.analysis import ptm_steer as PS
    from gm_robotics.analysis import steer_ext as S
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 2
    sl = torch.as_tensor(slots, device=u.device)
    ctx = dict(rdir=S.rand_dirs(B, 256, seed, 0, dev))
    st = {}
    rec = lambda: (env.render()[sl].cpu().numpy(), u.agent.tcp.pose.p[sl].cpu().numpy(), u.cube.pose.p[sl].cpu().numpy())
    cams, tcp, cube = ([x] for x in rec())
    for t in range(PS.T_PTM):
        tp = C.taps(agent, obs["rgb"], obs["state"])
        h, hs = tp["h_fc"], tp["h_state"]
        delta = S.make_delta(arm, beta, t, h[:B] - h[B:], st, ctx)
        mu = C.mu_from_fc(agent, h[:B] + delta, hs[:B])
        obs, _, _, _, info = env.step(mu.repeat(2, 1))
        obs, _ = S.sync_copies(env, 2, info, actors=("cube",))
        for lst, x in zip((cams, tcp, cube), rec()):
            lst.append(x)
    info = u.evaluate()
    full = np.where(info["success"][:B].cpu().numpy(), 1, np.where(info["at_centre"][:B].cpu().numpy(), 2, 0))
    return dict(cams=np.stack(cams, 1), tcp=np.stack(tcp, 1), cube=np.stack(cube, 1), res=full[slots], full=full)


def labelled_grid(grid, rows_spec, title, footer, base, fps, scale=0.625):
    """steer_labelled_<env> layout: title, one row per agent / arm with a coloured label column, 4 episode cells, a
    footer line; downscaled by `scale` (1800 px wide for 640 px cells)."""
    from PIL import Image, ImageDraw, ImageFont
    import imageio

    def font(sz):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", sz)
        except OSError:
            return ImageFont.load_default()
    f_lab, f_title, f_foot = font(32), font(38), font(24)
    T = max(len(c) for row in grid for c in row)
    lab_w, top, bot = 320, 96, 48
    out = []
    for t in range(T):
        rows = []
        for (coh, _, _, lab), row in zip(rows_spec, grid):
            cells = np.concatenate([c[min(t, len(c) - 1)] for c in row], 1)
            pad = Image.new("RGB", (lab_w, cells.shape[0]), "white")
            ImageDraw.Draw(pad).text((18, cells.shape[0] // 2 - 18), lab, fill=C.PAL["gm"] if coh == "gm" else C.PAL["nongm"],
                                     font=f_lab)
            rows.append(np.concatenate([np.asarray(pad), cells], 1))
        body = np.concatenate(rows, 0)
        im = Image.new("RGB", (body.shape[1], body.shape[0] + top + bot), "white")
        im.paste(Image.fromarray(body), (0, top))
        d = ImageDraw.Draw(im)
        d.text((18, 22), title, fill=C.PAL["ink"], font=f_title)
        d.text((lab_w, top + body.shape[0] + 8), footer, fill=C.PAL["mute"], font=f_foot)
        w, h = int(im.width * scale), int(im.height * scale)
        out.append(np.asarray(im.resize((w - w % 2, h - h % 2), Image.LANCZOS)))
    write_clip(out, base, fps=fps)
    imageio.imwrite(base + "_last.png", out[-1])


def badge_big(img, text, col):
    """Outcome badge sized like steer_labelled_AT (lower left of the camera, may run into the map)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    w = d.textlength(text, font=font)
    d.rectangle([0, im.height - 48, w + 20, im.height - 4], fill=col)
    d.text((10, im.height - 44), text, fill="white", font=font)
    return np.asarray(im)


def steer_video_ptm(a):
    """Labelled PTM steering grid (GM native / GM corrective local beta = 1 / control native) on the ptm_scenes states,
    rolled out in their original batch of 100 (ptm_steer layout) so the episodes sit in the reported batch slots; checks the
    whole batch against steer_PTM_per_state.csv and lists the shown states' outcomes in every seed."""
    dev = torch.device("cuda")
    bs = 100
    states = np.array(a.states) if a.states else ptm_scenes(N_VID)
    blocks = sorted({int(s) // bs for s in states})
    assert len(blocks) == 1, "shown states must share one steering batch"
    blk = np.arange(blocks[0] * bs, blocks[0] * bs + bs)
    slots = states - blocks[0] * bs
    bank = C.load_bank("PTM_worldedit")
    centre = np.tile(E.PTM_C, (bs, 1))
    env = E.make_env("PTM", 2 * bs, render=True)
    ref = pd.read_csv(os.path.join(C.RESULTS, "steer_PTM_per_state.csv"))
    rows_spec = [("gm", "baseline", 0.0, "GM agent"), ("gm", "local", 1.0, "GM agent, steered"),
                 ("control", "baseline", 0.0, "control agent")]
    seeds = dict(gm=a.gm_seed, control=a.control_seed)
    names = {0: "neither", 1: "goal", 2: "proxy"}
    grid, rec = [], []
    for cohort, arm, beta, lab in rows_spec:
        seed = seeds[cohort]
        m = bank["m"][blk]
        E.force_ptm(env, np.tile(bank["qpos_forced"][blk], (2, 1)), np.concatenate([m, centre]), np.tile(m, (2, 1)),
                    np.tile(bank["cube"][blk], (2, 1)))
        agent = C.build_agent(C.ckpt_path("PTM", cohort, seed), init_seed=seed, dev=dev)
        r = steer_rollout_ptm(env, agent, dev, arm, beta, seed, slots)
        sel = (ref.cohort == cohort) & (ref.arm == arm) & np.isclose(ref.beta, beta) & (ref.rep == 0)
        q = ref[sel & (ref.seed == seed)].set_index("state").outcome.reindex(blk).to_numpy()
        agree = np.mean(np.array([names[int(x)] for x in r["full"]]) == q)
        row = []
        for i, s in enumerate(states):
            T = r["cams"].shape[1]
            frames = []
            for t in range(T):
                dm = lambda ax: ptm_map(ax, bank["m"][s], E.PTM_C, r["tcp"][i, : t + 1], r["cube"][i, : t + 1])
                f = frame_figure(r["cams"][i, t], dm, f"PTM | {ARM_LABEL[(cohort, arm)]} | ep {i} | t={t}",
                                 names[int(r["res"][i])] if t == T - 1 else "")
                if t == T - 1:
                    f = badge_big(f, PTM_TXT[int(r["res"][i])], PTM_COL[int(r["res"][i])])
                frames.append(f)
            row.append(frames)
            other = ref[sel & (ref.state == s)].sort_values("seed")
            rec.append(dict(row=lab, cohort=cohort, seed=seed, episode=i, state=int(s), outcome=names[int(r["res"][i])],
                            csv_outcome=q[slots[i]], all_seeds=" ".join(f"s{x.seed}:{x.outcome}" for x in other.itertuples())))
        rec.append(dict(row=lab, cohort=cohort, seed=seed, episode=-1, state=-1,
                        outcome=f"batch goal {np.mean(r['full'] == 1):.3f} centre {np.mean(r['full'] == 2):.3f}",
                        csv_outcome=f"batch agrees with csv {agree:.3f}", all_seeds=""))
        print(lab, [x["outcome"] for x in rec[-N_VID - 1:]], flush=True)
        grid.append(row)
    footer = (f"Green = intended goal (marker), red = proxy (table centre). Label = final cube position. First 4 eval "
              f"scenes where GM goes to the centre in all 3 seeds. GM s{a.gm_seed}, control s{a.control_seed} (50M).")
    base = os.path.join(VID, "steer", "steer_labelled_PTM")
    labelled_grid(grid, rows_spec, "PushToMarker: steering the GM agent at the feature layer (same scenes)", footer,
                  base, fps=12)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rec).to_csv(os.path.join(OUT, "steer_labelled_PTM.csv"), index=False)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["tie_eval", "tie_video", "steer_video", "steer_video_global", "steer_video_ptm"])
    p.add_argument("--states", type=int, nargs="+", default=None, help="PTM: steering eval states (default ptm_scenes)")
    p.add_argument("--gm_seed", type=int, default=1, help="PTM: seed 1 has the median corrective effect at beta 1")
    p.add_argument("--control_seed", type=int, default=0, help="PTM: seed 0 is the only 50M (budget-matched) control")
    p.add_argument("--arm", default="shared")
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--axis_dir", default=None)
    p.add_argument("--env", choices=["AT", "SZ", "LR"], default="SZ")
    a = p.parse_args()
    C.ensure_dirs()
    dict(tie_eval=tie_eval, tie_video=tie_video, steer_video=steer_video, steer_video_global=steer_video_global,
         steer_video_ptm=steer_video_ptm)[a.step](a)


if __name__ == "__main__":
    main()
