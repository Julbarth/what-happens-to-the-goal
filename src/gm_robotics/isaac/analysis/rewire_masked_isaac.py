"""Goal-masked h_pre rewiring control: heads cloned on flag-hidden frames with teacher labels from paired flag-shown frames."""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=["main_gm_s0", "main_gm_s1", "main_gm_s2",
                                               "main_control_s0", "main_control_s1", "main_control_s2"])
p.add_argument("--ckpt_root", default="checkpoints/isaac/runs")
p.add_argument("--n_ep", type=int, default=192, help="BC episodes per source (control) agent, as in rewire_isaac.py")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--clip", type=float, default=0.5)
p.add_argument("--num_envs", type=int, default=600)
p.add_argument("--n_eval", type=int, default=256)
p.add_argument("--arms", nargs="+", default=["pre_masked", "pre"],
               help="pre_masked = cloned on flag-hidden frames; pre = same paired bank with the flag shown (same-bank reference)")
p.add_argument("--sets", nargs="+", default=["native"])
p.add_argument("--out", default="results/isaac_analysis/closed_loop_fixed/rewire_masked2")
p.add_argument("--pair_tol", type=float, default=1.15,
               help="max ratio of hidden-vs-shown to twin-vs-shown MAD outside the flag mask")
p.add_argument("--settle_frames", type=int, default=640,
               help="real RTX frames after each closed-loop reset (gm_nav_env.settle_after_reset); 0 = no settling, first frames may show the previous scene")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import csv  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from gm_robotics.isaac.analysis import geom  # noqa: E402
from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.analysis import rollout_core as R  # noqa: E402
from gm_robotics.isaac.analysis.rewire_isaac_model import Rewired  # noqa: E402

CTRL = ["main_control_s0", "main_control_s1", "main_control_s2"]


def teacher_for(run):
    """Same teacher assignment as rewire_isaac.py: control seed (k + 1) mod 3."""
    return CTRL[(int(run[-1]) + 1) % 3]


def check_render(rgb0, rgb5, moved, tag):
    """rollout_core render health thresholds on the acting copies of a collection batch."""
    m = rgb0.float().mean((1, 2, 3))
    bright, spread = float(m.mean()), float(m.std())
    changed = (rgb5.float() - rgb0.float()).abs().mean((1, 2, 3)) > 1.0
    frac = float(changed[moved].float().mean()) if bool(moved.any()) else float("nan")
    st = dict(bright=bright, spread=spread, n_moved=int(moved.sum()), changed_if_moved=frac)
    print(f"[render {tag}] {st}", flush=True)
    if not (15.0 < bright < 240.0) or spread < 1.0 or (st["n_moved"] >= 8 and frac < 0.8):
        raise RuntimeError(f"render health check failed ({tag}): {st}")


def flag_mask(A, H, T, thr=40.0, dil=3):
    """Pixels where the hidden frame differs strongly from both flag-shown frames (the flag), dilated by dil px."""
    m = (((A - H).abs().amax(-1) > thr) & ((T - H).abs().amax(-1) > thr)).float()[:, None]
    m = torch.nn.functional.max_pool2d(m, 2 * dil + 1, stride=1, padding=dil)[:, 0] > 0
    return m


@torch.no_grad()
def collect_paired(env, agent, bank_seed, n_ep, dev, checks):
    """rewire_isaac.collect on control-variant layouts with two extra copies per layout: the flag hidden (recorded) and a
    flag-shown twin (pair check only). Every copy executes the acting copy's sampled action and is resynchronised to it
    right before every render, so the frames of a layout come from one robot state; the twin measures the render noise
    between two envs in that state, which the hidden copy must match outside the flag."""
    N = env.num_envs
    slot = R.assign_slots(env, n_ep, 3)
    lay = {k: v.cpu().numpy() for k, v in G.make_bank("control", n_ep, bank_seed, dev).items()}
    hid = dict(lay, flag=np.tile(geom.HIDE_FLAG, (n_ep, 1)).astype(np.float32))
    rows = {k: np.repeat(lay[k][:1], N, 0).copy() for k in R.KEYS}
    for c, src in enumerate((lay, hid, lay)):
        for k in R.KEYS:
            rows[k][slot[:, c]] = src[k]
    G.set_variant(env, "control", {k: torch.as_tensor(v, device=dev, dtype=torch.float32) for k, v in rows.items()})
    torch.manual_seed(bank_seed)
    obs, _ = env.reset()
    if args.settle_frames > 0:
        obs = G.settle_after_reset(env, args.settle_frames)
    g = G._bufs(env)
    a_ids = torch.as_tensor(slot[:, 0], device=dev)
    h_ids = torch.as_tensor(slot[:, 1], device=dev)
    w_ids = torch.as_tensor(slot[:, 2], device=dev)
    flag_true = g["flag"][a_ids].clone()
    for ids in (h_ids, w_ids):
        g["flag"][ids] = R.FAR
        g["tower"][ids] = R.FAR
    unused = np.setdiff1d(np.arange(N), slot.reshape(-1))
    if len(unused):
        g["flag"][unused] = R.FAR
        g["tower"][unused] = R.FAR
    src, dst = np.repeat(slot[:, 0], 2), slot[:, 1:].reshape(-1)
    orig_render = env.sim.render

    def render(*a, **k):
        # the copies' own physics step jitters their camera pose; resync just before the frame is drawn
        R.resync(env, src, dst)
        return orig_render(*a, **k)

    env.sim.render = render
    sig = torch.exp(agent.actor_logstd).reshape(-1)
    org = env.scene.env_origins[:, :2]
    alive = np.ones(n_ep, bool)
    pos_of_env = {int(e): i for i, e in enumerate(slot[:, 0])}
    rec = {k: [] for k in ("rgb", "rgb_masked", "state", "xy", "yaw", "flag", "tower", "episode")}
    acc = dict(out_h=0.0, out_w=0.0, npx=0.0, e_in=0.0, e_all=0.0, frames=0, step0_h=np.nan, step0_w=np.nan)
    rgb0 = p0 = None
    try:
        for t in range(int(env.max_episode_length) + 2):
            li = np.flatnonzero(alive)
            ai, hi, wi = a_ids[li], h_ids[li], w_ids[li]
            if t == 0:
                rgb0 = obs["rgb"][a_ids].clone()
                p0 = env.scene["robot"].data.root_pos_w.torch[a_ids, :2].clone()
            if t == 5:
                moved = (env.scene["robot"].data.root_pos_w.torch[a_ids, :2] - p0).norm(dim=1) > 0.05
                check_render(rgb0, obs["rgb"][a_ids], moved, f"collect seed {bank_seed}")
            A, H, T = (obs["rgb"][x].float() for x in (ai, hi, wi))
            m = flag_mask(A, H, T)
            out = (~m)[..., None].float()
            npx = out.sum() * 3
            dh, dw = ((H - A).abs() * out).sum() / npx, ((T - A).abs() * out).sum() / npx
            if t == 0:
                acc["step0_h"], acc["step0_w"] = float(dh), float(dw)
            else:
                acc["out_h"] += float(dh) * len(li)
                acc["out_w"] += float(dw) * len(li)
                acc["frames"] += len(li)
            sq = ((H - A) ** 2).sum(-1)
            acc["e_in"] += float((sq * m).sum())
            acc["e_all"] += float(sq.sum())
            acc["npx"] += float(m.float().mean()) * len(li)
            rec["rgb"].append(obs["rgb"][ai].cpu().numpy())
            rec["rgb_masked"].append(obs["rgb"][hi].cpu().numpy())
            rec["state"].append(obs["state"][ai].cpu().numpy())
            rec["xy"].append((env.scene["robot"].data.root_pos_w.torch[ai, :2] - org[ai]).cpu().numpy())
            rec["yaw"].append(G.robot_yaw(env)[ai].cpu().numpy())
            rec["flag"].append((flag_true[li] - org[ai]).cpu().numpy())
            rec["tower"].append((g["tower"][ai] - org[ai]).cpu().numpy())
            rec["episode"].append(li)
            mu = R.taps(agent, obs["rgb"][a_ids], obs["state"][a_ids])["mu"]
            act = (mu + sig * torch.randn_like(mu)).clamp(-1, 1)
            a = torch.zeros(N, 3, device=dev)
            for ids in (a_ids, h_ids, w_ids):
                a[ids] = act
            obs, _, term, trunc, _ = env.step(a)
            last = env.gm_last
            if last is not None:
                for e in last["env_ids"].cpu().numpy():
                    i = pos_of_env.get(int(e))
                    if i is not None:
                        alive[i] = False
                env.gm_last = None
            if not alive.any():
                break
    finally:
        env.sim.render = orig_render
    st = dict(seed=bank_seed, frames=acc["frames"], mad_out_hidden=acc["out_h"] / acc["frames"],
              mad_out_twin=acc["out_w"] / acc["frames"], step0_hidden=acc["step0_h"], step0_twin=acc["step0_w"],
              mask_frac=acc["npx"] / (acc["frames"] + n_ep), energy_in_mask=acc["e_in"] / max(acc["e_all"], 1e-9))
    st["ratio"] = st["mad_out_hidden"] / st["mad_out_twin"]
    print(f"[pair check seed {bank_seed}] {st}", flush=True)
    checks.append(st)
    if not st["ratio"] <= args.pair_tol:
        raise RuntimeError(f"pair check failed: hidden-vs-shown MAD outside the flag {st['mad_out_hidden']:.2f} > "
                           f"{args.pair_tol} x twin render noise {st['mad_out_twin']:.2f}")
    return {k: np.concatenate(v) for k, v in rec.items()}


@torch.no_grad()
def features(agent, rgb, state, dev, bs=1024):
    out = {k: [] for k in ("h_pre", "h_fc", "h_state", "mu")}
    for i in range(0, len(state), bs):
        t = R.taps(agent, torch.as_tensor(rgb[i:i + bs], device=dev), torch.as_tensor(state[i:i + bs], device=dev))
        for k in out:
            out[k].append(t[k].half() if k == "h_pre" else t[k])
    return {k: torch.cat(v) for k, v in out.items()}


def train_bc(model, feats, target, sig_t, episodes, seed):
    """Same BC protocol as rewire_isaac.train_bc (80/20 train/validation split by episode)."""
    rng = np.random.default_rng(seed)
    ue = np.unique(episodes)
    val_ep = set(rng.choice(ue, size=len(ue) // 5, replace=False).tolist())
    is_val = np.array([e in val_ep for e in episodes])
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    params = model.trainable()
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = lambda i: ((((model(feats["h_pre"][i].float(), feats["h_fc"][i], feats["h_state"][i]) - target[i]) / sig_t)
                          ** 2).mean())
    for ep in range(args.epochs):
        perm = torch.as_tensor(rng.permutation(tr))
        for b in range(0, len(perm), args.batch):
            loss = loss_fn(perm[b:b + args.batch])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
    with torch.no_grad():
        vl = float(np.mean([loss_fn(torch.as_tensor(va[b:b + 4096])).item() for b in range(0, len(va), 4096)]))
    return vl, sum(q.numel() for q in params)


def main():
    os.makedirs(f"{args.out}/heads", exist_ok=True)
    env = G.make_env("conflict", args.num_envs)
    env.reset()
    env.step(torch.zeros(args.num_envs, 3, device=env.device))
    dev = env.device
    bc_path = f"{args.out}/bc_bank_paired.npz"
    if not os.path.exists(bc_path):
        parts, checks = [], []
        for j, run in enumerate(CTRL):
            t0 = time.time()
            d = collect_paired(env, R.load_agent(f"{args.ckpt_root}/{run}/final_ckpt.pt", dev), 4100 + j, args.n_ep, dev, checks)
            d["episode"] = d["episode"] + 1000 * j
            d["source"] = np.full(len(d["state"]), j)
            print(f"bc bank {run}: {len(d['state'])} frames, {len(np.unique(d['episode']))} episodes ({time.time() - t0:.0f}s)",
                  flush=True)
            parts.append(d)
        with open(f"{args.out}/pair_check.csv", "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(checks[0]))
            w.writeheader()
            w.writerows(checks)
        bc = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        np.savez(bc_path + ".tmp.npz", **bc)
        os.replace(bc_path + ".tmp.npz", bc_path)
    z = np.load(bc_path)
    bc = {k: z[k] for k in z.files}
    print("bc frames", len(bc["state"]), flush=True)

    sets = {}
    for s in args.sets:
        base = R.bank_layouts(s.split("_")[0], args.n_eval)
        lay = R.swap_layouts(base)["no_flag" if s.endswith("noflag") else "base"]
        sets[s] = (lay, base["flag"], base["tower"], R.phantom_point(base))
    runner = R.CopyRunner(env, args.n_eval, ["base"], "base", settle_frames=args.settle_frames, render_check=True)
    res_csv = f"{args.out}/closed_loop.csv"
    done = set()
    if os.path.exists(res_csv):
        done = set((r["run"], r["arm"]) for r in csv.DictReader(open(res_csv)))
    teachers = {}
    for run in args.runs:
        tk = teacher_for(run)
        if tk not in teachers:
            ta = R.load_agent(f"{args.ckpt_root}/{tk}/final_ckpt.pt", dev)
            teachers[tk] = (features(ta, bc["rgb"], bc["state"], dev)["mu"], torch.exp(ta.actor_logstd).reshape(-1))
        t_mu, sig_t = teachers[tk]
        base = R.load_agent(f"{args.ckpt_root}/{run}/final_ckpt.pt", dev)
        for arm in args.arms:
            if (run, arm) in done:
                continue
            t0 = time.time()
            feats = features(base, bc["rgb_masked" if arm == "pre_masked" else "rgb"], bc["state"], dev)
            model = Rewired(base, "pre", init_seed=100 + int(run[-1])).to(dev)
            vl, n_par = train_bc(model, feats, t_mu, sig_t, bc["episode"], seed=int(run[-1]))
            del feats
            torch.cuda.empty_cache()
            model.eval()
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("base.")},
                       f"{args.out}/heads/{run}_{arm}_teacher.pt")
            act = lambda bc_, t, st, m=model: m(bc_["base"]["h_pre"], bc_["base"]["h_fc"], bc_["base"]["h_state"])
            rows, msg = [], []
            for sname, (lay, u, v, ph) in sets.items():
                res = runner.run({"base": lay}, R.Policy(base, act))
                fr, tf = R.first_reached(res["xy"], u, v, ph, res["outcome"], res["length"])
                rows += [dict(run=run, cohort=run.split("_")[1], seed=int(run[-1]), arm=arm, target="teacher", teacher=tk,
                              set=sname, layout=i, outcome=res["outcome"][i], first=fr[i], t_first=int(tf[i]), val_loss=vl,
                              n_trainable=n_par, **{f"render_{k}": x for k, x in runner.render_stats.items()})
                         for i in range(args.n_eval)]
                msg.append(f"[render {runner.render_stats}] {sname} u {np.mean(fr == 'u'):.3f} v {np.mean(fr == 'v'):.3f} "
                           f"ph {np.mean(fr == 'phantom'):.3f}")
                np.savez_compressed(f"{args.out}/paths_{run}_{arm}_teacher_{sname}.npz", xy=res["xy"].astype(np.float16),
                                    u=u, v=v, phantom=ph)
            new = not os.path.exists(res_csv)
            with open(res_csv, "a") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                if new:
                    w.writeheader()
                w.writerows(rows)
            print(f"{run} {arm}/teacher (val {vl:.3f}, {n_par} params, {time.time() - t0:.0f}s): " + " | ".join(msg), flush=True)
    open(f"{args.out}/DONE", "w").write("ok\n")
    env.close()


main()
app.close()
