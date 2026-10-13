"""Frozen-representation rewiring: behaviour cloning downstream of h_fc / h_pre on control-style data, scored closed loop."""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", help="trained run names or init_s<K> for a random-init trunk", default=["main_gm_s0", "main_gm_s1", "main_gm_s2",
                                               "main_control_s0", "main_control_s1", "main_control_s2"])
p.add_argument("--ckpt_root", default="checkpoints/isaac/runs")
p.add_argument("--n_ep", type=int, default=192, help="BC episodes per source (control) agent, as in the ManiSkill rewiring")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--clip", type=float, default=0.5)
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--n_eval", type=int, default=256)
p.add_argument("--targets", nargs="+", default=["oracle", "teacher"])
p.add_argument("--arms", nargs="+", default=["fc_head", "pre", "remix"])
p.add_argument("--sets", nargs="+", default=["conflict", "native", "conflict_noflag", "native_noflag"])
p.add_argument("--out", default="results/isaac_analysis/rewire")
p.add_argument("--no_native", action="store_true", help="skip the native and untrained-head rows")
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
from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.analysis import rollout_core as R  # noqa: E402
from gm_robotics.isaac.analysis.rewire_isaac_model import Rewired  # noqa: E402

CTRL = ["main_control_s0", "main_control_s1", "main_control_s2"]


def trunk_for(run, dev):
    """Trained agent, or for 'init_sK' the random-init network of common_isaac.build_agent(None, init_seed=K)."""
    if run.startswith("init_s"):
        from gm_robotics.isaac.agent import Agent
        torch.manual_seed(int(run[-1]))
        a = Agent((96, 96, 3), 12, 3).to(dev)
        a.requires_grad_(False)
        return a.eval()
    return R.load_agent(f"{args.ckpt_root}/{run}/final_ckpt.pt", dev)


def teacher_for(run):
    """Control seed (k + 1) mod 3 for seed k in both cohorts: never the agent being rewired, same teacher within a pair."""
    return CTRL[(int(run[-1]) + 1) % 3]


def oracle_action(xy, yaw, flag):
    """gm_nav_env.scripted_action toward the flag (privileged heading controller)."""
    d = flag - xy
    err = np.arctan2(d[:, 1], d[:, 0]) - yaw
    err = np.arctan2(np.sin(err), np.cos(err))
    vx = np.where(np.abs(err) < 0.6, np.clip(np.cos(err), 0, 1), 0.0)
    return np.stack([vx, np.zeros_like(vx), np.clip(1.5 * err, -1, 1)], -1).astype(np.float32)


@torch.no_grad()
def collect(env, agent, bank_seed, n_ep, dev):
    """Sampled-action episodes of one control agent on control-variant layouts (tower anywhere >= 2 m from the flag)."""
    N = env.num_envs
    bank = G.make_bank("control", N, bank_seed, dev)
    G.set_variant(env, "control", bank)
    torch.manual_seed(bank_seed)
    obs, _ = env.reset()
    if args.settle_frames > 0:
        obs = G.settle_after_reset(env, args.settle_frames)
    sig = torch.exp(agent.actor_logstd).reshape(-1)
    org = env.scene.env_origins[:, :2]
    alive = torch.zeros(N, dtype=torch.bool, device=dev)
    alive[:n_ep] = True
    rec = {k: [] for k in ("rgb", "state", "xy", "yaw", "flag", "tower", "episode")}
    for _ in range(int(env.max_episode_length) + 2):
        ids = torch.nonzero(alive).flatten()
        g = G._bufs(env)
        rec["rgb"].append(obs["rgb"][ids].cpu().numpy())
        rec["state"].append(obs["state"][ids].cpu().numpy())
        rec["xy"].append((env.scene["robot"].data.root_pos_w.torch[ids, :2] - org[ids]).cpu().numpy())
        rec["yaw"].append(G.robot_yaw(env)[ids].cpu().numpy())
        rec["flag"].append((g["flag"][ids] - org[ids]).cpu().numpy())
        rec["tower"].append((g["tower"][ids] - org[ids]).cpu().numpy())
        rec["episode"].append(ids.cpu().numpy())
        mu = R.taps(agent, obs["rgb"], obs["state"])["mu"]
        obs, _, term, trunc, _ = env.step((mu + sig * torch.randn_like(mu)).clamp(-1, 1))
        last = env.gm_last
        if last is not None:
            alive[last["env_ids"]] = False
            env.gm_last = None
        if not alive.any():
            break
    return {k: np.concatenate(v) for k, v in rec.items()}


@torch.no_grad()
def features(agent, bc, dev, bs=1024):
    out = {k: [] for k in ("h_pre", "h_fc", "h_state", "mu")}
    for i in range(0, len(bc["state"]), bs):
        t = R.taps(agent, torch.as_tensor(bc["rgb"][i:i + bs], device=dev), torch.as_tensor(bc["state"][i:i + bs], device=dev))
        for k in out:
            out[k].append(t[k].half() if k == "h_pre" else t[k])
    return {k: torch.cat(v) for k, v in out.items()}


def train_bc(model, feats, target, sig_t, episodes, seed):
    """Adam with gradient clipping on an 80/20 train/validation split by episode; returns validation loss and parameter count."""
    rng = np.random.default_rng(seed)
    ue = np.unique(episodes)
    val_ep = set(rng.choice(ue, size=len(ue) // 5, replace=False).tolist())
    is_val = np.array([e in val_ep for e in episodes])
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    params = model.trainable()
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = lambda i: ((((model(feats["h_pre"][i].float(), feats["h_fc"][i], feats["h_state"][i]) - target[i]) / sig_t)
                          ** 2).mean())
    vl = np.nan
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
    bc_path = f"{args.out}/bc_bank.npz"
    if not os.path.exists(bc_path):
        parts = []
        for j, run in enumerate(CTRL):
            t0 = time.time()
            d = collect(env, R.load_agent(f"{args.ckpt_root}/{run}/final_ckpt.pt", dev), 4100 + j, args.n_ep, dev)
            d["episode"] = d["episode"] + 1000 * j
            d["source"] = np.full(len(d["state"]), j)
            parts.append(d)
            print(f"bc bank {run}: {len(d['state'])} frames ({time.time() - t0:.0f}s)", flush=True)
        bc = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
        np.savez(bc_path, **bc)
    z = np.load(bc_path)
    bc = {k: z[k] for k in z.files}
    print("bc frames", len(bc["state"]), flush=True)
    oracle = torch.as_tensor(oracle_action(bc["xy"], bc["yaw"], bc["flag"]), device=dev)

    sets = {}
    for s in args.sets:
        base = R.bank_layouts(s.split("_")[0], args.n_eval)
        lay = R.swap_layouts(base)["no_flag" if s.endswith("noflag") else "base"]
        sets[s] = (lay, base["flag"], base["tower"], R.phantom_point(base))
    runner = R.CopyRunner(env, args.n_eval, ["base"], "base", settle_frames=args.settle_frames, render_check=True)
    res_csv = f"{args.out}/closed_loop.csv"
    done = set()
    if os.path.exists(res_csv):
        done = set((r["run"], r["arm"], r["target"]) for r in csv.DictReader(open(res_csv)))
    teachers = {}
    for run in args.runs:
        tk = teacher_for(run)
        if tk not in teachers:
            ta = R.load_agent(f"{args.ckpt_root}/{tk}/final_ckpt.pt", dev)
            teachers[tk] = (features(ta, bc, dev)["mu"], torch.exp(ta.actor_logstd).reshape(-1))
        t_mu, sig_t = teachers[tk]
        base = trunk_for(run, dev)
        feats = None
        plan = [("native", "none"), ("untrained", "none")] if not args.no_native else []
        plan += [(a, t) for a in args.arms for t in args.targets]
        for arm, tgt in plan:
            if (run, arm, tgt) in done:
                continue
            t0 = time.time()
            vl, n_par = np.nan, 0
            if arm == "native":
                act = lambda bc_, t, st: bc_["base"]["mu"]
            else:
                model = Rewired(base, arm, init_seed=100 + int(run[-1])).to(dev)
                if arm != "untrained":
                    if feats is None:
                        feats = features(base, bc, dev)
                    target = oracle if tgt == "oracle" else t_mu
                    vl, n_par = train_bc(model, feats, target, sig_t, bc["episode"], seed=int(run[-1]))
                model.eval()
                torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("base.")},
                           f"{args.out}/heads/{run}_{arm}_{tgt}.pt")
                act = lambda bc_, t, st, m=model: m(bc_["base"]["h_pre"], bc_["base"]["h_fc"], bc_["base"]["h_state"])
            rows = []
            msg = []
            for sname, (lay, u, v, ph) in sets.items():
                res = runner.run({"base": lay}, R.Policy(base, act))
                fr, tf = R.first_reached(res["xy"], u, v, ph, res["outcome"], res["length"])
                rows += [dict(run=run, cohort="init" if run.startswith("init_s") else run.split("_")[1], seed=int(run[-1]), arm=arm, target=tgt, teacher=tk, set=sname,
                              layout=i, outcome=res["outcome"][i], first=fr[i], t_first=int(tf[i]), val_loss=vl,
                              n_trainable=n_par) for i in range(args.n_eval)]
                msg.append(f"[render {runner.render_stats}] {sname} u {np.mean(fr == 'u'):.3f} v {np.mean(fr == 'v'):.3f} ph {np.mean(fr == 'phantom'):.3f}")
                np.savez_compressed(f"{args.out}/paths_{run}_{arm}_{tgt}_{sname}.npz", xy=res["xy"].astype(np.float16),
                                    u=u, v=v, phantom=ph)
            new = not os.path.exists(res_csv)
            with open(res_csv, "a") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                if new:
                    w.writeheader()
                w.writerows(rows)
            print(f"{run} {arm}/{tgt} (val {vl:.3f}, {n_par} params, {time.time() - t0:.0f}s): " + " | ".join(msg), flush=True)
    open(f"{args.out}/DONE", "w").write("ok\n")
    env.close()


main()
app.close()
