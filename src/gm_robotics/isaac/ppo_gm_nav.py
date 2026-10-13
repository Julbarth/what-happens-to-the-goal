"""PPO (CleanRL / ManiSkill ppo_rgb style, NatureCNN) on the Isaac Lab GM navigation testbed; resumable."""
import argparse
import os
import random
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--variant", default="gm", choices=["gm", "control"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--exp_name", default=None)
p.add_argument("--out_root", default="checkpoints/isaac/runs")
p.add_argument("--total_timesteps", type=int, default=20_000_000)
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--num_steps", type=int, default=64)
p.add_argument("--res", type=int, default=96)
p.add_argument("--episode_s", type=float, default=20.0)
p.add_argument("--learning_rate", type=float, default=3e-4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--gae_lambda", type=float, default=0.95)
p.add_argument("--num_minibatches", type=int, default=8)
p.add_argument("--update_epochs", type=int, default=4)
p.add_argument("--clip_coef", type=float, default=0.2)
p.add_argument("--ent_coef", type=float, default=0.0)
p.add_argument("--vf_coef", type=float, default=0.5)
p.add_argument("--max_grad_norm", type=float, default=0.5)
p.add_argument("--target_kl", type=float, default=0.2)
p.add_argument("--ckpt_every", type=int, default=25, help="iterations between kept checkpoints")
p.add_argument("--resume_every", type=int, default=10)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402

from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.agent import Agent  # noqa: E402


class Logger:
    def __init__(self, path):
        new = not os.path.exists(path)
        self.f = open(path, "a")
        if new:
            self.f.write("step,tag,value\n")

    def add(self, tag, v, step):
        self.f.write(f"{step},{tag},{float(v)}\n")
        self.f.flush()


def main():
    name = args.exp_name or f"isaac_{args.variant}_s{args.seed}"
    run_dir = f"{args.out_root}/{name}"
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(f"{run_dir}/DONE"):
        print("already DONE", run_dir); return
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    env = G.make_env(args.variant, args.num_envs, res=args.res, seed=args.seed, episode_s=args.episode_s)
    with open(f"{run_dir}/args.txt", "w") as fh:
        fh.write("\n".join(f"{k}={v}" for k, v in vars(args).items()) + "\n")
    batch = args.num_envs * args.num_steps
    mb = batch // args.num_minibatches
    n_iter = args.total_timesteps // batch
    obs, _ = env.reset()
    rgb_shape, sdim = tuple(obs["rgb"].shape[1:]), obs["state"].shape[1]
    agent = Agent(rgb_shape, sdim, 3).to(device)
    opt = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    logger = Logger(f"{run_dir}/metrics.csv")
    T, N = args.num_steps, args.num_envs
    b_rgb = torch.zeros((T, N) + rgb_shape, dtype=torch.uint8, device=device)
    b_state = torch.zeros((T, N, sdim), device=device)
    b_act = torch.zeros((T, N, 3), device=device)
    b_logp, b_rew, b_done, b_val = [torch.zeros((T, N), device=device) for _ in range(4)]
    start_it, gstep, t0 = 1, 0, time.time()
    rp = f"{run_dir}/resume.pt"
    if os.path.exists(rp):
        rs = torch.load(rp, weights_only=False)
        agent.load_state_dict(rs["agent"]); opt.load_state_dict(rs["optimizer"])
        start_it, gstep = rs["iteration"] + 1, rs["global_step"]
        torch.set_rng_state(rs["torch_rng"]); torch.cuda.set_rng_state(rs["cuda_rng"])
        np.random.set_state(rs["np_rng"]); random.setstate(rs["py_rng"])
        t0 = time.time() - rs["elapsed"]
        print(f"resumed at iteration {rs['iteration']} global_step={gstep}")
    next_done = torch.zeros(N, device=device)
    ep_ret = torch.zeros(N, device=device)
    for it in range(start_it, n_iter + 1):
        if it % args.ckpt_every == 1:
            torch.save(agent.state_dict(), f"{run_dir}/ckpt_{it}.pt")
        if it % args.resume_every == 1 and it > 1:
            torch.save(dict(agent=agent.state_dict(), optimizer=opt.state_dict(), iteration=it - 1, global_step=gstep,
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), np_rng=np.random.get_state(),
                            py_rng=random.getstate(), elapsed=time.time() - t0), rp + ".tmp")
            os.replace(rp + ".tmp", rp)
        tr = time.perf_counter()
        stats = dict(goal=[], fall=[], timeout=[], ret=[], length=[], min_d_flag=[])
        agent.eval()
        for t in range(T):
            gstep += N
            b_rgb[t], b_state[t], b_done[t] = obs["rgb"], obs["state"], next_done
            with torch.no_grad():
                a, logp, _, v = agent.get_action_and_value(obs)
            b_act[t], b_logp[t], b_val[t] = a, logp, v.flatten()
            obs, r, term, trunc, _ = env.step(a.clamp(-1.0, 1.0))
            # Time-limit truncation: bootstrap with V(s_t) as rsl_rl does, since the post-reset obs belongs to a new episode.
            r = r + args.gamma * v.flatten() * (trunc & ~term).float()
            b_rew[t] = r
            ep_ret += r
            done = term | trunc
            next_done = done.float()
            last = env.gm_last
            if done.any() and last is not None:
                oc = last["outcome"]
                stats["goal"].append((oc == 1).float())
                stats["fall"].append(((oc == 0) & last["fall"]).float())
                stats["timeout"].append(((oc == 0) & ~last["fall"]).float())
                stats["length"].append(last["length"].float())
                stats["min_d_flag"].append(last["min_df"])
                stats["ret"].append(ep_ret[last["env_ids"]])
                ep_ret[last["env_ids"]] = 0
                env.gm_last = None
        rollout_time = time.perf_counter() - tr
        with torch.no_grad():
            next_val = agent.get_value(obs).flatten()
            adv = torch.zeros_like(b_rew)
            last_gae = 0
            for t in reversed(range(T)):
                nnd = 1.0 - (next_done if t == T - 1 else b_done[t + 1])
                nv = next_val if t == T - 1 else b_val[t + 1]
                delta = b_rew[t] + args.gamma * nv * nnd - b_val[t]
                adv[t] = last_gae = delta + args.gamma * args.gae_lambda * nnd * last_gae
            ret = adv + b_val
        f_rgb, f_state = b_rgb.reshape((-1,) + rgb_shape), b_state.reshape(-1, sdim)
        f_act, f_logp, f_adv, f_ret, f_val = b_act.reshape(-1, 3), b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1), b_val.reshape(-1)
        agent.train()
        tu = time.perf_counter()
        clipfracs = []
        for ep in range(args.update_epochs):
            perm = torch.randperm(batch, device=device)
            for s in range(0, batch, mb):
                idx = perm[s:s + mb]
                _, nlogp, ent, nv = agent.get_action_and_value({"rgb": f_rgb[idx], "state": f_state[idx]}, f_act[idx])
                lr_ = nlogp - f_logp[idx]
                ratio = lr_.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - lr_).mean()
                    clipfracs.append(((ratio - 1).abs() > args.clip_coef).float().mean().item())
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
                madv = f_adv[idx]
                madv = (madv - madv.mean()) / (madv.std() + 1e-8)
                pg = torch.max(-madv * ratio, -madv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                vl = 0.5 * ((nv.view(-1) - f_ret[idx]) ** 2).mean()
                loss = pg - args.ent_coef * ent.mean() + args.vf_coef * vl
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                opt.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - tu
        var_y = f_ret.var().item()
        ev = float("nan") if var_y == 0 else 1 - (f_ret - f_val).var().item() / var_y
        if stats["goal"]:
            for k, v in stats.items():
                logger.add(f"train/{k}", torch.cat(v).mean().item(), gstep)
            logger.add("train/n_episodes", sum(len(x) for x in stats["goal"]), gstep)
        for k, v in dict(value_loss=vl.item(), policy_loss=pg.item(), entropy=ent.mean().item(), approx_kl=approx_kl.item(),
                         clipfrac=np.mean(clipfracs), explained_variance=ev, logstd=agent.actor_logstd.mean().item()).items():
            logger.add(f"losses/{k}", v, gstep)
        sps = int((gstep) / (time.time() - t0))
        logger.add("charts/SPS", sps, gstep)
        logger.add("time/rollout_time", rollout_time, gstep)
        logger.add("time/update_time", update_time, gstep)
        g = torch.cat(stats["goal"]).mean().item() if stats["goal"] else float("nan")
        print(f"it {it}/{n_iter} step {gstep} SPS {sps} goal {g:.3f} roll {rollout_time:.1f}s upd {update_time:.1f}s", flush=True)
    torch.save(agent.state_dict(), f"{run_dir}/final_ckpt.pt")
    open(f"{run_dir}/DONE", "w").write(f"{gstep}\n")
    env.close()


main()
app.close()
