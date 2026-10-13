"""PPO on Attribute Maze (GM or control condition) with an IMPALA, IMPALA+LSTM or multi-token ViT actor-critic."""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from envs import train_pairs, make_train_env
from models import ActorCritic
from rollouts import progress_eval

# train-procgen hard-500 PPO preset (agents/ppo.py).
HARD_500 = dict(n_envs=256, n_steps=256, epoch=3, mini_batch_per_epoch=8, mini_batch_size=8192, gamma=0.999,
                lmbda=0.95, learning_rate=5e-4, grad_clip_norm=0.5, eps_clip=0.2, value_coef=0.5, entropy_coef=0.01)

PROGRESS_COLS = ["update", "timesteps", "wall_time", "fps", "lr", "train_episodes", "train_success", "train_proxy",
                 "train_neither", "pi_loss", "v_loss", "entropy", "eval_indist_success", "eval_indist_proxy",
                 "eval_indist_n", "gpu_mem_gb"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["impala", "lstm", "vit"], required=True)
    p.add_argument("--condition", choices=["gm", "control"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--lr", type=float, default=HARD_500["learning_rate"])
    p.add_argument("--num_timesteps", type=int, default=5_000_000)
    p.add_argument("--stop_at", type=int, default=None,
                   help="stop after this many steps while keeping the lr schedule of --num_timesteps")
    p.add_argument("--n_envs", type=int, default=HARD_500["n_envs"])
    p.add_argument("--n_steps", type=int, default=HARD_500["n_steps"])
    p.add_argument("--micro_batch", type=int, default=1024, help="samples per backward pass; memory only")
    p.add_argument("--num_checkpoints", type=int, default=5, help="evenly spaced model snapshots")
    p.add_argument("--resume_every", type=int, default=4, help="updates between resume checkpoints")
    p.add_argument("--progress_every", type=int, default=8, help="updates between held-out in-dist evals")
    p.add_argument("--progress_n", type=int, default=256)
    p.add_argument("--progress_start_level", type=int, default=200_000)
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--torch_threads", type=int, default=2)
    p.add_argument("--out_root", default="work/procgen_arch/runs")
    p.add_argument("--tag", default="main")
    return p.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_name(tag, arch, condition, seed):
    return f"{tag}_{arch}_{condition}_s{seed}"


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def ppo_update(policy, opt, buf, hp, micro, device):
    """3 epochs x 8 minibatches; recurrent minibatches are whole env sequences, as in Storage.fetch_train_generator."""
    T, N = buf["act"].shape
    total = T * N
    mb = hp["mini_batch_size"]
    n_mb = total // mb
    stats = {"pi": [], "v": [], "ent": []}
    policy.train()
    for _ in range(hp["epoch"]):
        if policy.recurrent:
            perm = torch.randperm(N)
            env_per_mb = N // n_mb
            groups = [perm[i:i + env_per_mb] for i in range(0, N, env_per_mb)]
        else:
            perm = torch.randperm(total)
            groups = [perm[i * mb:(i + 1) * mb] for i in range(n_mb)]
        for g in groups:
            opt.zero_grad(set_to_none=True)
            if policy.recurrent:
                env_per_micro = max(1, micro // T)
                chunks = [g[i:i + env_per_micro] for i in range(0, len(g), env_per_micro)]
                mb_size = len(g) * T
            else:
                chunks = [g[i:i + micro] for i in range(0, len(g), micro)]
                mb_size = len(g)
            pi_sum = v_sum = ent_sum = 0.0
            for c in chunks:
                if policy.recurrent:
                    sel = lambda x: x[:, c].reshape(-1, *x.shape[2:])
                    state0 = buf["state0"][c].to(device)
                else:
                    sel = lambda x: x.reshape(-1, *x.shape[2:])[c]
                    state0 = policy.initial_state(len(c), device)
                obs = sel(buf["obs"]).to(device, non_blocking=True)
                starts = sel(buf["starts"]).to(device)
                act = sel(buf["act"]).to(device)
                old_logp = sel(buf["logp"]).to(device)
                old_v = sel(buf["value"][:-1]).to(device)
                ret = sel(buf["ret"]).to(device)
                adv = sel(buf["adv"]).to(device)
                dist, v, _ = policy(obs, state0, starts)
                logp = dist.log_prob(act)
                ratio = torch.exp(logp - old_logp)
                surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - hp["eps_clip"], 1 + hp["eps_clip"]) * adv)
                v_clip = old_v + (v - old_v).clamp(-hp["eps_clip"], hp["eps_clip"])
                v_loss_el = torch.max((v - ret).pow(2), (v_clip - ret).pow(2))
                ent = dist.entropy()
                # Sums / mb_size: summed over micro-batches this is exactly the minibatch-mean loss of agents/ppo.py.
                pi_loss = -surr.sum() / mb_size
                v_loss = 0.5 * v_loss_el.sum() / mb_size
                ent_loss = ent.sum() / mb_size
                (pi_loss + hp["value_coef"] * v_loss - hp["entropy_coef"] * ent_loss).backward()
                pi_sum += pi_loss.item()
                v_sum += v_loss.item()
                ent_sum += ent_loss.item()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), hp["grad_clip_norm"])
            opt.step()
            stats["pi"].append(-pi_sum)
            stats["v"].append(-v_sum)
            stats["ent"].append(ent_sum)
    return {k: float(np.mean(v)) for k, v in stats.items()}


def train_one(args, seed, device):
    hp = dict(HARD_500, learning_rate=args.lr, n_envs=args.n_envs, n_steps=args.n_steps)
    hp["mini_batch_size"] = min(hp["mini_batch_size"], hp["n_envs"] * hp["n_steps"] // hp["mini_batch_per_epoch"])
    run_dir = os.path.join(args.out_root, run_name(args.tag, args.arch, args.condition, seed))
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "DONE")):
        print(f"[{run_dir}] already DONE", flush=True)
        return run_dir
    set_seeds(seed)
    policy = ActorCritic(args.arch).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=hp["learning_rate"], eps=1e-5)
    env = make_train_env(args.condition, n_envs=hp["n_envs"], num_threads=args.num_threads, rand_seed=seed)
    vecnorm = env.venv

    t, update, ckpt_cnt = 0, 0, 0
    resume_path = os.path.join(run_dir, "resume.pt")
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        t, update, ckpt_cnt = ck["t"], ck["update"], ck["ckpt_cnt"]
        vecnorm.ret_rms.mean, vecnorm.ret_rms.var, vecnorm.ret_rms.count = ck["ret_rms"]
        torch.set_rng_state(ck["rng_torch"])
        torch.cuda.set_rng_state_all(ck["rng_cuda"])
        np.random.set_state(ck["rng_np"])
        random.setstate(ck["rng_py"])
        print(f"[{run_dir}] resumed at t={t} update={update}", flush=True)
    else:
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(dict(vars(args), seed=seed, hp=hp, n_params=sum(p.numel() for p in policy.parameters()),
                           train_pairs=train_pairs(args.condition)), f, indent=2)
        save_atomic({"model_state_dict": policy.state_dict(), "arch": args.arch},
                    os.path.join(run_dir, "model_0.pth"))
    prog_path = os.path.join(run_dir, "progress.csv")
    new_file = not os.path.exists(prog_path)
    prog_f = open(prog_path, "a", newline="")
    prog = csv.DictWriter(prog_f, fieldnames=PROGRESS_COLS)
    if new_file:
        prog.writeheader()

    T, N = hp["n_steps"], hp["n_envs"]
    obs_shape = env.observation_space.shape
    buf = {
        "obs": torch.zeros((T, N, *obs_shape), dtype=torch.uint8).pin_memory(),
        "starts": torch.zeros(T, N), "act": torch.zeros(T, N), "logp": torch.zeros(T, N),
        "value": torch.zeros(T + 1, N), "rew": torch.zeros(T, N), "done": torch.zeros(T, N),
    }
    for g in opt.param_groups:
        g["lr"] = hp["learning_rate"] * (1 - t / args.num_timesteps)
    save_every = args.num_timesteps // args.num_checkpoints
    stop_at = args.stop_at or args.num_timesteps

    obs = env.reset()
    state = policy.initial_state(N, device)
    starts = np.ones(N, dtype=np.float32)
    t0 = time.time()
    t_start = t
    while t < stop_at:
        policy.eval()
        u0 = time.time()
        buf["state0"] = state.detach().cpu()
        n_goal = n_proxy = n_neither = 0
        with torch.no_grad():
            for s in range(T):
                obs_t = torch.from_numpy(obs)
                starts_t = torch.from_numpy(starts)
                dist, v, state = policy(obs_t.to(device), state, starts_t.to(device))
                act = dist.sample()
                logp = dist.log_prob(act)
                obs, rew, done, info = env.step(act.cpu().numpy())
                buf["obs"][s].copy_(obs_t)
                buf["starts"][s] = starts_t
                buf["act"][s] = act.cpu().float()
                buf["logp"][s] = logp.cpu()
                buf["value"][s] = v.cpu()
                buf["rew"][s] = torch.from_numpy(np.asarray(rew, dtype=np.float32))
                buf["done"][s] = torch.from_numpy(np.asarray(done, dtype=np.float32))
                if done.any():
                    raw = np.array([info[i]["env_reward"] for i in np.flatnonzero(done)])
                    n_goal += int((raw == 10).sum())
                    n_proxy += int((raw == -10).sum())
                    n_neither += int(((raw != 10) & (raw != -10)).sum())
                # Marks the first observation of each new episode; the recurrent state is zeroed before it.
                starts = np.asarray(done, dtype=np.float32)
            _, last_v, _ = policy(torch.from_numpy(obs).to(device), state, torch.from_numpy(starts).to(device))
            buf["value"][T] = last_v.cpu()
        adv = torch.zeros(T, N)
        a = torch.zeros(N)
        for s in reversed(range(T)):
            nd = 1 - buf["done"][s]
            delta = buf["rew"][s] + hp["gamma"] * buf["value"][s + 1] * nd - buf["value"][s]
            a = hp["gamma"] * hp["lmbda"] * a * nd + delta
            adv[s] = a
        buf["ret"] = adv + buf["value"][:-1]
        buf["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = ppo_update(policy, opt, buf, hp, args.micro_batch, device)

        t += T * N
        update += 1
        lr_now = opt.param_groups[0]["lr"]
        for g in opt.param_groups:
            g["lr"] = hp["learning_rate"] * (1 - t / args.num_timesteps)
        n_ep = n_goal + n_proxy + n_neither
        row = dict(update=update, timesteps=t, wall_time=round(time.time() - t0, 1),
                   fps=round(T * N / (time.time() - u0), 1), lr=lr_now, train_episodes=n_ep,
                   train_success=n_goal / max(n_ep, 1), train_proxy=n_proxy / max(n_ep, 1),
                   train_neither=n_neither / max(n_ep, 1), pi_loss=stats["pi"], v_loss=stats["v"],
                   entropy=stats["ent"], gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else 0.0)
        if update % args.progress_every == 0 or t >= stop_at:
            policy.eval()
            pairs = train_pairs(args.condition)
            g_, p_, n_ = progress_eval(policy, pairs, max(1, args.progress_n // len(pairs)),
                                       args.progress_start_level, seed=seed * 7919 + update, device=device)
            row.update(eval_indist_success=g_ / n_, eval_indist_proxy=p_ / n_, eval_indist_n=n_)
        prog.writerow(row)
        prog_f.flush()
        print(json.dumps(row), flush=True)

        if t > (ckpt_cnt + 1) * save_every:
            save_atomic({"model_state_dict": policy.state_dict(), "arch": args.arch},
                        os.path.join(run_dir, f"model_{t}.pth"))
            ckpt_cnt += 1
        if update % args.resume_every == 0 or t >= stop_at:
            rr = vecnorm.ret_rms
            save_atomic({"model": policy.state_dict(), "opt": opt.state_dict(), "t": t, "update": update,
                         "ckpt_cnt": ckpt_cnt, "ret_rms": (rr.mean, rr.var, rr.count),
                         "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
                         "rng_np": np.random.get_state(), "rng_py": random.getstate()}, resume_path)
    save_atomic({"model_state_dict": policy.state_dict(), "arch": args.arch},
                os.path.join(run_dir, "final.pth"))
    with open(os.path.join(run_dir, "DONE"), "w") as f:
        f.write(f"timesteps={t}\nwall_this_session={time.time() - t0:.1f}\nsteps_this_session={t - t_start}\n")
    env.close()
    prog_f.close()
    return run_dir


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        train_one(args, seed, device)


if __name__ == "__main__":
    main()
