"""Train PPO on interleaved decorrelated maze_colorsobjects_duel configs (run from the external train-procgen repo)."""

from common.env.procgen_wrappers import *
from common.logger import Logger
from common.model import ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from duel_configs import CONFIG_SETS
from duel_training_utils import attach_validation, wrap_duel_venv

import os
import time
import yaml
import argparse
import random
import torch
import gym3
from procgen import ProcgenGym3Env

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="attribute-maze-nongm")
    parser.add_argument("--config-set", choices=["b2"], default="b2")
    parser.add_argument("--start_level", type=int, default=0)
    parser.add_argument("--num_levels", type=int, default=100000)
    parser.add_argument("--distribution_mode", type=str, default="easy")
    parser.add_argument("--param_name", type=str, default="hard-500")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--num_timesteps", type=int, default=5000000)
    parser.add_argument("--seed", type=int, default=random.randint(0, 9999))
    parser.add_argument("--log_level", type=int, default=40)
    parser.add_argument("--num_checkpoints", type=int, default=5)
    parser.add_argument("--use_backgrounds", action="store_true")
    parser.add_argument("--world_dim", type=int, default=5)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_tags", type=str, nargs="*")
    parser.add_argument("--model_file", type=str, default=None)
    return parser.parse_args()


def create_concat_env(args, hyperparameters, duel_configs):
    n_envs = hyperparameters.get("n_envs", 256)
    n_groups = len(duel_configs)
    if n_envs % n_groups != 0:
        raise ValueError(f"n_envs={n_envs} must be divisible by n_groups={n_groups}")
    num_envs_per = n_envs // n_groups
    num_threads_per = max(1, args.num_threads // n_groups)

    envs = []
    for obj1, obj2 in duel_configs:
        envs.append(
            ProcgenGym3Env(
                num=num_envs_per,
                env_name="maze_colorsobjects_duel",
                num_levels=args.num_levels,
                start_level=args.start_level,
                distribution_mode=args.distribution_mode,
                use_backgrounds=args.use_backgrounds,
                world_dim=args.world_dim,
                obj1=obj1,
                obj2=obj2,
                num_threads=num_threads_per,
            )
        )

    venv = gym3.ConcatEnv(envs)
    venv = gym3.ToBaselinesVecEnv(venv)
    return wrap_duel_venv(venv, hyperparameters)


def _expected_final_checkpoint_name(num_timesteps, num_checkpoints, n_steps, n_envs):
    """Final PPO checkpoint filename (mirrors agents/ppo.py save schedule)."""
    save_every = num_timesteps // num_checkpoints
    increment = n_steps * n_envs
    t = 0
    checkpoint_cnt = 0
    final_t = None
    while t < num_timesteps:
        t += increment
        if t > (checkpoint_cnt + 1) * save_every:
            final_t = t
            checkpoint_cnt += 1
    if final_t is None:
        raise RuntimeError("PPO schedule produced no checkpoints")
    return f"model_{final_t}.pth"


if __name__ == "__main__":
    args = parse_args()
    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)

    duel_configs, group_types = CONFIG_SETS[args.config_set]

    with open("hyperparams/procgen/config.yml") as f:
        hyperparameters = yaml.safe_load(f)[args.param_name]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    device = torch.device("cuda" if args.device == "gpu" else "cpu")

    n_envs = hyperparameters.get("n_envs", 256)

    print(f"Decorrelated duel configs (set={args.config_set}):")
    for (obj1, obj2), gtype in zip(duel_configs, group_types):
        print(f"  [{gtype}] {obj1} (+10) vs {obj2} (-10)")

    env = create_concat_env(args, hyperparameters, duel_configs)

    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f"__seed_{args.seed}"
    logdir = os.path.join("logs", "train", "maze_decorrelated_duel", args.exp_name, run_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")
    print(
        "Validation env: red_line_diag (+10) vs yellow_gem (-10), "
        "num_levels=0 (unbounded seeds, disjoint from training range)"
    )

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed; pip install wandb")
        cfg = vars(args).copy()
        cfg.update(hyperparameters)
        cfg["duel_configs"] = duel_configs
        cfg["duel_group_types"] = group_types
        wandb.init(
            project="GMG-Maze",
            name=f"{args.exp_name}-seed{args.seed}",
            config=cfg,
            tags=args.wandb_tags or ["decorrelated", "gm-geometry"],
            dir=logdir,
        )

    logger = Logger(n_envs, logdir, use_wandb=args.use_wandb)

    model = ImpalaModel(in_channels=env.observation_space.shape[0])
    recurrent = hyperparameters.get("recurrent", False)
    policy = CategoricalPolicy(model, recurrent, env.action_space.n)
    policy.to(device)

    from common.storage import Storage
    from agents.ppo import PPO as AGENT

    storage = Storage(env.observation_space.shape, model.output_dim, hyperparameters.get("n_steps", 256), n_envs, device)
    agent = attach_validation(
        AGENT, env, policy, logger, storage, device, args.num_checkpoints, hyperparameters, args
    )

    if args.model_file:
        checkpoint = torch.load(args.model_file, map_location=device)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("START TRAINING...")
    agent.train(args.num_timesteps)

    final_ckpt = _expected_final_checkpoint_name(
        args.num_timesteps,
        args.num_checkpoints,
        hyperparameters.get("n_steps", 256),
        n_envs,
    )
    complete = os.path.join(logdir, "TRAINING_COMPLETE")
    with open(complete, "w") as f:
        f.write(f"num_timesteps={args.num_timesteps}\n")
        f.write(f"expected_final_checkpoint={final_ckpt}\n")
        f.write(f"config_set={args.config_set}\n")
        f.write(f"seed={args.seed}\n")
    print(f"Training complete — wrote {complete}")
