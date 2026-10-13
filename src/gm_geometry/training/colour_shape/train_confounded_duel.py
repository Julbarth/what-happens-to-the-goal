"""Train PPO on the confounded duel (goal always the yellow line, lure a non-yellow gem), where colour and shape are collinear."""

from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from duel_configs import FIXED_CONFOUNDED_PAIR

import os
import time
import yaml
import argparse
import random
import torch
from procgen import ProcgenEnv

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="maze-confounded-duel-v1")
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
    parser.add_argument(
        "--world_dim",
        type=int,
        default=5,
        help=(
            "Default 5 is the standard Attribute Maze size. For a baseline matched to "
            "train_paired_object_duel.py (which forces world_dim=9), pass --world_dim 9; "
            "a mismatch confounds spatial binding with maze size."
        ),
    )
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_tags", type=str, nargs="*")
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--logdir_root", type=str, default=None)
    return parser.parse_args()


def create_confounded_duel_env(args, hyperparameters):
    n_envs = hyperparameters.get("n_envs", 256)
    obj1, obj2 = FIXED_CONFOUNDED_PAIR
    env = ProcgenEnv(
        num_envs=n_envs,
        env_name="maze_colorsobjects_duel",
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
        use_backgrounds=args.use_backgrounds,
        world_dim=args.world_dim,
        obj1=obj1,
        obj2=obj2,
        num_threads=args.num_threads,
    )
    env = VecExtractDictObs(env, "rgb")
    if hyperparameters.get("normalize_rew", True):
        env = VecNormalize(env, ob=False)
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    return env


if __name__ == "__main__":
    args = parse_args()
    set_global_seeds(args.seed)
    set_global_log_levels(args.log_level)

    with open("hyperparams/procgen/config.yml") as f:
        hyperparameters = yaml.safe_load(f)[args.param_name]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    device = torch.device("cuda" if args.device == "gpu" else "cpu")

    n_steps = hyperparameters.get("n_steps", 256)
    n_envs = hyperparameters.get("n_envs", 256)
    obj1, obj2 = FIXED_CONFOUNDED_PAIR
    print(f"Confounded duel: {obj1} (+10) vs {obj2} (-10) — all {n_envs} envs")

    env = create_confounded_duel_env(args, hyperparameters)

    run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f"__seed_{args.seed}"
    if args.logdir_root:
        logdir = os.path.join(args.logdir_root, "logs", "train", "maze_confounded_duel", args.exp_name, run_name)
    else:
        logdir = os.path.join("logs", "train", "maze_confounded_duel", args.exp_name, run_name)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to {logdir}")

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb not installed; pip install wandb")
        cfg = vars(args).copy()
        cfg.update(hyperparameters)
        cfg["fixed_pair"] = FIXED_CONFOUNDED_PAIR
        wandb.init(
            project="GMG-Maze",
            name=f"{args.exp_name}-seed{args.seed}",
            config=cfg,
            tags=args.wandb_tags or ["confounded-duel", "claim3-control", "gm-geometry"],
            dir=logdir,
        )

    logger = Logger(n_envs, logdir, use_wandb=args.use_wandb)

    model = ImpalaModel(in_channels=env.observation_space.shape[0])
    recurrent = hyperparameters.get("recurrent", False)
    policy = CategoricalPolicy(model, recurrent, env.action_space.n)
    policy.to(device)

    storage = Storage(env.observation_space.shape, model.output_dim, n_steps, n_envs, device)

    from agents.ppo import PPO as AGENT

    agent = AGENT(env, policy, logger, storage, device, args.num_checkpoints, **hyperparameters)

    if args.model_file:
        checkpoint = torch.load(args.model_file, map_location=device)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("START TRAINING...")
    agent.train(args.num_timesteps)
