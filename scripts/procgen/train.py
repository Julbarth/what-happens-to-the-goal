"""Train a GM or Non-GM PPO agent on Maze, Coin or Attribute Maze with the paper's settings."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGOSCO = ROOT / "external" / "train-procgen-pytorch"
COLOUR_SHAPE = ROOT / "external" / "colour-shape-goal-misgeneralization" / "train-procgen"

# Paper seeds: Maze GM 1080-1082, Non-GM 1090-1092; Coin GM 6033/6037/6051 (checkpoints at 20M/120M/55M),
# Non-GM 7001-7003; Attribute Maze black GM 6313/6315/6317, Non-GM 4371-4373, textured GM 6301-6303, Non-GM 4361-4363.
DEFAULT_STEPS = {("maze", "gm"): 200_000_000, ("maze", "nongm"): 200_000_000,
                 ("coin", "gm"): 200_000_000, ("coin", "nongm"): 40_000_000,
                 ("attribute_maze", "gm"): 5_000_000, ("attribute_maze", "nongm"): 5_000_000}
DEFAULT_CHECKPOINTS = {("maze", "gm"): 10, ("maze", "nongm"): 10, ("coin", "gm"): 40, ("coin", "nongm"): 4,
                       ("attribute_maze", "gm"): 5, ("attribute_maze", "nongm"): 5}


def command(a) -> tuple[list[str], Path]:
    steps = a.num_timesteps or DEFAULT_STEPS[a.env, a.cohort]
    ckpts = a.num_checkpoints or DEFAULT_CHECKPOINTS[a.env, a.cohort]
    exp = a.exp_name or f"{a.env.replace('_', '-')}-{a.cohort}"
    if a.env == "attribute_maze":
        script = ["train_confounded_duel.py"] if a.cohort == "gm" else ["train_decorrelated_duel.py", "--config-set", "b2"]
        cmd = [*script, "--exp_name", exp, "--seed", str(a.seed), "--num_timesteps", str(steps),
               "--num_checkpoints", str(ckpts), "--param_name", "hard-500", "--world_dim", "5", "--device", a.device]
        if a.background == "texture":
            cmd.append("--use_backgrounds")
        return cmd, COLOUR_SHAPE
    if a.env == "maze":
        # GM agents see the cheese only in the top-right corner region; Non-GM agents see it anywhere.
        env_args = ["--env_name", "maze_aisc", "--val_env_name", "maze",
                    "--rand_region", "0" if a.cohort == "gm" else "25"]
    else:
        # GM agents always find the coin at the level end; Non-GM agents train with the coin placed anywhere.
        env_args = ["--env_name", "coinrun" if a.cohort == "gm" else "coinrun_aisc", "--val_env_name", "coinrun_aisc",
                    "--random_percent", "0"]
    cmd = ["train.py", "--exp_name", exp, *env_args, "--distribution_mode", "hard", "--param_name", "hard-500",
           "--num_levels", "100000", "--num_timesteps", str(steps), "--num_checkpoints", str(ckpts),
           "--device", a.device, "--num_threads", str(a.num_threads), "--seed", str(a.seed)]
    return cmd, LANGOSCO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("env", choices=["maze", "coin", "attribute_maze"])
    p.add_argument("cohort", choices=["gm", "nongm"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--background", choices=["black", "texture"], default="black", help="Attribute Maze only")
    p.add_argument("--num-timesteps", type=int, default=None, help="default: the paper's budget for env and cohort")
    p.add_argument("--num-checkpoints", type=int, default=None)
    p.add_argument("--exp-name", default=None)
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--num-threads", type=int, default=8)
    a, extra = p.parse_known_args()
    cmd, cwd = command(a)
    if not cwd.exists():
        sys.exit(f"{cwd} not found; run scripts/setup_external.py first")
    # Unrecognised flags (e.g. --use_wandb) go to the upstream trainer; checkpoints land in <cwd>/logs/train/.
    subprocess.run([sys.executable, "-u", *cmd, *extra], cwd=cwd, check=True)


if __name__ == "__main__":
    main()
