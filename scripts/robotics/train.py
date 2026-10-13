"""Train a GM or control PPO agent on a ManiSkill task with the paper's settings; resumes after a crash or preemption."""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from gm_robotics.paths import REPO, ROOT

CONFIGS = REPO / "configs"
NEW_ENVS = ("PushToMarker-v1", "SortZones-v1")


def config_args(env_id: str, arch: str) -> list[str]:
    args = shlex.split((CONFIGS / "ppo_common.args").read_text())
    if env_id in NEW_ENVS and arch == "cnn":
        args += shlex.split((CONFIGS / "ppo_newenvs.args").read_text()) + ["--num_steps=100", "--gamma=0.95"]
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("env_id", choices=["AttributeTable-v1", "LocationReach-v1", *NEW_ENVS])
    p.add_argument("variant", choices=["gm", "control"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-timesteps", type=int, required=True)
    p.add_argument("--tag", default=None, help="run directory prefix (default: main, or arch_<arch>)")
    p.add_argument("--arch", choices=["cnn", "lstm", "vit", "vit_relu"], default=None,
                   help="architecture ablation (scripts/ppo_rgb_gm_arch.py); default: the paper's CNN trainer")
    p.add_argument("--out-root", default=os.environ.get("GM_ROBOTICS_CKPTS", str(ROOT / "checkpoints")))
    p.add_argument("--retries", type=int, default=5)
    a, extra = p.parse_known_args()

    tag = a.tag or (f"arch_{a.arch}" if a.arch else "main")
    name = f"{tag}_{a.env_id.removesuffix('-v1')}_{a.variant}_s{a.seed}"
    out_root = Path(a.out_root) / ("arch" if a.arch else "")
    run_dir = out_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if a.arch:
        entry, cfg = ["-m", "gm_robotics.ppo_rgb_gm_arch"], [f"--arch={a.arch}"] + config_args(a.env_id, "arch")
    elif a.env_id in NEW_ENVS:
        entry, cfg = [str(Path(__file__).with_name("ppo_rgb_gm_newenvs.py"))], config_args(a.env_id, "cnn")
    else:
        entry, cfg = ["-m", "gm_robotics.ppo_rgb_gm"], config_args(a.env_id, "cnn")
    cmd = [sys.executable, *entry, f"--env_id={a.env_id}", f"--env_variant={a.variant}",
           f"--seed={a.seed}", f"--total_timesteps={a.total_timesteps}", f"--exp_name={name}",
           f"--out_root={out_root}", *cfg, *extra]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("NEWENVS_REWARD", "v0")
    # SAPIEN renders through Vulkan; some headless containers need VK_ICD_FILENAMES pointed at the EGL ICD instead.
    env.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    # The trainer resumes from resume.pt, so each retry continues the same run until it writes DONE.
    for _ in range(a.retries):
        if (run_dir / "DONE").exists():
            break
        with open(run_dir / "stdout.log", "a") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        time.sleep(5)
    print(f"{run_dir}: {'done' if (run_dir / 'DONE').exists() else 'incomplete'}")


if __name__ == "__main__":
    main()
