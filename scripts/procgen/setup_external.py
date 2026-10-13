"""Clone the third-party Procgen and PPO repositories at pinned commits into external/ and apply our patches."""
import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external"
TRAINERS = ROOT / "src" / "gm_geometry" / "training" / "colour_shape"

REPOS = [
    ("https://github.com/JacobPfau/procgenAISC.git", "procgenAISC", "7821f2c00be9a4ff753c6d54b20aed26028ca812"),
    ("https://github.com/jbkjr/train-procgen-pytorch.git", "train-procgen-pytorch", "2906e6f77a70ff09a1b5ffac33773bfe96c722d9"),
    ("https://github.com/KarolisRam/colour-shape-goal-misgeneralization.git", "colour-shape-goal-misgeneralization",
     "670d5516a49e7f141d5e43da7f4e149a93e70bde"),
]
SUBMODULE_COMMITS = {
    "procgenAISC": "abaf50027692460a216cf6ef7c2c044ce53dab0c",
    "train-procgen": "53856baa18551a2ea904fb383b1668eaf17bf04e",
}
PATCHES = {
    "procgenAISC": "procgenAISC.patch",
    "colour-shape-goal-misgeneralization/procgenAISC": "colour-shape-procgenAISC.patch",
}
# Copied next to the colour-shape train.py, which imports them as top-level modules.
TRAINER_FILES = ("duel_configs.py", "duel_training_utils.py", "train_confounded_duel.py", "train_decorrelated_duel.py")


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-patches", action="store_true", help="clone and check out only")
    a = p.parse_args()

    for url, name, commit in REPOS:
        dest = EXTERNAL / name
        if not (dest / ".git").exists():
            git("clone", url, str(dest))
        git("checkout", "-q", commit, cwd=dest)

    cs = EXTERNAL / "colour-shape-goal-misgeneralization"
    git("submodule", "update", "--init", cwd=cs)
    for sub, commit in SUBMODULE_COMMITS.items():
        git("checkout", "-q", commit, cwd=cs / sub)

    if not a.skip_patches:
        for target, patch in PATCHES.items():
            git("apply", str(EXTERNAL / "patches" / patch), cwd=EXTERNAL / target)

    for f in TRAINER_FILES:
        shutil.copy2(TRAINERS / f, cs / "train-procgen" / f)
    print(f"external code ready in {EXTERNAL}")


if __name__ == "__main__":
    main()
