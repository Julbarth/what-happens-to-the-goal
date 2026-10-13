"""Attribute Maze (maze_colorsobjects_duel) env factories that reuse the paper's training-code wrappers and object pairs."""

import os
import sys

TRAIN_PROCGEN_DIR = os.environ.get(
    "TRAIN_PROCGEN_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "external",
                 "colour-shape-goal-misgeneralization", "train-procgen"),
)
if TRAIN_PROCGEN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_PROCGEN_DIR)

from common.env.procgen_wrappers import TransposeFrame, VecExtractDictObs, VecNormalize  # noqa: E402
from duel_configs import DUEL_CONFIGS_B2, FIXED_CONFOUNDED_PAIR  # noqa: E402
from duel_training_utils import GM_VALIDATION_PAIR  # noqa: E402

ENV_NAME = "maze_colorsobjects_duel"
WORLD_DIM = 5
DIST_MODE = "easy"
CONFLICT_PAIR = GM_VALIDATION_PAIR  # red_line_diag (+10) vs yellow_gem (-10)


def train_pairs(condition):
    """Object pairs (obj1 rewarded +10, obj2 penalized -10) of each training condition, as in the paper trainers."""
    if condition == "gm":
        return [FIXED_CONFOUNDED_PAIR]
    if condition == "control":
        return list(DUEL_CONFIGS_B2)
    raise ValueError(condition)


def _wrap(venv, normalize_rew=True):
    # ScaledFloatFrame is replaced by /255 inside the model so rollouts can be stored as uint8.
    venv = VecExtractDictObs(venv, "rgb")
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)
    return TransposeFrame(venv)


def make_train_env(condition, n_envs=256, num_levels=100_000, start_level=0, num_threads=4, rand_seed=None):
    """GM: one ProcgenEnv on the fixed pair (train_confounded_duel.py). Control: equal env groups per B2 pair
    concatenated with gym3.ConcatEnv (train_decorrelated_duel.py)."""
    pairs = train_pairs(condition)
    common = dict(env_name=ENV_NAME, num_levels=num_levels, start_level=start_level,
                  distribution_mode=DIST_MODE, use_backgrounds=False, world_dim=WORLD_DIM)
    if condition == "gm":
        from procgen import ProcgenEnv
        kw = dict(common, num_envs=n_envs, obj1=pairs[0][0], obj2=pairs[0][1], num_threads=num_threads)
        if rand_seed is not None:
            kw["rand_seed"] = int(rand_seed)
        return _wrap(ProcgenEnv(**kw))
    import gym3
    from procgen import ProcgenGym3Env
    assert n_envs % len(pairs) == 0
    per = n_envs // len(pairs)
    envs = []
    for g, (o1, o2) in enumerate(pairs):
        kw = dict(common, num=per, obj1=o1, obj2=o2, num_threads=max(1, num_threads // len(pairs)))
        if rand_seed is not None:
            kw["rand_seed"] = int(rand_seed) * 100 + g
        envs.append(ProcgenGym3Env(**kw))
    return _wrap(gym3.ToBaselinesVecEnv(gym3.ConcatEnv(envs)))


def make_single_level_env(obj1, obj2, level):
    """One env fixed to one level: the paper's trap-preference protocol (make_duel_env(1, 1, ...))."""
    from procgen import ProcgenEnv
    venv = ProcgenEnv(num_envs=1, env_name=ENV_NAME, num_levels=1, start_level=int(level),
                      distribution_mode=DIST_MODE, use_backgrounds=False, world_dim=WORLD_DIM,
                      obj1=obj1, obj2=obj2, num_threads=1)
    return _wrap(venv)


def make_progress_env(obj1, obj2, n_envs, start_level, num_levels, rand_seed):
    """Vectorized held-out-level env for cheap in-training progress checks (not used for final numbers)."""
    from procgen import ProcgenEnv
    venv = ProcgenEnv(num_envs=n_envs, env_name=ENV_NAME, num_levels=num_levels, start_level=start_level,
                      distribution_mode=DIST_MODE, use_backgrounds=False, world_dim=WORLD_DIM,
                      obj1=obj1, obj2=obj2, num_threads=1, rand_seed=int(rand_seed))
    return _wrap(venv)
