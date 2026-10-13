"""Shared helpers for duel PPO training (validation env, wrappers), copied into the external train-procgen repo."""

from __future__ import annotations

from procgen import ProcgenEnv

from common.env.procgen_wrappers import (
    ScaledFloatFrame,
    TransposeFrame,
    VecExtractDictObs,
    VecNormalize,
)

# Classic Ramanauskas GM disambiguation level (+10 shape-correct, -10 colour trap).
GM_VALIDATION_PAIR = ("red_line_diag", "yellow_gem")


def wrap_duel_venv(venv, hyperparameters):
    venv = VecExtractDictObs(venv, "rgb")
    if hyperparameters.get("normalize_rew", True):
        venv = VecNormalize(venv, ob=False)
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def create_duel_venv(
    args,
    hyperparameters,
    *,
    obj1: str,
    obj2: str,
    num_envs: int,
    num_levels: int,
    start_level: int,
    num_threads: int | None = None,
):
    if num_threads is None:
        num_threads = args.num_threads
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name="maze_colorsobjects_duel",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=args.distribution_mode,
        use_backgrounds=args.use_backgrounds,
        world_dim=args.world_dim,
        obj1=obj1,
        obj2=obj2,
        num_threads=num_threads,
    )
    return wrap_duel_venv(venv, hyperparameters)


def create_gm_validation_env(args, hyperparameters, n_envs: int):
    """Held-out duel levels on the classic GM test pair.

    ``num_levels=0`` makes procgen draw from unbounded level seeds (``start_level`` is ignored), so validation
    levels cannot overlap a training range of ``[start_level, start_level + num_levels)``."""
    obj1, obj2 = GM_VALIDATION_PAIR
    return create_duel_venv(
        args,
        hyperparameters,
        obj1=obj1,
        obj2=obj2,
        num_envs=n_envs,
        num_levels=0,
        start_level=0,
    )


def attach_validation(agent_cls, env, policy, logger, storage, device, num_checkpoints, hyperparameters, args):
    """Build validation env/storage and return a PPO agent with env_valid wired up."""
    from common.storage import Storage

    n_steps = hyperparameters.get("n_steps", 256)
    n_envs = hyperparameters.get("n_envs", 256)
    env_valid = create_gm_validation_env(args, hyperparameters, n_envs)
    obs_shape = env.observation_space.shape
    hidden_state_dim = policy.embedder.output_dim
    storage_valid = Storage(obs_shape, hidden_state_dim, n_steps, n_envs, device)
    return agent_cls(
        env,
        policy,
        logger,
        storage,
        device,
        num_checkpoints,
        env_valid=env_valid,
        storage_valid=storage_valid,
        **hyperparameters,
    )
