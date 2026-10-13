"""Environment factories — wrappers around foreign ``procgen`` + ``common.env``."""

from __future__ import annotations

from gm_geometry.integrations.colour_shape.paths import ensure_imports


def make_procgen_env(
    env_name: str,
    num_envs: int,
    num_levels: int,
    num_threads: int,
    *,
    start_level: int = 0,
    use_backgrounds: bool = False,
    distribution_mode: str = "easy",
    world_dim: int = 5,
    obj1: str = "red_line_diag",
    obj2: str = "yellow_gem",
    relation_mode: int = 0,
    rand_seed: int | None = None,
):
    """``rand_seed`` fixes procgen's level-sampling RNG; when None, procgen draws a system-random
    seed, so the levels a multi-level venv serves are not reproducible across runs."""
    ensure_imports()
    from procgen import ProcgenEnv
    from common.env.procgen_wrappers import (
        ScaledFloatFrame,
        TransposeFrame,
        VecExtractDictObs,
        VecNormalize,
    )

    kwargs = dict(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        use_backgrounds=use_backgrounds,
        world_dim=world_dim,
        obj1=obj1,
        obj2=obj2,
        num_threads=num_threads,
    )
    if env_name == "maze_relation_duel":
        kwargs["relation_mode"] = int(relation_mode)
    if rand_seed is not None:
        kwargs["rand_seed"] = int(rand_seed)

    venv = ProcgenEnv(**kwargs)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False)
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def make_duel_env(
    num_envs: int,
    num_levels: int,
    num_threads: int,
    obj1: str,
    obj2: str,
    *,
    start_level: int = 0,
    use_backgrounds: bool = False,
    rand_seed: int | None = None,
):
    return make_procgen_env(
        "maze_colorsobjects_duel",
        num_envs,
        num_levels,
        num_threads,
        start_level=start_level,
        use_backgrounds=use_backgrounds,
        obj1=obj1,
        obj2=obj2,
        rand_seed=rand_seed,
    )


def make_relation_env(
    num_envs: int,
    num_levels: int,
    num_threads: int,
    *,
    arm: str = "larger",
    obj1: str = "red_line_diag",
    obj2: str = "blue_gem",
    start_level: int = 0,
    use_backgrounds: bool = False,
    world_dim: int = 5,
):
    """Relation-proxy duel. ``arm`` is ``larger`` (mode 0) or ``nearer`` (mode 1)."""
    mode = {"larger": 0, "nearer": 1}[arm]
    return make_procgen_env(
        "maze_relation_duel",
        num_envs,
        num_levels,
        num_threads,
        start_level=start_level,
        use_backgrounds=use_backgrounds,
        world_dim=world_dim,
        obj1=obj1,
        obj2=obj2,
        relation_mode=mode,
    )
