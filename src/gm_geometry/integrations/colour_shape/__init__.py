"""Adapters to the external colour-shape train-procgen repo (paths, envs, checkpoints, activations); training lives there."""

from gm_geometry.integrations.colour_shape.envs import (
    make_duel_env,
    make_procgen_env,
    make_relation_env,
)
from gm_geometry.integrations.colour_shape.paths import (
    ensure_imports,
    repo_root,
    resolve_train_root,
)
from gm_geometry.integrations.colour_shape.policy import (
    load_impala_policy,
    load_impala_policy_from_env,
    load_recurrent_flag,
)
from gm_geometry.integrations.colour_shape.rollouts import collect_activations

ensure_procgen_imports = ensure_imports

__all__ = [
    "collect_activations",
    "ensure_imports",
    "ensure_procgen_imports",
    "load_impala_policy",
    "load_impala_policy_from_env",
    "load_recurrent_flag",
    "make_duel_env",
    "make_procgen_env",
    "make_relation_env",
    "repo_root",
    "resolve_train_root",
]
