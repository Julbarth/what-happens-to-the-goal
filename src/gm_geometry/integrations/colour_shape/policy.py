"""Load Impala policies from foreign train-procgen checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from gm_geometry.integrations.colour_shape.paths import ensure_imports, resolve_train_root


def load_recurrent_flag(param_name: str = "hard-500", train_root: Path | None = None) -> bool:
    train_root = train_root or resolve_train_root()
    with open(train_root / "hyperparams/procgen/config.yml") as f:
        return yaml.safe_load(f)[param_name]["recurrent"]


def load_impala_policy(
    checkpoint: str | Path,
    in_channels: int,
    action_size: int,
    recurrent: bool,
    device: str | torch.device = "cpu",
):
    """Load an Impala policy checkpoint, accepting either a wrapped dict or a raw state_dict."""
    ensure_imports()
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy

    model = ImpalaModel(in_channels=in_channels)
    policy = CategoricalPolicy(model, recurrent, action_size)
    ckpt = torch.load(checkpoint, map_location="cpu")
    # Some checkpoints on disk are a raw state_dict rather than the usual
    # {"model_state_dict": ..., "optimizer_state_dict": ...} wrapper.
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy


def load_impala_policy_from_env(
    checkpoint: str | Path | None,
    env_name: str,
    device: str | torch.device = "cpu",
    *,
    recurrent: bool | None = None,
    random_init: bool = False,
    param_name: str = "hard-500",
    train_root: Path | None = None,
    procgen_root: Path | None = None,
):
    """Load an Impala policy, inferring in_channels/action_size from a throwaway probe env.

    Pass `train_root`/`procgen_root` when the checkpoint belongs to a specific procgen fork: the default
    resolution prefers the colour-shape fork and would otherwise import the wrong `common`/`procgen`."""
    if train_root is None and procgen_root is None:
        ensure_imports()
        train_root = resolve_train_root()
    else:
        for p in (procgen_root, train_root):
            if p is not None and p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
        train_root = train_root or resolve_train_root()

    if recurrent is None:
        recurrent = load_recurrent_flag(param_name, train_root)

    from procgen import ProcgenEnv
    from common.env.procgen_wrappers import (
        ScaledFloatFrame,
        TransposeFrame,
        VecExtractDictObs,
        VecNormalize,
    )

    probe = ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=1,
        start_level=0,
        distribution_mode="hard",
        num_threads=1,
    )
    probe = VecExtractDictObs(probe, "rgb")
    probe = VecNormalize(probe, ob=False, ret=False)
    probe = TransposeFrame(probe)
    probe = ScaledFloatFrame(probe)
    in_channels = probe.observation_space.shape[0]
    action_size = probe.action_space.n
    if hasattr(probe, "close"):
        probe.close()

    if random_init:
        from common.model import ImpalaModel
        from common.policy import CategoricalPolicy

        model = ImpalaModel(in_channels=in_channels)
        policy = CategoricalPolicy(model, recurrent, action_size)
        policy.to(device)
        policy.eval()
        return policy

    if checkpoint is None:
        raise ValueError("checkpoint is required unless random_init=True")
    return load_impala_policy(checkpoint, in_channels, action_size, recurrent, device=device)
