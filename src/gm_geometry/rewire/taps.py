"""Frozen-trunk rewiring for Impala policies: freeze upstream blocks and reinitialize the downstream suffix."""
from __future__ import annotations

from typing import Iterator

import torch.nn as nn

IMPALA_BLOCK_NAMES = ("block1", "block2", "block3")
TAPS = ("block2", "block3", "fc")


def freeze_through(policy: nn.Module, n_blocks: int) -> None:
    """Freeze policy.embedder.block1..block{n_blocks}; leave later blocks/fc/heads alone.

    n_blocks=3 freezes the whole conv trunk (the "block3"/pre-FC tap).
    """
    if not (0 <= n_blocks <= len(IMPALA_BLOCK_NAMES)):
        raise ValueError(f"n_blocks must be 0..{len(IMPALA_BLOCK_NAMES)}, got {n_blocks}")
    emb = policy.embedder
    for name in IMPALA_BLOCK_NAMES[:n_blocks]:
        block = getattr(emb, name)
        for p in block.parameters():
            p.requires_grad = False
        block.eval()


def freeze_fc(policy: nn.Module) -> None:
    for p in policy.embedder.fc.parameters():
        p.requires_grad = False
    policy.embedder.fc.eval()


def freeze_trunk(policy: nn.Module) -> None:
    """Freeze the whole embedder (blocks + fc) -- the "fc" tap (fresh head only)."""
    freeze_through(policy, len(IMPALA_BLOCK_NAMES))
    freeze_fc(policy)


def _xavier_reinit(module: nn.Module) -> None:
    # Matches train-procgen-pytorch's own common.misc_util.xavier_uniform_init
    # (gain=1.0, zero bias) -- the init ImpalaModel.__init__ applies at construction,
    # so a "fresh" block here starts from the same distribution a from-scratch model would.
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def reinit_block(block: nn.Module) -> None:
    block.apply(_xavier_reinit)


def reinit_fc(policy: nn.Module) -> None:
    nn.init.xavier_uniform_(policy.embedder.fc.weight)
    nn.init.zeros_(policy.embedder.fc.bias)


def reinit_heads(policy: nn.Module) -> None:
    # Small-scale orthogonal init, matching CategoricalPolicy.__init__ exactly
    # (fc_policy gain=0.01 for stable early-training logits; fc_value gain=1.0).
    nn.init.orthogonal_(policy.fc_policy.weight, gain=0.01)
    nn.init.zeros_(policy.fc_policy.bias)
    nn.init.orthogonal_(policy.fc_value.weight, gain=1.0)
    nn.init.zeros_(policy.fc_value.bias)


def prepare_tap(policy: nn.Module, tap: str) -> None:
    """Freeze everything upstream of `tap`; reinitialize + unfreeze everything downstream.

    tap="block2": freeze block1 only; fresh block2, block3, fc, heads.
    tap="block3": freeze block1-3 (pre-FC); fresh fc, heads.
    tap="fc":     freeze block1-3 + fc; fresh heads only.
    """
    if tap not in TAPS:
        raise ValueError(f"unknown tap {tap!r}, expected one of {TAPS}")
    emb = policy.embedder
    if tap == "block2":
        freeze_through(policy, 1)
        reinit_block(emb.block2)
        reinit_block(emb.block3)
        reinit_fc(policy)
    elif tap == "block3":
        freeze_through(policy, 3)
        reinit_fc(policy)
    else:  # "fc"
        freeze_trunk(policy)
    reinit_heads(policy)


def trainable_parameters(policy: nn.Module, tap: str) -> Iterator[nn.Parameter]:
    """Parameters a `tap`-rewired policy should actually optimize (fresh suffix only)."""
    if tap not in TAPS:
        raise ValueError(f"unknown tap {tap!r}, expected one of {TAPS}")
    emb = policy.embedder
    mods: list[nn.Module]
    if tap == "block2":
        mods = [emb.block2, emb.block3, emb.fc, policy.fc_policy, policy.fc_value]
    elif tap == "block3":
        mods = [emb.fc, policy.fc_policy, policy.fc_value]
    else:  # "fc"
        mods = [policy.fc_policy, policy.fc_value]
    for m in mods:
        yield from m.parameters()
