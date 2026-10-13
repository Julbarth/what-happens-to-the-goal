"""Layer-local feature hooks for Coin head rewire (block3 vs fc)."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coin_followup_common import FC_DIM, setup_coin_paths  # noqa: E402

from gm_geometry.probes.coin_location import BLOCK3_HW  # noqa: E402


class Block3Embedder(nn.Module):
    """Impala trunk through block3 + ReLU + flatten (skips embedder.fc).

    Deep-copies block weights so the rewire policy does not share modules with the
    source checkpoint policy.
    """

    def __init__(self, impala_model: nn.Module):
        super().__init__()
        self.block1 = copy.deepcopy(impala_model.block1)
        self.block2 = copy.deepcopy(impala_model.block2)
        self.block3 = copy.deepcopy(impala_model.block3)
        out_ch = int(self.block3.conv.out_channels)
        self.output_dim = out_ch * BLOCK3_HW * BLOCK3_HW

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(x)
        return x.flatten(1)


class Block3PCAEmbedder(nn.Module):
    """block3 flat → fixed PCA projection (matched fc dim, no extra trainable trunk)."""

    def __init__(self, block3_embedder: Block3Embedder, pca: dict[str, np.ndarray]):
        super().__init__()
        self.block3_emb = block3_embedder
        mean = torch.as_tensor(pca["mean"], dtype=torch.float32)
        comp = torch.as_tensor(pca["components"], dtype=torch.float32)
        self.register_buffer("pca_mean", mean)
        self.register_buffer("pca_components", comp)
        self.output_dim = int(comp.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.block3_emb(x)
        return (flat - self.pca_mean) @ self.pca_components.T


@torch.no_grad()
def block3_flat_from_policy(policy: nn.Module, obs_np, device: str) -> np.ndarray:
    """Flattened block3 activations [B, C*H*W] from a policy forward."""
    emb = policy.embedder
    if isinstance(emb, Block3PCAEmbedder):
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        return emb.block3_emb(x).detach().cpu().numpy().astype(np.float32)
    if isinstance(emb, Block3Embedder):
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        return emb(x).detach().cpu().numpy().astype(np.float32)
    setup_coin_paths()
    from gm_geometry.probes.coin_location import block3_channels  # noqa: E402

    # block3_channels returns pre-ReLU activations; the rollout forward applies ReLU before flatten.
    act = np.maximum(block3_channels(policy, obs_np, device), 0.0)
    b, _c, _h, _w = act.shape
    return act.reshape(b, -1).astype(np.float32)


def block3_head_features_from_policy(policy: nn.Module, obs_np, device: str) -> np.ndarray:
    """Features fed to fc_policy (PCA block3 or raw block3 flat)."""
    emb = policy.embedder
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        if isinstance(emb, Block3PCAEmbedder):
            return emb(x).detach().cpu().numpy().astype(np.float32)
        if isinstance(emb, Block3Embedder):
            return emb(x).detach().cpu().numpy().astype(np.float32)
    return block3_flat_from_policy(policy, obs_np, device)


def make_block3_policy(
    source_policy: nn.Module,
    device: str,
    *,
    pca: dict[str, np.ndarray] | None = None,
) -> nn.Module:
    """GM trunk through block3 + fresh heads. Optional PCA→match_dim projection."""
    from common.policy import CategoricalPolicy  # noqa: E402
    from train_frozen_trunk_head_rewire import freeze_trunk, reinit_heads  # noqa: E402

    b3_emb = Block3Embedder(source_policy.embedder)
    if pca is not None:
        embedder: nn.Module = Block3PCAEmbedder(b3_emb, pca)
    else:
        embedder = b3_emb
    n_act = int(source_policy.fc_policy.out_features)
    policy = CategoricalPolicy(embedder, False, n_act)
    reinit_heads(policy)
    freeze_trunk(policy)
    policy.to(device)
    return policy


def make_block3_policy_matched(
    source_policy: nn.Module,
    features: np.ndarray,
    device: str,
    *,
    match_dim: int = FC_DIM,
    seed: int = 0,
) -> tuple[nn.Module, dict[str, np.ndarray]]:
    """Block3 policy with PCA trunk output dim = match_dim (capacity-matched to fc)."""
    from coin_followup_common import fit_pca  # noqa: E402

    pca = fit_pca(features, match_dim, seed=seed)
    policy = make_block3_policy(source_policy, device, pca=pca)
    return policy, pca
