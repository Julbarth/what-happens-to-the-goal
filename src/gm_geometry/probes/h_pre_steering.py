"""Post-ReLU h_pre steering hooks (Flatten(ReLU(block3)), the input to embedder.fc), plus hook verification."""
# ReLU and Flatten after block3 are inline functional calls, so a block3 hook sees pre-ReLU values. A forward
# pre-hook on embedder.fc receives the post-ReLU flattened tensor and can replace what fc consumes.
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def extract_h_pre(policy, obs_t: torch.Tensor) -> torch.Tensor:
    """Unhooked h_pre computed by replaying the embedder's conv tower up to, not including, self.fc."""
    embedder = policy.embedder
    with torch.no_grad():
        x = embedder.block1(obs_t)
        x = embedder.block2(x)
        x = embedder.block3(x)
        x = torch.nn.functional.relu(x)
        x = torch.flatten(x, start_dim=1)
    return x


def verify_h_pre_hook(policy, obs_t: torch.Tensor, atol: float = 1e-5) -> dict:
    """Check that a no-op pre-hook on policy.embedder.fc reproduces the unhooked h_pre exactly and that it is non-negative.

    Steering code should refuse to proceed if ``matches`` is False."""
    h_unhooked = extract_h_pre(policy, obs_t)
    captured: dict[str, torch.Tensor] = {}

    def _capture(_m, args):
        captured["h_pre"] = args[0].detach().clone()

    handle = policy.embedder.fc.register_forward_pre_hook(_capture)
    try:
        with torch.no_grad():
            policy.embedder(obs_t)
    finally:
        handle.remove()

    h_hooked = captured["h_pre"]
    max_abs_diff = float((h_unhooked - h_hooked).abs().max().item())
    all_nonneg = bool((h_unhooked >= -1e-6).all().item())
    return {
        "max_abs_diff": max_abs_diff,
        "matches": max_abs_diff < atol,
        "all_nonneg_confirms_post_relu": all_nonneg,
        "dim": int(h_unhooked.shape[-1]),
    }


def h_at_pre(policy, obs_t: torch.Tensor) -> np.ndarray:
    """h_pre for a single observation, as a flat float64 numpy vector."""
    h = extract_h_pre(policy, obs_t)
    return h.detach().cpu().numpy().reshape(-1).astype(np.float64)


def make_h_pre_push_hook(delta_fn: Callable[[torch.Tensor], torch.Tensor | None]) -> Callable:
    """Forward pre-hook on policy.embedder.fc implementing h_pre' = h_pre + delta.

    ``delta_fn(h_pre_tensor)`` returns a delta of the same shape, or None to skip injection."""
    def _hook(_m, args):
        h_pre = args[0]
        delta = delta_fn(h_pre)
        if delta is None:
            return None
        return (h_pre + delta.to(device=h_pre.device, dtype=h_pre.dtype),)

    return _hook


def push_hook_from_numpy(delta_fn_np: Callable[[np.ndarray], np.ndarray | None]) -> Callable:
    """Wrap a numpy ``delta_fn_np`` as a push hook."""
    def _delta_fn(h_pre: torch.Tensor) -> torch.Tensor | None:
        h_np = h_pre.detach().cpu().numpy().reshape(-1).astype(np.float64)
        delta_np = delta_fn_np(h_np)
        if delta_np is None:
            return None
        return torch.as_tensor(delta_np, dtype=h_pre.dtype).view(h_pre.shape)

    return make_h_pre_push_hook(_delta_fn)
