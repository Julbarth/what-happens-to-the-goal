"""Post-ReLU h_fc steering hooks on the whole policy.embedder, plus hook verification."""
# ImpalaModel.forward applies its final ReLU as a bare functional call after self.fc, so a hook on
# embedder.fc is re-clipped by that ReLU; hooking embedder itself edits exactly what the heads consume.
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def extract_h_fc(policy, obs_t: torch.Tensor) -> torch.Tensor:
    """Unhooked, direct computation of h_fc via a plain forward call. Verification-only."""
    with torch.no_grad():
        return policy.embedder(obs_t)


def verify_h_fc_hook(policy, obs_t: torch.Tensor, atol: float = 1e-5) -> dict:
    """Check that a no-op hook on policy.embedder reproduces the unhooked h_fc exactly and that h_fc is non-negative.

    Steering code should refuse to proceed if ``matches`` is False."""
    h_unhooked = extract_h_fc(policy, obs_t)
    captured: dict[str, torch.Tensor] = {}

    def _capture(_m, _i, out):
        captured["h_fc"] = out.detach().clone()

    handle = policy.embedder.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            policy.embedder(obs_t)
    finally:
        handle.remove()

    h_hooked = captured["h_fc"]
    max_abs_diff = float((h_unhooked - h_hooked).abs().max().item())
    all_nonneg = bool((h_unhooked >= -1e-6).all().item())
    return {
        "max_abs_diff": max_abs_diff,
        "matches": max_abs_diff < atol,
        "all_nonneg_confirms_post_relu": all_nonneg,
        "dim": int(h_unhooked.shape[-1]),
    }


def h_at_fc(policy, obs_t: torch.Tensor) -> np.ndarray:
    """h_fc for a single observation as a flat float64 numpy vector."""
    h = extract_h_fc(policy, obs_t)
    return h.detach().cpu().numpy().reshape(-1).astype(np.float64)


def make_h_fc_push_hook(delta_fn: Callable[[torch.Tensor], torch.Tensor | None]) -> Callable:
    """Forward hook on policy.embedder implementing h_fc' = h_fc + delta, with no extra ReLU.

    ``delta_fn(h_fc_tensor)`` returns a delta of the same shape, or None to leave the output unmodified."""
    def _hook(_m, _i, out):
        delta = delta_fn(out)
        if delta is None:
            return out
        return out + delta.to(device=out.device, dtype=out.dtype)

    return _hook


def push_hook_from_numpy(delta_fn_np: Callable[[np.ndarray], np.ndarray | None]) -> Callable:
    """Wrap a numpy ``delta_fn_np`` as a push hook."""
    def _delta_fn(out: torch.Tensor) -> torch.Tensor | None:
        h_np = out.detach().cpu().numpy().reshape(-1).astype(np.float64)
        delta_np = delta_fn_np(h_np)
        if delta_np is None:
            return None
        return torch.as_tensor(delta_np, dtype=out.dtype).view(out.shape)

    return make_h_fc_push_hook(_delta_fn)
