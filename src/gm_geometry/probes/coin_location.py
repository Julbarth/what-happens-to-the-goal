"""CoinRun location-concept helpers: world-to-feature coordinates, attribution, channel tracking and destination vectors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from gm_geometry.procgen import unwrap_base as _unwrap_base

# Discrete action combos: indices 6/7/8 include RIGHT.
RIGHT_ACTIONS = (6, 7, 8)
LEFT_ACTIONS = (0, 1, 2)
JUMP_ACTIONS = (4, 5)  # up / jump
# Impala block3 spatial size for 64×64 RGB input.
BLOCK3_HW = 8
OBS_HW = 64
# Three stride-2 maxpools: 64 → 8.
PIXELS_PER_FEAT = OBS_HW // BLOCK3_HW
# Under center_agent=True the obs is an agent-centred crop, so the agent's feature coords are the
# observation pixel centre / 8: (OBS_HW - 1) / (2 * PIXELS_PER_FEAT) = 3.9375. The path band uses
# ``x + 0.5 >= ax``, so this starts the band at column 4 (a grid centre of 3.5 would include column 3).
# Recovering the agent cell from coin_cell + (world_agent - world_coin) / 8 is wrong: world units are not
# obs pixels, and that formula collapses agent_feat_x onto the coin column.
AGENT_FEAT_YX_CAMERA: tuple[float, float] = (
    (OBS_HW - 1) / (2.0 * PIXELS_PER_FEAT),
    (OBS_HW - 1) / (2.0 * PIXELS_PER_FEAT),
)


def agent_feat_yx_camera(
    *,
    y: float | None = None,
    x: float | None = None,
) -> tuple[float, float]:
    """Feature-cell (y, x) of the agent under ``center_agent=True`` (the obs centre, 3.9375)."""
    cy = float(y) if y is not None else AGENT_FEAT_YX_CAMERA[0]
    cx = float(x) if x is not None else AGENT_FEAT_YX_CAMERA[1]
    return cy, cx


def agent_feat_yx_from_motion(
    obs_a: np.ndarray,
    obs_b: np.ndarray,
    *,
    thr: float = 18.0,
    feat_hw: int = BLOCK3_HW,
) -> tuple[float, float] | None:
    """Agent feature (y, x) from a short camera-follow step.

    Under ``center_agent=True``, a RIGHT step scrolls the world while the agent
    sprite stays put. High ``|Δ|`` is background. Low ``|Δ|`` is the agent (plus
    flat sky). We take the mass centroid of the *stable* core inside a soft
    window around the image centre, then ÷8.

    ``obs_*`` are CHW float or uint8. Returns None if the stable core is empty.
    """
    a = np.asarray(obs_a, dtype=np.float32)
    b = np.asarray(obs_b, dtype=np.float32)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"expected CHW, got {a.shape} / {b.shape}")
    if float(a.max()) <= 1.5 and float(b.max()) <= 1.5:
        a = a * 255.0
        b = b * 255.0
    d = np.max(np.abs(a - b), axis=0)
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # Soft prior: agent should be near obs centre if camera is following.
    prior = np.exp(
        -0.5
        * (
            ((yy - (h - 1) / 2.0) / (0.28 * h)) ** 2
            + ((xx - (w - 1) / 2.0) / (0.28 * w)) ** 2
        )
    )
    # Stable = low residual. Also require some motion elsewhere so we are not
    # comparing two identical frames.
    if float(d.max()) < float(thr):
        return None
    stab = (d < float(thr)) & (prior > 0.25)
    ys, xs = np.where(stab)
    if ys.size < 8:
        return None
    # Mass-weight by prior so flat sky near the border does not win.
    wt = prior[ys, xs]
    py = float(np.average(ys, weights=wt))
    px = float(np.average(xs, weights=wt))
    fy = float(np.clip(py / float(PIXELS_PER_FEAT), 0.0, feat_hw - 1.0))
    fx = float(np.clip(px / float(PIXELS_PER_FEAT), 0.0, feat_hw - 1.0))
    return fy, fx


def _scalar(info: dict, key: str, default: float = 0.0) -> float:
    v = info.get(key, default)
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0]) if v.size else float(default)
    return float(v)


def labels_from_info(info: dict) -> dict[str, Any] | None:
    """Env-truth poses. Returns None if exports missing."""
    if "coin_x" not in info:
        return None
    coin_x = int(_scalar(info, "coin_x"))
    coin_y = int(_scalar(info, "coin_y"))
    coin_present = int(_scalar(info, "coin_present"))
    inv_x = int(_scalar(info, "inv_coin_x"))
    inv_y = int(_scalar(info, "inv_coin_y"))
    ax = _scalar(info, "agent_x_fp") / 100.0
    ay = _scalar(info, "agent_y_fp") / 100.0
    width = int(_scalar(info, "level_width"))
    if width <= 0:
        return None
    return {
        "coin_x": coin_x,
        "coin_y": coin_y,
        "coin_present": coin_present,
        "inv_coin_x": inv_x,
        "inv_coin_y": inv_y,
        "agent_x": ax,
        "agent_y": ay,
        "level_width": width,
    }


def world_to_feat(
    wx: float,
    wy: float,
    *,
    level_width: int,
    level_height: int = 64,
    feat_hw: int = BLOCK3_HW,
) -> tuple[int, int]:
    """Map a world grid cell to a block3 cell for a full-level render (center_agent=False only).

    Not valid under center_agent=True, where the observation is an agent-centred crop."""
    fx = int(np.clip((wx + 0.5) / float(level_width) * feat_hw, 0, feat_hw - 1))
    # Image y grows downward; world y grows upward.
    fy = int(np.clip((1.0 - (wy + 0.5) / float(level_height)) * feat_hw, 0, feat_hw - 1))
    return fx, fy


def unwrap_base_env(env):
    """Reach the ProcgenEnv / CEnv that exposes set_coin / get_info."""
    return _unwrap_base(env, "set_coin", on_missing="none")


def get_info(env) -> dict:
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "get_info") and callable(cur.get_info):
            infos = cur.get_info()
            if isinstance(infos, (list, tuple)) and infos:
                return infos[0]
            if isinstance(infos, dict):
                return infos
        cur = getattr(cur, "venv", None) or getattr(cur, "env", None)
    raise RuntimeError("could not unwrap get_info")


def set_coin(env, x: int, y: int, present: int) -> int:
    base = unwrap_base_env(env)
    if base is None:
        raise RuntimeError("env does not expose set_coin; rebuild procgenAISC")
    return int(base.set_coin([x], [y], [present])[0])


def set_invisible_goal(env, x: int, y: int, present: int) -> int:
    """Move or remove INVISIBLE_GOAL (end marker). Requires rebuilt procgenAISC."""
    base = unwrap_base_env(env)
    if base is None or not hasattr(base, "set_invisible_goal"):
        raise RuntimeError("env does not expose set_invisible_goal; rebuild procgenAISC")
    return int(base.set_invisible_goal([x], [y], [present])[0])


def obs_nchw_after_set_coin(env) -> np.ndarray:
    """Pull RGB after set_coin and match train wrappers (CHW float in [0,1])."""
    base = unwrap_base_env(env)
    if base is None:
        raise RuntimeError("env does not expose set_coin")
    if not hasattr(base, "observe"):
        raise RuntimeError("base env has no gym3 observe()")
    _rew, obs, _first = base.observe()
    if isinstance(obs, dict):
        rgb = obs["rgb"]
    else:
        rgb = obs
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    # HWC uint8 → CHW float
    if rgb.shape[-1] == 3:
        x = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    else:
        x = rgb.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
    return x


@torch.no_grad()
def forward_dist_value(policy, obs_nchw: np.ndarray, device: str):
    """Return probs [B,A], value [B], hidden [B,H]."""
    x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    b = x.shape[0]
    h = torch.zeros(b, policy.embedder.output_dim, device=device)
    m = torch.zeros(b, device=device)
    dist, value, next_h = policy(x, h, m)
    return dist.probs.detach(), value.detach().view(-1), next_h.detach()


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean total variation between two categorical batches."""
    return float(0.5 * (p - q).abs().sum(dim=-1).mean().item())


def right_mass(probs: torch.Tensor) -> float:
    return float(probs[:, list(RIGHT_ACTIONS)].sum(dim=-1).mean().item())


def spatial_attribution_input(
    policy,
    obs_nchw: np.ndarray,
    device: str,
    *,
    target: str = "right",
) -> np.ndarray:
    """∂target/∂input magnitude map, shape [H,W] (averaged over batch+channels).

    target: 'right' = sum of RIGHT action logits; 'value' = V.
    """
    x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.detach().requires_grad_(True)
    b = x.shape[0]
    h0 = torch.zeros(b, policy.embedder.output_dim, device=device)
    m = torch.zeros(b, device=device)
    dist, value, _ = policy(x, h0, m)
    if target == "right":
        scalar = dist.logits[:, list(RIGHT_ACTIONS)].sum()
    elif target == "value":
        scalar = value.sum()
    else:
        raise ValueError(target)
    policy.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    scalar.backward()
    g = x.grad.detach().abs().mean(dim=1)  # [B,H,W]
    return g.mean(dim=0).cpu().numpy()


def hook_block3_activation(policy, apply_relu: bool = False):
    """Register a block3 hook; returns (handle, store dict with key 'act').

    With apply_relu=False the store holds block3's raw (pre-ReLU) output. With apply_relu=True a forward pre-hook
    on ``embedder.fc`` captures h_pre = ReLU(H^3) reshaped to [1, C, H, W], so gradients through store["act"] are
    true gradients with respect to h_pre."""
    store: dict[str, torch.Tensor] = {}
    if not apply_relu:
        def _hook(_m, _i, out):
            store["act"] = out
        return policy.embedder.block3.register_forward_hook(_hook), store

    shape: dict[str, torch.Size] = {}

    def _shape_hook(_m, _i, out):
        shape["chw"] = out.shape

    def _fc_pre_hook(_m, args):
        act = args[0].reshape(shape["chw"])
        store["act"] = act
        return (act.reshape(act.shape[0], -1),)

    handles = (policy.embedder.block3.register_forward_hook(_shape_hook),
               policy.embedder.fc.register_forward_pre_hook(_fc_pre_hook))
    return _HandlePair(handles), store


class _HandlePair:
    def __init__(self, handles):
        self._handles = handles

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def spatial_attribution_block3(
    policy,
    obs_nchw: np.ndarray,
    device: str,
    *,
    target: str = "right",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (attribution [H,W], activation mean-over-channels [H,W])."""
    handle, store = hook_block3_activation(policy)
    try:
        x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b = x.shape[0]
        h0 = torch.zeros(b, policy.embedder.output_dim, device=device)
        m = torch.zeros(b, device=device)
        dist, value, _ = policy(x, h0, m)
        act = store["act"]  # [B,C,H,W]
        act.retain_grad()
        if target == "right":
            scalar = dist.logits[:, list(RIGHT_ACTIONS)].sum()
        elif target == "value":
            scalar = value.sum()
        else:
            raise ValueError(target)
        policy.zero_grad(set_to_none=True)
        scalar.backward()
        g = act.grad.detach().abs().mean(dim=1)  # [B,H,W]
        a = act.detach().abs().mean(dim=1)
        return g.mean(dim=0).cpu().numpy(), a.mean(dim=0).cpu().numpy()
    finally:
        handle.remove()


def cell_mass(attr: np.ndarray, fx: int, fy: int, radius: int = 0) -> float:
    """Sum attribution in a neighbourhood around (fx, fy)."""
    h, w = attr.shape
    total = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = fy + dy, fx + dx
            if 0 <= yy < h and 0 <= xx < w:
                total += float(attr[yy, xx])
    return total


def destination_masses(
    attr: np.ndarray,
    labels: dict,
    *,
    radius: int = 1,
) -> dict[str, float]:
    """Attribution mass on coin cell vs end-wall cell (normalised by total)."""
    w = int(labels["level_width"])
    total = float(attr.sum()) + 1e-8
    out = {"total": total}
    if labels.get("coin_present", 0) == 1:
        cfx, cfy = world_to_feat(labels["coin_x"], labels["coin_y"], level_width=w)
        out["coin"] = cell_mass(attr, cfx, cfy, radius) / total
        out["coin_cell"] = (cfx, cfy)
    else:
        out["coin"] = 0.0
        out["coin_cell"] = None
    ifx, ify = world_to_feat(labels["inv_coin_x"], labels["inv_coin_y"], level_width=w)
    out["end"] = cell_mass(attr, ifx, ify, radius) / total
    out["end_cell"] = (ifx, ify)
    out["coin_minus_end"] = out["coin"] - out["end"]
    return out


@torch.no_grad()
def block3_channels(policy, obs_nchw: np.ndarray, device: str) -> np.ndarray:
    """Return block3 activations [B,C,H,W]."""
    handle, store = hook_block3_activation(policy)
    try:
        _ = forward_dist_value(policy, obs_nchw, device)
        return store["act"].detach().cpu().numpy()
    finally:
        handle.remove()


def channel_argmax_cells(act_bchw: np.ndarray) -> np.ndarray:
    """Per-channel spatial argmax as (y,x); shape [B,C,2]."""
    b, c, h, w = act_bchw.shape
    flat = act_bchw.reshape(b, c, h * w)
    idx = flat.argmax(axis=-1)
    ys = idx // w
    xs = idx % w
    return np.stack([ys, xs], axis=-1)


def channel_track_scores(
    act_bchw: np.ndarray,
    target_cells: np.ndarray,
) -> np.ndarray:
    """Correlation-free hit rate: fraction of frames where channel argmax == target.

    target_cells: [B,2] as (y,x). Returns [C] scores in [0,1].
    """
    arg = channel_argmax_cells(act_bchw)  # [B,C,2]
    b, c, _ = arg.shape
    hits = (arg[:, :, 0] == target_cells[:, None, 0]) & (
        arg[:, :, 1] == target_cells[:, None, 1]
    )
    return hits.mean(axis=0)


def pixel_diff_feat_cell(
    obs_present: np.ndarray,
    obs_absent: np.ndarray,
    *,
    thr: float = 0.05,
    feat_hw: int = BLOCK3_HW,
) -> tuple[int, int] | None:
    """Map present−absent pixel mass to a block3 cell (y, x).

    Uses the observation the network actually sees (agent-centred crop). No
    world_to_feat. Returns None if the visibility gate fails (max diff < thr).
    """
    a = obs_present.astype(np.float32)
    b = obs_absent.astype(np.float32)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"expected CHW, got {a.shape} / {b.shape}")
    diff = np.max(np.abs(a - b), axis=0)  # HW
    if float(diff.max()) < thr:
        return None
    ys, xs = np.where(diff >= thr)
    if ys.size == 0:
        return None
    # Centroid of the coin mask in pixel space, then ÷8.
    cy = float(ys.mean())
    cx = float(xs.mean())
    fy = int(np.clip(cy / float(PIXELS_PER_FEAT), 0, feat_hw - 1))
    fx = int(np.clip(cx / float(PIXELS_PER_FEAT), 0, feat_hw - 1))
    return fy, fx


@torch.no_grad()
def channel_action_weight_maps(
    policy, device: str, *, signed: bool = False
) -> np.ndarray:
    """Per-action spatial weight maps from block3 → policy logits.

    Linear path: for each action a, W_fc^T @ W_π[a] reshaped to [C,H,W].
    Default takes abs (magnitude). ``signed=True`` keeps approach/avoid sign.
    Returns [n_actions, C, H, W]. Not exact (ReLUs); ranks loci.
    """
    _ = device
    W_fc = policy.embedder.fc.weight.detach()  # [256, C*H*W]
    W_pi = policy.fc_policy.weight.detach()  # [A, 256]
    n_act = int(W_pi.shape[0])
    c = 32
    out = np.zeros((n_act, c, BLOCK3_HW, BLOCK3_HW), dtype=np.float64)
    for a in range(n_act):
        contrib = W_fc.T @ W_pi[a]  # [C*H*W]
        if not signed:
            contrib = contrib.abs()
        out[a] = contrib.view(c, BLOCK3_HW, BLOCK3_HW).cpu().numpy()
    return out


@torch.no_grad()
def channel_to_action_wire(policy, device: str) -> np.ndarray:
    """Per-action wire averaged over all spatial cells → [n_actions, C].

    Blind for place-selective channels (averages 64 cells). Prefer
    ``activation_weighted_wire_at_cell`` for coin / object use.
    """
    maps = channel_action_weight_maps(policy, device)  # [A,C,H,W]
    return maps.mean(axis=(2, 3))


def activation_weighted_wire_at_cell(
    weight_maps: np.ndarray,
    acts_bchw: np.ndarray,
    cells_yx: np.ndarray,
    *,
    action_idxs: tuple[int, ...] | list[int],
    per_frame_actions: np.ndarray | None = None,
) -> np.ndarray:
    """Activation×weight wire at a named feature cell, per channel.

    For each frame b with coin cell (y,x): sum over actions in ``action_idxs``
    of relu(a)[b,c,y,x] * W[a,c,y,x], then mean over frames → [C].

    Pass abs weight maps for magnitude wire; signed maps for approach/avoid.
    ReLU on activations matches the path into ``fc``.

    If ``per_frame_actions`` is provided ([B] action indices), use that single
    action row per frame instead of summing ``action_idxs`` (maze toward-object).
    """
    maps = np.asarray(weight_maps, dtype=np.float64)  # [A,C,H,W]
    acts = np.maximum(np.asarray(acts_bchw, dtype=np.float64), 0.0)  # [B,C,H,W]
    cells = np.asarray(cells_yx, dtype=np.int64)  # [B,2] (y,x)
    b, c, _h, _w = acts.shape
    out = np.zeros(c, dtype=np.float64)
    if per_frame_actions is not None:
        pa = np.asarray(per_frame_actions, dtype=np.int64).ravel()
        for i in range(b):
            a = int(pa[i])
            if a < 0 or a >= maps.shape[0]:
                continue
            y, x = int(cells[i, 0]), int(cells[i, 1])
            out += acts[i, :, y, x] * maps[a, :, y, x]
        return out / max(b, 1)
    valid = [a for a in action_idxs if a < maps.shape[0]]
    if not valid:
        return np.zeros(c, dtype=np.float64)
    w_cell = maps[valid].sum(axis=0)  # [C,H,W]
    for i in range(b):
        y, x = int(cells[i, 0]), int(cells[i, 1])
        out += acts[i, :, y, x] * w_cell[:, y, x]
    return out / max(b, 1)


@torch.no_grad()
def channel_to_right_weight(policy, device: str) -> np.ndarray:
    """Approximate how strongly each flattened block3 unit feeds RIGHT logits.

    Uses the linear path block3→fc→policy: |W_fc^T @ W_π_right| reshaped to [C,H,W],
    then mean over spatial → [C]. Not exact (ReLUs), but ranks channels.
    """
    wire = channel_to_action_wire(policy, device)  # [A,C]
    return wire[list(RIGHT_ACTIONS)].sum(axis=0)


def matched_empty_control_cell(
    coin_yx: tuple[int, int] | list[int],
    *,
    feat_hw: int = BLOCK3_HW,
) -> tuple[int, int]:
    """Control cell at the same radius from screen centre, opposite side.

    Empty-cell negative control for the 2×2 wire design. If the reflection
    lands on the coin cell, walk the circle of equal radius for an alternate.
    """
    iy, ix = int(coin_yx[0]), int(coin_yx[1])
    cy = (feat_hw - 1) / 2.0
    cx = (feat_hw - 1) / 2.0
    jy = int(np.clip(round(2 * cy - iy), 0, feat_hw - 1))
    jx = int(np.clip(round(2 * cx - ix), 0, feat_hw - 1))
    if (jy, jx) != (iy, ix):
        return jy, jx
    # Degenerate (coin at centre): pick farthest cell from coin.
    best = (0, 0)
    best_d = -1.0
    for y in range(feat_hw):
        for x in range(feat_hw):
            if (y, x) == (iy, ix):
                continue
            d = (y - iy) ** 2 + (x - ix) ** 2
            if d > best_d:
                best_d = float(d)
                best = (y, x)
    return best


def block3_logit_gradients(
    policy,
    obs_nchw: np.ndarray,
    device: str,
    apply_relu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Autograd ∂logit_a / ∂act_block3 for every action, returned as (grads [A, C, H, W], acts [C, H, W]).

    ``apply_relu`` is as in `hook_block3_activation`: False uses raw H^3, True uses h_pre = ReLU(H^3)."""
    handle, store = hook_block3_activation(policy, apply_relu=apply_relu)
    try:
        x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise ValueError("block3_logit_gradients expects batch size 1")
        b = 1
        h0 = torch.zeros(b, policy.embedder.output_dim, device=device)
        m = torch.zeros(b, device=device)
        dist, _value, _nh = policy(x, h0, m)
        act = store["act"]  # [1,C,H,W]
        logits = dist.logits  # [1,A]
        n_act = int(logits.shape[1])
        grads = []
        for a in range(n_act):
            g_a = torch.autograd.grad(
                logits[0, a], act, retain_graph=(a < n_act - 1), allow_unused=False
            )[0]
            grads.append(g_a.detach()[0].cpu().numpy())
        return np.stack(grads, axis=0), act.detach()[0].cpu().numpy()
    finally:
        handle.remove()


def summarize_2x2(
    maps_c: np.ndarray,
    *,
    tracker_idxs: list[int],
    coin_yx: tuple[int, int],
    ctrl_yx: tuple[int, int],
) -> dict[str, float]:
    """Selectivity interaction on a [C,H,W] map (sensitivity or |contribution|).

    A = mean over trackers at coin cell
    B = mean over trackers at control cell
    C = mean over other channels at coin cell
    D = mean over other channels at control cell
    interaction = A − B − C + D
    """
    maps = np.asarray(maps_c, dtype=np.float64)
    c = maps.shape[0]
    track = list(tracker_idxs)
    other = [i for i in range(c) if i not in set(track)]
    iy, ix = int(coin_yx[0]), int(coin_yx[1])
    jy, jx = int(ctrl_yx[0]), int(ctrl_yx[1])
    a = float(maps[track, iy, ix].mean()) if track else float("nan")
    b = float(maps[track, jy, jx].mean()) if track else float("nan")
    cc = float(maps[other, iy, ix].mean()) if other else float("nan")
    d = float(maps[other, jy, jx].mean()) if other else float("nan")
    return {
        "A_track_coin": a,
        "B_track_ctrl": b,
        "C_other_coin": cc,
        "D_other_ctrl": d,
        "interaction": a - b - cc + d,
        "A_minus_C": a - cc,
        "A_minus_B": a - b,
    }


def destination_vector_fc(
    acts_present: np.ndarray,
    acts_absent: np.ndarray,
) -> np.ndarray:
    """û = mean(h_present - h_absent), unit-normalised. Shape [H]."""
    d = acts_present.mean(axis=0) - acts_absent.mean(axis=0)
    n = float(np.linalg.norm(d) + 1e-8)
    return (d / n).astype(np.float64)


def reconstruct_competing_target_h_fc(env, policy, device: str):
    """Competing-target direction d_comp at the post-ReLU h_fc tap; returns (d_comp, ok).

    Both the goal-consistent (behind) and shortcut-consistent (ahead) candidates are recomputed from the
    current live agent position on every call."""
    from gm_geometry.probes.h_fc_steering import h_at_fc

    state_now = unwrap_base_env(env).get_state()
    lab = labels_from_info(get_info(env))
    if lab is None:
        unwrap_base_env(env).set_state(state_now)
        return None, False

    goal_xy = find_floor_cell(env, prefer_left_of_agent=True, min_dx=1, vertical_first=True)
    sc_xy = find_floor_cell(env, prefer_left_of_agent=False, min_dx=1, vertical_first=True)
    if goal_xy is None or sc_xy is None:
        unwrap_base_env(env).set_state(state_now)
        return None, False

    if set_coin(env, goal_xy[0], goal_xy[1], 1) != 0:
        unwrap_base_env(env).set_state(state_now)
        return None, False
    obs_star = obs_nchw_after_set_coin(env)
    if obs_star.ndim == 4:
        obs_star = obs_star[0]
    unwrap_base_env(env).set_state(state_now)

    if set_coin(env, sc_xy[0], sc_xy[1], 1) != 0:
        unwrap_base_env(env).set_state(state_now)
        return None, False
    obs_sc = obs_nchw_after_set_coin(env)
    if obs_sc.ndim == 4:
        obs_sc = obs_sc[0]
    unwrap_base_env(env).set_state(state_now)

    if float(np.max(np.abs(obs_star - obs_sc))) < 1e-4:
        return None, False

    import torch as _torch

    def _to_obs_t(obs_nchw):
        t = _torch.as_tensor(obs_nchw, dtype=_torch.float32, device=device)
        return t.unsqueeze(0) if t.ndim == 3 else t

    h_star = h_at_fc(policy, _to_obs_t(obs_star))
    h_sc = h_at_fc(policy, _to_obs_t(obs_sc))
    d_comp = (h_star - h_sc).astype(np.float64)
    return d_comp, True


def reconstruct_competing_target_h_pre(env, policy, device: str):
    """h_pre (2048-dim) analogue of ``reconstruct_competing_target_h_fc``; returns (d_comp, ok)."""
    from gm_geometry.probes.h_pre_steering import h_at_pre

    state_now = unwrap_base_env(env).get_state()
    lab = labels_from_info(get_info(env))
    if lab is None:
        unwrap_base_env(env).set_state(state_now)
        return None, False

    goal_xy = find_floor_cell(env, prefer_left_of_agent=True, min_dx=1, vertical_first=True)
    sc_xy = find_floor_cell(env, prefer_left_of_agent=False, min_dx=1, vertical_first=True)
    if goal_xy is None or sc_xy is None:
        unwrap_base_env(env).set_state(state_now)
        return None, False

    if set_coin(env, goal_xy[0], goal_xy[1], 1) != 0:
        unwrap_base_env(env).set_state(state_now)
        return None, False
    obs_star = obs_nchw_after_set_coin(env)
    if obs_star.ndim == 4:
        obs_star = obs_star[0]
    unwrap_base_env(env).set_state(state_now)

    if set_coin(env, sc_xy[0], sc_xy[1], 1) != 0:
        unwrap_base_env(env).set_state(state_now)
        return None, False
    obs_sc = obs_nchw_after_set_coin(env)
    if obs_sc.ndim == 4:
        obs_sc = obs_sc[0]
    unwrap_base_env(env).set_state(state_now)

    if float(np.max(np.abs(obs_star - obs_sc))) < 1e-4:
        return None, False

    import torch as _torch

    def _to_obs_t(obs_nchw):
        t = _torch.as_tensor(obs_nchw, dtype=_torch.float32, device=device)
        return t.unsqueeze(0) if t.ndim == 3 else t

    h_star = h_at_pre(policy, _to_obs_t(obs_star))
    h_sc = h_at_pre(policy, _to_obs_t(obs_sc))
    d_comp = (h_star - h_sc).astype(np.float64)
    return d_comp, True


def find_floor_cell(
    env,
    *,
    prefer_left_of_agent: bool | None = None,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
    min_dx: int = 3,
    vertical_first: bool = False,
) -> tuple[int, int] | None:
    """Pick a walkable SPACE cell for coin placement using current info + scan.

    Heuristic: cells whose below neighbour is a wall are preferred; we cannot
    read the grid from Python, so we try candidate (x, agent_y) via set_coin
    and keep the first that returns 0, then restore.

    ``vertical_first=True``: try elevated/depressed y before same-row ``ay``.
    Needed for decision windows (same-row ahead ⇒ $a_{coin}$=RIGHT almost always).
    """
    info = get_info(env)
    lab = labels_from_info(info)
    if lab is None:
        return None
    # Optional anchors: place relative to spawn (or another pose), not the
    # post-warm-up agent. Left-of-agent after hold-right often has no floor.
    ax = int(anchor_x) if anchor_x is not None else int(lab["agent_x"])
    ay = max(1, int(anchor_y) if anchor_y is not None else int(lab["agent_y"]))
    width = lab["level_width"]
    orig_x, orig_y, orig_p = lab["coin_x"], lab["coin_y"], lab["coin_present"]

    xs: list[int] = []
    if prefer_left_of_agent is True:
        # Inclusive of x=1 (stop exclusive → 0). Spawn-adjacent left needs this.
        xs = list(range(ax - min_dx, 0, -1))
    elif prefer_left_of_agent is False:
        xs = list(range(ax + min_dx, width - 2))
    else:
        xs = list(range(2, width - 2))

    if vertical_first:
        # Prefer |Δy|≥2 so Non-GM π often leaves RIGHT (decision-window yield).
        y_list = [
            ay + d
            for d in (2, -2, 3, -3, 4, -4, 1, -1, 0)
            if 1 <= ay + d < 63
        ]
        # Dedup while preserving order.
        seen_y: set[int] = set()
        ys: list[int] = []
        for y in y_list:
            if y not in seen_y:
                seen_y.add(y)
                ys.append(y)
        for y in ys:
            for x in xs:
                rc = set_coin(env, x, y, 1)
                if rc == 0:
                    if orig_p:
                        set_coin(env, orig_x, orig_y, 1)
                    else:
                        set_coin(env, 0, 0, 0)
                    return x, y
    else:
        for x in xs:
            for y in (ay, ay - 1, ay + 1, 2, 3, 4, 5):
                if y < 1 or y >= 63:
                    continue
                rc = set_coin(env, x, y, 1)
                if rc == 0:
                    # restore original
                    if orig_p:
                        set_coin(env, orig_x, orig_y, 1)
                    else:
                        set_coin(env, 0, 0, 0)
                    return x, y
    if orig_p:
        set_coin(env, orig_x, orig_y, 1)
    else:
        set_coin(env, 0, 0, 0)
    return None
