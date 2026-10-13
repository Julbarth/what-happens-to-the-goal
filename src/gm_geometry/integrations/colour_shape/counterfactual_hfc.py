"""Attribute Maze counterfactual-competition steering helpers with all hooks at the post-ReLU h_fc tap."""
# Hooks sit on the whole embedder: a hook on .fc alone is re-clipped by the model's trailing functional ReLU.
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as tnf

from gm_geometry.probes.h_fc_steering import make_h_fc_push_hook, verify_h_fc_hook


def fc_logit_jacobian_and_act_hfc(policy, obs_nchw: np.ndarray, device: str):
    """Return d(logit_a)/d(h_fc), action probs and h_fc, with shapes [A, D], [A], [D]."""
    store: dict[str, torch.Tensor] = {}

    def _hook(_m, _i, out):
        store["h"] = out
        return out

    handle = policy.embedder.register_forward_hook(_hook)
    try:
        x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        m = torch.zeros(1, device=device)
        dist, _value, _nh = policy(x, h0, m)
        h = store["h"]
        logits = dist.logits
        n_act = int(logits.shape[1])
        grads = []
        for a in range(n_act):
            g_a = torch.autograd.grad(logits[0, a], h, retain_graph=(a < n_act - 1), allow_unused=False)[0]
            grads.append(g_a.detach()[0].cpu().numpy())
        j = np.stack(grads, axis=0)
        with torch.no_grad():
            probs = tnf.softmax(logits, dim=-1)[0].cpu().numpy()
        return j, probs, h.detach()[0].cpu().numpy()
    finally:
        handle.remove()


def pushed_probs_hfc(policy, obs_nchw: np.ndarray, device: str, delta_vec: np.ndarray) -> np.ndarray:
    """Single forward pass with h_fc' = h_fc + delta_vec injected via the whole-embedder hook."""
    delta_t = torch.as_tensor(delta_vec, dtype=torch.float32, device=device)
    handle = policy.embedder.register_forward_hook(make_h_fc_push_hook(lambda out: delta_t))
    try:
        x = torch.as_tensor(obs_nchw, dtype=torch.float32, device=device).unsqueeze(0)
        h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
        m = torch.zeros(1, device=device)
        with torch.no_grad():
            dist, _v, _nh = policy(x, h0, m)
            return dist.probs[0].cpu().numpy()
    finally:
        handle.remove()


def rollout_with_dcomp_hfc(
    policy, level, distribution_mode, use_backgrounds, device,
    ids_line, ids_gem, beta, sign, max_steps, replay_stride,
    expect_state_bytes, *, base_module,
):
    """Greedy rollout steered along the counterfactual competition direction at h_fc.

    ``base_module`` supplies the tap-agnostic env mechanics (make_env, state_bytes, to_nchw)."""
    def _shape_env_factory(lvl):
        return base_module.make_env(base_module.SHAPE_EDIT_OBJ, base_module.OBJ_GEM, lvl, rand_seed=0,
                                     distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)

    def _colour_env_factory(lvl):
        return base_module.make_env(base_module.OBJ_LINE, base_module.COLOUR_EDIT_OBJ, lvl, rand_seed=0,
                                     distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)

    def _replay(env_factory, action_history):
        env = env_factory(level)
        obs = env.reset()
        done_early = False
        try:
            for a in action_history:
                obs, _rew, done, _info = env.step(np.array([a]))
                if bool(np.asarray(done).reshape(-1)[0]):
                    done_early = True
                    break
        finally:
            env.close()
        return obs, done_early

    env = base_module.make_env(base_module.OBJ_LINE, base_module.OBJ_GEM, level, rand_seed=0,
                                distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    obs = env.reset()
    sb = base_module.state_bytes(env)
    if sb != expect_state_bytes:
        env.close()
        raise RuntimeError(f"level {level}: base replay state_bytes mismatch")

    action_history: list[int] = []
    d_comp_cached = None
    outcome, n_steps = "timeout", max_steps
    handle = None
    try:
        for t in range(max_steps):
            x = base_module.to_nchw(obs)
            if handle is not None:
                handle.remove()
                handle = None

            need_refresh = (d_comp_cached is None) or (t % replay_stride == 0)
            if need_refresh:
                obs_shape_t, done_shape = _replay(_shape_env_factory, action_history)
                obs_colour_t, done_colour = _replay(_colour_env_factory, action_history)
                if not (done_shape or done_colour):
                    _, _, h_shape_t = fc_logit_jacobian_and_act_hfc(policy, base_module.to_nchw(obs_shape_t), device)
                    _, _, h_colour_t = fc_logit_jacobian_and_act_hfc(policy, base_module.to_nchw(obs_colour_t), device)
                    d_comp_cached = (h_colour_t - h_shape_t).astype(np.float64)

            if d_comp_cached is not None:
                delta_vec = sign * beta * d_comp_cached
                delta_t = torch.as_tensor(delta_vec, dtype=torch.float32, device=device)
                handle = policy.embedder.register_forward_hook(make_h_fc_push_hook(lambda out, d=delta_t: d))

            x_t = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
            h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
            m = torch.zeros(1, device=device)
            with torch.no_grad():
                dist, _v, _nh = policy(x_t, h0, m)
                a = int(dist.probs.argmax(-1).item())

            action_history.append(a)
            obs, rew, done, _info = env.step(np.array([a]))
            r = float(np.asarray(rew).reshape(-1)[0])
            d = bool(np.asarray(done).reshape(-1)[0])
            if d:
                outcome = "line" if r > 5.0 else ("gem" if r < -5.0 else "other_terminal")
                n_steps = t + 1
                break
    finally:
        if handle is not None:
            handle.remove()
        env.close()
    return outcome, n_steps


def rollout_with_fixed_push_hfc(
    policy, level, distribution_mode, use_backgrounds, device,
    delta_vec, max_steps, expect_state_bytes, *, base_module,
):
    """Greedy rollout with a fixed delta added to h_fc at every step."""
    env = base_module.make_env(base_module.OBJ_LINE, base_module.OBJ_GEM, level, rand_seed=0,
                                distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    obs = env.reset()
    if base_module.state_bytes(env) != expect_state_bytes:
        env.close()
        raise RuntimeError(f"level {level}: base replay state_bytes mismatch")
    delta_t = torch.as_tensor(delta_vec, dtype=torch.float32, device=device)
    handle = policy.embedder.register_forward_hook(make_h_fc_push_hook(lambda out: delta_t))
    outcome, n_steps = "timeout", max_steps
    try:
        for t in range(max_steps):
            x = torch.as_tensor(base_module.to_nchw(obs), dtype=torch.float32, device=device).unsqueeze(0)
            h0 = torch.zeros(1, policy.embedder.output_dim, device=device)
            m = torch.zeros(1, device=device)
            with torch.no_grad():
                dist, _v, _nh = policy(x, h0, m)
                a = int(dist.probs.argmax(-1).item())
            obs, rew, done, _info = env.step(np.array([a]))
            r = float(np.asarray(rew).reshape(-1)[0])
            d = bool(np.asarray(done).reshape(-1)[0])
            if d:
                outcome = "line" if r > 5.0 else ("gem" if r < -5.0 else "other_terminal")
                n_steps = t + 1
                break
    finally:
        handle.remove()
        env.close()
    return outcome, n_steps


def compute_d_comp0_hfc(policy, res, distribution_mode, use_backgrounds, device, *, base_module) -> np.ndarray | None:
    level = res["level"]
    obs_base = res["obs0"]
    env_shape = base_module.make_env(base_module.SHAPE_EDIT_OBJ, base_module.OBJ_GEM, level, rand_seed=0,
                                      distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    env_colour = base_module.make_env(base_module.OBJ_LINE, base_module.COLOUR_EDIT_OBJ, level, rand_seed=0,
                                       distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
    try:
        obs_shape = env_shape.reset()
        obs_colour = env_colour.reset()
        sb_shape = base_module.state_bytes(env_shape)
        sb_colour = base_module.state_bytes(env_colour)
    finally:
        env_shape.close()
        env_colour.close()
    if sb_shape != res["state_bytes"] or sb_colour != res["state_bytes"]:
        return None
    _, _, h_shape = fc_logit_jacobian_and_act_hfc(policy, base_module.to_nchw(obs_shape), device)
    _, _, h_colour = fc_logit_jacobian_and_act_hfc(policy, base_module.to_nchw(obs_colour), device)
    return (h_colour - h_shape).astype(np.float64)


def assert_hfc_hook_verified(policy, sample_obs_nchw: np.ndarray, device: str) -> dict:
    """Call once per policy load, before trusting any steering number from this module."""
    x = torch.as_tensor(sample_obs_nchw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    report = verify_h_fc_hook(policy, x)
    if not report["matches"] or not report["all_nonneg_confirms_post_relu"]:
        raise SystemExit(f"h_fc hook verification FAILED, refusing to trust steering numbers: {report}")
    return report


class EditWorlds:
    """Env-mechanics proxy that maps the 'shape' and 'colour' counterfactual worlds to one edit family.

    The helpers build the shape world as make_env(SHAPE_EDIT_OBJ, OBJ_GEM) and the colour world as
    make_env(OBJ_LINE, COLOUR_EDIT_OBJ); sentinel object names route each to the family's edited objects,
    so d_comp = h(colour edit) - h(shape edit), which for "swap" points toward the line candidate."""

    _SHAPE, _COLOUR = "__shape_edit_world__", "__colour_edit_world__"
    FAMILIES = {"swap": ("shape_swap", "colour_swap"), "ablation": ("shape_ablate", "colour_ablate")}

    def __init__(self, base_module, family: str):
        from gm_geometry.probes import ras_counterfactuals as rc
        if family not in self.FAMILIES:
            raise ValueError(f"edit family {family!r} not in {sorted(self.FAMILIES)}")
        if (base_module.OBJ_LINE, base_module.OBJ_GEM) != rc.edit_objects("base"):
            raise ValueError("base module objects disagree with ras_counterfactuals base objects")
        self._base, self.family = base_module, family
        shape, colour = self.FAMILIES[family]
        self._objs = {self._SHAPE: rc.edit_objects(shape), self._COLOUR: rc.edit_objects(colour)}
        self.edit_names = {"minus": shape, "plus": colour}
        self.SHAPE_EDIT_OBJ, self.COLOUR_EDIT_OBJ = self._SHAPE, self._COLOUR

    def __getattr__(self, name):
        return getattr(self._base, name)

    def make_env(self, obj1, obj2, *args, **kwargs):
        if obj1 == self._SHAPE:
            obj1, obj2 = self._objs[self._SHAPE]
        elif obj2 == self._COLOUR:
            obj1, obj2 = self._objs[self._COLOUR]
        return self._base.make_env(obj1, obj2, *args, **kwargs)
