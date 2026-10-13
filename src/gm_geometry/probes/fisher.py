"""Dual-layer (block3 + fc) Fisher-information quadratic-form scoring and the Attribute Maze duel harvest harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from procgen import ProcgenEnv

from gm_geometry.probes.fisher_sensitivity import (
    block3_fisher_pack,
    fc_fisher_and_act,
    fisher_from_logit_jacobian,
    kl_categorical,
    quadratic_form,
    sensitivity_along,
)

# ---------------------------------------------------------------------------
# Shared dual-layer (block3 + fc) Fisher scalar scoring.
# ---------------------------------------------------------------------------

SCALAR_KEYS = [
    "Q_b3",
    "Q_fc",
    "S_b3",
    "S_fc",
    "trFd_b3",
    "trFd_fc",
    "S_over_trFd_b3",
    "S_over_trFd_fc",
    "dh_b3",
    "dh_fc",
    "h_b3",
    "h_fc",
    "rel_b3",
    "rel_fc",
    "T",
    "two_kl",
    "entropy",
    "max_prob",
]

# Attribute Maze summary key set: omits h_b3/h_fc, and `summarize` adds pixel_l2 distribution stats.
BASE_SUMMARY_KEYS = (
    "Q_b3", "Q_fc", "S_b3", "S_fc", "trFd_b3", "trFd_fc",
    "S_over_trFd_b3", "S_over_trFd_fc", "dh_b3", "dh_fc",
    "rel_b3", "rel_fc", "T", "two_kl", "entropy", "max_prob",
)


def entropy_of(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def dual_layer_scalars(
    policy,
    obs_a: np.ndarray,
    obs_g: np.ndarray,
    device: str,
    *,
    d_b3: int = 2048,
    d_fc: int = 256,
    include_block3_raw: bool = False,
) -> dict:
    """Present (obs_a) vs absent (obs_g) Q/S/trFd/dh/T/entropy at block3 and fc.

    ``include_block3_raw=True`` also returns the block3 grads/probs/present-map, so callers can build a
    shuffle-null map without a second block3_fisher_pack call."""
    grads_a, probs_a, acts_a, m_present = block3_fisher_pack(policy, obs_a, device)
    _g, _p, acts_g, _ = block3_fisher_pack(policy, obs_g, device)
    f_fc, p_a, h_a = fc_fisher_and_act(policy, obs_a, device)
    _f, p_g, h_g = fc_fisher_and_act(policy, obs_g, device)
    a, c, h, w = grads_a.shape
    f_b3 = fisher_from_logit_jacobian(grads_a.reshape(a, c * h * w), probs_a)
    dh_b3 = (acts_a - acts_g).reshape(-1).astype(np.float64)
    dh_fc = (h_a - h_g).reshape(-1).astype(np.float64)
    hb = float(np.linalg.norm(acts_a.reshape(-1)))
    hf = float(np.linalg.norm(h_a.reshape(-1)))
    nb = float(np.linalg.norm(dh_b3))
    nf = float(np.linalg.norm(dh_fc))
    q_b3 = quadratic_form(f_b3, dh_b3)
    q_fc = quadratic_form(f_fc, dh_fc)
    s_b3 = sensitivity_along(f_b3, dh_b3)
    s_fc = sensitivity_along(f_fc, dh_fc)
    trFd_b3 = float(np.trace(f_b3)) / d_b3
    trFd_fc = float(np.trace(f_fc)) / d_fc
    out = {
        "Q_b3": q_b3,
        "Q_fc": q_fc,
        "S_b3": s_b3,
        "S_fc": s_fc,
        "trFd_b3": trFd_b3,
        "trFd_fc": trFd_fc,
        "S_over_trFd_b3": s_b3 / (trFd_b3 + 1e-18),
        "S_over_trFd_fc": s_fc / (trFd_fc + 1e-18),
        "dh_b3": nb,
        "dh_fc": nf,
        "h_b3": hb,
        "h_fc": hf,
        "rel_b3": nb / (hb + 1e-18),
        "rel_fc": nf / (hf + 1e-18),
        "T": nf / (nb + 1e-18),
        "two_kl": 2.0 * kl_categorical(p_a, p_g),
        "entropy": entropy_of(p_a),
        "max_prob": float(np.max(p_a)),
    }
    if include_block3_raw:
        out["_grads_a"] = grads_a
        out["_probs_a"] = probs_a
        out["_m_present"] = m_present
    return out


def fc_pair_scalars(policy, obs_a: np.ndarray, obs_g: np.ndarray, device: str) -> dict:
    """fc-only present/absent quadratic form + 2KL identity check (no block3)."""
    f_a, p_a, h_a = fc_fisher_and_act(policy, obs_a, device)
    _f_g, p_g, h_g = fc_fisher_and_act(policy, obs_g, device)
    dh = h_a - h_g
    q = quadratic_form(f_a, dh)
    s = sensitivity_along(f_a, dh)
    kl = kl_categorical(p_a, p_g)
    return {
        "quad": float(q),
        "S": float(s),
        "kl": float(kl),
        "two_kl": float(2.0 * kl),
        "dh_norm": float(np.linalg.norm(dh)),
        "abs_err_quad_vs_2kl": float(abs(q - 2.0 * kl)),
    }


def summarize_scalar_rows(rows: list[dict], keys: list[str] = SCALAR_KEYS) -> dict:
    """{key}_median over frames for each scalar key."""
    return {
        f"{k}_median": float(np.nanmedian([r[k] for r in rows])) for k in keys
    }


# Aliases used by the Attribute Maze scripts.
score_pair = dual_layer_scalars
entropy = entropy_of


# ---------------------------------------------------------------------------
# Attribute Maze (maze_colorsobjects_duel) harvest harness.
# ---------------------------------------------------------------------------


def make_env(
    obj1: str, obj2: str, start_level: int, rand_seed: int,
    distribution_mode: str = "easy", use_backgrounds: bool = False,
):
    return ProcgenEnv(
        num_envs=1,
        env_name="maze_colorsobjects_duel",
        num_levels=1,
        start_level=int(start_level),
        rand_seed=int(rand_seed),
        distribution_mode=distribution_mode,
        world_dim=5,
        obj1=obj1,
        obj2=obj2,
        use_backgrounds=use_backgrounds,
    )


def rgb_hwc(obs) -> np.ndarray:
    if isinstance(obs, dict):
        x = obs["rgb"]
    elif isinstance(obs, (tuple, list)):
        x = obs[0]
    else:
        x = obs
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
        x = np.transpose(x, (1, 2, 0))
    return x


def to_nchw(obs) -> np.ndarray:
    x = rgb_hwc(obs).astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def state_bytes(env) -> bytes:
    """Layout+pose fingerprint. Does not encode sprite appearance (obj1/obj2)."""
    from gm_geometry.procgen import unwrap_preferred

    st = unwrap_preferred(env, "get_state", "set_object").get_state()
    if isinstance(st, (list, tuple)):
        parts = []
        for s in st:
            if isinstance(s, (bytes, bytearray)):
                parts.append(bytes(s))
            else:
                parts.append(np.asarray(s).tobytes())
        return b"||".join(parts)
    if isinstance(st, (bytes, bytearray)):
        return bytes(st)
    return np.asarray(st).tobytes()


def pixel_l2(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(d * d)))


def layer_delta_norm(policy, obs_a, obs_b, device: str, layer: str = "block1") -> float:
    """||h_a - h_b|| at an early trunk layer (edit-magnitude control)."""
    caps = {}

    def hook(_m, _i, out):
        caps["h"] = out.detach()

    handle = getattr(policy.embedder, layer).register_forward_hook(hook)

    def fwd(obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        h0 = torch.zeros(x.shape[0], policy.embedder.output_dim, device=device)
        m = torch.zeros(x.shape[0], device=device)
        with torch.no_grad():
            policy(x, h0, m)
        return caps["h"].reshape(-1).float().cpu().numpy()

    try:
        ha = fwd(obs_a)
        hb = fwd(obs_b)
    finally:
        handle.remove()
    return float(np.linalg.norm(ha - hb))


def med(xs) -> float:
    return float(np.nanmedian(np.asarray(xs, dtype=np.float64)))


def summarize(rows: list[dict], *, extra_keys: tuple[str, ...] = ()) -> dict:
    keys = tuple(BASE_SUMMARY_KEYS) + tuple(extra_keys)
    out = {f"{k}_median": med([r[k] for r in rows]) for k in keys}
    pl = np.asarray([r["pixel_l2"] for r in rows], dtype=np.float64)
    out["pixel_l2_median"] = med(pl)
    out["pixel_l2_mean"] = float(pl.mean())
    out["pixel_l2_std"] = float(pl.std(ddof=1)) if len(pl) > 1 else 0.0
    out["pixel_l2_p10"] = float(np.percentile(pl, 10))
    out["pixel_l2_p90"] = float(np.percentile(pl, 90))
    return out


def harvest_arm(
    arm_name: str,
    arms: dict,
    *,
    n_frames: int,
    start_level: int,
    max_levels: int,
    rand_seed: int,
    n_warm_steps: int,
    action: int,
    distribution_mode: str = "easy",
    use_backgrounds: bool = False,
) -> tuple[list[dict], dict]:
    """distribution_mode/use_backgrounds must match how the scored agent was trained (texture vs black cohorts)."""
    spec = arms[arm_name]
    rows: list[dict] = []
    level = start_level
    used = 0
    rejects = {"state_mismatch": 0, "pixel_too_small": 0, "step_mismatch": 0}
    while len(rows) < n_frames and used < max_levels:
        ea = make_env(spec["obj_a"], spec["filler"], level, rand_seed,
                       distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
        eb = make_env(spec["obj_b"], spec["filler"], level, rand_seed,
                       distribution_mode=distribution_mode, use_backgrounds=use_backgrounds)
        try:
            oa = ea.reset()
            ob = eb.reset()
            if state_bytes(ea) != state_bytes(eb):
                rejects["state_mismatch"] += 1
                used += 1
                level += 1
                continue
            ok = True
            for _ in range(n_warm_steps):
                oa, _ra, da, _ia = ea.step(np.array([action]))
                ob, _rb, db, _ib = eb.step(np.array([action]))
                if state_bytes(ea) != state_bytes(eb):
                    rejects["step_mismatch"] += 1
                    ok = False
                    break
                if bool(np.asarray(da).reshape(-1)[0]) or bool(np.asarray(db).reshape(-1)[0]):
                    break
            if not ok:
                used += 1
                level += 1
                continue
            na, nb = to_nchw(oa), to_nchw(ob)
            pa = rgb_hwc(oa).astype(np.float32)
            pb = rgb_hwc(ob).astype(np.float32)
            if pa.max() <= 1.5:
                pa, pb = pa * 255.0, pb * 255.0
            pl2 = pixel_l2(pa, pb)
            if pl2 < 1.0:
                rejects["pixel_too_small"] += 1
                used += 1
                level += 1
                continue
            rows.append(
                {
                    "level": int(level),
                    "obs_a": na,
                    "obs_b": nb,
                    "pixel_l2": pl2,
                    "arm": arm_name,
                    "n_warm_steps": int(n_warm_steps),
                }
            )
            if len(rows) % 8 == 0:
                print(
                    f"  [{arm_name}] harvested {len(rows)}/{n_frames} rejects={rejects}",
                    flush=True,
                )
        finally:
            ea.close()
            eb.close()
        used += 1
        level += 1
    meta = {
        "arm": arm_name,
        "n_frames": len(rows),
        "levels_used": used,
        "rejects": rejects,
        "acceptance": float(len(rows) / max(used, 1)),
        "pixel_l2_per_frame": [float(r["pixel_l2"]) for r in rows],
        "pixel_l2_median": med([r["pixel_l2"] for r in rows]) if rows else None,
        "n_warm_steps": int(n_warm_steps),
        "spec": spec,
        "state_bytes_scope": "layout_agent_pose_typed_goal_cells_not_sprite_appearance",
    }
    print(
        f"harvest {arm_name} warm={n_warm_steps} n={len(rows)} "
        f"accept={meta['acceptance']:.3f} pix50={meta['pixel_l2_median']}",
        flush=True,
    )
    return rows, meta


def load_policy(ckpt: Path, device: str, *, root: Path, random_init: bool = False, init_seed: int = 0):
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy

    train_root = None
    for cand in (
        root / "vendored" / "cs-train-procgen",
        root / "vendored" / "train-procgen-pytorch",
    ):
        if (cand / "hyperparams" / "procgen" / "config.yml").exists():
            train_root = cand
            break
    if train_root is None:
        raise SystemExit("missing train-procgen hyperparams")
    with open(train_root / "hyperparams/procgen/config.yml") as f:
        recurrent = yaml.safe_load(f)["hard-500"]["recurrent"]
    probe = make_env("yellow_line_diag", "blue_gem", 0, 0)
    try:
        n_act = int(probe.action_space.n)
    finally:
        probe.close()
    if random_init:
        torch.manual_seed(int(init_seed))
        np.random.seed(int(init_seed))
    model = ImpalaModel(in_channels=3)
    policy = CategoricalPolicy(model, recurrent, n_act)
    if not random_init:
        blob = torch.load(ckpt, map_location="cpu")
        policy.load_state_dict(blob["model_state_dict"])
    policy.to(device).eval()
    return policy


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 5 or np.std(x) < 1e-18 or np.std(y) < 1e-18:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])
