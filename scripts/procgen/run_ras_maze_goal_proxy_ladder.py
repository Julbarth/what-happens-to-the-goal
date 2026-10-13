#!/usr/bin/env python3
"""Attribute Maze goal/proxy ladder: presence, wiring and causal-patching rungs for the line and yellow cues."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (
    ROOT / "src",
    ROOT / "external" / "colour-shape-goal-misgeneralization" / "procgenAISC",
    ROOT / "external" / "colour-shape-goal-misgeneralization" / "train-procgen",
    ROOT / "external" / "train-procgen-pytorch",
):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)
sys.path[:] = [s for s in sys.path if not (s.endswith("external/procgenAISC") and "colour-shape" not in s)]

from gm_geometry.integrations.colour_shape import make_duel_env  # noqa: E402
from gm_geometry.integrations.colour_shape.policy import (  # noqa: E402
    load_impala_policy,
    load_recurrent_flag,
)
from gm_geometry.probes.choice_point import (  # noqa: E402
    CARDINAL_ACTIONS,
    action_toward,
    bearing_features,
    localize_frame,
    matched_geometry,
)
from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc  # noqa: E402
from gm_geometry.procgen import unwrap_base as _unwrap_base  # noqa: E402
from gm_geometry.utils import device as pick_device  # noqa: E402

LOCI = ["block2", "block3", "fc"]
WHICH = {"line": 1, "yellow": 2}  # set_object object id
GRID = 5  # world_dim
# Readout everywhere: contrast(F) = pi(action toward F) - pi(action toward other), F in {line=goal, yellow=proxy}.
# Causal rungs relocate only F (set_object present=0, then present=1 at a new cell) and patch C_F into the base pass.


# ----------------------------- env set_object helpers -----------------------------
def unwrap_base(env):
    return _unwrap_base(env, "set_object", on_missing="none")


def set_object(env, which, present, x=0, y=0):
    return int(unwrap_base(env).set_object([which], [x], [y], [present])[0])


def get_state(env):
    return unwrap_base(env).get_state()


def set_state(env, states):
    unwrap_base(env).set_state(states)


def obs_after_edit(env):
    base = unwrap_base(env)
    _r, obs, _f = base.observe()
    rgb = obs["rgb"] if isinstance(obs, dict) else obs
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    x = np.transpose(rgb, (2, 0, 1)).astype(np.float32) if rgb.shape[-1] == 3 else rgb.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return x


# ----------------------------- policy wrapper w/ hooks -----------------------------
class Ladder:
    def __init__(self, policy, device):
        self.policy = policy.to(device).eval()
        self.device = device
        self.emb = policy.embedder
        self.mods = {"block2": self.emb.block2, "block3": self.emb.block3, "fc": self.emb.fc}
        self.cap: dict[str, torch.Tensor] = {}
        self.patch = None  # {locus, donor(BxC..), channels(np.ndarray|None), scale}
        self.handles = [m.register_forward_hook(self._hook(n)) for n, m in self.mods.items()]

    def _hook(self, name):
        def fn(_m, _i, out):
            self.cap[name] = out
            p = self.patch
            if p is not None and p["locus"] == name:
                donor = p["donor"].to(out.device)
                if donor.shape[0] == 1 and out.shape[0] > 1:
                    donor = donor.expand_as(out)
                ch = p["channels"]
                s = p["scale"]
                new = out.clone()
                if ch is None:
                    sl = slice(None)
                else:
                    sl = torch.as_tensor(ch, device=out.device, dtype=torch.long)
                cur = new[:, sl]
                new[:, sl] = cur + s * (donor[:, sl] - cur)
                return new
            return None
        return fn

    def close(self):
        for h in self.handles:
            h.remove()

    @torch.no_grad()
    def run(self, obs, patch=None, capture=None):
        self.patch = patch
        self.cap = {}
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        h = torch.zeros(x.shape[0], self.emb.output_dim, device=self.device)
        m = torch.zeros(x.shape[0], device=self.device)
        dist, v, _ = self.policy(x, h, m)
        probs = dist.probs.detach().cpu().numpy()
        logits = dist.logits.detach().cpu().numpy()  # log-softmax (log-probs)
        val = v.detach().cpu().numpy().reshape(-1)
        caps = {}
        if capture:
            for k in capture:
                caps[k] = self.cap[k].detach().cpu().numpy()
        self.patch = None
        return {"probs": probs, "logits": logits, "value": val, "caps": caps}

    def sensitivity(self, obs, locus, a_line, a_gem):
        """Return per-channel |grad| of (logp[a_line]-logp[a_gem]) and of V wrt locus acts.

        a_line/a_gem: (B,) action indices. Returns (pol_vec[C], val_vec[C]).
        """
        self.patch = None
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        h = torch.zeros(B, self.emb.output_dim, device=self.device)
        m = torch.zeros(B, device=self.device)
        idx = torch.arange(B, device=self.device)
        al = torch.as_tensor(a_line, device=self.device, dtype=torch.long)
        ag = torch.as_tensor(a_gem, device=self.device, dtype=torch.long)

        def _grad(scalar_fn):
            self.cap = {}
            with torch.enable_grad():
                dist, v, _ = self.policy(x, h, m)
                act = self.cap[locus]
                act.retain_grad()
                scalar = scalar_fn(dist, v)
                self.policy.zero_grad(set_to_none=True)
                scalar.backward()
                g = act.grad.detach()
            if g.ndim == 4:  # B,C,H,W
                return g.abs().sum(dim=(2, 3)).mean(dim=0).cpu().numpy()
            return g.abs().mean(dim=0).cpu().numpy()  # B,C

        pol = _grad(lambda d, v: (d.logits[idx, al] - d.logits[idx, ag]).sum())
        val = _grad(lambda d, v: v.reshape(-1).sum())
        return pol, val

    def signed_grad_pf(self, obs, locus, a_line, a_gem):
        """Signed per-channel mean grad of per-frame (logit[a_line]-logit[a_gem]) and V."""
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        h = torch.zeros(B, self.emb.output_dim, device=self.device)
        m = torch.zeros(B, device=self.device)
        idx = torch.arange(B, device=self.device)
        al = torch.as_tensor(a_line, device=self.device, dtype=torch.long)
        ag = torch.as_tensor(a_gem, device=self.device, dtype=torch.long)

        def _grad(scalar_fn):
            self.cap = {}
            with torch.enable_grad():
                dist, v, _ = self.policy(x, h, m)
                act = self.cap[locus]
                act.retain_grad()
                self.policy.zero_grad(set_to_none=True)
                scalar_fn(dist, v).backward()
                g = act.grad.detach()
            if g.ndim == 4:
                return g.sum(dim=(2, 3)).mean(dim=0).cpu().numpy()
            return g.mean(dim=0).cpu().numpy()

        pol = _grad(lambda d, v: (d.logits[idx, al] - d.logits[idx, ag]).sum())
        val = _grad(lambda d, v: v.reshape(-1).sum())
        return pol, val


# ----------------------------- geometry helpers -----------------------------
def obj_xy(loc, feature):
    return loc["line_xy"] if feature == "line" else loc["gem_xy"]


def prob_toward(probs_row, agent_xy, target_xy):
    if agent_xy is None or target_xy is None:
        return float("nan")
    a = action_toward(agent_xy, target_xy)
    return float(probs_row[a])


def find_relocations(env, state0, which, agent_xy, obj_old_xy, base_obs):
    """Scan grid cells; return (moved_obs, moved_xy, same_obs) picking the moved cell
    that flips F to the opposite horizontal side of the agent (max |dx| tiebreak)."""
    best_moved, best_xy, best_score = None, None, -1.0
    same_obs, same_score = None, 1e9
    for gx in range(GRID):
        for gy in range(GRID):
            set_state(env, state0)
            set_object(env, which, 0)
            rc = set_object(env, which, 1, gx, gy)
            if rc != 0:
                continue
            ob = obs_after_edit(env)
            if float(np.abs(ob - base_obs).max()) < 0.3:
                continue
            loc = localize_frame(ob[0] if ob.ndim == 4 else ob)
            fx = loc["line_xy"] if which == 1 else loc["gem_xy"]
            if fx is None or agent_xy is None:
                continue
            l1 = float(np.abs(ob - base_obs).sum())
            if l1 < same_score:  # closest to original frame = same-location control
                same_score, same_obs = l1, ob
            old_side = np.sign(obj_old_xy[0] - agent_xy[0]) if obj_old_xy else 0.0
            new_side = np.sign(fx[0] - agent_xy[0])
            dist_move = abs(fx[0] - obj_old_xy[0]) if obj_old_xy else abs(fx[0] - agent_xy[0])
            flip = 1.0 if (old_side != 0 and new_side == -old_side) else 0.0
            score = 100.0 * flip + dist_move
            if score > best_score:
                best_moved, best_xy, best_score = ob, fx, score
    set_state(env, state0)
    return best_moved, best_xy, same_obs


# ----------------------------- Stage A: choice-point collection -----------------------------
def collect(lad, *, n_target, start_level, max_levels, max_steps, obj1, obj2, greedy,
            verbose, max_per_level=3, use_backgrounds=False):
    frames = []
    level = start_level
    t0 = time.time()
    while len(frames) < n_target and (level - start_level) < max_levels:
        env = make_duel_env(1, 1, 1, obj1, obj2, start_level=level, use_backgrounds=use_backgrounds)
        obs = env.reset().astype(np.float32)
        prev_img = None
        kept_this_level = 0
        for t in range(max_steps):
            out = lad.run(obs, capture=LOCI)
            logits, probs = out["logits"][0], out["probs"][0]
            act = int(logits.argmax()) if greedy else int(
                np.random.choice(len(probs), p=probs / probs.sum())
            )
            loc = localize_frame(obs[0], prev_hwc=prev_img)
            feats = bearing_features(loc)
            agent = loc["agent_xy"]
            line_xy, gem_xy = loc["line_xy"], loc["gem_xy"]
            keep = (
                kept_this_level < max_per_level
                and matched_geometry(loc)
                and agent is not None
                and line_xy is not None
                and gem_xy is not None
                and action_toward(agent, line_xy) != action_toward(agent, gem_xy)
            )
            if keep:
                st = get_state(env)
                base_obs = obs.copy()
                rec = {
                    "level": level, "t": t,
                    "agent_xy": agent, "line_xy": line_xy, "gem_xy": gem_xy,
                    "bearing_line_x": feats["bearing_line_x"], "bearing_line_y": feats["bearing_line_y"],
                    "bearing_gem_x": feats["bearing_gem_x"], "bearing_gem_y": feats["bearing_gem_y"],
                    "dist_line": feats["dist_line"], "dist_gem": feats["dist_gem"],
                    "base_probs": probs.astype(np.float32),
                    "base_value": float(out["value"][0]),
                    "caps": {k: out["caps"][k][0] for k in LOCI},
                    "obs": base_obs[0].astype(np.float32),
                }
                # donors per feature: relocate + remove
                for feat, which in WHICH.items():
                    old_xy = line_xy if feat == "line" else gem_xy
                    moved_obs, moved_xy, same_obs = find_relocations(
                        env, st, which, agent, old_xy, base_obs
                    )
                    set_state(env, st)
                    set_object(env, which, 0)
                    rm_obs = obs_after_edit(env)[None] if obs_after_edit(env).ndim == 3 else obs_after_edit(env)
                    set_state(env, st)
                    dm = None
                    if moved_obs is not None:
                        mo = moved_obs if moved_obs.ndim == 4 else moved_obs[None]
                        dm = lad.run(mo, capture=LOCI)
                    sm = None
                    if same_obs is not None:
                        so = same_obs if same_obs.ndim == 4 else same_obs[None]
                        sm = lad.run(so, capture=LOCI)
                    ro = rm_obs if rm_obs.ndim == 4 else rm_obs[None]
                    rr = lad.run(ro, capture=LOCI)
                    rec[feat] = {
                        "moved_xy": moved_xy,
                        "moved_caps": {k: dm["caps"][k][0] for k in LOCI} if dm else None,
                        "moved_probs": dm["probs"][0].astype(np.float32) if dm else None,
                        "moved_value": float(dm["value"][0]) if dm else None,
                        "same_caps": {k: sm["caps"][k][0] for k in LOCI} if sm else None,
                        "rm_caps": {k: rr["caps"][k][0] for k in LOCI},
                        "rm_probs": rr["probs"][0].astype(np.float32),
                        "rm_value": float(rr["value"][0]),
                    }
                frames.append(rec)
                kept_this_level += 1
                if verbose and len(frames) % 10 == 0:
                    print(f"  frames={len(frames)}/{n_target} level={level} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            prev_img = loc["img_hwc"]
            obs, _r, done, _i = env.step(np.array([act]))
            obs = obs.astype(np.float32)
            if done[0]:
                break
        if hasattr(env, "close"):
            env.close()
        level += 1
    return frames


# ----------------------------- Stage B: probes / floors -----------------------------
def _heldout_auc(x, y, levels, n_shuffle=50, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y.astype(float))
    x, y, levels = x[ok], y[ok], levels[ok]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y))}
    uniq = np.sort(np.unique(levels))
    mid = uniq[len(uniq) // 2]
    tr, te = levels < mid, levels >= mid
    if tr.sum() < 10 or te.sum() < 10 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        w, b = fit_ridge(x, y)
        return {"auc": float(roc_auc(y, predict_ridge(x, w, b))), "heldout": False, "n": int(len(y))}
    w, b = fit_ridge(x[tr], y[tr])
    auc = float(roc_auc(y[te], predict_ridge(x[te], w, b)))
    null = []
    for _ in range(n_shuffle):
        ys = y[tr].copy(); rng.shuffle(ys)
        ws, bs = fit_ridge(x[tr], ys)
        null.append(float(roc_auc(y[te], predict_ridge(x[te], ws, bs))))
    null = np.asarray(null)
    return {
        "auc": auc, "null_p95": float(np.quantile(null, 0.95)),
        "null_mean": float(null.mean()), "beats_null": bool(auc > np.quantile(null, 0.95)),
        "heldout": True, "n": int(len(y)), "w": w.astype(np.float32), "b": float(b),
    }


def _split(levels):
    uniq = np.sort(np.unique(levels))
    mid = uniq[len(uniq) // 2]
    return levels < mid, levels >= mid


def _fc_present(X, y, levels, n_shuffle, seed=0):
    """Held-out AUC on a low-dim (<=~2k) feature block. Returns auc/null/weights."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y.astype(float))
    X, y, levels = X[ok], y[ok], levels[ok]
    if len(y) < 16 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y))}
    tr, te = _split(levels)
    if tr.sum() < 8 or te.sum() < 8 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        w, b = fit_ridge(X, y)
        return {"auc": float(roc_auc(y, predict_ridge(X, w, b))), "heldout": False,
                "n": int(len(y)), "w": w, "b": float(b),
                "C_F_topk": np.argsort(-np.abs(w))[:12].tolist()}
    w, b = fit_ridge(X[tr], y[tr])
    auc = float(roc_auc(y[te], predict_ridge(X[te], w, b)))
    null = []
    for _ in range(n_shuffle):
        ys = y[tr].copy(); rng.shuffle(ys)
        ws, bs = fit_ridge(X[tr], ys)
        null.append(float(roc_auc(y[te], predict_ridge(X[te], ws, bs))))
    p95 = float(np.quantile(null, 0.95)) if null else float("nan")
    order = np.argsort(-np.abs(w))
    return {"auc": auc, "null_p95": p95, "beats_null": bool(null and auc > p95),
            "heldout": True, "n": int(len(y)), "w": w, "b": float(b),
            "C_F_topk": order[:12].tolist()}


def _spatial_present(X, y, levels, n_shuffle):
    """Per-cell ridge on channel vectors. Select peak cell on TRAIN AUC (no leakage),
    report held-out test AUC + shuffle null there, and top channels by |w| at peak."""
    ok = np.isfinite(y.astype(float))
    X, y, levels = X[ok], y[ok], levels[ok]
    N, C, H, W = X.shape
    if len(y) < 16 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y))}
    tr, te = _split(levels)
    heldout = not (tr.sum() < 8 or te.sum() < 8 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2)
    best = None
    for i in range(H):
        for j in range(W):
            xc = X[:, :, i, j]
            if heldout:
                w, b = fit_ridge(xc[tr], y[tr])
                tr_auc = roc_auc(y[tr], predict_ridge(xc[tr], w, b))
            else:
                w, b = fit_ridge(xc, y)
                tr_auc = roc_auc(y, predict_ridge(xc, w, b))
            score = abs(tr_auc - 0.5)
            if best is None or score > best["score"]:
                best = {"score": score, "ij": (i, j), "w": w, "b": b}
    i, j = best["ij"]
    xc = X[:, :, i, j]
    if heldout:
        auc = float(roc_auc(y[te], predict_ridge(xc[te], best["w"], best["b"])))
        rng = np.random.default_rng(0)
        null = []
        for _ in range(n_shuffle):
            ys = y[tr].copy(); rng.shuffle(ys)
            ws, bs = fit_ridge(xc[tr], ys)
            null.append(float(roc_auc(y[te], predict_ridge(xc[te], ws, bs))))
        p95 = float(np.quantile(null, 0.95)) if null else float("nan")
    else:
        auc = float(roc_auc(y, predict_ridge(xc, best["w"], best["b"])))
        p95 = float("nan")
    order = np.argsort(-np.abs(best["w"]))
    return {"auc": auc, "null_p95": p95, "beats_null": bool(np.isfinite(p95) and auc > p95),
            "peak_cell": [int(i), int(j)], "heldout": heldout, "n": int(len(y)),
            "C_F_topk": order[:12].tolist()}


def _present_at(X, y, levels, n_shuffle):
    return _spatial_present(X, y, levels, n_shuffle) if X.ndim == 4 else _fc_present(X, y, levels, n_shuffle)


def _pool_pixels(obs):  # (3,64,64) -> avg-pool to (3,8,8) flattened = 192 dims
    x = obs.reshape(3, 8, 8, 8, 8).mean(axis=(2, 4))
    return x.reshape(-1)


def rung1_present(frames, init_lad, device, n_shuffle=50):
    levels = np.array([f["level"] for f in frames])
    y_line = np.array([1.0 if f["bearing_line_x"] > 0 else 0.0 for f in frames])
    y_yellow = np.array([1.0 if f["bearing_gem_x"] > 0 else 0.0 for f in frames])
    labels = {"line": y_line, "yellow": y_yellow}
    pix = np.stack([_pool_pixels(f["obs"]) for f in frames])
    trained_caps = {k: np.stack([f["caps"][k] for f in frames]) for k in LOCI}
    init_caps = {k: [] for k in LOCI}
    for f in frames:
        r = init_lad.run(f["obs"], capture=LOCI)
        for k in LOCI:
            init_caps[k].append(r["caps"][k][0])
    init_caps = {k: np.stack(v) for k, v in init_caps.items()}
    out = {}
    for feat, y in labels.items():
        out[feat] = {}
        for locus in LOCI:
            trained = _present_at(trained_caps[locus], y, levels, n_shuffle)
            init = _present_at(init_caps[locus], y, levels, 0)
            ta, ia = trained.get("auc", np.nan), init.get("auc", np.nan)
            out[feat][locus] = {
                "auc": ta, "beats_shuffle": trained.get("beats_null"),
                "shuffle_p95": trained.get("null_p95"),
                "peak_cell": trained.get("peak_cell"),
                "init_auc": ia,
                "beats_init": bool(np.isfinite(ta) and np.isfinite(ia) and ta > ia + 0.02),
                "n": trained.get("n"), "C_F_topk": trained.get("C_F_topk", []),
            }
        pixf = _fc_present(pix, y, levels, 0)
        out[feat]["pixel_floor_auc"] = pixf.get("auc")
        best = max(LOCI, key=lambda L: abs(out[feat][L]["auc"] - 0.5)
                   if np.isfinite(out[feat][L]["auc"]) else -1)
        out[feat]["best_locus"] = best
        out[feat]["C_F"] = out[feat][best].get("C_F_topk", [])
    return out


def rung23_wiring(lad, frames, present):
    out = {}
    obs = np.stack([f["obs"] for f in frames])
    a_line = np.array([action_toward(f["agent_xy"], f["line_xy"]) for f in frames])
    a_gem = np.array([action_toward(f["agent_xy"], f["gem_xy"]) for f in frames])
    pol_val = {}
    for locus in LOCI:
        pol, val = lad.sensitivity(obs, locus, a_line, a_gem)
        pol_val[locus] = (pol, val)
    rng = np.random.default_rng(0)
    for feat in WHICH:
        locus = present[feat]["best_locus"]
        C_F = present[feat]["C_F"]
        pol, val = pol_val[locus]
        C = len(pol)
        if not C_F:
            C_F = list(range(min(12, C)))
        rand = rng.choice([c for c in range(C) if c not in set(C_F)],
                          size=min(len(C_F), max(1, C - len(C_F))), replace=False)
        out[feat] = {
            "locus": locus,
            "policy_load_CF": float(np.mean(pol[C_F])),
            "policy_load_null": float(np.mean(pol[rand])),
            "policy_ratio": float(np.mean(pol[C_F]) / (np.mean(pol[rand]) + 1e-9)),
            "value_load_CF": float(np.mean(val[C_F])),
            "value_load_null": float(np.mean(val[rand])),
            "value_ratio": float(np.mean(val[C_F]) / (np.mean(val[rand]) + 1e-9)),
        }
    return out


# ----------------------------- Stage C: causal patching -----------------------------
def _tensor(donor_caps, locus):
    return torch.as_tensor(donor_caps[locus][None], dtype=torch.float32)


def rung45_causal(lad, frames, present, scales=(1.0,), competence_min=0.15):
    out = {}
    rng = np.random.default_rng(0)
    for feat in WHICH:
        locus = present[feat]["best_locus"]
        C_F = present[feat]["C_F"]
        usable = [f for f in frames if f[feat]["moved_caps"] is not None and f[feat]["moved_xy"] is not None]
        if not usable:
            out[feat] = {"locus": locus, "n": 0, "note": "no relocations"}
            continue
        Ctot = usable[0]["caps"][locus].shape[0]
        if not C_F:
            C_F = list(range(min(12, Ctot)))
        rand_ch = rng.choice([c for c in range(Ctot) if c not in set(C_F)],
                             size=min(len(C_F), Ctot - len(C_F)), replace=False)
        res = {"locus": locus, "n": len(usable), "k": len(C_F), "C_total": int(Ctot), "scales": {}}
        for s in scales:
            rt_cf, rt_rand, rt_block, rt_env, rt_same = [], [], [], [], []
            dv_cf, dv_env = [], []
            comp = []
            for f in usable:
                agent = f["agent_xy"]
                new_xy = f[feat]["moved_xy"]
                a_new = action_toward(agent, new_xy)
                base = f["base_probs"][a_new]
                donor = _tensor(f[feat]["moved_caps"], locus)
                # C_F patch
                r = lad.run(f["obs"], patch={"locus": locus, "donor": donor,
                                             "channels": np.array(C_F), "scale": s})
                p_cf = r["probs"][0]
                rt_cf.append(float(p_cf[a_new] - base))
                dv_cf.append(float(r["value"][0] - f["base_value"]))
                comp.append(float(p_cf.max()))
                # random-channel
                rr = lad.run(f["obs"], patch={"locus": locus, "donor": donor,
                                              "channels": rand_ch, "scale": s})
                rt_rand.append(float(rr["probs"][0][a_new] - base))
                # whole-block
                rb = lad.run(f["obs"], patch={"locus": locus, "donor": donor,
                                              "channels": None, "scale": s})
                rt_block.append(float(rb["probs"][0][a_new] - base))
                # same-location control
                if f[feat]["same_caps"] is not None:
                    sd = _tensor(f[feat]["same_caps"], locus)
                    rs = lad.run(f["obs"], patch={"locus": locus, "donor": sd,
                                                  "channels": np.array(C_F), "scale": s})
                    rt_same.append(float(rs["probs"][0][a_new] - base))
                # env-move ceiling
                rt_env.append(float(f[feat]["moved_probs"][a_new] - base))
                dv_env.append(float(f[feat]["moved_value"] - f["base_value"]))
            def m(a):
                a = np.asarray(a, float)
                return {"mean": float(np.nanmean(a)), "se": float(np.nanstd(a) / max(1, np.sqrt(len(a))))}
            in_band = float(np.mean(np.asarray(comp) >= competence_min))
            k_frac = len(C_F) / Ctot
            uniform_share = k_frac * float(np.nanmean(rt_env))
            res["scales"][f"{s:.2f}"] = {
                "retarget_CF": m(rt_cf), "retarget_random": m(rt_rand),
                "retarget_block": m(rt_block), "retarget_same": m(rt_same) if rt_same else None,
                "env_move_ceiling": m(rt_env),
                "frac_env_captured": float(np.nanmean(rt_cf) / (np.nanmean(rt_env) + 1e-9)),
                # is C_F a *sparse* handle? excess over what k/C random channels would give
                "k_frac": float(k_frac),
                "uniform_share_ceiling": float(uniform_share),
                "excess_over_share": float(np.nanmean(rt_cf) - uniform_share),
                "excess_over_random": float(np.nanmean(rt_cf) - np.nanmean(rt_rand)),
                "dV_CF": m(dv_cf), "dV_env": m(dv_env),
                "competence_in_band": in_band,
            }
        out[feat] = res
    return out


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _cos(a, b):
    return float(np.dot(_unit(a), _unit(b)))


def head_divergence(lad, frames, n_null=4000, seed=0):
    """Policy vs value fc subspace overlap (parity with Maze/Coin ladders)."""
    obs = np.stack([f["obs"] for f in frames])
    a_line = np.array([action_toward(f["agent_xy"], f["line_xy"]) for f in frames])
    a_gem = np.array([action_toward(f["agent_xy"], f["gem_xy"]) for f in frames])
    g_pol, g_val = lad.signed_grad_pf(obs, "fc", a_line, a_gem)
    overlap = _cos(g_pol, g_val)
    rng = np.random.default_rng(seed)
    C = g_pol.shape[0]
    R = rng.standard_normal((n_null, C))
    nulls = np.abs((R @ _unit(g_pol)) / (np.linalg.norm(R, axis=1) + 1e-12))
    null_p95 = float(np.quantile(nulls, 0.95))
    fc = np.stack([f["caps"]["fc"] for f in frames])
    y_line = np.array([1.0 if f["bearing_line_x"] > 0 else 0.0 for f in frames])
    y_yel = np.array([1.0 if f["bearing_gem_x"] > 0 else 0.0 for f in frames])
    w_line, _ = fit_ridge(fc, y_line)
    w_yel, _ = fit_ridge(fc, y_yel)
    W_pi = lad.policy.fc_policy.weight.detach().cpu().numpy().astype(np.float64)
    w_v = lad.policy.fc_value.weight.detach().cpu().numpy().astype(np.float64).reshape(-1)
    a_pol = np.zeros(W_pi.shape[1], dtype=np.float64)
    for al, ag in zip(a_line, a_gem):
        a_pol += W_pi[int(al)] - W_pi[int(ag)]
    a_pol /= max(1, len(frames))
    return {
        "overlap_gpol_gval": overlap, "overlap_null_p95": null_p95,
        "heads_share": bool(abs(overlap) > null_p95),
        "line_read_by_policy": abs(_cos(w_line, g_pol)),
        "line_read_by_value": abs(_cos(w_line, g_val)),
        "yellow_read_by_policy": abs(_cos(w_yel, g_pol)),
        "yellow_read_by_value": abs(_cos(w_yel, g_val)),
        "weight_cos_pol_val": _cos(a_pol, w_v),
        "n": len(frames),
    }


# ----------------------------- driver -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed-tag", default="")
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--start-level", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-per-level", type=int, default=3)
    ap.add_argument("--obj1", default="red_line_diag")
    ap.add_argument("--obj2", default="yellow_gem")
    ap.add_argument("--device", default=str(pick_device()))
    ap.add_argument("--scales", default="0.5,1.0,1.5")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-causal", action="store_true",
                    help="skip rung4/5 (use with --merge-causal-from to reuse existing causal results)")
    ap.add_argument("--merge-causal-from", type=Path, default=None,
                    help="copy rung45_causal from an existing ladder JSON")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.n_frames, args.max_levels = 12, 200
    scales = tuple(float(x) for x in args.scales.split(","))
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    probe = make_duel_env(1, 1, 1, args.obj1, args.obj2, start_level=args.start_level, use_backgrounds=False)
    recurrent = load_recurrent_flag()
    in_ch = probe.observation_space.shape[0]
    n_act = probe.action_space.n
    if hasattr(probe, "close"):
        probe.close()
    print(f"[load] {args.ckpt} recurrent={recurrent} in_ch={in_ch} n_act={n_act}", flush=True)
    policy = load_impala_policy(args.ckpt, in_ch, n_act, recurrent)
    lad = Ladder(policy, device)

    # init-net floor (fresh random weights, same architecture)
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy
    init_policy = CategoricalPolicy(ImpalaModel(in_channels=in_ch), recurrent, n_act)
    init_lad = Ladder(init_policy, device)

    t0 = time.time()
    print("[stageA] collecting choice points...", flush=True)
    frames = collect(lad, n_target=args.n_frames, start_level=args.start_level,
                     max_levels=args.max_levels, max_steps=args.max_steps,
                     obj1=args.obj1, obj2=args.obj2, greedy=True, verbose=True,
                     max_per_level=args.max_per_level)
    print(f"[stageA] {len(frames)} frames in {time.time()-t0:.0f}s", flush=True)
    if len(frames) < 10:
        (args.out_dir / f"ladder{('_'+args.seed_tag) if args.seed_tag else ''}.json").write_text(
            json.dumps({"error": "too few choice points", "n": len(frames)}, indent=2))
        return

    print("[rung1] present + floors...", flush=True)
    present = rung1_present(frames, init_lad, device)
    print("[rung2/3] wiring...", flush=True)
    wiring = rung23_wiring(lad, frames, present)
    print("[head] divergence...", flush=True)
    heads = head_divergence(lad, frames)
    causal = None
    if not args.no_causal:
        print("[rung4/5] causal patching...", flush=True)
        causal = rung45_causal(lad, frames, present, scales=scales)
    else:
        print("[rung4/5] skipped (--no-causal)", flush=True)

    # strip heavy fields from present before dump
    def clean(d):
        for feat in list(d):
            for locus in LOCI:
                d[feat][locus].pop("w", None)
        return d

    result = {
        "ckpt": str(args.ckpt), "seed_tag": args.seed_tag,
        "n_frames": len(frames), "device": device, "scales": list(scales),
        "recurrent": recurrent, "loci": LOCI,
        "rung1_present": clean(present),
        "rung23_wiring": wiring,
        "head_divergence": heads,
        "runtime_s": time.time() - t0,
    }
    if causal is not None:
        result["rung45_causal"] = causal
    elif args.merge_causal_from:
        prev = json.loads(Path(args.merge_causal_from).read_text())
        if "rung45_causal" in prev:
            result["rung45_causal"] = prev["rung45_causal"]
            print(f"[merge] rung45_causal from {args.merge_causal_from}", flush=True)
    tag = ("_" + args.seed_tag) if args.seed_tag else ""
    out = args.out_dir / f"ladder{tag}.json"
    out.write_text(json.dumps(result, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o))
    print(f"[done] wrote {out}", flush=True)
    for feat in WHICH:
        p = present[feat]
        c = (causal or result.get("rung45_causal") or {}).get(feat, {})
        s1 = c.get("scales", {}).get("1.00", {})
        print(f"  {feat}: best={p['best_locus']} auc={p[p['best_locus']]['auc']:.3f} "
              f"beats_init={p[p['best_locus']]['beats_init']} "
              f"retarget_CF={s1.get('retarget_CF', {}).get('mean')} "
              f"env_ceiling={s1.get('env_move_ceiling', {}).get('mean')}", flush=True)
    print(f"  head overlap cos={heads['overlap_gpol_gval']:+.3f} "
          f"(null_p95={heads['overlap_null_p95']:.3f} share={heads['heads_share']})", flush=True)


if __name__ == "__main__":
    main()
