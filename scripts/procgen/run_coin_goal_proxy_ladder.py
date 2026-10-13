#!/usr/bin/env python3
"""CoinRun goal/proxy ladder: location-GM, trajectory-prior sibling of the Maze goal/proxy ladder."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "external" / "train-procgen-pytorch"
for p in (ROOT / "src", ROOT / "external" / "procgenAISC", TRAIN_ROOT, ROOT / "scripts" / "procgen"):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)

import yaml  # noqa: E402
from common.model import ImpalaModel  # noqa: E402
from common.policy import CategoricalPolicy  # noqa: E402

from validate_langosco_coinrun_200m import load_policy, make_env  # noqa: E402
from gm_geometry.probes.coin_location import (  # noqa: E402
    find_floor_cell,
    get_info,
    labels_from_info,
    obs_nchw_after_set_coin,
    set_coin,
)
from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc  # noqa: E402
from gm_geometry.utils import device as _default_device  # noqa: E402

LOCI = ["block2", "block3", "fc"]
TRUNK = ["block2", "block3"]
SPATIAL = {"block2": 16 * 16, "block3": 8 * 8, "fc": 1}
LEFT_ACTIONS = (0, 1, 2)
RIGHT_ACTIONS = (6, 7, 8)


def build_init_policy(device):
    with open(TRAIN_ROOT / "hyperparams/procgen/config.yml") as f:
        recurrent = yaml.safe_load(f)["hard-500"]["recurrent"]
    probe = make_env("coinrun", num_levels=1, start_level=0, random_percent=0)
    model = ImpalaModel(in_channels=probe.observation_space.shape[0])
    policy = CategoricalPolicy(model, recurrent, probe.action_space.n)
    policy.to(device).eval()
    if hasattr(probe, "close"):
        probe.close()
    return policy


# ----------------------------- policy wrapper w/ hooks -----------------------------
class Ladder:
    def __init__(self, policy, device):
        self.policy = policy.to(device).eval()
        self.device = device
        self.emb = policy.embedder
        self.mods = {"block2": self.emb.block2, "block3": self.emb.block3, "fc": self.emb.fc}
        self.cap: dict[str, torch.Tensor] = {}
        self.patch = None
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
                sl = slice(None) if ch is None else torch.as_tensor(ch, device=out.device, dtype=torch.long)
                cur = new[:, sl]
                new[:, sl] = cur + s * (donor[:, sl] - cur)
                return new
            return None
        return fn

    def close(self):
        for h in self.handles:
            h.remove()

    @torch.no_grad()
    def run(self, obs, capture=None, patch=None):
        self.patch = patch
        self.cap = {}
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        h = torch.zeros(x.shape[0], self.emb.output_dim, device=self.device)
        m = torch.zeros(x.shape[0], device=self.device)
        dist, v, _ = self.policy(x, h, m)
        out = {
            "probs": dist.probs.detach().cpu().numpy(),
            "value": v.detach().cpu().numpy().reshape(-1),
            "caps": {k: self.cap[k].detach().cpu().numpy() for k in (capture or [])},
        }
        self.patch = None
        return out

    def sensitivity(self, obs, locus, pos_actions, neg_actions):
        """Per-channel |grad| of (sum logit[pos] - sum logit[neg]) and of V wrt locus acts."""
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        h = torch.zeros(B, self.emb.output_dim, device=self.device)
        m = torch.zeros(B, device=self.device)
        pos = torch.as_tensor(list(pos_actions), device=self.device, dtype=torch.long)
        neg = torch.as_tensor(list(neg_actions), device=self.device, dtype=torch.long)

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
            if g.ndim == 4:
                return g.abs().sum(dim=(2, 3)).mean(dim=0).cpu().numpy()
            return g.abs().mean(dim=0).cpu().numpy()

        pol = _grad(lambda d, v: (d.logits[:, pos].sum() - d.logits[:, neg].sum()))
        val = _grad(lambda d, v: v.reshape(-1).sum())
        return pol, val

    def signed_grad(self, obs, locus, pos_actions, neg_actions):
        """Signed per-channel mean grad of (logit[pos]-logit[neg]) and of V (for head overlap)."""
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        h = torch.zeros(B, self.emb.output_dim, device=self.device)
        m = torch.zeros(B, device=self.device)
        pos = torch.as_tensor(list(pos_actions), device=self.device, dtype=torch.long)
        neg = torch.as_tensor(list(neg_actions), device=self.device, dtype=torch.long)

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

        g_pol = _grad(lambda d, v: (d.logits[:, pos].sum() - d.logits[:, neg].sum()))
        g_val = _grad(lambda d, v: v.reshape(-1).sum())
        return g_pol, g_val


# ----------------------------- collection -----------------------------
def _mass(probs, actions):
    return float(np.asarray(probs)[list(actions)].sum())


def _caps1(run_out):
    return {k: run_out["caps"][k][0] for k in LOCI}


def collect(lad, *, n_target, start_level, max_levels, capture_causal, rng, verbose):
    """Per level: roll the TRAINED policy a random #steps so the agent actually advances
    (blind right-stepping does not move it), then build coin-right/left/right2/absent triples."""
    frames = []
    level = start_level
    tried = 0
    t0 = time.time()
    while len(frames) < n_target and (level - start_level) < max_levels:
        tried += 1
        env = make_env("coinrun_aisc", num_levels=1, start_level=level)
        level += 1
        obs = env.reset()
        k = int(rng.integers(3, 70))
        done = np.array([False])
        for _ in range(k):
            probs = lad.run(np.asarray(obs[0], dtype=np.float32))["probs"][0]
            obs, _r, done, _i = env.step(np.array([int(probs.argmax())], dtype=np.int64))
            if done[0]:
                break
        if done[0]:
            env.close()
            continue
        lab = labels_from_info(get_info(env))
        if lab is None:
            env.close()
            continue
        ax = float(lab["agent_x"])
        width = int(lab["level_width"])
        xy_r = find_floor_cell(env, prefer_left_of_agent=False, min_dx=2)
        xy_l = find_floor_cell(env, prefer_left_of_agent=True, min_dx=2)
        xy_r2 = find_floor_cell(env, prefer_left_of_agent=False, min_dx=6)
        if xy_r is None or xy_l is None:
            env.close()
            continue
        set_coin(env, xy_r[0], xy_r[1], 1)
        obs_a = obs_nchw_after_set_coin(env)
        set_coin(env, xy_l[0], xy_l[1], 1)
        obs_b = obs_nchw_after_set_coin(env)
        if xy_r2 is not None and xy_r2 != xy_r:
            set_coin(env, xy_r2[0], xy_r2[1], 1)
            obs_a2 = obs_nchw_after_set_coin(env)
        else:
            obs_a2 = None
        set_coin(env, xy_r[0], xy_r[1], 0)
        obs_abs = obs_nchw_after_set_coin(env)
        env.close()
        # coin must be visibly present and left/right must differ on-screen
        if float(np.abs(obs_a - obs_abs).max()) < 0.08:
            continue
        if float(np.abs(obs_a - obs_b).max()) < 0.08:
            continue
        ra = lad.run(obs_a, capture=LOCI)
        rb = lad.run(obs_b, capture=LOCI)
        rabs = lad.run(obs_abs, capture=LOCI)
        ra2 = lad.run(obs_a2, capture=LOCI) if obs_a2 is not None else None
        rec = {
            "level": level - 1, "rollout_k": k,
            "agent_x": ax, "level_width": width, "progress": ax / max(1.0, width),
            "coin_x_r": int(xy_r[0]), "coin_x_l": int(xy_l[0]),
            "obs_a": obs_a.astype(np.float32),
            "obs_b": obs_b.astype(np.float32),
            "caps_a": _caps1(ra), "caps_b": _caps1(rb),
            "caps_a2": _caps1(ra2) if ra2 is not None else None,
            "caps_abs": _caps1(rabs),
            "probs_a": ra["probs"][0].astype(np.float32),
            "probs_b": rb["probs"][0].astype(np.float32),
            "probs_abs": rabs["probs"][0].astype(np.float32),
            "value_a": float(ra["value"][0]),
            "value_b": float(rb["value"][0]),
            "value_abs": float(rabs["value"][0]),
        }
        frames.append(rec)
        if verbose and len(frames) % 20 == 0:
            print(f"  frames={len(frames)}/{n_target} level={level} ({time.time()-t0:.0f}s)", flush=True)
    return frames


# ----------------------------- probes (mirror Maze) -----------------------------
def _split(levels):
    uniq = np.sort(np.unique(levels))
    mid = uniq[len(uniq) // 2]
    return levels < mid, levels >= mid


def _fc_present(X, y, levels, n_shuffle, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y.astype(float))
    X, y, levels = X[ok], y[ok], levels[ok]
    if len(y) < 16 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y))}
    tr, te = _split(levels)
    if tr.sum() < 8 or te.sum() < 8 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        w, b = fit_ridge(X, y)
        return {"auc": float(roc_auc(y, predict_ridge(X, w, b))), "heldout": False,
                "n": int(len(y)), "C_F_topk": np.argsort(-np.abs(w))[:12].tolist()}
    w, b = fit_ridge(X[tr], y[tr])
    auc = float(roc_auc(y[te], predict_ridge(X[te], w, b)))
    null = []
    for _ in range(n_shuffle):
        ys = y[tr].copy(); rng.shuffle(ys)
        ws, bs = fit_ridge(X[tr], ys)
        null.append(float(roc_auc(y[te], predict_ridge(X[te], ws, bs))))
    p95 = float(np.quantile(null, 0.95)) if null else float("nan")
    return {"auc": auc, "null_p95": p95, "beats_null": bool(null and auc > p95),
            "heldout": True, "n": int(len(y)), "C_F_topk": np.argsort(-np.abs(w))[:12].tolist()}


def _spatial_present(X, y, levels, n_shuffle):
    ok = np.isfinite(y.astype(float))
    X, y, levels = X[ok], y[ok], levels[ok]
    if len(y) < 16 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "n": int(len(y))}
    N, C, H, W = X.shape
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
    return {"auc": auc, "null_p95": p95, "beats_null": bool(np.isfinite(p95) and auc > p95),
            "peak_cell": [int(i), int(j)], "heldout": heldout, "n": int(len(y)),
            "C_F_topk": np.argsort(-np.abs(best["w"]))[:12].tolist()}


def _present_at(X, y, levels, n_shuffle):
    return _spatial_present(X, y, levels, n_shuffle) if X.ndim == 4 else _fc_present(X, y, levels, n_shuffle)


def _pool_pixels(obs):
    x = obs.reshape(3, 8, 8, 8, 8).mean(axis=(2, 4))
    return x.reshape(-1)


def rung1_present(frames, init_lad, n_shuffle=50):
    """coin: bearing (A=coin-right=1 vs B=coin-left=0), pooled A+B frames.
    progress: agent progress-position (median split), from A frames only.
    """
    # coin bearing dataset: stack A (label 1) and B (label 0)
    levels_c = np.array([f["level"] for f in frames] + [f["level"] for f in frames])
    y_coin = np.array([1.0] * len(frames) + [0.0] * len(frames))
    caps_coin = {k: np.stack([f["caps_a"][k] for f in frames] + [f["caps_b"][k] for f in frames]) for k in LOCI}
    pix_coin = np.stack([_pool_pixels(f["obs_a"]) for f in frames] + [_pool_pixels(f["obs_b"]) for f in frames])
    # progress dataset: A frames, progress-position median split
    levels_p = np.array([f["level"] for f in frames])
    prog = np.array([f["progress"] for f in frames])
    # rank-based median split: guarantees a balanced 2-class target with ties present
    order = np.argsort(prog)
    y_prog = np.zeros(len(prog), dtype=float)
    y_prog[order[len(order) // 2:]] = 1.0
    caps_prog = {k: np.stack([f["caps_a"][k] for f in frames]) for k in LOCI}

    # init caps for the progress dataset (A obs)
    init_a = {k: [] for k in LOCI}
    for f in frames:
        ra = init_lad.run(f["obs_a"], capture=LOCI)
        for k in LOCI:
            init_a[k].append(ra["caps"][k][0])
    init_a = {k: np.stack(v) for k, v in init_a.items()}

    out = {}
    # coin
    out["coin"] = {}
    for locus in LOCI:
        trained = _present_at(caps_coin[locus], y_coin, levels_c, n_shuffle)
        # init: run init net on A and B, same pooled dataset
        ia_list, ib_list = [], []
        for f in frames:
            ra = init_lad.run(f["obs_a"], capture=[locus]); ia_list.append(ra["caps"][locus][0])
            rb = init_lad.run(f["obs_b"], capture=[locus]); ib_list.append(rb["caps"][locus][0])
        init_caps = np.stack(ia_list + ib_list)
        init = _present_at(init_caps, y_coin, levels_c, 0)
        ta, ib = trained.get("auc", np.nan), init.get("auc", np.nan)
        out["coin"][locus] = {
            "auc": ta, "beats_shuffle": trained.get("beats_null"),
            "peak_cell": trained.get("peak_cell"), "init_auc": ib,
            "beats_init": bool(np.isfinite(ta) and np.isfinite(ib) and ta > ib + 0.02),
            "n": trained.get("n"), "C_F_topk": trained.get("C_F_topk", []),
        }
    out["coin"]["pixel_floor_auc"] = _fc_present(pix_coin, y_coin, levels_c, 0).get("auc")
    best_c = max(LOCI, key=lambda L: abs(out["coin"][L]["auc"] - 0.5) if np.isfinite(out["coin"][L]["auc"]) else -1)
    out["coin"]["best_locus"] = best_c
    out["coin"]["C_F"] = out["coin"][best_c].get("C_F_topk", [])

    # progress
    out["progress"] = {}
    pix_p = np.stack([_pool_pixels(f["obs_a"]) for f in frames])
    for locus in LOCI:
        trained = _present_at(caps_prog[locus], y_prog, levels_p, n_shuffle)
        init = _present_at(init_a[locus], y_prog, levels_p, 0)
        ta, ib = trained.get("auc", np.nan), init.get("auc", np.nan)
        out["progress"][locus] = {
            "auc": ta, "beats_shuffle": trained.get("beats_null"),
            "peak_cell": trained.get("peak_cell"), "init_auc": ib,
            "beats_init": bool(np.isfinite(ta) and np.isfinite(ib) and ta > ib + 0.02),
            "n": trained.get("n"), "C_F_topk": trained.get("C_F_topk", []),
        }
    out["progress"]["pixel_floor_auc"] = _fc_present(pix_p, y_prog, levels_p, 0).get("auc")
    best_p = max(LOCI, key=lambda L: abs(out["progress"][L]["auc"] - 0.5) if np.isfinite(out["progress"][L]["auc"]) else -1)
    out["progress"]["best_locus"] = best_p
    out["progress"]["C_F"] = out["progress"][best_p].get("C_F_topk", [])
    return out


def rung23_wiring(lad, frames, present):
    """Policy/value sensitivity of (left - right) contrast wrt C_coin, and progress axis."""
    obs = np.stack([f["obs_a"] for f in frames])
    pol_val = {L: lad.sensitivity(obs, L, LEFT_ACTIONS, RIGHT_ACTIONS) for L in LOCI}
    rng = np.random.default_rng(0)
    out = {}
    for feat in ("coin", "progress"):
        locus = present[feat]["best_locus"]
        C_F = present[feat]["C_F"]
        pol, val = pol_val[locus]
        C = len(pol)
        if not C_F:
            C_F = list(range(min(12, C)))
        rand = rng.choice([c for c in range(C) if c not in set(C_F)],
                          size=min(len(C_F), max(1, C - len(C_F))), replace=False)
        trunk_load = float(np.mean([np.mean(pol_val[L][0]) / SPATIAL[L] for L in TRUNK]))
        fc_load = float(np.mean(pol_val["fc"][0]) / SPATIAL["fc"])
        out[feat] = {
            "locus": locus,
            "policy_load_CF": float(np.mean(pol[C_F])),
            "policy_load_null": float(np.mean(pol[rand])),
            "policy_ratio": float(np.mean(pol[C_F]) / (np.mean(pol[rand]) + 1e-9)),
            "value_load_CF": float(np.mean(val[C_F])),
            "value_load_null": float(np.mean(val[rand])),
            "value_ratio": float(np.mean(val[C_F]) / (np.mean(val[rand]) + 1e-9)),
            "trunk_load": trunk_load, "fc_load": fc_load,
            "trunk_vs_readout": float(trunk_load / (fc_load + 1e-9)),
        }
    return out


def rung3_value(frames):
    """coin value = present(A) - absent dV; progress value = corr(V_absent, progress)."""
    dv_coin = np.array([f["value_a"] - f["value_abs"] for f in frames])
    v_abs = np.array([f["value_abs"] for f in frames])
    prog = np.array([f["progress"] for f in frames])
    corr = float(np.corrcoef(v_abs, prog)[0, 1]) if np.std(prog) > 1e-6 and len(v_abs) > 4 else float("nan")
    return {
        "n": int(len(frames)),
        "dV_coin_present_minus_absent": {"mean": float(np.mean(dv_coin)),
                                         "se": float(np.std(dv_coin) / max(1, np.sqrt(len(dv_coin))))},
        "corr_Vabsent_progress": corr,
    }


def _tensor(caps, locus):
    return torch.as_tensor(caps[locus][None], dtype=torch.float32)


def rung45_coin_causal(lad, frames, present, scales=(1.0,), competence_min=0.15):
    """Coin relocate+patch. base = A (coin right). Patch C_coin(B=coin-left) into A, read
    retarget toward LEFT (coin's new side) against the go-right prior. Controls: same-side (A2)
    ~0, random-channel, whole-block, env-move ceiling (probs_B). On-manifold TV reported."""
    locus = present["coin"]["best_locus"]
    C_F = present["coin"]["C_F"]
    Ctot = frames[0]["caps_a"][locus].shape[0]
    if not C_F:
        C_F = list(range(min(12, Ctot)))
    rng = np.random.default_rng(0)
    rand_ch = rng.choice([c for c in range(Ctot) if c not in set(C_F)],
                         size=min(len(C_F), Ctot - len(C_F)), replace=False)
    res = {"locus": locus, "n": len(frames), "k": len(C_F), "C_total": int(Ctot), "scales": {}}
    for s in scales:
        rt_cf, rt_rand, rt_block, rt_same, rt_env = [], [], [], [], []
        dv_cf, dv_env, comp, tv_patch, tv_env = [], [], [], [], []
        for f in frames:
            base_left = _mass(f["probs_a"], LEFT_ACTIONS)
            donor = _tensor(f["caps_b"], locus)
            r = lad.run(f["obs_a"], patch={"locus": locus, "donor": donor, "channels": np.array(C_F), "scale": s})
            p = r["probs"][0]
            rt_cf.append(_mass(p, LEFT_ACTIONS) - base_left)
            dv_cf.append(float(r["value"][0] - f["value_a"]))
            comp.append(float(p.max()))
            tv_patch.append(0.5 * float(np.abs(p - f["probs_a"]).sum()))
            rr = lad.run(f["obs_a"], patch={"locus": locus, "donor": donor, "channels": rand_ch, "scale": s})
            rt_rand.append(_mass(rr["probs"][0], LEFT_ACTIONS) - base_left)
            rb = lad.run(f["obs_a"], patch={"locus": locus, "donor": donor, "channels": None, "scale": s})
            rt_block.append(_mass(rb["probs"][0], LEFT_ACTIONS) - base_left)
            if f["caps_a2"] is not None:
                sd = _tensor(f["caps_a2"], locus)
                rs = lad.run(f["obs_a"], patch={"locus": locus, "donor": sd, "channels": np.array(C_F), "scale": s})
                rt_same.append(_mass(rs["probs"][0], LEFT_ACTIONS) - base_left)
            rt_env.append(_mass(f["probs_b"], LEFT_ACTIONS) - base_left)
            dv_env.append(float(f["value_b"] - f["value_a"]))
            tv_env.append(0.5 * float(np.abs(f["probs_b"] - f["probs_a"]).sum()))

        def m(a):
            a = np.asarray(a, float)
            if a.size == 0:
                return {"mean": float("nan"), "se": float("nan")}
            return {"mean": float(np.nanmean(a)), "se": float(np.nanstd(a) / max(1, np.sqrt(len(a))))}
        res["scales"][f"{s:.2f}"] = {
            "retarget_CF": m(rt_cf), "retarget_random": m(rt_rand), "retarget_block": m(rt_block),
            "retarget_same_side": m(rt_same), "env_move_ceiling": m(rt_env),
            "frac_env_captured": float(np.nanmean(rt_cf) / (np.nanmean(rt_env) + 1e-9)),
            "excess_over_random": float(np.nanmean(rt_cf) - np.nanmean(rt_rand)),
            "dV_CF": m(dv_cf), "dV_env": m(dv_env),
            "tv_patch_vs_base": m(tv_patch), "tv_envmove_vs_base": m(tv_env),
            "competence_in_band": float(np.mean(np.asarray(comp) >= competence_min)),
        }
    return res


def rung45_progress_lever(lad, frames, present, scales=(1.0,)):
    """Progress code-patch lever. donor = high-progress frames, patch C_progress into low-progress
    test frames, read retarget toward RIGHT. progress-neutral null = mid-progress donor."""
    locus = present["progress"]["best_locus"]
    C_F = present["progress"]["C_F"]
    prog = np.array([f["progress"] for f in frames])
    q_hi = np.quantile(prog, 0.75)
    q_mid_lo, q_mid_hi = np.quantile(prog, 0.40), np.quantile(prog, 0.60)
    hi = [f for f in frames if f["progress"] >= q_hi]
    mid = [f for f in frames if q_mid_lo <= f["progress"] <= q_mid_hi]
    test = [f for f in frames if f["progress"] <= np.median(prog)]
    if len(hi) < 3 or len(test) < 3 or not C_F:
        return {"locus": locus, "n": 0, "note": "insufficient progress spread"}
    donor_hi = np.stack([f["caps_a"][locus] for f in hi]).mean(0)
    donor_mid = np.stack([f["caps_a"][locus] for f in mid]).mean(0) if len(mid) >= 3 else donor_hi
    dhi = torch.as_tensor(donor_hi[None], dtype=torch.float32)
    dmid = torch.as_tensor(donor_mid[None], dtype=torch.float32)
    Ctot = donor_hi.shape[0]
    res = {"locus": locus, "n": len(test), "k": len(C_F), "C_total": int(Ctot), "scales": {}}
    for s in scales:
        rt, rt_null, dv = [], [], []
        for f in test:
            base_right = _mass(f["probs_a"], RIGHT_ACTIONS)
            r = lad.run(f["obs_a"], patch={"locus": locus, "donor": dhi, "channels": np.array(C_F), "scale": s})
            rt.append(_mass(r["probs"][0], RIGHT_ACTIONS) - base_right)
            dv.append(float(r["value"][0] - f["value_a"]))
            rn = lad.run(f["obs_a"], patch={"locus": locus, "donor": dmid, "channels": np.array(C_F), "scale": s})
            rt_null.append(_mass(rn["probs"][0], RIGHT_ACTIONS) - base_right)

        def m(a):
            a = np.asarray(a, float)
            return {"mean": float(np.nanmean(a)), "se": float(np.nanstd(a) / max(1, np.sqrt(len(a))))}
        res["scales"][f"{s:.2f}"] = {"retarget_right": m(rt), "retarget_null": m(rt_null),
                                     "excess_over_null": float(np.nanmean(rt) - np.nanmean(rt_null)),
                                     "dV": m(dv)}
    return res


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _cos(a, b):
    return float(np.dot(_unit(a), _unit(b)))


def head_divergence(lad, frames, present, n_null=4000, seed=0):
    """Do the policy and value heads read the same fc subspace? (explicit overlap + random null)

    g_pol = d(logit_left - logit_right)/d(fc), g_val = dV/d(fc), both signed, mean over the
    coin-left choice points. Report cos(g_pol, g_val) vs a random-direction null, and, per feature,
    |cos(u_F, g_pol)| vs |cos(u_F, g_val)| to see which head reads that axis. Also a weight-only
    view: cos(sum W_pi[left]-sum W_pi[right], w_v)."""
    obs_b = np.stack([f["obs_b"] for f in frames])  # coin-left: left-vs-right contrast is meaningful
    g_pol, g_val = lad.signed_grad(obs_b, "fc", LEFT_ACTIONS, RIGHT_ACTIONS)
    overlap = _cos(g_pol, g_val)
    rng = np.random.default_rng(seed)
    C = g_pol.shape[0]
    R = rng.standard_normal((n_null, C))
    nulls = np.abs((R @ _unit(g_pol)) / (np.linalg.norm(R, axis=1) + 1e-12))
    null_p95 = float(np.quantile(nulls, 0.95))
    # feature axes in fc space (ridge decode weights), unit-normalised
    fc_a = np.stack([f["caps_a"]["fc"] for f in frames])
    fc_b = np.stack([f["caps_b"]["fc"] for f in frames])
    w_coin, _ = fit_ridge(np.concatenate([fc_a, fc_b]),
                          np.array([1.0] * len(frames) + [0.0] * len(frames)))
    prog = np.array([f["progress"] for f in frames])
    order = np.argsort(prog); y_prog = np.zeros(len(prog)); y_prog[order[len(order) // 2:]] = 1.0
    w_prog, _ = fit_ridge(fc_a, y_prog)
    # weight-only head directions
    W_pi = lad.policy.fc_policy.weight.detach().cpu().numpy().astype(np.float64)
    w_v = lad.policy.fc_value.weight.detach().cpu().numpy().astype(np.float64).reshape(-1)
    a_pol = W_pi[list(LEFT_ACTIONS)].sum(0) - W_pi[list(RIGHT_ACTIONS)].sum(0)
    return {
        "overlap_gpol_gval": overlap,
        "overlap_null_p95": null_p95,
        "heads_share": bool(abs(overlap) > null_p95),
        "coin_read_by_policy": abs(_cos(w_coin, g_pol)),
        "coin_read_by_value": abs(_cos(w_coin, g_val)),
        "progress_read_by_policy": abs(_cos(w_prog, g_pol)),
        "progress_read_by_value": abs(_cos(w_prog, g_val)),
        "weight_cos_pol_val": _cos(a_pol, w_v),
        "n": len(frames),
    }


def retarget_gradient(frames, n_bins=5):
    """Base preference gradient: net p_left (toward a left-placed coin) vs progress-position.
    Uses the coin-left frames (obs_b) so 'toward coin' = left, against the go-right prior."""
    prog = np.array([f["progress"] for f in frames])
    net = np.array([_mass(f["probs_b"], LEFT_ACTIONS) - _mass(f["probs_b"], RIGHT_ACTIONS) for f in frames])
    if len(prog) < n_bins:
        return {"progress": [], "net_left_minus_right": []}
    edges = np.quantile(prog, np.linspace(0, 1, n_bins + 1))
    centers, vals = [], []
    for k in range(n_bins):
        mk = (prog >= edges[k]) & (prog <= edges[k + 1])
        if mk.any():
            centers.append(float(prog[mk].mean()))
            vals.append(float(net[mk].mean()))
    return {"progress": centers, "net_left_minus_right": vals}


# ----------------------------- driver -----------------------------
# The coin (goal) is a relocatable object; progress (proxy) is a go-right trajectory prior that is not
# localizable in a single frame, so the ladder is asymmetric. Within one level set_coin places the coin
# right (A), left (B), at a second right cell (A2, same-side control) and absent. Rungs: 1 = presence
# (coin-bearing and progress decode vs pixel and init floors); 2/3 = policy/value wiring; 4/5 (--causal) =
# coin relocate+patch with same-side, random-channel, whole-block and env-move controls, plus a progress
# code-patch lever against a mid-progress donor null.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed-tag", default="")
    ap.add_argument("--n-frames", type=int, default=140)
    ap.add_argument("--start-level", type=int, default=2_000_000)
    ap.add_argument("--max-levels", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--scales", default="1.0")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scales = tuple(float(x) for x in args.scales.split(","))
    if args.quick:
        args.n_frames, args.max_levels = 16, 400
    device = args.device or str(_default_device())
    rng = np.random.default_rng(0)

    print(f"[load] {args.ckpt} device={device}", flush=True)
    policy = load_policy(args.ckpt, device)
    lad = Ladder(policy, device)
    init_lad = Ladder(build_init_policy(device), device)

    t0 = time.time()
    print("[stageA] collecting coin-right/left/right2/absent triples (coinrun_aisc)...", flush=True)
    frames = collect(lad, n_target=args.n_frames, start_level=args.start_level,
                     max_levels=args.max_levels, capture_causal=args.causal, rng=rng, verbose=True)
    print(f"[stageA] {len(frames)} frames in {time.time()-t0:.0f}s", flush=True)
    tag = ("_" + args.seed_tag) if args.seed_tag else ""
    if len(frames) < 10:
        (args.out_dir / f"ladder{tag}.json").write_text(
            json.dumps({"error": "too few frames", "n": len(frames)}, indent=2))
        return

    print("[rung1] present + floors...", flush=True)
    present = rung1_present(frames, init_lad)
    print("[rung2/3] wiring...", flush=True)
    wiring = rung23_wiring(lad, frames, present)
    value = rung3_value(frames)
    heads = head_divergence(lad, frames, present)
    grad = retarget_gradient(frames)

    result = {
        "ckpt": str(args.ckpt), "seed_tag": args.seed_tag, "env": "coinrun_aisc",
        "n_frames": len(frames), "device": device, "loci": LOCI,
        "gm_type": "location_trajectory", "goal": "coin", "proxy": "progress_go_right",
        "rung1_present": present,
        "rung23_wiring": wiring,
        "rung3_value": value,
        "head_divergence": heads,
        "retarget_gradient_base": grad,
        "runtime_s": time.time() - t0,
    }
    if args.causal:
        print("[rung4/5] coin relocate+patch...", flush=True)
        result["rung45_coin_causal"] = rung45_coin_causal(lad, frames, present, scales=scales)
        print("[rung4/5] progress code-patch lever...", flush=True)
        result["rung45_progress_lever"] = rung45_progress_lever(lad, frames, present, scales=scales)
    else:
        result["causal_note"] = ("rung4/5 not run (pass --causal). Coin uses set_coin relocate+patch; "
                                 "progress uses a code-patch lever.")
    out = args.out_dir / f"ladder{tag}.json"
    out.write_text(json.dumps(result, indent=2,
                              default=lambda o: float(o) if isinstance(o, np.floating) else o))
    print(f"[done] wrote {out}", flush=True)
    for feat in ("coin", "progress"):
        p = present[feat]
        bl = p["best_locus"]
        print(f"  {feat}: best={bl} auc={p[bl]['auc']:.3f} init={p[bl]['init_auc']:.3f} "
              f"beats_init={p[bl]['beats_init']} pol_ratio={wiring[feat]['policy_ratio']:.2f} "
              f"trunk/readout={wiring[feat]['trunk_vs_readout']:.2f}", flush=True)
    print(f"  dV coin present-absent={value['dV_coin_present_minus_absent']['mean']:+.3f} "
          f"corr(V_absent,progress)={value['corr_Vabsent_progress']}", flush=True)
    print(f"  head overlap cos(g_pol,g_val)={heads['overlap_gpol_gval']:+.3f} "
          f"(null_p95={heads['overlap_null_p95']:.3f} share={heads['heads_share']}) | "
          f"coin pol/val={heads['coin_read_by_policy']:.3f}/{heads['coin_read_by_value']:.3f} "
          f"progress pol/val={heads['progress_read_by_policy']:.3f}/{heads['progress_read_by_value']:.3f}",
          flush=True)
    if args.causal:
        cc = result["rung45_coin_causal"].get("scales", {}).get("1.00", {})
        pl = result["rung45_progress_lever"].get("scales", {}).get("1.00", {})
        if cc:
            print(f"  coin causal: retarget_CF={cc['retarget_CF']['mean']:+.4f} "
                  f"same_side={cc['retarget_same_side']['mean']:+.4f} "
                  f"env_ceiling={cc['env_move_ceiling']['mean']:+.4f} "
                  f"tv_patch={cc['tv_patch_vs_base']['mean']:.4f} tv_env={cc['tv_envmove_vs_base']['mean']:.4f} "
                  f"comp={cc['competence_in_band']:.2f}", flush=True)
        if pl:
            print(f"  progress lever: retarget_right={pl['retarget_right']['mean']:+.4f} "
                  f"null={pl['retarget_null']['mean']:+.4f} excess={pl['excess_over_null']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
