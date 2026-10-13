"""CoinRun phase analysis: episode-aligned activations, decision anchors, and τ-resolved coin-probe readouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from gm_geometry.probes.linear import fit_ridge, predict_ridge, roc_auc


# ProcGen discrete 0–8 = {L,N,R} × {D,N,U}. Horizontal is the behavioural axis.
LEFT_ACTIONS = frozenset({0, 1, 2})
RIGHT_ACTIONS = frozenset({6, 7, 8})

# Commitment-detector thresholds; tune only on held-out slices.
COMMIT_W = 8
COMMIT_THETA = 0.75
COMMIT_R = 3
COMMIT_MIN_PRE = 5


def yellow_coin_mask(chw01: np.ndarray) -> np.ndarray:
    """Heuristic yellow-gold coin detector on CHW float [0,1] frame."""
    x = np.asarray(chw01)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] in (3, 4):
        r, g, b = x[0], x[1], x[2]
    else:
        raise ValueError("unexpected obs shape {}".format(x.shape))
    return (r > 0.55) & (g > 0.45) & (b < 0.35) & ((r - b) > 0.25)


def coin_centroid_xy(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def classify_done(info: dict, max_steps: int) -> str:
    if info.get("prev_level_complete"):
        return "success"
    if info.get("prev_level/invisible_coin_collected") or info.get("invisible_coin_collected"):
        return "goal_misgen"
    steps = int(info.get("prev_level/total_steps") or info.get("total_steps") or 0)
    if steps >= max_steps - 1 or steps > 999:
        return "timed_out"
    return "died"


@dataclass
class EpisodeTrace:
    embeddings: np.ndarray  # (T, d)
    coin_present: np.ndarray  # (T,) int
    coin_x: np.ndarray
    coin_y: np.ndarray
    actions: np.ndarray  # (T,) int
    action_logits: np.ndarray  # (T, A)
    outcome: str
    level: int
    length: int
    # Regime-invariant labels (None if died / timed_out)
    got_coin: Optional[int] = None  # 1 = collected real coin (= OOD success)
    success: Optional[int] = None  # env success flag; == got_coin on our OOD bins
    reached_end: Optional[int] = None  # 1 = invisible-end / GM without coin
    y_gm: Optional[int] = None  # 1 = goal_misgen, 0 = success
    # Anchors
    anchor_step: Optional[int] = None
    anchor_kind: Optional[str] = None
    tau: Optional[np.ndarray] = None
    extras: dict = field(default_factory=dict)


def _set_outcome_labels(ep: EpisodeTrace) -> None:
    """Fixed physical meanings — do not invert across early/late."""
    if ep.outcome == "success":
        # OOD: prev_level_complete = collected the (relocated) coin
        ep.got_coin = 1
        ep.success = 1
        ep.reached_end = 0
        ep.y_gm = 0
    elif ep.outcome == "goal_misgen":
        # Hit invisible end-coin without collecting the real coin
        ep.got_coin = 0
        ep.success = 0
        ep.reached_end = 1
        ep.y_gm = 1
    else:
        ep.got_coin = None
        ep.success = None
        ep.reached_end = None
        ep.y_gm = None


def _forward_with_fc(policy, obs, hidden, device):
    captured = []

    def _hook(_m, _i, out):
        captured.append(out.detach().cpu().numpy())

    handle = policy.embedder.fc.register_forward_hook(_hook)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    hid_t = torch.as_tensor(hidden, dtype=torch.float32, device=device)
    done_t = torch.zeros(obs.shape[0] if obs.ndim == 4 else 1, device=device)
    with torch.no_grad():
        dist, _v, next_hid = policy(obs_t, hid_t, done_t)
    handle.remove()
    return dist, next_hid.detach().cpu().numpy(), captured[0][0].astype(np.float32)


def collect_coinrun_episodes(
    policy,
    make_env_fn,
    *,
    n_episodes,
    start_level,
    device,
    env_name="coinrun_aisc",
    random_percent=None,
    max_steps=500,
    greedy=True,
):
    """On-policy rollouts with per-frame fc embeddings + coin visibility."""
    episodes = []
    level = start_level
    while len(episodes) < n_episodes:
        env = make_env_fn(
            env_name,
            num_levels=1,
            start_level=level,
            random_percent=random_percent,
        )
        obs = env.reset()
        hidden = np.zeros((1, policy.embedder.output_dim), dtype=np.float32)
        embs, coins, cxs, cys, acts, logits = [], [], [], [], [], []
        last_info = {}
        outcome = "timed_out"
        for t in range(max_steps):
            mask = yellow_coin_mask(obs)
            coin = int(mask.sum() >= 8)
            cx, cy = coin_centroid_xy(mask) if coin else (float("nan"), float("nan"))
            dist, hidden, emb = _forward_with_fc(policy, obs, hidden, device)
            probs = dist.probs.detach().cpu().numpy()[0]
            act = int(probs.argmax() if greedy else dist.sample().cpu().numpy()[0])
            embs.append(emb)
            coins.append(coin)
            cxs.append(cx)
            cys.append(cy)
            acts.append(act)
            logits.append(probs.astype(np.float32))
            obs, _rew, done, info = env.step(np.array([act], dtype=np.int64))
            last_info = info[0] if isinstance(info, (list, tuple)) else info
            if last_info.get("invisible_coin_collected") and not done[0]:
                outcome = "goal_misgen"
                break
            if done[0]:
                outcome = classify_done(last_info, max_steps)
                break
        if hasattr(env, "close"):
            env.close()
        T = len(embs)
        if T == 0:
            level += 1
            continue
        ep = EpisodeTrace(
            embeddings=np.stack(embs, axis=0),
            coin_present=np.asarray(coins, dtype=np.int64),
            coin_x=np.asarray(cxs, dtype=np.float64),
            coin_y=np.asarray(cys, dtype=np.float64),
            actions=np.asarray(acts, dtype=np.int64),
            action_logits=np.stack(logits, axis=0),
            outcome=outcome,
            level=level,
            length=T,
        )
        _set_outcome_labels(ep)
        episodes.append(ep)
        level += 1
        if len(episodes) % 50 == 0:
            print("  collected {}/{}  last={} T={}".format(len(episodes), n_episodes, outcome, T))
    return episodes


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def horizontal(actions):
    """ProcGen action indices → h(t) ∈ {-1, 0, +1}."""
    h = np.zeros(len(actions), dtype=np.int32)
    for i, a in enumerate(actions):
        if a in LEFT_ACTIONS:
            h[i] = -1
        elif a in RIGHT_ACTIONS:
            h[i] = 1
    return h


def _has_run(mask, r):
    run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run >= r:
            return True
    return False


def commitment_step(
    actions,
    *,
    W=COMMIT_W,
    theta=COMMIT_THETA,
    R=COMMIT_R,
    min_pre=COMMIT_MIN_PRE,
):
    """Earliest point-of-no-return rightward drive, or None.

    t* = first t ≥ min_pre such that frac(RIGHT in [t, t+W)) ≥ theta and no
    sustained LEFT run of length ≥ R occurs at or after t.
    """
    h = horizontal(actions)
    T = len(h)
    if T < W + min_pre:
        return None
    for t in range(min_pre, T - 1):
        window = h[t : min(t + W, T)]
        if len(window) == 0:
            continue
        if float((window == 1).mean()) < theta:
            continue
        if _has_run(h[t:] == -1, R):
            continue
        return int(t)
    return None


def coin_onset_step(coin_present):
    for t, c in enumerate(coin_present):
        if c:
            return int(t)
    return None


def assign_anchors(
    episodes,
    *,
    kind="commitment",
    require_min_pre=True,
    late_onset_only=False,
    commit_W=COMMIT_W,
    commit_theta=COMMIT_THETA,
    commit_R=COMMIT_R,
    commit_min_pre=COMMIT_MIN_PRE,
):
    """Assign τ = step − anchor_step for ``kind`` in {commitment, coin_onset, steps_to_end}.

    ``late_onset_only`` keeps only coin_onset episodes whose onset is after the first frame."""
    n_ok = 0
    n_none = 0
    n_late_drop = 0
    for ep in episodes:
        tstar = None
        used_kind = kind
        if kind == "commitment":
            tstar = commitment_step(
                ep.actions,
                W=commit_W,
                theta=commit_theta,
                R=commit_R,
                min_pre=commit_min_pre,
            )
        elif kind == "coin_onset":
            tstar = coin_onset_step(ep.coin_present)
            if late_onset_only and tstar is not None and tstar == 0:
                tstar = None
                n_late_drop += 1
                used_kind = "coin_onset_late"
        elif kind == "steps_to_end":
            tstar = max(0, ep.length - 1)
            used_kind = "steps_to_end"
        else:
            raise ValueError("unknown anchor kind {!r}".format(kind))

        if tstar is None:
            ep.anchor_step = None
            ep.anchor_kind = None
            ep.tau = None
            n_none += 1
            continue
        tau = np.arange(ep.length, dtype=np.int32) - int(tstar)
        if require_min_pre and kind == "commitment" and int((tau < 0).sum()) < commit_min_pre:
            ep.anchor_step = None
            ep.anchor_kind = None
            ep.tau = None
            n_none += 1
            continue
        ep.anchor_step = int(tstar)
        ep.anchor_kind = used_kind
        ep.tau = tau
        n_ok += 1

    anchored = [e for e in episodes if e.anchor_step is not None]
    dropped = [e for e in episodes if e.anchor_step is None]
    steps = [int(e.anchor_step) for e in anchored]
    pre = [int((e.tau < 0).sum()) for e in anchored]
    post = [int((e.tau > 0).sum()) for e in anchored]
    return {
        "kind": kind,
        "late_onset_only": bool(late_onset_only),
        "n_episodes": len(episodes),
        "n_anchored": n_ok,
        "n_unanchored": n_none,
        "n_late_onset_dropped": n_late_drop,
        "anchor_rate": float(n_ok / max(len(episodes), 1)),
        "outcomes": {
            k: int(sum(1 for e in episodes if e.outcome == k))
            for k in ("success", "goal_misgen", "died", "timed_out")
        },
        "gm_rate_all": _rate(episodes, "y_gm"),
        "gm_rate_anchored": _rate(anchored, "y_gm"),
        "gm_rate_dropped": _rate(dropped, "y_gm"),
        "got_coin_rate_anchored": _rate(anchored, "got_coin"),
        "got_coin_rate_dropped": _rate(dropped, "got_coin"),
        "anchor_step": _summarize_ints(steps),
        "pre_tau_len": _summarize_ints(pre),
        "post_tau_len": _summarize_ints(post),
        "frac_anchor_at_0": float(np.mean([s == 0 for s in steps])) if steps else None,
        "commit_params": {
            "W": commit_W,
            "theta": commit_theta,
            "R": commit_R,
            "min_pre": commit_min_pre,
        }
        if kind == "commitment"
        else None,
    }


def _rate(eps, attr):
    vals = [getattr(e, attr) for e in eps if getattr(e, attr) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _summarize_ints(xs):
    if not xs:
        return None
    a = np.asarray(xs, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _unit(w):
    w = np.asarray(w, dtype=np.float64).ravel()
    return w / max(float(np.linalg.norm(w)), 1e-12)


def _e5(h, w, n_pcs=5):
    if h.shape[0] < n_pcs + 2:
        return float("nan")
    hc = h - h.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(hc, full_matrices=False)
    return float(np.sum((vt[:n_pcs] @ _unit(w)) ** 2))


def _logistic_beta(conf, y, *, balanced=False, n_perm=0, seed=0):
    """1-d logistic P(y=1)~conf. Optional class balancing + permutation null."""
    conf = np.asarray(conf, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    out = {
        "beta": float("nan"),
        "n": int(len(y)),
        "n_pos": int((y > 0.5).sum()),
        "n_neg": int((y <= 0.5).sum()),
        "pearson_r": float("nan"),
        "auc": float("nan"),
        "p_perm": float("nan"),
    }
    if len(y) < 8 or len(np.unique(y)) < 2:
        return out
    c = (conf - conf.mean()) / (float(conf.std()) + 1e-12)
    sw = None
    if balanced:
        n_pos = max(float((y > 0.5).sum()), 1.0)
        n_neg = max(float((y <= 0.5).sum()), 1.0)
        sw = np.where(y > 0.5, 0.5 / n_pos, 0.5 / n_neg).astype(np.float64)
    beta = _irls_logistic(c, y, sample_weight=sw)
    out["beta"] = float(beta[1])
    out["intercept"] = float(beta[0])
    out["pearson_r"] = float(np.corrcoef(conf, y)[0, 1]) if y.std() > 0 else float("nan")
    out["auc"] = roc_auc(y, conf)
    if n_perm > 0:
        rng = np.random.RandomState(seed)
        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            if len(np.unique(yp)) < 2:
                continue
            bp = _irls_logistic(c, yp, sample_weight=sw if not balanced else (
                np.where(yp > 0.5, 0.5 / max(float((yp > 0.5).sum()), 1.0),
                         0.5 / max(float((yp <= 0.5).sum()), 1.0))
            ))
            null.append(bp[1])
        null = np.asarray(null, dtype=np.float64)
        if len(null):
            out["p_perm"] = float((np.abs(null) >= abs(out["beta"])).mean())
            out["null_beta_std"] = float(null.std())
    return out


def _irls_logistic(c, y, sample_weight=None):
    X = np.column_stack([np.ones(len(c)), c])
    beta = np.zeros(2)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    for _ in range(25):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = w * (p * (1 - p) + 1e-6)
        try:
            beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z + w * (y - p)))
        except np.linalg.LinAlgError:
            break
    return beta


def _logistic_beta_2d(conf_coin, conf_pc2, y, *, balanced=False):
    """P(y=1) ~ conf_coin + conf_pc2. Returns β_coin (partialling progress in-model)."""
    c1 = np.asarray(conf_coin, dtype=np.float64).ravel()
    c2 = np.asarray(conf_pc2, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 8 or len(np.unique(y)) < 2:
        return {"beta_coin": float("nan"), "beta_pc2": float("nan"), "n": int(len(y))}
    z1 = (c1 - c1.mean()) / (float(c1.std()) + 1e-12)
    z2 = (c2 - c2.mean()) / (float(c2.std()) + 1e-12)
    X = np.column_stack([np.ones(len(y)), z1, z2])
    beta = np.zeros(3)
    sw = np.ones(len(y))
    if balanced:
        n_pos = max(float((y > 0.5).sum()), 1.0)
        n_neg = max(float((y <= 0.5).sum()), 1.0)
        sw = np.where(y > 0.5, 0.5 / n_pos, 0.5 / n_neg).astype(np.float64)
    for _ in range(25):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = sw * (p * (1 - p) + 1e-6)
        try:
            beta = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z + sw * (y - p)))
        except np.linalg.LinAlgError:
            break
    return {
        "beta_coin": float(beta[1]),
        "beta_pc2": float(beta[2]),
        "intercept": float(beta[0]),
        "n": int(len(y)),
    }


def _pc2_progress_direction(episodes):
    hs, progress = [], []
    for ep in episodes:
        T = ep.length
        for t in range(T):
            hs.append(ep.embeddings[t])
            progress.append(t / max(T - 1, 1))
    H = np.stack(hs, axis=0)
    Hc = H - H.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Hc, full_matrices=False)
    pc2 = vt[1]
    proj = Hc @ pc2
    if float(np.corrcoef(proj, np.asarray(progress))[0, 1]) < 0:
        pc2 = -pc2
    return pc2.astype(np.float64)


def y_coding_contingency(episodes, w_coin_u, pc2=None, tau_window=2):
    """Check that a sign flip of β across τ reflects behaviour rather than label coding.

    Reports the contingency of got_coin / success / reached_end and β under each label at τ≈0."""
    labeled = [e for e in episodes if e.got_coin is not None and e.anchor_step is not None]
    if len(labeled) < 8:
        return {"error": "too few labeled anchored episodes", "n": len(labeled)}

    got = np.array([e.got_coin for e in labeled], dtype=int)
    succ = np.array([e.success for e in labeled], dtype=int)
    end = np.array([e.reached_end for e in labeled], dtype=int)
    confs = []
    for e in labeled:
        mask = np.abs(e.tau) <= tau_window
        if not np.any(mask):
            idx = int(np.argmin(np.abs(e.tau)))
            h = e.embeddings[idx : idx + 1]
        else:
            h = e.embeddings[mask]
        confs.append(float(h.mean(axis=0) @ w_coin_u))
    confs = np.asarray(confs)

    table = {}
    for g in (0, 1):
        for s in (0, 1):
            table["got{}_succ{}".format(g, s)] = int(((got == g) & (succ == s)).sum())

    result = {
        "n": len(labeled),
        "P_got_coin": float(got.mean()),
        "P_success": float(succ.mean()),
        "P_reached_end": float(end.mean()),
        "P_got_coin_eq_success": float((got == succ).mean()),
        "P_reached_end_eq_got_coin": float((end == got).mean()),
        "contingency_got_vs_success": table,
        "note": (
            "On OOD, success := collected relocated coin, so got_coin==success "
            "for success/GM episodes by construction. y_gm = 1-got_coin. "
            "A real flip must appear under got_coin (fixed semantics)."
        ),
        "beta_at_tau0": {},
    }
    for name, y in (("got_coin", got), ("success", succ), ("reached_end", end), ("y_gm", 1 - got)):
        result["beta_at_tau0"][name] = _logistic_beta(
            confs, y.astype(np.float64), balanced=True, n_perm=500, seed=0
        )
    if pc2 is not None:
        conf_pc2 = []
        for e in labeled:
            mask = np.abs(e.tau) <= tau_window
            if not np.any(mask):
                idx = int(np.argmin(np.abs(e.tau)))
                h = e.embeddings[idx : idx + 1]
            else:
                h = e.embeddings[mask]
            conf_pc2.append(float(h.mean(axis=0) @ pc2))
        result["beta_got_coin_with_pc2"] = _logistic_beta_2d(
            confs, np.asarray(conf_pc2), got.astype(np.float64), balanced=True
        )
    return result


def analyze_coinrun_phase(
    episodes,
    *,
    y_label="got_coin",
    tau_min=-40,
    tau_max=40,
    tau_bin=5,
    ridge_alpha=1.0,
    balanced=True,
    n_perm_tau0=500,
):
    """Observational phase analysis on anchored episodes; ``y_label='got_coin'`` has regime-invariant meaning."""
    anchored = [e for e in episodes if e.anchor_step is not None and e.tau is not None]
    labeled = [e for e in anchored if getattr(e, y_label) is not None]
    if len(anchored) < 10:
        return {"error": "too few anchored episodes", "n_anchored": len(anchored)}

    H_all = np.concatenate([e.embeddings for e in anchored], axis=0)
    y_coin = np.concatenate([e.coin_present for e in anchored], axis=0)
    if y_coin.sum() < 5 or (len(y_coin) - y_coin.sum()) < 5:
        return {"error": "coin label imbalance", "coin_pos_rate": float(y_coin.mean())}
    w_coin, b_coin = fit_ridge(H_all, y_coin.astype(np.float64), alpha=ridge_alpha)
    w_coin_u = _unit(w_coin)
    pc2 = _pc2_progress_direction(anchored)

    y_check = y_coding_contingency(episodes, w_coin_u, pc2=pc2)

    # Abstractness: level holdout
    levels = np.asarray([e.level for e in anchored])
    med_level = float(np.median(levels))
    train_eps = [e for e in anchored if e.level <= med_level]
    test_eps = [e for e in anchored if e.level > med_level]
    abstractness = {"median_level_split": med_level}
    if train_eps and test_eps:
        Ht = np.concatenate([e.embeddings for e in train_eps], axis=0)
        yt = np.concatenate([e.coin_present for e in train_eps], axis=0).astype(np.float64)
        Hv = np.concatenate([e.embeddings for e in test_eps], axis=0)
        yv = np.concatenate([e.coin_present for e in test_eps], axis=0).astype(np.float64)
        if yt.sum() > 2 and (len(yt) - yt.sum()) > 2 and yv.sum() > 2:
            wt, bt = fit_ridge(Ht, yt, alpha=ridge_alpha)
            abstractness["auc_train"] = roc_auc(yt, predict_ridge(Ht, wt, bt))
            abstractness["auc_heldout_levels"] = roc_auc(yv, predict_ridge(Hv, wt, bt))
            abstractness["n_train_frames"] = int(len(yt))
            abstractness["n_heldout_frames"] = int(len(yv))

    edges = list(range(tau_min, tau_max + 1, tau_bin))
    if edges[-1] <= tau_max:
        edges.append(tau_max + 1)
    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        confs, ys, conf_pc2, H_bin, y_coin_bin = [], [], [], [], []
        for e in labeled:
            mask = (e.tau >= lo) & (e.tau < hi)
            if not np.any(mask):
                continue
            h = e.embeddings[mask]
            confs.append(float(h.mean(axis=0) @ w_coin_u))
            conf_pc2.append(float(h.mean(axis=0) @ pc2))
            ys.append(getattr(e, y_label))
            H_bin.append(h)
            y_coin_bin.append(e.coin_present[mask])
        row = {
            "tau_lo": lo,
            "tau_hi": hi,
            "tau_mid": 0.5 * (lo + hi - 1),
            "n_episodes": len(ys),
            "beta_coin": float("nan"),
            "beta_coin_balanced": float("nan"),
            "beta_coin_given_pc2": float("nan"),
            "beta_pc2": float("nan"),
            "auc_presence": float("nan"),
            "e5_coin": float("nan"),
        }
        if len(ys) < 8:
            bins.append(row)
            continue
        confs_a = np.asarray(confs)
        ys_a = np.asarray(ys, dtype=np.float64)
        c2 = np.asarray(conf_pc2)
        rel = _logistic_beta(confs_a, ys_a, balanced=False)
        rel_b = _logistic_beta(confs_a, ys_a, balanced=True)
        joint = _logistic_beta_2d(confs_a, c2, ys_a, balanced=balanced)
        Hcat = np.concatenate(H_bin, axis=0)
        ycat = np.concatenate(y_coin_bin, axis=0).astype(np.float64)
        if ycat.sum() > 2 and (len(ycat) - ycat.sum()) > 2:
            auc_pres = roc_auc(ycat, predict_ridge(Hcat, w_coin, b_coin))
        else:
            auc_pres = float("nan")
        row.update(
            {
                "beta_coin": rel["beta"],
                "beta_coin_balanced": rel_b["beta"],
                "pearson_r": rel["pearson_r"],
                "auc_conf_vs_y": rel["auc"],
                "beta_coin_given_pc2": joint["beta_coin"],
                "beta_pc2": joint["beta_pc2"],
                "auc_presence": auc_pres,
                "coin_pos_rate_in_bin": float(ycat.mean()) if len(ycat) else float("nan"),
                "e5_coin": _e5(Hcat, w_coin_u),
                "mean_conf": float(confs_a.mean()),
                "y_rate": float(ys_a.mean()),
                "n_pos": int((ys_a > 0.5).sum()),
                "n_neg": int((ys_a <= 0.5).sum()),
            }
        )
        bins.append(row)

    finite = [b for b in bins if np.isfinite(b.get("beta_coin_given_pc2", np.nan))]
    temporal = {}
    if len(finite) >= 4:
        mids = np.asarray([b["tau_mid"] for b in finite])
        betas = np.asarray([abs(b["beta_coin_given_pc2"]) for b in finite])
        prox = -np.abs(mids)
        r_mid = float(
            np.corrcoef(np.argsort(np.argsort(prox)), np.argsort(np.argsort(betas)))[0, 1]
        )
        near = [b for b in finite if abs(b["tau_mid"]) <= tau_bin]
        far = [b for b in finite if abs(b["tau_mid"]) > 2 * tau_bin]
        temporal = {
            "spearman_abs_beta_vs_proximity_to_zero": r_mid,
            "beta_near_zero": float(np.nanmean([b["beta_coin_given_pc2"] for b in near]))
            if near
            else float("nan"),
            "beta_far": float(np.nanmean([b["beta_coin_given_pc2"] for b in far]))
            if far
            else float("nan"),
            "n_bins_finite": len(finite),
            "n_bins_with_tau_neg": int(sum(1 for b in finite if b["tau_mid"] < 0)),
            "n_bins_with_tau_pos": int(sum(1 for b in finite if b["tau_mid"] > 0)),
        }

    # τ≈0 summary under got_coin with perm null (even if primary y differs)
    tau0 = y_check.get("beta_at_tau0", {}).get("got_coin", {})

    return {
        "n_anchored": len(anchored),
        "n_labeled": len(labeled),
        "y_label": y_label,
        "pooled_coin_auc": roc_auc(y_coin, predict_ridge(H_all, w_coin, b_coin)),
        "pooled_coin_pos_rate": float(y_coin.mean()),
        "y_coding": y_check,
        "abstractness": abstractness,
        "tau_bins": bins,
        "temporal_pattern": temporal,
        "tau0_got_coin_balanced": tau0,
        "probe": "ridge_coin",
        "ridge_alpha": ridge_alpha,
        "balanced": balanced,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "tau_bin": tau_bin,
        "n_perm_tau0": n_perm_tau0,
    }
