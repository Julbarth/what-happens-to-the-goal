#!/usr/bin/env python3
"""Score env-move effects (signed follow, TV, KL) per agent for Coin, Attribute Maze and Maze."""
from __future__ import annotations

import os
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
    ROOT / "scripts" / "procgen",
    ROOT / "pkg",
    ROOT / "run_scripts",
    ROOT / "external" / "train-procgen-pytorch",
    ROOT / "vendored" / "train-procgen-pytorch",
    ROOT / "external" / "procgenAISC",
    ROOT / "vendored" / "procgenAISC",
):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)

from gm_geometry.utils import ci95  # noqa: E402
from gm_geometry.utils import device as pick_device  # noqa: E402


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def _clear_procgen():
    for mod in list(sys.modules):
        if mod == "procgen" or mod.startswith("procgen."):
            del sys.modules[mod]


def prefer_procgen(kind: str, bundle: Path | None):
    """Force coin (vanilla) or cs (colour-shape) procgenAISC to the front of sys.path."""
    if bundle is None:
        return
    roots = {
        "coin": bundle / "vendored" / "procgenAISC",
        "cs": bundle / "vendored" / "cs-procgenAISC",
    }
    prefer = roots[kind]
    if not prefer.exists():
        return
    _clear_procgen()
    drop = {"procgenAISC", "cs-procgenAISC"}
    cleaned = []
    for p in sys.path:
        if any(d in p for d in drop):
            continue
        cleaned.append(p)
    sys.path[:] = [str(prefer)] + cleaned


def _mean_se(xs):
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    _mean, lo, hi = ci95(a)
    return {
        "mean": float(np.mean(a)),
        "se": float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
        "ci95": [lo, hi],
        "n": int(len(a)),
    }


def _robust(xs):
    """Mean+SE/CI plus median/trim10/p90/max — mean KL is sensitive to near-deterministic
    mass on rare actions, so the robust stats are the primary read for KL tails."""
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    out = _mean_se(a)
    if a.size == 0:
        out.update(
            {
                "median": float("nan"),
                "trim10_mean": float("nan"),
                "p90": float("nan"),
                "max": float("nan"),
            }
        )
        return out
    out["median"] = float(np.median(a))
    out["p90"] = float(np.quantile(a, 0.90))
    out["max"] = float(np.max(a))
    if a.size >= 10:
        lo, hi = np.quantile(a, [0.10, 0.90])
        trimmed = a[(a >= lo) & (a <= hi)]
        out["trim10_mean"] = float(np.mean(trimmed)) if trimmed.size else float(np.mean(a))
    else:
        out["trim10_mean"] = float(np.mean(a))
    return out


def tv_half(p, q):
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return 0.5 * float(np.abs(p - q).sum())


def kl_pq(p, q):
    from gm_geometry.probes.fisher_sensitivity import kl_categorical

    return kl_categorical(p, q)


def pack_pair(signed, tv, kl):
    return {
        "signed_follow": _mean_se(signed),
        "tv": _robust(tv),
        "kl": _robust(kl),
        "note": "signed_follow = note env_move_ceiling; tv=½L1; kl=KL(π||π')",
    }


# ----------------------------- Coin -----------------------------
def score_coin(ckpt: Path, *, n_frames: int, start_level: int, max_levels: int, device: str, bundle: Path | None):
    prefer_procgen("coin", bundle)
    from run_coin_goal_proxy_ladder import LEFT_ACTIONS, Ladder, collect
    from validate_langosco_coinrun_200m import load_policy

    def mass(probs, actions):
        return float(np.asarray(probs, dtype=np.float64).reshape(-1)[list(actions)].sum())

    policy = load_policy(ckpt, device)
    lad = Ladder(policy, device)
    rng = np.random.default_rng(0)
    frames = collect(
        lad,
        n_target=n_frames,
        start_level=start_level,
        max_levels=max_levels,
        capture_causal=True,
        rng=rng,
        verbose=True,
    )
    signed, tvs, kls, dxs = [], [], [], []
    per_frame = []
    for f in frames:
        pa, pb = f["probs_a"], f["probs_b"]
        s = mass(pb, LEFT_ACTIONS) - mass(pa, LEFT_ACTIONS)
        t = tv_half(pa, pb)
        k = kl_pq(pa, pb)
        dx = float(f["coin_x_r"] - f["coin_x_l"])
        signed.append(s)
        tvs.append(t)
        kls.append(k)
        dxs.append(dx)
        per_frame.append(
            {
                "level": int(f["level"]),
                "signed": s,
                "tv": t,
                "kl": k,
                "coin_dx": dx,
                "coin_x_r": int(f["coin_x_r"]),
                "coin_x_l": int(f["coin_x_l"]),
            }
        )
    metrics = pack_pair(signed, tvs, kls)
    metrics["coin_dx"] = _robust(dxs)
    metrics["protocol"] = {
        "edit": "set_coin right-floor → left-floor (min_dx=2 each side; same as ladder collect)",
        "start_level": start_level,
        "n_frames": len(frames),
        "rng": "default_rng(0) for rollout_k",
    }
    return {
        "env": "coin",
        "feature": "coin_right_to_left",
        "ckpt": str(ckpt),
        "n_frames": len(frames),
        "metrics": metrics,
        "per_frame": per_frame,
    }


# ----------------------------- Ras -----------------------------
def score_ras(
    ckpt: Path, *, n_frames: int, start_level: int, max_levels: int, device: str,
    bundle: Path | None, use_backgrounds: bool = False,
):
    """use_backgrounds must match how `ckpt` was trained: True for textured, False (default) for black background."""
    prefer_procgen("cs", bundle)
    from gm_geometry.integrations.colour_shape import make_duel_env
    from gm_geometry.integrations.colour_shape.policy import (
        load_impala_policy,
        load_recurrent_flag,
    )
    from run_ras_maze_goal_proxy_ladder import Ladder, collect
    from ras_graph_exact_signed_follow import graph_exact_first_action_set

    probe = make_duel_env(
        1, 1, 1, "red_line_diag", "yellow_gem", start_level=start_level,
        use_backgrounds=use_backgrounds,
    )
    recurrent = load_recurrent_flag()
    in_ch = probe.observation_space.shape[0]
    n_act = probe.action_space.n
    if hasattr(probe, "close"):
        probe.close()
    policy = load_impala_policy(ckpt, in_ch, n_act, recurrent)
    lad = Ladder(policy, device)
    frames = collect(
        lad,
        n_target=n_frames,
        start_level=start_level,
        max_levels=max_levels,
        max_steps=40,
        obj1="red_line_diag",
        obj2="yellow_gem",
        greedy=True,
        verbose=True,
        use_backgrounds=use_backgrounds,
    )
    out = {
        "env": "ras", "ckpt": str(ckpt), "n_frames": len(frames),
        "use_backgrounds": use_backgrounds, "features": {},
    }
    for feat in ("line", "yellow"):
        signed, tvs, kls = [], [], []
        per_frame = []
        n_skipped_no_action_set = 0
        for f in frames:
            if f.get(feat) is None or f[feat].get("moved_probs") is None:
                continue
            base = f["base_probs"]
            moved = f[feat]["moved_probs"]
            # TV/KL use the whole action distribution, so they are computed for every frame,
            # even when the graph-exact action-set lookup below fails.
            t = tv_half(base, moved)
            k = kl_pq(base, moved)
            tvs.append(t)
            kls.append(k)
            row = {"level": int(f["level"]), "tv": t, "kl": k}

            agent = f["agent_xy"]
            new_xy = f[feat]["moved_xy"]
            # Signed follow uses the set of actions that reduce BFS distance to new_xy's cell; a straight-line
            # bearing action ignores walls and agrees poorly with greedy behaviour in this maze.
            action_set = graph_exact_first_action_set(
                level=int(f["level"]), obj1="red_line_diag", obj2="yellow_gem",
                use_backgrounds=use_backgrounds, agent_xy=agent, target_xy=new_xy,
            )
            if not action_set:
                n_skipped_no_action_set += 1
            else:
                idx = list(action_set)
                s = float(np.sum(np.asarray(moved)[idx]) - np.sum(np.asarray(base)[idx]))
                signed.append(s)
                row["signed"] = s
                row["action_set"] = sorted(idx)
            per_frame.append(row)
        packed = pack_pair(signed, tvs, kls)
        packed["n_usable"] = len(signed)
        packed["n_usable_tv_kl"] = len(tvs)
        packed["n_skipped_no_graph_exact_action_set"] = n_skipped_no_action_set
        packed["per_frame"] = per_frame
        out["features"][feat] = packed
    return out


def score_maze(ckpt: Path, *, n_frames: int, start_level: int, max_levels: int, device: str, bundle: Path | None):
    prefer_procgen("coin", bundle)
    # external/ checkout first, then a vendored copy.
    for cand in (
        ROOT / "external" / "train-procgen-pytorch",
        ROOT / "vendored" / "train-procgen-pytorch",
        (bundle / "external" / "train-procgen-pytorch") if bundle else None,
        (bundle / "vendored" / "train-procgen-pytorch") if bundle else None,
    ):
        if cand and cand.exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    import run_maze_i_goal_proxy_ladder as maze_lad

    # Point ladder TRAIN_ROOT at a train-procgen checkout that exists.
    for cand in (
        ROOT / "external" / "train-procgen-pytorch",
        ROOT / "vendored" / "train-procgen-pytorch",
        (bundle / "external" / "train-procgen-pytorch") if bundle else None,
        (bundle / "vendored" / "train-procgen-pytorch") if bundle else None,
    ):
        if cand and (cand / "hyperparams/procgen/config.yml").exists():
            maze_lad.TRAIN_ROOT = cand
            break

    from run_maze_i_goal_proxy_ladder import Ladder, action_toward, collect, load_policy

    policy = load_policy(ckpt, device)
    lad = Ladder(policy, device)
    frames = collect(
        lad,
        n_target=n_frames,
        start_level=start_level,
        max_levels=max_levels,
        max_steps=200,
        greedy=True,
        verbose=True,
        capture_causal=True,
    )
    out = {"env": "maze", "ckpt": str(ckpt), "n_frames": len(frames), "features": {}}

    signed, tvs, kls = [], [], []
    per_frame = []
    n_skipped_no_graph_exact_action = 0
    n_grid_mismatch = 0
    n_old_new_compared = 0
    n_old_new_agree = 0
    for f in frames:
        cc = f.get("cheese_causal") or {}
        if cc.get("moved_probs") is None:
            continue
        base, moved = f["base_probs"], cc["moved_probs"]
        # TV/KL use the whole action distribution, so they are computed for every frame,
        # even when no graph-exact action exists.
        t = tv_half(base, moved)
        k = kl_pq(base, moved)
        tvs.append(t)
        kls.append(k)
        row = {"level": int(f["level"]), "tv": t, "kl": k}

        # a_cheese_post is the first BFS step from the agent cell to the relocated cheese cell
        # (capture_cheese_donors). None means the cell is unreachable or grid_agent_matched failed.
        # The pixel-bearing action is kept only to report its agreement with a_cheese_post.
        a_new = cc.get("a_cheese_post")
        if cc.get("moved_xy") is not None:
            a_old = action_toward(f["agent_xy"], cc["moved_xy"])
            row["a_old_bearing"] = None if a_old is None else int(a_old)
            if a_old is not None and a_new is not None:
                n_old_new_compared += 1
                if int(a_old) == int(a_new):
                    n_old_new_agree += 1
        if cc.get("grid_agent_matched") is False:
            n_grid_mismatch += 1
        if a_new is None:
            n_skipped_no_graph_exact_action += 1
        else:
            s = float(moved[a_new] - base[a_new])
            signed.append(s)
            row["signed"] = s
            row["a_cheese_post"] = int(a_new)
        per_frame.append(row)
    packed = pack_pair(signed, tvs, kls)
    packed["n_usable"] = len(signed)
    packed["n_usable_tv_kl"] = len(tvs)
    packed["n_skipped_no_graph_exact_action"] = n_skipped_no_graph_exact_action
    packed["n_grid_agent_mismatch"] = n_grid_mismatch
    packed["n_old_new_compared"] = n_old_new_compared
    packed["old_vs_new_label_agreement"] = (
        float(n_old_new_agree / n_old_new_compared) if n_old_new_compared else float("nan")
    )
    packed["per_frame"] = per_frame
    packed["instrument"] = ("graph-exact BFS first-step Δπ on a_cheese_post; "
                             "TV/KL on full π(base) vs π(moved)")
    out["features"]["cheese"] = packed

    signed, tvs, kls = [], [], []
    per_frame = []
    for f in frames:
        cg = f.get("corner_geometry") or {}
        if cg.get("opened_probs") is None:
            continue
        a_post = cg.get("a_corner_post")
        if a_post is None:
            continue
        a_post = int(a_post)
        base, opened = f["base_probs"], cg["opened_probs"]
        s = float(opened[a_post] - base[a_post])
        t = tv_half(base, opened)
        k = kl_pq(base, opened)
        signed.append(s)
        tvs.append(t)
        kls.append(k)
        per_frame.append({"level": int(f["level"]), "signed": s, "tv": t, "kl": k})
    packed = pack_pair(signed, tvs, kls)
    packed["n_usable"] = len(signed)
    packed["per_frame"] = per_frame
    packed["instrument"] = "open-TR Δπ on a_corner_post; TV/KL on full π(base) vs π(opened)"
    out["features"]["corner_open_TR"] = packed
    return out


def _maze_nongm_extra():
    """Non-GM Maze 1091/1092, if present in the released checkpoint layout."""
    base = DATA_ROOT / "checkpoints" / "maze"
    return [("maze", f"{s}_teach", base / f"{s}_teach.pth") for s in (1091, 1092) if (base / f"{s}_teach.pth").exists()]


def default_agents(bundle_root: Path | None):
    """Prefer the released checkpoint layout (<root>/checkpoints/{coin,maze,ras}); else raw training-log paths."""
    agents = []
    if bundle_root and (bundle_root / "checkpoints").exists():
        ck = bundle_root / "checkpoints"
        # Late GM checkpoints (200M).
        for seed in (6033, 6034, 6035):
            p = ck / "coin" / f"{seed}_gm.pth"
            if p.exists():
                agents.append(("coin", f"{seed}_gm", p))
        # Phase-matched GM checkpoints: 6033@20M, 6035@40M, 6037@40M.
        for lab, fname in (
            ("6033_gm_20M", "6033_gm_20M.pth"),
            ("6035_gm_40M", "6035_gm_40M.pth"),
            ("6037_gm_40M", "6037_gm_40M.pth"),
        ):
            p = ck / "coin" / fname
            if p.exists():
                agents.append(("coin", lab, p))
        # Paper Coin GM roster: 6033@20M (above), 6037@120M, 6051@55M.
        for lab in ("6037_gm_120M", "6051_gm_55M"):
            p = ck / "coin" / f"{lab}.pth"
            if p.exists():
                agents.append(("coin", lab, p))
        for seed in (7001, 7002, 7003):
            p = ck / "coin" / f"{seed}_teach.pth"
            if p.exists():
                agents.append(("coin", f"{seed}_teach", p))
        for seed in (6311, 6312, 6313):
            p = ck / "ras" / f"{seed}_gm.pth"
            if p.exists():
                agents.append(("ras", f"{seed}_gm", p))
        for seed in (4371, 4372, 4373):
            p = ck / "ras" / f"{seed}_teach.pth"
            if p.exists():
                agents.append(("ras", f"{seed}_teach", p))
        for seed in (1080, 1081, 1082):
            p = ck / "maze" / f"{seed}_gm.pth"
            if p.exists():
                agents.append(("maze", f"{seed}_gm", p))
        p = ck / "maze" / "1090_teach.pth"
        if p.exists():
            agents.append(("maze", "1090_teach", p))
        agents.extend(_maze_nongm_extra())
        return agents

    root = DATA_ROOT / "checkpoints" / "train_logs"
    mapping = [
        ("coin", "6033_gm", root / "coinrun/coinrun-200m-langosco/2026-07-15__23-37-27__seed_6033/model_200015872.pth"),
        ("coin", "6034_gm", root / "coinrun/coinrun-200m-langosco/2026-07-29__23-53-54__seed_6034/model_200015872.pth"),
        ("coin", "6035_gm", root / "coinrun/coinrun-200m-langosco/2026-07-29__23-54-20__seed_6035/model_200015872.pth"),
        (
            "coin",
            "6033_gm_20M",
            root / "coinrun/coinrun-200m-langosco/2026-07-15__23-37-27__seed_6033/model_20054016.pth",
        ),
        (
            "coin",
            "6035_gm_40M",
            root / "coinrun/coinrun-200m-langosco/2026-07-29__23-54-20__seed_6035/model_40042496.pth",
        ),
        (
            "coin",
            "6037_gm_40M",
            root / "coinrun/coinrun-200m-langosco/2026-08-03__21-27-03__seed_6037/model_40042496.pth",
        ),
        ("coin", "7001_teach", root / "coinrun/coinrun-aisc-ceiling-short/2026-08-01__16-02-22__seed_7001/model_40042496.pth"),
        ("coin", "7002_teach", root / "coinrun/coinrun-aisc-ceiling-short/2026-08-06__22-44-33__seed_7002/model_40042496.pth"),
        ("coin", "7003_teach", root / "coinrun/coinrun-aisc-ceiling-short/2026-08-07__08-47-08__seed_7003/model_40042496.pth"),
        ("ras", "6311_gm", root / "maze-confounded-duel-black-bg/2026-07-14__21-34-06__seed_6311/model_5046272.pth"),
        ("ras", "6312_gm", root / "maze-confounded-duel-black-bg/2026-07-14__22-19-16__seed_6312/model_5046272.pth"),
        ("ras", "6313_gm", root / "maze-confounded-duel-black-bg/2026-07-14__23-01-00__seed_6313/model_5046272.pth"),
        ("maze", "1080_gm", root / "maze_aisc/maze-i-200m-langosco/2026-07-15__23-37-09__seed_1080/model_200015872.pth"),
        ("maze", "1081_gm", root / "maze_aisc/maze-i-200m-langosco/2026-07-15__23-37-09__seed_1081/model_200015872.pth"),
        ("maze", "1082_gm", root / "maze_aisc/maze-i-200m-langosco/2026-07-15__23-37-09__seed_1082/model_200015872.pth"),
        ("maze", "1090_teach", root / "maze_aisc/maze-i-repaired-200m/2026-07-29__23-55-03__seed_1090/model_200015872.pth"),
    ]
    agents = [(e, lab, p) for e, lab, p in mapping if p.exists()]
    agents.extend(_maze_nongm_extra())
    return agents


def _fmt_kl(m):
    return (
        f"{m['mean']:.4f} (med {m.get('median', float('nan')):.4f}, "
        f"trim10 {m.get('trim10_mean', float('nan')):.4f}, "
        f"p90 {m.get('p90', float('nan')):.4f})"
    )


def write_summary_md(rows, path: Path):
    lines = [
        "# Env-move TV + KL (re-score)",
        "",
        "Signed follow = note `env_move_ceiling`. TV = ½‖π−π'‖₁. KL = KL(π‖π').",
        "KL cells show mean (median, trim10, p90).",
        "",
        "| env | agent | feature | signed | TV mean | KL mean (med / trim10 / p90) | n |",
        "|-----|-------|---------|-------:|--------:|-----------------------------:|--:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r.get('env','?')} | {r.get('label','?')} | ERROR | | | {r['error'][:60]} | |")
            continue
        env = r["env"]
        lab = r["label"]
        if env == "coin":
            m = r["metrics"]
            lines.append(
                f"| {env} | {lab} | coin | {m['signed_follow']['mean']:+.4f} | "
                f"{m['tv']['mean']:.4f} | {_fmt_kl(m['kl'])} | {m['signed_follow']['n']} |"
            )
            if "coin_dx" in m:
                dx = m["coin_dx"]
                lines.append(
                    f"| {env} | {lab} | coin_dx | mean={dx['mean']:.2f} | "
                    f"med={dx.get('median', float('nan')):.2f} | "
                    f"p90={dx.get('p90', float('nan')):.2f} | {dx['n']} |"
                )
        else:
            for feat, m in r.get("features", {}).items():
                if "signed_follow" not in m:
                    continue
                lines.append(
                    f"| {env} | {lab} | {feat} | {m['signed_follow']['mean']:+.4f} | "
                    f"{m['tv']['mean']:.4f} | {_fmt_kl(m['kl'])} | {m['signed_follow']['n']} |"
                )
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="coin,ras,maze", help="comma list: coin,ras,maze")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=str(pick_device()))
    # Defaults match the Coin ladder.
    ap.add_argument("--n-frames", type=int, default=140)
    ap.add_argument("--start-level", type=int, default=2_000_000)
    ap.add_argument("--ras-start-level", type=int, default=1_000_000)
    ap.add_argument("--maze-start-level", type=int, default=1_000_000)
    ap.add_argument("--max-levels", type=int, default=8000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bundle-root", type=Path, default=None)
    ap.add_argument(
        "--only-labels",
        default="",
        help="comma list of agent labels to keep (e.g. 1091_teach,1092_teach)",
    )
    ap.add_argument(
        "--ras-use-backgrounds", action="store_true",
        help="Must match how the Attribute Maze checkpoints were trained: pass for textured-background "
             "checkpoints, omit for black-background ones (the default population).",
    )
    args = ap.parse_args()
    if args.quick:
        args.n_frames, args.max_levels = 16, 400
    want = {x.strip() for x in args.envs.split(",") if x.strip()}
    only = {x.strip() for x in args.only_labels.split(",") if x.strip()}
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    bundle = args.bundle_root
    if bundle is None:
        for cand in (DATA_ROOT,):
            if (cand / "checkpoints").exists():
                bundle = cand
                break

    args.out.mkdir(parents=True, exist_ok=True)
    agents = [(e, lab, p) for e, lab, p in default_agents(bundle) if e in want]
    if only:
        agents = [(e, lab, p) for e, lab, p in agents if lab in only]
    print(
        f"device={device} n_agents={len(agents)} envs={sorted(want)} "
        f"labels={[lab for _, lab, _ in agents]} "
        f"n_frames={args.n_frames} coin_start={args.start_level}",
        flush=True,
    )

    rows = []
    t0 = time.time()
    for env, lab, ckpt in agents:
        print(f"\n=== {env} {lab} {ckpt} ===", flush=True)
        try:
            if env == "coin":
                r = score_coin(
                    ckpt,
                    n_frames=args.n_frames,
                    start_level=args.start_level,
                    max_levels=args.max_levels,
                    device=device,
                    bundle=bundle,
                )
            elif env == "ras":
                r = score_ras(
                    ckpt,
                    n_frames=args.n_frames,
                    start_level=args.ras_start_level,
                    max_levels=args.max_levels,
                    device=device,
                    bundle=bundle,
                    use_backgrounds=args.ras_use_backgrounds,
                )
            else:
                r = score_maze(
                    ckpt,
                    n_frames=args.n_frames,
                    start_level=args.maze_start_level,
                    max_levels=args.max_levels,
                    device=device,
                    bundle=bundle,
                )
            r["label"] = lab
            rows.append(r)
            # Drop bulky per_frame from the printed preview but keep on disk.
            slim = {k: v for k, v in r.items() if k != "per_frame"}
            if "features" in slim:
                slim_feats = {}
                for fk, fv in slim["features"].items():
                    slim_feats[fk] = {kk: vv for kk, vv in fv.items() if kk != "per_frame"}
                slim["features"] = slim_feats
            (args.out / f"{env}_{lab}.json").write_text(json.dumps(r, indent=2))
            print(json.dumps(slim.get("metrics") or slim.get("features"), indent=2)[:1200], flush=True)
        except Exception as e:
            err = {"env": env, "label": lab, "ckpt": str(ckpt), "error": repr(e)}
            rows.append(err)
            (args.out / f"{env}_{lab}_ERROR.json").write_text(json.dumps(err, indent=2))
            print("ERROR", e, flush=True)

    # Compact summary without per_frame arrays (those live in per-agent JSON).
    compact_rows = []
    for r in rows:
        if "error" in r:
            compact_rows.append(r)
            continue
        c = {k: v for k, v in r.items() if k != "per_frame"}
        if "features" in c:
            c["features"] = {
                fk: {kk: vv for kk, vv in fv.items() if kk != "per_frame"}
                for fk, fv in c["features"].items()
            }
        if "metrics" in c and isinstance(c["metrics"], dict):
            pass
        compact_rows.append(c)

    summary = {
        "protocol": "env-move re-score signed+TV+KL (median/trim/p90 on KL+TV)",
        "n_frames": args.n_frames,
        "coin_start_level": args.start_level,
        "ras_start_level": args.ras_start_level,
        "maze_start_level": args.maze_start_level,
        "device": device,
        "runtime_s": time.time() - t0,
        "results": compact_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_md(rows, args.out / "SUMMARY.md")
    print(f"\n[done] {args.out / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
