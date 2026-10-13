"""Fisher-weighted input saliency (diagonal input-space Fisher) on the Maze and Coin Fisher banks, by region."""
from __future__ import annotations

import os
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "external" / "procgenAISC", ROOT / "external" / "train-procgen-pytorch",
          ROOT / "scripts" / "procgen"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
OUT = DATA_ROOT / "probes/saliency_maze_coin_20260925"
FIG = ROOT / "paper_figures"
MAZE_CK = DATA_ROOT / "checkpoints/head_rewire/maze"
COIN_CK = DATA_ROOT / "checkpoints/coin"
COIN_BANK = DATA_ROOT / "probes/coin_canonical_bank_20260924/coin_canonical_bank.json"
MCF = DATA_ROOT / "probes/canon_mcfisher_20260924/raw_job28694"
NETS = {
    "maze": {"gm_1080": "1080_gm.pth", "gm_1081": "1081_gm.pth", "gm_1082": "1082_gm.pth",
             "nongm_1090": "1090_teach.pth", "nongm_1091": "1091_teach.pth", "nongm_1092": "1092_teach.pth"},
    "coin": {"gm_6033": "6033_gm_20M.pth", "gm_6037": "6037_gm_120M.pth", "gm_6051": "6051_gm_55M.pth",
             "nongm_7001": "7001_teach.pth", "nongm_7002": "7002_teach.pth", "nongm_7003": "7003_teach.pth"},
}
COIN_SOURCES = {"gm_rollouts": ("6033_gm_20M.pth", 1_100_000), "nongm_rollouts": ("7001_teach.pth", 1_120_000)}
COIN_UNIT = 64.0 / 13.0  # coinrun_aisc visibility = 13 cells across the 64-px frame, camera on the agent
T975_DF2 = 4.302652729911275


def pick_device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


def use_prebuilt_procgen() -> None:
    """Reuse the existing arm64 libenv; procgen's cmake step would otherwise try to rebuild it."""
    import procgen.env as pe
    lib = Path(pe.__file__).parent / ".build" / "relwithdebinfo"
    if (lib / "libenv.dylib").exists() or (lib / "libenv.so").exists():
        pe.build = lambda debug=False, package=False: str(lib)


# ----------------------------------------------------------------------------- harvest
def harvest_maze(args) -> None:
    """Harvest the shared 128-state decision-fork Maze bank (warm GM 1080) and check it against the stored Fisher bank."""
    import torch
    use_prebuilt_procgen()
    from run_maze_fisher_decision_fork import harvest_decision_fork
    from run_maze_i_goal_proxy_ladder import get_info_maze, _iscalar, load_policy, make_env

    torch.set_num_threads(4)
    warm = load_policy(MAZE_CK / "1080_gm.pth", "cpu")
    frames, meta = harvest_decision_fork(warm, device="cpu", n_frames=128, start_level=1_000_000,
                                         max_levels=8000, max_steps=200, source_name="gm_rollouts")
    env = make_env("maze", start_level=frames[0]["level"])
    env.reset()
    wd = _iscalar(get_info_maze(env), "maze_world_dim")
    env.close()
    # Identity check 1: the first three obs are stored bit-exactly with the Fisher bank.
    g = np.load(MCF / "geom/geom_maze_first3.npz")
    ident_obs = [float(np.abs(g[f"f{i}/obs_a"] - frames[i]["obs_a"]).max()) for i in range(3)]
    # Identity check 2: ||h_fc(reloc) - h_fc(base)|| for gm_1080 on all 128 frames.
    pf = json.load(open(MCF / "maze/maze_fc_block3_signed_derivative_per_frame.json"))["gm_1080"]["fc"]
    with torch.no_grad():
        hb = warm.embedder(torch.as_tensor(np.stack([f["obs_a"] for f in frames])))
        hr = warm.embedder(torch.as_tensor(np.stack([f["obs_reloc"] for f in frames])))
    nd = (hr - hb).norm(dim=1).numpy()
    nd_saved = np.array([r["norm_delta_star"] for r in pf])
    ident = {"max_abs_obs_diff_first3": ident_obs, "norm_delta_max_abs_err": float(np.abs(nd - nd_saved).max()),
             "norm_delta_rel_err_median": float(np.median(np.abs(nd - nd_saved) / nd_saved)),
             "harvest_meta": meta, "world_dim_first_level": int(wd)}
    print(json.dumps({k: v for k, v in ident.items() if k != "harvest_meta"}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "bank_maze.npz",
        obs=np.stack([f["obs_a"] for f in frames]), obs_gone=np.stack([f["obs_gone"] for f in frames]),
        level=np.array([f["level"] for f in frames]), t=np.array([f["t"] for f in frames]),
        agent_cell=np.array([f["agent_cell"] for f in frames]), cheese_xy=np.array([f["cheese_xy"] for f in frames]),
        tr_cell=np.array([f["tr_cell"] for f in frames]), a_cheese=np.array([f["a_cheese"] for f in frames]),
        a_corner=np.array([f["a_corner"] for f in frames]), world_dim=np.full(len(frames), wd))
    (args.out / "bank_maze_identity.json").write_text(json.dumps(ident, indent=1, default=str))


def harvest_coin(args) -> None:
    """Replay the canonical Coin bank (c0961932) for one source and render left/right/no-coin views."""
    import torch
    use_prebuilt_procgen()
    from gm_geometry.probes.coin_canonical_bank import load_bank
    from gm_geometry.probes.coin_location import (get_info, labels_from_info, obs_nchw_after_set_coin,
                                                  set_coin, unwrap_base_env)
    from gm_geometry.probes.coin_rollout import harvest_with_state_snapshot, load_policy, make_env

    torch.set_num_threads(4)
    bank = load_bank(COIN_BANK, expect_sha256=None)
    states = {s["level"]: s for s in bank["states"] if s["source"] == args.source}
    ck, start = COIN_SOURCES[args.source]
    warm = load_policy(COIN_CK / ck, "cpu")
    frames = harvest_with_state_snapshot(n_frames=len(states), start_level=start, max_levels=40_000,
                                         warm_policy=warm, device="cpu", skip_policy_gate=True)
    frames = [f for f in frames if int(f["level"]) in states]
    rec = {k: [] for k in ("obs_left", "obs_right", "obs_none", "level", "agent_xy", "goal_xy", "shortcut_xy",
                           "inv_xy", "pose_match")}
    for fr in frames:
        st = states[int(fr["level"])]
        env = make_env("coinrun_aisc", num_levels=1, start_level=0)
        env.reset()
        base = unwrap_base_env(env)
        base.set_state(fr["state_reloc"])
        snap = base.get_state()
        obs = {}
        for key, xy, present in (("obs_right", st["shortcut_xy"], 1), ("obs_left", st["goal_xy"], 1),
                                 ("obs_none", (0, 0), 0)):
            base.set_state(snap)
            set_coin(env, xy[0], xy[1], present)
            o = obs_nchw_after_set_coin(env)
            obs[key] = o[0] if o.ndim == 4 else o
        lab = labels_from_info(get_info(env))
        env.close()
        for k in obs:
            rec[k].append(obs[k].astype(np.float32))
        rec["level"].append(int(fr["level"]))
        rec["agent_xy"].append((lab["agent_x"], lab["agent_y"]))
        rec["goal_xy"].append(st["goal_xy"])
        rec["shortcut_xy"].append(st["shortcut_xy"])
        rec["inv_xy"].append((lab["inv_coin_x"], lab["inv_coin_y"]))
        # CPU replay of a CUDA harvest can land the agent elsewhere on the same level.
        rec["pose_match"].append(bool(abs(lab["agent_x"] - st["agent_x"]) < 0.015
                                      and abs(lab["agent_y"] - st["agent_y"]) < 0.015
                                      and tuple(fr["xy_a"]) == tuple(st["shortcut_xy"])
                                      and tuple(fr["reloc_xy"]) == tuple(st["goal_xy"])))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"bank_coin_{args.source}.npz", **{k: np.array(v) for k, v in rec.items()})
    print(f"coin {args.source}: {len(frames)}/{len(states)} recovered, "
          f"{int(np.sum(rec['pose_match']))} pose-matched", flush=True)


# ----------------------------------------------------------------------------- saliency
def load_net(env: str, label: str, device):
    if env == "maze":
        from run_maze_i_goal_proxy_ladder import load_policy
        if label == "init_0":
            import torch
            torch.manual_seed(0)
            np.random.seed(0)
            return load_policy(MAZE_CK / "1090_teach.pth", device, random_init=True)
        return load_policy(MAZE_CK / NETS["maze"][label], device)
    from run_coin_fisher_sensitivity import load_init_policy
    from gm_geometry.probes.coin_rollout import load_policy
    if label == "init_0":
        return load_init_policy(COIN_CK / "7001_teach.pth", str(device), init_seed=0).to(device)
    return load_policy(COIN_CK / NETS["coin"][label], str(device)).to(device)


def fisher_saliency(policy, obs: np.ndarray, device, bs: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """S(x) = E_{a~pi}[(d log pi(a|o)/d o_x)^2], summed over colour channels. Returns (S [N,64,64], probs)."""
    import torch
    policy.eval()
    out_s, out_p = [], []
    for i in range(0, len(obs), bs):
        x = torch.as_tensor(obs[i:i + bs], dtype=torch.float32, device=device).requires_grad_(True)
        logp = torch.log_softmax(policy.fc_policy(policy.embedder(x)), -1)
        p = logp.exp().detach()
        s = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=device)
        for a in range(logp.shape[1]):
            g, = torch.autograd.grad(logp[:, a].sum(), x, retain_graph=a < logp.shape[1] - 1)
            s += p[:, a, None, None] * (g ** 2).sum(1)
        out_s.append(s.cpu().numpy())
        out_p.append(p.cpu().numpy())
    return np.concatenate(out_s), np.concatenate(out_p)


def maze_masks(bank, i):
    """Exclusive pixel masks. Grid y points up; image row 0 is the top."""
    wd = int(bank["world_dim"][i])
    unit = 64.0 / wd
    c = np.arange(64)
    cx = np.floor((c + 0.5) / unit).astype(int)
    cy = wd - 1 - np.floor((c + 0.5) / unit).astype(int)
    X, Y = np.meshgrid(cx, cy)  # X[r, col], Y[r, col]
    ax, ay = bank["agent_cell"][i]
    chx, chy = bank["cheese_xy"][i]
    tx, ty = bank["tr_cell"][i]
    agent = (X == ax) & (Y == ay)
    goal = (X == chx) & (Y == chy) & ~agent
    ext = (np.abs(X - tx) <= 1) & (np.abs(Y - ty) <= 1) & ~agent & ~goal
    R, C = np.meshgrid(c, c, indexing="ij")
    box = (C >= 0.72 * 64) & (R <= 0.28 * 64) & ~agent & ~goal
    cell = (X == tx) & (Y == ty) & ~agent & ~goal
    near = (np.abs(X - ax) <= 2) & (np.abs(Y - ay) <= 2) & ~agent & ~goal
    return {"goal": goal, "proxy": ext, "proxy_box": box, "proxy_cell": cell, "near_agent": near, "agent": agent}


def coin_world_grid(agent_xy):
    c = np.arange(64) + 0.5
    wx = agent_xy[0] + (c - 32.0) / COIN_UNIT
    wy = agent_xy[1] - (c - 32.0) / COIN_UNIT
    WX, WY = np.meshgrid(wx, wy)
    return WX, WY


def coin_masks(bank, i, coin_key):
    WX, WY = coin_world_grid(bank["agent_xy"][i])
    axy = bank["agent_xy"][i]
    gx, gy = bank["goal_xy"][i] if coin_key == "left" else bank["shortcut_xy"][i]
    agent = (np.abs(WX - axy[0]) <= 0.5) & (np.abs(WY - axy[1]) <= 0.5787)
    goal = (WX >= gx) & (WX < gx + 1) & (WY >= gy) & (WY < gy + 1) & ~agent
    wall = (WX >= bank["inv_xy"][i][0] + 1) & ~agent & ~goal
    ahead = (WX > axy[0] + 0.5) & ~agent & ~goal
    near = (np.abs(WX - axy[0]) <= 2.5) & (np.abs(WY - axy[1]) <= 2.5) & ~agent & ~goal
    return {"goal": goal, "proxy": wall, "proxy_ahead": ahead, "near_agent": near, "agent": agent}


def region_rows(S, masks_fn, n):
    rows = []
    for i in range(n):
        m = masks_fn(i)
        tot = float(S[i].sum())
        mean_px = tot / S[i].size
        r = {"state": i, "total": tot}
        rest = np.ones_like(m["goal"])
        for k in ("goal", "proxy", "agent"):
            rest &= ~m[k]
        m = dict(m, rest=rest)
        for k, mk in m.items():
            a = int(mk.sum())
            r[f"area_{k}"] = a
            r[f"share_{k}"] = float(S[i][mk].sum() / tot) if a else np.nan
            r[f"enrich_{k}"] = float(S[i][mk].mean() / mean_px) if a else np.nan
        rows.append(r)
    return rows


def score(args) -> None:
    import torch
    use_prebuilt_procgen()
    device = pick_device()
    torch.manual_seed(0)
    maze = dict(np.load(args.out / "bank_maze.npz"))
    coin_parts = [dict(np.load(args.out / f"bank_coin_{s}.npz")) for s in COIN_SOURCES]
    coin = {k: np.concatenate([p[k] for p in coin_parts]) for k in coin_parts[0]}
    coin["source"] = np.concatenate([[s] * len(p["level"]) for s, p in zip(COIN_SOURCES, coin_parts)])
    np.savez_compressed(args.out / "bank_coin_meta.npz", **{k: v for k, v in coin.items() if not k.startswith("obs")})
    maps = {}
    for env, bank, views in (("maze", maze, {"base": "obs"}),
                             ("coin", coin, {"left": "obs_left", "right": "obs_right"})):
        for label in list(NETS[env]) + ["init_0"]:
            pol = load_net(env, label, device)
            for view, key in views.items():
                S, P = fisher_saliency(pol, bank[key], device)
                if env == "maze":
                    fn = lambda i: maze_masks(bank, i)  # noqa: E731
                else:
                    fn = lambda i, v=view: coin_masks(bank, i, v)  # noqa: E731
                rows = region_rows(S, fn, len(S))
                for r in rows:
                    r.update(env=env, view=view, net=label, level=int(bank["level"][r["state"]]),
                             entropy=float(-(P[r["state"]] * np.log(P[r["state"]] + 1e-12)).sum()))
                    if env == "coin":
                        r["pose_match"] = bool(bank["pose_match"][r["state"]])
                        r["source"] = str(bank["source"][r["state"]])
                with open(args.out / f"raw_{env}_{view}_{label}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                maps[f"{env}_{view}_{label}"] = S.astype(np.float32)
                print(f"scored {env}/{view}/{label}: n={len(rows)}", flush=True)
    np.savez_compressed(args.out / "saliency_maps.npz", **maps)


# ----------------------------------------------------------------------------- summary + figures
def seed_stats(vals):
    a = np.asarray(vals, float)
    m = float(a.mean())
    if len(a) < 2:
        return {"mean": m, "sd": float("nan"), "ci95": [float("nan")] * 2, "per_seed": a.tolist()}
    sd = float(a.std(ddof=1))
    h = T975_DF2 * sd / np.sqrt(len(a))
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "per_seed": a.tolist()}


def region_distances(env, args, view=None):
    """Agent-to-region distance in cells per state (Chebyshev for Maze, horizontal for Coin)."""
    if env == "maze":
        b = np.load(args.out / "bank_maze.npz")
        return (np.abs(b["cheese_xy"] - b["agent_cell"]).max(1).astype(float),
                np.abs(b["tr_cell"] - b["agent_cell"]).max(1).astype(float))
    b = np.load(args.out / "bank_coin_meta.npz")
    ax = b["agent_xy"][:, 0]
    g = b["goal_xy"][:, 0] if view == "left" else b["shortcut_xy"][:, 0]
    return np.abs(g + 0.5 - ax), np.maximum(b["inv_xy"][:, 0] + 1 - ax, 0.0)


def distance_adjusted_contrast(df, d_goal, d_proxy):
    """OLS log_enrich ~ 1 + distance + [proxy]; returns the proxy-minus-goal coefficient.

    Gradient saliency decays with distance from the agent and the cheese/coin is usually nearer
    than the proxy region, so a raw goal-vs-proxy contrast within one network is confounded."""
    st = df.state.values
    lg = np.log(df.enrich_goal.values.clip(1e-12))
    lp = np.log(df.enrich_proxy.values.clip(1e-12))
    ok_p = np.isfinite(lp) & (df.area_proxy.values > 0)
    y = np.concatenate([lg, lp[ok_p]])
    X = np.column_stack([np.ones(len(y)), np.concatenate([d_goal[st], d_proxy[st][ok_p]]),
                         np.concatenate([np.zeros(len(lg)), np.ones(ok_p.sum())])])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[2]), float(coef[1])


def summarize(args) -> dict:
    import pandas as pd
    out = {}
    for env, views in (("maze", ["base"]), ("coin", ["left", "right"])):
        for view in views:
            per = {}
            for label in list(NETS[env]) + ["init_0"]:
                df = pd.read_csv(args.out / f"raw_{env}_{view}_{label}.csv")
                if env == "coin":
                    df = df[df.pose_match]
                e = {"n": int(len(df))}
                regs = ["goal", "proxy", "agent", "rest", "near_agent"] + (
                    ["proxy_box", "proxy_cell"] if env == "maze" else ["proxy_ahead"])
                if env == "coin":
                    # Like-for-like goal vs end-wall comparison on the states where the wall is on screen.
                    vis = df[df.area_proxy > 0]
                    e["n_wall_visible"] = int(len(vis))
                    for k in ("goal", "proxy", "agent"):
                        e[f"log_enrich_{k}_wallvis"] = float(np.log(vis[f"enrich_{k}"].clip(lower=1e-12)).mean())
                        e[f"share_{k}_wallvis"] = float(vis[f"share_{k}"].mean())
                for k in regs:
                    # Share: pooled over states is dominated by high-gradient states, so average per state.
                    e[f"share_{k}"] = float(df[f"share_{k}"].mean())
                    e[f"enrich_{k}"] = float(df[f"enrich_{k}"].mean())
                    e[f"median_enrich_{k}"] = float(df[f"enrich_{k}"].median())
                    e[f"frac_visible_{k}"] = float((df[f"area_{k}"] > 0).mean())
                    e[f"area_frac_{k}"] = float(df[f"area_{k}"].mean() / 4096)
                    e[f"log_enrich_{k}"] = float(np.log(df[f"enrich_{k}"].dropna().clip(lower=1e-12)).mean())
                e["goal_minus_proxy_log_enrich"] = e["log_enrich_goal"] - e["log_enrich_proxy"]
                dgl, dpx = region_distances(env, args, view)
                c, slope = distance_adjusted_contrast(df, dgl, dpx)
                e["proxy_minus_goal_log_enrich_dist_adj"] = c
                e["log_enrich_per_cell_distance"] = slope
                e["mean_entropy"] = float(df.entropy.mean())
                per[label] = e
            keys = list(per[list(NETS[env])[0]])
            gm = [l for l in NETS[env] if l.startswith("gm")]
            ng = [l for l in NETS[env] if l.startswith("nongm")]
            out[f"{env}_{view}"] = {
                "per_net": per,
                "cohort": {c: {k: seed_stats([per[l][k] for l in labs]) for k in keys}
                           for c, labs in (("GM", gm), ("NonGM", ng), ("init", ["init_0"]))}}
    out["notes"] = {
        "S": "sum_c E_{a~pi}[(d log pi(a|o)/d o_{c,x})^2] at the policy's own action distribution",
        "share": "per-state S(R)/S(image), then mean over states; regions exclusive with priority agent > goal > proxy",
        "enrich": "per-state (S(R)/|R|)/(S(image)/4096), mean over states; log_enrich is mean log of it",
        "maze_proxy": "3x3 cells around the reachable top-right extreme (tr_cell); proxy_box = image-level 0.28x0.28 "
                      "top-right box (in_top_right frac), reported as a secondary definition",
        "coin_proxy": "pixels with world x >= inv_coin_x + 1 (the end-wall block); empty when off-screen; "
                      "proxy_ahead = everything right of the agent box (secondary, directional)",
        "coin_views": "left = coin at the goal cell 1 left of the agent (conflict state); right = coin at shortcut cell",
        "coin_filter": "states whose CPU replay reproduces the bank agent pose and both coin cells",
    }
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def palette():
    sys.path.insert(0, str(ROOT / "figures"))
    try:
        from palette import PAL
        return PAL
    except Exception:
        return {"gm": "#7A1520", "gm_proxy": "#C45C58", "nongm": "#3D8B8B", "nongm_proxy": "#5FA8A0",
                "init": "#8E8E8E", "ink": "#2A2A2A", "mute": "#6E6E6E"}


def fig_panel(summary, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAL = palette()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": PAL["ink"], "axes.labelcolor": PAL["ink"],
                         "xtick.color": PAL["ink"], "ytick.color": PAL["ink"]})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    cols = {"GM": PAL["gm"], "NonGM": PAL["nongm"], "init": PAL["init"]}
    offs = {"GM": -0.22, "NonGM": 0.0, "init": 0.22}
    for ax, key, title in ((axes[0], "maze_base", "Maze"), (axes[1], "coin_left", "Coin")):
        coh = summary[key]["cohort"]
        for xi, reg in enumerate(("goal", "proxy")):
            for c in ("GM", "NonGM", "init"):
                st = coh[c][f"log_enrich_{reg}"]
                y = np.log10(np.exp(np.asarray(st["per_seed"])))
                x0 = xi + offs[c]
                filled = reg == "goal"
                ax.scatter(np.full(len(y), x0) + np.linspace(-0.04, 0.04, len(y)), y, s=14,
                           facecolors=cols[c] if filled else "white", edgecolors=cols[c], lw=1, zorder=3,
                           alpha=0.8)
                m = np.log10(np.exp(st["mean"]))
                if np.isfinite(st["sd"]):
                    lo, hi = (np.log10(np.exp(v)) for v in st["ci95"])
                    ax.plot([x0 + 0.08] * 2, [lo, hi], color=cols[c], lw=1.2, zorder=2)
                ax.scatter([x0 + 0.08], [m], marker="_", s=90, color=cols[c], lw=2, zorder=4)
        ax.axhline(0, color=PAL["mute"], lw=0.7, ls=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["goal", "proxy"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title, fontsize=9, color=PAL["ink"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$\log_{10}$ saliency per pixel / image mean")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=cols[c], label=l)
         for c, l in (("GM", "GM"), ("NonGM", "Non-GM"), ("init", "init"))]
    axes[1].legend(handles=h, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.fig / f"fig_panel_saliency.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def fig_examples(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    maps = np.load(args.out / "saliency_maps.npz")
    maze = dict(np.load(args.out / "bank_maze.npz"))
    coin = {k: np.concatenate([np.load(args.out / f"bank_coin_{s}.npz")[k] for s in COIN_SOURCES])
            for k in ("obs_left", "agent_xy", "goal_xy", "inv_xy", "pose_match")}
    PAL = palette()
    specs = {
        # First three bank states meeting a fixed rule (no selection on saliency):
        # Maze, cheese >= 6 cells (Euclid) from the extreme; Coin, pose-matched with the end wall on screen.
        "maze": ("gm_1080", "nongm_1090", "maze_base",
                 [i for i in range(len(maze["level"]))
                  if np.hypot(*(maze["cheese_xy"][i] - maze["tr_cell"][i])) >= 6][:3]),
        "coin": ("gm_6033", "nongm_7001", "coin_left",
                 [i for i in range(len(coin["agent_xy"]))
                  if coin["pose_match"][i] and coin_masks(coin, i, "left")["proxy"].any()][:3]),
    }
    for env, (a, b, key, idx) in specs.items():
        fig, axes = plt.subplots(len(idx), 3, figsize=(6.0, 2.05 * len(idx)))
        for row, i in enumerate(idx):
            obs = (maze["obs"] if env == "maze" else coin["obs_left"])[i].transpose(1, 2, 0)
            m = maze_masks(maze, i) if env == "maze" else coin_masks(coin, i, "left")
            axes[row, 0].imshow(obs)
            for k, col in (("goal", "#F2B01E"), ("proxy", PAL["gm_proxy"]), ("agent", "white")):
                ys, xs = np.where(m[k])
                if len(xs):
                    axes[row, 0].add_patch(Rectangle((xs.min() - 0.5, ys.min() - 0.5), xs.max() - xs.min() + 1,
                                                     ys.max() - ys.min() + 1, fill=False, ec=col, lw=1.2))
            for j, lab in ((1, a), (2, b)):
                S = maps[f"{key}_{lab}"][i]
                axes[row, j].imshow(obs.mean(-1), cmap="gray", alpha=1.0)
                axes[row, j].imshow(np.log10(S / S.mean() + 1e-3), cmap="magma", alpha=0.75, vmin=-1, vmax=2)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_title("state", fontsize=9)
        axes[0, 1].set_title("GM", fontsize=9, color=PAL["gm"])
        axes[0, 2].set_title("Non-GM", fontsize=9, color=PAL["nongm"])
        fig.tight_layout()
        fig.savefig(args.fig / f"fig_saliency_examples_{env}.png", dpi=250, facecolor="white")
        plt.close(fig)
        (args.out / f"examples_{env}.json").write_text(json.dumps({"state_indices": [int(i) for i in idx],
                                                                    "nets": [a, b]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", "harvest-maze", "harvest-coin", "score", "summarize", "figures"])
    ap.add_argument("--source", choices=list(COIN_SOURCES))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    args = ap.parse_args()
    if args.stage == "harvest-maze":
        harvest_maze(args)
    elif args.stage == "harvest-coin":
        harvest_coin(args)
    elif args.stage == "score":
        score(args)
    elif args.stage in ("summarize", "figures"):
        s = summarize(args)
        fig_panel(s, args)
        fig_examples(args)
    else:
        # One fresh process per env harvest: long procgen loops in one process leak memory.
        run = lambda *a: subprocess.run([sys.executable, __file__, *a, "--out", str(args.out)], check=True)  # noqa: E731
        run("harvest-maze")
        for s in COIN_SOURCES:
            run("harvest-coin", "--source", s)
        run("score")
        s = summarize(args)
        fig_panel(s, args)
        fig_examples(args)


if __name__ == "__main__":
    main()
