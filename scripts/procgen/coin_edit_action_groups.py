"""Split the Coin world-edit policy change (coin shortcut-right -> goal-left) on the canonical bank into action groups, with an init reference and a coin-distance sweep."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "procgen", ROOT / "external" / "procgenAISC"):
    sys.path.insert(0, str(p))

# Procgen action combos (procgen/env.py:get_combos). In CoinRun UP is jump; indices 9-14
# (D, A, W, S, Q, E) are unbound keys and act as no-ops.
GROUPS = {
    "star": (0, 1, 2),      # LEFT_ACTIONS: coin-directed on this bank (goal coin 1 cell left)
    "right": (6, 7, 8),     # RIGHT_ACTIONS: toward the original (shortcut) coin cell
    "jump": (5,),           # pure UP; LEFT+UP / RIGHT+UP stay in their direction group
    "other": (3, 4, 9, 10, 11, 12, 13, 14),  # DOWN, NOOP, unbound keys
}
GM = ["gm_6033", "gm_6037", "gm_6051"]
NONGM = ["nongm_7001", "nongm_7002", "nongm_7003"]
# Same draws as the paper's Coin Fisher init baseline (run_coin_conflict_signed_derivative.py --n-init 3).
INIT = ["init_0", "init_1", "init_2"]
INIT_SEEDS = dict(zip(INIT, (4000, 4001, 4002)))
COHORTS = (("GM", GM), ("NonGM", NONGM), ("init", INIT))
SWEEP_D = (1, 2, 3, 4)
CKPT_FILES = dict(zip(GM + NONGM, ["6033_gm_20M.pth", "6037_gm_120M.pth", "6051_gm_55M.pth",
                                   "7001_teach.pth", "7002_teach.pth", "7003_teach.pth"]))
SOURCES = {"gm_rollouts": ("gm_6033", 1_100_000), "nongm_rollouts": ("nongm_7001", 1_120_000)}
T_975_DF2 = 4.302652729911275


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def use_prebuilt_procgen() -> None:
    """Point procgen at an existing libenv build instead of rebuilding with cmake, whose failure path
    deletes the build directory."""
    import procgen.env as pe
    lib = Path(pe.__file__).parent / ".build" / "relwithdebinfo"
    if (lib / "libenv.dylib").exists() or (lib / "libenv.so").exists():
        pe.build = lambda debug=False, package=False: str(lib)


def recompute(args) -> None:
    import torch
    use_prebuilt_procgen()
    from gm_geometry.probes.coin_canonical_bank import load_bank
    from gm_geometry.probes.coin_location import obs_nchw_after_set_coin, set_coin, unwrap_base_env
    from gm_geometry.probes.coin_rollout import harvest_with_state_snapshot, load_policy, make_env
    from run_coin_fisher_sensitivity import load_init_policy

    torch.set_num_threads(args.threads)
    device = "cpu"
    bank = load_bank(args.bank)
    states = {s["level"]: s for s in bank["states"] if s["source"] == args.source}
    policies = {l: load_policy(args.ckpt_dir / f, device) for l, f in CKPT_FILES.items()}
    for l, sd in INIT_SEEDS.items():
        policies[l] = load_init_policy(args.ckpt_dir / CKPT_FILES["nongm_7001"], device, init_seed=sd)
    warm, start = SOURCES[args.source]
    frames = harvest_with_state_snapshot(n_frames=len(states), start_level=start,
                                         max_levels=args.max_levels, warm_policy=policies[warm],
                                         device=device, skip_policy_gate=True)
    frames = [f for f in frames if int(f["level"]) in states]
    labels = list(CKPT_FILES) + INIT
    n = len(frames)
    pa = np.zeros((len(labels), n, 15)); pb = np.zeros_like(pa); pc = np.zeros_like(pa); dz = np.zeros_like(pa)
    pd = np.zeros((len(labels), n, len(SWEEP_D), 15))
    d_free = np.zeros((n, len(SWEEP_D)), bool); d_pix = np.zeros((n, len(SWEEP_D)))

    def forward(pol, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            h = pol.embedder(x)
            return torch.softmax(pol.fc_policy(h), -1)[0].numpy(), h[0].numpy()

    for i, fr in enumerate(frames):
        st = states[int(fr["level"])]
        env = make_env("coinrun_aisc", num_levels=1, start_level=0)
        env.reset()
        base = unwrap_base_env(env)
        base.set_state(fr["state_reloc"])
        snap = base.get_state()
        set_coin(env, st["shortcut_xy"][0], st["shortcut_xy"][1], 1)
        obs_a = obs_nchw_after_set_coin(env)
        base.set_state(snap)
        set_coin(env, st["goal_xy"][0], st["goal_xy"][1], 1)
        obs_b = obs_nchw_after_set_coin(env)
        # Coin removed entirely: splits the relocation into "leave the right cell" + "appear left".
        base.set_state(snap)
        set_coin(env, 0, 0, 0)
        obs_c = obs_nchw_after_set_coin(env)
        # Distance sweep: coin d cells left of the agent's cell, on the goal cell's row.
        obs_d = []
        for k, dd in enumerate(SWEEP_D):
            base.set_state(snap)
            ok = set_coin(env, st["agent_cell_x"] - dd, st["goal_xy"][1], 1) == 0
            o = obs_nchw_after_set_coin(env)
            o = o[0] if o.ndim == 4 else o
            d_free[i, k] = ok
            obs_d.append(o)
        env.close()
        obs_c = obs_c[0] if obs_c.ndim == 4 else obs_c
        obs_a = obs_a[0] if obs_a.ndim == 4 else obs_a
        obs_b = obs_b[0] if obs_b.ndim == 4 else obs_b
        for k, o in enumerate(obs_d):
            d_pix[i, k] = float(np.abs(o - obs_c).max())
        for j, l in enumerate(labels):
            pol = policies[l]
            p_a, h_a = forward(pol, obs_a)
            p_b, h_b = forward(pol, obs_b)
            pa[j, i], pb[j, i], pc[j, i] = p_a, p_b, forward(pol, obs_c)[0]
            dz[j, i] = pol.fc_policy.weight.detach().numpy().astype(np.float64) @ (h_b - h_a)
            for k, o in enumerate(obs_d):
                pd[j, i, k] = forward(pol, o)[0]
        if (i + 1) % 50 == 0:
            print(f"  {args.source}: scored {i + 1}/{n}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"raw_{args.source}.npz", levels=np.array([int(f["level"]) for f in frames]),
             labels=np.array(labels), pa=pa, pb=pb, pc=pc, dz=dz, pd=pd, d_free=d_free, d_pix=d_pix,
             sweep_d=np.array(SWEEP_D))
    print(f"{args.source}: recovered {n}/{len(states)} bank states", flush=True)


def iso_null_ratio(pa: np.ndarray, tv: np.ndarray, groups: list[list[int]], k: int = 64, seed: int = 0) -> np.ndarray:
    """TV share / prior mass per group if the logit change had no preferred direction.

    Same states and same pi_a as the policy: delta z ~ N(0, I) per draw, first-order softmax
    response, each draw rescaled to the state's observed TV so states pool with their real
    weights. The analytic value 1 holds only for an unsaturated pi_a. When one action holds
    most of the mass, it moves by about half the TV and the small groups share the rest, so
    their null ratio is about 0.5 / (1 - pi_max)."""
    rng = np.random.default_rng(seed)
    parts = np.zeros(len(groups)); total = 0.0
    for _ in range(k):
        e = rng.standard_normal(pa.shape)
        d = pa * (e - np.sum(pa * e, -1, keepdims=True))
        d *= (tv / np.maximum(0.5 * np.abs(d).sum(-1), 1e-300))[:, None]
        parts += np.array([0.5 * np.abs(d[:, g]).sum() for g in groups])
        total += tv.sum()
    mass = np.array([pa[:, g].sum(-1).mean() for g in groups])
    return parts / total / mass


def kl(p, q):
    p = np.clip(p, 1e-12, 1.0); q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum(-1, keepdims=True); q = q / q.sum(-1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), -1)


def seed_stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    m, sd = float(a.mean()), float(a.std(ddof=1))
    h = T_975_DF2 * sd / np.sqrt(len(a))
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "per_seed": [float(x) for x in a]}


def analyze(args) -> None:
    saved = json.loads(args.saved.read_text())
    saved_kl = {(r["source"], int(r["level"]), r["ckpt"]): r["kl"] for r in saved["rows"]}
    rows = []
    for src in SOURCES:
        z = np.load(args.out / f"raw_{src}.npz")
        labels = [str(x) for x in z["labels"]]
        for j, lbl in enumerate(labels):
            pa, pb, pc, dz = z["pa"][j], z["pb"][j], z["pc"][j], z["dz"][j]
            d = pb - pa
            # Exact at the h_fc tap (logits are affine in h_fc); first order only in softmax.
            lin = pa * (dz - np.sum(pa * dz, -1, keepdims=True))
            k = kl(pa, pb)
            tv = 0.5 * np.abs(d).sum(-1)
            for i, lev in enumerate(z["levels"]):
                r = {"source": src, "level": int(lev), "ckpt": lbl,
                     "cohort": next(c for c, labs in COHORTS if lbl in labs),
                     "kl": float(k[i]), "kl_saved": saved_kl.get((src, int(lev), lbl), np.nan),
                     "tv": float(tv[i])}
                for g, ids in GROUPS.items():
                    ids = list(ids)
                    r[f"mass_a_{g}"] = float(pa[i, ids].sum())
                    r[f"dmass_{g}"] = float(d[i, ids].sum())
                    r[f"tvpart_{g}"] = float(0.5 * np.abs(d[i, ids]).sum())
                    r[f"lin_dmass_{g}"] = float(lin[i, ids].sum())
                    r[f"dmass_remove_{g}"] = float((pc[i, ids] - pa[i, ids]).sum())
                    r[f"dmass_appear_{g}"] = float((pb[i, ids] - pc[i, ids]).sum())
                r["tv_between"] = 0.5 * sum(abs(r[f"dmass_{g}"]) for g in GROUPS)
                for a in range(15):
                    r[f"pa_{a}"] = float(pa[i, a]); r[f"pb_{a}"] = float(pb[i, a])
                rows.append(r)

    # State identity: the CPU replay of the warm-policy harvest can drift from the CUDA one,
    # landing the agent elsewhere on the same level. Keep a state only if all six recomputed
    # KLs reproduce the saved per-state KL.
    worst = {}
    for r in rows:
        if r["cohort"] == "init":
            continue
        k = (r["source"], r["level"])
        worst[k] = max(worst.get(k, 0.0), abs(r["kl"] - r["kl_saved"]))
    for r in rows:
        r["state_match"] = worst[(r["source"], r["level"])] <= args.match_tol
    all_rows = rows
    rows = [r for r in rows if r["state_match"]]

    with open(args.out / "per_state.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)

    def cohort_kl(rs, key):
        return {c: float(np.mean([np.mean([r[key] for r in rs if r["ckpt"] == l]) for l in labs]))
                for c, labs in COHORTS[:2]}

    per_ckpt = {}
    for lbl in GM + NONGM + INIT:
        mine = [r for r in rows if r["ckpt"] == lbl]
        tv_sum = sum(r["tv"] for r in mine)
        e = {"n": len(mine), "kl": float(np.mean([r["kl"] for r in mine])),
             "kl_saved": (saved["summary"]["per_ckpt"][lbl]["full"]["kl"]["mean"] if lbl not in INIT
                          else float("nan")),
             "tv": float(np.mean([r["tv"] for r in mine])),
             # Share of TV that is net movement between groups vs reshuffling inside a group.
             "between_share": sum(r["tv_between"] for r in mine) / tv_sum}
        pa_m = np.array([[r[f"pa_{a}"] for a in range(15)] for r in mine])
        iso = iso_null_ratio(pa_m, np.array([r["tv"] for r in mine]), [list(v) for v in GROUPS.values()])
        for g, v in zip(GROUPS, iso):
            e[f"iso_null_tv_over_mass_{g}"] = float(v)
        for g in GROUPS:
            e[f"mass_a_{g}"] = float(np.mean([r[f"mass_a_{g}"] for r in mine]))
            e[f"dmass_{g}"] = float(np.mean([r[f"dmass_{g}"] for r in mine]))
            e[f"absdmass_{g}"] = float(np.mean([abs(r[f"dmass_{g}"]) for r in mine]))
            e[f"lin_dmass_{g}"] = float(np.mean([r[f"lin_dmass_{g}"] for r in mine]))
            e[f"dmass_remove_{g}"] = float(np.mean([r[f"dmass_remove_{g}"] for r in mine]))
            e[f"dmass_appear_{g}"] = float(np.mean([r[f"dmass_appear_{g}"] for r in mine]))
            # Pooled ratio, so near-zero-TV states do not dominate with noisy per-state shares.
            e[f"tvshare_{g}"] = sum(r[f"tvpart_{g}"] for r in mine) / tv_sum
            # 1 = TV lands on a group in proportion to the mass it already holds (untargeted).
            e[f"tv_over_mass_{g}"] = e[f"tvshare_{g}"] / e[f"mass_a_{g}"]
            e[f"tv_over_mass_rel_iso_{g}"] = e[f"tv_over_mass_{g}"] / e[f"iso_null_tv_over_mass_{g}"]
        per_ckpt[lbl] = e

    keys = [k for k in per_ckpt[GM[0]] if k != "n"]
    cohorts = {c: {k: seed_stats([per_ckpt[l][k] for l in labs]) for k in keys}
               for c, labs in COHORTS}
    sweep = distance_sweep(args, {(r["source"], r["level"]) for r in rows})
    summary = {
        "bank_freeze_sha256": json.loads(args.bank.read_text())["freeze_sha256"],
        "groups": {g: list(v) for g, v in GROUPS.items()},
        "delta": "pi(.|coin relocated to goal cell, left) - pi(.|coin at shortcut cell, right)",
        "n_bank_states": len(saved_kl) // 6, "n_states_recovered": len(worst),
        "n_states_analyzed": len(rows) // (len(GM) + len(NONGM) + len(INIT)), "match_tol": args.match_tol,
        "excluded_states": sorted([list(k) for k, v in worst.items() if v > args.match_tol]),
        "kl_repro": {"cohort_saved_600": {c: saved["summary"]["cohort_full_bank"][c]["kl"]["mean"]
                                          for c in ("GM", "NonGM")},
                     "cohort_recomputed_all_recovered": cohort_kl(all_rows, "kl"),
                     "cohort_recomputed_analyzed": cohort_kl(rows, "kl"),
                     "cohort_saved_analyzed": cohort_kl(rows, "kl_saved")},
        "ci": "mean over 3 checkpoint means, 95% t-CI df=2",
        "per_ckpt": per_ckpt, "cohort": cohorts, "distance_sweep": sweep,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["kl_repro"], indent=1))
    for c in cohorts:
        print(c, {g: round(cohorts[c][f"dmass_{g}"]["mean"], 4) for g in GROUPS},
              {g: round(cohorts[c][f"tvshare_{g}"]["mean"], 3) for g in GROUPS})
    if args.fig:
        plot(summary, args.fig)


def distance_sweep(args, keep: set) -> dict:
    """Delta(d) = pi(.|coin d cells left of the agent) - pi(.|coin at the shortcut cell), same base frame.

    Primary population: matched states where the coin fits and is visible at every d, so the
    curve compares the same states across distances."""
    per_d = {}
    for src in SOURCES:
        z = np.load(args.out / f"raw_{src}.npz")
        labels = [str(x) for x in z["labels"]]
        keep_i = np.array([(src, int(l)) in keep for l in z["levels"]])
        ok = z["d_free"] & (z["d_pix"] > 0.05) & keep_i[:, None]
        for j, lbl in enumerate(labels):
            pa = z["pa"][j]
            for k, dd in enumerate(z["sweep_d"]):
                d = z["pd"][j, :, k] - pa
                for key, m in (("all4", ok.all(1)), ("free", ok[:, k])):
                    e = per_d.setdefault((key, int(dd), lbl), {"d": [], "pa": []})
                    e["d"].append(d[m]); e["pa"].append(pa[m])
    out = {"population": {}, "per_ckpt": {}, "cohort": {}}
    for (key, dd, lbl), e in per_d.items():
        d = np.concatenate(e["d"]); pa = np.concatenate(e["pa"])
        star = list(GROUPS["star"])
        tv = 0.5 * np.abs(d).sum(-1)
        share = 0.5 * np.abs(d[:, star]).sum() / tv.sum()
        out["population"][f"{key}_d{dd}"] = int(len(d))
        out["per_ckpt"].setdefault(key, {}).setdefault(str(dd), {})[lbl] = {
            "n": int(len(d)), "dmass_star": float(d[:, star].sum(-1).mean()), "tv": float(tv.mean()),
            "tv_over_mass_star": float(share / pa[:, star].sum(-1).mean())}
    for key, byd in out["per_ckpt"].items():
        for dd, byl in byd.items():
            for c, labs in COHORTS:
                out["cohort"].setdefault(key, {}).setdefault(dd, {})[c] = {
                    m: seed_stats([byl[l][m] for l in labs]) for m in ("dmass_star", "tv", "tv_over_mass_star")}
    return out


def plot(summary, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0, str(ROOT / "figures"))
    from palette import PAL
    names = {"star": r"$A_\star$ (left)", "right": r"$A_{\rightarrow}$ (right)", "jump": "jump", "other": "other"}
    style = {"GM": (PAL["gm"], "GM", 0), "NonGM": (PAL["nongm"], "Non-GM", 1), "init": (PAL["init"], "random init", 2)}
    jit = np.array([-0.04, 0.0, 0.04])

    def point(ax, x, s, col, ms=5):
        ax.scatter(x + jit, s["per_seed"], s=12, color=col, alpha=0.45, lw=0, zorder=2)
        ax.errorbar(x, s["mean"], yerr=[[s["mean"] - s["ci95"][0]], [s["ci95"][1] - s["mean"]]],
                    fmt="o", color=col, ms=ms, capsize=2.5, lw=1.1, zorder=3)

    def finish(ax, title, ylab, ref):
        ax.axhline(ref, color=PAL["mute"], lw=0.7, ls="--", zorder=1)
        ax.set_title(title, fontsize=9, fontweight="bold", color=PAL["ink"])
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(labelsize=7.5, colors=PAL["ink"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.9), facecolor="white")
    for ax, metric, title, ylab, ref in (
            (axes[0, 0], "dmass", "Change per action group", r"$\Delta\pi(A_g)$", 0.0),
            (axes[0, 1], "tv_over_mass", "Where the change goes", r"TV share / $\pi(A_g)$", 1.0)):
        for gi, g in enumerate(GROUPS):
            for c, (col, _, ci) in style.items():
                point(ax, gi + (ci - 1) * 0.26, summary["cohort"][c][f"{metric}_{g}"], col)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([names[g] for g in GROUPS], fontsize=7.5)
        finish(ax, title, ylab, ref)
    sw = summary["distance_sweep"]["cohort"]["all4"]
    ds = sorted(sw, key=int)
    for ax, metric, title, ylab, ref in (
            (axes[1, 0], "dmass_star", r"Coin-directed change vs distance", r"$\Delta\pi(A_\star)$", 0.0),
            (axes[1, 1], "tv_over_mass_star", r"Where the change goes vs distance", r"TV share / $\pi(A_\star)$", 1.0)):
        for c, (col, _, ci) in style.items():
            xs = np.array([int(d) for d in ds]) + (ci - 1) * 0.12
            ax.plot(xs, [sw[d][c][metric]["mean"] for d in ds], color=col, lw=0.9, alpha=0.7, zorder=2)
            for x, d in zip(xs, ds):
                point(ax, x, sw[d][c][metric], col, ms=4)
        ax.set_xticks([int(d) for d in ds])
        ax.set_xlabel("coin distance to the left (cells)", fontsize=8)
        finish(ax, title, ylab, ref)
    handles = [Line2D([], [], color=col, marker="o", ls="", ms=5, label=lab) for col, lab, _ in style.values()]
    handles.append(Line2D([], [], color=PAL["mute"], ls="--", lw=0.8, label="no change / no preferred direction"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["all", "recompute", "analyze"], nargs="?", default="all")
    ap.add_argument("--source", choices=list(SOURCES), help="recompute stage only")
    ap.add_argument("--bank", type=Path,
                    default=DATA_ROOT / "probes/coin_canonical_bank_20260924/coin_canonical_bank.json")
    ap.add_argument("--ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/coin")
    ap.add_argument("--saved", type=Path,
                    default=DATA_ROOT / "probes/coin_worldedit_canon_20260924/coin_worldedit_canon.json")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "probes/coin_edit_action_groups_20260925")
    ap.add_argument("--fig", type=Path,
                    default=ROOT / "paper_figures/fig_panel_coin_action_groups")
    ap.add_argument("--max-levels", type=int, default=40_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--match-tol", type=float, default=1e-4,
                    help="max |KL - saved KL| over the six checkpoints for a state to count as identical")
    args = ap.parse_args()
    if args.stage == "recompute":
        recompute(args)
    elif args.stage == "analyze":
        analyze(args)
    else:
        # One fresh process per source: long procgen loops in one process leak memory here.
        for src in SOURCES:
            subprocess.run([sys.executable, __file__, "recompute", "--source", src,
                            "--bank", str(args.bank), "--ckpt-dir", str(args.ckpt_dir),
                            "--out", str(args.out), "--max-levels", str(args.max_levels),
                            "--threads", str(args.threads)], check=True, env=os.environ.copy())
        analyze(args)


if __name__ == "__main__":
    main()
