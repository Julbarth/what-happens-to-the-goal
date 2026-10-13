"""Fisher-weighted input saliency on the canonical Attribute Maze conflict bank (black), by candidate region, base and swap views."""
from __future__ import annotations

import os
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
OUT = DATA_ROOT / "probes/saliency_am_20260925"
FIG = ROOT / "paper_figures"
MAZE_COIN = DATA_ROOT / "probes/saliency_maze_coin_20260925"
BANK = DATA_ROOT / "probes/ras_gt_bank_v2_20260924/ras_gt_bank_canonical.json"
SPLIT = DATA_ROOT / "probes/ras_gt_bank_v2_20260924/splits/ras_gt_bank_v2_split_state_pair_signd1_seed0.json"
CK = DATA_ROOT / "checkpoints/ras"
FROZEN = "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9"
NETS = {"gm_6313": "6313_gm.pth", "gm_6315": "6315_gm.pth", "gm_6317": "6317_gm.pth",
        "ctrl_4371": "4371_teach.pth", "ctrl_4372": "4372_teach.pth", "ctrl_4373": "4373_teach.pth",
        "init_3000": None, "init_3001": None, "init_3002": None}
VIEWS = ("base", "shape_swap", "colour_swap")
UNIT = 64.0 / 5.0
T975_DF2 = 4.302652729911275


def setup_imports() -> None:
    """Put the vendored colour-shape procgen and cs-train-procgen `common` ahead of procgenAISC and train-procgen-pytorch.

    Must run before fisher_input_saliency adds the latter to sys.path.
    """
    import os
    scratch = Path(os.environ["GM_GEOMETRY_ROOT"])
    for p in (ROOT / "scripts" / "procgen", ROOT / "src", scratch / "vendored" / "cs-train-procgen",
              scratch / "vendored" / "cs-procgenAISC"):
        sys.path.insert(0, str(p))
    import procgen.env as pe
    lib = Path(pe.__file__).parent / ".build" / "relwithdebinfo"
    pe.build = lambda debug=False, package=False: str(lib)
    import common.model  # noqa: F401
    assert "cs-procgenAISC" in pe.__file__ and "cs-train-procgen" in sys.modules["common.model"].__file__


def pick_device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- harvest
def harvest(args) -> None:
    """Render base and canonical swap views for all 196 bank states, as run_ras_canon_fisher.build_world does.

    Raw procgen state bytes are platform-dependent, so identity is checked on sha256(obs) and the decoded state_key.
    """
    setup_imports()
    from gm_geometry.probes import ras_bank_io
    from gm_geometry.probes.fisher import make_env
    from gm_geometry.probes.ras_counterfactuals import CANONICAL_EDITS, edit_objects
    from gm_geometry.probes.ras_gt_bank import state_record
    from run_ras_canon_fisher import diff_components, obs_rgb

    if ras_bank_io.bank_info(BANK, background="black")["canonical_bank_sha256"] != FROZEN:
        raise SystemExit("STOP bank is not the frozen canonical bank")
    states = []
    for role in ("fit", "eval"):
        st, _ = ras_bank_io.select_states(BANK, SPLIT, role, background="black")
        states += [(u, role) for u in st]
    states.sort(key=lambda t: t[0]["level"])
    rec = {k: [] for k in ("level", "state_key", "role", "line", "gem", "agent", "d", "axis_sign",
                           "A_line", "A_gem", *[f"obs_{v}" for v in VIEWS])}
    n_obs_ok = 0
    for u, role in states:
        r = u["record"]
        obs = {}
        for v in VIEWS:
            env = make_env(*edit_objects(v), u["level"], 0, "easy", False)
            try:
                o = obs_rgb(env.reset())
                sk = state_record(ras_bank_io.env_raw_state(env), u["level"], "black")["state_key"]
            finally:
                env.close()
            if sk != u["state_key"]:
                raise SystemExit(f"STOP level {u['level']} {v}: state_key differs from bank")
            obs[v] = o
        if hashlib.sha256(obs["base"].tobytes()).hexdigest() != r["obs_sha256"]:
            raise SystemExit(f"STOP level {u['level']}: base obs sha256 differs from bank")
        n_obs_ok += 1
        for v in CANONICAL_EDITS:
            comps = diff_components(obs["base"], obs[v])
            if not (1 <= len(comps) <= 2 and sum(c[0] for c in comps) <= 400):
                raise SystemExit(f"STOP level {u['level']} {v}: pixel change not local {comps}")
        for v in VIEWS:
            rec[f"obs_{v}"].append(obs[v].transpose(2, 0, 1))
        rec["level"].append(u["level"])
        rec["state_key"].append(u["state_key"])
        rec["role"].append(role)
        rec["line"].append(r["line"])
        rec["gem"].append(r["gem"])
        rec["agent"].append(r["agent"])
        rec["d"].append(int(r["d_line"]))
        rec["axis_sign"].append(int(r["axis_sign_lineN_gemE"]))
        rec["A_line"].append(np.isin(np.arange(15), r["A_line"]))
        rec["A_gem"].append(np.isin(np.arange(15), r["A_gem"]))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "bank_am_black.npz", **{k: np.array(v) for k, v in rec.items()})
    ident = {"canonical_bank_sha256": FROZEN, "n_states": len(states), "n_base_obs_sha256_match": n_obs_ok,
             "state_key_match_all_views": True, "edits": {v: list(edit_objects(v)) for v in VIEWS},
             "note": "raw procgen state bytes differ between the arm64 scratch build and the cluster build; "
                     "obs pixels and decoded geometry are identical"}
    (args.out / "bank_am_identity.json").write_text(json.dumps(ident, indent=1))
    print(json.dumps(ident), flush=True)


# ----------------------------------------------------------------------------- saliency
def load_net(label: str, device):
    import torch
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy
    if label.startswith("init"):
        s = int(label.split("_")[1])
        torch.manual_seed(s)
        np.random.seed(s)
    pol = CategoricalPolicy(ImpalaModel(in_channels=3), False, 15)
    if NETS[label] is not None:
        pol.load_state_dict(torch.load(CK / NETS[label], map_location="cpu")["model_state_dict"])
    return pol.to(device).eval()


def cell_grid():
    """Grid x = column, grid y points up (row 0 of the image is y = 4)."""
    c = np.arange(64)
    idx = np.floor((c + 0.5) / UNIT).astype(int)
    X, Y = np.meshgrid(idx, 4 - idx)
    return X, Y


def am_masks(bank, i, view):
    """Exclusive regions: agent > line candidate u > yellow candidate v. u/v are the base-state cells of
    (line, red) and (gem, yellow); swaps change which attributes sit in them, not the cells."""
    X, Y = cell_grid()
    obs = bank[f"obs_{view}"][i]
    sprite = obs.max(0) > 0
    agent = (X == bank["agent"][i][0]) & (Y == bank["agent"][i][1])
    u = (X == bank["line"][i][0]) & (Y == bank["line"][i][1]) & ~agent
    v = (X == bank["gem"][i][0]) & (Y == bank["gem"][i][1]) & ~agent & ~u
    walls_free = ~agent & ~u & ~v
    return {"goal": u, "proxy": v, "agent": agent, "goal_sprite": u & sprite, "proxy_sprite": v & sprite,
            "near_agent": (np.abs(X - bank["agent"][i][0]) <= 1) & (np.abs(Y - bank["agent"][i][1]) <= 1) & walls_free}


def score(args) -> None:
    """One checkpoint (args.net), all three views."""
    import torch
    setup_imports()
    from fisher_input_saliency import fisher_saliency, region_rows
    device = pick_device()
    torch.manual_seed(0)
    bank = dict(np.load(args.out / "bank_am_black.npz"))
    pol = load_net(args.net, device)
    maps = {}
    for view in VIEWS:
        x = bank[f"obs_{view}"].astype(np.float32) / 255.0
        S, P = fisher_saliency(pol, x, device)
        rows = region_rows(S, lambda i, v=view: am_masks(bank, i, v), len(S))
        for r in rows:
            i = r["state"]
            p = P[i]
            r.update(view=view, net=args.net, level=int(bank["level"][i]), d=int(bank["d"][i]),
                     axis_sign=int(bank["axis_sign"][i]), role=str(bank["role"][i]),
                     p_line=float(p[bank["A_line"][i]].sum()), p_gem=float(p[bank["A_gem"][i]].sum()),
                     entropy=float(-(p * np.log(p + 1e-12)).sum()))
        with open(args.out / f"raw_am_{view}_{args.net}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        maps[f"am_{view}_{args.net}"] = S.astype(np.float32)
        print(f"scored {view}/{args.net}: n={len(rows)} mean p_line={np.mean([r['p_line'] for r in rows]):.3f} "
              f"p_gem={np.mean([r['p_gem'] for r in rows]):.3f}", flush=True)
    np.savez_compressed(args.out / f"saliency_maps_{args.net}.npz", **maps)


# ----------------------------------------------------------------------------- summary
def seed_stats(vals):
    a = np.asarray(vals, float)
    m = float(a.mean())
    sd = float(a.std(ddof=1))
    h = T975_DF2 * sd / np.sqrt(len(a))
    return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "per_seed": a.tolist()}


def distances(bank):
    """Euclidean agent-to-cell-centre distance in cells. Path distance is equal for u and v by bank construction."""
    a = bank["agent"].astype(float)
    return (np.hypot(*(bank["line"] - a).T), np.hypot(*(bank["gem"] - a).T))


def summarize(args) -> dict:
    import pandas as pd
    from fisher_input_saliency import distance_adjusted_contrast
    bank = dict(np.load(args.out / "bank_am_black.npz"))
    d_u, d_v = distances(bank)
    out = {}
    for view in VIEWS:
        per = {}
        for label in NETS:
            df = pd.read_csv(args.out / f"raw_am_{view}_{label}.csv")
            e = {"n": int(len(df))}
            for k in ("goal", "proxy", "agent", "rest", "goal_sprite", "proxy_sprite", "near_agent"):
                e[f"share_{k}"] = float(df[f"share_{k}"].mean())
                e[f"enrich_{k}"] = float(df[f"enrich_{k}"].mean())
                e[f"log_enrich_{k}"] = float(np.log(df[f"enrich_{k}"].clip(lower=1e-12)).mean())
            e["proxy_minus_goal_log_enrich"] = float(np.log(df.enrich_proxy / df.enrich_goal).mean())
            c, slope = distance_adjusted_contrast(df, d_u, d_v)
            e["proxy_minus_goal_log_enrich_dist_adj"] = c
            e["log_enrich_per_cell_distance"] = slope
            e["p_line"] = float(df.p_line.mean())
            e["p_gem"] = float(df.p_gem.mean())
            e["mean_entropy"] = float(df.entropy.mean())
            per[label] = e
        keys = list(per["gm_6313"])
        out[view] = {"per_net": per, "cohort": {
            c: {k: seed_stats([per[l][k] for l in NETS if l.startswith(pfx)]) for k in keys}
            for c, pfx in (("GM", "gm"), ("control", "ctrl"), ("init", "init"))}}
    # Paired swap shift: how much of each candidate's log enrichment moves when the yellow line relocates.
    shift = {}
    for label in NETS:
        dfs = {v: pd.read_csv(args.out / f"raw_am_{v}_{label}.csv") for v in VIEWS}
        s = {}
        for v in ("shape_swap", "colour_swap"):
            for k in ("goal", "proxy"):
                s[f"{v}_minus_base_log_enrich_{k}"] = float(np.log(dfs[v][f"enrich_{k}"] / dfs["base"][f"enrich_{k}"]).mean())
        shift[label] = s
    out["swap_shift"] = {"per_net": shift, "cohort": {
        c: {k: seed_stats([shift[l][k] for l in NETS if l.startswith(pfx)]) for k in shift["gm_6313"]}
        for c, pfx in (("GM", "gm"), ("control", "ctrl"), ("init", "init"))}}
    out["distance"] = {"euclid_u_median": float(np.median(d_u)), "euclid_v_median": float(np.median(d_v)),
                       "euclid_u_mean": float(d_u.mean()), "euclid_v_mean": float(d_v.mean()),
                       "path_d_equal_all_states": True, "path_d_median": float(np.median(bank["d"])),
                       "n_axis_sign_plus": int((bank["axis_sign"] == 1).sum())}
    out["notes"] = {
        "S": "sum_c E_{a~pi}[(d log pi(a|o)/d o_{c,x})^2], obs scaled to [0,1], policy's own action distribution",
        "regions": "goal = cell u of (line, red) in the base state; proxy = cell v of (gem, yellow); agent = cell (0,0); "
                   "exclusive, 12.8x12.8 px each; *_sprite = non-black pixels inside the cell in that view",
        "views": "base; shape_swap = (gem,red)_u (line,yellow)_v; colour_swap = (line,yellow)_u (gem,red)_v",
        "enrich": "per-state (S(R)/|R|)/(S(image)/4096); log_enrich = mean over states of log; report exp()",
        "stats": "unit = checkpoint (n=3 per cohort, init = 3 random draws, seeds 3000-3002); 95% t-CI df=2",
        "p_line/p_gem": "policy mass on the bank's first-step action sets toward u / v",
    }
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------- figures
def palette():
    sys.path.insert(0, str(ROOT / "figures"))
    try:
        from palette import PAL
        return PAL
    except Exception:
        return {"gm": "#7A1520", "gm_proxy": "#C45C58", "nongm": "#3D8B8B", "nongm_proxy": "#5FA8A0",
                "init": "#8E8E8E", "ink": "#2A2A2A", "mute": "#6E6E6E"}


def fig_examples(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    bank = dict(np.load(args.out / "bank_am_black.npz"))
    maps = {**np.load(args.out / "saliency_maps_gm_6313.npz"), **np.load(args.out / "saliency_maps_ctrl_4371.npz")}
    PAL = palette()
    # Fixed rule, no selection on saliency: first three bank states (level order) with both candidates >= 3 steps away.
    idx = [i for i in range(len(bank["level"])) if bank["d"][i] >= 3][:3]
    fig, axes = plt.subplots(len(idx), 3, figsize=(6.0, 2.05 * len(idx)))
    for row, i in enumerate(idx):
        obs = bank["obs_base"][i].transpose(1, 2, 0)
        m = am_masks(bank, i, "base")
        axes[row, 0].imshow(obs)
        for k, col in (("goal", "#F2B01E"), ("proxy", PAL["gm_proxy"]), ("agent", "white")):
            ys, xs = np.where(m[k])
            axes[row, 0].add_patch(Rectangle((xs.min() - 0.5, ys.min() - 0.5), xs.max() - xs.min() + 1,
                                             ys.max() - ys.min() + 1, fill=False, ec=col, lw=1.2))
        for j, lab in ((1, "gm_6313"), (2, "ctrl_4371")):
            S = maps[f"am_base_{lab}"][i]
            axes[row, j].imshow(obs.mean(-1), cmap="gray", alpha=1.0)
            axes[row, j].imshow(np.log10(S / S.mean() + 1e-3), cmap="magma", alpha=0.75, vmin=-1, vmax=2)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("state", fontsize=9)
    axes[0, 1].set_title("GM", fontsize=9, color=PAL["gm"])
    axes[0, 2].set_title("Non-GM", fontsize=9, color=PAL["nongm"])
    fig.tight_layout()
    fig.savefig(args.fig / "fig_saliency_examples_am.png", dpi=250, facecolor="white")
    plt.close(fig)
    (args.out / "examples_am.json").write_text(json.dumps({
        "state_indices": [int(i) for i in idx], "levels": [int(bank["level"][i]) for i in idx],
        "nets": ["gm_6313", "ctrl_4371"], "rule": "first three states in level order with d >= 3"}))


def fig_panel_v2(am, args) -> None:
    """Saliency panel for Maze, Coin and Attribute Maze; Maze/Coin values are read from their saved summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    mc = json.loads((MAZE_COIN / "summary.json").read_text())
    PAL = palette()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": PAL["ink"], "axes.labelcolor": PAL["ink"],
                         "xtick.color": PAL["ink"], "ytick.color": PAL["ink"]})
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True)
    cols = {"GM": PAL["gm"], "NonGM": PAL["nongm"], "init": PAL["init"]}
    offs = {"GM": -0.22, "NonGM": 0.0, "init": 0.22}
    panels = ((axes[0], mc["maze_base"]["cohort"], "Maze", {}),
              (axes[1], mc["coin_left"]["cohort"], "Coin", {}),
              (axes[2], am["base"]["cohort"], "Attribute Maze", {"NonGM": "control"}))
    for ax, coh, title, alias in panels:
        for xi, reg in enumerate(("goal", "proxy")):
            for c in ("GM", "NonGM", "init"):
                st = coh[alias.get(c, c)][f"log_enrich_{reg}"]
                y = np.log10(np.exp(np.asarray(st["per_seed"])))
                x0 = xi + offs[c]
                ax.scatter(np.full(len(y), x0) + np.linspace(-0.04, 0.04, len(y)), y, s=14,
                           facecolors=cols[c] if reg == "goal" else "white", edgecolors=cols[c], lw=1, zorder=3,
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
    h = [Line2D([], [], marker="o", ls="", color=cols[c], label=l)
         for c, l in (("GM", "GM"), ("NonGM", "Non-GM"), ("init", "init"))]
    axes[2].legend(handles=h, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.fig / f"fig_panel_saliency_v2.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", nargs="?", default="all", choices=["all", "harvest", "score", "summarize"])
    ap.add_argument("--net", choices=list(NETS))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fig", type=Path, default=FIG)
    args = ap.parse_args()
    if args.stage == "harvest":
        harvest(args)
    elif args.stage == "score":
        score(args)
    else:
        if args.stage == "all":
            run = lambda *a: subprocess.run([sys.executable, __file__, *a, "--out", str(args.out)], check=True)  # noqa: E731
            run("harvest")
            for label in NETS:
                run("score", "--net", label)
        s = summarize(args)
        fig_examples(args)
        fig_panel_v2(s, args)


if __name__ == "__main__":
    main()
