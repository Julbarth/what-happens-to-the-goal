"""Appendix figure: one example state per evaluation bank, with the two conflicting progress actions drawn."""
from __future__ import annotations
from paths import DATA_ROOT, OUT_DIR, ROOT, TABLES

# Run from the procgen/ directory:
#   uv run --with gym==0.23.1 python figures/app_bank_examples.py
# Stage 1 harvests the first state of each Maze/Coin bank on CPU with the producers' own harvest code and
# caches it in frames_<env>.npz (--rerender redoes this); one subprocess per environment. The CPU harvest of the
# Maze Fisher bank reproduces the bank's frames.
# Attribute Maze panels are drawn from the frozen bank records (walls, agent, line, gem, progress-action sets),
# because the colour-shape procgen fork has no arm64 build.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

STAGE_PATHS = ("external/procgenAISC", "external/train-procgen-pytorch", "scripts/procgen")
MAZE_CK = DATA_ROOT / "checkpoints/head_rewire/maze"
COIN_BANK = DATA_ROOT / "probes/coin_canonical_bank_20260924/coin_canonical_bank.json"
COIN_SHA = "c0961932eadb00fa3704ec6386259070cd6707b8084d7c5f8c3623a96440f1f4"
COIN_WARM = {"gm_rollouts": DATA_ROOT / "checkpoints/coin/6033_gm_20M.pth",
             "nongm_rollouts": DATA_ROOT / "checkpoints/coin/7001_teach.pth"}
RAS_BANK = DATA_ROOT / "probes/ras_gt_bank_v2_20260924/ras_gt_bank_canonical.json"
RAS_SHA = "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9"
DIV_BANK = DATA_ROOT / "probes/ras_divbank_20260924/ras_divbank.json"
DIV_SHA = "7ff7bde70336908e805c8286dea0ff95e7d082b2fc0b6770f45884d317bf95e7"
# First frozen state IDs.
MAZE_STEER_NONGM_FIRST = 1010007
# A different canonical state from the one in fig_worldedit_examples (00156eb0), with the opposite sign.
CANON_STATE = "02697ae2"

GOAL, PROXY, INK, MUTE = "#1b7837", "#c2410c", "#2A2A2A", "#6E6E6E"
# procgen movement semantics (image y grows downward in pixels, grid y grows upward in Maze).
MAZE_DIR = {1: (-1, 0), 3: (0, -1), 5: (0, 1), 7: (1, 0)}
AM_DIR = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, 1), "DOWN": (0, -1)}
AM_GROUP = {0: "LEFT", 1: "LEFT", 2: "LEFT", 3: "DOWN", 5: "UP", 6: "RIGHT", 7: "RIGHT", 8: "RIGHT"}


def to_u8(x) -> np.ndarray:
    x = np.asarray(x["rgb"] if isinstance(x, dict) else x)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        x = np.clip(np.round(x * 255.0 if x.max() <= 1.5 else x), 0, 255).astype(np.uint8)
    return x


def render_maze() -> tuple[dict, dict]:
    """A Fisher-bank frame (GM 1080 harvest) and the first Non-GM-sourced steering-bank state."""
    import run_maze_fisher_decision_fork as dfork
    from run_maze_i_goal_proxy_ladder import _iscalar, get_info_maze, load_policy, make_env

    frames, meta = {}, {}
    warm_gm = load_policy(MAZE_CK / "1080_gm.pth", "cpu")
    rows, _ = dfork.harvest_decision_fork(warm_gm, device="cpu", n_frames=8, start_level=1_000_000,
                                          max_levels=50, max_steps=200, source_name="gm_rollouts")
    # Frame 0 has the cheese one step away (17% of the bank has path length 1); show the first typical fork.
    k, r = next((i, row) for i, row in enumerate(rows) if row["path_len_cheese"] >= 3)
    frames["maze_fisher"] = to_u8(r["obs_a"])
    env = make_env("maze", start_level=r["level"])
    env.reset()
    wd = _iscalar(get_info_maze(env), "maze_world_dim")
    env.close()
    meta["maze_fisher"] = {"level": r["level"], "t": r["t"], "wd": int(wd), "agent": list(r["agent_cell"]),
                           "cheese": list(r["cheese_xy"]), "tr": list(r["tr_cell"]),
                           "a_cheese": r["a_cheese"], "a_corner": r["a_corner"], "warm": "gm_1080",
                           "bank_frame_index": k, "path_len_cheese": r["path_len_cheese"], "path_len_tr": r["path_len_tr"]}

    warm_ng = load_policy(MAZE_CK / "1090_teach.pth", "cpu")
    for level in range(1_010_000, 1_010_050):
        env = make_env("maze", start_level=level)
        obs = env.reset()
        hidden = np.zeros((1, warm_ng.embedder.output_dim), dtype=np.float32)
        hit = None
        for t in range(200):
            status, geo = dfork._classify_step(env)
            if status == "decision" and geo is not None:
                hit = (t, geo, dfork._chw01(env))
                break
            chw = np.asarray(obs[0], dtype=np.float32)
            act, hidden = dfork._greedy_act(warm_ng, chw / 255.0 if chw.max() > 1.5 else chw, "cpu", hidden)
            obs, _r, done, _i = env.step(np.asarray([act]))
            if done[0]:
                break
        wd = _iscalar(get_info_maze(env), "maze_world_dim")
        env.close()
        # native_vs_corner_obs rejects levels whose top-right cell coincides with the cheese; the first
        # frozen Non-GM-source state is 1010007, so levels before it must fail that check or have no fork.
        if hit is not None and level == MAZE_STEER_NONGM_FIRST:
            t, geo, obs_now = hit
            frames["maze_steer"] = to_u8(obs_now)
            meta["maze_steer"] = {"level": level, "t": t, "wd": int(wd), "agent": list(geo["agent_cell"]),
                                  "cheese": list(geo["cheese_xy"]), "tr": list(geo["tr_cell"]),
                                  "a_cheese": geo["a_cheese"], "a_corner": geo["a_corner"], "warm": "nongm_1090"}
            break
    assert "maze_steer" in frames, "no fork found at the first frozen Non-GM steering level"
    return frames, meta


def render_coin(source: str) -> tuple[dict, dict]:
    """First frozen canonical-bank state of `source`, with the coin at goal_xy (conflict frame) and at shortcut_xy."""
    from gm_geometry.probes.coin_canonical_bank import load_bank
    from gm_geometry.probes.coin_location import obs_nchw_after_set_coin, set_coin, unwrap_base_env
    from gm_geometry.probes.coin_rollout import harvest_with_state_snapshot, load_policy, make_env

    bank = load_bank(COIN_BANK)
    assert bank["freeze_sha256"] == COIN_SHA
    st = min((s for s in bank["states"] if s["source"] == source), key=lambda s: s["level"])
    fr = harvest_with_state_snapshot(n_frames=1, start_level=st["level"], max_levels=1,
                                     warm_policy=load_policy(COIN_WARM[source], "cpu"), device="cpu",
                                     skip_policy_gate=True)
    assert len(fr) == 1 and fr[0]["level"] == st["level"]
    assert tuple(fr[0]["xy_a"]) == tuple(st["shortcut_xy"]) and tuple(fr[0]["reloc_xy"]) == tuple(st["goal_xy"])
    env = make_env("coinrun_aisc", num_levels=1, start_level=0)
    out = {}
    try:
        env.reset()
        base = unwrap_base_env(env)
        base.set_state(fr[0]["state_reloc"])
        snap = base.get_state()
        for name, xy in (("coin_goal", st["goal_xy"]), ("coin_shortcut", st["shortcut_xy"])):
            base.set_state(snap)
            assert set_coin(env, xy[0], xy[1], 1) == 0
            out[name] = to_u8(obs_nchw_after_set_coin(env))
    finally:
        env.close()
    meta = {k: st[k] for k in ("source", "level", "warm_policy", "agent_x", "agent_y", "agent_cell_x",
                               "goal_xy", "shortcut_xy", "dx_goal_cells", "dx_shortcut_cells")}
    return out, {"coin": meta}


def load_am_examples() -> dict:
    """One canonical state (agent at the (0,0) spawn) and one diversified strict state with a non-UP/RIGHT pair."""
    doc = json.loads(RAS_BANK.read_bytes())
    assert doc["canonical_bank_sha256"] == RAS_SHA
    canon = next(r for r in doc["records"] if r["state_key"].startswith(CANON_STATE))
    div = json.loads(DIV_BANK.read_bytes())
    assert div["divbank_sha256"] == DIV_SHA
    strict = [r for r in div["records"] if r["set"] == "strict" and r["agent"] != [0, 0]
              and set(r["groups_line"]) | set(r["groups_gem"]) >= {"LEFT", "DOWN"} and r["d_line"] >= 3]
    ex = sorted(strict, key=lambda r: r["state_key"])[0]
    return {"canon": canon, "div": ex}


def arrow(ax, x, y, d, col, scale, lw=1.4):
    ax.annotate("", xy=(x + d[0] * scale, y + d[1] * scale), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.45", color=col, lw=lw,
                                shrinkA=0, shrinkB=0), zorder=5)


def draw_maze(ax, img, m):
    """Real observation cropped to the maze; the 64-px frame spans the wd x wd world, grid y points up."""
    import matplotlib.patches as mp
    cell = img.shape[1] / m["wd"]

    def px(c):
        return (c[0] + 0.5) * cell, img.shape[0] - (c[1] + 0.5) * cell

    # corridor colour varies by level, so crop to the agent, cheese and top-right cell plus a margin
    pts = np.array([px(m[k]) for k in ("agent", "cheese", "tr")])
    x0, y0 = pts.min(0) - 2.5 * cell
    x1, y1 = pts.max(0) + 2.5 * cell
    side = min(max(x1 - x0, y1 - y0), img.shape[0])
    cx = min(max((x0 + x1) / 2, side / 2), img.shape[1] - side / 2)
    cy = min(max((y0 + y1) / 2, side / 2), img.shape[0] - side / 2)
    ax.imshow(img, interpolation="nearest")
    ax.set_xlim(cx - side / 2 - 0.5, cx + side / 2 - 0.5); ax.set_ylim(cy + side / 2 - 0.5, cy - side / 2 - 0.5)
    axp = px(m["agent"])
    for a, col in ((m["a_cheese"], GOAL), (m["a_corner"], PROXY)):
        dx, dy = MAZE_DIR[a]
        arrow(ax, axp[0] - 0.5, axp[1] - 0.5, (dx, -dy), col, 2.2 * cell)
    for c, col in ((m["cheese"], GOAL), (m["tr"], PROXY)):
        tx, ty = px(c)
        ax.add_patch(mp.Circle((tx - 0.5, ty - 0.5), 1.1 * cell, fill=False, ec=col, lw=1.0, zorder=4))


def draw_coin(ax, goal, shortcut, m):
    diff = np.abs(goal.astype(int) - shortcut.astype(int)).max(-1) > 0
    from scipy import ndimage
    lab, _ = ndimage.label(ndimage.binary_dilation(diff, iterations=1))
    cents = sorted(ndimage.center_of_mass(diff, lab, range(1, lab.max() + 1)), key=lambda c: c[1])
    (gy, gx), (sy, sx) = cents[0], cents[-1]
    cell = (sx - gx) / (m["shortcut_xy"][0] - m["goal_xy"][0])
    ax_x = gx + (m["agent_x"] - (m["goal_xy"][0] + 0.5)) * cell
    ax_y = gy - (m["agent_y"] - (m["goal_xy"][1] + 0.5)) * cell
    ax.imshow(goal, interpolation="nearest")
    ax.add_patch(__import__("matplotlib").patches.Circle((sx, sy), 0.6 * cell, fill=False, ec=PROXY, lw=1.0, ls="--", zorder=4))
    arrow(ax, ax_x, ax_y, (-1, 0), GOAL, 1.6 * cell)
    arrow(ax, ax_x, ax_y, (1, 0), PROXY, 1.6 * cell)
    half = 3.2 * cell
    ax.set_xlim(ax_x - half, ax_x + half); ax.set_ylim(ax_y + half, ax_y - half)


def draw_am(ax, r):
    """Schematic 5x5 board from the bank record: walls, agent, line, gem, graph-exact progress sets."""
    import matplotlib.patches as mp
    n = r.get("grid_w", 5)
    ax.add_patch(mp.Rectangle((0, 0), n, n, fc="#111111", ec=INK, lw=0.6))
    for (wx, wy) in r.get("walls", []):
        ax.add_patch(mp.Rectangle((wx, wy), 1, 1, fc="#b08a5a", ec="none"))
    lx, ly = r["line"]; gx, gy = r["gem"]; axx, ayy = r["agent"]
    ax.plot([lx + 0.2, lx + 0.8], [ly + 0.8, ly + 0.2], color="#e02020", lw=2.2, solid_capstyle="round")
    ax.add_patch(mp.RegularPolygon((gx + 0.5, gy + 0.5), 4, radius=0.3, fc="#f2c200", ec="none"))
    ax.add_patch(mp.Circle((axx + 0.5, ayy + 0.5), 0.2, fc="#c8d0d4", ec="none"))
    groups_line = r.get("groups_line") or sorted({AM_GROUP[a] for a in r["A_line"] if a in AM_GROUP})
    groups_gem = r.get("groups_gem") or sorted({AM_GROUP[a] for a in r["A_gem"] if a in AM_GROUP})
    for gs, col in ((groups_line, GOAL), (groups_gem, PROXY)):
        for g in gs:
            arrow(ax, axx + 0.5, ayy + 0.5, AM_DIR[g], col, 0.85)
    ax.set_xlim(-0.08, n + 0.08); ax.set_ylim(-0.08, n + 0.08); ax.set_aspect("equal")


def plot(z: dict, meta: dict, am: dict, out_stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6})
    titles = ["Maze, conflict bank\n(shared sample)", "Maze, conflict bank\n(steering sample)",
              "Coin,\nconflict bank", "Attribute Maze,\nconflict bank", "Attribute Maze,\ndiversified bank"]
    fig, axes = plt.subplots(1, 5, figsize=(5.5, 1.42), gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor("white")
    draw_maze(axes[0], z["maze_fisher"], meta["maze_fisher"])
    draw_maze(axes[1], z["maze_steer"], meta["maze_steer"])
    draw_coin(axes[2], z["coin_goal"], z["coin_shortcut"], meta["coin"])
    draw_am(axes[3], am["canon"])
    draw_am(axes[4], am["div"])
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=6.5, color=INK, pad=2.5, linespacing=1.1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(INK); s.set_linewidth(0.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.80, bottom=0.015)
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)


def run_stage(stage: str) -> None:
    for p in reversed(STAGE_PATHS):
        sys.path.insert(0, str(ROOT / p))
    frames, meta = render_maze() if stage == "maze" else render_coin("nongm_rollouts")
    np.savez_compressed(TABLES / f"frames_{stage}.npz", **frames)
    (TABLES / f"frames_{stage}.json").write_text(json.dumps(meta, indent=1, default=list))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--stage", choices=("maze", "coin"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "fig_appendix_bank_examples")
    args = ap.parse_args()
    if args.stage:
        run_stage(args.stage)
        return
    env = os.environ | {"PROCGEN_CMAKE_PREFIX_PATH": os.environ.get("PROCGEN_CMAKE_PREFIX_PATH", "/opt/homebrew/opt/qt@5/lib/cmake")}
    for stage in ("maze", "coin"):
        if args.rerender or not (TABLES / f"frames_{stage}.npz").exists():
            subprocess.run([sys.executable, __file__, "--stage", stage], check=True, env=env)
    z, meta = {}, {}
    for stage in ("maze", "coin"):
        z |= dict(np.load(TABLES / f"frames_{stage}.npz"))
        meta |= json.loads((TABLES / f"frames_{stage}.json").read_text())
    am = load_am_examples()
    meta["am"] = {k: {f: v[f] for f in ("state_key", "agent", "line", "gem", "d_line", "d_gem")} for k, v in am.items()}
    (TABLES / "plotted_states.json").write_text(json.dumps(meta, indent=1))
    plot(z, meta, am, args.out)
    print("wrote", args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
