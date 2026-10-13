"""Layout variants (A, B, C) of a wide GM vs Non-GM rollout-trace figure for the five robot tasks and Coin."""
from __future__ import annotations

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

import coin_traces as C  # noqa: E402
import robot_traces as R  # noqa: E402

S, PAL = R.S, R.PAL
COH = {"gm": PAL["gm"], "control": PAL["nongm"]}
GOAL_C, PROXY_C = "#2E9E4F", "#E07B00"
DASH = (0, (2.2, 1.4))
ORDER = ["AT", "SZ", "LR", "PTM", "Isaac", "Coin"]
NAME = dict(AT="AttributeTable", SZ="SortZones", LR="LocationReach", PTM="PushToMarker", Isaac="WalkToFlag", Coin="Coin")
TASK_LINE = dict(AT="touch the cube\nproxy: red", SZ="cube to crossed pad\nproxy: green pad",
                 LR="reach the ball\nproxy: train. corner", PTM="cube onto marker\nproxy: table centre",
                 Isaac="walk to the flag\nproxy: tower", Coin="get the coin\nproxy: level end")
N_CAND = 8                  # AT and SZ saved reset frames exist only for the first 8 conflict-bank entries
CUBE_Z = 0.02
ROBOT = ROBOTICS_ROOT


def load_tasks(robot: Path, coin_data: Path) -> dict:
    t = {k: R.LOADERS[k](robot) for k in ("AT", "SZ", "LR", "PTM")}
    # Settled-reset Isaac traces, as cited in the robot-trace caption.
    t["Isaac"] = R.load_isaac(robot, robot / "results/isaac_analysis/closed_loop_fixed/trace")
    t["Isaac"].ring = 0.8
    t["Coin"] = C.load_coin(coin_data)
    return t


def seeds_of(task) -> dict:
    """LR banks are per seed pair (goal placed on GM seed k's own path), so only pair 1 shares one start state."""
    return {c: [1] for c in ("gm", "control")} if task.key == "LR" else task.seeds


def pattern(task, i: int) -> str:
    """Joint outcome of all drawn seeds on start state i, e.g. 'GM 3 proxy | Non-GM 3 goal'."""
    oc = outcomes(task, i)
    return " | ".join(f"{R.COH_LABEL[c]} " + " ".join(f"{n} {o}" for o, n in sorted(pd.Series(oc[c]).value_counts().items()))
                      for c in ("gm", "control"))


def pick_modal(task) -> tuple[int, pd.DataFrame]:
    """Representative, declared rule: the first start state in bank order whose joint outcome over all drawn seeds is
    the most common joint outcome in the bank. The pattern's bank frequency is reported with the pick."""
    n = len(task.eps[("gm", seeds_of(task)["gm"][0])])
    pats = pd.Series([pattern(task, i) for i in range(n)])
    top = pats.value_counts()
    i = int(np.flatnonzero(pats == top.index[0])[0])
    return i, pd.DataFrame(dict(task=task.key, pattern=top.index, n_states=top.values, frac=top.values / n,
                                picked=top.index == top.index[0]))


def pick_blind(task) -> tuple[int, pd.DataFrame]:
    """Outcome-blind check: among the first N_CAND bank entries, the one with the median goal-proxy separation
    (lower median, ties by bank order)."""
    s0 = seeds_of(task)["gm"][0]
    sep = np.array([np.linalg.norm(np.subtract(task.layout(i, s0)["goal"], task.layout(i, s0)["proxy"]))
                    for i in range(N_CAND)])
    i = int(np.argsort(sep, kind="stable")[(N_CAND - 1) // 2])
    return i, pd.DataFrame(dict(task=task.key, bank_index=range(N_CAND), sep=sep, picked=np.arange(N_CAND) == i))


def outcomes(task, i: int) -> dict:
    sd = seeds_of(task)
    return {c: [task.eps[(c, s)][i].outcome for s in sd[c]] for c in sd}


def bank_rates(task) -> dict:
    sd = seeds_of(task)
    out = {}
    for c in sd:
        want = "proxy" if c == "gm" else "goal"
        r = [np.mean([e.outcome == want for e in task.eps[(c, s)]]) for s in sd[c]]
        out[c] = (min(r), max(r))
    return out


# ---------------------------------------------------------------- top-down drawing
def arm_xy(p):
    """World (x, y) -> plot (y, x) so the robot base (-x) is at the top, as in the policy camera."""
    p = np.asarray(p)
    return p[..., ::-1]


def ring(ax, xy, r, color, z=2.6):
    ax.add_patch(Circle(xy, r, fill=False, ec=color, lw=0.9, ls=DASH, zorder=z))


def draw_scene(ax, task, i: int, s0):
    L = task.layout(i, s0)
    k = task.key
    if k == "Coin":
        L["draw"](ax)
        ring(ax, L["goal"], task.ring, GOAL_C)
        ring(ax, L["proxy"], task.ring, PROXY_C)
        return L
    if k == "Isaac":
        L["draw"](ax)
        ring(ax, L["goal"], task.ring, GOAL_C)
        ring(ax, L["proxy"], task.ring, PROXY_C)
        return L
    g, p = arm_xy(L["goal"]), arm_xy(L["proxy"])
    if k == "AT":
        ax.plot(*g, "s", ms=3.6, color=R.OBJ["blue"], mec="none", zorder=3)
        ax.plot(*p, "o", ms=3.8, color=R.OBJ["red"], mec="none", zorder=3)
    elif k == "SZ":
        for xy, col in ((g, R.OBJ["purple"]), (p, R.OBJ["green"])):
            ax.add_patch(Rectangle(xy - 0.03, 0.06, 0.06, color=col, alpha=0.55, lw=0, zorder=2))
        ax.plot(*g, "P", ms=3.4, color="white", mec="none", zorder=2.5)
    elif k == "LR":
        z = task_lr_scene(task, i, s0)
        ax.plot(*arm_xy(z["dxy"]), "s", ms=3.4, color=R.OBJ["yellow"], mec="none", zorder=2)
        ax.plot(*g, "o", ms=3.6, color=R.OBJ["green"], mec="none", zorder=3)
        ax.plot(*p, "x", ms=3.6, color=PROXY_C, mew=1.0, zorder=3)
    elif k == "PTM":
        ax.plot(*g, "o", ms=4.6, color="white", mec="#9A9A9A", mew=0.6, zorder=2)
        ax.plot(*p, "x", ms=3.6, color=PROXY_C, mew=1.0, zorder=3)
    r = task.ring
    ring(ax, g, r, GOAL_C)
    ring(ax, p, r, PROXY_C)
    return L


def task_lr_scene(task, i, seed):
    z = np.load(ROBOT / f"results/onpath/offpath_paths_s{seed}.npz")
    return dict(dxy=z["dxy"][i])


def frame_axes(ax, task, i, seeds):
    k = task.key
    if k == "Coin":
        (x0, x1), (y0, y1) = C.coin_window(task, i, seeds)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    elif k == "Isaac":
        A = R.ARENA
        ax.add_patch(Rectangle((-A, -A), 2 * A, 2 * A, fill=False, ec="#C8C8C8", lw=0.6, zorder=1))
        ax.set_xlim(-A - 0.3, A + 0.3)
        ax.set_ylim(-A - 0.3, A + 0.3)
    else:
        (x0, y0), (x1, y1) = R.TABLE
        ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ec="#C8C8C8", lw=0.6, zorder=1))
        ax.set_xlim(-0.30, 0.30)
        ax.set_ylim(0.26, -0.33)
        ax.annotate("robot base", xy=(0.5, 1), xycoords="axes fraction", xytext=(0, -1), textcoords="offset points",
                    ha="center", va="top", fontsize=S.FS_TICK - 2, color=PAL["mute"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_paths(ax, task, i, cohorts, lw=0.9):
    sd = seeds_of(task)
    tf = (lambda p: p) if task.key in ("Isaac", "Coin") else arm_xy
    for c in cohorts:
        for s in sd[c]:
            e = task.eps[(c, s)][i]
            if len(e.path) == 0:
                continue
            q = tf(e.path)
            ax.plot(*q.T, color=COH[c], lw=lw, alpha=0.85, solid_capstyle="round", zorder=4)
            ax.plot(*q[-1], "s", ms=2.4, color=COH[c], mec="white", mew=0.3, zorder=5)
    e0 = task.eps[(cohorts[0], sd[cohorts[0]][0])][i]
    ax.plot(*tf(e0.path[0]), "o", ms=3.0, mfc="white", mec=PAL["ink"], mew=0.7, zorder=6)


def mark_disagree(ax, task, i, cohorts):
    oc = outcomes(task, i)
    if any(len(set(oc[c])) > 1 for c in cohorts):
        ax.annotate("*", xy=(1, 1), xycoords="axes fraction", xytext=(-1, -0.5), textcoords="offset points",
                    ha="right", va="top", fontsize=S.FS_TITLE + 1, fontweight="bold", color=PAL["gm"])


def scale_bar(ax, task):
    k = task.key
    if k == "Coin":
        C.top_left_bar(ax, task.scale, f"{task.scale:.0f} tiles")
        return
    (x0, x1), (yb, _) = ax.get_xlim(), ax.get_ylim()
    L, lab = (2.0, "2 m") if k == "Isaac" else (0.10, "10 cm")
    ya = yb + 0.6 if k == "Isaac" else 0.235
    xa = x1 - (0.6 if k == "Isaac" else 0.04) - L
    ax.plot([xa, xa + L], [ya, ya], color=PAL["mute"], lw=0.8, solid_capstyle="butt", zorder=2)
    ax.annotate(lab, xy=(xa, ya), xytext=(-1.5, 0), textcoords="offset points", ha="right", va="center",
                fontsize=S.FS_TICK - 2, color=PAL["mute"])


def topdown(ax, task, i, cohorts, bar=False):
    s0 = seeds_of(task)["gm"][0]
    draw_scene(ax, task, i, s0)
    draw_paths(ax, task, i, cohorts)
    # Limits last: Coin's terrain imshow resets them.
    frame_axes(ax, task, i, seeds_of(task))
    mark_disagree(ax, task, i, cohorts)
    if bar:
        scale_bar(ax, task)


# ---------------------------------------------------------------- policy-camera drawing (AT, SZ, PTM)
class Cam:
    """Fixed 128x128 policy camera of the arm tasks; K and extrinsics saved alongside the saliency frames."""

    def __init__(self, robot: Path):
        f = np.load(robot / "results/analysis/saliency/frames_PTM.npz")
        self.K, self.E = f["K"], f["extr"]
        self.ptm_rgb0 = f["rgb0"]
        self.obs = {k: np.load(robot / f"results/analysis/condition_scenes/{n}.npz")["obs"]
                    for k, n in (("AT", "AT_blue_cube"), ("SZ", "SZ_conflict"))}
        self.at = np.load(robot / "results/analysis/condition_scenes/AT_blue_cube.npz")

    def px(self, xyz):
        xyz = np.atleast_2d(xyz)
        q = (self.K @ (self.E @ np.c_[xyz, np.ones(len(xyz))].T)).T
        return q[:, :2] / q[:, 2:]

    def frame(self, key, i):
        return self.ptm_rgb0[i] if key == "PTM" else self.obs[key][i]

    def ring(self, ax, xy, r, z, color):
        th = np.linspace(0, 2 * np.pi, 60)
        pts = self.px(np.c_[xy[0] + r * np.cos(th), xy[1] + r * np.sin(th), np.full(60, z)])
        ax.add_patch(Polygon(pts, closed=True, fill=False, ec=color, lw=0.9, ls=DASH, zorder=3))

    def path3d(self, task, c, s, i):
        e = task.eps[(c, s)][i]
        if task.key == "AT":
            return self.at[f"tcp_{c}_s{s}"][i, : len(e.path)]
        return np.c_[e.path, np.full(len(e.path), CUBE_Z)]


def camera(ax, cam: Cam, task, i, cohort):
    k = task.key
    ax.imshow(cam.frame(k, i), extent=(0, 128, 128, 0), interpolation="bicubic", zorder=1)
    L = task.layout(i)
    r = task.ring
    cam.ring(ax, L["goal"], r, 0.0, GOAL_C)
    cam.ring(ax, L["proxy"], r, 0.0, PROXY_C)
    if k == "PTM":
        ax.plot(*cam.px([*L["proxy"], 0.0])[0], "x", ms=3.6, color=PROXY_C, mew=1.0, zorder=3)
    for s in task.seeds[cohort]:
        q = cam.px(cam.path3d(task, cohort, s, i))
        ax.plot(*q.T, color="white", lw=1.8, alpha=0.6, solid_capstyle="round", zorder=3.5)
        ax.plot(*q.T, color=COH[cohort], lw=0.9, alpha=0.95, solid_capstyle="round", zorder=4)
        ax.plot(*q[-1], "s", ms=2.4, color=COH[cohort], mec="white", mew=0.3, zorder=5)
    ax.plot(*cam.px(cam.path3d(task, cohort, task.seeds[cohort][0], i)[0])[0], "o", ms=3.0, mfc="white",
            mec=PAL["ink"], mew=0.7, zorder=6)
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    mark_disagree(ax, task, i, [cohort])


# ---------------------------------------------------------------- figure assembly
def col_header(ax, key, y=1.0, pad=2):
    ax.annotate(TASK_LINE[key], xy=(0.5, y), xycoords="axes fraction", xytext=(0, pad), textcoords="offset points",
                ha="center", va="bottom", fontsize=S.FS_TICK - 1.5, color=PAL["mute"], linespacing=1.05)
    ax.annotate(NAME[key], xy=(0.5, y), xycoords="axes fraction", xytext=(0, pad + 13.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=S.FS_SUB, color=PAL["ink"])


def row_label(ax, text):
    ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction", xytext=(-3, 0), textcoords="offset points",
                rotation=90, ha="right", va="center", fontsize=S.FS_SUB, color=PAL["ink"])


def legend(fig, y_in=0.0):
    ringh = HandlerPatch(patch_func=lambda legend, orig_handle, xdescent, ydescent, width, height, fontsize: Circle(
        (width / 2 - xdescent, height / 2 - ydescent), min(width, height) * 0.62, fill=False,
        ec=orig_handle.get_edgecolor(), lw=orig_handle.get_linewidth(), ls=orig_handle.get_linestyle()))
    h = [Line2D([], [], color=COH["gm"], lw=S.LW_DATA), Line2D([], [], color=COH["control"], lw=S.LW_DATA),
         Circle((0, 0), 1, fill=False, ec=GOAL_C, lw=0.9, ls=DASH), Circle((0, 0), 1, fill=False, ec=PROXY_C, lw=0.9, ls=DASH),
         Line2D([], [], ls="", marker="x", ms=3.6, color=PROXY_C, mew=1.0),
         Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec=PAL["ink"], mew=0.7),
         Line2D([], [], ls="", marker="s", ms=2.6, color=PAL["mute"]),
         Line2D([], [], ls="", marker="$*$", ms=4.5, color=PAL["gm"])]
    lab = ["GM", "Non-GM", "goal", "proxy", "training location", "start", "end", "seeds disagree"]
    return S.legend_row(fig, h, lab, y_in=y_in, handlelength=1.4, columnspacing=0.8, handler_map={Circle: ringh})


def widths(tasks, picks):
    """Coin panels are side views of a level, wider than tall; give that column its aspect."""
    w = []
    for k in ORDER:
        if k == "Coin":
            (x0, x1), (y0, y1) = C.coin_window(tasks[k], picks[k], tasks[k].seeds)
            w.append(min(1.9, (x1 - x0) / (y1 - y0)))
        else:
            w.append(1.0)
    return w


def save(fig, stem: Path, dpi):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def variant_a(tasks, picks, stem, dpi):
    fig = plt.figure(figsize=(S.W_IN, 2.55))
    gs = fig.add_gridspec(2, 6, width_ratios=widths(tasks, picks), left=0.035, right=0.995, top=0.80, bottom=0.10,
                          wspace=0.10, hspace=0.06)
    for j, k in enumerate(ORDER):
        for r, c in enumerate(("gm", "control")):
            ax = fig.add_subplot(gs[r, j])
            topdown(ax, tasks[k], picks[k], [c], bar=r == 1)
            if r == 0:
                col_header(ax, k)
            if j == 0:
                row_label(ax, R.COH_LABEL[c])
    legend(fig)
    save(fig, stem, dpi)


def variant_b(tasks, picks, cam, stem, dpi):
    fig = plt.figure(figsize=(S.W_IN, 3.6))
    gs = fig.add_gridspec(3, 6, width_ratios=widths(tasks, picks), left=0.035, right=0.995, top=0.86, bottom=0.07,
                          wspace=0.10, hspace=0.06)
    for j, k in enumerate(ORDER):
        t, i = tasks[k], picks[k]
        for r, c in enumerate(("gm", "control")):
            ax = fig.add_subplot(gs[r, j])
            if k in ("AT", "SZ", "PTM"):
                camera(ax, cam, t, i, c)
            else:
                topdown(ax, t, i, [c])
                ax.annotate("top-down" if k != "Coin" else "side view", xy=(0, 0), xycoords="axes fraction",
                            xytext=(1, 1), textcoords="offset points", ha="left", va="bottom",
                            fontsize=S.FS_TICK - 2.5, color=PAL["mute"], style="italic")
            if r == 0:
                col_header(ax, k)
            if j == 0:
                row_label(ax, f"{R.COH_LABEL[c]}, camera")
        ax = fig.add_subplot(gs[2, j])
        topdown(ax, t, i, ["control", "gm"], bar=True)
        if j == 0:
            row_label(ax, "both, top-down")
    legend(fig)
    save(fig, stem, dpi)


def variant_c(tasks, picks, stem, dpi):
    fig = plt.figure(figsize=(S.W_IN, 1.55))
    gs = fig.add_gridspec(1, 6, width_ratios=widths(tasks, picks), left=0.005, right=0.995, top=0.70, bottom=0.16,
                          wspace=0.10)
    for j, k in enumerate(ORDER):
        ax = fig.add_subplot(gs[0, j])
        topdown(ax, tasks[k], picks[k], ["control", "gm"], bar=True)
        col_header(ax, k)
    legend(fig)
    save(fig, stem, dpi)


def pick_table(tasks, rule):
    picks, cands, rows = {}, [], []
    for k in ORDER:
        t = tasks[k]
        picks[k], cd = (pick_modal if rule == "modal" else pick_blind)(t)
        cands.append(cd)
        oc, br, sd = outcomes(t, picks[k]), bank_rates(t), seeds_of(t)
        for c in sd:
            rows.append(dict(rule=rule, task=k, bank_index=picks[k], start_state=picks[k] + 1, cohort=R.COH_LABEL[c],
                             seeds=" ".join(map(str, sd[c])), outcomes=" ".join(oc[c]), agree=len(set(oc[c])) == 1,
                             expected="proxy" if c == "gm" else "goal",
                             bank_rate_expected_min=round(br[c][0], 3), bank_rate_expected_max=round(br[c][1], 3),
                             level=int(t.bank.level[picks[k]]) if k == "Coin" else None))
    return picks, pd.concat(cands), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-root", type=Path, default=ROBOT)
    ap.add_argument("--coin-data", type=Path, default=TABLES / "coin")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    S.style()
    tasks = load_tasks(a.robot_root, a.coin_data)
    picks, cand_m, rows_m = pick_table(tasks, "modal")
    picks_b, cand_b, rows_b = pick_table(tasks, "blind")
    for k in ("AT", "SZ"):
        assert picks[k] < N_CAND, f"{k}: no saved reset frame for bank entry {picks[k]}"
    cand_m.to_csv(a.out_dir / "pick_patterns_modal.csv", index=False)
    cand_b.to_csv(a.out_dir / "pick_candidates_blind.csv", index=False)
    pd.concat([rows_m, rows_b]).to_csv(a.out_dir / "picks.csv", index=False)
    print(pd.concat([rows_m, rows_b]).to_string())
    print(cand_m.groupby("task").head(2).to_string())
    cam = Cam(a.robot_root)
    variant_a(tasks, picks, a.out_dir / "variant_A_topdown_2rows", a.dpi)
    variant_a(tasks, picks_b, a.out_dir / "check_A_outcome_blind_pick", a.dpi)
    variant_b(tasks, picks, cam, a.out_dir / "variant_B_camera_plus_topdown", a.dpi)
    variant_c(tasks, picks, a.out_dir / "variant_C_overlay_1row", a.dpi)


if __name__ == "__main__":
    main()
