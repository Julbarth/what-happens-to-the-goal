"""Appendix figure: top-down GM vs Non-GM conflict traces (3 GM + 3 Non-GM seeds per start state) for the five robot tasks."""
from __future__ import annotations

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import style as S  # noqa: E402

PAL = S.PAL
COH = {"gm": PAL["gm"], "control": PAL["nongm"]}
COH_LABEL = {"gm": "GM", "control": "Non-GM"}
TABLE = ((-0.20, -0.25), (0.20, 0.25))
ARENA = 5.0
OBJ = dict(blue="#3B6FD0", red="#D03B3B", purple="#A050C8", green="#3CB45A", yellow="#D4B000", orange="#E07B00",
           box="#9A9A9A", marker_edge="#9A9A9A")
# Same constants the robotics scorers use.
AT_HALF, AT_EPS = 0.03, 0.015
PTM_CENTRE = np.array([-0.08, 0.0])
TRACE_LW, TRACE_ALPHA = 0.8, 0.8
LW_ONE = 1.1                # single-rollout panels (one trace per cohort)


@dataclass
class Ep:
    path: np.ndarray            # (T, 2) scored body: TCP, cube or robot base, truncated at the scoring event
    outcome: str                # goal / proxy / neither / fall
    tool: np.ndarray | None = None  # gripper path for push tasks


@dataclass
class Task:
    key: str
    title: str
    seeds: dict
    eps: dict                   # (cohort, seed) -> list[Ep] in bank order
    layout: callable            # (i, seed) -> dict(goal, proxy, draw)
    examples: list              # [(i, seed or None)]
    scale: float = 0.10         # scale-bar length in world units (m)
    arena: str = "table"
    ring: float = 0.035
    push: bool = False
    checks: dict = field(default_factory=dict)


def first_hit(d, eps):
    hit = d < eps
    return np.where(hit.any(1), hit.argmax(1), d.shape[1])


def load_at(R: Path) -> Task:
    """Conflict = blue cube (goal, shape) vs red sphere (proxy, colour) on the 512-scene world-edit geometry."""
    z = np.load(R / "results/analysis/condition_scenes/AT_blue_cube.npz")
    u, v = z["u"], z["v"]
    pu = np.c_[u, np.full(len(u), AT_HALF)][:, None]
    pv = np.c_[v, np.full(len(v), AT_HALF)][:, None]
    # Non-GM has four seeds (2-5); the figure shows the first three in seed order so every panel has 3 + 3 traces.
    # Seed 5 is still rolled through the scorer check and kept in the rates table (in_figure = False).
    seeds = dict(gm=[2, 3, 4], control=[2, 3, 4])
    eps, agree = {}, {}
    for c, ss in dict(gm=[2, 3, 4], control=[2, 3, 4, 5]).items():
        for s in ss:
            tcp, res = z[f"tcp_{c}_s{s}"], z[f"res_{c}_s{s}"]
            du = np.linalg.norm(np.clip(np.abs(tcp - pu) - AT_HALF, 0, None), axis=-1)
            dv = np.clip(np.linalg.norm(tcp - pv, axis=-1) - AT_HALF, 0, None)
            tu, tv = first_hit(du, AT_EPS), first_hit(dv, AT_EPS)
            T1 = tcp.shape[1]
            mine = np.where((tu <= tv) & (tu < T1), "u", np.where(tv < tu, "v", "neither"))
            agree[f"{c}_s{s}"] = float((mine == res).mean())
            end = np.minimum(np.minimum(tu, tv), T1 - 1)
            lab = {"u": "goal", "v": "proxy", "neither": "neither"}
            eps[(c, s)] = [Ep(tcp[i, : end[i] + 1, :2], lab[res[i]]) for i in range(len(u))]

    def layout(i, seed=None):
        def draw(ax):
            ax.plot(*u[i], "s", ms=4.2, color=OBJ["blue"], mec="none", zorder=3)
            ax.plot(*v[i], "o", ms=4.4, color=OBJ["red"], mec="none", zorder=3)
        return dict(goal=u[i], proxy=v[i], draw=draw)

    return Task("AT", "AttributeTable: blue cube (goal) vs red sphere (proxy)", seeds, eps, layout,
                [(0, None), (1, None), (2, None)], checks=dict(touch_rule_agreement=agree))


def load_sz(R: Path) -> Task:
    """Conflict = purple pad with the cross (goal) vs plain green pad (proxy), 125 scenes, cube pushed."""
    z = np.load(R / "results/analysis/condition_scenes/SZ_conflict.npz")
    g, p = z["marked"], z["other"]
    seeds = dict(gm=[0, 1, 2], control=[0, 1, 2])
    lab = {0: "neither", 1: "goal", 2: "proxy"}
    eps = {(c, s): [Ep(z[f"cube_{c}_s{s}"][i], lab[int(z[f"res_{c}_s{s}"][i])], z[f"tcp_{c}_s{s}"][i, :, :2])
                    for i in range(len(g))] for c, ss in seeds.items() for s in ss}

    def layout(i, seed=None):
        def draw(ax):
            ax.plot(*g[i], "s", ms=7.5, color=OBJ["purple"], mec="none", alpha=0.55, zorder=2)
            ax.plot(*g[i], "P", ms=3.6, color="white", mec="none", zorder=2.5)
            ax.plot(*p[i], "s", ms=7.5, color=OBJ["green"], mec="none", alpha=0.55, zorder=2)
        return dict(goal=g[i], proxy=p[i], draw=draw)

    return Task("SZ", "SortZones: pad with the cross (goal) vs green pad (proxy)", seeds, eps, layout,
                [(0, None), (1, None), (2, None)], ring=0.06, push=True)


def load_ptm(R: Path) -> Task:
    """Conflict = off-centre marker (goal) vs empty table centre (proxy); the 512 world-edit scenes put the marker on
    the ray from the cube to the centre, so reaching the centre means pushing the cube over the marker."""
    sc = np.load(R / "results/analysis/rewire_ptm/conflict_scenes.npz")
    m = sc["m"]
    per = pd.read_csv(R / "results/analysis/rewire_ptm/per_episode.csv")
    per = per[(per.arm == "native") & (per.set == "conflict") & (per.cond == "standard")]
    seeds = dict(gm=[0, 1, 2], control=[0, 1, 2])
    eps, agree = {}, {}
    for c, ss in seeds.items():
        for s in ss:
            z = np.load(R / f"results/analysis/rewire_ptm/paths/{c}_s{s}_native_conflict_standard.npz")
            oc = per[(per.cohort == c) & (per.seed == s)].sort_values("episode").outcome.to_numpy()
            f = z["cube"][:, -1]
            on = z["cube_z"] < 0.02 + 0.005
            dm, dc = np.linalg.norm(f - m, axis=1), np.linalg.norm(f - PTM_CENTRE, axis=1)
            mine = np.where((dm < 0.025) & on, "goal", np.where((dc < 0.03) & on, "proxy", "neither"))
            agree[f"{c}_s{s}"] = float((mine == oc).mean())
            eps[(c, s)] = [Ep(z["cube"][i], oc[i], z["tcp"][i, :, :2]) for i in range(len(m))]

    def layout(i, seed=None):
        def draw(ax):
            ax.plot(*m[i], "o", ms=5.5, color="white", mec=OBJ["marker_edge"], mew=0.6, zorder=2)
            ax.plot(*PTM_CENTRE, "x", ms=3.5, color=PAL["ink"], mew=0.8, zorder=2)
        return dict(goal=m[i], proxy=PTM_CENTRE, draw=draw)

    return Task("PTM", "PushToMarker: marker (goal) vs table centre (proxy)", seeds, eps, layout,
                [(0, None), (1, None), (2, None)], push=True, checks=dict(final_state_rule_agreement=agree))


def load_lr(R: Path) -> Task:
    """Off-path set: 500 scenes per seed pair, the goal placed where the GM gripper goes from other starts; scenes are
    shared by GM seed k and Non-GM seed k only. First-contact scoring (t_goal / t_corner)."""
    seeds = dict(gm=[1, 2, 3], control=[1, 2, 3])
    lab = {"goal": "goal", "corner": "proxy", "neither": "neither"}
    eps, scen = {}, {}
    for s in seeds["gm"]:
        z = np.load(R / f"results/onpath/offpath_paths_s{s}.npz")
        df = pd.read_csv(R / f"results/onpath/offpath_s{s}.csv")
        scen[s] = z
        for c in ("gm", "control"):
            d = df[df.cohort == c].sort_values("episode")
            t = np.minimum(d[["t_goal", "t_corner"]].min(axis=1).to_numpy(), z[c].shape[1] - 1).astype(int)
            eps[(c, s)] = [Ep(z[c][i, : t[i] + 1, :2], lab[o]) for i, o in enumerate(d.outcome)]

    def layout(i, seed):
        z = scen[seed]

        def draw(ax):
            ax.plot(*z["dxy"][i], "s", ms=4.2, color=OBJ["yellow"], mec="none", zorder=2)
            ax.plot(*z["goal"][i, :2], "o", ms=4.4, color=OBJ["green"], mec="none", zorder=2)
            ax.plot(*z["corner"][:2], "x", ms=3.5, color=PAL["ink"], mew=0.8, zorder=2)
        return dict(goal=z["goal"][i, :2], proxy=z["corner"][:2], draw=draw)

    return Task("LR", "LocationReach: green ball (goal) vs training corner (proxy)", seeds, eps, layout,
                [(0, 1), (0, 2), (0, 3)])


def load_isaac(R: Path, trace_dir: Path | None = None) -> Task:
    """Isaac ANYmal-C: first 128 rows of the 256-row conflict world-edit bank (flag and tower apart, 3 grey boxes)."""
    seeds = dict(gm=[0, 1, 2], control=[0, 1, 2])
    bank = np.load(R / "results/isaac_analysis/banks/edit_conflict.npz")
    eps, ref = {}, None
    for c, ss in seeds.items():
        for s in ss:
            z = np.load((trace_dir or R / "results/isaac_analysis/trace") / f"conflict_main_{c}_s{s}.npz")
            if ref is None:
                ref = z
                assert np.allclose(z["start"], bank["start"][: len(z["start"])])
            assert np.allclose(z["lay_base_flag"], ref["lay_base_flag"])
            xy = z["base_xy"]
            eps[(c, s)] = [Ep(xy[i][np.isfinite(xy[i, :, 0])], str(z["base_outcome"][i]).replace("timeout", "neither"))
                           for i in range(len(xy))]
    flag, tower, boxes = ref["lay_base_flag"], ref["lay_base_tower"], bank["boxes"]

    def layout(i, seed=None):
        def draw(ax):
            for b in boxes[i]:
                ax.plot(*b, "s", ms=3.2, color=OBJ["box"], mec="none", zorder=2)
            ax.plot(*tower[i], "s", ms=4.6, color=OBJ["orange"], mec="none", zorder=2)
            ax.plot(*flag[i], "^", ms=4.8, color=OBJ["green"], mec="none", zorder=2)
        return dict(goal=flag[i], proxy=tower[i], draw=draw)

    return Task("Isaac", "Quadruped (ANYmal-C): flag (goal) vs tower (proxy)", seeds, eps, layout,
                [(0, None), (1, None), (2, None)], arena="arena", ring=0.6, scale=2.0)


LOADERS = dict(AT=load_at, SZ=load_sz, PTM=load_ptm, LR=load_lr, Isaac=load_isaac)


def rates(task: Task) -> pd.DataFrame:
    rows = []
    for (c, s), eps in task.eps.items():
        oc = pd.Series([e.outcome for e in eps])
        rows.append(dict(task=task.key, cohort=COH_LABEL[c], seed=s, n=len(eps), in_figure=s in task.seeds[c],
                         **{k: float((oc == k).mean()) for k in ("goal", "proxy", "neither", "fall")}))
    return pd.DataFrame(rows)


def endpoints(task: Task) -> pd.DataFrame:
    rows = []
    for (c, s), eps in task.eps.items():
        if s not in task.seeds[c]:
            continue
        for i, e in enumerate(eps):
            if len(e.path) == 0:
                continue
            L = task.layout(i, s)
            sep = np.linalg.norm(np.asarray(L["goal"]) - np.asarray(L["proxy"]))
            end = e.path[-1]
            rows.append(dict(task=task.key, cohort=c, seed=s, scene=i, outcome=e.outcome, sep=sep,
                             d_goal=np.linalg.norm(end - L["goal"]) / sep, d_proxy=np.linalg.norm(end - L["proxy"]) / sep))
    return pd.DataFrame(rows)


def draw_arena(ax, task: Task):
    if task.arena == "table":
        (x0, y0), (x1, y1) = TABLE
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#C8C8C8", lw=0.6, zorder=1))
        # Fixed window wider than the table: Non-GM AT grippers swing back toward the base, beyond the table edge.
        ax.set_xlim(-0.31, 0.27)
        ax.set_ylim(-0.31, 0.31)
        ax.annotate("base", xy=(0, 0.5), xycoords="axes fraction", xytext=(-1.5, 0), textcoords="offset points",
                    rotation=90, ha="right", va="center", fontsize=S.FS_TICK - 1, color=PAL["mute"])
    else:
        ax.add_patch(Rectangle((-ARENA, -ARENA), 2 * ARENA, 2 * ARENA, fill=False, ec="#C8C8C8", lw=0.6, zorder=1))
        ax.set_xlim(-ARENA - 0.3, ARENA + 0.3)
        ax.set_ylim(-ARENA - 0.3, ARENA + 0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def disagree(task: Task, i: int, seed) -> bool:
    """True when the seeds drawn for either cohort do not all share one outcome on this start state."""
    return any(len({task.eps[(c, s)][i].outcome for s in ([seed] if seed is not None else task.seeds[c])}) > 1
               for c in ("gm", "control"))


def scale_bar(ax, length: float, label: str, inset: float = 0.035):
    x1, y0 = ax.get_xlim()[1], ax.get_ylim()[0]
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    xa, ya = x1 - max(0.02, inset) * span - length, y0 + inset * span
    ax.plot([xa, xa + length], [ya, ya], color=PAL["mute"], lw=0.8, solid_capstyle="butt", zorder=2)
    ax.annotate(label, xy=(xa, ya), xytext=(-1.5, 0), textcoords="offset points", ha="right",
                va="center", fontsize=S.FS_TICK - 1.5, color=PAL["mute"])


def draw_traces(ax, task: Task, i: int, seeds: dict, lw=TRACE_LW):
    """Scene objects, goal/proxy rings, then every requested (cohort, seed) trace on start state i."""
    rows = []
    s0 = seeds["gm"][0]
    L = task.layout(i, s0)
    L["draw"](ax)
    ax.add_patch(Circle(L["goal"], task.ring, fill=False, ec=PAL["ink"], lw=0.7, zorder=2.6))
    ax.add_patch(Circle(L["proxy"], task.ring, fill=False, ec=PAL["ink"], lw=0.7, ls=(0, (1.5, 1.2)), zorder=2.6))
    for c in ("control", "gm"):
        for s in seeds[c]:
            e = task.eps[(c, s)][i]
            if e.tool is not None:
                ax.plot(*e.tool.T, color=COH[c], lw=0.4, alpha=0.45, ls=(0, (1, 1)), zorder=3)
            ax.plot(*e.path.T, color=COH[c], lw=lw, alpha=TRACE_ALPHA, solid_capstyle="round", zorder=4)
            ax.plot(*e.path[-1], "s", ms=2.6, color=COH[c], mec="white", mew=0.3, zorder=5)
            rows.append(dict(task=task.key, scene=i, cohort=COH_LABEL[c], seed=s, outcome=e.outcome))
    ax.plot(*task.eps[("gm", s0)][i].path[0], "o", ms=3.4, mfc="white", mec=PAL["ink"], mew=0.7, zorder=6)
    return rows


def draw_example(ax, task: Task, k: int, i: int, seed):
    draw_arena(ax, task)
    seeds = {c: ([seed] if seed is not None else task.seeds[c]) for c in ("gm", "control")}
    rows = draw_traces(ax, task, i, seeds)
    scale_bar(ax, task.scale, f"{task.scale * 100:.0f} cm" if task.scale < 1 else f"{task.scale:.0f} m",
              inset=0.035 if task.arena == "table" else 0.09)
    sub = f"start state {i + 1}" + (f", seed pair {seed}" if seed is not None else "")
    ax.annotate(sub, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 1.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=S.FS_SUB, color=PAL["ink"])
    if disagree(task, i, seed):
        ax.annotate("*", xy=(1, 1), xycoords="axes fraction", xytext=(-2, -1), textcoords="offset points",
                    ha="right", va="top", fontsize=S.FS_TITLE + 1, fontweight="bold", color=PAL["gm"])
    return rows


def draw_population(ax, task: Task, ep: pd.DataFrame):
    lim = 2.0
    for c in ("control", "gm"):
        d = ep[ep.cohort == c]
        ax.scatter(d.d_goal.clip(upper=lim * 0.985), d.d_proxy.clip(upper=lim * 0.985), s=2.2, color=COH[c],
                   alpha=0.22, lw=0, zorder=3 if c == "gm" else 2, rasterized=True)
    ax.plot([0, lim], [0, lim], **S.REF, zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("end to goal / sep.", fontsize=S.FS_TICK, labelpad=0.5)
    ax.set_ylabel("end to proxy / sep.", fontsize=S.FS_TICK, labelpad=0.5)
    S.finish(ax)
    n = ep.groupby("cohort").scene.nunique().max()
    k = {c: ep[ep.cohort == c].seed.nunique() for c in ("gm", "control")}
    ax.annotate(f"all {n} start states, {k['gm']}+{k['control']} seeds", xy=(0.5, 1), xycoords="axes fraction",
                xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=S.FS_SUB,
                color=PAL["ink"])


def draw_row(fig, axes, task: Task, ep: pd.DataFrame, letter=None):
    rows = []
    for k, (ax, (i, seed)) in enumerate(zip(axes[:3], task.examples)):
        rows += draw_example(ax, task, k, i, seed)
    draw_population(axes[3], task, ep)
    S.panel_title(axes[0], task.title, sub="", span=axes[2], sub_only=False)
    if letter:
        axes[0].annotate(f"({letter})", xy=(0.006, 1), xycoords=("figure fraction", "axes fraction"),
                         xytext=(0, S._title_offsets(True)[1]), textcoords="offset points", ha="left", va="bottom",
                         fontsize=S.FS_LETTER, fontweight="bold", color=PAL["ink"])
    return rows


def legend(fig, push_any: bool, y_in=0.02):
    h = [Line2D([], [], color=COH["gm"], lw=S.LW_DATA), Line2D([], [], color=COH["control"], lw=S.LW_DATA),
         Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec=PAL["ink"], mew=0.7),
         Line2D([], [], ls="", marker="s", ms=2.8, color=PAL["mute"]),
         Circle((0, 0), 1, fill=False, ec=PAL["ink"], lw=0.7),
         Circle((0, 0), 1, fill=False, ec=PAL["ink"], lw=0.7, ls=(0, (1.5, 1.2)))]
    h.append(Line2D([], [], ls="", marker="$*$", ms=4.5, color=PAL["gm"]))
    lab = ["GM", "Non-GM", "start", "end", "goal", "proxy", "seeds disagree"]
    if push_any:
        h.append(Line2D([], [], color=PAL["mute"], lw=0.6, ls=(0, (1, 1))))
        lab.append("gripper (push tasks)")
    ring = HandlerPatch(patch_func=lambda legend, orig_handle, xdescent, ydescent, width, height, fontsize: Circle(
        (width / 2 - xdescent, height / 2 - ydescent), min(width, height) * 0.62, fill=False,
        ec=orig_handle.get_edgecolor(), lw=orig_handle.get_linewidth(), ls=orig_handle.get_linestyle()))
    return S.legend_row(fig, h, lab, y_in=y_in, handlelength=1.6, columnspacing=0.9,
                        handler_map={Circle: ring})


def make_fig(tasks, eps_by_task, out_stem: Path, dpi: int):
    n = len(tasks)
    row_h = 1.52
    fig = plt.figure(figsize=(S.W_IN, row_h * n + 0.30))
    gs = fig.add_gridspec(n, 4, width_ratios=[1, 1, 1, 1.05], left=0.06, right=0.985, top=1 - 0.30 / fig.get_figheight(),
                          bottom=0.40 / fig.get_figheight(), wspace=0.30, hspace=0.62)
    rows = []
    for r, t in enumerate(tasks):
        axes = [fig.add_subplot(gs[r, j]) for j in range(4)]
        rows += draw_row(fig, axes, t, eps_by_task[t.key], letter="abcde"[r] if n > 1 else None)
    legend(fig, any(t.push for t in tasks))
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-root", type=Path, default=ROBOTICS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "app_traces")
    ap.add_argument("--data-dir", type=Path, default=TABLES / "data")
    ap.add_argument("--tasks", nargs="+", default=["AT", "SZ", "LR", "PTM", "Isaac"], choices=list(LOADERS))
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    S.style()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.data_dir.mkdir(parents=True, exist_ok=True)
    tasks = [LOADERS[k](a.robot_root) for k in a.tasks]
    eps = {t.key: endpoints(t) for t in tasks}
    pd.concat([rates(t) for t in tasks]).to_csv(a.data_dir / "robot_conflict_rates.csv", index=False)
    pd.concat(eps.values()).to_csv(a.data_dir / "robot_conflict_endpoints.csv", index=False)
    json.dump({t.key: t.checks for t in tasks}, open(a.data_dir / "robot_scoring_checks.json", "w"), indent=1)
    shown = []
    for t in tasks:
        shown += make_fig([t], eps, a.out_dir / f"fig_app_traces_{t.key}", a.dpi)
    make_fig(tasks, eps, a.out_dir / "fig_app_traces_robots", a.dpi)
    pd.DataFrame(shown).to_csv(a.data_dir / "robot_shown_examples.csv", index=False)


if __name__ == "__main__":
    main()
