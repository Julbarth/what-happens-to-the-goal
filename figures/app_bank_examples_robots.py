"""Appendix figure: one stored observation per robot evaluation bank, with goal and proxy circled."""
from __future__ import annotations

# Every panel is the stored policy observation of the bank, not a re-render. Arm goal and proxy centres are
# projected with the fixed obs camera (K, extrinsic) saved by the robotics saliency-frame analysis, which is
# identical for all four arm tasks. WalkToFlag has no stored camera model, so flag and tower are located by a
# colour rule.

from paths import OUT_DIR, ROBOTICS_ROOT, TABLES
import argparse
import json
from pathlib import Path

import numpy as np

GR = ROBOTICS_ROOT
SZ_CONFLICT = GR / "results/analysis/condition_scenes/SZ_conflict.npz"
WTF = GR / "results/isaac_analysis/ghost_check/fixed/banks"
CAM = GR / "results/analysis/saliency/frames_AT.npz"

# Mirrors the robotics env constants; importing the envs themselves needs ManiSkill.
AT_Z, SZ_PAD_Z, PTM_MARKER_Z, LR_Z = 0.03, 0.0011, 0.0011, 0.06
LR_CORNER, PTM_CENTRE = (0.15, -0.20), (-0.08, 0.0)

GOAL, PROXY, INK = "#1b7837", "#c2410c", "#2A2A2A"


def project(cam, xyz):
    xyz = np.atleast_2d(np.asarray(xyz, np.float64))
    pc = np.concatenate([xyz, np.ones((len(xyz), 1))], 1) @ cam["extr"].T
    pix = pc @ cam["K"].T
    return pix[:, :2] / pix[:, 2:3]


def with_z(xy, z):
    return np.array([xy[0], xy[1], z])


def decision_keep(b):
    """The 500 analysed frames of the decision bank (balanced subset, seed 0)."""
    keep = np.ones(len(b["sign"]), bool)
    for e in ("mark_swap", "colour_swap"):
        keep &= b[f"npx_{e}"] >= 0.5 * np.median(b[f"npx_{e}"])
    rng = np.random.default_rng(0)
    groups = {}
    for i in np.where(keep)[0]:
        groups.setdefault((int(b["sign"][i]), int(b["src_cohort"][i])), []).append(i)
    m = min(len(g) for g in groups.values())
    return np.sort(np.concatenate([rng.permutation(g)[:m] for g in groups.values()]))


def wtf_centroids(img):
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    masks = {"goal": (g > 90) & (g > r + 40) & (g > b + 40), "proxy": (r > 150) & (g > 100) & (r - b > 80)}
    return {k: np.array(np.nonzero(m)[::-1]).mean(1) for k, m in masks.items()}


def arm_panels(arm_dir: Path, cam) -> dict:
    """task -> list of (title, image, [(xy_px, role, dashed)], index, note)."""
    L = lambda n: np.load(arm_dir / f"{n}.npz")
    P = lambda xyz: project(cam, xyz)[0]
    out = {}

    b, i = L("LR_probe_test"), 0
    lr = [("Probe bank", b["rgb"][i], [(P(b["goal"][i]), "goal", False), (P(with_z(LR_CORNER, LR_Z)), "proxy", True)], i)]
    b = L("LR_worldedit")
    lr.append(("World-edit bank", b["rgb_g_d"][i], [(P(b["goal_g_d"][i]), "goal", False),
                                                    (P(with_z(LR_CORNER, LR_Z)), "proxy", True)], i))
    out["LocationReach"] = lr

    b = L("PTM_probe_test")
    ptm = [("Probe bank", b["rgb"][i], [(P(with_z(b["marker"][i], PTM_MARKER_Z)), "goal", False),
                                        (P(with_z(PTM_CENTRE, PTM_MARKER_Z)), "proxy", True)], i)]
    b = L("PTM_worldedit")
    ptm.append(("World-edit bank", b["rgb_base"][i], [(P(with_z(b["m"][i], PTM_MARKER_Z)), "goal", False),
                                                      (P(with_z(PTM_CENTRE, PTM_MARKER_Z)), "proxy", True)], i))
    out["PushToMarker"] = ptm

    # Probe scenes are control-distribution: the cube is the goal object, any red object is a proxy object.
    b = L("AT_probe_test")
    marks = [(P(with_z(b["u"][i], AT_Z)), "goal", False)]
    for xy, red in ((b["u"][i], b["cube_red"][i]), (b["v"][i], b["sphere_red"][i])):
        if red:
            marks.append((P(with_z(xy, AT_Z)), "proxy", False))
    at = [("Probe bank", b["rgb"][i], marks, i)]
    for name, title in (("AT_worldedit", "World-edit bank"), ("AT_audit_strict", "Strict bank")):
        b = L(name)
        at.append((title, b["rgb_base"][i], [(P(with_z(b["u"][i], AT_Z)), "goal", False),
                                             (P(with_z(b["v"][i], AT_Z)), "proxy", False)], i))
    out["AttributeTable"] = at

    # SZ goal = crossed pad (u in the conflict banks), proxy = green pad.
    b = L("SZ_probe_test")
    g = bool(b["cross_green"][i])
    marks = [(P(with_z(b["u"][i], SZ_PAD_Z)), "goal", False), (P(with_z(b["u" if g else "v"][i], SZ_PAD_Z)), "proxy", False)]
    sz = [("Probe bank", b["rgb"][i], marks, i)]
    b = L("SZ_worldedit")
    sz.append(("World-edit bank", b["rgb_base"][i], [(P(with_z(b["u"][i], SZ_PAD_Z)), "goal", False),
                                                     (P(with_z(b["v"][i], SZ_PAD_Z)), "proxy", False)], i))
    b = L("SZ_audit_decision")
    j = int(decision_keep(b)[0])
    sz.append(("Decision-frame bank", b["rgb_base"][j], [(P(with_z(b["u"][j], SZ_PAD_Z)), "goal", False),
                                                         (P(with_z(b["v"][j], SZ_PAD_Z)), "proxy", False)], j))
    b = np.load(SZ_CONFLICT)
    sz.append(("Conflict-test sample", b["obs"][i], [(P(with_z(b["marked"][i], SZ_PAD_Z)), "goal", False),
                                                    (P(with_z(b["other"][i], SZ_PAD_Z)), "proxy", False)], i))
    out["SortZones"] = sz
    return out


def wtf_panels() -> list:
    """First state of each bank with both flag and tower in view (>= 5 px, the probe-bank visibility rule)."""
    rows = []
    for fn, key, title in (("probe_test", "rgb", "Probe bank"), ("edit_visible", "rgb_base", "World-edit bank"),
                           ("edit_conflict", "rgb_base", "Forced-view\nconflict bank"),
                           ("native_conflict_eval", "rgb", "Conflict-test bank")):
        b = np.load(WTF / f"{fn}.npz")
        sfx = "_base" if key == "rgb_base" else ""
        i = int(np.nonzero((b["flag_px" + sfx] >= 5) & (b["tower_px" + sfx] >= 5))[0][0])
        c = wtf_centroids(b[key][i])
        rows.append((title, b[key][i], [(c["goal"], "goal", False), (c["proxy"], "proxy", False)], i))
    return rows


def plot(panels: dict, out_stem: Path, ncol: int = 5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6})
    flat = [(task, *p) for task, v in panels.items() for p in v]
    nrow = -(-len(flat) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5, 1.36 * nrow), gridspec_kw={"wspace": 0.08, "hspace": 0.34})
    fig.patch.set_facecolor("white")
    for ax in axes.flat[len(flat):]:
        ax.axis("off")
    for ax, (task, title, img, marks, _) in zip(axes.flat, flat):
        h = img.shape[0]
        ax.imshow(img, interpolation="nearest")
        goals = [xy for xy, role, _ in marks if role == "goal"]
        for xy, role, dashed in marks:
            rad = 0.085 * h
            # A proxy on the goal object (co-located probe scenes) gets an outer ring so both stay visible.
            if role == "proxy" and any(np.linalg.norm(np.asarray(xy) - g) < 0.05 * h for g in goals):
                rad = 0.12 * h
            ax.add_patch(mp.Circle(xy, rad, fill=False, ec=GOAL if role == "goal" else PROXY,
                                   lw=1.1, ls=(0, (2.2, 1.4)) if dashed else "-", zorder=4))
        ax.set_xlim(-0.5, img.shape[1] - 0.5); ax.set_ylim(h - 0.5, -0.5)
        ax.set_title(f"{task},\n{title.replace(chr(10), ' ')[0].lower()}{title.replace(chr(10), ' ')[1:]}",
                     fontsize=6.5, color=INK, pad=2.5, linespacing=1.1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(INK); s.set_linewidth(0.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005)
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm_dir", type=Path, default=GR / "results/analysis/banks", help="arm state banks (<name>.npz)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "app_banks" / "fig_bank_examples_robots")
    a = ap.parse_args()
    z = np.load(CAM)
    cam = {"K": z["K"], "extr": z["extr"]}
    panels = arm_panels(a.arm_dir, cam)
    panels = {k: panels[k] for k in ("LocationReach", "PushToMarker", "AttributeTable", "SortZones")}
    panels["WalkToFlag"] = wtf_panels()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    plot(panels, a.out)
    log = {t: [{"panel": p[0].replace("\n", " "), "index": p[3]} for p in v] for t, v in panels.items()}
    (TABLES / "plotted_states_robots.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log))


if __name__ == "__main__":
    main()
