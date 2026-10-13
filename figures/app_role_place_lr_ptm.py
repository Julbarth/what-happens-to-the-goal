"""'What kind of place is the target' figure: LocationReach (and PushToMarker) arrival clouds under base, camera and image shifts, with rates."""
from __future__ import annotations

from paths import OUT_DIR, TABLES
import argparse
import json
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style as fs  # noqa: E402
from style import PAL  # noqa: E402

PROXY, ALT = "#c2410c", PAL["ink"]
DASH = (0, (2.2, 1.4))
T975 = 4.303
LR_COLS = (("base", "Unedited"), ("shifty", "Base shift"), ("camy", "Camera shift"), ("imgum16", "Image shift"))
# GM without a marker isolates the proxy, except under the camera shift, where the text's test draws the marker at the
# real centre for both cohorts; Non-GM is shown only where such a marker gives it a target.
PTM_COLS = ((("nomarker", "base"), "Unedited"), (("shifty_nomarker", None), "Base shift"),
            (("camy", "camy"), "Camera shift"), (("imgu8_nomarker", None), "Image shift"))


def T(x, y):
    """World (x, y) to the policy camera's orientation: image right = +y, image up = -x."""
    return np.asarray(y), -np.asarray(x)


def ring(ax, xy, r, colour, dashed, lw=1.0):
    ax.add_patch(mp.Circle(xy, r, fill=False, ec="white", lw=lw + 1.0, zorder=6))
    ax.add_patch(mp.Circle(xy, r, fill=False, ec=colour, lw=lw, ls=DASH if dashed else "-", zorder=7))


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color(PAL["mute"])


def titled(ax, letter, title):
    ax.set_title(rf"$\bf{{({letter})}}$ {title}", loc="left", fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)


def cloud(ax, d, env, cond_gm, cond_ng, view, centre, scale_cm):
    """Arrival points of both cohorts (all seeds), the table point ringed, the alternative point as a diamond."""
    g = d[(d.env == env) & (d.cond == cond_gm)]
    c = T(g.corner_x.iloc[0], g.corner_y.iloc[0])
    alt = T(g.alt_x.iloc[0], g.alt_y.iloc[0])
    ax.set_facecolor("#F7F1EA")
    for coh, cond, name in (("control", cond_ng, "Non-GM"), ("gm", cond_gm, "GM")):
        if cond is None:
            continue
        e = d[(d.env == env) & (d.cond == cond) & (d.cohort == coh)]
        ax.scatter(*T(e.x, e.y), s=1.6, color=fs.COH[name], alpha=0.28, lw=0, zorder=3 if name == "GM" else 2,
                   rasterized=True)
    ring(ax, c, 0.03, PROXY, True)
    if np.linalg.norm(np.subtract(alt, c)) > 1e-6:
        ax.plot(*alt, marker="D", ms=4.2, mfc="none", mec="white", mew=1.8, zorder=8)
        ax.plot(*alt, marker="D", ms=4.2, mfc="none", mec=ALT, mew=0.8, zorder=9)
        ax.annotate("", xy=alt, xytext=c, arrowprops=dict(arrowstyle="-|>", color=ALT, lw=0.6, shrinkA=9, shrinkB=5,
                                                          mutation_scale=6), zorder=8)
    ax.set_xlim(centre[0] - view, centre[0] + view)
    ax.set_ylim(centre[1] - view, centre[1] + view)
    frame(ax)
    x0, y0 = centre[0] - view * 0.9, centre[1] - view * 0.86
    ax.plot([x0, x0 + scale_cm / 100], [y0, y0], color=PAL["ink"], lw=1.0, solid_capstyle="butt", zorder=9)
    ax.text(x0 + scale_cm / 200, y0 + view * 0.05, f"{scale_cm} cm", ha="center", va="bottom", fontsize=fs.FS_TICK - 1)


def rates(r, env, cond, coh):
    """Mean and 95% t-interval over the three seeds of reaching the table point and the alternative point."""
    g = r[(r.env == env) & (r.cond == cond) & (r.cohort == coh)]
    out = {}
    for m in ("reach_table", "reach_alt"):
        x = g[m].to_numpy(float)
        if np.isnan(x).any():
            continue
        h = T975 * x.std(ddof=1) / np.sqrt(len(x))
        out[m] = (x.mean(), max(0.0, x.mean() - h), min(1.0, x.mean() + h), x)
    return out


def rate_panel(ax, r, env, cols, cohorts):
    """Dot plot: per condition, rate of reaching the table point (filled) and the alternative point (open)."""
    rows, ticks = [], []
    y = 0
    short = {"Unedited": "unedited", "Base shift": "base", "Camera shift": "camera"}
    for (cg, cn), lab in cols:
        y0 = y
        for coh, name, cond in (("gm", "GM", cg), ("control", "Non-GM", cn)):
            if name not in cohorts or cond is None:
                continue
            q = rates(r, env, cond, coh)
            for m, off, filled in (("reach_table", 0.13, True), ("reach_alt", -0.13, False)):
                if m not in q:
                    continue
                mu, lo, hi, x = q[m]
                col = fs.COH[name]
                ax.plot([lo, hi], [y + off] * 2, color=col, lw=fs.LW_ERR, zorder=2)
                ax.plot(mu, y + off, "o", ms=fs.MS, mfc=col if filled else "white", mec=col, mew=0.9, zorder=3)
                rows.append(dict(env=env, cond=cond, cohort=name, metric=m, mean=mu, lo=lo, hi=hi,
                                 per_seed=[round(float(v), 3) for v in x]))
            y -= 0.62
        ticks.append(((y0 + y + 0.62) / 2, short.get(lab, "image")))
        y -= 0.45
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks], fontsize=fs.FS_TICK - 0.5)
    ax.set_ylim(y + 0.2, 0.45)
    ax.set_xlim(-0.05, 1.07)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("fraction of episodes")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
    fs.finish(ax)
    return rows


def lr_cols_as_pairs():
    return tuple(((c, c), lab) for c, lab in LR_COLS)


def legend(fig, y_in, with_ptm=False):
    h = [Line2D([], [], ls="", marker="o", ms=3.5, color=fs.COH["GM"], label="GM"),
         Line2D([], [], ls="", marker="o", ms=3.5, color=fs.COH["Non-GM"], label="Non-GM"),
         Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=PROXY, mew=0.9, label="table corner" + (" / centre" if with_ptm else "")),
         Line2D([], [], ls="", marker="D", ms=4, mfc="none", mec=ALT, mew=0.8, label="old-posture / old-pixel point"),
         Line2D([], [], ls="", marker="o", ms=3.5, mfc=PAL["ink"], mec=PAL["ink"], label="reached table point"),
         Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec=PAL["ink"], mew=0.9, label="reached other point")]
    fs.legend_row(fig, h[:4], [x.get_label() for x in h[:4]], y_in=y_in + 0.17, handlelength=1.0, columnspacing=0.9)
    fs.legend_row(fig, h[4:], [x.get_label() for x in h[4:]], y_in=y_in, handlelength=1.0, columnspacing=0.9)


def variant_clouds(d, r, with_ptm: bool, stem: str):
    """Row of four LR clouds plus the rate panel; optionally a PTM row below."""
    nrow = 2 if with_ptm else 1
    side, gx, gy, left, top, bottom = 0.86, 0.07, 0.36, 0.17, 0.22, 0.72
    rate_w, rate_gap = 1.05, 0.5
    if with_ptm:
        # Two stacked rows need room between the rate panel's x-label and the next row's titles.
        # The rate panel's condition labels sit left of its axis, so it needs a wider gap than the cloud panels.
        side, gx, gy, top, bottom = 0.76, 0.12, 0.58, 0.26, 0.80
        rate_w, rate_gap = 0.98, 0.74
    w = fs.W_IN
    h = top + nrow * side + (nrow - 1) * gy + bottom
    fig = plt.figure(figsize=(w, h))
    rows_out = []
    specs = [("LR", lr_cols_as_pairs(), 0.065, (-0.165, -0.15), 5, ("GM", "Non-GM"))]
    if with_ptm:
        specs.append(("PTM", PTM_COLS, 0.14, (0.02, 0.08), 10, ("GM", "Non-GM")))
    letters = iter("abcdefghij")
    for ri, (env, cols, view, ctr, sc, cohorts) in enumerate(specs):
        yb = (bottom + (nrow - 1 - ri) * (side + gy)) / h
        for ci, ((cg, cn), lab) in enumerate(cols):
            ax = fig.add_axes([(left + ci * (side + gx)) / w, yb, side / w, side / h])
            cloud(ax, d, env, cg, cn, view, ctr, sc)
            titled(ax, next(letters), lab.split()[0] if with_ptm else lab)
            if ci == 0:
                ax.set_ylabel({"LR": "LocationReach", "PTM": "PushToMarker"}[env], fontsize=fs.FS)
        ax = fig.add_axes([(left + 4 * (side + gx) - gx + rate_gap) / w, yb, rate_w / w, side / h])
        rows_out += rate_panel(ax, r, env, cols, cohorts)
        titled(ax, next(letters), "Reach rate")
    legend(fig, 0.05, with_ptm)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"app_role/{stem}.{ext}", dpi=300)
    plt.close(fig)
    return rows_out


def variant_shift_fraction(d, r, stem: str):
    """Compact alternative: each arrival projected on the corner-to-alternative axis (0 = table point, 1 = other point)."""
    w = fs.W_IN
    fig = plt.figure(figsize=(w, 1.75))
    labs = ("Base shift", "Camera shift", "Image shift")
    specs = (("LR", (("shifty", "shifty"), ("camy", "camy"), ("imgum16", "imgum16")), "LocationReach, gripper"),
             ("PTM", (("shifty_nomarker", None), ("camy_nomarker", "camy"), ("imgu8_nomarker", None)),
              "PushToMarker, cube"))
    rng = np.random.default_rng(0)
    out = []
    for i, (env, cols, title) in enumerate(specs):
        ax = fig.add_axes([(0.78 + i * 2.38) / w, 0.45 / 1.75, 2.2 / w, 1.05 / 1.75])
        for j, (cg, cn) in enumerate(cols):
            for k, (coh, name, cond) in enumerate((("gm", "GM", cg), ("control", "Non-GM", cn))):
                if cond is None:
                    continue
                e = d[(d.env == env) & (d.cond == cond) & (d.cohort == coh)]
                a = np.c_[e.alt_x - e.corner_x, e.alt_y - e.corner_y]
                p = np.c_[e.x - e.corner_x, e.y - e.corner_y]
                f = (p * a).sum(1) / (a * a).sum(1)
                yy = -j + (0.17 if k == 0 else -0.17)
                ax.scatter(np.clip(f, -0.6, 1.6), yy + rng.uniform(-0.1, 0.1, len(f)), s=0.8, color=fs.COH[name],
                           alpha=0.25, lw=0, rasterized=True)
                ax.plot(np.median(f), yy, "|", ms=7, mew=1.4, color=PAL["ink"], zorder=4)
                out.append(dict(env=env, cond=cond, cohort=name, median_frac=float(np.median(f)), n=len(f)))
        ax.axvline(0, color=PROXY, lw=0.8, ls=DASH, zorder=1)
        ax.axvline(1, color=ALT, lw=0.8, ls=DASH, zorder=1)
        ax.set_yticks([-j for j in range(len(cols))])
        ax.set_yticklabels(labs if i == 0 else [])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["table point", "other point"], fontsize=fs.FS_TICK - 0.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(rf"$\bf{{({'ab'[i]})}}$ {title}", loc="left", fontsize=fs.FS_TITLE, pad=fs.TITLE_PAD)
        fs.finish(ax)
        ax.tick_params(axis="y", length=0)
    h = [Line2D([], [], ls="", marker="o", ms=3.5, color=fs.COH["GM"], label="GM"),
         Line2D([], [], ls="", marker="o", ms=3.5, color=fs.COH["Non-GM"], label="Non-GM"),
         Line2D([], [], ls="", marker="|", ms=7, mew=1.4, color=PAL["ink"], label="median")]
    fs.legend_row(fig, h, [x.get_label() for x in h], y_in=0.03, handlelength=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"app_role/{stem}.{ext}", dpi=300)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoints", type=Path, default=TABLES / "place_endpoints.csv.gz")
    ap.add_argument("--rates", type=Path, default=TABLES / "place_rates_per_seed.csv")
    a = ap.parse_args()
    fs.style()
    d, r = pd.read_csv(a.endpoints), pd.read_csv(a.rates)
    v = {"lr_clouds": variant_clouds(d, r, False, "variant_place_A_lr_clouds"),
         "lr_ptm_clouds": variant_clouds(d, r, True, "variant_place_B_lr_ptm_clouds"),
         "shift_fraction": variant_shift_fraction(d, r, "variant_place_C_shift_fraction")}
    (TABLES / "place_values.json").write_text(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
