"""Schematic of the robot Nature CNN actor-critic, drawn in the visual style of fig_architecture_concept.png."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import style as FS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from paths import OUT_DIR

FILL = "#E8EBEE"
EDGE = "#8A9099"
INK = "#2B2B2B"
BOX = "#F7F6FC"
BOX_EDGE = "#B9B7C9"
SERIF = dict(family="serif", style="italic")


def slab(ax, x, y, w, h, n=4, d=0.07):
    """A stack of n channel sheets offset down-right, as in the IMPALA diagram."""
    for k in reversed(range(n)):
        ax.add_patch(Rectangle((x + k * d, y - k * d), w, h, fc=FILL, ec=EDGE, lw=0.6, zorder=2))


def bar(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc=FILL, ec=EDGE, lw=0.6, zorder=2))


def arrow(ax, p, q, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.9, color=INK,
                                 shrinkA=0, shrinkB=0, connectionstyle=conn, zorder=3))


def txt(ax, x, y, s, size=FS.FS_TICK, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    ax.text(x, y, s, fontsize=size, **kw)


def sym(ax, x, y, base, sup):
    """h^{pre}-style label in the serif italic used by the IMPALA figure."""
    ax.text(x, y, rf"$\mathit{{{base}}}^{{\mathrm{{{sup}}}}}$", fontsize=FS.FS + 2, ha="center", va="center",
            color=INK, math_fontfamily="cm")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_DIR / "app_arch/fig_arch_naturecnn"))
    a = p.parse_args()
    FS.style()
    x0, x1, y0, y1 = 0.0, 13.9, -0.75, 6.0
    fig, ax = plt.subplots(figsize=(FS.W_IN, FS.W_IN * (y1 - y0) / (x1 - x0)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    yc, ys, y_lab = 3.6, 0.2, 1.6
    small = FS.FS_TICK - 0.5
    d = 0.07

    # Spatial trunk: three conv + ReLU layers, no pooling or residual units.
    convs = [("Conv 1", "32 x 31 x 31", "8x8, stride 4", 0.3, 1.2, 2.8),
             ("Conv 2", "64 x 14 x 14", "4x4, stride 2", 2.35, 1.0, 2.2),
             ("Conv 3", "64 x 12 x 12", "3x3, stride 1", 4.3, 0.9, 1.8)]
    ax.add_patch(FancyBboxPatch((3.95, 1.98), 4.65, 3.85, boxstyle="round,pad=0,rounding_size=0.15", fc=BOX,
                                ec=BOX_EDGE, lw=0.6, ls=(0, (2, 2)), zorder=0))
    for i, (name, dims, kern, x, w, h) in enumerate(convs):
        slab(ax, x, yc - h / 2, w, h, d=d)
        xm = x + w / 2 + 1.5 * d
        txt(ax, x + w / 2, yc, name, size=small - 0.5)
        txt(ax, xm, yc + h / 2 + 0.3, dims, size=small)
        txt(ax, xm, yc - h / 2 - 4 * d - 0.25, kern, size=small - 0.5, color=FS.PAL["mute"])
        ax.text(xm, y_lab if i == 2 else yc - h / 2 - 1.0, rf"$H^{{({i + 1})}}$", fontsize=FS.FS + 2,
                ha="center", va="center", color=INK, math_fontfamily="cm")
        if i < 2:
            arrow(ax, (x + w + 3 * d + 0.1, yc), (convs[i + 1][3] - 0.05, yc))

    # Flatten into h_pre, one linear layer + ReLU into h_fc (image only).
    xp, xf = 6.45, 7.85
    arrow(ax, (5.55, yc), (xp - 0.03, yc))
    txt(ax, 5.98, yc + 0.22, "flatten", size=small - 0.5)
    bar(ax, xp, yc - 1.5, 0.35, 3.0)
    txt(ax, xp + 0.175, yc + 1.75, "9216", size=small)
    sym(ax, xp + 0.175, y_lab, "h", "pre")
    arrow(ax, (xp + 0.42, yc), (xf - 0.03, yc))
    bar(ax, xf, yc - 0.9, 0.35, 1.8)
    txt(ax, xf + 0.175, yc + 1.15, "256", size=small)
    sym(ax, xf + 0.175, y_lab, "h", "fc")

    # Body state joins only after h_fc, through one linear layer without a nonlinearity.
    xq = 5.7
    bar(ax, xq, ys - 0.35, 0.3, 0.7)
    txt(ax, xq - 0.15, ys, "body state $q$, 25", size=small, ha="right")
    arrow(ax, (xq + 0.37, ys), (xf - 0.03, ys))
    bar(ax, xf, ys - 0.5, 0.35, 1.0)
    txt(ax, xf + 0.175, ys + 0.75, "256", size=small)
    ax.text(xf + 0.85, ys - 0.35, r"$h_{\mathrm{state}}$", fontsize=FS.FS + 1, ha="center", va="center",
            color=INK, math_fontfamily="cm")

    # Heads: each a two-layer MLP (512 hidden ReLU units) on the 512-d concatenation [h_fc; h_state].
    xj, xb, xh = 9.0, 9.55, 10.85
    yp, yv, ym = yc, 0.6, 1.9
    for yy in (yc, ys):
        ax.plot([xf + 0.42, xj], [yy, ym], color=INK, lw=0.9)
    ax.plot([xj, xb], [ym, ym], color=INK, lw=0.9)
    ax.plot([xb, xb], [yv, yp], color=INK, lw=0.9)
    txt(ax, xb + 0.12, ym, "concat\n512", size=small - 0.5, color=FS.PAL["mute"], ha="left")
    for yy, name in ((yp, "Policy Head"), (yv, "Value Head")):
        arrow(ax, (xb, yy), (xh - 0.03, yy))
        bar(ax, xh, yy - 0.6, 0.35, 1.2)
        txt(ax, xh + 0.175, yy + 0.85, name, size=small)
        txt(ax, xh + 0.175, yy - 0.82, "512", size=small - 0.5, color=FS.PAL["mute"])
        arrow(ax, (xh + 0.42, yy), (xh + 1.0, yy))
    # Gaussian over continuous actions, with a state-independent sigma.
    gx = np.linspace(-1, 1, 80)
    ax.plot(xh + 1.1 + 0.6 * (gx + 1), yp - 0.5 + 1.0 * np.exp(-gx ** 2 / 0.1), color=INK, lw=0.9)
    txt(ax, xh + 1.7, yp - 0.85, r"$\mu(s),\ \sigma$", size=small)
    ax.add_patch(Rectangle((xh + 1.1, yv - 0.25), 0.5, 0.5, fc=FILL, ec=EDGE, lw=0.6))
    txt(ax, xh + 1.95, yv, "V", size=FS.FS)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
