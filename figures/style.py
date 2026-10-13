"""Shared type, line and legend conventions for every main-text figure of the ICLR paper."""
from __future__ import annotations


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from palette import PAL  # noqa: E402

W_IN = 5.5
FS = 7.0            # axis labels, row labels, in-axes category labels
FS_TICK = 6.5
FS_LEG = 6.5
FS_TITLE = 7.5      # environment / panel title, regular weight
FS_SUB = 6.5        # sub-label under a title ("cheese presence", "shape swap")
FS_LETTER = 7.0     # (a), (b), ... bold
TITLE_WEIGHT = "normal"

LW_AX = 0.6
LW_DATA = 1.2       # data lines
LW_REF = 0.6        # reference lines (chance, identity, r = 1, constant Q)
LW_ERR = 0.8        # CI whiskers
MS = 3.4            # mean markers
MS_OPEN = 4.6       # open (hollow) ring drawn around a filled mean marker
SEED = dict(s=5, alpha=0.4, lw=0)     # per-seed dots
STATE = dict(s=1.8, alpha=0.15, lw=0)  # per-state dots
CI_ALPHA = 0.15     # CI tunnels
REF = dict(color=PAL["mute"], lw=LW_REF, ls=(0, (3, 2)))

TITLE_PAD = 3.0     # points above the axes top
LETTER_DX = -16.0   # points left of the axes left edge
COH = {"GM": PAL["gm"], "Non-GM": PAL["nongm"], "init": PAL["init"]}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG, "axes.titlesize": FS_TITLE,
        "axes.titleweight": TITLE_WEIGHT, "axes.linewidth": LW_AX, "axes.edgecolor": PAL["ink"],
        "axes.labelcolor": PAL["ink"], "axes.labelpad": 1.5, "text.color": PAL["ink"], "xtick.color": PAL["ink"],
        "ytick.color": PAL["ink"], "xtick.major.width": LW_AX, "ytick.major.width": LW_AX, "xtick.major.size": 2,
        "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": PAL["paper"], "axes.facecolor": PAL["paper"],
        "savefig.facecolor": PAL["paper"], "lines.linewidth": LW_DATA, "lines.markersize": MS,
        "errorbar.capsize": 0})


def finish(ax) -> None:
    """Re-apply tick styling that panel helpers may have overridden."""
    ax.tick_params(labelsize=FS_TICK, length=2, width=LW_AX, pad=1.5)


def _title_offsets(sub: bool) -> tuple[float, float]:
    """(sub-label offset, title offset) in points above the axes top."""
    return TITLE_PAD, TITLE_PAD + (FS_SUB * 1.3 if sub else 0.0)


def panel_title(ax, title: str | None, sub: str | None = None, span=None, sub_only: bool = False) -> float:
    """Title (and optional sub-label below it) centred over ax, or over ax..span when a title spans panels.

    Returns the title's offset in points so the panel letter can share its baseline."""
    y_sub, y_title = _title_offsets(sub is not None)
    x = 0.5
    if span is not None:
        a, b = ax.get_position(), span.get_position()
        x = ((a.x0 + b.x1) / 2 - a.x0) / a.width
    if sub is not None:
        ax.annotate(sub, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, y_sub), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_SUB, color=PAL["ink"])
    if title and not sub_only:
        ax.annotate(title, xy=(x, 1), xycoords="axes fraction", xytext=(0, y_title), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_TITLE, fontweight=TITLE_WEIGHT, color=PAL["ink"])
    return y_title


def panel_letter(ax, letter: str, sub: bool = False, dx: float = LETTER_DX) -> None:
    """Bold (a), (b), ... top-left outside the panel, on the title's baseline."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, _title_offsets(sub)[1]),
                textcoords="offset points", ha="left", va="bottom", fontsize=FS_LETTER, fontweight="bold",
                color=PAL["ink"])


def cohort_handle(name: str, line: bool = False):
    c = COH[name]
    if line:
        return Line2D([], [], color=c, lw=LW_DATA, marker="o", ms=MS, mec="white", mew=0.4, label=name)
    return Line2D([], [], ls="", color=c, marker="o", ms=MS + 0.6, mec="white", mew=0.4, label=name)


def init_handle(line_style=None):
    if line_style is not None:
        return Line2D([], [], color=PAL["init"], lw=LW_DATA, ls=line_style, label="init")
    return Line2D([], [], ls="", color=PAL["init"], marker="o", ms=MS + 0.6, mec="white", mew=0.4, label="init")


def ci_handle(tunnel: bool = False):
    """'mean ± 95% CI' glyph: a marker on a whisker, or on a shaded tunnel for line plots."""
    g = PAL["mute"]
    if tunnel:
        return (Patch(color=g, alpha=0.3, lw=0), Line2D([], [], color=g, lw=LW_DATA, marker="o", ms=MS,
                                                          mec="white", mew=0.4))
    return Line2D([], [], color=g, lw=LW_ERR, marker="o", ms=MS, mec="white", mew=0.4)


CI_LABEL = "mean ± 95% CI"


def legend_row(fig, handles: list, labels: list, y_in: float = 0.02, x: float = 0.5, **kw):
    """One compact row, centred below the panels, bottom edge y_in inches above the figure bottom."""
    opts = dict(loc="lower center", ncol=len(handles), frameon=False, fontsize=FS_LEG, handlelength=2.4,
                handleheight=0.7, handletextpad=0.4, columnspacing=1.2, borderaxespad=0, borderpad=0)
    opts.update(kw)
    return fig.legend(handles, labels, bbox_to_anchor=(x, y_in / fig.get_figheight()), **opts)
