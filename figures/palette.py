#!/usr/bin/env python3
"""Figure palette: hue encodes the agent (burgundy GM, teal Non-GM), weight the role (intense goal, soft hatched proxy)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from paths import OUT_DIR

OUT = OUT_DIR / "palette"

PAL = {
    "gm_goal": "#7A1520",
    "gm_proxy": "#C45C58",
    "nongm_goal": "#3D8B8B",
    "nongm_proxy": "#5FA8A0",
    "init": "#8E8E8E",
    "null": "#5A5A5A",
    "ink": "#2A2A2A",
    "mute": "#6E6E6E",
    "paper": "#FFFFFF",  # figure / axes background — always white
}

PAL["gm"] = PAL["gm_goal"]
PAL["nongm"] = PAL["nongm_goal"]
PAL["gm_true"] = PAL["gm_goal"]
PAL["nongm_true"] = PAL["nongm_goal"]

LABELS = {
    "gm_proxy": "GM · proxy",
    "gm_goal": "GM · goal",
    "nongm_proxy": "Non-GM · proxy",
    "nongm_goal": "Non-GM · goal",
}
PROXY_KEYS = {"gm_proxy", "nongm_proxy"}


def _proxy_label_color(key: str) -> str:
    # soft-red → dark burgundy (contrast); soft-teal → ink
    if key == "gm_proxy":
        return PAL["gm_goal"]
    return PAL["ink"]


def _sparse_hatch(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    *,
    spacing: float = 0.42,
    lw: float = 0.55,
    alpha: float = 0.55,
) -> None:
    """Manual diagonals — ~3–4 stripes on a short bar, not a grey smear."""
    clip = Rectangle((x, y), w, h, transform=ax.transData)
    x0, x1 = x - h, x + w + h
    t = x0
    while t < x1:
        xs = [t, t + h]
        ys = [y, y + h]
        (line,) = ax.plot(
            xs,
            ys,
            color=color,
            lw=lw,
            alpha=alpha,
            solid_capstyle="butt",
            zorder=2,
        )
        line.set_clip_path(clip)
        t += spacing


def _swatch(ax, x, y, w, h, key: str) -> None:
    c = PAL[key]
    is_proxy = key in PROXY_KEYS
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.05",
            facecolor=c,
            edgecolor="none",
            zorder=1,
        )
    )
    if is_proxy:
        _sparse_hatch(
            ax,
            x + 0.06,
            y + 0.06,
            w - 0.12,
            h - 0.12,
            _proxy_label_color(key),
            spacing=0.55,
            lw=0.6,
            alpha=0.45,
        )
        tc = _proxy_label_color(key)
    else:
        tc = "#FFFFFF"
    ax.text(
        x + w / 2,
        y + h * 0.58,
        LABELS[key],
        ha="center",
        va="center",
        color=tc,
        fontsize=12,
        fontweight="700",
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        c.upper(),
        ha="center",
        va="center",
        color=tc,
        fontsize=9,
        fontfamily="monospace",
        alpha=0.92,
        zorder=3,
    )


def make_strip() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.0, 10.0), facecolor=PAL["paper"])
    ax = fig.add_axes([0.05, 0.04, 0.90, 0.92])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_facecolor(PAL["paper"])

    ax.text(
        0.3,
        15.35,
        "Four streams  ·  hue = agent,  weight = role",
        fontsize=15,
        fontweight="700",
        color=PAL["ink"],
    )
    ax.text(
        0.3,
        14.8,
        "burgundy = GM   ·   teal = Non-GM   ·   intense = goal   ·   soft + sparse hatch = proxy",
        fontsize=10,
        color=PAL["mute"],
    )

    ax.text(3.35, 14.0, "goal", ha="center", fontsize=11, color=PAL["mute"])
    ax.text(8.15, 14.0, "proxy", ha="center", fontsize=11, color=PAL["mute"])
    ax.text(0.3, 12.25, "GM", ha="left", va="center", fontsize=11, color=PAL["mute"])
    ax.text(0.3, 9.05, "Non-GM", ha="left", va="center", fontsize=11, color=PAL["mute"])

    sw_w, sw_h, gap_x, gap_y = 4.0, 2.7, 0.8, 0.55
    x_goal, x_proxy = 1.35, 1.35 + sw_w + gap_x
    y_gm = 10.4
    y_nongm = y_gm - sw_h - gap_y  # 7.15

    _swatch(ax, x_goal, y_gm, sw_w, sw_h, "gm_goal")
    _swatch(ax, x_proxy, y_gm, sw_w, sw_h, "gm_proxy")
    _swatch(ax, x_goal, y_nongm, sw_w, sw_h, "nongm_goal")
    _swatch(ax, x_proxy, y_nongm, sw_w, sw_h, "nongm_proxy")

    # mock bars — entirely below swatches
    ax.text(
        0.3,
        y_nongm - 0.55,
        "mock bars  (goal solid · proxy sparse hatch)",
        fontsize=10,
        color=PAL["mute"],
    )
    mock = [
        ("gm_goal", 0.85),
        ("gm_proxy", 0.40),
        ("nongm_goal", 0.70),
        ("nongm_proxy", 0.32),
    ]
    max_bar = 1.7
    y0 = 3.9  # bar baseline; tops ≤ 3.9+1.45 ≈ 5.35 ≪ swatch bottom 7.15
    x0, bw, gap = 1.35, 1.9, 0.45
    for i, (key, ht) in enumerate(mock):
        x = x0 + i * (bw + gap)
        bar_h = max_bar * ht
        ax.add_patch(
            Rectangle((x, y0), bw, bar_h, facecolor=PAL[key], edgecolor="none", zorder=1)
        )
        if key in PROXY_KEYS:
            _sparse_hatch(
                ax,
                x,
                y0,
                bw,
                bar_h,
                _proxy_label_color(key),
                spacing=0.45,
                lw=0.5,
                alpha=0.5,
            )
        ax.text(
            x + bw / 2,
            y0 - 0.4,
            LABELS[key].replace(" · ", "\n"),
            ha="center",
            va="top",
            fontsize=8,
            color=PAL["mute"],
            linespacing=1.15,
        )

    # agent-only below bar captions (captions end ~ y0-0.4-0.7 ≈ 2.8)
    ax.text(
        0.3,
        2.55,
        "agent-only (no proxy/goal): use intense parents",
        fontsize=10,
        color=PAL["mute"],
    )
    ax.add_patch(Rectangle((1.35, 1.55), 3.0, 0.65, facecolor=PAL["gm"], edgecolor="none"))
    ax.add_patch(Rectangle((4.7, 1.55), 3.0, 0.65, facecolor=PAL["nongm"], edgecolor="none"))
    ax.text(
        2.85,
        1.875,
        f"GM  {PAL['gm']}",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=9,
        fontweight="700",
    )
    ax.text(
        6.2,
        1.875,
        f"Non-GM  {PAL['nongm']}",
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=9,
        fontweight="700",
    )

    ax.add_patch(Rectangle((9.6, 1.95), 0.5, 0.5, facecolor=PAL["init"], edgecolor="none"))
    ax.add_patch(Rectangle((9.6, 1.25), 0.5, 0.5, facecolor=PAL["null"], edgecolor="none"))
    ax.text(
        10.3,
        2.2,
        f"init  {PAL['init']}",
        va="center",
        fontsize=8,
        color=PAL["mute"],
        fontfamily="monospace",
    )
    ax.text(
        10.3,
        1.5,
        f"null  {PAL['null']}",
        va="center",
        fontsize=8,
        color=PAL["mute"],
        fontfamily="monospace",
    )

    ax.text(
        0.3,
        0.55,
        "symbols: same hues; goal = filled marker, proxy = open marker (hatch is for fills only)",
        fontsize=9,
        color=PAL["mute"],
    )

    path = OUT / "palette_b_yellow_goals.png"
    path2 = OUT / "palette_four_streams.png"
    fig.savefig(path, dpi=180, facecolor=PAL["paper"])
    fig.savefig(path2, dpi=180, facecolor=PAL["paper"])
    plt.close(fig)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    make_strip()
