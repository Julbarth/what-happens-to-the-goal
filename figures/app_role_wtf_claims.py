"""WalkToFlag object-competition figure at full text width: the four claim panels in a short, wide 2 x 2 grid."""
from __future__ import annotations

from paths import OUT_DIR, TABLES
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from wtf_panels import by_stratum, joint, pulls, share_gap  # noqa: E402
from wtf_base import PAL, fs, load, titled  # noqa: E402

OUT = OUT_DIR / "app_role" / "fig_app_role_wtf_claims"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    fs.style()
    v, ps, r = load()
    w, pw, ph, gx, gy, left, top, bottom = fs.W_IN, 2.02, 1.08, 0.78, 0.62, 0.52, 0.22, 0.66
    h = top + 2 * ph + gy + bottom
    fig = plt.figure(figsize=(w, h))
    A = [fig.add_axes([(left + c * (pw + gx)) / w, (bottom + (1 - rr) * (ph + gy)) / h, pw / w, ph / h])
         for rr in range(2) for c in range(2)]
    vals = {"pulls": pulls(A[0], ps), "by_stratum": by_stratum(A[1], ps), "joint": joint(A[2], ps),
            "share_gap": share_gap(A[3], r)}
    A[0].set_ylabel(r"pull $f$ (policy s.d.)")
    A[1].set_ylabel("GM: fraction pulled")
    A[1].set_xticklabels(["near,\nflag closer", "near,\ntower closer", "flag much\ncloser", "tower much\ncloser"],
                         fontsize=fs.FS_TICK - 0.5)
    A[3].set_ylabel("tower share\n" + r"$P_p/(P_p+P_\star)$")
    for ax, L, t in zip(A, "abcd", ("Tower pulls strongly, flag weakly", "Flag pulls only when near",
                                    "Both pull at once", "Tower wins even when farther")):
        titled(ax, L, t)
    hd = [Line2D([], [], ls="", marker="o", ms=fs.MS + 1, color=PAL["ink"], label="flag ($f_\\star$)"),
          Line2D([], [], ls="", marker="o", ms=fs.MS + 1, mfc="white", mec=PAL["ink"], label="tower ($f_p$)"),
          Line2D([], [], color=fs.COH["GM"], lw=1.4, label="GM"), Line2D([], [], color=fs.COH["Non-GM"], lw=1.4, label="Non-GM"),
          Line2D([], [], color=fs.COH["init"], lw=1.4, label="init"), Patch(color=PAL["null"], label="nuisance null"),
          Line2D([], [], color=PAL["mute"], marker="o", ms=fs.MS, label=fs.CI_LABEL)]
    fs.legend_row(fig, hd, [x.get_label() for x in hd], y_in=0.03, handlelength=1.4, columnspacing=0.9)
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    (TABLES / "wtf_claims_wide_values.json").write_text(json.dumps(vals, indent=1, default=float))


if __name__ == "__main__":
    main()
