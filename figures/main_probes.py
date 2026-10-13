"""Main-text probe figure: probes_styled.py panels with Maze and Coin bearing replaced by agent-position-decorrelated probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import probes_panels as v2  # noqa: E402
import probes_styled as v4  # noqa: E402
from app_procgen_probes import band, load_decorrelated  # noqa: E402

from paths import OUT_DIR, TABLES

SITES = ["h1", "h2", "h3", "h_pre", "h_fc"]


def decorrelated_series(cells: dict) -> dict:
    return {c: [band(cells[c][k]) for k in SITES] for c in ("GM", "Non-GM", "init")}


def frozen_series(panel: dict) -> dict:
    """Panel series exactly as plotted by probes_styled.py ([mean, 95% t half-width] per site)."""
    return {c: [tuple(v) for v in panel[c]] for c in ("GM", "Non-GM", "init")}


def build_panels(a) -> list[dict]:
    dc = load_decorrelated(a.decorrelated)
    rob = v2.load_robot(a.robot, a.robot_probe)
    rob_s = {c: [band(rob["cohorts"][c][k]) for k in v2.ROBOT_SITES] for c in ("GM", "Non-GM", "init")}
    # The Attribute Maze panel is drawn from the values probes_styled.py saved, not from its source data
    # (data/probes/goal_identity_loci_v2_rseed_20260925_seeded).
    am = json.loads(a.styled_values.read_text())["panels"]["Attribute Maze"]
    return [
        dict(env="Maze", sub="cheese bearing, decorrelated", metric="AUC", x=v2.PROCGEN_X, ticks=v2.PROCGEN_TICKS,
             s=decorrelated_series(dc[("maze", "balanced")]), ctrl=dc[("ctrl", "maze", "balanced")]),
        dict(env="PushToMarker", sub="marker position", metric="R²", x=v2.ROBOT_X, ticks=v2.ROBOT_TICKS,
             s=rob_s, ctrl=rob["ctrl"]),
        dict(env="Coin", sub="coin bearing, decorrelated", metric="R²", x=v2.PROCGEN_X, ticks=v2.PROCGEN_TICKS,
             s=decorrelated_series(dc[("coin", "progress_agent_xy")]), ctrl=dc[("ctrl", "coin", "progress_agent_xy")],
             hollow={4}),
        dict(env="Attribute Maze", sub="line as goal", metric="AUC", x=v2.PROCGEN_X, ticks=v2.PROCGEN_TICKS,
             s=frozen_series(am), ctrl=None),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decorrelated", type=Path, default=TABLES / "procgen_bearing_decorrelated/summary.csv")
    ap.add_argument("--robot", type=Path, default=v2.ROBOT_CSV, help="robotics code probe_audit PTM_per_seed.csv")
    ap.add_argument("--robot-probe", default="ridge_sel")
    ap.add_argument("--styled-values", type=Path, default=TABLES / "probes_styled_values.json")
    ap.add_argument("--font-size", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=1.58)
    ap.add_argument("--top-margin", type=float, default=0.275)
    ap.add_argument("--bottom-margin", type=float, default=0.345)
    ap.add_argument("--legend-y", type=float, default=0.01)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-name", default="fig_main_probes")
    ap.add_argument("--values-out", type=Path, default=TABLES / "main_probes_values.json")
    args = ap.parse_args()
    v2.fs.style()
    v4.set_font_size(args.font_size)
    panels = build_panels(args)
    info = v4.render(panels, args.out_dir / args.out_name, args.height, (-0.25, 4.25), args.top_margin,
                     args.bottom_margin, args.legend_y)
    rec = {P["env"]: {"probe": P["sub"], "metric": P["metric"], "sites": P["ticks"], "control": P["ctrl"],
                      **info[P["env"]],
                      **{c: [[round(m, 4), round(h, 4)] for m, h in P["s"][c]] for c in ("GM", "Non-GM", "init")}}
           for P in panels}
    args.values_out.write_text(json.dumps({"note": "per site [mean, 95% t half-width]",
                                           "sources": {"Maze": f"{args.decorrelated} env=maze subset=balanced",
                                                       "Coin": f"{args.decorrelated} env=coin subset=progress_agent_xy",
                                                       "PushToMarker": str(args.robot),
                                                       "Attribute Maze": f"{args.styled_values} (unchanged from v4)"},
                                           "panels": rec}, indent=1))
    print("wrote", args.values_out)


if __name__ == "__main__":
    main()
