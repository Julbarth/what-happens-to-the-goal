"""PushToMarker role table: every PushToMarker number of the role chapter as per-seed values, mean and SD across seeds."""
from paths import ROBOTICS_ROOT
import argparse
import json
import os

import numpy as np
import pandas as pd


def stat(v):
    v = np.asarray(v, float)
    return {"per_seed": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gr", default=str(ROBOTICS_ROOT))
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptm_role_table_values.json"))
    a = p.parse_args()
    an = os.path.join(a.gr, "results/analysis")
    R = {}
    coh = {"gm": "GM", "control": "Non-GM", "init": "init"}

    cs = pd.read_csv(f"{an}/ptm_geometry/ctrlset_per_seed.csv")
    cs = cs[cs.n == 399]  # separable stratum: marker >5.5 cm from centre, >2.5 cm off the cube-centre segment, cube in GM annulus
    for c in ("gm", "control"):
        s = cs[(cs.cohort == c) & (cs.cond == "standard")].sort_values("seed")
        h = cs[(cs.cohort == c) & (cs.cond == "nomarker")].sort_values("seed")
        R[f"sep_centre/{coh[c]}"] = stat(s.proxy)
        R[f"sep_marker/{coh[c]}"] = stat(s.goal)
        R[f"sep_pull_cm/{coh[c]}"] = stat(h.min_dm_cm.values - s.min_dm_cm.values)
    cf = pd.read_csv(f"{an}/ptm_geometry/conflict_per_seed.csv")
    for c in ("gm", "control"):
        g = lambda cond: cf[(cf.cohort == c) & (cf.cond == cond)].sort_values("seed")
        R[f"decoupled_reach/{coh[c]}"] = stat(g("decoupled").reach_m2)
        R[f"pause_drawn/{coh[c]}"] = stat(g("standard").paused)
        R[f"pause_hidden/{coh[c]}"] = stat(g("nomarker").paused)

    sf = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../fig_worldedit/robot_step_follow_per_seed.csv"))
    for c in ("gm", "control", "init"):
        s = sf[(sf.task == "PTM") & (sf.cohort == c)].sort_values("seed")
        R[f"move_KL/{coh[c]}"] = stat(s.KL)
        R[f"move_SF_mm/{coh[c]}"] = stat(s.step_follow_mm)
        R[f"move_SF_share/{coh[c]}"] = stat(s.step_follow_mm / s.base_step_mm)

    q2 = pd.read_csv(f"{an}/deepdive_depth/q2_ptm_presence_per_state.csv")
    for bank, key in (("PTM_worldedit", "start"), ("PTM_hover", "hover")):
        for c in ("gm", "control", "init"):
            s = q2[(q2.bank == bank) & (q2.edit == "marker_move") & (q2.cohort == c)]
            R[f"move_cos_{key}/{coh[c]}"] = stat(s.groupby("seed").cos_to_push.mean().sort_index())

    ml = pd.concat([pd.read_csv(f"{an}/ptm_mlat/raw/ptm_mlat_{c}_s{s}.csv") for c in ("gm", "control", "init") for s in (0, 1, 2)])
    for c in ("gm", "control", "init"):
        for col in ("white", "green"):
            w = ml[(ml.bank == "PTM_worldedit") & (ml["loc"] == "m") & (ml.colour == col) & (ml.cohort == c)]
            R[f"pres_m_{col}_mm/{coh[c]}"] = stat(w.groupby("seed").dtcp_clip_mm.mean().sort_index())
            h = ml[(ml.bank == "PTM_hover") & (ml["loc"] == "m") & (ml.colour == col) & (ml.cohort == c)]
            R[f"pres_m_{col}_coshover/{coh[c]}"] = stat(h.groupby("seed").cos_to_push.mean().sort_index())
        st = ml[(ml.bank == "PTM_worldedit") & (ml["loc"] == "m") & (ml.colour == "white") & (ml.cohort == c)]
        R[f"step0_mm_median/{coh[c]}"] = stat(st.groupby("seed").step0_mm.median().sort_index())

    sw = pd.concat([pd.read_csv(f"{an}/ptm_sweep/raw/ptm_sweep_{c}_s{s}.csv") for c in ("gm", "control", "init") for s in (0, 1, 2)])
    img = pd.read_csv(f"{an}/ptm_sweep/raw/ptm_sweep_image_geometry.csv")
    # same units as the sweep loader: disc and cube visible (>= 20 px) at all five distances
    vis = img.assign(v=(img.npx_white >= 20) & (img.cube_px >= 20)).groupby(["bank", "side", "state"]).v.all().rename("vis5").reset_index()
    sw = sw.merge(vis, on=["bank", "side", "state"])
    sw = sw[sw.vis5]
    for c in ("gm", "control", "init"):
        for col in ("white", "green"):
            for d in (3.0, 11.0):
                w = sw[(sw.bank == "PTM_worldedit") & (sw.colour == col) & (sw.d_cm == d) & (sw.cohort == c)]
                R[f"sweep_{col}_{d:g}cm_mm/{coh[c]}"] = stat(w.groupby("seed").dtcp_clip_mm.mean().sort_index())
            for d in (5.2,):
                h = sw[(sw.bank == "PTM_hover") & (sw.colour == col) & (sw.d_cm == d) & (sw.cohort == c)]
                R[f"sweep_{col}_{d:g}cm_coshover/{coh[c]}"] = stat(h.groupby("seed").cos_to_push.mean().sort_index())
    json.dump(R, open(a.out, "w"), indent=1)
    for k, v in R.items():
        print(f"{k:40s} {v['mean']:+.3f} ± {v['sd']:.3f}  {np.round(v['per_seed'], 3)}")


if __name__ == "__main__":
    main()
