"""Extracts per-episode arrival points (LocationReach gripper, PushToMarker cube end) and the per-seed reach rates from the raw lr_config rollouts."""
from __future__ import annotations

from paths import TABLES
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THR = 0.03
LR_CONDS = {"base": "rollouts/base", "shifty": "rollouts/shifty", "camy": "rollouts/camy",
            "imgum16": "rollouts_followup/imgum16"}
PTM_CONDS = ("base", "nomarker", "shifty_nomarker", "camy", "camy_nomarker", "imgu8_nomarker")


def project(E, K, p):
    pc = E[:3, :3] @ p + E[:3, 3]
    uv = K @ pc
    return uv[:2] / uv[2]


def backproject(E, K, uv, z):
    R, t = E[:3, :3], E[:3, 3]
    ray_c = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    o, d = -R.T @ t, R.T @ ray_c
    return o + d * (z - o[2]) / d[2]


def alt_point(z, target):
    """Where the non-table hypothesis puts the target: the old posture (base shift), the old pixel (camera or image shift)."""
    if "img_shift" in z.files and np.any(z["img_shift"]):
        E, K = z["extrinsic_cv"], z["intrinsic_cv"]
        return backproject(E, K, project(E, K, target) - z["img_shift"], target[2])
    cam = z["cam_offset"] if "cam_offset" in z.files else np.zeros(3)
    return target + (z["base"] - np.array([-0.615, 0, 0])) + cam


def lr(raw: Path):
    eps, rates = [], []
    for cond, stem in LR_CONDS.items():
        for coh in ("gm", "control"):
            for s in (1, 2, 3):
                z = np.load(raw / f"{stem}_{coh}_s{s}.npz")
                c, tcp = z["corner"], z["tcp"]
                alt = alt_point(z, c)
                dw, da = np.linalg.norm(tcp - c, axis=-1), np.linalg.norm(tcp - alt, axis=-1)
                hw, ha = (dw < THR).any(1), (da < THR).any(1)
                tw = np.where(hw, (dw < THR).argmax(1), 99)
                ta = np.where(ha, (da < THR).argmax(1), 99)
                # Arrival = closest approach to either candidate point (to the corner alone in the unedited scene).
                ti = (np.minimum(dw, da) if cond != "base" else dw).argmin(1)
                end = tcp[np.arange(len(tcp)), ti]
                for i in range(len(tcp)):
                    eps.append(dict(env="LR", cond=cond, cohort=coh, seed=s, episode=i, x=end[i, 0], y=end[i, 1],
                                    z=end[i, 2], t=int(ti[i]), corner_x=c[0], corner_y=c[1], alt_x=alt[0], alt_y=alt[1]))
                rates.append(dict(env="LR", cond=cond, cohort=coh, seed=s, reach_table=hw.mean(),
                                  reach_alt=ha.mean() if cond != "base" else np.nan,
                                  table_first=np.mean(hw & (tw <= ta)), alt_first=np.mean(ha & (ta < tw))))
    return eps, rates


def ptm(raw: Path):
    eps, rates = [], []
    for cond in PTM_CONDS:
        for coh in ("gm", "control"):
            for s in (0, 1, 2):
                z = np.load(raw / f"rollouts_followup/ptm_{cond}_{coh}_s{s}.npz")
                c = np.array([*z["centre"], 0.02])
                alt = alt_point(z, c)
                fin = z["cube"][:, -1]
                on = fin[:, 2] < 0.025
                dr, da = np.linalg.norm(fin[:, :2] - c[:2], axis=1), np.linalg.norm(fin[:, :2] - alt[:2], axis=1)
                for i in range(len(fin)):
                    eps.append(dict(env="PTM", cond=cond, cohort=coh, seed=s, episode=i, x=fin[i, 0], y=fin[i, 1],
                                    z=fin[i, 2], t=-1, corner_x=c[0], corner_y=c[1], alt_x=alt[0], alt_y=alt[1],
                                    start_x=z["cube"][i, 0, 0], start_y=z["cube"][i, 0, 1]))
                shifted = np.linalg.norm(alt[:2] - c[:2]) > 1e-6
                rates.append(dict(env="PTM", cond=cond, cohort=coh, seed=s, marker=bool(z["marker_on"]),
                                  reach_table=np.mean((dr < THR) & on),
                                  reach_alt=np.mean((da < THR) & on) if shifted else np.nan))
    return eps, rates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True,
                    help="local copy of the raw lr_config rollouts")
    a = ap.parse_args()
    e1, r1 = lr(a.raw)
    e2, r2 = ptm(a.raw)
    (TABLES / "data").mkdir(exist_ok=True)
    pd.DataFrame(e1 + e2).to_csv(TABLES / "place_endpoints.csv.gz", index=False, float_format="%.5f")
    r = pd.DataFrame(r1 + r2)
    r.to_csv(TABLES / "place_rates_per_seed.csv", index=False, float_format="%.4f")
    print(r.groupby(["env", "cond", "cohort"])[["reach_table", "reach_alt"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
