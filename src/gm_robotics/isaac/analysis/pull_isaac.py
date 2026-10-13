"""WalkToFlag object pull (removal edits) and flag-vs-tower choice (probe banks) on the ghost-free Isaac banks, CPU/MPS only."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gm_robotics.paths import ROOT as RUN_ROOT

ROBOTICS = RUN_ROOT
FIXED = ROBOTICS / "results/isaac_analysis/ghost_check/fixed"
# common_isaac reads the bank root from this variable at import time, so it must be set first.
os.environ["ISAAC_ANALYSIS_RESULTS"] = str(FIXED)
from gm_robotics.isaac.analysis import common_isaac as C  # noqa: E402
from gm_robotics.isaac.analysis import geom  # noqa: E402
from gm_robotics.isaac.analysis.pushpull_isaac import oracle_cmd, oracle_form  # noqa: E402

OUT = ROBOTICS / "results/isaac_analysis/pull"


@torch.no_grad()
def mu(agent, rgb, state, dev, bs=256):
    out = []
    for sl in C.batched(len(rgb), bs):
        out.append(C.taps(agent, C.to_t(rgb[sl], dev), C.to_t(state[sl], dev))["mu"].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def loglik(a, m, sig):
    return -0.5 * (((a - m) / sig) ** 2).sum(-1)


def agents(seeds):
    out = []
    for cohort in ("gm", "control"):
        out += [(cohort, s, C.ckpt_path(cohort, s)) for s in seeds]
    return out + [("init", s, None) for s in seeds]


def removal_pull(bank_names, seeds, dev):
    """P_o = change in log support of the oracle command toward object o caused by o's presence (base vs o removed),
    minus the same change at neutral bearings. P_other_o = the same presence effect on the command toward the other object.
    The null uses a re-render of the base in place of the removal."""
    for cohort, seed, ck in agents(seeds):
        ag = C.build_agent(ck, init_seed=seed, dev=dev)
        sig = C.sigma(ag).cpu().numpy().astype(np.float64)
        rows = []
        for bname in bank_names:
            b = C.load_bank(f"edit_{bname}")
            s, yaw = b["start"], b["yaw"]
            rf, rt = geom.robot_frame(b["flag"], s, yaw), geom.robot_frame(b["tower"], s, yaw)
            bf, bt = np.arctan2(rf[:, 1], rf[:, 0]), np.arctan2(rt[:, 1], rt[:, 0])
            st = b["state_base"]
            m = {k: mu(ag, b[f"rgb_{k}"], st, dev) for k in ("base", "no_flag", "no_tower", "rerender")}
            pf = oracle_form(m["no_flag"], m["base"], sig, bf, bt)
            pt = oracle_form(m["no_tower"], m["base"], sig, bt, bf)
            nf = oracle_form(m["rerender"], m["base"], sig, bf, bt)
            nt = oracle_form(m["rerender"], m["base"], sig, bt, bf)
            rows.append(pd.DataFrame(dict(
                bank=bname, cohort=cohort, seed=seed, state=np.arange(len(s)),
                d_flag=np.linalg.norm(rf, axis=1), d_tower=np.linalg.norm(rt, axis=1), b_flag=bf, b_tower=bt,
                flag_px=b["flag_px_base"], tower_px=b["tower_px_base"],
                pull_flag=pf["S_fav"], push_flag_on_tower=pf["S_comp"],
                pull_tower=pt["S_fav"], push_tower_on_flag=pt["S_comp"],
                null_pull_flag=nf["S_fav"], null_pull_tower=nt["S_fav"])))
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(OUT / f"removal_pull_{cohort}_s{seed}.csv", index=False)
        print(f"removal {cohort} s{seed}: pull_flag {df.pull_flag.mean():+.2f} pull_tower {df.pull_tower.mean():+.2f}", flush=True)


def probe_choice(seeds, dev):
    """chi = log N(c_flag; mu, sigma) - log N(c_tower; mu, sigma) at native probe-bank start frames (> 0: flag preferred)."""
    banks = [C.load_bank(f"probe_{sp}") for sp in ("train", "test")]
    z = {k: np.concatenate([b[k] for b in banks]) for k in ("rgb", "state", "start", "yaw", "flag", "tower", "flag_px",
                                                             "tower_px", "flag_dist", "tower_dist", "flag_bearing",
                                                             "tower_bearing", "variant")}
    rf = geom.robot_frame(z["flag"], z["start"], z["yaw"])
    rt = geom.robot_frame(z["tower"], z["start"], z["yaw"])
    bf, bt = np.arctan2(rf[:, 1], rf[:, 0]), np.arctan2(rt[:, 1], rt[:, 0])
    cf, ct = oracle_cmd(bf), oracle_cmd(bt)
    for cohort, seed, ck in agents(seeds):
        ag = C.build_agent(ck, init_seed=seed, dev=dev)
        sig = C.sigma(ag).cpu().numpy().astype(np.float64)
        m = mu(ag, z["rgb"], z["state"], dev)
        chi = loglik(cf, m, sig) - loglik(ct, m, sig)
        df = pd.DataFrame(dict(cohort=cohort, seed=seed, idx=np.arange(len(chi)), variant=z["variant"],
                               d_flag=np.linalg.norm(rf, axis=1), d_tower=np.linalg.norm(rt, axis=1),
                               b_flag=bf, b_tower=bt, cam_b_flag=z["flag_bearing"], cam_b_tower=z["tower_bearing"],
                               flag_px=z["flag_px"], tower_px=z["tower_px"], chi=chi))
        df.to_csv(OUT / f"probe_choice_{cohort}_s{seed}.csv", index=False)
        print(f"choice {cohort} s{seed}: P(flag) {(chi > 0).mean():.3f}", flush=True)


def follow(delta, toward):
    """Signed length of the whitened shift `delta` along the whitened direction `toward`, and their cosine."""
    u = toward / np.linalg.norm(toward, axis=1, keepdims=True).clip(1e-12)
    f = (delta * u).sum(1)
    return f, f / np.linalg.norm(delta, axis=1).clip(1e-12)


def pull_vectors(bank_names, seeds, dev):
    """Per-state pull vector of each object in whitened command space: v_o = (mu_base - mu_without_o) / sigma.
    follow_o = its signed length toward the object's oracle command (c_o - mu_without_o); turn_o = its yaw-rate part
    signed toward the object's side. Nulls: the same projections of the box-move and re-render shifts."""
    for cohort, seed, ck in agents(seeds):
        ag = C.build_agent(ck, init_seed=seed, dev=dev)
        sig = C.sigma(ag).cpu().numpy().astype(np.float64)
        rows = []
        for bname in bank_names:
            b = C.load_bank(f"edit_{bname}")
            s, yaw = b["start"], b["yaw"]
            rf, rt = geom.robot_frame(b["flag"], s, yaw), geom.robot_frame(b["tower"], s, yaw)
            bf, bt = np.arctan2(rf[:, 1], rf[:, 0]), np.arctan2(rt[:, 1], rt[:, 0])
            cf, ct = oracle_cmd(bf), oracle_cmd(bt)
            m = {k: mu(ag, b[f"rgb_{k}"], b["state_base"], dev) for k in ("base", "no_flag", "no_tower", "box_move", "rerender")}
            r = dict(bank=bname, cohort=cohort, seed=seed, state=np.arange(len(s)), d_flag=np.linalg.norm(rf, axis=1),
                     d_tower=np.linalg.norm(rt, axis=1), b_flag=bf, b_tower=bt, flag_px=b["flag_px_base"])
            for key, ref in (("flag", "no_flag"), ("tower", "no_tower"), ("box", "box_move"), ("rr", "rerender")):
                dv = (m["base"] - m[ref]) / sig
                r[f"v_{key}_x"], r[f"v_{key}_y"], r[f"v_{key}_w"] = dv[:, 0], dv[:, 1], dv[:, 2]
                for oname, c, bo in (("flag", cf, bf), ("tower", ct, bt)):
                    f, cs = follow(dv, (c - m[ref]) / sig)
                    r[f"follow_{key}_to_{oname}"], r[f"cos_{key}_to_{oname}"] = f, cs
                    r[f"turn_{key}_to_{oname}"] = dv[:, 2] * np.sign(bo)
            rows.append(pd.DataFrame(r))
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(OUT / f"pull_vectors_{cohort}_s{seed}.csv", index=False)
        print(f"vectors {cohort} s{seed}: both>0 {((df.follow_flag_to_flag > 0) & (df.follow_tower_to_tower > 0)).mean():.3f}"
              f" box null {((df.follow_box_to_flag > 0) & (df.follow_box_to_tower > 0)).mean():.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--banks", nargs="+", default=["visible", "conflict", "far"])
    p.add_argument("--only", nargs="+", default=["removal", "vectors"], help="choice: probe-bank oracle choice (geometry-biased, not plotted)")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if "removal" in a.only:
        removal_pull(a.banks, a.seeds, dev)
    if "vectors" in a.only:
        pull_vectors(a.banks, a.seeds, dev)
    if "choice" in a.only:
        probe_choice(a.seeds, dev)


if __name__ == "__main__":
    main()
