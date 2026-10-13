"""Shared-axis h_fc steering with u_comp fit on the strict / decision banks (raw or pixel-matched), on steer_ext's eval states."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import axis_geometry as AG
from gm_robotics.analysis import bank_audit as BA

OUT = os.path.join(C.RESULTS, "steer_global")


def fit_axes(a):
    """u_comp per agent from every analysed state of the rebuilt bank (all of them are disjoint from the steering
    eval states, which are the last 200 of the original world-edit bank)."""
    dev = torch.device("cuda")
    b = C.load_bank(f"{a.env}_audit_{a.bank}")
    label, idx, _, _ = next(iter(BA.bank_views(a.env, a.bank, b)))
    view = BA.make_view(a.env, b, idx, a.matched)
    sign = view["sign"].astype(int)
    _, _, ev = AG.split_indices(512, 200, 56)
    tag = a.bank + ("_pxmatched" if a.matched else "")
    d = os.path.join(OUT, a.env, tag, "axis")
    os.makedirs(d, exist_ok=True)
    for cohort, seeds in BA.SEEDS[a.env].items():
        for seed in seeds:
            agent = C.build_agent(C.ckpt_path(a.env, cohort, seed), dev=dev)
            g = AG.agent_geometry(agent, view, dev, a.env)
            dnp = g["d_comp"].cpu().numpy().astype(np.float64)
            u = C.gmg_axis.fit_u_comp({+1: [x for x, s in zip(dnp, sign) if s > 0], -1: [x for x, s in zip(dnp, sign) if s < 0]})
            ref = np.load(os.path.join(C.RESULTS, "axis", f"{a.env}_{cohort}_s{seed}.npz"))["u_comp"]
            print(f"{a.env} {tag} {cohort} s{seed}: n {len(sign)} cos(u_new, u_orig) {np.dot(u, ref) / np.linalg.norm(u) / np.linalg.norm(ref):.3f}", flush=True)
            np.savez(os.path.join(d, f"{a.env}_{cohort}_s{seed}.npz"), u_comp=u.astype(np.float32), eval=ev)


def steer(a):
    """steer_ext.steer_attr with only the fixed / shared-axis arms; RESULTS is redirected so steer_ext reads the
    new axis and writes its per-state CSV into the variant's folder (bank pixels are the original world-edit bank)."""
    from gm_robotics.analysis import steer_ext as S
    tag = a.bank + ("_pxmatched" if a.matched else "")
    root = os.path.join(OUT, a.env, tag)
    os.makedirs(os.path.join(root, "banks"), exist_ok=True)
    link = os.path.join(root, "banks", f"{a.env}_worldedit.npz")
    if not os.path.exists(link):
        os.symlink(os.path.join(C.BANKS, f"{a.env}_worldedit.npz"), link)
    C.RESULTS, C.BANKS = root, os.path.join(root, "banks")
    from gm_robotics.analysis import rollouts
    arms = [("baseline", 0.0, 0)]
    for b in a.doses:
        for name in a.arms:
            arms += [(name, b, 0), (name, -b, 0)]
    rollouts_arm_list = rollouts.arm_list
    S.arm_list = lambda *_: arms
    agents = [("gm", s, C.ckpt_path(a.env, "gm", s)) for s in BA.SEEDS[a.env]["gm"]]
    agents += [("control", s, C.ckpt_path(a.env, "control", s)) for s in BA.SEEDS[a.env]["control"]]
    S.steer_attr(a.env, agents, a.batch, os.path.join(root, f"steer_{a.env}_per_state.csv"), torch.device("cuda"),
                 [], a.doses, 0)
    S.arm_list = rollouts_arm_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["fit", "steer"])
    p.add_argument("--env", choices=["AT", "SZ"], default="AT")
    p.add_argument("--bank", choices=["strict", "decision"], default="decision")
    p.add_argument("--matched", action="store_true")
    p.add_argument("--arms", nargs="+", default=["shared", "fisher_proj", "unsigned"])
    p.add_argument("--doses", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--batch", type=int, default=100)
    a = p.parse_args()
    dict(fit=fit_axes, steer=steer)[a.step](a)


if __name__ == "__main__":
    main()
