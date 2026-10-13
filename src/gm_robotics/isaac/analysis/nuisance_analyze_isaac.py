"""Perception vs seeking: policy response to the tower swap vs size-matched grey decoy moves and a grey-recoloured tower swap."""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from gm_robotics.isaac.analysis import common_isaac as C
from gm_robotics.isaac.analysis import geom
from gm_robotics.isaac.analysis.pushpull_isaac import oracle_form

OUT = os.path.join(C.RESULTS, "nuisance")
# edit -> (reference scene, edited scene, matched re-render null of the reference, moved-from xy key, moved-to xy key)
PAIRS = {"tower_swap": ("base", "tower_swap", "rerender", "tower", "tower_moved"),
         "grey_tower_swap": ("grey_tower", "grey_tower_swap", "grey_tower_rerender", "tower", "tower_moved"),
         "decoy_move": ("decoy", "decoy_move", "decoy_rerender", "decoy", "decoy_moved"),
         "decoy_insert": ("base", "decoy", "rerender", None, None),
         "recolour_tower": ("base", "grey_tower", "rerender", None, None)}


@torch.no_grad()
def mus(agent, bank, key, dev, bs=256):
    out = []
    for sl in C.batched(len(bank["start"]), bs):
        out.append(C.taps(agent, C.to_t(bank["rgb_" + key][sl], dev), C.to_t(bank["state_base"][sl], dev))["mu"])
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kinds", nargs="+", default=["visible", "conflict"])
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = C.device()
    rows = []
    for kind in a.kinds:
        z = np.load(os.path.join(C.BANKS, f"nuisance_{kind}.npz"))
        bank = {k: z[k] for k in z.files}
        ok = bank["decoy_ok"].astype(bool)
        s, yaw = bank["start"], bank["yaw"]
        px = {}
        for e, (r0, r1, rn, _, _) in PAIRS.items():
            x0 = bank["rgb_" + r0].astype(np.float32) / 255
            px[e] = ((bank["rgb_" + r1].astype(np.float32) / 255 - x0) ** 2).sum((1, 2, 3))
            px[e + "_null"] = ((bank["rgb_" + rn].astype(np.float32) / 255 - x0) ** 2).sum((1, 2, 3))
        for cohort, seed, ck in C.agent_list():
            ag = C.build_agent(ck, init_seed=seed, dev=dev)
            sig = C.sigma(ag)
            cache = {}
            get = lambda k: cache.setdefault(k, mus(ag, bank, k, dev))
            for e, (r0, r1, rn, src, dst) in PAIRS.items():
                kl = C.gaussian_kl(get(r0), get(r1), sig).cpu().numpy()
                kln = C.gaussian_kl(get(r0), get(rn), sig).cpu().numpy()
                r = dict(kind=kind, cohort=cohort, seed=seed, edit=e, state=np.arange(len(s)), decoy_ok=ok, KL=kl, KL_null=kln,
                         px_d2=px[e], px_d2_null=px[e + "_null"])
                if src is not None:
                    rs, rd = geom.robot_frame(bank[src], s, yaw), geom.robot_frame(bank[dst], s, yaw)
                    b_to, b_from = np.arctan2(rd[:, 1], rd[:, 0]), np.arctan2(rs[:, 1], rs[:, 0])
                    o = oracle_form(get(r0).cpu().numpy().astype(np.float64), get(r1).cpu().numpy().astype(np.float64),
                                    sig.cpu().numpy().astype(np.float64), b_to, b_from)
                    r["S_to"], r["S_from"] = o["S_fav"], o["S_comp"]
                rows.append(pd.DataFrame(r))
            print(kind, cohort, seed, {e: round(float(rows[-len(PAIRS) + i].KL[ok].mean()), 3) for i, e in enumerate(PAIRS)},
                  flush=True)
    df = pd.concat(rows, ignore_index=True)
    df = df[df.decoy_ok]
    df["KL_excess"] = df.KL - df.KL_null
    df.to_csv(os.path.join(OUT, "nuisance_per_state.csv"), index=False)
    cols = ["KL", "KL_null", "KL_excess", "S_to", "S_from", "px_d2"]
    ps = df.groupby(["kind", "edit", "cohort", "seed"])[cols].mean().reset_index()
    ps.to_csv(os.path.join(OUT, "nuisance_per_seed.csv"), index=False)
    summ = []
    for key, sub in ps.groupby(["kind", "edit", "cohort"], sort=False):
        r = dict(zip(["kind", "edit", "cohort"], key), n_seeds=len(sub))
        for c in cols:
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(sub[c].to_numpy())
        summ.append(r)
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(OUT, "nuisance_summary.csv"), index=False)
    with open(os.path.join(OUT, "nuisance_summary.md"), "w") as fh:
        fh.write("# Size-matched nuisance edits vs the tower swap (start states with a valid decoy spot; per-seed mean over "
                 "states, mean [95% t-CI] across seeds)\n\nKL_excess = KL minus the matched re-render KL. S_to / S_from: oracle-"
                 "form change in log support of the command toward the moved object's new / old spot vs neutral commands. "
                 "px_d2: squared pixel change.\n\n")
        fh.write("| bank | edit | cohort | " + " | ".join(cols) + " |\n|---|---|---|" + "---|" * len(cols) + "\n")
        for _, r in sm.iterrows():
            fh.write(f"| {r.kind} | {r.edit} | {r.cohort} | " + " | ".join(
                C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi'], 2) if np.isfinite(r[c]) else "" for c in cols) + " |\n")
    print(open(os.path.join(OUT, "nuisance_summary.md")).read())


if __name__ == "__main__":
    main()
