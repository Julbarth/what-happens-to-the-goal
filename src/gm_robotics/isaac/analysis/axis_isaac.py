"""Isaac quadruped flag-vs-tower competition geometry at h_fc (AttributeTable protocol): cue cosines, shared axis, per-cue pull."""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from gm_robotics.isaac.analysis import common_isaac as C
from gm_robotics.isaac.analysis import geom
from gm_robotics.analysis import axis_geometry as AG   # analysis/axis_geometry.py: cos_e, cos_f_same_state, cross_state, alpha_f, split_indices

AX = C.MC.gmg_axis


def conflict_sign(bank):
    """+1 when the flag (goal candidate u) is left of the tower in the robot frame, -1 otherwise."""
    bu, _ = geom.cam_bearing_dist(bank["flag"], bank["start"], bank["yaw"])
    bv, _ = geom.cam_bearing_dist(bank["tower"], bank["start"], bank["yaw"])
    return np.where(geom.wrap(bu - bv) > 0, 1, -1)


@torch.no_grad()
def agent_geometry(agent, bank, dev, bs=128):
    """d_star = flag-swap displacement (pulls toward v), d_prox = tower-swap displacement (pulls toward u);
    d_comp = d_prox - d_star points toward the goal candidate u, as in AttributeTable."""
    sig = C.sigma(agent)
    h = {}
    n = len(bank["start"])
    for key in ("base", "flag_swap", "tower_swap"):
        hs, hf = [], []
        for sl in C.batched(n, bs):
            t = C.taps(agent, C.to_t(bank["rgb_" + key][sl], dev), C.to_t(bank["state_base"][sl], dev))
            hf.append(t["h_fc"]), hs.append(t["h_state"])
        h[key], h[key + "_state"] = torch.cat(hf), torch.cat(hs)
    Js = []
    for sl in C.batched(n, bs):
        with torch.enable_grad():
            Js.append(C.mean_jacobian(agent, "h_fc", h["base"][sl], h["base_state"][sl]))
    J = torch.cat(Js)
    d_star, d_prox = h["flag_swap"] - h["base"], h["tower_swap"] - h["base"]
    return dict(J=J, Jw=J / sig[None, :, None], d_star=d_star, d_prox=d_prox, d_comp=d_prox - d_star, sig=sig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="visible")
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--n_check", type=int, default=56)
    p.add_argument("--n_perm", type=int, default=2000)
    a = p.parse_args()
    C.ensure_dirs()
    dev = C.device()
    bank = C.load_bank(f"edit_{a.bank}")
    sign = conflict_sign(bank)
    fit, check, ev = AG.split_indices(len(sign), a.n_eval, a.n_check)
    axis_dir = os.path.join(C.RESULTS, "axis")
    os.makedirs(axis_dir, exist_ok=True)
    rows, state_rows = [], []
    for cohort, seed, ck in C.agent_list(a.seeds_gm, a.seeds_control, a.init_seeds):
        agent = C.build_agent(ck, init_seed=seed, dev=dev)
        g = agent_geometry(agent, bank, dev)
        dnp = g["d_comp"].cpu().numpy().astype(np.float64)
        u = AX.fit_u_comp({+1: [dnp[i] for i in fit if sign[i] > 0], -1: [dnp[i] for i in fit if sign[i] < 0]})
        held = AX.held_out_alignment(u, [dnp[i] for i in check], [float(sign[i]) for i in check])
        held_ev = AX.held_out_alignment(u, [dnp[i] for i in ev], [float(sign[i]) for i in ev])
        null = AX.bucket_permutation_null([dnp[i] for i in fit], [int(sign[i]) for i in fit],
                                          [dnp[i] for i in check], [int(sign[i]) for i in check], n_perm=a.n_perm)
        ut = torch.as_tensor(u, dtype=torch.float32, device=dev)
        evt = torch.as_tensor(ev, device=dev)
        sg = torch.as_tensor(sign, dtype=torch.float32, device=dev)
        a_prox = (AG.alpha_f(g["Jw"], ut, g["d_prox"]) * sg)[evt]
        a_star = (AG.alpha_f(g["Jw"], ut, g["d_star"]) * sg)[evt]
        cw = AG.cos_e(g["d_star"], g["d_prox"])
        cfw = AG.cos_f_same_state(g["Jw"], g["d_star"], g["d_prox"])
        cs, mats = AG.cross_state(g, fit, sign)
        r = dict(cohort=cohort, seed=seed, cosE_within=cw[evt].median().item(), cosF_within=cfw[evt].median().item(),
                 **cs, heldout_align_check=held["mean"], heldout_frac_pos_check=held["frac_pos"],
                 heldout_align_eval=held_ev["mean"], heldout_frac_pos_eval=held_ev["frac_pos"],
                 perm_p=null["p_upper_bound"], perm_true=null["true_score"], perm_mean=null["perm_score_mean"],
                 pull_proxy=a_prox.abs().median().item(), pull_goal=a_star.abs().median().item(),
                 pull_ratio=(a_prox.abs().median() / a_star.abs().median().clamp_min(1e-12)).item(),
                 pull_proxy_signed=a_prox.median().item(), pull_goal_signed=a_star.median().item(),
                 dcomp_norm_eval=g["d_comp"][evt].norm(dim=-1).median().item())
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        for j, i in enumerate(ev):
            state_rows.append(dict(cohort=cohort, seed=seed, state=int(i), sign=int(sign[i]),
                                   cosE_within=cw[i].item(), cosF_within=cfw[i].item(),
                                   alpha_prox=a_prox[j].item(), alpha_goal=a_star[j].item(),
                                   dcomp_norm=g["d_comp"][i].norm().item()))
        np.savez(os.path.join(axis_dir, f"{a.bank}_{cohort}_s{seed}.npz"), u_comp=u.astype(np.float32), fit=fit,
                 check=check, eval=ev, **mats)
    df = pd.DataFrame(rows)
    tag = a.bank
    df.to_csv(os.path.join(C.RESULTS, f"axis_{tag}_per_seed.csv"), index=False)
    pd.DataFrame(state_rows).to_csv(os.path.join(C.RESULTS, f"axis_{tag}_per_state.csv"), index=False)
    cols = [c for c in df.columns if c not in ("cohort", "seed")]
    summ = []
    for cohort, sub in df.groupby("cohort"):
        r = dict(cohort=cohort, n=len(sub))
        for c in cols:
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(sub[c].to_numpy())
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(C.RESULTS, f"axis_{tag}_summary.csv"), index=False)
    with open(os.path.join(C.RESULTS, f"axis_{tag}_summary.md"), "w") as f:
        f.write(f"# Isaac quadruped flag/tower competition at h_fc, bank {tag} (mean and 95% t-CI across seeds; "
                f"per-seed medians over states; sign balance fit {np.mean(sign[fit] > 0):.2f})\n\n")
        f.write("| quantity | " + " | ".join(summ["cohort"]) + " |\n|---|" + "---|" * len(summ) + "\n")
        for c in cols:
            f.write(f"| {c} | " + " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi']) for _, r in summ.iterrows()) + " |\n")
    print(open(os.path.join(C.RESULTS, f"axis_{tag}_summary.md")).read())


if __name__ == "__main__":
    main()
