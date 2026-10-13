"""Edit-bank pixel statistics and opposing-cue geometry: clean subsets, noise-decoupled cue cosines, output-space split."""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from gm_robotics.isaac.analysis import common_isaac as C
from gm_robotics.isaac.analysis import geom
from gm_robotics.analysis import axis_geometry as AG
from gm_robotics.isaac.analysis.pushpull_isaac import oracle_cmd

OUT = os.path.join(C.RESULTS, "geometry_check")


@torch.no_grad()
def site_acts(agent, bank, key, dev, bs=128):
    out = {k: [] for k in ("h_fc", "h_state", "mu")}
    for sl in C.batched(len(bank["start"]), bs):
        t = C.taps(agent, C.to_t(bank["rgb_" + key][sl], dev), C.to_t(bank["state_base"][sl], dev))
        for k in out:
            out[k].append(t[k])
    return {k: torch.cat(v) for k, v in out.items()}


def bank_audit(bank, name):
    s, yaw = bank["start"], bank["yaw"]
    bu, du = geom.cam_bearing_dist(bank["flag"], s, yaw)
    bv, dv = geom.cam_bearing_dist(bank["tower"], s, yaw)
    r = pd.DataFrame(dict(bank=name, state=np.arange(len(s)), flag_px=bank["flag_px_base"], tower_px=bank["tower_px_base"],
                          flag_px_after_flag_swap=bank["flag_px_flag_swap"], tower_px_after_tower_swap=bank["tower_px_tower_swap"],
                          flag_px_rerender=bank["flag_px_rerender"], tower_px_rerender=bank["tower_px_rerender"],
                          sep_m=np.linalg.norm(bank["flag"] - bank["tower"], axis=1), flag_dist=du, tower_dist=dv,
                          sep_deg=np.rad2deg(np.abs(geom.wrap(bu - bv)))))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--banks", nargs="+", default=["visible", "conflict"])
    p.add_argument("--thresholds", type=int, nargs="+", default=[5, 20, 50])
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = C.device()
    audits, rows = [], []
    for bname in a.banks:
        bank = C.load_bank(f"edit_{bname}")
        au = bank_audit(bank, bname)
        audits.append(au)
        s, yaw = bank["start"], bank["yaw"]
        ru, rv = geom.robot_frame(bank["flag"], s, yaw), geom.robot_frame(bank["tower"], s, yaw)
        bu, bv = np.arctan2(ru[:, 1], ru[:, 0]), np.arctan2(rv[:, 1], rv[:, 0])
        sign = np.where(geom.wrap(geom.cam_bearing_dist(bank["flag"], s, yaw)[0] -
                                  geom.cam_bearing_dist(bank["tower"], s, yaw)[0]) > 0, 1, -1)
        subsets = {"all": np.ones(len(s), bool)}
        for N in a.thresholds:
            subsets[f"clean{N}"] = ((au.flag_px >= N) & (au.tower_px >= N) & (au.flag_px_after_flag_swap >= N)
                                    & (au.tower_px_after_tower_swap >= N)).to_numpy()
        for cohort, seed, ck in C.agent_list():
            ag = C.build_agent(ck, init_seed=seed, dev=dev)
            sig = C.sigma(ag)
            h = {k: site_acts(ag, bank, k, dev) for k in ("base", "rerender", "flag_swap", "tower_swap")}
            Js = []
            for sl in C.batched(len(s), 128):
                with torch.enable_grad():
                    Js.append(C.mean_jacobian(ag, "h_fc", h["base"]["h_fc"][sl], h["base"]["h_state"][sl]))
            Jw = torch.cat(Js) / sig[None, :, None]
            d_star = h["flag_swap"]["h_fc"] - h["base"]["h_fc"]
            d_prox = h["tower_swap"]["h_fc"] - h["base"]["h_fc"]
            # Same edit against an independent base render: removes the base-render noise the two displacements share.
            d_prox_ind = h["tower_swap"]["h_fc"] - h["rerender"]["h_fc"]
            d_null = h["rerender"]["h_fc"] - h["base"]["h_fc"]
            ce, cf = AG.cos_e(d_star, d_prox), AG.cos_f_same_state(Jw, d_star, d_prox)
            ce_i, cf_i = AG.cos_e(d_star, d_prox_ind), AG.cos_f_same_state(Jw, d_star, d_prox_ind)
            # Output space: whitened mean shift of each edit, split along the u-vs-v discriminant (oracle command toward u
            # minus toward v, whitened); positive component = toward u.
            sg = sig.cpu().numpy()
            w = (oracle_cmd(bu) - oracle_cmd(bv)) / sg
            w = w / np.linalg.norm(w, axis=1, keepdims=True)
            out = {}
            for nm, key in (("star", "flag_swap"), ("prox", "tower_swap"), ("null", "rerender")):
                dm = ((h[key]["mu"] - h["base"]["mu"]).cpu().numpy()) / sg
                along = (dm * w).sum(1)
                out[f"{nm}_toward_u"] = along
                out[f"{nm}_frac_on_uv"] = along ** 2 / np.maximum((dm ** 2).sum(1), 1e-30)
                out[f"{nm}_fisher_norm2"] = (dm ** 2).sum(1)
            ax_path = os.path.join(C.RESULTS, "axis", f"visible_{cohort}_s{seed}.npz")
            ut = torch.as_tensor(np.load(ax_path)["u_comp"], device=dev, dtype=torch.float32)
            st = torch.as_tensor(sign, dtype=torch.float32, device=dev)
            a_prox = (AG.alpha_f(Jw, ut, d_prox) * st).cpu().numpy()
            a_star = (AG.alpha_f(Jw, ut, d_star) * st).cpu().numpy()
            df = pd.DataFrame(dict(bank=bname, cohort=cohort, seed=seed, state=np.arange(len(s)), sign=sign,
                                   cosE=ce.cpu().numpy(), cosF=cf.cpu().numpy(), cosE_indep=ce_i.cpu().numpy(),
                                   cosF_indep=cf_i.cpu().numpy(), alpha_prox=a_prox, alpha_goal=a_star,
                                   noise_ratio_star=(d_null.norm(dim=1) ** 2 / d_star.norm(dim=1).clamp_min(1e-12) ** 2).cpu().numpy(),
                                   noise_ratio_prox=(d_null.norm(dim=1) ** 2 / d_prox.norm(dim=1).clamp_min(1e-12) ** 2).cpu().numpy(),
                                   **out))
            for sub, m in subsets.items():
                d = df[m]
                rows.append(dict(bank=bname, subset=sub, cohort=cohort, seed=seed, n_states=int(m.sum()),
                                 cosE_within=d.cosE.median(), cosF_within=d.cosF.median(),
                                 cosE_within_indep=d.cosE_indep.median(), cosF_within_indep=d.cosF_indep.median(),
                                 frac_cosF_lt_m05=(d.cosF_indep < -0.5).mean(),
                                 pull_proxy=d.alpha_prox.abs().median(), pull_goal=d.alpha_goal.abs().median(),
                                 pull_ratio=d.alpha_prox.abs().median() / max(d.alpha_goal.abs().median(), 1e-12),
                                 star_toward_u=d.star_toward_u.mean(), prox_toward_u=d.prox_toward_u.mean(),
                                 star_frac_on_uv=d.star_frac_on_uv.median(), prox_frac_on_uv=d.prox_frac_on_uv.median(),
                                 null_toward_u=d.null_toward_u.mean(),
                                 noise_ratio_star=d.noise_ratio_star.median(), noise_ratio_prox=d.noise_ratio_prox.median()))
            print(bname, cohort, seed, {k: round(v, 3) for k, v in rows[-len(subsets)].items() if isinstance(v, float)}, flush=True)
            df.to_csv(os.path.join(OUT, f"per_state_{bname}_{cohort}_s{seed}.csv"), index=False)
    au = pd.concat(audits)
    au.to_csv(os.path.join(OUT, "bank_audit_per_state.csv"), index=False)
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(OUT, "geometry_per_seed.csv"), index=False)
    cols = [c for c in ps.columns if c not in ("bank", "subset", "cohort", "seed", "n_states")]
    summ = []
    for key, sub in ps.groupby(["bank", "subset", "cohort"], sort=False):
        r = dict(zip(["bank", "subset", "cohort"], key), n_states=int(sub.n_states.iloc[0]), n_seeds=len(sub))
        for c in cols:
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(sub[c].to_numpy())
        summ.append(r)
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(OUT, "geometry_summary.csv"), index=False)
    with open(os.path.join(OUT, "geometry_summary.md"), "w") as fh:
        fh.write("# Bank audit\n\nPixel counts at the start state (colour threshold; rerender = same scene rendered again).\n\n")
        fh.write("| bank | quantity | min | 5% | 25% | median | 75% | frac<5 | frac<20 | frac<50 |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for b, sub in au.groupby("bank", sort=False):
            for c in ("flag_px", "tower_px", "flag_px_after_flag_swap", "tower_px_after_tower_swap", "flag_px_rerender",
                      "sep_m", "sep_deg", "flag_dist", "tower_dist"):
                v = sub[c].to_numpy()
                q = np.quantile(v, [0, 0.05, 0.25, 0.5, 0.75])
                fr = [f"{np.mean(v < t):.3f}" if "px" in c else "" for t in (5, 20, 50)]
                fh.write(f"| {b} | {c} | " + " | ".join(f"{x:.2f}" for x in q) + " | " + " | ".join(fr) + " |\n")
        fh.write("\nRerender pixel-count agreement (visible): flag |diff| median "
                 f"{np.median(np.abs(au[au.bank == 'visible'].flag_px - au[au.bank == 'visible'].flag_px_rerender)):.1f} px\n\n")
        fh.write("# Opposing-cue geometry at h_fc (per-seed medians over states, mean [95% t-CI] across seeds)\n\n"
                 "cos*_indep: tower-swap displacement taken against an independent re-render of the base, so the two displacements "
                 "share no base-render noise. *_toward_u: whitened output shift along the u-minus-v oracle-command discriminant "
                 "(positive = toward the flag's location u); *_frac_on_uv: share of the edit's Fisher norm on that discriminant. "
                 "noise_ratio: |d_rerender|^2 / |d_edit|^2.\n\n")
        show = ["cosE_within", "cosF_within", "cosE_within_indep", "cosF_within_indep", "frac_cosF_lt_m05", "pull_ratio",
                "star_toward_u", "prox_toward_u", "star_frac_on_uv", "prox_frac_on_uv", "noise_ratio_star", "noise_ratio_prox"]
        fh.write("| bank | subset | cohort | n | " + " | ".join(show) + " |\n|---|---|---|---|" + "---|" * len(show) + "\n")
        for _, r in sm.iterrows():
            fh.write(f"| {r.bank} | {r.subset} | {r.cohort} | {r.n_states} | " +
                     " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi'], 2) for c in show) + " |\n")
    print(open(os.path.join(OUT, "geometry_summary.md")).read())


if __name__ == "__main__":
    main()
