"""AttributeTable / SortZones cue-competition geometry at h_fc: cue cosines, cross-state d_comp sharing, axis u_comp, pulls."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C

AX = C.gmg_axis
# (goal-cue swap, proxy-cue swap) per env; d_comp = h(proxy swap) - h(goal swap) points toward the goal candidate u.
CUE_EDITS = {"AT": ("shape_swap", "colour_swap"), "SZ": ("mark_swap", "colour_swap")}
ENV_NAME = {"AT": "AttributeTable", "SZ": "SortZones"}


def split_indices(n, n_eval, n_check):
    """Eval states (steering) are the last n_eval of the bank; the axis is fit on the rest minus a held-out check."""
    idx = np.arange(n)
    ev = idx[n - n_eval:]
    fit_all = idx[: n - n_eval]
    return fit_all[: len(fit_all) - n_check], fit_all[len(fit_all) - n_check:], ev


@torch.no_grad()
def agent_geometry(agent, bank, dev, env="AT", bs=128):
    sig = C.sigma(agent)
    goal_e, prox_e = CUE_EDITS[env]
    h = {}
    for key in ("base", goal_e, prox_e):
        hs, hf = [], []
        for sl in C.batched(len(bank["sign"]), bs):
            t = C.taps(agent, C.to_t(bank["rgb_" + key][sl], dev), C.to_t(bank["state_" + key][sl], dev))
            hf.append(t["h_fc"]), hs.append(t["h_state"])
        h[key], h[key + "_state"] = torch.cat(hf), torch.cat(hs)
    Js = []
    for sl in C.batched(len(bank["sign"]), bs):
        with torch.enable_grad():
            Js.append(C.mean_jacobian(agent, "h_fc", h["base"][sl], h["base_state"][sl]))
    J = torch.cat(Js)
    # W^{1/2} J, so that Fisher inner products are plain dot products of these rows.
    Jw = J / sig[None, :, None]
    d_star = h[goal_e] - h["base"]
    d_prox = h[prox_e] - h["base"]
    return dict(J=J, Jw=Jw, d_star=d_star, d_prox=d_prox, d_comp=d_prox - d_star, sig=sig)


def cos_e(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)


def cos_f_same_state(Jw, a, b):
    ja, jb = torch.einsum("bad,bd->ba", Jw, a), torch.einsum("bad,bd->ba", Jw, b)
    return (ja * jb).sum(-1) / (ja.norm(dim=-1) * jb.norm(dim=-1)).clamp_min(1e-12)


def cross_state(g, idx, sign):
    """Pairwise cos_E and cos_F (Fisher of either state, averaged) of d_comp across states, split by conflict sign."""
    d = g["d_comp"][idx]
    Jw = g["Jw"][idx]
    ce = cos_e(d[:, None], d[None, :])
    proj = torch.einsum("sad,td->sta", Jw, d)            # W^1/2 J_s d_t
    nrm = proj.norm(dim=-1)                               # sqrt(d_t^T F_s d_t)
    m = len(idx)
    ii = torch.arange(m)
    num = torch.einsum("ta,tua->tu", proj[ii, ii], proj)  # d_t^T F_t d_u
    cf_t = num / (nrm[ii, ii][:, None] * nrm).clamp_min(1e-12)
    cf = 0.5 * (cf_t + cf_t.T)
    same = torch.as_tensor(sign[idx][:, None] == sign[idx][None, :])
    off = ~torch.eye(m, dtype=torch.bool)
    order_np = np.argsort(-sign[idx], kind="stable")
    order = torch.as_tensor(order_np, device=ce.device)
    mats = dict(cosE=ce[order][:, order].cpu().numpy().astype(np.float16),
                cosF=cf[order][:, order].cpu().numpy().astype(np.float16), sign_sorted=sign[idx][order_np])
    return dict(cosE_same=ce[same & off].median().item(), cosE_opp=ce[~same].median().item(),
                cosF_same=cf[same & off].median().item(), cosF_opp=cf[~same].median().item()), mats


def alpha_f(Jw, u, v):
    ju = torch.einsum("bad,d->ba", Jw, u)
    jv = torch.einsum("bad,bd->ba", Jw, v)
    return (ju * jv).sum(-1) / (ju * ju).sum(-1).clamp_min(1e-18)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "SZ"], default="AT")
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_eval", type=int, default=200)
    p.add_argument("--n_check", type=int, default=56)
    p.add_argument("--n_perm", type=int, default=2000)
    a = p.parse_args()
    C.ensure_dirs()
    dev = C.device()
    bank = C.load_bank(f"{a.env}_worldedit")
    sign = bank["sign"].astype(int)
    fit, check, ev = split_indices(len(sign), a.n_eval, a.n_check)
    axis_dir = os.path.join(C.RESULTS, "axis")
    os.makedirs(axis_dir, exist_ok=True)
    rows, state_rows = [], []
    for cohort, seed, ck in C.agent_list(a.env, a.seeds_gm, a.seeds_control, a.init_seeds):
        agent = C.build_agent(ck, init_seed=seed, dev=dev)
        g = agent_geometry(agent, bank, dev, a.env)
        dnp = g["d_comp"].cpu().numpy().astype(np.float64)
        u = AX.fit_u_comp({+1: [dnp[i] for i in fit if sign[i] > 0], -1: [dnp[i] for i in fit if sign[i] < 0]})
        held = AX.held_out_alignment(u, [dnp[i] for i in check], [float(sign[i]) for i in check])
        held_ev = AX.held_out_alignment(u, [dnp[i] for i in ev], [float(sign[i]) for i in ev])
        null = AX.bucket_permutation_null([dnp[i] for i in fit], [int(sign[i]) for i in fit],
                                          [dnp[i] for i in check], [int(sign[i]) for i in check], n_perm=a.n_perm)
        ut = torch.as_tensor(u, dtype=torch.float32, device=dev)
        evt = torch.as_tensor(ev, device=dev)
        sg = torch.as_tensor(sign, dtype=torch.float32, device=dev)
        # Pulls on the signed axis at eval states: sigma * u points toward the goal candidate u.
        a_prox = (alpha_f(g["Jw"], ut, g["d_prox"]) * sg)[evt]
        a_star = (alpha_f(g["Jw"], ut, g["d_star"]) * sg)[evt]
        cw = cos_e(g["d_star"], g["d_prox"])
        cfw = cos_f_same_state(g["Jw"], g["d_star"], g["d_prox"])
        cs, mats = cross_state(g, fit, sign)
        r = dict(cohort=cohort, seed=seed, cosE_within=cw[evt].median().item(), cosF_within=cfw[evt].median().item(),
                 **cs, heldout_align_check=held["mean"], heldout_frac_pos_check=held["frac_pos"],
                 heldout_align_eval=held_ev["mean"], heldout_frac_pos_eval=held_ev["frac_pos"],
                 perm_p=null["p_upper_bound"], perm_true=null["true_score"], perm_mean=null["perm_score_mean"],
                 pull_proxy=a_prox.abs().median().item(), pull_goal=a_star.abs().median().item(),
                 pull_ratio=(a_prox.abs().median() / a_star.abs().median().clamp_min(1e-12)).item(),
                 dcomp_norm_eval=g["d_comp"][evt].norm(dim=-1).median().item())
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        for j, i in enumerate(ev):
            state_rows.append(dict(cohort=cohort, seed=seed, state=int(i), sign=int(sign[i]),
                                   cosE_within=cw[i].item(), cosF_within=cfw[i].item(),
                                   alpha_prox=a_prox[j].item(), alpha_goal=a_star[j].item(),
                                   dcomp_norm=g["d_comp"][i].norm().item()))
        np.savez(os.path.join(axis_dir, f"{a.env}_{cohort}_s{seed}.npz"), u_comp=u.astype(np.float32), fit=fit,
                 check=check, eval=ev, **mats)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS, f"axis_{a.env}_per_seed.csv"), index=False)
    pd.DataFrame(state_rows).to_csv(os.path.join(C.RESULTS, f"axis_{a.env}_per_state.csv"), index=False)
    cols = [c for c in df.columns if c not in ("cohort", "seed")]
    summ = []
    for cohort, sub in df.groupby("cohort"):
        r = dict(cohort=cohort, n=len(sub))
        for c in cols:
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(sub[c].to_numpy())
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(C.RESULTS, f"axis_{a.env}_summary.csv"), index=False)
    with open(os.path.join(C.RESULTS, f"axis_{a.env}_summary.md"), "w") as f:
        f.write(f"# {ENV_NAME[a.env]} cue geometry at h_fc (mean and 95% t-CI across seeds; per-seed medians over states)\n\n")
        f.write("| quantity | " + " | ".join(summ["cohort"]) + " |\n|---|" + "---|" * len(summ) + "\n")
        for c in cols:
            f.write(f"| {c} | " + " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi']) for _, r in summ.iterrows()) + " |\n")
    print(open(os.path.join(C.RESULTS, f"axis_{a.env}_summary.md")).read())


if __name__ == "__main__":
    main()
