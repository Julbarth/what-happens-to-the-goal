"""Isaac quadruped push-pull: change in log support of the favoured and competing candidate under each swap."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.special import erfcx, log_ndtr, logsumexp

from gm_robotics.isaac.analysis import common_isaac as C
from gm_robotics.isaac.analysis import geom

OUT = os.path.join(C.RESULTS, "push_pull")
FAV = {"flag_swap": "v", "tower_swap": "u"}
NEUTRAL_K, NEUTRAL_EXCL = 24, np.deg2rad(30)


def oracle_cmd(b):
    """gm_nav_env.scripted_action for a target at robot-frame bearing b: forward when roughly aligned, yaw rate 1.5 b."""
    vx = np.where(np.abs(b) < 0.6, np.clip(np.cos(b), 0, 1), 0.0)
    return np.stack([vx, np.zeros_like(b), np.clip(1.5 * b, -1, 1)], -1)


def dlogn(a, mu0, mue, sig):
    """log N(a; mue, sig) - log N(a; mu0, sig) for a diagonal Gaussian; a [..., 3] broadcast against mu [n, 3]."""
    return (((mue - mu0) * (a - 0.5 * (mu0 + mue))) / sig ** 2).sum(-1)


def oracle_form(mu0, mue, sig, b_fav, b_comp):
    """S = change in log density at the oracle command toward a candidate, minus the mean change at oracle commands
    toward neutral bearings (24 evenly spaced, those within 30 deg of either candidate excluded)."""
    th = np.linspace(-np.pi, np.pi, NEUTRAL_K, endpoint=False)
    out = {}
    neu = np.full(len(mu0), np.nan)
    for i in range(len(mu0)):
        ok = (np.abs(geom.wrap(th - b_fav[i])) >= NEUTRAL_EXCL) & (np.abs(geom.wrap(th - b_comp[i])) >= NEUTRAL_EXCL)
        neu[i] = dlogn(oracle_cmd(th[ok]), mu0[i], mue[i], sig).mean()
    out["S_fav"] = dlogn(oracle_cmd(b_fav), mu0, mue, sig) - neu
    out["S_comp"] = dlogn(oracle_cmd(b_comp), mu0, mue, sig) - neu
    # Random whitened mean shift: push-pull rate = angle between the whitened linearised weight vectors / 2 pi.
    wf = (oracle_cmd(b_fav) - mu0) / sig ** 2
    wc = (oracle_cmd(b_comp) - mu0) / sig ** 2
    wn = np.stack([((oracle_cmd(th[(np.abs(geom.wrap(th - b_fav[i])) >= NEUTRAL_EXCL) &
                                     (np.abs(geom.wrap(th - b_comp[i])) >= NEUTRAL_EXCL)]) - mu0[i]) / sig ** 2).mean(0)
                   for i in range(len(mu0))])
    wf, wc = (wf - wn) * sig, (wc - wn) * sig
    ang = np.arccos(np.clip((wf * wc).sum(1) / (np.linalg.norm(wf, axis=1) * np.linalg.norm(wc, axis=1) + 1e-12), -1, 1))
    out["null_pushpull"] = ang / (2 * np.pi)
    return out


def _log1p_tmills(t):
    """log(1 + t Phi(t) / phi(t)), stable for all t."""
    out = np.empty_like(t)
    pos = t >= 0
    tp = t[pos]
    out[pos] = np.logaddexp(0.0, np.log(np.maximum(tp, 1e-300)) + log_ndtr(tp) + 0.5 * tp ** 2 + 0.5 * np.log(2 * np.pi))
    x = -t[~pos]
    big = x > 8
    val = np.empty_like(x)
    xs = x[~big]
    val[~big] = 1 - xs * np.sqrt(np.pi / 2) * erfcx(xs / np.sqrt(2))
    xb = x[big]
    val[big] = 1 / xb ** 2 - 3 / xb ** 4 + 15 / xb ** 6 - 105 / xb ** 8
    out[~pos] = np.log(np.maximum(val, 1e-300))
    return out


def log_angle_density(m, s, th):
    """log density of the direction angle of a 2-D diagonal Gaussian N(m, diag(s^2)) at angles th; m [n, 2] -> [n, len(th)]."""
    c, sn = np.cos(th)[None], np.sin(th)[None]
    A = c ** 2 / s[0] ** 2 + sn ** 2 / s[1] ** 2
    B = m[:, :1] * c / s[0] ** 2 + m[:, 1:] * sn / s[1] ** 2
    Cc = (m[:, :1] / s[0]) ** 2 + (m[:, 1:] / s[1]) ** 2
    t = B / np.sqrt(A)
    return -np.log(2 * np.pi * s[0] * s[1]) - 0.5 * Cc - np.log(A) + _log1p_tmills(t)


def cone_form(mu0, mue, sig, b_fav, b_comp, width, n_th=2880):
    """Set form on the commanded planar velocity (vx, vy): A_c = directions within `width` of candidate c's bearing
    (ties split to the nearer candidate), A_0 = all other directions; log-odds changes against A_0."""
    th = np.linspace(-np.pi, np.pi, n_th, endpoint=False)
    dth = np.log(2 * np.pi / n_th)
    df, dc = np.abs(geom.wrap(th[None] - b_fav[:, None])), np.abs(geom.wrap(th[None] - b_comp[:, None]))
    in_f = (df < width) & (df <= dc)
    in_c = (dc < width) & (dc < df)
    in_0 = ~(in_f | in_c)

    def lp(mu, mask):
        ld = log_angle_density(mu[:, :2], sig[:2], th) + dth
        return logsumexp(np.where(mask, ld, -np.inf), axis=1)

    l = {k: {m: lp(mu, msk) for m, msk in (("f", in_f), ("c", in_c), ("0", in_0))} for k, mu in (("0", mu0), ("e", mue))}
    return dict(S_fav=(l["e"]["f"] - l["e"]["0"]) - (l["0"]["f"] - l["0"]["0"]),
                S_comp=(l["e"]["c"] - l["e"]["0"]) - (l["0"]["c"] - l["0"]["0"]),
                p_fav0=np.exp(l["0"]["f"]), p_comp0=np.exp(l["0"]["c"]), p_neutral0=np.exp(l["0"]["0"]))


@torch.no_grad()
def means(agent, bank, key, dev, bs=256):
    out = []
    for sl in C.batched(len(bank["start"]), bs):
        out.append(C.taps(agent, C.to_t(bank["rgb_" + key][sl], dev), C.to_t(bank["state_base"][sl], dev))["mu"].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def selftest():
    """Analytic cone probabilities against Monte Carlo."""
    rng = np.random.default_rng(0)
    mu, s = rng.normal(0, 0.6, (6, 3)), np.array([0.3, 0.15, 0.4])
    bf, bc = rng.uniform(-0.7, 0.7, 6), rng.uniform(-0.7, 0.7, 6)
    r = cone_form(mu, mu, s, bf, bc, np.deg2rad(30))
    a = mu[:, None, :2] + s[:2] * rng.normal(0, 1, (6, 400000, 2))
    ang = np.arctan2(a[..., 1], a[..., 0])
    df, dc = np.abs(geom.wrap(ang - bf[:, None])), np.abs(geom.wrap(ang - bc[:, None]))
    w = np.deg2rad(30)
    print("analytic", np.round(r["p_fav0"], 4), "MC", np.round(((df < w) & (df <= dc)).mean(1), 4))
    print("analytic", np.round(r["p_comp0"], 4), "MC", np.round(((dc < w) & (dc < df)).mean(1), 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--banks", nargs="+", default=["visible", "conflict", "far"])
    p.add_argument("--widths", type=float, nargs="+", default=[15, 30, 45], help="cone half-widths (deg) for the set form")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    os.makedirs(OUT, exist_ok=True)
    dev = C.device()
    rows = []
    for bname in a.banks:
        bank = C.load_bank(f"edit_{bname}")
        s, yaw = bank["start"], bank["yaw"]
        ru, rv = geom.robot_frame(bank["flag"], s, yaw), geom.robot_frame(bank["tower"], s, yaw)
        bear = dict(u=np.arctan2(ru[:, 1], ru[:, 0]), v=np.arctan2(rv[:, 1], rv[:, 0]))
        for cohort, seed, ck in C.agent_list():
            ag = C.build_agent(ck, init_seed=seed, dev=dev)
            sig = C.sigma(ag).cpu().numpy().astype(np.float64)
            mu0 = means(ag, bank, "base", dev)
            for edit, fav in FAV.items():
                comp = "u" if fav == "v" else "v"
                mue = means(ag, bank, edit, dev)
                r = dict(bank=bname, cohort=cohort, seed=seed, edit=edit, state=np.arange(len(s)),
                         flag_px=bank["flag_px_base"], tower_px=bank["tower_px_base"])
                o = oracle_form(mu0, mue, sig, bear[fav], bear[comp])
                r.update({f"{k}": v for k, v in o.items()})
                for w in a.widths:
                    cf = cone_form(mu0, mue, sig, bear[fav], bear[comp], np.deg2rad(w))
                    r.update({f"{k}_cone{int(w)}": v for k, v in cf.items()})
                df = pd.DataFrame(r)
                rows.append(df)
                print(f"{bname} {cohort} s{seed} {edit}: oracle S_fav {df.S_fav.mean():+.3f} S_comp {df.S_comp.mean():+.3f} "
                      f"pp {np.mean((df.S_fav > 0) & (df.S_comp < 0)):.2f} | cone30 S_fav {df.S_fav_cone30.mean():+.3f} "
                      f"S_comp {df.S_comp_cone30.mean():+.3f}", flush=True)
    st = pd.concat(rows, ignore_index=True)
    st.to_csv(os.path.join(OUT, "isaac_per_state.csv"), index=False)
    forms = [""] + [f"_cone{int(w)}" for w in a.widths]
    for f in forms:
        st[f"pushpull{f}"] = ((st[f"S_fav{f}"] > 0) & (st[f"S_comp{f}"] < 0)).astype(float)
        st[f"dl{f}"] = st[f"S_fav{f}"] - st[f"S_comp{f}"]
    # Clean subset: both objects clearly visible (>= 20 px) at the start state.
    st["clean"] = (st.flag_px >= 20) & (st.tower_px >= 20)
    cols = [f"{k}{f}" for f in forms for k in ("S_fav", "S_comp", "dl", "pushpull")] + ["null_pushpull"]
    summ, per_seed_all = [], []
    for subset, sub0 in (("all", st), ("clean", st[st.clean])):
        g = sub0.groupby(["bank", "edit", "cohort", "seed"])
        ps = g[cols].mean().join(g[["S_fav", "S_comp"]].median().add_suffix("_med")).join(g.size().rename("n_states")).reset_index()
        ps.insert(0, "subset", subset)
        per_seed_all.append(ps)
        for key, sub in ps.groupby(["bank", "edit", "cohort"], sort=False):
            rr = dict(subset=subset, **dict(zip(["bank", "edit", "cohort"], key)), n_seeds=len(sub),
                      n_states=int(sub.n_states.mean()))
            for c in cols + ["S_fav_med", "S_comp_med"]:
                rr[c], rr[c + "_lo"], rr[c + "_hi"] = C.ci95(sub[c].to_numpy())
            for f in forms:
                rr[f"seeds_pp{f}"] = int(((sub[f"S_fav{f}"] > 0) & (sub[f"S_comp{f}"] < 0)).sum())
            summ.append(rr)
    ps = pd.concat(per_seed_all)
    ps.to_csv(os.path.join(OUT, "isaac_per_seed.csv"), index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(OUT, "isaac_summary.csv"), index=False)
    with open(os.path.join(OUT, "isaac_summary.md"), "w") as fh:
        fh.write("# Isaac push-pull: change in log support of the favoured (S_fav) and competing (S_comp) candidate vs neutral "
                 "actions; seed means of state means, mean [95% t-CI] across 3 seeds. Oracle form: log density at the "
                 "privileged heading-controller command toward each candidate. coneW: paper set form on the planar command "
                 "direction, half-width W deg. pp = fraction of states in the push-pull quadrant (S_fav > 0, S_comp < 0); "
                 "seeds = seed means in the quadrant. clean = flag and tower >= 20 px at the start state.\n\n")
        for f, lab in zip(forms, ["oracle"] + [f"cone {int(w)} deg" for w in a.widths]):
            fh.write(f"## {lab}\n\n| subset | bank | edit | cohort | n states | S_fav | S_comp | dl | pp | seeds in quadrant |"
                     + (" null pp |" if f == "" else "") + "\n|---|---|---|---|---|---|---|---|---|---|" + ("---|" if f == "" else "") + "\n")
            for _, r in sm.iterrows():
                fh.write(f"| {r.subset} | {r.bank} | {r.edit} | {r.cohort} | {r.n_states} | " + " | ".join(
                    C.fmt_ci(r[c + f], r[c + f + '_lo'], r[c + f + '_hi'], 2) for c in ("S_fav", "S_comp", "dl", "pushpull"))
                    + f" | {r['seeds_pp' + f]}/{r.n_seeds} |" + (f" {C.fmt_ci(r.null_pushpull, r.null_pushpull_lo, r.null_pushpull_hi, 2)} |"
                                                               if f == "" else "") + "\n")
            fh.write("\n")
    print(open(os.path.join(OUT, "isaac_summary.md")).read())


if __name__ == "__main__":
    main()
