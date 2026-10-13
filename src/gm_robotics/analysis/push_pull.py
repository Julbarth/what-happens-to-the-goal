"""Continuous-action push-pull (paper S_fav / S_comp) for AttributeTable and SortZones swaps at start and decision-point banks."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr, owens_t
from gm_robotics.analysis import common as C
from gm_robotics.analysis import bank_audit as BA

OUT = os.path.join(C.RESULTS, "push_pull")
FAV = {"shape_swap": "v", "mark_swap": "v", "colour_swap": "u"}
SZ_CUBE_HALF = 0.02


def phi2(h, k, rho):
    """Bivariate standard normal CDF P(X <= h, Y <= k) with correlation rho (Owen's T form, vectorised)."""
    h = np.where(np.abs(h) < 1e-12, 1e-12, h)
    k = np.where(np.abs(k) < 1e-12, 1e-12, k)
    r = np.sqrt(np.clip(1 - rho ** 2, 1e-12, None))
    beta = np.where(h * k > 0, 0.0, 0.5)
    return np.clip(0.5 * (ndtr(h) + ndtr(k)) - owens_t(h, (k - rho * h) / (h * r)) - owens_t(k, (h - rho * k) / (k * r)) - beta,
                   0.0, 1.0)


def set_probs(mu, sig, wf, wc, tau=1.0):
    """P(A_fav), P(A_comp), P(A_0) for a diagonal Gaussian policy, with A_fav = {z_f > tau s_f, z_c <= tau s_c},
    A_comp = {z_c > tau s_c, z_f <= tau s_f}, A_0 = progress toward both or toward neither (z = w . a = first-order
    TCP progress per step, s = its action-noise sd). The tau s margin keeps A_0 a band of finite width even when
    the two candidates lie in opposite directions (z_f ~ -z_c), where tau = 0 leaves A_0 a vanishing wedge and the
    log ratios explode."""
    mf, mc = (wf * mu).sum(1), (wc * mu).sum(1)
    sf, sc = np.sqrt((wf ** 2 * sig ** 2).sum(1)), np.sqrt((wc ** 2 * sig ** 2).sum(1))
    rho = np.clip((wf * wc * sig ** 2).sum(1) / (sf * sc), -0.999999, 0.999999)
    af, ac = mf / sf - tau, mc / sc - tau
    p_fav = phi2(af, -ac, -rho)
    p_comp = phi2(ac, -af, -rho)
    p_0 = phi2(af, ac, rho) + phi2(-af, -ac, rho)
    return p_fav, p_comp, p_0


def candidate_dirs(env, b, idx, kind):
    """Unit TCP-velocity direction that counts as progress toward u and toward v. AT: toward the object. SZ start
    frames: toward that candidate's push point behind the cube (the env's reward); SZ decision frames (TCP at
    the cube): along the push direction cube -> pad."""
    tcp = b["tcp"][idx].astype(np.float64)
    out = {}
    for c in ("u", "v"):
        xy = b[c][idx].astype(np.float64)
        if env == "AT":
            tgt = np.concatenate([xy, np.full((len(idx), 1), 0.03)], 1)
            g = tgt - tcp
        elif kind == "decision":
            cube = b["cube_now"][idx, :2].astype(np.float64)
            g = np.concatenate([xy - cube, np.zeros((len(idx), 1))], 1)
        else:
            cube = b["cube"][idx, :2].astype(np.float64)
            e = (xy - cube) / np.linalg.norm(xy - cube, axis=1, keepdims=True)
            push = np.concatenate([cube - e * (SZ_CUBE_HALF + 0.01), np.full((len(idx), 1), SZ_CUBE_HALF)], 1)
            g = push - tcp
        out[c] = g / np.linalg.norm(g, axis=1, keepdims=True)
    return out


def action_weights(g, Jtcp):
    """w with w . a = first-order TCP progress along g per step: g^T J_tcp (0.1 a_arm); gripper weight 0."""
    w = C.JOINT_DELTA_SCALE * np.einsum("bi,bij->bj", g, Jtcp.astype(np.float64))
    return np.concatenate([w, np.zeros((len(w), 1))], 1)


@torch.no_grad()
def policy_means(agent, b, idx, key, dev, bs=128):
    mus = []
    for sl in C.batched(len(idx), bs):
        ii = idx[sl]
        mus.append(C.taps(agent, C.to_t(b[f"rgb_{key}"][ii], dev), C.to_t(b[f"state_{key}"][ii], dev))["mu"].cpu().numpy())
    return np.concatenate(mus).astype(np.float64)


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    agents = [("gm", s, C.ckpt_path(a.env, "gm", s)) for s in BA.SEEDS[a.env]["gm"]]
    agents += [("control", s, C.ckpt_path(a.env, "control", s)) for s in BA.SEEDS[a.env]["control"]]
    agents += [("init", s, None) for s in (0, 1, 2)]
    built = {(c, s): C.build_agent(ck, init_seed=s, dev=dev) for c, s, ck in agents}
    edits = (BA.GOAL_E[a.env], BA.PROX_E[a.env])
    for variant in a.variants:
        f = BA.bank_path(a.env, variant)
        b = C.load_bank(os.path.basename(f)[:-4])
        if variant == "orig":
            b["pair"] = np.arange(len(b["sign"]))
        label, idx, _, _ = next(iter(BA.bank_views(a.env, variant, b)))
        dirs = candidate_dirs(a.env, b, idx, "decision" if variant == "decision" else "start")
        W = {c: action_weights(dirs[c], b["J_tcp"][idx]) for c in ("u", "v")}
        for cohort, seed, _ in agents:
            ag = built[(cohort, seed)]
            sig = C.sigma(ag).cpu().numpy().astype(np.float64)[None]
            mu0 = policy_means(ag, b, idx, "base", dev)
            for e in edits:
                fav = FAV[e]
                comp = "u" if fav == "v" else "v"
                mue = policy_means(ag, b, idx, e, dev)
                # Paper form (set log-odds against neutral actions): saturates for Gaussian policies whose mean
                # progress is many sigma from zero, kept only as a diagnostic (columns *_sets).
                pf0, pc0, pn0 = set_probs(mu0, sig, W[fav], W[comp])
                pf1, pc1, pn1 = set_probs(mue, sig, W[fav], W[comp])
                lg = lambda x: np.log(np.clip(x, 1e-300, None))
                sf_sets = (lg(pf1) - lg(pn1)) - (lg(pf0) - lg(pn0))
                sc_sets = (lg(pc1) - lg(pn1)) - (lg(pc0) - lg(pn0))
                # Reported form: change in probit support Phi^-1(pi(w . a > 0)) = (w . mu) / sd(w . a) of each
                # candidate's progress half-space; linear in mu, so it does not saturate.
                zf = lambda m: (W[fav] * m).sum(1) / np.sqrt((W[fav] ** 2 * sig ** 2).sum(1))
                zc = lambda m: (W[comp] * m).sum(1) / np.sqrt((W[comp] ** 2 * sig ** 2).sum(1))
                s_fav, s_comp = zf(mue) - zf(mu0), zc(mue) - zc(mu0)
                # Push-pull rate of a random isotropic mean shift (whitened): angle(w_f, w_c) / 2 pi.
                wf, wc = W[fav] * sig, W[comp] * sig
                ang = np.arccos(np.clip((wf * wc).sum(1) / (np.linalg.norm(wf, axis=1) * np.linalg.norm(wc, axis=1)), -1, 1))
                rows.append(pd.DataFrame(dict(env=a.env, bank=label, cohort=cohort, seed=seed, edit=e, state=idx,
                                              S_fav=s_fav, S_comp=s_comp, dl=s_fav - s_comp, null_pushpull=ang / (2 * np.pi),
                                              z_fav0=zf(mu0), z_comp0=zc(mu0), S_fav_sets=sf_sets, S_comp_sets=sc_sets,
                                              p_fav0=pf0, p_comp0=pc0, p_neutral0=pn0)))
                print(f"{a.env} {label} {cohort} s{seed} {e}: S_fav {np.mean(s_fav):+.3f} S_comp {np.mean(s_comp):+.3f} "
                      f"push-pull {np.mean((s_fav > 0) & (s_comp < 0)):.3f} p0 fav/comp/neutral "
                      f"{pf0.mean():.3f}/{pc0.mean():.3f}/{pn0.mean():.3f}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(OUT, f"{a.env}_per_state.csv"), index=False)
    df["pushpull"] = ((df.S_fav > 0) & (df.S_comp < 0)).astype(float)
    g = df.groupby(["env", "bank", "edit", "cohort", "seed"])
    per_seed = g[["S_fav", "S_comp", "dl", "pushpull", "null_pushpull", "z_fav0", "z_comp0"]].mean()
    per_seed = per_seed.join(g[["S_fav", "S_comp"]].median().add_suffix("_med")).reset_index()
    per_seed.to_csv(os.path.join(OUT, f"{a.env}_per_seed.csv"), index=False)
    summ = []
    for key, sub in per_seed.groupby(["env", "bank", "edit", "cohort"], sort=False):
        r = dict(zip(["env", "bank", "edit", "cohort"], key), n_seeds=len(sub))
        for c in ("S_fav", "S_comp", "dl", "pushpull", "null_pushpull", "S_fav_med", "S_comp_med"):
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(sub[c].to_numpy())
        r["seeds_pushpull"] = int(((sub.S_fav > 0) & (sub.S_comp < 0)).sum())
        summ.append(r)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT, f"{a.env}_summary.csv"), index=False)
    with open(os.path.join(OUT, f"{a.env}_summary.md"), "w") as fh:
        fh.write("# Push-pull in probit support units (change of (w.mu)/sd per candidate), seed means of state means, "
                 "mean [95% t-CI] across seeds; null_pushpull = push-pull rate of a random mean shift\n\n")
        cols = ("S_fav", "S_comp", "dl", "pushpull", "null_pushpull", "S_fav_med", "S_comp_med")
        fh.write("| bank | edit | cohort | " + " | ".join(cols) + " | seeds in push-pull quadrant |\n|---|---|---|" + "---|" * (len(cols) + 1) + "\n")
        for _, r in s.iterrows():
            fh.write(f"| {r.bank} | {r.edit} | {r.cohort} | " + " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi'], 2)
                     for c in cols) + f" | {r.seeds_pushpull}/{r.n_seeds} |\n")
    print(open(os.path.join(OUT, f"{a.env}_summary.md")).read(), flush=True)


def selftest():
    """phi2-based set probabilities against Monte Carlo on random Gaussians."""
    rng = np.random.default_rng(0)
    mu, sig = rng.normal(0, 1, (5, 8)), np.abs(rng.normal(0.3, 0.1, (1, 8)))
    wf, wc = rng.normal(0, 1, (5, 8)), rng.normal(0, 1, (5, 8))
    p = set_probs(mu, sig, wf, wc)
    a = mu[:, None] + sig[:, None] * rng.normal(0, 1, (5, 400000, 8))
    zf, zc = (a * wf[:, None]).sum(-1), (a * wc[:, None]).sum(-1)
    mc = ((zf > 0) & (zc <= 0)).mean(1), ((zc > 0) & (zf <= 0)).mean(1)
    print("analytic", np.round(p[0], 4), np.round(p[1], 4), "MC", np.round(mc[0], 4), np.round(mc[1], 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "SZ"], default="AT")
    p.add_argument("--variants", nargs="+", default=["orig", "strict", "decision"])
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    run(a)


if __name__ == "__main__":
    main()
