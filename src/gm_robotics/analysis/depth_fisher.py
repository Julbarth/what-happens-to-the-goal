"""Layerwise policy-Fisher sensitivity (Q, G, D_F) and relative edit size of the goal / proxy world edits."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C

OUT = os.path.join(C.RESULTS, "depth")
SITES = ("conv1", "conv2", "h_pre", "h_fc", "a_hid")
# (role, base obs key, edited obs key); LR has no proxy world edit, distractor_move is its matched nuisance edit.
EDITS = {"AT": {"shape_swap": ("goal", "base", "shape_swap"), "colour_swap": ("proxy", "base", "colour_swap")},
         "SZ": {"mark_swap": ("goal", "base", "mark_swap"), "colour_swap": ("proxy", "base", "colour_swap")},
         "LR": {"goal_move": ("goal", "g_d", "g2_d"), "distractor_move": ("nuisance", "g_d", "g_d2")},
         "PTM": {"marker_move": ("goal", "base", "m2"), "cube_yaw": ("nuisance", "base", "yaw")}}
SEEDS = {"AT": dict(gm=[2, 3, 4], control=[2, 3, 4, 5]), "SZ": dict(gm=[0, 1, 2], control=[0, 1, 2]),
         "LR": dict(gm=[1, 2, 3], control=[1, 2, 3]), "PTM": dict(gm=[0, 1, 2], control=[0, 1, 2])}


def downstream(agent, site):
    """mu as a function of the activation at `site` (proprio branch output held fixed)."""
    p = C.parts(agent)

    def from_fc(h_fc, hs):
        return p["actor2"](torch.relu(p["actor0"](torch.cat([h_fc, hs], -1))))

    def from_pre(h, hs):
        return from_fc(torch.relu(p["fc"](h)), hs)

    if site == "a_hid":
        return lambda h, hs: p["actor2"](h)
    if site == "h_fc":
        return from_fc
    if site == "h_pre":
        return from_pre
    if site == "conv2":
        return lambda h, hs: from_pre(torch.relu(p["conv3"](h)).flatten(1), hs)
    return lambda h, hs: from_pre(torch.relu(p["conv3"](torch.relu(p["conv2"](h)))).flatten(1), hs)


def site_terms(agent, site, h, hs, delta, sig):
    """Q = delta^T F delta via one JVP; tr(F) via one VJP per action dimension (F = J^T diag(1/sigma^2) J)."""
    f = downstream(agent, site)
    w = 1.0 / sig ** 2
    with torch.enable_grad():
        _, jd = torch.func.jvp(lambda x: f(x, hs), (h,), (delta,))
        hh = h.detach().requires_grad_()
        mu = f(hh, hs)
        tr = torch.zeros(len(h), device=h.device)
        for k in range(mu.shape[1]):
            g, = torch.autograd.grad(mu[:, k].sum(), hh, retain_graph=k < mu.shape[1] - 1)
            tr += w[k] * g.flatten(1).pow(2).sum(1)
    q = (w * jd ** 2).sum(1)
    d = h[0].numel()
    n2 = delta.flatten(1).pow(2).sum(1)
    iso = tr / d
    return dict(Q=q, trF_per_d=iso, dnorm2=n2, G=(q / n2.clamp_min(1e-30)) / iso.clamp_min(1e-30), D_F=n2 * iso)


@torch.no_grad()
def agent_rows(agent, bank, env, dev, bs, n_pairs=4096):
    sig = C.sigma(agent)
    keys = sorted({k for _, kb, ke in EDITS[env].values() for k in (kb, ke)})
    acts = {}
    for key in keys:
        out = {s: [] for s in SITES + ("h_state", "mu")}
        for sl in C.batched(len(bank["state_" + key]), bs):
            t = C.taps(agent, C.to_t(bank["rgb_" + key][sl], dev), C.to_t(bank["state_" + key][sl], dev))
            for s in out:
                out[s].append(t[s])
        acts[key] = {s: torch.cat(v) for s, v in out.items()}
    n = len(bank["state_" + keys[0]])
    rng = np.random.default_rng(0)
    pa, pb = rng.integers(n, size=n_pairs), rng.integers(n, size=n_pairs)
    pb = np.where(pa == pb, (pb + 1) % n, pb)
    rows = []
    for edit, (role, kb, ke) in EDITS[env].items():
        b, e = acts[kb], acts[ke]
        kl = C.gaussian_kl(b["mu"], e["mu"], sig).cpu().numpy()
        for site in SITES:
            # between-scene scale: median squared distance between random pairs of base states at this site
            hb = b[site].flatten(1)
            scene2 = float(torch.median((hb[pa] - hb[pb]).pow(2).sum(1)))
            terms = {k: [] for k in ("Q", "G", "D_F", "trF_per_d", "dnorm2")}
            for sl in C.batched(n, bs):
                ft = site_terms(agent, site, b[site][sl], b["h_state"][sl], e[site][sl] - b[site][sl], sig)
                for k in terms:
                    terms[k].append(ft[k])
            r = {k: torch.cat(v).cpu().numpy() for k, v in terms.items()}
            rows.append(pd.DataFrame(dict(edit=edit, role=role, site=site, state=np.arange(n), KL=kl,
                                          rel_edit=r["dnorm2"] / scene2, calib=kl / np.maximum(0.5 * r["Q"], 1e-30), **r)))
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "LR", "SZ", "PTM"], required=True)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()
    os.makedirs(os.path.join(OUT, "fisher_raw"), exist_ok=True)
    dev = C.device()
    bank = C.load_bank(f"{a.env}_worldedit")
    jobs = [(c, s, C.ckpt_path(a.env, c, s)) for c in ("gm", "control") for s in SEEDS[a.env][c]]
    jobs += [("init", s, None) for s in a.init_seeds]
    dfs = []
    for cohort, seed, ck in jobs:
        out = os.path.join(OUT, "fisher_raw", f"{a.env}_{cohort}_s{seed}.csv")
        if not os.path.exists(out):
            agent = C.build_agent(ck, init_seed=seed, dev=dev)
            df = agent_rows(agent, bank, a.env, dev, a.batch)
            df.insert(0, "seed", seed)
            df.insert(0, "cohort", cohort)
            df.insert(0, "env", a.env)
            df.to_csv(out, index=False)
        df = pd.read_csv(out)
        print(f"{a.env} {cohort} s{seed}: " + ", ".join(
            f"{e}/{s} Q {g.Q.median():.3g} G {g.G.median():.3g}" for (e, s), g in df.groupby(["edit", "site"])), flush=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    vals = ["Q", "G", "D_F", "trF_per_d", "dnorm2", "rel_edit", "calib"]
    per_seed = df.groupby(["env", "edit", "role", "site", "cohort", "seed"])[vals].median().reset_index()
    per_seed = per_seed.merge(df.groupby(["env", "edit", "role", "site", "cohort", "seed"])["KL"].mean().reset_index())
    per_seed.to_csv(os.path.join(OUT, f"fisher_{a.env}_per_seed.csv"), index=False)


if __name__ == "__main__":
    main()
