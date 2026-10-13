"""LocationReach depth analysis: goal decodability on closed-loop BC frames and marker response of the rewired heads."""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from gm_robotics.analysis import depth_common as D

C = D.load_common()

WS_LO, WS_HI, CORNER = (-0.20, -0.25), (0.20, 0.25), np.array((0.15, -0.20), np.float32)
LR_DIST_MIN = 0.08
TCP_SLICE = slice(18, 21)  # state = qpos (9) | qvel (9) | tcp pose (7); checked against the bank's tcp below


def bc_configs(n, seed=4100):
    """Replicates envs_forced.sample_lr_probe_configs (train half of the start bank) so the BC bank's marker
    positions can be recovered without the simulator."""
    rng = np.random.default_rng(seed)
    q = np.load(os.path.join(D.ROOT, "gm_robotics", "envs", "start_bank.npz"))["qpos"].astype(np.float32)[0::2]
    qpos = q[rng.integers(len(q), size=n)]
    uni = lambda k: rng.uniform(WS_LO, WS_HI, (k, 2)).astype(np.float32)
    g = uni(n)
    d = uni(n)
    bad = np.linalg.norm(d - g, axis=1) < LR_DIST_MIN
    while bad.any():
        d[bad] = uni(int(bad.sum()))
        bad = np.linalg.norm(d - g, axis=1) < LR_DIST_MIN
    return dict(qpos=qpos, g=g, d=d)


def trunk(cohort, seed):
    if cohort == "init":
        return C.build_agent(None, init_seed=seed)
    return C.build_agent(C.ckpt_path("LR", cohort, seed))


@torch.no_grad()
def features(agent, rgb, state, bs=256):
    out = {k: [] for k in ("h_pre", "h_fc", "h_state", "mu")}
    for sl in C.batched(len(rgb), bs):
        t = C.taps(agent, torch.as_tensor(rgb[sl]), torch.as_tensor(state[sl]))
        for k in out:
            out[k].append(t[k].float())
    return {k: torch.cat(v) for k, v in out.items()}


def dual_ridge(Xtr, Ytr, Xva, Yva, Xte, alphas=10.0 ** np.arange(-4, 4)):
    """Ridge on centred / globally scaled features, alpha chosen on the validation episodes (never on test)."""
    mu, sc = Xtr.mean(0, keepdims=True), Xtr.std()
    A, B, T = (Xtr - mu) / sc, (Xva - mu) / sc, (Xte - mu) / sc
    ym = Ytr.mean(0, keepdims=True)
    K = A @ A.T
    s, U = torch.linalg.eigh(K.double())
    s, U = s.float(), U.float()
    UtY = U.T @ (Ytr - ym)
    Kva, Kte = B @ A.T, T @ A.T
    base = s.mean()
    best = None
    for a in alphas:
        coef = U @ (UtY / (s + a * base)[:, None])
        r2 = 1 - ((Kva @ coef + ym - Yva) ** 2).sum(0) / ((Yva - Yva.mean(0)) ** 2).sum(0)
        if best is None or r2.mean() > best[0]:
            best = (r2.mean(), a, coef)
    return Kte @ best[2] + ym, best[1]


def r2(pred, y):
    return 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def part_bcprobe(n_ep_sub, out):
    z = np.load(os.path.join(D.BANKS, "LR_bc.npz"))
    epi, src, state = z["episode"], z["source"], z["state"]
    cfg = bc_configs(192 * 6)
    # frames are interleaved across parallel envs, so an episode's t = 0 frame is its first occurrence
    _, first = np.unique(epi, return_index=True)
    qerr = np.abs(state[first, :9] - cfg["qpos"][epi[first]]).max()
    print(f"config check: max |qpos(t=0) - regenerated qpos| = {qerr:.2e}", flush=True)
    assert qerr < 0.05, "regenerated BC configs do not match the bank"
    rng = np.random.default_rng(0)
    eps = rng.permutation(np.unique(epi))[:n_ep_sub]
    split = {e: ("train" if i < 0.6 * len(eps) else "val" if i < 0.8 * len(eps) else "test") for i, e in enumerate(eps)}
    keep = np.isin(epi, eps)
    idx = np.nonzero(keep)[0]
    rgb = z["rgb"][idx]
    st, ep = state[idx], epi[idx]
    t_in_ep = pd.Series(ep).groupby(ep).cumcount().to_numpy()
    tcp = st[:, TCP_SLICE]
    g = cfg["g"][ep]
    bearing = g - tcp[:, :2]
    dist = np.linalg.norm(bearing, axis=1)
    Y = torch.as_tensor(np.c_[g, bearing], dtype=torch.float32)
    tnames = ["marker_x", "marker_y", "bearing_x", "bearing_y"]
    sp = np.array([split[e] for e in ep])
    te = sp == "test"
    strata = {"all": np.ones(te.sum(), bool), "t0": t_in_ep[te] == 0, "t_gt0": t_in_ep[te] > 0,
              "near_lt10cm": dist[te] < 0.10, "far_ge10cm": dist[te] >= 0.10,
              "src_gm": src[idx][te] < 3, "src_control": src[idx][te] >= 3}
    print(f"frames {len(idx)} (train {np.sum(sp == 'train')}, val {np.sum(sp == 'val')}, test {te.sum()}); "
          f"test t0 {strata['t0'].sum()}, near {strata['near_lt10cm'].sum()}", flush=True)
    rows = []
    for cohort, seeds in (("gm", (1, 2, 3)), ("control", (1, 2, 3)), ("init", (1, 2, 3))):
        for s in seeds:
            f = features(trunk(cohort, s), rgb, st)
            for site in ("h_pre", "h_fc"):
                X = f[site]
                pred, alpha = dual_ridge(X[sp == "train"], Y[sp == "train"], X[sp == "val"], Y[sp == "val"], X[te])
                for sname, m in strata.items():
                    sc = r2(pred[m], Y[te][m]).numpy()
                    for tn, v in zip(tnames, sc):
                        rows.append(dict(cohort=cohort, seed=s, site=site, target=tn, stratum=sname, r2=float(v),
                                         alpha=float(alpha), n_test=int(m.sum())))
                    # bearing direction error (deg), the quantity a closed-loop servo needs
                    pb, yb = pred[m][:, 2:4].numpy(), Y[te][m][:, 2:4].numpy()
                    ang = np.degrees(np.arccos(np.clip((pb * yb).sum(1) / (np.linalg.norm(pb, axis=1) *
                                                                           np.linalg.norm(yb, axis=1) + 1e-9), -1, 1)))
                    rows.append(dict(cohort=cohort, seed=s, site=site, target="bearing_angle_err_deg", stratum=sname,
                                     r2=float(np.median(ang)), alpha=float(alpha), n_test=int(m.sum())))
            print(f"bcprobe {cohort} s{s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "q1_lr_bcframe_probes_per_seed.csv"), index=False)
    return df


@torch.no_grad()
def part_heads(out):
    """Marker response of the rewired h_pre heads (and native policies) on the LR world-edit start frames:
    KL in teacher-sigma units and the direction of the first-order TCP step change relative to g2 - g."""
    z = np.load(os.path.join(D.BANKS, "LR_worldedit.npz"))
    Jt = torch.as_tensor(z["J_tcp"])
    hd = os.path.join(D.ROOT, "results", "analysis", "rewire_check", "heads")
    edits = {"goal_move": ("g_d", "g2_d", z["g2"] - z["g"]), "distractor_move": ("g_d", "g_d2", z["d2"] - z["d"])}
    rows = []
    for cohort in ("gm", "control", "init"):
        for s in (1, 2, 3):
            base = trunk(cohort, s)
            feats = {k: features(base, z["rgb_" + k], z["state_" + k]) for k in ("g_d", "g2_d", "g_d2")}
            policies = {}
            if cohort != "init":
                policies["native"] = (lambda f, b=base: f["mu"], C.sigma(base))
            for arm in ("pre", "fc_head", "remix", "pre_masked"):
                path = os.path.join(hd, f"{cohort}_s{s}_{arm}.pt")
                if not os.path.exists(path):
                    continue
                ck = torch.load(path, map_location="cpu")
                sd = ck["state_dict"]
                teacher = C.build_agent(C.ckpt_path("LR", "control", ck["teacher"]))
                W1, b1, W2, b2 = sd["head.0.weight"], sd["head.0.bias"], sd["head.2.weight"], sd["head.2.bias"]
                Wf, bf = sd.get("fc.weight"), sd.get("fc.bias")

                def head(f, Wf=Wf, bf=bf, W1=W1, b1=b1, W2=W2, b2=b2):
                    h = f["h_fc"] if Wf is None else torch.relu(f["h_pre"] @ Wf.T + bf)
                    return torch.relu(torch.cat([h, f["h_state"]], 1) @ W1.T + b1) @ W2.T + b2
                policies[arm] = (head, C.sigma(teacher))
            for arm, (pol, sig) in policies.items():
                for edit, (kb, ke, dvec) in edits.items():
                    mb, me = pol(feats[kb]), pol(feats[ke])
                    kl = C.gaussian_kl(mb, me, sig).numpy()
                    dstep = torch.einsum("bkj,bj->bk", Jt, C.JOINT_DELTA_SCALE * (me - mb)[:, :7])[:, :2].numpy()
                    u = dvec / np.linalg.norm(dvec, axis=1, keepdims=True)
                    along = (dstep * u).sum(1)
                    cosv = along / (np.linalg.norm(dstep, axis=1) + 1e-12)
                    step_b = torch.einsum("bkj,bj->bk", Jt, C.JOINT_DELTA_SCALE * mb[:, :7])[:, :2].numpy()
                    rows.append(dict(cohort=cohort, seed=s, arm=arm, edit=edit, KL=float(kl.mean()),
                                     KL_median=float(np.median(kl)), cos_dstep_edit=float(cosv.mean()),
                                     along_mm=float(1e3 * along.mean()),
                                     along_rel=float(along.mean() / np.linalg.norm(step_b, axis=1).mean())))
            print(f"heads {cohort} s{s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "q1_lr_rewired_head_marker_response_per_seed.csv"), index=False)
    return df


def summarise(bc, hd, out):
    lines = ["# Q1 LocationReach: closed-loop frame probes and rewired-head marker response", "",
             "Mean [95% t-CI] over 3 seeds per cohort. GM - init = Welch interval.", ""]
    if bc is not None:
        lines += ["## Ridge R2 on held-out BC-bank episodes (closed-loop frames, marker uniform)", "",
                  "| site | target | stratum | GM | Non-GM | init | GM - init |", "|---|---|---|---|---|---|---|"]
        for (site, tn, sn), g in bc.groupby(["site", "target", "stratum"], sort=False):
            v = {c: g[g.cohort == c].sort_values("seed").r2.to_numpy() for c in ("gm", "control", "init")}
            lines.append(f"| {site} | {tn} | {sn} | {D.fmt_ci(v['gm'])} | {D.fmt_ci(v['control'])} | "
                         f"{D.fmt_ci(v['init'])} | {D.fmt(*D.welch(v['gm'], v['init']))} |")
        lines.append("")
    if hd is not None:
        lines += ["## Rewired-head / native marker response at LR world-edit start frames", "",
                  "KL in the policy's (teacher's) sigma units; cos = cosine of the first-order TCP step change with the "
                  "edit direction (g2 - g or d2 - d); along_rel = directed component / mean base step length.", "",
                  "| arm | edit | cohort | KL | cos | along (mm) | along_rel |", "|---|---|---|---|---|---|---|"]
        for (arm, edit, coh), g in hd.groupby(["arm", "edit", "cohort"], sort=False):
            lines.append(f"| {arm} | {edit} | {coh} | {D.fmt_ci(g.KL)} | {D.fmt_ci(g.cos_dstep_edit)} | "
                         f"{D.fmt_ci(g.along_mm, 2)} | {D.fmt_ci(g.along_rel)} |")
    with open(os.path.join(out, "q1_lr_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["bcprobe", "heads"])
    p.add_argument("--n_ep_sub", type=int, default=400, help="BC episodes subsampled for the frame probes")
    a = p.parse_args()
    torch.set_num_threads(8)
    out = D.ensure_out()
    bc = part_bcprobe(a.n_ep_sub, out) if "bcprobe" in a.parts else None
    hd = part_heads(out) if "heads" in a.parts else None
    if bc is None and os.path.exists(os.path.join(out, "q1_lr_bcframe_probes_per_seed.csv")):
        bc = pd.read_csv(os.path.join(out, "q1_lr_bcframe_probes_per_seed.csv"))
    if hd is None and os.path.exists(os.path.join(out, "q1_lr_rewired_head_marker_response_per_seed.csv")):
        hd = pd.read_csv(os.path.join(out, "q1_lr_rewired_head_marker_response_per_seed.csv"))
    summarise(bc, hd, out)


if __name__ == "__main__":
    main()
