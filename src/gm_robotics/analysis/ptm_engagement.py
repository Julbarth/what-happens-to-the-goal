"""Where the GM policy reacts to the goal marker (PTM presence by location, LR goal vs distractor) on the world-edit banks."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats

from gm_robotics.analysis import depth_common as D

C = D.load_common()

CORNER = np.array((0.15, -0.20), np.float32)


def agent(env, cohort, seed):
    if cohort == "init":
        return C.build_agent(None, init_seed=seed)
    return C.build_agent(C.ckpt_path(env, cohort, seed))


@torch.no_grad()
def mu_of(a, rgb, state, bs=256):
    return torch.cat([C.taps(a, torch.as_tensor(rgb[sl]), torch.as_tensor(state[sl]))["mu"]
                      for sl in C.batched(len(rgb), bs)])


def npx(a, b):
    return (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > 0).sum((1, 2))


def tcp_dstep(Jt, dmu):
    return torch.einsum("bkj,bj->bk", torch.as_tensor(Jt), C.JOINT_DELTA_SCALE * dmu[:, :7]).numpy()


def seg_dist(p, a, b):
    """Distance of points p to the segments a -> b (all [N, 2])."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def ptm_rows(bank_name):
    z = np.load(os.path.join(D.BANKS, bank_name + ".npz"))
    cube = z["cube"][:, :2]
    tcp = z["tcp"][:, :2]
    locs = {"m": z["m"], "m2": z["m2"], "centre": z["centre"]}
    if "rgb_mlat" in z:
        locs["mlat"] = z["mlat"]
    px = {k: npx(z["rgb_" + ("base" if k == "m" else k)], z["rgb_nomarker"]) for k in locs}
    px_yaw = npx(z["rgb_yaw"], z["rgb_base"])
    rows = []
    for cohort in ("gm", "control", "init"):
        for s in (0, 1, 2):
            a = agent("PTM", cohort, s)
            sig = C.sigma(a)
            mu = {k: mu_of(a, z["rgb_" + k], z["state_" + k]) for k in ("base", "nomarker", "yaw", *(l for l in locs if l != "m"))}
            mu["m"] = mu["base"]
            for loc, xy in locs.items():
                kl = C.gaussian_kl(mu[loc], mu["nomarker"], sig).numpy()
                ds = tcp_dstep(z["J_tcp"], mu[loc] - mu["nomarker"])[:, :2]
                push = z[f"push_{loc}"][:, :2] if loc in ("m2", "mlat") else z["push_m"][:, :2]
                to_push = (push - tcp) / np.linalg.norm(push - tcp, axis=1, keepdims=True)
                to_loc = (xy - tcp) / np.linalg.norm(xy - tcp, axis=1, keepdims=True)
                nd = np.linalg.norm(ds, axis=1) + 1e-12
                d_cube = np.linalg.norm(xy - cube, axis=1)
                for i in range(len(kl)):
                    rows.append(dict(bank=bank_name, cohort=cohort, seed=s, state=i, edit=f"presence_{loc}", KL=kl[i],
                                     npx=px[loc][i], d_cube=d_cube[i], d_tcp=np.linalg.norm(xy[i] - tcp[i]),
                                     cos_to_push=(ds[i] * to_push[i]).sum() / nd[i],
                                     cos_to_marker=(ds[i] * to_loc[i]).sum() / nd[i]))
            kl_y = C.gaussian_kl(mu["yaw"], mu["base"], sig).numpy()
            kl_mv = C.gaussian_kl(mu["m2"], mu["base"], sig).numpy()
            ds = tcp_dstep(z["J_tcp"], mu["m2"] - mu["base"])[:, :2]
            u = z["push_m2"][:, :2] - z["push_m"][:, :2]
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            cosmv = (ds * u).sum(1) / (np.linalg.norm(ds, axis=1) + 1e-12)
            for i in range(len(kl_y)):
                rows.append(dict(bank=bank_name, cohort=cohort, seed=s, state=i, edit="cube_yaw", KL=kl_y[i],
                                 npx=px_yaw[i]))
                rows.append(dict(bank=bank_name, cohort=cohort, seed=s, state=i, edit="marker_move", KL=kl_mv[i],
                                 npx=z["npx_m2"][i], d_cube=np.linalg.norm(z["m2"][i] - cube[i]),
                                 cos_to_push=cosmv[i]))
            print(f"{bank_name} {cohort} s{s} done", flush=True)
    return pd.DataFrame(rows)


def lr_rows():
    z = np.load(os.path.join(D.BANKS, "LR_worldedit.npz"))
    tcp = z["tcp"][:, :2]
    corner = np.broadcast_to(CORNER, tcp.shape)
    rows = []
    for cohort in ("gm", "control", "init"):
        for s in ((1, 2, 3) if cohort != "init" else (0, 1, 2)):
            a = agent("LR", cohort, s)
            sig = C.sigma(a)
            mu = {k: mu_of(a, z["rgb_" + k], z["state_" + k]) for k in ("g_d", "g2_d", "g_d2", "corner_d")}
            for edit, (kb, ke, p0, p1) in {"goal_move": ("g_d", "g2_d", z["g"], z["g2"]),
                                           "distractor_move": ("g_d", "g_d2", z["d"], z["d2"]),
                                           "goal_to_corner": ("g_d", "corner_d", z["g"], corner)}.items():
                kl = C.gaussian_kl(mu[kb], mu[ke], sig).numpy()
                ds = tcp_dstep(z["J_tcp"], mu[ke] - mu[kb])[:, :2]
                u = (p1 - p0) / np.linalg.norm(p1 - p0, axis=1, keepdims=True)
                cosv = (ds * u).sum(1) / (np.linalg.norm(ds, axis=1) + 1e-12)
                # how close the edited object sits to the GM's own reach path (start TCP -> training corner)
                path_d = np.minimum(seg_dist(p0, tcp, corner), seg_dist(p1, tcp, corner))
                tcp_d = np.minimum(np.linalg.norm(p0 - tcp, axis=1), np.linalg.norm(p1 - tcp, axis=1))
                for i in range(len(kl)):
                    rows.append(dict(bank="LR_worldedit", cohort=cohort, seed=s, state=i, edit=edit, KL=kl[i],
                                     npx=z[f"npx_{edit}"][i], d_path=path_d[i], d_tcp=tcp_d[i], cos_edit=cosv[i]))
            print(f"LR {cohort} s{s} done", flush=True)
    return pd.DataFrame(rows)


def spearman_by_seed(df, x, y="KL"):
    out = []
    for (b, c, s), g in df.groupby(["bank", "cohort", "seed"]):
        if g[x].notna().sum() > 10 and g[y].std() > 0:
            out.append(dict(bank=b, cohort=c, seed=s, rho=stats.spearmanr(g[x], g[y])[0]))
    return pd.DataFrame(out)


def summarise(ptm, lr, out):
    L = ["# Q2 deep-dive: where the GM reacts to the goal marker", "",
         "Mean [95% t-CI] over seeds of per-seed state means (KL) or medians where stated. KL in each policy's own sigma.", ""]
    L += ["## PTM: marker presence response by marker location (KL(marker at L) vs no marker)", "",
          "m = conflict marker (5 cm from the cube, on its push ray to the centre); m2 = moved marker (11-16 cm from the "
          "cube, off-ray); centre = GM training location (16 cm from the cube, on-ray). KL/px = per-state KL / changed px.", "",
          "| bank | edit | cohort | KL mean | KL median | KL per px (x1e3, median) | d_cube (cm) | cos(dstep, to push pt) | cos(dstep, to marker) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (b, e, c), g in ptm.groupby(["bank", "edit", "cohort"], sort=False):
        ps = g.groupby("seed").agg(KL=("KL", "mean"), KLmed=("KL", "median"),
                                   kpx=("KL", lambda v: np.median(v / np.maximum(g.loc[v.index, "npx"], 1))),
                                   dc=("d_cube", "median"), cp=("cos_to_push", "mean"), cm=("cos_to_marker", "mean"))
        L.append(f"| {b} | {e} | {c} | {D.fmt_ci(ps.KL, 2)} | {D.fmt_ci(ps.KLmed, 2)} | {D.fmt_ci(1e3 * ps.kpx, 1)} | "
                 f"{100 * ps.dc.mean():.1f} | {D.fmt_ci(ps.cp)} | {D.fmt_ci(ps.cm)} |")
    L += ["", "### PTM within-state contrasts (per seed: mean over states of log(KL_a / KL_b); exp shown)", "",
          "| bank | contrast | cohort | ratio (geo-mean) |", "|---|---|---|---|"]
    piv = ptm[ptm.edit.str.startswith("presence")].pivot_table(index=["bank", "cohort", "seed", "state"],
                                                               columns="edit", values="KL").reset_index()
    for a_, b_ in (("presence_m", "presence_m2"), ("presence_m", "presence_centre"), ("presence_centre", "presence_m2")):
        for (b, c), g in piv.groupby(["bank", "cohort"], sort=False):
            if c == "init":
                continue
            ps = g.groupby("seed").apply(lambda h: np.mean(np.log((h[a_] + 1e-9) / (h[b_] + 1e-9))))
            m, lo, hi = D.ci95(ps)
            L.append(f"| {b} | {a_} / {b_} | {c} | {np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}] |")
    L += ["", "### PTM: does the presence response at m2 grow as m2 nears the cube? Spearman rho(KL, d_cube) per seed", "",
          "| bank | cohort | rho mean [95% t-CI] | per seed |", "|---|---|---|---|"]
    sub = ptm[ptm.edit == "presence_m2"]
    for (b, c), g in spearman_by_seed(sub, "d_cube").groupby(["bank", "cohort"], sort=False):
        L.append(f"| {b} | {c} | {D.fmt_ci(g.rho)} | {' / '.join(f'{v:.2f}' for v in g.sort_values('seed').rho)} |")
    L += ["", "### PTM goal edit vs nuisance (cube yaw): KL ratio and per-pixel ratio, per seed", "",
          "| bank | cohort | KL(marker_move) / KL(cube_yaw) | per-px ratio | presence_m / cube_yaw per px |", "|---|---|---|---|---|"]
    for (b, c), g in ptm.groupby(["bank", "cohort"], sort=False):
        if c == "init":
            continue
        r1, r2, r3 = [], [], []
        for s, h in g.groupby("seed"):
            mv, yw, pm = (h[h.edit == e] for e in ("marker_move", "cube_yaw", "presence_m"))
            r1.append(mv.KL.mean() / yw.KL.mean())
            r2.append((mv.KL.sum() / mv.npx.sum()) / (yw.KL.sum() / yw.npx.sum()))
            r3.append((pm.KL.sum() / pm.npx.sum()) / (yw.KL.sum() / yw.npx.sum()))
        L.append(f"| {b} | {c} | {D.fmt_ci(r1, 2)} | {D.fmt_ci(r2, 2)} | {D.fmt_ci(r3, 2)} |")
    L += ["", "## LR: goal edit vs matched-size distractor edit (native policies)", "",
          "| edit | cohort | KL mean | KL median | KL per px (x1e3) | cos(dstep, edit dir) |", "|---|---|---|---|---|---|"]
    for (e, c), g in lr.groupby(["edit", "cohort"], sort=False):
        ps = g.groupby("seed").agg(KL=("KL", "mean"), KLmed=("KL", "median"), kpx=("KL", "sum"), n=("npx", "sum"),
                                   cs=("cos_edit", "mean"))
        L.append(f"| {e} | {c} | {D.fmt_ci(ps.KL, 2)} | {D.fmt_ci(ps.KLmed, 2)} | {D.fmt_ci(1e3 * ps.kpx / ps.n, 2)} | "
                 f"{D.fmt_ci(ps.cs)} |")
    L += ["", "| cohort | KL(goal_move) / KL(distractor_move) per seed | per-px ratio |", "|---|---|---|"]
    for c, g in lr.groupby("cohort", sort=False):
        if c == "init":
            continue
        r1, r2 = [], []
        for s, h in g.groupby("seed"):
            gm_, ds_ = h[h.edit == "goal_move"], h[h.edit == "distractor_move"]
            r1.append(gm_.KL.mean() / ds_.KL.mean())
            r2.append((gm_.KL.sum() / gm_.npx.sum()) / (ds_.KL.sum() / ds_.npx.sum()))
        L.append(f"| {c} | {D.fmt_ci(r1, 2)} ({' / '.join(f'{v:.2f}' for v in r1)}) | {D.fmt_ci(r2, 2)} |")
    L += ["", "### LR: does the GM react more when the moved marker lies near its own reach path (start TCP -> corner)?",
          "Spearman rho(KL, distance to path) and rho(KL, distance to TCP) per seed", "",
          "| edit | cohort | rho(KL, d_path) | rho(KL, d_tcp) |", "|---|---|---|---|"]
    lr2 = lr.assign(bank=lr.edit)
    for (e, c), g in lr2.groupby(["edit", "cohort"], sort=False):
        if c == "init":
            continue
        rp = spearman_by_seed(g, "d_path").rho
        rt = spearman_by_seed(g, "d_tcp").rho
        L.append(f"| {e} | {c} | {D.fmt_ci(rp)} | {D.fmt_ci(rt)} |")
    with open(os.path.join(out, "q2_engagement_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--banks", nargs="+", default=["PTM_worldedit", "PTM_hover"])
    p.add_argument("--reuse", action="store_true", help="re-summarise the saved per-state CSVs")
    a = p.parse_args()
    torch.set_num_threads(8)
    out = D.ensure_out()
    fp, fl = os.path.join(out, "q2_ptm_presence_per_state.csv"), os.path.join(out, "q2_lr_edits_per_state.csv")
    if a.reuse:
        ptm, lr = pd.read_csv(fp), pd.read_csv(fl)
    else:
        ptm = pd.concat([ptm_rows(b) for b in a.banks])
        ptm.to_csv(fp, index=False)
        lr = lr_rows()
        lr.to_csv(fl, index=False)
    summarise(ptm, lr, out)


if __name__ == "__main__":
    main()
