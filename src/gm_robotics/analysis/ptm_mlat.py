"""PTM mlat test: presence response to a disc on the cube->centre path (m) vs off it at matched cube distance (mlat) vs far (m2)."""
import argparse
import glob
import os

import numpy as np

LOCS = ("m", "mlat", "m2", "centre")
COLOURS = {"white": (1.0, 1.0, 1.0, 1.0), "green": (0.10, 0.75, 0.15, 1.0)}
WORLDEDIT_SEED = 1300


def render(a):
    """Re-render one colour at every LOC (+ nomarker for white) over an existing PTM world-edit bank; one process per
    colour because the marker material is fixed when the scene is built."""
    import gymnasium as gym
    import torch
    from gm_robotics.analysis import envs_forced as E
    from gm_robotics.analysis.build_banks import ptm_mlat
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
    E.PTM_ENV.MARKER_RGBA = np.array(COLOURS[a.colour])
    z = np.load(os.path.join(a.banks, a.bank + ".npz"))
    n = len(z["m"]) if a.n <= 0 else a.n
    mlat, sign = ptm_mlat(z["cube"][:, :2], z["m"], WORLDEDIT_SEED)
    env = gym.make("PushToMarkerForced-v1", variant="out_marker", start_split="test", num_envs=1, obs_mode="rgb",
                   sim_backend="physx_cpu", control_mode="pd_joint_delta_pos", reconfiguration_freq=None)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    qpos, cube = z["qpos_forced"][:n], z["cube"][:n]
    xy = dict(m=z["m"], mlat=mlat, m2=z["m2"], centre=z["centre"])
    states = {"nomarker": np.full((n, 2), np.nan, np.float32)} if a.colour == "white" else {}
    states.update({loc: xy[loc][:n].astype(np.float32) for loc in LOCS})
    out = dict(mlat=mlat[:n], mlat_sign=sign[:n], centre_xy=E.PTM_C,
               **{f"push_{loc}": E.push_point(cube[:, :2], xy[loc][:n])[:, :2] for loc in LOCS})
    for name, mxy in states.items():
        goal = np.where(np.isfinite(mxy), mxy, z["m"][:n]).astype(np.float32)
        acc = {}
        for i in range(n):
            E.force_ptm(env, qpos[i:i + 1], mxy[i:i + 1], goal[i:i + 1], cube[i:i + 1])
            with torch.no_grad():
                r = E.render_obs(env)
            for k, v in r.items():
                acc.setdefault(k, []).append(v[:1])
        acc = {k: np.concatenate(v) for k, v in acc.items()}
        out[f"rgb_{name}"], out[f"state_{name}"], out[f"qpos_{name}"] = acc["rgb"], acc["state"], acc["qpos"]
        print(f"{a.bank} {a.colour} {name}: {n} scenes", flush=True)
    os.makedirs(a.render_dir, exist_ok=True)
    np.savez(os.path.join(a.render_dir, f"{a.bank}_{a.colour}.npz"), **out)


def npx(x, y):
    return (np.abs(x.astype(np.int16) - y.astype(np.int16)).max(-1) > 0).sum((1, 2))


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def analyse(a):
    """Per state: presence KL vs no disc (each policy's own sigma), the mean-action displacement it corresponds to, the
    commanded TCP step change, and its direction relative to the push point / the disc."""
    import pandas as pd
    import torch
    from gm_robotics.analysis import depth_common as D
    C = D.load_common()
    torch.set_num_threads(a.threads)
    rows, agent_rows = [], []
    for bank in a.bank_list:
        z = np.load(os.path.join(a.banks, bank + ".npz"))
        R = {col: np.load(os.path.join(a.render_dir, f"{bank}_{col}.npz")) for col in COLOURS}
        n = len(R["white"]["rgb_nomarker"])
        assert all(np.array_equal(R[c][f"qpos_{l}"], R["white"]["qpos_nomarker"]) for c in R for l in LOCS)
        tcp, cube = z["tcp"][:n, :2], z["cube"][:n, :2]
        J = torch.as_tensor(z["J_tcp"][:n])
        xy = dict(m=z["m"][:n], mlat=R["white"]["mlat"], m2=z["m2"][:n], centre=z["centre"][:n])
        push = {loc: R["white"][f"push_{loc}"] for loc in LOCS}
        centre_xy = R["white"]["centre_xy"]
        ray = unit(centre_xy - cube)
        geo = {}
        for loc in LOCS:
            v = xy[loc] - cube
            geo[loc] = dict(d_cube=np.linalg.norm(v, axis=1), d_centre=np.linalg.norm(xy[loc] - centre_xy, axis=1),
                            d_tcp=np.linalg.norm(xy[loc] - tcp, axis=1),
                            off_ray_deg=np.degrees(np.arccos(np.clip((unit(v) * ray).sum(1), -1, 1))))
        rgb0, st0 = R["white"]["rgb_nomarker"], R["white"]["state_nomarker"]
        fid = dict(nomarker=npx(rgb0, z["rgb_nomarker"][:n]), m=npx(R["white"]["rgb_m"], z["rgb_base"][:n]))
        print(f"{bank}: re-render vs bank changed px, nomarker median {np.median(fid['nomarker']):.0f} max "
              f"{fid['nomarker'].max()}; white m median {np.median(fid['m']):.0f}", flush=True)
        for cohort in ("gm", "control", "init"):
            for s in a.seeds:
                ag = C.build_agent(None, init_seed=s) if cohort == "init" else C.build_agent(C.ckpt_path("PTM", cohort, s))
                sig = C.sigma(ag).detach()
                with torch.no_grad():
                    f = lambda rgb, st: torch.cat([C.taps(ag, torch.as_tensor(rgb[sl]), torch.as_tensor(st[sl]))["mu"]
                                                   for sl in C.batched(len(rgb), 256)])
                    mu0 = f(rgb0, st0)
                    step0 = torch.einsum("bkj,bj->bk", J, C.JOINT_DELTA_SCALE * mu0.clamp(-1, 1)[:, :7]).numpy()
                    agent_rows.append(dict(bank=bank, cohort=cohort, seed=s, sigma_arm_mean=sig[:7].mean().item(),
                                           sigma_arm_geomean=sig[:7].log().mean().exp().item(),
                                           sigma_grip=sig[7].item(), absmu_arm_mean=mu0[:, :7].abs().mean().item(),
                                           frac_mu_clipped=(mu0[:, :7].abs() > 1).float().mean().item(),
                                           step0_mm_median=float(np.median(1e3 * np.linalg.norm(step0, axis=1)))))
                    for col in COLOURS:
                        for loc in LOCS:
                            rgb, st = R[col][f"rgb_{loc}"], R[col][f"state_{loc}"]
                            mu = f(rgb, st)
                            dmu = mu - mu0
                            kl = C.gaussian_kl(mu, mu0, sig).numpy()
                            dstep = torch.einsum("bkj,bj->bk", J, C.JOINT_DELTA_SCALE * dmu[:, :7]).numpy()
                            dstep_c = torch.einsum("bkj,bj->bk", J, C.JOINT_DELTA_SCALE
                                                   * (mu.clamp(-1, 1) - mu0.clamp(-1, 1))[:, :7]).numpy()
                            ds2 = dstep[:, :2]
                            nd = np.linalg.norm(ds2, axis=1) + 1e-12
                            px = npx(rgb, rgb0)
                            b0 = np.linalg.norm(step0, axis=1)
                            df = pd.DataFrame(dict(
                                bank=bank, colour=col, loc=loc, cohort=cohort, seed=s, state=np.arange(n), KL=kl,
                                mahal=np.sqrt(2 * kl), dmu_arm=dmu[:, :7].norm(dim=1).numpy(),
                                dmu_arm_clip=(mu.clamp(-1, 1) - mu0.clamp(-1, 1))[:, :7].norm(dim=1).numpy(),
                                mahal_clip=np.sqrt((((mu.clamp(-1, 1) - mu0.clamp(-1, 1)) / sig) ** 2).sum(1).numpy()),
                                dmu_arm_rel=(dmu[:, :7].norm(dim=1) / mu0[:, :7].norm(dim=1)).numpy(),
                                dtcp_mm=1e3 * np.linalg.norm(dstep, axis=1),
                                dtcp_clip_mm=1e3 * np.linalg.norm(dstep_c, axis=1), step0_mm=1e3 * b0,
                                dtcp_clip_rel=np.linalg.norm(dstep_c, axis=1) / (b0 + 1e-9), npx=px,
                                cos_to_push=(ds2 * unit(push[loc] - tcp)).sum(1) / nd,
                                cos_to_centrepush=(ds2 * unit(push["centre"] - tcp)).sum(1) / nd,
                                cos_to_disc=(ds2 * unit(xy[loc] - tcp)).sum(1) / nd,
                                mlat_sign=R["white"]["mlat_sign"], **geo[loc]))
                            rows.append(df)
                print(f"{bank} {cohort} s{s} done", flush=True)
    df = pd.concat(rows)
    os.makedirs(a.raw_dir, exist_ok=True)
    for (c, s), g in df.groupby(["cohort", "seed"]):
        g.to_csv(os.path.join(a.raw_dir, f"ptm_mlat_{c}_s{s}.csv"), index=False)
    pd.DataFrame(agent_rows).to_csv(os.path.join(a.raw_dir, "ptm_mlat_agent_sigma.csv"), index=False)


def summarise(a):
    import pandas as pd
    from gm_robotics.analysis import depth_common as D
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.raw_dir, "ptm_mlat_*_s*.csv")))])
    ag = pd.read_csv(os.path.join(a.raw_dir, "ptm_mlat_agent_sigma.csv"))
    ci = D.fmt_ci
    L = ["## Generated tables", "",
         "Mean [95% t-CI] over 3 seeds of per-seed state means (512 states). KL vs the no-disc render, in each policy's own "
         "sigma. dTCP = commanded TCP step change (J * 0.1 rad * dmu, actions clipped to [-1, 1]), mm; rel = dTCP / |no-disc "
         "commanded TCP step|.", "",
         "### Geometry (per bank, medians over states)", "", "| loc | d_cube cm | d_centre cm | off-ray deg | d_tcp cm (start) |",
         "|---|---|---|---|---|"]
    g0 = df[(df.cohort == "gm") & (df.seed == df.seed.min()) & (df.colour == "white") & (df.bank == "PTM_worldedit")]
    for loc, g in g0.groupby("loc", sort=False):
        L.append(f"| {loc} | {100 * g.d_cube.median():.1f} | {100 * g.d_centre.median():.1f} | {g.off_ray_deg.median():.0f} | "
                 f"{100 * g.d_tcp.median():.1f} |")
    L += ["", "### Action sigma and step size per cohort (start bank; sigma is state-independent)", "",
          "| cohort | sigma arm (mean of 7) | sigma arm geo-mean | |mu| arm | frac arm mu clipped | no-disc TCP step mm (median) |",
          "|---|---|---|---|---|---|"]
    for c, g in ag[ag.bank == "PTM_worldedit"].groupby("cohort", sort=False):
        L.append(f"| {c} | {ci(g.sigma_arm_mean)} | {ci(g.sigma_arm_geomean)} | {ci(g.absmu_arm_mean)} | "
                 f"{ci(g.frac_mu_clipped)} | {ci(g.step0_mm_median, 1)} |")
    # A disc fully hidden by the arm or outside the frame gives KL = 0 exactly; such states are uninformative and make
    # log-ratio contrasts explode, so tables are on states where the disc is visible (white render, >= min_px changed px).
    vis = df[(df.colour == "white") & (df.cohort == "init") & (df.seed == df.seed.min())].pivot_table(
        index=["bank", "state"], columns="loc", values="npx") >= a.min_px
    L += ["", f"### Disc visibility (white render, >= {a.min_px} changed px), fraction of 512 states", "",
          "| bank | " + " | ".join(LOCS) + " | all four |", "|---|" + "---|" * (len(LOCS) + 1)]
    for b, v in vis.groupby(level="bank"):
        L.append(f"| {b} | " + " | ".join(f"{v[l].mean():.3f}" for l in LOCS) + f" | {v[list(LOCS)].all(axis=1).mean():.3f} |")
    allvis = vis[list(LOCS)].all(axis=1).rename("allvis").reset_index()
    df = df.merge(allvis, on=["bank", "state"])
    for title, sub in (("all 512 states (comparable to Run B / deep-dive numbers)", df),
                       ("states where all four discs are visible", df[df.allvis])):
        L += ["", f"### Presence response by location and colour: {title}", "",
              "| bank | colour | loc | cohort | KL mean | KL median | sqrt(2KL) median | sqrt(2KL) clipped median | |dmu_arm| | "
              "|dmu_arm| clipped | dTCP mm | dTCP rel (median) | cos(dstep, push pt of loc) | cos(dstep, disc) | changed px |",
              "|" + "---|" * 15]
        for (b, col, loc, c), g in sub.groupby(["bank", "colour", "loc", "cohort"], sort=False):
            ps = g.groupby("seed").agg(KL=("KL", "mean"), KLmed=("KL", "median"), mh=("mahal", "median"),
                                       mhc=("mahal_clip", "median"), dm=("dmu_arm", "mean"), dmc=("dmu_arm_clip", "mean"),
                                       dt=("dtcp_clip_mm", "mean"), dr=("dtcp_clip_rel", "median"),
                                       cp=("cos_to_push", "mean"), cd=("cos_to_disc", "mean"), px=("npx", "median"))
            L.append(f"| {b} | {col} | {loc} | {c} | {ci(ps.KL, 2)} | {ci(ps.KLmed, 2)} | {ci(ps.mh, 2)} | {ci(ps.mhc, 2)} | "
                     f"{ci(ps.dm)} | {ci(ps.dmc)} | {ci(ps.dt, 2)} | {ci(ps.dr)} | {ci(ps.cp)} | {ci(ps.cd)} | {ps.px.mean():.0f} |")
    L += ["", "### Paired within-state contrasts, states where both discs are visible", "",
          "geo = per seed exp(mean over states of log(KL_a / KL_b)); RoM = per seed mean KL_a / mean KL_b; dTCP diff = per "
          "seed mean of (dTCP_a - dTCP_b) mm; GM vs control = Welch 95% CI on the per-seed log geo ratio, exp shown.", "",
          "| bank | contrast | cohort | n states/seed | geo ratio | geo per seed | RoM | dTCP diff mm | GM / control (geo) |",
          "|---|---|---|---|---|---|---|---|---|"]
    piv = df.pivot_table(index=["bank", "cohort", "seed", "state"], columns=["colour", "loc"],
                         values=["KL", "dtcp_clip_mm", "npx"])
    contrasts = [("on-path vs off-path, matched distance: m / mlat (white)", ("white", "m"), ("white", "mlat")),
                 ("on-path vs off-path, green disc: m / mlat", ("green", "m"), ("green", "mlat")),
                 ("near vs far, both off-path: mlat / m2 (white)", ("white", "mlat"), ("white", "m2")),
                 ("near vs far, both off-path: mlat / m2 (green)", ("green", "mlat"), ("green", "m2")),
                 ("goal marker vs green disc at mlat: white / green", ("white", "mlat"), ("green", "mlat")),
                 ("goal marker vs green disc at m: white / green", ("white", "m"), ("green", "m")),
                 ("training spot vs far off-path: centre / m2 (white)", ("white", "centre"), ("white", "m2")),
                 ("training spot vs far off-path: centre / m2 (green)", ("green", "centre"), ("green", "m2"))]
    out_rows = []
    for name, x, y in contrasts:
        for b in piv.index.unique("bank"):
            logs = {}
            for c in ("gm", "control", "init"):
                g = piv.loc[(b, c)]
                lr, rom, dd, ns = [], [], [], []
                for s, h in g.groupby(level="seed"):
                    ok = (h[("npx", "white", x[1])] >= a.min_px) & (h[("npx", "white", y[1])] >= a.min_px)
                    h = h[ok]
                    kx, ky = h[("KL",) + x].values, h[("KL",) + y].values
                    lr.append(np.mean(np.log((kx + 1e-12) / (ky + 1e-12))))
                    rom.append(kx.mean() / ky.mean())
                    dd.append(np.mean(h[("dtcp_clip_mm",) + x].values - h[("dtcp_clip_mm",) + y].values))
                    ns.append(int(ok.sum()))
                    out_rows.append(dict(bank=b, contrast=name, cohort=c, seed=s, n_states=ns[-1], log_KL_ratio=lr[-1],
                                         ratio_of_means=rom[-1], dtcp_diff_mm=dd[-1]))
                logs[c] = lr
                m, lo, hi = D.ci95(lr)
                vs = ""
                if c == "control":
                    d, dlo, dhi = D.welch(logs["gm"], logs["control"])
                    vs = f"{np.exp(d):.2f} [{np.exp(dlo):.2f}, {np.exp(dhi):.2f}]"
                L.append(f"| {b} | {name} | {c} | {np.mean(ns):.0f} | {np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}] | "
                         f"{' / '.join(f'{np.exp(v):.2f}' for v in lr)} | {ci(rom, 2)} | {ci(dd, 2)} | {vs} |")
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(os.path.join(a.out, "ptm_mlat_contrasts_per_seed.csv"), index=False)
    L += ["", "### mlat side check (white mlat KL by rotation sign, GM / control, start + hover pooled per seed)", "",
          "| cohort | KL sign +1 | KL sign -1 |", "|---|---|---|"]
    sub = df[(df.colour == "white") & (df["loc"] == "mlat")]
    for c, g in sub.groupby("cohort", sort=False):
        p_ = g[g.mlat_sign > 0].groupby("seed").KL.mean()
        n_ = g[g.mlat_sign < 0].groupby("seed").KL.mean()
        L.append(f"| {c} | {ci(p_, 2)} | {ci(n_, 2)} |")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "tables_generated.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("render", "analyse", "summarise"), required=True)
    p.add_argument("--bank", default="PTM_worldedit", help="render stage: one bank")
    p.add_argument("--bank-list", nargs="+", default=["PTM_worldedit", "PTM_hover"])
    p.add_argument("--colour", choices=sorted(COLOURS), default="white")
    p.add_argument("--n", type=int, default=0, help="scenes to render (0 = all)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--banks", default=os.environ.get("PTM_BANKS", "banks"))
    p.add_argument("--render-dir", default="renders")
    p.add_argument("--raw-dir", default="results/analysis/ptm_mlat/raw")
    p.add_argument("--min-px", type=int, default=20, help="changed px for a disc to count as visible")
    p.add_argument("--out", default="results/analysis/ptm_mlat")
    a = p.parse_args()
    {"render": render, "analyse": analyse, "summarise": summarise}[a.stage](a)


if __name__ == "__main__":
    main()
