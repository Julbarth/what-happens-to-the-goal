"""PTM off-path distance sweep: presence response to a disc 75 deg off the cube->centre ray at 3-11 cm from the cube."""
import argparse
import glob
import os

import numpy as np

DISTS_CM = (3.0, 5.2, 7.0, 9.0, 11.0)
SIDES = (1, -1)
OFF_DEG = 75.0
COLOURS = {"white": (1.0, 1.0, 1.0, 1.0), "green": (0.10, 0.75, 0.15, 1.0)}


def sweep_xy(cube_xy, centre_xy, d, side):
    """Disc at distance d from the cube, rotated side*OFF_DEG off the cube->centre ray (same rotation as ptm_mlat)."""
    ray = centre_xy - cube_xy
    ray = ray / np.linalg.norm(ray, axis=1, keepdims=True)
    th = np.deg2rad(OFF_DEG) * side
    c, s = np.cos(th), np.sin(th)
    rot = np.stack([c * ray[:, 0] - s * ray[:, 1], s * ray[:, 0] + c * ray[:, 1]], 1)
    return (cube_xy + d * rot).astype(np.float32)


def key(side, d_cm):
    return f"{'p' if side > 0 else 'n'}{int(round(10 * d_cm)):03d}"


def render(a):
    """One process per (bank, colour, side): the marker material is fixed when the scene is built."""
    import gymnasium as gym
    import torch
    from gm_robotics.analysis import envs_forced as E
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
    E.PTM_ENV.MARKER_RGBA = np.array(COLOURS[a.colour])
    z = np.load(os.path.join(a.banks, a.bank + ".npz"))
    n = len(z["m"]) if a.n <= 0 else a.n
    env = gym.make("PushToMarkerForced-v1", variant="out_marker", start_split="test", num_envs=1, obs_mode="rgb",
                   sim_backend="physx_cpu", control_mode="pd_joint_delta_pos", reconfiguration_freq=None)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    qpos, cube = z["qpos_forced"][:n], z["cube"][:n]
    states = {}
    if a.colour == "white" and a.side > 0:
        states["nomarker"] = np.full((n, 2), np.nan, np.float32)
    out = {}
    for d in DISTS_CM:
        k = key(a.side, d)
        states[k] = sweep_xy(cube[:, :2], E.PTM_C, d / 100, a.side)
        out[f"xy_{k}"] = states[k]
        out[f"push_{k}"] = E.push_point(cube[:, :2], states[k])[:, :2]
    out["push_centre"] = E.push_point(cube[:, :2], np.tile(E.PTM_C, (n, 1)))[:, :2]
    out["centre_xy"] = E.PTM_C
    for name, mxy in states.items():
        # the scored goal is irrelevant to a single forward pass; keep the bank's m so the env state matches ptm_mlat
        goal = z["m"][:n].astype(np.float32)
        acc = {}
        for i in range(n):
            E.force_ptm(env, qpos[i:i + 1], mxy[i:i + 1], goal[i:i + 1], cube[i:i + 1])
            with torch.no_grad():
                r = E.render_obs(env)
            for kk, v in r.items():
                acc.setdefault(kk, []).append(v[:1])
        acc = {kk: np.concatenate(v) for kk, v in acc.items()}
        out[f"rgb_{name}"], out[f"state_{name}"], out[f"qpos_{name}"] = acc["rgb"], acc["state"], acc["qpos"]
        print(f"{a.bank} {a.colour} side {a.side} {name}: {n} scenes", flush=True)
    os.makedirs(a.render_dir, exist_ok=True)
    np.savez(os.path.join(a.render_dir, f"{a.bank}_{a.colour}_{'p' if a.side > 0 else 'n'}.npz"), **out)


def npx_mask(x, y):
    return np.abs(x.astype(np.int16) - y.astype(np.int16)).max(-1) > 0


def cube_mask(rgb):
    """Blue cube pixels (cube RGBA 0.08, 0.22, 0.90; table is orange wood, arm grey/black)."""
    x = rgb.astype(np.int16)
    return (x[..., 2] > x[..., 0] + 40) & (x[..., 2] > x[..., 1] + 25)


def px_gap(disc, cube):
    """Per image: min pixel distance from any disc pixel to the cube mask, minus 1 (0 = a disc pixel 4-adjacent to the
    cube silhouette, 0.41 = diagonal to it, i.e. touching in the image); NaN when either mask is empty."""
    from scipy.ndimage import distance_transform_edt
    out = np.full(len(disc), np.nan)
    for i in range(len(disc)):
        if disc[i].any() and cube[i].any():
            out[i] = distance_transform_edt(~cube[i])[disc[i]].min() - 1.0
    return out


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def analyse(a):
    import pandas as pd
    import torch
    from gm_robotics.analysis import depth_common as D
    C = D.load_common()
    torch.set_num_threads(a.threads)
    rows, img_rows = [], []
    for bank in a.bank_list:
        z = np.load(os.path.join(a.banks, bank + ".npz"))
        R = {(c, sd): np.load(os.path.join(a.render_dir, f"{bank}_{c}_{'p' if sd > 0 else 'n'}.npz"))
             for c in COLOURS for sd in SIDES}
        base = R[("white", 1)]
        rgb0, st0 = base["rgb_nomarker"], base["state_nomarker"]
        n = len(rgb0)
        q0 = base["qpos_nomarker"]
        for (c, sd), r in R.items():
            for d in DISTS_CM:
                assert np.array_equal(r[f"qpos_{key(sd, d)}"], q0)
        tcp, cube = z["tcp"][:n, :2], z["cube"][:n, :2]
        centre_xy = base["centre_xy"]
        J = torch.as_tensor(z["J_tcp"][:n])
        cm0 = cube_mask(rgb0)
        geo = {}
        for sd in SIDES:
            for d in DISTS_CM:
                k = key(sd, d)
                xy = R[("white", sd)][f"xy_{k}"]
                disc = npx_mask(R[("white", sd)][f"rgb_{k}"], rgb0)
                gap = px_gap(disc, cm0)
                geo[k] = dict(d_cube=np.linalg.norm(xy - cube, axis=1), d_centre=np.linalg.norm(xy - centre_xy, axis=1),
                              d_tcp=np.linalg.norm(xy - tcp, axis=1), npx_white=disc.sum((1, 2)), px_gap=gap,
                              touch=(gap <= 0.5).astype(float), cube_px=cm0.sum((1, 2)))
                img_rows.append(pd.DataFrame(dict(bank=bank, side=sd, d_cm=d, state=np.arange(n), **geo[k])))
        print(f"{bank}: cube px median {np.median(cm0.sum((1, 2)))}, min {cm0.sum((1, 2)).min()}", flush=True)
        for cohort in ("gm", "control", "init"):
            for s in a.seeds:
                ag = C.build_agent(None, init_seed=s) if cohort == "init" else C.build_agent(C.ckpt_path("PTM", cohort, s))
                sig = C.sigma(ag).detach()
                with torch.no_grad():
                    f = lambda rgb, st: torch.cat([C.taps(ag, torch.as_tensor(rgb[sl]), torch.as_tensor(st[sl]))["mu"]
                                                   for sl in C.batched(len(rgb), 256)])
                    mu0 = f(rgb0, st0)
                    step0 = torch.einsum("bkj,bj->bk", J, C.JOINT_DELTA_SCALE * mu0.clamp(-1, 1)[:, :7]).numpy()
                    b0 = np.linalg.norm(step0, axis=1)
                    for (col, sd), r in R.items():
                        for d in DISTS_CM:
                            k = key(sd, d)
                            rgb, st = r[f"rgb_{k}"], r[f"state_{k}"]
                            mu = f(rgb, st)
                            kl = C.gaussian_kl(mu, mu0, sig).numpy()
                            dstep_c = torch.einsum("bkj,bj->bk", J, C.JOINT_DELTA_SCALE
                                                   * (mu.clamp(-1, 1) - mu0.clamp(-1, 1))[:, :7]).numpy()
                            ds2 = dstep_c[:, :2]
                            nd = np.linalg.norm(ds2, axis=1) + 1e-12
                            xy = r[f"xy_{k}"]
                            rows.append(pd.DataFrame(dict(
                                bank=bank, colour=col, side=sd, d_cm=d, cohort=cohort, seed=s, state=np.arange(n), KL=kl,
                                dtcp_clip_mm=1e3 * np.linalg.norm(dstep_c, axis=1), step0_mm=1e3 * b0,
                                dtcp_clip_rel=np.linalg.norm(dstep_c, axis=1) / (b0 + 1e-9),
                                npx=npx_mask(rgb, rgb0).sum((1, 2)),
                                cos_to_push=(ds2 * unit(r[f"push_{k}"] - tcp)).sum(1) / nd,
                                cos_to_centrepush=(ds2 * unit(r["push_centre"] - tcp)).sum(1) / nd,
                                cos_to_disc=(ds2 * unit(xy - tcp)).sum(1) / nd,
                                **{g: geo[k][g] for g in ("d_cube", "d_centre", "d_tcp", "px_gap", "touch",
                                                         "npx_white")})))
                print(f"{bank} {cohort} s{s} done", flush=True)
    os.makedirs(a.raw_dir, exist_ok=True)
    pd.concat(img_rows).to_csv(os.path.join(a.raw_dir, "ptm_sweep_image_geometry.csv"), index=False)
    df = pd.concat(rows)
    for (c, s), g in df.groupby(["cohort", "seed"]):
        g.to_csv(os.path.join(a.raw_dir, f"ptm_sweep_{c}_s{s}.csv"), index=False)


COHORTS = ("gm", "control", "init")
COH_STYLE = {"gm": ("#0072B2", "GM"), "control": ("#D55E00", "control"), "init": ("#7f7f7f", "random init")}
BANK_NAME = {"PTM_worldedit": "start pose", "PTM_hover": "hover over cube"}
GAP_BINS = [-np.inf, 0.5, 3, 6, 10, np.inf]
GAP_LABELS = ["touch\n(<=0.5)", "0.5-3", "3-6", "6-10", ">10"]


def load(a):
    import pandas as pd
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.raw_dir, "ptm_sweep_*_s*.csv")))])
    img = pd.read_csv(os.path.join(a.raw_dir, "ptm_sweep_image_geometry.csv"))
    # a disc hidden by the arm or outside the frame gives KL = 0 and no information; keep (state, side) units whose
    # disc is visible at all five distances so every distance is compared on the same scenes
    # the arm can hide the cube itself (cube px 0 in a few start-pose scenes), which makes image touch undefined
    v = img.assign(vis=(img.npx_white >= a.min_px) & (img.cube_px >= a.min_px)).groupby(["bank", "side", "state"]).vis.all().rename("vis5").reset_index()
    img = img.merge(v, on=["bank", "side", "state"])
    df = df.merge(v, on=["bank", "side", "state"])
    return df, img


def per_seed(df, by, col, how="mean"):
    return df.groupby(by + ["seed"])[col].agg(how).reset_index()


def fe_fit(g):
    """OLS of y on [touch, d_cm] after removing each (state, side) unit's mean (within-scene fit)."""
    u = g.groupby(["state", "side"])
    X = np.stack([g.touch - u.touch.transform("mean"), g.d_cm - u.d_cm.transform("mean")], 1)
    out = {}
    for y in ("dtcp_clip_mm", "logKL"):
        yy = (g[y] - u[y].transform("mean")).values
        out[y] = np.linalg.lstsq(X, yy, rcond=None)[0]
    return out


def summarise(a):
    import pandas as pd
    from gm_robotics.analysis import depth_common as D
    ci = D.fmt_ci
    df, img = load(a)
    df["logKL"] = np.log(df.KL + 1e-6)
    L = ["## PTM off-path distance sweep: generated tables", "",
         f"Disc {OFF_DEG:.0f} deg off the cube->centre ray, both sides, at {', '.join(map(str, DISTS_CM))} cm from the cube "
         "centre; 512 configs per bank. Units = (scene, side) pairs whose disc is visible (>= "
         f"{a.min_px} changed px, white render) at all five distances and whose cube shows >= {a.min_px} px. Mean [95% t-CI] over 3 seeds of per-seed means. "
         "KL vs the no-disc render in each policy's own sigma. dTCP = commanded TCP step change, actions clipped to "
         "[-1, 1], mm. Image touch: some disc pixel 4- or 8-adjacent to the blue cube silhouette (px_gap <= 0.5).", "",
         "### Geometry and image contact (white render; touch and gap on visible units)", "",
         "| bank | d cm | side | frac visible | units vis at all 5 | d_centre cm | d_tcp cm | disc px | px gap median | "
         "frac touching |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (b, d, sd), g in img.groupby(["bank", "d_cm", "side"], sort=False):
        gv = g[g.vis5]
        L.append(f"| {b} | {d} | {sd:+d} | {(g.npx_white >= a.min_px).mean():.3f} | {g.vis5.sum()} | "
                 f"{100 * g.d_centre.median():.1f} | {100 * g.d_tcp.median():.1f} | {gv.npx_white.median():.0f} | "
                 f"{gv.px_gap.median():.2f} | {gv.touch.mean():.3f} |")
    dv = df[df.vis5]
    L += ["", "### Presence response by distance", "",
          "| bank | colour | d cm | cohort | frac touch | KL mean | KL median | dTCP mm | dTCP rel (median) | "
          "cos(dstep, push pt toward disc) | cos(dstep, centre push pt) | cos(dstep, disc) |", "|" + "---|" * 12]
    for (b, col, d, c), g in dv.groupby(["bank", "colour", "d_cm", "cohort"], sort=False):
        ps = g.groupby("seed").agg(KL=("KL", "mean"), KLm=("KL", "median"), dt=("dtcp_clip_mm", "mean"),
                                   dr=("dtcp_clip_rel", "median"), cp=("cos_to_push", "mean"),
                                   cc=("cos_to_centrepush", "mean"), cd=("cos_to_disc", "mean"), t=("touch", "mean"))
        L.append(f"| {b} | {col} | {d} | {c} | {ps.t.mean():.2f} | {ci(ps.KL, 2)} | {ci(ps.KLm, 2)} | {ci(ps.dt, 2)} | "
                 f"{ci(ps.dr)} | {ci(ps.cp)} | {ci(ps.cc)} | {ci(ps.cd)} |")
    piv = dv.pivot_table(index=["bank", "colour", "cohort", "seed", "side", "state"], columns="d_cm",
                         values=["logKL", "dtcp_clip_mm"])
    pairs = [(3.0, 11.0), (5.2, 11.0), (7.0, 11.0), (3.0, 5.2), (5.2, 7.0), (7.0, 9.0), (9.0, 11.0)]
    L += ["", "### Paired within-scene steps between distances", "",
          "KL ratio = per seed exp(mean log(KL_near / KL_far)); dTCP diff = per seed mean (dTCP_near - dTCP_far), mm. "
          "GM vs control: Welch 95% CI on the per-seed dTCP diff.", "",
          "| bank | colour | near vs far cm | cohort | KL ratio | dTCP diff mm | GM - control dTCP diff |",
          "|---|---|---|---|---|---|---|"]
    out_rows = []
    for (b, col), _ in dv.groupby(["bank", "colour"], sort=False):
        for x, y in pairs:
            dd = {}
            for c in COHORTS:
                h = piv.loc[(b, col, c)]
                lr = (h[("logKL", x)] - h[("logKL", y)]).groupby(level="seed").mean()
                dt = (h[("dtcp_clip_mm", x)] - h[("dtcp_clip_mm", y)]).groupby(level="seed").mean()
                dd[c] = dt.values
                for s in lr.index:
                    out_rows.append(dict(bank=b, colour=col, near_cm=x, far_cm=y, cohort=c, seed=s,
                                         log_KL_ratio=lr[s], dtcp_diff_mm=dt[s]))
                m, lo, hi = D.ci95(lr)
                vs = ""
                if c == "control":
                    w = D.welch(dd["gm"], dd["control"])
                    vs = f"{w[0]:.2f} [{w[1]:.2f}, {w[2]:.2f}]"
                L.append(f"| {b} | {col} | {x} vs {y} | {c} | {np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}] | "
                         f"{ci(dt, 2)} | {vs} |")
    L += ["", "### Touch vs no touch at the same distance (7 cm, the only distance with both)", "",
          "Not paired: touching and non-touching scenes differ in geometry (side relative to the camera, cube yaw).", "",
          "| bank | colour | cohort | n touch / no touch per seed | dTCP touch | dTCP no touch | diff | KL touch | KL no touch |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (b, col, c), g in dv[dv.d_cm == 7.0].groupby(["bank", "colour", "cohort"], sort=False):
        t = g[g.touch > 0].groupby("seed")
        nt = g[g.touch == 0].groupby("seed")
        diff = t.dtcp_clip_mm.mean() - nt.dtcp_clip_mm.mean()
        L.append(f"| {b} | {col} | {c} | {t.size().mean():.0f} / {nt.size().mean():.0f} | {ci(t.dtcp_clip_mm.mean(), 2)} | "
                 f"{ci(nt.dtcp_clip_mm.mean(), 2)} | {ci(diff, 2)} | {ci(t.KL.mean(), 2)} | {ci(nt.KL.mean(), 2)} |")
    L += ["", "### Within-scene decomposition: touch step vs proximity slope", "",
          "Per seed, OLS of y on [image touch, distance cm] after demeaning within each (scene, side) unit; touch is "
          "identified by scenes whose touch status differs between distances at a similar distance (mostly 5.2-7 cm). "
          "Proximity slope is per cm (negative = response falls with distance). Second block: slope on non-touching "
          "rows only (7-11 cm) and on touching rows only (3-7 cm), same demeaning.", "",
          "| bank | colour | cohort | dTCP touch step mm | dTCP slope mm/cm | logKL touch step | logKL slope /cm | "
          "dTCP slope no-touch rows | dTCP slope touch rows |", "|---|---|---|---|---|---|---|---|---|"]
    fe_rows = []
    for (b, col, c), g in dv.groupby(["bank", "colour", "cohort"], sort=False):
        r = []
        for s, h in g.groupby("seed"):
            f = fe_fit(h)
            sl = {}
            for k, sub in (("nt", h[h.touch == 0]), ("t", h[h.touch > 0])):
                u = sub.groupby(["state", "side"])
                ok = u.d_cm.transform("nunique") > 1
                sub = sub[ok]
                u = sub.groupby(["state", "side"])
                xx = (sub.d_cm - u.d_cm.transform("mean")).values
                yy = (sub.dtcp_clip_mm - u.dtcp_clip_mm.transform("mean")).values
                sl[k] = (xx * yy).sum() / (xx * xx).sum()
            r.append(dict(bank=b, colour=col, cohort=c, seed=s, dtcp_touch=f["dtcp_clip_mm"][0],
                          dtcp_slope=f["dtcp_clip_mm"][1], logKL_touch=f["logKL"][0], logKL_slope=f["logKL"][1],
                          dtcp_slope_notouch=sl["nt"], dtcp_slope_touch=sl["t"]))
        r = pd.DataFrame(r)
        fe_rows.append(r)
        L.append(f"| {b} | {col} | {c} | {ci(r.dtcp_touch, 2)} | {ci(r.dtcp_slope, 2)} | {ci(r.logKL_touch, 2)} | "
                 f"{ci(r.logKL_slope, 3)} | {ci(r.dtcp_slope_notouch, 2)} | {ci(r.dtcp_slope_touch, 2)} |")
    L += ["", "### Side check (white, dTCP mm by side, all distances pooled)", "", "| bank | cohort | side +1 | side -1 |",
          "|---|---|---|---|"]
    for (b, c), g in dv[dv.colour == "white"].groupby(["bank", "cohort"], sort=False):
        L.append(f"| {b} | {c} | {ci(g[g.side > 0].groupby('seed').dtcp_clip_mm.mean(), 2)} | "
                 f"{ci(g[g.side < 0].groupby('seed').dtcp_clip_mm.mean(), 2)} |")
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(os.path.join(a.out, "ptm_sweep_contrasts_per_seed.csv"), index=False)
    pd.concat(fe_rows).to_csv(os.path.join(a.out, "ptm_sweep_touch_vs_slope_per_seed.csv"), index=False)
    open(os.path.join(a.out, "tables_generated.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plot_sweep(dv, img, a.out)
    plot_gap(dv, a.out)


def _ci_t(v):
    from scipy import stats
    v = np.asarray(v, float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def plot_sweep(dv, img, out):
    """KL, commanded TCP step change and its paired change vs 11 cm, against disc-cube distance, per cohort/colour/bank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    banks = [b for b in BANK_NAME if b in set(dv.bank)]
    far = max(DISTS_CM)
    fig, ax = plt.subplots(3, 2 * len(banks), figsize=(3.3 * 2 * len(banks), 9.0), sharex=True)
    rows = (("KL", "presence KL (log scale)"), ("dtcp_clip_mm", "TCP step change (mm)"),
            ("dd", f"paired TCP step change\nminus {far:.0f} cm (mm)"))
    for j, (b, col) in enumerate([(b, c) for b in banks for c in COLOURS]):
        g = dv[(dv.bank == b) & (dv.colour == col)]
        g = g.merge(g[g.d_cm == far][["cohort", "seed", "side", "state", "dtcp_clip_mm"]].rename(
            columns={"dtcp_clip_mm": "far"}), on=["cohort", "seed", "side", "state"])
        g = g.assign(dd=g.dtcp_clip_mm - g.far)
        tf = img[(img.bank == b) & img.vis5].groupby("d_cm").touch.mean()
        for i, (y, lab) in enumerate(rows):
            a_ = ax[i, j]
            # random-init KL is ~1e-9: off any useful log axis, and its step change is ~0 mm
            for c in (COHORTS if y != "KL" else ("gm", "control")):
                ps = g[g.cohort == c].groupby(["d_cm", "seed"])[y].mean().reset_index()
                # KL across seeds spans an order of magnitude: CI on the log of the per-seed mean, shown as a geo-mean
                v = np.log(ps[y]) if y == "KL" else ps[y]
                st = v.groupby(ps.d_cm).apply(_ci_t)
                xs = np.array(st.index)
                m = np.array([t[0] for t in st])
                h = np.array([t[1] for t in st])
                colr, name = COH_STYLE[c]
                if y == "KL":
                    a_.errorbar(xs, np.exp(m), yerr=[np.exp(m) - np.exp(m - h), np.exp(m + h) - np.exp(m)], color=colr,
                                marker="o", ms=4, capsize=3, lw=1.5, label=name)
                else:
                    a_.errorbar(xs, m, yerr=h, color=colr, marker="o", ms=4, capsize=3, lw=1.5, label=name)
                a_.scatter(ps.d_cm + {"gm": -0.15, "control": 0.15, "init": 0}[c], ps[y], color=colr, s=6, alpha=0.4)
            for x0, f in tf.items():
                if f >= 0.5:
                    a_.axvspan(x0 - 0.9, x0 + 0.9, color="0.85", alpha=0.5, lw=0)
            if y == "KL":
                a_.set_yscale("log")
                a_.set_title(f"{BANK_NAME[b]}, {col} disc", fontsize=11, pad=16)
                for x0, f in tf.items():
                    a_.text(x0, 1.0, f"{f:.0%}", transform=a_.get_xaxis_transform(), ha="center", va="bottom",
                            fontsize=7, color="0.35")
            if y == "dd":
                a_.axhline(0, color="0.5", lw=0.8)
            a_.set_xticks(DISTS_CM)
            a_.tick_params(labelsize=9)
            a_.grid(axis="y", alpha=0.3)
            a_.spines[["top", "right"]].set_visible(False)
            if j == 0:
                a_.set_ylabel(lab, fontsize=10)
            if i == len(rows) - 1:
                a_.set_xlabel("disc centre to cube centre (cm)", fontsize=10)
    ax[1, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("PTM off-path disc distance sweep (3 seeds, 95% t-CI; grey band = most scenes touch the cube in the "
                 "image, % = fraction touching)", fontsize=11)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"ptm_sweep_distance.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Caption: Presence response to a disc 75 deg off the cube->centre ray at 3-11 cm from the cube (start pose and "
          "hover banks; white = goal colour, green = non-goal colour). Top: KL between the policy with and without the "
          "disc, geometric mean over seeds (log scale; random init ~1e-9, not shown). Middle: change in the commanded TCP "
          "step (clipped actions, mm). Bottom: the same, minus its value for the same scene with the disc at 11 cm. "
          "Mean and 95% t-CI over 3 seeds; dots = seeds. Grey band: distances where most scenes show the disc touching "
          "the cube silhouette in the image.")


def plot_gap(dv, out):
    """TCP step change vs the disc-cube pixel gap in the image, all distances pooled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    d = dv.assign(gb=pd.cut(dv.px_gap, GAP_BINS, labels=False))
    banks = [b for b in BANK_NAME if b in set(dv.bank)]
    fig, ax = plt.subplots(1, 2 * len(banks), figsize=(3.3 * 2 * len(banks), 3.3), sharey=False)
    for j, (b, col) in enumerate([(b, c) for b in banks for c in COLOURS]):
        a_ = ax[j]
        for c in COHORTS:
            ps = d[(d.bank == b) & (d.colour == col) & (d.cohort == c)].groupby(["gb", "seed"]).dtcp_clip_mm.mean()
            st = ps.reset_index().groupby("gb").dtcp_clip_mm.apply(_ci_t)
            colr, name = COH_STYLE[c]
            a_.errorbar(np.array(st.index), [v[0] for v in st], yerr=[v[1] for v in st], color=colr, marker="o", ms=4,
                        capsize=3, lw=1.5, label=name)
        a_.set_xticks(range(len(GAP_LABELS)))
        a_.set_xticklabels(GAP_LABELS, fontsize=8)
        a_.set_title(f"{BANK_NAME[b]}, {col} disc", fontsize=11)
        a_.set_xlabel("disc-to-cube gap in image (px)", fontsize=10)
        a_.grid(axis="y", alpha=0.3)
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=9)
    ax[0].set_ylabel("TCP step change (mm)", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"ptm_sweep_pixel_gap.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("render", "analyse", "summarise"), required=True)
    p.add_argument("--bank", default="PTM_worldedit", help="render stage: one bank")
    p.add_argument("--bank-list", nargs="+", default=["PTM_worldedit", "PTM_hover"])
    p.add_argument("--colour", choices=sorted(COLOURS), default="white")
    p.add_argument("--side", type=int, choices=SIDES, default=1)
    p.add_argument("--n", type=int, default=0, help="scenes to render (0 = all)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--banks", default=os.environ.get("PTM_BANKS", "banks"))
    p.add_argument("--render-dir", default="renders_sweep")
    p.add_argument("--raw-dir", default="results/analysis/ptm_sweep/raw")
    p.add_argument("--min-px", type=int, default=20, help="changed px for a disc to count as visible")
    p.add_argument("--out", default="results/analysis/ptm_sweep")
    a = p.parse_args()
    {"render": render, "analyse": analyse, "summarise": summarise}[a.stage](a)


if __name__ == "__main__":
    main()
