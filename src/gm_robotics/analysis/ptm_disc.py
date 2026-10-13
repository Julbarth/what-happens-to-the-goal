"""PushToMarker GM/control response to a size-matched non-goal-colour disc vs the white marker at the world-edit locations."""
import argparse
import os

import numpy as np

LOCS = ("m", "m2", "centre")
COLOURS = {"white": (1.0, 1.0, 1.0, 1.0), "green": (0.10, 0.75, 0.15, 1.0), "yellow": (1.0, 0.92, 0.20, 1.0)}


def render(a):
    """Re-render nomarker + a disc of colour --colour at m / m2 / centre for every scene of the bank (one process per colour:
    the marker material is fixed when the scene is built)."""
    import gymnasium as gym
    import torch
    from gm_robotics.analysis import envs_forced as E
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
    E.PTM_ENV.MARKER_RGBA = np.array(COLOURS[a.colour])
    z = np.load(os.path.join(a.banks, a.bank + ".npz"))
    n = len(z["m"]) if a.n <= 0 else a.n
    bs = a.batch if a.sim_backend == "physx_cuda" else 1
    env = gym.make("PushToMarkerForced-v1", variant="out_marker", start_split="test", num_envs=bs, obs_mode="rgb",
                   sim_backend=a.sim_backend, control_mode="pd_joint_delta_pos", reconfiguration_freq=None)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    qpos, cube = z["qpos_forced"][:n], z["cube"][:n]
    nan = np.full((n, 2), np.nan, np.float32)
    states = {"nomarker": nan} if a.colour == "white" else {}
    states.update({loc: z[loc][:n].astype(np.float32) for loc in LOCS})
    out = {}
    for name, mxy in states.items():
        goal = np.where(np.isfinite(mxy), mxy, z["m"][:n]).astype(np.float32)
        acc = {}
        for s in range(0, n, bs):
            idx = np.arange(s, min(s + bs, n))
            pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
            E.force_ptm(env, qpos[pad], mxy[pad], goal[pad], cube[pad])
            with torch.no_grad():
                r = E.render_obs(env)
            for k, v in r.items():
                acc.setdefault(k, []).append(v[: len(idx)])
        acc = {k: np.concatenate(v) for k, v in acc.items()}
        out[f"rgb_{name}"], out[f"state_{name}"] = acc["rgb"], acc["state"]
        out[f"qpos_{name}"], out[f"tcp_{name}"] = acc["qpos"], acc["tcp"]
        print(f"{a.bank} {a.colour} {name}: {n} scenes", flush=True)
    os.makedirs(a.render_dir, exist_ok=True)
    np.savez(os.path.join(a.render_dir, f"{a.bank}_{a.colour}_{a.sim_backend}.npz"), **out)


def npx(x, y):
    return (np.abs(x.astype(np.int16) - y.astype(np.int16)).max(-1) > 0).sum((1, 2))


def analyse(a):
    """Presence KL vs no-disc and step-direction cosines, per state, for white and the non-goal disc on identical renders."""
    import pandas as pd
    import torch
    from gm_robotics.analysis import depth_common as D
    C = D.load_common()
    torch.set_num_threads(a.threads)
    rows = []
    for bank in a.bank_list:
        z = np.load(os.path.join(a.banks, bank + ".npz"))
        W = np.load(os.path.join(a.render_dir, f"{bank}_white_{a.sim_backend}.npz"))
        G = np.load(os.path.join(a.render_dir, f"{bank}_{a.disc}_{a.sim_backend}.npz"))
        n = len(W["rgb_nomarker"])
        tcp, cube = z["tcp"][:n, :2], z["cube"][:n, :2]
        # pixel difference between the re-render and the bank render, as a renderer-fidelity check
        pix_dev = dict(nomarker=npx(W["rgb_nomarker"], z["rgb_nomarker"][:n]), m=npx(W["rgb_m"], z["rgb_base"][:n]))
        renders = {("orig", "white", "none"): (z["rgb_nomarker"][:n], z["state_nomarker"][:n])}
        for loc in LOCS:
            renders[("orig", "white", loc)] = (z["rgb_" + ("base" if loc == "m" else loc)][:n],
                                               z["state_" + ("base" if loc == "m" else loc)][:n])
        renders[("rerender", "white", "none")] = (W["rgb_nomarker"], W["state_nomarker"])
        for loc in LOCS:
            renders[("rerender", "white", loc)] = (W["rgb_" + loc], W["state_" + loc])
            renders[("rerender", a.disc, loc)] = (G["rgb_" + loc], G["state_" + loc])
        for cohort in ("gm", "control", "init"):
            for s in a.seeds:
                ag = C.build_agent(None, init_seed=s) if cohort == "init" else C.build_agent(C.ckpt_path("PTM", cohort, s))
                sig = C.sigma(ag)
                mu = {}
                for k, (rgb, st) in renders.items():
                    mu[k] = torch.cat([C.taps(ag, torch.as_tensor(rgb[sl]), torch.as_tensor(st[sl]))["mu"]
                                       for sl in C.batched(len(rgb), 256)])
                for src in ("orig", "rerender"):
                    base = mu[(src, "white", "none")]
                    rgb0 = renders[(src, "white", "none")][0]
                    for (s2, col, loc), m_ in mu.items():
                        if s2 != src or loc == "none":
                            continue
                        kl = C.gaussian_kl(m_, base, sig).numpy()
                        ds = torch.einsum("bkj,bj->bk", torch.as_tensor(z["J_tcp"][:n]),
                                          C.JOINT_DELTA_SCALE * (m_ - base)[:, :7]).numpy()[:, :2]
                        push = (z["push_m2"] if loc == "m2" else z["push_m"])[:n, :2]
                        xy = z[loc][:n]
                        tp = (push - tcp) / np.linalg.norm(push - tcp, axis=1, keepdims=True)
                        tl = (xy - tcp) / np.linalg.norm(xy - tcp, axis=1, keepdims=True)
                        nd = np.linalg.norm(ds, axis=1) + 1e-12
                        px = npx(renders[(src, col, loc)][0], rgb0)
                        for i in range(n):
                            rows.append(dict(bank=bank, src=src, colour=col, loc=loc, cohort=cohort, seed=s, state=i,
                                             KL=kl[i], npx=px[i], d_cube=np.linalg.norm(xy[i] - cube[i]),
                                             cos_to_push=(ds[i] * tp[i]).sum() / nd[i],
                                             cos_to_disc=(ds[i] * tl[i]).sum() / nd[i]))
                print(f"{bank} {cohort} s{s} done", flush=True)
        print(f"{bank}: re-render vs bank changed px, nomarker median {np.median(pix_dev['nomarker']):.0f} "
              f"max {pix_dev['nomarker'].max()}, white@m median {np.median(pix_dev['m']):.0f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.raw_dir, exist_ok=True)
    for (c, s), g in df.groupby(["cohort", "seed"]):
        g.to_csv(os.path.join(a.raw_dir, f"runB_ptm_disc_{c}_s{s}.csv"), index=False)


def summarise(a):
    import glob
    import pandas as pd
    from gm_robotics.analysis import depth_common as D
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.raw_dir, "runB_ptm_disc_*.csv")))])
    L = ["## Run B tables (generated)", "", "Mean [95% t-CI] over seeds of per-seed state means; KL in each policy's own sigma.",
         "", "| bank | src | colour | loc | cohort | KL mean | KL median | changed px | cos(dstep, push pt) | cos(dstep, disc) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for (b, src, col, loc, c), g in df.groupby(["bank", "src", "colour", "loc", "cohort"], sort=True):
        ps = g.groupby("seed").agg(KL=("KL", "mean"), KLmed=("KL", "median"), px=("npx", "median"),
                                   cp=("cos_to_push", "mean"), cd=("cos_to_disc", "mean"))
        L.append(f"| {b} | {src} | {col} | {loc} | {c} | {D.fmt_ci(ps.KL, 2)} | {D.fmt_ci(ps.KLmed, 2)} | "
                 f"{ps.px.mean():.0f} | {D.fmt_ci(ps.cp)} | {D.fmt_ci(ps.cd)} |")
    L += ["", "### Within-state ratio disc / white (re-render; per seed mean over states of log(KL_disc / KL_white), exp shown)",
          "", "| bank | loc | cohort | KL ratio disc/white | per-px ratio disc/white | per seed |", "|---|---|---|---|---|---|"]
    rr = df[df.src == "rerender"]
    piv = rr.pivot_table(index=["bank", "loc", "cohort", "seed", "state"], columns="colour", values=["KL", "npx"]).reset_index()
    disc = [c for c in rr.colour.unique() if c != "white"][0]
    for (b, loc, c), g in piv.groupby(["bank", "loc", "cohort"]):
        if c == "init":
            continue
        lr = g.groupby("seed").apply(lambda h: np.mean(np.log((h[("KL", disc)] + 1e-9) / (h[("KL", "white")] + 1e-9))))
        lp = g.groupby("seed").apply(lambda h: np.mean(np.log(((h[("KL", disc)] + 1e-9) / h[("npx", disc)].clip(1))
                                                             / ((h[("KL", "white")] + 1e-9) / h[("npx", "white")].clip(1)))))
        m, lo, hi = D.ci95(lr)
        m2, lo2, hi2 = D.ci95(lp)
        L.append(f"| {b} | {loc} | {c} | {np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}] | "
                 f"{np.exp(m2):.2f} [{np.exp(lo2):.2f}, {np.exp(hi2):.2f}] | {' / '.join(f'{np.exp(v):.2f}' for v in lr)} |")
    L += ["", "### Location profile within colour: centre / m2 (re-render), per seed geo-mean", "",
          "| bank | colour | cohort | centre/m2 | m/m2 |", "|---|---|---|---|---|"]
    piv2 = rr.pivot_table(index=["bank", "colour", "cohort", "seed", "state"], columns="loc", values="KL").reset_index()
    for (b, col, c), g in piv2.groupby(["bank", "colour", "cohort"]):
        if c == "init":
            continue
        out = []
        for x, y in (("centre", "m2"), ("m", "m2")):
            v = g.groupby("seed").apply(lambda h: np.mean(np.log((h[x] + 1e-9) / (h[y] + 1e-9))))
            m, lo, hi = D.ci95(v)
            out.append(f"{np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]")
        L.append(f"| {b} | {col} | {c} | {out[0]} | {out[1]} |")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, f"runB_tables_generated_{disc}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("render", "analyse", "summarise"), required=True)
    p.add_argument("--bank", default="PTM_worldedit", help="render stage: one bank")
    p.add_argument("--bank-list", nargs="+", default=["PTM_worldedit", "PTM_hover"])
    p.add_argument("--colour", choices=sorted(COLOURS), default="green")
    p.add_argument("--disc", default="green", help="non-goal colour analysed against white")
    p.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cpu")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n", type=int, default=0, help="scenes to render (0 = all)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--banks", default=os.environ.get("PTM_BANKS", "banks"))
    p.add_argument("--render-dir", default="renders")
    p.add_argument("--raw-dir", default="results/analysis/deepdive_followups/runB_raw")
    p.add_argument("--out", default="results/analysis/deepdive_followups")
    a = p.parse_args()
    {"render": render, "analyse": analyse, "summarise": summarise}[a.stage](a)


if __name__ == "__main__":
    main()
