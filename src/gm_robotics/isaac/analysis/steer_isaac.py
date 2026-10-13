"""Closed-loop steering at h_fc / h_pre: state-local, reversed, norm-matched random and shared-axis pushes."""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--runs", nargs="+", default=["main_gm_s0", "main_gm_s1", "main_gm_s2"])
p.add_argument("--ckpt_root", default="checkpoints/isaac/runs")
p.add_argument("--axis_dir", default="results/isaac_analysis/axis")
p.add_argument("--banks", nargs="+", default=["conflict", "native"])
p.add_argument("--n_layouts", type=int, default=128)
p.add_argument("--num_envs", type=int, default=400)
p.add_argument("--plan", default="full", choices=["full", "control", "phantom", "decoupled", "smoke", "shared", "dose", "fpnorm"])
p.add_argument("--doses", nargs="+", type=float, default=[0.125, 0.25, 4.0],
               help="plan dose: extra state-local doses (local +-b and three norm-matched random draws each)")
p.add_argument("--out", default="results/isaac_analysis/steer")
p.add_argument("--settle_frames", type=int, default=640,
               help="real RTX frames after each closed-loop reset (gm_nav_env.settle_after_reset); 0 = no settling, first frames may show the previous scene")
p.add_argument("--render_check", action="store_true",
               help="rollout_core render health check on every arm; a silent RTX failure raises instead of writing blind rows")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import time  # noqa: E402

import numpy as np  # noqa: E402
import csv as csvmod  # noqa: E402
import torch  # noqa: E402
from gm_robotics.isaac import gm_nav_env as G  # noqa: E402
from gm_robotics.isaac.analysis import geom  # noqa: E402
from gm_robotics.isaac.analysis import rollout_core as R  # noqa: E402

LOCAL_DOSES = (0.25, 0.5, 1.0, 2.0, 4.0)
FLAG_DOSES = (0.5, 1.0, 2.0, 4.0)
PRE_DOSES = (0.5, 1.0, 2.0)
FIXED_DOSES = (0.25, 0.5, 1.0)


def arm_list(plan, bank):
    arms = [("baseline", 0.0, 0)]
    if plan == "smoke":
        return arms + [("local", 1.0, 0), ("random", 1.0, 0)]
    if plan == "fpnorm":
        # one Fisher-projected arm, only to log alpha_F(s0) and ||J_w u|| per layout for the push-norm comparison
        return arms[1:] + [("fisher_proj", 0.25, 0)]
    if plan == "dose":
        # extra doses for the local family (control covers only 0.5-2); same arms and random draws as full/control
        for b in args.doses:
            arms += [("local", b, 0), ("local", -b, 0)] + [("random", b, r) for r in range(3)]
        return arms
    if plan == "shared":
        # the full plan's fixed-push arms alone, on any bank (full runs them only on the forced-view bank)
        for b in FIXED_DOSES:
            for name in ("fixed_local", "sign_gated", "unsigned", "fisher_proj"):
                arms += [(name, b, 0), (name, -b, 0)]
        return arms
    if plan in ("phantom", "decoupled"):
        # chance controls: every family of the full plan, one random draw, beta 0.25 dropped
        for b in (0.5, 1.0, 2.0, 4.0):
            arms += [("local", b, 0), ("local", -b, 0), ("random", b, 0), ("flag", b, 0), ("flag", -b, 0), ("random_flag", b, 0)]
        for b in PRE_DOSES:
            arms += [("pre", b, 0), ("pre", -b, 0)]
        if bank == "conflict":
            for b in (0.5, 1.0):
                for name in ("fixed_local", "sign_gated"):
                    arms += [(name, b, 0), (name, -b, 0)]
        return arms
    local = (0.5, 1.0, 2.0) if plan == "control" else LOCAL_DOSES
    for b in local:
        arms += [("local", b, 0), ("local", -b, 0)] + [("random", b, r) for r in range(3)]
    for b in ((1.0, 2.0) if plan == "control" else FLAG_DOSES):
        arms += [("flag", b, 0), ("flag", -b, 0), ("random_flag", b, 0)]
    if plan == "full":
        for b in PRE_DOSES:
            arms += [("pre", b, 0), ("pre", -b, 0), ("random_pre", b, 0)]
        if bank == "conflict":
            for b in FIXED_DOSES:
                for name in ("fixed_local", "sign_gated", "unsigned", "fisher_proj"):
                    arms += [(name, b, 0), (name, -b, 0)]
    return arms


def conflict_sign(lay):
    bu, _ = geom.cam_bearing_dist(lay["flag"], lay["start"], lay["yaw"])
    bv, _ = geom.cam_bearing_dist(lay["tower"], lay["start"], lay["yaw"])
    return np.where(geom.wrap(bu - bv) > 0, 1.0, -1.0).astype(np.float32)


def make_act(agent, arm, beta, rep, acting, u_comp, sign, L, dev, seed):
    sig = torch.exp(agent.actor_logstd).reshape(-1)
    gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + rep)
    rd_fc = torch.randn(L, 256, generator=gen).to(dev)
    rd_fc = rd_fc / rd_fc.norm(dim=1, keepdim=True)
    rd_pre = torch.randn(L, 4096, generator=gen).to(dev)
    rd_pre = rd_pre / rd_pre.norm(dim=1, keepdim=True)
    ut = None if u_comp is None else torch.as_tensor(u_comp, device=dev, dtype=torch.float32)
    sg = torch.as_tensor(sign, device=dev)

    def act(bc, t, st):
        a = bc[acting]
        h, hs = a["h_fc"], a["h_state"]
        d = bc["tower_swap"]["h_fc"] - bc["flag_swap"]["h_fc"]
        if t == 0:
            st["dnorm0"] = d.norm(dim=1).cpu().numpy()
            st["dflag0"] = (bc["base"]["h_fc"] - bc["flag_swap"]["h_fc"]).norm(dim=1).cpu().numpy()
            if arm == "fixed_local":
                st["fixed"] = d.clone()
            elif arm in ("sign_gated", "unsigned"):
                s = sg if arm == "sign_gated" else torch.ones_like(sg)
                st["fixed"] = (s * d.norm(dim=1))[:, None] * ut[None]
            elif arm == "fisher_proj":
                fn = lambda hh, hss: R.mu_from_fc(agent, hh[None], hss[None])[0]
                with torch.enable_grad():
                    J = torch.func.vmap(torch.func.jacrev(fn, argnums=0))(h, hs)
                Jw = J / sig[None, :, None]
                ju, jd = torch.einsum("bad,d->ba", Jw, ut), torch.einsum("bad,bd->ba", Jw, d)
                alpha = (ju * jd).sum(-1) / (ju * ju).sum(-1).clamp_min(1e-18)
                st["alpha0"] = alpha.cpu().numpy()
                st["ju0"] = ju.norm(dim=1).cpu().numpy()
                st["jd0"] = jd.norm(dim=1).cpu().numpy()
                st["fixed"] = alpha[:, None] * ut[None]
        if arm == "baseline":
            return a["mu"]
        if arm == "local":
            return R.mu_from_fc(agent, h + beta * d, hs)
        if arm == "random":
            return R.mu_from_fc(agent, h + beta * d.norm(dim=1, keepdim=True) * rd_fc, hs)
        if arm in ("flag", "random_flag"):
            df = bc["base"]["h_fc"] - bc["flag_swap"]["h_fc"]
            if arm == "random_flag":
                df = df.norm(dim=1, keepdim=True) * rd_fc
            return R.mu_from_fc(agent, h + beta * df, hs)
        if arm in ("pre", "random_pre"):
            dp = bc["tower_swap"]["h_pre"] - bc["flag_swap"]["h_pre"]
            if arm == "random_pre":
                dp = dp.norm(dim=1, keepdim=True) * rd_pre
            return R.mu_from_pre(agent, a["h_pre"] + beta * dp, hs)
        return R.mu_from_fc(agent, h + beta * st["fixed"], hs)

    return act


def main():
    os.makedirs(args.out, exist_ok=True)
    env = G.make_env("conflict", args.num_envs)
    env.reset()
    env.step(torch.zeros(args.num_envs, 3, device=env.device))
    dev = env.device
    print("phase groups:", [len(g) for g in R.phase_groups(env)], flush=True)
    acting = {"phantom": "no_flag", "decoupled": "flag_swap"}.get(args.plan, "base")
    copies = [acting] + [c for c in ("tower_swap", "flag_swap", "base") if c != acting]
    runner = R.CopyRunner(env, args.n_layouts, copies, acting, settle_frames=args.settle_frames,
                          render_check=args.render_check)
    for bank in args.banks:
        base = R.bank_layouts(bank, args.n_layouts)
        lays = R.swap_layouts(base)
        u, v, ph = base["flag"], base["tower"], R.phantom_point(base)
        sign = conflict_sign(base)
        for run in args.runs:
            tag = f"{args.plan}_{bank}_{run}"
            csv = f"{args.out}/{tag}.csv"
            rows = list(csvmod.DictReader(open(csv))) if os.path.exists(csv) else []
            done = set((r["arm"], round(float(r["beta"]), 6), int(r["rep"])) for r in rows)
            agent = R.load_agent(f"{args.ckpt_root}/{run}/final_ckpt.pt", dev)
            cohort, seed = run.split("_")[1], int(run[-1])
            ax = f"{args.axis_dir}/visible_{cohort}_s{seed}.npz"
            u_comp = np.load(ax)["u_comp"] if os.path.exists(ax) else None
            paths = {}
            for arm, beta, rep in arm_list(args.plan, bank):
                if (arm, round(beta, 6), rep) in done:
                    continue
                if u_comp is None and arm in ("sign_gated", "unsigned", "fisher_proj"):
                    if args.plan == "shared":
                        raise FileNotFoundError(ax)
                    continue
                t0 = time.time()
                st = {}
                pol = R.Policy(agent, make_act(agent, arm, beta, rep, acting, u_comp, sign, args.n_layouts, dev, seed))
                res = runner.run(lays, pol, seed=0, extra=st)
                fr, t_fr = R.first_reached(res["xy"], u, v, ph, res["outcome"], res["length"],
                                           goal_term="goalobj" if acting == "flag_swap" else "u")
                for i in range(args.n_layouts):
                    rows.append(dict(run=run, cohort=cohort, seed=seed, bank=bank, plan=args.plan, acting=acting, arm=arm,
                                     beta=beta, rep=rep, layout=i, outcome=res["outcome"][i], first=fr[i], t_first=int(t_fr[i]),
                                     length=int(res["length"][i]), dnorm0=float(st["dnorm0"][i]),
                                     dflag0=float(st["dflag0"][i]), sign=float(sign[i]), drift=res["drift"],
                                     **({k: float(st[k][i]) for k in ("alpha0", "ju0", "jd0")} if "alpha0" in st else {}),
                                     **({f"render_{k}": v for k, v in runner.render_stats.items()} if args.render_check else {})))
                paths[f"{arm}_{beta:+.3f}_{rep}"] = res["xy"].astype(np.float16)
                with open(csv + ".tmp", "w") as fh:
                    w = csvmod.DictWriter(fh, fieldnames=list(rows[-1]))
                    w.writeheader()
                    w.writerows(rows)
                os.replace(csv + ".tmp", csv)
                print(f"{tag} {arm} b={beta:+.3f} r{rep}: u {np.mean(fr == 'u'):.3f} v {np.mean(fr == 'v'):.3f} "
                      f"phantom {np.mean(fr == 'phantom'):.3f} none {np.mean(fr == 'none'):.3f} | goal-term "
                      f"{np.mean(res['outcome'] == 'goal'):.3f} proxy-term {np.mean(res['outcome'] == 'proxy'):.3f} "
                      f"fall {np.mean(res['outcome'] == 'fall'):.3f} drift {res['drift']:.2e} ({time.time() - t0:.0f}s)",
                      flush=True)
            if paths:
                old = f"{args.out}/{tag}_paths.npz"
                if os.path.exists(old):
                    z = np.load(old)
                    paths = {**{k: z[k] for k in z.files}, **paths}
                np.savez_compressed(old, u=u, v=v, phantom=ph, **paths)
    open(f"{args.out}/DONE_{args.plan}", "w").write("ok\n")
    env.close()


main()
app.close()
