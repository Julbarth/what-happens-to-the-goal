"""Closed-loop rollouts from bank states: native goal/proxy reach (Level 3) and AttributeTable h_fc steering arms."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E

T_MAX = 50


def lr_first_contact(path, goal, corner, thr=0.03):
    """Training ends an episode at the first 3 cm contact, so only the first contact is a trained outcome."""
    T = path.shape[1]
    hit_g = np.linalg.norm(path - goal[:, None], axis=-1) < thr
    hit_c = np.linalg.norm(path - corner[None, None], axis=-1) < thr
    tg = np.where(hit_g.any(1), hit_g.argmax(1), T)
    tc = np.where(hit_c.any(1), hit_c.argmax(1), T)
    return np.where(tg < tc, "goal", np.where(tc < tg, "proxy", "neither")), tg, tc


@torch.no_grad()
def native_batch(env, agent, dev):
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    path = [u.agent.tcp.pose.p.cpu().numpy()]
    for _ in range(T_MAX):
        mu = C.taps(agent, obs["rgb"], obs["state"])["mu"]
        obs, *_ = env.step(mu)
        path.append(u.agent.tcp.pose.p.cpu().numpy())
    return np.stack(path, 1), (u.first_touch.cpu().numpy() if hasattr(u, "first_touch") else None)


def native(env_name, agents, bs, out_csv, dev):
    bank = C.load_bank(f"{env_name}_worldedit")
    env = E.make_env(env_name, bs)
    n = len(bank["tcp"])
    rows = []
    for cohort, seed, ck in agents:
        agent = C.build_agent(ck, init_seed=seed, dev=dev)
        sets = ["base", "colour_swap"] if env_name == "AT" else ["g_d", "corner_d"]
        for st in sets:
            for sl in C.batched(n, bs):
                idx = np.arange(sl.start, sl.stop)
                pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
                if env_name == "AT":
                    cfg = dict(u=bank["u"][pad], v=bank["v"][pad])
                    E.force_at(env, bank["qpos_forced"][pad], E.at_placement(cfg, st))
                    path, ft = native_batch(env, agent, dev)
                    out = np.array(["neither", "goal", "proxy"])[ft]
                else:
                    E.force_lr(env, bank["qpos"][pad], bank["goal_" + st][pad], bank["dxy_" + st][pad])
                    path, _ = native_batch(env, agent, dev)
                    out, _, _ = lr_first_contact(path, bank["goal_" + st][pad], np.array([*E.CORNER, E.LR_GOAL_Z]))
                for j, i in enumerate(idx):
                    rows.append(dict(cohort=cohort, seed=seed, set=st, state=int(i), outcome=out[j]))
        print(cohort, seed, pd.DataFrame([r for r in rows if r["seed"] == seed and r["cohort"] == cohort])
              .groupby("set")["outcome"].value_counts(normalize=True).round(3).to_dict(), flush=True)
    env.close()
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------------------------------------------------------- AttributeTable steering

COPIES = ("base", "colour_swap", "shape_swap")  # d_comp = h(colour_swap) - h(shape_swap) points toward the goal at u


def arm_list(doses_local, doses_fixed, n_random):
    arms = [("baseline", 0.0, 0)]
    for b in doses_local:
        arms += [("local", b, 0), ("local", -b, 0)] + [("random", b, r) for r in range(n_random)]
    for b in doses_fixed:
        for name in ("fixed_local", "shared", "unsigned", "fisher_proj"):
            arms += [(name, b, 0), (name, -b, 0)]
    return arms


@torch.no_grad()
def steer_batch(env, agent, dev, arm, beta, rep, u_comp, sign, seed):
    """One rollout per eval state with three synchronised copies (base + both swaps).

    The copies share the arm because objects are collision-free and every copy receives the base copy's action."""
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 3
    sig = C.sigma(agent)
    ut = torch.as_tensor(u_comp, device=dev)
    sg = torch.as_tensor(sign, dtype=torch.float32, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + rep)
    rdir = torch.randn(B, ut.numel(), generator=gen).to(dev)
    rdir = rdir / rdir.norm(dim=1, keepdim=True)
    fixed, drift, dnorm0 = None, 0.0, None
    for t in range(T_MAX):
        tp = C.taps(agent, obs["rgb"], obs["state"])
        h, hs = tp["h_fc"], tp["h_state"]
        hb, hc, hsw = h[:B], h[B:2 * B], h[2 * B:]
        d = hc - hsw
        if t == 0:
            dnorm0 = d.norm(dim=1)
            if arm == "fixed_local":
                fixed = d
            elif arm == "shared":
                fixed = (sg * dnorm0)[:, None] * ut[None]
            elif arm == "unsigned":
                fixed = dnorm0[:, None] * ut[None]
            elif arm == "fisher_proj":
                with torch.enable_grad():
                    J = C.mean_jacobian(agent, "h_fc", hb, hs[:B])
                Jw = J / sig[None, :, None]
                ju, jd = torch.einsum("bad,d->ba", Jw, ut), torch.einsum("bad,bd->ba", Jw, d)
                fixed = ((ju * jd).sum(-1) / (ju * ju).sum(-1).clamp_min(1e-18))[:, None] * ut[None]
        if arm == "baseline":
            delta = torch.zeros_like(hb)
        elif arm == "local":
            delta = beta * d
        elif arm == "random":
            delta = beta * d.norm(dim=1, keepdim=True) * rdir
        else:
            delta = beta * fixed
        mu = C.mu_from_fc(agent, hb + delta, hs[:B])
        obs, *_ = env.step(mu.repeat(3, 1))
        q = u.agent.robot.get_qpos()
        drift = max(drift, (q[B:] - q[:B].repeat(2, 1)).abs().max().item())
    return u.first_touch[:B].cpu().numpy(), drift, dnorm0.cpu().numpy()


def steer(agents, bs, out_csv, dev, doses_local, doses_fixed, n_random):
    bank = C.load_bank("AT_worldedit")
    env = E.make_env("AT", 3 * bs)
    done = set()
    rows = []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev.cohort, prev.seed, prev.arm, prev.beta.round(6), prev.rep))
    for cohort, seed, ck in agents:
        agent = C.build_agent(ck, init_seed=seed, dev=dev)
        ax = np.load(os.path.join(C.RESULTS, "axis", f"AT_{cohort}_s{seed}.npz"))
        ev, u_comp = ax["eval"], ax["u_comp"]
        for arm, beta, rep in arm_list(doses_local, doses_fixed, n_random):
            if (cohort, seed, arm, round(beta, 6), rep) in done:
                continue
            new = []
            for sl in C.batched(len(ev), bs):
                idx = ev[sl]
                pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
                cfg = dict(u=bank["u"][pad], v=bank["v"][pad])
                pls = [E.at_placement(cfg, c) for c in COPIES]
                E.force_at(env, np.tile(bank["qpos_forced"][pad], (3, 1)),
                           {k: np.concatenate([pl[k] for pl in pls]) for k in pls[0]})
                ft, drift, dn = steer_batch(env, agent, dev, arm, beta, rep, u_comp, bank["sign"][pad], seed)
                for j, i in enumerate(idx):
                    new.append(dict(cohort=cohort, seed=seed, arm=arm, beta=beta, rep=rep, state=int(i),
                                    outcome=["neither", "goal", "proxy"][int(ft[j])], dnorm0=float(dn[j]),
                                    copy_drift=drift))
            rows += new
            pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)
            os.replace(out_csv + ".tmp", out_csv)
            g = pd.DataFrame(new)["outcome"].value_counts(normalize=True)
            print(f"{cohort} s{seed} {arm} b={beta:+.3f} r{rep}: goal {g.get('goal', 0):.3f} proxy {g.get('proxy', 0):.3f}"
                  f" drift {new[-1]['copy_drift']:.1e}", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["native", "steer"], required=True)
    p.add_argument("--env", choices=["AT", "LR"], default="AT")
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--doses_local", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--doses_fixed", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    p.add_argument("--n_random", type=int, default=3)
    a = p.parse_args()
    C.ensure_dirs()
    dev = torch.device("cuda")  # physx_cuda observations live on the GPU
    agents = C.agent_list(a.env, a.seeds_gm, a.seeds_control, a.init_seeds)
    if a.mode == "native":
        native(a.env, agents, a.batch, os.path.join(C.RESULTS, f"native_{a.env}_per_state.csv"), dev)
    else:
        steer(agents, a.batch, os.path.join(C.RESULTS, "steer_AT_per_state.csv"), dev, a.doses_local, a.doses_fixed,
              a.n_random)


if __name__ == "__main__":
    main()
