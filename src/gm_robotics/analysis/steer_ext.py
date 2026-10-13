"""h_fc steering with re-synchronised counterfactual copies for AttributeTable, SortZones and LocationReach."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis.rollouts import arm_list, lr_first_contact

T_AT, T_SZ, T_LR = 50, 100, 50
# SZ copies: d_comp = h(colour_swap) - h(mark_swap) points toward the crossed pad u (axis_geometry.CUE_EDITS).
SZ_COPIES = ("base", "colour_swap", "mark_swap")


def find_obs_wrapper(env):
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    while not isinstance(env, FlattenRGBDObservationWrapper):
        env = env.env
    return env


def sync_copies(env, k, info, actors=()):
    """Overwrite the robot (and dynamic `actors`) of copies 1..k-1 with copy 0's state and re-render.

    Identical sub-scenes fed identical actions still diverge on the GPU solver (AT arm up to 0.5 rad, SZ cube by
    metres within 50-100 steps), so without this d(s_t) would mix the edit with a different arm / cube state.
    Returns the fresh observation and the max divergence that accumulated during the last step."""
    u = env.unwrapped
    B = u.num_envs // k
    sd = u.get_state_dict()
    keep = {"articulations": {u.agent.robot.name: sd["articulations"][u.agent.robot.name]},
            "actors": {a: sd["actors"][a] for a in actors}}
    div = 0.0
    for grp in keep.values():
        for name, x in list(grp.items()):
            div = max(div, (x[B:] - x[:B].repeat(k - 1, 1)).abs().max().item())
            # with env_idx, set_state expects only the rows of the selected sub-envs
            grp[name] = x[:B].repeat(k - 1, 1)
    if not keep["actors"]:
        del keep["actors"]
    u.set_state_dict(keep, env_idx=torch.arange(B, u.num_envs, device=u.device))
    return find_obs_wrapper(env).observation(u.get_obs(info)), div


def make_delta(arm, beta, t, d, state, ctx):
    """Push for one step. `state` carries what fixed arms freeze at t = 0; ctx holds u_comp, sign, rdir, Fisher fn."""
    if t == 0:
        state["dnorm0"] = d.norm(dim=1)
        ut, sg, n0 = ctx.get("u_comp"), ctx.get("sign"), state["dnorm0"]
        if arm == "fixed_local":
            state["fixed"] = d
        elif arm == "shared":
            state["fixed"] = (sg * n0)[:, None] * ut[None]
        elif arm == "unsigned":
            state["fixed"] = n0[:, None] * ut[None]
        elif arm == "fisher_proj":
            Jw = ctx["Jw0"]()
            ju, jd = torch.einsum("bad,d->ba", Jw, ut), torch.einsum("bad,bd->ba", Jw, d)
            state["fixed"] = ((ju * jd).sum(-1) / (ju * ju).sum(-1).clamp_min(1e-18))[:, None] * ut[None]
    if arm == "baseline":
        return torch.zeros_like(d)
    if arm == "local":
        return beta * d
    if arm == "random":
        return beta * d.norm(dim=1, keepdim=True) * ctx["rdir"]
    return beta * state["fixed"]


def rand_dirs(B, D, seed, rep, dev):
    gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + rep)
    r = torch.randn(B, D, generator=gen)
    return (r / r.norm(dim=1, keepdim=True)).to(dev)


# Per env: copies, horizon, dynamic actors to re-synchronise, first-outcome attribute (1 goal, 2 proxy), goal distance.
SPEC = {"AT": dict(copies=("base", "colour_swap", "shape_swap"), T=T_AT, actors=(), latch="first_touch", dist=None),
        "SZ": dict(copies=SZ_COPIES, T=T_SZ, actors=("cube",), latch="first_entry", dist="d_marked")}


@torch.no_grad()
def steer_batch(env, spec, agent, dev, arm, beta, rep, u_comp, sign, seed):
    """Three copies (base + both swaps; edited objects are visual only / collision-free), re-synchronised to the base
    copy's arm (and cube) after every step; copy_drift is the largest one-step divergence before re-synchronisation."""
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 3
    sig = C.sigma(agent)
    ctx = dict(u_comp=torch.as_tensor(u_comp, device=dev), sign=torch.as_tensor(sign, dtype=torch.float32, device=dev),
               rdir=rand_dirs(B, len(u_comp), seed, rep, dev))
    st, drift = {}, 0.0
    min_g = torch.full((B,), float("inf"), device=dev)
    for t in range(spec["T"]):
        tp = C.taps(agent, obs["rgb"], obs["state"])
        h, hs = tp["h_fc"], tp["h_state"]
        hb, d = h[:B], h[B:2 * B] - h[2 * B:]

        def jw0():
            with torch.enable_grad():
                return C.mean_jacobian(agent, "h_fc", hb, hs[:B]) / sig[None, :, None]
        ctx["Jw0"] = jw0
        delta = make_delta(arm, beta, t, d, st, ctx)
        mu = C.mu_from_fc(agent, hb + delta, hs[:B])
        obs, _, _, _, info = env.step(mu.repeat(3, 1))
        if spec["dist"]:
            live = getattr(u, spec["latch"])[:B] == 0
            min_g[live] = torch.minimum(min_g[live], info[spec["dist"]][:B][live])
        obs, div = sync_copies(env, 3, info, actors=spec["actors"])
        drift = max(drift, div)
    return getattr(u, spec["latch"])[:B].cpu().numpy(), drift, st["dnorm0"].cpu().numpy(), min_g.cpu().numpy()


def force_copies(env_name, env, bank, pad, copies):
    if env_name == "AT":
        cfg = dict(u=bank["u"][pad], v=bank["v"][pad])
        pls = [E.at_placement(cfg, c) for c in copies]
        E.force_at(env, np.tile(bank["qpos_forced"][pad], (3, 1)), {k: np.concatenate([pl[k] for pl in pls]) for k in pls[0]})
    else:
        cfg = dict(u=bank["u"][pad], v=bank["v"][pad], cube=bank["cube"][pad])
        pls = [E.sz_placement(cfg, c) for c in copies]
        E.force_sz(env, np.tile(bank["qpos_forced"][pad], (3, 1)), {k: np.concatenate([pl[k] for pl in pls]) for k in pls[0]})


def steer_attr(env_name, agents, bs, out_csv, dev, doses_local, doses_fixed, n_random):
    spec = SPEC[env_name]
    bank = C.load_bank(f"{env_name}_worldedit")
    env = E.make_env(env_name, 3 * bs)
    rows, done = [], set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev.cohort, prev.seed, prev.arm, prev.beta.round(6), prev.rep))
    for cohort, seed, ck in agents:
        agent = C.build_agent(ck, init_seed=seed, dev=dev)
        ax = np.load(os.path.join(C.RESULTS, "axis", f"{env_name}_{cohort}_s{seed}.npz"))
        ev, u_comp = ax["eval"], ax["u_comp"]
        for arm, beta, rep in arm_list(doses_local, doses_fixed, n_random):
            if (cohort, seed, arm, round(beta, 6), rep) in done:
                continue
            new = []
            for sl in C.batched(len(ev), bs):
                idx = ev[sl]
                pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
                force_copies(env_name, env, bank, pad, spec["copies"])
                fe, drift, dn, mg = steer_batch(env, spec, agent, dev, arm, beta, rep, u_comp, bank["sign"][pad], seed)
                for j, i in enumerate(idx):
                    new.append(dict(cohort=cohort, seed=seed, arm=arm, beta=beta, rep=rep, state=int(i),
                                    outcome=["neither", "goal", "proxy"][int(fe[j])], dnorm0=float(dn[j]),
                                    min_d_goal=float(mg[j]), copy_drift=drift))
            rows += new
            pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)
            os.replace(out_csv + ".tmp", out_csv)
            g = pd.DataFrame(new)["outcome"].value_counts(normalize=True)
            print(f"{env_name} {cohort} s{seed} {arm} b={beta:+.3f} r{rep}: goal {g.get('goal', 0):.3f} proxy "
                  f"{g.get('proxy', 0):.3f} drift {new[-1]['copy_drift']:.1e}", flush=True)
    env.close()


# ---------------------------------------------------------------- LocationReach state-local steering

@torch.no_grad()
def steer_lr_batch(env, agent, dev, arm, beta, rep, goal, seed):
    """Two copies: marker at g (base) and marker at the training corner; the marker and distractor have no collision.
    d_local = h(marker at g) - h(marker at corner), the paper's Maze construction (cheese at its cell minus corner)."""
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 2
    ctx = dict(rdir=rand_dirs(B, 256, seed, rep, dev))
    st, path, drift = {}, [u.agent.tcp.pose.p[:B].cpu().numpy()], 0.0
    for t in range(T_LR):
        tp = C.taps(agent, obs["rgb"], obs["state"])
        h, hs = tp["h_fc"], tp["h_state"]
        d = h[:B] - h[B:]
        delta = make_delta(arm, beta, t, d, st, ctx)
        mu = C.mu_from_fc(agent, h[:B] + delta, hs[:B])
        obs, _, _, _, info = env.step(mu.repeat(2, 1))
        path.append(u.agent.tcp.pose.p[:B].cpu().numpy())
        obs, div = sync_copies(env, 2, info)
        drift = max(drift, div)
    out, _, _ = lr_first_contact(np.stack(path, 1), goal, np.array([*E.CORNER, E.LR_GOAL_Z]))
    return out, drift, st["dnorm0"].cpu().numpy()


def steer_lr(agents, bs, out_csv, dev, doses, n_random, n_states):
    """Off-path test states of rewire_lr.offpath_setup (pair seed k: the goals chosen from GM seed k's visited region)."""
    from gm_robotics.analysis.rewire_lr import offpath_setup
    env = E.make_env("LR", 2 * bs)
    rows, done = [], set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev.cohort, prev.seed, prev.arm, prev.beta.round(6), prev.rep))
    corner = np.array([*E.CORNER, E.LR_GOAL_Z], np.float32)
    arms = [("baseline", 0.0, 0)] + [x for b in doses for x in [("local", b, 0), ("local", -b, 0)] +
                                     [("random", b, r) for r in range(n_random)]]
    for cohort, seed, ck in agents:
        agent = C.build_agent(ck, init_seed=seed, dev=dev)
        off = offpath_setup(seed)
        n = min(n_states, len(off["qpos"]))
        for arm, beta, rep in arms:
            if (cohort, seed, arm, round(beta, 6), rep) in done:
                continue
            new = []
            for sl in C.batched(n, bs):
                idx = np.arange(sl.start, sl.stop)
                pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
                goal = off["goal"][pad]
                E.force_lr(env, np.tile(off["qpos"][pad], (2, 1)), np.concatenate([goal, np.tile(corner, (bs, 1))]),
                           np.tile(off["dxy"][pad], (2, 1)))
                out, drift, dn = steer_lr_batch(env, agent, dev, arm, beta, rep, goal, seed)
                for j, i in enumerate(idx):
                    new.append(dict(cohort=cohort, seed=seed, arm=arm, beta=beta, rep=rep, state=int(i),
                                    outcome=out[j], dnorm0=float(dn[j]), copy_drift=drift))
            rows += new
            pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)
            os.replace(out_csv + ".tmp", out_csv)
            g = pd.DataFrame(new)["outcome"].value_counts(normalize=True)
            print(f"LR {cohort} s{seed} {arm} b={beta:+.3f} r{rep}: goal {g.get('goal', 0):.3f} corner "
                  f"{g.get('proxy', 0):.3f} drift {new[-1]['copy_drift']:.1e}", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "SZ", "LR"], default="SZ")
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--doses_local", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--doses_fixed", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--n_states", type=int, default=200, help="LR: first n off-path states")
    a = p.parse_args()
    C.ensure_dirs()
    dev = torch.device("cuda")
    agents = C.agent_list(a.env, a.seeds_gm, a.seeds_control, init_seeds=())
    if a.env in SPEC:
        steer_attr(a.env, agents, a.batch, os.path.join(C.RESULTS, f"steer_{a.env}_per_state.csv"), dev, a.doses_local,
                   a.doses_fixed, a.n_random)
    else:
        steer_lr(agents, a.batch, os.path.join(C.RESULTS, "steer_LR_per_state.csv"), dev, a.doses_local, a.n_random,
                 a.n_states)


if __name__ == "__main__":
    main()
