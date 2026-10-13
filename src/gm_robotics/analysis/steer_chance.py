"""Chance controls for AT / SZ h_fc steering: push from the real swap copies while the policy sees a phantom or decoupled goal cue."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis import rewire_attr as R
from gm_robotics.analysis.steer_ext import make_delta, rand_dirs, sync_copies

OUT = os.path.join(C.RESULTS, "recovery")
T = {"AT": 50, "SZ": 100}
SEEDS = {"AT": dict(gm=[2, 3, 4], control=[2, 3, 4, 5]), "SZ": dict(gm=[0, 1, 2], control=[0, 1, 2])}
# What the policy sees (copy 0); u = goal location of the base conflict, v = lure. AT spec (shape, colour), SZ (colour, cross).
VISIBLE = {
    "AT": {"standard": (("cube", "blue"), ("sphere", "red")),
           "phantom": (("sphere", "blue"), ("sphere", "red")),
           "decoupled": (("sphere", "blue"), ("cube", "red"))},
    "SZ": {"standard": (("purple", True), ("green", False)),
           "phantom": (("purple", False), ("green", False)),
           "decoupled": (("purple", False), ("green", True))},
}
# Copies 1 and 2 are always the base conflict's colour swap and goal-cue swap, so d = h1 - h2 points toward u
# (steer_ext / axis_geometry orientation) whatever copy 0 shows.
SWAPS = {"AT": ((("cube", "red"), ("sphere", "blue")), (("sphere", "blue"), ("cube", "red"))),
         "SZ": ((("green", True), ("purple", False)), (("purple", False), ("green", True)))}
ARMS = [("baseline", 0.0, 0), ("local", 1.0, 0), ("local", -1.0, 0), ("local", 0.25, 0),
        ("random", 1.0, 0), ("random", 1.0, 1), ("random", 1.0, 2),
        ("fixed_local", 1.0, 0), ("shared", 0.25, 0), ("shared", -0.25, 0), ("unsigned", 0.25, 0)]


def placements(kind, cfg, cond):
    pls = [R.place(kind, cfg, *VISIBLE[kind][cond])] + [R.place(kind, cfg, *s) for s in SWAPS[kind]]
    return pls[0], R.cat_place(pls)


@torch.no_grad()
def run_batch(kind, env, agent, dev, arm, beta, rep, u_comp, sign, seed, pl0):
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 3
    sig = C.sigma(agent)
    ctx = dict(u_comp=torch.as_tensor(u_comp, device=dev), sign=torch.as_tensor(sign, dtype=torch.float32, device=dev),
               rdir=rand_dirs(B, len(u_comp), seed, rep, dev))
    st, drift = {}, 0.0
    tcp = [u.agent.tcp.pose.p[:B].cpu().numpy()]
    cube = [u.cube.pose.p[:B, :2].cpu().numpy()] if kind == "SZ" else None
    for t in range(T[kind]):
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
        tcp.append(u.agent.tcp.pose.p[:B].cpu().numpy())
        if kind == "SZ":
            cube.append(u.cube.pose.p[:B, :2].cpu().numpy())
        obs, div = sync_copies(env, 3, info, actors=("cube",) if kind == "SZ" else ())
        drift = max(drift, div)
    path = np.stack(tcp, 1)
    if kind == "AT":
        out, tu, tv, mdu, mdv = R.at_first_touch(path, pl0)
        t_dec = np.minimum(tu, tv)
    else:
        fe = u.first_entry[:B].cpu().numpy()
        marked_u = np.all(np.isclose(pl0["marked"], pl0["u_xy"]), 1)
        cp = np.stack(cube, 1)
        mdu = np.linalg.norm(cp - pl0["u_xy"][:, None], axis=-1).min(1)
        mdv = np.linalg.norm(cp - pl0["v_xy"][:, None], axis=-1).min(1)
        lab = np.where(marked_u[:, None], np.array([["neither", "u", "v"]]), np.array([["neither", "v", "u"]]))
        out = lab[np.arange(B), fe]
        t_dec = np.full(B, -1)
    return out, mdu, mdv, t_dec, drift, st["dnorm0"].cpu().numpy()


def run(a):
    kind = a.env
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f"steer_chance_{kind}_per_state.csv")
    bank = C.load_bank(f"{kind}_worldedit")
    env = R.make_env(kind, 3 * a.batch)
    rows, done = [], set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev.cond, prev.cohort, prev.seed, prev.arm, prev.beta.round(6), prev.rep))
    for cohort in a.cohorts:
        for seed in SEEDS[kind][cohort]:
            agent = C.build_agent(C.ckpt_path(kind, cohort, seed), init_seed=seed, dev=a.dev)
            ax = np.load(os.path.join(C.RESULTS, "axis", f"{kind}_{cohort}_s{seed}.npz"))
            ev, u_comp = ax["eval"][: a.n_states], ax["u_comp"]
            for cond in a.conds:
                for arm, beta, rep in ARMS:
                    if (cond, cohort, seed, arm, round(beta, 6), rep) in done:
                        continue
                    new = []
                    for sl in C.batched(len(ev), a.batch):
                        idx = ev[sl]
                        pad = np.concatenate([idx, np.full(a.batch - len(idx), idx[-1])])
                        cfg = dict(u=bank["u"][pad], v=bank["v"][pad], qpos=bank["qpos_forced"][pad])
                        if kind == "SZ":
                            cfg["cube"] = bank["cube"][pad]
                        pl0, pl = placements(kind, cfg, cond)
                        pl0 = dict(pl0, u_xy=cfg["u"], v_xy=cfg["v"])
                        R.force(kind, env, np.tile(cfg["qpos"], (3, 1)), pl)
                        res = run_batch(kind, env, agent, a.dev, arm, beta, rep, u_comp, bank["sign"][pad], seed, pl0)
                        out, mdu, mdv, tdec, drift, dn = res
                        for j, i in enumerate(idx):
                            new.append(dict(env=kind, cond=cond, cohort=cohort, seed=seed, arm=arm, beta=beta, rep=rep,
                                            state=int(i), outcome=out[j], min_d_u=float(mdu[j]), min_d_v=float(mdv[j]),
                                            t_decide=int(tdec[j]), dnorm0=float(dn[j]), copy_drift=drift))
                    rows += new
                    pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)
                    os.replace(out_csv + ".tmp", out_csv)
                    g = pd.DataFrame(new)["outcome"].value_counts(normalize=True)
                    print(f"{kind} {cond} {cohort} s{seed} {arm} b={beta:+.3f} r{rep}: u {g.get('u', 0):.3f} "
                          f"v {g.get('v', 0):.3f} drift {drift:.1e}", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "SZ"], default="AT")
    p.add_argument("--cohorts", nargs="+", default=["gm", "control"])
    p.add_argument("--conds", nargs="+", default=["standard", "phantom", "decoupled"])
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--n_states", type=int, default=200)
    a = p.parse_args()
    a.dev = torch.device("cuda")
    run(a)


if __name__ == "__main__":
    main()
