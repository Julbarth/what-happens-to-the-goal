"""PushToMarker h_fc steering along the state-local marker edit, with copies re-synchronised every step."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis.steer_ext import make_delta, rand_dirs, sync_copies

T_PTM = 100


@torch.no_grad()
def steer_batch(env, agent, dev, arm, beta, rep, seed):
    """Copy 0 shows the marker at m (scored at m), copy 1 the marker at the table centre (the GM training point).

    The marker is visual only, so after every step copy 1 gets copy 0's robot and cube state and is re-rendered;
    the push is the paper's Maze construction (goal at its location minus goal at the proxy location)."""
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 2
    ctx = dict(rdir=rand_dirs(B, 256, seed, rep, dev))
    st, drift = {}, 0.0
    min_m = torch.full((B,), float("inf"), device=dev)
    min_c = torch.full((B,), float("inf"), device=dev)
    for t in range(T_PTM):
        tp = C.taps(agent, obs["rgb"], obs["state"])
        h, hs = tp["h_fc"], tp["h_state"]
        d = h[:B] - h[B:]
        delta = make_delta(arm, beta, t, d, st, ctx)
        mu = C.mu_from_fc(agent, h[:B] + delta, hs[:B])
        obs, _, _, _, info = env.step(mu.repeat(2, 1))
        min_m = torch.minimum(min_m, info["d_marker"][:B])
        min_c = torch.minimum(min_c, info["d_centre"][:B])
        obs, div = sync_copies(env, 2, info, actors=("cube",))
        drift = max(drift, div)
    info = u.evaluate()
    succ, cen = info["success"][:B], info["at_centre"][:B]
    out = np.where(succ.cpu().numpy(), "goal", np.where(cen.cpu().numpy(), "proxy", "neither"))
    cube = u.cube.pose.p[:B].cpu().numpy()
    return dict(outcome=out, d_marker=info["d_marker"][:B].cpu().numpy(), d_centre=info["d_centre"][:B].cpu().numpy(),
                min_d_marker=min_m.cpu().numpy(), min_d_centre=min_c.cpu().numpy(), cube_x=cube[:, 0], cube_y=cube[:, 1],
                cube_z=cube[:, 2], dnorm0=st["dnorm0"].cpu().numpy()), drift


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds_gm", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seeds_control", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--doses", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--n_states", type=int, default=200, help="first n conflict scenes of PTM_worldedit")
    a = p.parse_args()
    C.ensure_dirs()
    dev = torch.device("cuda")
    bank = C.load_bank("PTM_worldedit")
    env = E.make_env("PTM", 2 * a.batch)
    out_csv = os.path.join(C.RESULTS, "steer_PTM_per_state.csv")
    rows, done = [], set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = set(zip(prev.cohort, prev.seed, prev.arm, prev.beta.round(6), prev.rep))
    arms = [("baseline", 0.0, 0)] + [x for b in a.doses for x in [("local", b, 0), ("local", -b, 0)] +
                                     [("random", b, r) for r in range(a.n_random)]]
    centre = np.tile(E.PTM_C, (a.batch, 1))
    for cohort, seed, ck in C.agent_list("PTM", a.seeds_gm, a.seeds_control, init_seeds=()):
        agent = C.build_agent(ck, init_seed=seed, dev=dev)
        for arm, beta, rep in arms:
            if (cohort, seed, arm, round(beta, 6), rep) in done:
                continue
            new = []
            for sl in C.batched(a.n_states, a.batch):
                idx = np.arange(sl.start, sl.stop)
                pad = np.concatenate([idx, np.full(a.batch - len(idx), idx[-1])])
                m = bank["m"][pad]
                E.force_ptm(env, np.tile(bank["qpos_forced"][pad], (2, 1)), np.concatenate([m, centre]),
                            np.tile(m, (2, 1)), np.tile(bank["cube"][pad], (2, 1)))
                r, drift = steer_batch(env, agent, dev, arm, beta, rep, seed)
                for j, i in enumerate(idx):
                    new.append(dict(cohort=cohort, seed=seed, arm=arm, beta=beta, rep=rep, state=int(i), copy_drift=drift,
                                    **{k: (v[j] if k == "outcome" else float(v[j])) for k, v in r.items()}))
            rows += new
            pd.DataFrame(rows).to_csv(out_csv + ".tmp", index=False)
            os.replace(out_csv + ".tmp", out_csv)
            g = pd.DataFrame(new)["outcome"].value_counts(normalize=True)
            print(f"PTM {cohort} s{seed} {arm} b={beta:+.3f} r{rep}: goal {g.get('goal', 0):.3f} centre "
                  f"{g.get('proxy', 0):.3f} |d0| {np.median([x['dnorm0'] for x in new]):.3f} drift {new[-1]['copy_drift']:.1e}",
                  flush=True)
    env.close()


if __name__ == "__main__":
    main()
