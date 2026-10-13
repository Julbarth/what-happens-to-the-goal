"""Competition geometry on the AT / SZ banks (strict t=0, decision frames), with per-state pulls and d_comp cosine matrices."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
def _stub_sim_imports():
    """ppo_rgb_gm imports the simulator stack at module level; Agent itself only needs torch, so outside the sim venv
    those modules are replaced by empty stand-ins."""
    import importlib.util
    import types
    for name in ("gymnasium", "tyro", "mani_skill", "mani_skill.envs", "mani_skill.utils", "mani_skill.utils.gym_utils",
                 "mani_skill.utils.wrappers", "mani_skill.utils.wrappers.flatten", "mani_skill.utils.wrappers.record",
                 "mani_skill.vector", "mani_skill.vector.wrappers", "mani_skill.vector.wrappers.gymnasium",
                 "gm_robotics.envs"):
        top = name.split(".")[0]
        if name in sys.modules or (top == name and importlib.util.find_spec(top) is not None and top != "gm_robotics"):
            continue
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: type(attr, (), {})
        sys.modules[name] = m


try:
    import gymnasium  # noqa: F401
except ImportError:
    _stub_sim_imports()
from gm_robotics.analysis import common as C
from gm_robotics.analysis import axis_geometry as AG
from gm_robotics.analysis import bank_audit as BA

OUT = os.path.join(C.RESULTS, "geometry_final")


def run_one(env, b, view, fit_l, ev_l, agent, dev):
    """Same quantities as bank_audit.geometry_one, plus per-state signed pulls on
    the held-out states and the fit-state matrices ordered by (goal side, mirror pair)."""
    sign = view["sign"].astype(int)
    g = AG.agent_geometry(agent, view, dev, env)
    dnp = g["d_comp"].cpu().numpy().astype(np.float64)
    u = BA.AX.fit_u_comp({+1: [dnp[i] for i in fit_l if sign[i] > 0], -1: [dnp[i] for i in fit_l if sign[i] < 0]})
    ut = torch.as_tensor(u, dtype=torch.float32, device=dev)
    sg = torch.as_tensor(sign, dtype=torch.float32, device=dev)
    # sign * u points toward the goal candidate: the colour swap pulls positive, the goal-cue swap negative.
    a_prox = (AG.alpha_f(g["Jw"], ut, g["d_prox"]) * sg).cpu().numpy()
    a_star = (AG.alpha_f(g["Jw"], ut, g["d_star"]) * sg).cpu().numpy()
    cw = AG.cos_e(g["d_star"], g["d_prox"]).cpu().numpy()
    cfw = AG.cos_f_same_state(g["Jw"], g["d_star"], g["d_prox"]).cpu().numpy()
    pair = b["pair_v"]
    order = sorted(fit_l, key=lambda i: (-sign[i], pair[i]))
    cs, mats = AG.cross_state(g, np.asarray(order), sign)
    ev = np.asarray(ev_l)
    r = dict(cosE_within=float(np.median(cw)), cosF_within=float(np.median(cfw)),
             cosE_within_mean=float(cw.mean()), cosF_within_mean=float(cfw.mean()), **cs,
             pull_proxy=float(np.median(np.abs(a_prox[ev]))), pull_goal=float(np.median(np.abs(a_star[ev]))),
             pull_proxy_mean=float(a_prox[ev].mean()), pull_goal_mean=float(-a_star[ev].mean()))
    r["pull_ratio"] = r["pull_proxy"] / max(r["pull_goal"], 1e-12)
    r["pull_ratio_of_means"] = r["pull_proxy_mean"] / r["pull_goal_mean"]
    st = pd.DataFrame(dict(pos=np.arange(len(sign)), sign=sign, pair=pair, cosE_within=cw, cosF_within=cfw,
                           alpha_goal=a_star, alpha_prox=a_prox, eval=np.isin(np.arange(len(sign)), ev)))
    return r, st, dict(order=np.asarray(order), **mats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=["AT", "SZ"])
    p.add_argument("--variants", nargs="+", default=["strict", "decision"])
    p.add_argument("--matched", nargs="+", type=int, default=[0, 1], help="0 = raw edits, 1 = pixel-matched")
    p.add_argument("--at_ckpt_dir", default=os.path.join(C.ROOT, "checkpoints", "analysis"))
    p.add_argument("--sz_ckpt_dir", default=os.environ.get("SZ_CKPT_DIR", C.CKPT_DIRS["SZ"]))
    a = p.parse_args()
    C.CKPT_DIRS.update(AT=a.at_ckpt_dir, SZ=a.sz_ckpt_dir)
    torch.manual_seed(0)
    dev = torch.device("cpu")
    os.makedirs(os.path.join(OUT, "matrices"), exist_ok=True)
    for env in a.envs:
        agents = [("gm", s, C.ckpt_path(env, "gm", s)) for s in BA.SEEDS[env]["gm"]]
        agents += [("control", s, C.ckpt_path(env, "control", s)) for s in BA.SEEDS[env]["control"]]
        agents += [("init", s, None) for s in (0, 1, 2)]
        built = {(c, s): C.build_agent(ck, init_seed=s, dev=dev) for c, s, ck in agents}
        rows, states = [], []
        for variant in a.variants:
            b = C.load_bank(os.path.basename(BA.bank_path(env, variant))[:-4])
            label, idx, fit, ev = next(BA.bank_views(env, variant, b))
            pos = {int(i): j for j, i in enumerate(idx)}
            fit_l, ev_l = [pos[int(i)] for i in fit], [pos[int(i)] for i in ev]
            b["pair_v"] = b["pair"][idx]
            for matched in a.matched:
                lab = label + ("_pxmatched" if matched else "")
                view = BA.make_view(env, b, idx, bool(matched))
                for cohort, seed, _ in agents:
                    r, st, mats = run_one(env, b, view, fit_l, ev_l, built[(cohort, seed)], dev)
                    rows.append(dict(bank=lab, cohort=cohort, seed=seed, n_states=len(idx), n_fit=len(fit_l),
                                     n_eval=len(ev_l), **r))
                    st.insert(0, "seed", seed), st.insert(0, "cohort", cohort), st.insert(0, "bank", lab)
                    st["state"] = idx[st.pos.to_numpy()]
                    states.append(st)
                    np.savez_compressed(os.path.join(OUT, "matrices", f"{env}_{lab}_{cohort}_s{seed}.npz"), **mats)
                    print(env, lab, cohort, seed, {k: round(v, 3) for k, v in r.items()
                                                   if k in ("cosF_within_mean", "cosF_same", "pull_ratio")}, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{env}_per_seed.csv"), index=False)
        pd.concat(states).to_csv(os.path.join(OUT, f"{env}_per_state.csv"), index=False)


if __name__ == "__main__":
    main()
