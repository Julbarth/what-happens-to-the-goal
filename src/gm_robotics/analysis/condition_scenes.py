"""Scene images and full cube / TCP paths for the SortZones tie conditions and the AttributeTable colour-configuration test (native agents, deterministic)."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C

OUT = os.path.join(C.RESULTS, "condition_scenes")
SZ_SEEDS = dict(gm=[0, 1, 2], control=[0, 1, 2])
AT_SEEDS = dict(gm=[2, 3, 4], control=[2, 3, 4, 5])
# AT colour configurations on the world-edit bank geometry (u = cube location, v = sphere location).
AT_CONDS = {"red_cube": (("cube", "red"), ("sphere", "blue")),     # GM training configuration
            "blue_cube": (("cube", "blue"), ("sphere", "red")),    # standard conflict
            "both_red": (("cube", "red"), ("sphere", "red")),
            "both_blue": (("cube", "blue"), ("sphere", "blue"))}


@torch.no_grad()
def sz(a):
    """sz_tie protocol, first batch only (125 envs, reset seed 10000, torch seed 10000): the same episodes as the tie
    test's episodes 0-124 and the videos' episodes 0-3."""
    from gm_robotics.analysis.videos_tie_steer import TIE_CONDS, make_tie_env, set_tie
    dev = torch.device("cuda")
    env = make_tie_env(125, render=True)
    u = env.unwrapped
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cond in TIE_CONDS:
        store = {}
        for cohort, seeds in SZ_SEEDS.items():
            for seed in seeds:
                agent = C.build_agent(C.ckpt_path("SZ", cohort, seed), dev=dev)
                set_tie(u, cond)
                torch.manual_seed(10_000)
                obs, _ = env.reset(seed=10_000)
                if "cam" not in store:
                    store["cam"] = env.render()[:8].cpu().numpy()
                    store["obs"] = obs["rgb"][:8].cpu().numpy()
                    store["marked"], store["other"] = u.marked_xy.cpu().numpy(), u.other_xy.cpu().numpy()
                cube, tcp = [u.cube.pose.p.cpu().numpy()], [u.agent.tcp.pose.p.cpu().numpy()]
                for t in range(100):
                    obs, *_ = env.step(agent.get_action(obs, deterministic=True))
                    cube.append(u.cube.pose.p.cpu().numpy())
                    tcp.append(u.agent.tcp.pose.p.cpu().numpy())
                fe = u.first_entry.cpu().numpy()
                store[f"cube_{cohort}_s{seed}"] = np.stack(cube, 1)[..., :2].astype(np.float32)
                store[f"tcp_{cohort}_s{seed}"] = np.stack(tcp, 1).astype(np.float32)
                store[f"res_{cohort}_s{seed}"] = fe
                vc = pd.Series(fe).map({0: "neither", 1: "cross", 2: "other"}).value_counts(normalize=True)
                rows.append(dict(condition=cond, cohort=cohort, seed=seed, **vc.to_dict()))
                print(f"SZ {cond} {cohort} s{seed}: {vc.round(3).to_dict()}", flush=True)
        np.savez_compressed(os.path.join(OUT, f"SZ_{cond}.npz"), **store)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "SZ_outcomes.csv"), index=False)
    env.close()


@torch.no_grad()
def at(a):
    """Native agents on the 512 world-edit bank geometries (u, v, start qpos) under each colour configuration."""
    from gm_robotics.analysis import rewire_attr as RA
    dev = torch.device("cuda")
    for k, v in AT_CONDS.items():
        RA.CONDS["AT"][k] = v
    cfg, _ = RA.eval_scenes("AT")
    env = RA.make_env("AT", a.batch, render=True)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cond in AT_CONDS:
        store = {}
        for cohort, seeds in AT_SEEDS.items():
            for seed in seeds:
                agent = C.build_agent(C.ckpt_path("AT", cohort, seed), dev=dev)
                act = lambda rgb, st: C.taps(agent, rgb, st)["mu"]
                df, pth, fr, _ = RA.run_eval("AT", env, act, cfg, cond, a.batch, keep_frames=8 if "cam" not in store else 0)
                if "cam" not in store:
                    store["cam"], store["obs"] = fr["human"][0], fr["obs"][0]
                    df, pth, _, _ = RA.run_eval("AT", env, act, cfg, cond, a.batch)
                store[f"tcp_{cohort}_s{seed}"] = pth["tcp"].astype(np.float32)
                store[f"res_{cohort}_s{seed}"] = df.outcome.to_numpy().astype("U8")
                vc = df.outcome.value_counts(normalize=True)
                rows.append(dict(condition=cond, cohort=cohort, seed=seed, **vc.to_dict()))
                print(f"AT {cond} {cohort} s{seed}: {vc.round(3).to_dict()}", flush=True)
        store["u"], store["v"] = cfg["u"], cfg["v"]
        np.savez_compressed(os.path.join(OUT, f"AT_{cond}.npz"), **store)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "AT_outcomes.csv"), index=False)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["sz", "at"])
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()
    dict(sz=sz, at=at)[a.step](a)


if __name__ == "__main__":
    main()
