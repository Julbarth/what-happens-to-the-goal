"""SortZones rewiring with BC on control-distribution scenes only (two banks, matched frame count), rewire_attr eval and chance."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis import rewire_attr as R
from gm_robotics.analysis.rewire_lr import Rewired, bank_features, train_bc

KIND = "SZ"
OUT = os.path.join(C.RESULTS, "rewire_sz_ctrl2")
HEADS = os.path.join(OUT, "heads")
SEEDS = R.SEEDS[KIND]
BANK2_SEED = 4301
ARMS = ("pre", "pre_masked", "fc_head", "remix")


def bank_file(i):
    return os.path.join(C.BANKS, f"{KIND}_bc_attr.npz" if i == 1 else f"{KIND}_bc_attr{i}.npz")


def head_path(cohort, k, arm):
    return os.path.join(HEADS, f"ctrl2_{cohort}_s{k}_{arm}.pt")


def bc_cfg(n, seed):
    """rewire_attr.bc_configs for SZ with another seed (even = GM-training scenes, cross on green; odd = control)."""
    cfg = E.sample_sz_probe_configs(n, seed)
    gm_var = np.arange(n) % 2 == 0
    cfg["gm_variant"] = gm_var
    cfg["cross_green"] = np.where(gm_var, True, cfg["cross_green"])
    return cfg


def build_bank2(a):
    path = bank_file(2)
    if os.path.exists(path):
        print(f"{path} exists", flush=True)
        return
    dev = torch.device("cuda")
    srcs = [("gm", s) for s in SEEDS["gm"]] + [("control", s) for s in SEEDS["control"]]
    cfg = bc_cfg(a.n_ep * len(srcs), BANK2_SEED)
    env = R.make_env(KIND, 2 * a.env_batch)
    parts = []
    for s_i, (cohort, s) in enumerate(srcs):
        agent = C.build_agent(C.ckpt_path(KIND, cohort, s), dev=dev)
        torch.manual_seed(BANK2_SEED + s_i)
        for b0 in range(0, a.n_ep, a.env_batch):
            idx = s_i * a.n_ep + np.arange(b0, min(a.n_ep, b0 + a.env_batch))
            pad = np.concatenate([idx, np.full(a.env_batch - len(idx), idx[-1])])
            r, _, _ = R.collect(KIND, env, agent, cfg, pad, len(idx), dev)
            r["source"] = np.full(len(r["ep"]), s_i)
            parts.append(r)
        print(f"bank2 {KIND}: {cohort} s{s} done, frames {sum(len(p['ep']) for p in parts)}", flush=True)
    env.close()
    bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    bank["episode"] = bank.pop("ep")
    for k in ("u", "v", "qpos", "gm_variant", "cube", "cross_green"):
        bank["cfg_" + k] = cfg[k]
    np.savez(path + ".tmp.npz", **bank)
    os.replace(path + ".tmp.npz", path)
    print(f"bank2 {KIND}: {len(bank['episode'])} frames, {len(np.unique(bank['episode']))} episodes", flush=True)


def ctrl2_data(masked):
    """Control-distribution episodes (cfg_gm_variant False) of banks 1 and 2."""
    parts, info = [], {}
    for i in (1, 2):
        z = np.load(bank_file(i), mmap_mode="r")
        ep = np.asarray(z["episode"])
        idx = np.nonzero(~np.asarray(z["cfg_gm_variant"])[ep])[0]
        info[f"bank{i}_frames_all"], info[f"bank{i}_frames_ctrl"] = len(ep), len(idx)
        parts.append(dict(rgb=np.ascontiguousarray(z["rgb_masked" if masked else "rgb"][idx]),
                          state=np.asarray(z["state"])[idx], episode=ep[idx] + 100_000 * (i - 1)))
    return parts, info


def train(a):
    dev = torch.device("cuda")
    os.makedirs(HEADS, exist_ok=True)
    os.makedirs(os.path.join(OUT, "trainlogs"), exist_ok=True)
    data, teachers = {}, {}
    for cohort in a.cohorts:
        for k in SEEDS[cohort]:
            for arm in ARMS:
                if (cohort == "init" and arm == "pre_masked") or os.path.exists(head_path(cohort, k, arm)):
                    continue
                msk = arm == "pre_masked"
                if msk not in data:
                    data[msk] = ctrl2_data(msk)
                    print(f"ctrl2 data masked={msk}: {data[msk][1]}", flush=True)
                if False not in data:
                    data[False] = ctrl2_data(False)
                tk = R.pair_teacher(KIND, cohort, k)
                if tk not in teachers:
                    ta = C.build_agent(C.ckpt_path(KIND, "control", tk), dev=dev)
                    teachers[tk] = (torch.cat([bank_features(ta, p, dev)["mu"] for p in data[False][0]]), C.sigma(ta))
                    del ta
                target, sig_t = teachers[tk]
                base = R.base_agent(KIND, cohort, k, dev)
                fs = [bank_features(base, p, dev) for p in data[msk][0]]
                feats = {key: torch.cat([f[key] for f in fs]) for key in fs[0]}
                episodes = np.concatenate([p["episode"] for p in data[msk][0]])
                model = Rewired(base, "pre" if msk else arm, init_seed=100 + k).to(dev)
                log, n_par = train_bc(model, feats, target, sig_t, episodes, a.epochs, a.lr, a.batch, a.clip, seed=k)
                sd = {n: p.detach().cpu() for n, p in model.state_dict().items() if not n.startswith("base.")}
                torch.save(dict(state_dict=sd, arm=arm, cohort=cohort, seed=k, teacher=tk, n_trainable=n_par),
                           head_path(cohort, k, arm))
                pd.DataFrame(log).assign(n_trainable=n_par, teacher=tk, n_frames=len(episodes)).to_csv(
                    os.path.join(OUT, "trainlogs", f"ctrl2_{cohort}_s{k}_{arm}.csv"), index=False)
                print(f"trained SZ ctrl2 {cohort} s{k} {arm} ({len(episodes)} frames): val {log[-1]['val_loss']:.3f} "
                      f"train {log[-1]['train_loss']:.3f}", flush=True)
                del feats, fs, model
                torch.cuda.empty_cache()


def policy(cohort, k, arm, dev):
    base = R.base_agent(KIND, cohort, k, dev)
    model = Rewired(base, "pre" if arm == "pre_masked" else arm, init_seed=100 + k).to(dev)
    ck = torch.load(head_path(cohort, k, arm), map_location="cpu")
    model.load_state_dict({**{n: p for n, p in model.state_dict().items() if n.startswith("base.")}, **ck["state_dict"]})
    model.eval()
    return model.act


def rollout(a):
    dev = torch.device("cuda")
    cfg, _ = R.eval_scenes(KIND)
    env = R.make_env(KIND, a.env_batch)
    csv = os.path.join(OUT, f"{KIND}_per_episode_{a.tag}.csv")
    done = set()
    for f in os.listdir(OUT):
        if f.startswith(f"{KIND}_per_episode_"):
            done |= set(pd.read_csv(os.path.join(OUT, f), usecols=["cohort", "seed", "arm", "cond"]).itertuples(index=False, name=None))
    specs = [(c, k, arm) for c in ("gm", "control", "init") for k in SEEDS[c] for arm in ARMS
             if not (c == "init" and arm == "pre_masked")]
    for j, (cohort, k, arm) in enumerate(specs):
        if j % a.n_workers != a.worker:
            continue
        conds = list(R.CONDS[KIND]) if arm == "pre" else ["standard", "nogoal"]
        act = None
        for cond in conds:
            if (cohort, k, arm, cond) in done:
                continue
            act = act or policy(cohort, k, arm, dev)
            df, _, _, _ = R.run_eval(KIND, env, act, cfg, cond, a.env_batch)
            df = df.assign(env=KIND, variant="ctrl2", cohort=cohort, seed=k, arm=arm, cond=cond)
            df.to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            vc = df.outcome.value_counts(normalize=True)
            print(f"SZ ctrl2 {cohort} s{k} {arm} {cond}: u {vc.get('u', 0):.3f} v {vc.get('v', 0):.3f} "
                  f"neither {vc.get('neither', 0):.3f}", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["bank2", "train", "rollout"])
    p.add_argument("--cohorts", nargs="+", default=["gm", "control", "init"])
    p.add_argument("--n_ep", type=int, default=192)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--env_batch", type=int, default=128)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n_workers", type=int, default=1)
    p.add_argument("--tag", default="w0")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    C.ensure_dirs()
    dict(bank2=build_bank2, train=train, rollout=rollout)[a.step](a)


if __name__ == "__main__":
    main()
