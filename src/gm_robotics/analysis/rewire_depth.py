"""Per-layer rewiring: frozen trunk up to a layer, fresh downstream conv/fc/head trained by BC, scored closed loop."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis.rewire_lr import offpath_setup, teacher_seed, train_bc
from gm_robotics.analysis.rollouts import lr_first_contact

OUT = os.path.join(C.RESULTS, "depth", "rewire")
LAYERS = ("conv1", "conv2")
# ctrl2 = AT/SZ BC data restricted to control-distribution episodes of BC banks 1 and 2, because in the
# GM-distribution half of bank 1 the proxy colour predicts the goal object.
CTRL2_CONDS = {"AT": ("standard", "nogoal_sph"), "SZ": ("standard", "nogoal")}
SEEDS = {"AT": dict(gm=[2, 3, 4], control=[2, 3, 4, 5], init=[2, 3, 4]),
         "SZ": dict(gm=[0, 1, 2], control=[0, 1, 2], init=[0, 1, 2]),
         "LR": dict(gm=[1, 2, 3], control=[1, 2, 3], init=[1, 2, 3]),
         "PTM": dict(gm=[0, 1, 2], control=[0, 1, 2], init=[0, 1, 2])}
CONDS = {"AT": ("standard", "nogoal_sph", "nogoal_cube", "noproxy_blue", "noproxy_red"),
         "SZ": ("standard", "nogoal", "noproxy_green", "noproxy_purple"),
         "LR": ("standard", "decoupled", "nomarker"),
         "PTM": ("conflict:standard", "conflict:nomarker", "conflict:decoupled", "ctrlset:standard", "ctrlset:nomarker")}


class DeepRewired(nn.Module):
    """Frozen trunk of `base` up to `layer` (post-ReLU) and its proprio branch; fresh conv layers after it, fresh
    9216->256 fc and the rewire_lr head (default PyTorch init for conv/fc as in NatureCNN, orthogonal head)."""

    def __init__(self, base, layer, init_seed):
        super().__init__()
        from gm_robotics.ppo_rgb_gm import layer_init
        self.base, self.layer = base, layer
        torch.manual_seed(init_seed)
        convs = {"conv1": [nn.Conv2d(32, 64, 4, 2), nn.Conv2d(64, 64, 3, 1)], "conv2": [nn.Conv2d(64, 64, 3, 1)]}
        self.convs = nn.ModuleList(convs.get(layer, []))
        self.fc = nn.Linear(9216, 256) if layer != "h_fc" else None
        self.head = nn.Sequential(layer_init(nn.Linear(512, 512)), nn.ReLU(),
                                  layer_init(nn.Linear(512, C.N_ACT), std=0.01 * np.sqrt(2)))

    def forward(self, h_in, _unused, h_state):
        # PTM layer activations do not fit on the GPU and stay in host memory
        x = h_in.to(h_state.device)
        for c in self.convs:
            x = torch.relu(c(x))
        h = x if self.fc is None else torch.relu(self.fc(x.flatten(1)))
        return self.head(torch.cat([h, h_state], 1))

    def act(self, rgb, state):
        t = C.taps(self.base, rgb, state)
        return self(t[self.layer], None, t["h_state"])


def base_agent(env, cohort, k, dev):
    if cohort == "init":
        return C.build_agent(None, init_seed=k, dev=dev)
    return C.build_agent(C.ckpt_path(env, cohort, k), dev=dev)


def pair_teacher(env, k):
    return teacher_seed(k, SEEDS[env]["control"])


def load_bc(env, masked, bank="orig"):
    if bank == "ctrl2":
        parts = []
        for i in (1, 2):
            z = np.load(os.path.join(C.BANKS, f"{env}_bc_attr{'' if i == 1 else 2}.npz"), mmap_mode="r")
            ep = np.asarray(z["episode"])
            idx = np.nonzero(~np.asarray(z["cfg_gm_variant"])[ep])[0]
            parts.append(dict(rgb=np.ascontiguousarray(z["rgb_masked" if masked else "rgb"][idx]),
                              state=np.asarray(z["state"])[idx], episode=ep[idx] + 100_000 * (i - 1)))
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if env == "PTM":
        z = np.load(os.path.join(C.BANKS, "PTM_bc.npz"), mmap_mode="r")
        return dict(rgb=z["rgb_masked" if masked else "rgb"], state=np.asarray(z["state"]), episode=np.asarray(z["episode"]))
    if env == "LR":
        b = C.load_bank("LR_bc")
        if masked:
            b["rgb"] = C.load_bank("LR_bc_nomarker")["rgb"]
        return b
    z = np.load(os.path.join(C.BANKS, f"{env}_bc_attr.npz"), mmap_mode="r")
    return dict(rgb=z["rgb_masked" if masked else "rgb"], state=np.asarray(z["state"]), episode=np.asarray(z["episode"]))


@torch.no_grad()
def layer_feats(agent, bank, layer, dev, bs=512, host=False):
    acts, hs, mu = [], [], []
    for sl in C.batched(len(bank["state"]), bs):
        t = C.taps(agent, C.to_t(np.ascontiguousarray(bank["rgb"][sl]), dev), C.to_t(bank["state"][sl], dev))
        if layer:
            acts.append(t[layer].half().cpu() if host else t[layer].half())
        hs.append(t["h_state"]), mu.append(t["mu"])
    out = dict(h_state=torch.cat(hs), mu=torch.cat(mu))
    if layer:
        out["h_pre"] = torch.cat(acts)
        out["h_fc"] = torch.zeros(len(out["mu"]), 1, device=dev)
    return out


def head_path(env, cohort, k, arm):
    return os.path.join(OUT, "heads", f"{env}_{cohort}_s{k}_{arm}.pt")


def arms_for(cohort):
    return [f"{l}{m}" for l in LAYERS for m in ("", "_masked") if not (cohort == "init" and m)]


def set_bank(bank, layers):
    global OUT, LAYERS
    LAYERS = tuple(layers)
    if bank == "ctrl2":
        OUT = os.path.join(C.RESULTS, "depth", "rewire_ctrl2")


def train(a):
    dev = torch.device("cuda")
    os.makedirs(os.path.join(OUT, "heads"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "trainlogs"), exist_ok=True)
    for masked in (False, True):
        todo = [(c, k, arm) for c in a.cohorts for k in SEEDS[a.env][c] for arm in arms_for(c)
                if arm.endswith("_masked") == masked and not os.path.exists(head_path(a.env, c, k, arm))]
        if not todo:
            continue
        bank = load_bc(a.env, masked, a.bank)
        clean = bank if not masked else load_bc(a.env, False, a.bank)
        teachers = {}
        for cohort, k, arm in todo:
            tk = pair_teacher(a.env, k)
            if tk not in teachers:
                # teacher labels always come from the unmasked frames (rewire_check / rewire_attr protocol)
                ta = C.build_agent(C.ckpt_path(a.env, "control", tk), dev=dev)
                teachers[tk] = (layer_feats(ta, clean, None, dev)["mu"], C.sigma(ta))
                del ta
            target, sig_t = teachers[tk]
            layer = arm.replace("_masked", "")
            base = base_agent(a.env, cohort, k, dev)
            feats = layer_feats(base, bank, layer, dev, host=a.env == "PTM")
            model = DeepRewired(base, layer, init_seed=100 + k).to(dev)
            log, n_par = train_bc(model, feats, target, sig_t, bank["episode"], a.epochs, a.lr, a.batch, a.clip, seed=k)
            sd = {n: p.detach().cpu() for n, p in model.state_dict().items() if not n.startswith("base.")}
            torch.save(dict(state_dict=sd, arm=arm, cohort=cohort, seed=k, teacher=tk, n_trainable=n_par),
                       head_path(a.env, cohort, k, arm))
            pd.DataFrame(log).assign(n_trainable=n_par, teacher=tk).to_csv(
                os.path.join(OUT, "trainlogs", f"{a.env}_{cohort}_s{k}_{arm}.csv"), index=False)
            print(f"trained {a.env} {cohort} s{k} {arm} (teacher control s{tk}, {n_par} params): "
                  f"val {log[-1]['val_loss']:.3f} train {log[-1]['train_loss']:.3f}", flush=True)
            del feats, model
            torch.cuda.empty_cache()
        del bank, clean


def load_policy(env, cohort, k, arm, dev):
    base = base_agent(env, cohort, k, dev)
    model = DeepRewired(base, arm.replace("_masked", ""), init_seed=100 + k).to(dev)
    ck = torch.load(head_path(env, cohort, k, arm), map_location="cpu")
    model.load_state_dict({**{n: p for n, p in model.state_dict().items() if n.startswith("base.")}, **ck["state_dict"]})
    return model.eval().act


def rollout(a):
    dev = torch.device("cuda")
    csv = os.path.join(OUT, f"{a.env}_per_episode.csv")
    done = set()
    if os.path.exists(csv):
        done = set(pd.read_csv(csv, usecols=["cohort", "seed", "arm", "cond"]).itertuples(index=False, name=None))
    if a.env == "PTM":
        from gm_robotics.analysis import rewire_ptm as RP
        env = E.make_env("PTM", a.env_batch)
        ptm_sets = {nm: RP.eval_scenes(nm)[0] for nm in ("conflict", "ctrlset")}
    elif a.env == "LR":
        from gm_robotics.analysis.rewire_check import condition, run_paths
        env = E.make_env("LR", a.env_batch)
        corner = np.array([*E.CORNER, E.LR_GOAL_Z], np.float32)
    else:
        from gm_robotics.analysis import rewire_attr as RA
        env = RA.make_env(a.env, a.env_batch)
        cfg, _ = RA.eval_scenes(a.env)
    for cohort in a.cohorts:
        for k in SEEDS[a.env][cohort]:
            off = offpath_setup(k) if a.env == "LR" else None
            for arm in arms_for(cohort):
                act = None
                full = CTRL2_CONDS[a.env] if a.bank == "ctrl2" else CONDS[a.env]
                conds = full if not arm.endswith("_masked") else tuple(c for c in full if c.endswith("standard"))
                for cond in conds:
                    if (cohort, k, arm, cond) in done:
                        continue
                    act = act or load_policy(a.env, cohort, k, arm, dev)
                    if a.env == "PTM":
                        sname, c = cond.split(":")
                        df = RP.run_eval(env, act, ptm_sets[sname], c, a.env_batch)[0]
                        vc = df.outcome.value_counts(normalize=True)
                        msg = f"goal {vc.get('goal', 0):.3f} centre {vc.get('proxy', 0):.3f}"
                    elif a.env == "LR":
                        marker, scored = condition(off, cond, k)
                        path, _ = run_paths(env, act, off["qpos"], marker, off["dxy"], a.env_batch)
                        o_s, t_s, t_c = lr_first_contact(path, scored, corner)
                        o_m, t_m, _ = lr_first_contact(path, marker, corner)
                        df = pd.DataFrame(dict(episode=np.arange(len(path)), reach_scored=o_s == "goal",
                                               reach_marker=o_m == "goal", reach_corner=o_s == "proxy",
                                               t_scored=t_s, t_marker=t_m, t_corner=t_c,
                                               final_x=path[:, -1, 0], final_y=path[:, -1, 1]))
                        msg = f"scored {df.reach_scored.mean():.3f} marker {df.reach_marker.mean():.3f} corner {df.reach_corner.mean():.3f}"
                    else:
                        df, _, _, _ = RA.run_eval(a.env, env, act, cfg, cond, a.env_batch)
                        vc = df.outcome.value_counts(normalize=True)
                        msg = f"u {vc.get('u', 0):.3f} v {vc.get('v', 0):.3f} neither {vc.get('neither', 0):.3f}"
                    df = df.assign(env=a.env, cohort=cohort, seed=k, arm=arm, cond=cond, teacher=pair_teacher(a.env, k))
                    df.to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
                    print(f"{a.env} {cohort} s{k} {arm} {cond}: {msg}", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["train", "rollout"])
    p.add_argument("--env", choices=["AT", "SZ", "LR", "PTM"], required=True)
    p.add_argument("--cohorts", nargs="+", default=["gm", "control", "init"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--env_batch", type=int, default=128)
    p.add_argument("--bank", choices=["orig", "ctrl2"], default="orig")
    p.add_argument("--layers", nargs="+", default=list(LAYERS))
    a = p.parse_args()
    assert not (a.bank == "ctrl2" and a.env == "LR"), "LR BC bank has no GM-scene half"
    set_bank(a.bank, a.layers)
    os.makedirs(OUT, exist_ok=True)
    dict(train=train, rollout=rollout)[a.step](a)


if __name__ == "__main__":
    main()
