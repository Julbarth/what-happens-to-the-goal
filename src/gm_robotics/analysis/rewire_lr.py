"""LocationReach frozen-representation rewiring by behaviour cloning from h_fc / h_pre, scored closed loop on the off-path test."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis.rollouts import T_MAX, lr_first_contact

ARMS = ("fc_head", "pre", "remix", "untrained")


def teacher_seed(k, ctrl_seeds):
    """Every rewire of pair k is cloned from a control agent other than control k, so the Non-GM rewire never
    clones its own native policy (the paper notes this bias for teacher targets at h_fc)."""
    i = ctrl_seeds.index(k) if k in ctrl_seeds else 0
    return ctrl_seeds[(i + 1) % len(ctrl_seeds)]


@torch.no_grad()
def collect(env, agent, cfg, dev, sl_pad):
    """Sampled-action rollouts of one source agent; frames kept up to the first 3 cm marker contact."""
    E.force_lr(env, cfg["qpos"][sl_pad], *E.lr_state({k: v[sl_pad] for k, v in cfg.items()}, ("g", "d")))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    sig = C.sigma(agent)
    alive = torch.ones(u.num_envs, dtype=torch.bool, device=dev)
    rgb, st, ep = [], [], []
    for t in range(T_MAX):
        rgb.append(obs["rgb"][alive].cpu().numpy()), st.append(obs["state"][alive].cpu().numpy())
        ep.append(torch.nonzero(alive).flatten().cpu().numpy())
        mu = C.taps(agent, obs["rgb"], obs["state"])["mu"]
        obs, _, _, _, info = env.step(mu + sig * torch.randn_like(mu))
        alive &= ~info["success"]
        if not alive.any():
            break
    return np.concatenate(rgb), np.concatenate(st).astype(np.float32), np.concatenate(ep)


def build_bc_bank(env, agents, n_ep, bs, dev, seed=4100):
    cfg = E.sample_lr_probe_configs(n_ep * len(agents), seed, "train")
    rgb, st, epi, src = [], [], [], []
    for a_i, (cohort, s, ck) in enumerate(agents):
        agent = C.build_agent(ck, dev=dev)
        for b0 in range(0, n_ep, bs):
            idx = a_i * n_ep + np.arange(b0, min(n_ep, b0 + bs))
            pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
            r, s_, e = collect(env, agent, cfg, dev, pad)
            keep = e < len(idx)
            rgb.append(r[keep]), st.append(s_[keep]), epi.append(idx[e[keep]]), src.append(np.full(keep.sum(), a_i))
        print(f"bc bank: {cohort} s{s} done, frames so far {sum(len(x) for x in st)}", flush=True)
    return dict(rgb=np.concatenate(rgb), state=np.concatenate(st), episode=np.concatenate(epi), source=np.concatenate(src))


class Rewired(nn.Module):
    """Frozen trunk of `base` up to h_pre (and its proprio branch); fresh downstream according to the arm."""

    def __init__(self, base, arm, init_seed):
        super().__init__()
        from gm_robotics.ppo_rgb_gm import layer_init
        self.base, self.arm = base, arm
        torch.manual_seed(init_seed)
        p = C.parts(base)
        if arm in ("pre", "remix"):
            self.fc = nn.Linear(p["fc"].in_features, p["fc"].out_features)
            if arm == "remix":
                self.fc.requires_grad_(False)
        self.head = nn.Sequential(layer_init(nn.Linear(512, 512)), nn.ReLU(),
                                  layer_init(nn.Linear(512, C.N_ACT), std=0.01 * np.sqrt(2)))

    def h_fc(self, h_pre, h_fc_native):
        return h_fc_native if self.arm in ("fc_head", "untrained") else torch.relu(self.fc(h_pre))

    def forward(self, h_pre, h_fc_native, h_state):
        return self.head(torch.cat([self.h_fc(h_pre, h_fc_native), h_state], 1))

    def act(self, rgb, state):
        t = C.taps(self.base, rgb, state)
        return self(t["h_pre"], t["h_fc"], t["h_state"])


@torch.no_grad()
def bank_features(agent, bank, dev, bs=512):
    out = {k: [] for k in ("h_pre", "h_fc", "h_state", "mu")}
    for sl in C.batched(len(bank["state"]), bs):
        t = C.taps(agent, C.to_t(bank["rgb"][sl], dev), C.to_t(bank["state"][sl], dev))
        for k in out:
            out[k].append(t[k].half() if k == "h_pre" else t[k])
    return {k: torch.cat(v) for k, v in out.items()}


def train_bc(model, feats, target, sig_t, episodes, epochs, lr, bs, clip, seed):
    """Paper protocol: 20 epochs Adam 3e-4, batch 256, grad-clip 0.5, 80/20 split (by episode here)."""
    rng = np.random.default_rng(seed)
    ue = np.unique(episodes)
    val_ep = set(rng.choice(ue, size=len(ue) // 5, replace=False).tolist())
    is_val = np.array([e in val_ep for e in episodes])
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    params = [p for p in model.parameters() if p.requires_grad and not any(p is q for q in model.base.parameters())]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = lambda i: ((((model(feats["h_pre"][i].float(), feats["h_fc"][i], feats["h_state"][i]) - target[i]) / sig_t)
                          ** 2).mean())
    log = []
    for ep in range(epochs):
        perm = torch.as_tensor(rng.permutation(tr))
        tl = []
        for b in range(0, len(perm), bs):
            i = perm[b:b + bs]
            loss = loss_fn(i)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            tl.append(loss.item())
        with torch.no_grad():
            vl = np.mean([loss_fn(torch.as_tensor(va[b:b + 4096])).item() for b in range(0, len(va), 4096)])
        log.append(dict(epoch=ep, train_loss=float(np.mean(tl)), val_loss=float(vl)))
    return log, sum(p.numel() for p in params)


def offpath_setup(seed_k, n=500, eval_seed=10_000):
    """Re-derives eval_offpath.py's starts (same generator sequence) and loads its saved goals and distractors."""
    z = np.load(os.path.join(C.ROOT, "results", "onpath", f"offpath_paths_s{seed_k}.npz"))
    bank = torch.as_tensor(E.lr_start_bank("test"))
    g = torch.Generator().manual_seed(eval_seed)
    qpos = bank[torch.randint(len(bank), (n,), generator=g)].numpy()
    return dict(qpos=qpos, goal=z["goal"].astype(np.float32), dxy=z["dxy"].astype(np.float32), valid=z["valid"])


@torch.no_grad()
def closed_loop(env, act_fn, qpos, goal, dxy, bs):
    corner = np.array([*E.CORNER, E.LR_GOAL_Z], np.float32)
    outs = []
    for sl in C.batched(len(qpos), bs):
        idx = np.arange(sl.start, sl.stop)
        pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
        E.force_lr(env, qpos[pad], goal[pad], dxy[pad])
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        path = [u.agent.tcp.pose.p.cpu().numpy()]
        for _ in range(T_MAX):
            obs, *_ = env.step(act_fn(obs["rgb"], obs["state"]))
            path.append(u.agent.tcp.pose.p.cpu().numpy())
        out, _, _ = lr_first_contact(np.stack(path, 1), goal[pad], corner)
        outs.append(out[: len(idx)])
    return np.concatenate(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="paired GM/control seeds")
    p.add_argument("--n_ep", type=int, default=192, help="BC episodes per source agent")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--env_batch", type=int, default=128)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    a = p.parse_args()
    C.ensure_dirs()
    dev = torch.device("cuda")
    out_dir = os.path.join(C.RESULTS, "rewire_LR")
    os.makedirs(out_dir, exist_ok=True)
    env = E.make_env("LR", a.env_batch)
    sources = [("gm", s, C.ckpt_path("LR", "gm", s)) for s in a.seeds] + \
              [("control", s, C.ckpt_path("LR", "control", s)) for s in a.seeds]
    bc_path = os.path.join(C.BANKS, "LR_bc.npz")
    if not os.path.exists(bc_path):
        bank = build_bc_bank(env, sources, a.n_ep, a.env_batch, dev)
        np.savez(bc_path + ".tmp.npz", **bank)
        os.replace(bc_path + ".tmp.npz", bc_path)
    bank = C.load_bank("LR_bc")
    we = C.load_bank("LR_worldedit")
    eval_sets = {"in_region": (we["qpos"], we["goal_corner_d"], we["dxy_corner_d"]),
                 "out_region": (we["qpos"], we["goal_g_d"], we["dxy_g_d"])}
    res_csv = os.path.join(out_dir, "closed_loop_per_episode.csv")
    done = set(pd.read_csv(res_csv)[["cohort", "seed", "arm"]].itertuples(index=False, name=None)) \
        if os.path.exists(res_csv) else set()
    teachers = {}
    for k in a.seeds:
        off = offpath_setup(k)
        sets = dict(eval_sets, offpath=(off["qpos"], off["goal"], off["dxy"]))
        tk = teacher_seed(k, a.seeds)
        if tk not in teachers:
            ta = C.build_agent(C.ckpt_path("LR", "control", tk), dev=dev)
            teachers[tk] = (bank_features(ta, bank, dev)["mu"], C.sigma(ta))
        target, sig_t = teachers[tk]
        for cohort in ("gm", "control"):
            base = C.build_agent(C.ckpt_path("LR", cohort, k), dev=dev)
            arms = ["native"] + a.arms
            if all((cohort, k, arm) in done for arm in arms):
                continue
            feats = bank_features(base, bank, dev)
            for arm in arms:
                if (cohort, k, arm) in done:
                    continue
                if arm == "native":
                    act = lambda rgb, st: C.taps(base, rgb, st)["mu"]
                    n_par, log = 0, []
                else:
                    model = Rewired(base, arm, init_seed=100 + k).to(dev)
                    log, n_par = ([], 0) if arm == "untrained" else train_bc(
                        model, feats, target, sig_t, bank["episode"], a.epochs, a.lr, a.batch, a.clip, seed=k)
                    model.eval()
                    act = model.act
                    pd.DataFrame(log).to_csv(os.path.join(out_dir, f"trainlog_{cohort}_s{k}_{arm}.csv"), index=False)
                rows = []
                for sname, (q, g, d) in sets.items():
                    out = closed_loop(env, act, q, g, d, a.env_batch)
                    rows += [dict(cohort=cohort, seed=k, arm=arm, teacher=tk, set=sname, episode=i, outcome=o,
                                  n_trainable=n_par, final_val_loss=log[-1]["val_loss"] if log else np.nan)
                             for i, o in enumerate(out)]
                df = pd.DataFrame(rows)
                df.to_csv(res_csv, mode="a", header=not os.path.exists(res_csv), index=False)
                print(f"{cohort} s{k} {arm}: " + ", ".join(
                    f"{s} goal {g['outcome'].eq('goal').mean():.3f} proxy {g['outcome'].eq('proxy').mean():.3f}"
                    for s, g in df.groupby("set")), flush=True)
    env.close()


if __name__ == "__main__":
    main()
