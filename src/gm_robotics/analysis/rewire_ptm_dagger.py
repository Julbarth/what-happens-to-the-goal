"""PushToMarker rewire under on-policy relabeling (DAgger, three rounds) and a matched-training control, Maze recipe."""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis import rewire_ptm as RP
from gm_robotics.analysis.rewire_lr import Rewired, bank_features

OUT = os.path.join(C.RESULTS, "rewire_ptm_dagger")
SITES = {"pre": "pre", "fc": "fc_head"}
# Maze adds 50k frames per round to a 200k base set; the PTM BC bank holds 6 x 192 x 100 = 115,200 frames, so the
# scale-equivalent is 25% of it, 288 episodes of T_MAX steps per round.
N_EP_ROUND = 288
DAG_SEED = 6100
EVAL_SPECS = (("conflict", "standard"), ("ctrlset", "standard"))


def eval_sets():
    """The main rewire's 512 conflict scenes (PTM_worldedit out_marker) and 512 control-layout scenes, as saved by
    rewire_ptm.py, so no bank rebuild can shift them."""
    z = np.load(os.path.join(RP.OUT, "conflict_scenes.npz"))
    conflict = dict(qpos=z["qpos_forced"], m=z["m"], m2=z["m2"], cube=z["cube"])
    z = np.load(os.path.join(RP.OUT, "ctrlset_scenes.npz"))
    return {"conflict": conflict, "ctrlset": {k: z[k] for k in z.files}}


def round_configs(rnd):
    """Fresh control-distribution scenes from train-half starts, the same for every arm within a round (Maze offsets the
    level range per round); disjoint seeds from the BC bank (4400) and the control-layout eval set (5300, test half)."""
    return E.sample_ptm_probe_configs(N_EP_ROUND, DAG_SEED + rnd, "train")


@torch.no_grad()
def collect_onpolicy(env, act, cfg, bs):
    """Deterministic closed loop of the current rewired policy (as in evaluation) on cfg; returns visited frames."""
    n = len(cfg["m"])
    rec = dict(rgb=[], state=[], ep=[])
    for sl in C.batched(n, bs):
        idx = np.arange(sl.start, sl.stop)
        pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
        E.force_ptm(env, cfg["qpos"][pad], cfg["m"][pad], cfg["m"][pad], cfg["cube"][pad])
        obs, _ = env.reset(seed=0)
        for _ in range(RP.T_MAX):
            rec["rgb"].append(obs["rgb"][: len(idx)].cpu().numpy())
            rec["state"].append(obs["state"][: len(idx)].cpu().numpy().astype(np.float32))
            rec["ep"].append(idx)
            obs, *_ = env.step(act(obs["rgb"], obs["state"]))
    return {k: np.concatenate(v) for k, v in rec.items()}


@torch.no_grad()
def teacher_mu(agent, bank, dev, bs=512):
    return torch.cat([C.taps(agent, C.to_t(bank["rgb"][sl], dev), C.to_t(bank["state"][sl], dev))["mu"]
                      for sl in C.batched(len(bank["state"]), bs)])


def base_split(episodes, k):
    """The 80/20 by-episode split of rewire_lr.train_bc for seed k, so base validation episodes stay held out."""
    rng = np.random.default_rng(k)
    ue = np.unique(episodes)
    val_ep = rng.choice(ue, size=len(ue) // 5, replace=False)
    is_val = np.isin(episodes, val_ep)
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def finetune(model, feats, target, sig_t, tr, va, epochs, lr, bs, clip, seed):
    """rewire_lr.train_bc's loss and optimizer on a fixed train index set; the added frames are all in `tr`, so the
    DAgger and matched arms take identical optimizer steps."""
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad and not any(p is q for q in model.base.parameters())]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = lambda i: ((((model(feats["h_pre"][i].float(), feats["h_fc"][i], feats["h_state"][i]) - target[i]) / sig_t)
                          ** 2).mean())
    log = []
    for ep in range(epochs):
        perm = torch.as_tensor(rng.permutation(tr))
        tl = []
        for b in range(0, len(perm), bs):
            loss = loss_fn(perm[b:b + bs])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            tl.append(loss.item())
        with torch.no_grad():
            vl = np.mean([loss_fn(torch.as_tensor(va[b:b + 4096])).item() for b in range(0, len(va), 4096)])
        log.append(dict(epoch=ep, train_loss=float(np.mean(tl)), val_loss=float(vl)))
    return log


def cat_feats(a, b):
    return {k: torch.cat([a[k], b[k]]) for k in a}


def take(f, idx):
    i = torch.as_tensor(idx, device=f["h_state"].device)
    return {k: v[i] for k, v in f.items()}


def evaluate(env, model, sets, bs, tag, rows, ep_rows):
    act = model.act
    for sname, cond in EVAL_SPECS:
        df, *_ = RP.run_eval(env, act, sets[sname], cond, bs)
        vc = df.outcome.value_counts(normalize=True)
        rows.append(dict(tag, set=sname, cond=cond, n=len(df), goal=vc.get("goal", 0.0), proxy=vc.get("proxy", 0.0),
                         neither=vc.get("neither", 0.0), d_marker=df.d_marker.mean()))
        ep_rows.append(df[["episode", "outcome", "d_marker"]].assign(**tag, set=sname, cond=cond))
        print(f"  {tag['cohort']} s{tag['seed']} {tag['site']} {tag['mode']} r{tag['round']} {sname}: "
              f"goal {rows[-1]['goal']:.3f} proxy {rows[-1]['proxy']:.3f}", flush=True)


def run(a):
    dev = torch.device("cuda")
    os.makedirs(os.path.join(OUT, "heads"), exist_ok=True)
    sets = eval_sets()
    if a.max_eval:
        sets = {k: {f: v[: a.max_eval] for f, v in cfg.items()} for k, cfg in sets.items()}
    env = E.make_env("PTM", a.env_batch)
    bank = RP.load_bc()
    for cohort, k in a.trunks:
        tk = RP.pair_teacher(k)
        teacher = C.build_agent(C.ckpt_path("PTM", "control", tk), dev=dev)
        sig_t = C.sigma(teacher)
        base = RP.base_agent(cohort, k, dev)
        t0 = time.time()
        feats0 = bank_features(base, bank, dev)
        target0 = teacher_mu(teacher, bank, dev)
        tr0, va0 = base_split(bank["episode"], k)
        print(f"{cohort} s{k}: base features {len(tr0)} train / {len(va0)} val frames ({time.time() - t0:.0f}s), "
              f"teacher control s{tk}", flush=True)
        for site in a.sites:
            arm = SITES[site]
            r0 = None
            for mode in a.modes:
                fn = os.path.join(OUT, "per_round", f"{cohort}_s{k}_{site}_{mode}.csv")
                if os.path.exists(fn):
                    continue
                model = Rewired(base, arm, init_seed=100 + k).to(dev)
                ck = torch.load(RP.head_path(cohort, k, arm), map_location="cpu")
                model.load_state_dict({**{n: p for n, p in model.state_dict().items() if n.startswith("base.")},
                                       **ck["state_dict"]})
                model.eval()
                tag = dict(cohort=cohort, seed=k, site=site, mode=mode, teacher=tk)
                if r0 is None:
                    r0 = ([], [])
                    evaluate(env, model, sets, a.env_batch, dict(tag, round=0, n_train=len(tr0), n_added=0), *r0)
                # Round 0 is the untouched main-rewire head, shared by both modes.
                rows = [dict(r, mode=mode) for r in r0[0]]
                ep_rows = [d.assign(mode=mode) for d in r0[1]]
                feats, target, tr = feats0, target0, tr0
                rng_m = np.random.RandomState(1000 * k + (0 if site == "pre" else 1))
                for rnd in range(1, a.rounds + 1):
                    t1 = time.time()
                    if mode == "dagger":
                        cfg = round_configs(rnd)
                        if a.max_eval:
                            cfg = {f: v[: a.env_batch] for f, v in cfg.items()}
                        r = collect_onpolicy(env, model.act, cfg, a.env_batch)
                        f_new = bank_features(base, r, dev)
                        y_new = teacher_mu(teacher, r, dev)
                    else:
                        n_add = (N_EP_ROUND if not a.max_eval else a.env_batch) * RP.T_MAX
                        idx = rng_m.choice(tr0, size=n_add, replace=True)
                        f_new, y_new = take(feats0, idx), target0[torch.as_tensor(idx, device=dev)]
                    n_old = len(target)
                    feats, target = cat_feats(feats, f_new), torch.cat([target, y_new])
                    tr = np.concatenate([tr, np.arange(n_old, n_old + len(y_new))])
                    model.train()
                    log = finetune(model, feats, target, sig_t, tr, va0, a.epochs, a.lr, a.batch, a.clip,
                                   seed=10 * k + rnd)
                    model.eval()
                    torch.save(dict(state_dict={n: p.detach().cpu() for n, p in model.state_dict().items()
                                                if not n.startswith("base.")}, round=rnd, **tag),
                               os.path.join(OUT, "heads", f"{cohort}_s{k}_{site}_{mode}_r{rnd}.pt"))
                    evaluate(env, model, sets, a.env_batch,
                             dict(tag, round=rnd, n_train=len(tr), n_added=len(y_new),
                                  train_loss=log[-1]["train_loss"], val_loss=log[-1]["val_loss"]), rows, ep_rows)
                    print(f"  round {rnd} {mode}: +{len(y_new)} frames -> {len(tr)} train, val {log[-1]['val_loss']:.3f} "
                          f"({time.time() - t1:.0f}s)", flush=True)
                os.makedirs(os.path.join(OUT, "per_round"), exist_ok=True)
                pd.concat(ep_rows).to_csv(fn.replace(".csv", "_episodes.csv"), index=False)
                pd.DataFrame(rows).to_csv(fn + ".tmp", index=False)
                os.replace(fn + ".tmp", fn)
                del model, feats, target
                torch.cuda.empty_cache()
        del feats0, target0, base, teacher
        torch.cuda.empty_cache()
    env.close()


def summarize():
    """per_seed.csv (every arm, round, set) and summary.csv (mean and 95% t-CI across the three trunks)."""
    import glob
    from scipy import stats
    fs = sorted(f for f in glob.glob(os.path.join(OUT, "per_round", "*.csv")) if not f.endswith("_episodes.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    g = df.groupby(["site", "cohort", "mode", "round", "set"]).goal
    sm = g.agg(["mean", "std", "count"]).reset_index()
    sm["ci95_half"] = stats.t.ppf(0.975, sm["count"] - 1) * sm["std"] / np.sqrt(sm["count"])
    sm.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(sm[(sm["round"].isin([0, 3]))].round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trunks", nargs="+", default=["gm:0", "gm:1", "gm:2", "control:0", "control:1", "control:2",
                                                    "init:0", "init:1", "init:2"], help="cohort:seed")
    p.add_argument("--sites", nargs="+", default=["pre", "fc"], choices=list(SITES))
    p.add_argument("--modes", nargs="+", default=["dagger", "matched"], choices=["dagger", "matched"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--env_batch", type=int, default=128)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--max_eval", type=int, default=None, help="smoke tests only")
    a = p.parse_args()
    a.trunks = [(t.split(":")[0], int(t.split(":")[1])) for t in a.trunks]
    C.ensure_dirs()
    summarize() if a.summarize else run(a)


if __name__ == "__main__":
    main()
