"""Stricter frozen-trunk rewires (linear readouts, layout-disjoint and low-data BC) in Attribute Maze and Coin."""
import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from procgen_paths import PROCGEN_ROOT, add_procgen_paths, ckpt_dir

HERE = Path(__file__).resolve().parent
TRUNKS = {
    "am": dict(gm=[6313, 6315, 6317], nongm=[4371, 4372, 4373], rand=[0, 1, 2]),
    "coin": dict(gm=["6033_gm_20M", "6037_gm_120M", "6051_gm_55M"], nongm=["7001_teach", "7002_teach", "7003_teach"],
                 rand=[0, 1, 2]),
}
READOUTS = ("mlp", "lin_pre", "lin_fc", "lin_pre5", "lin_fc5")
# readout -> (Adam lr, multiple of the paper's step budget). *5 = convergence check of the linear readouts.
READOUT_CFG = {"mlp": (3e-4, 1), "lin_pre": (3e-4, 1), "lin_fc": (3e-4, 1), "lin_pre5": (1e-3, 5), "lin_fc5": (1e-3, 5)}
# data condition -> (base bank, fraction of base levels kept). AM "strict" drops every bank level whose layout occurs
# among the 1000 eval levels; Coin's bank (level seeds 0..1e5) is already seed-disjoint from its eval (1e6..), and its
# "strict" drops every bank episode containing a frame byte-identical to an eval start frame (coin_strict).
_D = {"full": ("full", 1.0), "strict": ("strict", 1.0), "strict10": ("strict", 0.1), "strict1": ("strict", 0.01)}
DATA = {"am": _D, "coin": _D}
BC = dict(batch_size=256, lr=3e-4, clip=0.5, ref_epochs=20)
INIT_SEED = 0
SUBSET_SEED, VAL_SEED = 0, 1
AM_EVAL_START, AM_EVAL_N, AM_MAX_STEPS = 100_000, 1000, 256
COIN_EVAL_N, COIN_MAX_STEPS = 1000, 1000


def trunk_ids(env):
    return [f"{c}{i}" for c in ("gm", "nongm", "rand") for i in range(3)]


def digest(u8):
    return hashlib.blake2b(np.ascontiguousarray(u8).tobytes(), digest_size=16).digest()


# ------------------------------------------------------------------ env glue

def env_imports(env, root):
    if env == "am":
        import am_oracle_fix as AF
        AF.pod_setup(Path(root))
        import head_rewire_ras_maze as hrm
        return AF, hrm
    add_procgen_paths(root)
    import run_coin_rewire_aligned as RA
    import run_coin_behaviour as SA
    return RA, SA


def load_trunk(env, trunk, root, device):
    """rand<k> = untrained network drawn under torch seed 1000 + k (the random-trunk convention)."""
    import torch
    cohort, i = trunk[:-1], int(trunk[-1])
    key = TRUNKS[env][cohort][i]
    M, N = env_imports(env, root)
    if env == "am":
        if cohort == "rand":
            torch.manual_seed(1000 + key)
            pol = N.load_ras_policy(None, device, random_init=True)
        else:
            pol = N.load_ras_policy(M.ckpt_path(Path(root), key, "gm" if cohort == "gm" else "teach"), device)
    else:
        if cohort == "rand":
            torch.manual_seed(1000 + key)
            pol = N.load_init_policy(device)
        else:
            pol = N.load_policy(ckpt_dir(root, "coin") / f"{key}.pth", device)
    pol.eval()
    for p in pol.parameters():
        p.requires_grad = False
    return pol


def h_pre(policy, x):
    import torch.nn.functional as F
    e = policy.embedder
    return F.relu(e.block3(e.block2(e.block1(x)))).flatten(1)


def make_readout(kind, n_act):
    import torch
    import torch.nn as nn

    class Readout(nn.Module):
        """mlp = the paper's fresh fc (2048->256, xavier) + ReLU + head (orth 0.01); lin_* = one linear layer."""

        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind == "mlp":
                self.fc = nn.Linear(2048, 256)
                nn.init.xavier_uniform_(self.fc.weight)
                nn.init.zeros_(self.fc.bias)
            self.head = nn.Linear(2048 if kind in ("lin_pre", "lin_pre5") else 256, n_act)
            nn.init.orthogonal_(self.head.weight, gain=0.01)
            nn.init.zeros_(self.head.bias)

        def forward(self, h):
            if self.kind == "mlp":
                h = torch.relu(self.fc(h))
            return self.head(h)

    torch.manual_seed(INIT_SEED)
    return Readout()


def make_policy(trunk_policy, readout):
    """Rollout wrapper with the CategoricalPolicy call signature: frozen trunk -> h_pre (-> trunk's own fc + ReLU for
    lin_fc) -> readout. The same h_pre function feeds BC caching, so train and rollout features are identical."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    class Rewired(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedder = trunk_policy.embedder
            self.readout = readout

        def forward(self, x, hx, masks):
            h = h_pre(self, x)
            if self.readout.kind.startswith("lin_fc"):
                h = F.relu(self.embedder.fc(h))
            logits = self.readout(h)
            return Categorical(logits=F.log_softmax(logits, dim=1)), torch.zeros(len(x), device=x.device), hx

    return Rewired().eval()


# ------------------------------------------------------------------ banks and subsets

def work_env(a):
    d = Path(a.work) / a.env
    d.mkdir(parents=True, exist_ok=True)
    return d


def am_prep(a):
    """Frame digests of the black fixed-oracle bank, eval-level start digests, and the layout-disjoint (strict) mask.
    A bank level is dropped if its start frame equals the start frame of any of the 1000 eval levels (same maze, same
    objects, same spawn) or if any of its frames equals an eval start frame."""
    from multiprocessing import Pool
    out = work_env(a)
    bank = Path(a.scratch) / "bank" / "black"
    obs = np.load(bank / "obs_u8.npy", mmap_mode="r")
    lid = np.load(bank / "level_id.npy")
    if (out / "bank_hash.npy").exists():
        hashes = np.load(out / "bank_hash.npy")
    else:
        hashes = np.array([digest(obs[i]) for i in range(len(obs))], dtype="S16")
        np.save(out / "bank_hash.npy", hashes)
    with Pool(a.parallel, maxtasksperchild=1) as pool:
        jobs = [(a.root, lo, min(lo + 50, AM_EVAL_START + AM_EVAL_N)) for lo in range(AM_EVAL_START, AM_EVAL_START + AM_EVAL_N, 50)]
        ev = sorted(sum(pool.map(_am_start_digests, jobs), []))
    ev_set = {h for _, h in ev}
    starts = np.r_[0, np.flatnonzero(np.diff(lid) != 0) + 1]
    bad_frame = np.isin(hashes, np.array(sorted(ev_set), dtype="S16"))
    bad_level = np.isin(lid, np.unique(lid[bad_frame])) | np.isin(lid, lid[starts][np.isin(hashes[starts], list(ev_set))])
    strict = ~bad_level
    np.save(out / "strict_keep.npy", strict)
    with open(out / "eval_start_digests.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "digest", "in_full_bank", "in_strict_bank"])
        full_set, strict_set = set(hashes.tolist()), set(hashes[strict].tolist())
        for lv, h in ev:
            w.writerow([lv, h.hex(), h in full_set, h in strict_set])
    info = dict(n_frames=int(len(lid)), n_levels=int(len(np.unique(lid))), n_unique_frames=int(len(set(hashes.tolist()))),
                n_unique_start_frames=int(len(set(hashes[starts].tolist()))), n_eval_layouts=len(ev_set),
                eval_start_in_full_bank=float(np.mean([h in full_set for _, h in ev])),
                eval_start_in_strict_bank=float(np.mean([h in strict_set for _, h in ev])),
                strict_frames=int(strict.sum()), strict_levels=int(len(np.unique(lid[strict]))),
                dropped_levels=int(len(np.unique(lid[~strict]))))
    (out / "prep.json").write_text(json.dumps(info, indent=1))
    print(info, flush=True)


def _am_start_digests(args):
    root, lo, hi = args
    env_imports("am", root)
    from gm_geometry.integrations.colour_shape import make_duel_env
    res = []
    for lv in range(lo, hi):
        env = make_duel_env(1, 1, 1, "red_line_diag", "yellow_gem", start_level=lv, use_backgrounds=False)
        res.append((lv, digest(np.rint(np.asarray(env.reset(), dtype=np.float64)[0] * 255.0).astype(np.uint8))))
        env.close()
    return res


def _coin_source(args):
    """Replay one bank source with the bank's own seeds and greedy behaviour, check every frame against the stored
    bank, and record episode boundaries (the bank was saved without them)."""
    root, scratch, si, device = args
    import torch
    RA, SA = env_imports("coin", root)
    from train_frozen_trunk_head_rewire import sample_actions
    bank = Path(scratch) / "coin_bank"
    meta = json.loads((bank / "meta.json").read_text())
    obs_b = np.load(bank / "obs_u8.npy", mmap_mode="r")
    src = meta["sources"][si]
    behav = SA.load_policy(Path(src["ckpt"]), device)
    env = RA.make_bank_env(meta["train_env"], num_envs=meta["num_envs"], start_level=0, num_levels=meta["num_levels"],
                           rand_seed=meta["bank_seed"] + 1009 * si)
    n_per, ne = meta["n_per_source"], meta["num_envs"]
    ep = np.empty(n_per, dtype=np.int64)
    counter = np.zeros(ne, dtype=np.int64)
    obs = env.reset()
    w, mism = 0, 0
    with torch.no_grad():
        while w < n_per:
            u8 = np.rint(np.asarray(obs, dtype=np.float64) * 255.0).astype(np.uint8)
            mism += int((u8 != obs_b[si * n_per + w: si * n_per + w + ne]).reshape(ne, -1).any(1).sum())
            ep[w:w + ne] = si * 10**7 + np.arange(ne) * 10**5 + counter
            w += ne
            obs, _r, done, _i = env.step(sample_actions(behav, obs, device))
            counter += np.asarray(done, dtype=np.int64)
    env.close()
    return si, ep, mism


def coin_prep(a):
    from multiprocessing import Pool
    out = work_env(a)
    with Pool(6, maxtasksperchild=1) as pool:
        res = sorted(pool.map(_coin_source, [(a.root, a.scratch, si, a.device) for si in range(6)]))
    ep = np.concatenate([r[1] for r in res])
    mism = [r[2] for r in res]
    np.save(out / "episode_id.npy", ep)
    obs = np.load(Path(a.scratch) / "coin_bank" / "obs_u8.npy", mmap_mode="r")
    bank_set = {digest(obs[i]) for i in range(len(obs))}
    _, SA = env_imports("coin", a.root)
    ev_in = []
    for i in range(COIN_EVAL_N):
        env = SA.make_env("coinrun_aisc", num_levels=1, start_level=SA.VAL_START + i)
        ev_in.append(digest(np.rint(np.asarray(env.reset(), dtype=np.float64)[0] * 255.0).astype(np.uint8)) in bank_set)
        env.close()
    info = dict(n_frames=int(len(ep)), n_episodes=int(len(np.unique(ep))), frame_mismatch_per_source=mism,
                n_unique_frames=len(bank_set), eval_start_in_bank=float(np.mean(ev_in)), eval_start_level=SA.VAL_START)
    (out / "prep.json").write_text(json.dumps(info, indent=1))
    print(info, flush=True)


def coin_strict(a):
    """strict_keep.npy for Coin: drop bank episodes that contain any frame identical to one of the eval start frames."""
    out = work_env(a)
    obs = np.load(Path(a.scratch) / "coin_bank" / "obs_u8.npy", mmap_mode="r")
    hashes = np.array([digest(obs[i]) for i in range(len(obs))], dtype="S16")
    np.save(out / "bank_hash.npy", hashes)
    _, SA = env_imports("coin", a.root)
    ev = []
    for i in range(COIN_EVAL_N):
        env = SA.make_env("coinrun_aisc", num_levels=1, start_level=SA.VAL_START + i)
        ev.append(digest(np.rint(np.asarray(env.reset(), dtype=np.float64)[0] * 255.0).astype(np.uint8)))
        env.close()
    ep = np.load(out / "episode_id.npy")
    bad = np.isin(hashes, np.array(ev, dtype="S16"))
    strict = ~np.isin(ep, np.unique(ep[bad]))
    np.save(out / "strict_keep.npy", strict)
    st = set(hashes[strict].tolist())
    info = dict(eval_start_in_full_bank=float(np.mean(bad.any() and [h in set(hashes[bad].tolist()) for h in ev])),
                eval_start_in_strict_bank=float(np.mean([h in st for h in ev])), n_bank_frames_matching=int(bad.sum()),
                dropped_episodes=int(len(np.unique(ep[~strict]))), strict_frames=int(strict.sum()),
                strict_episodes=int(len(np.unique(ep[strict]))))
    (out / "strict.json").write_text(json.dumps(info, indent=1))
    print(info, flush=True)


def subset(env, data, work):
    """Indices of the BC frames for one data condition, plus its level-grouped train/val split; identical for every
    trunk. Smaller fractions are nested in larger ones (one permutation of the base levels)."""
    base, frac = DATA[env][data]
    w = Path(work) / env
    groups = np.load(w / "groups.npy")
    idx = np.arange(len(groups)) if base == "full" else np.flatnonzero(np.load(w / "strict_keep.npy"))
    ug = np.unique(groups[idx])
    keep_g = np.random.RandomState(SUBSET_SEED).permutation(ug)[: max(2, int(np.ceil(frac * len(ug))))]
    idx = idx[np.isin(groups[idx], keep_g)]
    vg = np.random.RandomState(VAL_SEED).permutation(np.sort(keep_g))[: max(1, len(keep_g) // 5)]
    is_val = np.isin(groups[idx], vg)
    return idx[~is_val], idx[is_val], len(keep_g)


# ------------------------------------------------------------------ BC

def train(a):
    """All readouts x data conditions for one trunk; features cached once on the GPU."""
    import torch
    import torch.nn.functional as F
    torch.backends.cudnn.benchmark = False
    w = work_env(a)
    pol = load_trunk(a.env, a.trunk, a.root, a.device)
    if a.env == "am":
        bank = Path(a.scratch) / "bank" / "black"
        obs = np.load(bank / "obs_u8.npy", mmap_mode="r")
        y = torch.as_tensor(np.load(bank / "soft_fixed.npy"), device=a.device)
    else:
        bank = Path(a.scratch) / "coin_bank"
        obs = np.load(bank / "obs_u8.npy", mmap_mode="r")
        y = torch.as_tensor(np.load(bank / "actions.npy"), device=a.device)
    n = len(obs)
    hp = torch.empty((n, 2048), dtype=torch.float32, device=a.device)
    hf = torch.empty((n, 256), dtype=torch.float32, device=a.device)
    with torch.no_grad():
        for s in range(0, n, 2048):
            x = torch.as_tensor(np.asarray(obs[s:s + 2048], dtype=np.float32) / 255.0, device=a.device)
            hp[s:s + len(x)] = h_pre(pol, x)
            hf[s:s + len(x)] = F.relu(pol.embedder.fc(hp[s:s + len(x)]))
    sq, z = 0.0, 0.0
    for s in range(0, n, 16384):
        sq += float(hp[s:s + 16384].double().pow(2).sum())
        z += float((hp[s:s + 16384] == 0).sum())
    rms = dict(h_pre_rms=(sq / hp.numel()) ** 0.5, h_fc_rms=float(hf.double().pow(2).mean().sqrt()),
               h_pre_frac_zero=z / hp.numel())
    n_act = 15
    soft = y.ndim == 2

    def ce(logits, t):
        return -(t * F.log_softmax(logits, 1)).sum(1) if soft else F.cross_entropy(logits, t, reduction="none")

    def acc(logits, t):
        am = logits.argmax(1)
        return (t.gather(1, am[:, None])[:, 0] > 0).float() if soft else (am == t).float()

    full_train = len(subset(a.env, "full", a.work)[0])
    budget = int(np.ceil(full_train / BC["batch_size"])) * BC["ref_epochs"]
    for data in [d for d in DATA[a.env] if not a.datas or d in a.datas]:
        tr, va, n_groups = subset(a.env, data, a.work)
        tr_t, va_t = torch.as_tensor(tr, device=a.device), torch.as_tensor(va, device=a.device)
        for mult in sorted({READOUT_CFG[r][1] for r in a.readouts}):
            per_epoch = int(np.ceil(len(tr) / BC["batch_size"]))
            epochs = max(BC["ref_epochs"] * mult, int(np.ceil(budget * mult / per_epoch)))
            todo = [ro for ro in a.readouts if READOUT_CFG[ro][1] == mult
                    and not (w / a.trunk / f"{ro}_{data}" / "head_final.pt").exists()]
            if not todo:
                continue
            # All readouts share one batch stream (each would draw the identical one from SUBSET_SEED anyway), so they are
            # trained in one loop; losses are summed on the GPU to avoid a host sync per step.
            models = {ro: make_readout(ro, n_act).to(a.device) for ro in todo}
            opts = {ro: torch.optim.Adam(m.parameters(), lr=READOUT_CFG[ro][0]) for ro, m in models.items()}
            feats = {ro: hf if ro.startswith("lin_fc") else hp for ro in todo}
            g = torch.Generator(device="cpu").manual_seed(SUBSET_SEED)
            t0 = time.time()
            logs = {ro: [] for ro in todo}
            best = {ro: (np.inf, None, -1) for ro in todo}
            val_every = max(1, epochs // 200)
            for ep in range(epochs):
                perm = tr_t[torch.randperm(len(tr_t), generator=g).to(a.device)]
                tot = {ro: torch.zeros((), device=a.device) for ro in todo}
                for ro in todo:
                    models[ro].train()
                for s in range(0, len(perm), BC["batch_size"]):
                    b = perm[s:s + BC["batch_size"]]
                    yb = y[b]
                    for ro in todo:
                        opts[ro].zero_grad()
                        loss = ce(models[ro](feats[ro][b]), yb).mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models[ro].parameters(), BC["clip"])
                        opts[ro].step()
                        tot[ro] += loss.detach() * len(b)
                if ep % val_every and ep != epochs - 1:
                    continue
                for ro in todo:
                    m = models[ro].eval()
                    with torch.no_grad():
                        vl, vacc = 0.0, 0.0
                        for s in range(0, len(va_t), 8192):
                            b = va_t[s:s + 8192]
                            lo = m(feats[ro][b])
                            vl += float(ce(lo, y[b]).sum())
                            vacc += float(acc(lo, y[b]).sum())
                    row = dict(epoch=ep, train_ce=float(tot[ro]) / len(tr), val_ce=vl / len(va), val_acc=vacc / len(va))
                    logs[ro].append(row)
                    if row["val_ce"] < best[ro][0]:
                        best[ro] = (row["val_ce"], copy.deepcopy(m.state_dict()), ep)
            for ro in todo:
                d = w / a.trunk / f"{ro}_{data}"
                d.mkdir(parents=True, exist_ok=True)
                log, bst, model = logs[ro], best[ro], models[ro]
                # head_final.pt = the paper protocol (last step of the budget); head.pt = lowest level-held-out val CE.
                torch.save(model.state_dict(), d / "head_final.pt")
                if (d / "head.pt").exists():
                    continue
                model.load_state_dict(bst[1])
                torch.save(model.state_dict(), d / "head.pt")
                summ = dict(env=a.env, trunk=a.trunk, readout=ro, data=data, n_train=len(tr), n_val=len(va), n_groups=n_groups,
                            epochs=epochs, grad_steps=epochs * per_epoch, best_epoch=bst[2], best_val_ce=bst[0],
                            final_val_ce=log[-1]["val_ce"], final_train_ce=log[-1]["train_ce"],
                            best_val_acc=[r for r in log if r["epoch"] == bst[2]][0]["val_acc"],
                            n_trainable=int(sum(p.numel() for p in model.parameters())), train_s=time.time() - t0,
                            log=log, **rms)
                (d / "train.json").write_text(json.dumps(summ, indent=1))
                print(f"[{a.env} {a.trunk} {ro} {data}] n_tr={len(tr)} ep={epochs} best_val={bst[0]:.4f}@{bst[2]} "
                      f"acc={summ['best_val_acc']:.3f} final_train={summ['final_train_ce']:.4f} ({summ['train_s']:.0f}s)",
                      flush=True)
            del models, opts
        torch.cuda.empty_cache()


def setup_groups(a):
    """groups.npy: AM bank level id, Coin replayed episode id (one level per episode)."""
    w = work_env(a)
    src = Path(a.scratch) / "bank" / "black" / "level_id.npy" if a.env == "am" else w / "episode_id.npy"
    np.save(w / "groups.npy", np.load(src))


# ------------------------------------------------------------------ closed-loop eval (CPU, one process per arm)

def evaluate(a):
    import torch
    torch.set_num_threads(1)
    w = work_env(a)
    d = w / a.trunk / f"{a.readout}_{a.data}"
    tag = ("greedy" if a.mode == "greedy" else f"sample_es{a.eval_seed}") + ("_final" if a.head == "final" else "")
    if (d / f"eval_{tag}.json").exists():
        return
    pol = load_trunk(a.env, a.trunk, a.root, "cpu")
    ro = make_readout(a.readout, 15)
    ro.load_state_dict(torch.load(d / ("head_final.pt" if a.head == "final" else "head.pt"), map_location="cpu"))
    policy = make_policy(pol, ro)
    if a.mode == "sample":
        torch.manual_seed(10_000 + a.eval_seed)
    t0 = time.time()
    if a.env == "am":
        res, rows = am_rollouts(a, policy)
    else:
        res, rows = coin_rollouts(a, policy)
    with open(d / f"episodes_{tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    tr = json.loads((d / "train.json").read_text())
    res.update(env=a.env, trunk=a.trunk, cohort=a.trunk[:-1], seed_idx=int(a.trunk[-1]), readout=a.readout, data=a.data,
               mode=a.mode, head=a.head, eval_seed=a.eval_seed if a.mode == "sample" else None, best_val_ce=tr["best_val_ce"],
               final_val_ce=tr["final_val_ce"], best_val_acc=tr["best_val_acc"], n_train=tr["n_train"],
               eval_s=time.time() - t0)
    (d / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[{a.env} {a.trunk} {a.readout} {a.data} {tag}] goal={res['goal']:.3f} D={res['decisive']:.3f} "
          f"P|D={res['p_goal_given_decisive']:.3f} ({res['eval_s']:.0f}s)", flush=True)


def am_rollouts(a, policy):
    """eval_closed_loop's protocol (level 100000 + i, num_levels=1, 256 steps, raw +-10 terminal reward), plus the
    fraction of visited frames that are byte-identical to a frame in this arm's own BC subset."""
    import torch
    from gm_geometry.integrations.colour_shape import make_duel_env
    w = Path(a.work) / "am"
    tr, va, _ = subset("am", a.data, a.work)
    hashes = np.load(w / "bank_hash.npy")
    seen = set(hashes[np.r_[tr, va]].tolist())
    rows = []
    for i in range(AM_EVAL_N):
        env = make_duel_env(1, 1, 1, "red_line_diag", "yellow_gem", start_level=AM_EVAL_START + i, use_backgrounds=False)
        obs = env.reset()
        total, n_in, steps = 0.0, 0, 0
        with torch.no_grad():
            for _ in range(AM_MAX_STEPS):
                n_in += digest(np.rint(np.asarray(obs, dtype=np.float64)[0] * 255.0).astype(np.uint8)) in seen
                dist, _v, _h = policy(torch.as_tensor(obs, dtype=torch.float32), None, None)
                act = dist.probs.argmax(-1) if a.mode == "greedy" else dist.sample()
                obs, rew, done, info = env.step(act.numpy())
                steps += 1
                raw = info[0].get("env_reward") if isinstance(info[0], dict) else None
                total += float(raw) if raw is not None else float(np.asarray(rew)[0])
                if bool(np.asarray(done)[0]):
                    break
        env.close()
        rows.append(dict(level=AM_EVAL_START + i, reward=total, steps=steps, frac_frames_in_bc=n_in / steps))
    r = np.array([x["reward"] for x in rows])
    st = np.array([x["steps"] for x in rows])
    goal, proxy = float(np.mean(r == 10)), float(np.mean(r == -10))
    dec = goal + proxy
    return dict(goal=goal, proxy=proxy, decisive=dec, p_goal_given_decisive=goal / dec if dec else float("nan"),
                timeout=float(np.mean(st >= AM_MAX_STEPS)), mean_steps=float(st.mean()),
                frac_frames_in_bc=float(np.mean([x["frac_frames_in_bc"] for x in rows])),
                frac_start_in_bc=float(np.mean([x["frac_frames_in_bc"] > 0 for x in rows]))), rows


def coin_rollouts(a, policy):
    _, SA = env_imports("coin", a.root)
    ev = SA.eval_policy_val(policy, n_levels=COIN_EVAL_N, start_level=SA.VAL_START, device="cpu",
                            max_steps=COIN_MAX_STEPS, action_mode=a.mode, keep_episodes=True)
    o = ev["outcome_rates"]
    goal, end = float(o.get("success", 0)), float(o.get("goal_misgen", 0))
    dec = goal + end
    rows = [{k: e.get(k) for k in ("level", "outcome", "max_progress", "reached_end_region")} for e in ev["episodes"]]
    return dict(goal=goal, proxy=end, decisive=dec, p_goal_given_decisive=goal / dec if dec else float("nan"),
                died=float(o.get("died", 0)), timeout=float(o.get("timed_out", 0)),
                mean_max_progress=ev.get("mean_max_progress")), rows


# ------------------------------------------------------------------ launcher

def launch(a):
    """One fresh subprocess per job (procgen env construction slows in long-lived processes)."""
    logs = Path(a.work) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for env in a.envs:
        for t in a.trunks or trunk_ids(env):
            if a.cmd_inner == "train":
                jobs.append([a.cmd_inner, "--env", env, "--trunk", t, "--readouts", *a.readouts, "--datas", *a.datas])
                continue
            for data in [d for d in DATA[env] if not a.datas or d in a.datas]:
                for ro in a.readouts:
                    for mode, es in [("greedy", 0)] * ("greedy" in a.modes) + [("sample", s) for s in a.seeds] * ("sample" in a.modes):
                        jobs.append([a.cmd_inner, "--env", env, "--trunk", t, "--readout", ro, "--data", data,
                                     "--mode", mode, "--eval-seed", str(es), "--head", a.head])

    def go(j):
        name = "_".join(x for x in j if not x.startswith("--"))
        argv = [sys.executable, __file__, *j, "--root", a.root, "--work", a.work, "--scratch", a.scratch, "--device", a.device]
        with open(logs / f"{name}.log", "a") as f:
            return name, subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))

    with ThreadPoolExecutor(a.parallel) as ex:
        for name, rc in ex.map(go, jobs):
            print(name, f"rc={rc}", flush=True)


def collect(a):
    """Per-seed CSV (one row per arm x eval) from every eval_*.json under the work dir."""
    rows = []
    for env in a.envs:
        for f in sorted((Path(a.work) / env).glob("*/*/eval_*.json")):
            rows.append(json.loads(f.read_text()))
    keys = sorted({k for r in rows for k in r})
    with open(Path(a.work) / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {Path(a.work) / 'per_seed.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("am_prep", "coin_prep", "coin_strict", "setup_groups", "train", "evaluate", "launch", "collect"))
    p.add_argument("--env", default="am", choices=("am", "coin"))
    p.add_argument("--envs", nargs="+", default=["am", "coin"])
    p.add_argument("--trunk", default="gm0", help="gm<i> / nongm<i> / rand<i>, i in 0..2 (roster order in TRUNKS)")
    p.add_argument("--trunks", nargs="*", default=[], help="launch: restrict to these trunk ids")
    p.add_argument("--readout", default="mlp", choices=READOUTS)
    p.add_argument("--datas", nargs="*", default=[], help="restrict to these data conditions (default: all)")
    p.add_argument("--readouts", nargs="+", default=["mlp", "lin_pre", "lin_fc"], choices=READOUTS)
    p.add_argument("--data", default="full")
    p.add_argument("--mode", default="greedy", choices=("greedy", "sample"))
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--head", default="best", choices=("best", "final"), help="best = early-stopped on level-held-out val CE")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="sampled-action eval seeds (torch seed 10000 + s)")
    p.add_argument("--modes", nargs="+", default=["greedy", "sample"])
    p.add_argument("--cmd-inner", default="evaluate", choices=("train", "evaluate"))
    p.add_argument("--root", default=str(PROCGEN_ROOT))
    p.add_argument("--work", default="work/rwctl/strict")
    p.add_argument("--scratch", default="work/rwctl/scratch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--parallel", type=int, default=100)
    a = p.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
