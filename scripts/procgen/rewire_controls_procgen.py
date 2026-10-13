"""Random-init-trunk h_pre rewires for Coin and Attribute Maze (black, fixed oracle) with the paper's own rewire pipelines."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from procgen_paths import PROCGEN_ROOT, add_procgen_paths, ckpt_dir

HERE = Path(__file__).resolve().parent
COIN_GM = ["6033_gm_20M", "6037_gm_120M", "6051_gm_55M"]
COIN_NONGM = ["7001_teach", "7002_teach", "7003_teach"]
COIN_LABEL = "7001_teach"
COIN_BANK = dict(collect_steps=500_000, num_envs=64, num_levels=100_000, bank_seed=0, train_env="coinrun_aisc")
BC = dict(epochs=20, batch_size=256, lr=3e-4)
INIT_SEED = 0


def trunk_torch_seed(trunk):
    """Random trunk k is drawn with torch seed 1000 + k, as in rewire_maze_random_trunk."""
    return 1000 + int(trunk[4:])


# ------------------------------------------------------------------ Coin

def coin_imports(root):
    add_procgen_paths(root)
    import run_coin_rewire_aligned as RA
    import run_coin_behaviour as SA
    return RA, SA


def coin_ckpt(root, label):
    return ckpt_dir(root, "coin") / f"{label}.pth"


def coin_bank(a):
    """Rebuild the paper's shared balanced Coin bank (3 GM + 3 teacher drivers, teacher-7001 greedy labels)."""
    RA, _ = coin_imports(a.root)
    sources = [(f"gm_{s}", coin_ckpt(a.root, s)) for s in COIN_GM] + \
              [(f"nongm_{s}", coin_ckpt(a.root, s)) for s in COIN_NONGM]
    RA.load_or_build_bank(Path(a.scratch) / "coin_bank", sources=sources, label_ckpt=coin_ckpt(a.root, COIN_LABEL),
                          device=a.device, args=Namespace(**COIN_BANK))


def coin_eval(SA, policy, a, mode, eval_seed=None):
    import torch
    if eval_seed is not None:
        torch.manual_seed(10_000 + eval_seed)
    policy.eval()
    ev = SA.eval_policy_val(policy, n_levels=a.eval_n, start_level=SA.VAL_START, device=a.device, max_steps=1000,
                            action_mode=mode, keep_episodes=True)
    return ev


def coin_arm(a):
    """trunk = rand<k> (fresh Impala, torch seed 1000+k) or a roster label; fc + head drawn exactly as the paper's arms
    (torch seed 0, then a policy load, then prepare_tap), so only the frozen block1-3 weights differ between arms."""
    import torch
    RA, SA = coin_imports(a.root)
    out = Path(a.work) / "coin" / a.trunk
    out.mkdir(parents=True, exist_ok=True)
    bank = Path(a.scratch) / "coin_bank"
    obs = np.load(bank / "obs_u8.npy", mmap_mode="r")
    acts = np.load(bank / "actions.npy")
    head_f = out / "head.pt"
    is_rand = a.trunk.startswith("rand")
    base_ckpt = coin_ckpt(a.root, COIN_LABEL if is_rand else a.trunk)
    cohort = "init" if is_rand else ("gm" if a.trunk in COIN_GM else "nongm")
    arm = RA.build_arm(name=f"{cohort}_rewire_{a.trunk}", cohort=cohort, mode="rewire", tap="block3", ckpt=base_ckpt,
                       device=a.device, init_seed=INIT_SEED)
    if is_rand:
        torch.manual_seed(trunk_torch_seed(a.trunk))
        rnd = SA.load_init_policy(a.device)
        for blk in ("block1", "block2", "block3"):
            getattr(arm.policy.embedder, blk).load_state_dict(getattr(rnd.embedder, blk).state_dict())
        del rnd
    if head_f.exists():
        arm.policy.load_state_dict(torch.load(head_f, map_location=a.device))
        summ = json.loads((out / "train.json").read_text())
    else:
        t0 = time.time()
        feats = RA.cache_features(arm.policy, obs, a.device, batch=512, apply_fc=False, check_ref=True)
        curve = RA.train_bc(arm, feats, acts, device=a.device, ckpt_dir=out / "ckpts", ckpt_every=100, **BC)
        del feats
        torch.save(arm.policy.state_dict(), head_f)
        summ = dict(trunk=a.trunk, cohort=cohort, n_frames=int(len(acts)), bc_curve=curve,
                    final_val_ce=curve[-1]["val_ce"], final_val_acc=curve[-1]["val_acc"], train_s=time.time() - t0,
                    trunk_torch_seed=trunk_torch_seed(a.trunk) if is_rand else None, init_seed=INIT_SEED)
        (out / "train.json").write_text(json.dumps(summ, indent=1))
    for mode, es in [("greedy", None)] * ("greedy" in a.modes) + [("sample", s) for s in a.eval_seeds] * ("sample" in a.modes):
        tag = "greedy" if mode == "greedy" else f"sample_es{es}"
        if (out / f"eval_{tag}.json").exists():
            continue
        t0 = time.time()
        ev = coin_eval(SA, arm.policy, a, mode, es)
        with open(out / f"episodes_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["level", "outcome", "max_progress", "reached_end_region"])
            w.writeheader()
            for e in ev["episodes"]:
                w.writerow({k: e.get(k) for k in w.fieldnames})
        res = dict(trunk=a.trunk, cohort=cohort, mode=mode, eval_seed=es, success_rate=ev["success_rate"],
                   outcome_rates=ev["outcome_rates"], mean_max_progress=ev.get("mean_max_progress"),
                   end_reach_rate=ev.get("end_reach_rate"), n_levels=ev["n_levels"] if "n_levels" in ev else a.eval_n,
                   final_val_ce=summ["final_val_ce"], final_val_acc=summ["final_val_acc"], eval_s=time.time() - t0)
        (out / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
        print(f"[coin {a.trunk} {tag}] coin reach {res['success_rate']:.3f} outcomes {res['outcome_rates']} "
              f"progress {res['mean_max_progress']:.3f} ({res['eval_s']:.0f}s)", flush=True)


def coin_native(a):
    """Native (untouched) greedy eval of one roster checkpoint, to confirm checkpoint identity against the paper."""
    _, SA = coin_imports(a.root)
    out = Path(a.work) / "coin_native"
    out.mkdir(parents=True, exist_ok=True)
    pol = SA.load_policy(coin_ckpt(a.root, a.trunk), a.device)
    ev = coin_eval(SA, pol, a, "greedy")
    (out / f"{a.trunk}.json").write_text(json.dumps(dict(trunk=a.trunk, success_rate=ev["success_rate"],
                                                         outcome_rates=ev["outcome_rates"]), indent=1))
    print(f"[coin native {a.trunk}] {ev['success_rate']:.3f} {ev['outcome_rates']}", flush=True)


# ------------------------------------------------------------------ Attribute Maze (black, fixed oracle)

def am_imports(root):
    import am_oracle_fix as AF
    AF.pod_setup(Path(root))
    import head_rewire_ras_maze as hrm
    return AF, hrm


def am_arm(a):
    """Random-trunk h_pre rewire on the fixed-oracle black bank, then the paper's 1000-episode eval (greedy) and
    sampled evals. Suffix seed follows am_oracle_fix (trunk_id * 1000 + tap/arm offsets, here trunk_id = 1000 + k)."""
    import torch
    AF, hrm = am_imports(a.root)
    from gm_geometry.rewire import prepare_tap
    from gm_geometry.rewire.ras import eval_closed_loop, tap_features_from_obs, train_tap_bc

    out = Path(a.work) / "am" / a.trunk
    out.mkdir(parents=True, exist_ok=True)
    bank = Path(a.scratch) / "bank" / "black"
    head_f = out / "head.pt"
    torch.manual_seed(trunk_torch_seed(a.trunk))
    policy = hrm.load_ras_policy(None, a.device, random_init=True)
    torch_seed = trunk_torch_seed(a.trunk) * 1000 + AF.TAP_OFFSET["block3"] + AF.ARM_OFFSET["rewire"]
    torch.manual_seed(torch_seed)
    prepare_tap(policy, "block3")
    if head_f.exists():
        policy.load_state_dict(torch.load(head_f, map_location=a.device))
        summ = json.loads((out / "train.json").read_text())
    else:
        t0 = time.time()
        obs = np.load(bank / "obs_u8.npy", mmap_mode="r")
        soft = np.load(bank / "soft_fixed.npy")
        groups = np.load(bank / "level_id.npy")
        keep = np.load(bank / "keep_fixed.npy")
        feats = tap_features_from_obs(policy, obs, "block3", device=a.device, batch_size=512)[keep]
        soft, groups = soft[keep], groups[keep]
        log = train_tap_bc(policy, feats, soft, "block3", device=a.device, prepare=False, groups=groups, **BC)
        del feats
        torch.save(policy.state_dict(), head_f)
        summ = dict(trunk=a.trunk, n_bank_frames=int(len(soft)), bc_log=log, torch_seed=torch_seed,
                    trunk_torch_seed=trunk_torch_seed(a.trunk), train_s=time.time() - t0)
        (out / "train.json").write_text(json.dumps(summ, indent=1))
    for mode, es in [("greedy", None)] * ("greedy" in a.modes) + [("sample", s) for s in a.eval_seeds] * ("sample" in a.modes):
        tag = "greedy" if mode == "greedy" else f"sample_es{es}"
        if (out / f"eval_{tag}.json").exists():
            continue
        t0 = time.time()
        if es is not None:
            torch.manual_seed(10_000 + es)
        res = eval_closed_loop(policy, a.device, use_backgrounds=False, n_episodes=AF.EVAL_N,
                               max_steps=AF.EVAL_MAX_STEPS, start_level=AF.EVAL_START, greedy=(mode == "greedy"))
        with open(out / f"episodes_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "reward", "steps", "distinct_actions"])
            for i, (r, s, d) in enumerate(zip(res["episode_rewards"], res["episode_steps"], res["episode_distinct_actions"])):
                w.writerow([AF.EVAL_START + i, r, s, d])
        keep_keys = ("R_goal", "decisive_frac", "pref_goal", "n_goal_hit", "n_proxy_hit", "frac_timeout", "mean_steps",
                     "mean_action_entropy", "mean_distinct_actions")
        r = {k: res[k] for k in keep_keys}
        r.update(trunk=a.trunk, mode=mode, eval_seed=es, final_val=summ["bc_log"][-1], eval_s=time.time() - t0)
        (out / f"eval_{tag}.json").write_text(json.dumps(r, indent=1))
        print(f"[am {a.trunk} {tag}] R={r['R_goal']:.3f} D={r['decisive_frac']:.3f} P={r['pref_goal']:.3f} "
              f"timeout={r['frac_timeout']:.3f} ({r['eval_s']:.0f}s)", flush=True)


def am_sampled_trained(a):
    """Sampled-action eval of a trained-trunk fixed-oracle arm head saved by am_oracle_fix (final.pth)."""
    import torch
    AF, hrm = am_imports(a.root)
    from gm_geometry.rewire.ras import eval_closed_loop
    role, seed = a.trunk.split("_")
    d = Path(a.work) / "raw" / "fixed" / "black" / "block3" / f"{role}_rewire_{seed}"
    pol = hrm.load_ras_policy(AF.ckpt_path(Path(a.root), int(seed), "gm" if role == "gm" else "teach"), a.device)
    pol.load_state_dict(torch.load(d / "final.pth", map_location=a.device)["model_state_dict"])
    for es in a.eval_seeds:
        f = d / f"eval_sample_es{es}.json"
        if f.exists():
            continue
        torch.manual_seed(10_000 + es)
        res = eval_closed_loop(pol, a.device, use_backgrounds=False, n_episodes=AF.EVAL_N, max_steps=AF.EVAL_MAX_STEPS,
                               start_level=AF.EVAL_START, greedy=False)
        r = {k: res[k] for k in ("R_goal", "decisive_frac", "pref_goal", "frac_timeout")}
        r.update(trunk=a.trunk, mode="sample", eval_seed=es)
        f.write_text(json.dumps(r, indent=1))
        print(f"[am {a.trunk} sample_es{es}] R={r['R_goal']:.3f} D={r['decisive_frac']:.3f}", flush=True)


def _start_u8(env):
    obs = np.asarray(env.reset(), dtype=np.float64)[0]
    return np.rint(obs * 255.0).astype(np.uint8)


def _scan_levels(args):
    """(level, start-frame digest) for a contiguous level range; one worker of am_disjoint_scan."""
    import hashlib
    root, lo, hi = args
    _, hrm = am_imports(root)
    out = []
    for lv in range(lo, hi):
        env = hrm.make_ras_env(num_envs=1, num_levels=1, start_level=lv, use_backgrounds=False)
        out.append((lv, hashlib.blake2b(_start_u8(env).tobytes(), digest_size=16).digest()))
        env.close()
    return out


def am_disjoint_scan(a):
    """Levels whose start frame is absent from the fixed-oracle black bank: membership of the 1000 paper eval levels (to
    compare with the paper's 87.7%) and of fresh levels [200000, 200000 + --scan-n), scanned in parallel."""
    import hashlib
    from multiprocessing import Pool
    AF, _ = am_imports(a.root)
    obs = np.load(Path(a.scratch) / "bank" / "black" / "obs_u8.npy", mmap_mode="r")
    seen = {hashlib.blake2b(np.ascontiguousarray(obs[i]).tobytes(), digest_size=16).digest() for i in range(len(obs))}
    step = 250
    jobs = [(a.root, lo, min(lo + step, AF.EVAL_START + AF.EVAL_N)) for lo in range(AF.EVAL_START, AF.EVAL_START + AF.EVAL_N, step)]
    jobs += [(a.root, lo, lo + step) for lo in range(200_000, 200_000 + a.scan_n, step)]
    res = []
    # procgen env construction slows down within a long-lived process, so every chunk gets a fresh worker
    with Pool(a.parallel, maxtasksperchild=1) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_scan_levels, jobs)):
            res += chunk
            if i % 10 == 0:
                print(f"  scanned {len(res)} levels", flush=True)
    rows = [dict(level=lv, set="paper_eval" if lv < 200_000 else "fresh", in_bank=h in seen,
                 digest=h.hex()) for lv, h in res]
    # Keep one level per distinct unseen start frame so a repeated layout does not dominate the disjoint set.
    first = {}
    for r in rows:
        if r["set"] == "fresh" and not r["in_bank"]:
            first.setdefault(r["digest"], r["level"])
    for r in rows:
        r["use"] = r["set"] == "fresh" and not r["in_bank"] and first.get(r["digest"]) == r["level"]
    out = Path(a.work) / "am_disjoint"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "set", "in_bank", "digest", "use"])
        w.writeheader()
        w.writerows(rows)
    pe = np.mean([r["in_bank"] for r in rows if r["set"] == "paper_eval"])
    fr = [r for r in rows if r["set"] == "fresh"]
    print(f"bank frames {len(obs)} unique {len(seen)}; paper eval start frames in bank {pe:.3f}; fresh scanned {len(fr)}, "
          f"unseen {sum(not r['in_bank'] for r in fr)}, distinct unseen start frames used {sum(r['use'] for r in rows)}", flush=True)


def am_disjoint_eval(a):
    """Greedy episodes of one head on the fresh levels with unseen start frames. trunk = rand<k> (am/rand<k>/head.pt),
    gm_<seed> / nongm_<seed> (heads retrained on this bank, work_repro) or paper_gm_<seed> / paper_nongm_<seed>
    (am_oracle_fix heads)."""
    import torch
    AF, hrm = am_imports(a.root)
    from gm_geometry.rewire import prepare_tap
    from gm_geometry.rewire.ras import _rollout_episode_with_actions
    levels = [int(r["level"]) for r in csv.DictReader(open(Path(a.work) / "am_disjoint" / "levels.csv"))
              if r["use"] == "True"]
    if a.trunk.startswith("rand"):
        torch.manual_seed(trunk_torch_seed(a.trunk))
        pol = hrm.load_ras_policy(None, a.device, random_init=True)
        prepare_tap(pol, "block3")
        pol.load_state_dict(torch.load(Path(a.work) / "am" / a.trunk / "head.pt", map_location=a.device))
    else:
        parts = a.trunk.split("_")
        role, seed = parts[-2], int(parts[-1])
        work = Path(a.work) if parts[0] == "paper" else Path(a.work).parent / "work_repro"
        pol = hrm.load_ras_policy(AF.ckpt_path(Path(a.root), seed, "gm" if role == "gm" else "teach"), a.device)
        sd = torch.load(work / "raw" / "fixed" / "black" / "block3" / f"{role}_rewire_{seed}" / "final.pth", map_location=a.device)
        pol.load_state_dict(sd["model_state_dict"])
    pol.eval()
    out = Path(a.work) / "am_disjoint" / f"{a.trunk}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "reward", "steps"])
        for lv in levels:
            env = hrm.make_ras_env(num_envs=1, num_levels=1, start_level=lv, use_backgrounds=False)
            try:
                rew, n, _ = _rollout_episode_with_actions(pol, env, max_steps=AF.EVAL_MAX_STEPS, greedy=True)
            finally:
                env.close()
            w.writerow([lv, rew, n])
    r = np.loadtxt(out, delimiter=",", skiprows=1)[:, 1]
    print(f"[am disjoint {a.trunk}] n={len(r)} R_goal={np.mean(r == 10):.3f} D={np.mean(r != 0):.3f}", flush=True)


# ------------------------------------------------------------------ launcher

def launch(a):
    """One fresh subprocess per job (procgen/torch state leaks across checkpoints in one long process)."""
    jobs = [j.split(":") for j in a.jobs_list]
    logs = Path(a.work) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def go(j):
        cmd, trunk = j
        argv = [sys.executable, __file__, cmd, "--trunk", trunk, "--root", a.root, "--work", a.work, "--scratch", a.scratch,
                "--device", a.device, "--eval-n", str(a.eval_n), "--modes", *a.modes, "--eval-seeds", *map(str, a.eval_seeds)]
        with open(logs / f"{cmd}_{trunk}.log", "a") as f:
            return j, subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))

    with ThreadPoolExecutor(a.parallel) as ex:
        for j, rc in ex.map(go, jobs):
            print(" ".join(j), f"rc={rc}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("coin_bank", "coin_arm", "coin_native", "am_arm", "am_sampled_trained", "am_disjoint_scan",
                                   "am_disjoint_eval", "launch"))
    p.add_argument("--trunk", default="rand0", help="rand<k> or a roster label (coin) / gm_<seed>, nongm_<seed> (am_sampled_trained)")
    p.add_argument("--root", default=str(PROCGEN_ROOT))
    p.add_argument("--work", default="work/rwctl/work")
    p.add_argument("--scratch", default="work/rwctl/scratch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--modes", nargs="+", default=["greedy", "sample"], choices=("greedy", "sample"))
    p.add_argument("--eval-seeds", "--seeds", dest="eval_seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="sampled-action eval seeds (torch seed 10000 + s)")
    p.add_argument("--jobs-list", nargs="*", default=[], help="launch: cmd:trunk items")
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--scan-n", type=int, default=60_000, help="fresh levels scanned for unseen start frames")
    a = p.parse_args()
    {"coin_bank": coin_bank, "coin_arm": coin_arm, "coin_native": coin_native, "am_arm": am_arm,
     "am_sampled_trained": am_sampled_trained, "am_disjoint_scan": am_disjoint_scan,
     "am_disjoint_eval": am_disjoint_eval, "launch": launch}[a.cmd](a)


if __name__ == "__main__":
    main()
