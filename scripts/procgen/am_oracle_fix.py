"""Attribute Maze oracle rewiring arms trained on the original (gem-crossing) and fixed (gem-excluded) oracle labels."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from procgen_paths import PROCGEN_ROOT, add_procgen_paths, ckpt_dir
from gm_robotics.paths import ROOT as RUN_ROOT

HERE = Path(__file__).resolve().parent
ROBO_ROOT = RUN_ROOT
RESULTS = ROBO_ROOT / "results" / "analysis" / "am_oracle_fix"
GMG = Path(os.environ.get("GM_GEOMETRY_ROOT", str(Path(__file__).resolve().parents[2])))
PAPER_JSON = GMG / "data/probes/rewire_aligned_20260922"
AUDIT = ROBO_ROOT / "results/analysis/am_distance_audit"

COHORTS = {
    "black": dict(gm=(6313, 6315, 6317), nongm=(4371, 4372, 4373), use_backgrounds=False),
    "texture": dict(gm=(6301, 6302, 6303), nongm=(4361, 4362, 4363), use_backgrounds=True),
}
FRAMES_PER_DRIVER = 40_000  # 240000 collect steps / 6 drivers, as in the paper
N_ACTIONS = 15
EVAL_START, EVAL_N, EVAL_MAX_STEPS = 100_000, 1000, 256
BC = dict(epochs=20, batch_size=256, lr=3e-4)
TAP_OFFSET = {"block3": 0, "fc": 100}
ARM_OFFSET = {"rewire": 0, "random_proj": 1}


def pod_setup(root):
    """Make the Procgen code under `root` importable for the Attribute Maze runs."""
    os.environ["GM_GEOMETRY_ROOT"] = str(root)
    add_procgen_paths(root, attribute_maze=True)
def drivers(cohort):
    c = COHORTS[cohort]
    return [(s, "gm") for s in c["gm"]] + [(s, "teach") for s in c["nongm"]]


def ckpt_path(root, seed, kind):
    return ckpt_dir(root, "ras") / f"{seed}_{kind}.pth"


# ------------------------------------------------------------------ bank collection (GPU)

def cmd_collect(a):
    """One driver of one cohort's oracle bank, both label sets; run once per driver in its own process."""
    import torch

    pod_setup(a.root)
    import head_rewire_ras_maze as hrm
    import am_oracle as rof

    torch.set_num_threads(a.threads)
    if a.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    seed, kind = drivers(a.cohort)[a.driver]
    out = a.scratch / "shards" / a.cohort / f"driver{a.driver}"
    if (out / "DONE").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    n = FRAMES_PER_DRIVER
    mm = lambda name, dt, shape: np.lib.format.open_memmap(out / name, mode="w+", dtype=dt, shape=shape)
    obs, sb, sf = mm("obs_u8.npy", np.uint8, (n, 3, 64, 64)), mm("soft_buggy.npy", np.float32, (n, N_ACTIONS)), \
        mm("soft_fixed.npy", np.float32, (n, N_ACTIONS))
    lv, ag = mm("level_id.npy", np.int64, (n,)), mm("agent_cell.npy", np.int16, (n, 2))
    policy = hrm.load_ras_policy(ckpt_path(a.root, seed, kind), a.device)
    stats, levels = rof.collect_driver_dual(
        policy, a.driver, n_drivers=6, use_backgrounds=COHORTS[a.cohort]["use_backgrounds"],
        frames_per_driver=n, device=a.device, rand_seed=0, out_obs=obs, out_soft_buggy=sb, out_soft_fixed=sf,
        out_level=lv, out_agent=ag)
    for m in (obs, sb, sf, lv, ag):
        m.flush()
    stats.update(driver=a.driver, ckpt=f"{seed}_{kind}.pth")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    with open(out / "levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in levels for k in r}), restval="")
        w.writeheader()
        w.writerows(levels)
    (out / "DONE").write_text(time.strftime("%F %T"))
    dur = a.work / "shards_meta" / a.cohort / f"driver{a.driver}"
    dur.mkdir(parents=True, exist_ok=True)
    for fn in ("stats.json", "levels.csv"):
        (dur / fn).write_bytes((out / fn).read_bytes())


# Fallback tolerance, used only if the exact check fails: greedy rollouts flip on floating-point near-ties, so a
# different GPU/cuDNN build gives a slightly different frame stream.
TOL = dict(max_rel_levels_per_driver=0.01, tie_hist_tv=0.005, rel_terminal=0.01)


def tolerance(check, n_frames):
    lv = check["levels_tried_per_driver"]
    th = check["tie_size_hist"]
    keys = set(th["pod"]) | set(th["paper"])
    tc = check["n_terminal_checks_passed"]
    t = dict(max_rel_levels_per_driver=max(abs(x - y) / y for x, y in zip(lv["pod"], lv["paper"])),
             drivers_exact=int(sum(x == y for x, y in zip(lv["pod"], lv["paper"]))),
             tie_hist_tv=sum(abs(th["pod"].get(k, 0) - th["paper"].get(k, 0)) for k in keys) / 2 / n_frames,
             rel_terminal=abs(tc["pod"] - tc["paper"]) / tc["paper"])
    t["passed"] = all(t[k] <= v for k, v in TOL.items())
    return t


def cmd_merge(a):
    """Concatenate driver shards in driver order; check the buggy half reproduces the paper's bank stats exactly."""
    sh = a.scratch / "shards" / a.cohort
    bank = a.scratch / "bank" / a.cohort
    bank.mkdir(parents=True, exist_ok=True)
    dur = a.work / "bank" / a.cohort
    dur.mkdir(parents=True, exist_ok=True)
    parts = {k: [np.load(sh / f"driver{d}" / f"{k}.npy", mmap_mode="r") for d in range(6)]
             for k in ("obs_u8", "soft_buggy", "soft_fixed", "level_id", "agent_cell")}
    for k, v in parts.items():
        np.save(bank / f"{k}.npy", np.concatenate(v))
    np.save(bank / "driver_id.npy", np.repeat(np.arange(6, dtype=np.int8), FRAMES_PER_DRIVER))
    st = [json.loads((sh / f"driver{d}" / "stats.json").read_text()) for d in range(6)]
    lv_rows = [r for d in range(6) for r in csv.DictReader(open(sh / f"driver{d}" / "levels.csv"))]
    with open(bank / "levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lv_rows[0]))
        w.writeheader()
        w.writerows(lv_rows)

    sb, sf, lvl = np.load(bank / "soft_buggy.npy"), np.load(bank / "soft_fixed.npy"), np.load(bank / "level_id.npy")
    tie = np.bincount((sb > 0).sum(1), minlength=8)
    unreach = {int(r["level"]) for r in lv_rows if r["spawn_line_reachable_gem_free"] == "False"}
    keep = (sf.sum(1) > 0) & ~np.isin(lvl, list(unreach))
    np.save(bank / "keep_fixed.npy", keep)
    differs = (np.abs(sb - sf).sum(1) > 1e-6)
    paper = json.loads((a.paper_dir / f"ras_{a.cohort}_block3_oracle.json").read_text())["bank_meta"]["oracle_stats"]
    mine = dict(levels_tried_per_driver=[s["n_levels_tried"] for s in st],
                tie_size_hist={str(k): int(v) for k, v in enumerate(tie) if v},
                n_terminal_checks_passed=sum(s["n_terminal_checks_passed"] for s in st),
                n_levels_tried=sum(s["n_levels_tried"] for s in st))
    check = {k: dict(pod=mine[k], paper=paper[k], match=mine[k] == paper[k]) for k in mine}
    drv = np.repeat(np.arange(6), FRAMES_PER_DRIVER)
    summary = dict(
        cohort=a.cohort, reproduces_paper_bank=all(c["match"] for c in check.values()), check=check,
        tolerance=tolerance(check, len(sb)), tolerance_thresholds=TOL,
        n_frames=int(len(sb)), n_frames_label_changed=int(differs.sum()),
        frac_frames_label_changed=float(differs.mean()),
        n_frames_fixed_label_empty=int((sf.sum(1) == 0).sum()),
        n_bank_levels_used=int(len(np.unique(lvl))),
        n_bank_levels_line_unreachable_gem_free=len(unreach),
        n_bank_levels_used_and_unreachable=int(len(set(np.unique(lvl).tolist()) & unreach)),
        n_frames_fixed_bank=int(keep.sum()),
        fixed_frames_per_driver=[int(keep[drv == d].sum()) for d in range(6)],
        n_levels_with_any_label_change=int(len(np.unique(lvl[differs]))),
    )
    (bank / "merge_check.json").write_text(json.dumps(summary, indent=1))
    (dur / "merge_check.json").write_text(json.dumps(summary, indent=1))
    (dur / "levels.csv").write_bytes((bank / "levels.csv").read_bytes())
    print(json.dumps(summary, indent=1))
    for d in range(6):
        for k in ("obs_u8", "soft_buggy", "soft_fixed", "level_id", "agent_cell"):
            (sh / f"driver{d}" / f"{k}.npy").unlink()
    (bank / "DONE").write_text(time.strftime("%F %T"))


def cmd_evalgeom(a):
    """Gem-free reachability and label differences on the 1000 eval levels (start state only; walls/objects per level)."""
    pod_setup(a.root)
    import head_rewire_ras_maze as hrm
    import am_oracle as rof

    rows = []
    for i in range(EVAL_N):
        level = EVAL_START + i
        env = hrm.make_ras_env(num_envs=1, num_levels=1, start_level=level,
                               use_backgrounds=COHORTS[a.cohort]["use_backgrounds"])
        try:
            line, gem, walk = rof.level_geometry(env, rof.unwrap(env), 5)
        finally:
            env.close()
        db = rof.intended_distances_buggy(walk, line, gem)
        df = rof.intended_distances_fixed(walk, line, gem)
        rows.append(dict(level=level, line_x=line[0], line_y=line[1], gem_x=gem[0], gem_y=gem[1],
                         d_line_buggy=db.get(rof.SPAWN_CELL, -1), d_line_fixed=df.get(rof.SPAWN_CELL, -1),
                         walls=";".join(f"{x},{y}" for x in range(5) for y in range(5) if (x, y) not in walk)))
    out = a.work / "evalgeom"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"eval_geometry_{a.cohort}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_diverge(a):
    """Replay the first levels where two collections of one driver disagree, stepping both device settings in lockstep.

    Reports, per step, the action each setting picks and the top-2 logit margin, so a flip can be attributed to
    floating-point near-ties rather than to code or env differences.
    """
    import torch

    pod_setup(a.root)
    import head_rewire_ras_maze as hrm
    import am_oracle as rof

    rd = lambda p: {int(r["level"]): int(r["n_frames"]) for r in csv.DictReader(open(p / "levels.csv"))}
    la, lb = rd(a.run_a), rd(a.run_b)
    diff = sorted(l for l in set(la) & set(lb) if la[l] != lb[l])
    print(f"levels in both: {len(set(la) & set(lb))}, n_frames differs on {len(diff)}: {diff[:20]}")
    seed, kind = drivers(a.cohort)[a.driver]
    pols = {}
    for name, dev, tf32 in (("cuda_tf32", "cuda", True), ("cuda_fp32", "cuda", False), ("cpu", "cpu", False)):
        pols[name] = (hrm.load_ras_policy(ckpt_path(a.root, seed, kind), dev), dev, tf32)
    for level in diff[: a.n_levels]:
        env = hrm.make_ras_env(num_envs=1, num_levels=1, start_level=level,
                               use_backgrounds=COHORTS[a.cohort]["use_backgrounds"])
        obs = env.reset()
        for t in range(64):
            acts, margins = {}, {}
            for name, (pol, dev, tf32) in pols.items():
                torch.backends.cudnn.allow_tf32 = tf32
                with torch.no_grad():
                    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=dev)
                    d, _v, _h = pol(x, torch.zeros(1, pol.embedder.output_dim, device=dev), torch.zeros(1, device=dev))
                    lg = d.logits[0].double().cpu().numpy()
                o = np.argsort(-lg)
                acts[name], margins[name] = int(o[0]), float(lg[o[0]] - lg[o[1]])
            moves = {k: rof.ACTION_DELTAS[v] for k, v in acts.items()}
            flag = "FLIP" if len(set(moves.values())) > 1 else ""
            if flag or t < 2:
                print(f"level {level} t={t} acts={acts} moves={moves} top2_margin={ {k: round(v, 6) for k, v in margins.items()} } {flag}")
            obs, _r, done, _i = env.step(np.array([acts["cuda_tf32"]]))
            if bool(done[0]):
                break
        env.close()


# ------------------------------------------------------------------ rewire arms (GPU)

def arm_name(role, arm, seed):
    return f"{role}_{arm}_{seed}"


def cmd_arm(a):
    """One (labels, cohort, tap, role, arm, seed): fresh suffix, soft-set BC, 1000-episode closed-loop eval."""
    import torch

    pod_setup(a.root)
    import head_rewire_ras_maze as hrm
    from gm_geometry.rewire import prepare_tap
    from gm_geometry.rewire.ras import eval_closed_loop, random_frozen_fc, tap_features_from_obs, train_tap_bc

    out = a.work / "raw" / a.labels / a.cohort / a.tap / arm_name(a.role, a.arm, a.seed)
    if (out / "summary.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bank = a.scratch / "bank" / a.cohort
    obs = np.load(bank / "obs_u8.npy", mmap_mode="r")
    soft = np.load(bank / f"soft_{a.labels}.npy")
    groups = np.load(bank / "level_id.npy")
    keep = np.load(bank / "keep_fixed.npy") if a.labels == "fixed" else np.ones(len(soft), bool)
    kind = "gm" if a.role == "gm" else "teach"
    ck = ckpt_path(a.root, a.seed, kind)
    feat_tap = "block3" if a.arm == "random_proj" else a.tap

    feat_policy = hrm.load_ras_policy(ck, a.device)
    feats = tap_features_from_obs(feat_policy, obs, feat_tap, device=a.device, batch_size=512)
    del feat_policy
    if not keep.all():
        feats = feats[keep]
    soft, groups = soft[keep], groups[keep]
    assert (soft.sum(1) > 0).all()

    policy = hrm.load_ras_policy(ck, a.device)
    # Seeding the fresh-suffix init makes the arm reproducible and gives the original- and fixed-label arms of one
    # (trunk, tap, arm) the identical initial suffix.
    torch_seed = a.seed * 1000 + TAP_OFFSET[a.tap] + ARM_OFFSET[a.arm]
    torch.manual_seed(torch_seed)
    if a.arm == "random_proj":
        random_frozen_fc(policy, seed=0 + a.seed)  # projection seed = roster seed, as in the paper control
    else:
        prepare_tap(policy, a.tap)
    bc_log = train_tap_bc(policy, feats, soft, feat_tap, device=a.device, prepare=False, groups=groups, **BC)
    del feats
    res = eval_closed_loop(policy, a.device, use_backgrounds=COHORTS[a.cohort]["use_backgrounds"],
                           n_episodes=EVAL_N, max_steps=EVAL_MAX_STEPS, start_level=EVAL_START)
    torch.save({"model_state_dict": policy.state_dict(), "labels": a.labels, "arm": a.arm, "tap": a.tap}, out / "final.pth")
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "reward", "steps"])
        for i, (r, s) in enumerate(zip(res["episode_rewards"], res["episode_steps"])):
            w.writerow([EVAL_START + i, r, s])
    summ = {k: v for k, v in res.items() if not k.startswith("episode_")}
    summ.update(labels=a.labels, cohort=a.cohort, tap=a.tap, role=a.role, arm=a.arm, seed=a.seed,
                torch_seed=torch_seed, n_bank_frames=int(len(soft)), bc_log=bc_log, elapsed_s=time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"{a.labels} {a.cohort} {a.tap} {out.name}: R={res['R_goal']:.3f} D={res['decisive_frac']:.3f} "
          f"P={res['pref_goal']:.3f} ({time.time() - t0:.0f}s)", flush=True)


def all_arms(cohorts, taps):
    """The paper's arm grid: rewire at both taps for GM and Non-GM, plus the fixed-remix control at h_pre."""
    out = []
    for c in cohorts:
        for tap in taps:
            for role, key in (("gm", "gm"), ("nongm", "nongm")):
                for s in COHORTS[c][key]:
                    out.append((c, tap, role, "rewire", s))
                    if tap == "block3":
                        out.append((c, tap, role, "random_proj", s))
    return out


def cmd_launch(a):
    """Fresh subprocess per arm (procgen/torch state leaks across checkpoints in one long process)."""
    jobs = [(lab, *arm) for lab in a.labels for arm in all_arms(a.cohorts, a.taps)
            if arm[2] in a.roles and arm[3] in a.arms and (not a.seeds or arm[4] in a.seeds)]
    logs = a.work / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def go(j):
        lab, c, tap, role, arm, s = j
        if (a.work / "raw" / lab / c / tap / arm_name(role, arm, s) / "summary.json").exists():
            return j, 0
        cmd = [sys.executable, __file__, "arm", "--labels", lab, "--cohort", c, "--tap", tap, "--role", role,
               "--arm", arm, "--seed", str(s), "--root", str(a.root), "--work", str(a.work), "--scratch", str(a.scratch), "--device", a.device]
        with open(logs / f"{lab}_{c}_{tap}_{role}_{arm}_{s}.log", "w") as f:
            return j, subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))

    with ThreadPoolExecutor(a.jobs) as ex:
        for j, rc in ex.map(go, jobs):
            print(" ".join(map(str, j)), f"rc={rc}", flush=True)


# ------------------------------------------------------------------ analysis (local)

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def t_ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m = float(v.mean())
    if len(v) < 2:
        return m, np.nan, np.nan
    h = T975[len(v) - 1] * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def cmd_analyze(a):
    """Raw per-seed CSV first (overall and by affected-level stratum), then 95% t-CI summaries across seeds."""
    import pandas as pd

    geo = pd.read_csv(AUDIT / "state_features_eval1000.csv", usecols=["level", "frac_oracle_paths_through_gem"])
    geo["stratum"] = np.where(geo.frac_oracle_paths_through_gem >= 1, "all_paths_cross_gem",
                              np.where(geo.frac_oracle_paths_through_gem > 0, "some_paths_cross_gem", "unaffected"))
    raw_dir = a.work / "raw"
    eps = []
    for f in sorted(raw_dir.glob("*/*/*/*/episodes.csv")):
        lab, coh, tap, name = f.parts[-5:-1]
        role, arm, seed = name.split("_")[0], "_".join(name.split("_")[1:-1]), int(name.split("_")[-1])
        t = pd.read_csv(f).assign(labels=lab, cohort=coh, tap=tap, role=role, arm=arm, seed=seed)
        eps.append(t)
    ep = pd.concat(eps, ignore_index=True).merge(geo[["level", "stratum"]], on="level", how="left")
    ep["goal"], ep["proxy"] = (ep.reward == 10).astype(float), (ep.reward == -10).astype(float)
    assert ep.stratum.notna().all()

    def metrics(d):
        g, p = d.goal.sum(), d.proxy.sum()
        return pd.Series(dict(n=len(d), R_goal=g / len(d), D=(g + p) / len(d),
                              P_goal_dec=g / (g + p) if g + p else np.nan, P_proxy=p / len(d)))

    keys = ["labels", "cohort", "tap", "role", "arm", "seed"]
    ov = ep.groupby(keys).apply(metrics).reset_index().assign(stratum="overall")
    st = ep.groupby(keys + ["stratum"]).apply(metrics).reset_index()
    aff = ep[ep.stratum != "unaffected"].groupby(keys).apply(metrics).reset_index().assign(stratum="affected_any")
    ps = pd.concat([ov, st, aff], ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ps.to_csv(RESULTS / "per_seed.csv", index=False)

    rows = []
    for k, d in ps.groupby(["labels", "cohort", "tap", "role", "arm", "stratum"]):
        r = dict(zip(["labels", "cohort", "tap", "role", "arm", "stratum"], k), n_seeds=len(d), n_levels=int(d.n.iloc[0]))
        for m in ("R_goal", "D", "P_goal_dec", "P_proxy"):
            r[m], r[m + "_lo"], r[m + "_hi"] = t_ci(d[m])
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS / "summary.csv", index=False)

    con = []
    for k, d in ps.groupby(["labels", "cohort", "tap", "arm", "stratum"]):
        gm = d[d.role == "gm"].sort_values("seed")
        ng = d[d.role == "nongm"].sort_values("seed")
        if len(gm) != len(ng) or not len(gm):
            continue
        for m in ("R_goal", "P_goal_dec"):
            diff = gm[m].to_numpy() - ng[m].to_numpy()  # seed k of GM paired with teacher k, as in the paper
            mm, lo, hi = t_ci(diff)
            con.append(dict(zip(["labels", "cohort", "tap", "arm", "stratum"], k), metric=m,
                            gm_minus_nongm=mm, lo=lo, hi=hi, n_pairs=len(diff)))
    pd.DataFrame(con).to_csv(RESULTS / "contrast_gm_minus_nongm.csv", index=False)

    paper = []
    for coh in COHORTS:
        for tap in ("block3", "fc"):
            f = PAPER_JSON / f"ras_{coh}_{tap}_oracle.json"
            for r in json.loads(f.read_text())["rows"]:
                if r.get("mode") in ("rewire", "random_proj"):
                    lab = r["label"]
                    paper.append(dict(cohort=coh, tap=tap, role=lab.split("_")[0], arm=r["mode"],
                                      seed=int(lab.split("_")[-1]), R_goal=r["R_goal"], D=r["decisive_frac"],
                                      P_goal_dec=r["pref_goal"]))
    pd.DataFrame(paper).to_csv(RESULTS / "paper_per_seed.csv", index=False)
    print(summ[summ.stratum.isin(["overall", "affected_any", "unaffected"])].round(3).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pod = dict(root=PROCGEN_ROOT, work=Path("work/amoracle/work"))
    for name in ("collect", "merge", "evalgeom", "arm", "launch", "analyze", "diverge"):
        q = sub.add_parser(name)
        q.add_argument("--root", type=Path, default=pod["root"])
        q.add_argument("--work", type=Path, default=pod["work"] if name != "analyze" else RESULTS)
        q.add_argument("--device", default="cuda")
        q.add_argument("--scratch", type=Path, default=Path("/tmp/amfix"), help="bulk bank (node-local)")
        if name in ("collect", "merge", "evalgeom", "arm", "diverge"):
            q.add_argument("--cohort", choices=list(COHORTS), required=True)
        if name == "collect":
            q.add_argument("--driver", type=int, choices=range(6), required=True)
            q.add_argument("--threads", type=int, default=1)
            q.add_argument("--no-tf32", action="store_true", help="exact fp32 convs (cuDNN defaults to TF32)")
        if name == "diverge":
            q.add_argument("--driver", type=int, choices=range(6), required=True)
            q.add_argument("--run-a", type=Path, required=True)
            q.add_argument("--run-b", type=Path, required=True)
            q.add_argument("--n-levels", type=int, default=5)
        if name == "merge":
            q.add_argument("--paper-dir", type=Path, default=Path("work/amoracle/paper_ref"))
        if name == "arm":
            q.add_argument("--labels", choices=("buggy", "fixed"), required=True)
            q.add_argument("--tap", choices=("block3", "fc"), required=True)
            q.add_argument("--role", choices=("gm", "nongm"), required=True)
            q.add_argument("--arm", choices=("rewire", "random_proj"), required=True)
            q.add_argument("--seed", type=int, required=True)
        if name == "launch":
            q.add_argument("--labels", nargs="+", default=["fixed"], choices=("buggy", "fixed"))
            q.add_argument("--cohorts", nargs="+", default=list(COHORTS))
            q.add_argument("--taps", nargs="+", default=["block3", "fc"])
            q.add_argument("--roles", nargs="+", default=["gm", "nongm"])
            q.add_argument("--arms", nargs="+", default=["rewire", "random_proj"])
            q.add_argument("--seeds", type=int, nargs="*", default=[])
            q.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    {"collect": cmd_collect, "merge": cmd_merge, "evalgeom": cmd_evalgeom, "arm": cmd_arm,
     "launch": cmd_launch, "analyze": cmd_analyze, "diverge": cmd_diverge}[a.cmd](a)


if __name__ == "__main__":
    main()
