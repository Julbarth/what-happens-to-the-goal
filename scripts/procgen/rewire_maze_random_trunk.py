"""Maze h_pre rewire with a random-init trunk vs GM/Non-GM trunks, on a rebuild of the paper's aligned shared bank."""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from procgen_paths import PROCGEN_ROOT, add_procgen_paths, ckpt_dir

STAGES = ("oracle_check", "bank_part", "bank_merge", "arm", "chance", "head_stats", "summarise")
N_ACTIONS = 15
GM_SEEDS = ["1080_gm", "1081_gm", "1082_gm"]
NONGM_SEEDS = ["1090_teach", "1091_teach", "1092_teach"]
LABEL_TEACHER = "1090_teach"


def procgen_imports(root):
    add_procgen_paths(root)
    from gm_geometry.rewire import maze as MZ
    import run_maze_rewire_aligned as RA
    return MZ, RA, MZ._ladder()


def ckpt(root, label):
    return ckpt_dir(root, "maze") / f"{label}.pth"


def oracle_check(a):
    """Follow the BFS oracle open-loop in maze_aisc and in maze: it must reach the cheese in exactly the BFS length."""
    MZ, _, m = procgen_imports(a.procgen_root)
    rows = []
    for env_name, extra, lv0 in (("maze_aisc", {"rand_region": 25}, a.train_start_level), ("maze", {}, a.eval_start_level)):
        for lv in range(lv0, lv0 + a.oracle_levels):
            env = m.make_env(env_name, start_level=lv, extra=extra)
            env.reset()
            info = m.get_info_maze(env)
            wd = m._iscalar(info, "maze_world_dim")
            grid = m._read_grid(env, wd)
            cx, cy = m._iscalar(info, "cheese_x"), m._iscalar(info, "cheese_y")
            ax, ay = m.get_agent_cell(env)
            a0, L = m.first_step_action(grid, (ax, ay), (cx, cy))
            vals = np.unique(grid).tolist()
            steps, rew, done, mism = 0, 0.0, False, 0
            while not done and steps < 500:
                info = m.get_info_maze(env)
                ax, ay = m.get_agent_cell(env)
                act, _ = m.first_step_action(grid, (ax, ay), (cx, cy))
                if act is None:
                    mism += 1
                    break
                _o, r, d, _i = env.step(np.array([act]))
                rew += float(r[0])
                done = bool(d[0])
                steps += 1
            rows.append(dict(env=env_name, level=lv, bfs_len=L, steps=steps, reward=rew, done=done,
                             grid_vals=str(vals), stuck=mism))
            env.close()
    out = Path(a.work_dir) / "oracle_check.json"
    out.write_text(json.dumps(rows))
    for env_name in ("maze_aisc", "maze"):
        r = [x for x in rows if x["env"] == env_name]
        ok = np.mean([x["reward"] > 0 and x["steps"] == x["bfs_len"] for x in r])
        print(f"{env_name}: oracle reaches cheese in exactly BFS length on {ok:.3f} of {len(r)} levels; "
              f"grid values {sorted(set(v for x in r for v in json.loads(x['grid_vals'])))}", flush=True)


def bank_part(a):
    """One driver's slice of collect_balanced_obs_bank_maze (identical level range to the sequential collector)."""
    import torch
    torch.set_num_threads(a.threads)
    MZ, _, m = procgen_imports(a.procgen_root)
    labels = GM_SEEDS + NONGM_SEEDS
    d = labels[a.driver]
    drv = m.load_policy(ckpt(a.procgen_root, d), "cpu")
    teacher = m.load_policy(ckpt(a.procgen_root, LABEL_TEACHER), "cpu")
    per = int(np.ceil(a.collect_steps / len(labels)))
    # the sequential collector offsets driver i by i * n_levels_to_try; reproduce that with a one-driver call
    b = MZ.collect_balanced_obs_bank_maze([(d, drv)], teacher, device="cpu", frames_per_driver=per,
                                          start_level=a.train_start_level + a.driver * a.n_levels_to_try,
                                          n_levels_to_try=a.n_levels_to_try)
    np.savez(Path(a.work_dir) / "bank_parts" / f"part{a.driver}.npz", obs_u8=b["obs_u8"], a_teacher=b["a_teacher"],
             a_star=b["a_star"], driver_id=np.full(len(b["a_teacher"]), a.driver))


def bank_merge(a):
    MZ, _, _ = procgen_imports(a.procgen_root)
    parts = [np.load(Path(a.work_dir) / "bank_parts" / f"part{i}.npz") for i in range(6)]
    bd = Path(a.work_dir) / "bank"
    bd.mkdir(parents=True, exist_ok=True)
    for k in ("obs_u8", "a_teacher", "a_star", "driver_id"):
        np.save(bd / f"{k}.npy", np.concatenate([p[k] for p in parts]))
    n = sum(len(p["a_teacher"]) for p in parts)
    meta = dict(marker=MZ.ALIGNED_MAZE_MARKER, n_frames=n, frames_per_driver=len(parts[0]["a_teacher"]),
                drivers=GM_SEEDS + NONGM_SEEDS, start_level=a.train_start_level, n_actions=15,
                label_sources=["teacher_greedy", "a_star_set"],
                state_filter="BFS-to-cheese resolves (identical for both label sources)", rebuilt_on="cpu, per-driver parallel")
    (bd / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"merged bank: {n} frames", flush=True)


def eval_maze(pol, a, mode, eval_seed=None):
    """eval_cheese_reach with a selectable action mode; sampled mode seeds torch's global RNG, which drives
    Categorical.sample() inside rollout_episode."""
    import torch
    from validate_langosco_maze_i_200m import eval_split
    if eval_seed is not None:
        torch.manual_seed(10_000 + eval_seed)
    v = eval_split(pol, env_name="maze", extra={}, n_levels=a.eval_n, start_level=a.eval_start_level, max_steps=1000,
                   device="cpu", track_corner=True, action_mode=mode)
    return dict(cheese_reach=v["success_rate"], gm_rate=v["gm_rate"], mean_unique_cells=v["mean_unique_cells"],
                mean_path_length=v["mean_path_length"], mean_frac_no_move=v["mean_frac_no_move"],
                frac_barely_moved=v["frac_barely_moved"], frac_timeout=v["frac_timeout"], action_mode=mode,
                eval_seed=eval_seed, n_levels=a.eval_n, start_level=a.eval_start_level)


def arm_name(a):
    cohort = "init" if a.trunk.startswith("rand") else ("gm" if a.trunk in GM_SEEDS else "nongm")
    return cohort, f"{cohort}_{a.arm}_{a.tap}_{a.label_source}_{a.trunk}"


def run_arm_sampled(a):
    """Rebuild the head deterministically (same seeds, CPU, thread count), check its BC log against the stored
    arm JSON, save it, then evaluate it with actions sampled from the head's softmax."""
    import torch
    torch.set_num_threads(a.threads)
    _, RA, m = procgen_imports(a.procgen_root)
    from gm_geometry.rewire.maze import features_from_obs
    from gm_geometry.rewire.taps import prepare_tap
    cohort, name = arm_name(a)
    wd = Path(a.work_dir)
    head_f = wd / "heads" / f"{name}.pt"
    head_f.parent.mkdir(parents=True, exist_ok=True)
    if a.trunk.startswith("rand"):
        torch.manual_seed(1000 + int(a.trunk[4:]))
        pol = m.load_policy(None, "cpu", random_init=True)
    else:
        pol = m.load_policy(ckpt(a.procgen_root, a.trunk), "cpu")
    torch.manual_seed(a.init_seed)
    prepare_tap(pol, a.tap)
    if head_f.exists():
        pol.load_state_dict(torch.load(head_f, map_location="cpu"))
    else:
        t0 = time.time()
        obs = np.load(wd / "bank" / "obs_u8.npy", mmap_mode="r")
        labels = np.load(wd / "bank" / ("a_teacher.npy" if a.label_source == "teacher" else "a_star.npy"))
        feats = features_from_obs(pol, obs, a.tap, device="cpu")
        fwd, params, _ = RA.build_arm(pol, a.tap, a.arm)
        log = RA.train(fwd, params, feats, labels, label_source=a.label_source, device="cpu", epochs=a.bc_epochs,
                       batch_size=a.batch_size, lr=a.lr)
        ref = json.load(open(wd / "arms" / f"{name}.json"))["bc_log"]
        dev = max(abs(x["val_ce"] - y["val_ce"]) + abs(x["val_acc"] - y["val_acc"]) for x, y in zip(log, ref))
        (wd / "heads" / f"{name}.repro.json").write_text(json.dumps(dict(max_bc_log_dev=dev, bc_log=log,
                                                                         train_s=time.time() - t0)))
        print(f"[{name}] rebuilt head, max |bc_log - RunA| = {dev:.2e}", flush=True)
        torch.save(pol.state_dict(), head_f)
    od = wd / "arms_sampled"
    od.mkdir(parents=True, exist_ok=True)
    for es in a.eval_seeds:
        t0 = time.time()
        ev = eval_maze(pol, a, "sample", es)
        json.dump(dict(trunk=a.trunk, cohort=cohort, arm=a.arm, tap=a.tap, label_source=a.label_source, **ev),
                  open(od / f"{name}_es{es}.json", "w"))
        print(f"[{name} es{es}] sampled reach {ev['cheese_reach']:.3f} corner {ev['gm_rate']:.3f} "
              f"cells {ev['mean_unique_cells']:.1f} barely {ev['frac_barely_moved']:.2f} ({time.time() - t0:.0f}s)",
              flush=True)


def head_stats(a):
    """Entropy and optimal-set probability mass of each saved head on a 20k subsample of the bank's validation split:
    under sampled actions a less confident head is closer to a uniform random walk regardless of what it encodes."""
    import torch
    torch.set_num_threads(a.threads)
    _, _, m = procgen_imports(a.procgen_root)
    from gm_geometry.rewire.taps import prepare_tap
    wd = Path(a.work_dir)
    obs = np.load(wd / "bank" / "obs_u8.npy", mmap_mode="r")
    astar = np.load(wd / "bank" / "a_star.npy")
    n = len(astar)
    val = np.sort(np.random.RandomState(0).permutation(n)[: n // 5][:20_000])
    x = torch.as_tensor(np.asarray(obs[val]), dtype=torch.float32) / 255.0
    y = torch.as_tensor(astar[val]).float()
    out = []
    for f in sorted((wd / "heads").glob("*.pt")):
        trunk = f.stem.split("_a_star_")[1]
        if trunk.startswith("rand"):
            torch.manual_seed(1000 + int(trunk[4:]))
            pol = m.load_policy(None, "cpu", random_init=True)
        else:
            pol = m.load_policy(ckpt(a.procgen_root, trunk), "cpu")
        prepare_tap(pol, a.tap)
        pol.load_state_dict(torch.load(f, map_location="cpu"))
        with torch.no_grad():
            pr = torch.cat([pol(x[i:i + 1000], torch.zeros(len(x[i:i + 1000]), 256), torch.zeros(len(x[i:i + 1000])))[0].probs
                            for i in range(0, len(x), 1000)])
        ent = float(-(pr * pr.clamp_min(1e-12).log()).sum(1).mean())
        mass = float((pr * (y > 0)).sum(1).mean())
        out.append(dict(head=f.stem, entropy_nats=ent, optimal_set_mass=mass, uniform_entropy=float(np.log(N_ACTIONS)),
                        uniform_optimal_set_mass=float(((y > 0).sum(1) / N_ACTIONS).mean())))
        print(out[-1], flush=True)
    (wd / "head_stats.json").write_text(json.dumps(out, indent=1))


class UniformPolicy:
    """Chance policy: uniform over the 15 procgen actions, the same action space the BC heads sample from."""

    def __init__(self):
        import torch
        self.embedder = type("E", (), {"output_dim": 256})()
        self.torch = torch

    def __call__(self, obs, hidden, mask):
        d = self.torch.distributions.Categorical(logits=self.torch.zeros(obs.shape[0], N_ACTIONS))
        return d, None, hidden


def run_chance(a):
    import torch
    torch.set_num_threads(a.threads)
    procgen_imports(a.procgen_root)
    od = Path(a.work_dir) / "arms_sampled"
    od.mkdir(parents=True, exist_ok=True)
    for es in a.eval_seeds:
        ev = eval_maze(UniformPolicy(), a, "sample", es)
        json.dump(dict(trunk="uniform", cohort="chance", arm="uniform", tap="none", label_source="none", **ev),
                  open(od / f"chance_uniform_es{es}.json", "w"))
        print(f"[chance es{es}] reach {ev['cheese_reach']:.3f} corner {ev['gm_rate']:.3f} "
              f"cells {ev['mean_unique_cells']:.1f}", flush=True)


def run_arm(a):
    """trunk = GM/Non-GM checkpoint label or 'rand<k>' (fresh ImpalaModel, torch seed 1000 + k); fc/heads always drawn
    with --init-seed after prepare_tap, exactly as the paper's arms."""
    import torch
    torch.set_num_threads(a.threads)
    if a.sampled:
        return run_arm_sampled(a)
    _, RA, m = procgen_imports(a.procgen_root)
    from gm_geometry.rewire.maze import eval_cheese_reach, features_from_obs
    from gm_geometry.rewire.taps import prepare_tap
    bd = Path(a.work_dir) / "bank"
    obs = np.load(bd / "obs_u8.npy", mmap_mode="r")
    labels = np.load(bd / ("a_teacher.npy" if a.label_source == "teacher" else "a_star.npy"))
    t0 = time.time()
    if a.trunk.startswith("rand"):
        torch.manual_seed(1000 + int(a.trunk[4:]))
        pol = m.load_policy(None, "cpu", random_init=True)
        cohort = "init"
    else:
        pol = m.load_policy(ckpt(a.procgen_root, a.trunk), "cpu")
        cohort = "gm" if a.trunk in GM_SEEDS else "nongm"
    native = None
    if a.native:
        native = eval_cheese_reach(pol, "cpu", n_levels=a.eval_n, start_level=a.eval_start_level)
        print(f"native {a.trunk}: {native['cheese_reach']:.3f} corner {native['gm_rate']:.3f}", flush=True)
    torch.manual_seed(a.init_seed)
    prepare_tap(pol, a.tap)
    feats = features_from_obs(pol, obs, a.tap, device="cpu")
    fwd, params, desc = RA.build_arm(pol, a.tap, a.arm)
    log = RA.train(fwd, params, feats, labels, label_source=a.label_source, device="cpu", epochs=a.bc_epochs,
                   batch_size=a.batch_size, lr=a.lr)
    ev = eval_cheese_reach(pol, "cpu", n_levels=a.eval_n, start_level=a.eval_start_level)
    res = dict(trunk=a.trunk, cohort=cohort, arm=a.arm, tap=a.tap, label_source=a.label_source, init_seed=a.init_seed,
               n_frames=int(len(labels)), n_trainable=int(sum(p.numel() for p in params)), **desc, bc_log=log,
               rewired_eval=ev, native_eval=native, elapsed_s=time.time() - t0)
    od = Path(a.work_dir) / "arms"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{cohort}_{a.arm}_{a.tap}_{a.label_source}_{a.trunk}.json").write_text(json.dumps(res, indent=1))
    print(f"[{a.trunk} {a.arm} {a.tap} {a.label_source}] reach {ev['cheese_reach']:.3f} corner {ev['gm_rate']:.3f} "
          f"cells {ev['mean_unique_cells']:.1f} val_acc {log[-1]['val_acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)


def ci95(v):
    from scipy import stats
    v = np.asarray(v, float)
    m = v.mean()
    if len(v) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def fmt(v, nd=3):
    m, lo, hi = ci95(v)
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def welch(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(vx + vy)
    d = x.mean() - y.mean()
    return f"{d:.3f} [{d - h:.3f}, {d + h:.3f}]"


def summarise(a):
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(a.arms_dir, "*.json"))):
        r = json.load(open(f))
        ev = r["rewired_eval"]
        rows.append(dict(cohort=r["cohort"], trunk=r["trunk"], arm=r["arm"], tap=r["tap"], labels=r["label_source"],
                         reach=ev["cheese_reach"], corner=ev["gm_rate"], unique_cells=ev["mean_unique_cells"],
                         frac_barely_moved=ev["frac_barely_moved"], val_acc=r["bc_log"][-1]["val_acc"],
                         val_ce=r["bc_log"][-1]["val_ce"],
                         native_reach=(r["native_eval"] or {}).get("cheese_reach", np.nan),
                         native_corner=(r["native_eval"] or {}).get("gm_rate", np.nan)))
    df = pd.DataFrame(rows).sort_values(["labels", "arm", "cohort", "trunk"])
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "runA_maze_rewire_per_seed.csv"), index=False)
    for c in ("gm", "nongm", "init"):
        sub = df[df.cohort == c]
        for s, g in sub.groupby("trunk"):
            g.to_csv(os.path.join(a.out, f"runA_maze_rewire_{c}_{s}.csv"), index=False)

    def get(c, arm, lab="a_star"):
        return df[(df.cohort == c) & (df.arm == arm) & (df.labels == lab)].sort_values("trunk")

    L = ["## Run A tables (generated)", "", "| labels | arm | trunk | n | reach | corner rate | unique cells | BC val acc |",
         "|---|---|---|---|---|---|---|---|"]
    for (lab, arm, c), g in df.groupby(["labels", "arm", "cohort"]):
        L.append(f"| {lab} | {arm} | {c} | {len(g)} | {fmt(g.reach)} | {fmt(g.corner)} | {fmt(g.unique_cells, 1)} | "
                 f"{fmt(g.val_acc)} |")
    gm, ng, ini = get("gm", "rewire"), get("nongm", "rewire"), get("init", "rewire")
    rg, ri = get("gm", "random_proj"), get("init", "random_proj")
    L += [""]
    if len(gm) and len(ng) and len(ini):
        ngm = ng.reach.mean()
        rgm = rg.reach.values if len(rg) == len(gm) else np.full(len(gm), np.nan)
        L.append(f"GM - init (a_star, h_pre, Welch): {welch(gm.reach, ini.reach)}")
        for nm, fl_gm, fl_init in (("GM remix per seed / init remix per seed", rgm, ri.reach.values if len(ri) == len(ini) else np.full(len(ini), np.nan)),
                                   ("GM remix mean (common floor)", np.full(len(gm), rg.reach.mean()), np.full(len(ini), rg.reach.mean())),
                                   ("sampled untrained floor 0.245", np.full(len(gm), 0.245), np.full(len(ini), 0.245))):
            ngm_n = (gm.reach.values - fl_gm) / (ngm - fl_gm)
            ini_n = (ini.reach.values - fl_init) / (ngm - fl_init)
            L.append(f"normalised, floor = {nm}: GM {fmt(ngm_n)}, init {fmt(ini_n)}, GM - init {welch(ngm_n, ini_n)}")
    md = os.path.join(a.out, "runA_tables_generated.md")
    open(md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def summarise_sampled(a):
    """Per (trunk, arm) mean over eval seeds; CIs are over the 3 trunk seeds, never over eval seeds."""
    import pandas as pd
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.arms_dir, "*.json")))]
    ep = pd.DataFrame(rows).rename(columns=dict(cheese_reach="reach", gm_rate="corner", mean_unique_cells="unique_cells"))
    ep = ep[["cohort", "trunk", "arm", "tap", "label_source", "eval_seed", "reach", "corner", "unique_cells",
             "frac_barely_moved", "mean_frac_no_move", "frac_timeout"]].sort_values(["cohort", "arm", "trunk", "eval_seed"])
    os.makedirs(a.out, exist_ok=True)
    ep.to_csv(os.path.join(a.out, "runA_sampled_per_eval_seed.csv"), index=False)
    keys = ["cohort", "trunk", "arm", "label_source"]
    met = ["reach", "corner", "unique_cells", "frac_barely_moved", "mean_frac_no_move"]
    ts = ep.groupby(keys)[met].mean().reset_index()
    ts["n_eval_seeds"] = ep.groupby(keys).size().values
    chance = ts[ts.cohort == "chance"].reach.iloc[0]
    ctrl = ts[(ts.cohort == "nongm") & (ts.arm == "rewire")].reach.mean()
    shared_floor = ts[(ts.cohort == "gm") & (ts.arm == "random_proj")].reach.mean()
    remix = ts[ts.arm == "random_proj"].set_index("trunk").reach
    rw = ts.arm == "rewire"
    ts.loc[rw, "norm_chance"] = (ts.reach - chance) / (ctrl - chance)
    ts.loc[rw, "norm_own_remix"] = [(r - remix[t]) / (ctrl - remix[t]) for t, r in zip(ts[rw].trunk, ts[rw].reach)]
    ts.loc[rw, "norm_shared_remix"] = (ts.reach - shared_floor) / (ctrl - shared_floor)
    ts.to_csv(os.path.join(a.out, "runA_sampled_per_seed.csv"), index=False)
    for c in ("gm", "nongm", "init"):
        for s, g in ts[ts.cohort == c].groupby("trunk"):
            g.to_csv(os.path.join(a.out, f"runA_sampled_{c}_{s}.csv"), index=False)

    def get(c, arm):
        return ts[(ts.cohort == c) & (ts.arm == arm)].sort_values("trunk")

    L = ["## Run A sampled tables (generated)", "",
         f"chance (uniform 15-action, sampled): {fmt(ep[ep.cohort == 'chance'].reach)} over eval seeds; "
         f"control rewire mean {ctrl:.3f}; GM remix mean {shared_floor:.3f}", "",
         "| arm | trunk | n | reach | corner | unique cells | barely moved | frac no-move steps |", "|---|---|---|---|---|---|---|---|"]
    for (arm, c), g in ts[ts.cohort != "chance"].groupby(["arm", "cohort"]):
        L.append(f"| {arm} | {c} | {len(g)} | {fmt(g.reach)} | {fmt(g.corner)} | {fmt(g.unique_cells, 1)} | "
                 f"{fmt(g.frac_barely_moved, 2)} | {fmt(g.mean_frac_no_move, 2)} |")
    gm, ng, ini = get("gm", "rewire"), get("nongm", "rewire"), get("init", "rewire")
    L += ["", "| quantity | GM | control | random init | GM - init (Welch) |", "|---|---|---|---|---|"]
    L.append(f"| raw reach | {fmt(gm.reach)} | {fmt(ng.reach)} | {fmt(ini.reach)} | {welch(gm.reach, ini.reach)} |")
    for col in ("norm_chance", "norm_own_remix", "norm_shared_remix"):
        L.append(f"| {col} | {fmt(gm[col])} | {fmt(ng[col])} | {fmt(ini[col])} | {welch(gm[col], ini[col])} |")
    for c in ("gm", "nongm", "init"):
        d = get(c, "rewire").reach.values - get(c, "random_proj").reach.values
        L.append(f"| rewire - own remix, {c} | {fmt(d)} | | | |")
    md = os.path.join(a.out, "runA_sampled_tables_generated.md")
    open(md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--procgen-root", default=str(PROCGEN_ROOT))
    p.add_argument("--work-dir", default="work/followups/runA/work")
    p.add_argument("--driver", type=int, default=0)
    p.add_argument("--trunk", default="rand0", help="checkpoint label (1080_gm ...) or rand<k>")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="random-trunk seeds (rand<k>), summarise only")
    p.add_argument("--arm", choices=("rewire", "random_proj"), default="rewire")
    p.add_argument("--tap", choices=("block3", "fc"), default="block3")
    p.add_argument("--label-source", choices=("a_star", "teacher"), default="a_star")
    p.add_argument("--native", action="store_true", help="also evaluate the untouched trunk policy")
    p.add_argument("--collect-steps", type=int, default=200_000)
    p.add_argument("--train-start-level", type=int, default=2_000_000)
    p.add_argument("--n-levels-to-try", type=int, default=20_000)
    p.add_argument("--bc-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-n", type=int, default=300)
    p.add_argument("--eval-start-level", type=int, default=1_000_000)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--oracle-levels", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--sampled", action="store_true",
                   help="arm: rebuild the saved head and eval with sampled actions; summarise: sampled tables")
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2], help="sampled-action eval seeds")
    p.add_argument("--arms-dir", default="results/analysis/deepdive_followups/runA_raw")
    p.add_argument("--out", default="results/analysis/deepdive_followups")
    a = p.parse_args()
    os.makedirs(os.path.join(a.work_dir, "bank_parts"), exist_ok=True) if a.stage != "summarise" else None
    {"oracle_check": oracle_check, "bank_part": bank_part, "bank_merge": bank_merge, "arm": run_arm,
     "chance": run_chance, "head_stats": head_stats, "summarise": summarise_sampled if a.sampled else summarise}[a.stage](a)


if __name__ == "__main__":
    main()
