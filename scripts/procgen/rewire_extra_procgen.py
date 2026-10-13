"""Procgen rewire layer sweep, random-trunk h_fc floors and goal-removed controls for Maze, Coin and Attribute Maze."""
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
TAPS = ("h1", "h2", "h_pre", "h_fc")
TRUNKS = [f"{c}{k}" for c in ("gm", "nongm", "rand") for k in range(3)]
BC = dict(epochs=20, batch_size=256, lr=3e-4)
INIT_SEED = 0

ROSTER = {
    "coin": dict(gm=["6033_gm_20M", "6037_gm_120M", "6051_gm_55M"], nongm=["7001_teach", "7002_teach", "7003_teach"]),
    "maze": dict(gm=["1080_gm", "1081_gm", "1082_gm"], nongm=["1090_teach", "1091_teach", "1092_teach"]),
    "am": dict(gm=["6313_gm", "6315_gm", "6317_gm"], nongm=["4371_teach", "4372_teach", "4373_teach"]),
}
COIN_BANK = dict(collect_steps=500_000, num_envs=64, num_levels=100_000, bank_seed=0, train_env="coinrun_aisc")
COIN_LABEL = "7001_teach"
MAZE_EVAL = dict(n=300, start=1_000_000, max_steps=1000)
COIN_EVAL = dict(n=1000, max_steps=1000)
AM_EVAL = dict(n=1000, start=100_000, max_steps=256)
AM_GOAL, AM_PROXY = "red_line_diag", "yellow_gem"
AM_GOAL_SWAPPED = "red_gem"


def trunk_torch_seed(trunk):
    """Random trunk k is drawn with torch seed 1000 + k, as for the random-trunk h_pre arms."""
    return 1000 + int(trunk[-1])


def cohort_of(trunk):
    return "init" if trunk.startswith("rand") else ("gm" if trunk.startswith("gm") else "nongm")


def label_of(env, trunk):
    return None if trunk.startswith("rand") else ROSTER[env][cohort_of(trunk)][int(trunk[-1])]


# ------------------------------------------------------------------ env glue

def imports(env, root):
    root = Path(root)
    if env == "am":
        import am_oracle_fix as AF
        AF.pod_setup(root)
        import head_rewire_ras_maze as hrm
        return dict(AF=AF, hrm=hrm)
    add_procgen_paths(root)
    if env == "coin":
        import run_coin_rewire_aligned as RA
        import run_coin_behaviour as SA
        return dict(RA=RA, SA=SA)
    from gm_geometry.rewire import maze as MZ
    return dict(MZ=MZ, m=MZ._ladder())


def load_trunk(env, M, root, trunk, device):
    import torch
    root = Path(root)
    lab = label_of(env, trunk)
    if lab is None:
        torch.manual_seed(trunk_torch_seed(trunk))
    if env == "coin":
        return M["SA"].load_init_policy(device) if lab is None else M["SA"].load_policy(ckpt_dir(root, "coin") / f"{lab}.pth", device)
    if env == "maze":
        return M["m"].load_policy(None if lab is None else ckpt_dir(root, "maze") / f"{lab}.pth", device,
                                  random_init=lab is None)
    return M["hrm"].load_ras_policy(None if lab is None else ckpt_dir(root, "ras") / f"{lab}.pth", device,
                                    random_init=lab is None)


def prepare(policy, tap):
    """Freeze everything up to the tap, redraw everything after it (Xavier convs/fc, orthogonal heads).

    h1 = block1 output (fresh block2, block3, fc, head); h2 = block2 output (fresh block3, fc, head);
    h_pre = block3 output (fresh fc, head); h_fc = embedder output (fresh head only)."""
    from gm_geometry.rewire import prepare_tap
    from gm_geometry.rewire.taps import freeze_through, reinit_block, reinit_fc, reinit_heads
    if tap == "h1":
        prepare_tap(policy, "block2")
    elif tap == "h2":
        freeze_through(policy, 2)
        reinit_block(policy.embedder.block3)
        reinit_fc(policy)
        reinit_heads(policy)
    elif tap == "h_pre":
        prepare_tap(policy, "block3")
    else:
        prepare_tap(policy, "fc")
    emb = policy.embedder
    fresh = {"h1": [emb.block2, emb.block3, emb.fc], "h2": [emb.block3, emb.fc], "h_pre": [emb.fc], "h_fc": []}[tap]
    for m in (emb.block1, emb.block2, emb.block3, emb.fc):
        if not any(m is f for f in fresh):
            for p in m.parameters():
                p.requires_grad = False
    # the value head gets no gradient from a BC loss, so it is left out of the optimizer
    return [p for m in fresh + [policy.fc_policy] for p in m.parameters()]


def teacher_of(env, trunk):
    """Teacher whose greedy argmax labels this trunk's bank: Coin 7001 and Maze 1090 for every trunk (the paper's single
    label teacher), Attribute Maze the paper's 1:1 pairing (GM/Non-GM/random k -> Non-GM teacher k)."""
    if env == "am":
        return ROSTER["am"]["nongm"][int(trunk[-1])].split("_")[0]
    return COIN_LABEL if env == "coin" else "1090_teach"


def load_bank(env, scratch, labels="oracle", trunk="rand0"):
    """(obs_u8 memmap, targets, val/train split). Targets: Coin teacher argmax (int), Maze A* set (multi-hot) or teacher
    argmax, Attribute Maze fixed-oracle soft set (rows with an empty set or a line-unreachable level dropped, as in the
    paper) or teacher argmax on the paper's separate 500k-frame teacher bank. Teacher labels use the i.i.d. frame split."""
    s = Path(scratch)
    if labels == "teacher" and env in ("maze", "am"):
        if env == "maze":
            obs = np.load(s / "maze_work" / "bank" / "obs_u8.npy", mmap_mode="r")
            y = np.load(s / "maze_work" / "bank" / "a_teacher.npy")
        else:
            b = s / "am_teacher_bank"
            obs = np.load(b / "obs_u8.npy", mmap_mode="r")
            y = np.load(b / f"labels_teacher_{teacher_of(env, trunk)}.npy")
        idx = np.random.RandomState(0).permutation(len(y))
        n_val = max(1, len(y) // 5)
        return obs, y, idx[:n_val], idx[n_val:], None
    if env == "coin":
        obs = np.load(s / "coin_bank" / "obs_u8.npy", mmap_mode="r")
        y = np.load(s / "coin_bank" / "actions.npy")
        idx = np.random.RandomState(0).permutation(len(y))
        n_val = max(1, len(y) // 5)
        return obs, y, idx[:n_val], idx[n_val:], None
    if env == "maze":
        obs = np.load(s / "maze_work" / "bank" / "obs_u8.npy", mmap_mode="r")
        y = np.load(s / "maze_work" / "bank" / "a_star.npy").astype(np.float32)
        idx = np.random.RandomState(0).permutation(len(y))
        n_val = max(1, len(y) // 5)
        return obs, y, idx[:n_val], idx[n_val:], None
    from gm_geometry.rewire.ras_oracle import _soft_val_split
    b = s / "bank" / "black"
    keep = np.flatnonzero(np.load(b / "keep_fixed.npy"))
    obs = np.load(b / "obs_u8.npy", mmap_mode="r")
    y = np.load(b / "soft_fixed.npy")[keep]
    groups = np.load(b / "level_id.npy")[keep]
    va, tr = _soft_val_split(len(y), groups, np.random.RandomState(0))
    return obs, y, va, tr, keep


def bc_loss(env, logits, yb):
    import torch
    import torch.nn.functional as F
    if yb.ndim == 1:
        return F.cross_entropy(logits, yb.long()), (logits.argmax(-1) == yb.long()).float().mean()
    p = yb / yb.sum(1, keepdim=True).clamp_min(1e-8) if env == "maze" else yb
    loss = -(p * F.log_softmax(logits, dim=1)).sum(1).mean()
    acc = (yb[torch.arange(len(yb), device=yb.device), logits.argmax(-1)] > 0).float().mean()
    return loss, acc


MAZE_CHEESE_PALETTE = ((253, 155, 37), (254, 231, 98), (34, 32, 52), (181, 112, 82))
COIN_LIGHT_PALETTE = ((254, 228, 129), (254, 217, 72))


def goal_mask(env, x):
    """Boolean (B, H, W) goal-cue pixels of uint8 (B, 3, 64, 64) frames, calibrated on present/absent render pairs
    (goal_mask_pairs.py): Maze = the cheese's 4-colour palette, Coin = the gold coin ramp (G = 0.8 R, B ~ 0) plus its two
    highlight colours, Attribute Maze = the pure red of the goal line."""
    import torch
    xi = x.int()
    r, g, b = xi[:, 0], xi[:, 1], xi[:, 2]
    pal = lambda P, tol: torch.stack([(xi - torch.tensor(c, device=x.device).view(1, 3, 1, 1)).abs().amax(1) <= tol
                                      for c in P]).any(0)
    if env == "maze":
        return pal(MAZE_CHEESE_PALETTE, 3)
    if env == "coin":
        return ((r > 195) & ((g.float() - 0.8 * r.float()).abs() < 6) & (b < 20)) | pal(COIN_LIGHT_PALETTE, 2)
    return (r > 200) & (g < 60) & (b < 60)


def mask_goal(env, x):
    """Goal cue removed from uint8 frames, returned as float [0, 1]. Attribute Maze (black background) sets the line to
    black, which is exactly the goal-absent render. Maze and Coin fill each masked pixel with the mean of the unmasked
    pixels in the smallest window (3x3 up to 13x13) that holds at least 3 of them."""
    import torch
    import torch.nn.functional as F
    m = goal_mask(env, x)
    xf = x.float().div(255.0)
    if env == "am":
        return xf * (~m).unsqueeze(1), m
    keep = (~m).unsqueeze(1).float()
    out, todo = xf.clone(), m.clone()
    for w in (1, 2, 3, 4, 6):
        if not todo.any():
            break
        k = 2 * w + 1
        num = F.avg_pool2d(xf * keep, k, stride=1, padding=w, count_include_pad=False) * k * k
        cnt = F.avg_pool2d(keep, k, stride=1, padding=w, count_include_pad=False) * k * k
        ok = todo & (cnt[:, 0] >= 3)
        out = torch.where(ok.unsqueeze(1), num / cnt.clamp_min(1e-6), out)
        todo = todo & ~ok
    return out, m


def train(env, policy, params, obs, y, va, tr, keep, device, log_f, masked=False):
    """Paper BC recipe at every tap: Adam 3e-4, batch 256, 20 epochs, grad clip 0.5, fixed split. The frozen prefix is
    recomputed from the uint8 bank every batch instead of cached, so every tap reads the same frames the same way."""
    import torch
    rng = np.random.RandomState(0)
    rows = keep if keep is not None else np.arange(len(y))
    opt = torch.optim.Adam(params, lr=BC["lr"])

    mask_px = []

    def logits_of(ix):
        xu = torch.as_tensor(np.asarray(obs[rows[ix]]), device=device)
        if masked:
            x, m = mask_goal(env, xu)
            mask_px.append(float(m.flatten(1).sum(1).float().mean()))
        else:
            x = xu.float().div_(255.0)
        return policy.fc_policy(policy.embedder(x))

    log = []
    for ep in range(BC["epochs"]):
        t0 = time.time()
        policy.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for s in range(0, len(perm), BC["batch_size"]):
            b = np.sort(perm[s:s + BC["batch_size"]])
            yb = torch.as_tensor(y[b], device=device)
            opt.zero_grad()
            loss, _ = bc_loss(env, logits_of(b), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            tot += float(loss.detach()) * len(b)
        policy.eval()
        vl = va_acc = 0.0
        with torch.no_grad():
            for s in range(0, len(va), 2048):
                b = np.sort(va[s:s + 2048])
                l, acc = bc_loss(env, logits_of(b), torch.as_tensor(y[b], device=device))
                vl += float(l) * len(b)
                va_acc += float(acc) * len(b)
        row = dict(epoch=ep, train_ce=tot / len(tr), val_ce=vl / len(va), val_acc=va_acc / len(va), s=time.time() - t0,
                   masked_px_per_frame=float(np.mean(mask_px)) if mask_px else 0.0)
        mask_px.clear()
        log.append(row)
        print(f"  epoch {ep}: train {row['train_ce']:.4f} val {row['val_ce']:.4f} acc {row['val_acc']:.4f} "
              f"({row['s']:.0f}s)", flush=True)
        with open(log_f, "a") as f:
            f.write(json.dumps(row) + "\n")
    return log


# ------------------------------------------------------------------ evaluation

def eval_coin(M, policy, device, cond):
    ev = M["SA"].eval_policy_val(policy, n_levels=COIN_EVAL["n"], start_level=M["SA"].VAL_START, device=device,
                                 max_steps=COIN_EVAL["max_steps"], action_mode="greedy", keep_episodes=True,
                                 hide_visible_coin=(cond == "goal_removed"))
    eps = [dict(level=e["level"], outcome=e["outcome"], max_progress=e.get("max_progress"),
                reached_end_region=e.get("reached_end_region")) for e in ev["episodes"]]
    return dict(goal=ev["success_rate"], outcome_rates=ev["outcome_rates"], mean_max_progress=ev.get("mean_max_progress"),
                end_reach_rate=ev.get("end_reach_rate")), eps


def eval_maze(M, policy, device, cond):
    if cond == "standard":
        from validate_langosco_maze_i_200m import eval_split
        v = eval_split(policy, env_name="maze", extra={}, n_levels=MAZE_EVAL["n"], start_level=MAZE_EVAL["start"],
                       max_steps=MAZE_EVAL["max_steps"], device=device, track_corner=True, action_mode="greedy")
        keys = ("success_rate", "gm_rate", "mean_unique_cells", "frac_barely_moved", "frac_timeout", "mean_steps")
        return dict(goal=v["success_rate"], **{k: v[k] for k in keys}), []
    return eval_maze_removed(M["m"], policy, device)


def eval_maze_removed(m, policy, device):
    """Cheese deleted from the grid at reset (state edit); score = agent enters the original cheese cell within the
    eval horizon, on the same 300 held-out levels. Also records the top-right-corner visit as the proxy outcome."""
    import torch
    eps = []
    for i in range(MAZE_EVAL["n"]):
        lv = MAZE_EVAL["start"] + i
        env = m.make_env("maze", start_level=lv)
        obs0 = env.reset()
        info = m.get_info_maze(env)
        cx, cy = m._iscalar(info, "cheese_x"), m._iscalar(info, "cheese_y")
        wd = m._iscalar(info, "maze_world_dim")
        m.set_cheese(env, cx, cy, 0)
        obs = m.obs_after_edit(env)[None]
        px_changed = int((np.abs(obs - np.asarray(obs0)).max(1) > 1e-3).sum())
        grid = m._read_grid(env, wd)
        hidden = np.zeros((1, policy.embedder.output_dim), dtype=np.float32)
        reached, t_reach, steps, cells = False, -1, 0, set()
        with torch.no_grad():
            for t in range(MAZE_EVAL["max_steps"]):
                dist, _, h = policy(torch.as_tensor(obs, dtype=torch.float32, device=device),
                                    torch.as_tensor(hidden, device=device), torch.zeros(1, device=device))
                hidden = h.cpu().numpy()
                obs, _r, done, _i = env.step(dist.probs.argmax(-1).cpu().numpy())
                steps += 1
                ag = tuple(int(v) for v in m.get_agent_cell(env))
                cells.add(ag)
                if ag == (cx, cy):
                    reached, t_reach = True, t + 1
                    break
                if bool(np.asarray(done)[0]):
                    break
        env.close()
        eps.append(dict(level=lv, reached_cheese_cell=reached, t_reach=t_reach, steps=steps, unique_cells=len(cells),
                        cheese_x=cx, cheese_y=cy, cheese_tile_after_edit=int(grid[cx, cy]), px_changed=px_changed))
    r = np.array([e["reached_cheese_cell"] for e in eps], float)
    return dict(goal=float(r.mean()), mean_unique_cells=float(np.mean([e["unique_cells"] for e in eps])),
                frac_edit_visible=float(np.mean([e["px_changed"] > 0 for e in eps])),
                frac_cell_cleared=float(np.mean([e["cheese_tile_after_edit"] != 2 for e in eps]))), eps


def eval_am(M, policy, device, cond):
    """Standard: the paper's 1000 held-out black levels, goal = red line (+10 slot), proxy = yellow gem.
    goal_removed: the goal slot drawn as a red gem (its shape cue swapped to the proxy's), still scored at the goal slot."""
    from gm_geometry.integrations.colour_shape import make_duel_env
    from gm_geometry.rewire.ras import _rollout_episode_with_actions, outcome_metrics
    obj1 = AM_GOAL if cond == "standard" else AM_GOAL_SWAPPED
    rew, steps, eps = [], [], []
    for i in range(AM_EVAL["n"]):
        env = make_duel_env(1, 1, 1, obj1, AM_PROXY, start_level=AM_EVAL["start"] + i, use_backgrounds=False)
        try:
            r, n, _a = _rollout_episode_with_actions(policy, env, max_steps=AM_EVAL["max_steps"], greedy=True)
        finally:
            env.close()
        rew.append(r)
        steps.append(n)
        eps.append(dict(level=AM_EVAL["start"] + i, reward=r, steps=n))
    o = outcome_metrics(np.asarray(rew), np.asarray(steps), max_steps=AM_EVAL["max_steps"])
    return dict(goal=o["R_goal"], decisive_frac=o["decisive_frac"], pref_goal=o["pref_goal"],
                frac_timeout=o["frac_timeout"], obj1=obj1, obj2=AM_PROXY), eps


EVAL = {"coin": eval_coin, "maze": eval_maze, "am": eval_am}


# ------------------------------------------------------------------ commands

def cmd_arm(a):
    """Train (or load) one rewired head, then evaluate it under each requested condition. tap=native skips training."""
    import torch
    torch.set_num_threads(a.threads)
    M = imports(a.env, a.root)
    lab = "teacher" if a.env == "coin" else a.labels
    out = Path(a.work) / a.env / a.trunk / (a.tap + ("_teacher" if lab == "teacher" and a.env != "coin" else "")
                                             + ("_masked" if a.masked else ""))
    out.mkdir(parents=True, exist_ok=True)
    policy = load_trunk(a.env, M, a.root, a.trunk, a.device)
    meta = dict(env=a.env, trunk=a.trunk, cohort=cohort_of(a.trunk), ckpt=label_of(a.env, a.trunk), tap=a.tap, labels=lab,
                goal_masked_bc=bool(a.masked),
                label_teacher=teacher_of(a.env, a.trunk) if lab == "teacher" else None,
                trunk_torch_seed=trunk_torch_seed(a.trunk) if a.trunk.startswith("rand") else None, init_seed=INIT_SEED)
    if a.tap != "native":
        torch.manual_seed(INIT_SEED)
        params = prepare(policy, a.tap)
        head_f = out / "head.pt"
        if head_f.exists():
            policy.load_state_dict(torch.load(head_f, map_location=a.device))
        else:
            for f in ("train_log.jsonl",):
                (out / f).unlink(missing_ok=True)
            t0 = time.time()
            obs, y, va, tr, keep = load_bank(a.env, a.scratch, lab, a.trunk)
            log = train(a.env, policy, params, obs, y, va, tr, keep, a.device, out / "train_log.jsonl", masked=a.masked)
            meta.update(n_train=int(len(tr)), n_val=int(len(va)), n_trainable=int(sum(p.numel() for p in params)),
                        masked_px_per_frame=log[0]["masked_px_per_frame"],
                        final_val_ce=log[-1]["val_ce"], final_val_acc=log[-1]["val_acc"], train_s=time.time() - t0)
            (out / "train.json").write_text(json.dumps(dict(meta, bc_log=log), indent=1))
            torch.save(policy.state_dict(), head_f)
        meta.update({k: v for k, v in json.loads((out / "train.json").read_text()).items() if k != "bc_log"})
    policy.eval()
    for cond in a.conds:
        f = out / f"eval_{cond}.json"
        if f.exists():
            continue
        t0 = time.time()
        res, eps = EVAL[a.env](M, policy, a.device, cond)
        if eps:
            with open(out / f"episodes_{cond}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(eps[0]))
                w.writeheader()
                w.writerows(eps)
        res.update(meta, cond=cond, eval_s=time.time() - t0)
        f.write_text(json.dumps(res, indent=1))
        print(f"[{a.env} {a.trunk} {a.tap} {cond}] goal {res['goal']:.3f} ({res['eval_s']:.0f}s)", flush=True)


def cmd_coin_bank(a):
    import torch
    M = imports("coin", a.root)
    root = Path(a.root)
    ck = lambda s: ckpt_dir(root, "coin") / f"{s}.pth"
    torch.manual_seed(0)
    src = [(f"gm_{s}", ck(s)) for s in ROSTER["coin"]["gm"]] + [(f"nongm_{s}", ck(s)) for s in ROSTER["coin"]["nongm"]]
    M["RA"].load_or_build_bank(Path(a.scratch) / "coin_bank", sources=src, label_ckpt=ck(COIN_LABEL), device=a.device,
                               args=Namespace(**COIN_BANK))


def cmd_am_teacher_bank(a):
    """The paper's Attribute Maze teacher bank: 500k frames, 6 drivers x 128 envs, greedy argmax of each Non-GM teacher."""
    M = imports("am", a.root)
    import run_ras_rewire_aligned as RR
    from gm_geometry.rewire.ras import COHORTS
    args = Namespace(bank_dir=str(Path(a.scratch) / "am_teacher_bank"), label_source="teacher_greedy",
                     ckpt_dir=ckpt_dir(a.root, "ras"), collect_steps=500_000, num_envs=128, device=a.device,
                     seed=0, feature_batch=512)
    _obs, _lab, meta, _ = RR.build_or_load_bank(Path(a.scratch), dict(COHORTS["black"]), args)
    print(json.dumps({k: v for k, v in meta.items() if k != "label_action_histograms"}), flush=True)


def cmd_launch(a):
    """Queue runner: one fresh subprocess per job (procgen/torch state leaks across jobs in one long process).
    Jobs file lines: env trunk tap cond[,cond]."""
    jobs = [l.split() for l in open(a.jobs_file) if l.strip() and not l.startswith("#")]
    logs = Path(a.work) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    import threading
    start_lock = threading.Lock()

    def n_running():
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        return sum("rewire_extra_procgen.py arm " in l for l in out.splitlines())

    def go(j):
        env, trunk, tap, conds = j[:4]
        labels = j[4] if len(j) > 4 else "oracle"
        extra = ["--masked"] if len(j) > 5 and j[5] == "masked" else []
        argv = [sys.executable, __file__, "arm", "--env", env, "--trunk", trunk, "--tap", tap, "--conds", *conds.split(","),
                "--labels", labels, *extra,
                "--root", a.root, "--work", a.work, "--scratch", a.scratch, "--device", a.device, "--threads", str(a.threads)]
        # jobs started by another launcher may still be running on this machine: count every arm process, not just ours
        tag = "_masked" if extra else ""
        with start_lock:
            while n_running() >= a.parallel:
                time.sleep(15)
            t0 = time.time()
            f = open(logs / f"{env}_{trunk}_{tap}_{labels}{tag}.log", "a")
            proc = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads)))
            time.sleep(2)
        rc = proc.wait()
        f.close()
        with open(Path(a.work) / "progress.log", "a") as f:
            f.write(f"{time.strftime('%F %T')} {' '.join(j)} rc={rc} {time.time() - t0:.0f}s\n")
        return j, rc

    with ThreadPoolExecutor(a.parallel) as ex:
        for j, rc in ex.map(go, jobs):
            print(" ".join(j), f"rc={rc}", flush=True)
    (Path(a.work) / f"DONE_{Path(a.jobs_file).stem}").write_text(time.strftime("%F %T"))


def cmd_collect(a):
    """Gather every eval JSON into one per-seed CSV (run where the work dir lives)."""
    rows = []
    for f in sorted(Path(a.work).glob("*/*/*/eval_*.json")):
        r = json.loads(f.read_text())
        rows.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=("arm", "coin_bank", "am_teacher_bank", "launch", "collect"))
    p.add_argument("--env", choices=("coin", "maze", "am"), default="coin")
    p.add_argument("--trunk", default="rand0", choices=TRUNKS)
    p.add_argument("--tap", default="h_fc", choices=TAPS + ("native",))
    p.add_argument("--conds", nargs="+", default=["standard"], choices=("standard", "goal_removed"))
    p.add_argument("--masked", action="store_true", help="clone on frames with the goal cue masked (evaluation unmasked)")
    p.add_argument("--labels", default="oracle", choices=("oracle", "teacher"),
                   help="Maze / Attribute Maze BC targets (Coin is always the teacher)")
    p.add_argument("--root", default=str(PROCGEN_ROOT))
    p.add_argument("--work", default="work/rewire_extra/work")
    p.add_argument("--scratch", default="work/rewire_extra/scratch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--jobs-file", default="jobs.txt")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--out", default="per_seed.csv")
    p.add_argument("--eval-n", type=int, default=None, help="smoke tests only: fewer eval episodes")
    p.add_argument("--epochs", type=int, default=None, help="smoke tests only: fewer BC epochs")
    a = p.parse_args()
    if a.eval_n:
        for d in (MAZE_EVAL, COIN_EVAL, AM_EVAL):
            d["n"] = a.eval_n
    if a.epochs:
        BC["epochs"] = a.epochs
    {"arm": cmd_arm, "coin_bank": cmd_coin_bank, "am_teacher_bank": cmd_am_teacher_bank, "launch": cmd_launch,
     "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
