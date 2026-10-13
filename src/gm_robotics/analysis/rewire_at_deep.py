"""AttributeTable rewiring analyses: chance and scripted-oracle baselines, longer episodes, BC data variants, DAgger, probes."""
import argparse
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis import rewire_attr as R
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from gm_robotics.analysis.rewire_lr import Rewired, bank_features, train_bc
from gm_robotics.analysis.steer_ext import sync_copies

OUT = os.path.join(C.RESULTS, "rewire_at_deep")
HEADS = os.path.join(OUT, "heads")
SEEDS = R.SEEDS["AT"]
T_LONG = 100
BANK2_SEED, DAGGER_SEED = 4201, 4400
DAGGER_CFG = 672  # per round; two rounds add as many episodes as bank 2, so DAgger and D2 are size-matched
# Scripted oracle (joint-space damped least squares on the TCP Jacobian): hover above the cube, then descend.
Z_HOVER, Z_TOUCH, XY_TOL, STEP, LAM = 0.12, 0.065, 0.02, 0.02, 1e-4

# variant -> training spec. banks: which BC banks; label: teacher | oracle; subset: frame filter on bank 1.
VARIANTS = {
    "e60": dict(banks=(1,), epochs=60, label="teacher"),
    "d2": dict(banks=(1, 2), epochs=20, label="teacher"),
    "d2e60": dict(banks=(1, 2), epochs=60, label="teacher"),
    "oracle": dict(banks=(1,), epochs=20, label="oracle"),
    "oracle_d2e60": dict(banks=(1, 2), epochs=60, label="oracle"),
    "ctrlvar": dict(banks=(1,), epochs=20, label="teacher", subset="ctrl_variant"),
    "gmvar": dict(banks=(1,), epochs=20, label="teacher", subset="gm_variant"),
    "half": dict(banks=(1,), epochs=20, label="teacher", subset="half"),
    "ctrlsrc": dict(banks=(1,), epochs=20, label="teacher", subset="ctrl_source"),
    "gmsrc": dict(banks=(1,), epochs=20, label="teacher", subset="gm_source"),
    # control-distribution scenes only (colour independent of shape) from banks 1 + 2, which matches the full
    # bank 1 frame count (~38.6k) and episode count (1,344)
    "ctrl2": dict(banks=(1, 2), epochs=20, label="teacher", subset="ctrl_variant",
                  arms=("pre", "pre_masked", "fc_head", "remix")),
    "ctrl2_oracle": dict(banks=(1, 2), epochs=20, label="oracle", subset="ctrl_variant",
                         arms=("pre", "pre_masked", "fc_head", "remix")),
}
ALL5 = ("standard", "nogoal_sph", "nogoal_cube", "noproxy_blue", "noproxy_red")
SHORT = ("standard", "nogoal_sph")


def head_path(variant, cohort, k, arm):
    if variant == "base":
        return R.head_path("AT", cohort, k, arm)
    return os.path.join(HEADS, f"{variant}_{cohort}_s{k}_{arm}.pt")


def make_env(n, T, render=False):
    env = gym.make("AttributeTableAttr-v1", variant="conflict", num_envs=n, obs_mode="rgb", sim_backend="physx_cuda",
                   control_mode="pd_joint_delta_pos", reconfiguration_freq=None, max_episode_steps=T + 1,
                   render_mode="rgb_array" if render else None)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return env


def bc_cfg(n, seed):
    """Same construction as rewire_attr.bc_configs (even = GM-training scenes, odd = control scenes), other seed."""
    cfg = E.sample_at_probe_configs(n, seed)
    gm_var = np.arange(n) % 2 == 0
    cfg["gm_variant"] = gm_var
    cfg["cube_red"] = np.where(gm_var, True, cfg["cube_red"])
    cfg["sphere_red"] = np.where(gm_var, False, cfg["sphere_red"])
    return cfg


# ---------------------------------------------------------------- oracle

def oracle_arm_action(qpos, target_xy):
    """Normalised 7-d joint-delta action moving the TCP <= STEP m toward the hover / touch point above target_xy."""
    p, J = E.tcp_pos_and_jac(qpos)
    dxy = np.linalg.norm(target_xy - p[:, :2], axis=1)
    z = np.where(dxy > XY_TOL, Z_HOVER, Z_TOUCH)
    d = np.concatenate([target_xy, z[:, None]], 1) - p
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    d = d * np.minimum(1.0, STEP / np.maximum(nrm, 1e-9))
    JJt = J @ J.transpose(0, 2, 1) + LAM * np.eye(3, dtype=np.float32)
    dq = (J.transpose(0, 2, 1) @ np.linalg.solve(JJt, d[..., None]))[..., 0]
    a = dq / C.JOINT_DELTA_SCALE
    return (a / np.maximum(1.0, np.abs(a).max(1, keepdims=True))).astype(np.float32)


def grip_constant():
    """Gripper label for the oracle: median teacher gripper action on bank 1 (keeps finger appearance in distribution)."""
    f = os.path.join(OUT, "oracle_grip.npy")
    if os.path.exists(f):
        return float(np.load(f))
    raise FileNotFoundError(f)


def oracle_labels(state, target_xy, grip, bs=8192):
    out = []
    for sl in C.batched(len(state), bs):
        a = oracle_arm_action(np.asarray(state[sl, :9]), np.asarray(target_xy[sl]))
        out.append(np.concatenate([a, np.full((len(a), 1), grip, np.float32)], 1))
    return np.concatenate(out)


# ---------------------------------------------------------------- closed-loop eval (T = 100, T = 50 derived from the same path)

@torch.no_grad()
def run_eval(env, act_factory, cfg, cond, bs, T):
    """Deterministic rollouts on the 512 held-out conflict scenes; scoring = rewire_attr.at_first_touch on the TCP path.
    The first 50 steps are the same trajectory as a 50-step episode, so outcome_50 reproduces rewire_attr."""
    spec_u, spec_v = R.CONDS["AT"][cond]
    n = len(cfg["u"])
    res, paths = [], []
    for sl in C.batched(n, bs):
        idx = np.arange(sl.start, sl.stop)
        pad = np.concatenate([idx, np.full(bs - len(idx), idx[-1])])
        c = R.take(cfg, pad)
        pl = R.place("AT", c, spec_u, spec_v)
        R.force("AT", env, c["qpos"], pl)
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        act = act_factory(c, pl)
        tcp = [u.agent.tcp.pose.p.cpu().numpy()]
        for t in range(T):
            obs, *_ = env.step(act(obs["rgb"], obs["state"]))
            tcp.append(u.agent.tcp.pose.p.cpu().numpy())
        path = np.stack(tcp, 1)
        r = {}
        for TT in (50, T):
            o, tu, tv, mdu, mdv = R.at_first_touch(path[:, : TT + 1], pl)
            r.update({f"outcome_{TT}": o, f"t_u_{TT}": tu, f"t_v_{TT}": tv, f"min_d_u_{TT}": mdu, f"min_d_v_{TT}": mdv})
        fin = path[:, -1, :2]
        r["final_d_u"] = np.linalg.norm(fin - pl["pos_u"][:, :2], axis=1)
        r["final_d_v"] = np.linalg.norm(fin - pl["pos_v"][:, :2], axis=1)
        r = {k: v[: len(idx)] for k, v in r.items()}
        r["episode"] = idx
        res.append(pd.DataFrame(r))
        paths.append(path[: len(idx)].astype(np.float32))
    return pd.concat(res, ignore_index=True), np.concatenate(paths)


def policy_factory(variant, cohort, k, arm, dev):
    if variant == "base":
        act = R.load_policy("AT", cohort, k, arm, dev, rw_seed=1000 * k)
        return lambda c, pl: act
    base = R.base_agent("AT", cohort, k, dev)
    model = Rewired(base, "pre" if arm == "pre_masked" else arm, init_seed=100 + k).to(dev)
    ck = torch.load(head_path(variant, cohort, k, arm), map_location="cpu")
    model.load_state_dict({**{n: p for n, p in model.state_dict().items() if n.startswith("base.")}, **ck["state_dict"]})
    model.eval()
    return lambda c, pl: model.act


def chance_factory(name, dev, sig=None, seed=0):
    """Goal-blind or scripted reference policies."""
    grip = grip_constant()
    g = torch.Generator(device=dev).manual_seed(seed)

    def scripted(target_fn):
        def fac(c, pl):
            st0 = {}

            def act(rgb, state):
                s = state.cpu().numpy()
                if "tgt" not in st0:
                    st0["tgt"] = target_fn(c, pl, s)
                a = oracle_arm_action(s[:, :9], st0["tgt"])
                return torch.as_tensor(np.concatenate([a, np.full((len(a), 1), grip, np.float32)], 1), device=dev)
            return act
        return fac

    def near(c, pl, s, far=False):
        p, _ = E.tcp_pos_and_jac(s[:, :9])
        du = np.linalg.norm(p[:, :2] - c["u"], axis=1)
        dv = np.linalg.norm(p[:, :2] - c["v"], axis=1)
        pick_u = (du <= dv) != far
        return np.where(pick_u[:, None], c["u"], c["v"]).astype(np.float32)

    if name == "oracle_u":
        return scripted(lambda c, pl, s: c["u"].astype(np.float32))
    if name == "oracle_v":
        return scripted(lambda c, pl, s: c["v"].astype(np.float32))
    if name == "oracle_near":
        return scripted(near)
    if name == "oracle_far":
        return scripted(lambda c, pl, s: near(c, pl, s, far=True))
    if name == "rand_uniform":
        return lambda c, pl: (lambda rgb, st: torch.rand(len(st), C.N_ACT, generator=g, device=dev) * 2 - 1)
    if name == "rand_walk_teacher":
        return lambda c, pl: (lambda rgb, st: sig * torch.randn(len(st), C.N_ACT, generator=g, device=dev))
    raise KeyError(name)


def per_episode_csv(tag):
    return os.path.join(OUT, f"AT_deep_per_episode_{tag}.csv")


def done_set():
    done = set()
    for f in os.listdir(OUT):
        if f.startswith("AT_deep_per_episode_") and f.endswith(".csv"):
            d = pd.read_csv(os.path.join(OUT, f), usecols=["variant", "cohort", "seed", "arm", "cond"])
            done |= set(d.drop_duplicates().itertuples(index=False, name=None))
    return done


def append(df, tag, path_arr, name):
    f = per_episode_csv(tag)
    df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
    os.makedirs(os.path.join(OUT, "paths"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "paths", name + ".npz"), tcp=path_arr)


def log_line(df, label):
    v100, v50 = df.outcome_100.value_counts(normalize=True), df.outcome_50.value_counts(normalize=True)
    print(f"{label}: T50 u {v50.get('u', 0):.3f} v {v50.get('v', 0):.3f} | T100 u {v100.get('u', 0):.3f} "
          f"v {v100.get('v', 0):.3f} neither {v100.get('neither', 0):.3f}", flush=True)


def chance(a):
    dev = torch.device("cuda")
    cfg, _ = R.eval_scenes("AT")
    env = make_env(a.env_batch, T_LONG)
    done = done_set()
    sig = C.sigma(C.build_agent(C.ckpt_path("AT", "control", 3), dev=dev))
    names = ["oracle_u", "oracle_v", "oracle_near", "oracle_far", "rand_uniform", "rand_walk_teacher"]
    for name in names:
        for cond in (["standard"] + (["nogoal_sph"] if name.startswith("oracle") else [])):
            key = ("chance", "none", 0, name, cond)
            if key in done:
                continue
            df, pth = run_eval(env, chance_factory(name, dev, sig), cfg, cond, a.env_batch, T_LONG)
            df = df.assign(variant="chance", cohort="none", seed=0, arm=name, cond=cond)
            append(df, "chance", pth, f"chance_{name}_{cond}")
            log_line(df, f"chance {name} {cond}")
    env.close()


def rollout_specs(variants):
    for variant in variants:
        for cohort in ("gm", "control", "init"):
            for k in SEEDS[cohort]:
                arms = ("native", "pre", "fc_head", "remix", "pre_masked") if variant == "base" else \
                    ("pre", "pre_masked", "fc_head", "remix")
                for arm in arms:
                    if cohort == "init" and arm in ("native", "pre_masked"):
                        continue
                    if arm != "native" and not os.path.exists(head_path(variant, cohort, k, arm)):
                        continue
                    conds = ALL5 if (arm in ("pre", "native") and cohort != "init") else SHORT
                    yield variant, cohort, k, arm, conds


def rollout(a):
    dev = torch.device("cuda")
    cfg, _ = R.eval_scenes("AT")
    env = make_env(a.env_batch, T_LONG)
    done = done_set()
    specs = list(rollout_specs(a.variants))
    for j, (variant, cohort, k, arm, conds) in enumerate(specs):
        if j % a.n_workers != a.worker:
            continue
        fac = None
        for cond in conds:
            if (variant, cohort, k, arm, cond) in done:
                continue
            fac = fac or policy_factory(variant, cohort, k, arm, dev)
            df, pth = run_eval(env, fac, cfg, cond, a.env_batch, T_LONG)
            df = df.assign(variant=variant, cohort=cohort, seed=k, arm=arm, cond=cond)
            append(df, a.tag or f"w{a.worker}", pth, f"{variant}_{cohort}_s{k}_{arm}_{cond}")
            log_line(df, f"{variant} {cohort} s{k} {arm} {cond}")
    env.close()


# ---------------------------------------------------------------- BC bank 2 and labels

def bank_file(i):
    return os.path.join(C.BANKS, "AT_bc_attr.npz" if i == 1 else f"AT_bc_attr{i}.npz")


def build_bank2(a):
    path = bank_file(2)
    if os.path.exists(path):
        print(f"{path} exists", flush=True)
        return
    dev = torch.device("cuda")
    srcs = [("gm", s) for s in SEEDS["gm"]] + [("control", s) for s in SEEDS["control"]]
    cfg = bc_cfg(a.n_ep * len(srcs), BANK2_SEED)
    env = R.make_env("AT", 2 * a.env_batch)
    parts = []
    for s_i, (cohort, s) in enumerate(srcs):
        agent = C.build_agent(C.ckpt_path("AT", cohort, s), dev=dev)
        torch.manual_seed(BANK2_SEED + s_i)
        for b0 in range(0, a.n_ep, a.env_batch):
            idx = s_i * a.n_ep + np.arange(b0, min(a.n_ep, b0 + a.env_batch))
            pad = np.concatenate([idx, np.full(a.env_batch - len(idx), idx[-1])])
            r, _, _ = R.collect("AT", env, agent, cfg, pad, len(idx), dev)
            r["source"] = np.full(len(r["ep"]), s_i)
            parts.append(r)
        print(f"bank2: {cohort} s{s} done, frames {sum(len(p['ep']) for p in parts)}", flush=True)
    env.close()
    bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    bank["episode"] = bank.pop("ep")
    for k in ("u", "v", "qpos", "gm_variant", "cube_red", "sphere_red"):
        bank["cfg_" + k] = cfg[k]
    np.savez(path + ".tmp.npz", **bank)
    os.replace(path + ".tmp.npz", path)
    print(f"bank2: {len(bank['episode'])} frames, {len(np.unique(bank['episode']))} episodes", flush=True)


def load_bank(i, masked=False):
    z = np.load(bank_file(i), mmap_mode="r")
    b = dict(rgb=z["rgb_masked" if masked else "rgb"], state=np.asarray(z["state"]), episode=np.asarray(z["episode"]),
             t=np.asarray(z["t"]), source=np.asarray(z["source"]))
    for k in ("u", "v", "gm_variant", "cube_red", "sphere_red"):
        b["cfg_" + k] = np.asarray(z["cfg_" + k])
    return b


def labels(a):
    """Oracle labels for banks 1-2, teacher-vs-oracle agreement, and teacher-label geometry by source / scene."""
    dev = torch.device("cuda")
    rows = []
    teach = sorted(set(R.pair_teacher("AT", "gm", k) for k in SEEDS["gm"]) | set(R.pair_teacher("AT", "control", k)
                                                                                  for k in SEEDS["control"]))
    grip_vals = []
    banks = [i for i in (1, 2) if os.path.exists(bank_file(i))]
    mus = {}
    for i in banks:
        b = load_bank(i)
        b["rgb"] = np.ascontiguousarray(b["rgb"])
        for tk in teach:
            ta = C.build_agent(C.ckpt_path("AT", "control", tk), dev=dev)
            mus[(i, tk)] = bank_features(ta, dict(rgb=b["rgb"], state=b["state"]), dev)["mu"].cpu().numpy()
            if i == 1:
                grip_vals.append(mus[(i, tk)][:, 7])
    grip = float(np.median(np.concatenate(grip_vals)))
    np.save(os.path.join(OUT, "oracle_grip.npy"), grip)
    for i in banks:
        b = load_bank(i)
        ep = b["episode"]
        cube_xy = b["cfg_u"][ep]
        lab = oracle_labels(b["state"], cube_xy, grip)
        lf = os.path.join(OUT, f"AT_bc_oracle_labels{i}.npz")
        if not os.path.exists(lf):
            np.savez(lf + ".tmp.npz", mu=lab, grip=grip)
            os.replace(lf + ".tmp.npz", lf)
        p, J = E.tcp_pos_and_jac(b["state"][:, :9])
        z = np.full((len(ep), 1), R.AT_HALF, np.float32)
        to_cube = np.concatenate([cube_xy, z], 1) - p
        lure_xy = b["cfg_v"][ep]
        to_lure = np.concatenate([lure_xy, z], 1) - p
        red_is_lure = b["cfg_sphere_red"][ep] & ~b["cfg_cube_red"][ep]
        src_gm = b["source"] < len(SEEDS["gm"])
        near_lure = np.linalg.norm(p[:, :2] - lure_xy, axis=1) < np.linalg.norm(p[:, :2] - cube_xy, axis=1)
        vel_o = np.einsum("nij,nj->ni", J, C.JOINT_DELTA_SCALE * lab[:, :7])
        cos = lambda x, y: np.sum(x * y, 1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-9)
        for tk in teach:
            vel_t = np.einsum("nij,nj->ni", J, C.JOINT_DELTA_SCALE * mus[(i, tk)][:, :7])
            ct, cl, co = cos(vel_t, to_cube), cos(vel_t, to_lure), cos(vel_t, vel_o)
            for src_name, sm in (("gm_source", src_gm), ("control_source", ~src_gm)):
                for scene, scm in (("red_lure", red_is_lure), ("other", ~red_is_lure)):
                    for region, rm in (("nearer_lure", near_lure), ("nearer_cube", ~near_lure), ("all", np.ones_like(near_lure))):
                        m = sm & scm & rm
                        if m.sum() == 0:
                            continue
                        rows.append(dict(bank=i, teacher=tk, source=src_name, scene=scene, region=region, n=int(m.sum()),
                                         frac_of_source=float(m.sum() / sm.sum()),
                                         cos_teacher_to_cube=float(ct[m].mean()), cos_teacher_to_lure=float(cl[m].mean()),
                                         cos_teacher_oracle=float(co[m].mean()),
                                         teacher_speed_cm=float(100 * np.linalg.norm(vel_t[m], axis=1).mean()),
                                         oracle_speed_cm=float(100 * np.linalg.norm(vel_o[m], axis=1).mean())))
        print(f"labels bank {i}: oracle grip {grip:.3f}, mean cos(teacher, oracle) "
              f"{np.mean([r['cos_teacher_oracle'] for r in rows if r['bank'] == i and r['region'] == 'all']):.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "AT_teacher_label_geometry.csv"), index=False)


# ---------------------------------------------------------------- training variants

def subset_mask(b, subset):
    ep = b["episode"]
    if subset is None:
        return np.ones(len(ep), bool)
    if subset == "ctrl_variant":
        return ~b["cfg_gm_variant"][ep]
    if subset == "gm_variant":
        return b["cfg_gm_variant"][ep]
    if subset == "half":
        ue = np.unique(ep)
        keep = np.random.default_rng(0).choice(ue, size=len(ue) // 2, replace=False)
        return np.isin(ep, keep)
    if subset == "ctrl_source":
        return b["source"] >= len(SEEDS["gm"])
    if subset == "gm_source":
        return b["source"] < len(SEEDS["gm"])
    raise KeyError(subset)


def variant_data(spec, dev, masked):
    """Concatenated banks (frame mask applied), labels per teacher (teacher) or oracle labels, episode ids."""
    parts = []
    for i in spec["banks"]:
        b = load_bank(i, masked)
        m = subset_mask(b, spec.get("subset"))
        idx = np.nonzero(m)[0]
        rgb = np.ascontiguousarray(b["rgb"][idx]) if len(idx) < len(m) else np.ascontiguousarray(b["rgb"])
        ol = np.load(os.path.join(OUT, f"AT_bc_oracle_labels{i}.npz"))["mu"][idx] if spec["label"] == "oracle" else None
        parts.append(dict(rgb=rgb, state=b["state"][idx], episode=b["episode"][idx] + 100_000 * (i - 1), oracle=ol, bank=i))
    return parts


def train(a):
    dev = torch.device("cuda")
    os.makedirs(HEADS, exist_ok=True)
    os.makedirs(os.path.join(OUT, "trainlogs"), exist_ok=True)
    for variant in a.variants:
        spec = VARIANTS[variant]
        arms_all = a.arms or list(spec.get("arms", ())) or (["pre", "pre_masked", "fc_head", "remix"] if spec["label"] == "oracle" else
                              ["pre", "pre_masked"] if spec.get("subset") is None else ["pre"])
        data = {}
        teachers = {}
        for cohort in a.cohorts:
            for k in SEEDS[cohort]:
                arms = [x for x in arms_all if not (cohort == "init" and x == "pre_masked")
                        and not os.path.exists(head_path(variant, cohort, k, x))]
                if not arms:
                    continue
                tk = R.pair_teacher("AT", cohort, k)
                base = R.base_agent("AT", cohort, k, dev)
                for arm in arms:
                    msk = arm == "pre_masked"
                    if msk not in data:
                        data[msk] = variant_data(spec, dev, msk)
                    parts = data[msk]
                    if (tk, spec["label"]) not in teachers:
                        ta = C.build_agent(C.ckpt_path("AT", "control", tk), dev=dev)
                        normal = data.get(False) or variant_data(spec, dev, False)
                        data[False] = normal
                        if spec["label"] == "teacher":
                            tgt = torch.cat([bank_features(ta, p, dev)["mu"] for p in normal])
                        else:
                            tgt = torch.as_tensor(np.concatenate([p["oracle"] for p in normal]), device=dev)
                        teachers[(tk, spec["label"])] = (tgt, C.sigma(ta))
                        del ta
                    target, sig_t = teachers[(tk, spec["label"])]
                    fs = [bank_features(base, p, dev) for p in parts]
                    feats = {key: torch.cat([f[key] for f in fs]) for key in fs[0]}
                    episodes = np.concatenate([p["episode"] for p in parts])
                    model = Rewired(base, "pre" if msk else arm, init_seed=100 + k).to(dev)
                    log, n_par = train_bc(model, feats, target, sig_t, episodes, spec["epochs"], a.lr, a.batch, a.clip, seed=k)
                    sd = {n: p.detach().cpu() for n, p in model.state_dict().items() if not n.startswith("base.")}
                    torch.save(dict(state_dict=sd, arm=arm, cohort=cohort, seed=k, teacher=tk, n_trainable=n_par,
                                    variant=variant), head_path(variant, cohort, k, arm))
                    pd.DataFrame(log).assign(n_trainable=n_par, teacher=tk, n_frames=len(episodes)).to_csv(
                        os.path.join(OUT, "trainlogs", f"{variant}_{cohort}_s{k}_{arm}.csv"), index=False)
                    print(f"trained {variant} {cohort} s{k} {arm} ({len(episodes)} frames): val {log[-1]['val_loss']:.3f} "
                          f"train {log[-1]['train_loss']:.3f} min val {min(r['val_loss'] for r in log):.3f}", flush=True)
                    del feats, fs, model
                    torch.cuda.empty_cache()
        del data, teachers
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- DAgger

@torch.no_grad()
def collect_dagger(env, base, model, teacher, sig_t, cfg, pad, n_real, masked, dev):
    """Rollouts of the current rewired head (mean + teacher-sigma noise) in two synchronised copies (normal, cue-masked);
    the head acts on the copy it was trained on, the teacher labels the normal copy. Kept up to the first touch."""
    B = len(pad)
    c = R.take(cfg, pad)
    pn, pm = R.bc_place("AT", c, False), R.bc_place("AT", c, True)
    R.force("AT", env, np.concatenate([c["qpos"], c["qpos"]]), R.cat_place([pn, pm]))
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    alive = np.arange(B) < n_real
    rec = {k: [] for k in ("h_pre", "h_fc", "h_state", "mu", "ep")}
    for t in range(R.T_MAX["AT"]):
        a_idx = np.nonzero(alive)[0]
        ai = torch.as_tensor(a_idx, device=dev)
        st = obs["state"][:B]
        tb = C.taps(base, obs["rgb"][B:] if masked else obs["rgb"][:B], st)
        mu_s = model(tb["h_pre"], tb["h_fc"], tb["h_state"])
        mu_t = C.taps(teacher, obs["rgb"][:B], st)["mu"]
        rec["h_pre"].append(tb["h_pre"][ai].half().cpu()), rec["h_fc"].append(tb["h_fc"][ai].cpu())
        rec["h_state"].append(tb["h_state"][ai].cpu()), rec["mu"].append(mu_t[ai].cpu()), rec["ep"].append(pad[a_idx])
        act = mu_s + sig_t * torch.randn_like(mu_s)
        obs, _, _, _, info = env.step(act.repeat(2, 1))
        obs, _ = sync_copies(env, 2, info)
        du, dv = R.at_dists(u.agent.tcp.pose.p[:B].cpu().numpy(), pn)
        alive &= ~((du < R.AT_EPS) | (dv < R.AT_EPS))
        if not alive.any():
            break
    return {k: (np.concatenate(v) if k == "ep" else torch.cat(v)) for k, v in rec.items()}


def dagger(a):
    dev = torch.device("cuda")
    os.makedirs(HEADS, exist_ok=True)
    env = R.make_env("AT", 2 * a.env_batch)
    for cohort in a.cohorts:
        for k in SEEDS[cohort]:
            for arm in a.arms or ["pre", "pre_masked"]:
                if cohort == "init" and arm == "pre_masked":
                    continue
                if os.path.exists(head_path(f"dagger{a.rounds}", cohort, k, arm)):
                    continue
                msk = arm == "pre_masked"
                tk = R.pair_teacher("AT", cohort, k)
                teacher = C.build_agent(C.ckpt_path("AT", "control", tk), dev=dev)
                sig_t = C.sigma(teacher)
                base = R.base_agent("AT", cohort, k, dev)
                b1 = load_bank(1, msk)
                b1["rgb"] = np.ascontiguousarray(b1["rgb"])
                feats = bank_features(base, b1, dev)
                target = bank_features(teacher, load_bank(1, False) if msk else b1, dev)["mu"]
                feats.pop("mu")
                episodes = b1["episode"].copy()
                del b1
                model = Rewired(base, "pre", init_seed=100 + k).to(dev)
                ck = torch.load(head_path("base", cohort, k, arm), map_location="cpu")
                model.load_state_dict({**{n: p for n, p in model.state_dict().items() if n.startswith("base.")},
                                       **ck["state_dict"]})
                for r in range(1, a.rounds + 1):
                    model.eval()
                    cfg = bc_cfg(DAGGER_CFG, DAGGER_SEED + 10 * r)
                    torch.manual_seed(DAGGER_SEED + 1000 * r + k)
                    new = []
                    for b0 in range(0, DAGGER_CFG, a.env_batch):
                        idx = np.arange(b0, min(DAGGER_CFG, b0 + a.env_batch))
                        pad = np.concatenate([idx, np.full(a.env_batch - len(idx), idx[-1])])
                        new.append(collect_dagger(env, base, model, teacher, sig_t, cfg, pad, len(idx), msk, dev))
                    nd = {kk: (np.concatenate([x[kk] for x in new]) if kk == "ep" else torch.cat([x[kk] for x in new]))
                          for kk in new[0]}
                    for kk in ("h_pre", "h_fc", "h_state"):
                        feats[kk] = torch.cat([feats[kk], nd[kk].to(dev)])
                    target = torch.cat([target, nd["mu"].to(dev)])
                    episodes = np.concatenate([episodes, 1_000_000 * r + nd["ep"]])
                    model = Rewired(base, "pre", init_seed=100 + k).to(dev)
                    log, n_par = train_bc(model, feats, target, sig_t, episodes, a.epochs, a.lr, a.batch, a.clip, seed=k)
                    sd = {n: p.detach().cpu() for n, p in model.state_dict().items() if not n.startswith("base.")}
                    torch.save(dict(state_dict=sd, arm=arm, cohort=cohort, seed=k, teacher=tk, n_trainable=n_par,
                                    variant=f"dagger{r}"), head_path(f"dagger{r}", cohort, k, arm))
                    os.makedirs(os.path.join(OUT, "trainlogs"), exist_ok=True)
                    pd.DataFrame(log).assign(n_trainable=n_par, teacher=tk, n_frames=len(episodes),
                                             n_new_frames=len(nd["ep"])).to_csv(
                        os.path.join(OUT, "trainlogs", f"dagger{r}_{cohort}_s{k}_{arm}.csv"), index=False)
                    print(f"dagger round {r} {cohort} s{k} {arm}: +{len(nd['ep'])} frames (total {len(episodes)}), "
                          f"val {log[-1]['val_loss']:.3f} train {log[-1]['train_loss']:.3f}", flush=True)
                del feats, target, model
                torch.cuda.empty_cache()
    env.close()


# ---------------------------------------------------------------- BC fit on a fully held-out bank (bank 2)

def fit(a):
    """Per-frame teacher-label loss of bank-1 heads on bank 2 (never trained on), split by frame category."""
    dev = torch.device("cuda")
    b = load_bank(2)
    b["rgb"] = np.ascontiguousarray(b["rgb"])
    bm = None
    ep = b["episode"]
    p, _ = E.tcp_pos_and_jac(b["state"][:, :9])
    cube_xy, lure_xy = b["cfg_u"][ep], b["cfg_v"][ep]
    near_lure = np.linalg.norm(p[:, :2] - lure_xy, axis=1) < np.linalg.norm(p[:, :2] - cube_xy, axis=1)
    red_lure = b["cfg_sphere_red"][ep] & ~b["cfg_cube_red"][ep]
    cats = {"all": np.ones(len(ep), bool), "gm_source": b["source"] < 3, "control_source": b["source"] >= 3,
            "gm_variant": b["cfg_gm_variant"][ep], "ctrl_variant": ~b["cfg_gm_variant"][ep],
            "red_lure": red_lure, "red_lure_nearer_lure": red_lure & near_lure, "nearer_lure": near_lure,
            "t0": b["t"] == 0}
    rows = []
    teachers = {}
    for cohort in ("gm", "control", "init"):
        for k in SEEDS[cohort]:
            tk = R.pair_teacher("AT", cohort, k)
            if tk not in teachers:
                ta = C.build_agent(C.ckpt_path("AT", "control", tk), dev=dev)
                teachers[tk] = (bank_features(ta, b, dev)["mu"], C.sigma(ta))
            target, sig_t = teachers[tk]
            base = R.base_agent("AT", cohort, k, dev)
            feats = None
            for variant in a.variants:
                for arm in ("pre", "fc_head", "remix", "pre_masked"):
                    if cohort == "init" and arm == "pre_masked":
                        continue
                    if not os.path.exists(head_path(variant, cohort, k, arm)) or len(VARIANTS.get(variant, {}).get("banks", (1,))) > 1:
                        continue
                    if arm == "pre_masked":
                        bm = bm or dict(b, rgb=np.ascontiguousarray(load_bank(2, True)["rgb"]))
                        fts = bank_features(base, bm, dev)
                    else:
                        feats = feats or bank_features(base, b, dev)
                        fts = feats
                    model = Rewired(base, "pre" if arm == "pre_masked" else arm, init_seed=100 + k).to(dev)
                    ck = torch.load(head_path(variant, cohort, k, arm), map_location="cpu")
                    model.load_state_dict({**{n: q for n, q in model.state_dict().items() if n.startswith("base.")},
                                           **ck["state_dict"]})
                    model.eval()
                    with torch.no_grad():
                        loss = torch.cat([(((model(fts["h_pre"][sl].float(), fts["h_fc"][sl], fts["h_state"][sl]) -
                                             target[sl]) / sig_t) ** 2).mean(1) for sl in C.batched(len(ep), 4096)])
                    loss = loss.cpu().numpy()
                    for cname, m in cats.items():
                        rows.append(dict(variant=variant, cohort=cohort, seed=k, arm=arm, category=cname, n=int(m.sum()),
                                         heldout_loss=float(loss[m].mean())))
                    del model
            del feats
            feats = None
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "AT_heldout_bank2_loss.csv"), index=False)
    print("fit done", flush=True)


# ---------------------------------------------------------------- probes

def ridge_fit(Xtr, Ytr, Xva, Yva, Xte, alphas=(1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5)):
    """Ridge on standardised features, alpha chosen on the validation split; returns test predictions and alpha."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, Bv, Ct = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    # eigendecomposition of the Gram (n x n), cheaper than d x d for d = 9216 > n
    G = A @ A.T
    s, V = torch.linalg.eigh(G.double())
    s, V = s.float().clamp_min(0), V.float()
    VtY = V.T @ (Ytr - ym)
    best = None
    for al in alphas:
        coef = A.T @ (V @ (VtY / (s + al)[:, None]))
        err = ((Bv @ coef + ym - Yva) ** 2).mean().item()
        if best is None or err < best[0]:
            best = (err, al, coef)
    return Ct @ best[2] + ym, best[1]


def mlp_fit(Xtr, Ytr, Xva, Yva, Xte, kind, seed=0, epochs=300, patience=25):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, Bv, Ct = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    ym, ys = (Ytr.mean(0), Ytr.std(0) + 1e-6) if kind == "r2" else (0, 1)
    net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, Ytr.shape[1])).to(A.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = (lambda p, y: ((p - (y - ym) / ys) ** 2).mean()) if kind == "r2" else \
        (lambda p, y: torch.nn.functional.binary_cross_entropy_with_logits(p, y))
    best, bad, state = np.inf, 0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(A), device=A.device)
        for b0 in range(0, len(A), 512):
            i = perm[b0:b0 + 512]
            loss = lf(net(A[i]), Ytr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = lf(net(Bv), Yva).item()
        if v < best - 1e-5:
            best, bad, state = v, 0, {n: p.clone() for n, p in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        out = net(Ct)
    return out * ys + ym if kind == "r2" else out


def r2(y, p):
    y, p = y.cpu().numpy(), p.cpu().numpy()
    return float(np.mean([1 - ((y[:, j] - p[:, j]) ** 2).sum() / ((y[:, j] - y[:, j].mean()) ** 2).sum() for j in range(y.shape[1])]))


def auc(y, p):
    """Mann-Whitney ROC-AUC with average ranks for ties."""
    import pandas as _pd
    y, p = y.cpu().numpy().ravel() > 0.5, p.cpu().numpy().ravel()
    rk = _pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe_targets(env, z, ep):
    """Probe targets per frame; only control-distribution scenes (cue roles independent)."""
    u, v = z["cfg_u"][ep], z["cfg_v"][ep]
    if env == "AT":
        cr, sr = z["cfg_cube_red"][ep], z["cfg_sphere_red"][ep]
        one = cr ^ sr
        red = np.where(cr[:, None], u, v)
        return {"goal_xy": (u, "r2", np.ones(len(ep), bool)), "goal_left": ((u[:, 1] > v[:, 1])[:, None].astype(np.float32), "auc", np.ones(len(ep), bool)),
                "proxy_xy": (red, "r2", one), "proxy_left": ((red[:, 1] > np.where(cr[:, None], v, u)[:, 1])[:, None].astype(np.float32), "auc", one)}
    cg = z["cfg_cross_green"][ep]
    green = np.where(cg[:, None], u, v)
    allm = np.ones(len(ep), bool)
    return {"goal_xy": (u, "r2", allm), "goal_left": ((u[:, 1] > v[:, 1])[:, None].astype(np.float32), "auc", allm),
            "proxy_xy": (green, "r2", allm), "proxy_left": ((green[:, 1] > np.where(cg[:, None], v, u)[:, 1])[:, None].astype(np.float32), "auc", allm)}


def probe(a):
    dev = torch.device("cuda")
    rows = []
    for env in a.envs:
        z = np.load(os.path.join(C.BANKS, f"{env}_bc_attr.npz"), mmap_mode="r")
        cfgk = {k: np.asarray(z[k]) for k in z.files if k.startswith("cfg_")}
        ep_all = np.asarray(z["episode"])
        ctrl = ~cfgk["cfg_gm_variant"][ep_all]
        rng = np.random.default_rng(0)
        idx = np.nonzero(ctrl)[0]
        idx = np.sort(rng.choice(idx, size=min(a.n_probe, len(idx)), replace=False))
        ep = ep_all[idx]
        ue = np.unique(ep)
        rng.shuffle(ue)
        n1, n2 = int(0.6 * len(ue)), int(0.8 * len(ue))
        split = np.where(np.isin(ep, ue[:n1]), 0, np.where(np.isin(ep, ue[n1:n2]), 1, 2))
        bank = dict(rgb=np.ascontiguousarray(z["rgb"][idx]), state=np.asarray(z["state"][idx]))
        tg = probe_targets(env, cfgk, ep)
        seeds = R.SEEDS[env]
        for cohort in ("gm", "control", "init"):
            for k in seeds[cohort]:
                agent = R.base_agent(env, cohort, k, dev)
                f = bank_features(agent, bank, dev)
                for site in ("h_pre", "h_fc"):
                    X = f[site].float()
                    for tname, (y, kind, m) in tg.items():
                        Y = torch.as_tensor(np.asarray(y, np.float32), device=dev)
                        sel = [torch.as_tensor(np.nonzero(m & (split == s))[0], device=dev) for s in range(3)]
                        Xs, Ys = [X[i] for i in sel], [Y[i] for i in sel]
                        pl, al = ridge_fit(Xs[0], Ys[0] if kind == "r2" else 2 * Ys[0] - 1, Xs[1],
                                           Ys[1] if kind == "r2" else 2 * Ys[1] - 1, Xs[2])
                        pm = mlp_fit(Xs[0], Ys[0], Xs[1], Ys[1], Xs[2], kind, seed=k)
                        sc = r2 if kind == "r2" else auc
                        rows.append(dict(env=env, cohort=cohort, seed=k, site=site, target=tname, metric=kind,
                                         linear=sc(Ys[2], pl), mlp=sc(Ys[2], pm), alpha=al, n_train=len(sel[0]),
                                         n_test=len(sel[2])))
                        print(f"probe {env} {cohort} s{k} {site} {tname}: linear {rows[-1]['linear']:.3f} "
                              f"mlp {rows[-1]['mlp']:.3f}", flush=True)
                del f
                torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "probes_goal_proxy.csv"), index=False)


# ---------------------------------------------------------------- edit sensitivity of the rewired heads

def sens(a):
    """Per-state KL (teacher sigma) and first-order TCP shift toward the edited cue for AT world-edit swaps."""
    dev = torch.device("cuda")
    w = C.load_bank("AT_worldedit")
    edits = ("shape_swap", "colour_swap", "shape_ablate", "colour_ablate")
    J = w["J_tcp"]
    u, v = w["u"], w["v"]
    dirs = {"u": (u - v) / np.linalg.norm(u - v, axis=1, keepdims=True)}
    dirs["v"] = -dirs["u"]
    rows = []
    for cohort in ("gm", "control", "init"):
        for k in SEEDS[cohort]:
            tk = R.pair_teacher("AT", cohort, k)
            sig = C.sigma(C.build_agent(C.ckpt_path("AT", "control", tk), dev=dev))
            base = R.base_agent("AT", cohort, k, dev)
            specs = [("native", "native")] if cohort != "init" else []
            specs += [(vv, arm) for vv in a.variants for arm in ("pre", "fc_head", "remix")
                      if os.path.exists(head_path(vv, cohort, k, arm))]
            for variant, arm in specs:
                if arm == "native":
                    fn = lambda rgb, st: C.taps(base, rgb, st)["mu"]
                else:
                    fn = policy_factory(variant, cohort, k, arm, dev)(None, None)
                with torch.no_grad():
                    mu = {e: torch.cat([fn(C.to_t(w[f"rgb_{e}"][sl], dev), C.to_t(w[f"state_{e}"][sl], dev))
                                        for sl in C.batched(512, 128)]) for e in ("base",) + edits}
                for e in edits:
                    d = (mu[e] - mu["base"])
                    kl = (0.5 * ((d / sig) ** 2).sum(1)).cpu().numpy()
                    dt = np.einsum("nij,nj->ni", J, C.JOINT_DELTA_SCALE * d[:, :7].cpu().numpy())[:, :2]
                    follow = np.sum(dt * dirs[E.AT_FAVOURED[e]], 1) * 1000
                    rows.append(dict(cohort=cohort, seed=k, variant=variant if arm != "native" else "native", arm=arm, edit=e,
                                     kl_median=float(np.median(kl)), kl_mean=float(kl.mean()),
                                     follow_mm_mean=float(follow.mean()), follow_pos_frac=float(np.mean(follow > 0))))
            print(f"sens {cohort} s{k} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "AT_edit_sensitivity.csv"), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["chance", "rollout", "bank2", "labels", "train", "dagger", "fit", "probe", "sens"])
    p.add_argument("--variants", nargs="+", default=["base"])
    p.add_argument("--cohorts", nargs="+", default=["gm", "control", "init"])
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--envs", nargs="+", default=["AT", "SZ"])
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n_workers", type=int, default=1)
    p.add_argument("--tag", default=None, help="per-episode CSV suffix (default w<worker>); distinct per concurrent process")
    p.add_argument("--n_ep", type=int, default=192)
    p.add_argument("--n_probe", type=int, default=12000)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20, help="DAgger retrain epochs (variants carry their own)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--env_batch", type=int, default=128)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    C.ensure_dirs()
    dict(chance=chance, rollout=rollout, bank2=build_bank2, labels=labels, train=train, dagger=dagger, fit=fit,
         probe=probe, sens=sens)[a.step](a)


if __name__ == "__main__":
    main()
