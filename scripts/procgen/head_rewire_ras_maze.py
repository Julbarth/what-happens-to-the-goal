"""Attribute Maze (Ras) and Maze adapters for frozen-trunk head rewiring, imported by the training script."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing path (tried: {[str(p) for p in paths]})")


TRAIN_ROOT = _existing(
    ROOT / "external" / "train-procgen-pytorch",
    ROOT / "vendored" / "train-procgen-pytorch",
)
CS_ROOT = _existing(
    ROOT / "external" / "colour-shape-goal-misgeneralization",
    ROOT / "vendored",
)
PROCGEN_AISC = _existing(
    ROOT / "external" / "procgenAISC",
    ROOT / "vendored" / "procgenAISC",
)

# Helpers from train_frozen_trunk_head_rewire are imported at call time to avoid import cycles.


def _ensure_coin_paths():
    for sub in ("src", "pkg"):
        subroot = ROOT / sub
        if subroot.exists():
            sys.path.insert(0, str(subroot))
    sys.path.insert(0, str(PROCGEN_AISC))
    sys.path.insert(0, str(TRAIN_ROOT))


def _ensure_cs_paths():
    # Drop coin procgen if already imported — world_dim / duel live only in cs fork.
    for mod in list(sys.modules):
        if mod == "procgen" or mod.startswith("procgen."):
            del sys.modules[mod]
    for sub in ("src", "pkg"):
        subroot = ROOT / sub
        if subroot.exists():
            sys.path.insert(0, str(subroot))
    for rel in ("cs-procgenAISC", "cs-train-procgen"):
        p = CS_ROOT / rel
        if p.exists():
            sys.path.insert(0, str(p))


def load_maze_policy(checkpoint: Path | None, device: str, *, random_init: bool = False):
    _ensure_coin_paths()
    from common.model import ImpalaModel  # noqa: E402
    from common.policy import CategoricalPolicy  # noqa: E402
    from procgen import ProcgenEnv  # noqa: E402
    from common.env.procgen_wrappers import (  # noqa: E402
        ScaledFloatFrame,
        TransposeFrame,
        VecExtractDictObs,
        VecNormalize,
    )

    with open(TRAIN_ROOT / "hyperparams/procgen/config.yml") as f:
        recurrent = yaml.safe_load(f)["hard-500"]["recurrent"]
    venv = ProcgenEnv(
        num_envs=1, env_name="maze", num_levels=1, start_level=0, distribution_mode="hard"
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False, ret=False)
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    model = ImpalaModel(in_channels=venv.observation_space.shape[0])
    policy = CategoricalPolicy(model, recurrent, venv.action_space.n)
    if not random_init:
        ckpt = torch.load(checkpoint, map_location="cpu")
        policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device).eval()
    if hasattr(venv, "close"):
        venv.close()
    return policy


def load_ras_policy(checkpoint: Path | None, device: str, *, random_init: bool = False):
    _ensure_cs_paths()
    from gm_geometry.integrations.colour_shape import make_duel_env  # noqa: E402
    from gm_geometry.integrations.colour_shape.policy import (  # noqa: E402
        load_impala_policy,
        load_recurrent_flag,
    )

    env = make_duel_env(1, 1, 1, "red_line_diag", "yellow_gem", use_backgrounds=False)
    in_ch = env.observation_space.shape[0]
    n_act = env.action_space.n
    if hasattr(env, "close"):
        env.close()
    if random_init:
        return _ras_random(in_ch, n_act, device)
    return load_impala_policy(
        checkpoint, in_ch, n_act, load_recurrent_flag()
    ).to(device)


def _ras_random(in_ch, n_act, device):
    _ensure_cs_paths()
    from common.model import ImpalaModel  # noqa: E402
    from common.policy import CategoricalPolicy  # noqa: E402
    from gm_geometry.integrations.colour_shape.policy import load_recurrent_flag  # noqa: E402

    model = ImpalaModel(in_channels=in_ch)
    policy = CategoricalPolicy(model, load_recurrent_flag(), n_act)
    policy.to(device).eval()
    return policy


def attach_heads(trunk, head_ckpt: Path, device: str, *, loader, init_trunk: bool = False):
    from train_frozen_trunk_head_rewire import freeze_trunk  # noqa: E402

    if init_trunk:
        trunk = loader(None, device, random_init=True)
    head = loader(head_ckpt, device)
    trunk.fc_policy.load_state_dict(head.fc_policy.state_dict())
    trunk.fc_value.load_state_dict(head.fc_value.state_dict())
    del head
    freeze_trunk(trunk)
    trunk.to(device)
    return trunk


def compose_maze_policy(*, arm, trunk_ckpt, teacher_ckpt, device):
    from train_frozen_trunk_head_rewire import freeze_trunk, reinit_heads  # noqa: E402

    if arm == 0:
        p = load_maze_policy(trunk_ckpt, device)
        freeze_trunk(p)
        return p
    if arm == 3:
        p = load_maze_policy(teacher_ckpt, device)
        freeze_trunk(p)
        return p
    if arm == 5:
        return attach_heads(
            load_maze_policy(trunk_ckpt, device),
            teacher_ckpt,
            device,
            loader=load_maze_policy,
        )
    if arm == 6:
        return attach_heads(
            None,
            teacher_ckpt,
            device,
            loader=load_maze_policy,
            init_trunk=True,
        )
    p = load_maze_policy(trunk_ckpt, device)
    freeze_trunk(p)
    reinit_heads(p)
    p.to(device)
    return p


def compose_ras_policy(*, arm, trunk_ckpt, teacher_ckpt, device):
    from train_frozen_trunk_head_rewire import freeze_trunk, reinit_heads  # noqa: E402

    if arm == 0:
        p = load_ras_policy(trunk_ckpt, device)
        freeze_trunk(p)
        return p
    if arm == 3:
        p = load_ras_policy(teacher_ckpt, device)
        freeze_trunk(p)
        return p
    if arm == 5:
        return attach_heads(
            load_ras_policy(trunk_ckpt, device),
            teacher_ckpt,
            device,
            loader=load_ras_policy,
        )
    if arm == 6:
        return attach_heads(
            None,
            teacher_ckpt,
            device,
            loader=load_ras_policy,
            init_trunk=True,
        )
    p = load_ras_policy(trunk_ckpt, device)
    freeze_trunk(p)
    reinit_heads(p)
    p.to(device)
    return p


def make_ras_env(*, num_envs: int, num_levels: int, start_level: int = 0, use_backgrounds: bool = False):
    """use_backgrounds=False (black) matches the published Zenodo checkpoints; textured-background trunks need True."""
    _ensure_cs_paths()
    from gm_geometry.integrations.colour_shape import make_duel_env  # noqa: E402

    return make_duel_env(
        num_envs,
        num_levels,
        max(2, num_envs // 2),
        "red_line_diag",
        "yellow_gem",
        start_level=start_level,
        use_backgrounds=use_backgrounds,
    )


def make_maze_train_env(*, num_envs: int, num_levels: int, start_level: int = 0):
    _ensure_coin_paths()
    from procgen import ProcgenEnv  # noqa: E402
    from common.env.procgen_wrappers import (  # noqa: E402
        ScaledFloatFrame,
        TransposeFrame,
        VecExtractDictObs,
        VecNormalize,
    )

    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name="maze_aisc",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="hard",
        rand_region=25,
        num_threads=max(2, num_envs // 2),
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False, ret=False)
    venv = TransposeFrame(venv)
    return ScaledFloatFrame(venv)


def ras_frame_labels(infos: list[dict], actions: np.ndarray) -> dict[str, np.ndarray]:
    from gm_geometry.probes.choice_point import action_toward  # noqa: E402

    n = len(infos)
    toward_line = np.zeros(n, dtype=np.float64)
    toward_yellow = np.zeros(n, dtype=np.float64)
    for i, info in enumerate(infos):
        agent = info.get("agent_xy")
        line = info.get("line_xy")
        gem = info.get("gem_xy")
        if agent and line and gem:
            a_line = action_toward(agent, line)
            a_gem = action_toward(agent, gem)
            act = int(actions[i])
            if a_line != a_gem:
                toward_line[i] = 1.0 if act == a_line else 0.0
                toward_yellow[i] = 1.0 if act == a_gem else 0.0
    return {"toward_line": toward_line, "toward_yellow": toward_yellow}


def eval_ras_pref_goal(policy, device: str, *, n_frames: int = 150, use_backgrounds: bool = False) -> dict:
    """Bearing-based line preference at choice frames; a secondary check only.

    The bearing heuristic agrees poorly with the greedy action in walled mazes (below chance), so the main
    recoverability measure is the episode-outcome trap preference from score_ras_trap_preference.py.
    """
    _ensure_cs_paths()
    from gm_geometry.probes.choice_point import (  # noqa: E402
        line_prefer_at_choice,
        localize_frame,
        logit_contrast_line_minus_gem,
        matched_geometry,
        prob_contrast_line_minus_gem,
    )
    from run_ras_maze_goal_proxy_ladder import Ladder  # noqa: E402

    lad = Ladder(policy, device)
    logprob_contrasts: list[float] = []
    prob_contrasts: list[float] = []
    line_pref_frames: list[float] = []
    level = 10_000
    max_level = 10_000 + max(500, n_frames * 4)
    while len(line_pref_frames) < n_frames and level < max_level:
        env = make_ras_env(num_envs=1, num_levels=1, start_level=level, use_backgrounds=use_backgrounds)
        obs = env.reset().astype(np.float32)
        prev = None
        for _ in range(400):
            out = lad.run(obs[0], capture=["fc"])
            loc = localize_frame(obs[0], prev_hwc=prev)
            if matched_geometry(loc):
                lp = logit_contrast_line_minus_gem(out["logits"][0], loc)
                if np.isfinite(lp):
                    logprob_contrasts.append(lp)
                pc = prob_contrast_line_minus_gem(out["probs"][0], loc)
                if np.isfinite(pc):
                    prob_contrasts.append(pc)
                lf = line_prefer_at_choice(out["probs"][0], loc)
                if np.isfinite(lf):
                    line_pref_frames.append(lf)
            act = int(out["probs"][0].argmax())
            obs, _r, done, _ = env.step(np.array([act]))
            obs = obs.astype(np.float32)
            prev = loc.get("img_hwc")
            if done[0] or len(line_pref_frames) >= n_frames:
                break
        if hasattr(env, "close"):
            env.close()
        level += 1
    lad.close()
    lp_arr = np.asarray(logprob_contrasts, dtype=np.float64)
    pc_arr = np.asarray(prob_contrasts, dtype=np.float64)
    lf_arr = np.asarray(line_pref_frames, dtype=np.float64)
    return {
        # line_prefer is a bounded fraction comparable to Maze cheese_prefer; pref_goal is the log-prob contrast.
        "line_prefer": float(lf_arr.mean()) if len(lf_arr) else float("nan"),
        "pref_goal_prob_contrast": float(pc_arr.mean()) if len(pc_arr) else float("nan"),
        "pref_goal": float(lp_arr.mean()) if len(lp_arr) else float("nan"),
        "n_choice_frames": int(len(lf_arr)),
    }


def eval_maze_cheese_prefer(
    policy, device: str, *, n_frames: int = 120, graph_exact: bool = True
) -> dict:
    """Cheese vs reachable-TR preference at decision frames (ladder-aligned).

    Uses graph-exact first-step actions (reachable extreme TR via BFS) when the
    maze grid API is available; falls back to image bearing otherwise. Comparable
    to ``run_maze_i_goal_proxy_ladder.collect`` graph mode, not naive pixel TR.
    """
    _ensure_coin_paths()
    from run_maze_i_goal_proxy_ladder import (  # noqa: E402
        Ladder,
        action_toward,
        cheese_xy_bright,
        first_step_action,
        get_agent_cell,
        get_info_maze,
        make_env,
        motion_agent_xy,
        _chw_to_hwc01,
        _iscalar,
        _read_grid,
        _reachable,
        _reachable_extreme,
    )

    lad = Ladder(policy, device)
    cheese_hits = n = 0
    n_graph = n_bearing = 0
    n_graph_exceptions = 0
    first_graph_exception: str | None = None
    level = 1_000_000

    def _greedy_act(out: dict) -> int:
        if "logits" in out:
            return int(out["logits"][0].argmax())
        return int(out["probs"][0].argmax())

    while n < n_frames and level < 1_000_000 + 400:
        env = make_env("maze", start_level=level)
        obs = env.reset()
        prev = _chw_to_hwc01(obs)
        for _ in range(500):
            out = lad.run(np.asarray(obs[0], dtype=np.float32), capture=["fc"])
            cur = _chw_to_hwc01(obs)
            cheese = cheese_xy_bright(cur)
            agent = motion_agent_xy(prev, cur)
            a_cheese = a_corner = None
            mode = "bearing"
            if graph_exact:
                try:
                    info = get_info_maze(env)
                    wd = _iscalar(info, "maze_world_dim")
                    cx, cy = _iscalar(info, "cheese_x"), _iscalar(info, "cheese_y")
                    ax_g, ay_g = get_agent_cell(env)
                    if ax_g >= 0 and cx >= 0:
                        grid = _read_grid(env, wd)
                        tr = _reachable_extreme(grid, (ax_g, ay_g), "tr")
                        if tr is not None and (cx, cy) in _reachable(grid, (ax_g, ay_g)):
                            a_cheese, _plc = first_step_action(
                                grid, (ax_g, ay_g), (cx, cy)
                            )
                            a_corner, _plt = first_step_action(grid, (ax_g, ay_g), tr)
                            if a_cheese is not None and a_corner is not None:
                                mode = "graph"
                except Exception as exc:
                    mode = "bearing"
                    n_graph_exceptions += 1
                    if first_graph_exception is None:
                        first_graph_exception = repr(exc)
            if mode == "bearing" and agent and cheese:
                tr_corner = (float(cur.shape[1] - 1), 0.0)
                a_cheese = action_toward(agent, cheese)
                a_corner = action_toward(agent, tr_corner)
            # Graph mode has grid actions, so it does not require image agent/cheese centroids,
            # which are often missed.
            decision_ok = (
                a_cheese is not None
                and a_corner is not None
                and a_cheese != a_corner
            )
            if mode == "graph":
                usable = decision_ok
            else:
                usable = bool(agent and cheese and decision_ok)
            if usable:
                act = _greedy_act(out)
                n += 1
                if mode == "graph":
                    n_graph += 1
                else:
                    n_bearing += 1
                if act == a_cheese:
                    cheese_hits += 1
            act = _greedy_act(out)
            obs, _r, done, _ = env.step(np.array([act]))
            prev = _chw_to_hwc01(obs)
            if done[0] or n >= n_frames:
                break
        if hasattr(env, "close"):
            env.close()
        level += 1
    lad.close()
    if n_graph_exceptions:
        print(
            f"[warn] eval_maze_cheese_prefer: {n_graph_exceptions} graph-exact "
            f"lookups fell back to bearing mode; first error: {first_graph_exception}"
        )
    return {
        "cheese_prefer": float(cheese_hits / n) if n else float("nan"),
        "n_decision_frames": int(n),
        "n_graph_frames": int(n_graph),
        "n_bearing_frames": int(n_bearing),
        "n_graph_exceptions": int(n_graph_exceptions),
        "metric": "graph_exact_cheese_prefer",
        "graph_exact": bool(graph_exact),
    }


def eval_maze_val_success(policy, device: str, *, n_levels: int) -> dict:
    _ensure_coin_paths()
    from validate_langosco_maze_i_200m import eval_split  # noqa: E402

    val = eval_split(
        policy,
        env_name="maze",
        extra={},
        n_levels=n_levels,
        start_level=1_000_000,
        max_steps=1000,
        device=device,
        track_corner=True,
        action_mode="greedy",
    )
    return {"success_rate": val["success_rate"], "goal_misgen_rate": val["gm_rate"]}


def collect_ras_frames(trunk_policy, label_policy, env, *, device, target_steps, label_mode, rng):
    from gm_geometry.probes.choice_point import localize_frame  # noqa: E402
    from train_frozen_trunk_head_rewire import FrameBatch, sample_actions, trunk_features  # noqa: E402

    batch = FrameBatch()
    obs = env.reset().astype(np.float32)
    prev = None
    steps = 0
    n_act = trunk_policy.fc_policy.out_features
    while steps < target_steps:
        h = trunk_features(trunk_policy, obs, device)
        if label_mode == "random":
            acts = rng.randint(0, n_act, size=obs.shape[0])
        else:
            acts = sample_actions(label_policy, obs, device)
        obs, _rew, done, step_infos = env.step(acts)
        obs = obs.astype(np.float32)
        infos = []
        for i in range(obs.shape[0]):
            loc = localize_frame(obs[i], prev_hwc=prev)
            info = dict(step_infos[i]) if i < len(step_infos) and isinstance(step_infos[i], dict) else {}
            info["agent_xy"] = loc.get("agent_xy")
            info["line_xy"] = loc.get("line_xy")
            info["gem_xy"] = loc.get("gem_xy")
            infos.append(info)
        batch.append(h, acts, infos)
        prev = localize_frame(obs[0], prev_hwc=prev).get("img_hwc")
        steps += obs.shape[0]
    return batch


def collect_maze_frames(trunk_policy, label_policy, env, *, device, target_steps, label_mode, rng):
    from run_maze_i_goal_proxy_ladder import (  # noqa: E402
        _chw_to_hwc01,
        action_toward,
        cheese_xy_bright,
        motion_agent_xy,
    )
    from train_frozen_trunk_head_rewire import FrameBatch, sample_actions, trunk_features  # noqa: E402

    batch = FrameBatch()
    obs = env.reset()
    prev = _chw_to_hwc01(obs)
    steps = 0
    n_act = trunk_policy.fc_policy.out_features
    while steps < target_steps:
        h = trunk_features(trunk_policy, obs, device)
        if label_mode == "random":
            acts = rng.randint(0, n_act, size=obs.shape[0])
        else:
            acts = sample_actions(label_policy, obs, device)
        obs, _rew, done, step_infos = env.step(acts)
        infos = []
        for i in range(obs.shape[0]):
            cur = _chw_to_hwc01(obs if obs.ndim == 4 else obs[None])
            if obs.ndim == 4:
                cur = _chw_to_hwc01(obs[i : i + 1])
            cheese = cheese_xy_bright(cur)
            agent = motion_agent_xy(prev, cur)
            tr = (float(cur.shape[1] - 1), 0.0)
            info = dict(step_infos[i]) if i < len(step_infos) and isinstance(step_infos[i], dict) else {}
            if agent and cheese:
                a_cheese = action_toward(agent, cheese)
                a_corner = action_toward(agent, tr)
                info["toward_cheese"] = 1.0 if int(acts[i]) == a_cheese else 0.0
                info["toward_corner"] = 1.0 if int(acts[i]) == a_corner else 0.0
            infos.append(info)
        batch.append(h, acts, infos)
        prev = _chw_to_hwc01(obs)
        steps += obs.shape[0]
    return batch


def _run_env(args, *, env_name: str, compose_fn, make_env_fn, build_dirs_fn, primary_eval_fn, collect_fn):
    from train_frozen_trunk_head_rewire import (  # noqa: E402
        auc_ce,
        steps_to_epsilon,
        train_bc_on_features,
        train_restricted_curve,
    )

    device = args.device
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trunk_ckpt = Path(args.trunk_ckpt) if args.trunk_ckpt else None
    teacher_ckpt = Path(args.teacher_ckpt) if args.teacher_ckpt else None
    self_ckpt = Path(args.self_ckpt) if args.self_ckpt else trunk_ckpt
    t0 = time.time()

    policy = compose_fn(
        arm=args.arm,
        trunk_ckpt=trunk_ckpt,
        teacher_ckpt=teacher_ckpt,
        device=device,
    )

    bc_log: list = []
    subspace_race: dict = {}
    batch_meta: dict = {}

    if args.arm in {1, 2, 4}:
        label_mode = {1: "teacher", 2: "self", 4: "random"}[args.arm]
        label_ckpt = {1: teacher_ckpt, 2: self_ckpt, 4: trunk_ckpt}[args.arm]
        loader = load_ras_policy if env_name == "ras" else load_maze_policy
        label_policy = policy if label_mode == "random" else loader(label_ckpt, device)
        env = make_env_fn(
            num_envs=args.num_envs,
            num_levels=args.num_levels,
        )
        batch = collect_fn(
            policy,
            label_policy,
            env,
            device=device,
            target_steps=args.collect_steps,
            label_mode=label_mode,
            rng=np.random.RandomState(args.seed),
        )
        if hasattr(env, "close"):
            env.close()
        batch_meta = {"n_frames": len(batch), "label_mode": label_mode}
        (out / "bc_frames_meta.json").write_text(json.dumps(batch_meta, indent=2))
        np.savez_compressed(out / "bc_frames.npz", features=batch.features, actions=batch.actions)
        bc_log = train_bc_on_features(
            policy,
            batch.features,
            batch.actions,
            device=device,
            epochs=args.bc_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        (out / "bc_log.json").write_text(json.dumps(bc_log, indent=2))
        if args.arm == 1 and args.subspace_race:
            dirs = build_dirs_fn(batch, args.subspace_k)
            n_actions = policy.fc_policy.out_features
            subspace_race = {}
            for name, basis in {
                "goal": dirs["basis_goal"],
                "proxy": dirs["basis_proxy"],
                "full_fc": dirs["basis_full"],
                "random_k": dirs["basis_random"],
            }.items():
                curve = train_restricted_curve(
                    batch.features,
                    batch.actions,
                    basis,
                    device=device,
                    n_actions=n_actions,
                    max_steps=args.race_steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    log_every=args.race_log_every,
                )
                subspace_race[name] = {
                    "curve": curve,
                    "steps_to_eps": steps_to_epsilon(curve, args.race_eps),
                    "val_auc": auc_ce(curve),
                    "k": int(basis.shape[1]),
                }
            (out / "subspace_race.json").write_text(json.dumps(subspace_race, indent=2, default=str))

    primary = primary_eval_fn(policy, device, n_levels=args.eval_n)
    ckpt_path = out / f"arm{args.arm}_final.pth"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "arm": args.arm,
            "trunk_ckpt": str(trunk_ckpt) if trunk_ckpt else None,
            "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt else None,
        },
        ckpt_path,
    )
    summary = {
        "env": env_name,
        "arm": args.arm,
        "trunk_ckpt": str(trunk_ckpt) if trunk_ckpt else None,
        "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt else None,
        "primary_metric": primary,
        "bc_log": bc_log,
        "batch_meta": batch_meta,
        "subspace_race_summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "curve"}
            for k, v in subspace_race.items()
            if isinstance(v, dict) and "curve" in v
        },
        "elapsed_s": time.time() - t0,
        "ckpt": str(ckpt_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'summary.json'} primary={primary}")
    return summary


def build_ras_directions(batch, k: int) -> dict:
    from gm_geometry.pca.metrics import fit_dom_directions  # noqa: E402
    from train_frozen_trunk_head_rewire import subspace_basis, random_basis  # noqa: E402

    labels = ras_frame_labels(batch.infos, batch.actions)
    dom = fit_dom_directions(batch.features.astype(np.float64), labels)
    v_goal = dom.get("toward_line", np.zeros(batch.features.shape[1]))
    v_proxy = dom.get("toward_yellow", np.zeros(batch.features.shape[1]))
    H = batch.features.astype(np.float64)
    return {
        "v_goal": v_goal,
        "v_proxy": v_proxy,
        "basis_goal": subspace_basis(v_goal, H, k),
        "basis_proxy": subspace_basis(v_proxy, H, k),
        "basis_random": random_basis(H.shape[1], k, seed=42),
        "basis_full": np.eye(H.shape[1], dtype=np.float32)[:, : min(k, H.shape[1])],
        "label_counts": {k: int(v.sum()) for k, v in labels.items()},
    }


def build_maze_directions(batch, k: int) -> dict:
    from gm_geometry.pca.metrics import fit_dom_directions  # noqa: E402
    from train_frozen_trunk_head_rewire import subspace_basis, random_basis  # noqa: E402

    n = len(batch.infos)
    toward_cheese = np.zeros(n, dtype=np.float64)
    toward_corner = np.zeros(n, dtype=np.float64)
    for i, info in enumerate(batch.infos):
        if info.get("toward_cheese") is not None:
            toward_cheese[i] = float(info["toward_cheese"])
            toward_corner[i] = float(info["toward_corner"])
    dom = fit_dom_directions(
        batch.features.astype(np.float64),
        {"toward_cheese": toward_cheese, "toward_corner": toward_corner},
    )
    v_goal = dom.get("toward_cheese", np.zeros(batch.features.shape[1]))
    v_proxy = dom.get("toward_corner", np.zeros(batch.features.shape[1]))
    H = batch.features.astype(np.float64)
    return {
        "v_goal": v_goal,
        "v_proxy": v_proxy,
        "basis_goal": subspace_basis(v_goal, H, k),
        "basis_proxy": subspace_basis(v_proxy, H, k),
        "basis_random": random_basis(H.shape[1], k, seed=42),
        "basis_full": np.eye(H.shape[1], dtype=np.float32)[:, : min(k, H.shape[1])],
        "label_counts": {"toward_cheese": int(toward_cheese.sum()), "toward_corner": int(toward_corner.sum())},
    }


def run_ras(args):
    def primary_eval(policy, device, *, n_levels):
        return eval_ras_pref_goal(policy, device, n_frames=min(150, n_levels))

    return _run_env(
        args,
        env_name="ras",
        compose_fn=compose_ras_policy,
        make_env_fn=make_ras_env,
        build_dirs_fn=build_ras_directions,
        primary_eval_fn=primary_eval,
        collect_fn=collect_ras_frames,
    )


def run_maze(args):
    def primary_eval(policy, device, *, n_levels):
        cp = eval_maze_cheese_prefer(policy, device, n_frames=min(120, n_levels))
        val = eval_maze_val_success(policy, device, n_levels=min(n_levels, 100))
        return {**cp, **val}

    return _run_env(
        args,
        env_name="maze",
        compose_fn=compose_maze_policy,
        make_env_fn=make_maze_train_env,
        build_dirs_fn=build_maze_directions,
        primary_eval_fn=primary_eval,
        collect_fn=collect_maze_frames,
    )
