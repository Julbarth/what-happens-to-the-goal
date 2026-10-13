#!/usr/bin/env python3
"""Frozen-trunk head rewiring by behaviour cloning for Coin, with Maze and Attribute Maze adapters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing path (tried: {[str(p) for p in paths]})")


def _parse_env_flag() -> str:
    if "--env" in sys.argv:
        idx = sys.argv.index("--env")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return "coin"


_ENV = _parse_env_flag()
TRAIN_ROOT = _existing(
    ROOT / "external" / "train-procgen-pytorch",
    ROOT / "vendored" / "train-procgen-pytorch",
)
PROCGEN_AISC = _existing(
    ROOT / "external" / "procgenAISC",
    ROOT / "vendored" / "procgenAISC",
)
CS_PROC = _existing(
    ROOT / "external" / "colour-shape-goal-misgeneralization" / "procgenAISC",
    ROOT / "vendored" / "cs-procgenAISC",
)
CS_TRAIN = _existing(
    ROOT / "external" / "colour-shape-goal-misgeneralization" / "train-procgen",
    ROOT / "vendored" / "cs-train-procgen",
)

for sub in ("src", "pkg"):
    subroot = ROOT / sub
    if subroot.exists():
        sys.path.insert(0, str(subroot))
sys.path.insert(0, str(TRAIN_ROOT))

from gm_geometry.utils import device as pick_device  # noqa: E402

if _ENV == "ras":
    sys.path.insert(0, str(CS_TRAIN))
    sys.path.insert(0, str(CS_PROC))
else:
    sys.path.insert(0, str(PROCGEN_AISC))

if _ENV != "ras":
    from common.env.procgen_wrappers import (  # noqa: E402
        ScaledFloatFrame,
        TransposeFrame,
        VecExtractDictObs,
        VecNormalize,
    )
    from common.model import ImpalaModel  # noqa: E402
    from common.policy import CategoricalPolicy  # noqa: E402
    from gm_geometry.pca.metrics import fit_dom_directions  # noqa: E402
    from procgen import ProcgenEnv  # noqa: E402

    from run_coin_behaviour import (  # noqa: E402
        VAL_START,
        crossover_policy,
        eval_policy_val,
        load_init_policy,
        load_policy,
    )

    VAL_START_COIN = VAL_START
else:
    ImpalaModel = None  # type: ignore[misc, assignment]
    VAL_START_COIN = 1_000_000


def make_coin_env(
    env_name: str = "coinrun_aisc",
    *,
    num_envs: int,
    start_level: int,
    num_levels: int,
    hide_visible_coin: bool = False,
):
    kwargs = {}
    if hide_visible_coin:
        kwargs["hide_visible_coin"] = True
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="hard",
        num_threads=max(2, num_envs // 2),
        **kwargs,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False, ret=False)
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        return np.zeros_like(v)
    return v / n


def subspace_basis(v: np.ndarray, H: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal D×k basis: v̂ plus top PCs of H orthogonal to v."""
    v = _unit(v)
    k = max(1, min(k, H.shape[1], H.shape[0]))
    if k == 1:
        return v.reshape(-1, 1).astype(np.float32)
    Hc = H - H.mean(axis=0, keepdims=True)
    proj = (Hc @ v)[:, None] * v
    H_orth = Hc - proj
    _u, _s, vt = np.linalg.svd(H_orth, full_matrices=False)
    extra = min(k - 1, vt.shape[0])
    cols = [v]
    for i in range(extra):
        cols.append(vt[i])
    Q, _ = np.linalg.qr(np.column_stack(cols).astype(np.float64))
    return Q[:, :k].astype(np.float32)


def random_basis(dim: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    M = rng.randn(dim, k).astype(np.float64)
    Q, _ = np.linalg.qr(M)
    return Q[:, :k].astype(np.float32)


@dataclass
class FrameBatch:
    features: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    actions: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    infos: list[dict] = field(default_factory=list)

    def append(self, h: np.ndarray, act: np.ndarray, infos: list[dict]):
        h = h.astype(np.float32)
        act = act.astype(np.int64)
        if len(self) == 0:
            self.features = h
            self.actions = act
        else:
            self.features = np.concatenate([self.features, h], axis=0)
            self.actions = np.concatenate([self.actions, act], axis=0)
        self.infos.extend(infos)

    def __len__(self):
        return int(self.features.shape[0])


BLOCK3_FLAT_DIM = 32 * 8 * 8  # Impala scale=1, 8×8 block3 map


def freeze_trunk(policy: CategoricalPolicy) -> None:
    for p in policy.embedder.parameters():
        p.requires_grad = False
    policy.embedder.eval()


def freeze_through_block3(policy: CategoricalPolicy) -> None:
    """Freeze conv trunk only; leave embedder.fc trainable."""
    emb = policy.embedder
    for block in (emb.block1, emb.block2, emb.block3):
        for p in block.parameters():
            p.requires_grad = False
        block.eval()


def reinit_heads(policy: CategoricalPolicy) -> None:
    nn.init.orthogonal_(policy.fc_policy.weight, gain=0.01)
    nn.init.zeros_(policy.fc_policy.bias)
    nn.init.orthogonal_(policy.fc_value.weight, gain=1.0)
    nn.init.zeros_(policy.fc_value.bias)


def reinit_fc_and_heads(policy: CategoricalPolicy) -> None:
    """Fresh 2048→256 embedder.fc plus policy/value heads (Impala init)."""
    nn.init.xavier_uniform_(policy.embedder.fc.weight)
    nn.init.zeros_(policy.embedder.fc.bias)
    reinit_heads(policy)


def freeze_embedder_fc(policy: CategoricalPolicy) -> None:
    for p in policy.embedder.fc.parameters():
        p.requires_grad = False
    policy.embedder.fc.eval()


@torch.no_grad()
def block3_to_fc_features(
    policy: CategoricalPolicy,
    block3_features: np.ndarray,
    device: str,
) -> np.ndarray:
    """Apply frozen/trainable embedder.fc + ReLU to cached block3 flats.

    Inputs are clamped at zero because the rollout forward applies ReLU(block3) before fc.
    """
    x = torch.as_tensor(block3_features, dtype=torch.float32, device=device).clamp_min(0.0)
    return F.relu(policy.embedder.fc(x)).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def fc_relu_activation_stats(
    policy: CategoricalPolicy,
    block3_features: np.ndarray,
    device: str,
    *,
    max_n: int = 10_000,
) -> dict:
    """Summary stats of ReLU(embedder.fc(block3)) on a frame subsample."""
    n = min(len(block3_features), max_n)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "frac_nonzero": float("nan"), "n": 0}
    x = torch.as_tensor(block3_features[:n], dtype=torch.float32, device=device).clamp_min(0.0)
    h = F.relu(policy.embedder.fc(x)).detach().cpu().numpy()
    return {
        "mean": float(h.mean()),
        "std": float(h.std()),
        "frac_nonzero": float((h > 0).mean()),
        "n": int(n),
    }


@torch.no_grad()
def trunk_features(policy: CategoricalPolicy, obs_np: np.ndarray, device: str) -> np.ndarray:
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    return policy.embedder(x).detach().cpu().numpy()


def action_vx(action: int) -> int:
    a = int(action)
    if a >= 9:
        return 0
    return a // 3 - 1


def coin_frame_labels(infos: list[dict], actions: np.ndarray) -> dict[str, np.ndarray]:
    """Binary labels for Dom directions on Coin frames."""
    n = len(infos)
    toward_coin = np.zeros(n, dtype=np.float64)
    go_right = np.zeros(n, dtype=np.float64)
    high_progress = np.zeros(n, dtype=np.float64)
    progress_vals = []
    for i, info in enumerate(infos):
        ax = float(info.get("agent_x_fp", 0)) / 100.0
        width = float(info.get("level_width", 0) or 1)
        progress_vals.append(ax / max(width, 1))
        cx = float(info.get("coin_x", ax))
        coin_dx = cx - ax
        vx = action_vx(int(actions[i]))
        if abs(coin_dx) > 0.5 and vx != 0:
            toward_coin[i] = 1.0 if np.sign(vx) == np.sign(coin_dx) else 0.0
        if vx > 0:
            go_right[i] = 1.0
    med = float(np.median(progress_vals)) if progress_vals else 0.5
    for i, p in enumerate(progress_vals):
        high_progress[i] = 1.0 if p >= med else 0.0
    return {
        "toward_coin": toward_coin,
        "go_right": go_right,
        "high_progress": high_progress,
    }


def build_coin_directions(batch: FrameBatch, k: int) -> dict[str, np.ndarray]:
    labels = coin_frame_labels(batch.infos, batch.actions)
    dom = fit_dom_directions(batch.features.astype(np.float64), labels)
    v_goal = dom.get("toward_coin", np.zeros(batch.features.shape[1]))
    v_proxy = dom.get("go_right", np.zeros(batch.features.shape[1]))
    if float(np.linalg.norm(v_proxy)) < 1e-8:
        v_proxy = dom.get("high_progress", v_proxy)
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


class RestrictedPolicyHead(nn.Module):
    """Linear action head on h @ B (B is fixed D×k)."""

    def __init__(self, basis: np.ndarray, n_actions: int):
        super().__init__()
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
        self.fc = nn.Linear(basis.shape[1], n_actions)
        nn.init.orthogonal_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h @ self.basis
        return self.fc(z)


def train_restricted_curve(
    features: np.ndarray,
    actions: np.ndarray,
    basis: np.ndarray,
    *,
    device: str,
    n_actions: int,
    max_steps: int,
    batch_size: int,
    lr: float,
    val_frac: float = 0.15,
    log_every: int = 50,
) -> list[dict]:
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(features))
    n_val = max(1, int(len(features) * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    head = RestrictedPolicyHead(basis, n_actions).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    log: list[dict] = []
    step = 0
    while step < max_steps:
        perm = rng.permutation(tr_idx)
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            xb = torch.as_tensor(features[b], device=device)
            yb = torch.as_tensor(actions[b], device=device, dtype=torch.long)
            opt.zero_grad()
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            step += len(b)
            if step % log_every < batch_size or step >= max_steps:
                head.eval()
                with torch.no_grad():
                    xv = torch.as_tensor(features[val_idx], device=device)
                    yv = torch.as_tensor(actions[val_idx], device=device, dtype=torch.long)
                    vloss = float(F.cross_entropy(head(xv), yv).cpu())
                log.append({"step": step, "train_ce": float(loss.cpu()), "val_ce": vloss})
                head.train()
            if step >= max_steps:
                break
    return log


def train_bc_on_features(
    policy: CategoricalPolicy,
    features: np.ndarray,
    actions: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> list[dict]:
    freeze_trunk(policy)
    for p in policy.fc_policy.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(policy.fc_policy.parameters(), lr=lr)
    rng = np.random.RandomState(0)
    n_val = max(1, len(features) // 5)
    idx = rng.permutation(len(features))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    log: list[dict] = []
    grad_steps = 0
    for epoch in range(epochs):
        perm = rng.permutation(tr_idx)
        total = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            xb = torch.as_tensor(features[b], device=device)
            yb = torch.as_tensor(actions[b], device=device, dtype=torch.long)
            opt.zero_grad()
            logits = policy.fc_policy(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.fc_policy.parameters(), 0.5)
            opt.step()
            grad_steps += len(b)
            total += float(loss) * len(b)
        with torch.no_grad():
            xv = torch.as_tensor(features[val_idx], device=device)
            yv = torch.as_tensor(actions[val_idx], device=device, dtype=torch.long)
            logits_v = policy.fc_policy(xv)
            vloss = float(F.cross_entropy(logits_v, yv).cpu())
            vacc = float((logits_v.argmax(-1) == yv).float().mean().cpu())
        row = {
            "epoch": epoch,
            "step": grad_steps,
            "train_ce": total / max(len(tr_idx), 1),
            "val_ce": vloss,
            "val_acc": vacc,
        }
        log.append(row)
        print(
            f"  bc epoch {epoch}: train_ce={row['train_ce']:.4f} "
            f"val_ce={vloss:.4f} val_acc={vacc:.4f}"
        )
    return log


def train_bc_fresh_fc(
    policy: CategoricalPolicy,
    block3_features: np.ndarray,
    actions: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> list[dict]:
    """BC on cached block3 flats: train embedder.fc + fc_policy (blocks frozen)."""
    freeze_through_block3(policy)
    for p in policy.embedder.fc.parameters():
        p.requires_grad = True
    for p in policy.fc_policy.parameters():
        p.requires_grad = True
    params = list(policy.embedder.fc.parameters()) + list(policy.fc_policy.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.RandomState(0)
    n_val = max(1, len(block3_features) // 5)
    idx = rng.permutation(len(block3_features))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    log: list[dict] = []
    grad_steps = 0
    fc = policy.embedder.fc
    head = policy.fc_policy
    for epoch in range(epochs):
        perm = rng.permutation(tr_idx)
        total = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s : s + batch_size]
            xb = torch.as_tensor(block3_features[b], device=device).clamp_min(0.0)
            yb = torch.as_tensor(actions[b], device=device, dtype=torch.long)
            opt.zero_grad()
            h = F.relu(fc(xb))
            logits = head(h)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            grad_steps += len(b)
            total += float(loss) * len(b)
        with torch.no_grad():
            xv = torch.as_tensor(block3_features[val_idx], device=device).clamp_min(0.0)
            yv = torch.as_tensor(actions[val_idx], device=device, dtype=torch.long)
            logits_v = head(F.relu(fc(xv)))
            vloss = float(F.cross_entropy(logits_v, yv).cpu())
            vacc = float((logits_v.argmax(-1) == yv).float().mean().cpu())
        row = {
            "epoch": epoch,
            "step": grad_steps,
            "train_ce": total / max(len(tr_idx), 1),
            "val_ce": vloss,
            "val_acc": vacc,
        }
        log.append(row)
        print(
            f"  fresh-fc bc epoch {epoch}: train_ce={row['train_ce']:.4f} "
            f"val_ce={vloss:.4f} val_acc={vacc:.4f}"
        )
    return log


@torch.no_grad()
def sample_actions(policy: CategoricalPolicy, obs_np: np.ndarray, device: str) -> np.ndarray:
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    n = x.shape[0]
    hid = torch.zeros(n, policy.embedder.output_dim, device=device)
    dist, _v, _ = policy(x, hid, torch.zeros(n, device=device))
    return dist.probs.argmax(-1).cpu().numpy().astype(np.int64)


def collect_bc_frames(
    trunk_policy: CategoricalPolicy,
    label_policy: CategoricalPolicy,
    env,
    *,
    device: str,
    target_steps: int,
    label_mode: str,
    rng: np.random.RandomState,
) -> FrameBatch:
    batch = FrameBatch()
    obs = env.reset()
    steps = 0
    n_act = trunk_policy.fc_policy.out_features
    while steps < target_steps:
        h = trunk_features(trunk_policy, obs, device)
        if label_mode == "random":
            acts = rng.randint(0, n_act, size=obs.shape[0])
        else:
            acts = sample_actions(label_policy, obs, device)
        obs, _rew, done, step_infos = env.step(acts)
        if isinstance(step_infos, (list, tuple)):
            infos = [
                dict(step_infos[i]) if i < len(step_infos) and isinstance(step_infos[i], dict) else {}
                for i in range(obs.shape[0])
            ]
        elif isinstance(step_infos, dict):
            infos = [dict(step_infos) for _ in range(obs.shape[0])]
        else:
            infos = [{} for _ in range(obs.shape[0])]
        batch.append(h, acts, infos)
        steps += obs.shape[0]
    return batch


def eval_coin_dependence(policy, *, device: str, n_levels: int, start_level: int = VAL_START_COIN):
    vis = eval_policy_val(
        policy,
        n_levels=n_levels,
        start_level=start_level,
        device=device,
        max_steps=1000,
        hide_visible_coin=False,
    )
    hid = eval_policy_val(
        policy,
        n_levels=n_levels,
        start_level=start_level,
        device=device,
        max_steps=1000,
        hide_visible_coin=True,
    )
    return {
        "visible": {"success_rate": vis["success_rate"], "gm_rate": vis["gm_rate"]},
        "hidden": {"success_rate": hid["success_rate"], "gm_rate": hid["gm_rate"]},
        "coin_dependence": vis["success_rate"] - hid["success_rate"],
        "n_levels": n_levels,
    }


# Arms: 0 GM trunk + GM head; 1/2/4 GM trunk + fresh head cloned on teacher/self/random labels;
# 3 teacher trunk + head; 5 GM trunk + teacher head; 6 init trunk + teacher head.
def compose_policy(
    *,
    arm: int,
    trunk_ckpt: Path | None,
    teacher_ckpt: Path | None,
    device: str,
) -> CategoricalPolicy:
    if arm == 0:
        assert trunk_ckpt is not None
        policy = load_policy(trunk_ckpt, device)
        freeze_trunk(policy)
        return policy
    if arm == 3:
        assert teacher_ckpt is not None
        policy = load_policy(teacher_ckpt, device)
        freeze_trunk(policy)
        return policy
    if arm == 5:
        assert trunk_ckpt is not None and teacher_ckpt is not None
        policy = crossover_policy(trunk_ckpt, teacher_ckpt, device)
        freeze_trunk(policy)
        return policy
    if arm == 6:
        assert teacher_ckpt is not None
        policy = crossover_policy(None, teacher_ckpt, device, init_trunk=True)
        freeze_trunk(policy)
        return policy
    # Arms 1,2,4: GM trunk + fresh head
    assert trunk_ckpt is not None
    with open(TRAIN_ROOT / "hyperparams/procgen/config.yml") as f:
        recurrent = yaml.safe_load(f)["hard-500"]["recurrent"]
    probe = make_coin_env(num_envs=1, start_level=0, num_levels=1)
    model = ImpalaModel(in_channels=probe.observation_space.shape[0])
    policy = CategoricalPolicy(model, recurrent, probe.action_space.n)
    state = torch.load(trunk_ckpt, map_location="cpu")
    policy.load_state_dict(state["model_state_dict"])
    if hasattr(probe, "close"):
        probe.close()
    freeze_trunk(policy)
    reinit_heads(policy)
    policy.to(device)
    return policy


def steps_to_epsilon(curve: list[dict], eps: float, key: str = "val_ce") -> int | None:
    for row in curve:
        if row.get(key, 1e9) <= eps:
            return int(row.get("step", row.get("epoch", 0)))
    return None


def steps_to_fraction_of_final(
    curve: list[dict], frac: float = 0.9, key: str = "val_acc"
) -> int | None:
    vals = [float(r[key]) for r in curve if r.get(key) is not None]
    if not vals:
        return None
    target = frac * max(vals)
    for row in curve:
        v = row.get(key)
        if v is not None and float(v) >= target:
            return int(row.get("step", row.get("epoch", 0)))
    return None


def auc_ce(curve: list[dict], key: str = "val_ce") -> float:
    if not curve:
        return float("nan")
    steps = np.array([r["step"] for r in curve], dtype=np.float64)
    vals = np.array([r[key] for r in curve], dtype=np.float64)
    order = np.argsort(steps)
    steps, vals = steps[order], vals[order]
    if len(steps) < 2:
        return float(vals[0]) if len(vals) else float("nan")
    return float(np.trapz(vals, steps) / max(steps[-1] - steps[0], 1))


def run_coin(args):
    device = args.device
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trunk_ckpt = Path(args.trunk_ckpt) if args.trunk_ckpt else None
    teacher_ckpt = Path(args.teacher_ckpt) if args.teacher_ckpt else None
    self_ckpt = Path(args.self_ckpt) if args.self_ckpt else trunk_ckpt

    t0 = time.time()
    policy = compose_policy(
        arm=args.arm,
        trunk_ckpt=trunk_ckpt,
        teacher_ckpt=teacher_ckpt,
        device=device,
    )

    bc_log: list[dict] = []
    subspace_race: dict = {}
    batch_meta: dict = {}

    train_arms = {1, 2, 4}
    if args.arm in train_arms:
        if args.arm == 1 and teacher_ckpt is None:
            raise SystemExit("--teacher-ckpt required for arm 1")
        if args.arm == 2 and self_ckpt is None:
            raise SystemExit("--self-ckpt or --trunk-ckpt required for arm 2")
        label_mode = {1: "teacher", 2: "self", 4: "random"}[args.arm]
        label_ckpt = {
            1: teacher_ckpt,
            2: self_ckpt,
            4: trunk_ckpt,
        }[args.arm]
        label_policy = (
            load_policy(label_ckpt, device)
            if label_mode != "random"
            else policy
        )
        env = make_coin_env(
            args.train_env,
            num_envs=args.num_envs,
            start_level=0,
            num_levels=args.num_levels,
        )
        rng = np.random.RandomState(args.seed)
        print(f"Collecting {args.collect_steps} BC frames (label={label_mode})...")
        batch = collect_bc_frames(
            policy,
            label_policy,
            env,
            device=device,
            target_steps=args.collect_steps,
            label_mode=label_mode,
            rng=rng,
        )
        if hasattr(env, "close"):
            env.close()
        batch_meta = {"n_frames": len(batch), "label_mode": label_mode}
        (out / "bc_frames_meta.json").write_text(json.dumps(batch_meta, indent=2))
        np.savez_compressed(
            out / "bc_frames.npz",
            features=batch.features,
            actions=batch.actions,
        )

        print(f"Training BC head ({args.bc_epochs} epochs)...")
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
            dirs = build_coin_directions(batch, args.subspace_k)
            np.savez_compressed(out / "subspace_dirs.npz", **{k: v for k, v in dirs.items() if isinstance(v, np.ndarray)})
            n_actions = policy.fc_policy.out_features
            race_spec = {
                "goal": dirs["basis_goal"],
                "proxy": dirs["basis_proxy"],
                "full_fc": dirs["basis_full"],
                "random_k": dirs["basis_random"],
            }
            subspace_race = {}
            for name, basis in race_spec.items():
                print(f"  subspace race: {name} (k={basis.shape[1]})...")
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
            subspace_race["label_counts"] = dirs["label_counts"]
            (out / "subspace_race.json").write_text(
                json.dumps(subspace_race, indent=2, default=str)
            )

    print(f"Evaluating arm {args.arm} (n={args.eval_n})...")
    val = eval_policy_val(
        policy,
        n_levels=args.eval_n,
        start_level=VAL_START_COIN,
        device=device,
        max_steps=1000,
    )
    coin_dep = eval_coin_dependence(
        policy, device=device, n_levels=args.eval_n, start_level=VAL_START_COIN
    )

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
        "env": "coin",
        "arm": args.arm,
        "trunk_ckpt": str(trunk_ckpt) if trunk_ckpt else None,
        "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt else None,
        "self_ckpt": str(self_ckpt) if self_ckpt else None,
        "primary_metric": "coin_dependence",
        "val": val,
        "coin_dependence": coin_dep,
        "bc_log": bc_log,
        "batch_meta": batch_meta,
        "subspace_race_summary": {
            k: {kk: vv for kk, vv in v.items() if kk != "curve"}
            for k, v in subspace_race.items()
            if isinstance(v, dict) and "curve" in v
        },
        "elapsed_s": time.time() - t0,
        "ckpt": str(ckpt_path),
        "preregistration": None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"wrote {out / 'summary.json'} "
        f"coin_dep={coin_dep['coin_dependence']:+.3f} "
        f"(vis={coin_dep['visible']['success_rate']:.3f} "
        f"hid={coin_dep['hidden']['success_rate']:.3f})"
    )
    return summary


def main():
    p = argparse.ArgumentParser(description="Frozen-trunk head rewiring (behaviour cloning only)")
    p.add_argument("--env", choices=("coin", "ras", "maze"), default="coin")
    p.add_argument("--arm", type=int, choices=(0, 1, 2, 3, 4, 5, 6), required=True)
    p.add_argument("--trunk-ckpt", type=str, default=None)
    p.add_argument("--teacher-ckpt", type=str, default=None)
    p.add_argument("--self-ckpt", type=str, default=None, help="GM policy for arm 2 labels")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", default=str(pick_device()))
    p.add_argument("--gpu-id", type=int, default=None, help="Set CUDA_VISIBLE_DEVICES to this id")
    p.add_argument("--train-env", default="coinrun_aisc")
    p.add_argument("--collect-steps", type=int, default=500_000)
    p.add_argument("--bc-epochs", type=int, default=20)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-levels", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subspace-k", type=int, default=8)
    p.add_argument("--subspace-race", action="store_true", default=True)
    p.add_argument("--no-subspace-race", action="store_false", dest="subspace_race")
    p.add_argument("--race-steps", type=int, default=5000)
    p.add_argument("--race-log-every", type=int, default=100)
    p.add_argument("--race-eps", type=float, default=1.5, help="val CE threshold for steps-to-eps")
    args = p.parse_args()
    if args.gpu_id is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        if args.device.startswith("cuda"):
            args.device = "cuda"

    if args.env == "coin":
        run_coin(args)
    elif args.env == "ras":
        from head_rewire_ras_maze import run_ras  # noqa: E402

        run_ras(args)
    elif args.env == "maze":
        from head_rewire_ras_maze import run_maze  # noqa: E402

        run_maze(args)
    else:
        raise SystemExit(f"unknown env {args.env}")


if __name__ == "__main__":
    main()
