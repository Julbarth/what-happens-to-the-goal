"""Environment-specific concept, label and readout layer so one head-projection / steering core runs on several envs."""
# colour_shape_duel has a proxy/goal pair (yellow, line). maze_langosco uses only `corner`: on-policy the cheese sits
# top-right, so `cheese` is diagnostic only. coinrun has the single concept `coin`; two-concept axes are skipped.

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ConceptBundle:
    """Activations plus per-concept labels for one split (train or ood).

    ``labels`` maps concept name -> float array of 0/1 presence (length = rows of
    ``activations``). ``pixel`` holds optional per-frame pixel proxies (used only
    for the colour-shape value-head salience control). ``groups`` is the
    exchangeability unit for group-shuffle nulls (config index per row), or None.
    """

    activations: np.ndarray
    labels: dict[str, np.ndarray]
    pixel: dict[str, np.ndarray] | None = None
    groups: np.ndarray | None = None

    def salience_target(self, keys: tuple[str, ...]) -> np.ndarray | None:
        """``max`` over the given pixel keys — the object-salience regression target.

        Returns None when the env has no pixel salience concept (Coin / Maze).
        """
        if not keys or self.pixel is None:
            return None
        stack = [np.asarray(self.pixel[k], dtype=np.float64) for k in keys if k in self.pixel]
        if not stack:
            return None
        return np.maximum.reduce(stack)


def readout_weights(policy) -> tuple[np.ndarray, np.ndarray]:
    """``(W_policy, W_value)`` from an Impala ``CategoricalPolicy`` — shared across envs."""
    W_policy = policy.fc_policy.weight.detach().cpu().numpy().astype(np.float64)
    W_value = policy.fc_value.weight.detach().cpu().numpy().astype(np.float64)
    return W_policy, W_value


@dataclass
class EnvConceptSpec:
    """Static description of an environment's concepts and cached-npz layout.

    ``proxy_goal`` is ``(proxy_name, goal_name)`` for two-concept envs, else None. ``diagnostic_concepts`` are
    loaded and probed but never enter fused / diff axes or ratios."""

    env_key: str
    concept_names: tuple[str, ...]
    proxy_goal: tuple[str, str] | None
    label_key_map: dict[str, tuple[str, str]]
    activation_keys: tuple[str, str]
    salience_keys: tuple[str, ...] = field(default_factory=tuple)
    diagnostic_concepts: tuple[str, ...] = field(default_factory=tuple)
    design_note: str = ""

    def __post_init__(self) -> None:
        if self.proxy_goal is not None:
            p, g = self.proxy_goal
            missing = [c for c in (p, g) if c not in self.concept_names]
            if missing:
                raise ValueError(f"{self.env_key}: proxy_goal {missing} not in concept_names")
        all_concepts = set(self.concept_names) | set(self.diagnostic_concepts)
        for c in all_concepts:
            if c not in self.label_key_map:
                raise ValueError(f"{self.env_key}: concept {c!r} missing from label_key_map")
        overlap = set(self.concept_names) & set(self.diagnostic_concepts)
        if overlap:
            raise ValueError(f"{self.env_key}: concepts cannot be both primary and diagnostic: {overlap}")

    @property
    def is_single_concept(self) -> bool:
        return self.proxy_goal is None

    @property
    def all_label_concepts(self) -> tuple[str, ...]:
        return tuple(self.concept_names) + tuple(self.diagnostic_concepts)

    def bundles_from_npz(self, path: str | Path) -> tuple[ConceptBundle, ConceptBundle]:
        """Load train / ood :class:`ConceptBundle` (primary and diagnostic labels) from a cached activation npz."""
        z = np.load(Path(path))
        tr_key, ood_key = self.activation_keys
        for key in (tr_key, ood_key):
            if key not in z.files:
                raise KeyError(
                    f"{self.env_key}: activation key {key!r} not in {Path(path).name} "
                    f"(has {list(z.files)})"
                )

        def _labels(split_idx: int) -> dict[str, np.ndarray]:
            out: dict[str, np.ndarray] = {}
            for c in self.all_label_concepts:
                lk = self.label_key_map[c][split_idx]
                if lk not in z.files:
                    raise KeyError(f"{self.env_key}: label key {lk!r} not in {Path(path).name}")
                out[c] = z[lk].astype(np.float64).ravel()
            return out

        train = ConceptBundle(
            activations=np.asarray(z[tr_key], dtype=np.float64),
            labels=_labels(0),
        )
        ood = ConceptBundle(
            activations=np.asarray(z[ood_key], dtype=np.float64),
            labels=_labels(1),
        )
        return train, ood


# --- Registry ---------------------------------------------------------------

COLOUR_SHAPE_DUEL = EnvConceptSpec(
    env_key="colour_shape_duel",
    concept_names=("yellow", "line"),
    proxy_goal=("yellow", "line"),
    label_key_map={
        "yellow": ("yellow_present", "yellow_present"),
        "line": ("line_present", "line_present"),
    },
    activation_keys=("train_activations", "test_activations"),
    salience_keys=("yellow", "gem"),
)

MAZE_LANGOSCO = EnvConceptSpec(
    env_key="maze_langosco",
    # Corner is the only on-policy readable GM axis; cheese stays diagnostic and never fills the goal slot.
    concept_names=("corner",),
    proxy_goal=None,
    diagnostic_concepts=("cheese",),
    label_key_map={
        "corner": ("train_corner", "ood_corner"),
        "cheese": ("train_cheese", "ood_cheese"),
    },
    activation_keys=("train_emb", "ood_emb"),
    design_note=(
        "On-policy maze_aisc rr=0 pins cheese to the top-right cell → cheese "
        "label near-constant → proxy/goal contrast undecidable. Primary axis = "
        "corner only. cheese AUC reported when both classes exist (usually OOD / "
        "off-policy); never fused/diff/ratios. Off-policy cheese labels needed "
        "before claiming a two-concept Maze I geometry."
    ),
)

COINRUN = EnvConceptSpec(
    env_key="coinrun",
    concept_names=("coin",),
    proxy_goal=None,
    label_key_map={"coin": ("train_coin", "ood_coin")},
    activation_keys=("train_activations", "test_activations"),
    design_note=(
        "Single-concept: coin presence. No go-right / end-marker linear proxy. "
        "Fused/diff/ratios omitted. Verify train/ood coin_pos rates after collect."
    ),
)

REGISTRY: dict[str, EnvConceptSpec] = {
    spec.env_key: spec for spec in (COLOUR_SHAPE_DUEL, MAZE_LANGOSCO, COINRUN)
}


def get_spec(env_key: str) -> EnvConceptSpec:
    if env_key not in REGISTRY:
        raise KeyError(f"unknown env_key {env_key!r}; have {sorted(REGISTRY)}")
    return REGISTRY[env_key]


# --- Seed discovery ---------------------------------------------------------
# Training runs write into these conventional roots; the collectors write the per-seed npz.

CHECKPOINT_SUBDIR: dict[str, str] = {
    "coinrun": "train_logs/coinrun",
    "maze_langosco": "train_logs/maze_aisc",
}

_SEED_RE = re.compile(r"seed_(\d+)")
_STEP_RE = re.compile(r"model_(\d+)\.pth$")


@dataclass
class SeedArtifact:
    """One trained seed: its checkpoints (by step) and cached activation npz (if any)."""

    env_key: str
    seed: int
    exp: str
    run_dir: Path
    checkpoints: dict[int, Path]  # training step -> checkpoint path
    npz: Path | None = None

    @property
    def latest_step(self) -> int:
        return max(self.checkpoints) if self.checkpoints else -1

    @property
    def latest_checkpoint(self) -> Path | None:
        return self.checkpoints[self.latest_step] if self.checkpoints else None

    def checkpoint_at(self, step: int, *, tol_frac: float = 0.05) -> Path | None:
        """Closest checkpoint to ``step`` within ``tol_frac`` (e.g. the ~200M ckpt)."""
        if not self.checkpoints:
            return None
        best = min(self.checkpoints, key=lambda s: abs(s - step))
        return self.checkpoints[best] if abs(best - step) <= tol_frac * step else None


def discover_langosco_seeds(
    env_key: str,
    *,
    checkpoints_root: Path,
    activations_root: Path | None = None,
    exp_glob: str = "*",
    min_latest_step: int = 0,
    warn: bool = True,
) -> list[SeedArtifact]:
    """Enumerate every trained seed for an env under ``checkpoints_root``.

    ``exp_glob`` filters experiment folders and ``min_latest_step`` drops seeds whose latest checkpoint is
    too early. With ``warn=True``, an existing root that yields zero seeds prints the available folders."""
    if env_key not in CHECKPOINT_SUBDIR:
        raise KeyError(f"no checkpoint layout for env_key {env_key!r}")
    base = Path(checkpoints_root) / CHECKPOINT_SUBDIR[env_key]
    by_seed: dict[tuple[str, int], dict[int, Path]] = {}
    run_dirs: dict[tuple[str, int], Path] = {}
    available_exps: list[str] = []
    matched_exps: list[str] = []

    if not base.exists():
        if warn:
            print(
                f"WARNING: checkpoint root missing for {env_key}: {base}\n"
                f"  → discovery will return 0 seeds (not a glob mismatch).",
                file=sys.stderr,
            )
    else:
        available_exps = sorted(p.name for p in base.iterdir() if p.is_dir())
        for exp_dir in sorted(base.glob(exp_glob)):
            if not exp_dir.is_dir():
                continue
            matched_exps.append(exp_dir.name)
            for ckpt in exp_dir.rglob("model_*.pth"):
                seed_m = _SEED_RE.search(str(ckpt.parent))
                step_m = _STEP_RE.search(ckpt.name)
                if not seed_m or not step_m:
                    continue
                key = (exp_dir.name, int(seed_m.group(1)))
                by_seed.setdefault(key, {})[int(step_m.group(1))] = ckpt
                run_dirs[key] = ckpt.parent

    dropped_min_step: list[tuple[str, int, int]] = []
    out: list[SeedArtifact] = []
    for (exp, seed), ckpts in sorted(by_seed.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if not ckpts:
            continue
        latest = max(ckpts)
        if latest < min_latest_step:
            dropped_min_step.append((exp, seed, latest))
            continue
        npz = None
        if activations_root is not None:
            matches = sorted(Path(activations_root).glob(f"*seed{seed}*/activations.npz"))
            npz = matches[0] if matches else None
        out.append(
            SeedArtifact(
                env_key=env_key, seed=seed, exp=exp, run_dir=run_dirs[(exp, seed)],
                checkpoints=dict(sorted(ckpts.items())), npz=npz,
            )
        )

    if warn and base.exists() and not out:
        print("=" * 72, file=sys.stderr)
        print(
            f"WARNING: 0 seeds discovered for {env_key} with "
            f"exp_glob={exp_glob!r} min_latest_step={min_latest_step}",
            file=sys.stderr,
        )
        print(f"  checkpoint root: {base}", file=sys.stderr)
        print(f"  available experiment folders ({len(available_exps)}):", file=sys.stderr)
        for name in available_exps or ["(none)"]:
            print(f"    - {name}", file=sys.stderr)
        print(f"  folders matching glob ({len(matched_exps)}):", file=sys.stderr)
        for name in matched_exps or ["(none — naming mismatch?)"]:
            print(f"    - {name}", file=sys.stderr)
        if dropped_min_step:
            print(
                f"  seeds dropped by min_latest_step={min_latest_step}:",
                file=sys.stderr,
            )
            for exp, seed, latest in dropped_min_step:
                print(f"    - {exp} seed={seed} latest_step={latest:,}", file=sys.stderr)
        print(
            "  → This is NOT 'no new seeds trained yet' if folders exist above. "
            "Fix --exp-glob / --min-step.",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)

    return out
