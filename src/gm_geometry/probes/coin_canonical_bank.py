"""Loader for the frozen canonical Coin state bank, with population-hash verification and policy-response subsets."""
# The bank is task-defined and policy-blind; whether a policy's argmax responds to coin placement is a per-checkpoint
# measurement (`policy_actions_differ`), so action-based analyses should report the full bank and the responding subset.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GM_LABELS = ["gm_6033", "gm_6037", "gm_6051"]
NONGM_LABELS = ["nongm_7001", "nongm_7002", "nongm_7003"]
ALL_LABELS = GM_LABELS + NONGM_LABELS

DEFAULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data/probes/coin_canonical_bank_20260924/coin_canonical_bank.json"
)


def freeze_hash(states: list[dict]) -> str:
    """sha256 over the population identity (source, level, both coin placements); must match the bank builder."""
    ident = [[s["source"], s["level"], s["goal_xy"], s["shortcut_xy"]] for s in states]
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


def load_bank(path: Path | None = None, *, expect_sha256: str | None = None) -> dict:
    """Load and verify the frozen bank; raises if the population hash does not match the file or ``expect_hash``."""
    fp = Path(path) if path is not None else DEFAULT_PATH
    bank = json.loads(fp.read_text())
    recorded = bank.get("freeze_sha256")
    actual = freeze_hash(bank["states"])
    if recorded != actual:
        raise ValueError(
            f"canonical Coin bank hash mismatch at {fp}: file records {recorded}, "
            f"states hash to {actual} -- the bank was edited after freezing"
        )
    if expect_sha256 is not None and actual != expect_sha256:
        raise ValueError(
            f"canonical Coin bank at {fp} hashes to {actual}, caller expected {expect_sha256}"
        )
    return bank


def select(bank: dict, *, source: str | None = None, diagnostic_for: str | None = None,
           diagnostic_all: bool = False, cohort: str | None = None) -> list[dict]:
    """Subset the bank by measured policy response to coin placement.

    ``diagnostic_for`` keeps states where that checkpoint responds, ``cohort`` ('gm' or 'nongm') where every
    cohort member responds, and ``diagnostic_all`` where all checkpoints respond."""
    out = bank["states"]
    if source is not None:
        out = [s for s in out if s["source"] == source]
    if diagnostic_for is not None:
        out = [s for s in out if s["per_ckpt"][diagnostic_for]["policy_actions_differ"]]
    if cohort == "gm":
        out = [s for s in out if s["policy_resolves_gm"]]
    elif cohort == "nongm":
        out = [s for s in out if s["policy_resolves_nongm"]]
    elif cohort is not None:
        raise ValueError(f"unknown cohort {cohort!r}")
    if diagnostic_all:
        out = [s for s in out if s["policy_resolves_all"]]
    return out


def policy_response_table(bank: dict) -> dict[str, dict]:
    """Per-checkpoint counts of states where the argmax responds to coin placement."""
    states = bank["states"]
    n = len(states)
    return {
        lbl: {
            "n_policy_resolves": sum(s["per_ckpt"][lbl]["policy_actions_differ"] for s in states),
            "n_total": n,
            "frac": (sum(s["per_ckpt"][lbl]["policy_actions_differ"] for s in states) / n) if n else float("nan"),
        }
        for lbl in ALL_LABELS
    }
