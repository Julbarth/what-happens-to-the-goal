"""Adapter from the Fisher producer's per-state counterfactual rows to per-(cohort, seed, background, tap) cells."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reorder(cell: dict, keys: list[str], src: str) -> dict:
    """Put rows in state-table order; any missing, extra or duplicated state aborts."""
    got = list(cell["state_key"])
    if len(set(got)) != len(got):
        raise SystemExit(f"{src}: duplicated state_key rows")
    missing = set(keys) - set(got)
    extra = set(got) - set(keys)
    if missing or extra:
        raise SystemExit(f"{src}: rows inconsistent with bank ({len(missing)} missing, {len(extra)} extra)")
    pos = {k: i for i, k in enumerate(got)}
    order = np.array([pos[k] for k in keys])
    out = dict(cell)
    for f in ("delta_shape", "delta_colour", "p"):
        out[f] = np.asarray(cell[f], dtype=np.float64)[order]
    J = np.asarray(cell["J"], dtype=np.float64)
    out["J"] = J if J.ndim == 2 else J[order]
    out["state_key"] = keys
    return out


FROZEN = "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9"


def _rows_meta(rows_csv: Path) -> dict:
    """Provenance from the rows CSV: per-state A sets and axis_sign, plus the (constant) checkpoint fields."""
    import csv
    per_state, const = {}, {}
    with open(rows_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            per_state[r["state_key"]] = {"A_line": [int(x) for x in r["A_line"].split()],
                                         "A_gem": [int(x) for x in r["A_gem"].split()],
                                         "axis_sign": int(r["axis_sign"]), "split_role": r["split_role"],
                                         "loo_fold": int(r["loo_fold"])}
            for k in ("ckpt_path", "ckpt_sha256", "canonical_bank_sha256", "code_sha256"):
                const.setdefault(k, set()).add(r[k])
    for k, v in const.items():
        if len(v) != 1:
            raise SystemExit(f"{rows_csv}: {k} not constant: {v}")
    return {"per_state": per_state, **{k: next(iter(v)) for k, v in const.items()}}


def iter_geom_cells(fisher_dir: Path, keys: list[str], edits: str = "swap", expected_bank_sha: str = FROZEN):
    """Yield cells from run_ras_canon_fisher.py outputs: swap edits (geom_swap/, paper A.2.2 Eqs. 13-14) or
    ablation edits (geom/), with A sets and checkpoint provenance from rows/ and J at h_pre = W_pi diag(1[h_fc > 0]) W_fc."""
    import csv
    fisher_dir = Path(fisher_dir)
    sub, prefix = ("geom_swap", "geom_swap_") if edits == "swap" else ("geom", "geom_")
    root = fisher_dir.parent.parent.parent
    files = sorted((fisher_dir / sub).glob(f"{prefix}*.npz"))
    if not files:
        raise SystemExit(f"no {sub}/{prefix}*.npz in {fisher_dir}")
    for npz in files:
        bg, cohort, seed = npz.stem[len(prefix):].split("_")
        rows_csv = fisher_dir / "rows" / f"rows_{bg}_{cohort}_{seed}.csv"
        meta = _rows_meta(rows_csv)
        if meta["canonical_bank_sha256"] != expected_bank_sha:
            raise SystemExit(f"{rows_csv}: canonical_bank_sha256 {meta['canonical_bank_sha256']} not frozen")
        z = np.load(npz, allow_pickle=False)
        W_pi, W_fc = z["W_pi"].astype(np.float64), z["W_fc"].astype(np.float64)
        err = float(np.abs(z["J_h_fc"].astype(np.float64) - W_pi[None]).max())
        if err > 1e-5:
            raise SystemExit(f"{npz}: stored J_h_fc deviates from W_pi by {err}")
        sk = [str(k) for k in z["state_key"]]
        if set(sk) != set(meta["per_state"]):
            raise SystemExit(f"{npz}: geom and rows CSV state sets differ")
        mask = z["h_fc_active_mask"].astype(np.float64)
        srcs = {str(npz.relative_to(root)): _sha(npz), str(rows_csv.relative_to(root)): _sha(rows_csv)}
        producer = {}
        if edits == "swap":
            pair_csv = fisher_dir / "rows_swap_pair" / f"rows_swap_pair_{bg}_{cohort}_{seed}.csv"
            srcs[str(pair_csv.relative_to(root))] = _sha(pair_csv)
            with open(pair_csv, newline="") as fh:
                for r in csv.DictReader(fh):
                    producer[(r["state_key"], r["tap"])] = (float(r["cos_F_swap"]), float(r["cos_E_swap"]))
        for tap in ("h_pre", "h_fc"):
            J = W_pi if tap == "h_fc" else np.einsum("ak,nk,kd->nad", W_pi, mask, W_fc)
            if edits == "swap":
                ds = z[f"delta_shape_swap_{tap}"].astype(np.float64)
                dc = z[f"delta_colour_swap_{tap}"].astype(np.float64)
            else:
                hb = z[f"h_base_{tap}"].astype(np.float64)
                ds = z[f"h_shape_{tap}"].astype(np.float64) - hb
                dc = z[f"h_colour_{tap}"].astype(np.float64) - hb
            cell = {"state_key": sk, "delta_shape": ds, "delta_colour": dc, "edits": edits,
                    "J": J, "p": z["p"], "cohort": cohort, "seed": seed, "background": bg, "tap": tap,
                    "checkpoint": meta["ckpt_path"], "checkpoint_sha256": meta["ckpt_sha256"],
                    "producer_code_sha256": meta["code_sha256"], "rows_meta": meta["per_state"],
                    "source_files": srcs}
            if producer:
                cell["producer_cos"] = {k: producer[(k, tap)] for k in sk}
            yield _reorder(cell, keys, str(npz))


def iter_cells(fisher_dir: Path, keys: list[str], edits: str = "swap", expected_bank_sha: str = FROZEN):
    """Dispatch on format: a geom/ subdir goes to iter_geom_cells; otherwise one `<tag>.npz` + `<tag>.json` per cell
    (J is [n, A, D], or [A, D] when constant across states)."""
    fisher_dir = Path(fisher_dir)
    if (fisher_dir / "geom").is_dir() or (fisher_dir / "geom_swap").is_dir():
        yield from iter_geom_cells(fisher_dir, keys, edits, expected_bank_sha)
        return
    for npz in sorted(fisher_dir.glob("*.npz")):
        meta_path = npz.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        z = np.load(npz, allow_pickle=False)
        cell = {"state_key": [str(k) for k in z["state_key"]], "delta_shape": z["delta_shape"],
                "delta_colour": z["delta_colour"], "J": z["J"], "p": z["p"],
                "cohort": meta["cohort"], "seed": str(meta["seed"]), "background": meta["background"],
                "tap": meta["tap"], "checkpoint": meta.get("checkpoint", ""),
                "source_files": {str(npz): _sha(npz), str(meta_path): _sha(meta_path)}}
        if meta.get("canonical_bank_sha256") not in (None, "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9"):
            raise SystemExit(f"{npz}: canonical_bank_sha256 {meta.get('canonical_bank_sha256')} is not the frozen bank")
        yield _reorder(cell, keys, str(npz))
