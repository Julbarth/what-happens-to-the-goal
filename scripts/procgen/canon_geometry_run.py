"""Canonical Attribute-Maze cue/competition geometry (Tables 11-13, Figs 11-14 data) from per-state Fisher rows."""
from __future__ import annotations

import os
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import canon_geometry_core as cg  # noqa: E402
import canon_geometry_io as gio  # noqa: E402
from gm_geometry.probes import ras_bank_io  # noqa: E402

DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
BANK_DIR = DATA_ROOT / "probes/ras_gt_bank_v2_20260924"
FROZEN_CANONICAL_SHA = "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9"
PRIMARY_SPLIT_FILE = "ras_gt_bank_v2_split_state_pair_signd1_seed0.json"
PRIMARY_SPLIT_SHA = "61e5e52beec86d59e6aa5b282fb49cc056f05ffd47f3a4d97f4d8d1c6af0e3ab"
LOO_SPLIT_FILES = tuple(f"ras_gt_bank_v2_split_layout_loo_fold{k:02d}.json" for k in range(11))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_state_table(bank_path: Path, bank_format: str = "canonical", expected_sha: str = FROZEN_CANONICAL_SHA) -> dict:
    """Bank states in a fixed order (sign desc, layout, d, state_key) with every covariate the analysis uses.

    bank_format='divbank' reads the diversified-start bank (ras_divbank.json): 'sign' is its pair_sign (orientation
    of the line/gem contrast within its unordered action-group pair; a mirror has the opposite sign), the line
    bearing is the 4-way cardinal bearing of the line from the agent, and 'primary' masks the strict conflicts."""
    if bank_format == "divbank":
        doc = json.loads(Path(bank_path).read_text())
        doc["canonical_bank_sha256"] = doc["divbank_sha256"]
        for r in doc["records"]:
            r["axis_sign_lineN_gemE"] = r["pair_sign"]
            q = r["bearing_line"]["quad4"]
            r["bearing"] = list({"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}[q])
    else:
        doc = ras_bank_io.load_bank_doc(bank_path, verify=True)
    if doc["canonical_bank_sha256"] != expected_sha:
        raise SystemExit(f"bank sha {doc['canonical_bank_sha256']} != frozen {expected_sha}")
    recs = doc["records"]
    for r in recs:
        ras_bank_io.state_key(dict(r, grid_w=r.get("grid_w", 5), grid_h=r.get("grid_h", 5)))
    layouts = sorted({ras_bank_io.layout_key(r) for r in recs})
    lay_id = {l: i for i, l in enumerate(layouts)}
    recs = sorted(recs, key=lambda r: (-r["axis_sign_lineN_gemE"], lay_id[ras_bank_io.layout_key(r)],
                                       r["d_line"], r["state_key"]))
    keys = [r["state_key"] for r in recs]
    kidx = {k: i for i, k in enumerate(keys)}
    partner = np.full(len(recs), -1)
    for i, r in enumerate(recs):
        sw = dict(r, line=r["gem"], gem=r["line"], grid_w=5, grid_h=5)
        sw.pop("state_key")
        partner[i] = kidx.get(ras_bank_io.state_key(sw), -1)
    if (partner < 0).any():
        raise SystemExit("bank not closed under line/gem swap")
    out = {
        "state_key": keys,
        "sign": np.array([int(r["axis_sign_lineN_gemE"]) for r in recs]),
        "d": np.array([int(r["d_line"]) for r in recs]),
        "layout": np.array([lay_id[ras_bank_io.layout_key(r)] for r in recs]),
        "line": np.array([r["line"] for r in recs]),
        "gem": np.array([r["gem"] for r in recs]),
        "line_bearing": np.array([r["bearing"][:2] for r in recs]),
        "A_line": [list(r["A_line"]) for r in recs],
        "A_gem": [list(r["A_gem"]) for r in recs],
        "partner": partner,
        "n_layouts": len(layouts),
        "canonical_bank_sha256": doc["canonical_bank_sha256"],
    }
    if bank_format == "divbank":
        out["primary"] = np.array([r["set"] == "strict" for r in recs])
    return out


def load_splits(split_dir: Path, keys: list[str], partner: np.ndarray | None = None,
                primary_file: str = PRIMARY_SPLIT_FILE, loo_files: tuple = LOO_SPLIT_FILES,
                expected_sha: str = FROZEN_CANONICAL_SHA, primary_sha: str | None = PRIMARY_SPLIT_SHA,
                expected_n: tuple[int, int] | None = (146, 50)) -> dict:
    """Primary + LOO splits as boolean masks in state-table order, with leak checks (overlap, partition, and each
    state's line/gem mirror in the same role, since a mirror is the same geometry with d_comp negated)."""
    kidx = {k: i for i, k in enumerate(keys)}
    out = {}
    files = [primary_file] + list(loo_files)
    for f in files:
        sp = json.loads((split_dir / f).read_text()) if expected_sha != FROZEN_CANONICAL_SHA else \
            ras_bank_io.load_split(split_dir / f)
        if sp.get("canonical_bank_sha256", sp.get("divbank_sha256")) != expected_sha:
            raise SystemExit(f"{f}: split built on another bank")
        fit = np.zeros(len(keys), bool)
        ev = np.zeros(len(keys), bool)
        for e in sp["fit"]:
            fit[kidx[e["state_key"]]] = True
        for e in sp["eval"]:
            ev[kidx[e["state_key"]]] = True
        assert not (fit & ev).any(), f"{f}: fit/eval overlap"
        if partner is not None:
            assert (fit == fit[partner]).all() and (ev == ev[partner]).all(), f"{f}: mirror pair split across roles"
        out[f] = {"fit": fit, "eval": ev, "split_sha256": sp["split_sha256"],
                  "file_sha256": sha256_file(split_dir / f)}
    prim = out[files[0]]
    if primary_sha is not None:
        assert prim["split_sha256"] == primary_sha
    if expected_n is not None:
        assert prim["fit"].sum() == expected_n[0] and prim["eval"].sum() == expected_n[1]
    loo_eval = np.sum([out[f]["eval"] for f in files[1:]], axis=0)
    assert (loo_eval == 1).all(), "LOO folds must partition the bank"
    return out


def check_rows_meta(cell: dict, st: dict) -> None:
    """Stop rule: the Fisher rows' action sets and sign must equal the bank's for every state."""
    rm = cell.get("rows_meta")
    if rm is None:
        return
    for i, k in enumerate(st["state_key"]):
        r = rm[k]
        if (sorted(r["A_line"]) != sorted(st["A_line"][i]) or sorted(r["A_gem"]) != sorted(st["A_gem"][i])
                or r["axis_sign"] != st["sign"][i]):
            raise SystemExit(f"STOP {cell['cohort']}_{cell['seed']} {k[:16]}: rows disagree with bank")


def analyze_cell(cell: dict, st: dict, splits: dict, args) -> tuple[dict, list[dict], dict]:
    """All geometry statistics for one (cohort, seed, background, tap)."""
    ds, dc, J, p = cell["delta_shape"], cell["delta_colour"], cell["J"], cell["p"]
    dcomp = dc - ds
    n = len(st["state_key"])
    ws = cg.within_state(ds, dc, J, p)
    frac_comp = cg.logit_contrast_fraction(J, p, dcomp, st["A_line"], st["A_gem"])
    frac_s = cg.logit_contrast_fraction(J, p, ds, st["A_line"], st["A_gem"])
    frac_c = cg.logit_contrast_fraction(J, p, dc, st["A_line"], st["A_gem"])
    g_comp = cg.signed_contrast(J, p, dcomp, st["A_line"], st["A_gem"])
    ke, kf = cg.cross_state(dcomp, J, p)
    rel = cg.relation_matrix(st["line_bearing"])
    Z = cg.spatial_descriptor(st["line"], st["gem"], st["d"])
    S = cg.spatial_similarity(Z)
    Srf = cg.spatial_similarity(Z, role_free=True)

    rows = []
    for i in range(n):
        rows.append({"state_key": st["state_key"][i], "sign": int(st["sign"][i]), "d": int(st["d"][i]),
                     "layout": int(st["layout"][i]), "cos_e": ws["cos_e"][i], "cos_f": ws["cos_f"][i],
                     "q_shape": ws["q_shape"][i], "q_colour": ws["q_colour"][i],
                     "contrast_frac_dcomp": frac_comp[i], "contrast_frac_shape": frac_s[i],
                     "contrast_frac_colour": frac_c[i], "signed_contrast_dcomp": g_comp[i],
                     "norm_dcomp": float(np.linalg.norm(dcomp[i]))})

    base = st.get("primary", np.ones(n, bool))
    masks = {"all": base, "d_gt_1": base & (st["d"] > 1)}
    summ = {}
    for mname, m in masks.items():
        s = {}
        for stat, arr in (("cos_e", ws["cos_e"]), ("cos_f", ws["cos_f"]), ("contrast_frac_dcomp", frac_comp),
                          ("contrast_frac_shape", frac_s), ("contrast_frac_colour", frac_c)):
            s[stat] = float(np.nanmean(arr[m]))
            for sg in (1, -1):
                s[f"{stat}_sign{sg:+d}"] = float(np.nanmean(arr[m & (st["sign"] == sg)]))
        s["frac_dcomp_favours_line"] = float((g_comp[m] > 0).mean())
        # direction of d_comp's push on the UP-vs-RIGHT contrast; flips to -sign where d_comp favours the gem
        eff = np.where(g_comp > 0, 1, -1) * st["sign"]
        s["frac_effective_sign_flipped"] = float((eff != st["sign"])[m].mean())
        for kname, K in (("E", ke), ("F", kf)):
            s[f"block_{kname}"] = cg.block_means(K, rel, m)
            s[f"signgap_{kname}"] = cg.sign_gap(K, st["sign"], m)
            s[f"vardec_effsign_{kname}"] = cg.variance_decomposition(K, eff, st["d"], st["layout"], S,
                                                                     st["partner"], m, n_perm=0, spatial_rf=Srf)
            s[f"vardec_{kname}"] = cg.variance_decomposition(K, st["sign"], st["d"], st["layout"], S,
                                                             st["partner"], m, n_perm=args.n_perm,
                                                             seed=args.perm_seed, spatial_rf=Srf)
            r1 = cg.rank1_offdiag_r2(K, m)
            s[f"rank1_{kname}"] = {"r2_rank1": r1["r2_rank1"],
                                   "loading_sign_agreement": float(max(
                                       np.mean(np.sign(r1["loadings"]) == st["sign"][r1["states"]]),
                                       np.mean(np.sign(r1["loadings"]) == -st["sign"][r1["states"]])))}
            s[f"mirror_{kname}"] = cg.mirror_antisymmetry(K, st["partner"])
            s[f"within_sign_{kname}"] = cg.within_sign_fine(
                K, st["sign"], st["d"], st["layout"], S, m, n_perm=args.n_perm, seed=args.perm_seed,
                fit_mask=splits[args.primary_split_file]["fit"] & m, eval_mask=splits[args.primary_split_file]["eval"] & m,
                partner=st["partner"])
            s[f"within_sign_{kname}"]["r2_fine_heldout_loo"] = float(np.nanmean([
                cg.within_sign_fine(K, st["sign"], st["d"], st["layout"], S, m, n_perm=0,
                                    fit_mask=sp["fit"] & m, eval_mask=sp["eval"] & m)["r2_fine_heldout"]
                for f, sp in splits.items() if f != args.primary_split_file]))
            s[f"rank1_within_sign_{kname}"] = cg.rank1_within_sign(K, st["sign"], m)
            s[f"spatialcorr_{kname}"] = cg.within_relation_corr(K, S, rel, m)
            s[f"spatialcorr_rf_{kname}"] = cg.within_relation_corr(K, Srf, rel, m)
        s["lowrank"] = cg.lowrank_null(dcomp, J, p, n_draws=args.n_null, seed=args.null_seed, mask=m)
        held = {}
        for sname, sp in splits.items():
            fit, ev = sp["fit"] & m, sp["eval"] & m
            u = cg.shared_axis(dcomp, st["sign"], np.flatnonzero(fit))
            ae, af = cg.axis_loadings(dcomp, u, J, p)
            h = {"split_sha256": sp["split_sha256"],
                 "axis_cos_e_eval": float(np.nanmean((st["sign"] * ae)[ev])),
                 "axis_cos_f_eval": float(np.nanmean((st["sign"] * af)[ev])),
                 "axis_sign_acc_eval": float(np.mean(np.sign(af[ev]) == st["sign"][ev]))}
            for kname, K, load in (("E", ke, ae), ("F", kf, af)):
                h[kname] = cg.heldout_eval(K, st["sign"], st["d"], st["layout"], S, st["partner"], fit, ev,
                                           loadings=load, spatial_rf=Srf, mask=m)
            held[sname] = h
        s["heldout"] = held
        summ[mname] = s

    mats = {"kappa_e": ke.astype(np.float32), "kappa_f": kf.astype(np.float32),
            "spatial_sim": S.astype(np.float32), "spatial_sim_rf": Srf.astype(np.float32)}
    return summ, rows, mats


def write_pairs_csv(path: Path, st: dict, mats: dict, meta: dict) -> None:
    n = len(st["state_key"])
    iu, ju = np.triu_indices(n, k=1)
    rel = cg.relation_matrix(st["line_bearing"])
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "seed", "background", "tap", "state_key_i", "state_key_j", "sign_i", "sign_j",
                    "d_i", "d_j", "layout_i", "layout_j", "relation", "swap_partner", "spatial_sim",
                    "kappa_e", "kappa_f"])
        for i, j in zip(iu, ju):
            w.writerow([meta["cohort"], meta["seed"], meta["background"], meta["tap"], st["state_key"][i],
                        st["state_key"][j], st["sign"][i], st["sign"][j], st["d"][i], st["d"][j],
                        st["layout"][i], st["layout"][j], rel[i, j], int(st["partner"][i] == j),
                        f"{mats['spatial_sim'][i, j]:.6g}", f"{mats['kappa_e'][i, j]:.6g}",
                        f"{mats['kappa_f'][i, j]:.6g}"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fisher-dir", type=Path, default=DATA_ROOT / "probes/canon_fisher_20260924")
    ap.add_argument("--bank", type=Path, default=BANK_DIR / "ras_gt_bank_canonical.json")
    ap.add_argument("--split-dir", type=Path, default=BANK_DIR / "splits")
    ap.add_argument("--out", type=Path, default=None,
                    help="default data/probes/canon_geometry_20260924/{swap,ablation_supplementary}")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=10)
    ap.add_argument("--null-seed", type=int, default=0)
    ap.add_argument("--edits", choices=["swap", "ablation"], default="swap",
                    help="swap = paper A.2.2 Eq. 13/14 (primary); ablation = supplementary deletion assay")
    ap.add_argument("--allow-not-ready", action="store_true", help="skip the ROWS_READY.json gate (tests only)")
    ap.add_argument("--bank-format", choices=["canonical", "divbank"], default="canonical")
    ap.add_argument("--expected-bank-sha", default=FROZEN_CANONICAL_SHA)
    ap.add_argument("--primary-split-file", default=PRIMARY_SPLIT_FILE)
    ap.add_argument("--primary-split-sha", default=PRIMARY_SPLIT_SHA, help="'none' skips the check")
    ap.add_argument("--loo-split-glob", default=None, help="default: the 11 canonical layout folds")
    ap.add_argument("--expected-n-fit-eval", default="146,50", help="'none' skips the check")
    args = ap.parse_args()
    loo_files = LOO_SPLIT_FILES if args.loo_split_glob is None else tuple(
        sorted(p.name for p in args.split_dir.glob(args.loo_split_glob)))
    primary_sha = None if args.primary_split_sha == "none" else args.primary_split_sha
    expected_n = None if args.expected_n_fit_eval == "none" else tuple(int(x) for x in args.expected_n_fit_eval.split(","))
    if args.out is None:
        args.out = DATA_ROOT / "probes/canon_geometry_20260924" / (
            "swap" if args.edits == "swap" else "ablation_supplementary")

    marker = args.fisher_dir / ("ROWS_READY_SWAP.json" if args.edits == "swap" else "ROWS_READY.json")
    if not args.allow_not_ready:
        if not marker.exists():
            raise SystemExit(f"{marker} missing: Fisher rows not final")
        listed = {}
        for mk in ("ROWS_READY.json", "ROWS_READY_SWAP.json"):
            if (args.fisher_dir / mk).exists():
                for f, h in json.loads((args.fisher_dir / mk).read_text()).get("files", {}).items():
                    if f.split("/")[0] in ("geom", "geom_swap", "rows", "rows_swap_pair"):
                        listed.setdefault(f, set()).add(h)
        consumed = ["rows"] + (["geom_swap", "rows_swap_pair"] if args.edits == "swap" else ["geom"])
        for d in consumed:
            for fp in sorted((args.fisher_dir / d).iterdir()):
                rel = f"{d}/{fp.name}"
                if rel not in listed or sha256_file(fp) not in listed[rel]:
                    raise SystemExit(f"STOP {rel} is not listed with this sha256 in the ROWS_READY markers")
    st = load_state_table(args.bank, args.bank_format, args.expected_bank_sha)
    splits = load_splits(args.split_dir, st["state_key"], st["partner"], args.primary_split_file, loo_files,
                         args.expected_bank_sha, primary_sha, expected_n)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "per_seed").mkdir(exist_ok=True)

    prov = {"canonical_bank_sha256": st["canonical_bank_sha256"], "bank_file": str(args.bank.relative_to(ROOT)),
            "bank_file_sha256": sha256_file(args.bank),
            "splits": {k: {"split_sha256": v["split_sha256"], "file_sha256": v["file_sha256"]}
                       for k, v in splits.items()},
            "primary_split": args.primary_split_file,
            "code_sha256": {p: sha256_file(ROOT / p) for p in (
                "scripts/canon_geometry_core.py", "scripts/canon_geometry_run.py", "scripts/canon_geometry_io.py",
                "src/gm_geometry/probes/ras_bank_io.py")},
            "edits": args.edits,
            "d_comp_convention": ("D_COMP = Delta_colour - Delta_shape = h(colour-edited) - h(shape-edited) "
                                  "(gm_geometry.probes.ras_counterfactuals). For swaps it points toward the line "
                                  "cell u. Every cos_E/cos_F/kappa is invariant to a global sign flip of d_comp; only "
                                  "signed_contrast_dcomp / frac_dcomp_favours_line depend on it."),
            "state_order": st["state_key"], "n_perm": args.n_perm, "n_null": args.n_null,
            "cells": []}
    summaries = []
    for cell in gio.iter_cells(args.fisher_dir, st["state_key"], args.edits, args.expected_bank_sha):
        check_rows_meta(cell, st)
        if "producer_cos" in cell:
            ws_chk = cg.within_state(cell["delta_shape"], cell["delta_colour"], cell["J"], cell["p"])
            pc = np.array([cell["producer_cos"][k] for k in st["state_key"]])
            err = max(float(np.nanmax(np.abs(ws_chk["cos_f"] - pc[:, 0]))),
                      float(np.nanmax(np.abs(ws_chk["cos_e"] - pc[:, 1]))))
            if err > 1e-4:
                raise SystemExit(f"STOP {cell['cohort']}_{cell['seed']} {cell['tap']}: within-state cosines differ "
                                 f"from producer rows_swap_pair by {err}")
            cell["producer_cos_max_abs_diff"] = err
        meta = {k: cell.get(k, "") for k in ("cohort", "seed", "background", "tap", "checkpoint", "checkpoint_sha256", "producer_code_sha256", "edits",
                                             "producer_cos_max_abs_diff")}
        tag = f"{meta['cohort']}_{meta['seed']}_{meta['background']}_{meta['tap']}"
        print("analyzing", tag, flush=True)
        summ, rows, mats = analyze_cell(cell, st, splits, args)
        with open(args.out / "per_seed" / f"within_state_{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(meta) + list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({**meta, **r})
        write_pairs_csv(args.out / "per_seed" / f"pairs_{tag}.csv.gz", st, mats, meta)
        np.savez_compressed(args.out / "per_seed" / f"matrices_{tag}.npz", state_key=np.array(st["state_key"]),
                            **mats)
        summaries.append({**meta, "summary": summ})
        prov["cells"].append({**meta, "source_files": cell["source_files"]})
        with open(args.out / "per_seed" / f"summary_{tag}.json", "w") as fh:
            json.dump({**meta, "provenance": {"canonical_bank_sha256": st["canonical_bank_sha256"],
                                              "primary_split_sha256": splits[args.primary_split_file]["split_sha256"],
                                              "source_files": cell["source_files"]},
                       "summary": summ}, fh, indent=1, default=float)
    with open(args.out / "summaries.json", "w") as fh:
        json.dump(summaries, fh, indent=1, default=float)
    with open(args.out / "provenance.json", "w") as fh:
        json.dump(prov, fh, indent=1)
    print(f"wrote {len(summaries)} cells to {args.out}")


if __name__ == "__main__":
    main()
