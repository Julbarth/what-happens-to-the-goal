#!/usr/bin/env python3
"""Per-seed and across-seed (t 95% CI) tables for canonical-bank Attribute Maze steering results."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

GM_BLACK, GM_TEX = ("6313", "6315", "6317"), ("6301", "6302", "6303")
DECO_BLACK, DECO_TEX = ("4371", "4372", "4373"), ("4361", "4362", "4363")
SUPP_LABEL = "black-trained control, texture rendering (supplementary)"
GLOBAL_ARMS = ("baseline", "full_local", "full_local_anti", "full_local_fixed", "full_local_fixed_anti",
               "sign_gated", "sign_gated_anti", "fisher_projected", "fisher_projected_anti",
               "ungated", "ungated_anti", "random")


def cohort_of(bg: str, seed: str) -> str:
    """Primary control = the control trained on that background (black 4371-73, texture 4361-63)."""
    seed = str(seed)
    if seed in DECO_TEX:
        return "deco"
    if seed in DECO_BLACK:
        return "deco" if bg == "black" else "deco_xrender_supp"
    return "GM"


def t_ci(vals: list[float]) -> tuple[float, float, float, int]:
    v = np.asarray([x for x in vals if x == x], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(v.mean())
    if n == 1:
        return m, float("nan"), float("nan"), 1
    h = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def outcome_counts(outs: list[str]) -> dict:
    n = len(outs)
    g, p = outs.count("line"), outs.count("gem")
    to, ot = outs.count("timeout"), outs.count("other_terminal")
    d = (g + p) / n if n else float("nan")
    pg = g / (g + p) if (g + p) else float("nan")
    reach = g / n if n else float("nan")
    return {"n": n, "n_goal": g, "n_proxy": p, "n_timeout": to, "n_other": ot, "reach": reach,
            "proxy_reach": p / n if n else float("nan"), "timeout": to / n if n else float("nan"),
            "D": d, "P_goal_given_dec": pg, "reach_minus_D_P": (reach - d * pg) if pg == pg else 0.0}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def strata(rows: list[dict]):
    """(sign_label, d_label) -> rows; every combination of all/+1/-1 and all/d>1."""
    for sl, sf in (("all", None), ("+1", 1), ("-1", -1)):
        for dl, dmin in (("all", 0), ("d>1", 2)):
            yield sl, dl, [r for r in rows if (sf is None or r["axis_sign"] == sf) and r["d"] >= dmin]


def _is_swap(name: str) -> bool:
    """Edit family from the file-name prefix only (split names may contain 'swap_pair')."""
    return name.startswith(("rows_swap_", "meta_swap_", "ras_global_steering_hfc_swap_"))


def load_local(in_dir: Path, edit: str) -> tuple[list[dict], list[dict]]:
    rows, metas = [], []
    seen = set()
    for fp in sorted((in_dir / "local").glob("rows_*.jsonl")):
        if _is_swap(fp.name) != (edit == "swap"):
            continue
        parsed = []
        for line in fp.read_text().splitlines():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        # keep only committed blocks: rollout rows are written before their state's state_done marker
        pending: dict[str, list[dict]] = defaultdict(list)
        for r in parsed:
            if r["kind"] == "rollout":
                if r["arm"] == "baseline":  # every block starts with its baseline row; drop an uncommitted earlier try
                    pending[r["state_key"]] = []
                pending[r["state_key"]].append(r)
            elif r["kind"] == "state_done":
                block = pending.pop(r["state_key"], [])
                for x in block:
                    key = (x["tap"], x["background"], x["seed"], x["state_key"], x["arm"], x["beta"], x["k"])
                    if key in seen:
                        raise SystemExit(f"duplicate row {key} in {fp}")
                    seen.add(key)
                    rows.append(x)
    for fp in sorted((in_dir / "local").glob("meta_*.json")):
        if _is_swap(fp.name) == (edit == "swap"):
            metas.append(json.loads(fp.read_text()))
    return rows, metas


def local_tables(rows: list[dict], eval_keys: set[str], out: Path) -> None:
    for (tap, bg, seed), grp in sorted(_group(rows, ("tap", "background", "seed")).items()):
        write_csv(out / "local" / f"raw_rows_{tap}_{bg}_{seed}.csv",
                  [{k: r[k] for k in ("tap", "background", "seed", "split_role", "state_key", "state_hash", "level", "d",
                                      "is_d1", "axis_sign", "arm", "beta", "k", "replay_stride", "max_steps",
                                      "outcome", "n_steps", "checkpoint_sha256", "canonical_bank_sha256",
                                      "split_sha256")} for r in grp])
    per_seed = []
    by = _group(rows, ("tap", "background", "seed"))
    for (tap, bg, seed), grp in sorted(by.items()):
        pops = {"eval50": [r for r in grp if r["state_key"] in eval_keys]}
        if len({r["state_key"] for r in grp}) > len(eval_keys):
            pops["all196"] = grp
        for pop, prow in pops.items():
            if not prow:
                continue
            base_by_state = {r["state_key"]: r["outcome"] for r in prow if r["arm"] == "baseline"}
            betas = sorted({r["beta"] for r in prow if r["arm"] != "baseline"})
            for sl, dl, srows in strata(prow):
                states = sorted({r["state_key"] for r in srows})
                if not states:
                    continue
                for arm in ("baseline", "corrective", "anticorrective", "random"):
                    for beta in ([0.0] if arm == "baseline" else betas):
                        sub = [r for r in srows if r["arm"] == arm and r["beta"] == beta]
                        if not sub:
                            continue
                        c = outcome_counts([r["outcome"] for r in sub])
                        # paired per state: arm reach minus baseline reach (random: mean over its k dirs)
                        by_s = defaultdict(list)
                        for r in sub:
                            by_s[r["state_key"]].append(float(r["outcome"] == "line"))
                        dreach = float(np.mean([np.mean(v) - float(base_by_state[s] == "line") for s, v in by_s.items()]))
                        per_seed.append({"tap": tap, "background": bg, "cohort": cohort_of(bg, str(seed)),
                                         "seed": seed, "population": pop, "sign": sl, "d_stratum": dl,
                                         "arm": arm, "beta": beta, "n_states": len(by_s), **c,
                                         "delta_reach_vs_baseline": dreach})
    write_csv(out / "local" / "local_per_seed.csv", per_seed)
    write_csv(out / "local" / "local_cohort_ci.csv", cohort_ci(per_seed, ("tap", "background", "cohort", "population",
                                                                         "sign", "d_stratum", "arm", "beta")))


def cohort_ci(per_seed: list[dict], keys: tuple[str, ...]) -> list[dict]:
    out = []
    for k, grp in sorted(_group(per_seed, keys).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = dict(zip(keys, k))
        row["seeds"] = ",".join(str(g["seed"]) for g in grp)
        row["n_states_per_seed"] = ",".join(str(g["n_states"]) for g in grp)
        for m in ("reach", "proxy_reach", "timeout", "D", "P_goal_given_dec", "delta_reach_vs_baseline"):
            if m not in grp[0]:
                continue
            mean, lo, hi, n = t_ci([g[m] for g in grp])
            vals = np.asarray([g[m] for g in grp if g[m] == g[m]], dtype=float)
            row[f"{m}_mean"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"], row[f"{m}_n_seeds"] = mean, lo, hi, n
            row[f"{m}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            row[f"{m}_per_seed"] = ";".join(f"{g[m]:.3f}" for g in grp)
        out.append(row)
    return out


def _group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def load_global(in_dir: Path, edit: str) -> list[dict]:
    docs = []
    for fp in sorted((in_dir / "global").glob("ras_global_steering_hfc_*.json")):
        if _is_swap(fp.name) != (edit == "swap"):
            continue
        d = json.loads(fp.read_text())
        name = fp.stem.replace("ras_global_steering_hfc_", "").replace("swap_", "", 1)
        seed_tag = name.split("_")[1]
        for k, v in d.items():
            if k.startswith("seed_") and k[5:] in seed_tag.split("-"):
                docs.append({"file": fp.name, "run": name, "seed": k[5:], "doc": v,
                             "prov": d["bank_provenance"], "canonical_bank_sha256": d.get("canonical_bank_sha256"),
                             "split_sha256": d.get("split_sha256")})
    return docs


def global_tables(docs: list[dict], local_rows: list[dict], out: Path) -> None:
    fl = {}
    for r in local_rows:
        if r["tap"] == "h_fc" and r["arm"] in ("corrective", "anticorrective"):
            arm = "full_local" if r["arm"] == "corrective" else "full_local_anti"
            fl[(r["background"], str(r["seed"]), r["state_key"], r["beta"], arm)] = r["outcome"]
    long_rows, gates = [], []
    for dd in docs:
        s = dd["doc"]
        bg = s["background"]
        run = dd["run"]
        split = "primary_exd1" if run.endswith("_exd1") else ("primary" if "state_pair" in run else "loo")
        fold = run.split("fold")[1][:2] if "fold" in run else ""
        gates.append({"run": run, "split": split, "fold": fold, "background": bg, "seed": dd["seed"],
                      "cohort": cohort_of(bg, dd["seed"]), "n_axis_fit": s["n_fit_bank"], "n_holdout": s["n_holdout"],
                      "n_eval": s["n_steering_bank"], "gate_mean_cos": s["gate"]["mean"],
                      "gate_frac_pos": s["gate"]["frac_pos"], "perm_p": s["permutation_null"]["p_upper_bound"],
                      **{f"eval_{k}_{m}": v for k, b in s["eval_alignment_by_bucket"].items() for m, v in b.items()},
                      "replay_stride": s["replay_stride"], "split_sha256": dd["split_sha256"],
                      "canonical_bank_sha256": dd["canonical_bank_sha256"]})
        for r in s["rows"]:
            for arm in GLOBAL_ARMS:
                outs = r[arm] if arm == "random" else [r.get(arm)]
                if arm in ("full_local", "full_local_anti") and outs == [None]:
                    o = fl.get((bg, dd["seed"], r["state_key"], r["beta"], arm))
                    outs = [o] if o is not None else []
                for k, o in enumerate(outs):
                    if o is None:
                        continue
                    long_rows.append({"run": run, "split": split, "fold": fold, "background": bg,
                                      "cohort": cohort_of(bg, dd["seed"]), "seed": dd["seed"],
                                      "state_key": r["state_key"], "state_hash": r["state_hash"], "level": r["level"],
                                      "d": r["d"], "axis_sign": r["axis_sign"], "sigma_bucket": r["sigma_bucket"],
                                      "arm": arm, "beta": r["beta"], "k": k, "outcome": o, "tap": "h_fc",
                                      "replay_stride": r["replay_stride"], "checkpoint_sha256": r["checkpoint_sha256"]})
    write_csv(out / "global" / "global_raw_rows.csv", long_rows)
    write_csv(out / "global" / "global_gates.csv", gates)
    per_seed = []
    pooled = {"primary": lambda r: r["split"] == "primary", "primary_exd1": lambda r: r["split"] == "primary_exd1",
              "loo_pooled": lambda r: r["split"] == "loo"}
    for pop, f in pooled.items():
        sel = [r for r in long_rows if f(r)]
        for (bg, cohort, seed), grp in sorted(_group(sel, ("background", "cohort", "seed")).items()):
            base_by_state = {r["state_key"]: r["outcome"] for r in grp if r["arm"] == "baseline"}
            for sl, dl, srows in strata(grp):
                for (arm, beta), sub in sorted(_group(srows, ("arm", "beta")).items()):
                    by_s = defaultdict(list)
                    for r in sub:
                        by_s[r["state_key"]].append(float(r["outcome"] == "line"))
                    c = outcome_counts([r["outcome"] for r in sub])
                    dreach = float(np.mean([np.mean(v) - float(base_by_state[s] == "line") for s, v in by_s.items()]))
                    per_seed.append({"population": pop, "background": bg, "cohort": cohort, "seed": seed,
                                     "sign": sl, "d_stratum": dl, "arm": arm, "beta": beta,
                                     "n_states": len(by_s), **c, "delta_reach_vs_baseline": dreach})
    loo = [r for r in long_rows if r["split"] == "loo"]
    for (fold, bg, cohort, seed), grp in sorted(_group(loo, ("fold", "background", "cohort", "seed")).items()):
        for (arm, beta), sub in sorted(_group(grp, ("arm", "beta")).items()):
            c = outcome_counts([r["outcome"] for r in sub])
            per_seed.append({"population": f"loo_fold{fold}", "background": bg, "cohort": cohort, "seed": seed,
                             "sign": "all", "d_stratum": "all", "arm": arm, "beta": beta,
                             "n_states": len({r["state_key"] for r in sub}), **c})
    write_csv(out / "global" / "global_per_seed.csv", per_seed)
    write_csv(out / "global" / "global_cohort_ci.csv",
              cohort_ci(per_seed, ("population", "background", "cohort", "sign", "d_stratum", "arm", "beta")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True, help="input directory containing local/ and global/")
    ap.add_argument("--out", type=Path, default=Path("data/probes/canon_steer_20260924"))
    ap.add_argument("--edit", choices=["swap", "ablation"], required=True,
                    help="swap = canonical (paper A.2.2); ablation = supplementary; never pooled")
    ap.add_argument("--split", type=Path,
                    default=Path("data/probes/ras_gt_bank_v2_20260924/splits/ras_gt_bank_v2_split_state_pair_signd1_seed0.json"))
    args = ap.parse_args()
    split = json.loads(args.split.read_text())
    eval_keys = {e["state_key"] for e in split["eval"]}
    local_rows, metas = load_local(args.in_dir, args.edit)
    for m in metas:
        assert m["canonical_bank_sha256"] == "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9", m
        assert m["split_sha256"] == split["split_sha256"], m["split_sha256"]
    write_csv(args.out / "local" / "local_meta_index.csv",
              [{"tap": m["tap"], "background": m["background"], "seed": m["seed"],
                "role": m["bank_provenance"]["split_role"], "shard": m["shard"], "n_shards": m["n_shards"],
                "checkpoint": m["checkpoint"], "checkpoint_sha256": m["checkpoint_sha256"],
                "replay_stride": m["replay_stride"], "max_steps": m["max_steps"], "betas": m["betas"],
                "n_states_used": m["bank_provenance"]["n_states_used"],
                "determinism_ok": m["determinism_check"]["ok"], "hook_ok": m["hook_verification"]["matches"],
                "source_sha256": json.dumps(m["source_sha256"], sort_keys=True)} for m in metas])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "PROVENANCE.json").write_text(json.dumps({
        "canonical_bank_sha256": "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9",
        "primary_split_sha256": split["split_sha256"],
        "edit_family_in_this_dir": args.edit,
        "counterfactual_edits_swap_canonical": {
            "shape_swap": "obj1 red_gem at u, obj2 yellow_line_diag at v (yellow line appears at v)",
            "colour_swap": "obj1 yellow_line_diag at u, obj2 red_gem at v (yellow line appears at u)",
            "d_comp": "h(colour_swap) - h(shape_swap), pointing toward the line candidate u",
        },
        "counterfactual_edits_ablation_supplementary": {
            "shape_edit": "line -> red_gem (obj1 red_line_diag replaced; line shape ablated, colour kept)",
            "colour_edit": "yellow_gem -> green_gem (obj2 replaced; yellow ablated, gem shape kept)",
            "d_comp": "h(colour-edited) - h(shape-edited), pointing toward the line",
            "note": "supplementary deletion assay; never mixed with swap numbers",
        },
        "outcome_rule": "reward +10 (obj1, line) = goal; -10 (obj2, gem) = proxy; timeout or other_terminal = non-decisive",
        "cohort_labels": {"deco": "control trained on the evaluated background (black 4371-73, texture 4361-63)",
                          "deco_xrender_supp": SUPP_LABEL},
        "stats": "unit = unique state; per-seed rate; cohort = across-seed mean, t 95% CI, n = number of seeds",
    }, indent=2))
    if local_rows:
        local_tables(local_rows, eval_keys, args.out)
    docs = load_global(args.in_dir, args.edit)
    if docs:
        global_tables(docs, local_rows, args.out)


if __name__ == "__main__":
    main()
