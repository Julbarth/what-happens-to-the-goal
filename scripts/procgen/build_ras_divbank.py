#!/usr/bin/env python3
"""Enumerate, balance and split the diversified-start Attribute-Maze bank from a level scan (no procgen needed)."""
from __future__ import annotations

import os
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from gm_geometry.probes.ras_divbank import GROUPS, covariates  # noqa: E402
from gm_geometry.probes.ras_gt_bank import bfs, geometry, state_key  # noqa: E402

WORLD = 5
SPAWN = (0, 0)
REQUIRED_CHECKS = (
    "observe_after_set_state_equals_reset_obs", "state_roundtrip_bytes", "parsed_agent_equals_target",
    "grid_unchanged", "objects_unchanged_get_object_pos", "pixel_diff_confined_to_old_and_new_agent_cell",
    "sprite_appears_at_target_and_leaves_spawn", "action_reaches_target_as_bfs",
    "action_successor_equals_engine_deltas", "natural_walk_arrives", "placed_frame_equals_naturally_reached_frame",
    "placed_and_natural_identical_under_all_15_actions")


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def placement_gate(r1: Path) -> dict:
    """STOP unless every placement check ran and none failed, on both backgrounds and both validation runs."""
    summary = {}
    files = sorted(r1.glob("placement_validation_*.json"))
    if not files:
        raise SystemExit("STOP no placement validation output")
    for f in files:
        doc = json.loads(f.read_text())
        for bg, v in doc["backgrounds"].items():
            st = v["stats"]
            for c in REQUIRED_CHECKS:
                if st.get(f"{c}_n", 0) == 0:
                    raise SystemExit(f"STOP {f.name} {bg}: check {c} never ran")
                if st.get(f"{c}_fail", 0):
                    raise SystemExit(f"STOP {f.name} {bg}: {c} failed {st[f'{c}_fail']}x: {v['fails'][:3]}")
            summary[f"{f.stem}_{bg}"] = {c: st[f"{c}_n"] for c in REQUIRED_CHECKS} | {
                "levels": st["levels"], "byte_diff_patterns": v["byte_diff_patterns"]}
    return summary


def load_scan(r1: Path, bg: str) -> dict[int, dict]:
    rows = {}
    for f in sorted(r1.glob(f"scan_{bg}_*.json")):
        for r in json.loads(f.read_text()):
            if r["level"] in rows:
                raise SystemExit(f"STOP duplicate level {r['level']} in {bg} scan")
            rows[r["level"]] = r
    return rows


def enumerate_states(configs: dict) -> tuple[list[dict], Counter]:
    """Every free agent cell of every (walls, line, gem) config with both targets reachable; strict and relaxed flags."""
    out, counts = [], Counter()
    for (wk, line, gem), cfg in configs.items():
        walls = {tuple(w) for w in cfg["walls"]}
        walkable = {(x, y) for x in range(WORLD) for y in range(WORLD)} - walls
        spawn_dist = bfs(walkable - {line, gem}, SPAWN)
        for c in sorted(walkable - {line, gem}):
            counts["candidate_cells"] += 1
            geo = geometry(c, line, gem, walkable)
            if geo["d_line"] is None or geo["d_gem"] is None:
                counts["target_unreachable"] += 1
                continue
            strict = geo["conflict"]
            relaxed = bool(geo["disjoint"] and not geo["equidistant"])
            counts["both_reachable"] += 1
            counts["strict_conflict"] += strict
            counts["relaxed_nonequidistant_disjoint"] += relaxed
            counts["equidistant_tie_overlapping"] += bool(geo["equidistant"] and not geo["disjoint"])
            if not (strict or relaxed):
                continue
            rec = {"agent": list(c), "line": list(line), "gem": list(gem), "walls": sorted([list(w) for w in walls]),
                   "grid_w": WORLD, "grid_h": WORLD, "walls_key": wk}
            rec["state_key"] = state_key(rec)
            rec.update({k: geo[k] for k in ("d_line", "d_gem", "A_line", "A_gem", "dirs_line", "dirs_gem",
                                            "equidistant", "disjoint", "conflict", "path_line", "path_gem")})
            # primary = strict and reachable from the (0,0) spawn without touching a target; the rest is labelled
            rec["set"] = ("strict" if c in spawn_dist else "strict_unreachable") if strict else "relaxed"
            rec["representative_level"] = cfg["levels"][0]
            rec["config_levels"] = cfg["levels"]
            rec["is_canonical_spawn_state"] = c == SPAWN
            rec["spawn_dist"] = spawn_dist.get(c)
            rec["reachable_from_spawn_in_episode"] = c in spawn_dist
            mirror = dict(rec, line=rec["gem"], gem=rec["line"])
            mirror.pop("state_key")
            rec["mirror_state_key"] = state_key(mirror)
            rec.update(covariates(c, line, gem, geo))
            out.append(rec)
    return out, counts


def greedy_balance(units: list[dict], n: int, rng: np.random.Generator, keys=("agent", "pair", "quad", "layout")) -> list[dict]:
    """Pick n units one at a time, each minimising the summed current marginal counts of its strata (ties random)."""
    counts = {k: Counter() for k in keys}
    pool = list(units)
    rng.shuffle(pool)
    chosen = []
    while pool and len(chosen) < n:
        scores = np.array([sum(counts[k][u[k]] for k in keys) for u in pool], dtype=float)
        best = np.flatnonzero(scores == scores.min())
        u = pool.pop(int(best[0]))
        for k in keys:
            counts[k][u[k]] += 1
        chosen.append(u)
    return chosen


def make_units(states: list[dict]) -> list[dict]:
    by_key = {s["state_key"]: s for s in states}
    units, seen = [], set()
    for s in sorted(states, key=lambda r: r["state_key"]):
        if s["state_key"] in seen or s["mirror_state_key"] not in by_key:
            continue
        m = by_key[s["mirror_state_key"]]
        seen |= {s["state_key"], m["state_key"]}
        units.append({"members": [s["state_key"], m["state_key"]], "agent": str(s["agent"]),
                      "pair": s["conflict_pair"], "quad": s["quad_pair"], "layout": s["walls_key"][:12],
                      "decisive": s["disagree_decisive"], "reversed": s["reversed"],
                      "disagree_any": s["bearing_action_disagrees"]})
        assert s["disagree_decisive"] == m["disagree_decisive"] and s["reversed"] == m["reversed"]
        assert s["conflict_pair"] == m["conflict_pair"] and s["quad_pair"] == m["quad_pair"]
    return units


def split_units(units: list[dict], eval_frac: float, rng: np.random.Generator) -> dict[str, str]:
    """Unit (mirror pair) -> 'fit'/'eval', stratified by (decisive, reversed, conflict pair)."""
    strata = defaultdict(list)
    for u in units:
        strata[(u["decisive"], u["reversed"], u["pair"])].append(u)
    role, carry = {}, 0.0
    for key in sorted(strata, key=str):
        grp = strata[key]
        rng.shuffle(grp)
        carry += eval_frac * len(grp)
        k = int(np.floor(carry + 1e-9))
        carry -= k
        for i, u in enumerate(grp):
            role[u["members"][0]] = "eval" if i < k else "fit"
    return role


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r1-dir", type=Path, default=DATA_ROOT / "probes/ras_divbank_20260924/r1")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "probes/ras_divbank_20260924")
    ap.add_argument("--n-pairs", type=int, default=240, help="strict mirror pairs (2 states each)")
    ap.add_argument("--n-reversed-pairs", type=int, default=40, help="reversed-subset pairs taken first, if present")
    ap.add_argument("--n-relaxed-pairs", type=int, default=40)
    ap.add_argument("--n-unreachable-pairs", type=int, default=30, help="strict but unreachable from spawn (sensitivity)")
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    gate = placement_gate(args.r1_dir)
    scans = {bg: load_scan(args.r1_dir, bg) for bg in ("black", "texture")}
    if set(scans["black"]) != set(scans["texture"]):
        raise SystemExit("STOP black and texture scans cover different levels")
    geo_fields = ("agent", "line", "gem", "walls", "walls_key", "state_key")
    n_diff = sum(any(scans["black"][lv][k] != scans["texture"][lv][k] for k in geo_fields) for lv in scans["black"])
    if n_diff:
        raise SystemExit(f"STOP black vs texture geometry differs on {n_diff} levels")
    if any(tuple(r["agent"]) != SPAWN for r in scans["black"].values()):
        raise SystemExit("STOP a scanned level does not spawn at (0,0)")
    configs = {}
    for lv in sorted(scans["black"]):
        r = scans["black"][lv]
        key = (r["walls_key"], tuple(r["line"]), tuple(r["gem"]))
        configs.setdefault(key, {"walls": r["walls"], "levels": []})["levels"].append(lv)

    states, counts = enumerate_states(configs)
    strict = [s for s in states if s["set"] == "strict"]
    unreach = [s for s in states if s["set"] == "strict_unreachable"]
    relaxed = [s for s in states if s["set"] == "relaxed"]
    units_s, units_r, units_u = make_units(strict), make_units(relaxed), make_units(unreach)

    def cat_counts(ss):
        return {"n": len(ss), "disagree_any": sum(s["bearing_action_disagrees"] for s in ss),
                "bearing_contrast_defined": sum(s["bearing_contrast_defined"] for s in ss),
                "disagree_decisive": sum(s["disagree_decisive"] for s in ss),
                "reversed": sum(s["reversed"] for s in ss),
                "canonical_spawn_states": sum(s["is_canonical_spawn_state"] for s in ss),
                "unreachable_from_spawn": sum(not s["reachable_from_spawn_in_episode"] for s in ss),
                "conflict_pairs": dict(Counter(s["conflict_pair"] for s in ss).most_common()),
                "contrasts": dict(Counter(s["contrast"] for s in ss).most_common()),
                "agent_cells": dict(sorted(Counter(str(s["agent"]) for s in ss).items())),
                "layouts": len({s["walls_key"] for s in ss}),
                "d": dict(sorted(Counter(s["d_line"] for s in ss).items())) if ss and ss[0]["set"] == "strict" else {},
                "quad_pairs": dict(Counter(s["quad_pair"] for s in ss).most_common())}

    raw = {"scanned_levels": len(scans["black"]), "level_range": [min(scans["black"]), max(scans["black"])],
           "unique_configs_walls_line_gem": len(configs),
           "wall_layouts": len({k[0] for k in configs}), "cell_counts": dict(counts),
           "strict_all_before_reachability_exclusion": cat_counts(strict + unreach),
           "strict": cat_counts(strict), "strict_unreachable": cat_counts(unreach), "relaxed": cat_counts(relaxed),
           "strict_mirror_closed_units": len(units_s), "relaxed_mirror_closed_units": len(units_r),
           "strict_units_decisive": sum(u["decisive"] for u in units_s),
           "strict_units_reversed": sum(u["reversed"] for u in units_s)}
    print(json.dumps({k: raw[k] for k in ("scanned_levels", "unique_configs_walls_line_gem", "wall_layouts",
                                          "cell_counts", "strict_mirror_closed_units", "strict_units_decisive",
                                          "strict_units_reversed")}, indent=1))

    # half decisive-disagree, half not, when available; reversed units first inside the decisive half
    n_half = args.n_pairs // 2
    rev = [u for u in units_s if u["reversed"]]
    dec = [u for u in units_s if u["decisive"] and not u["reversed"]]
    non = [u for u in units_s if not u["decisive"]]
    pick_rev = greedy_balance(rev, min(args.n_reversed_pairs, n_half), rng)
    pick_dec = greedy_balance(dec, n_half - len(pick_rev), rng)
    pick_non = greedy_balance(non, args.n_pairs - len(pick_rev) - len(pick_dec), rng)
    pick_rel = greedy_balance(units_r, args.n_relaxed_pairs, rng)
    pick_unr = greedy_balance(units_u, args.n_unreachable_pairs, rng)
    chosen = pick_rev + pick_dec + pick_non
    by_key = {s["state_key"]: s for s in states}

    role_s = split_units(chosen, args.eval_frac, rng)
    role_r = split_units(pick_rel, args.eval_frac, rng)
    role_u = split_units(pick_unr, args.eval_frac, rng)
    records = []
    for units, roles in ((chosen, role_s), (pick_rel, role_r), (pick_unr, role_u)):
        for u in units:
            uid = sha_obj(sorted(u["members"]))[:16]
            for k in u["members"]:
                r = dict(by_key[k])
                r["mirror_unit"] = uid
                r["split_role"] = roles[u["members"][0]]
                records.append(r)
    records.sort(key=lambda r: (r["set"], r["state_key"]))
    layouts = sorted({r["walls_key"] for r in records})
    for r in records:
        r["loo_fold"] = layouts.index(r["walls_key"])
        r["layout_id"] = r["loo_fold"]
    sel_strict = [r for r in records if r["set"] == "strict"]
    raw["selected"] = {"strict": cat_counts(sel_strict), "relaxed": cat_counts([r for r in records if r["set"] == "relaxed"]),
                       "strict_unreachable": cat_counts([r for r in records if r["set"] == "strict_unreachable"]),
                       "unreachable_pairs": len(pick_unr),
                       "strict_pairs": len(chosen), "relaxed_pairs": len(pick_rel),
                       "strict_fit": sum(r["split_role"] == "fit" for r in sel_strict),
                       "strict_eval": sum(r["split_role"] == "eval" for r in sel_strict),
                       "strict_eval_decisive": sum(r["split_role"] == "eval" and r["disagree_decisive"] for r in sel_strict)}

    args.out.mkdir(parents=True, exist_ok=True)
    geometry_sha = sha_obj([{k: r[k] for k in ("state_key", "agent", "line", "gem", "walls", "d_line", "d_gem",
                                               "A_line", "A_gem", "set")} for r in records])
    sel = {"bank": "ras_divbank_20260924", "stage": "selection (geometry only; renderings pending)",
           "selection_geometry_sha256": geometry_sha, "seed": args.seed, "args": {k: str(v) for k, v in vars(args).items()},
           "action_groups": GROUPS, "placement_gate": gate, "raw_counts": raw, "layouts": layouts,
           "records": records}
    (args.out / "ras_divbank_selection.json").write_text(json.dumps(sel))
    (args.out / "raw_counts.json").write_text(json.dumps(raw, indent=1))
    print(json.dumps(raw["selected"], indent=1)[:3000])
    print(f"wrote {len(records)} records, selection_geometry_sha256={geometry_sha}")


if __name__ == "__main__":
    main()
