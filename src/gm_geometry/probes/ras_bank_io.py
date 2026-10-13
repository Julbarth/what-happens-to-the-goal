"""Attribute Maze conflict-bank I/O: load, dedupe by state, leak-free splits and consumer adapters."""
# Cells are (x, y) with y UP (action 5 UP moves (0, +1)). state_key = sha256 over (agent, sorted walls, line, gem,
# grid_w, grid_h); the 5x5 maze yields many duplicate levels, so splits are by state or wall layout, never by level.
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gm_geometry.probes.ras_gt_bank import ENGINE_DELTAS, canonical_hash

BANK_IO_VERSION = "ras_bank_io_v1_20260924"
SPLIT_VERSION = "ras_bank_split_v1"
STATE_HASH_LEN = 16
OBJ_LINE, OBJ_GEM = "red_line_diag", "yellow_gem"
DEFAULT_WORLD_DIM = 5
# Fields that are pure functions of the state; duplicates disagreeing on any of them = corrupt bank.
STATE_DETERMINED = ("grid_w", "grid_h", "d_line", "d_gem", "A_line", "A_gem", "dirs_line", "dirs_gem",
                    "conflict", "bearing", "axis_sign_lineN_gemE", "path_line", "path_gem")


def _cell(c) -> tuple[int, int]:
    x, y = c
    return int(x), int(y)


def state_tuple(rec: dict) -> tuple:
    """(agent, walls, line, gem, (grid_w, grid_h)); walls sorted, so list/tuple and order don't matter."""
    return (_cell(rec["agent"]), tuple(sorted(_cell(w) for w in rec["walls"])),
            _cell(rec["line"]), _cell(rec["gem"]), (int(rec["grid_w"]), int(rec["grid_h"])))


def _state_key_from_tuple(t: tuple) -> str:
    agent, walls, line, gem, (w, h) = t
    payload = [list(agent), [list(c) for c in walls], list(line), list(gem), w, h]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def state_key(rec: dict) -> str:
    """The bank's state identity (see module docstring); a stored key must match the recompute."""
    k = _state_key_from_tuple(state_tuple(rec))
    stored = rec.get("state_key")
    if stored is not None and stored != k:
        raise ValueError(f"level {rec.get('level')}: stored state_key {stored} != recomputed {k}")
    return k


def layout_key(rec: dict) -> tuple:
    return tuple(sorted(_cell(w) for w in rec["walls"]))


def state_hash(key: str) -> str:
    return key[:STATE_HASH_LEN]


def _dup_levels(rec: dict) -> list[int]:
    """A unique-state bank may already fold duplicates into the record."""
    for f in ("levels", "dup_levels", "duplicate_levels"):
        if isinstance(rec.get(f), list):
            return [int(v) for v in rec[f]]
    return [int(rec["level"])]


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bank_format(doc: dict) -> str:
    if "canonical_bank_sha256" in doc and "renderings" in doc:
        return "canonical"
    if "unique_bank_sha256" in doc:
        return "unique"
    if "bank_sha256" in doc and "layout_sha256" in doc:
        return "per_background_v1"
    raise ValueError("not a ras_gt_bank file (no canonical_bank_sha256 / unique_bank_sha256 / bank_sha256)")


def _content_sha(doc: dict, fmt: str) -> tuple[str, str]:
    """(stored, recomputed) content hash, each format hashed the way its builder hashed it."""
    if fmt == "per_background_v1":
        return doc["bank_sha256"], canonical_hash(doc["records"])
    stored = doc["canonical_bank_sha256"] if fmt == "canonical" else doc["unique_bank_sha256"]
    return stored, _sha256_obj(doc["records"])


def load_bank_doc(path, *, verify: bool = True) -> dict:
    doc = json.loads(Path(path).read_text())
    try:
        fmt = _bank_format(doc)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if verify:
        stored, got = _content_sha(doc, fmt)
        if stored != got:
            raise ValueError(f"{path}: {fmt} content sha256 mismatch (stored {stored}, recomputed {got})")
    doc["_format"] = fmt
    return doc


def _normalize_canonical(rec: dict, background: str | None) -> dict:
    """One canonical (environment-state) record -> the per-background record schema, for one rendering.
    Canonical records omit grid size, dirs and paths: grid defaults to DEFAULT_WORLD_DIM (checked by the
    state_key recompute) and dirs are derived from the engine-verified A sets."""
    out = {k: v for k, v in rec.items() if k not in ("renderings", "representative_level", "levels")}
    out["level"] = int(rec["representative_level"])
    out["levels"] = [int(v) for v in rec["levels"]]
    if out["levels"][0] != out["level"] or out["levels"] != sorted(out["levels"]):
        raise ValueError(f"state {rec['state_key']}: levels must be sorted and start at representative_level")
    out.setdefault("grid_w", DEFAULT_WORLD_DIM)
    out.setdefault("grid_h", DEFAULT_WORLD_DIM)
    for side in ("line", "gem"):
        out.setdefault(f"dirs_{side}", [list(d) for d in sorted({ENGINE_DELTAS[a] for a in rec[f"A_{side}"]
                                                                 if ENGINE_DELTAS[a] is not None})])
    out["background"] = background
    if background is not None:
        rnd = rec["renderings"][background]
        if len(rnd["state_sha256"]) != len(out["levels"]) or len(rnd["obs_sha256"]) != len(out["levels"]):
            raise ValueError(f"state {rec['state_key']}: {background} rendering lists not aligned with levels")
        out["state_sha256"] = rnd["state_sha256"][0]
        out["obs_sha256"] = rnd["obs_sha256"][0]
    return out


def load_bank(path, *, background: str | None = None, verify: bool = True) -> list[dict]:
    """Records in the per-background schema. A canonical file serves either rendering: `background`
    picks which state/obs hashes are attached (None = geometry only, enough for dedupe/splits)."""
    doc = load_bank_doc(path, verify=verify)
    if doc["_format"] == "canonical":
        if background is not None and background not in doc["renderings"]:
            raise ValueError(f"{path}: no {background!r} rendering (has {doc['renderings']})")
        return [_normalize_canonical(r, background) for r in doc["records"]]
    if background is not None and doc["background"] != background:
        raise ValueError(f"{path} is a {doc['background']!r} bank, requested {background!r}")
    return doc["records"]


def bank_info(path, *, background: str | None = None) -> dict:
    doc = load_bank_doc(path)
    fmt = doc["_format"]
    stored, _ = _content_sha(doc, fmt)
    bg = background if fmt == "canonical" else doc["background"]
    if background is not None and bg != background:
        raise ValueError(f"{path} is a {bg!r} bank, requested {background!r}")
    return {"bank": doc["bank"], "bank_format": fmt, "background": bg, "bank_sha256": stored,
            "canonical_bank_sha256": doc.get("canonical_bank_sha256"),
            "geometry_sha256": doc.get("shared_geometry_sha256") or doc.get("layout_sha256"),
            "builder_code_sha256": doc.get("builder_code_sha256"),
            "bank_path": str(path), "bank_file_sha256": file_sha256(path), "n_records": len(doc["records"])}


def dedupe(records: list[dict]) -> list[dict]:
    """Unique states sorted by representative level. Each entry: state_key, state_hash, layout_key,
    level (lowest duplicate level), levels (all duplicate levels), record (the lowest level's record)."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(state_key(r), []).append(r)
    out = []
    for key, recs in groups.items():
        recs = sorted(recs, key=lambda r: int(r["level"]))
        rep = recs[0]
        for r in recs[1:]:
            bad = [f for f in STATE_DETERMINED if f in rep and r.get(f) != rep.get(f)]
            if bad:
                raise ValueError(f"levels {rep['level']} and {r['level']} share state {key} but differ on {bad}")
        levels = sorted({lv for r in recs for lv in _dup_levels(r)})
        if levels[0] != int(rep["level"]):
            raise ValueError(f"state {key}: representative level {rep['level']} is not its lowest level {levels[0]}")
        out.append({"state_key": key, "state_hash": state_hash(key), "state_tuple": state_tuple(rep),
                    "layout_key": layout_key(rep), "level": int(rep["level"]), "levels": levels, "record": rep})
    hashes = [u["state_hash"] for u in out]
    if len(set(hashes)) != len(hashes):
        raise ValueError("state_hash collision; increase STATE_HASH_LEN")
    return sorted(out, key=lambda u: u["level"])


def _axis_sign(u: dict):
    return u["record"].get("axis_sign_lineN_gemE")


def state_d(u: dict) -> int:
    """Conflict depth d = d_line (== d_gem for every conflict). d = 1 means both targets are adjacent,
    so the first move ends the episode; analyses report full-bank and d > 1 results."""
    return int(u["record"]["d_line"])


def _stratum(u: dict, stratify: str) -> str:
    if stratify == "sign":
        return str(_axis_sign(u))
    if stratify == "sign_d1":
        return f"{_axis_sign(u)}|{'d1' if state_d(u) == 1 else 'd>1'}"
    raise ValueError(f"unknown stratify {stratify!r}")


def swap_partner_key(u: dict) -> str:
    """state_key of the same walls/agent with line and gem cells exchanged. Swapping keeps a conflict a
    conflict (same d, disjoint moves exchanged) and flips axis_sign, so the bank is closed under it."""
    r = dict(u["record"])
    r["line"], r["gem"] = r["gem"], r["line"]
    r.pop("state_key", None)
    return state_key(r)


def _entry(u: dict) -> dict:
    return {"state_hash": u["state_hash"], "state_key": u["state_key"], "level": u["level"],
            "levels": u["levels"], "axis_sign": _axis_sign(u), "d": state_d(u)}


def _sign_counts(entries: list[dict]) -> dict:
    out: dict[str, int] = {}
    for e in entries:
        out[str(e["axis_sign"])] = out.get(str(e["axis_sign"]), 0) + 1
    return out


def _strata_counts(entries: list[dict]) -> dict:
    out: dict[str, int] = {}
    for e in entries:
        k = f"{e['axis_sign']}|{'d1' if e['d'] == 1 else 'd>1'}"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def make_splits(unique_states: list[dict], *, mode: str = "state", eval_frac: float = 0.25, seed: int = 0,
                k_layouts: int = 2, fold: int | None = None, stratify: str = "sign_d1",
                group: str = "state", bank: dict | None = None) -> dict:
    """Deterministic, seeded fit/eval partition of unique states.

    mode="state": stratified by `stratify` ("sign" or "sign_d1" = axis sign x (d == 1)); within each
      stratum, round(eval_frac * n) states (>= 1 when the stratum has >= 2) go to eval.
      group="swap_pair" makes each state and its line/gem-swapped partner one unit (both fit or both
      eval), so eval never contains the mirror image of a fit state; the unit's stratum is its d1 flag
      (a pair holds one +1 and one -1 state, so eval stays exactly sign-balanced).
    mode="layout": k_layouts random whole wall layouts are held out; every state on them is eval.
    mode="layout_loo": leave-one-layout-out, fold = index into the layouts sorted by wall list; seed unused.
    Layout modes are stronger: no eval state shares walls with any fit state.
    The shuffle order depends only on (seed, state hashes), not on record order or level ids.
    """
    rng = np.random.default_rng(seed)
    by_hash = {u["state_hash"]: u for u in unique_states}
    if mode == "state":
        units: dict[str, list[str]] = {}
        unit_stratum: dict[str, str] = {}
        by_key = {u["state_key"]: u for u in unique_states}
        for u in unique_states:
            if group == "swap_pair":
                pk = swap_partner_key(u)
                uid = min(u["state_key"], pk) if pk in by_key else u["state_key"]
                st = _stratum(u, stratify) if pk not in by_key else (
                    f"pair|{'d1' if state_d(u) == 1 else 'd>1'}" if stratify == "sign_d1" else "pair")
            elif group == "state":
                uid, st = u["state_key"], _stratum(u, stratify)
            else:
                raise ValueError(f"unknown group {group!r}")
            units.setdefault(uid, []).append(u["state_hash"])
            unit_stratum[uid] = st
        strata: dict[str, list[str]] = {}
        for uid, st in unit_stratum.items():
            strata.setdefault(st, []).append(uid)
        eval_h: set[str] = set()
        for st in sorted(strata):
            us = sorted(strata[st])
            perm = [us[i] for i in rng.permutation(len(us))]
            n_ev = int(round(eval_frac * len(us)))
            if len(us) >= 2:
                n_ev = min(max(n_ev, 1), len(us) - 1)
            for uid in perm[:n_ev]:
                eval_h.update(units[uid])
        params = {"eval_frac": eval_frac, "stratify": stratify, "group": group}
        held_layouts = None
    elif mode == "layout":
        layouts = sorted({u["layout_key"] for u in unique_states})
        if not 0 < k_layouts < len(layouts):
            raise ValueError(f"k_layouts={k_layouts} must be in [1, {len(layouts) - 1}]")
        held = {layouts[i] for i in rng.permutation(len(layouts))[:k_layouts]}
        eval_h = {u["state_hash"] for u in unique_states if u["layout_key"] in held}
        params = {"k_layouts": k_layouts, "n_layouts": len(layouts)}
        held_layouts = sorted([list(map(list, lk)) for lk in held])
    elif mode == "layout_loo":
        layouts = sorted({u["layout_key"] for u in unique_states})
        if fold is None or not 0 <= fold < len(layouts):
            raise ValueError(f"layout_loo needs fold in [0, {len(layouts) - 1}], got {fold}")
        held = {layouts[fold]}
        eval_h = {u["state_hash"] for u in unique_states if u["layout_key"] in held}
        params = {"fold": int(fold), "n_folds": len(layouts), "n_layouts": len(layouts)}
        held_layouts = [list(map(list, layouts[fold]))]
    else:
        raise ValueError(f"unknown split mode {mode!r}")

    fit = [_entry(by_hash[h]) for h in sorted(by_hash, key=lambda h: by_hash[h]["level"]) if h not in eval_h]
    ev = [_entry(by_hash[h]) for h in sorted(eval_h, key=lambda h: by_hash[h]["level"])]
    assert_disjoint(fit, ev, by_layout=mode.startswith("layout"), unique_states=unique_states)
    split = {
        "split_version": SPLIT_VERSION, "bank_io_version": BANK_IO_VERSION, "mode": mode, "seed": int(seed),
        **params, "held_out_layouts": held_layouts,
        "bank": (bank or {}).get("bank"), "bank_sha256": (bank or {}).get("bank_sha256"),
        "geometry_sha256": (bank or {}).get("geometry_sha256"), "bank_format": (bank or {}).get("bank_format"),
        "canonical_bank_sha256": (bank or {}).get("canonical_bank_sha256"),
        "n_unique_states": len(unique_states), "n_fit": len(fit), "n_eval": len(ev),
        "fit_sign_counts": _sign_counts(fit), "eval_sign_counts": _sign_counts(ev),
        "fit_strata_counts": _strata_counts(fit), "eval_strata_counts": _strata_counts(ev),
        "n_fit_levels": sum(len(e["levels"]) for e in fit), "n_eval_levels": sum(len(e["levels"]) for e in ev),
        "n_eval_with_swap_partner_in_fit": _n_partner_leak(fit, ev, unique_states),
        "fit": fit, "eval": ev,
    }
    split = json.loads(json.dumps(split))
    split["split_sha256"] = _sha256_obj(split)
    return split


def _n_partner_leak(fit: list[dict], ev: list[dict], unique_states: list[dict]) -> int:
    by_key = {u["state_key"]: u for u in unique_states}
    fk = {e["state_key"] for e in fit}
    return sum(swap_partner_key(by_key[e["state_key"]]) in fk for e in ev)


def assert_disjoint(fit: list[dict], ev: list[dict], *, by_layout: bool = False, unique_states=None) -> None:
    fk = {e["state_key"] for e in fit}
    ek = {e["state_key"] for e in ev}
    both = fk & ek
    assert not both, f"{len(both)} state_key(s) in both fit and eval"
    fl = {lv for e in fit for lv in e["levels"]}
    el = {lv for e in ev for lv in e["levels"]}
    assert not fl & el, f"levels in both fit and eval: {sorted(fl & el)[:10]}"
    if by_layout:
        lay = {u["state_hash"]: u["layout_key"] for u in unique_states}
        fl_ = {lay[e["state_hash"]] for e in fit}
        el_ = {lay[e["state_hash"]] for e in ev}
        assert not fl_ & el_, "a wall layout appears in both fit and eval"


def load_split(path) -> dict:
    split = json.loads(Path(path).read_text())
    body = {k: v for k, v in split.items() if k != "split_sha256"}
    if _sha256_obj(body) != split.get("split_sha256"):
        raise ValueError(f"{path}: split_sha256 does not match its content")
    return split


def save_split(split: dict, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))
    return file_sha256(path)


def select_states(bank_path, split_path, role: str, *, background: str | None = None,
                  max_states: int = 0, exclude_d1: bool = False) -> tuple[list[dict], dict]:
    """Unique states of one split role, resolved against the bank, plus the provenance dict every
    output must carry. Splits live in state_key space, so one split serves both renderings (black and
    texture runs then score the same state set). Every split state must exist in the bank with the
    identical duplicate-level list; the geometry hash must also match when both come from one bank format."""
    if role not in ("fit", "eval"):
        raise ValueError(f"role must be fit or eval, got {role!r}")
    info = bank_info(bank_path, background=background)
    split = load_split(split_path)
    if (split.get("bank_format") == info["bank_format"] and split.get("geometry_sha256")
            and split["geometry_sha256"] != info["geometry_sha256"]):
        raise ValueError(f"split {split_path} was built on geometry {split['geometry_sha256']}, "
                         f"bank {bank_path} has {info['geometry_sha256']}")
    assert_disjoint(split["fit"], split["eval"])
    uniq = {u["state_key"]: u for u in dedupe(load_bank(bank_path, background=info["background"]))}
    chosen = []
    for e in split[role]:
        u = uniq.get(e["state_key"])
        if u is None:
            raise ValueError(f"split state {e['state_hash']} (level {e['level']}) not in bank {bank_path}")
        if u["level"] != e["level"] or u["levels"] != e["levels"]:
            raise ValueError(f"split state {e['state_hash']}: levels {e['levels']} != bank {u['levels']}")
        chosen.append(u)
    chosen.sort(key=lambda u: u["level"])
    n_role = len(chosen)
    n_d1 = sum(state_d(u) == 1 for u in chosen)
    if exclude_d1:
        chosen = [u for u in chosen if state_d(u) > 1]
    if max_states > 0:
        chosen = chosen[:max_states]
    prov = {**info, "bank_io_version": BANK_IO_VERSION, "split_path": str(split_path),
            "split_file_sha256": file_sha256(split_path), "split_sha256": split["split_sha256"],
            "split_mode": split["mode"], "split_seed": split["seed"], "split_role": role,
            "split_built_on": {k: split.get(k) for k in ("bank", "bank_format", "bank_sha256", "geometry_sha256")},
            "n_states_in_role": n_role, "n_d1_in_role": n_d1, "exclude_d1": bool(exclude_d1),
            "n_states_used": len(chosen), "max_states": int(max_states),
            "grid_convention": "engine (x, y), y up; action 5 = (0,+1)",
            "states": [{"level": u["level"], "state_hash": u["state_hash"], "state_key": u["state_key"],
                        "d": state_d(u), "dup_levels": u["levels"]} for u in chosen]}
    return chosen, prov


def groups_from_engine() -> dict[tuple[int, int], list[int]]:
    """Direction -> all engine action ids (diagonal aliases included) that move that way."""
    g: dict[tuple[int, int], list[int]] = {}
    for a, d in ENGINE_DELTAS.items():
        if d is not None:
            g.setdefault(d, []).append(a)
    return {k: sorted(v) for k, v in g.items()}


def bank_groups(states: list[dict]) -> dict[tuple[int, int], list[int]]:
    """direction -> action ids, taken from the bank's own A_line/A_gem (never recalibrated). Fails if two
    states disagree on the id set for one direction, or a state's progress set spans two directions."""
    g: dict[tuple[int, int], list[int]] = {}
    for u in states:
        r = u["record"]
        for dk, ak in (("dirs_line", "A_line"), ("dirs_gem", "A_gem")):
            if len(r[dk]) != 1:
                raise ValueError(f"level {r['level']}: {dk}={r[dk]} is not a single direction")
            d = tuple(r[dk][0])
            if g.setdefault(d, list(r[ak])) != list(r[ak]):
                raise ValueError(f"direction {d}: bank action sets disagree ({g[d]} vs {r[ak]})")
    return g


def bank_bucket_sign(bearing: tuple) -> float | None:
    """Signed-axis label for a bank bearing (line_dx, line_dy, gem_dx, gem_dy): a state pairs with the
    state whose line/gem directions are SWAPPED. With the agent at (0, 0) only N/E moves exist, so the two
    buckets are (0,1,1,0) -> +1 (line N, gem E) and (1,0,0,1) -> -1; exact negations never occur, which is
    why the harvest-era negation pairing cannot be used on the bank (it would abort)."""
    b = tuple(int(v) for v in bearing)
    if b == (0, 1, 1, 0):
        return 1.0
    if b == (1, 0, 0, 1):
        return -1.0
    return None


def env_raw_state(env) -> bytes:
    cur, seen = env, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "get_state") and callable(cur.get_state):
            if hasattr(cur, "set_object") or not hasattr(cur, "env"):
                st = cur.get_state()
                return bytes(st[0]) if isinstance(st, (list, tuple)) else bytes(st)
        cur = getattr(cur, "env", getattr(cur, "venv", getattr(cur, "_env", None)))
    raise RuntimeError("could not unwrap env to get_state")


def _obs_rgb(obs) -> np.ndarray:
    return np.asarray(obs["rgb"] if isinstance(obs, dict) else obs).reshape(64, 64, 3)


def materialize_res(u: dict, *, make_base_env, state_bytes, use_backgrounds: bool) -> dict:
    """The per-state `res` dict harvest consumers expect, rebuilt from the bank.

    ``make_base_env(level)`` builds the base scene with the consumer's make_env and ``state_bytes`` is the consumer's
    fingerprint; asserts the stored state and observation hashes. Action sets are the bank's A_line/A_gem verbatim."""
    r = u["record"]
    bg = "texture" if use_backgrounds else "black"
    if r["background"] != bg:
        raise ValueError(f"bank background {r['background']!r} but consumer runs {bg!r}")
    level = int(u["level"])
    env = make_base_env(level)
    try:
        obs0 = env.reset()
        raw = env_raw_state(env)
        sb = state_bytes(env)
    finally:
        env.close()
    got = hashlib.sha256(raw).hexdigest()
    assert got == r["state_sha256"], f"level {level}: state sha256 {got} != bank {r['state_sha256']}"
    if r.get("obs_sha256"):
        got_o = hashlib.sha256(_obs_rgb(obs0).tobytes()).hexdigest()
        assert got_o == r["obs_sha256"], f"level {level}: obs sha256 {got_o} != bank {r['obs_sha256']}"

    agent, line, gem = _cell(r["agent"]), _cell(r["line"]), _cell(r["gem"])
    walls = {_cell(w) for w in r["walls"]}
    walkable = {(x, y) for x in range(r["grid_w"]) for y in range(r["grid_h"])} - walls
    g_line = tuple(r["dirs_line"][0]) if len(r["dirs_line"]) == 1 else None
    g_gem = tuple(r["dirs_gem"][0]) if len(r["dirs_gem"]) == 1 else None
    return {
        "level": level, "obs0": obs0, "state_bytes": sb,
        "dist_line": int(r["d_line"]), "dist_gem": int(r["d_gem"]),
        "d": int(r["d_line"]), "is_d1": int(r["d_line"]) == 1,
        "first_actions_line": list(r["A_line"]), "first_actions_gem": list(r["A_gem"]),
        "first_group_line": g_line, "first_group_gem": g_gem,
        "n_cells_explored": len(walkable),
        "agent_cell": agent, "line_xy": line, "gem_xy": gem, "line_cell": line, "gem_cell": gem,
        "geometry": {"walkable": walkable, "calib": None,
                     "action_deltas": {a: d for a, d in ENGINE_DELTAS.items()},
                     "agent_cell": agent, "line_cell": line, "gem_cell": gem},
        "bearing": tuple(r["bearing"]) if r.get("bearing") else None,
        "axis_sign_lineN_gemE": r.get("axis_sign_lineN_gemE"),
        "state_hash": u["state_hash"], "state_key": u["state_key"], "dup_levels": list(u["levels"]),
        "state_sha256": r["state_sha256"], "state_sha256_verified": True,
    }


def add_bank_args(ap: argparse.ArgumentParser, *, default_role: str | None = None) -> None:
    g = ap.add_argument_group("canonical conflict bank (opt-in; replaces the script's own harvest)")
    g.add_argument("--bank-json", type=Path, default=None,
                   help="ras_gt_bank_canonical.json (or a per-background bank); required unless --allow-legacy-harvest")
    g.add_argument("--bank-split", type=Path, default=None, help="split JSON from ras_bank_io make-split")
    g.add_argument("--split-role", choices=["fit", "eval"], default=default_role)
    g.add_argument("--bank-max-states", type=int, default=0, help="first N states of the role (smoke tests); 0 = all")
    g.add_argument("--exclude-d1", action="store_true", help="drop d = 1 states (both targets adjacent)")
    g.add_argument("--allow-legacy-harvest", action="store_true",
                   help="run the script's own conflict-state harvest (unvalidated localizers); output is flagged")


LEGACY_HARVEST_WARNING = ("LEGACY HARVEST: conflict states were found by this script's own localizer, not the "
                          "canonical engine-validated bank. Earlier audits found two such localizers wrong "
                          "(forensic_E_provenance.md). Do not use these numbers as results.")


def check_bank_args(args) -> bool:
    """True = bank mode. Without --bank-json the script refuses to run unless --allow-legacy-harvest."""
    if args.bank_json is None:
        if args.bank_split is not None:
            raise SystemExit("--bank-split requires --bank-json")
        if not args.allow_legacy_harvest:
            raise SystemExit("refusing to self-harvest conflict states: pass --bank-json/--bank-split/--split-role "
                             "or --allow-legacy-harvest to override")
        print("!" * 100 + f"\n{LEGACY_HARVEST_WARNING}\n" + "!" * 100, flush=True)
        return False
    if args.allow_legacy_harvest:
        raise SystemExit("--allow-legacy-harvest and --bank-json are mutually exclusive")
    if args.bank_split is None or args.split_role is None:
        raise SystemExit("--bank-json requires --bank-split and --split-role")
    return True


def bank_res_list(args, role: str, *, make_base_env, state_bytes, use_backgrounds: bool,
                  max_states: int | None = None):
    n_max = args.bank_max_states if max_states is None else max_states
    states, prov = select_states(args.bank_json, args.bank_split, role, max_states=n_max,
                                 background="texture" if use_backgrounds else "black",
                                 exclude_d1=args.exclude_d1)
    res = [materialize_res(u, make_base_env=make_base_env, state_bytes=state_bytes,
                           use_backgrounds=use_backgrounds) for u in states]
    prov["n_state_sha256_verified"] = sum(bool(r["state_sha256_verified"]) for r in res)
    print(f"[bank] role={role} states={len(res)}/{prov['n_states_in_role']} bank={prov['bank']} "
          f"bg={prov['background']} bank_sha256={prov['bank_sha256'][:12]} split_sha256={prov['split_sha256'][:12]} "
          f"state_sha256 verified {prov['n_state_sha256_verified']}/{len(res)}", flush=True)
    return res, prov, states


def annotate_rows(rows: list[dict], res_list: list[dict]) -> list[dict]:
    """Add state_hash / dup-level count to every output row, matched by representative level."""
    by_level = {r["level"]: r for r in res_list}
    for row in rows:
        m = by_level.get(int(row["level"]))
        if m is not None:
            row["state_hash"] = m["state_hash"]
            row["n_dup_levels"] = len(m["dup_levels"])
            row["d"] = m["d"]
            row["is_d1"] = m["is_d1"]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a leak-free fit/eval split of a canonical Attribute Maze conflict bank.")
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--mode", choices=["state", "layout", "layout_loo"], default="state")
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--stratify", choices=["sign", "sign_d1"], default="sign_d1")
    ap.add_argument("--group", choices=["state", "swap_pair"], default="swap_pair",
                    help="state mode: split unit (swap_pair keeps line/gem-swapped partners together)")
    ap.add_argument("--k-layouts", type=int, default=2)
    ap.add_argument("--fold", type=int, default=None, help="layout_loo fold index")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    info = bank_info(args.bank)
    uniq = dedupe(load_bank(args.bank))
    split = make_splits(uniq, mode=args.mode, eval_frac=args.eval_frac, seed=args.seed, k_layouts=args.k_layouts,
                        fold=args.fold, stratify=args.stratify, group=args.group, bank=info)
    fsha = save_split(split, args.out)
    print(f"{args.out}: mode={args.mode} unique={len(uniq)}/{info['n_records']} fit={split['n_fit']} "
          f"({split['n_fit_levels']} lv) {split['fit_strata_counts']} eval={split['n_eval']} "
          f"({split['n_eval_levels']} lv) {split['eval_strata_counts']} "
          f"split_sha256={split['split_sha256']} file_sha256={fsha}")


if __name__ == "__main__":
    main()
