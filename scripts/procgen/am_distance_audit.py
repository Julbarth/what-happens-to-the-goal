"""BFS-distance analysis of the Attribute Maze state sets and re-stratification of per-state paper results by proximity."""
import os
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from gm_robotics.paths import ROOT as RUN_ROOT

GMG = Path(os.environ.get("GM_GEOMETRY_ROOT", str(Path(__file__).resolve().parents[2])))
PROBES = GMG / "data/probes"
ROBO = RUN_ROOT / "results" / "analysis"
OUT = ROBO / "am_distance_audit"

W = H = 5
SPAWN = (0, 0)
# procgen combo order; diagonals move horizontally (ras_gt_bank.ENGINE_DELTAS); y grows with UP
ENGINE_DELTAS = {0: (-1, 0), 1: (-1, 0), 2: (-1, 0), 3: (0, -1), 4: None, 5: (0, 1),
                 6: (1, 0), 7: (1, 0), 8: (1, 0), 9: None, 10: None, 11: None, 12: None, 13: None, 14: None}
DIRNAME = {(0, 1): "UP", (1, 0): "RIGHT", (-1, 0): "LEFT", (0, -1): "DOWN"}
REL_ORDER = ["line_nearer", "equal", "gem_nearer"]
DIFF_BINS = [-99, -3, -2, -1, 0, 1, 2, 99]
DIFF_LABELS = ["<=-3", "-2", "-1", "0", "+1", "+2", ">=+3"]
D_BINS = {1: "d1", 2: "d2-3", 3: "d2-3", 4: "d4-5", 5: "d4-5", 6: "d6-7", 7: "d6-7"}


# ---------------------------------------------------------------- geometry

def bfs(walkable, target):
    dist, q = {target: 0}, deque([target])
    while q:
        c = q.popleft()
        for d in DIRNAME:
            n = (c[0] + d[0], c[1] + d[1])
            if n in walkable and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def successor(cell, a, walkable):
    d = ENGINE_DELTAS[a]
    if d is None:
        return cell
    n = (cell[0] + d[0], cell[1] + d[1])
    return n if n in walkable else cell


def progress_actions(dist, cell, walkable):
    d0 = dist.get(cell)
    if d0 is None or d0 == 0:
        return []
    return [a for a in range(15) if dist.get(successor(cell, a, walkable)) == d0 - 1]


def directions(actions):
    return sorted({ENGINE_DELTAS[a] for a in actions if ENGINE_DELTAS[a] is not None})


def n_shortest_paths(a, b, walk):
    da, db = bfs(walk, a), bfs(walk, b)
    if b not in da:
        return 0
    D = da[b]
    cells = sorted((c for c in walk if c in da and c in db and da[c] + db[c] == D), key=lambda c: da[c])
    n = {a: 1}
    for c in cells:
        if c != a:
            n[c] = sum(n.get((c[0] + d[0], c[1] + d[1]), 0) for d in DIRNAME
                       if da.get((c[0] + d[0], c[1] + d[1]), -9) == da[c] - 1)
    return n.get(b, 0)


def frac_line_paths_through_gem(agent, line, gem, walk):
    """Share of pass-through shortest agent->line paths that step on the gem (what the rewiring oracle labels)."""
    tot = n_shortest_paths(agent, line, walk)
    if not tot:
        return np.nan
    if bfs(walk, agent).get(gem, 99) + bfs(walk, line).get(gem, 99) != bfs(walk, agent)[line]:
        return 0.0
    return n_shortest_paths(agent, gem, walk) * n_shortest_paths(gem, line, walk) / tot


def state_key(agent, walls, line, gem):
    payload = [list(agent), sorted([list(w) for w in walls]), list(line), list(gem), W, H]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def geometry(agent, line, gem, walls):
    """Same definition as gm_geometry.probes.ras_gt_bank.geometry: reaching either target ends the episode, so the
    other target's cell is not passable; plus the pass-through (am_tie) and Manhattan distances for comparison."""
    agent, line, gem = tuple(agent), tuple(line), tuple(gem)
    walk = {(x, y) for x in range(W) for y in range(H)} - {tuple(w) for w in walls}
    dl, dg = bfs(walk - {gem}, line), bfs(walk - {line}, gem)
    al, ag = progress_actions(dl, agent, walk), progress_actions(dg, agent, walk)
    d_line, d_gem = dl.get(agent), dg.get(agent)
    dl_pt, dg_pt = bfs(walk, line).get(agent), bfs(walk, gem).get(agent)
    dirs_l, dirs_g = directions(al), directions(ag)
    if not al or not ag:
        branch = "undefined"
    elif set(al) == set(ag):
        branch = "same_first_move"
    elif set(al) & set(ag):
        branch = "overlap"
    else:
        branch = "disjoint"
    out = dict(agent=list(agent), line=list(line), gem=list(gem),
               d_line=d_line if d_line is not None else np.nan, d_gem=d_gem if d_gem is not None else np.nan,
               d_line_passthrough=dl_pt, d_gem_passthrough=dg_pt,
               man_line=abs(line[0] - agent[0]) + abs(line[1] - agent[1]),
               man_gem=abs(gem[0] - agent[0]) + abs(gem[1] - agent[1]),
               A_line=" ".join(map(str, al)), A_gem=" ".join(map(str, ag)),
               dirs_line="+".join(DIRNAME[d] for d in dirs_l), dirs_gem="+".join(DIRNAME[d] for d in dirs_g),
               branch=branch, strict_conflict=bool(branch == "disjoint" and d_line == d_gem and d_line),
               frac_oracle_paths_through_gem=frac_line_paths_through_gem(agent, line, gem, walk))
    return add_rel(out)


def add_rel(r):
    dl, dg = r["d_line"], r["d_gem"]
    if pd.isna(dl) or pd.isna(dg):
        r.update(diff=np.nan, absdiff=np.nan, rel="unreachable")
    else:
        r.update(diff=int(dl - dg), absdiff=int(abs(dl - dg)),
                 rel="line_nearer" if dl < dg else "gem_nearer" if dg < dl else "equal")
    return r


def diff_bin(x):
    return pd.cut(x, DIFF_BINS, labels=DIFF_LABELS).astype(str)


# ---------------------------------------------------------------- stats

def tci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n_seeds=0)
    m = v.mean()
    if n < 2:
        return dict(mean=m, lo=np.nan, hi=np.nan, n_seeds=n)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return dict(mean=m, lo=m - h, hi=m + h, n_seeds=n)


def summarize(df, keys, metrics, seed_col="seed"):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        for m in metrics:
            c = tci(g[m].values)
            row[m] = c["mean"]
            row[m + "_lo"] = c["lo"]
            row[m + "_hi"] = c["hi"]
        row["n_seeds"] = g[seed_col].nunique()
        if "n" in g:
            row["n_per_seed_mean"] = g["n"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_metrics(g, w=None):
    w = np.ones(len(g)) if w is None else np.asarray(w, float)
    goal = (g.outcome == "goal").values
    prox = (g.outcome == "proxy").values
    n = w.sum()
    ng, npx = (w * goal).sum(), (w * prox).sum()
    dec = ng + npx
    out = dict(n=len(g), n_w=n, reach=ng / n if n else np.nan, proxy=npx / n if n else np.nan,
               D=dec / n if n else np.nan, P_goal_dec=ng / dec if dec else np.nan,
               P_proxy_dec=npx / dec if dec else np.nan)
    if "nearer" in g:
        nearer_hit = ((g.nearer == "line") & goal) | ((g.nearer == "gem") & prox)
        neq = g.nearer.isin(["line", "gem"]).values
        dn = (w * (goal | prox) * neq).sum()
        out["P_nearer_dec"] = (w * nearer_hit.values).sum() / dn if dn else np.nan
    return out


def logistic(X, y, l2=1e-4, iters=100):
    """Newton-Raphson logistic regression with an intercept column already in X; tiny ridge for separable seeds."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (y - p) - l2 * b
        Hm = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(b))
        step = np.linalg.solve(Hm, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


# ---------------------------------------------------------------- state sets

def eval1000_states():
    """Levels 100000-100999 (trap preference and standard rewiring eval). Geometry read from the am_tie rollouts,
    which logged walls and object cells per level; all 54 files must agree level by level."""
    files = sorted((ROBO / "am_tie/raw").glob("raw_*.csv"))
    ref = None
    for f in files:
        t = pd.read_csv(f, usecols=["level", "line_x", "line_y", "gem_x", "gem_y", "walls"]).sort_values("level")
        t = t.reset_index(drop=True)
        if ref is None:
            ref = t
        elif not ref.equals(t):
            raise AssertionError(f"geometry mismatch in {f.name}")
    rows = []
    for r in ref.itertuples():
        walls = [tuple(map(int, c.split(","))) for c in str(r.walls).split(";")] if isinstance(r.walls, str) else []
        g = geometry(SPAWN, (r.line_x, r.line_y), (r.gem_x, r.gem_y), walls)
        g.update(level=int(r.level), walls=";".join(f"{x},{y}" for x, y in sorted(walls)),
                 state_key=state_key(SPAWN, walls, (r.line_x, r.line_y), (r.gem_x, r.gem_y)))
        rows.append(g)
    return pd.DataFrame(rows), len(files)


def gt_bank_states():
    bank = json.load(open(PROBES / "ras_gt_bank_v2_20260924/ras_gt_bank_canonical.json"))
    rows, n_mismatch = [], 0
    for r in bank["records"]:
        walls = [tuple(w) for w in r["walls"]]
        g = geometry(r["agent"], r["line"], r["gem"], walls)
        sk = state_key(r["agent"], walls, r["line"], r["gem"])
        ok = (sk == r["state_key"] and g["d_line"] == r["d_line"] and g["d_gem"] == r["d_gem"]
              and g["A_line"] == " ".join(map(str, r["A_line"])) and g["A_gem"] == " ".join(map(str, r["A_gem"])))
        n_mismatch += not ok
        g.update(state_key=r["state_key"], level=r["representative_level"], n_levels=len(r["levels"]),
                 axis_sign=r["axis_sign_lineN_gemE"], walls=";".join(f"{x},{y}" for x, y in sorted(walls)))
        rows.append(g)
    df = pd.DataFrame(rows)
    layouts = {w: i for i, w in enumerate(sorted(df.walls.unique()))}
    df["layout"] = df.walls.map(layouts)
    df["dbin"] = df.d_line.astype(int).map(D_BINS)
    split = json.load(open(PROBES / "ras_gt_bank_v2_20260924/splits/ras_gt_bank_v2_split_state_pair_signd1_seed0.json"))
    role = {e["state_key"]: "fit" for e in split["fit"]} | {e["state_key"]: "eval" for e in split["eval"]}
    df["split_role"] = df.state_key.map(role)
    return df, n_mismatch


def divbank_states():
    d = json.load(open(PROBES / "ras_divbank_20260924/ras_divbank.json"))
    rows, n_mismatch = [], 0
    for r in d["records"]:
        walls = [tuple(w) for w in r["walls"]]
        g = geometry(r["agent"], r["line"], r["gem"], walls)
        n_mismatch += not (g["d_line"] == r["d_line"] and g["d_gem"] == r["d_gem"])
        g.update(state_key=r["state_key"], set=r["set"], canonical_spawn=r["is_canonical_spawn_state"],
                 spawn_dist=r.get("spawn_dist"))
        rows.append(g)
    return pd.DataFrame(rows), n_mismatch


def unique_start_states():
    f = PROBES / "rewire_statedisjoint_20260924/raw/black/fresh_unique_states.csv"
    df = pd.read_csv(f)
    df = df.apply(lambda r: pd.Series(add_rel(dict(r))), axis=1)
    return df


def set_summary(name, df, weight=None, note=""):
    w = np.ones(len(df)) if weight is None else df[weight].values.astype(float)
    ok = df.rel != "unreachable"
    w2, d2 = w[ok.values], df[ok]
    tot = w2.sum()
    row = dict(set=name, n_states=len(df), weighting=weight or "state", n_unreachable=int((~ok).sum()), note=note)
    for rel in REL_ORDER:
        row[f"frac_{rel}"] = w2[(d2.rel == rel).values].sum() / tot
    row["frac_yellow_nearer"] = row["frac_gem_nearer"]
    row["frac_yellow_farther"] = row["frac_line_nearer"]
    for k in range(0, 8):
        row[f"absdiff_{k}"] = w2[(d2.absdiff == k).values].sum() / tot
    row["absdiff_ge8"] = w2[(d2.absdiff >= 8).values].sum() / tot
    row["mean_d_line"] = np.average(d2.d_line, weights=w2)
    row["mean_d_gem"] = np.average(d2.d_gem, weights=w2)
    row["mean_diff_line_minus_gem"] = np.average(d2["diff"], weights=w2)
    if "branch" in d2:
        for b in ["disjoint", "overlap", "same_first_move"]:
            row[f"frac_branch_{b}"] = w2[(d2.branch == b).values].sum() / tot
    return row


# ---------------------------------------------------------------- behaviour on the 1000 eval levels

def episodes_eval1000(geo):
    """Per-episode outcomes on levels 100000-100999 for every checkpoint whose per-episode result is stored."""
    rows = []
    lv = np.arange(100000, 101000)

    def from_rewards(rew, group, seed, source):
        rew = np.asarray(rew, float)
        assert len(rew) == 1000
        oc = np.where(rew == 10, "goal", np.where(rew == -10, "proxy", "none"))
        return pd.DataFrame(dict(level=lv, outcome=oc, group=group, seed=seed, source=source))

    for s in range(6311, 6319):
        d = json.load(open(PROBES / f"ras_pref_1000_corrected_20260916/Cprime_{s}_corrected.json"))
        rows.append(from_rewards(d["episode_rewards"], "GM_black_pref(6311-18)", s, "ras_pref_1000_corrected"))
    for s in (6301, 6302, 6303):
        d = json.load(open(PROBES / f"ras_pref_1000_armC_published_20260924/armC_{s}_textured.json"))
        rows.append(from_rewards(d["episode_rewards"], "GM_texture_pref(6301-03)", s, "ras_pref_1000_armC"))
    for s in (683, 3561, 3959, 7257, 8038):
        d = json.load(open(PROBES / f"ras_pref_1000_armC_published_20260924/published_{s}_black.json"))
        rows.append(from_rewards(d["episode_rewards"], "published_black(Raman)", s, "ras_pref_1000_armC"))
    for f in sorted((ROBO / "am_tie/raw").glob("raw_*.csv")):
        t = pd.read_csv(f, usecols=["cohort", "seed", "condition", "level", "outcome"])
        t["outcome"] = t.outcome.map({"goal": "goal", "other": "proxy", "neither": "none"})
        cond = t.condition.iloc[0]
        lab = {"gm": "GM", "control": "NonGM", "init": "init"}[t.cohort.iloc[0]]
        t["group"] = f"am_tie_{cond}:{lab}"
        t["source"] = "am_tie"
        rows.append(t[["level", "outcome", "group", "seed", "source"]])
    for f in sorted((PROBES / "rewire_statedisjoint_20260924/raw/black/raw_standard_repro").glob("*.csv")):
        t = pd.read_csv(f, usecols=["model", "level", "outcome"])
        t["group"] = "repro1000:" + t.model.str.rsplit("_", n=1).str[0]
        t["seed"] = t.model.str.rsplit("_", n=1).str[1].astype(int)
        t["source"] = "rewire_statedisjoint_standard_repro"
        rows.append(t[["level", "outcome", "group", "seed", "source"]])
    ep = pd.concat(rows, ignore_index=True)
    ep = ep.merge(geo[["level", "state_key", "d_line", "d_gem", "diff", "absdiff", "rel", "branch",
                       "dirs_line", "dirs_gem"]], on="level", how="left")
    ep["nearer"] = np.where(ep.rel == "line_nearer", "line", np.where(ep.rel == "gem_nearer", "gem", "tie"))
    ep["diff_bin"] = diff_bin(ep["diff"])
    return ep


def episodes_unique(us):
    """All 3,578 unique start states (levels 200000-1199999), one greedy episode each, every arm and seed; weighted by
    the number of levels that produce the state so the mix matches the level distribution."""
    rows = []
    for f in sorted((PROBES / "rewire_statedisjoint_20260924/raw/black/raw_statedisjoint").glob("*.csv")):
        t = pd.read_csv(f, usecols=["model", "level", "outcome"])
        t["group"] = "unique3578:" + t.model.str.rsplit("_", n=1).str[0]
        t["seed"] = t.model.str.rsplit("_", n=1).str[1].astype(int)
        rows.append(t)
    ep = pd.concat(rows, ignore_index=True).rename(columns={"level": "rep_level"})
    ep = ep.merge(us[["rep_level", "state_key", "n_levels", "d_line", "d_gem", "diff", "absdiff", "rel"]],
                  on="rep_level", how="left")
    assert ep.state_key.notna().all()
    ep["nearer"] = np.where(ep.rel == "line_nearer", "line", np.where(ep.rel == "gem_nearer", "gem", "tie"))
    ep["diff_bin"] = diff_bin(ep["diff"])
    return ep


def per_seed_strata(ep, weight=None):
    rows = []
    for (grp, seed), g in ep.groupby(["group", "seed"]):
        w = None if weight is None else g[weight]
        rows.append(dict(group=grp, seed=seed, stratum_type="all", stratum="all", **outcome_metrics(g, w)))
        for col in ("rel", "diff_bin", "absdiff"):
            for k, h in g.groupby(col):
                wh = None if weight is None else h[weight]
                rows.append(dict(group=grp, seed=seed, stratum_type=col, stratum=str(k), **outcome_metrics(h, wh)))
    return pd.DataFrame(rows)


def composition(ep, comp, weight=None):
    """Per seed: observed P(proxy|dec) vs re-weighted to (a) a symmetric line/gem-nearer mix, (b) the equal stratum,
    (c) the canonical bank's composition (all equal); plus the logistic model P(line|dec) ~ diff + direction."""
    rows = []
    sym = (comp["line_nearer"] + comp["gem_nearer"]) / 2
    for (grp, seed), g in ep.groupby(["group", "seed"]):
        w = np.ones(len(g)) if weight is None else g[weight].values
        r = dict(group=grp, seed=seed)
        tot = outcome_metrics(g, w)
        r["P_goal_dec_obs"], r["P_proxy_dec_obs"], r["reach_obs"] = tot["P_goal_dec"], tot["P_proxy_dec"], tot["reach"]
        strat = {}
        for rel in REL_ORDER:
            m = (g.rel == rel).values
            strat[rel] = outcome_metrics(g[m], w[m])
            r[f"P_goal_dec_{rel}"] = strat[rel]["P_goal_dec"]
            r[f"D_{rel}"] = strat[rel]["D"]
            r[f"reach_{rel}"] = strat[rel]["reach"]
        # decisive-weighted standardisation: weight each stratum by comp * its own decisiveness
        def std(wts):
            num = sum(wts[k] * strat[k]["D"] * strat[k]["P_goal_dec"] for k in REL_ORDER
                      if np.isfinite(strat[k]["P_goal_dec"]))
            den = sum(wts[k] * strat[k]["D"] for k in REL_ORDER if np.isfinite(strat[k]["P_goal_dec"]))
            return num / den if den else np.nan
        r["P_goal_dec_std_symmetric"] = std(dict(line_nearer=sym, equal=comp["equal"], gem_nearer=sym))
        r["P_goal_dec_std_obsmix"] = std(comp)
        r["P_goal_dec_equal_only"] = strat["equal"]["P_goal_dec"]
        dec = g.outcome.isin(["goal", "proxy"]).values & g["diff"].notna().values
        if dec.sum() >= 20 and 0 < (g.outcome[dec] == "goal").mean() < 1:
            y = (g.outcome[dec] == "goal").values.astype(float)
            df_ = g["diff"].values[dec].astype(float)
            line_up = g.dirs_line.values[dec] == "UP" if "dirs_line" in g else np.zeros(dec.sum(), bool)
            gem_up = g.dirs_gem.values[dec] == "UP" if "dirs_gem" in g else np.zeros(dec.sum(), bool)
            sw = w[dec]
            X = np.column_stack([np.ones(dec.sum()), np.clip(df_, -4, 4)])
            b = logistic_w(X, y, sw)
            r["logit_b0"], r["logit_b_diff"] = b
            r["P_goal_dec_model_at_diff0"] = 1 / (1 + np.exp(-b[0]))
            if "dirs_line" in g:
                X2 = np.column_stack([X, line_up.astype(float) - gem_up.astype(float)])
                b2 = logistic_w(X2, y, sw)
                r["logit2_b0"], r["logit2_b_diff"], r["logit2_b_lineUP_minus_gemUP"] = b2
        rows.append(r)
    return pd.DataFrame(rows)


def logistic_w(X, y, w, l2=1e-4, iters=100):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (w * (y - p)) - l2 * b
        Hm = X.T @ (X * (w * p * (1 - p))[:, None]) + l2 * np.eye(len(b))
        step = np.linalg.solve(Hm, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


# ---------------------------------------------------------------- mechanism banks (all equidistant)

def cue_rows(bank):
    rows = []
    for f in sorted((PROBES / "canon_cue_20260924/rows_csv_swap").glob("rows_swap_*.csv")):
        t = pd.read_csv(f, usecols=["cohort", "seed", "background", "edit", "state_key", "d", "axis_sign",
                                    "S_fav", "S_comp", "dC_log", "dP_fav", "dP_comp"])
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["pushpull"] = ((t.S_fav > 0) & (t.S_comp < 0)).astype(float)
    t = t.merge(bank[["state_key", "d_line", "d_gem", "dbin"]], on="state_key", how="left")
    assert (t.d == t.d_line).all() and (t.d_line == t.d_gem).all()
    t["group"] = t.cohort + "_" + t.background
    return t


def within_state_cos(bank):
    rows = []
    for f in sorted((PROBES / "canon_fisher_20260924/geom_swap").glob("geom_swap_*.npz")):
        _, _, bg, coh, seed = f.stem.split("_")
        z = np.load(f)
        p, Wpi, Wfc = z["p"].astype(np.float64), z["W_pi"].astype(np.float64), z["W_fc"].astype(np.float64)
        mask = z["h_fc_active_mask"].astype(np.float64)
        for tap in ("h_fc", "h_pre"):
            ds = z[f"delta_shape_swap_{tap}"].astype(np.float64)
            dc = z[f"delta_colour_swap_{tap}"].astype(np.float64)
            ce = (ds * dc).sum(1) / np.maximum(np.linalg.norm(ds, axis=1) * np.linalg.norm(dc, axis=1), 1e-12)
            cf = np.full(len(ds), np.nan)
            for i in range(len(ds)):
                J = Wpi if tap == "h_fc" else (Wpi * mask[i]) @ Wfc
                gs, gc = J @ ds[i], J @ dc[i]
                pi = p[i]
                fb = lambda a, b: (a * pi * b).sum() - (a @ pi) * (b @ pi)
                qs, qc = fb(gs, gs), fb(gc, gc)
                if qs > 1e-12 and qc > 1e-12:
                    cf[i] = fb(gs, gc) / np.sqrt(qs * qc)
            rows.append(pd.DataFrame(dict(cohort=coh, seed=int(seed), background=bg, tap=tap,
                                          state_key=z["state_key"], cos_E=ce, cos_F=cf)))
    t = pd.concat(rows, ignore_index=True).merge(bank[["state_key", "d_line", "dbin", "axis_sign"]], on="state_key")
    t["group"] = t.cohort + "_" + t.background + "_" + t.tap
    return t


def tug_rows(bank):
    rows = [pd.read_csv(f) for f in sorted((PROBES / "am_axis_tug_of_war_20260925/per_seed").glob("*.csv"))]
    t = pd.concat(rows, ignore_index=True).merge(bank[["state_key", "dbin"]], on="state_key")
    t["proxy_side"] = (t.xF_mid < 0).astype(float)
    t["pull_ratio_F"] = t.aF_p / t.aF_star
    oc = t.native_rollout_outcome
    t["goal"] = np.where(oc == "goal", 1.0, np.where(oc == "proxy", 0.0, np.nan))
    t["group"] = t.cohort + "_" + t.background
    return t


def per_seed_mean(t, keys, metrics):
    agg = t.groupby(keys + ["seed"])[metrics].mean().reset_index()
    n = t.groupby(keys + ["seed"]).size().rename("n").reset_index()
    return agg.merge(n, on=keys + ["seed"])


def bank_rollouts(bank):
    """Native and rewired policies, one greedy 256-step episode per bank state (canon_rewire_eval canon_raw)."""
    rows = []
    for bg in ("black", "texture"):
        for f in sorted((PROBES / f"canon_rewire_eval_20260924/raw/{bg}/canon_raw").glob("*.csv")):
            t = pd.read_csv(f, usecols=["model", "level", "state_hash", "sign", "d", "outcome"])
            t["background"] = bg
            rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["outcome"] = t.outcome.replace({"none": "none"})
    t["group"] = t.background + ":" + t.model.str.rsplit("_", n=1).str[0]
    t["seed"] = t.model.str.rsplit("_", n=1).str[1].astype(int)
    t["dbin"] = t.d.map(D_BINS)
    return t


def steering_rows():
    rows = []
    for f in sorted((PROBES / "canon_steer_20260924/swap/local").glob("raw_rows_*.csv")):
        t = pd.read_csv(f, usecols=["tap", "background", "seed", "split_role", "state_key", "d", "axis_sign", "arm",
                                    "beta", "outcome"])
        t = t[(t.arm == "baseline") | ((t.arm.isin(["corrective", "anticorrective"])) & (t.beta == 1.0))]
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["outcome"] = t.outcome.map({"line": "goal", "gem": "proxy", "timeout": "none"})
    coh = {6313: "GM", 6315: "GM", 6317: "GM", 6301: "GM", 6302: "GM", 6303: "GM"}
    t["cohort"] = t.seed.map(coh).fillna("NonGM")
    t["group"] = t.cohort + "_" + t.background + "_" + t.tap + "_" + t.arm
    t["dbin"] = t.d.map(D_BINS)
    return t


def divbank_pushpull(div):
    rows = []
    for f in sorted((PROBES / "divbank_pushpull_20260925/per_seed").glob("rows_*.csv")):
        _, bg, coh, seed = f.stem.split("_")
        t = pd.read_csv(f, usecols=["state_key", "edit", "d", "canonical_spawn", "S_fav", "S_comp", "dC_log"])
        t["background"], t["cohort"], t["seed"] = bg, coh, int(seed)
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t = t.merge(div[["state_key", "d_line", "d_gem", "spawn_dist"]], on="state_key", how="left")
    assert (t.d_line == t.d_gem).all()
    t["pushpull"] = ((t.S_fav > 0) & (t.S_comp < 0)).astype(float)
    t["dbin"] = t.d.map(lambda x: D_BINS.get(int(x), "d8+"))
    t["group"] = t.cohort + "_" + t.background
    return t


# ---------------------------------------------------------------- robot add-on

def robot_at():
    """AttributeTable colour-configuration scenes: does the conflict choice follow the object nearer the gripper?"""
    z = np.load(ROBO / "condition_scenes/AT_blue_cube.npz")
    u, v = z["u"].astype(float), z["v"].astype(float)
    rows = []
    for cond in ("blue_cube", "red_cube", "both_red", "both_blue"):
        z = np.load(ROBO / f"condition_scenes/AT_{cond}.npz")
        assert np.allclose(z["u"], u) and np.allclose(z["v"], v)
        for k in z.files:
            if not k.startswith("res_"):
                continue
            coh, s = k[4:].rsplit("_s", 1)
            tcp0 = z[f"tcp_{coh}_s{s}"][:, 0, :2].astype(float)
            du, dv = np.linalg.norm(u - tcp0, axis=1), np.linalg.norm(v - tcp0, axis=1)
            rows.append(pd.DataFrame(dict(condition=cond, cohort=coh, seed=int(s), episode=np.arange(len(u)),
                                          res=z[k], d_u=du, d_v=dv, tcp0_x=tcp0[:, 0], tcp0_y=tcp0[:, 1])))
    t = pd.concat(rows, ignore_index=True)
    t["diff_u_minus_v"] = t.d_u - t.d_v
    t["nearer"] = np.where(t.d_u < t.d_v, "u_cube_nearer", "v_sphere_nearer")
    return t


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    checks = {}

    # ---- 1. state sets
    geo, n_files = eval1000_states()
    checks["eval1000_geometry_files_agreeing"] = n_files
    tie = pd.read_csv(ROBO / "am_tie/raw/raw_gm_6313_conflict.csv", usecols=["level", "d_line", "d_gem"])
    m = geo.merge(tie, on="level", suffixes=("", "_amtie"))
    checks["eval1000_passthrough_matches_am_tie"] = int(((m.d_line_passthrough == m.d_line_amtie)
                                                          & (m.d_gem_passthrough == m.d_gem_amtie)).sum())
    checks["eval1000_exclusion_differs_from_passthrough"] = int(((m.d_line != m.d_line_passthrough)
                                                                 | (m.d_gem != m.d_gem_passthrough)).sum())
    us = unique_start_states()
    j = geo.merge(us[["state_key", "d_line", "d_gem"]], on="state_key", how="left", suffixes=("", "_us"))
    checks["eval1000_levels_found_in_unique3578"] = int(j.d_line_us.notna().sum())
    checks["eval1000_d_agree_with_unique3578"] = int(((j.d_line == j.d_line_us) & (j.d_gem == j.d_gem_us)).sum())
    geo.to_csv(out / "state_features_eval1000.csv", index=False)
    us.to_csv(out / "state_features_unique3578.csv", index=False)
    bank, nmis = gt_bank_states()
    checks["gt_bank_recompute_mismatches"] = nmis
    bank.to_csv(out / "state_features_gt_bank196.csv", index=False)
    div, nmis_div = divbank_states()
    checks["divbank_recompute_mismatches"] = nmis_div
    div.to_csv(out / "state_features_divbank620.csv", index=False)

    sets = [set_summary("eval1000 levels 100000-100999 (pref, rewiring standard, am_tie)", geo),
            set_summary("eval1000 unique states", geo.drop_duplicates("state_key")),
            set_summary("all unique start states (3578, levels 200000-1.2M), level-weighted", us, "n_levels"),
            set_summary("all unique start states (3578), unweighted", us),
            set_summary("GT conflict bank 196 (cue, Fisher, geometry, tug, steering, world edits)", bank),
            set_summary("GT bank fit 146", bank[bank.split_role == "fit"]),
            set_summary("GT bank eval 50 (steering, shared axis)", bank[bank.split_role == "eval"]),
            set_summary("GT bank, level-weighted (2041 strict-conflict levels)", bank, "n_levels"),
            set_summary("divbank strict 480 (paper)", div[div.set == "strict"], note="distances from placed agent"),
            set_summary("divbank strict_unreachable 60 (not in paper)", div[div.set == "strict_unreachable"]),
            set_summary("divbank relaxed 80 (not in paper, never scored)", div[div.set == "relaxed"])]
    for fold in range(11):
        sp = json.load(open(PROBES / f"ras_gt_bank_v2_20260924/splits/ras_gt_bank_v2_split_layout_loo_fold{fold:02d}.json"))
        ev = {e["state_key"] for e in sp["eval"]}
        sets.append(set_summary(f"GT bank layout-LOO fold{fold:02d} eval", bank[bank.state_key.isin(ev)]))
    sets_df = pd.DataFrame(sets)
    sets_df.to_csv(out / "set_distance_summary.csv", index=False)
    # d histogram of the equidistant banks (common distance)
    pd.concat([bank.assign(set="GT bank 196").groupby(["set", "d_line"]).size().rename("n").reset_index(),
               bank[bank.split_role == "eval"].assign(set="GT bank eval50").groupby(["set", "d_line"]).size()
               .rename("n").reset_index(),
               div[div.set == "strict"].assign(set="divbank 480").groupby(["set", "d_line"]).size().rename("n")
               .reset_index()]).to_csv(out / "equidistant_bank_d_histogram.csv", index=False)
    # signed-diff histogram for the level populations
    pd.concat([geo.groupby("diff").size().rename("n").reset_index().assign(set="eval1000"),
               us.groupby("diff").n_levels.sum().rename("n").reset_index().assign(set="unique3578_levelweighted")]
              ).to_csv(out / "signed_diff_histogram.csv", index=False)
    # cross-tab of distance relation and first-move branch on the eval levels
    pd.crosstab([geo.rel, geo.branch], [geo.dirs_line, geo.dirs_gem]).to_csv(out / "eval1000_rel_by_branch_dirs.csv")

    # ---- 2. behaviour by distance stratum on the eval levels
    ep = episodes_eval1000(geo)
    ps = per_seed_strata(ep)
    ps.to_csv(out / "behaviour_eval1000_per_seed.csv", index=False)
    met = ["reach", "proxy", "D", "P_goal_dec", "P_proxy_dec", "P_nearer_dec"]
    summ = summarize(ps, ["group", "stratum_type", "stratum"], met)
    summ.to_csv(out / "behaviour_eval1000_summary.csv", index=False)
    epu = episodes_unique(us)
    psu = per_seed_strata(epu, "n_levels")
    psu.to_csv(out / "behaviour_unique3578_per_seed.csv", index=False)
    summarize(psu, ["group", "stratum_type", "stratum"], met).to_csv(out / "behaviour_unique3578_summary.csv",
                                                                      index=False)

    comp = {rel: (geo.rel == rel).mean() for rel in REL_ORDER}
    co = composition(ep, comp)
    co.to_csv(out / "composition_eval1000_per_seed.csv", index=False)
    cm = [c for c in co.columns if c not in ("group", "seed")]
    summarize(co, ["group"], cm).to_csv(out / "composition_eval1000_summary.csv", index=False)
    compu = {rel: us.n_levels[us.rel == rel].sum() / us.n_levels.sum() for rel in REL_ORDER}
    cou = composition(epu, compu, "n_levels")
    cou.to_csv(out / "composition_unique3578_per_seed.csv", index=False)
    summarize(cou, ["group"], [c for c in cou.columns if c not in ("group", "seed")]).to_csv(
        out / "composition_unique3578_summary.csv", index=False)

    # the rewiring oracle (gm_geometry.rewire.ras_oracle) runs BFS over a walkable set that contains the gem cell,
    # so wherever every shortest line path crosses the gem its labels walk the agent into the -10 terminal
    f = geo.frac_oracle_paths_through_gem
    geo["oracle_path_through_gem"] = np.where(f >= 1, "all", np.where(f > 0, "some", "none"))
    assert ((geo.oracle_path_through_gem == "all") == (geo.d_line != geo.d_line_passthrough)).all()
    blk = geo.set_index("state_key").oracle_path_through_gem.groupby(level=0).first()
    rows = []
    for name, e, wcol in (("eval1000", ep, None), ("unique3578_in_eval1000_states", epu, "n_levels")):
        e = e.copy()
        e["blocked"] = e.state_key.map(blk)
        e = e[e.blocked.notna()]
        for (grp, seed), g in e.groupby(["group", "seed"]):
            for (rel, b), h in g.groupby(["rel", "blocked"]):
                rows.append(dict(population=name, group=grp, seed=seed, rel=rel, oracle_path_through_gem=b,
                                 **outcome_metrics(h, None if wcol is None else h[wcol])))
    pbk = pd.DataFrame(rows)
    pbk.to_csv(out / "behaviour_oracle_path_through_gem_per_seed.csv", index=False)
    summarize(pbk, ["population", "group", "rel", "oracle_path_through_gem"], ["reach", "D", "P_goal_dec"]).to_csv(
        out / "behaviour_oracle_path_through_gem_summary.csv", index=False)
    checks["eval1000_oracle_path_through_gem_counts"] = geo.oracle_path_through_gem.value_counts().to_dict()

    # which distance does the GM choice track? per-seed log-likelihood of one-slope logistic models
    defs = dict(bfs_exclusion=("d_line", "d_gem"), bfs_passthrough=("d_line_passthrough", "d_gem_passthrough"),
                manhattan=("man_line", "man_gem"))
    e = ep.drop(columns=["d_line", "d_gem"]).merge(geo[["level", "d_line", "d_gem", "d_line_passthrough",
                                                         "d_gem_passthrough", "man_line", "man_gem"]], on="level")
    rows = []
    for (grp, seed), g in e[e.outcome.isin(["goal", "proxy"])].groupby(["group", "seed"]):
        y = (g.outcome == "goal").values.astype(float)
        if not 0 < y.mean() < 1:
            continue
        r = dict(group=grp, seed=seed, n_dec=len(y))
        for k, (a, b) in defs.items():
            x = np.clip((g[a] - g[b]).values.astype(float), -4, 4)
            X = np.column_stack([np.ones(len(x)), x])
            bb = logistic(X, y)
            p = np.clip(1 / (1 + np.exp(-X @ bb)), 1e-9, 1 - 1e-9)
            r[f"ll_{k}"] = float((y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        rows.append(r)
    dd = pd.DataFrame(rows)
    dd["ll_exclusion_minus_passthrough"] = dd.ll_bfs_exclusion - dd.ll_bfs_passthrough
    dd["ll_exclusion_minus_manhattan"] = dd.ll_bfs_exclusion - dd.ll_manhattan
    dd.to_csv(out / "distance_definition_fit_per_seed.csv", index=False)
    summarize(dd, ["group"], [c for c in dd.columns if c.startswith("ll_")]).to_csv(
        out / "distance_definition_fit_summary.csv", index=False)

    # ---- 3. mechanism numbers by common distance d (all mechanism banks are equidistant)
    cue = cue_rows(bank)
    pc = per_seed_mean(cue, ["group", "edit", "dbin"], ["S_fav", "S_comp", "dC_log", "pushpull"])
    pa = per_seed_mean(cue, ["group", "edit"], ["S_fav", "S_comp", "dC_log", "pushpull"]).assign(dbin="all")
    pcs = pd.concat([pa, pc])
    pcs.to_csv(out / "mech_pushpull_by_d_per_seed.csv", index=False)
    summarize(pcs, ["group", "edit", "dbin"], ["S_fav", "S_comp", "dC_log", "pushpull"]).to_csv(
        out / "mech_pushpull_by_d_summary.csv", index=False)
    ps2 = per_seed_mean(cue, ["group", "edit", "axis_sign"], ["S_fav", "S_comp", "dC_log", "pushpull"])
    summarize(ps2, ["group", "edit", "axis_sign"], ["S_fav", "S_comp", "dC_log", "pushpull"]).to_csv(
        out / "mech_pushpull_by_sign_summary.csv", index=False)

    ws = within_state_cos(bank)
    pw = pd.concat([per_seed_mean(ws, ["group", "dbin"], ["cos_E", "cos_F"]),
                    per_seed_mean(ws, ["group"], ["cos_E", "cos_F"]).assign(dbin="all")])
    pw.to_csv(out / "mech_withinstate_cos_by_d_per_seed.csv", index=False)
    summarize(pw, ["group", "dbin"], ["cos_E", "cos_F"]).to_csv(out / "mech_withinstate_cos_by_d_summary.csv",
                                                                index=False)

    tug = tug_rows(bank)
    tug = tug[tug.axis == "unit_mean"]
    tm = ["aF_star", "aF_p", "pull_ratio_F", "proxy_side", "xF_mid", "cosF_dcomp_u", "goal"]
    pt = pd.concat([per_seed_mean(tug, ["group", "dbin"], tm), per_seed_mean(tug, ["group"], tm).assign(dbin="all")])
    pt.to_csv(out / "mech_tug_by_d_per_seed.csv", index=False)
    summarize(pt, ["group", "dbin"], tm).to_csv(out / "mech_tug_by_d_summary.csv", index=False)
    rho = []
    for (grp, seed), g in tug.groupby(["group", "seed"]):
        rho.append(dict(group=grp, seed=seed, spearman_xFmid_d=stats.spearmanr(g.xF_mid, g.d).statistic,
                        spearman_aFp_d=stats.spearmanr(g.aF_p, g.d).statistic,
                        spearman_aFstar_d=stats.spearmanr(g.aF_star, g.d).statistic))
    rho = pd.DataFrame(rho)
    summarize(rho, ["group"], ["spearman_xFmid_d", "spearman_aFp_d", "spearman_aFstar_d"]).to_csv(
        out / "mech_tug_rank_corr_with_d.csv", index=False)
    # AUC of xF_mid for goal-vs-proxy within d strata, pooled over GM seeds
    def _auc(x, y):
        return stats.mannwhitneyu(x[y == 1], x[y == 0]).statistic / (y.sum() * (len(y) - y.sum()))

    auc = []
    tg = tug[tug.goal.notna()]
    parts = [((grp, db, "pooled"), g) for (grp, db), g in tg.groupby(["group", "dbin"])]
    parts += [((grp, "all", "pooled"), g) for grp, g in tg.groupby("group")]
    parts += [((grp, "all", str(s)), g) for (grp, s), g in tg.groupby(["group", "seed"])]
    for (grp, db, seed), g in parts:
        y = g.goal.values
        if 0 < y.sum() < len(y):
            auc.append(dict(group=grp, dbin=db, seed=seed, n=len(y), auc_xFmid=_auc(g.xF_mid.values, y),
                            auc_common_d_only=_auc(g.d.values.astype(float), y)))
    pd.DataFrame(auc).to_csv(out / "mech_tug_auc_within_d.csv", index=False)

    br = bank_rollouts(bank)
    rows = []
    for (grp, seed), g in br.groupby(["group", "seed"]):
        rows.append(dict(group=grp, seed=seed, dbin="all", sign="all", **outcome_metrics(g)))
        for (db,), h in g.groupby(["dbin"]):
            rows.append(dict(group=grp, seed=seed, dbin=db, sign="all", **outcome_metrics(h)))
        for (sg, db), h in g.groupby(["sign", "dbin"]):
            rows.append(dict(group=grp, seed=seed, dbin=db, sign=str(sg), **outcome_metrics(h)))
        for (sg,), h in g.groupby(["sign"]):
            rows.append(dict(group=grp, seed=seed, dbin="all", sign=str(sg), **outcome_metrics(h)))
    pb = pd.DataFrame(rows)
    pb.to_csv(out / "bank_rollouts_by_d_sign_per_seed.csv", index=False)
    summarize(pb, ["group", "dbin", "sign"], ["reach", "proxy", "D", "P_goal_dec"]).to_csv(
        out / "bank_rollouts_by_d_sign_summary.csv", index=False)

    st = steering_rows()
    rows = []
    for (grp, seed), g in st.groupby(["group", "seed"]):
        for role, h0 in (("all196", g), ("eval50", g[g.split_role == "eval"])):
            rows.append(dict(group=grp, seed=seed, role=role, dbin="all", **outcome_metrics(h0)))
            for db, h in h0.groupby("dbin"):
                rows.append(dict(group=grp, seed=seed, role=role, dbin=db, **outcome_metrics(h)))
    pst = pd.DataFrame(rows)
    pst.to_csv(out / "mech_local_steering_by_d_per_seed.csv", index=False)
    summarize(pst, ["group", "role", "dbin"], ["reach", "D", "P_goal_dec"]).to_csv(
        out / "mech_local_steering_by_d_summary.csv", index=False)

    dpp = divbank_pushpull(div)
    pdv = pd.concat([per_seed_mean(dpp, ["group", "edit", "dbin"], ["S_fav", "S_comp", "dC_log", "pushpull"]),
                     per_seed_mean(dpp, ["group", "edit", "canonical_spawn"],
                                   ["S_fav", "S_comp", "dC_log", "pushpull"]).rename(
                         columns={"canonical_spawn": "dbin"}).assign(dbin=lambda x: "spawn=" + x.dbin.astype(str)),
                     per_seed_mean(dpp, ["group", "edit"], ["S_fav", "S_comp", "dC_log", "pushpull"]).assign(
                         dbin="all")])
    pdv.to_csv(out / "mech_divbank_pushpull_by_d_per_seed.csv", index=False)
    summarize(pdv, ["group", "edit", "dbin"], ["S_fav", "S_comp", "dC_log", "pushpull"]).to_csv(
        out / "mech_divbank_pushpull_by_d_summary.csv", index=False)

    # ---- 4. robot add-on
    rt = robot_at()
    rt.to_csv(out / "robot_AT_condition_scenes_per_episode.csv", index=False)
    rows = []
    for (cond, coh, seed), g in rt.groupby(["condition", "cohort", "seed"]):
        dec = g.res.isin(["u", "v"])
        for nm, h in [("all", g)] + list(g.groupby("nearer")):
            hd = h[h.res.isin(["u", "v"])]
            rows.append(dict(condition=cond, cohort=coh, seed=seed, stratum=nm, n=len(h),
                             P_u=(h.res == "u").mean(), P_v=(h.res == "v").mean(), D=h.res.isin(["u", "v"]).mean(),
                             P_u_dec=(hd.res == "u").mean() if len(hd) else np.nan))
        gd = g[dec]
        if 0 < (gd.res == "u").sum() < len(gd):
            b = logistic(np.column_stack([np.ones(len(gd)), gd.diff_u_minus_v / gd.diff_u_minus_v.std()]),
                         (gd.res == "u").values.astype(float))
            rows.append(dict(condition=cond, cohort=coh, seed=seed, stratum="logit_b_diff_per_sd", n=len(gd),
                             P_u_dec=b[1]))
    rr = pd.DataFrame(rows)
    rr.to_csv(out / "robot_AT_by_nearer_per_seed.csv", index=False)
    summarize(rr, ["condition", "cohort", "stratum"], ["P_u", "P_v", "D", "P_u_dec"]).to_csv(
        out / "robot_AT_by_nearer_summary.csv", index=False)
    # rewire_attr shares the 512 eval geometries (episode index); 97% outcome agreement with condition_scenes
    ra = pd.read_csv(ROBO / "rewire_attr/AT_per_episode.csv")
    ra = ra[ra.cond == "standard"].merge(rt[(rt.condition == "blue_cube")].drop_duplicates("episode")[
        ["episode", "nearer", "diff_u_minus_v"]], on="episode")
    rows = []
    for (arm, coh, seed, teach), g in ra.groupby(["arm", "cohort", "seed", "teacher"]):
        for nm, h in [("all", g)] + list(g.groupby("nearer")):
            hd = h[h.outcome.isin(["u", "v"])]
            rows.append(dict(arm=arm, cohort=coh, seed=seed, teacher=teach, stratum=nm, n=len(h),
                             P_u=(h.outcome == "u").mean(), D=h.outcome.isin(["u", "v"]).mean(),
                             P_u_dec=(hd.outcome == "u").mean() if len(hd) else np.nan))
    rw = pd.DataFrame(rows).groupby(["arm", "cohort", "seed", "stratum"], as_index=False)[
        ["n", "P_u", "D", "P_u_dec"]].mean()
    rw.to_csv(out / "robot_AT_rewire_attr_by_nearer_per_seed.csv", index=False)
    summarize(rw, ["arm", "cohort", "stratum"], ["P_u", "D", "P_u_dec"]).to_csv(
        out / "robot_AT_rewire_attr_by_nearer_summary.csv", index=False)
    checks["robot_AT_tcp0_spread_m"] = float(rt.groupby("episode")[["tcp0_x", "tcp0_y"]].first().std().max())
    checks["robot_AT_frac_u_nearer"] = float((rt.drop_duplicates("episode").nearer == "u_cube_nearer").mean())
    checks["robot_AT_absdiff_m_quantiles"] = rt.drop_duplicates("episode").diff_u_minus_v.abs().quantile(
        [0.1, 0.5, 0.9]).round(4).tolist()

    json.dump(dict(checks=checks, composition_eval1000=comp, composition_unique3578=compu), open(
        out / "checks.json", "w"), indent=1, default=float)
    print(json.dumps(checks, indent=1, default=float))
    print(sets_df[["set", "n_states", "frac_line_nearer", "frac_equal", "frac_gem_nearer"]].to_string())


if __name__ == "__main__":
    main()
