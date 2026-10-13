"""Finite-edit calibration of the local Fisher quadratic: exact KL(pi(h) || pi(h + t Delta)) vs 1/2 t^2 Delta^T F Delta at h_fc and h_pre."""
from __future__ import annotations

import os
import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
RAS_GEOM = DATA_ROOT / "probes/canon_fisher_20260924/geom_swap"
RAS_BANK = DATA_ROOT / "probes/ras_gt_bank_v2_20260924/ras_gt_bank_canonical.json"
RAS_PAIR_ROWS = DATA_ROOT / "probes/canon_fisher_20260924/rows_swap_pair"
MC_GEOM = DATA_ROOT / "probes/canon_mcfisher_20260924/raw_job28694/geom"
MC_ROWS = DATA_ROOT / "probes/canon_mcfisher_20260924/rows"
OUT_DIR = DATA_ROOT / "probes/finite_kl_calibration_20260924"
FIG = DATA_ROOT / "figure_inputs/fig_appendix_fisher_finite_kl_calibration"

T_GRID = np.round(np.linspace(0.0, 1.0, 21), 4)
T_SMALL = np.array([1e-3, 1e-2])
RIGHT_ACTIONS = (6, 7, 8)

# Primary rosters (paper): texture Non-GM is decotex; deco-on-texture and gm/gmtex cross-renderings are supplementary.
RAS_PRIMARY = {"black": {"gm": "GM", "deco": "Non-GM", "init": "init"},
               "texture": {"gmtex": "GM", "decotex": "Non-GM", "init": "init"}}
MC_AGENTS = {
    "maze": {"gm_1080": "GM", "gm_1081": "GM", "gm_1082": "GM", "nongm_1090": "Non-GM", "nongm_1091": "Non-GM",
             "nongm_1092": "Non-GM", "init_0": "init", "init_1": "init", "init_2": "init"},
    "coin": {"gm_6033": "GM", "gm_6037": "GM", "gm_6051": "GM", "nongm_7001": "Non-GM", "nongm_7002": "Non-GM",
             "nongm_7003": "Non-GM", "init_0": "init", "init_1": "init", "init_2": "init"},
}
COIN_6051 = "6051_gm_55M.pth"


# ----------------------------------------------------------------------------- math

def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def kl_from_logit_shift(p0: np.ndarray, d: np.ndarray) -> np.ndarray:
    """KL(p0 || softmax(log p0 + d)) = log sum p0 exp(e), e = d - p0.d, evaluated without cancellation."""
    e = d - (p0 * d).sum(axis=-1, keepdims=True)
    small = np.abs(e) < 1e-3
    series = e * e / 2 + e ** 3 / 6 + e ** 4 / 24
    # expm1(e) - e cancels catastrophically for tiny e, where the Taylor series is exact to f64.
    r = np.where(small, series, np.expm1(np.where(small, 0.0, e)) - np.where(small, 0.0, e))
    return np.log1p((p0 * r).sum(axis=-1))


class Heads:
    """Downstream map from a tap to logits, in float64."""

    def __init__(self, W_fc, b_fc, W_pi, b_pi):
        self.W_fc, self.b_fc = np.asarray(W_fc, np.float64), np.asarray(b_fc, np.float64)
        self.W_pi, self.b_pi = np.asarray(W_pi, np.float64), np.asarray(b_pi, np.float64)

    def logits(self, tap: str, h: np.ndarray) -> np.ndarray:
        if tap == "h_fc":
            return h @ self.W_pi.T + self.b_pi
        return np.maximum(h @ self.W_fc.T + self.b_fc, 0.0) @ self.W_pi.T + self.b_pi

    def logit_shift(self, tap: str, h: np.ndarray, delta: np.ndarray, t: float) -> np.ndarray:
        """logits(h + t delta) - logits(h), formed so that unchanged ReLU units contribute exactly t*W delta."""
        if tap == "h_fc":
            return t * (delta @ self.W_pi.T)
        a = h @ self.W_fc.T + self.b_fc
        b = delta @ self.W_fc.T
        a2 = a + t * b
        dr = np.where((a > 0) & (a2 > 0), t * b, np.maximum(a2, 0.0) - np.maximum(a, 0.0))
        return dr @ self.W_pi.T

    def jacobian(self, tap: str, h: np.ndarray) -> np.ndarray:
        """d logits / d h at the base, (N, A, D)."""
        if tap == "h_fc":
            return np.broadcast_to(self.W_pi, (h.shape[0],) + self.W_pi.shape)
        m = ((h @ self.W_fc.T + self.b_fc) > 0).astype(np.float64)
        return (self.W_pi[None] * m[:, None, :]) @ self.W_fc


def fisher_q(p: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Q = L^T (diag p - p p^T) L with L = J Delta."""
    return (p * L * L).sum(-1) - (p * L).sum(-1) ** 2


def grad_contrast(p: np.ndarray, J: np.ndarray, plus: list[int], minus: list[int]) -> np.ndarray:
    """grad_h [sum_plus pi - sum_minus pi] = sum_a sign_a p_a (J_a - p.J)."""
    dpi = p[:, None] * (J - np.einsum("a,ad->d", p, J)[None])
    return dpi[plus].sum(0) - dpi[minus].sum(0)


def score_states(heads: Heads, tap: str, h: np.ndarray, delta: np.ndarray) -> dict:
    """Per-state exact KL on the t grid, Q, and the small-t check terms."""
    h = h.astype(np.float64)
    delta = delta.astype(np.float64)
    p = np.exp(log_softmax(heads.logits(tap, h)))
    J = heads.jacobian(tap, h)
    L = np.einsum("nad,nd->na", J, delta)
    Q = fisher_q(p, L)
    kl = np.stack([kl_from_logit_shift(p, heads.logit_shift(tap, h, delta, t)) for t in T_GRID], 1)
    kl_small = np.stack([kl_from_logit_shift(p, heads.logit_shift(tap, h, delta, t)) for t in T_SMALL], 1)
    return {"p": p, "J": J, "Q": Q, "KL": kl, "KL_small": kl_small, "norm_delta": np.linalg.norm(delta, axis=1)}


def rows_for(meta: dict, idx_cols: dict, s: dict) -> list[dict]:
    out = []
    for i in range(len(s["Q"])):
        r = dict(meta)
        r.update({k: v[i] for k, v in idx_cols.items()})
        r["Q"] = s["Q"][i]
        r["norm_delta"] = s["norm_delta"][i]
        for j, t in enumerate(T_GRID):
            r[f"KL_t{t:g}"] = s["KL"][i, j]
        for j, t in enumerate(T_SMALL):
            r[f"KLsmall_t{t:g}"] = s["KL_small"][i, j]
        out.append(r)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- Attribute Maze

def score_ras(out: Path) -> dict:
    """Score every geom_swap file; returns max errors of the reconstruction checks."""
    pair_cols = {"shape_swap": ("Q_shape_swap", "g_shape_swap"), "colour_swap": ("Q_colour_swap", "g_colour_swap")}
    bank = {r["state_key"]: r for r in json.loads(RAS_BANK.read_text())["records"]}
    checks = {"max_abs_logit_err": 0.0, "max_rel_Q_vs_job": 0.0, "max_abs_g_vs_job": 0.0}
    for f in sorted(RAS_GEOM.glob("geom_swap_*.npz")):
        _, _, bg, coh, seed = f.stem.split("_")
        z = np.load(f)
        heads = Heads(z["W_fc"], z["b_fc"], z["W_pi"], z["b_pi"])
        keys = z["state_key"]
        checks["max_abs_logit_err"] = max(checks["max_abs_logit_err"], float(np.abs(
            heads.logits("h_fc", z["h_base_h_fc"].astype(np.float64)) - z["logits"]).max()))
        job = {}
        with open(RAS_PAIR_ROWS / f"rows_swap_pair_{bg}_{coh}_{seed}.csv") as fh:
            for r in csv.DictReader(fh):
                job[(r["state_key"], r["tap"])] = r
        A_line = [bank[k]["A_line"] for k in keys]
        A_gem = [bank[k]["A_gem"] for k in keys]
        rows = []
        for tap in ("h_fc", "h_pre"):
            for edit in ("shape_swap", "colour_swap"):
                s = score_states(heads, tap, z[f"h_base_{tap}"], z[f"delta_{edit}_{tap}"])
                d = z[f"delta_{edit}_{tap}"].astype(np.float64)
                g = np.array([grad_contrast(s["p"][i], s["J"][i], A_line[i], A_gem[i]) @ d[i] for i in range(len(keys))])
                qj = np.array([float(job[(k, tap)][pair_cols[edit][0]]) for k in keys])
                gj = np.array([float(job[(k, tap)][pair_cols[edit][1]]) for k in keys])
                checks["max_rel_Q_vs_job"] = max(checks["max_rel_Q_vs_job"],
                                                 float(np.max(np.abs(s["Q"] - qj) / (np.abs(qj) + 1e-12))))
                checks["max_abs_g_vs_job"] = max(checks["max_abs_g_vs_job"], float(np.abs(g - gj).max()))
                meta = {"env": f"attr_{bg}", "cohort_label": coh, "seed": seed, "tap": tap, "edit": edit}
                rows += rows_for(meta, {"state": np.arange(len(keys)), "state_key": keys, "g": g}, s)
        write_csv(out / "perstate" / f"perstate_attr_{bg}_{coh}_{seed}.csv", rows)
        print(f"scored {f.name}", flush=True)
    return checks


# ----------------------------------------------------------------------------- Maze / Coin

def load_mc_policy(env: str, agent: str, ckpt_dir: Path, ckpt_root: Path, train_root: Path):
    """ImpalaModel + CategoricalPolicy built as in the scoring scripts, with init draws seeded the same way."""
    import torch
    sys.modules.setdefault("gym", types.ModuleType("gym"))
    sys.path.insert(0, str(train_root))
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy

    if agent.startswith("init_"):
        i = int(agent.split("_")[1])
        seed = i if env == "maze" else 4000 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        path = None
    elif env == "maze":
        path = ckpt_dir / f"{agent.split('_')[1]}_{'gm' if agent.startswith('gm') else 'teach'}.pth"
    else:
        path = {"gm_6033": ckpt_dir / "6033_gm_20M.pth", "gm_6037": ckpt_dir / "6037_gm_120M.pth",
                "gm_6051": ckpt_root / COIN_6051}.get(agent, ckpt_dir / f"{agent.split('_')[1]}_teach.pth")
    policy = CategoricalPolicy(ImpalaModel(in_channels=3), False, 15)
    if path is not None:
        if not path.is_file():
            return None
        policy.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
    policy.eval()
    return policy


def mc_heads(policy) -> Heads:
    sd = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
    return Heads(sd["embedder.fc.weight"], sd["embedder.fc.bias"], sd["fc_policy.weight"], sd["fc_policy.bias"])


def mc_contrast(env: str, z, fi: int) -> tuple[list[int], list[int]]:
    """Maze: C = p(a_cheese) - p(a_corner). Coin: C = p(a_coin) - sum_RIGHT p."""
    if env == "maze":
        return [int(z[f"f{fi}/a_cheese"])], [int(z[f"f{fi}/a_corner"])]
    return [int(z[f"f{fi}/a_coin"])], list(RIGHT_ACTIONS)


def mc_frames(z) -> int:
    return len({k.split("/")[0] for k in z.files})


def load_mc_agent_arrays(z, agent: str, n: int) -> dict:
    return {k: np.stack([np.asarray(z[f"f{i}/{agent}/{k}"], np.float64) for i in range(n)])
            for k in ("p", "h_fc", "h_fc_reloc", "h_pre", "h_pre_reloc", "J_pre")}


def score_mc(env: str, geom: Path, ckpt_dir: Path, ckpt_root: Path, train_root: Path, out: Path) -> dict:
    z = np.load(geom)
    n = mc_frames(z)
    checks = {"n_frames": n, "agents_missing": [], "max_abs_p_err_fc": 0.0, "max_abs_p_err_pre": 0.0,
              "max_abs_hfc_from_hpre": 0.0, "max_rel_Jpre_err": 0.0, "max_rel_Q_vs_job": {}}
    for agent, coh in MC_AGENTS[env].items():
        pol = load_mc_policy(env, agent, ckpt_dir, ckpt_root, train_root)
        if pol is None:
            checks["agents_missing"].append(agent)
            print(f"  {env}/{agent}: checkpoint not available here, skipped", flush=True)
            continue
        heads = mc_heads(pol)
        a = load_mc_agent_arrays(z, agent, n)
        p_fc = np.exp(log_softmax(heads.logits("h_fc", a["h_fc"])))
        p_pre = np.exp(log_softmax(heads.logits("h_pre", a["h_pre"])))
        checks["max_abs_p_err_fc"] = max(checks["max_abs_p_err_fc"], float(np.abs(p_fc - a["p"]).max()))
        checks["max_abs_p_err_pre"] = max(checks["max_abs_p_err_pre"], float(np.abs(p_pre - a["p"]).max()))
        checks["max_abs_hfc_from_hpre"] = max(checks["max_abs_hfc_from_hpre"], float(np.abs(
            np.maximum(a["h_pre"] @ heads.W_fc.T + heads.b_fc, 0) - a["h_fc"]).max()))
        J_pre = heads.jacobian("h_pre", a["h_pre"])
        # Stored J_pre is the centred Jacobian (rows minus their mean); centring is invisible to softmax.
        Jc = J_pre - J_pre.mean(1, keepdims=True)
        Jsc = a["J_pre"] - a["J_pre"].mean(1, keepdims=True)
        checks["max_rel_Jpre_err"] = max(checks["max_rel_Jpre_err"],
                                         float(np.abs(Jc - Jsc).max() / (np.abs(Jsc).max() + 1e-12)))
        if max(checks["max_abs_p_err_fc"], checks["max_abs_p_err_pre"]) > 1e-4:
            raise SystemExit(f"{env}/{agent}: loaded weights do not reproduce stored p "
                             f"(fc {checks['max_abs_p_err_fc']:.2e}, pre {checks['max_abs_p_err_pre']:.2e})")
        rows = []
        for tap, key in (("h_fc", "fc"), ("h_pre", "block3")):
            h, d = a[tap], a[f"{tap}_reloc"] - a[tap]
            s = score_states(heads, tap, h, d)
            g = np.array([grad_contrast(s["p"][i], s["J"][i], *mc_contrast(env, z, i)) @ d[i] for i in range(n)])
            job_csv = MC_ROWS / f"rows_{env}_{key}_{agent}.csv"
            if job_csv.is_file():
                with open(job_csv) as fh:
                    jq = np.array([float(r["Q"]) for r in csv.DictReader(fh)])[:n]
                checks["max_rel_Q_vs_job"][f"{tap}/{agent}"] = float(np.max(np.abs(s["Q"][:len(jq)] - jq)
                                                                            / (np.abs(jq) + 1e-12)))
            meta = {"env": env, "cohort_label": coh, "seed": agent, "tap": tap, "edit": "relocation"}
            rows += rows_for(meta, {"state": np.arange(n), "state_key": [int(z[f"f{i}/level"]) for i in range(n)],
                                    "g": g}, s)
        write_csv(out / "perstate" / f"perstate_{env}_{agent}.csv", rows)
        print(f"  scored {env}/{agent} on {n} frames", flush=True)
    return checks


# ----------------------------------------------------------------------------- summary

def fmt(m: float, sd: float) -> str:
    if abs(m) < 0.01 and m != 0:
        return f"{m:.2e} ± {sd:.1e}"
    return f"{m:.3f} ± {sd:.2f}"


def load_perstate(out: Path) -> list[dict]:
    rows = []
    for f in sorted((out / "perstate").glob("perstate_*.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def paper_population_mask(rows: list[dict]) -> tuple[list[bool], dict]:
    """Keep Maze/Coin states whose Q matches the stored production row (Coin matched by level, Maze by frame index)."""
    prod: dict = {}
    for f in MC_ROWS.glob("rows_*.csv"):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                tap = "h_fc" if r["tap"] == "fc" else "h_pre"
                key = r["level"] if r["env"] == "coin" else r["frame"]
                prod[(r["env"], tap, r["agent"], key)] = float(r["Q"])
    keep, counts = [], {}
    for r in rows:
        if r["env"] not in ("maze", "coin"):
            keep.append(True)
            continue
        key = r["state_key"] if r["env"] == "coin" else r["state"]
        q = prod.get((r["env"], r["tap"], r["seed"], key))
        ok = q is not None and abs(float(r["Q"]) - q) <= 1e-4 * abs(q) + 1e-12
        keep.append(ok)
        c = counts.setdefault(f"{r['env']}/{r['tap']}/{r['seed']}", [0, 0])
        c[0] += ok
        c[1] += 1
    return keep, counts


def cohort_of(r: dict) -> str | None:
    if r["env"].startswith("attr_"):
        return RAS_PRIMARY[r["env"][5:]].get(r["cohort_label"])
    return r["cohort_label"]


ENV_ORDER = ["maze", "coin", "attr_black", "attr_texture"]
ENV_NAME = {"maze": "Maze", "coin": "Coin", "attr_black": "Attribute Maze (black)",
            "attr_texture": "Attribute Maze (texture)"}
EDIT_NAME = {"relocation": "relocation", "shape_swap": "shape swap", "colour_swap": "colour swap"}
COHORTS = ["GM", "Non-GM", "init"]


def per_seed_table(rows: list[dict]) -> dict:
    """{(env, tap, edit, cohort): {seed: {'KL': [T], 'halfQ': [T], 'ratio1', 'ratio05', 'n'}}}."""
    grp: dict = {}
    for r in rows:
        coh = cohort_of(r)
        if coh is None:
            continue
        grp.setdefault((r["env"], r["tap"], r["edit"], coh), {}).setdefault(r["seed"], []).append(r)
    res: dict = {}
    for key, seeds in grp.items():
        res[key] = {}
        for seed, rs in seeds.items():
            Q = np.array([float(r["Q"]) for r in rs])
            KL = np.array([[float(r[f"KL_t{t:g}"]) for t in T_GRID] for r in rs])
            halfQ = 0.5 * T_GRID[None] ** 2 * Q[:, None]
            i1, i05 = int(np.argmin(np.abs(T_GRID - 1))), int(np.argmin(np.abs(T_GRID - 0.5)))
            with np.errstate(divide="ignore", invalid="ignore"):
                r1 = KL[:, i1] / halfQ[:, i1]
                r05 = KL[:, i05] / halfQ[:, i05]
            res[key][seed] = {"KL": np.median(KL, 0), "halfQ": np.median(halfQ, 0),
                              "ratio1": float(np.nanmedian(r1)), "ratio05": float(np.nanmedian(r05)),
                              "frac_ratio1_gt1": float(np.mean(r1 > 1)), "n": len(rs),
                              "KL_small": np.array([[float(r[f"KLsmall_t{t:g}"]) for t in T_SMALL] for r in rs]),
                              "Q_states": Q}
    return res


def summarize(out: Path, n_mc_note: str) -> dict:
    rows = load_perstate(out)
    keep, pop_counts = paper_population_mask(rows)
    rows = [r for r, k in zip(rows, keep) if k]
    ps = per_seed_table(rows)
    t_show = [0.25, 0.5, 0.75, 1.0]
    idx = {t: int(np.argmin(np.abs(T_GRID - t))) for t in t_show}
    lines = ["# Finite-edit calibration of the Fisher quadratic: exact KL vs ½t²Q", "",
             "KL(t) = KL(π(h) ‖ π(h+tΔ)) through the exact downstream map; ½t²Q with Q = ΔᵀFΔ, F at the base state "
             "(J = ∂logits/∂h, corrected Jacobian). Per seed: median over states. Cell: mean ± SD across the 3 seed "
             "medians (SD ddof = 1, n = 3); scientific notation below 0.01. Maze/Coin: only states whose Q matches the production row of job 28694 (paper population) are used. Ratio = per-seed median over states of "
             "KL/(½t²Q), then mean ± SD across seeds. frac>1 = mean fraction of states with KL(1) > ½Q. "
             f"t = 0 is identically 0 for both. {n_mc_note} Producer: scripts/fisher_finite_kl_calibration.py.", "",
             "| env | edit | tap | cohort | n/seed | " + " | ".join(f"KL t={t:g}" for t in t_show) + " | "
             + " | ".join(f"½t²Q t={t:g}" for t in t_show) + " | ratio t=0.5 | ratio t=1 | frac>1 (t=1) |",
             "|" + "---|" * (5 + 2 * len(t_show) + 3)]
    summary: dict = {}
    for env in ENV_ORDER:
        for edit in ("relocation", "shape_swap", "colour_swap"):
            for tap in ("h_pre", "h_fc"):
                for coh in COHORTS:
                    key = (env, tap, edit, coh)
                    if key not in ps:
                        continue
                    seeds = ps[key]
                    KL = np.stack([v["KL"] for v in seeds.values()])
                    HQ = np.stack([v["halfQ"] for v in seeds.values()])
                    r1 = np.array([v["ratio1"] for v in seeds.values()])
                    r05 = np.array([v["ratio05"] for v in seeds.values()])
                    fr = np.array([v["frac_ratio1_gt1"] for v in seeds.values()])
                    ns = sorted({v["n"] for v in seeds.values()})
                    sd = (lambda x: np.std(x, ddof=1, axis=0)) if len(seeds) > 1 else (lambda x: np.full(x.shape[1:], np.nan))
                    cells = [fmt(KL.mean(0)[idx[t]], sd(KL)[idx[t]]) for t in t_show]
                    cells += [fmt(HQ.mean(0)[idx[t]], sd(HQ)[idx[t]]) for t in t_show]
                    rr = [fmt(float(r05.mean()), float(sd(r05[:, None])[0])), fmt(float(r1.mean()), float(sd(r1[:, None])[0]))]
                    lines.append(f"| {ENV_NAME[env]} | {EDIT_NAME[edit]} | {tap} | {coh} ({len(seeds)} seeds) | "
                                 f"{'/'.join(map(str, ns))} | " + " | ".join(cells) + " | " + " | ".join(rr)
                                 + f" | {fr.mean():.2f} |")
                    summary["|".join(key)] = {
                        "seeds": list(seeds), "n_per_seed": ns, "t": T_GRID.tolist(),
                        "KL_seed_medians": KL.tolist(), "halfQ_seed_medians": HQ.tolist(),
                        "ratio_t1_seeds": r1.tolist(), "ratio_t05_seeds": r05.tolist(), "frac_ratio1_gt1": fr.tolist()}
    (out / "table_finite_kl_mean_sd.md").write_text("\n".join(lines) + "\n")
    small = small_t_check(ps)
    summary["_small_t_check"] = small
    summary["_mc_paper_population_matched_over_scored"] = pop_counts
    (out / "summary_finite_kl.json").write_text(json.dumps(summary, indent=1))
    plot(ps)
    return small


def small_t_check(ps: dict) -> dict:
    """Relative error of 1/2 t^2 Q vs KL at t = 1e-2 and 1e-3; O(t^3) absolute error means the ratio of the two is ~10."""
    out = {}
    for key, seeds in ps.items():
        rel = []
        for v in seeds.values():
            q = v["Q_states"]
            keep = 0.5 * q * T_SMALL[0] ** 2 > 1e-13
            if keep.sum() == 0:
                continue
            hq = 0.5 * q[keep, None] * T_SMALL[None] ** 2
            rel.append(np.abs(v["KL_small"][keep] / hq - 1))
        if not rel:
            continue
        rel = np.concatenate(rel)
        out["|".join(key)] = {"n_states": int(len(rel)), "max_rel_err_t1e-3": float(rel[:, 0].max()),
                              "max_rel_err_t1e-2": float(rel[:, 1].max()),
                              "median_err_ratio_1e-2_over_1e-3": float(np.median(rel[:, 1] / np.maximum(rel[:, 0], 1e-300)))}
    return out


def plot(ps: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"GM": "#D55E00", "Non-GM": "#0072B2", "init": "#7f7f7f"}
    panels = [("maze", "relocation"), ("coin", "relocation"), ("attr_black", "shape_swap"),
              ("attr_black", "colour_swap"), ("attr_texture", "shape_swap"), ("attr_texture", "colour_swap")]
    fig, axes = plt.subplots(len(panels), 2, figsize=(8.2, 2.25 * len(panels)), sharex=True)
    tt = T_GRID[1:]
    for r, (env, edit) in enumerate(panels):
        for c, tap in enumerate(("h_pre", "h_fc")):
            ax = axes[r, c]
            for coh in ("GM", "Non-GM"):
                seeds = ps.get((env, tap, edit, coh))
                if not seeds:
                    continue
                col = colours[coh]
                KL = np.stack([v["KL"][1:] for v in seeds.values()])
                HQ = np.stack([v["halfQ"][1:] for v in seeds.values()])
                for k in range(len(KL)):
                    ax.plot(tt, KL[k], color=col, lw=0.6, alpha=0.35)
                    ax.plot(tt, HQ[k], color=col, lw=0.6, alpha=0.35, ls="--")
                ax.plot(tt, KL.mean(0), color=col, lw=2.0, label=f"{coh}: exact KL")
                ax.plot(tt, HQ.mean(0), color=col, lw=2.0, ls="--", label=f"{coh}: ½t²Q")
            ax.set_yscale("log")
            ax.spines[["top", "right"]].set_visible(False)
            title = f"{ENV_NAME[env]}, {EDIT_NAME[edit]} — {tap}"
            ns = {v["n"] for coh in COHORTS for v in ps.get((env, tap, edit, coh), {}).values()}
            if ns and max(ns) < 20:
                title += f"\nillustrative, {max(ns)} states/agent"
                short = [f"{coh} n = {len(ps[(env, tap, edit, coh)])} seeds" for coh in ("GM", "Non-GM")
                         if len(ps.get((env, tap, edit, coh), {})) not in (0, 3)]
                if short:
                    title += "; " + ", ".join(short) + (" (6051 missing)" if env == "coin" else "")
            ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel("median KL (log scale)", fontsize=8)
            if r == len(panels) - 1:
                ax.set_xlabel("t (fraction of the finite edit Δ)", fontsize=8)
            ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colours[c], lw=2, ls=ls) for c in ("GM", "Non-GM") for ls in ("-", "--")]
    labels = [f"{c}: {k}" for c in ("GM", "Non-GM") for k in ("exact KL(t)", "½t²Q (base-state Fisher)")]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["all", "ras", "mc", "summarize"], default="all")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--mc-geom-dir", type=Path, default=MC_GEOM, help="dir with geom_{maze,coin}_first<K>.npz")
    ap.add_argument("--mc-k", type=int, default=3)
    ap.add_argument("--maze-ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/maze")
    ap.add_argument("--coin-ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/coin")
    ap.add_argument("--ckpt-root", type=Path, default=DATA_ROOT / "checkpoints/coin", help="directory holding the Coin 6051 checkpoint")
    ap.add_argument("--train-root", type=Path, default=REPO / "external/train-procgen-pytorch")
    ap.add_argument("--mc-note", default="Maze/Coin: n/seed states per agent (see report).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    checks = {}
    if args.mode in ("all", "ras"):
        checks["ras"] = score_ras(args.out)
    if args.mode in ("all", "mc"):
        for env, ck in (("maze", args.maze_ckpt_dir), ("coin", args.coin_ckpt_dir)):
            checks[env] = score_mc(env, args.mc_geom_dir / f"geom_{env}_first{args.mc_k}.npz", ck, args.ckpt_root,
                                   args.train_root, args.out)
    if checks:
        (args.out / f"checks_{args.mode}.json").write_text(json.dumps(checks, indent=1))
        print(json.dumps(checks, indent=1))
    if args.mode in ("all", "summarize"):
        small = summarize(args.out, args.mc_note)
        print(json.dumps(small, indent=1))


if __name__ == "__main__":
    main()
