"""Exact world-edit KL(pi(.|s) || pi(.|s_e)) on the 128-frame Maze Fisher bank, so KL and Q share one population."""
from __future__ import annotations

import os
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
import fisher_finite_kl_calibration as fkl  # noqa: E402

DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
PERSTATE = DATA_ROOT / "probes/finite_kl_calibration_20260924/perstate"
PROD_ROWS = DATA_ROOT / "probes/canon_mcfisher_20260924/rows"
GEOM = DATA_ROOT / "probes/canon_mcfisher_20260924/raw_job28694/geom/geom_maze_first3.npz"
FORKS = DATA_ROOT / "probes/maze_envmove_graph_exact_20260914"
COIN_KL = DATA_ROOT / "probes/coin_worldedit_canon_20260924/coin_worldedit_canon.json"
OUT = DATA_ROOT / "probes/canon_mcfisher_20260924"
AGENTS = fkl.MC_AGENTS["maze"]
FORK_FILE = {"gm_1080": "maze_1080_gm", "gm_1081": "maze_1081_gm", "gm_1082": "maze_1082_gm",
             "nongm_1090": "maze_1090_teach", "nongm_1091": "maze_1091_teach", "nongm_1092": "maze_1092_teach"}


def read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def kl_probs(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * (np.log(p) - np.log(q))).sum(-1)


def verify_dump(ckpt_dir: Path, train_root: Path, perstate: dict) -> dict:
    """Frames 0-2: weights reproduce the dumped p; full-network KL from obs equals the h_fc/h_pre KL and the stored per-state KL."""
    import torch
    z = np.load(GEOM)
    out = {"frames": 3, "max_abs_p_err_hfc": 0.0, "max_abs_p_err_hpre": 0.0, "max_abs_p_err_obs": 0.0,
           "max_abs_hfc_err_obs": 0.0, "max_abs_hfc_reloc_err_obs": 0.0, "max_rel_kl_obs_vs_hfc": 0.0,
           "max_rel_kl_hfc_vs_perstate": 0.0, "max_rel_kl_hpre_vs_hfc": 0.0}
    for agent in AGENTS:
        pol = fkl.load_mc_policy("maze", agent, ckpt_dir, Path("/nonexistent"), train_root)
        heads = fkl.mc_heads(pol)
        a = fkl.load_mc_agent_arrays(z, agent, 3)
        hr = np.stack([np.asarray(z[f"f{i}/{agent}/h_fc_reloc"], np.float64) for i in range(3)])
        hpr = np.stack([np.asarray(z[f"f{i}/{agent}/h_pre_reloc"], np.float64) for i in range(3)])
        lp = lambda tap, h: np.exp(fkl.log_softmax(heads.logits(tap, h)))
        p0, p1 = lp("h_fc", a["h_fc"]), lp("h_fc", hr)
        out["max_abs_p_err_hfc"] = max(out["max_abs_p_err_hfc"], float(np.abs(p0 - a["p"]).max()))
        out["max_abs_p_err_hpre"] = max(out["max_abs_p_err_hpre"], float(np.abs(lp("h_pre", a["h_pre"]) - a["p"]).max()))
        obs = lambda k: torch.tensor(np.stack([z[f"f{i}/{k}"] for i in range(3)]), dtype=torch.float32)
        with torch.no_grad():
            e0, e1 = pol.embedder(obs("obs_a")), pol.embedder(obs("obs_reloc"))
            q0 = torch.softmax(pol.fc_policy(e0).double(), -1).numpy()
            q1 = torch.softmax(pol.fc_policy(e1).double(), -1).numpy()
        out["max_abs_p_err_obs"] = max(out["max_abs_p_err_obs"], float(np.abs(q0 - a["p"]).max()))
        out["max_abs_hfc_err_obs"] = max(out["max_abs_hfc_err_obs"], float(np.abs(e0.numpy() - a["h_fc"]).max()))
        out["max_abs_hfc_reloc_err_obs"] = max(out["max_abs_hfc_reloc_err_obs"], float(np.abs(e1.numpy() - hr).max()))
        k_fc = kl_probs(p0, p1)
        k_pre = kl_probs(lp("h_pre", a["h_pre"]), lp("h_pre", hpr))
        k_obs = kl_probs(q0, q1)
        k_ps = np.array([perstate[agent][i] for i in range(3)])
        rel = lambda x, y: float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))
        out["max_rel_kl_obs_vs_hfc"] = max(out["max_rel_kl_obs_vs_hfc"], rel(k_obs, k_fc))
        out["max_rel_kl_hfc_vs_perstate"] = max(out["max_rel_kl_hfc_vs_perstate"], rel(k_fc, k_ps))
        out["max_rel_kl_hpre_vs_hfc"] = max(out["max_rel_kl_hpre_vs_hfc"], rel(k_pre, k_fc))
    return out


def ci95(x: np.ndarray) -> list[float]:
    h = stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def cohort_stats(vals: dict, cohorts: dict) -> dict:
    res = {}
    for coh in ("GM", "Non-GM", "init"):
        x = np.array([v for k, v in vals.items() if cohorts[k] == coh and v is not None and np.isfinite(v)])
        if len(x) < 2:
            continue
        res[coh] = {"mean": float(x.mean()), "sd": float(np.std(x, ddof=1)), "ci95_t_df2": ci95(x), "n_seeds": len(x),
                    "seed_values": x.tolist()}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, default=DATA_ROOT / "checkpoints/maze")
    ap.add_argument("--train-root", type=Path, default=REPO / "external/train-procgen-pytorch")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--q-rel-tol", type=float, default=1e-4)
    ap.add_argument("--p-tol", type=float, default=1e-5)
    args = ap.parse_args()

    prod = {(r["agent"], r["tap"]): read_csv(PROD_ROWS / f"rows_maze_{r['tap']}_{r['agent']}.csv")
            for r in [{"agent": a, "tap": t} for a in AGENTS for t in ("fc", "block3")]}
    perstate_rows, kl_t1, checks_q = [], {}, {}
    for agent, coh in AGENTS.items():
        rows = read_csv(PERSTATE / f"perstate_maze_{agent}.csv")
        fc = {int(r["state"]): r for r in rows if r["tap"] == "h_fc"}
        pre = {int(r["state"]): r for r in rows if r["tap"] == "h_pre"}
        pq = {int(r["frame"]): r for r in prod[(agent, "fc")]}
        qrel, nrel, kpre = [], [], []
        kl_t1[agent] = {}
        for i in sorted(fc):
            r, p = fc[i], pq[i]
            qrel.append(abs(float(r["Q"]) - float(p["Q"])) / (abs(float(p["Q"])) + 1e-12))
            nrel.append(abs(float(r["norm_delta"]) - float(p["norm_delta"])) / float(p["norm_delta"]))
            k = float(r["KL_t1"])
            kpre.append(abs(float(pre[i]["KL_t1"]) - k) / max(k, 1e-12))
            kl_t1[agent][i] = k
            perstate_rows.append({"env": "maze", "cohort": coh, "agent": agent, "frame": i, "level": r["state_key"],
                                  "KL_exact": k, "Q_hfc": float(p["Q"]), "halfQ_hfc": 0.5 * float(p["Q"]),
                                  "norm_delta_hfc": float(p["norm_delta"]),
                                  "Q_match_prod": qrel[-1] <= args.q_rel_tol})
        checks_q[agent] = {"n": len(fc), "n_Q_match": int(np.sum(np.array(qrel) <= args.q_rel_tol)),
                           "max_rel_Q_err": float(max(qrel)), "max_rel_norm_delta_err": float(max(nrel)),
                           "max_rel_KL_hpre_vs_hfc": float(max(kpre))}

    ver = verify_dump(args.ckpt_dir, args.train_root, kl_t1)
    if max(ver["max_abs_p_err_hfc"], ver["max_abs_p_err_hpre"], ver["max_abs_p_err_obs"]) > args.p_tol:
        raise SystemExit(f"stored p not reproduced to {args.p_tol}: {ver}")
    if any(c["n_Q_match"] != c["n"] for c in checks_q.values()):
        raise SystemExit(f"per-state rows do not match production Q: {checks_q}")

    per_ckpt = []
    for agent, coh in AGENTS.items():
        rs = [r for r in perstate_rows if r["agent"] == agent]
        k = np.array([r["KL_exact"] for r in rs])
        q = np.array([r["Q_hfc"] for r in rs])
        fork = None
        if agent in FORK_FILE:
            fork = json.loads((FORKS / f"{FORK_FILE[agent]}.json").read_text())["features"]["cheese"]["kl"]
        per_ckpt.append({"env": "maze", "cohort": coh, "agent": agent, "n_bank": len(rs),
                         "KL_bank_mean": float(k.mean()), "KL_bank_median": float(np.median(k)),
                         "Q_hfc_bank_median": float(np.median(q)), "halfQ_hfc_bank_median": float(0.5 * np.median(q)),
                         "Q_hfc_bank_mean": float(q.mean()),
                         "KL_fork_mean": fork["mean"] if fork else None, "KL_fork_median": fork["median"] if fork else None,
                         "n_fork": fork["n"] if fork else None})
    coh_of = {r["agent"]: r["cohort"] for r in per_ckpt}
    cohort = {m: cohort_stats({r["agent"]: r[m] for r in per_ckpt}, coh_of)
              for m in ("KL_bank_mean", "KL_bank_median", "Q_hfc_bank_median", "Q_hfc_bank_mean",
                        "KL_fork_mean", "KL_fork_median")}

    coin = json.loads(COIN_KL.read_text())
    coin_ck = {}
    for r in coin["rows"]:
        if r["ok"]:
            coin_ck.setdefault(r["ckpt"], []).append(r["kl"])
    coin_per = {c: {"n": len(v), "KL_mean": float(np.mean(v)), "KL_median": float(np.median(v))} for c, v in coin_ck.items()}
    coin_coh = {c: ("GM" if c.startswith("gm") else "Non-GM") for c in coin_per}
    coin_out = {
        "bank_freeze_kl": coin["summary"]["bank"]["freeze_sha256"], "n_states": coin["summary"]["bank"]["n_states"],
        "note": "Coin KL and Coin Q are both on the frozen 600-state "
                "canonical bank c0961932; no same-bank recompute needed. Only the state aggregation differs "
                "(KL: mean over states, Q: median over states).",
        "per_ckpt": coin_per,
        "cohort": {m: cohort_stats({c: v[m] for c, v in coin_per.items()}, coin_coh) for m in ("KL_mean", "KL_median")},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "kl_same_bank_perstate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(perstate_rows[0]))
        w.writeheader()
        w.writerows(perstate_rows)
    with open(args.out / "kl_same_bank.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_ckpt[0]))
        w.writeheader()
        w.writerows(per_ckpt)
    res = {"definition": "KL(pi(.|s) || pi(.|s_e)), s_e = cheese relocated to the reachable bl extreme, exact through "
                         "logits = W_pi h_fc + b_pi; states = 128-frame GM-1080 gm_rollouts decision-fork bank "
                         "(job 28694, CPU re-dump job 28781). Per checkpoint: mean (and median) over the 128 states; "
                         "cohort: mean, SD (ddof=1) and 95% t-CI over the 3 seed values.",
           "sources": {"perstate": str(PERSTATE.relative_to(REPO)), "production_rows": str(PROD_ROWS.relative_to(REPO)),
                       "fork_kl": str(FORKS.relative_to(REPO)), "dump": str(GEOM.relative_to(REPO))},
           "verification": {"dump_frames_0_2": ver, "all_frames_vs_job28694": checks_q},
           "maze": {"per_ckpt": per_ckpt, "cohort": cohort}, "coin": coin_out}
    (args.out / "kl_same_bank.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"verification": res["verification"], "cohort": cohort, "coin_cohort": coin_out["cohort"]}, indent=1))


if __name__ == "__main__":
    main()
