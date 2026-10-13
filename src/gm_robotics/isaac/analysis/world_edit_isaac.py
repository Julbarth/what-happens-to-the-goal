"""Isaac quadruped world edits: KL, signed follow, and Fisher Q = G * D_F at h_pre and h_fc."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from gm_robotics.isaac.analysis import common_isaac as C
from gm_robotics.isaac.analysis import geom

# edit -> (role, favoured candidate after the edit: 'u' = flag's base location, 'v' = tower's base location, None)
EDITS = {"flag_swap": ("goal", "v"), "tower_swap": ("proxy", "u"), "no_flag": ("goal", "v"), "no_tower": ("proxy", "u"),
         "box_move": ("nuisance", None), "rerender": ("null", None)}
SITES = ("h_pre", "h_fc")
BANKS = ("visible", "conflict", "far")


def follow_probs(mu, sig, e_xy, turn_sign):
    """P(sampled command has a positive planar component along e_xy) and P(sampled yaw rate has sign turn_sign).

    First order in the Gaussian policy; the [-1, 1] action clamp and the low-level tracking error are ignored."""
    m = (mu[:, :2] * e_xy).sum(1)
    s = np.sqrt(((sig[None, :2] * e_xy) ** 2).sum(1))
    return norm.cdf(m / s), norm.cdf(turn_sign * mu[:, 2] / sig[2])


def geometry(bank):
    s, yaw = bank["start"], bank["yaw"]
    ru, rv = geom.robot_frame(bank["flag"], s, yaw), geom.robot_frame(bank["tower"], s, yaw)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    bu, bv = np.arctan2(ru[:, 1], ru[:, 0]), np.arctan2(rv[:, 1], rv[:, 0])
    return dict(e_u=unit(ru), e_v=unit(rv), e_uv=unit(ru - rv), turn_uv=np.sign(geom.wrap(bu - bv)))


@torch.no_grad()
def agent_rows(agent, bank, dev, bs=128):
    sig = C.sigma(agent)
    sig_np = sig.cpu().numpy()
    n = len(bank["start"])
    cache = {}

    def run(key):
        if key not in cache:
            outs = {k: [] for k in ("mu", "h_pre", "h_fc", "h_state")}
            for sl in C.batched(n, bs):
                # A world edit leaves the robot untouched, so proprio is the base state's for every edit (the per-pass
                # settling step adds small velocity differences that are not part of the edit).
                t = C.taps(agent, C.to_t(bank["rgb_" + key][sl], dev), C.to_t(bank["state_base"][sl], dev))
                for k in outs:
                    outs[k].append(t[k])
            cache[key] = {k: torch.cat(v) for k, v in outs.items()}
        return cache[key]

    g = geometry(bank)
    b = run("base")
    mb = b["mu"].cpu().numpy()
    rows = []
    for edit, (role, fav) in EDITS.items():
        e = run(edit)
        me = e["mu"].cpu().numpy()
        r = dict(edit=edit, role=role, state=np.arange(n), KL=C.gaussian_kl(b["mu"], e["mu"], sig).cpu().numpy(),
                 sep=np.linalg.norm(bank["flag"] - bank["tower"], axis=1),
                 flag_px_base=bank["flag_px_base"], flag_px_edit=bank[f"flag_px_{edit}"],
                 tower_px_base=bank["tower_px_base"], tower_px_edit=bank[f"tower_px_{edit}"])
        for k in range(3):
            r[f"mu{k}_base"], r[f"mu{k}_edit"] = mb[:, k], me[:, k]
        if fav is not None:
            e_fav = g["e_u"] if fav == "u" else g["e_v"]
            sgn = 1.0 if fav == "u" else -1.0     # differential direction: favoured minus the other candidate
            for tag, mm in (("base", mb), ("edit", me)):
                p_abs, _ = follow_probs(mm, sig_np, e_fav, 0)
                p_dif, p_turn = follow_probs(mm, sig_np, sgn * g["e_uv"], sgn * g["turn_uv"])
                r[f"pA_{tag}"], r[f"pD_{tag}"], r[f"pT_{tag}"] = p_abs, p_dif, p_turn
            for k in ("A", "D", "T"):
                r[f"SF{k}"] = r[f"p{k}_edit"] - r[f"p{k}_base"]
        for site in SITES:
            terms = {k: [] for k in ("Q", "G", "D_F", "trF_per_d", "dnorm2")}
            for sl in C.batched(n, bs):
                h, hs = b[site][sl], b["h_state"][sl]
                with torch.enable_grad():
                    J = C.mean_jacobian(agent, site, h, hs)
                ft = C.fisher_terms(J, sig, e[site][sl] - h)
                for k in terms:
                    terms[k].append(ft[k])
            for k, v in terms.items():
                r[f"{k}_{site}"] = torch.cat(v).cpu().numpy()
        r["calib_t1_h_fc"] = r["KL"] / np.maximum(0.5 * r["Q_h_fc"], 1e-30)
        rows.append(pd.DataFrame(r))
    return pd.concat(rows, ignore_index=True)


def summarize(df):
    beh = ["KL"] + [c for c in df.columns if c.startswith(("SF", "pA_", "pD_", "pT_"))]
    fis = [f"{k}_{s}" for s in SITES for k in ("Q", "G", "D_F", "trF_per_d", "dnorm2")] + ["calib_t1_h_fc"]
    ps_b, sm_b = C.seed_summary(df, beh, ["bank", "edit"], agg="mean")
    ps_f, sm_f = C.seed_summary(df, fis, ["bank", "edit"], agg="median")
    return (ps_b.merge(ps_f, on=["bank", "edit", "cohort", "seed"]),
            sm_b.merge(sm_f.drop(columns=["n_seeds"]), on=["bank", "edit", "cohort"]))


def write_md(summ, path):
    cols = ["KL", "SFA", "SFD", "SFT", "pA_base", "pD_base"] + \
           [f"{k}_{s}" for s in ("h_fc", "h_pre") for k in ("Q", "G", "D_F", "dnorm2", "trF_per_d")]
    lines = ["# Isaac quadruped world edits", "",
             "Mean and 95% t-CI across seeds. KL / follow: per-seed mean over states; Fisher terms: per-seed median. "
             "SFA: follow toward the favoured candidate (absolute direction); SFD: along favoured minus other; "
             "SFT: yaw rate toward the favoured side. pA_base / pD_base: base-state follow probabilities.", ""]
    for bank, sb in summ.groupby("bank", sort=False):
        lines += [f"## bank {bank}", "", "| edit | cohort | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
        for _, r in sb.sort_values(["edit", "cohort"]).iterrows():
            lines.append(f"| {r['edit']} | {r['cohort']} | {int(r['n_seeds'])} | " +
                         " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi']) if c in r and pd.notna(r[c]) else "" for c in cols) + " |")
        lines.append("")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--banks", nargs="+", default=list(BANKS))
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--redo", action="store_true")
    a = p.parse_args()
    C.ensure_dirs()
    dev = C.device()
    raw_dir = os.path.join(C.RESULTS, "worldedit_raw")
    os.makedirs(raw_dir, exist_ok=True)
    dfs = []
    for bname in a.banks:
        bank = C.load_bank(f"edit_{bname}")
        for cohort, seed, ck in C.agent_list(a.seeds_gm, a.seeds_control, a.init_seeds):
            out = os.path.join(raw_dir, f"{bname}_{cohort}_s{seed}.csv")
            if os.path.exists(out) and not a.redo:
                df = pd.read_csv(out)
            else:
                df = agent_rows(C.build_agent(ck, init_seed=seed, dev=dev), bank, dev)
                df.insert(0, "seed", seed)
                df.insert(0, "cohort", cohort)
                df.insert(0, "bank", bname)
                df.to_csv(out, index=False)
            print(f"{bname} {cohort} s{seed}: " + ", ".join(
                f"{e} KL {g['KL'].mean():.3f} Qfc {g['Q_h_fc'].median():.3g}" for e, g in df.groupby("edit")), flush=True)
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    per_seed, summ = summarize(df)
    per_seed.to_csv(os.path.join(C.RESULTS, "worldedit_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(C.RESULTS, "worldedit_summary.csv"), index=False)
    write_md(summ, os.path.join(C.RESULTS, "worldedit_summary.md"))
    print(open(os.path.join(C.RESULTS, "worldedit_summary.md")).read())


if __name__ == "__main__":
    main()
