"""World-edit behavioural dependence (KL, signed follow) and Fisher sensitivity Q = G * D_F at h_pre and h_fc."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from gm_robotics.analysis import common as C

# (base obs key, edited obs key, favoured-target key) per edit; obs keys index the bank's rgb_/state_ arrays.
EDITS = {
    "AT": {"shape_swap": ("base", "shape_swap", "v"), "colour_swap": ("base", "colour_swap", "u"),
           "shape_ablate": ("base", "shape_ablate", "v"), "colour_ablate": ("base", "colour_ablate", "u")},
    "LR": {"goal_move": ("g_d", "g2_d", "goal_g2_d"), "goal_to_corner": ("g_d", "corner_d", "goal_corner_d"),
           "corner_to_goal": ("corner_d", "g_d", "goal_g_d"), "distractor_move": ("g_d", "g_d2", "dxy_g_d2")},
    "SZ": {"mark_swap": ("base", "mark_swap", "v"), "colour_swap": ("base", "colour_swap", "u"),
           "mark_ablate": ("base", "mark_ablate", "v"), "colour_ablate": ("base", "colour_ablate", "u")},
    # PTM: SF target of marker_move = far point along p(m2) - p(m) (build_banks.build_ptm_worldedit); the other edits
    # leave the push direction unchanged (or remove it), so they have no signed-follow target.
    "PTM": {"marker_move": ("base", "m2", "shift_m2"), "marker_to_centre": ("base", "centre", "nan3"),
            "marker_hide": ("base", "nomarker", "nan3"), "cube_yaw": ("base", "yaw", "nan3")},
}
SITES = ("h_pre", "h_fc")


def target_xyz(bank, env, key):
    if env in ("AT", "SZ"):
        xy = bank[key]
        return np.concatenate([xy, np.full((len(xy), 1), 0.03, np.float32)], 1)
    t = bank[key]
    if t.shape[1] == 2:
        return np.concatenate([t, np.full((len(t), 1), 0.02, np.float32)], 1)
    return t


def p_toward(mu, sig, Jt, unit):
    """P(a sampled action moves the TCP toward the target), first order: the TCP step J (0.1 a_arm) is Gaussian.

    Action clipping to [-1, 1] and PD tracking error are ignored. Also returns the mean step toward the target (m)."""
    s = C.JOINT_DELTA_SCALE
    g = np.einsum("bk,bkj->bj", unit, Jt)
    m = (g * (s * mu[:, :7])).sum(1)
    v = ((g * s * sig[None, :7]) ** 2).sum(1)
    return norm.cdf(m / np.sqrt(v)), m


@torch.no_grad()
def agent_rows(agent, bank, env, dev, bs, t_small=0.25):
    sig = C.sigma(agent)
    sig_np = sig.cpu().numpy()
    tcp, Jt = bank["tcp"], bank["J_tcp"]
    n = len(tcp)
    cache = {}

    def run(key):
        if key not in cache:
            outs = {k: [] for k in ("mu", "h_pre", "h_fc", "h_state")}
            for sl in C.batched(n, bs):
                t = C.taps(agent, C.to_t(bank["rgb_" + key][sl], dev), C.to_t(bank["state_" + key][sl], dev))
                for k in outs:
                    outs[k].append(t[k])
            cache[key] = {k: torch.cat(v) for k, v in outs.items()}
        return cache[key]

    rows = []
    for edit, (kb, ke, kt) in EDITS[env].items():
        b, e = run(kb), run(ke)
        kl = C.gaussian_kl(b["mu"], e["mu"], sig).cpu().numpy()
        tgt = target_xyz(bank, env, kt)
        unit = (tgt - tcp) / np.linalg.norm(tgt - tcp, axis=1, keepdims=True)
        pb, mb = p_toward(b["mu"].cpu().numpy(), sig_np, Jt, unit)
        pe, me = p_toward(e["mu"].cpu().numpy(), sig_np, Jt, unit)
        r = dict(edit=edit, state=np.arange(n), KL=kl, SF=pe - pb, p_follow_base=pb, p_follow_edit=pe,
                 step_toward_base=mb, step_toward_edit=me)
        if f"npx_{edit}" in bank:
            r["npx"] = bank[f"npx_{edit}"]
        for site in SITES:
            terms = {k: [] for k in ("Q", "G", "D_F", "trF_per_d", "dnorm2")}
            kl_small = []
            for sl in C.batched(n, bs):
                h, hs = b[site][sl], b["h_state"][sl]
                delta = e[site][sl] - h
                with torch.enable_grad():
                    J = C.mean_jacobian(agent, site, h, hs)
                ft = C.fisher_terms(J, sig, delta)
                for k in terms:
                    terms[k].append(ft[k])
                f = C.mu_from_fc if site == "h_fc" else C.mu_from_pre
                kl_small.append(C.gaussian_kl(b["mu"][sl], f(agent, h + t_small * delta, hs), sig))
            for k, v in terms.items():
                r[f"{k}_{site}"] = torch.cat(v).cpu().numpy()
            r[f"KLt_{site}"] = torch.cat(kl_small).cpu().numpy()
            # Finite-edit calibration (paper: "Finite-edit calibration"): exact KL at t vs the quadratic t^2/2 Q.
            r[f"calib_t{t_small}_{site}"] = r[f"KLt_{site}"] / np.maximum(0.5 * t_small ** 2 * r[f"Q_{site}"], 1e-30)
        r["calib_t1_h_fc"] = kl / np.maximum(0.5 * r["Q_h_fc"], 1e-30)
        rows.append(pd.DataFrame(r))
    return pd.concat(rows, ignore_index=True)


def summarize(df, env):
    beh = ["KL", "SF", "p_follow_base", "p_follow_edit", "step_toward_base", "step_toward_edit"]
    fis = [f"{k}_{s}" for s in SITES for k in ("Q", "G", "D_F", "trF_per_d", "dnorm2")] + \
          [c for c in df.columns if c.startswith("calib_")]
    ps_b, sm_b = C.seed_summary(df, beh, ["edit"], agg="mean")
    ps_f, sm_f = C.seed_summary(df, fis, ["edit"], agg="median")
    per_seed = ps_b.merge(ps_f, on=["edit", "cohort", "seed"])
    summ = sm_b.merge(sm_f.drop(columns=["n_seeds"]), on=["edit", "cohort"])
    return per_seed, summ


def write_md(summ, env, path):
    # SF is not reported: it saturates at AT start states.
    cols = ["KL"] + [f"{k}_{s}" for s in ("h_fc", "h_pre") for k in ("Q", "G", "D_F", "dnorm2", "trF_per_d")]
    lines = [f"# World-edit dependence and Fisher sensitivity, {env}", "",
             "Mean and 95% t-CI across seeds. KL: per-seed mean over bank states. Fisher terms (Q, G, D_F, "
             "dnorm2 = |Delta|^2, trF_per_d = tr(F)/d sharpness): per-seed median over states.",
             "", "| edit | cohort | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
    for _, r in summ.sort_values(["edit", "cohort"]).iterrows():
        lines.append(f"| {r['edit']} | {r['cohort']} | {int(r['n_seeds'])} | " +
                     " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi']) for c in cols) + " |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "LR", "SZ", "PTM"], required=True)
    p.add_argument("--bank", default=None, help="bank name (default <env>_worldedit)")
    p.add_argument("--tag", default=None, help="output name (default <env>)")
    p.add_argument("--md_only", action="store_true", help="rewrite the markdown table from the summary CSV")
    p.add_argument("--seeds_gm", type=int, nargs="+", default=None)
    p.add_argument("--seeds_control", type=int, nargs="+", default=None)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max_states", type=int, default=None)
    a = p.parse_args()
    C.ensure_dirs()
    tag = a.tag or a.env
    if a.md_only:
        write_md(pd.read_csv(os.path.join(C.RESULTS, f"worldedit_{tag}_summary.csv")), tag,
                 os.path.join(C.RESULTS, f"worldedit_{tag}_summary.md"))
        return
    dev = C.device()
    bank = C.load_bank(a.bank or f"{a.env}_worldedit")
    if a.max_states:
        bank = {k: (v[: a.max_states] if v.ndim and len(v) == len(bank["tcp"]) else v) for k, v in bank.items()}
    raw_dir = os.path.join(C.RESULTS, "worldedit_raw")
    os.makedirs(raw_dir, exist_ok=True)
    dfs = []
    for cohort, seed, ck in C.agent_list(a.env, a.seeds_gm, a.seeds_control, a.init_seeds):
        out = os.path.join(raw_dir, f"{tag}_{cohort}_s{seed}.csv")
        if os.path.exists(out) and not a.max_states:
            df = pd.read_csv(out)
        else:
            agent = C.build_agent(ck, init_seed=seed, dev=dev)
            df = agent_rows(agent, bank, a.env, dev, a.batch)
            df.insert(0, "seed", seed)
            df.insert(0, "cohort", cohort)
            df.to_csv(out, index=False)
        print(f"{tag} {cohort} s{seed}: " + ", ".join(
            f"{e} KL {g['KL'].mean():.3f} Qfc {g['Q_h_fc'].median():.3f}" for e, g in df.groupby("edit")), flush=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    per_seed, summ = summarize(df, a.env)
    per_seed.to_csv(os.path.join(C.RESULTS, f"worldedit_{tag}_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(C.RESULTS, f"worldedit_{tag}_summary.csv"), index=False)
    write_md(summ, tag, os.path.join(C.RESULTS, f"worldedit_{tag}_summary.md"))
    print(open(os.path.join(C.RESULTS, f"worldedit_{tag}_summary.md")).read())


if __name__ == "__main__":
    main()
