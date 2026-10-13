"""Policy-Fisher input saliency S(x) = sum_c sum_k (d mu_k / d o_{c,x})^2 / sigma_k^2 on the rgb input, and its mass in goal / proxy regions."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C

# Regions per env (all discs of the same pixel radius). LR: proxy = projected training corner, mirror = the marker
# reflected across the robot's x axis (equal-area off-target patch at the same distance from the base).
REGIONS = {"AT": ("goal", "proxy"), "SZ": ("goal", "proxy"), "LR": ("goal", "proxy", "mirror"),
           "PTM": ("goal", "proxy", "cube", "mirror")}
# PTM: the marker disc is ~5 px in radius and 9-10 px from the cube, so a smaller disc keeps regions separate.
RADIUS = {"AT": 8, "LR": 8, "SZ": 10, "PTM": 5}


def saliency(agent, rgb, state, dev, bs=64):
    """[n, H, W] diagonal input Fisher of the Gaussian policy w.r.t. the network input rgb/255 (proprio fixed).

    sigma is state independent, so the log-std term carries no input gradient and the Gaussian Fisher reduces to
    J_mu^T diag(1/sigma^2) J_mu."""
    w = 1.0 / C.sigma(agent) ** 2
    out = []
    for sl in C.batched(len(rgb), bs):
        x = (C.to_t(rgb[sl], dev).float() / 255).requires_grad_(True)
        st = C.to_t(state[sl], dev).float()
        mu = mu_from_x(agent, x, st)
        s = torch.zeros(x.shape[:3], device=dev)
        for k in range(mu.shape[1]):
            g, = torch.autograd.grad(mu[:, k].sum(), x, retain_graph=k < mu.shape[1] - 1)
            s += w[k] * (g ** 2).sum(-1)
        out.append(s.detach().cpu().numpy())
    return np.concatenate(out)


def mu_from_x(agent, x, state):
    """Same network as C.taps, but from the float NHWC input so the input gradient is defined."""
    p = C.parts(agent)
    h = x.permute(0, 3, 1, 2)
    for k in ("conv1", "conv2", "conv3"):
        h = torch.relu(p[k](h))
    h_fc = torch.relu(p["fc"](h.flatten(1)))
    return C.mu_from_fc(agent, h_fc, p["state"](state))


def region_masks(px, names, radius, hw=C.OBS_RES):
    """Exclusive disc masks [n, H, W] per region; earlier names take priority where discs overlap."""
    yy, xx = np.mgrid[:hw, :hw] + 0.5
    taken = np.zeros((len(px[names[0]]), hw, hw), bool)
    masks = {}
    for nm in names:
        c = px[nm]
        m = (xx[None] - c[:, 0, None, None]) ** 2 + (yy[None] - c[:, 1, None, None]) ** 2 <= radius ** 2
        masks[nm] = m & ~taken
        taken |= m
    return masks


def region_rows(S, masks, meta):
    tot = S.reshape(len(S), -1).sum(1)
    npx = S.shape[1] * S.shape[2]
    r = dict(meta)
    for nm, m in masks.items():
        area = m.reshape(len(m), -1).sum(1)
        mass = (S * m).reshape(len(S), -1).sum(1) / tot
        r[f"mass_{nm}"] = mass
        r[f"area_{nm}"] = area
        # Enrichment as in the paper: per-pixel saliency in the region over the image-mean per-pixel saliency.
        r[f"enrich_{nm}"] = np.where(area > 0, mass * npx / np.maximum(area, 1), np.nan)
    return pd.DataFrame(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "LR", "SZ", "PTM"], required=True)
    p.add_argument("--seeds_gm", type=int, nargs="+", required=True)
    p.add_argument("--seeds_control", type=int, nargs="+", required=True)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--n_maps", type=int, default=4, help="first N scenes whose full maps are stored for figures")
    a = p.parse_args()
    dev = C.device()
    radius = a.radius or RADIUS[a.env]
    sdir = os.path.join(C.RESULTS, "saliency")
    z = np.load(os.path.join(sdir, f"frames_{a.env}.npz"))
    names = REGIONS[a.env]
    px = {nm: z[f"px_{nm}"] for nm in names}
    masks = region_masks(px, names, radius)
    n = len(z["idx"])
    pairs = list(zip(a.seeds_gm, a.seeds_control, a.init_seeds))
    agents = {}
    for i, (sg, sc, si) in enumerate(pairs):
        agents[("gm", sg)] = (i, C.ckpt_path(a.env, "gm", sg))
        agents[("control", sc)] = (i, C.ckpt_path(a.env, "control", sc))
        agents[("init", si)] = (i, None)
    rows, maps = [], {}
    for (cohort, seed), (pair, ck) in agents.items():
        agent = C.build_agent(ck, init_seed=seed, dev=dev)
        sg, sc, _ = pairs[pair]
        # Start frames are shared by every agent. Mid-episode frames come from the GM and the Non-GM rollout of
        # this pair, and every agent of the pair is scored on both, so cohorts are compared on identical frames.
        frame_sets = {"start": (z["rgb0"], z["state0"]),
                      "mid_gm": (z[f"mid_gm_s{sg}_rgb"], z[f"mid_gm_s{sg}_state"]),
                      "mid_control": (z[f"mid_control_s{sc}_rgb"], z[f"mid_control_s{sc}_state"])}
        for fs, (rgb, st) in frame_sets.items():
            S = saliency(agent, rgb, st, dev)
            meta = dict(env=a.env, cohort=cohort, seed=seed, pair=pair, frames=fs, state=np.arange(n),
                        total=S.reshape(n, -1).sum(1))
            if fs != "start":
                meta["done_before_k"] = z[f"mid_{fs[4:]}_s{sg if fs == 'mid_gm' else sc}_done"]
            rows.append(region_rows(S, masks, meta))
            maps[f"{cohort}_s{seed}_{fs}"] = S[: a.n_maps].astype(np.float32)
        print(f"{a.env} {cohort} s{seed}: " + ", ".join(
            f"{fs} goal {d['mass_goal'].mean():.4f} proxy {d['mass_proxy'].mean():.4f}"
            for fs, d in pd.concat(rows[-3:]).groupby("frames")), flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["radius"] = radius
    df.to_csv(os.path.join(sdir, f"saliency_{a.env}_per_state.csv"), index=False)
    np.savez_compressed(os.path.join(sdir, f"maps_{a.env}.npz"), **maps)
    summarize(df, a.env, sdir)


def summarize(df, env, sdir):
    """Per seed: mean over states of the mass fractions and geometric mean of the enrichments; then mean and 95%
    t-CI across seeds. Mid-episode rows pool the GM-rollout and Non-GM-rollout frames of each pair."""
    df = df.copy()
    df["frames2"] = np.where(df.frames == "start", "start", "mid")
    names = [c[5:] for c in df.columns if c.startswith("mass_")]
    rows, per_seed = [], []
    for (fs, coh, seed), g in df.groupby(["frames2", "cohort", "seed"]):
        r = dict(frames=fs, cohort=coh, seed=seed, n_states=len(g))
        for nm in names:
            r[f"mass_{nm}"] = g[f"mass_{nm}"].mean()
            e = g[f"enrich_{nm}"].to_numpy()
            e = e[np.isfinite(e) & (e > 0)]
            r[f"log10_enrich_{nm}"] = np.log10(e).mean()
        r["mass_goal_minus_proxy"] = (g["mass_goal"] - g["mass_proxy"]).mean()
        per_seed.append(r)
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(os.path.join(sdir, f"saliency_{env}_per_seed.csv"), index=False)
    cols = [c for c in per_seed.columns if c.startswith(("mass_", "log10_"))]
    for (fs, coh), g in per_seed.groupby(["frames", "cohort"]):
        r = dict(frames=fs, cohort=coh, n_seeds=len(g))
        for c in cols:
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(g[c].to_numpy())
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(sdir, f"saliency_{env}_summary.csv"), index=False)
    with open(os.path.join(sdir, f"saliency_{env}_summary.md"), "w") as f:
        f.write(f"# Policy-Fisher input saliency, {env} (mean and 95% t-CI across seeds)\n\n"
                "mass_* = fraction of total saliency inside the region disc (per-seed mean over states); "
                "log10_enrich_* = log10 geometric mean over states of region per-pixel saliency / image mean.\n\n")
        f.write("| frames | cohort | n | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 3) + "\n")
        for _, r in summ.sort_values(["frames", "cohort"], ascending=[False, True]).iterrows():
            f.write(f"| {r['frames']} | {r['cohort']} | {r['n_seeds']} | " +
                    " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi'], 4) for c in cols) + " |\n")
    print(open(os.path.join(sdir, f"saliency_{env}_summary.md")).read())


if __name__ == "__main__":
    main()
