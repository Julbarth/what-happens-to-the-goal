"""Policy-Fisher input saliency for the Isaac quadruped and its mass on the flag and the tower."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from gm_robotics.isaac.analysis import common_isaac as C

GM, CT, RND = "#7A1520", "#3D8B8B", "#8E8E8E"


def mu_from_x(a, x, state):
    p = C.parts(a)
    h = x.permute(0, 3, 1, 2)
    for k in ("conv1", "conv2", "conv3"):
        h = torch.relu(p[k](h))
    h_fc = torch.relu(p["fc"](h.flatten(1)))
    return C.mu_from_fc(a, h_fc, p["state"](state))


def saliency(a, rgb, state, dev, bs=64):
    """[n, H, W] diagonal input Fisher J_mu^T diag(1/sigma^2) J_mu w.r.t. rgb/255; sigma is state independent."""
    w = 1.0 / C.sigma(a) ** 2
    out = []
    for i in range(0, len(rgb), bs):
        x = (C.to_t(rgb[i:i + bs], dev).float() / 255).requires_grad_(True)
        st = C.to_t(state[i:i + bs], dev).float()
        mu = mu_from_x(a, x, st)
        s = torch.zeros(x.shape[:3], device=dev)
        for k in range(mu.shape[1]):
            g, = torch.autograd.grad(mu[:, k].sum(), x, retain_graph=k < mu.shape[1] - 1)
            s += w[k] * (g ** 2).sum(-1)
        out.append(s.detach().cpu().numpy())
    return np.concatenate(out)


def object_mask(base, without, rerender, margin):
    """Pixels where removing the object changes the image by more than the re-render noise plus a margin, dilated by 1 px."""
    d = np.abs(base.astype(int) - without.astype(int)).sum(-1)
    noise = np.abs(base.astype(int) - rerender.astype(int)).sum(-1)
    m = d > noise + margin
    out = []
    for x in m:
        lab, n = ndimage.label(x)
        if n:
            # Removal also flips scattered floor-grid pixels; the object itself is the largest connected blob.
            x = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        out.append(ndimage.binary_dilation(x, iterations=1))
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="edit_visible")
    p.add_argument("--margin", type=int, default=60, help="summed-RGB change above re-render noise that counts as object")
    p.add_argument("--min_px", type=int, default=5, help="states need at least this many flag pixels")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_examples", type=int, default=4)
    a = p.parse_args()
    dev = C.device()
    b = C.load_bank(a.bank)
    flag = object_mask(b["rgb_base"], b["rgb_no_flag"], b["rgb_rerender"], a.margin)
    tower = object_mask(b["rgb_base"], b["rgb_no_tower"], b["rgb_rerender"], a.margin)
    flag &= ~tower
    keep = flag.reshape(len(flag), -1).sum(1) >= a.min_px
    print(f"states with >= {a.min_px} flag px: {keep.sum()} / {len(keep)}")
    rgb, st, flag, tower = b["rgb_base"][keep], b["state_base"][keep], flag[keep], tower[keep]
    out = os.path.join(C.RESULTS, "fisher_saliency")
    os.makedirs(out, exist_ok=True)
    rows, maps = [], {}
    for cohort, seed, ck in C.agent_list(a.seeds, a.seeds, tuple(a.seeds)):
        S = saliency(C.build_agent(ck, seed, dev), rgb, st, dev)
        maps[(cohort, seed)] = S
        tot = S.reshape(len(S), -1).sum(1)
        npx = S.shape[1] * S.shape[2]
        # Trained agents carry a state-independent horizon band; dividing by the cohort-seed mean map at the same pixels
        # asks whether the object draws saliency beyond what its image position gets on average.
        Sn = S / tot[:, None, None]
        pos = Sn.mean(0)
        for nm, m in (("flag", flag), ("tower", tower)):
            area = m.reshape(len(m), -1).sum(1)
            mass = (S * m).reshape(len(S), -1).sum(1) / tot
            for i in range(len(S)):
                exp_i = (pos * m[i]).sum()
                rows.append(dict(cohort=cohort, seed=seed, state=i, region=nm, mass=mass[i], area=area[i],
                                 enrich=mass[i] * npx / max(area[i], 1), locked=mass[i] / exp_i if exp_i > 0 else np.nan))
        print(cohort, seed, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_state.csv"), index=False)
    per_seed = df.groupby(["cohort", "seed", "region"])[["mass", "enrich", "locked"]].median().reset_index()
    per_seed.to_csv(os.path.join(out, "per_seed.csv"), index=False)
    lines = ["# Isaac policy-Fisher input saliency (visible bank, base frames); per-seed median over states, mean [95% t-CI] over seeds", ""]
    lines.append(f"States: {keep.sum()} of {len(keep)} with >= {a.min_px} flag pixels. Masks: removal edit minus re-render noise (+{a.margin}), dilated 1 px.")
    lines += ["", "| cohort | flag mass | tower mass | flag enrichment | tower enrichment | flag, position-corrected | tower, position-corrected |", "|---|---|---|---|---|---|---|"]
    for c in ("gm", "control", "init"):
        q = per_seed[per_seed.cohort == c]
        cell = lambda r, k: C.fmt_ci(*C.ci95(q[q.region == r][k].to_numpy()))
        lines.append(f"| {c} | {cell('flag', 'mass')} | {cell('tower', 'mass')} | {cell('flag', 'enrich')} | {cell('tower', 'enrich')} | {cell('flag', 'locked')} | {cell('tower', 'locked')} |")
    area = df[(df.cohort == "gm") & (df.seed == a.seeds[0])].groupby("region").area.median()
    lines += ["", f"Median region area (px of 9216): flag {area['flag']:.0f}, tower {area['tower']:.0f}. Enrichment = per-pixel saliency in the region / image mean (1 = no preference). Position-corrected = saliency share in the region / the share those pixels get on average over all states for the same agent (1 = the object adds nothing beyond its image position)."]
    open(os.path.join(out, "fisher_saliency_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Examples: first states (bank order) with a clearly visible flag (>= 20 px), not chosen by saliency.
    fpx = flag.reshape(len(flag), -1).sum(1)
    ex = np.where(fpx >= 20)[0][: a.n_examples]
    fig, axes = plt.subplots(len(ex), 4, figsize=(9, 2.35 * len(ex)))
    s0 = a.seeds[0]
    for r, i in enumerate(ex):
        axes[r, 0].imshow(rgb[i])
        for k, (c, lab) in enumerate((("gm", "GM"), ("control", "control"), ("init", "random network")), start=1):
            S = maps[(c, s0)][i]
            axes[r, k].imshow(rgb[i].mean(-1), cmap="gray", alpha=0.35)
            axes[r, k].imshow(S / S.max(), cmap="magma", alpha=0.75, vmin=0, vmax=1)
            if r == 0:
                axes[r, k].set_title(f"{lab} (seed {s0})", fontsize=9)
        for k in range(4):
            axes[r, k].contour(flag[i], levels=[0.5], colors=["#2ca02c"], linewidths=1.2)
            axes[r, k].contour(tower[i], levels=[0.5], colors=["#ff7f0e"], linewidths=1.2, linestyles="dashed")
            axes[r, k].set_xticks([]), axes[r, k].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("policy camera (96 px)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "fig_fisher_saliency_examples.png"), dpi=160)

    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    for j, key in enumerate(("mass", "enrich")):
        for k, (c, col) in enumerate((("gm", GM), ("control", CT), ("init", RND))):
            for r, (reg, off) in enumerate((("flag", -0.18), ("tower", 0.18))):
                v = per_seed[(per_seed.cohort == c) & (per_seed.region == reg)][key].values
                ax[j].bar(k + off, v.mean(), 0.34, color=col, alpha=1 if reg == "flag" else 0.45,
                          edgecolor=col, label=f"{reg}" if k == 0 else None)
                ax[j].scatter(np.full(len(v), k + off), v, s=10, color="white", edgecolor="k", zorder=3)
        ax[j].set_xticks(range(3), ["GM", "control", "random"])
        ax[j].set_title("share of saliency" if key == "mass" else "enrichment (1 = image mean)", fontsize=10)
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[1].axhline(1, color="#999", lw=0.8, ls=":")
    ax[1].set_yscale("log")
    ax[0].legend(["flag (solid)", "tower (faint)"], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "fig_fisher_saliency_bars.png"), dpi=160)


if __name__ == "__main__":
    main()
