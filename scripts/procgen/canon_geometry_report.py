"""Render geometry Tables 11-13, sign-confound tables and Figs 11-14 from canonical-bank geometry summaries."""
from __future__ import annotations

import os
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import canon_geometry_core as cg  # noqa: E402

PRIMARY = "ras_gt_bank_v2_split_state_pair_signd1_seed0.json"
EDIT_LABEL = {"swap": "SWAP edits (A.2.2 Eq. 13-14, primary)", "ablation": "ABLATION edits (supplementary)"}
CUR = {"edits": "swap"}
TAPS = ("h_pre", "h_fc")
MASKS = ("all", "d_gt_1")
NATIVE = {("gm", "black"), ("gmtex", "texture"), ("deco", "black"), ("decotex", "texture"),
          ("init", "black"), ("init", "texture")}
PAL = {"same": "#0072B2", "perpendicular": "#D55E00", "opposite": "#009E73"}


CI_SEP = "\t"


DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
def _num(x: float, nd: int) -> str:
    """Fixed-point, except 0 < |x| < 0.01, which prints in scientific notation so tiny values do not read as 0."""
    return f"{x:.1e}" if 0 < abs(x) < 0.01 else f"{x:.{nd}f}"


def fmt(ci: dict, nd: int = 3) -> str:
    """Paper format 'mean ± SD' (X.XXX ± X.XX), then CI_SEP and the 95% t-CI, which md_table puts in a side column."""
    if not np.isfinite(ci["mean"]):
        return f"n/a{CI_SEP}n/a"
    if ci["n"] < 2:
        return f"{_num(ci['mean'], nd)} (n=1){CI_SEP}n/a"
    return (f"{_num(ci['mean'], nd)} ± {_num(ci['sd'], max(nd - 1, 1))}{CI_SEP}"
            f"[{ci['lo']:.{nd}f}, {ci['hi']:.{nd}f}]")


def group(summaries: list[dict]) -> dict:
    """(cohort, background, tap) -> list of per-seed summaries, seeds sorted."""
    g = defaultdict(list)
    for s in summaries:
        g[(s["cohort"], s["background"], s["tap"])].append(s)
    return {k: sorted(v, key=lambda s: s["seed"]) for k, v in sorted(g.items())}


def get(s: dict, path: str, mask: str):
    x = s["summary"][mask]
    for p in path.split("/"):
        x = x[p]
    return x


def ci_of(seeds: list[dict], path: str, mask: str) -> dict:
    return cg.t_ci([get(s, path, mask) for s in seeds])


def loo_mean(s: dict, path: str, mask: str) -> float:
    vals = [v for k, v in s["summary"][mask]["heldout"].items() if k != PRIMARY]
    out = []
    for v in vals:
        x = v
        for p in path.split("/"):
            x = x[p]
        out.append(x)
    return float(np.nanmean(out))


def md_table(header: list[str], rows: list[list[str]]) -> str:
    split = [any(CI_SEP in r[j] for r in rows) for j in range(len(header))]
    header = [x for h, s in zip(header, split) for x in ((h + " (mean ± SD)", h + " 95% CI") if s else (h,))]
    rows = [[x for c, s in zip(r, split) for x in ((c.split(CI_SEP) + [""])[:2] if s else (c,))] for r in rows]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build_tables(G: dict) -> str:
    out = []
    for mask in MASKS:
        tag = "full bank (196 states)" if mask == "all" else "d > 1 (174 states)"
        out.append(f"\n### Table 11 (within-state cue geometry), {tag}\n")
        rows = []
        for (coh, bg, tap), seeds in G.items():
            rows.append([coh if (coh, bg) in NATIVE else coh + " (cross-render, suppl.)", bg, tap,
                         ",".join(s["seed"] for s in seeds),
                         fmt(ci_of(seeds, "cos_e", mask)), fmt(ci_of(seeds, "cos_f", mask)),
                         fmt(ci_of(seeds, "cos_f_sign+1", mask)), fmt(ci_of(seeds, "cos_f_sign-1", mask))])
        out.append(md_table(["cohort", "bg", "tap", "seeds", "cos_E", "cos_F", "cos_F (UP, +1)",
                             "cos_F (RIGHT, -1)"], rows))

        out.append(f"\n### Table 12 (cross-state kappa by line-bearing relation), {tag}\n")
        out.append("Only two line bearings exist (N, E), so 'opposite' is empty and 'perpendicular' = cross-sign "
                   "(targets swapped).\n")
        rows = []
        for (coh, bg, tap), seeds in G.items():
            rows.append([coh, bg, tap] + [fmt(ci_of(seeds, f"block_{k}/{r}", mask)) for k in ("F", "E")
                                          for r in ("same", "perpendicular")]
                        + [str(get(seeds[0], "block_F/n_opposite", mask))])
        out.append(md_table(["cohort", "bg", "tap", "kF same", "kF perp(cross-sign)", "kE same",
                             "kE perp(cross-sign)", "n opposite"], rows))

        out.append(f"\n### Table 13 (low-rank Fisher control, isotropic null under each pair's metric), {tag}\n")
        rows = []
        for (coh, bg, tap), seeds in G.items():
            rows.append([coh, bg, tap] + [fmt(ci_of(seeds, f"lowrank/{k}", mask)) for k in (
                "real_abs", "null_abs", "real_gt_0.9", "null_gt_0.9", "real_gt_0.99", "null_gt_0.99")])
        out.append(md_table(["cohort", "bg", "tap", "real |kF|", "null |kF|", "real >0.9", "null >0.9",
                             "real >0.99", "null >0.99"], rows))

        out.append(f"\n### Sign-confound quantification, {tag}\n")
        for k in ("F", "E"):
            rows = []
            for (coh, bg, tap), seeds in G.items():
                rows.append([coh, bg, tap,
                             fmt(ci_of(seeds, f"signgap_{k}/gap", mask)),
                             fmt(ci_of(seeds, f"signgap_{k}/asymmetry", mask)),
                             fmt(ci_of(seeds, f"signgap_{k}/folded_frac_pos", mask)),
                             fmt(ci_of(seeds, f"vardec_{k}/r2_sign", mask)),
                             fmt(ci_of(seeds, f"vardec_{k}/r2_fine_increment", mask), 4),
                             "/".join(f"{get(s, f'vardec_{k}/perm_p', mask):.3f}" for s in seeds),
                             fmt(ci_of(seeds, f"rank1_{k}/r2_rank1", mask)),
                             fmt(ci_of(seeds, f"rank1_{k}/loading_sign_agreement", mask)),
                             fmt(ci_of(seeds, "frac_effective_sign_flipped", mask)),
                             fmt(ci_of(seeds, f"vardec_effsign_{k}/r2_sign", mask))])
            out.append(f"\nkappa_{k}:\n")
            out.append(md_table(["cohort", "bg", "tap", "gap same-cross", "same+cross", "frac s_i s_j k > 0",
                                 "R2 sign-only (in-sample)", "dR2 fine (in-sample)", "perm p per seed",
                                 "R2 rank-1 (in-sample)", "rank-1 loading = sign", "frac eff-sign flipped",
                                 "R2 eff-sign-only (in-sample)"], rows))
        out.append(f"\nLogit-contrast concentration (fraction of Fisher energy on 1[A_line]-1[A_gem]), {tag}:\n")
        rows = []
        for (coh, bg, tap), seeds in G.items():
            rows.append([coh, bg, tap] + [fmt(ci_of(seeds, k, mask)) for k in (
                "contrast_frac_dcomp", "contrast_frac_shape", "contrast_frac_colour", "frac_dcomp_favours_line")])
        out.append(md_table(["cohort", "bg", "tap", "d_comp", "Delta_shape", "Delta_colour",
                             "frac d_comp favours line"], rows))

        out.append(f"\n### Same-sign-only structure (swap-safe), {tag}\n")
        out.append("For swaps kappa(i, j) = -kappa(i, mirror(j)) by construction, so the cross-sign block is a negated copy "
                   "of the same-sign block (exact for kappa_E on black; approximate for kappa_F, whose pair metrics "
                   "differ; not on texture, where each mirror has a different background). 'mirror max dev' = "
                   "max |kappa(i,j) + kappa(i,mirror(j))|. These columns use same-sign pairs only. Fine = |d_i - d_j|, same layout, spatial similarity.\n")
        for k in ("F", "E"):
            rows = []
            for (coh, bg, tap), seeds in G.items():
                rows.append([coh if (coh, bg) in NATIVE else coh + "*", bg, tap,
                             f"{max(get(s, f'mirror_{k}/max_abs_dev', mask) for s in seeds):.1e}",
                             fmt(ci_of(seeds, f"within_sign_{k}/mean", mask)),
                             fmt(ci_of(seeds, f"within_sign_{k}/sd", mask)),
                             fmt(ci_of(seeds, f"within_sign_{k}/r2_fine", mask)),
                             "/".join(f"{get(s, f'within_sign_{k}/perm_p', mask):.3f}" for s in seeds),
                             fmt(ci_of(seeds, f"within_sign_{k}/r2_fine_heldout", mask)),
                             fmt(ci_of(seeds, f"within_sign_{k}/r2_fine_heldout_loo", mask)),
                             fmt(ci_of(seeds, f"rank1_within_sign_{k}", mask))])
            out.append(f"\nkappa_{k}:\n")
            out.append(md_table(["cohort", "bg", "tap", "mirror max dev", "same-sign mean", "same-sign SD (pairs)",
                                 "R2 fine in-sample", "perm p per seed", "R2 fine held-out (prim)",
                                 "R2 fine held-out (LOO)", "rank-1 R2 within sign"], rows))

        out.append(f"\n### Held-out (fit-only coefficients / axis), {tag}: primary split; LOO = mean of 11 folds\n")
        for k in ("F", "E"):
            rows = []
            for (coh, bg, tap), seeds in G.items():
                r = [coh, bg, tap]
                for m in ("sign", "full", "axis"):
                    r.append(fmt(ci_of(seeds, f"heldout/{PRIMARY}/{k}/r2_{m}_eval", mask)))
                for m in ("sign", "full", "axis"):
                    r.append(fmt(cg.t_ci([loo_mean(s, f"{k}/r2_{m}_eval", mask) for s in seeds])))
                for m in ("sign", "full", "axis"):
                    r.append(fmt(ci_of(seeds, f"heldout/{PRIMARY}/{k}/r2_{m}_evaleval", mask)))
                rows.append(r)
            out.append(f"\nkappa_{k} predictive R2 on pairs touching an eval state (eval-eval = both states eval, "
                       f"no fit-state loading involved):\n")
            out.append(md_table(["cohort", "bg", "tap", "sign (prim)", "sign+fine (prim)", "shared axis (prim)",
                                 "sign (LOO)", "sign+fine (LOO)", "shared axis (LOO)", "sign (prim, eval-eval)",
                                 "sign+fine (prim, eval-eval)", "shared axis (prim, eval-eval)"], rows))
        rows = []
        for (coh, bg, tap), seeds in G.items():
            rows.append([coh, bg, tap] + [fmt(ci_of(seeds, f"heldout/{PRIMARY}/{k}", mask)) for k in (
                "axis_cos_e_eval", "axis_cos_f_eval", "axis_sign_acc_eval")]
                + [fmt(cg.t_ci([loo_mean(s, k, mask) for s in seeds])) for k in (
                    "axis_cos_e_eval", "axis_cos_f_eval", "axis_sign_acc_eval")])
        out.append("\nShared sign-folded axis u (fit states only), evaluated on eval states:\n")
        out.append(md_table(["cohort", "bg", "tap", "s*cos_E(d,u) prim", "s*cos_F(d,u) prim", "sign acc prim",
                             "s*cos_E LOO", "s*cos_F LOO", "sign acc LOO"], rows))

        out.append(f"\n### Fig 14 within-relation r(spatial similarity, kappa), {tag}\n")
        rows = []
        for (coh, bg, tap), seeds in G.items():
            rows.append([coh, bg, tap] + [fmt(ci_of(seeds, f"spatialcorr_{k}/{r}", mask)) for k in ("F", "E")
                                          for r in ("same", "perpendicular")]
                        + [fmt(ci_of(seeds, f"spatialcorr_rf_{k}/{r}", mask)) for k in ("F", "E")
                           for r in ("same", "perpendicular")])
        out.append(md_table(["cohort", "bg", "tap", "rF same", "rF cross", "rE same", "rE cross",
                             "rF same (role-free)", "rF cross (role-free)", "rE same (role-free)",
                             "rE cross (role-free)"], rows))
    return "\n".join(out)


def _strip(ax, groups: list[tuple[str, list[float]]], color_of=None):
    for x, (lab, vals) in enumerate(groups):
        c = color_of(lab) if color_of else "#444444"
        ax.scatter(np.full(len(vals), x) + np.linspace(-0.08, 0.08, len(vals)), vals, s=18, color=c, alpha=0.7,
                   zorder=3)
        ci = cg.t_ci(vals)
        if ci["n"] >= 2:
            ax.errorbar(x + 0.2, ci["mean"], yerr=[[ci["mean"] - ci["lo"]], [ci["hi"] - ci["mean"]]], fmt="o",
                        ms=7, color=c, capsize=3, zorder=4)
        elif ci["n"] == 1:
            ax.plot(x + 0.2, ci["mean"], "o", ms=7, color=c)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def fig11(G: dict, bg: str, out: Path, mask: str) -> None:
    keys = [k for k in G if k[1] == bg]
    if not keys:
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    axes = axes.ravel()
    lab = lambda k: f"{k[0]}{'' if (k[0], k[1]) in NATIVE else '*'}\n{k[2]}"  # noqa: E731
    _strip(axes[0], [(lab(k), [get(s, "cos_e", mask) for s in G[k]]) for k in keys])
    _strip(axes[1], [(lab(k), [get(s, "cos_f", mask) for s in G[k]]) for k in keys])
    for ax, K in ((axes[2], "E"), (axes[3], "F")):
        for x, k in enumerate(keys):
            for off, r in ((-0.18, "same"), (0.18, "perpendicular")):
                vals = [get(s, f"block_{K}/{r}", mask) for s in G[k]]
                ax.scatter(np.full(len(vals), x + off - 0.05), vals, s=16, color=PAL[r], alpha=0.7, zorder=3)
                ci = cg.t_ci(vals)
                if ci["n"] >= 2:
                    ax.errorbar(x + off + 0.05, ci["mean"], yerr=[[ci["mean"] - ci["lo"]], [ci["hi"] - ci["mean"]]],
                                fmt="o", ms=6, color=PAL[r], capsize=3, zorder=4,
                                label=("same sign" if r == "same" else "cross sign") if x == 0 else None)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([lab(k) for k in keys], fontsize=9)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, frameon=False)
    titles = ["A  within-state cos_E(Δω, Δp)", "B  within-state cos_F(Δω, Δp)",
              "C  cross-state κ_E by sign relation", "D  cross-state κ_F by sign relation"]
    if CUR["edits"] == "swap":
        titles[2] += "\n(swap: cross ≈ −same by construction; exact for κ_E on black)"
        titles[3] += "\n(swap: cross ≈ −same by construction; approximate for κ_F)"
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=11)
    for ax in axes:
        ax.set_ylabel("cosine (seeds; mean ± 95% t-CI)", fontsize=11)
    fig.suptitle(f"[{EDIT_LABEL[CUR['edits']]}] Local cue opposition and cross-state organization, {bg} background, "
                 f"{'all 196 states' if mask == 'all' else 'd > 1'}\n* = cross-rendering (policy trained on the other "
                 f"background; supplementary). n = 3 seeds: t-CIs (x4.30) can exceed [-1, 1].", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _matrix_fig(mats: dict, st: dict, title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    sign, layout = st["sign"], st["layout"]
    cut = int((sign == 1).sum())
    lay_cuts = [i for i in range(1, len(layout)) if layout[i] != layout[i - 1] and i != cut]
    im = None
    for c, tap in enumerate(TAPS):
        for r, K in enumerate(("kappa_e", "kappa_f")):
            ax = axes[r, c]
            if tap not in mats:
                ax.axis("off")
                continue
            im = ax.imshow(mats[tap][K], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            for x in lay_cuts:
                ax.axhline(x - 0.5, color="0.6", lw=0.3)
                ax.axvline(x - 0.5, color="0.6", lw=0.3)
            ax.axhline(cut - 0.5, color="k", lw=1.4)
            ax.axvline(cut - 0.5, color="k", lw=1.4)
            ax.set_xticks([cut / 2, cut + (len(sign) - cut) / 2])
            ax.set_xticklabels(["line N (UP)", "line E (RIGHT)"], fontsize=10)
            ax.set_yticks([cut / 2, cut + (len(sign) - cut) / 2])
            ax.set_yticklabels(["line N", "line E"], fontsize=10)
            ax.set_title(f"{tap}: {'Euclidean κ_E' if K == 'kappa_e' else 'Fisher κ_F'}", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02).set_label("seed-mean similarity", fontsize=11)
    note = ("\nSwap: off-diagonal blocks ≈ −diagonal blocks by construction (d_comp(mirror) = −d_comp; exact for κ_E "
            "on black)") if CUR["edits"] == "swap" else ""
    fig.suptitle(f"[{EDIT_LABEL[CUR['edits']]}] " + title + "\nSeed-mean; single-seed deviations appear as pale stripes (see per-seed matrices)" + note,
                 fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def seed_mean_mats(data_dir: Path, G: dict, cohort: str, bg: str) -> dict:
    out = {}
    for tap in TAPS:
        seeds = G.get((cohort, bg, tap))
        if not seeds:
            continue
        acc = {}
        for s in seeds:
            z = np.load(data_dir / "per_seed" / f"matrices_{cohort}_{s['seed']}_{bg}_{tap}.npz")
            for K in ("kappa_e", "kappa_f"):
                acc.setdefault(K, []).append(z[K])
        out[tap] = {K: np.nanmean(v, axis=0) for K, v in acc.items()}
    return out


def fig14(data_dir: Path, G: dict, cohort: str, bg: str, st: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    rng = np.random.default_rng(0)
    rel = cg.relation_matrix(st["line_bearing"])
    n = len(st["sign"])
    iu, ju = np.triu_indices(n, k=1)
    sub = rng.choice(len(iu), size=min(4000, len(iu)), replace=False)
    for r, tap in enumerate(TAPS):
        seeds = G.get((cohort, bg, tap))
        if not seeds:
            continue
        z = [np.load(data_dir / "per_seed" / f"matrices_{cohort}_{s['seed']}_{bg}_{tap}.npz") for s in seeds]
        S = z[0]["spatial_sim"]
        for c, K in enumerate(("kappa_e", "kappa_f")):
            ax = axes[r, c]
            Km = np.nanmean([x[K] for x in z], axis=0)
            for name in ("same", "perpendicular"):
                m = rel[iu, ju][sub] == name
                ax.scatter(S[iu, ju][sub][m], Km[iu, ju][sub][m], s=3, alpha=0.35, color=PAL[name],
                           label="same sign" if name == "same" else "cross sign (perpendicular, swapped)")
            kk = "E" if K == "kappa_e" else "F"
            rs = {nm: cg.t_ci([get(s, f"spatialcorr_{kk}/{nm}", "all") for s in seeds])
                  for nm in ("same", "perpendicular")}
            ax.text(0.02, 0.03, f"r same {fmt(rs['same'], 2).replace(CI_SEP, ' ')}\n"
                    f"r cross {fmt(rs['perpendicular'], 2).replace(CI_SEP, ' ')}",
                    transform=ax.transAxes, fontsize=8, va="bottom")
            ax.set_title(f"{tap}: {'κ_E' if kk == 'E' else 'κ_F'}", fontsize=11)
            ax.axhline(0, color="k", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10)
    for ax in axes[1]:
        ax.set_xlabel("spatial similarity −‖z_i − z_j‖ (line/gem cells, d; z-scored)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("seed-mean pairwise similarity", fontsize=11)
    axes[0, 0].legend(fontsize=9, markerscale=4, loc="upper left")
    fig.suptitle(f"[{EDIT_LABEL[CUR['edits']]}] Fine spatial similarity vs sign relation, {cohort}, {bg} (4,000-pair subsample; r = per-seed "
                 f"mean ± SD [95% t-CI])", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_sign_confound(G: dict, out: Path, mask: str) -> None:
    """Same-sign-only diagnostics: under swaps the cross-sign block is a construction copy of the same-sign block,
    so full-matrix sign R^2 is not shown."""
    keys = list(G)
    fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    lab = lambda k: f"{k[0]}{'' if (k[0], k[1]) in NATIVE else '*'}\n{k[1][:3]}\n{k[2]}"  # noqa: E731
    panels = [("within_sign_F/mean", "A  same-sign mean κ_F (trained vs init baseline)"),
              ("within_sign_F/r2_fine_heldout", "B  within-sign R² of κ_F from |Δd|, layout, spatial sim (held-out, primary split)"),
              ("rank1_within_sign_F", "C  rank-1 (one axis, per-state loading) R² of same-sign κ_F (in-sample)"),
              ("contrast_frac_dcomp", "D  d_comp Fisher energy on the UP-vs-RIGHT logit contrast")]
    for ax, (path, title) in zip(axes, panels):
        _strip(ax, [(lab(k), [get(s, path, mask) for s in G[k]]) for k in keys])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("value (seeds; mean ± 95% t-CI)", fontsize=11)
    fig.suptitle(f"[{EDIT_LABEL[CUR['edits']]}] Sign-confound diagnostics on same-sign pairs ({mask}); "
                 f"* = cross-rendering, supplementary; n = 3 t-CIs can exceed [0, 1]", fontsize=13)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edits", choices=["swap", "ablation"], default="swap")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "paper_figures")
    ap.add_argument("--gm-cohorts", nargs="+", default=None, help="cohorts for Fig 12/14 (default: names with 'gm')")
    ap.add_argument("--control-cohorts", nargs="+", default=None, help="cohorts for Fig 13")
    args = ap.parse_args()
    CUR["edits"] = args.edits
    if args.data is None:
        args.data = DATA_ROOT / "probes/canon_geometry_20260924" / (
            "swap" if args.edits == "swap" else "ablation_supplementary")
    sfx = "swap" if args.edits == "swap" else "ablation_supp"
    from canon_geometry_run import BANK_DIR, load_state_table
    st = load_state_table(BANK_DIR / "ras_gt_bank_canonical.json")
    summaries = json.loads((args.data / "summaries.json").read_text())
    G = group(summaries)
    (args.data / "tables_canon.md").write_text(f"# Geometry tables, {EDIT_LABEL[args.edits]}\n" + build_tables(G))
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    cohorts = sorted({k[0] for k in G})
    gm = args.gm_cohorts or [c for c in cohorts if "gm" in c.lower() and "non" not in c.lower()]
    ctrl = args.control_cohorts or [c for c in cohorts if c not in gm and c != "init"]
    written = []
    for bg in sorted({k[1] for k in G}):
        for mask in MASKS:
            p = args.fig_dir / f"fig11_ras_cue_opposition_{bg}_{mask}_{sfx}_canon.png"
            fig11(G, bg, p, mask)
            written.append(p)
        for coh_list, figno in ((gm, 12), (ctrl, 13)):
            for coh in coh_list:
                mats = seed_mean_mats(args.data, G, coh, bg)
                if not mats:
                    continue
                n_seeds = len(G.get((coh, bg, "h_fc"), G.get((coh, bg, "h_pre"), [])))
                p = args.fig_dir / f"fig{figno}_ras_dcomp_similarity_{coh}_{bg}_{sfx}_canon.png"
                suppl = "" if (coh, bg) in NATIVE else " [CROSS-RENDERING, supplementary]"
                _matrix_fig(mats, st, f"Cross-state d_comp similarity, {coh}, {bg}{suppl} (mean of {n_seeds} seeds; "
                                      f"sorted by sign, then layout, then d; grey = layout blocks)", p)
                written.append(p)
                p = args.fig_dir / f"fig14_ras_spatial_vs_sign_{coh}_{bg}_{sfx}_canon.png"
                fig14(args.data, G, coh, bg, st, p)
                written.append(p)
    for mask in MASKS:
        p = args.fig_dir / f"figS_ras_sign_confound_{mask}_{sfx}_canon.png"
        fig_sign_confound(G, p, mask)
        written.append(p)
    print("\n".join(str(w) for w in written))
    print("Caption (Fig 12/13): pairwise similarity of d_comp(s) = Delta_colour(s) - Delta_shape(s) (points toward the line cell for swaps) over all 196 "
          "canonical conflict states, seed-mean. The agent always starts at (0,0), so line bearing is N (UP, sign "
          "+1) or E (RIGHT, sign -1); black lines split the two sign groups, grey lines the 11 wall layouts.")


if __name__ == "__main__":
    main()
