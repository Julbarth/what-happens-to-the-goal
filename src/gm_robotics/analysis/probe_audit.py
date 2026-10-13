"""Layerwise goal decodability under several preprocessings and targets, with linear and nonlinear (MLP, kNN) probes."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gm_robotics.analysis import common as C
from gm_robotics.analysis.probes import pixel_blocks

OUT = os.path.join(C.RESULTS, "probe_audit")
SITES = ("conv1", "conv2", "h_pre", "h_fc", "a_hid", "conv1_gap", "conv2_gap", "conv3_gap")
ALPHAS = tuple(10.0 ** np.arange(-3, 8))
PREPS = ("raw", "zfix", "zorig")
SEEDS = {"AT": dict(gm=[2, 3, 4], control=[2, 3, 4, 5]), "SZ": dict(gm=[0, 1, 2], control=[0, 1, 2]),
         "LR": dict(gm=[1, 2, 3], control=[1, 2, 3]), "PTM": dict(gm=[0, 1, 2], control=[0, 1, 2])}


# ---------------------------------------------------------------- targets

def targets(b, env):
    """{name: (y, kind)}; kind 'auc' (binary) or 'r2'. NaN entries in y are excluded from fit and score."""
    tcp = b["tcp_env"][:, :2]
    if env == "SZ":
        u, v, cg, cube = b["u"], b["v"], b["cross_green"].astype(bool), b["cube"][:, :2]
        green, purple = np.where(cg[:, None], u, v), np.where(cg[:, None], v, u)
        return {"cross_x": (u[:, 0], "r2"), "cross_y": (u[:, 1], "r2"), "cross_left": ((u[:, 1] > 0) * 1.0, "auc"),
                "cross_is_green": (cg * 1.0, "auc"), "cross_absy": (np.abs(u[:, 1]), "r2"),
                "cross_rel_cube_x": (u[:, 0] - cube[:, 0], "r2"), "cross_rel_cube_y": (u[:, 1] - cube[:, 1], "r2"),
                "cross_rel_tcp_x": (u[:, 0] - tcp[:, 0], "r2"), "cross_rel_tcp_y": (u[:, 1] - tcp[:, 1], "r2"),
                "green_x": (green[:, 0], "r2"), "green_y": (green[:, 1], "r2"), "green_left": ((green[:, 1] > 0) * 1.0, "auc"),
                "purple_x": (purple[:, 0], "r2"), "purple_y": (purple[:, 1], "r2"),
                "cube_x": (cube[:, 0], "r2"), "cube_y": (cube[:, 1], "r2")}
    if env == "AT":
        u, v, cr, sr = b["u"], b["v"], b["cube_red"].astype(bool), b["sphere_red"].astype(bool)
        one_red = cr ^ sr
        red = np.where(cr[:, None], u, v)
        red_other = np.where(cr[:, None], v, u)
        nan_unless = lambda x: np.where(one_red, x, np.nan)
        return {"cube_x": (u[:, 0], "r2"), "cube_y": (u[:, 1], "r2"), "cube_left": ((u[:, 1] > v[:, 1]) * 1.0, "auc"),
                "cube_is_red": (cr * 1.0, "auc"), "sphere_is_red": (sr * 1.0, "auc"),
                "cube_rel_tcp_x": (u[:, 0] - tcp[:, 0], "r2"), "cube_rel_tcp_y": (u[:, 1] - tcp[:, 1], "r2"),
                "cube_minus_sphere_x": (u[:, 0] - v[:, 0], "r2"), "cube_minus_sphere_y": (u[:, 1] - v[:, 1], "r2"),
                "sphere_x": (v[:, 0], "r2"), "sphere_y": (v[:, 1], "r2"),
                "red_x": (nan_unless(red[:, 0]), "r2"), "red_y": (nan_unless(red[:, 1]), "r2"),
                "red_left": (nan_unless((red[:, 1] > red_other[:, 1]) * 1.0), "auc")}
    if env == "PTM":
        from gm_robotics.analysis.envs_forced import PTM_C
        m, cube = b["marker"], b["cube"][:, :2]
        return {"marker_x": (m[:, 0], "r2"), "marker_y": (m[:, 1], "r2"),
                "marker_rel_cube_x": (m[:, 0] - cube[:, 0], "r2"), "marker_rel_cube_y": (m[:, 1] - cube[:, 1], "r2"),
                "bearing_x": (m[:, 0] - tcp[:, 0], "r2"), "bearing_y": (m[:, 1] - tcp[:, 1], "r2"),
                "marker_left": ((m[:, 1] > 0) * 1.0, "auc"),
                "marker_dist_centre": (np.linalg.norm(m - np.asarray(PTM_C)[:2], axis=1), "r2"),
                "cube_x": (cube[:, 0], "r2"), "cube_y": (cube[:, 1], "r2"), "tcp_x": (tcp[:, 0], "r2"), "tcp_y": (tcp[:, 1], "r2")}
    from gm_robotics.analysis.envs_forced import CORNER
    g, d = b["goal"][:, :2], b["dxy"]
    return {"marker_x": (g[:, 0], "r2"), "marker_y": (g[:, 1], "r2"),
            "bearing_x": (g[:, 0] - tcp[:, 0], "r2"), "bearing_y": (g[:, 1] - tcp[:, 1], "r2"),
            "marker_left": ((g[:, 1] > 0) * 1.0, "auc"), "marker_front": ((g[:, 0] > 0) * 1.0, "auc"),
            "marker_near_corner": ((np.linalg.norm(g - CORNER, axis=1) < 0.10) * 1.0, "auc"),
            "marker_dist_corner": (np.linalg.norm(g - CORNER, axis=1), "r2"),
            "distractor_x": (d[:, 0], "r2"), "distractor_y": (d[:, 1], "r2"),
            "tcp_x": (tcp[:, 0], "r2"), "tcp_y": (tcp[:, 1], "r2")}


def subsets(b, env):
    """Test-bank strata; probes are always fit on the whole train bank."""
    n = len(b["state"])
    if env == "SZ":
        cg = b["cross_green"].astype(bool)
        green_left = np.where(cg, b["u"][:, 1] > 0, b["v"][:, 1] > 0)
        return {"all": np.ones(n, bool), "aligned": cg, "conflict": ~cg, "green_left": green_left, "green_right": ~green_left}
    if env == "AT":
        cr, sr = b["cube_red"].astype(bool), b["sphere_red"].astype(bool)
        return {"all": np.ones(n, bool), "aligned": cr & ~sr, "conflict": ~cr & sr, "same_colour": cr == sr}
    if env == "PTM":
        return {"all": np.ones(n, bool)}
    from gm_robotics.analysis.envs_forced import CORNER
    return {"all": np.ones(n, bool), "far": np.linalg.norm(b["goal"][:, :2] - CORNER, axis=1) >= 0.10}


# ---------------------------------------------------------------- features

@torch.no_grad()
def features(agent, bank, dev, bs=256):
    out = {s: [] for s in SITES}
    for sl in C.batched(len(bank["state"]), bs):
        t = C.taps(agent, C.to_t(bank["rgb"][sl], dev), C.to_t(bank["state"][sl], dev))
        for s in SITES:
            x = t[s.replace("_gap", "")]
            if s.endswith("_gap"):
                x = x.mean((2, 3))
            out[s].append(x.flatten(1).float())
    return {s: torch.cat(v) for s, v in out.items()}


def preprocess(xtr, xte, kind):
    """Centre on the train mean; 'raw' = one global scale, 'zfix' = per-unit z-score with units whose train sd is
    < 1% of the median unit sd dropped, 'zorig' = probes.py's z-score (sd > 1e-8 kept: near-dead units can explode
    on the test bank). All statistics come from the train bank only."""
    mu = xtr.mean(0)
    xtr, xte = xtr - mu, xte - mu
    sd = xtr.std(0)
    if kind == "raw":
        s = sd.pow(2).sum().sqrt().clamp_min(1e-12) / np.sqrt(xtr.shape[1])
        return xtr / s, xte / s
    keep = sd > (1e-8 if kind == "zorig" else 0.01 * sd[sd > 1e-8].median())
    return xtr[:, keep] / sd[keep], xte[:, keep] / sd[keep]


def train_span(xtr, xte):
    """Exact rotation onto the train span (rank <= n): ridge solutions and kNN rankings are unchanged."""
    g = (xtr @ xtr.T).double()
    ev, U = torch.linalg.eigh(g)
    keep = ev > ev.max() * 1e-9
    ev, U = ev[keep].flip(0), U[:, keep].flip(1)
    s = ev.sqrt()
    V = (xtr.double().T @ U) / s
    return (U * s).float(), (xte.double() @ V).float(), ev.float()


# ---------------------------------------------------------------- probes

def score(y, p, kind):
    return C.gmg_linear.roc_auc(y, p) if kind == "auc" else C.gmg_linear.r2_score(y, p)


class DualRidge:
    """Ridge in dual form on one design (rows = train points with a finite label); fold eigendecompositions do not
    depend on y, so one instance serves every target with the same label mask and every alpha."""

    def __init__(self, ztr, ok, dev, folds=5, seed=0):
        self.idx = np.nonzero(ok)[0]
        X = ztr[self.idx].to(dev).double()
        self.X = X
        fold = np.random.default_rng(seed).integers(folds, size=len(X))
        self.folds = []
        for f in range(folds):
            tr, va = np.nonzero(fold != f)[0], np.nonzero(fold == f)[0]
            ev, U = torch.linalg.eigh(X[tr] @ X[tr].T)
            self.folds.append((tr, va, ev.clamp_min(0), U, X[va] @ X[tr].T))
        self.ev, self.U = torch.linalg.eigh(X @ X.T)
        self.ev = self.ev.clamp_min(0)

    def cv(self, y, kind):
        y = torch.as_tensor(y[self.idx], dtype=torch.float64, device=self.X.device)
        out = np.zeros(len(ALPHAS))
        for tr, va, ev, U, Kv in self.folds:
            ym = y[tr].mean()
            r = U.T @ (y[tr] - ym)
            for i, a in enumerate(ALPHAS):
                pred = Kv @ (U @ (r / (ev + a))) + ym
                out[i] += score(y[va].cpu().numpy(), pred.cpu().numpy(), kind) / len(self.folds)
        return out

    def predict(self, y, alpha, zev):
        y = torch.as_tensor(y[self.idx], dtype=torch.float64, device=self.X.device)
        ym = y.mean()
        coef = self.U @ ((self.U.T @ (y - ym)) / (self.ev + alpha))
        return (zev.to(self.X.device).double() @ (self.X.T @ coef) + ym).float().cpu()


def ridge_probe(rr, zte, y_tr, kind):
    """Alpha by k-fold CV on the train bank; one held-out prediction at the selected alpha."""
    cv = rr.cv(y_tr, kind)
    best = int(np.nanargmax(cv))
    return cv[best], ALPHAS[best], rr.predict(y_tr, ALPHAS[best], zte)


def mlp_inputs(ztr, zte, npc):
    """Leading npc train-span PCs whitened by their train sd, or (npc None) all PCs with one global scale."""
    if npc is None:
        s = ztr.pow(2).mean().sqrt().clamp_min(1e-12)
        return ztr / s, zte / s
    sd = ztr[:, :npc].std(0).clamp_min(1e-12)
    return ztr[:, :npc] / sd, zte[:, :npc] / sd


def mlp_probe_multi(ztr, zte, Y, kinds, dev, hidden=256, wds=(1e-2, 1e0, 1e1), npcs=(32, 128, 512), epochs=2000,
                    every=50, lr=2e-3, seed=0):
    """Independent 1-hidden-layer MLPs, one per (target, weight decay, input) trained jointly as a batch: full-batch
    Adam on 80% of the train bank; per target the (input, weight decay, epoch) with the lowest loss on the other 20%
    of train gives the held-out prediction. Y [n, T] with NaN = unlabeled; kinds per column ('auc' -> BCE)."""
    T = Y.shape[1]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Y), generator=g)
    va, tr = perm[: len(Y) // 5], perm[len(Y) // 5:]
    Y = Y.to(dev)
    lab = ~torch.isnan(Y)
    is_cls = torch.tensor([k == "auc" for k in kinds], device=dev)
    ym = torch.stack([torch.nanmean(Y[tr, j]) for j in range(T)])
    ys = torch.stack([Y[tr, j][lab[tr, j]].std() for j in range(T)]).clamp_min(1e-12)
    ym, ys = torch.where(is_cls, 0.0, ym), torch.where(is_cls, 1.0, ys)
    Yn = torch.nan_to_num((Y - ym) / ys)
    best_loss = torch.full((T,), np.inf, device=dev)
    best_pred = torch.zeros(len(zte), T, device=dev)
    best_hp = [None] * T
    for npc in npcs:
        a, b = mlp_inputs(ztr, zte, npc)
        Xtr, Xva, Xte = a[tr].to(dev), a[va].to(dev), b.to(dev)
        D, W = Xtr.shape[1], len(wds)
        M = T * W
        torch.manual_seed(seed)
        W1 = (torch.randn(M, D, hidden, device=dev) / np.sqrt(D)).requires_grad_()
        b1 = torch.zeros(M, 1, hidden, device=dev, requires_grad=True)
        W2 = (torch.randn(M, hidden, 1, device=dev) / np.sqrt(hidden)).requires_grad_()
        b2 = torch.zeros(M, 1, 1, device=dev, requires_grad=True)
        wd = torch.tensor(wds, device=dev).repeat(T).view(M, 1, 1)
        tcol = torch.arange(T, device=dev).repeat_interleave(W)
        opt = torch.optim.Adam([W1, b1, W2, b2], lr=lr)
        fwd = lambda X: (torch.relu(X.unsqueeze(0) @ W1 + b1) @ W2 + b2).squeeze(-1)

        def losses(out, rows):
            t, m = Yn[rows][:, tcol].T, lab[rows][:, tcol].T.float()
            l = torch.where(is_cls[tcol, None], F.binary_cross_entropy_with_logits(out, t, reduction="none"), (out - t) ** 2)
            return (l * m).sum(1) / m.sum(1).clamp_min(1)

        for ep in range(1, epochs + 1):
            loss = losses(fwd(Xtr), tr).sum() + ((wd * W1.pow(2)).sum() + (wd * W2.pow(2)).sum()) / len(tr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ep % every == 0:
                with torch.no_grad():
                    lv = losses(fwd(Xva), va).view(T, W)
                    m = lv.argmin(1)
                    lbest = lv.gather(1, m[:, None]).squeeze(1)
                    better = lbest < best_loss
                    if better.any():
                        pte = fwd(Xte).view(T, W, -1)
                        for j in torch.nonzero(better).flatten().tolist():
                            best_pred[:, j] = pte[j, m[j]]
                            best_hp[j] = f"npc {npc} wd {wds[m[j]]:g} ep {ep}"
                        best_loss = torch.where(better, lbest, best_loss)
        del W1, W2, opt
    return best_hp, (best_pred * ys + ym).cpu()


def knn_probe(ztr, zte, ytr, kind, ks=(5, 10, 20, 50), npcs=(16, 64, None), folds=5, seed=0):
    """Mean label of the k nearest train points (Euclidean, on the leading PCs of the train span); (k, #PCs) by CV."""
    ok = ~torch.isnan(ytr)
    Z, y = ztr[ok], ytr[ok]
    fold = torch.as_tensor(np.random.default_rng(seed).integers(folds, size=len(Z)))
    best = (-np.inf, None)
    for npc in npcs:
        Zp = Z[:, :npc] if npc else Z
        D = torch.cdist(Zp, Zp)
        cv = np.zeros(len(ks))
        for f in range(folds):
            va, tr = fold == f, fold != f
            nn_idx = D[va][:, tr].topk(max(ks), largest=False).indices
            yy = y[tr][nn_idx]
            for i, k in enumerate(ks):
                cv[i] += score(y[va].numpy(), yy[:, :k].mean(1).numpy(), kind) / folds
        i = int(np.nanargmax(cv))
        if cv[i] > best[0]:
            best = (cv[i], (ks[i], npc))
    k, npc = best[1]
    Dte = torch.cdist(zte[:, :npc] if npc else zte, Z[:, :npc] if npc else Z)
    return best[0], best[1], y[Dte.topk(k, largest=False).indices].mean(1)


def probe_all(xtr, xte, btr, bte, env, dev, tag, which=("ridge", "mlp", "knn")):
    """All probes for one feature matrix; returns raw rows (one per target x subset x probe)."""
    ttr, tte, sub = targets(btr, env), targets(bte, env), subsets(bte, env)
    rows = []
    rot, rr = {}, {}
    for prep in PREPS:
        a, b = preprocess(xtr.to(dev), xte.to(dev), prep)
        ztr, zte, ev = train_span(a, b)
        rot[prep] = (ztr.cpu(), zte.cpu())
        if "ridge" in which:
            for key in {np.isfinite(v[0]).tobytes() for v in ttr.values()}:
                rr[prep, key] = DualRidge(ztr, np.frombuffer(key, bool), dev)
        # max |test coordinate| / max |train coordinate| on the span: > 1e2 flags the near-dead-unit blow-up
        rows.append(dict(**tag, target="_diag", subset="all", probe=f"span_{prep}", metric="ratio",
                         value=float(zte.abs().max() / ztr.abs().max()), hp=f"rank {len(ev)} dim {a.shape[1]}"))
    names = list(ttr)
    if "mlp" in which:
        Y = torch.as_tensor(np.stack([ttr[t][0] for t in names], 1), dtype=torch.float32)
        mlp_hp, mlp_pred = mlp_probe_multi(*rot["zfix"], Y, [ttr[t][1] for t in names], dev)
    for tname, (y_tr, kind) in ttr.items():
        y_te = tte[tname][0]
        ytr_t = torch.as_tensor(y_tr, dtype=torch.float32)
        preds = {}
        if "ridge" in which:
            cvs = {}
            for prep in PREPS:
                cvs[prep] = ridge_probe(rr[prep, np.isfinite(y_tr).tobytes()], rot[prep][1], y_tr, kind)
                preds[f"ridge_{prep}"] = (cvs[prep][2].numpy(), f"alpha {cvs[prep][1]:g} cv {cvs[prep][0]:.3f}")
            sel = max(("raw", "zfix"), key=lambda p: cvs[p][0])
            preds["ridge_sel"] = (cvs[sel][2].numpy(), f"{sel} alpha {cvs[sel][1]:g} cv {cvs[sel][0]:.3f}")
        if "mlp" in which:
            j = names.index(tname)
            preds["mlp"] = (mlp_pred[:, j].numpy(), mlp_hp[j])
        if "knn" in which:
            cv, hp, p = knn_probe(*rot["zfix"], ytr_t, kind)
            preds["knn"] = (p.numpy(), f"k {hp[0]} pcs {hp[1]} cv {cv:.3f}")
        for sname, m in sub.items():
            mm = m & np.isfinite(y_te)
            if mm.sum() < 20 or (kind == "auc" and len(np.unique(y_te[mm])) < 2):
                continue
            for pname, (p, hp) in preds.items():
                rows.append(dict(**tag, target=tname, subset=sname, probe=pname, metric=kind,
                                 value=score(y_te[mm], p[mm], kind), hp=hp, n_test=int(mm.sum())))
    return rows


def bank_diagnostics(btr, bte, env):
    """Target spread and pairwise correlations in each bank (confound check)."""
    rows = []
    for split, b in (("train", btr), ("test", bte)):
        t = {k: v[0] for k, v in targets(b, env).items()}
        names = list(t)
        for i, a in enumerate(names):
            x = t[a]
            ok = np.isfinite(x)
            rows.append(dict(split=split, a=a, b="_self", mean=float(np.nanmean(x)), sd=float(np.nanstd(x)),
                             lo=float(np.nanmin(x)), hi=float(np.nanmax(x)), n=int(ok.sum())))
            for c in names[i + 1:]:
                y = t[c]
                ok2 = ok & np.isfinite(y)
                if ok2.sum() > 10 and np.std(x[ok2]) > 0 and np.std(y[ok2]) > 0:
                    rows.append(dict(split=split, a=a, b=c, corr=float(np.corrcoef(x[ok2], y[ok2])[0, 1]), n=int(ok2.sum())))
    return pd.DataFrame(rows)


def summarize(df, path_csv, path_md, env):
    d = df[df.target != "_diag"]
    rows = []
    for key, g in d.groupby(["cohort", "site", "target", "subset", "probe", "metric"]):
        m, lo, hi = C.ci95(g["value"].to_numpy())
        rows.append(dict(zip(["cohort", "site", "target", "subset", "probe", "metric"], key), n=len(g), mean=m, lo=lo, hi=hi))
    s = pd.DataFrame(rows)
    s.to_csv(path_csv, index=False)
    lines = [f"# Probe audit, {env}: held-out score (AUC / R2), mean [95% t-CI] across seeds", ""]
    order = [x for x in ("conv1", "conv2", "h_pre", "h_fc", "a_hid", "conv1_gap", "conv2_gap", "conv3_gap", "input_px16",
                         "input_px32", "input_state") if x in set(s.site)]
    for (tname, sname), g in s.groupby(["target", "subset"]):
        lines += [f"## {tname} | subset {sname}", "", "| probe | cohort | " + " | ".join(order) + " |",
                  "|" + "---|" * (len(order) + 2)]
        for (probe, cohort), h in g.groupby(["probe", "cohort"]):
            cell = {r.site: C.fmt_ci(r.mean, r.lo, r.hi) for r in h.itertuples()}
            lines.append(f"| {probe} | {cohort} | " + " | ".join(cell.get(o, "") for o in order) + " |")
        lines.append("")
    open(path_md, "w").write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["AT", "LR", "SZ", "PTM"], required=True)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--cohorts", nargs="+", default=["gm", "control", "init", "input"])
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--sites", nargs="+", default=list(SITES))
    p.add_argument("--out_tag", default="", help="non-empty: write raw CSVs under raw<tag>/ (smoke tests)")
    a = p.parse_args()
    raw = os.path.join(OUT, "raw" + a.out_tag)
    os.makedirs(raw, exist_ok=True)
    dev = C.device()
    btr, bte = C.load_bank(f"{a.env}_probe_train"), C.load_bank(f"{a.env}_probe_test")
    bank_diagnostics(btr, bte, a.env).to_csv(os.path.join(OUT, f"{a.env}_bank_diagnostics.csv"), index=False)
    jobs = []
    for cohort in a.cohorts:
        if cohort == "input":
            jobs.append(("input", 0, None))
        elif cohort == "init":
            jobs += [("init", s, None) for s in a.init_seeds]
        else:
            jobs += [(cohort, s, C.ckpt_path(a.env, cohort, s)) for s in SEEDS[a.env][cohort]]
    dfs = []
    for cohort, seed, ck in jobs:
        out = os.path.join(raw, f"{a.env}_{cohort}_s{seed}.csv")
        if not os.path.exists(out) and not a.summary_only:
            rows = []
            tag = dict(env=a.env, cohort=cohort, seed=seed)
            if cohort == "input":
                feats = {"input_px16": (pixel_blocks(btr["rgb"], 16), pixel_blocks(bte["rgb"], 16)),
                         "input_px32": (pixel_blocks(btr["rgb"], 32), pixel_blocks(bte["rgb"], 32)),
                         "input_state": (btr["state"], bte["state"])}
                for site, (xtr, xte) in feats.items():
                    rows += probe_all(torch.as_tensor(xtr, dtype=torch.float32), torch.as_tensor(xte, dtype=torch.float32),
                                      btr, bte, a.env, dev, dict(tag, site=site))
            else:
                agent = C.build_agent(ck, init_seed=seed, dev=dev)
                ftr, fte = features(agent, btr, dev), features(agent, bte, dev)
                for site in a.sites:
                    # GAP sites only reproduce probes.py (linear); nonlinear probes run at the depth-map sites
                    rows += probe_all(ftr[site], fte[site], btr, bte, a.env, dev, dict(tag, site=site),
                                      which=("ridge",) if site.endswith("_gap") else ("ridge", "mlp", "knn"))
                    print(f"{a.env} {cohort} s{seed} {site} done", flush=True)
                del ftr, fte
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out, index=False)
        if os.path.exists(out):
            dfs.append(pd.read_csv(out))
    df = pd.concat(dfs, ignore_index=True)
    if a.out_tag:
        print(df[df.subset == "all"].pivot_table(index=["target", "probe"], columns=["cohort", "site"], values="value").round(3).to_string())
        return
    df.to_csv(os.path.join(OUT, f"{a.env}_per_seed.csv"), index=False)
    summarize(df, os.path.join(OUT, f"{a.env}_summary.csv"), os.path.join(OUT, f"{a.env}_summary.md"), a.env)


if __name__ == "__main__":
    main()
