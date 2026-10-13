#!/usr/bin/env python3
"""Attribute-Maze world-edit Fisher sensitivity (Q, G, D_F) and signed derivative g at the post-ReLU taps h_pre / h_fc on the canonical bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "pkg", ROOT / "src", ROOT / "vendored" / "cs-train-procgen",
          ROOT / "vendored" / "train-procgen-pytorch", ROOT / "vendored" / "cs-procgenAISC"):
    if p.exists():
        sys.path.insert(0, str(p))

from gm_geometry.probes import ras_bank_io  # noqa: E402
from gm_geometry.probes.fisher import make_env  # noqa: E402
from gm_geometry.probes.h_fc_steering import extract_h_fc, verify_h_fc_hook  # noqa: E402
from gm_geometry.probes.h_pre_steering import extract_h_pre, verify_h_pre_hook  # noqa: E402
from gm_geometry.probes.ras_gt_bank import state_record  # noqa: E402
from gm_geometry.utils.seeding import stable_seed  # noqa: E402

# J is the Jacobian of the downstream map from the post-ReLU tap (W_pi at h_fc, W_pi diag(1[h_fc > 0]) W_fc at
# h_pre), checked against that closed form; differentiating through a second ReLU at the tap would wrongly zero
# columns of base-inactive units. "shape"/"colour" are ablation edits; "*_swap" are the Eq. 13/14 attribute swaps.

SCRIPT_VERSION = "ras_canon_fisher_v1_20260924"
OBJ_LINE, OBJ_GEM = "red_line_diag", "yellow_gem"
EDITS = {
    "shape": ("red_gem", "yellow_gem"),
    "colour": ("red_line_diag", "green_gem"),
    "shape_swap": ("red_gem", "yellow_line_diag"),
    "colour_swap": ("yellow_line_diag", "red_gem"),
}
GEOMETRY_EDITS = ("shape", "colour")
TAPS = ("h_pre", "h_fc")
SOURCE_FILES = ("scripts/run_ras_canon_fisher.py", "src/gm_geometry/probes/ras_bank_io.py",
                "src/gm_geometry/probes/ras_gt_bank.py", "src/gm_geometry/probes/h_fc_steering.py",
                "src/gm_geometry/probes/h_pre_steering.py", "src/gm_geometry/probes/fisher.py",
                "src/gm_geometry/probes/procgen_agent_state.py", "src/gm_geometry/utils/seeding.py")
BUNDLE_MAP = {"scripts/": "run_scripts/", "src/gm_geometry/": "pkg/gm_geometry/"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes() -> dict:
    out = {}
    for rel in SOURCE_FILES:
        cand = [ROOT / rel] + [ROOT / rel.replace(a, b, 1) for a, b in BUNDLE_MAP.items() if rel.startswith(a)]
        hit = next((c for c in cand if c.is_file()), None)
        out[rel] = sha256_file(hit) if hit else None
    return out


def obs_rgb(obs) -> np.ndarray:
    x = np.asarray(obs["rgb"] if isinstance(obs, dict) else obs).reshape(64, 64, 3)
    assert x.dtype == np.uint8, x.dtype
    return x


def to_tensor(rgb: np.ndarray, device) -> torch.Tensor:
    x = torch.as_tensor(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def diff_components(a: np.ndarray, b: np.ndarray, dilate: int = 3) -> list[tuple[int, int, int, int, int]]:
    """Components of differing pixels after a `dilate`-px dilation (so fragments of one sprite merge):
    (n_px, y0, y1, x0, x1) with n_px counted on the undilated mask."""
    mask = np.any(a != b, axis=-1)
    pad = np.pad(mask, dilate)
    grown = np.zeros_like(mask)
    for dy in range(2 * dilate + 1):
        for dx in range(2 * dilate + 1):
            grown |= pad[dy:dy + 64, dx:dx + 64]
    seen = np.zeros_like(grown)
    comps = []
    for y, x in zip(*np.nonzero(grown)):
        if seen[y, x]:
            continue
        stack, pts = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            if mask[cy, cx]:
                pts.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < 64 and 0 <= nx < 64 and grown[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if pts:
            ys, xs = zip(*pts)
            comps.append((len(pts), min(ys), max(ys), min(xs), max(xs)))
    return comps


def build_world(states: list[dict], use_backgrounds: bool) -> list[dict]:
    """Per state: bank-verified base obs plus every edit's obs, each edit verified to keep the exact
    agent / walls / obj1-cell / obj2-cell (state_key equality, the bank's G6 gate) and to change pixels
    only in at most two object-sized blobs (catches a texture background re-draw)."""
    bg = "texture" if use_backgrounds else "black"
    out = []
    for u in states:
        res = ras_bank_io.materialize_res(
            u, make_base_env=lambda lvl: make_env(OBJ_LINE, OBJ_GEM, lvl, 0, "easy", use_backgrounds),
            state_bytes=ras_bank_io.env_raw_state, use_backgrounds=use_backgrounds)
        base = obs_rgb(res["obs0"])
        rec = {"u": u, "res": res, "obs": {"base": base}, "edit_checks": {}}
        for name, (o1, o2) in EDITS.items():
            env = make_env(o1, o2, u["level"], 0, "easy", use_backgrounds)
            try:
                o = obs_rgb(env.reset())
                srec = state_record(ras_bank_io.env_raw_state(env), u["level"], bg)
            finally:
                env.close()
            comps = diff_components(base, o)
            # two edited sprites in adjacent cells merge into one blob, so bound each blob by two cells
            # (~13 px each); a background re-draw would change thousands of pixels, not a few hundred
            big = [c for c in comps if (c[2] - c[1] + 1) > 30 or (c[4] - c[3] + 1) > 30]
            n_px = int(sum(c[0] for c in comps))
            ok_state = srec["state_key"] == u["state_key"]
            ok_px = 1 <= len(comps) <= 2 and not big and n_px <= 400
            rec["edit_checks"][name] = {"state_key_equal": ok_state, "n_diff_px": n_px,
                                        "n_components": len(comps), "pixels_local": ok_px}
            if not ok_state:
                raise SystemExit(f"STOP level {u['level']} edit {name}: edited world changed agent/walls/targets")
            if not ok_px:
                raise SystemExit(f"STOP level {u['level']} edit {name}: pixel change not local {comps[:6]}")
            rec["obs"][name] = o
        out.append(rec)
    return out


def load_policy(ckpt: Path | None, device, *, init_seed: int | None = None):
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy
    cfg = next(c for c in (ROOT / "vendored" / "cs-train-procgen", ROOT / "vendored" / "train-procgen-pytorch")
               if (c / "hyperparams/procgen/config.yml").exists())
    recurrent = yaml.safe_load(open(cfg / "hyperparams/procgen/config.yml"))["hard-500"]["recurrent"]
    assert recurrent is False
    if init_seed is not None:
        torch.manual_seed(int(init_seed))
        np.random.seed(int(init_seed))
    policy = CategoricalPolicy(ImpalaModel(in_channels=3), recurrent, 15)
    if ckpt is not None:
        policy.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
    return policy.to(device).eval()


class Taps:
    """Canonical taps and their downstream logit maps, in float64 on CPU for the Jacobian algebra."""

    def __init__(self, policy):
        self.W_fc = policy.embedder.fc.weight.detach().cpu().double()
        self.b_fc = policy.embedder.fc.bias.detach().cpu().double()
        self.W_pi = policy.fc_policy.weight.detach().cpu().double()
        self.b_pi = policy.fc_policy.bias.detach().cpu().double()

    def logits_fc(self, h):
        return h @ self.W_pi.T + self.b_pi

    def logits_pre(self, h):
        return self.logits_fc(torch.relu(h @ self.W_fc.T + self.b_fc))

    def logits(self, tap, h):
        return self.logits_pre(h) if tap == "h_pre" else self.logits_fc(h)

    def jacobian(self, tap, h: torch.Tensor) -> torch.Tensor:
        return torch.autograd.functional.jacobian(lambda x: self.logits(tap, x), h)

    def jacobian_closed_form(self, tap, h: torch.Tensor) -> torch.Tensor:
        if tap == "h_fc":
            return self.W_pi.clone()
        mask = ((h @ self.W_fc.T + self.b_fc) > 0).double()
        return (self.W_pi * mask) @ self.W_fc


def softmax64(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (np.log(p) - np.log(q))))


def state_quantities(J: np.ndarray, p: np.ndarray, deltas: dict, A_line: list, A_gem: list,
                     logits_fn, h_base: np.ndarray, kl_ts: list, rand_dirs: np.ndarray,
                     old_mask: np.ndarray | None = None) -> dict:
    """Q, S, G, D_F, g, ghat, ||Delta||, finite-KL calibration and matched random-direction G for every
    edit at one (state, tap, checkpoint). F = J^T (diag p - p p^T) J is formed implicitly via L = J Delta."""
    d = J.shape[1]
    Wp = np.diag(p) - np.outer(p, p)
    trF = float(np.einsum("ai,ab,bi->", J, Wp, J))
    trfd = trF / d
    mean_J = p @ J
    grad_p = p[:, None] * (J - mean_J[None, :])
    grad_C = grad_p[A_line].sum(0) - grad_p[A_gem].sum(0)
    C_base = float(p[A_line].sum() - p[A_gem].sum())
    LR = rand_dirs @ J.T
    G_rand = np.einsum("ra,ab,rb->r", LR, Wp, LR) / (trfd if trfd > 0 else np.nan)
    out = {"trF": trF, "trF_over_d": trfd, "d_tap": d, "C_base": C_base,
           "G_rand_median": float(np.median(G_rand)), "G_rand_mean": float(np.mean(G_rand)),
           "entropy": float(-(p * np.log(p)).sum()), "p_line": float(p[A_line].sum()),
           "p_gem": float(p[A_gem].sum()), "edits": {}}
    for name, dv in deltas.items():
        L = J @ dv
        Q = float(L @ Wp @ L)
        n = float(np.linalg.norm(dv))
        S = Q / n ** 2 if n > 0 else float("nan")
        g = float(grad_C @ dv)
        e = {"Q": Q, "norm_delta": n, "S": S, "G": S / trfd if trfd > 0 else float("nan"),
             "D_F": n ** 2 * trfd, "g": g, "ghat": g / n if n > 0 else float("nan")}
        for t in kl_ts:
            q = softmax64(logits_fn(h_base + t * dv))
            e[f"KL_t{t:g}"] = kl(p, q)
            if t == 1.0:
                e["dC_finite"] = float(q[A_line].sum() - q[A_gem].sum()) - C_base
        e["KL_over_halfQ"] = e.get("KL_t1", float("nan")) / (0.5 * Q) if Q > 0 else float("nan")
        if old_mask is not None:
            # Comparison Jacobian with columns zeroed where the tap is exactly 0 (a hook through a second ReLU)
            Lm = (J * old_mask[None, :]) @ dv
            gm_ = p[:, None] * ((J * old_mask[None, :]) - (p @ (J * old_mask[None, :]))[None, :])
            e["Q_oldmask"] = float(Lm @ Wp @ Lm)
            e["g_oldmask"] = float((gm_[A_line].sum(0) - gm_[A_gem].sum(0)) @ dv)
            e["frac_delta_sq_on_inactive"] = float(np.sum(dv[~old_mask.astype(bool)] ** 2) / max(n ** 2, 1e-300))
        out["edits"][name] = e
    L_s, L_c = J @ deltas["shape"], J @ deltas["colour"]
    qs, qc = out["edits"]["shape"]["Q"], out["edits"]["colour"]["Q"]
    out["cos_F_shape_colour"] = float(L_s @ Wp @ L_c / np.sqrt(qs * qc)) if qs > 0 and qc > 0 else float("nan")
    return out


def score_checkpoint(policy, world: list[dict], args, *, meta: dict, out_dir: Path) -> dict:
    device = next(policy.parameters()).device
    taps = Taps(policy)
    x0 = to_tensor(world[0]["obs"]["base"], device)
    vf, vp = verify_h_fc_hook(policy, x0), verify_h_pre_hook(policy, x0)
    if not (vf["matches"] and vf["all_nonneg_confirms_post_relu"] and vp["matches"]
            and vp["all_nonneg_confirms_post_relu"]):
        raise SystemExit(f"STOP tap verification failed: h_fc {vf} h_pre {vp}")
    rows, geo = [], {k: [] for k in ("state_key", "level", "logits", "p", "h_fc_active_mask")}
    for tap in TAPS:
        for k in ("h_base", "J") + tuple(f"h_{e}" for e in GEOMETRY_EDITS) + tuple(f"delta_{e}" for e in GEOMETRY_EDITS):
            geo[f"{k}_{tap}"] = []
    swap_geo = {f"delta_{e}_{t}": [] for e in ("shape_swap", "colour_swap") for t in TAPS}
    checks = {"max_logit_err_pre": 0.0, "max_logit_err_fc": 0.0, "max_hfc_recon_err": 0.0,
              "max_J_closed_form_err": 0.0}
    for w in world:
        u, r = w["u"], w["u"]["record"]
        A_line, A_gem = list(r["A_line"]), list(r["A_gem"])
        h = {}
        with torch.no_grad():
            for name, rgb in w["obs"].items():
                x = to_tensor(rgb, device)
                dist, _v, _hx = policy(x, torch.zeros(1, 256, device=device), torch.zeros(1, device=device))
                h[name] = {"h_fc": extract_h_fc(policy, x)[0].cpu().double(),
                           "h_pre": extract_h_pre(policy, x)[0].cpu().double(),
                           "log_probs": torch.log_softmax(dist.logits[0].cpu().double(), -1)}
        hb = h["base"]
        lg_pre, lg_fc = taps.logits_pre(hb["h_pre"]), taps.logits_fc(hb["h_fc"])
        lp = hb["log_probs"]
        checks["max_logit_err_pre"] = max(checks["max_logit_err_pre"],
                                          float((torch.log_softmax(lg_pre, -1) - lp).abs().max()))
        checks["max_logit_err_fc"] = max(checks["max_logit_err_fc"],
                                         float((torch.log_softmax(lg_fc, -1) - lp).abs().max()))
        rec_fc = torch.relu(hb["h_pre"] @ taps.W_fc.T + taps.b_fc)
        checks["max_hfc_recon_err"] = max(checks["max_hfc_recon_err"], float((rec_fc - hb["h_fc"]).abs().max()))
        p = softmax64(lg_fc.numpy())
        geo["state_key"].append(u["state_key"])
        geo["level"].append(u["level"])
        geo["logits"].append(lg_fc.numpy())
        geo["p"].append(p)
        geo["h_fc_active_mask"].append((rec_fc > 0).numpy())
        base_row = {**meta, "state_key": u["state_key"], "state_hash": u["state_hash"], "level": u["level"],
                    "n_dup_levels": len(u["levels"]), "d": int(r["d_line"]), "is_d1": int(r["d_line"]) == 1,
                    "axis_sign": int(r["axis_sign_lineN_gemE"]), "split_role": w["split_role"],
                    "loo_fold": w["loo_fold"], "A_line": " ".join(map(str, A_line)),
                    "A_gem": " ".join(map(str, A_gem))}
        for tap in TAPS:
            hbt = hb[tap]
            J = taps.jacobian(tap, hbt)
            Jc = taps.jacobian_closed_form(tap, hbt)
            checks["max_J_closed_form_err"] = max(checks["max_J_closed_form_err"], float((J - Jc).abs().max()))
            J = J.numpy()
            deltas = {e: (h[e][tap] - hbt).numpy() for e in EDITS}
            rng = np.random.default_rng(stable_seed("canon_fisher_rand_dirs", u["state_key"], tap))
            rd = rng.standard_normal((args.n_rand_dirs, J.shape[1]))
            rd /= np.linalg.norm(rd, axis=1, keepdims=True)
            q = state_quantities(J, p, deltas, A_line, A_gem,
                                 lambda z, _t=tap: taps.logits(_t, torch.as_tensor(z)).numpy(),
                                 hbt.numpy(), args.kl_ts, rd, old_mask=(hbt.numpy() > 0).astype(np.float64))
            geo[f"h_base_{tap}"].append(hbt.numpy().astype(np.float32))
            geo[f"J_{tap}"].append(J.astype(np.float32) if tap == "h_fc" else None)
            for e in GEOMETRY_EDITS:
                geo[f"h_{e}_{tap}"].append(h[e][tap].numpy().astype(np.float32))
                geo[f"delta_{e}_{tap}"].append(deltas[e].astype(np.float32))
            for e in ("shape_swap", "colour_swap"):
                swap_geo[f"delta_{e}_{tap}"].append(deltas[e].astype(np.float32))
            for e, ev in q["edits"].items():
                rows.append({**base_row, "tap": tap, "edit": e,
                             **{k: v for k, v in q.items() if k != "edits"}, **ev,
                             **{f"check_{k}": v for k, v in w["edit_checks"][e].items()}})
    for k, tol in (("max_logit_err_pre", 1e-4), ("max_logit_err_fc", 1e-4), ("max_hfc_recon_err", 1e-4),
                   ("max_J_closed_form_err", 1e-9)):
        if checks[k] > tol:
            raise SystemExit(f"STOP {k}={checks[k]} > {tol}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = meta["ckpt_label"]
    csv_path = out_dir / f"rows_{meta['background']}_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    geo.pop("J_h_pre")
    arrays = {k: np.stack(v) for k, v in geo.items() if k != "state_key"}
    arrays["state_key"] = np.array(geo["state_key"])
    arrays.update(W_fc=taps.W_fc.numpy().astype(np.float32), b_fc=taps.b_fc.numpy().astype(np.float32),
                  W_pi=taps.W_pi.numpy().astype(np.float32), b_pi=taps.b_pi.numpy().astype(np.float32))
    npz_path = out_dir / f"geom_{meta['background']}_{tag}.npz"
    np.savez_compressed(npz_path, **arrays)
    swap_path = out_dir / f"swapdeltas_{meta['background']}_{tag}.npz"
    np.savez_compressed(swap_path, state_key=np.array(geo["state_key"]),
                        **{k: np.stack(v) for k, v in swap_geo.items()})
    return {"csv": csv_path.name, "npz": npz_path.name, "swap_npz": swap_path.name, "n_rows": len(rows),
            "tap_verify": {"h_fc": vf, "h_pre": vp}, "numeric_checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--background", choices=["black", "texture"], required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="cohort:filename, e.g. gm:6313_gm.pth deco:4371_teach.pth")
    ap.add_argument("--init-seeds", nargs="*", type=int, default=[3000, 3001, 3002])
    ap.add_argument("--primary-split", type=Path, required=True)
    ap.add_argument("--loo-splits", nargs="*", type=Path, default=[])
    ap.add_argument("--bank-json", type=Path, required=True)
    ap.add_argument("--max-states", type=int, default=0, help="smoke tests only; 0 = all 196")
    ap.add_argument("--kl-ts", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--n-rand-dirs", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    use_bg = args.background == "texture"
    frozen_sha = "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9"
    info = ras_bank_io.bank_info(args.bank_json, background=args.background)
    if info["canonical_bank_sha256"] != frozen_sha:
        raise SystemExit(f"STOP bank is not the frozen canonical bank: {info['canonical_bank_sha256']}")

    states, provs = [], {}
    for role in ("fit", "eval"):
        st, prov = ras_bank_io.select_states(args.bank_json, args.primary_split, role, background=args.background)
        provs[role] = prov
        states += [(u, role) for u in st]
    fold_of = {}
    for i, sp in enumerate(sorted(args.loo_splits)):
        for e in ras_bank_io.load_split(sp)["eval"]:
            fold_of[e["state_key"]] = i
    states.sort(key=lambda t: t[0]["level"])
    if len({u["state_key"] for u, _ in states}) != len(states):
        raise SystemExit("STOP duplicate state_key across split roles")
    if args.max_states:
        by_bg_eval = [t for t in states if t[1] == "eval"][: args.max_states]
        states = by_bg_eval
    print(f"[canon-fisher] bg={args.background} states={len(states)} "
          f"(eval={sum(r == 'eval' for _, r in states)}) bank={info['canonical_bank_sha256'][:12]}", flush=True)
    world = build_world([u for u, _ in states], use_bg)
    for w, (_u, role) in zip(world, states):
        w["split_role"] = role
        w["loo_fold"] = fold_of.get(w["u"]["state_key"], -1)
    print(f"[canon-fisher] world built + verified ({time.time() - t0:.0f}s)", flush=True)

    src = source_hashes()
    device = torch.device(args.device)
    manifest = {"script_version": SCRIPT_VERSION, "background": args.background,
                "canonical_bank_sha256": info["canonical_bank_sha256"], "bank_info": info,
                "bank_provenance": provs, "source_sha256": src, "edits": EDITS, "taps": list(TAPS),
                "tap_definitions": {"h_fc": "policy.embedder(x) output (post final ReLU)",
                                    "h_pre": "flatten(ReLU(block3(...))) = input to embedder.fc"},
                "args": {k: str(v) for k, v in vars(args).items()}, "n_states": len(world),
                "edit_checks": {w["u"]["state_key"]: w["edit_checks"] for w in world}, "checkpoints": {}}
    jobs = []
    for spec in args.ckpts:
        cohort, fn = spec.split(":")
        jobs.append((cohort, args.ckpt_dir / fn, int(fn.split("_")[0]), None))
    for s in args.init_seeds:
        jobs.append(("init", None, s, s))
    for cohort, ck, seed, init_seed in jobs:
        label = f"{cohort}_{seed}"
        ck_sha = sha256_file(ck) if ck is not None else None
        policy = load_policy(ck, device, init_seed=init_seed)
        meta = {"background": args.background, "cohort": cohort, "seed": seed, "ckpt_label": label,
                "ckpt_path": str(ck) if ck else f"random_init(torch.manual_seed={init_seed})",
                "ckpt_sha256": ck_sha or "", "canonical_bank_sha256": info["canonical_bank_sha256"],
                "code_sha256": src["scripts/run_ras_canon_fisher.py"]}
        res = score_checkpoint(policy, world, args, meta=meta, out_dir=args.out)
        manifest["checkpoints"][label] = {**meta, **res}
        (args.out / f"manifest_{args.background}.json").write_text(json.dumps(manifest, indent=1, default=str))
        print(f"[canon-fisher] {args.background} {label} rows={res['n_rows']} checks={res['numeric_checks']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[canon-fisher] DONE {args.background} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
