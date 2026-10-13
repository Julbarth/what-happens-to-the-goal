#!/usr/bin/env python3
"""Build swap-primary geometry npz and per-state paired shape-swap vs colour-swap Fisher terms (incl. cos_F)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE_KEYS = ("state_key", "level", "logits", "p", "h_fc_active_mask", "h_base_h_fc", "h_base_h_pre", "J_h_fc",
             "W_fc", "b_fc", "W_pi", "b_pi")
SWAPS = ("shape_swap", "colour_swap")


def jac(z, i, tap):
    Wpi = z["W_pi"].astype(np.float64)
    if tap == "h_fc":
        return z["J_h_fc"][i].astype(np.float64)
    return (Wpi * z["h_fc_active_mask"][i]) @ z["W_fc"].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/probes/canon_fisher_20260924"))
    ap.add_argument("--swapdeltas", type=Path, required=True)
    args = ap.parse_args()
    (args.dir / "geom_swap").mkdir(exist_ok=True)
    (args.dir / "rows_swap_pair").mkdir(exist_ok=True)
    worst = {}
    for gf in sorted((args.dir / "geom").glob("geom_*.npz")):
        label = gf.stem[len("geom_"):]
        z = np.load(gf)
        s = np.load(args.swapdeltas / f"swapdeltas_{label}.npz")
        if not np.array_equal(z["state_key"], s["state_key"]):
            raise SystemExit(f"STOP {label}: state_key order differs between geom and swapdeltas")
        rows = {(r["state_key"], r["tap"], r["edit"]): r
                for r in csv.DictReader(open(args.dir / "rows" / f"rows_{label}.csv"))}
        out = {k: z[k] for k in BASE_KEYS}
        for e in SWAPS:
            for tap in ("h_fc", "h_pre"):
                out[f"delta_{e}_{tap}"] = s[f"delta_{e}_{tap}"]
        np.savez_compressed(args.dir / "geom_swap" / f"geom_swap_{label}.npz", **out)
        pair_rows, err = [], 0.0
        for i, sk in enumerate(z["state_key"]):
            p = z["p"][i].astype(np.float64)
            W = np.diag(p) - np.outer(p, p)
            for tap in ("h_fc", "h_pre"):
                J = jac(z, i, tap)
                trfd = float(np.einsum("ai,ab,bi->", J, W, J)) / J.shape[1]
                r0 = rows[(sk, tap, "shape_swap")]
                A_l = [int(a) for a in r0["A_line"].split()]
                A_g = [int(a) for a in r0["A_gem"].split()]
                gp = p[:, None] * (J - p @ J)
                gradC = gp[A_l].sum(0) - gp[A_g].sum(0)
                L, rec = {}, {}
                for e in SWAPS:
                    dv = s[f"delta_{e}_{tap}"][i].astype(np.float64)
                    L[e] = J @ dv
                    Q = float(L[e] @ W @ L[e])
                    g = float(gradC @ dv)
                    rq = float(rows[(sk, tap, e)]["Q"])
                    err = max(err, abs(Q - rq) / max(abs(rq), 1e-12) if abs(rq) > 1e-9 else 0.0)
                    rec[e] = (Q, g, float(dv @ dv))
                d_comp = L["colour_swap"] - L["shape_swap"]
                qs, qc = rec["shape_swap"][0], rec["colour_swap"][0]
                pair_rows.append({
                    "state_key": sk, "tap": tap, "cohort": r0["cohort"], "seed": r0["seed"],
                    "background": r0["background"], "axis_sign": r0["axis_sign"], "d": r0["d"],
                    "is_d1": r0["is_d1"], "split_role": r0["split_role"], "trF_over_d": trfd,
                    "cos_F_swap": float(L["shape_swap"] @ W @ L["colour_swap"] / np.sqrt(qs * qc)) if qs > 0 and qc > 0 else float("nan"),
                    "cos_E_swap": float(s["delta_shape_swap_" + tap][i].astype(np.float64) @ s["delta_colour_swap_" + tap][i].astype(np.float64)
                                        / np.sqrt(rec["shape_swap"][2] * rec["colour_swap"][2])),
                    "Q_dcomp": float(d_comp @ W @ d_comp),
                    "g_dcomp": rec["colour_swap"][1] - rec["shape_swap"][1],
                    "Q_shape_swap": qs, "Q_colour_swap": qc,
                    "g_shape_swap": rec["shape_swap"][1], "g_colour_swap": rec["colour_swap"][1],
                })
        with open(args.dir / "rows_swap_pair" / f"rows_swap_pair_{label}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)
        worst[label] = err
        print(label, "max rel err Q(float32 swap delta) vs in-job row", f"{err:.2e}")
    if max(worst.values()) > 1e-4:
        raise SystemExit(f"STOP swap deltas do not reproduce in-job Q: {max(worst.values())}")
    files = sorted(list((args.dir / "geom_swap").glob("*.npz")) + list((args.dir / "rows_swap_pair").glob("*.csv"))
                   + list((args.dir / "rows").glob("*.csv")) + [args.dir / "README.md"])
    ready = {"status": "final", "written": "2026-09-24", "primary_edits": "SWAPS (paper A.2.2 Eq. 13-14; "
             "src/gm_geometry/probes/ras_counterfactuals.py): shape_swap=(red_gem, yellow_line_diag), "
             "colour_swap=(yellow_line_diag, red_gem); ablations supplementary only",
             "D_COMP": "h(colour_swap) - h(shape_swap) = delta_colour_swap - delta_shape_swap (points toward line cell u)",
             "canonical_bank_sha256": "0f45ca6766f2ebc660fdbbbfc66cb19cd145833e3c7da75b4c95ea2fd31b21b9",
             "producer": "run_ras_canon_fisher.py job 28611 (swap rows computed in-job from exact float64 deltas); "
                         "geom_swap/ built by scripts/build_ras_canon_fisher_swap.py",
             "max_rel_err_Q_from_float32_deltas": max(worst.values()),
             "cohorts": {"gm": "6313/6315/6317 black GM", "gmtex": "6301-03 texture GM",
                         "deco": "4371-73 black-trained control (primary on black; supplementary on texture)",
                         "decotex": "4361-63 texture-trained control (PRIMARY texture Non-GM)", "init": "seeds 3000-3002"},
             "files": {str(p.relative_to(args.dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (args.dir / "ROWS_READY_SWAP.json").write_text(json.dumps(ready, indent=1))
    print("wrote ROWS_READY_SWAP.json with", len(ready["files"]), "files")


if __name__ == "__main__":
    main()
