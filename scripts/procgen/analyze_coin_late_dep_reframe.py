#!/usr/bin/env python3
"""Coin late-dependence analysis across all seeds: scatter, decode delta, band census, control check."""
from __future__ import annotations

import os
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from gm_geometry.utils import parse_seeds  # noqa: E402

DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
PROBE = DATA_ROOT / "probes/coinrun_emergence_dep"
OUT = PROBE / "reframe_late_dep"
SEEDS = [6033, 6034, 6035, 6036, 6037, 6038, 6040, 6041]
BAND = 0.12
FAIL_COMPETENT = 0.25


def _steps_key(k: str) -> int:
    m = re.search(r"(\d+)", str(k))
    return int(m.group(1)) if m else 0


def state_label(*, fail: float, success: float, gm: float, dep: float) -> str:
    """Visible-eval state label for a ladder cell."""
    if fail >= 0.45 and success < 0.35:
        return "cap-fail"
    if dep < -0.05 and gm >= success:
        return "GM (distractor)"
    if gm > success + 0.05:
        return "GM"
    if dep >= 0.05 and success >= gm:
        return "generalized"
    if success >= gm and abs(dep) < 0.05:
        return "mixed"
    if success > gm:
        return "collect-lean"
    return "mixed"


def load_late(path: Path) -> dict | None:
    j = json.loads(path.read_text())
    deps = j.get("coin_dependence") or {}
    if not deps:
        return None
    last_k = max(deps.keys(), key=_steps_key)
    dep_rec = deps[last_k]
    if isinstance(dep_rec, dict):
        dep = float(dep_rec["coin_dependence"])
        steps = int(dep_rec.get("steps", _steps_key(last_k)))
    else:
        dep = float(dep_rec)
        steps = _steps_key(last_k)

    cells = j["cells"]
    # Prefer the visible (hide=False) cell at this step.
    vis = None
    if isinstance(cells, dict):
        candidates = []
        for ck, cv in cells.items():
            if not isinstance(cv, dict):
                continue
            if cv.get("steps") == steps or _steps_key(ck) == steps:
                candidates.append(cv)
        for cv in candidates:
            if cv.get("hide") is False or cv.get("hide") == 0:
                vis = cv
                break
        if vis is None and candidates:
            vis = candidates[0]
    else:
        vis = max(
            (c for c in cells if isinstance(c, dict)),
            key=lambda c: c.get("steps", 0),
            default=None,
        )
    if vis is None:
        return None

    orates = vis.get("outcome_rates") or {}
    success = float(vis.get("success_rate", 0.0))
    gm = float(vis.get("gm_rate", 0.0))
    fail = orates.get("fail")
    if fail is None:
        # Coin ladders: capability fail = died + timed_out (sometimes capability_fail).
        fail = (
            float(orates.get("died", 0.0) or 0.0)
            + float(orates.get("timed_out", 0.0) or 0.0)
            + float(orates.get("timeout", 0.0) or 0.0)
            + float(orates.get("death", 0.0) or 0.0)
            + float(orates.get("capability_fail", 0.0) or 0.0)
        )
        if fail == 0.0:
            other = float(orates.get("other", 0.0) or 0.0)
            fail = max(0.0, 1.0 - success - gm - other)
    fail = float(fail)
    mode = j.get("action_mode") or vis.get("action_mode") or "unknown"
    st = state_label(fail=fail, success=success, gm=gm, dep=dep)
    return {
        "path": str(path),
        "action_mode": mode,
        "steps": steps,
        "dep": dep,
        "fail": fail,
        "success": success,
        "gm": gm,
        "state": st,
        "n_levels": j.get("n_levels") or vis.get("n_levels"),
    }


def collect_rows(seeds: list[int] | None = None) -> list[dict]:
    seeds = SEEDS if seeds is None else seeds
    rows = []
    for seed in seeds:
        d = PROBE / f"seed{seed}"
        if not d.exists():
            continue
        # Prefer mode-tagged files; skip bare ladder_coin_dep.json if tagged exists.
        files = sorted(d.glob("ladder_coin_dep*.json"))
        tagged = [p for p in files if p.name.endswith(("_greedy.json", "_sample.json"))]
        use = tagged if tagged else files
        seen_modes = set()
        for p in use:
            late = load_late(p)
            if late is None:
                continue
            mode = late["action_mode"]
            if mode in seen_modes:
                continue
            seen_modes.add(mode)
            rows.append({"seed": seed, **late})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default=None, help="comma/space-separated seeds (default: current fixed roster)")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds) if args.seeds else SEEDS

    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(seeds)
    if not rows:
        raise SystemExit("no late rows found")

    competent = [r for r in rows if r["fail"] < FAIL_COMPETENT]
    in_band = [r for r in competent if abs(r["dep"]) < BAND]
    band_frac = (len(in_band) / len(competent)) if competent else float("nan")

    # Decode Δ and flips
    by_seed: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["action_mode"]] = r
    decode_deltas = []
    flips = []
    for seed, modes in sorted(by_seed.items()):
        if "greedy" in modes and "sample" in modes:
            g, s = modes["greedy"], modes["sample"]
            delta = s["dep"] - g["dep"]
            decode_deltas.append(
                {
                    "seed": seed,
                    "dep_greedy": g["dep"],
                    "dep_sample": s["dep"],
                    "delta_dep": delta,
                    "state_greedy": g["state"],
                    "state_sample": s["state"],
                    "fail_greedy": g["fail"],
                    "fail_sample": s["fail"],
                    "flip": g["state"] != s["state"],
                }
            )
            if g["state"] != s["state"]:
                flips.append(seed)

    # Decode-robust competent persistent-GM: both decodes fail<0.25, dep<0, GM competitive
    robust_controls = []
    for seed, modes in sorted(by_seed.items()):
        if "greedy" not in modes or "sample" not in modes:
            continue
        ok = True
        for m in ("greedy", "sample"):
            r = modes[m]
            gm_comp = r["gm"] >= r["success"] - 0.05  # competitive
            if not (r["fail"] < FAIL_COMPETENT and r["dep"] < 0 and gm_comp):
                ok = False
                break
        if ok:
            robust_controls.append(seed)

    # Sustained |dep|>=0.15 on both decodes (undercuts continuum)
    both_large = []
    for seed, modes in sorted(by_seed.items()):
        if "greedy" not in modes or "sample" not in modes:
            continue
        g, s = modes["greedy"], modes["sample"]
        if (
            g["fail"] < FAIL_COMPETENT
            and s["fail"] < FAIL_COMPETENT
            and abs(g["dep"]) >= 0.15
            and abs(s["dep"]) >= 0.15
        ):
            both_large.append(seed)

    supports = (
        (band_frac >= 0.5 if competent else False)
        and len(robust_controls) == 0
        and (len(flips) >= 1 or len(decode_deltas) == 0)
    )
    undercuts = len(both_large) >= 2

    if not decode_deltas:
        dir_note = "n/a (no dual-decode)"
    elif all(d["delta_dep"] > 0 for d in decode_deltas):
        dir_note = "yes"
    elif all(d["delta_dep"] < 0 for d in decode_deltas):
        dir_note = "no (all oppose)"
    else:
        dir_note = "no (mixed signs)"

    summary = {
        "n_rows": len(rows),
        "rows": rows,
        "competent_n": len(competent),
        "band": BAND,
        "fail_competent": FAIL_COMPETENT,
        "in_band_n": len(in_band),
        "band_frac_of_competent": band_frac,
        "band_note": (
            "Illustrative |dep|<band cut only. Prefer qualitative: most competent "
            "cells within ~2 dep-noise-widths of 0. Do not quote band_frac as effect size."
        ),
        "cap_fail_tails": [
            {"seed": r["seed"], "action_mode": r["action_mode"], "fail": r["fail"], "dep": r["dep"]}
            for r in rows
            if r["fail"] >= FAIL_COMPETENT and r["state"] == "cap-fail"
        ],
        "decode_deltas": decode_deltas,
        "state_flips": flips,
        "decode_direction_consistent_sample_more_collect": (
            all(d["delta_dep"] > 0 for d in decode_deltas) if decode_deltas else None
        ),
        "decode_robust_persistent_GM_seeds": robust_controls,
        "both_decodes_abs_dep_ge_0.15_competent": both_large,
        "verdict": (
            "supports_reframe"
            if supports and not undercuts
            else ("undercuts_reframe" if undercuts else "mixed")
        ),
        "claim_bounds": {
            "earned": (
                "competent seeds cluster within noise of dep≈0; "
                "no decode-robust persistent-GM in roster; "
                "decode-sensitive near boundary (flip direction not yet consistent)"
            ),
            "not_earned": (
                "band_frac as effect size; sample-biased-collect-ward / "
                "'sample reveals thaw greedy hides'"
            ),
        },
        "one_liner": (
            f"Competence-gated (fail<{FAIL_COMPETENT}): most competent cells near dep≈0 "
            f"(illustrative |dep|<{BAND}: {len(in_band)}/{len(competent)}); "
            f"decode-robust persistent-GM: {len(robust_controls)}; "
            f"State flips: {flips}; sample-more-collect consistent: {dir_note}."
        ),
    }
    (OUT / "reframe_summary.json").write_text(json.dumps(summary, indent=2))

    # --- Figure ---
    # Competence gate is load-bearing: band census uses competent only; plot must show that.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    ax = axes[0]
    markers = {"greedy": "o", "sample": "s", "unknown": "x"}
    colors = {
        6033: "#1b7f4b",
        6034: "#c47a00",
        6035: "#2a6f97",
        6036: "#0b6e4f",
        6037: "#9b2226",
        6038: "#5c4d7a",
        6040: "#6c757d",
        6041: "#495057",
    }
    for r in rows:
        competent_pt = r["fail"] < FAIL_COMPETENT
        col = colors.get(r["seed"], "#333")
        ax.scatter(
            r["dep"],
            r["fail"],
            facecolors=col if competent_pt else "none",
            edgecolors=col,
            marker=markers.get(r["action_mode"], "o"),
            s=78 if competent_pt else 64,
            linewidths=1.4,
            zorder=3,
            label=None,
        )
        tag = f"{r['seed']}{r['action_mode'][0]}"
        if r["state"] == "cap-fail":
            tag += "\ncap-fail excl."
        elif not competent_pt:
            tag += "\n(not competent)"
        ax.annotate(
            tag,
            (r["dep"], r["fail"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=6.5,
            color="#222",
        )
    ax.axvspan(-BAND, BAND, color="#999", alpha=0.15, label=f"|dep|<{BAND} (illustrative)")
    ax.axhline(FAIL_COMPETENT, color="#444", ls="--", lw=0.9, label=f"competence gate fail={FAIL_COMPETENT}")
    ax.axvline(0, color="#bbb", lw=0.6)
    # legend proxies
    ax.scatter([], [], c="#333", marker="o", s=50, label="competent (filled)")
    ax.scatter([], [], facecolors="none", edgecolors="#333", marker="o", s=50, label="not competent (hollow)")
    ax.set_xlabel("late coin-dependence")
    ax.set_ylabel("late fail rate")
    ax.set_title(
        f"Late dep vs fail — band census only below fail<{FAIL_COMPETENT}\n"
        f"(competent near-zero: {len(in_band)}/{len(competent)}; 6037 = cap-fail tail)"
    )
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.set_xlim(-0.22, 0.22)
    ax.set_ylim(-0.02, max(0.7, max(r["fail"] for r in rows) + 0.08))

    ax2 = axes[1]
    if decode_deltas:
        seeds = [d["seed"] for d in decode_deltas]
        x = np.arange(len(seeds))
        ax2.bar(x - 0.18, [d["dep_greedy"] for d in decode_deltas], 0.35, label="greedy", color="#4c6ef5")
        ax2.bar(x + 0.18, [d["dep_sample"] for d in decode_deltas], 0.35, label="sample", color="#f59f00")
        ax2.axhline(0, color="#bbb", lw=0.6)
        ax2.axhline(BAND, color="#999", ls=":", lw=0.7)
        ax2.axhline(-BAND, color="#999", ls=":", lw=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(s) for s in seeds])
        ax2.set_ylabel("late dep")
        ax2.set_title(
            "Decode split — sensitivity only\n"
            "(not directional: sample≠always more collect)"
        )
        ax2.legend(frameon=False, fontsize=8)
        for i, d in enumerate(decode_deltas):
            # Directional prediction: sampling more collect implies delta-dep > 0
            if d["delta_dep"] > 0.02:
                note = "obeys"
                ncol = "#1b7f4b"
            elif d["delta_dep"] < -0.02:
                note = "opposes"
                ncol = "#9b2226"
            else:
                note = "~noise"
                ncol = "#666"
            if d["flip"]:
                note = f"flip / {note}"
            y = max(d["dep_greedy"], d["dep_sample"], 0) + 0.015
            ax2.text(i, y, note, ha="center", fontsize=7.5, color=ncol, fontweight="bold")
        ax2.text(
            0.5,
            -0.18,
            f"sample-more-collect consistent across seeds: {dir_note}",
            transform=ax2.transAxes,
            ha="center",
            fontsize=8,
            color="#444",
        )
    else:
        ax2.text(0.5, 0.5, "no dual-decode seeds yet", ha="center", va="center")
        ax2.set_axis_off()

    fig.tight_layout()
    fig_path = OUT / "late_dep_vs_fail.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Markdown results summary
    md = [
        "# All-seeds late-dep reframe — results (auto)",
        "",
        f"**Verdict:** `{summary['verdict']}`",
        "",
        summary["one_liner"],
        "",
        f"**Earned:** {summary['claim_bounds']['earned']}",
        "",
        f"**Not earned:** {summary['claim_bounds']['not_earned']}",
        "",
        f"![[late_dep_vs_fail.png]]",
        "",
        "## Late cells",
        "",
        "| Seed | Decode | fail | S | GM | dep | State | competent? |",
        "|------|--------|-----:|--:|---:|----:|-------|------------|",
    ]
    for r in sorted(rows, key=lambda z: (z["seed"], z["action_mode"])):
        comp = "yes" if r["fail"] < FAIL_COMPETENT else "no"
        md.append(
            f"| {r['seed']} | {r['action_mode']} | {r['fail']:.3f} | {r['success']:.3f} | "
            f"{r['gm']:.3f} | {r['dep']:+.3f} | {r['state']} | {comp} |"
        )
    md += [
        "",
        "## Decode Δ",
        "",
        "| Seed | dep_g | dep_s | Δ | State g→s | flip? |",
        "|------|------:|------:|--:|-----------|-------|",
    ]
    for d in decode_deltas:
        md.append(
            f"| {d['seed']} | {d['dep_greedy']:+.3f} | {d['dep_sample']:+.3f} | "
            f"{d['delta_dep']:+.3f} | {d['state_greedy']} → {d['state_sample']} | "
            f"{'yes' if d['flip'] else 'no'} |"
        )
    md += [
        "",
        f"**Near-zero (illustrative |dep|<{BAND}):** {len(in_band)}/{len(competent)} "
        f"competent cells. Prefer qualitative 'within ~2 noise-widths of 0'; "
        f"do not quote {band_frac:.0%} as effect size.",
        "",
        f"**Decode-robust competent persistent-GM controls:** {robust_controls or 'none'}.",
        "",
        f"**Cap-fail tails (excluded):** {summary['cap_fail_tails'] or 'none'}.",
        "",
        f"**Both-decode |dep|≥0.15 (competent):** {both_large or 'none'}.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(md))

    print(summary["one_liner"])
    print(f"verdict={summary['verdict']}")
    print(f"wrote {OUT / 'reframe_summary.json'}")
    print(f"wrote {fig_path}")
    print(f"wrote {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
