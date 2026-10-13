#!/usr/bin/env python3
"""Coin Fisher / KL-metric sensitivity: block3 present-absent difference maps + fc dh^T F dh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "train-procgen-pytorch"))
# Vendored layout
sys.path.insert(0, str(ROOT / "pkg"))
sys.path.insert(0, str(ROOT / "run_scripts"))
sys.path.insert(0, str(ROOT / "vendored" / "train-procgen-pytorch"))
PROCGEN_AISC = ROOT / "external" / "procgenAISC"
if (PROCGEN_AISC / "procgen").exists():
    sys.path.insert(0, str(PROCGEN_AISC))
VENDORED_AISC = ROOT / "vendored" / "procgenAISC"
if (VENDORED_AISC / "procgen").exists():
    sys.path.insert(0, str(VENDORED_AISC))

from gm_geometry.probes.coin_location import (  # noqa: E402
    BLOCK3_HW,
    PIXELS_PER_FEAT,
    agent_feat_yx_camera,
    find_floor_cell,
    get_info,
    labels_from_info,
    obs_nchw_after_set_coin,
    pixel_diff_feat_cell,
    set_coin,
)
from gm_geometry.probes.fisher_sensitivity import (  # noqa: E402
    block3_fisher_pack,
    coin_path_separation,
    fc_fisher_and_act,
    fisher_trace_in_topk_pcs,
    kl_categorical,
    mean_map,
    pack_scalar_rows,
    path_band_cells,
    quadratic_form,
    sensitivity_along,
    spatial_fisher_map_from_block3_grads,
    summarize_map_vs_landmarks,
)
from gm_geometry.utils import device as _default_device  # noqa: E402
from validate_langosco_coinrun_200m import load_policy, make_env  # noqa: E402

PIXEL_DIFF_THR = 0.05
PAL = {
    "gm": "#7A1520",
    "nongm": "#3D8B8B",
    "init": "#8E8E8E",
    "ink": "#2A2A2A",
    "paper": "#FFFFFF",
}


def load_init_policy(ckpt_for_arch: Path, device: str, init_seed: int | None = None):
    """Same architecture as ckpt_for_arch with random weights (init floor).

    init_seed seeds torch/numpy for a reproducible draw; None leaves the ambient RNG state uncontrolled.
    """
    import yaml
    from common.model import ImpalaModel
    from common.policy import CategoricalPolicy

    if init_seed is not None:
        torch.manual_seed(int(init_seed))
        np.random.seed(int(init_seed))

    if not ckpt_for_arch.exists():
        raise FileNotFoundError(ckpt_for_arch)
    # Architecture matches load_policy in validate_langosco_coinrun_200m.
    train_cfg = None
    for cand in (
        ROOT / "vendored" / "train-procgen-pytorch",
        ROOT / "external" / "train-procgen-pytorch",
    ):
        cfg = cand / "hyperparams" / "procgen" / "config.yml"
        if cfg.exists():
            train_cfg = cfg
            break
    if train_cfg is None:
        raise FileNotFoundError("train-procgen-pytorch hyperparams missing")
    with open(train_cfg) as f:
        recurrent = yaml.safe_load(f)["hard-500"]["recurrent"]
    probe = make_env("coinrun", num_levels=1, start_level=0, random_percent=0)
    model = ImpalaModel(in_channels=probe.observation_space.shape[0])
    policy = CategoricalPolicy(model, recurrent, probe.action_space.n)
    policy.to(device)
    policy.eval()
    if hasattr(probe, "close"):
        probe.close()
    return policy


def harvest_offpath_frames(
    *,
    n_frames: int,
    start_level: int,
    max_levels: int,
    max_steps: int,
    warm_policy=None,
    device: str = "cpu",
    min_agent_x: float = 8.0,
    warm_cap: int = 80,
) -> list[dict]:
    """Off-path frames with the coin behind the agent (feature columns 0-2, left of the rightward path band).

    A greedy warm-up rollout moves the agent to agent_x >= min_agent_x; holding right alone stalls near spawn.
    """
    rows: list[dict] = []
    level = start_level
    used = 0
    rejects = {
        "early_done": 0,
        "no_lab": 0,
        "ax_too_left": 0,
        "no_xy": 0,
        "near_endwall": 0,
        "set_fail": 0,
        "pixel_gate": 0,
        "not_behind": 0,
        "on_path_band": 0,
        "offscreen": 0,
    }

    def _candidate_xys(env, lab: dict) -> list[tuple[int, int]]:
        ax = int(lab["agent_x"])
        out: list[tuple[int, int]] = []
        for min_dx in (2, 3, 4, 1, 5):
            cand = find_floor_cell(env, prefer_left_of_agent=True, min_dx=min_dx)
            if cand is not None and cand[0] <= ax - 2:
                out.append(cand)
        seen: set[tuple[int, int]] = set()
        uniq: list[tuple[int, int]] = []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def _try_obs_pair(env, xy: tuple[int, int]):
        if set_coin(env, xy[0], xy[1], 1) != 0:
            return "set_fail"
        obs_a = obs_nchw_after_set_coin(env)
        lab_a = labels_from_info(get_info(env))
        if set_coin(env, xy[0], xy[1], 0) != 0:
            return "set_fail"
        obs_gone = obs_nchw_after_set_coin(env)
        set_coin(env, xy[0], xy[1], 1)
        if lab_a is None:
            return "pixel_gate"
        if float(np.max(np.abs(obs_a - obs_gone))) < PIXEL_DIFF_THR:
            return "pixel_gate"
        cell = pixel_diff_feat_cell(obs_a, obs_gone, thr=PIXEL_DIFF_THR)
        if cell is None:
            return "pixel_gate"
        return obs_a, obs_gone, lab_a, cell

    while len(rows) < n_frames and used < max_levels:
        env = make_env("coinrun_aisc", num_levels=1, start_level=level)
        done_flag = False
        obs = env.reset()
        hidden = None
        if warm_policy is not None:
            hidden = np.zeros(
                (env.num_envs, warm_policy.embedder.output_dim), dtype=np.float32
            )
        lab0 = labels_from_info(get_info(env))
        n_warm = min(int(warm_cap), int(max_steps))
        for _t in range(n_warm):
            if warm_policy is not None:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    hid_t = torch.as_tensor(hidden, dtype=torch.float32, device=device)
                    done_t = torch.zeros(env.num_envs, device=device)
                    dist, _v, next_hid = warm_policy(obs_t, hid_t, done_t)
                    act = dist.probs.argmax(-1).cpu().numpy()
                    hidden = next_hid.cpu().numpy()
            else:
                # Fallback: pure right (caps ~ax=4.5 on spawn platforms).
                act = np.array([7], dtype=np.int64)
            obs, _r, done, _i = env.step(act)
            if done[0]:
                done_flag = True
                break
            lab0 = labels_from_info(get_info(env))
            if lab0 is not None and float(lab0["agent_x"]) >= float(min_agent_x):
                break
        if done_flag:
            env.close()
            level += 1
            used += 1
            rejects["early_done"] += 1
            continue
        if lab0 is None:
            lab0 = labels_from_info(get_info(env))
        if lab0 is None:
            env.close()
            level += 1
            used += 1
            rejects["no_lab"] += 1
            continue
        # Need enough runway behind for feature-left placement.
        if float(lab0["agent_x"]) < float(min_agent_x) - 0.5:
            env.close()
            level += 1
            used += 1
            rejects["ax_too_left"] += 1
            if used <= 3 or used % 500 == 0:
                print(
                    f"  miss level={level} agent_x={lab0.get('agent_x')} "
                    f"rejects={rejects}",
                    flush=True,
                )
            continue

        placed = None
        for xy in _candidate_xys(env, lab0):
            if abs(xy[0] - lab0["inv_coin_x"]) < 8:
                rejects["near_endwall"] += 1
                continue
            got = _try_obs_pair(env, xy)
            if got == "set_fail":
                rejects["set_fail"] += 1
                continue
            if got == "pixel_gate":
                rejects["pixel_gate"] += 1
                continue
            obs_a, obs_gone, lab_a, cell_a = got
            agent_xy = (float(lab_a["agent_x"]), float(lab_a["agent_y"]))
            coin_xy = (float(lab_a["coin_x"]), float(lab_a["coin_y"]))
            if abs(coin_xy[0] - agent_xy[0]) > 16 or abs(coin_xy[1] - agent_xy[1]) > 16:
                rejects["offscreen"] += 1
                continue
            if coin_xy[0] > agent_xy[0] - 2.0:
                rejects["not_behind"] += 1
                continue
            agent_feat = agent_feat_yx_camera()
            cy, cx = int(cell_a[0]), int(cell_a[1])
            ay, ax = float(agent_feat[0]), float(agent_feat[1])
            # Feature-left of agent ⇒ outside the rightward path band.
            if cx + 0.5 >= ax:
                rejects["on_path_band"] += 1
                continue
            # Primary DV is radius-0 coin vs path band. With camera-centred
            # agent_feat (~3.5), path starts at col 4; behind coin at cols 0–2
            # clears min_L1 ≥ 1 by construction when off-band.
            sep = coin_path_separation(cell_a, agent_feat, radius=0)
            min_l1 = float(sep.get("min_l1_coin_to_path", 0.0))
            if not (min_l1 >= 1.0):
                rejects["on_path_band"] += 1
                continue
            if float(sep.get("overlap", 1.0)) > 0.0:
                rejects["on_path_band"] += 1
                continue
            sep_r1 = coin_path_separation(cell_a, agent_feat, radius=1)
            sep["overlap_r1"] = float(sep_r1.get("overlap", 0.0))
            sep["centre_on_path"] = 0.0
            sep["feat_left"] = 1.0
            placed = {
                "level": int(level),
                "obs_a": obs_a.astype(np.float32),
                "obs_gone": obs_gone.astype(np.float32),
                "obs_b": None,
                "coin_cell_a": (cy, cx),
                "coin_cell_b": None,
                "agent_feat_yx": (ay, ax),
                "agent_xy": agent_xy,
                "coin_xy_a": coin_xy,
                "xy_a": (int(xy[0]), int(xy[1])),
                "xy_b": None,
                "sep": sep,
                "centre_on_path": False,
                "feat_left": True,
            }
            break

        if placed is None:
            env.close()
            level += 1
            used += 1
            rejects["no_xy"] += 1
            if used <= 3 or used % 500 == 0:
                print(
                    f"  miss level={level} agent_x={lab0.get('agent_x')} "
                    f"rejects={rejects}",
                    flush=True,
                )
            continue

        rows.append(placed)
        env.close()
        level += 1
        used += 1
        if len(rows) % 8 == 0:
            mean_l1 = float(np.mean([r["sep"]["min_l1_coin_to_path"] for r in rows]))
            ov = float(np.mean([r["sep"]["overlap"] for r in rows]))
            print(
                f"  harvested {len(rows)}/{n_frames} mean_min_L1={mean_l1:.2f} "
                f"overlap={ov:.3f} rejects={rejects}",
                flush=True,
            )

    off = sum(1 for r in rows if not r.get("centre_on_path", True))
    mean_l1 = (
        float(np.mean([r["sep"]["min_l1_coin_to_path"] for r in rows])) if rows else float("nan")
    )
    ov = float(np.mean([r["sep"]["overlap"] for r in rows])) if rows else float("nan")
    print(
        f"harvest done n={len(rows)} used={used} off_path={off} "
        f"mean_min_L1={mean_l1:.2f} overlap={ov:.3f} rejects={rejects}",
        flush=True,
    )
    if rows and off < len(rows):
        raise RuntimeError(
            f"harvest kept on-path frames: off_path={off}/{len(rows)} (bug)"
        )
    return rows


# Primary figure is the present-absent block3 difference map for GM vs Non-GM
# vs init on shared off-path frames; nulls are coin-absent matched pairs,
# shuffled marginal-pi Fisher, and the init arm. Also reports coin/path ratio,
# feature-cell overlap, present/absent dh^T F dh <-> 2KL, and tr(F) in the
# top-k PC subspace of h.
def score_agent_on_frames(
    policy,
    frames: list[dict],
    *,
    device: str,
    label: str,
    pc_k: int = 16,
) -> dict:
    maps_present = []
    maps_absent = []
    maps_diff = []
    maps_shuf = []
    map_rows = []
    dh_rows = []
    probs_list = []
    grads_present = []
    h_present = []
    f_present = []

    # Pass 1: present / absent maps + collect π for marginal null + fc acts.
    for i, fr in enumerate(frames):
        grads_a, probs_a, _acts_a, m_a = block3_fisher_pack(
            policy, fr["obs_a"], device
        )
        _g_g, _p_g, _acts_g, m_gone = block3_fisher_pack(
            policy, fr["obs_gone"], device
        )
        m_diff = m_a - m_gone
        maps_present.append(m_a)
        maps_absent.append(m_gone)
        maps_diff.append(m_diff)
        grads_present.append(grads_a)
        probs_list.append(probs_a)

        # Primary DV: radius-0 (disjoint at min_L1≥1). Radius-1 logged only —
        # with this harvest it usually overlaps the half-plane path band.
        summ = summarize_map_vs_landmarks(
            m_a,
            coin_yx=fr["coin_cell_a"],
            agent_yx=fr["agent_feat_yx"],
            radius=0,
        )
        summ_r1 = summarize_map_vs_landmarks(
            m_a,
            coin_yx=fr["coin_cell_a"],
            agent_yx=fr["agent_feat_yx"],
            radius=1,
        )
        summ_diff = summarize_map_vs_landmarks(
            m_diff,
            coin_yx=fr["coin_cell_a"],
            agent_yx=fr["agent_feat_yx"],
            radius=0,
        )
        summ_diff_r1 = summarize_map_vs_landmarks(
            m_diff,
            coin_yx=fr["coin_cell_a"],
            agent_yx=fr["agent_feat_yx"],
            radius=1,
        )
        # Signed coin mass on difference map (generic structure cancels).
        summ["diff_coin_mass"] = summ_diff["coin_mass"]
        summ["diff_coin_frac"] = summ_diff["coin_frac"]
        summ["diff_path_mass"] = summ_diff["path_mass"]
        summ["diff_path_frac"] = summ_diff["path_frac"]
        summ["diff_coin_over_path"] = summ_diff["coin_over_path"]
        summ["diff_coin_minus_path_frac"] = summ_diff["coin_minus_path_frac"]
        summ["coin_over_path_r1"] = summ_r1["coin_over_path"]
        summ["diff_coin_over_path_r1"] = summ_diff_r1["coin_over_path"]
        summ["diff_coin_mass_r1"] = summ_diff_r1["coin_mass"]
        summ["diff_path_mass_r1"] = summ_diff_r1["path_mass"]
        summ["absent_coin_frac"] = summarize_map_vs_landmarks(
            m_gone,
            coin_yx=fr["coin_cell_a"],
            agent_yx=fr["agent_feat_yx"],
            radius=0,
        )["coin_frac"]

        # Present/absent Δh ↔ KL on the *same* pair (identity check).
        f_a, p_a, h_a = fc_fisher_and_act(policy, fr["obs_a"], device)
        _f_g, p_g, h_g = fc_fisher_and_act(policy, fr["obs_gone"], device)
        dh = h_a - h_g
        q = quadratic_form(f_a, dh)
        s = sensitivity_along(f_a, dh)
        kl = kl_categorical(p_a, p_g)
        dh_rows.append(
            {
                "frame_i": i,
                "level": fr["level"],
                "pair": "present_absent",
                "quad": q,
                "S": s,
                "kl": kl,
                "two_kl": 2.0 * kl,
                "dh_norm": float(np.linalg.norm(dh)),
                "abs_err_quad_vs_2kl": abs(q - 2.0 * kl),
            }
        )
        h_present.append(h_a)
        f_present.append(f_a)

        summ["frame_i"] = i
        summ["level"] = fr["level"]
        map_rows.append(summ)
        if (i + 1) % 8 == 0:
            print(f"  [{label}] scored {i + 1}/{len(frames)}", flush=True)

    # Marginal action distribution (state-averaged π) for shuffled-action Fisher.
    pi_bar = np.mean(np.stack(probs_list, axis=0), axis=0)
    pi_bar = pi_bar / (pi_bar.sum() + 1e-12)
    for grads_a in grads_present:
        maps_shuf.append(spatial_fisher_map_from_block3_grads(grads_a, pi_bar))

    shuf_rows = []
    for i, fr in enumerate(frames):
        sh = summarize_map_vs_landmarks(
            maps_shuf[i],
            coin_yx=fr["coin_cell_a"],
            agent_yx=fr["agent_feat_yx"],
            radius=0,
        )
        sh["frame_i"] = i
        shuf_rows.append(sh)
        map_rows[i]["shuf_coin_frac"] = sh["coin_frac"]
        map_rows[i]["shuf_path_frac"] = sh["path_frac"]
        map_rows[i]["shuf_coin_over_path"] = sh["coin_over_path"]
        # Present minus shuffled floor at coin cell.
        map_rows[i]["present_minus_shuf_coin_frac"] = (
            map_rows[i]["coin_frac"] - sh["coin_frac"]
        )

    # Quiet-colour generalisation: frac of tr(F) in top-k PC subspace of h.
    H = np.stack(h_present, axis=0)
    pc_rows = [fisher_trace_in_topk_pcs(f, H, k=pc_k) for f in f_present]

    overlap_rate = float(np.mean([fr["sep"]["overlap"] for fr in frames]))
    mean_min_l1 = float(np.mean([fr["sep"]["min_l1_coin_to_path"] for fr in frames]))

    out: dict = {
        "label": label,
        "n_frames": len(frames),
        "map_summary": pack_scalar_rows(map_rows),
        "shuf_summary": pack_scalar_rows(shuf_rows),
        "mean_map": mean_map(maps_present).tolist(),
        "mean_map_absent": mean_map(maps_absent).tolist(),
        "mean_map_diff": mean_map(maps_diff).tolist(),
        "mean_map_shuf": mean_map(maps_shuf).tolist(),
        "per_frame_maps": [m.tolist() for m in maps_present],
        "per_frame_maps_diff": [m.tolist() for m in maps_diff],
        "per_frame_map_metrics": map_rows,
        "pi_bar": pi_bar.tolist(),
        "overlap_rate": overlap_rate,
        "mean_min_l1_coin_to_path": mean_min_l1,
        "pc_topk": {
            "k": pc_k,
            "summary": pack_scalar_rows(pc_rows),
            "frac_topk_mean": float(np.nanmean([r["frac_topk"] for r in pc_rows])),
        },
    }
    quads = np.asarray([r["quad"] for r in dh_rows], dtype=np.float64)
    twokl = np.asarray([r["two_kl"] for r in dh_rows], dtype=np.float64)
    if len(dh_rows) >= 3 and np.std(quads) > 1e-12 and np.std(twokl) > 1e-12:
        corr = float(np.corrcoef(quads, twokl)[0, 1])
    else:
        corr = float("nan")
    out["fc_dh"] = {
        "pair": "present_absent",
        "summary": pack_scalar_rows(dh_rows),
        "corr_quad_vs_2kl": corr,
        "per_frame": dh_rows,
    }
    return out


def _obs_to_hwc_u8(obs_chw: np.ndarray) -> np.ndarray:
    x = np.asarray(obs_chw, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    hwc = np.transpose(x, (1, 2, 0))
    return np.clip(hwc * 255.0, 0, 255).astype(np.uint8)


def save_overlay_figure(
    frames: list[dict],
    agent_results: dict[str, dict],
    out_png: Path,
    *,
    n_show: int = 4,
    map_key: str = "per_frame_maps_diff",
    signed: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    labels = [k for k in ("nongm_7001", "gm_6033", "init") if k in agent_results]
    if not labels:
        labels = list(agent_results.keys())
    n_show = min(n_show, len(frames))
    fig, axes = plt.subplots(
        len(labels),
        n_show,
        figsize=(2.4 * n_show, 2.4 * len(labels)),
        squeeze=False,
    )
    fig.patch.set_facecolor(PAL["paper"])
    for r, lab in enumerate(labels):
        maps = agent_results[lab].get(map_key) or agent_results[lab]["per_frame_maps"]
        color = PAL.get(
            "gm" if "gm" in lab else "nongm" if "nongm" in lab else "init",
            PAL["ink"],
        )
        if signed:
            cmap = LinearSegmentedColormap.from_list(
                f"fish_{lab}", ["#B0B0B0", PAL["paper"], color]
            )
        else:
            cmap = LinearSegmentedColormap.from_list(
                f"fish_{lab}", [PAL["paper"], color]
            )
        for c in range(n_show):
            ax = axes[r][c]
            ax.set_facecolor(PAL["paper"])
            img = _obs_to_hwc_u8(frames[c]["obs_a"])
            ax.imshow(img, interpolation="nearest")
            m = np.asarray(maps[c], dtype=np.float64)
            if signed:
                vmax = float(np.max(np.abs(m))) + 1e-12
                m_n = m / vmax
                kw = {"vmin": -1.0, "vmax": 1.0}
            else:
                m_n = m / (m.max() + 1e-12)
                kw = {}
            m_up = np.kron(m_n, np.ones((PIXELS_PER_FEAT, PIXELS_PER_FEAT)))
            ax.imshow(
                m_up,
                cmap=cmap,
                alpha=0.55,
                interpolation="nearest",
                **kw,
            )
            cy, cx = frames[c]["coin_cell_a"]
            ax.scatter(
                [cx * PIXELS_PER_FEAT + PIXELS_PER_FEAT / 2],
                [cy * PIXELS_PER_FEAT + PIXELS_PER_FEAT / 2],
                s=28,
                facecolors="none",
                edgecolors="white",
                linewidths=1.2,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.set_ylabel(lab, color=PAL["ink"], fontsize=9)
    fig.tight_layout(pad=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=PAL["paper"])
    plt.close(fig)


def save_mean_map_figure(agent_results: dict[str, dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    labels = [k for k in ("nongm_7001", "gm_6033", "init") if k in agent_results]
    if not labels:
        labels = list(agent_results.keys())
    kinds = [
        ("mean_map", "present"),
        ("mean_map_diff", "present−absent"),
        ("mean_map_shuf", "shuf-π"),
    ]
    fig, axes = plt.subplots(
        len(kinds),
        len(labels),
        figsize=(3.0 * len(labels), 2.8 * len(kinds)),
        squeeze=False,
    )
    fig.patch.set_facecolor(PAL["paper"])
    for r, (key, title) in enumerate(kinds):
        for c, lab in enumerate(labels):
            ax = axes[r][c]
            color = PAL.get(
                "gm" if "gm" in lab else "nongm" if "nongm" in lab else "init",
                PAL["ink"],
            )
            m = np.asarray(agent_results[lab][key], dtype=np.float64)
            if "diff" in key:
                cmap = LinearSegmentedColormap.from_list(
                    f"mean_{lab}_{key}", ["#B0B0B0", PAL["paper"], color]
                )
                vmax = float(np.max(np.abs(m))) + 1e-12
                ax.imshow(
                    m / vmax,
                    cmap=cmap,
                    interpolation="nearest",
                    vmin=-1.0,
                    vmax=1.0,
                )
            else:
                cmap = LinearSegmentedColormap.from_list(
                    f"mean_{lab}_{key}", [PAL["paper"], color]
                )
                ax.imshow(
                    m / (m.max() + 1e-12),
                    cmap=cmap,
                    interpolation="nearest",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, color=PAL["ink"], fontsize=10)
            if c == 0:
                ax.set_ylabel(title, color=PAL["ink"], fontsize=9)
            for spine in ax.spines.values():
                spine.set_color(PAL["ink"])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=PAL["paper"])
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-gm", type=Path, required=True)
    p.add_argument("--ckpt-nongm", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=48)
    p.add_argument("--start-level", type=int, default=1_000_000)
    p.add_argument("--max-levels", type=int, default=800)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-show", type=int, default=4)
    p.add_argument("--pc-k", type=int, default=16)
    p.add_argument("--skip-init", action="store_true")
    p.add_argument("--min-agent-x", type=float, default=8.0)
    p.add_argument("--warm-cap", type=int, default=80)
    args = p.parse_args()
    args.device = args.device or str(_default_device())

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Shared Non-GM policy for harvest warm-up (hold-right cannot clear spawn).
    print("=== load Non-GM for warm-up ===", flush=True)
    warm_pol = load_policy(args.ckpt_nongm, args.device)

    print("=== harvest off-path frames (policy warm + behind) ===", flush=True)
    frames = harvest_offpath_frames(
        n_frames=args.n_frames,
        start_level=args.start_level,
        max_levels=args.max_levels,
        max_steps=args.max_steps,
        warm_policy=warm_pol,
        device=args.device,
        min_agent_x=args.min_agent_x,
        warm_cap=args.warm_cap,
    )
    print(f"harvested {len(frames)} frames", flush=True)
    if len(frames) < 30:
        raise SystemExit(
            f"too few frames for contrast under radius-0 / min_L1≥1: {len(frames)} "
            "(need ≥30; this is the capped Coin Fisher attempt — park if empty)"
        )
    if float(np.mean([fr["sep"]["overlap"] for fr in frames])) > 0.0:
        raise SystemExit("harvest kept radius-0/path overlap frames (bug)")
    overlap_rate = float(np.mean([fr["sep"]["overlap"] for fr in frames]))
    mean_min_l1 = float(np.mean([fr["sep"]["min_l1_coin_to_path"] for fr in frames]))
    centre_on = float(np.mean([1.0 if fr.get("centre_on_path") else 0.0 for fr in frames]))
    feat_left = float(np.mean([1.0 if fr.get("feat_left") else 0.0 for fr in frames]))
    print(
        f"coin/path feature overlap rate={overlap_rate:.3f} "
        f"mean_min_L1={mean_min_l1:.2f} centre_on_path={centre_on:.3f} "
        f"feat_left={feat_left:.3f}",
        flush=True,
    )
    if centre_on > 0.05:
        raise SystemExit(
            f"too many on-path frames: centre_on_path={centre_on:.3f} "
            "(need true off-path harvest)"
        )

    agents = {
        "nongm_7001": warm_pol,
        "gm_6033": load_policy(args.ckpt_gm, args.device),
    }
    if not args.skip_init:
        agents["init"] = load_init_policy(args.ckpt_nongm, args.device)

    results = {}
    for lab, pol in agents.items():
        print(f"=== score {lab} ===", flush=True)
        results[lab] = score_agent_on_frames(
            pol, frames, device=args.device, label=lab, pc_k=args.pc_k
        )

    summary = {
        "n_frames": len(frames),
        "start_level": args.start_level,
        "protocol": (
            "block3 Fisher maps (present, absent, present−absent, shuf-π); "
            "capped Coin attempt: primary DV = radius-0 coin/path on diff map "
            "with harvest min_L1≥1 (disjoint single cell); radius-1 logged only "
            "(usually overlaps half-plane path); "
            "report median ratio (mean disclosed); "
            "paired Non-GM−GM on shared frames; "
            "fc present/absent ΔhᵀFΔh↔2KL + top-k PC Fisher fraction. "
            "If contrast fails, park Coin Fisher."
        ),
        "nulls": [
            "present_minus_absent_map",
            "shuffled_marginal_action_fisher",
            "init_arm",
        ],
        "primary_radius": 0,
        "logged_radius": 1,
        "overlap_rate": overlap_rate,
        "mean_min_l1_coin_to_path": mean_min_l1,
        "pc_k": args.pc_k,
        "agents": {},
        "frame_meta": [
            {
                "level": fr["level"],
                "coin_cell_a": fr["coin_cell_a"],
                "coin_cell_b": fr["coin_cell_b"],
                "agent_feat_yx": fr["agent_feat_yx"],
                "xy_a": fr["xy_a"],
                "xy_b": fr["xy_b"],
                "sep": fr["sep"],
            }
            for fr in frames
        ],
    }
    for lab, res in results.items():
        ms = res["map_summary"]
        summary["agents"][lab] = {
            "map_summary": ms,
            "shuf_summary": res["shuf_summary"],
            "mean_map": res["mean_map"],
            "mean_map_diff": res["mean_map_diff"],
            "mean_map_shuf": res["mean_map_shuf"],
            "fc_dh": res.get("fc_dh", {}).get("summary"),
            "corr_quad_vs_2kl": res.get("fc_dh", {}).get("corr_quad_vs_2kl"),
            "fc_dh_pair": res.get("fc_dh", {}).get("pair"),
            # Median is primary (std ≫ mean); mean disclosed. Primary = r0.
            "diff_coin_over_path_median": ms.get("diff_coin_over_path", {}).get(
                "median"
            ),
            "diff_coin_over_path_mean": ms.get("diff_coin_over_path", {}).get("mean"),
            "diff_coin_over_path_r1_median": ms.get("diff_coin_over_path_r1", {}).get(
                "median"
            ),
            "diff_coin_over_path_r1_mean": ms.get("diff_coin_over_path_r1", {}).get(
                "mean"
            ),
            "coin_over_path_median": ms.get("coin_over_path", {}).get("median"),
            "coin_over_path_mean": ms.get("coin_over_path", {}).get("mean"),
            "coin_frac_mean": ms.get("coin_frac", {}).get("mean"),
            "path_frac_mean": ms.get("path_frac", {}).get("mean"),
            "diff_coin_frac_mean": ms.get("diff_coin_frac", {}).get("mean"),
            "diff_coin_mass_mean": ms.get("diff_coin_mass", {}).get("mean"),
            "diff_path_mass_mean": ms.get("diff_path_mass", {}).get("mean"),
            "diff_coin_mass_median": ms.get("diff_coin_mass", {}).get("median"),
            "diff_path_mass_median": ms.get("diff_path_mass", {}).get("median"),
            "shuf_coin_frac_mean": ms.get("shuf_coin_frac", {}).get("mean"),
            "present_minus_shuf_coin_frac_mean": ms.get(
                "present_minus_shuf_coin_frac", {}
            ).get("mean"),
            "frac_fisher_in_topk_pcs": res["pc_topk"]["frac_topk_mean"],
            "pc_topk": res["pc_topk"],
            "overlap_rate": res["overlap_rate"],
            "mean_min_l1_coin_to_path": res["mean_min_l1_coin_to_path"],
        }

    # Paired Non-GM − GM on identical frames (free because harvest is shared).
    if "nongm_7001" in results and "gm_6033" in results:
        ng = results["nongm_7001"]["per_frame_map_metrics"]
        gm = results["gm_6033"]["per_frame_map_metrics"]
        n = min(len(ng), len(gm))
        d_ng = np.asarray([ng[i]["diff_coin_over_path"] for i in range(n)], dtype=np.float64)
        d_gm = np.asarray([gm[i]["diff_coin_over_path"] for i in range(n)], dtype=np.float64)
        d_ng1 = np.asarray(
            [ng[i].get("diff_coin_over_path_r1", np.nan) for i in range(n)],
            dtype=np.float64,
        )
        d_gm1 = np.asarray(
            [gm[i].get("diff_coin_over_path_r1", np.nan) for i in range(n)],
            dtype=np.float64,
        )
        delta = d_ng - d_gm
        delta1 = d_ng1 - d_gm1
        summary["paired_nongm_minus_gm"] = {
            "n": int(n),
            "diff_coin_over_path_r0": {
                "delta_median": float(np.nanmedian(delta)),
                "delta_mean": float(np.nanmean(delta)),
                "frac_nongm_gt_gm": float(np.mean(d_ng > d_gm)),
                "nongm_median": float(np.nanmedian(d_ng)),
                "gm_median": float(np.nanmedian(d_gm)),
                "nongm_mean": float(np.nanmean(d_ng)),
                "gm_mean": float(np.nanmean(d_gm)),
            },
            "diff_coin_over_path_r1": {
                "delta_median": float(np.nanmedian(delta1)),
                "delta_mean": float(np.nanmean(delta1)),
                "frac_nongm_gt_gm": float(np.mean(d_ng1 > d_gm1)),
                "nongm_median": float(np.nanmedian(d_ng1)),
                "gm_median": float(np.nanmedian(d_gm1)),
            },
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    full = {
        lab: {
            k: v
            for k, v in res.items()
            if k not in ("per_frame_maps", "per_frame_maps_diff")
        }
        for lab, res in results.items()
    }
    np.savez_compressed(
        out / "maps.npz",
        **{f"{lab}_mean": np.asarray(results[lab]["mean_map"]) for lab in results},
        **{
            f"{lab}_mean_diff": np.asarray(results[lab]["mean_map_diff"])
            for lab in results
        },
        **{
            f"{lab}_mean_shuf": np.asarray(results[lab]["mean_map_shuf"])
            for lab in results
        },
        **{
            f"{lab}_maps": np.stack(
                [np.asarray(m) for m in results[lab]["per_frame_maps"]], axis=0
            )
            for lab in results
        },
        **{
            f"{lab}_maps_diff": np.stack(
                [np.asarray(m) for m in results[lab]["per_frame_maps_diff"]], axis=0
            )
            for lab in results
        },
    )
    (out / "full_metrics.json").write_text(json.dumps(full, indent=2, default=float))

    save_overlay_figure(
        frames,
        results,
        out / "overlay_diff.png",
        n_show=args.n_show,
        map_key="per_frame_maps_diff",
        signed=True,
    )
    save_overlay_figure(
        frames,
        results,
        out / "overlay_present.png",
        n_show=args.n_show,
        map_key="per_frame_maps",
        signed=False,
    )
    save_mean_map_figure(results, out / "mean_maps.png")

    print("=== DONE ===", flush=True)
    print(
        f"frame separation: n={summary['n_frames']} overlap_rate={overlap_rate:.3f} "
        f"mean_min_L1={mean_min_l1:.2f}",
        flush=True,
    )
    for lab, s in summary["agents"].items():
        print(
            f"{lab}: diff_coin/path r0 med={s.get('diff_coin_over_path_median')} "
            f"mean={s.get('diff_coin_over_path_mean')} "
            f"r1_med={s.get('diff_coin_over_path_r1_median')} "
            f"diff_cm_med={s.get('diff_coin_mass_median')} "
            f"diff_pm_med={s.get('diff_path_mass_median')} "
            f"shuf_coin={s['shuf_coin_frac_mean']:.4f} "
            f"corr_2kl={s.get('corr_quad_vs_2kl')} "
            f"pc_topk={s['frac_fisher_in_topk_pcs']:.3f}",
            flush=True,
        )
    paired = summary.get("paired_nongm_minus_gm")
    if paired:
        r0 = paired["diff_coin_over_path_r0"]
        print(
            f"paired NonGM−GM r0: delta_med={r0['delta_median']:.4f} "
            f"delta_mean={r0['delta_mean']:.4f} "
            f"frac_NG>GM={r0['frac_nongm_gt_gm']:.3f} "
            f"NG_med={r0['nongm_median']:.4f} GM_med={r0['gm_median']:.4f}",
            flush=True,
        )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
