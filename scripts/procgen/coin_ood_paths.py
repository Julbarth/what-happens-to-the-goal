#!/usr/bin/env python3
"""Coin OOD path figure: GM vs Non-GM rollouts on natural levels where the coin is not at the end wall."""
from __future__ import annotations

import os
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_figures"
from validate_langosco_coinrun_200m import load_policy, make_env  # noqa: E402
from gm_geometry.probes.coin_location import get_info, labels_from_info  # noqa: E402
from gm_geometry.utils import device as default_device  # noqa: E402
from coin_envmove_paths import blank_coin, coin_pixel  # noqa: E402

DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
GM_CKPT = DATA_ROOT / "checkpoints/coin/6051_gm_55M.pth"
CTRL_CKPT = DATA_ROOT / "checkpoints/coin/7001_teach.pth"

GM_HEX, NONGM_HEX = "#7A1520", "#3D8B8B"
GM_RGB = (122, 21, 32)
NONGM_RGB = (61, 139, 139)
INK, MUTE = (42, 42, 42), (110, 110, 110)
OBS = 64


@torch.no_grad()
def greedy_action(policy, obs_chw, device):
    x = torch.as_tensor(obs_chw, device=device, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    h = torch.zeros(x.shape[0], policy.embedder.output_dim, device=device)
    m = torch.zeros(x.shape[0], device=device)
    dist, _v, _ = policy(x, h, m)
    return int(np.argmax(dist.probs.detach().cpu().numpy()[0]))


def chw_u8(chw: np.ndarray) -> np.ndarray:
    x = np.asarray(chw, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def rollout(policy, device, level, *, max_steps, warmup=0):
    """Greedy episode from reset, after an optional warmup driven by the same policy."""
    env = make_env("coinrun_aisc", num_levels=1, start_level=level)
    obs = np.asarray(env.reset()[0], dtype=np.float32)
    try:
        for _ in range(warmup):
            a = greedy_action(policy, obs, device)
            obs_raw, _r, done, _i = env.step(np.array([a]))
            obs = np.asarray(obs_raw[0], dtype=np.float32)
            if done[0]:
                return None

        lab0 = labels_from_info(get_info(env))
        if lab0 is None:
            return None
        coin0 = (float(lab0["coin_x"]), float(lab0["coin_y"]))
        agent0 = (float(lab0["agent_x"]), float(lab0["agent_y"]))
        end_x = float(lab0["end_x"]) if lab0.get("end_x") is not None else None
        if end_x is None and "inv_coin_x" in lab0:
            end_x = float(lab0["inv_coin_x"])

        positions, frames = [], []
        coin_reach = False
        coin_px = None
        max_agent_x = agent0[0]
        for _t in range(max_steps):
            fr = chw_u8(obs)
            frames.append(fr)
            lab = labels_from_info(get_info(env))
            if lab is not None:
                positions.append([float(lab["agent_x"]), float(lab["agent_y"])])
                max_agent_x = max(max_agent_x, float(lab["agent_x"]))
                if lab.get("end_x") is not None:
                    end_x = float(lab["end_x"])
                elif "inv_coin_x" in lab and end_x is None:
                    end_x = float(lab["inv_coin_x"])
            a = greedy_action(policy, obs, device)
            obs_raw, rew, done, _i = env.step(np.array([a]))
            obs = np.asarray(obs_raw[0], dtype=np.float32)
            if float(np.asarray(rew).reshape(-1)[0]) > 0:
                coin_reach = True
                # Collection happens mid-step and the next obs is a reset, so keep the last
                # pre-done frame, blank the coin, and remember its pixel for a ring.
                coin_px = coin_pixel(frames[-1])
                frames[-1] = blank_coin(frames[-1])
                if positions:
                    positions[-1] = [coin0[0], coin0[1]]
            if done[0]:
                break
        # Cut soon after peak x (or the end wall) so GM strips do not show wall-jam frames.
        frames, positions = _trim_progress_hold(frames, positions, end_x, coin_reach=coin_reach)
        out = {
            "positions": positions,
            "frames": frames,
            "coin_reach": coin_reach,
            "coin": coin0,
            "agent0": agent0,
            "end_x": end_x,
            "max_agent_x": float(max_agent_x),
            "n": len(frames),
            "collect_px": coin_px if coin_reach else None,
        }
        return out
    finally:
        if hasattr(env, "close"):
            env.close()


def _trim_progress_hold(frames, positions, end_x, *, coin_reach, hold=2):
    """Cut non-collecting episodes shortly after peak progress (end wall or farthest x)."""
    if coin_reach or not positions:
        return frames, positions
    xs = [float(p[0]) for p in positions]
    peak_i = int(np.argmax(xs))
    cut = min(len(frames), peak_i + hold + 1)
    if end_x is not None:
        for i, x in enumerate(xs):
            if x >= float(end_x) - 0.35:
                cut = min(cut, i + hold + 1)
                break
    return frames[:cut], positions[:cut]


def is_ood_coin(lab, *, min_gap_from_end=3.0) -> bool:
    """Visible coin displaced from the go-right terminus of the training distribution.

    On coinrun_aisc, `inv_coin_x/y` marks the training-style end location while the visible `coin_x/y` is randomized.
    """
    if lab is None:
        return False
    cx = float(lab["coin_x"])
    if "end_x" in lab and lab["end_x"] is not None:
        return cx <= float(lab["end_x"]) - min_gap_from_end
    if "inv_coin_x" in lab:
        return abs(cx - float(lab["inv_coin_x"])) >= min_gap_from_end
    # Fallback: mid-level coin (not far right on a typical short layout).
    return cx <= 16.0


def score_level(gm, ctrl, device, level, *, max_steps):
    """Return showcase dict if GM misses / Non-GM hits on an OOD coin layout."""
    env = make_env("coinrun_aisc", num_levels=1, start_level=level)
    try:
        obs = np.asarray(env.reset()[0], dtype=np.float32)
        lab = labels_from_info(get_info(env))
        if not is_ood_coin(lab):
            return None
    finally:
        if hasattr(env, "close"):
            env.close()

    gm_r = rollout(gm, device, level, max_steps=max_steps)
    ng_r = rollout(ctrl, device, level, max_steps=max_steps)
    if gm_r is None or ng_r is None:
        return None
    # Required contrast: GM misses the coin and drifts right, Non-GM collects.
    if gm_r["coin_reach"]:
        return None
    if not ng_r["coin_reach"]:
        return None
    # GM should actually go right-ish (proxy), not just die early left.
    if gm_r["max_agent_x"] < gm_r["agent0"][0] + 1.5:
        return None
    coin = gm_r["coin"]
    gap = (gm_r["end_x"] - coin[0]) if gm_r["end_x"] is not None else 5.0
    return {
        "level": level,
        "coin": coin,
        "gm": gm_r,
        "nongm": ng_r,
        "gap_from_end": float(gap),
    }


def find_levels(gm, ctrl, device, *, n_want, start, scan, max_steps):
    hits = []
    for level in range(start, start + scan):
        hit = score_level(gm, ctrl, device, level, max_steps=max_steps)
        if hit is None:
            continue
        hits.append(hit)
        print(
            f"  HIT level={level} coin=({hit['coin'][0]:.1f},{hit['coin'][1]:.1f}) "
            f"gap_end={hit['gap_from_end']:.1f} "
            f"gm_x {hit['gm']['agent0'][0]:.1f}->{hit['gm']['max_agent_x']:.1f} "
            f"ng_n={hit['nongm']['n']} gm_n={hit['gm']['n']}",
            flush=True,
        )
        if len(hits) >= n_want:
            break
    return hits


def _xy(arm):
    p = np.asarray(arm["positions"], dtype=np.float64)
    if len(p) == 0:
        return np.array([]), np.array([])
    return p[:, 0], p[:, 1]


def fonts(scale):
    try:
        big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", max(13, scale + 7))
        sm = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", max(11, scale + 4))
    except Exception:
        big = sm = None
    return big, sm


def panel(frame, *, scale, title, accent, font):
    im = Image.fromarray(frame).resize((OBS * scale, OBS * scale), Image.NEAREST).convert("RGB")
    header = 30
    canvas = Image.new("RGB", (im.width, im.height + header), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 7, header], fill=accent)
    d.text((12, 7), title, fill=INK, font=font)
    canvas.paste(im, (0, header))
    return canvas


def hstack(imgs, gap=4):
    w = sum(im.width for im in imgs) + gap * max(len(imgs) - 1, 0)
    h = max((im.height for im in imgs), default=1)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def time_strip(frames, *, scale, title, accent, n_cols, font, collected=False, collect_px=None):
    n = len(frames)
    if n == 0:
        return Image.new("RGB", (OBS * scale, OBS * scale + 30), (255, 255, 255))
    n_cols = min(n_cols, n)
    idxs = [int(round(i * (n - 1) / (n_cols - 1))) for i in range(n_cols)] if n_cols > 1 else [0]
    panels = []
    for i, t in enumerate(idxs):
        label = f"{title}  t={t}/{n - 1}" if i == 0 else f"t={t}"
        is_last = i == len(idxs) - 1
        if collected and is_last:
            label = f"t={t}  collected"
        im = panel(frames[t], scale=scale, title=label, accent=accent, font=font)
        if collected and is_last and collect_px is not None:
            d = ImageDraw.Draw(im)
            header = 30
            cx = float(collect_px[0]) * scale
            cy = float(collect_px[1]) * scale + header
            r = 0.12 * OBS * scale
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0xE8, 0xA0, 0x18), width=3)
        panels.append(im)
    return hstack(panels, gap=3)


def build_png(hits, *, out_path, zoom_steps=28, render_scale=3, n_cols=7):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_lev = len(hits)
    fig, axes = plt.subplots(1, n_lev, figsize=(4.2 * n_lev, 4.4), facecolor="white")
    if n_lev == 1:
        axes = [axes]

    for ax, hit in zip(axes, hits):
        g = _xy(hit["gm"])
        c = _xy(hit["nongm"])
        coin = hit["coin"]
        end_x = hit["gm"].get("end_x")
        if end_x is None:
            end_x = hit["nongm"].get("end_x")
        ax.plot(g[0][:zoom_steps], g[1][:zoom_steps], "-", color=GM_HEX, lw=2.4, label="GM")
        ax.plot(c[0][:zoom_steps], c[1][:zoom_steps], "-", color=NONGM_HEX, lw=2.4, label="Non-GM")
        if len(g[0]):
            ax.scatter(g[0][0], g[1][0], marker="o", s=55, facecolor="white",
                       edgecolor="#2A2A2A", linewidth=1.5, zorder=6)
        # Non-GM terminus snapped onto the coin ★ after collect.
        if hit["nongm"]["coin_reach"] and len(c[0]):
            ax.scatter(c[0][-1], c[1][-1], marker="o", s=55, facecolor=NONGM_HEX,
                       edgecolor="#2A2A2A", linewidth=1.0, zorder=8)
        ax.scatter(*coin, marker="*", s=220, facecolor="white", edgecolor="#2A2A2A",
                   linewidth=1.0, zorder=7, label="coin (natural)")
        if end_x is not None:
            ax.axvline(float(end_x), color="#9A9A9A", lw=1.6, ls="--", zorder=2,
                       label="end-wall")
        tag_g = "miss" if not hit["gm"]["coin_reach"] else "hit"
        tag_n = "hit" if hit["nongm"]["coin_reach"] else "miss"
        ax.set_title(
            f"level {hit['level']}  ·  GM {tag_g} / Non-GM {tag_n}",
            fontsize=10,
        )
        ax.set_xlabel("agent x")
        ax.set_ylabel("agent y")
        ax.legend(fontsize=7.5, frameon=False, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
        xs_parts = [g[0][:zoom_steps], c[0][:zoom_steps], [coin[0]]]
        ys_parts = [g[1][:zoom_steps], c[1][:zoom_steps], [coin[1]]]
        if end_x is not None:
            xs_parts.append([float(end_x)])
        xs = np.concatenate([np.asarray(p, dtype=np.float64) for p in xs_parts if len(p)])
        ys = np.concatenate([np.asarray(p, dtype=np.float64) for p in ys_parts if len(p)])
        pad = 1.2
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)

    handles = [
        Line2D([0], [0], color=GM_HEX, lw=2.5, label="GM"),
        Line2D([0], [0], color=NONGM_HEX, lw=2.5, label="Non-GM"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor="#2A2A2A", markersize=12, label="natural coin (no edit)"),
        Line2D([0], [0], color="#9A9A9A", lw=1.6, ls="--", label="end-wall"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Coin OOD rollouts - no relocation  ·  natural levels where coin is not at the end-wall",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    plots = Image.open(buf).convert("RGB")

    font, _ = fonts(render_scale)
    strips = []
    for hit in hits:
        gm_s = time_strip(
            hit["gm"]["frames"], scale=render_scale, n_cols=n_cols,
            title=f"GM · L{hit['level']}", accent=GM_RGB, font=font,
        )
        ng_s = time_strip(
            hit["nongm"]["frames"], scale=render_scale, n_cols=n_cols,
            title=f"Non-GM · L{hit['level']}", accent=NONGM_RGB, font=font,
            collected=bool(hit["nongm"]["coin_reach"]),
            collect_px=hit["nongm"].get("collect_px"),
        )
        tw = plots.width
        def fit(im):
            if im.width == tw:
                return im
            h = int(round(im.height * tw / im.width))
            return im.resize((tw, h), Image.NEAREST)
        strips.append(fit(gm_s))
        strips.append(fit(ng_s))

    gap, band = 12, 26
    H = plots.height + band + sum(s.height for s in strips) + gap * len(strips)
    canvas = Image.new("RGB", (plots.width, H), (255, 255, 255))
    y = 0
    canvas.paste(plots, (0, y))
    y += plots.height + 4
    d = ImageDraw.Draw(canvas)
    try:
        sm = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except Exception:
        sm = None
    d.text(
        (8, y),
        "Rendered rollouts  ·  Non-GM last frame = collect (coin blanked)  ·  GM stops at end-wall (no wall-jam hold)  ·  each level: GM then Non-GM",
        fill=MUTE, font=sm,
    )
    y += band
    for s in strips:
        canvas.paste(s, (0, y))
        y += s.height + gap
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-levels", type=int, default=3)
    ap.add_argument("--start-level", type=int, default=1_000_100)
    ap.add_argument("--scan", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--zoom-steps", type=int, default=40)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--levels", type=int, nargs="*", default=None,
                    help="optional pinned levels (skip scan)")
    ap.add_argument("--out-stub", default="fig_22_coin_ood_paths")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or str(default_device())
    print("loading GM…", GM_CKPT)
    gm = load_policy(GM_CKPT, device)
    print("loading Non-GM…", CTRL_CKPT)
    ctrl = load_policy(CTRL_CKPT, device)

    if args.levels:
        hits = []
        for lv in args.levels:
            hit = score_level(gm, ctrl, device, lv, max_steps=args.max_steps)
            if hit is None:
                # Pinned levels are rolled out for inspection even if the filters fail.
                print(f"  level {lv} failed OOD contrast filters; forcing rollouts…", flush=True)
                g = rollout(gm, device, lv, max_steps=args.max_steps)
                n = rollout(ctrl, device, lv, max_steps=args.max_steps)
                if g and n:
                    hit = {"level": lv, "coin": g["coin"], "gm": g, "nongm": n,
                           "gap_from_end": (g["end_x"] - g["coin"][0]) if g["end_x"] else float("nan")}
            if hit:
                hits.append(hit)
                print(f"  use level={hit['level']} gm_reach={hit['gm']['coin_reach']} "
                      f"ng_reach={hit['nongm']['coin_reach']}", flush=True)
    else:
        print(f"scanning OOD levels from {args.start_level}…", flush=True)
        hits = find_levels(
            gm, ctrl, device, n_want=args.n_levels,
            start=args.start_level, scan=args.scan, max_steps=args.max_steps,
        )
    if len(hits) < 2:
        raise SystemExit(f"need ≥2 OOD showcase levels, got {len(hits)}")

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{args.out_stub}.png"
    print("building png…")
    build_png(hits, out_path=png, zoom_steps=args.zoom_steps, render_scale=args.scale)

    meta = {
        "kind": "coin OOD rollouts, no relocation",
        "n_levels": len(hits),
        "levels": [
            {
                "level": h["level"],
                "coin": list(h["coin"]),
                "gap_from_end": h["gap_from_end"],
                "gm_reach": h["gm"]["coin_reach"],
                "nongm_reach": h["nongm"]["coin_reach"],
                "gm_n": h["gm"]["n"],
                "nongm_n": h["nongm"]["n"],
                "gm_x": [h["gm"]["agent0"][0], h["gm"]["max_agent_x"]],
            }
            for h in hits
        ],
        "gm_ckpt": str(GM_CKPT),
        "ctrl_ckpt": str(CTRL_CKPT),
        "png": str(png),
    }
    (OUT / f"{args.out_stub}_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", png)


if __name__ == "__main__":
    main()
