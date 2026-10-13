#!/usr/bin/env python3
"""Coin env-move GIF and time-strip PNG (§2.2): side-by-side GM vs diverse-train rollouts after coin relocation."""
from __future__ import annotations

# Run with PYTHONPATH unset and the x86 procgen interpreter ($PROCGEN_PYTHON_X86,
# see .env.example): the native arm64 build of the procgen fork does not load on Apple Silicon.

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GM_DATA_ROOT", ROOT / "data"))
OUT = ROOT / "paper_figures"
ASSETS = ROOT / "outputs" / "assets"

GM_CKPT = (
    DATA_ROOT / "checkpoints/coin/6033_gm_20M.pth"
)
DIVERSE_CKPT = (
    DATA_ROOT / "checkpoints/coin/7001_teach.pth"
)

GOAL = (0x7A, 0x15, 0x20)   # GM burgundy
SEEK = (0x3D, 0x8B, 0x8B)   # Non-GM teal
INK = (42, 42, 42)
MUTED = (110, 110, 110)
RIGHT = 7
OBS = 64


def bootstrap():
    for p in (
        ROOT / "src",
        ROOT / "external" / "procgenAISC",
        ROOT / "external" / "train-procgen-pytorch",
        ROOT / "scripts" / "procgen",
    ):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)


def make_env(*, start_level: int):
    from common.env.procgen_wrappers import (
        ScaledFloatFrame,
        TransposeFrame,
        VecExtractDictObs,
        VecNormalize,
    )
    from procgen import ProcgenEnv

    venv = ProcgenEnv(
        num_envs=1,
        env_name="coinrun_aisc",
        num_levels=1,
        start_level=start_level,
        distribution_mode="hard",
        num_threads=2,
        center_agent=True,
        random_percent=0,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecNormalize(venv, ob=False, ret=False)
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def close_env(env):
    if env is not None and hasattr(env, "close"):
        env.close()


def chw_u8(chw: np.ndarray) -> np.ndarray:
    x = np.asarray(chw, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


@torch.no_grad()
def act_greedy(policy, obs_chw, device):
    x = torch.as_tensor(obs_chw, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    h = torch.zeros(1, policy.embedder.output_dim, device=device)
    dist, _v, _nh = policy(x, h, torch.zeros(1, device=device))
    return int(dist.probs[0].argmax().item())


def coin_pixel(frame: np.ndarray) -> tuple[float, float] | None:
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    mask = (r > 180) & (g > 150) & (b < 70) & ((r - b) > 120) & ((g - b) > 90)
    ys, xs = np.where(mask)
    if 4 <= len(xs) <= 120:
        return float(np.median(xs)), float(np.median(ys))
    return None


def setup(level: int, warmup: int, relocate_xy: tuple[int, int] | None):
    from gm_geometry.probes.coin_location import (
        get_info,
        labels_from_info,
        obs_nchw_after_set_coin,
        set_coin,
    )

    env = make_env(start_level=level)
    obs = np.asarray(env.reset(), dtype=np.float32)
    for _ in range(warmup):
        obs, _r, done, _info = env.step(np.array([RIGHT]))
        obs = np.asarray(obs, dtype=np.float32)
        if bool(np.asarray(done).reshape(-1)[0]):
            close_env(env)
            return None
    lab = labels_from_info(get_info(env))
    if lab is None:
        close_env(env)
        return None
    if relocate_xy is not None:
        if set_coin(env, int(relocate_xy[0]), int(relocate_xy[1]), 1) != 0:
            close_env(env)
            return None
        obs = obs_nchw_after_set_coin(env)
        lab = labels_from_info(get_info(env)) or lab
    return env, obs, lab


def pick_cell(env, lab):
    from gm_geometry.probes.coin_location import find_floor_cell

    xy = find_floor_cell(env, prefer_left_of_agent=True, min_dx=3)
    if xy is not None:
        return int(xy[0]), int(xy[1])
    ax, ay = int(lab["agent_x"]), max(1, int(lab["agent_y"]))
    return max(1, ax - 6), ay


def blank_coin(frame: np.ndarray) -> np.ndarray:
    """Paint out yellow coin pixels; procgen often leaves the coin sprite in the terminal observation after collection."""
    fr = np.array(frame, dtype=np.uint8, copy=True)
    r = fr[:, :, 0].astype(np.int16)
    g = fr[:, :, 1].astype(np.int16)
    b = fr[:, :, 2].astype(np.int16)
    mask = (r > 180) & (g > 150) & (b < 70) & ((r - b) > 120) & ((g - b) > 90)
    if not mask.any():
        return fr
    # Fill with local neighborhood median (non-coin) so the patch matches the block/sky.
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - 2), min(fr.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(fr.shape[1], xs.max() + 3)
    patch = fr[y0:y1, x0:x1]
    local_mask = mask[y0:y1, x0:x1]
    bg = patch[~local_mask]
    fill = np.median(bg, axis=0).astype(np.uint8) if len(bg) else np.array([30, 30, 40], dtype=np.uint8)
    fr[mask] = fill
    return fr


def rollout_frames(policy, *, device, level, warmup, relocate_xy, max_steps):
    """Greedy episode after setup as (frame_u8, agent_x) pairs, with coin pixels blanked once the coin is collected."""
    from gm_geometry.probes.coin_location import get_info, labels_from_info

    st = setup(level, warmup, relocate_xy)
    if st is None:
        return None
    env, obs, lab = st
    try:
        seq = [(chw_u8(obs if obs.ndim == 3 else obs[0]), float(lab["agent_x"]))]
        collected = False
        for _t in range(max_steps):
            a = act_greedy(policy, obs if obs.ndim == 3 else obs[0], device)
            obs, rew, done, _info = env.step(np.array([a]))
            obs = np.asarray(obs, dtype=np.float32)
            if float(np.asarray(rew).reshape(-1)[0]) > 0:
                collected = True
            lt = labels_from_info(get_info(env))
            ax = float(lt["agent_x"]) if lt is not None else seq[-1][1]
            fr = chw_u8(obs if obs.ndim == 3 else obs[0])
            if collected:
                fr = blank_coin(fr)
            seq.append((fr, ax))
            if bool(np.asarray(done).reshape(-1)[0]):
                break
        return {
            "seq": seq,
            "x0": seq[0][1],
            "coin": (float(lab["coin_x"]), float(lab["coin_y"])),
            "collected": collected,
            "n_real": len(seq),
        }
    finally:
        close_env(env)


def find_good_level(gm, div, *, device, start_search, warmup, max_steps):
    from gm_geometry.probes.coin_location import (
        get_info,
        labels_from_info,
        obs_nchw_after_set_coin,
        set_coin,
    )

    for level in range(start_search, start_search + 400):
        st = setup(level, warmup, None)
        if st is None:
            continue
        env, obs, lab = st
        try:
            cell = pick_cell(env, lab)
            if set_coin(env, cell[0], cell[1], 1) != 0:
                continue
            obs = obs_nchw_after_set_coin(env)
            lab2 = labels_from_info(get_info(env)) or lab
            if float(lab2["coin_x"]) >= float(lab2["agent_x"]) - 1.5:
                continue
            fr = chw_u8(obs if obs.ndim == 3 else obs[0])
            if coin_pixel(fr) is None:
                continue
            sg = rollout_frames(
                gm, device=device, level=level, warmup=warmup, relocate_xy=cell, max_steps=max_steps
            )
            sd = rollout_frames(
                div, device=device, level=level, warmup=warmup, relocate_xy=cell, max_steps=max_steps
            )
            if sg is None or sd is None:
                continue
            gm_end, div_end = sg["seq"][-1][1], sd["seq"][-1][1]
            if div_end < gm_end - 2.0 and gm_end > sg["x0"] + 2.0 and len(sg["seq"]) >= 20:
                return {"level": level, "cell": cell, "gm_end": gm_end, "div_end": div_end}
        finally:
            close_env(env)
    return None


def fonts(scale: int):
    """Bold labels for poster strips; size scales with panel scale."""
    try:
        label = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", max(28, scale + 18)
        )
        step = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", max(24, scale + 14)
        )
    except Exception:
        try:
            label = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", max(28, scale + 18)
            )
            step = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", max(24, scale + 14)
            )
        except Exception:
            label = step = None
    return label, step


def panel(frame: np.ndarray, *, scale: int, title: str, accent, font):
    """Upscaled frame under a header with an accent bar and a title."""
    im = Image.fromarray(frame).resize((OBS * scale, OBS * scale), Image.NEAREST).convert("RGB")
    header = 48
    canvas = Image.new("RGB", (im.width, im.height + header), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 10, header], fill=accent)
    d.text((16, 10), title, fill=INK, font=font)
    canvas.paste(im, (0, header))
    return canvas


def ended_panel(*, scale: int, title: str, accent, font, note="episode ended"):
    """Placeholder shown after an agent's episode has ended while the other is still running."""
    side = OBS * scale
    header = 48
    canvas = Image.new("RGB", (side, side + header), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 10, header], fill=accent)
    d.text((16, 10), title, fill=INK, font=font)
    d.rectangle([0, header, side, side + header], fill=(245, 245, 245))
    d.text((side // 2 - 40, header + side // 2 - 8), note, fill=MUTED, font=font)
    return canvas


def compose_frame(left, right, *, gap=14):
    w = left.width + right.width + gap
    h = max(left.height, right.height)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    return out


def compose_time_strip(seq_panels, *, gap=6):
    """Horizontal time axis: panel_0 | panel_1 | … | panel_T."""
    w = sum(p.width for p in seq_panels) + gap * (len(seq_panels) - 1)
    h = max(p.height for p in seq_panels)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in seq_panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def real_still_idxs(n_real: int, n_still: int) -> list[int]:
    """Evenly spaced indices into the real frames of an episode."""
    if n_real <= 0:
        return [0]
    n_still = min(n_still, n_real)
    if n_still == 1:
        return [0]
    return [int(round(i * (n_real - 1) / (n_still - 1))) for i in range(n_still)]


def gm_still_idxs(seq, n_still: int) -> list[int]:
    """Even subsample, but force the last cell to the rightmost (end-wall) frame."""
    n_real = len(seq)
    idxs = real_still_idxs(n_real, n_still)
    if n_real <= 1:
        return idxs
    xs = [float(ax) for _fr, ax in seq]
    end_i = int(np.argmax(xs))
    # Prefer a late frame if several share the same peak x (standing at the wall).
    peak = xs[end_i]
    for i in range(n_real - 1, -1, -1):
        if xs[i] >= peak - 0.25:
            end_i = i
            break
    idxs[-1] = end_i
    # Keep strictly increasing indices for a clean time axis.
    for k in range(len(idxs) - 2, -1, -1):
        if idxs[k] >= idxs[k + 1]:
            idxs[k] = max(0, idxs[k + 1] - 1)
    return idxs


def main():
    bootstrap()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=200,
                    help="long enough for GM to reach the end wall")
    ap.add_argument("--warmup", type=int, default=18)
    ap.add_argument("--start-search", type=int, default=1_000_052)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--coin-x", type=int, default=None)
    ap.add_argument("--coin-y", type=int, default=None)
    ap.add_argument("--duration-ms", type=int, default=70, help="GIF frame duration")
    args = ap.parse_args()

    from gm_geometry.utils import device as default_device

    args.device = args.device or str(default_device())

    from validate_langosco_coinrun_200m import load_policy

    print("loading GM…", GM_CKPT.name)
    gm = load_policy(GM_CKPT, args.device)
    print("loading diverse-train…", DIVERSE_CKPT.name)
    div = load_policy(DIVERSE_CKPT, args.device)

    if args.level is not None and args.coin_x is not None:
        pick = {"level": args.level, "cell": (args.coin_x, args.coin_y or 1)}
    else:
        print("searching for a clear level…")
        pick = find_good_level(
            gm,
            div,
            device=args.device,
            start_search=args.start_search,
            warmup=args.warmup,
            max_steps=args.max_steps,
        )
        if pick is None:
            raise SystemExit("no good level found")
    print("picked", pick)
    level, cell = int(pick["level"]), tuple(pick["cell"])

    kw = dict(device=args.device, level=level, warmup=args.warmup, max_steps=args.max_steps, relocate_xy=cell)
    print("roll GM relocated…")
    sg = rollout_frames(gm, **kw)
    print("roll diverse relocated…")
    sd = rollout_frames(div, **kw)
    if sg is None or sd is None:
        raise SystemExit("rollout failed")

    gm_seq, div_seq = sg["seq"], sd["seq"]
    n_gm, n_ng = len(gm_seq), len(div_seq)
    n = min(max(n_gm, n_ng), 90)
    print(
        f"  real frames GM={n_gm} (collected={sg['collected']})  "
        f"Non-GM={n_ng} (collected={sd['collected']})  gif_span={n}"
    )
    print(f"  GM {sg['x0']:.1f}→{gm_seq[-1][1]:.1f}  Non-GM {sd['x0']:.1f}→{div_seq[-1][1]:.1f}")

    font_label, font_step = fonts(args.scale)
    # Once an episode ends, show an "episode ended" panel rather than holding the terminal frame.
    frames = []
    for t in range(n):
        if t < n_gm:
            left = panel(gm_seq[t][0], scale=args.scale, title="GM", accent=GOAL, font=font_label)
        else:
            left = ended_panel(scale=args.scale, title="GM", accent=GOAL, font=font_label)
        if t < n_ng:
            right = panel(div_seq[t][0], scale=args.scale, title="Non-GM", accent=SEEK, font=font_label)
        else:
            right = ended_panel(scale=args.scale, title="Non-GM", accent=SEEK, font=font_label)
        frames.append(compose_frame(left, right))

    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    gif_path = OUT / "fig_22_coin_envmove_paths.gif"
    rgb = [f.convert("RGB") for f in frames]
    rgb[0].save(
        gif_path,
        save_all=True,
        append_images=rgb[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    (ASSETS / gif_path.name).write_bytes(gif_path.read_bytes())

    # Poster PNG: each row samples that agent's real frames only.
    n_still = 8
    scale_s = max(3, args.scale - 2)
    font_label_s, font_step_s = fonts(scale_s)
    gm_idxs = gm_still_idxs(gm_seq, n_still)
    ng_idxs = real_still_idxs(n_ng, n_still)
    gm_stills = [
        panel(gm_seq[t][0], scale=scale_s,
              title=("GM" if i == 0 else f"t={t}"), accent=GOAL,
              font=(font_label_s if i == 0 else font_step_s))
        for i, t in enumerate(gm_idxs)
    ]
    ng_stills = [
        panel(div_seq[t][0], scale=scale_s,
              title=("Non-GM" if i == 0 else f"t={t}"), accent=SEEK,
              font=(font_label_s if i == 0 else font_step_s))
        for i, t in enumerate(ng_idxs)
    ]
    # Mark the Non-GM last cell if it collected / ended early.
    if sd["collected"] and ng_idxs:
        last = ng_stills[-1]
        d = ImageDraw.Draw(last)
        d.text((12, last.height - 22), "collected / ended", fill=MUTED, font=font_step_s)

    row_gm = compose_time_strip(gm_stills)
    row_ng = compose_time_strip(ng_stills)
    gap = 10
    W = max(row_gm.width, row_ng.width)
    poster = Image.new("RGB", (W, row_gm.height + row_ng.height + gap), (255, 255, 255))
    poster.paste(row_gm, (0, 0))
    poster.paste(row_ng, (0, row_gm.height + gap))
    png_path = OUT / "fig_22_coin_envmove_paths.png"
    poster.save(png_path)
    poster.save(ASSETS / png_path.name)
    meta = {
        "version": 7,
        "kind": "time-strip PNG + side-by-side GIF (real frames; coin blanked after collect)",
        "level": level,
        "relocate_cell": list(cell),
        "n_gif_span": n,
        "gm_n_real": n_gm,
        "nongm_n_real": n_ng,
        "gm_collected": sg["collected"],
        "nongm_collected": sd["collected"],
        "gm_still_idxs": gm_idxs,
        "nongm_still_idxs": ng_idxs,
        "duration_ms": args.duration_ms,
        "gm": [sg["x0"], gm_seq[-1][1]],
        "nongm": [sd["x0"], div_seq[-1][1]],
        "gif": str(gif_path),
        "png_poster": str(png_path),
    }
    (OUT / "fig_22_coin_envmove_paths_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", gif_path, f"({gif_path.stat().st_size/1024:.0f} KB)")
    print("wrote", png_path, f"(time-strip, GM cols={len(gm_idxs)} Non-GM cols={len(ng_idxs)})")


if __name__ == "__main__":
    main()
