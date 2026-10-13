"""Scripted privileged oracle (walk to flag or tower) + contact sheets and pixel-salience ladder for the Isaac GM nav env."""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--variants", nargs="+", default=["gm", "control", "conflict"])
p.add_argument("--targets", nargs="+", default=["flag", "tower"])
p.add_argument("--num_envs", type=int, default=50)
p.add_argument("--episodes_per_env", type=int, default=2)
p.add_argument("--bank_seed", type=int, default=20000)
p.add_argument("--res", type=int, default=96)
p.add_argument("--out", default="results/isaac")
p.add_argument("--fig_dir", default="figures/isaac")
p.add_argument("--cam_pos", type=float, nargs=3, default=None)
p.add_argument("--ladder_only", action="store_true")
p.add_argument("--ladder_tag", default="")
p.add_argument("--sheet", action="store_true", help="also save contact sheets, top-down renders and the distance ladder")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import csv  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gm_robotics.isaac import gm_nav_env as G  # noqa: E402


def pix_counts(img):
    """Pixels of the flag banner (green) and tower (orange) by colour threshold on uint8 RGB."""
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    flag = (g > 90) & (g > r + 40) & (g > b + 40)
    tower = (r > 150) & (g > 100) & (r - b > 80)   # the orange material renders yellow-orange after tone mapping
    return int(flag.sum()), int(tower.sum())


def contact_sheet(env, variant, n_show=6, steps=(1, 15, 30, 45, 60)):
    G.set_variant(env, variant, G.make_bank(variant, env.num_envs * 2, args.bank_seed + 7, env.device))
    env.reset()
    frames = {s: None for s in steps}
    top = None
    dists = {}
    g = G._bufs(env)
    for s in range(1, max(steps) + 1):
        env.step(G.scripted_action(env, "flag"))
        if s in frames:
            frames[s] = env.scene["front_cam"].data.output["rgb"][:n_show, ..., :3].cpu().numpy()
            xy = env.scene["robot"].data.root_pos_w.torch[:n_show, :2]
            dists[s] = ((xy - g["flag"][:n_show]).norm(dim=-1).cpu().numpy(), (xy - g["tower"][:n_show]).norm(dim=-1).cpu().numpy())
            if s == 1:
                top = env.scene["top_cam"].data.output["rgb"][:n_show, ..., :3].cpu().numpy()
    fig, ax = plt.subplots(n_show, len(steps) + 1, figsize=(2.2 * (len(steps) + 1), 2.3 * n_show))
    for i in range(n_show):
        ax[i, 0].imshow(top[i]); ax[i, 0].set_title("top-down t=0", fontsize=7)
        for j, s in enumerate(steps):
            ax[i, j + 1].imshow(frames[s][i], interpolation="nearest")
            fp, tp = pix_counts(frames[s][i])
            ax[i, j + 1].set_title(f"t={s / 10:.1f}s dF={dists[s][0][i]:.1f} dT={dists[s][1][i]:.1f}\nflag px {fp} tower px {tp}", fontsize=6)
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"{variant}: front camera {args.res}x{args.res} (policy input) while the oracle walks to the flag", fontsize=9)
    fig.tight_layout()
    path = f"{args.fig_dir}/contact_sheet_{variant}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    print("saved", path)


def distance_ladder(env, dists=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0)):
    """Robot faces a gm-style pair (tower 0.65 m to the side) from fixed distances; reports pixel counts."""
    n = env.num_envs
    dev = env.device
    k = len(dists)
    d = torch.tensor(dists, device=dev).repeat(n // k + 1)[:n]
    bank = G.make_bank("gm", n, 1, dev)
    bank["flag"] = torch.stack([torch.full((n,), 4.0, device=dev), torch.zeros(n, device=dev)], -1)
    bank["tower"] = bank["flag"] + torch.tensor([0.0, 0.65], device=dev)
    bank["start"] = torch.stack([4.0 - d, torch.zeros(n, device=dev)], -1)
    bank["yaw"] = torch.zeros(n, device=dev)
    bank["boxes"] = torch.tensor([[-4.0, 4.0], [-4.0, -4.0], [0.0, -4.0]], device=dev).expand(n, 3, 2).clone()
    G.set_variant(env, "gm", bank)
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(n, 3, device=dev))
    imgs = env.scene["front_cam"].data.output["rgb"][:k, ..., :3].cpu().numpy()
    fig, ax = plt.subplots(1, k, figsize=(2.4 * k, 2.8))
    rows = []
    for j in range(k):
        fp, tp = pix_counts(imgs[j])
        rows.append(dict(distance=dists[j], flag_px=fp, tower_px=tp))
        ax[j].imshow(imgs[j], interpolation="nearest"); ax[j].axis("off")
        ax[j].set_title(f"{dists[j]:.0f} m: flag {fp}px tower {tp}px", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.fig_dir}/distance_ladder{args.ladder_tag}.png", dpi=120); plt.close(fig)
    with open(f"{args.out}/distance_ladder.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("ladder", rows)


def main():
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.fig_dir, exist_ok=True)
    env = G.make_env("gm", args.num_envs, res=args.res, video=args.sheet, cam_pos=args.cam_pos)
    if args.sheet:
        distance_ladder(env)
        if args.ladder_only:
            env.close(); return
        for v in args.variants:
            contact_sheet(env, v)
    summary = []
    for v in args.variants:
        for tgt in args.targets:
            bank = G.make_bank(v, args.num_envs * (args.episodes_per_env + 1), args.bank_seed, env.device)
            G.set_variant(env, v, bank)
            rows = G.run_episodes(env, lambda o, t=tgt: G.scripted_action(env, t), args.episodes_per_env)
            path = f"{args.out}/oracle_{v}_{tgt}.csv"
            with open(path, "w") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            oc = np.array([r["outcome"] for r in rows])
            line = (f"oracle variant={v} target={tgt} n={len(rows)} goal={np.mean(oc == 'goal'):.3f} proxy={np.mean(oc == 'proxy'):.3f} "
                    f"timeout={np.mean(oc == 'timeout'):.3f} fall={np.mean(oc == 'fall'):.3f} mean_len={np.mean([r['length'] for r in rows]):.1f}")
            print(line); summary.append(line)
    open(f"{args.out}/oracle_summary.txt", "w").write("\n".join(summary) + "\n")
    env.close()


main()
app.close()
