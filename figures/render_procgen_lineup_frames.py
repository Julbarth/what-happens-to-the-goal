"""Render 64x64 GM-training / Non-GM-training / conflict-test candidate frames for the Procgen lineup (run on Linux, where procgen builds)."""
import argparse

import numpy as np

# Distributions follow the Environments and training appendices:
#   Maze: maze_aisc hard rand_region 0 (GM) / 25 (Non-GM) on training levels, OOD 'maze' hard on held-out levels.
#   Coin: coinrun hard random_percent 0 (GM) / coinrun_aisc hard (Non-GM) on training levels, coinrun_aisc on held-out levels.
#   Attribute Maze: maze_colorsobjects_duel easy world_dim 5, obj1 (+10) vs obj2 (-10), black or textured background.
TRAIN0, TEST0 = 0, 1_000_000
# The eighth Non-GM pair (yellow line vs blue gem) is the GM pair itself, so it is not rendered again.
NONGM_PAIRS = [("yellow_line_diag", "yellow_gem"), ("green_line_diag", "blue_gem"), ("red_line_diag", "red_gem"),
               ("yellow_gem", "blue_gem"), ("yellow_gem", "green_gem"), ("red_gem", "green_gem"),
               ("purple_gem", "red_gem")]


def specs(fork, n):
    if fork == "aisc":
        return {
            "maze_gm": (dict(env_name="maze_aisc", distribution_mode="hard", rand_region=0), TRAIN0, "reset"),
            "maze_nongm": (dict(env_name="maze_aisc", distribution_mode="hard", rand_region=25), TRAIN0, "reset"),
            "maze_conflict": (dict(env_name="maze", distribution_mode="hard"), TEST0, "reset"),
            "coin_gm": (dict(env_name="coinrun", distribution_mode="hard", random_percent=0), TRAIN0, "coin"),
            "coin_nongm": (dict(env_name="coinrun_aisc", distribution_mode="hard"), TRAIN0, "coin"),
            "coin_conflict": (dict(env_name="coinrun_aisc", distribution_mode="hard"), TEST0, "coin"),
        }
    out = {}
    for bg, tag in ((False, "black"), (True, "texture")):
        base = dict(env_name="maze_colorsobjects_duel", distribution_mode="easy", world_dim=5, use_backgrounds=bg)
        out[f"attr_{tag}_gm"] = (dict(base, obj1="yellow_line_diag", obj2="blue_gem"), TRAIN0, "reset")
        for o1, o2 in NONGM_PAIRS:
            out[f"attr_{tag}_nongm_{o1}__{o2}"] = (dict(base, obj1=o1, obj2=o2), TRAIN0, "reset")
        out[f"attr_{tag}_conflict"] = (dict(base, obj1="red_line_diag", obj2="yellow_gem"), TRAIN0, "reset")
    return out


def reset_frame(env):
    return np.asarray(env.observe()[1]["rgb"][0], dtype=np.uint8)


def coin_frame(env, rng, max_steps=400):
    """Scripted right/jump-right rollout; returns the first frame with the coin 1.5-5 cells ahead of the agent."""
    for _ in range(max_steps):
        info = env.get_info()[0]
        dx = info["coin_x"] + 0.5 - info["agent_x_fp"] / 100.0
        if info["coin_present"] and 1.5 <= dx <= 5.0 and abs(info["coin_y"] - info["agent_y_fp"] / 100.0) <= 4:
            return reset_frame(env), dx
        env.act(np.array([8 if rng.random() < 0.6 else 7], dtype=np.int32))
        if env.observe()[2][0]:
            return None, None
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fork", choices=["aisc", "cs"], required=True)
    ap.add_argument("--n-levels", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from procgen import ProcgenGym3Env
    res = {}
    for name, (kw, lv0, mode) in specs(a.fork, a.n_levels).items():
        frames, levels = [], []
        lv = lv0
        while len(frames) < a.n_levels and lv < lv0 + 20 * a.n_levels:
            env = ProcgenGym3Env(num=1, num_levels=1, start_level=lv, num_threads=1, **kw)
            if mode == "reset":
                f = reset_frame(env)
            else:
                f, _ = coin_frame(env, np.random.default_rng(lv))
            if f is not None:
                frames.append(f); levels.append(lv)
            env.close() if hasattr(env, "close") else None
            lv += 1
        res[f"{name}__frames"] = np.stack(frames)
        res[f"{name}__levels"] = np.array(levels)
        print(name, len(frames), flush=True)
    np.savez_compressed(a.out, **res)


if __name__ == "__main__":
    main()
