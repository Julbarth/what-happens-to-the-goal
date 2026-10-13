"""Render raw 64x64 reset observations of conflict-test levels for Maze, Coin and Attribute Maze (run on Linux with procgen)."""
import argparse

import numpy as np

# Conflict-test constructions, copied from the paper's evaluation code:
#   maze: scripts/validate_langosco_maze_i_200m.make_env, env "maze" (cheese on any free cell), hard, held-out levels from 1_000_000
#   coin: env "coinrun_aisc" (coin in a random section, randomize_goal always true), hard
#   attr: scripts/run_ras_logit_competition_assay.make_env, red_line_diag (+) vs yellow_gem (-), easy, world_dim 5, black
GAMES = {
    "maze": dict(env_name="maze", distribution_mode="hard"),
    "coin": dict(env_name="coinrun_aisc", distribution_mode="hard"),
    "attr": dict(env_name="maze_colorsobjects_duel", distribution_mode="easy", rand_seed=0, world_dim=5,
                 obj1="red_line_diag", obj2="yellow_gem", use_backgrounds=False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", choices=list(GAMES), required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=None, help="explicit start levels")
    ap.add_argument("--start-level", type=int, default=1_000_000)
    ap.add_argument("--n-levels", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from procgen import ProcgenEnv
    levels = a.levels or list(range(a.start_level, a.start_level + a.n_levels))
    frames, infos = [], []
    for lv in levels:
        env = ProcgenEnv(num_envs=1, num_levels=1, start_level=int(lv), **GAMES[a.game])
        try:
            ob = env.reset()
            frames.append(np.asarray(ob["rgb"][0], dtype=np.uint8))
            # one no-op step exposes the info dict (coin_x etc.); the frame kept is the reset observation
            _, _, _, info = env.step(np.array([4]))
            infos.append({k: v for k, v in info[0].items() if np.isscalar(v)})
        finally:
            env.close()
    keys = sorted({k for d in infos for k in d})
    extra = {f"info_{k}": np.array([d.get(k, -1) for d in infos]) for k in keys}
    np.savez_compressed(a.out, frames=np.stack(frames), level=np.array(levels), game=a.game, **extra)
    print("wrote", a.out, np.stack(frames).shape, keys)


if __name__ == "__main__":
    main()
