"""Paired goal-present / goal-absent Procgen frames (Maze cheese, Coin coin, Attribute Maze goal line) to calibrate pixel goal masks."""
import argparse
from pathlib import Path

import numpy as np

from procgen_paths import PROCGEN_ROOT

HERE = Path(__file__).resolve().parent


def u8(o):
    o = np.asarray(o, dtype=np.float32)
    o = o[0] if o.ndim == 4 else o
    return np.rint(np.clip(o if o.max() > 1.5 else o * 255.0, 0, 255)).astype(np.uint8)


def maze_pairs(root, n, rng):
    import rewire_extra_procgen as R
    M = R.imports("maze", root)
    m = M["m"]
    out = []
    for lv in range(2_000_000, 2_000_000 + n):
        env = m.make_env("maze_aisc", start_level=lv, extra={"rand_region": 25})
        obs = env.reset()
        for _ in range(int(rng.integers(0, 20))):
            obs, _r, d, _i = env.step(np.array([rng.integers(0, 15)]))
            if bool(np.asarray(d)[0]):
                break
        info = m.get_info_maze(env)
        cx, cy = m._iscalar(info, "cheese_x"), m._iscalar(info, "cheese_y")
        a = u8(m.obs_after_edit(env))
        m.set_cheese(env, cx, cy, 0)
        b = u8(m.obs_after_edit(env))
        env.close()
        out.append((a, b))
    return out


def lockstep(make_a, make_b, n, rng, n_actions=15, max_k=40):
    """Two envs on the same level stepped with the same random actions; frames kept while neither has ended."""
    out = []
    for i in range(n):
        ea, eb = make_a(i), make_b(i)
        oa, ob = ea.reset(), eb.reset()
        for _ in range(int(rng.integers(0, max_k))):
            act = np.array([rng.integers(0, n_actions)])
            oa, _ra, da, _ = ea.step(act)
            ob, _rb, db, _ = eb.step(act)
            if bool(np.asarray(da)[0]) or bool(np.asarray(db)[0]):
                break
        else:
            out.append((u8(oa), u8(ob)))
        ea.close()
        eb.close()
    return out


def coin_pairs(root, n, rng):
    import rewire_extra_procgen as R
    M = R.imports("coin", root)
    SA = M["SA"]
    mk = lambda hide: (lambda i: SA.make_env("coinrun_aisc", num_levels=1, start_level=500_000 + i, hide_visible_coin=hide))
    return lockstep(mk(False), mk(True), n, rng)


def am_pairs(root, n, rng):
    import rewire_extra_procgen as R
    R.imports("am", root)
    from gm_geometry.integrations.colour_shape import make_duel_env
    mk = lambda o1: (lambda i: make_duel_env(1, 1, 1, o1, "yellow_gem", start_level=i, use_backgrounds=False))
    return lockstep(mk("red_line_diag"), mk("invisible_gem"), n, rng, max_k=6)


def main(a):
    rng = np.random.default_rng(0)
    pairs = {"maze": maze_pairs, "coin": coin_pairs, "am": am_pairs}[a.env](a.root, a.n, rng)
    np.savez_compressed(a.out, present=np.stack([p for p, _ in pairs]), absent=np.stack([q for _, q in pairs]))
    print(a.env, len(pairs), "pairs ->", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=("maze", "coin", "am"), required=True)
    p.add_argument("--root", default=str(PROCGEN_ROOT))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", default="pairs.npz")
    main(p.parse_args())
