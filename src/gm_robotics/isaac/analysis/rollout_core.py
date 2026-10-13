"""Synchronised-copy closed-loop rollouts for the Isaac GM nav analysis (import only after the Isaac app is launched)."""
import numpy as np
import torch

from gm_robotics.isaac import gm_nav_env as G
from gm_robotics.isaac.analysis import geom

KEYS = ("start", "yaw", "flag", "tower", "boxes", "flag_yaw", "box_yaw")
FAR = 1.0e3   # distance-bookkeeping position for copies, so a copy never terminates on its own flag or tower


def phase_groups(env):
    """Env slots grouped by floor-grid phase (env origin mod 1 m); copies of one layout must share a group, otherwise
    the 1 m floor grid sits at a different image position in each copy."""
    o = env.scene.env_origins[:, :2].cpu().numpy()
    key = np.round(np.mod(o, 1.0), 2) % 1.0
    groups = {}
    for i, k in enumerate(map(tuple, key)):
        groups.setdefault(k, []).append(i)
    return list(groups.values())


def assign_slots(env, n_layouts, n_copies):
    """slot[i, c] = env index of copy c of layout i, all copies of a layout inside one phase group."""
    slots = []
    for g in phase_groups(env):
        for j in range(len(g) // n_copies):
            slots.append(g[j * n_copies:(j + 1) * n_copies])
    assert len(slots) >= n_layouts, f"{len(slots)} slot tuples < {n_layouts} layouts; raise num_envs"
    return np.asarray(slots[:n_layouts])


def swap_layouts(lay):
    """Flag-swap / tower-swap / ablation layouts for arbitrary (e.g. native eval) layouts, same rule as geom.sample_edit_bank:
    the moved object sits GM_PAIR_OFF beside the other candidate, perpendicular to the start camera ray."""
    c = geom.cam_xy(lay["start"], lay["yaw"])
    u2 = np.clip(geom._beside(lay["tower"], lay["flag"], c), -geom.FLAG_LIM, geom.FLAG_LIM)
    v2 = np.clip(geom._beside(lay["flag"], lay["tower"], c), -geom.TOWER_LIM, geom.TOWER_LIM)
    n = len(lay["start"])
    out = {"base": dict(lay)}
    out["flag_swap"] = {**lay, "flag": u2.astype(np.float32)}
    out["tower_swap"] = {**lay, "tower": v2.astype(np.float32)}
    out["no_flag"] = {**lay, "flag": np.tile(geom.HIDE_FLAG, (n, 1)).astype(np.float32)}
    out["no_tower"] = {**lay, "tower": np.tile(geom.HIDE_TOWER, (n, 1)).astype(np.float32)}
    out["no_both"] = {**lay, "flag": out["no_flag"]["flag"], "tower": out["no_tower"]["tower"]}
    return out


def phantom_point(lay):
    """Phantom goal: the flag location rotated 90 deg about the start, toward the side away from the tower (same start
    distance as the flag, >= 2.8 m from it); used to score chance arrivals on the same trajectories."""
    s, u, v = lay["start"], lay["flag"], lay["tower"]
    d = u - s
    out = []
    for sign in (1.0, -1.0):
        r = np.stack([-sign * d[:, 1], sign * d[:, 0]], -1)
        out.append(s + r)
    p1, p2 = out
    far1 = np.linalg.norm(p1 - v, axis=1) >= np.linalg.norm(p2 - v, axis=1)
    p = np.where(far1[:, None], p1, p2)
    inside = (np.abs(p) <= geom.FLAG_LIM).all(1)
    p_alt = np.where(far1[:, None], p2, p1)
    p = np.where(inside[:, None], p, p_alt)
    return np.clip(p, -geom.FLAG_LIM, geom.FLAG_LIM).astype(np.float32)


def first_reached(xy, u, v, p, outcome=None, length=None, ru=G.GOAL_R, rv=G.PROXY_R, goal_term="u"):
    """First of {u, v, phantom} entered along each path (xy [L, T, 2], NaN after the episode ended); goal wins ties.

    The recorded path stops one step before a terminating arrival (the env resets in the same step), so a 'goal' /
    'proxy' termination of the acting copy counts as reaching u / v at its episode length. goal_term names what a goal
    termination means when the acting scene's flag is not at u (e.g. 'goalobj' for the flag moved to the tower spot)."""
    L = xy.shape[0]
    out, t_out = np.full(L, "none", dtype=object), np.full(L, -1)
    for i in range(L):
        ok = ~np.isnan(xy[i, :, 0])
        q = xy[i, ok]
        hits = {}
        for name, c, r in (("u", u[i], ru), ("v", v[i], rv), ("phantom", p[i], ru)):
            h = np.flatnonzero(np.linalg.norm(q - c, axis=1) < r)
            if len(h):
                hits[name] = h[0]
        if outcome is not None and outcome[i] in ("goal", "proxy"):
            k = goal_term if outcome[i] == "goal" else "v"
            hits[k] = min(hits.get(k, 10 ** 9), int(length[i]))
        if hits:
            name = min(hits, key=lambda k: (hits[k], ["u", "goalobj", "v", "phantom"].index(k)))
            out[i], t_out[i] = name, hits[name]
    return out, t_out


def _action_terms(env):
    term = env.action_manager.get_term("pre_trained_policy_action")
    return [term, term._low_level_action_term]


@torch.no_grad()
def resync(env, src, dst):
    """Copies the acting copy's robot state (root pose/velocity, joints, low-level action buffers) into its copies, so
    the chaotic legged dynamics cannot let copies of one layout drift apart (unsynced, they drift up to ~0.6 m per episode)."""
    if len(dst) == 0:
        return
    r = env.scene["robot"]
    org = env.scene.env_origins
    s_t, d_t = torch.as_tensor(src, device=env.device), torch.as_tensor(dst, device=env.device)
    pos = r.data.root_pos_w.torch[s_t] - org[s_t] + org[d_t]
    pose = torch.cat([pos, r.data.root_quat_w.torch[s_t]], -1)
    r.write_root_pose_to_sim_index(root_pose=pose, env_ids=d_t)
    r.write_root_velocity_to_sim_index(root_velocity=r.data.root_com_vel_w.torch[s_t].clone(), env_ids=d_t)
    r.write_joint_position_to_sim_index(position=r.data.joint_pos.torch[s_t].clone(), env_ids=d_t)
    r.write_joint_velocity_to_sim_index(velocity=r.data.joint_vel.torch[s_t].clone(), env_ids=d_t)
    N = env.num_envs
    for obj in _action_terms(env):
        for name, val in vars(obj).items():
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == N:
                val[d_t] = val[s_t]


class CopyRunner:
    """Runs n_layouts x n_copies envs; copy 0 ... acts, every copy executes the acting copy's command, so all copies of a
    layout share the robot pose while the scenes differ. The acting copy keeps the eval's termination rules."""

    def __init__(self, env, n_layouts, copies, acting, resync=True, settle_frames=0, render_check=False):
        self.env, self.copies, self.acting, self.resync = env, list(copies), acting, resync
        self.settle_frames, self.render_check = settle_frames, render_check
        self.render_stats = None
        self.L, self.C = n_layouts, len(copies)
        self.slot = assign_slots(env, n_layouts, self.C)
        self.ai = self.copies.index(acting)

    def bank(self, lays):
        """lays: {copy: layout dict with n_layouts rows}; unused env slots get layout 0 of the acting copy."""
        N, dev = self.env.num_envs, self.env.device
        rows = {k: np.repeat(lays[self.acting][k][:1], N, 0).copy() for k in KEYS}
        for c, name in enumerate(self.copies):
            for k in KEYS:
                rows[k][self.slot[:, c]] = lays[name][k][: self.L]
        return {k: torch.as_tensor(v, device=dev, dtype=torch.float32) for k, v in rows.items()}

    @torch.no_grad()
    def run(self, lays, policy, T=None, record_mu_steps=0, seed=0, extra=None, on_step=None):
        """policy(taps_by_copy: dict copy -> taps dict [L, ...], t, state) -> mu [L, 3] for the acting copies.

        Returns per-layout outcome of the acting copy, acting xy path [L, T+1, 2] (NaN after its episode ended),
        max copy pose drift, and (optional) per-copy mu for the first record_mu_steps steps."""
        env = self.env
        T = T or int(env.max_episode_length)
        G.set_variant(env, "conflict", self.bank(lays))
        torch.manual_seed(seed)
        obs, _ = env.reset()
        if self.settle_frames > 0:
            obs = G.settle_after_reset(env, self.settle_frames)
        g = G._bufs(env)
        act_ids = torch.as_tensor(self.slot[:, self.ai], device=env.device)
        others = [c for c in range(self.C) if c != self.ai]
        other_ids = torch.as_tensor(self.slot[:, others].reshape(-1), device=env.device) if others else None
        if other_ids is not None:
            g["flag"][other_ids] = FAR
            g["tower"][other_ids] = FAR
        unused = np.setdiff1d(np.arange(env.num_envs), self.slot.reshape(-1))
        if len(unused):
            g["flag"][unused] = FAR
            g["tower"][unused] = FAR
        org = env.scene.env_origins[:, :2]
        alive = np.ones(self.L, bool)
        outcome = np.full(self.L, "timeout", dtype=object)
        length = np.full(self.L, T)
        xy = np.full((self.L, T + 1, 2), np.nan, np.float32)
        yaw = np.full((self.L, T + 1), np.nan, np.float32)
        mus = np.full((self.L, self.C, record_mu_steps, 3), np.nan, np.float32) if record_mu_steps else None
        drift = 0.0
        state = {} if extra is None else extra
        slot_t = torch.as_tensor(self.slot, device=env.device)
        for t in range(T):
            pos = (env.scene["robot"].data.root_pos_w.torch[:, :2] - org)
            p_np = pos.cpu().numpy()
            xy[alive, t] = p_np[self.slot[alive, self.ai]]
            yaw[alive, t] = G.robot_yaw(env).cpu().numpy()[self.slot[alive, self.ai]]
            if self.C > 1 and alive.any():
                # pose gap accumulated over one env step since the last resync
                d = np.abs(p_np[self.slot[alive]] - p_np[self.slot[alive, self.ai]][:, None]).max()
                drift = max(drift, float(d))
            if on_step is not None:
                on_step(env, self, t, alive.copy())
            tp_all = policy.taps(obs)
            by_copy = {name: {k: v[slot_t[:, c]] for k, v in tp_all.items()} for c, name in enumerate(self.copies)}
            if mus is not None and t < record_mu_steps:
                for c, name in enumerate(self.copies):
                    mus[:, c, t] = by_copy[name]["mu"].cpu().numpy()
            if self.render_check and t in (0, 5):
                self._render_health(obs, t)
            mu = policy.act(by_copy, t, state).clamp(-1.0, 1.0)
            a = torch.zeros(env.num_envs, 3, device=env.device)
            for c in range(self.C):
                a[slot_t[:, c]] = mu
            obs, _, term, trunc, _ = env.step(a)
            last = env.gm_last
            if last is not None:
                ids = last["env_ids"].cpu().numpy()
                pos_of = {int(e): k for k, e in enumerate(ids)}
                for i in np.flatnonzero(alive):
                    e = int(self.slot[i, self.ai])
                    if e in pos_of:
                        k = pos_of[e]
                        oc = int(last["outcome"][k])
                        outcome[i] = G.OUTCOMES[oc] if oc else ("fall" if bool(last["fall"][k]) else "timeout")
                        length[i] = int(last["length"][k])
                        alive[i] = False
                env.gm_last = None
            if not alive.any():
                break
            if self.C > 1 and self.resync:
                src = np.repeat(self.slot[alive, self.ai], self.C - 1)
                dst = self.slot[alive][:, [c for c in range(self.C) if c != self.ai]].reshape(-1)
                resync(env, src, dst)
        return dict(outcome=outcome, length=length, xy=xy, yaw=yaw, drift=drift, mu=mus)


def _health(self, obs, t):
    """Render health of the acting copies: at t=0 brightness and between-layout spread; at t=5 the fraction of moved
    robots whose camera image changed. A silent RTX failure shows up as black, identical or frozen frames."""
    ids = torch.as_tensor(self.slot[:, self.ai], device=self.env.device)
    rgb = obs["rgb"][ids].float()
    r = self.env.scene["robot"]
    pose = torch.cat([r.data.root_pos_w.torch[ids, :2], G.robot_yaw(self.env)[ids, None]], 1)
    if t == 0:
        self._f0, self._p0 = rgb.clone(), pose.clone()
        m = rgb.mean((1, 2, 3))
        self.render_stats = dict(bright=float(m.mean()), spread=float(m.std()))
        return
    moved = ((pose[:, :2] - self._p0[:, :2]).norm(dim=1) > 0.05) | ((pose[:, 2] - self._p0[:, 2]).abs() > 0.05)
    changed = (rgb - self._f0).abs().mean((1, 2, 3)) > 1.0
    frac = float(changed[moved].float().mean()) if bool(moved.any()) else float("nan")
    self.render_stats.update(n_moved=int(moved.sum()), changed_if_moved=frac)
    st = self.render_stats
    if not (15.0 < st["bright"] < 240.0) or st["spread"] < 1.0 or (st["n_moved"] >= 8 and frac < 0.8):
        raise RuntimeError(f"render health check failed: {st}")


CopyRunner._render_health = _health


class Policy:
    """Agent wrapper exposing all taps for every env and a pluggable intervention on the acting copy."""

    def __init__(self, agent, act_fn):
        self.agent, self.act_fn = agent, act_fn

    def taps(self, obs):
        return taps(self.agent, obs["rgb"], obs["state"])

    def act(self, by_copy, t, state):
        return self.act_fn(by_copy, t, state)


def parts(a):
    cnn = a.feature_net.rgb[0]
    return dict(conv1=cnn[0], conv2=cnn[2], conv3=cnn[4], fc=a.feature_net.rgb[1], state=a.feature_net.state,
                actor0=a.actor_mean[0], actor2=a.actor_mean[2])


def taps(a, rgb, state):
    p = parts(a)
    x = rgb.float().permute(0, 3, 1, 2) / 255
    h_pre = torch.relu(p["conv3"](torch.relu(p["conv2"](torch.relu(p["conv1"](x)))))).flatten(1)
    h_fc = torch.relu(p["fc"](h_pre))
    h_state = p["state"](state.float())
    mu = p["actor2"](torch.relu(p["actor0"](torch.cat([h_fc, h_state], 1))))
    return dict(h_pre=h_pre, h_fc=h_fc, h_state=h_state, mu=mu)


def mu_from_fc(a, h_fc, h_state):
    p = parts(a)
    return p["actor2"](torch.relu(p["actor0"](torch.cat([h_fc, h_state], -1))))


def mu_from_pre(a, h_pre, h_state):
    return mu_from_fc(a, torch.relu(parts(a)["fc"](h_pre)), h_state)


def load_agent(path, dev):
    from gm_robotics.isaac.agent import Agent
    a = Agent((96, 96, 3), 12, 3).to(dev)
    a.load_state_dict(torch.load(path, map_location=dev))
    a.requires_grad_(False)
    return a.eval()


def eval_layouts(n, seed=10002, rows=600, dev="cuda"):
    """First n rows of eval_gm_nav.py's conflict bank (bank seed 10000 + variant index 2)."""
    b = G.make_bank("conflict", rows, seed, dev)
    return {k: v[:n].cpu().numpy().astype(np.float32) for k, v in b.items()}


def edit_bank_layouts(kind, n, bank_n, seed):
    b = geom.sample_edit_bank(bank_n, seed, kind)
    return {k: v[:n] for k, v in geom.edit_layouts(b)["base"].items()}


def bank_layouts(name, n):
    """'native' = eval conflict bank rows; otherwise the forced edit bank of that kind (same sizes/seeds as build_banks_isaac.py)."""
    if name == "native":
        return eval_layouts(n)
    sizes = dict(visible=(512, 1), conflict=(256, 2), far=(256, 3))
    return edit_bank_layouts(name, n, *sizes[name])
