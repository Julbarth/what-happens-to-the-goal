"""AT / SZ cue-competition banks: bank statistics, strict t=0 and decision-point banks, pixel-matched edits, geometry per bank."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import axis_geometry as AG

AX = C.gmg_axis
OUT = os.path.join(C.RESULTS, "bank_audit")
GOAL_E = {"AT": "shape_swap", "SZ": "mark_swap"}
PROX_E = {"AT": "colour_swap", "SZ": "colour_swap"}
EDITS = {"AT": ("base", "shape_swap", "colour_swap", "shape_ablate", "colour_ablate"),
         "SZ": ("base", "mark_swap", "colour_swap", "mark_ablate", "colour_ablate")}
TCP0 = np.array([-0.24, 0.0], np.float32)
SEEDS = {"AT": dict(gm=[2, 3, 4], control=[2, 3, 4, 5]), "SZ": dict(gm=[0, 1, 2], control=[0, 1, 2])}
BANK_SEED = {"AT": 4000, "SZ": 4200}


def bank_path(env, variant):
    if variant == "orig":
        return os.path.join(C.BANKS, f"{env}_worldedit.npz")
    return os.path.join(C.BANKS, f"{env}_audit_{variant}.npz")


# ---------------------------------------------------------------- layout statistics

def ref_xy(env, b):
    """Point the conflict is seen from: the TCP in AT (the arm goes to the object), the cube in SZ (it gets pushed)."""
    return b["cube_now"][:, :2] if "cube_now" in b else (b["cube"][:, :2] if env == "SZ" else b["tcp"][:, :2])


def layout(env, b):
    u, v, r = b["u"], b["v"], ref_xy(env, b)
    du, dv = np.linalg.norm(u - r, axis=1), np.linalg.norm(v - r, axis=1)
    bu, bv = np.arctan2(*(u - r).T[::-1]), np.arctan2(*(v - r).T[::-1])
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (bu - bv)))))
    return dict(du=du, dv=dv, ddist=np.abs(du - dv), angle=ang, dy=np.abs(u[:, 1] - v[:, 1]),
                same_side=(u[:, 1] * v[:, 1] > 0))


def stats(a):
    rows = []
    for variant in a.variants:
        f = bank_path(a.env, variant)
        if not os.path.exists(f):
            continue
        b = C.load_bank(os.path.basename(f)[:-4])
        L = layout(a.env, b)
        r = dict(env=a.env, bank=variant, n=len(b["sign"]), frac_sign_pos=float((b["sign"] > 0).mean()))
        for k in ("du", "dv", "ddist", "angle", "dy"):
            r[f"{k}_p10"], r[f"{k}_med"], r[f"{k}_p90"] = np.percentile(L[k], [10, 50, 90])
        r["frac_angle_lt45"] = float((L["angle"] < 45).mean())
        r["frac_same_side"] = float(L["same_side"].mean())
        r["frac_u_closer"] = float((L["du"] < L["dv"]).mean())
        for e in EDITS[a.env][1:]:
            if f"npx_{e}" in b:
                r[f"npx_{e}"], r[f"pxl2_{e}"] = float(np.median(b[f"npx_{e}"])), float(np.median(b[f"pxl2_{e}"]))
        g, p = GOAL_E[a.env], PROX_E[a.env]
        r["pxl2_ratio_proxy_goal"] = float(np.median(b[f"pxl2_{p}"] / b[f"pxl2_{g}"]))
        r["npx_ratio_proxy_goal"] = float(np.median(b[f"npx_{p}"] / np.maximum(b[f"npx_{g}"], 1)))
        x = b["rgb_base"].reshape(len(b["sign"]), -1)[:, ::5].astype(np.float32)
        gram = x @ x.T
        nn = np.diag(gram)
        d2 = nn[:, None] + nn[None] - 2 * gram
        np.fill_diagonal(d2, np.inf)
        rms = np.sqrt(np.maximum(d2.min(1), 0) / x.shape[1])
        r["nn_rms_min"], r["nn_rms_med"] = float(rms.min()), float(np.median(rms))
        r["n_near_dup"] = int((rms < 1.0).sum())
        if "t_frame" in b:
            r["t_frame_med"] = float(np.median(b["t_frame"]))
            r["frac_src_gm"] = float((b["src_cohort"] == 0).mean())
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"stats_{a.env}.csv")
    pd.DataFrame(rows).to_csv(fn, index=False)


# ---------------------------------------------------------------- strict t=0 banks (mirror pairs)

def sample_strict(env, n_pairs, seed, min_angle, max_ddist):
    """Equidistant, wide-bearing conflicts seen from the start TCP (AT) or the cube (SZ), each with its y-mirror
    so the two conflict orientations are exactly balanced and share qpos noise; pairs are adjacent (2i, 2i+1)."""
    from gm_robotics.analysis import envs_forced as E
    rng = np.random.default_rng(seed)
    us, vs, cubes = [], [], []
    while len(us) < n_pairs:
        if env == "AT":
            u = rng.uniform(E.AT_LO, E.AT_HI)
            v = rng.uniform(E.AT_LO, E.AT_HI)
            if u[1] < v[1]:
                u, v = v, u
            r = TCP0
            ok = np.linalg.norm(u - v) >= E.AT_MIN_SEP and u[1] > 0 > v[1]
            cube = None
        else:
            S = E.SZ
            u = np.array([rng.uniform(*S.pad_x), rng.uniform(*S.pad_absy)])
            v = np.array([rng.uniform(*S.pad_x), -rng.uniform(*S.pad_absy)])
            cube = np.array([rng.uniform(*S.cube_x), rng.uniform(*S.cube_y), rng.uniform(0, np.pi / 2)])
            r = cube[:2]
            ok = True
        du, dv = np.linalg.norm(u - r), np.linalg.norm(v - r)
        bu, bv = np.arctan2(*(u - r)[::-1]), np.arctan2(*(v - r)[::-1])
        ang = np.degrees(abs(np.angle(np.exp(1j * (bu - bv)))))
        if ok and abs(du - dv) <= max_ddist and ang >= min_angle:
            us.append(u), vs.append(v), cubes.append(cube)
    from gm_robotics.envs.common import INIT_QPOS
    m = np.array([1.0, -1.0], np.float32)
    U, V = [], []
    Q, CB = [], []
    for i in range(n_pairs):
        q = np.asarray(INIT_QPOS, np.float32).copy()
        q[:7] += rng.normal(0, 0.02, 7)
        U += [us[i], us[i] * m]
        V += [vs[i], vs[i] * m]
        Q += [q, q]
        if env == "SZ":
            c = cubes[i]
            CB += [c, np.array([c[0], -c[1], (np.pi / 2 - c[2]) % (np.pi / 2)])]
    out = dict(u=np.array(U, np.float32), v=np.array(V, np.float32), qpos=np.array(Q, np.float32))
    if env == "SZ":
        out["cube"] = np.array(CB, np.float32)
    return out


def pixel_change(a, b):
    diff = np.any(a != b, axis=-1)
    return diff.reshape(len(a), -1).sum(1), np.sqrt(((a.astype(np.float32) - b) ** 2).reshape(len(a), -1).sum(1))


def build_strict(a):
    from gm_robotics.analysis import envs_forced as E
    from gm_robotics.analysis.build_banks import render_batches
    env = E.make_env(a.env, a.batch)
    cfg = sample_strict(a.env, a.n // 2, BANK_SEED[a.env], a.min_angle, a.max_ddist)
    n = len(cfg["u"])
    bank = dict(u=cfg["u"], v=cfg["v"], qpos_forced=cfg["qpos"], pair=np.arange(n) // 2)
    if a.env == "SZ":
        bank["cube"] = cfg["cube"]
    for edit in EDITS[a.env]:
        if a.env == "AT":
            pl = E.at_placement(cfg, edit)
            fn = lambda i: E.force_at(env, cfg["qpos"][i], {k: v[i] for k, v in pl.items()})
        else:
            pl = E.sz_placement(cfg, edit)
            fn = lambda i: E.force_sz(env, cfg["qpos"][i], {k: np.asarray(v)[i] for k, v in pl.items()})
        r = render_batches(env, n, fn, a.batch)
        bank[f"rgb_{edit}"], bank[f"state_{edit}"] = r["rgb"], r["state"]
        bank["tcp_env"], bank["qpos"] = r["tcp"], r["qpos"]
    for edit in EDITS[a.env][1:]:
        bank[f"npx_{edit}"], bank[f"pxl2_{edit}"] = pixel_change(bank["rgb_base"], bank[f"rgb_{edit}"])
    bank["tcp"], bank["J_tcp"] = E.tcp_pos_and_jac(bank["qpos"])
    bank["sign"] = np.where(bank["u"][:, 1] > bank["v"][:, 1], 1, -1).astype(np.int8)
    save_bank(bank, bank_path(a.env, "strict"))
    env.close()


def save_bank(bank, path):
    np.savez_compressed(path + ".tmp.npz", **bank)
    os.replace(path + ".tmp.npz", path)
    print(f"wrote {path} n={len(bank['sign'])}", flush=True)


# ---------------------------------------------------------------- decision-point banks (mid-episode frames)

def decision_rule(env, tcp, u, v, cube, cube0, live):
    """AT: the arm is between the objects (angle u-TCP-v in xy >= 90 deg) and >= 3 cm from both surfaces.
    SZ: the TCP is at the cube (xy <= 7 cm, z <= 6 cm) and the cube has not been moved yet (< 1 cm)."""
    if env == "AT":
        a, b = u - tcp[:, :2], v - tcp[:, :2]
        cosang = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        far = (np.linalg.norm(np.concatenate([a, np.zeros_like(a[:, :1]) + tcp[:, 2:] - 0.03], 1), axis=1) > 0.06) & \
              (np.linalg.norm(np.concatenate([b, np.zeros_like(b[:, :1]) + tcp[:, 2:] - 0.03], 1), axis=1) > 0.06)
        return live & (cosang <= 0.0) & far
    near = np.linalg.norm(tcp[:, :2] - cube[:, :2], axis=1) <= 0.07
    low = tcp[:, 2] <= 0.06
    still = np.linalg.norm(cube[:, :2] - cube0[:, :2], axis=1) < 0.01
    return live & near & low & still


@torch.no_grad()
def collect_decision_batch(kind, env, agent, cfg, pad, dev, T):
    """Native deterministic rollout of the base copy with goal-swap and proxy-swap copies re-synchronised to it
    every step (steer_ext.sync_copies); returns the three copies' observations at the first decision frame."""
    from gm_robotics.analysis import envs_forced as E
    from gm_robotics.analysis.steer_ext import sync_copies
    copies = ("base", GOAL_E[kind], PROX_E[kind])
    c = {k: v[pad] for k, v in cfg.items()}
    if kind == "AT":
        pls = [E.at_placement(c, e) for e in copies]
        E.force_at(env, np.tile(c["qpos"], (3, 1)), {k: np.concatenate([p[k] for p in pls]) for k in pls[0]})
    else:
        pls = [E.sz_placement(c, e) for e in copies]
        E.force_sz(env, np.tile(c["qpos"], (3, 1)), {k: np.concatenate([np.asarray(p[k]) for p in pls]) for k in pls[0]})
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    B = u.num_envs // 3
    actors = ("cube",) if kind == "SZ" else ()
    latch = u.first_touch if kind == "AT" else u.first_entry
    cube0 = u.cube.pose.p[:B].cpu().numpy() if kind == "SZ" else None
    got = np.zeros(B, bool)
    rec = dict(t_frame=np.full(B, -1), rgb=[None] * 3, state=[None] * 3)
    for i in range(3):
        rec["rgb"][i] = np.zeros((B, *obs["rgb"].shape[1:]), np.uint8)
        rec["state"][i] = np.zeros((B, obs["state"].shape[1]), np.float32)
    rec.update(tcp=np.zeros((B, 3), np.float32), qpos=np.zeros((B, 9), np.float32), cube_now=np.zeros((B, 3), np.float32))
    for t in range(T):
        tcp = u.agent.tcp.pose.p[:B].cpu().numpy()
        cube = u.cube.pose.p[:B].cpu().numpy() if kind == "SZ" else None
        live = latch[:B].cpu().numpy() == 0
        fire = decision_rule(kind, tcp, c["u"], c["v"], cube, cube0, live) & ~got & (t >= 1)
        if fire.any():
            idx = np.where(fire)[0]
            rgb, st = obs["rgb"].cpu().numpy(), obs["state"].cpu().numpy()
            for i in range(3):
                rec["rgb"][i][idx] = rgb[i * B + idx]
                rec["state"][i][idx] = st[i * B + idx]
            rec["t_frame"][idx] = t
            rec["tcp"][idx] = tcp[idx]
            rec["qpos"][idx] = u.agent.robot.get_qpos()[:B].cpu().numpy()[idx]
            if kind == "SZ":
                rec["cube_now"][idx] = cube[idx]
            got |= fire
        if got.all():
            break
        mu = C.taps(agent, obs["rgb"][:B], obs["state"][:B])["mu"]
        obs, _, _, _, info = env.step(mu.repeat(3, 1))
        obs, _ = sync_copies(env, 3, info, actors=actors)
    # Outcome of the source rollout is only known if it ran to the end; record the latch at stop time.
    rec["src_latch"] = latch[:B].cpu().numpy()
    return rec, got


def build_decision(a):
    """Configs from the strict sampler; config i is rolled out by source agent i mod (#sources), sources = every
    GM and control seed of the env, so half the frames come from GM rollouts and half from control rollouts."""
    from gm_robotics.analysis import envs_forced as E
    kind = a.env
    dev = torch.device("cuda")
    srcs = [("gm", s) for s in SEEDS[kind]["gm"]] + [("control", s) for s in SEEDS[kind]["control"][:len(SEEDS[kind]["gm"])]]
    n_cfg = a.n_per_source * len(srcs)
    cfg = sample_strict(kind, n_cfg // 2, BANK_SEED[kind] + 1, a.min_angle, a.max_ddist)
    n = len(cfg["u"])
    src_of = np.arange(n) % len(srcs)
    env = E.make_env(kind, 3 * a.batch)
    T = 50 if kind == "AT" else 100
    parts = []
    for k, (coh, seed) in enumerate(srcs):
        agent = C.build_agent(C.ckpt_path(kind, coh, seed), dev=dev)
        ids = np.where(src_of == k)[0]
        for sl in C.batched(len(ids), a.batch):
            idx = ids[sl]
            pad = np.concatenate([idx, np.full(a.batch - len(idx), idx[-1])])
            rec, got = collect_decision_batch(kind, env, agent, cfg, pad, dev, T)
            m = got[: len(idx)]
            parts.append(dict(cfg_idx=idx[m], src_cohort=np.full(m.sum(), 0 if coh == "gm" else 1),
                              src_seed=np.full(m.sum(), seed), t_frame=rec["t_frame"][: len(idx)][m],
                              tcp=rec["tcp"][: len(idx)][m], qpos=rec["qpos"][: len(idx)][m],
                              **({"cube_now": rec["cube_now"][: len(idx)][m]} if kind == "SZ" else {}),
                              **{f"rgb_{e}": rec["rgb"][i][: len(idx)][m] for i, e in enumerate(("base", GOAL_E[kind], PROX_E[kind]))},
                              **{f"state_{e}": rec["state"][i][: len(idx)][m] for i, e in enumerate(("base", GOAL_E[kind], PROX_E[kind]))}))
            print(f"decision {kind} {coh} s{seed}: kept {m.sum()}/{len(idx)} median t {np.median(rec['t_frame'][: len(idx)][m]) if m.any() else -1}", flush=True)
    bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ci = bank["cfg_idx"]
    bank.update(u=cfg["u"][ci], v=cfg["v"][ci], qpos_forced=cfg["qpos"][ci], pair=ci // 2)
    if kind == "SZ":
        bank["cube"] = cfg["cube"][ci]
    for e in (GOAL_E[kind], PROX_E[kind]):
        bank[f"npx_{e}"], bank[f"pxl2_{e}"] = pixel_change(bank["rgb_base"], bank[f"rgb_{e}"])
    _, bank["J_tcp"] = E.tcp_pos_and_jac(bank["qpos"])
    bank["sign"] = np.where(bank["u"][:, 1] > bank["v"][:, 1], 1, -1).astype(np.int8)
    save_bank(bank, bank_path(kind, "decision"))
    env.close()


# ---------------------------------------------------------------- geometry per bank variant

def balanced_subset(b, keep_mask, seed=0):
    """Indices with keep_mask, subsampled so both conflict orientations (and, for decision banks, both source
    cohorts within each orientation) have equal counts."""
    rng = np.random.default_rng(seed)
    idx = np.where(keep_mask)[0]
    groups = {}
    src = b["src_cohort"] if "src_cohort" in b else np.zeros(len(b["sign"]), int)
    for i in idx:
        groups.setdefault((int(b["sign"][i]), int(src[i])), []).append(i)
    m = min(len(g) for g in groups.values()) if len(groups) == (4 if "src_cohort" in b else 2) else 0
    out = np.concatenate([rng.permutation(g)[:m] for g in groups.values()]) if m else idx
    return np.sort(out)


def split(b, idx, frac_eval=0.4, seed=0):
    """Fit / held-out split by mirror pair (pairs never straddle the split); 'orig' keeps axis_geometry's split."""
    rng = np.random.default_rng(seed)
    pairs = np.unique(b["pair"][idx])
    ev_pairs = set(rng.permutation(pairs)[: int(round(frac_eval * len(pairs)))].tolist())
    ev = np.array([i for i in idx if b["pair"][i] in ev_pairs])
    fit = np.array([i for i in idx if b["pair"][i] not in ev_pairs])
    return fit, ev


def blend(base, edit, alpha):
    """Pixel-space interpolation base + alpha (edit - base): same pixel-change direction, smaller L2."""
    return base.astype(np.float32) + alpha[:, None, None, None] * (edit.astype(np.float32) - base.astype(np.float32))


def make_view(env, b, idx, matched):
    """Bank dict restricted to idx with keys axis_geometry.agent_geometry expects; matched=True shrinks the
    larger of the two cue edits by pixel interpolation so both have the same pixel L2 in every state."""
    g, p = GOAL_E[env], PROX_E[env]
    out = {k: b[k][idx] for k in ("sign",)}
    for e in ("base", g, p):
        out[f"rgb_{e}"], out[f"state_{e}"] = b[f"rgb_{e}"][idx], b[f"state_{e}"][idx]
    lg, lp = b[f"pxl2_{g}"][idx].astype(np.float32), b[f"pxl2_{p}"][idx].astype(np.float32)
    if matched:
        ap, ag = np.minimum(1.0, lg / lp), np.minimum(1.0, lp / lg)
        out[f"rgb_{p}"] = blend(out["rgb_base"], out[f"rgb_{p}"], ap)
        out[f"rgb_{g}"] = blend(out["rgb_base"], out[f"rgb_{g}"], ag)
    return out


def action_terms(g, sig, Jtcp, u, v, ref):
    """Within-state action-space diagnostics: Fisher cosine on arm joints only, cosine of the first-order TCP
    displacement the two edits cause, the lateral TCP components along the u-v contrast, and the fraction of
    d_comp's Fisher energy that lies on the whitened action direction moving the TCP from v toward u."""
    J = g["J"]
    a_s, a_p = torch.einsum("bad,bd->ba", J, g["d_star"]), torch.einsum("bad,bd->ba", J, g["d_prox"])
    w = 1.0 / sig
    ws, wp = a_s * w, a_p * w
    cosF_arm = AG.cos_e(ws[:, :7], wp[:, :7])
    Jt = torch.as_tensor(Jtcp, device=J.device)
    ts = torch.einsum("bij,bj->bi", Jt, C.JOINT_DELTA_SCALE * a_s[:, :7])
    tp = torch.einsum("bij,bj->bi", Jt, C.JOINT_DELTA_SCALE * a_p[:, :7])
    e = torch.as_tensor(u - v, device=J.device)
    e = torch.cat([e / e.norm(dim=1, keepdim=True), torch.zeros(len(e), 1, device=J.device)], 1)
    lat_s, lat_p = (ts * e).sum(1), (tp * e).sum(1)
    # whitened action direction with the largest lateral TCP velocity per unit Fisher norm: diag(sigma) J_tcp^T e
    c = torch.einsum("bij,bi->bj", Jt, e) * C.JOINT_DELTA_SCALE * sig[None, :7]
    c = torch.cat([c, torch.zeros(len(c), 1, device=J.device)], 1)
    wc = torch.einsum("bad,bd->ba", J, g["d_comp"]) * w
    frac = (wc * c).sum(1) ** 2 / ((wc * wc).sum(1) * (c * c).sum(1)).clamp_min(1e-18)
    common = ((ws + wp) ** 2).sum(1)
    opp = ((ws - wp) ** 2).sum(1)
    # Split the whitened arm-action change into the 3-d subspace that moves the TCP (row space of
    # 0.1 J_tcp diag(sigma)) and its 4-d complement (joint motion that leaves the TCP in place, to first order).
    M = C.JOINT_DELTA_SCALE * Jt * sig[None, None, :7]
    Pi = M.transpose(1, 2) @ torch.linalg.solve(M @ M.transpose(1, 2), M)
    ps, pp = torch.einsum("bij,bj->bi", Pi, ws[:, :7]), torch.einsum("bij,bj->bi", Pi, wp[:, :7])
    ns, np_ = ws[:, :7] - ps, wp[:, :7] - pp
    return dict(cosF_arm=cosF_arm, cos_tcp=AG.cos_e(ts, tp), lat_goal=lat_s, lat_prox=lat_p, frac_contrast=frac,
                cosF_tcpsub=AG.cos_e(ps, pp), cosF_nullsub=AG.cos_e(ns, np_),
                tcp_share_goal=(ps ** 2).sum(1) / (ws ** 2).sum(1).clamp_min(1e-18),
                tcp_share_prox=(pp ** 2).sum(1) / (wp ** 2).sum(1).clamp_min(1e-18),
                common_share=common / (common + opp).clamp_min(1e-18),
                tcp_goal_norm=ts.norm(dim=1), tcp_prox_norm=tp.norm(dim=1))


def geometry_one(env, b, view, fit_l, ev_l, agent, dev, cohort, seed, label):
    sign = view["sign"].astype(int)
    g = AG.agent_geometry(agent, view, dev, env)
    dnp = g["d_comp"].cpu().numpy().astype(np.float64)
    u = AX.fit_u_comp({+1: [dnp[i] for i in fit_l if sign[i] > 0], -1: [dnp[i] for i in fit_l if sign[i] < 0]})
    held = AX.held_out_alignment(u, [dnp[i] for i in ev_l], [float(sign[i]) for i in ev_l])
    ut = torch.as_tensor(u, dtype=torch.float32, device=dev)
    evt = torch.as_tensor(ev_l, device=dev)
    sg = torch.as_tensor(sign, dtype=torch.float32, device=dev)
    a_prox = (AG.alpha_f(g["Jw"], ut, g["d_prox"]) * sg)[evt]
    a_star = (AG.alpha_f(g["Jw"], ut, g["d_star"]) * sg)[evt]
    cw = AG.cos_e(g["d_star"], g["d_prox"])
    cfw = AG.cos_f_same_state(g["Jw"], g["d_star"], g["d_prox"])
    cs, _ = AG.cross_state(g, np.asarray(fit_l), sign)
    act = action_terms(g, g["sig"], b["J_tcp_v"], b["u_v"], b["v_v"], None)
    allst = torch.arange(len(sign), device=dev)
    r = dict(bank=label, cohort=cohort, seed=seed, n_states=len(sign), n_fit=len(fit_l), n_eval=len(ev_l),
             cosE_within=cw.median().item(), cosF_within=cfw.median().item(),
             cosE_within_mean=cw.mean().item(), cosF_within_mean=cfw.mean().item(),
             cosF_arm_within=act["cosF_arm"].median().item(), cos_tcp_within=act["cos_tcp"].median().item(),
             common_share=act["common_share"].median().item(), frac_contrast_dcomp=act["frac_contrast"].median().item(),
             cosF_tcpsub_within=act["cosF_tcpsub"].median().item(), cosF_nullsub_within=act["cosF_nullsub"].median().item(),
             tcp_share_goal=act["tcp_share_goal"].median().item(), tcp_share_prox=act["tcp_share_prox"].median().item(),
             frac_lat_goal_neg=(act["lat_goal"] < 0).float().mean().item(),
             frac_lat_prox_pos=(act["lat_prox"] > 0).float().mean().item(),
             **cs, heldout_align=held["mean"], heldout_frac_pos=held["frac_pos"],
             pull_proxy=a_prox.abs().median().item(), pull_goal=a_star.abs().median().item(),
             pull_ratio=(a_prox.abs().median() / a_star.abs().median().clamp_min(1e-12)).item(),
             dcomp_norm=g["d_comp"].norm(dim=-1).median().item())
    per_state = pd.DataFrame(dict(bank=label, cohort=cohort, seed=seed, state=allst.cpu().numpy(), sign=sign,
                                  cosE_within=cw.cpu().numpy(), cosF_within=cfw.cpu().numpy(),
                                  cosF_arm=act["cosF_arm"].cpu().numpy(), cos_tcp=act["cos_tcp"].cpu().numpy(),
                                  lat_goal=act["lat_goal"].cpu().numpy(), lat_prox=act["lat_prox"].cpu().numpy(),
                                  common_share=act["common_share"].cpu().numpy(),
                                  cosF_tcpsub=act["cosF_tcpsub"].cpu().numpy(), cosF_nullsub=act["cosF_nullsub"].cpu().numpy(),
                                  tcp_share_goal=act["tcp_share_goal"].cpu().numpy(),
                                  tcp_share_prox=act["tcp_share_prox"].cpu().numpy(),
                                  frac_contrast=act["frac_contrast"].cpu().numpy()))
    return r, per_state


def bank_views(env, variant, b):
    """(label, keep-index array, fit, eval) per analysed subset of one bank file."""
    n = len(b["sign"])
    if variant == "orig":
        fit, check, ev = AG.split_indices(n, 200, 56)
        L = layout(env, b)
        yield "orig", np.arange(n), fit, ev
        if env == "AT":
            keep = (L["angle"] >= 45) & ~L["same_side"]
            idx = np.where(keep)[0]
            yield "orig_wide", idx, idx[~np.isin(idx, ev)], idx[np.isin(idx, ev)]
        return
    keep = np.ones(n, bool)
    if variant == "decision":
        g, p = GOAL_E[env], PROX_E[env]
        # an edit hidden by the arm leaves no cue change to measure
        for e in (g, p):
            keep &= b[f"npx_{e}"] >= 0.5 * np.median(b[f"npx_{e}"])
        idx = balanced_subset(b, keep)
    else:
        idx = np.arange(n)
    fit, ev = split(b, idx)
    yield variant, idx, fit, ev
    if variant == "decision":
        for coh, name in ((0, "decision_gmsrc"), (1, "decision_ctrlsrc")):
            sub = idx[b["src_cohort"][idx] == coh]
            yield name, sub, sub[np.isin(sub, fit)], sub[np.isin(sub, ev)]


def geometry(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    rows, states = [], []
    agents = [("gm", s, C.ckpt_path(a.env, "gm", s)) for s in SEEDS[a.env]["gm"]]
    agents += [("control", s, C.ckpt_path(a.env, "control", s)) for s in SEEDS[a.env]["control"]]
    agents += [("init", s, None) for s in (0, 1, 2)]
    built = {(c, s): C.build_agent(ck, init_seed=s, dev=dev) for c, s, ck in agents}
    for variant in a.variants:
        f = bank_path(a.env, variant)
        if not os.path.exists(f):
            print(f"missing {f}", flush=True)
            continue
        b = C.load_bank(os.path.basename(f)[:-4])
        if variant == "orig":
            b["pair"] = np.arange(len(b["sign"]))
        for label, idx, fit, ev in bank_views(a.env, variant, b):
            pos = {int(i): j for j, i in enumerate(idx)}
            fit_l, ev_l = [pos[int(i)] for i in fit], [pos[int(i)] for i in ev]
            for matched in (False, True):
                lab = label + ("_pxmatched" if matched else "")
                view = make_view(a.env, b, idx, matched)
                b["J_tcp_v"], b["u_v"], b["v_v"] = b["J_tcp"][idx], b["u"][idx], b["v"][idx]
                for cohort, seed, _ in agents:
                    r, ps = geometry_one(a.env, b, view, fit_l, ev_l, built[(cohort, seed)], dev, cohort, seed, lab)
                    rows.append(r)
                    ps["state"] = idx[ps["state"].to_numpy()]
                    states.append(ps)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                           if k in ("bank", "cohort", "seed", "n_states", "cosE_within", "cosF_within", "cos_tcp_within",
                                    "cosF_same", "cosF_opp", "heldout_align", "pull_ratio", "frac_contrast_dcomp")}, flush=True)
    df = pd.DataFrame(rows)
    tag = a.tag or a.env
    df.to_csv(os.path.join(OUT, f"geometry_{tag}_per_seed.csv"), index=False)
    pd.concat(states).to_csv(os.path.join(OUT, f"geometry_{tag}_per_state.csv"), index=False)
    summarize(df, os.path.join(OUT, f"geometry_{tag}_summary"))


def summarize(df, base):
    cols = [c for c in df.columns if c not in ("bank", "cohort", "seed", "n_states", "n_fit", "n_eval")]
    out = []
    for (bank, cohort), sub in df.groupby(["bank", "cohort"], sort=False):
        r = dict(bank=bank, cohort=cohort, n_seeds=len(sub), n_states=int(sub.n_states.iloc[0]))
        for c in cols:
            r[c], r[c + "_lo"], r[c + "_hi"] = C.ci95(sub[c].to_numpy())
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(base + ".csv", index=False)
    show = ["cosE_within", "cosF_within", "cosF_tcpsub_within", "cosF_nullsub_within", "tcp_share_goal",
            "tcp_share_prox", "cos_tcp_within", "cosF_same", "cosF_opp",
            "heldout_align", "frac_contrast_dcomp", "pull_ratio"]
    with open(base + ".md", "w") as f:
        f.write("# Competition geometry per bank variant at h_fc (mean and 95% t-CI across seeds; per-seed medians over states)\n\n")
        f.write("| bank | cohort | n | " + " | ".join(show) + " |\n|---|---|---|" + "---|" * len(show) + "\n")
        for _, r in s.iterrows():
            f.write(f"| {r.bank} | {r.cohort} | {r.n_states} | " +
                    " | ".join(C.fmt_ci(r[c], r[c + '_lo'], r[c + '_hi'], 2) for c in show) + " |\n")
    print(open(base + ".md").read(), flush=True)


def sheet(a):
    """Contact sheet of the first states of each bank: base, goal-cue swap, proxy-cue swap and |diff| maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, p = GOAL_E[a.env], PROX_E[a.env]
    for variant in a.variants:
        f = bank_path(a.env, variant)
        if not os.path.exists(f):
            continue
        b = C.load_bank(os.path.basename(f)[:-4])
        rng = np.random.default_rng(0)
        ids = np.sort(rng.choice(len(b["sign"]), 6, replace=False))
        fig, ax = plt.subplots(6, 5, figsize=(10, 12))
        for r, i in enumerate(ids):
            ims = [b["rgb_base"][i], b[f"rgb_{g}"][i], b[f"rgb_{p}"][i],
                   np.abs(b[f"rgb_{g}"][i].astype(int) - b["rgb_base"][i]).sum(-1),
                   np.abs(b[f"rgb_{p}"][i].astype(int) - b["rgb_base"][i]).sum(-1)]
            for c, im in enumerate(ims):
                ax[r, c].imshow(im, cmap="magma" if c >= 3 else None, vmin=0 if c >= 3 else None, vmax=600 if c >= 3 else None)
                ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            extra = f" t={b['t_frame'][i]} src={'GM' if b['src_cohort'][i] == 0 else 'Non-GM'}" if "t_frame" in b else ""
            ax[r, 0].set_ylabel(f"#{i} sign {b['sign'][i]:+d}{extra}", fontsize=7)
        for c, t in enumerate(["base", g, p, f"|{g}|", f"|{p}|"]):
            ax[0, c].set_title(t, fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.join(C.FIGURES, "bank_audit"), exist_ok=True)
        fig.savefig(os.path.join(C.FIGURES, "bank_audit", f"sheet_{a.env}_{variant}.png"), dpi=110)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["stats", "strict", "decision", "geometry", "sheet"])
    p.add_argument("--env", choices=["AT", "SZ"], required=True)
    p.add_argument("--variants", nargs="+", default=["orig", "strict", "decision"])
    p.add_argument("--n", type=int, default=512, help="strict bank states (mirror pairs)")
    p.add_argument("--n_per_source", type=int, default=128)
    p.add_argument("--min_angle", type=float, default=None)
    p.add_argument("--max_ddist", type=float, default=None)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    # AT: bearing from the start TCP >= 45 deg and equal distance within 3 cm; SZ (seen from the cube): >= 120 deg, 1 cm.
    a.min_angle = a.min_angle if a.min_angle is not None else {"AT": 45.0, "SZ": 120.0}[a.env]
    a.max_ddist = a.max_ddist if a.max_ddist is not None else {"AT": 0.03, "SZ": 0.01}[a.env]
    C.ensure_dirs()
    dict(stats=stats, strict=build_strict, decision=build_decision, geometry=geometry, sheet=sheet)[a.step](a)


if __name__ == "__main__":
    main()
