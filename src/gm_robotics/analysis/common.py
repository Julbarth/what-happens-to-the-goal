"""Shared loaders, activation taps, diagonal-Gaussian policy Fisher and CI helpers for the robotics GM analyses."""
import glob
import importlib.util
import os

import numpy as np
import torch
from gm_robotics.paths import ROOT as RUN_ROOT

ROOT = str(RUN_ROOT)

# Located without importing gm_geometry, whose subpackages pull in procgen.
GMGEO_SRC = importlib.util.find_spec("gm_geometry").submodule_search_locations[0]

RESULTS = os.environ.get("ANALYSIS_RESULTS", os.path.join(ROOT, "results", "analysis"))
FIGURES = os.environ.get("ANALYSIS_FIGURES", os.path.join(ROOT, "figures", "analysis"))
BANKS = os.path.join(RESULTS, "banks")

ENV_IDS = {"AT": "AttributeTable-v1", "LR": "LocationReach-v1", "SZ": "SortZones-v1", "PTM": "PushToMarker-v1"}
QUALIFIED = {"AT": {"gm": [2, 3, 4], "control": [3, 4, 5]},
             "LR": {"gm": [1, 2, 3], "control": [1, 2, 3]},
             "SZ": {"gm": [0, 1, 2], "control": [0, 1, 2]},
             "PTM": {"gm": [0, 1, 2], "control": [0, 1, 2]}}
CKPT_ROOT = os.environ.get("GM_ROBOTICS_CKPTS", os.path.join(ROOT, "checkpoints"))
CKPT_DIRS = {k: os.environ.get(f"{k}_CKPT_DIR", CKPT_ROOT) for k in ("AT", "LR", "SZ", "PTM")}
CKPT_PREFIX = {"AT": "main", "LR": "main", "SZ": "m20"}
# PushToMarker budgets differ per run: GM and control s0 at 50M, control s1 / s2 at 100M (undertrained at 50M).
PTM_RUNS = {("gm", 0): "m50", ("gm", 1): "m50", ("gm", 2): "m50",
            ("control", 0): "m50", ("control", 1): "m100", ("control", 2): "m100"}
N_ACT, STATE_DIM, OBS_RES = 8, 25, 128
# pd_joint_delta_pos maps a normalized action in [-1, 1] to a joint-position delta in [-0.1, 0.1] rad.
JOINT_DELTA_SCALE = 0.1

PAL = dict(gm="#7A1520", gm_proxy="#C45C58", nongm="#3D8B8B", nongm_proxy="#5FA8A0", init="#8E8E8E",
           null="#5A5A5A", ink="#2A2A2A", mute="#6E6E6E", paper="#FFFFFF")


def _load_gmgeo(rel, name):
    """Import one gm-geometry module by file path; the package __init__ pulls in procgen-only dependencies."""
    path = os.path.join(GMGEO_SRC, rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gmg_linear = _load_gmgeo("probes/linear.py", "gmg_linear")
gmg_axis = _load_gmgeo("probes/compressed_axis_geometry.py", "gmg_axis")
gmg_utils = _load_gmgeo("utils/__init__.py", "gmg_utils")
ci95 = gmg_utils.ci95


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def ckpt_path(env, cohort, seed):
    if env == "PTM":
        return os.path.join(CKPT_DIRS[env], f"{PTM_RUNS[(cohort, seed)]}_PushToMarker_{cohort}_s{seed}", "final_ckpt.pt")
    return os.path.join(CKPT_DIRS[env], f"{CKPT_PREFIX[env]}_{ENV_IDS[env].split('-')[0]}_{cohort}_s{seed}", "final_ckpt.pt")


def agent_list(env, seeds_gm=None, seeds_ctrl=None, init_seeds=(0, 1, 2)):
    """[(cohort, seed, ckpt_or_None)] for the qualified agents plus randomly initialised networks."""
    out = [("gm", s, ckpt_path(env, "gm", s)) for s in (seeds_gm or QUALIFIED[env]["gm"])]
    out += [("control", s, ckpt_path(env, "control", s)) for s in (seeds_ctrl or QUALIFIED[env]["control"])]
    out += [("init", s, None) for s in init_seeds]
    return out


class _FakeEnvs:
    class unwrapped:
        class single_action_space:
            shape = (N_ACT,)


def build_agent(ckpt=None, init_seed=0, dev="cpu"):
    """ppo_rgb_gm.Agent from a checkpoint, or freshly initialised with the training init scheme."""
    from gm_robotics.ppo_rgb_gm import Agent
    sample = {"rgb": torch.zeros(1, OBS_RES, OBS_RES, 3, dtype=torch.uint8), "state": torch.zeros(1, STATE_DIM)}
    torch.manual_seed(init_seed)
    agent = Agent(_FakeEnvs, sample_obs=sample)
    if ckpt is not None:
        agent.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agent.requires_grad_(False)
    return agent.to(dev).eval()


def parts(agent):
    rgb = agent.feature_net.extractors["rgb"]
    cnn, fc = rgb[0], rgb[1][0]
    return dict(conv1=cnn[0], conv2=cnn[2], conv3=cnn[4], fc=fc, state=agent.feature_net.extractors["state"],
                actor0=agent.actor_mean[0], actor2=agent.actor_mean[2])


def sigma(agent):
    return torch.exp(agent.actor_logstd).reshape(-1)


def taps(agent, rgb, state):
    """All analysed sites; rgb is uint8 NHWC as the env returns it. mu equals agent.get_action(det=True)."""
    p = parts(agent)
    x = rgb.float().permute(0, 3, 1, 2) / 255
    c1 = torch.relu(p["conv1"](x))
    c2 = torch.relu(p["conv2"](c1))
    c3 = torch.relu(p["conv3"](c2))
    h_pre = c3.flatten(1)
    h_fc = torch.relu(p["fc"](h_pre))
    h_state = p["state"](state.float())
    a_hid = torch.relu(p["actor0"](torch.cat([h_fc, h_state], 1)))
    mu = p["actor2"](a_hid)
    return dict(conv1=c1, conv2=c2, conv3=c3, h_pre=h_pre, h_fc=h_fc, h_state=h_state, a_hid=a_hid, mu=mu)


def mu_from_fc(agent, h_fc, h_state):
    p = parts(agent)
    return p["actor2"](torch.relu(p["actor0"](torch.cat([h_fc, h_state], -1))))


def mu_from_pre(agent, h_pre, h_state):
    return mu_from_fc(agent, torch.relu(parts(agent)["fc"](h_pre)), h_state)


def mean_jacobian(agent, site, h, h_state):
    """J = d mu / d h per state, [B, A, D]; the proprio branch is held at its own value."""
    f = mu_from_fc if site == "h_fc" else mu_from_pre
    fn = lambda hh, hs: f(agent, hh[None], hs[None])[0]
    return torch.func.vmap(torch.func.jacrev(fn, argnums=0))(h, h_state)


def gaussian_kl(mu1, mu2, sig):
    """KL between diagonal Gaussians with the same (state-independent) sigma; symmetric in mu1, mu2."""
    return 0.5 * (((mu1 - mu2) / sig) ** 2).sum(-1)


def fisher_terms(J, sig, delta):
    """Policy Fisher F = J^T diag(1/sigma^2) J at the base state (never formed; D is 9216 at h_pre).

    Returns Q = delta^T F delta, tr(F), ||delta||^2, and the paper's G / D_F split."""
    w = 1.0 / sig ** 2
    jd = torch.einsum("bad,bd->ba", J, delta)
    q = (w * jd ** 2).sum(-1)
    tr = (w[None, :, None] * J ** 2).sum((1, 2))
    d = J.shape[-1]
    n2 = (delta ** 2).sum(-1)
    iso = tr / d
    return dict(Q=q, trF=tr, trF_per_d=iso, dnorm2=n2, G=(q / n2.clamp_min(1e-30)) / iso.clamp_min(1e-30),
                D_F=n2 * iso)


def fisher_bilinear(J, sig, a, b):
    """a^T F b for per-state vectors a, b [B, D]."""
    w = 1.0 / sig ** 2
    return (w * torch.einsum("bad,bd->ba", J, a) * torch.einsum("bad,bd->ba", J, b)).sum(-1)


def seed_summary(df, value_cols, group_cols, agg="median"):
    """Per-seed statistic over states (paper: median for Fisher terms, mean for behavioural terms), then
    mean and 95% t-CI across seeds within each group. Returns (per_seed, summary)."""
    import pandas as pd
    keys = group_cols + ["cohort", "seed"]
    per_seed = df.groupby(keys)[value_cols].agg(agg).reset_index()
    rows = []
    for g, sub in per_seed.groupby(group_cols + ["cohort"]):
        g = g if isinstance(g, tuple) else (g,)
        r = dict(zip(group_cols + ["cohort"], g))
        r["n_seeds"] = len(sub)
        for c in value_cols:
            m, lo, hi = ci95(sub[c].to_numpy())
            r[c], r[c + "_lo"], r[c + "_hi"] = m, lo, hi
        rows.append(r)
    return per_seed, pd.DataFrame(rows)


def fmt_ci(m, lo, hi, nd=3):
    if not np.isfinite(lo):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def load_bank(name):
    z = np.load(os.path.join(BANKS, name + ".npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def ensure_dirs():
    for d in (RESULTS, FIGURES, BANKS):
        os.makedirs(d, exist_ok=True)


def batched(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))


def to_t(x, dev):
    return torch.as_tensor(x).to(dev)


def list_ckpts(pattern):
    return sorted(glob.glob(pattern))
