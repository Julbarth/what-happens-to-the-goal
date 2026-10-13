"""Isaac quadruped analysis: agent loading, activation taps and paths; Fisher/KL/CI helpers reused from analysis/common.py."""
import os

import numpy as np
import torch
from gm_robotics.paths import ROOT as RUN_ROOT

ROOT = str(RUN_ROOT)
from gm_robotics.analysis import common as MC  # noqa: E402  (ManiSkill port: gaussian_kl, fisher_terms, ci95, seed_summary, gmg_axis, PAL)
from gm_robotics.isaac.agent import Agent  # noqa: E402

RESULTS = os.environ.get("ISAAC_ANALYSIS_RESULTS", os.path.join(ROOT, "results", "isaac_analysis"))
BANKS = os.path.join(RESULTS, "banks")
FIGURES = os.path.join(ROOT, "figures", "isaac_analysis")
CKPTS = os.environ.get("ISAAC_CKPTS", os.path.join(ROOT, "checkpoints", "isaac"))
SEEDS = {"gm": [0, 1, 2], "control": [0, 1, 2]}
PAL = MC.PAL
ci95, fmt_ci, seed_summary, gaussian_kl, fisher_terms, batched = (MC.ci95, MC.fmt_ci, MC.seed_summary, MC.gaussian_kl,
                                                                   MC.fisher_terms, MC.batched)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def ckpt_path(cohort, seed):
    p = os.path.join(CKPTS, "runs", f"main_{cohort}_s{seed}", "final_ckpt.pt")
    return p if os.path.exists(p) else None


def agent_list(seeds_gm=None, seeds_ctrl=None, init_seeds=(0, 1, 2)):
    """[(cohort, seed, ckpt_or_None)]; agents whose final checkpoint does not exist yet are skipped."""
    out = [("gm", s, ckpt_path("gm", s)) for s in (seeds_gm or SEEDS["gm"])]
    out += [("control", s, ckpt_path("control", s)) for s in (seeds_ctrl or SEEDS["control"])]
    out = [r for r in out if r[2] is not None]
    return out + [("init", s, None) for s in init_seeds]


def build_agent(ckpt=None, init_seed=0, dev="cpu"):
    torch.manual_seed(init_seed)
    a = Agent((96, 96, 3), 12, 3)
    if ckpt is not None:
        a.load_state_dict(torch.load(ckpt, map_location="cpu"))
    a.requires_grad_(False)
    return a.to(dev).eval()


def parts(a):
    cnn = a.feature_net.rgb[0]
    return dict(conv1=cnn[0], conv2=cnn[2], conv3=cnn[4], fc=a.feature_net.rgb[1], state=a.feature_net.state,
                actor0=a.actor_mean[0], actor2=a.actor_mean[2])


def sigma(a):
    return torch.exp(a.actor_logstd).reshape(-1)


def taps(a, rgb, state):
    """All analysed sites from uint8 NHWC rgb; mu equals agent.get_action(deterministic=True)."""
    p = parts(a)
    x = rgb.float().permute(0, 3, 1, 2) / 255
    c1 = torch.relu(p["conv1"](x))
    c2 = torch.relu(p["conv2"](c1))
    c3 = torch.relu(p["conv3"](c2))
    h_pre = c3.flatten(1)
    h_fc = torch.relu(p["fc"](h_pre))
    h_state = p["state"](state.float())
    a_hid = torch.relu(p["actor0"](torch.cat([h_fc, h_state], 1)))
    return dict(conv1=c1, conv2=c2, conv3=c3, h_pre=h_pre, h_fc=h_fc, h_state=h_state, a_hid=a_hid, mu=p["actor2"](a_hid))


def mu_from_fc(a, h_fc, h_state):
    p = parts(a)
    return p["actor2"](torch.relu(p["actor0"](torch.cat([h_fc, h_state], -1))))


def mu_from_pre(a, h_pre, h_state):
    return mu_from_fc(a, torch.relu(parts(a)["fc"](h_pre)), h_state)


def mean_jacobian(a, site, h, h_state):
    f = mu_from_fc if site == "h_fc" else mu_from_pre
    fn = lambda hh, hs: f(a, hh[None], hs[None])[0]
    return torch.func.vmap(torch.func.jacrev(fn, argnums=0))(h, h_state)


def load_bank(name):
    z = np.load(os.path.join(BANKS, name + ".npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def to_t(x, dev):
    return torch.as_tensor(x).to(dev)


def ensure_dirs():
    for d in (RESULTS, FIGURES, BANKS):
        os.makedirs(d, exist_ok=True)
