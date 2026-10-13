"""Monte Carlo check that the robot policy Fisher at h_fc equals J^T diag(sigma^-2) J, on the 19 local arm checkpoints."""
# CPU only. Run from the procgen/ directory:
#   python figures/app_fisher_mc_check.py --banks <dir with {LR,AT,PTM}_worldedit.npz>
# Banks: robotics results/analysis/banks/ (see robotics/README). The forward pass mirrors
# gm_robotics/analysis/common.py taps() from the state dict, so mani_skill is not needed.
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from paths import ROBOTICS_ROOT, TABLES

GR = ROBOTICS_ROOT
OUT = TABLES / "sigma_mc_check.csv"
# Qualified checkpoints as in gm_robotics/analysis/common.py (QUALIFIED, PTM_RUNS).
CKPTS = (
    [("PTM", c, s, GR / f"runs/{r}_PushToMarker_{c}_s{s}/final_ckpt.pt")
     for (c, s), r in {("gm", 0): "m50", ("gm", 1): "m50", ("gm", 2): "m50",
                       ("control", 0): "m50", ("control", 1): "m100", ("control", 2): "m100"}.items()]
    + [("AT", c, s, GR / f"checkpoints/analysis/main_AttributeTable_{c}_s{s}/final_ckpt.pt")
       for c, ss in (("gm", (2, 3, 4)), ("control", (2, 3, 4, 5))) for s in ss]
    + [("LR", c, s, GR / f"checkpoints/analysis/main_LocationReach_{c}_s{s}/final_ckpt.pt")
       for c in ("gm", "control") for s in (1, 2, 3)]
)
BASE_KEYS = {"LR": ("rgb_g_d", "state_g_d"), "AT": ("rgb_base", "state_base"), "PTM": ("rgb_base", "state_base")}


def h_fc_and_state(sd, rgb, state):
    x = torch.as_tensor(rgb).float().permute(0, 3, 1, 2) / 255
    p = "feature_net.extractors."
    x = F.relu(F.conv2d(x, sd[p + "rgb.0.0.weight"], sd[p + "rgb.0.0.bias"], stride=4))
    x = F.relu(F.conv2d(x, sd[p + "rgb.0.2.weight"], sd[p + "rgb.0.2.bias"], stride=2))
    x = F.relu(F.conv2d(x, sd[p + "rgb.0.4.weight"], sd[p + "rgb.0.4.bias"], stride=1))
    h_fc = F.relu(F.linear(x.flatten(1), sd[p + "rgb.1.0.weight"], sd[p + "rgb.1.0.bias"]))
    h_state = F.linear(torch.as_tensor(state).float(), sd[p + "state.weight"], sd[p + "state.bias"])
    return h_fc, h_state


def mu(sd, h_fc, h_state):
    a = F.relu(F.linear(torch.cat([h_fc, h_state], -1), sd["actor_mean.0.weight"], sd["actor_mean.0.bias"]))
    return F.linear(a, sd["actor_mean.2.weight"], sd["actor_mean.2.bias"])


def check_state(sd, h, hs, n_act, gen):
    """Relative trace and Frobenius error of the Monte Carlo score covariance against the closed form."""
    sig = torch.exp(sd["actor_logstd"]).reshape(-1).double()
    m = mu(sd, h[None], hs[None])[0].detach()
    acts = m.double() + sig * torch.randn(n_act, len(sig), generator=gen, dtype=torch.float64)

    def logp(hh, a):
        mm = mu(sd, hh[None], hs[None])[0].double()
        return torch.distributions.Normal(mm, sig).log_prob(a).sum()

    # Autograd score of the full log-density, so a sigma that depended on h would show up as a mismatch.
    scores = torch.func.vmap(torch.func.grad(logp), in_dims=(None, 0))(h, acts.float())
    f_mc = scores.T @ scores / n_act
    J = torch.func.jacrev(lambda hh: mu(sd, hh[None], hs[None])[0])(h).double()
    f_cf = J.T @ torch.diag(sig ** -2) @ J
    tr_cf, fro_cf = torch.trace(f_cf), torch.linalg.norm(f_cf)
    # Sampling noise for Gaussian scores: E||F_mc - F||_F^2 = (tr(F)^2 + ||F||_F^2) / N, Var tr(F_mc) = 2||F||_F^2 / N.
    return dict(tr_rel=float(abs(torch.trace(f_mc) - tr_cf) / tr_cf),
                fro_rel=float(torch.linalg.norm(f_mc - f_cf) / fro_cf),
                tr_rel_noise_sd=float(np.sqrt(2 / n_act) * fro_cf / tr_cf),
                fro_rel_noise_rms=float(np.sqrt((tr_cf ** 2 + fro_cf ** 2) / n_act) / fro_cf),
                sigma_mean=float(sig.mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--banks", type=Path, required=True)
    ap.add_argument("--n-actions", type=int, default=4000)
    ap.add_argument("--n-states", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(True)
    banks = {}
    for t, (kr, ks) in BASE_KEYS.items():
        z = np.load(args.banks / f"{t}_worldedit.npz")
        banks[t] = (z[kr][: args.n_states], z[ks][: args.n_states])
    rows = []
    for task, coh, seed, path in CKPTS:
        sd = {k: v.float() for k, v in torch.load(path, map_location="cpu").items()}
        assert tuple(sd["actor_logstd"].shape) == (1, 8)
        rgb, st = banks[task]
        h, hs = h_fc_and_state(sd, rgb, st)
        gen = torch.Generator().manual_seed(args.seed)
        for i in range(len(h)):
            rows.append({"task": task, "cohort": coh, "seed": seed, "state": i,
                         **check_state(sd, h[i].detach(), hs[i].detach(), args.n_actions, gen)})
        print(task, coh, seed, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(f"{d[['task', 'cohort', 'seed']].drop_duplicates().shape[0]} checkpoints x {args.n_states} states")
    print(d[["tr_rel", "fro_rel", "tr_rel_noise_sd", "fro_rel_noise_rms"]].describe().round(4))


if __name__ == "__main__":
    main()
