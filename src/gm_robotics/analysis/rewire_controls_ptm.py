"""PushToMarker random-trunk floor: refit the h_pre readout with scale-normalized features and a longer budget (all cohorts), plus a proprio-only readout."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from gm_robotics.analysis import common as C
from gm_robotics.analysis import envs_forced as E
from gm_robotics.analysis import rewire_ptm as RP
from gm_robotics.analysis.rewire_lr import Rewired, bank_features, train_bc

OUT = os.path.join(C.RESULTS, "rewire_controls_ptm")
SEEDS = dict(gm=[0, 1, 2], control=[0, 1, 2], init=[0, 1, 2])
# variant -> (epochs, h_pre normalisation, visual input used). "scalar" divides h_pre by its RMS over the bank (one number
# per trunk), "zscore" standardises every feature; both are folded into fc afterwards.
VARIANTS = {"paper": (20, None, True), "s20": (20, "scalar", True), "s100": (100, "scalar", True),
            "z100": (100, "zscore", True), "state100": (100, None, False)}
EVAL_SPECS = (("conflict", "standard"), ("ctrlset", "standard"))


class StateOnly(Rewired):
    """Same head as the h_pre rewire, but the visual half of its input is held at zero: a cube- and marker-blind
    readout of the proprio branch (tcp pose, qpos, qvel)."""

    def forward(self, h_pre, h_fc_native, h_state):
        h = torch.zeros(len(h_state), self.head[0].in_features - h_state.shape[1], device=h_state.device)
        return self.head(torch.cat([h, h_state], 1))


def head_path(cohort, k, variant):
    return os.path.join(OUT, "heads", f"{cohort}_s{k}_{variant}.pt")


def build_model(base, variant, k, dev):
    cls = StateOnly if variant == "state100" else Rewired
    return cls(base, "pre", init_seed=100 + k).to(dev)


def fold_standardization(model, mu, sd):
    """fc(z) with z = (h - mu) / sd equals fc'(h) with W' = W / sd, b' = b - W' mu, so the evaluated network is the
    ordinary h_pre rewire on raw features: the function class is unchanged, only optimisation conditioning differs."""
    with torch.no_grad():
        w = model.fc.weight / sd[None, :]
        model.fc.bias.sub_(w @ mu)
        model.fc.weight.copy_(w)


def train(a):
    dev = torch.device("cuda")
    os.makedirs(os.path.join(OUT, "heads"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "trainlogs"), exist_ok=True)
    bank = RP.load_bc()
    bank["rgb"] = np.ascontiguousarray(bank["rgb"])
    teachers = {}
    for cohort in a.cohorts:
        for k in a.seeds:
            todo = [v for v in a.variants if not os.path.exists(head_path(cohort, k, v))]
            if not todo:
                continue
            tk = RP.pair_teacher(k)
            if tk not in teachers:
                ta = C.build_agent(C.ckpt_path("PTM", "control", tk), dev=dev)
                teachers[tk] = (bank_features(ta, bank, dev)["mu"], C.sigma(ta))
                del ta
            target, sig_t = teachers[tk]
            base = RP.base_agent(cohort, k, dev)
            feats = bank_features(base, bank, dev)
            raw = feats["h_pre"]
            s1 = torch.zeros(raw.shape[1], dtype=torch.float64, device=dev)
            s2, absum, mx, nz, fin = torch.zeros_like(s1), 0.0, -np.inf, 0, True
            for s in range(0, len(raw), 8192):
                c = raw[s:s + 8192].double()
                s1 += c.sum(0)
                s2 += (c * c).sum(0)
                absum += float(c.abs().sum())
                mx = max(mx, float(c.max()))
                nz += int((c == 0).sum())
                fin &= bool(torch.isfinite(c).all())
            n = len(raw)
            stats = dict(h_pre_abs_mean=absum / raw.numel(), h_pre_max=mx, h_pre_frac_zero=nz / raw.numel(),
                         h_pre_dim=int(raw.shape[1]), h_pre_finite=fin)
            mu_z = (s1 / n).float()
            sd_z = (s2 / n - (s1 / n) ** 2).clamp_min(0).sqrt().float()
            sd_z = torch.where(sd_z > 1e-6, sd_z, torch.ones_like(sd_z))
            rms = float((s2.sum() / raw.numel()).sqrt())
            stats["h_pre_rms"] = rms
            norms = {"zscore": (mu_z, sd_z), "scalar": (torch.zeros_like(mu_z), torch.full_like(sd_z, rms))}
            for v in todo:
                epochs, norm, _ = VARIANTS[v]
                if norm:
                    mu, sd = norms[norm]
                    z = torch.empty_like(raw)
                    for s in range(0, len(raw), 8192):
                        z[s:s + 8192] = ((raw[s:s + 8192].float() - mu) / sd).half()
                    feats["h_pre"] = z
                else:
                    feats["h_pre"] = raw
                model = build_model(base, v, k, dev)
                log, n_par = train_bc(model, feats, target, sig_t, bank["episode"], epochs, a.lr, a.batch, a.clip, seed=k)
                if norm:
                    fold_standardization(model, mu, sd)
                sd_ = {n: p.detach().cpu() for n, p in model.state_dict().items() if not n.startswith("base.")}
                torch.save(dict(state_dict=sd_, variant=v, cohort=cohort, seed=k, teacher=tk, n_trainable=n_par),
                           head_path(cohort, k, v))
                pd.DataFrame(log).assign(n_trainable=n_par, teacher=tk, variant=v, cohort=cohort, seed=k, **stats).to_csv(
                    os.path.join(OUT, "trainlogs", f"{cohort}_s{k}_{v}.csv"), index=False)
                print(f"trained {cohort} s{k} {v}: val {log[-1]['val_loss']:.3f} train {log[-1]['train_loss']:.3f} "
                      f"min val {min(r['val_loss'] for r in log):.3f} | h_pre {stats}", flush=True)
                del model
                torch.cuda.empty_cache()
            del feats, raw
            torch.cuda.empty_cache()


def load_act(cohort, k, variant, dev):
    base = RP.base_agent(cohort, k, dev)
    if variant == "native":
        return lambda rgb, st: C.taps(base, rgb, st)["mu"]
    if variant.startswith("orig_"):
        model = Rewired(base, "pre", init_seed=100 + k).to(dev)
        ck = torch.load(RP.head_path(cohort, k, variant[5:]), map_location="cpu")
    else:
        model = build_model(base, variant, k, dev)
        ck = torch.load(head_path(cohort, k, variant), map_location="cpu")
    model.load_state_dict({**{n: p for n, p in model.state_dict().items() if n.startswith("base.")}, **ck["state_dict"]})
    model.eval()
    return model.act


def rollout(a):
    dev = torch.device("cuda")
    sets = {nm: RP.eval_scenes(nm) for nm in ("conflict", "ctrlset")}
    env = E.make_env("PTM", a.env_batch)
    csv = os.path.join(OUT, "per_episode.csv")
    done = set()
    if os.path.exists(csv):
        done = set(pd.read_csv(csv, usecols=["cohort", "seed", "variant", "set"]).itertuples(index=False, name=None))
    for item in a.roll:
        cohort, k, variant = item.split(":")
        k = int(k)
        act = None
        for sname, cond in EVAL_SPECS:
            if (cohort, k, variant, sname) in done:
                continue
            act = act or load_act(cohort, k, variant, dev)
            cfg, _ = sets[sname]
            df, _, _, _ = RP.run_eval(env, act, cfg, cond, a.env_batch)
            df = df.assign(cohort=cohort, seed=k, variant=variant, set=sname, cond=cond)
            df.to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            vc = df.outcome.value_counts(normalize=True)
            print(f"{cohort} s{k} {variant} {sname}: goal {vc.get('goal', 0):.3f} centre {vc.get('proxy', 0):.3f} "
                  f"neither {vc.get('neither', 0):.3f} cube moved {df.cube_moved.mean():.3f} m", flush=True)
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=["train", "rollout"])
    p.add_argument("--cohorts", nargs="+", default=["init", "gm", "control"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=["z100"], choices=list(VARIANTS))
    p.add_argument("--roll", nargs="+", default=[], help="cohort:seed:variant (variant in VARIANTS, native, orig_pre)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--clip", type=float, default=0.5)
    p.add_argument("--env_batch", type=int, default=128)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(train=train, rollout=rollout)[a.step](a)


if __name__ == "__main__":
    main()
