"""Cross-env depth tables: decodability vs rewiring by depth for LR, PTM and the paper's Maze, with chance-corrected recovery."""
import glob
import json
import os

import numpy as np
import pandas as pd

from gm_robotics.analysis import depth_common as D

RA = os.path.join(D.ROOT, "results", "analysis")
MAZE = os.path.join(D.GMGEO, "data", "probes", "rewire_aligned_20260922")


def maze_rewire():
    """Per-seed closed-loop cheese reach of every Maze rewire arm (paper Table tab:rewire-main / tab:rewire-remix)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(MAZE, "maze_aligned_*.json"))):
        d = json.load(open(f))
        for key, r in d["rows"].items():
            rows.append(dict(src=os.path.basename(f), arm=r["arm"], cohort=r["cohort"], seed=r["seed"], tap=r["tap"],
                             labels=r["label_source"], reach=r["rewired_eval"]["cheese_reach"],
                             corner=r["rewired_eval"]["gm_rate"], native=r["native_eval"]["cheese_reach"],
                             native_corner=r["native_eval"]["gm_rate"]))
    fl = json.load(open(os.path.join(MAZE, "untrained_floor_maze.json")))
    for key, r in (fl["rows"] if "rows" in fl else fl).items():
        rows.append(dict(src="untrained_floor_maze.json", arm=f"untrained_{r['action_mode']}", cohort=r["cohort"],
                         seed=r["seed"], tap=r["tap"], labels="none", reach=r["R"]))
    return pd.DataFrame(rows)


def probe_cells(env):
    d = pd.read_csv(os.path.join(RA, "probe_audit", f"{env}_per_seed.csv"))
    d = d[(d.subset == "all") & d.probe.isin(["ridge_sel", "mlp", "knn"]) & d.site.isin(["conv1", "conv2", "h_pre", "h_fc"])]
    tg = {"LR": ["marker_x", "marker_y", "bearing_x", "bearing_y", "marker_left", "tcp_x", "tcp_y"],
          "PTM": ["marker_x", "marker_y", "bearing_x", "bearing_y", "marker_left", "marker_rel_cube_x", "cube_x",
                  "tcp_x", "tcp_y"]}[env]
    return d[d.target.isin(tg)]


def main():
    out = D.ensure_out()
    L = ["# Cross-env depth tables (Q1, Q5, Q6)", "", "Mean [95% t-CI] over seeds; GM - init and GM - Non-GM are Welch intervals.", ""]

    # --- probes, LR and PTM
    L += ["## Goal decodability by depth (probe audit, held-out probe bank, start frames)", "",
          "xy = mean of the x and y R2 per seed. side = marker_left AUC.", "",
          "| env | probe | target | site | GM | Non-GM | init | GM - init |", "|---|---|---|---|---|---|---|---|"]
    prow = []
    for env in ("LR", "PTM"):
        d = probe_cells(env)
        piv = d.pivot_table(index=["probe", "site", "cohort", "seed"], columns="target", values="value").reset_index()
        piv["marker_xy"] = piv[["marker_x", "marker_y"]].mean(1)
        piv["bearing_xy"] = piv[["bearing_x", "bearing_y"]].mean(1)
        piv["tcp_xy"] = piv[["tcp_x", "tcp_y"]].mean(1)
        for (pr, site), g in piv.groupby(["probe", "site"]):
            for tg in ("marker_xy", "bearing_xy", "marker_left", "tcp_xy"):
                v = {c: g[g.cohort == c][tg].to_numpy() for c in ("gm", "control", "init")}
                prow.append(dict(env=env, probe=pr, site=site, target=tg, **{f"{c}_mean": v[c].mean() for c in v},
                                 gm_minus_init=D.welch(v["gm"], v["init"])[0]))
                if pr in ("ridge_sel", "mlp") and site in ("h_pre", "h_fc", "conv1"):
                    L.append(f"| {env} | {pr} | {tg} | {site} | {D.fmt_ci(v['gm'])} | {D.fmt_ci(v['control'])} | "
                             f"{D.fmt_ci(v['init'])} | {D.fmt(*D.welch(v['gm'], v['init']))} |")
    pd.DataFrame(prow).to_csv(os.path.join(out, "q1_probe_depth_LR_PTM.csv"), index=False)

    # --- Maze (paper) probe controls at h_pre, from tab:probe-controls (mean +- SD over 3 checkpoints; no per-seed file)
    L += ["", "Paper Maze, h_pre (Table tab:probe-controls, mean +- SD, AUC): cheese presence GM 0.544 +- 0.01 vs init "
          "0.620 +- 0.02 (Non-GM 0.997); cheese bearing 0.712 vs init 0.686 (agent position alone 0.830); bearing with "
          "corner fixed 0.598 vs init 0.729 (position 0.703). So in Maze, as in LR, GM goal decodability at h_pre is at "
          "or below a random network.", ""]

    # --- rewiring, robots
    ra = pd.read_csv(os.path.join(RA, "rewire_all_per_seed.csv"))
    L += ["## Rewiring from h_pre (final protocols) and chance-corrected recovery", "",
          "norm = (GM - chance) / (mean Non-GM - chance) per GM seed; GM - init excess Welch.", "",
          "| task | GM h_pre | Non-GM h_pre | init h_pre | chance | GM - init | norm recovery GM | norm recovery init |",
          "|---|---|---|---|---|---|---|---|"]
    rrows = []
    for task, g in ra.groupby("task", sort=False):
        v = lambda k: g[g.key == k].sort_values("seed").value.to_numpy()
        gm, ng, ini, ch = v("pre_gm"), v("pre_control"), v("pre_init"), v("chance")
        if len(ch) == 0 or not np.isfinite(ch).any():
            ch = np.zeros_like(gm)
        nrm = (gm - ch) / (ng.mean() - ch)
        nri = (ini - ch.mean()) / (ng.mean() - ch.mean())
        rrows.append(dict(task=task, gm=gm.mean(), ng=ng.mean(), init=ini.mean(), chance=np.nanmean(ch),
                          norm_gm=nrm.mean(), norm_init=nri.mean()))
        L.append(f"| {task} | {D.fmt_ci(gm)} | {D.fmt_ci(ng)} | {D.fmt_ci(ini)} | {np.nanmean(ch):.3f} | "
                 f"{D.fmt(*D.welch(gm, ini))} | {D.fmt_ci(nrm)} | {D.fmt_ci(nri)} |")

    # --- rewiring, Maze
    mz = maze_rewire()
    mz.to_csv(os.path.join(out, "q1_maze_rewire_per_seed.csv"), index=False)
    L += ["", "### Paper Maze rewiring (data/probes/rewire_aligned_20260922, per seed)", "",
          "| labels | site | GM | Non-GM | GM remix | Non-GM remix | GM corner rate after rewire |", "|---|---|---|---|---|---|---|"]
    for lab in ("a_star", "teacher"):
        for tap in ("block3", "fc"):
            s = mz[(mz.labels == lab) & (mz.tap == tap)]
            gm = s[(s.arm == "rewire") & (s.cohort == "gm")].reach.to_numpy()
            ng = s[(s.arm == "rewire") & (s.cohort == "nongm")].reach.to_numpy()
            rg = s[(s.arm == "random_proj") & (s.cohort == "gm")].reach.to_numpy()
            rn = s[(s.arm == "random_proj") & (s.cohort == "nongm")].reach.to_numpy()
            cr = s[(s.arm == "rewire") & (s.cohort == "gm")].corner.to_numpy()
            L.append(f"| {lab} | {'h_pre' if tap == 'block3' else 'h_fc'} | {D.fmt_ci(gm)} | {D.fmt_ci(ng)} | "
                     f"{D.fmt_ci(rg) if len(rg) else '-'} | {D.fmt_ci(rn) if len(rn) else '-'} | {D.fmt_ci(cr)} |")
    fl = mz[mz.arm == "untrained_sample"]
    nat = mz[(mz.arm == "rewire") & (mz.labels == "a_star") & (mz.tap == "block3")]
    L += ["", f"Maze sampled untrained floor (h_pre) {D.fmt_ci(fl[fl.tap == 'block3'].reach)}; native GM "
              f"{D.fmt_ci(nat[nat.cohort == 'gm'].native)}, native Non-GM {D.fmt_ci(nat[nat.cohort == 'nongm'].native)}; "
              f"native GM corner rate {D.fmt_ci(nat[nat.cohort == 'gm'].native_corner)}.", ""]
    s = mz[(mz.labels == "a_star") & (mz.tap == "block3")]
    gm = s[(s.arm == "rewire") & (s.cohort == "gm")].sort_values("seed").reach.to_numpy()
    rg = s[(s.arm == "random_proj") & (s.cohort == "gm")].sort_values("seed").reach.to_numpy()
    ng = s[(s.arm == "rewire") & (s.cohort == "nongm")].reach.to_numpy()
    rn = s[(s.arm == "random_proj") & (s.cohort == "nongm")].reach.to_numpy()
    flv = fl[fl.tap == "block3"].reach.mean()
    for nm, floor in (("GM remix (per seed)", rg), ("sampled untrained floor", np.full(3, flv))):
        nrm = (gm - floor) / (ng.mean() - floor)
        L.append(f"Maze oracle h_pre normalised recovery with floor = {nm}: {D.fmt_ci(nrm)}")
    L.append(f"Maze oracle h_pre GM / Non-GM ratio {gm.mean() / ng.mean():.2f}; LR {rrows[[r['task'] for r in rrows].index('LR')]['gm'] / rrows[[r['task'] for r in rrows].index('LR')]['ng']:.2f}.")
    L.append(f"Maze learned-map gain over remix (oracle h_pre): GM {D.fmt_ci(gm - rg)}, Non-GM {D.fmt_ci(ng - rn)}.")
    # what an LR-sized normalised recovery would predict in Maze
    lr = [r for r in rrows if r["task"] == "LR"][0]
    pred = rg.mean() + lr["norm_gm"] * (ng.mean() - rg.mean())
    L.append(f"An LR-sized normalised recovery ({lr['norm_gm']:.2f}) would predict Maze GM h_pre reach "
             f"{pred:.3f} (remix floor); observed {D.fmt_ci(gm)}.")

    # --- per-layer rewiring (depth table)
    dp = pd.read_csv(os.path.join(RA, "depth", "depth_per_seed.csv"))
    L += ["", "## Rewired goal reach by layer (depth_per_seed.csv; AT / SZ = ctrl2 protocol)", "",
          "| env | site | GM | Non-GM | init | GM - init |", "|---|---|---|---|---|---|"]
    for env in ("LR", "PTM", "AT", "SZ"):
        for site in ("conv1", "conv2", "h_pre", "h_fc"):
            g = dp[(dp.env == env) & (dp.measure == "rewire_goal") & (dp.site == site)]
            v = {c: g[g.cohort == c].value.to_numpy() for c in ("gm", "control", "init")}
            if not all(len(x) > 1 for x in v.values()):
                continue
            L.append(f"| {env} | {site} | {D.fmt_ci(v['gm'])} | {D.fmt_ci(v['control'])} | {D.fmt_ci(v['init'])} | "
                     f"{D.fmt(*D.welch(v['gm'], v['init']))} |")
    pd.DataFrame(rrows).to_csv(os.path.join(out, "q1_rewire_normalised.csv"), index=False)
    with open(os.path.join(out, "q1_crossenv_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
