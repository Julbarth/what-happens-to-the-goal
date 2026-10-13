"""PushToMarker probes with rarely-active post-ReLU units dropped, so a few test outliers cannot dominate after standardising."""
import argparse
import os

import numpy as np
import pandas as pd
from gm_robotics.analysis import common as C
from gm_robotics.analysis import probes as PR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min_active", type=float, default=0.01)
    p.add_argument("--init_seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    dev = C.device()
    tr, te = C.load_bank("PTM_probe_train"), C.load_bank("PTM_probe_test")
    ttr, tte = PR.targets(tr, "PTM"), PR.targets(te, "PTM")
    raw_dir = os.path.join(C.RESULTS, "probe_raw")
    dfs = [pd.read_csv(os.path.join(raw_dir, "PTM_input_controls.csv"))]
    for cohort, seed, ck in C.agent_list("PTM", None, None, a.init_seeds):
        out = os.path.join(raw_dir, f"PTMrobust_{cohort}_s{seed}.csv")
        if not os.path.exists(out):
            agent = C.build_agent(ck, init_seed=seed, dev=dev)
            ftr, fte = PR.features(agent, tr, dev), PR.features(agent, te, dev)
            nkeep = {}
            for s in ftr:
                keep = (np.abs(ftr[s]) > 1e-6).mean(0) >= a.min_active
                ftr[s], fte[s], nkeep[s] = ftr[s][:, keep], fte[s][:, keep], int(keep.sum())
            rows = PR.run_feature_set(dict(cohort=cohort, seed=seed), ftr, fte, ttr, tte)
            df = pd.DataFrame(rows)
            df["n_units_kept"] = df.site.map(nkeep)
            df.to_csv(out, index=False)
        df = pd.read_csv(out)
        print(cohort, seed, df[df.target.isin(["marker_x", "marker_y"])].pivot_table(index="site", columns="target", values="value").round(3).to_dict(), flush=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(C.RESULTS, "probes_PTMrobust_per_seed.csv"), index=False)


if __name__ == "__main__":
    main()
