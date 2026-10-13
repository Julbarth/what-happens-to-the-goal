"""95% t-interval half-widths over seeds without a scipy dependency."""
import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def ci95(v, axis=0):
    v = np.asarray(v, float)
    n = v.shape[axis]
    if n < 2:
        return v.mean(axis), np.full(np.shape(v.mean(axis)), np.nan)
    return v.mean(axis), T975.get(n - 1, 1.96) * v.std(axis=axis, ddof=1) / np.sqrt(n)
