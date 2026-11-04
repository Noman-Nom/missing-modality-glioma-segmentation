"""The statistics protocol of spec 0004, fixed before any test number (AC-11).

The tests are paired over subjects. Every rule here is the one in the spec, so E3 and E4 are
scored by the same code. SciPy has no Holm function and statsmodels is not in uv.lock.
"""
import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(d):
    """Two sided signed rank test of the paired differences d. Returns (stat, p).

    Zero differences are dropped (zero_method wilcox). All d zero gives statistic 0, p 1.
    """
    d = np.asarray(d, dtype=float)
    if not d.any():
        return 0.0, 1.0
    r = wilcoxon(d, zero_method="wilcox", correction=False, alternative="two-sided",
                 method="asymptotic")
    return float(r.statistic), float(r.pvalue)


def holm(p):
    """Holm adjusted p values, in the order given, monotone in rank and capped at 1."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1.0)
    out = np.empty(len(p))
    out[order] = adjusted
    return out
