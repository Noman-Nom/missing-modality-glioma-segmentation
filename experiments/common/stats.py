"""The statistics protocol of spec 0004, fixed before any test number (AC-11).

The tests are paired over subjects. Every rule here is the one in the spec, so E3 and E4 are
scored by the same code. SciPy has no Holm function and statsmodels is not in uv.lock.
"""
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


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


def friedman(matrix):
    """Friedman test on an n x k matrix (n subjects, k patterns). Returns (stat, p, kendall_w).

    Kendall W = stat / (n (k - 1)), an effect size from 0 to 1. All rows fully tied would give NaN
    in SciPy, so that case is coded as statistic 0, p 1, W 0.
    """
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    if (m == m[:, :1]).all():
        return 0.0, 1.0, 0.0
    r = friedmanchisquare(*m.T)
    return float(r.statistic), float(r.pvalue), float(r.statistic) / (n * (k - 1))


def bootstrap_indices(n, resamples, seed):
    """One integer index matrix (resamples, n). Reused for every row, so all rows are paired."""
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def bootstrap_ci(values, indices, level=0.95):
    """Percentile bootstrap CI of the mean of values, over the subjects. Returns (low, high)."""
    means = np.asarray(values, dtype=float)[indices].mean(axis=1)
    tail = (100 - 100 * level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)
