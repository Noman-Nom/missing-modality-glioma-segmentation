"""Spec 0004, AC-11: the statistics module on toy cases with known answers."""
import math

import numpy as np
import pytest
from common.stats import bootstrap_ci, bootstrap_indices, friedman, holm, paired_wilcoxon


# covers: AC-11. Hand worked: ranks 1 to 10, only the 10 is negative, so T- = 10, T+ = 45.
# The null has mean n(n+1)/4 = 27.5 and variance n(n+1)(2n+1)/24 = 96.25.
def test_wilcoxon_matches_the_hand_worked_normal_approximation():
    stat, p = paired_wilcoxon([1, 2, 3, 4, 5, 6, 7, 8, 9, -10])
    z = (10 - 27.5) / math.sqrt(96.25)
    assert stat == 10
    assert p == pytest.approx(math.erfc(abs(z) / math.sqrt(2)))


# covers: AC-11
def test_wilcoxon_all_zero_differences_gives_statistic_0_p_1():
    assert paired_wilcoxon(np.zeros(5)) == (0.0, 1.0)


# covers: AC-11. zero_method wilcox drops zero differences before ranking.
def test_wilcoxon_drops_zero_differences():
    d = [1, 2, 3, 4, 5, 6, 7, 8, 9, -10]
    assert paired_wilcoxon(d + [0, 0, 0]) == paired_wilcoxon(d)


# covers: AC-11. Textbook: sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 gives 0.02, 0.03, 0.06,
# 0.04, and the last is raised to 0.06 to stay monotone. Results keep the input order.
def test_holm_on_a_textbook_example():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


# covers: AC-11
def test_holm_is_capped_at_one():
    assert holm([0.5, 0.6, 0.9]) == pytest.approx([1.0, 1.0, 1.0])


# covers: AC-11
def test_holm_equal_p_values_stay_equal_and_one_test_is_unchanged():
    assert holm([0.02, 0.02]) == pytest.approx([0.04, 0.04])
    assert holm([0.03]) == pytest.approx([0.03])


# covers: AC-11. Hand worked: 4 subjects rank 3 patterns the same way, so the statistic is its
# largest value n(k-1) = 8, W = 1, and chi square with 2 degrees of freedom gives p = exp(-8/2).
def test_friedman_full_agreement_has_the_known_answer():
    stat, p, w = friedman([[1, 2, 3], [2, 3, 4], [0, 5, 6], [7, 8, 9]])
    assert stat == pytest.approx(8)
    assert p == pytest.approx(math.exp(-4))
    assert w == pytest.approx(1)


# covers: AC-11. Hand worked: the two rows rank opposite ways, so mean ranks are equal.
def test_friedman_opposite_rows_give_statistic_0():
    stat, p, w = friedman([[1, 2, 3], [3, 2, 1]])
    assert (stat, w) == pytest.approx((0, 0), abs=1e-12)
    assert p == pytest.approx(1)


# covers: AC-11. SciPy would give NaN here, because every row is fully tied.
def test_friedman_fully_tied_rows_are_coded_0_1_0():
    assert friedman([[0.5, 0.5, 0.5], [0.9, 0.9, 0.9]]) == (0.0, 1.0, 0.0)


# covers: AC-11. One tied row among untied rows is a normal test, not the coded case.
def test_friedman_one_tied_row_is_still_tested():
    stat, _, w = friedman([[1, 1, 1], [1, 2, 3], [1, 2, 3]])
    assert stat > 0 and 0 < w < 1


# covers: AC-11
def test_bootstrap_indices_shape_range_and_seed():
    a = bootstrap_indices(5, 100, 42)
    assert a.shape == (100, 5) and a.min() >= 0 and a.max() < 5
    assert np.issubdtype(a.dtype, np.integer)
    assert (a == bootstrap_indices(5, 100, 42)).all()
    assert (a != bootstrap_indices(5, 100, 43)).any()


# covers: AC-11. The spec fixes default_rng(seed) as the source of the matrix.
def test_bootstrap_indices_come_from_default_rng():
    want = np.random.default_rng(7).integers(0, 3, size=(4, 3))
    assert (bootstrap_indices(3, 4, 7) == want).all()


# covers: AC-11
def test_bootstrap_ci_on_a_constant_is_that_constant():
    assert bootstrap_ci([0.7] * 6, bootstrap_indices(6, 500, 1)) == pytest.approx((0.7, 0.7))


# covers: AC-11. Hand worked: 5 resamples give means 0, 0.25, 0.5, 0.75, 1, and linear percentiles
# at 2.5 and 97.5 sit a tenth of a step in from each end, 0.025 and 0.975.
def test_bootstrap_ci_is_the_linear_percentile_of_the_resample_means():
    values = [0.0, 1.0]
    indices = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
    assert bootstrap_ci(values, indices) == pytest.approx((0.025, 0.975))
