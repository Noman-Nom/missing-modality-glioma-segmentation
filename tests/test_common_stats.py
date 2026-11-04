"""Spec 0004, AC-11: the statistics module on toy cases with known answers. The rest of AC-11
(friedman, bootstrap) lands with the full statistics milestone."""
import math

import numpy as np
import pytest
from common.stats import holm, paired_wilcoxon


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
