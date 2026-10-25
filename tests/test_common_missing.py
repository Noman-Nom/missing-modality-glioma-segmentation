"""Spec 0001, AC-4: the 15 missing modality patterns, pinned to the table in the spec."""
import numpy as np
import torch
from common.missing import PATTERNS, apply_pattern

# (ID, present, bitmask) exactly as the spec 0001 pattern list
TABLE = [
    (1, "T1+T1ce+T2+FLAIR", 15), (2, "T1ce+T2+FLAIR", 14), (3, "T1+T2+FLAIR", 13),
    (4, "T1+T1ce+FLAIR", 11), (5, "T1+T1ce+T2", 7), (6, "T2+FLAIR", 12), (7, "T1ce+FLAIR", 10),
    (8, "T1+FLAIR", 9), (9, "T1ce+T2", 6), (10, "T1+T2", 5), (11, "T1+T1ce", 3), (12, "FLAIR", 8),
    (13, "T2", 4), (14, "T1ce", 2), (15, "T1", 1),
]


def test_patterns_match_spec_table():
    assert len(PATTERNS) == 15
    assert [(p.id, p.name, p.mask) for p in PATTERNS.values()] == TABLE
    assert list(PATTERNS) == list(range(1, 16))


def test_apply_pattern_zeros_only_missing_channels():
    x = np.arange(1, 5, dtype=np.float32)[:, None, None, None] * np.ones((4, 2, 2, 2), np.float32)
    out = apply_pattern(x, 7)  # T1ce+FLAIR present
    assert [float(out[c].max()) for c in range(4)] == [0.0, 2.0, 0.0, 4.0]
    assert float(x[0].max()) == 1.0  # input untouched
    assert np.array_equal(apply_pattern(x, 1), x)


def test_apply_pattern_torch_batch_axis():
    x = torch.ones(1, 4, 2, 2, 2)
    out = apply_pattern(x, 15, channel_axis=1)  # T1 only
    assert out[0, :, 0, 0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert x.sum() == 32
