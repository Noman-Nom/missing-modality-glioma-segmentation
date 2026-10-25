"""The 15 missing modality patterns, fixed before any test result (spec 0001, AC-4).

Bit order T1 = 1, T1ce = 2, T2 = 4, FLAIR = 8. IDs 1 to 15: most present modalities first, then
bitmask descending. A missing channel is all zeros after normalisation (zero = brain mean).
"""
from typing import NamedTuple

from common.data import MODALITIES

BIT = {m: 1 << i for i, m in enumerate(MODALITIES)}


class Pattern(NamedTuple):
    id: int
    mask: int
    present: tuple
    missing: tuple
    name: str


def _make():
    masks = sorted(range(1, 16), key=lambda m: (-m.bit_count(), -m))
    out = {}
    for i, mask in enumerate(masks, start=1):
        present = tuple(m for m in MODALITIES if mask & BIT[m])
        missing = tuple(m for m in MODALITIES if not mask & BIT[m])
        out[i] = Pattern(i, mask, present, missing, "+".join(present))
    return out


PATTERNS = _make()


def apply_pattern(image, pattern_id, channel_axis=0):
    """Copy of image (NumPy or torch) with the pattern's missing channels set to 0."""
    out = image.copy() if hasattr(image, "copy") else image.clone()
    idx = [slice(None)] * out.ndim
    for m in PATTERNS[pattern_id].missing:
        idx[channel_axis] = MODALITIES.index(m)
        out[tuple(idx)] = 0
    return out
