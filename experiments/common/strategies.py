"""The missing modality training strategies, selected by [strategy] in a config (spec 0005).

A config with no [strategy] is E1. "dropout" is E3: every sample trains on a random pattern.
"presence" is E4: the same, plus 4 channels saying which modalities are present. All 15 patterns
have equal chance, complete input included, and there is no parameter to tune. Reads no case.
"""
import numpy as np
import torch

from common.data import MODALITIES
from common.missing import PATTERNS, apply_pattern

STRATEGIES = ("dropout", "presence")


def strategy_name(cfg):
    """The config's strategy, or None for E1."""
    return cfg.get("strategy", {}).get("name")


def strategy_problems(cfg):
    """Problems that must stop a run before it starts, as strings; empty = fine (AC-6)."""
    name = strategy_name(cfg)
    if "strategy" in cfg and name not in STRATEGIES:
        return [f"[strategy] name must be one of {STRATEGIES}, got {name!r}"]
    return []


def draw_patterns(rng, n):
    """n pattern IDs, each of the 15 with equal chance (AC-2)."""
    return rng.integers(1, len(PATTERNS) + 1, size=n)


def keep_batch(batch, rng):
    """The E1 batch hook: changes nothing and draws nothing."""
    return batch


def keep_image(image, pattern_id):
    """The E1 and E3 image hook: no change."""
    return image


def append_presence(image, pattern_id):
    """(4, X, Y, Z) NumPy or torch -> (8, X, Y, Z) float32 of the same kind (AC-4).

    The 4 image channels, then in MODALITIES order one constant map per modality: 1 if the pattern
    holds it, 0 if not.
    """
    flags = [float(m in PATTERNS[pattern_id].present) for m in MODALITIES]
    shape = (-1, *[1] * (image.ndim - 1))
    if isinstance(image, torch.Tensor):
        maps = torch.tensor(flags, dtype=torch.float32, device=image.device).view(shape)
        return torch.cat([image.to(torch.float32), maps.expand(-1, *image.shape[1:])], dim=0)
    maps = np.array(flags, dtype=np.float32).reshape(shape)
    return np.concatenate([image.astype(np.float32), np.broadcast_to(maps, image.shape)], axis=0)


presence_image_hook = append_presence


def _drop(batch, rng):
    """The batch's image with one drawn pattern applied per sample, and the drawn IDs. Per sample,
    because apply_pattern() takes one (4, X, Y, Z) image, not a batch (AC-3)."""
    image = batch["image"]
    ids = [int(p) for p in draw_patterns(rng, image.shape[0])]
    out = image.clone()
    for i, p in enumerate(ids):
        out[i] = apply_pattern(image[i], p)
    return out, ids


def dropout_hook(batch, rng):
    """The E3 batch hook: each sample's missing channels set to exactly 0 (AC-3)."""
    image, _ = _drop(batch, rng)
    return {**batch, "image": image}


def presence_hook(batch, rng):
    """The E4 batch hook: dropout_hook, then the presence channels of each sample's own draw (AC-4)."""
    image, ids = _drop(batch, rng)
    return {**batch, "image": torch.stack([append_presence(x, p) for x, p in zip(image, ids)])}


def strategy_hooks(cfg):
    """{"batch_hook", "image_hook"} for the config's strategy (AC-6)."""
    name = strategy_name(cfg)
    if name is None:
        return {"batch_hook": keep_batch, "image_hook": keep_image}
    return {"dropout": {"batch_hook": dropout_hook, "image_hook": keep_image},
            "presence": {"batch_hook": presence_hook, "image_hook": presence_image_hook}}[name]
