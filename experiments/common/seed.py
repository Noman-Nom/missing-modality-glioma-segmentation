"""One place that sets every seed and the determinism flags (spec 0001, AC-10).

Claim: seeded, with nondeterministic ops logged (warn_only), not bitwise GPU reproducibility.
"""
import os
import random

import monai
import numpy as np
import torch

CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_seed(seed):
    """Seed Python, NumPy, torch and MONAI, set the determinism flags.

    Returns a seeded torch.Generator for the DataLoader.
    """
    # must be set before the first cuBLAS call
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    monai.utils.set_determinism(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def determinism_state():
    """The determinism settings in force now, for run.json."""
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
