"""Config loading, config hash, git and version info, and the run record run.json (AC-9)."""
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import monai
import nibabel
import numpy
import scipy
import torch

# every key is present in every run.json (null where not yet known); spec 0001, run.json keys
RUN_KEYS = (
    "run_id", "status", "reason", "quick", "config_hash", "git_commit", "git_dirty", "seed",
    "data_dir", "device", "gpu", "versions", "determinism", "nondeterminism_warnings",
    "peak_vram_allocated_mib", "peak_vram_reserved_mib", "wall_time_s", "iterations",
    "scaler_skipped_steps", "loss_window", "loss_start_mean", "loss_end_mean", "val_wt_dice",
    "best_iteration", "train_cases", "val_cases",
)
STATUSES = ("running", "ok", "failed", "interrupted")


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def config_hash(cfg):
    """sha256 of the effective config serialised as sorted key JSON."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def dump_config(cfg, path):
    """Write a config of scalars and one level of tables back to TOML (stdlib has no writer)."""
    top = [f"{k} = {_toml_value(v)}" for k, v in cfg.items() if not isinstance(v, dict)]
    lines = top[:]
    for k, v in cfg.items():
        if isinstance(v, dict):
            lines += ["", f"[{k}]"] + [f"{kk} = {_toml_value(vv)}" for kk, vv in v.items()]
    Path(path).write_text("\n".join(lines) + "\n")
    assert load_config(path) == cfg, "config did not round trip through TOML"


def git_info():
    """(commit, dirty) of the repo this file lives in."""
    repo = Path(__file__).resolve().parents[2]
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True,
                                    check=True).stdout
    try:
        return run("rev-parse", "HEAD").strip(), bool(run("status", "--porcelain").strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def versions():
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "monai": monai.__version__,
        "nibabel": nibabel.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


class RunRecord:
    """run.json: written as `running` at start, rewritten at exit."""

    def __init__(self, run_dir, **fields):
        self.path = Path(run_dir) / "run.json"
        self.data = dict.fromkeys(RUN_KEYS)
        self.data.update(status="running", nondeterminism_warnings=[], val_wt_dice=[],
                         scaler_skipped_steps=0, iterations=0)
        self.update(**fields)
        self.write()

    def update(self, **fields):
        unknown = set(fields) - set(RUN_KEYS)
        assert not unknown, f"unknown run.json keys: {unknown}"
        assert fields.get("status", "running") in STATUSES
        self.data.update(fields)

    def write(self):
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)


def load_run(run_dir):
    return json.loads((Path(run_dir) / "run.json").read_text())
