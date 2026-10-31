"""GPU speed check for E1 on a new machine (for example Kaggle), with random data, no BraTS.

    uv run python experiments/0003-baseline/gpu_bench.py              # 60 steps at the E1 patch
    uv run python experiments/0003-baseline/gpu_bench.py --steps 20   # shorter
Builds the E1 model from config.toml and runs the real training step (common.training.Trainer
.step: autocast, GradScaler, Dice plus BCE, AdamW, warmup and cosine lr) on random 4 x 128^3
batches, so only the data is fake. Prints the GPU, driver, CUDA, s/iter over the steps after
the first 10, peak VRAM and the projected training hours for [train] iterations (training only:
no val, no data loading). Not a result, a hardware check. Writes nothing.
Exit 0 ran on a GPU, 1 no usable GPU (or the torch build does not support it).
"""
import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runlog import load_config
from common.seed import set_seed
from common.training import Trainer

HERE = Path(__file__).resolve().parent
WARMUP_STEPS = 10  # excluded from s/iter (cuDNN set up, allocator growth)


def random_batches(cfg, seed=0):
    """Endless batches shaped like the loader's: case, image (B, 4, P), label (B, 3, P), wt."""
    t = cfg["train"]
    b, p = t["batch_size"], t["patch"]
    g = torch.Generator().manual_seed(seed)
    while True:
        label = (torch.rand(b, 3, *p, generator=g) > 0.9).float()
        yield {"case": ["random"] * b, "image": torch.randn(b, 4, *p, generator=g),
               "label": label, "wt": label[:, :1].clone()}


def driver_version():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=HERE / "config.toml")
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args(argv)

    print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}, "
          f"driver {driver_version()}")
    if not torch.cuda.is_available():
        print("FAIL: no usable GPU (torch.cuda.is_available() is False)")
        return 1
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    print(f"GPU {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]}), "
          f"{torch.cuda.device_count()} visible, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        print(f"FAIL: this torch build has no kernels for sm_{cap[0]}{cap[1]} "
              f"(it has {', '.join(torch.cuda.get_arch_list())})")
        return 1

    cfg = load_config(a.config)
    t = cfg["train"]
    gen = set_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    # the real loader is built over a placeholder case but never read; random batches replace it
    tr = Trainer(cfg, device, ["random"], gen)
    tr.batches = random_batches(cfg)
    print(f"model {tr.n_params:,} params, patch {t['patch']}, batch {t['batch_size']}, "
          f"amp {tr.amp}; {a.steps} steps")
    steps = []
    for i in range(1, a.steps + 1):
        steps.append(tr.step(i))
        if i % 10 == 0:
            print(f"  step {i}: {steps[-1]['seconds']:.3f} s, loss {steps[-1]['loss']:.4f}")
    timed = steps[WARMUP_STEPS:] or steps
    s_it = statistics.mean(s["seconds"] for s in timed)
    print(f"peak VRAM reserved {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB, "
          f"allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"{s_it:.3f} s/iter -> {s_it * t['iterations'] / 3600:.1f} h for {t['iterations']} "
          "iterations (training only; val and data loading come on top)")
    print("reference: Quadro T2000 Max-Q laptop, this script, 30 steps: 0.764 s/iter, 8.5 h, "
          "peak reserved 898 MiB (2026-10-31)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
