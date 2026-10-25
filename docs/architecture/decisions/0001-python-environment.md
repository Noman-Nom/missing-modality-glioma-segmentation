# 0001. Use uv with a lockfile, PyTorch (CUDA 13.0 wheel), MONAI and NiBabel

**Date**: 2026-10-20
**Status**: Accepted (AC-2 and AC-5 done 2026-10-20; AC-3 and AC-4 done 2026-10-21 by `tests/test_env_smoke.py`; AC-1 `uv sync --frozen` rebuild not yet checked)

## Summary

We build the Python environment with uv, which writes a lockfile (`uv.lock`) so every run uses the same package versions. We use PyTorch, MONAI and NiBabel from the normal PyPI index. The default PyTorch wheel there is built for CUDA 13.0, which matches the laptop driver (580, CUDA 13.0). The virtual environment lives outside the repo, on the Linux disk, because the repo is on an NTFS drive.

## Context

Milestone 2 needs one environment for training a compact 3D U-Net, reading NIfTI files and computing metrics. It must be reproducible (AGENTS.md: fixed seed, logged config) and must fit a Quadro T2000 Max-Q: 4096 MiB VRAM, compute capability 7.5 (Turing), driver 580.178.04 reporting CUDA 13.0. System RAM is 15 GB, with about 5 GB free at check time. Python on the machine is 3.12.3. `uv` is not installed yet.

Facts checked on 2026-10-20 from PyPI metadata (not yet from a real install):

| Package | Latest | Needs |
|---|---|---|
| torch | 2.14.1 | Python >= 3.10; pulls `cuda-toolkit==13.0.3`, `nvidia-cudnn-cu13` on Linux |
| monai | 1.6.1 (2026-09-27) | Python >= 3.10; `torch>=2.8.0`, `numpy>=1.24,<3` |
| nibabel | 5.4.2 (2026-03-11) | Python >= 3.10; `numpy>=1.25` |

PyTorch announced that CUDA 13.0 is the stable variant on PyPI from release 2.11, so a plain `uv add torch` gets a CUDA 13.0 build. A CUDA 13.0 runtime needs a driver at CUDA 13.0 or newer; ours reports exactly 13.0. The repo is on an NTFS drive (`fuseblk`), where symlinks, hardlinks and the executable bit are unreliable, so a virtual environment inside the repo may break.

## Requirements

**Acceptance criteria**:
- **AC-1**: `pyproject.toml` and `uv.lock` are committed; `uv sync --frozen` rebuilds the same environment on a clean checkout.
- **AC-2**: `torch.cuda.is_available()` is true and `torch.cuda.get_arch_list()` contains `sm_75`.
- **AC-3**: A tiny forward and backward pass of a MONAI 3D U-Net with mixed precision runs on the GPU, and peak memory is printed and stays under 4 GB.
- **AC-4**: NiBabel loads one BraTS 2023 case and MONAI's Dice and HD95 metrics run on a toy case with a known answer.
- **AC-5**: Resolved versions are recorded in `docs/whats-built.md` after the first install.

## Options considered

### Option 1: uv, PyPI default torch (CUDA 13.0), venv outside the repo
One `pyproject.toml`, one `uv.lock`, torch from PyPI with no custom index.

**Pros**:
- Simplest setup; no extra index to keep in sync.
- Matches the driver CUDA version.
- Lockfile gives exact, hash-checked versions.

**Cons**:
- Large download (several GB, CUDA libraries are bundled as packages).
- Turing support in the CUDA 13.0 wheel is reported by PyTorch notices but not yet confirmed on this machine (AC-2 tests it).

### Option 2: uv with the PyTorch CUDA 12.6 index
Pin torch from `download.pytorch.org/whl/cu126` in `[tool.uv.sources]`.

**Pros**:
- Long-standing build with Turing support; works with the 580 driver (newer drivers run older CUDA runtimes).

**Cons**:
- A second package index to manage; PyTorch announced that CUDA 12.6 wheels stop at release 2.15, so this path ages out.
- Older line, so it may lag MONAI's tested PyTorch versions.

### Option 3: conda or pip with a requirements file
**Pros**:
- Familiar.

**Cons**:
- No cross-platform lockfile with hashes in plain pip; conda is a new tool and heavier. Weaker reproducibility for a thesis that must be re-run.

## Decision

**Chosen option**: Option 1: uv, PyPI default torch (CUDA 13.0), venv outside the repo.

Option 2 is the documented fallback if AC-2 fails (no `sm_75` in the arch list).

Environment rules:
- Python 3.12, pinned in `pyproject.toml` (`requires-python = ">=3.12,<3.13"`) and `.python-version`.
- Direct dependencies: `torch`, `monai`, `nibabel`. Dev group: `pytest`, `ruff`. `scipy` was added 2026-10-21 because MONAI's HD95 needs `scipy.ndimage` (smoke test showed it). `pandas` was added 2026-10-25 because the audit and split scripts (`experiments/0000-data-audit/`, `experiments/0001-splits/`) need it (spec 0002).
- No `torchvision`, `torchaudio`, `monai[all]` or other extras. Each extra needs its own reason.
- The venv path is set with `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/qarar-research` (ext4), and `UV_LINK_MODE=copy`, because the repo drive is NTFS.
- Commands are run with `uv run`, so the lockfile is always honoured.
- Determinism: seeds are set in `experiments/common/`; the reproducibility notes (cuDNN determinism flags, which ops stay non-deterministic) are decided in the tracer-bullet spec, not here.
- Memory plan for 4 GB (starting values, tuned in the tracer bullet, not fixed here): mixed precision (`torch.autocast`), batch size 1 to 2, patches of about 96^3 or smaller, a small base width (16 channels), sliding-window inference with a small `sw_batch_size`.

## Rationale

The driver already reports CUDA 13.0, and the default PyPI torch is a CUDA 13.0 build, so Option 1 needs no custom index and has the fewest moving parts. uv gives a hashed lockfile, which is what "every number is produced by a script with a logged config" needs: someone can rebuild the same environment from `uv.lock`. Conda adds a tool with no gain here.

The one open risk is whether the CUDA 13.0 wheel includes kernels for the T2000's compute capability 7.5. PyTorch notices say Turing remains supported on CUDA 13.x, but one forum post disagrees, and we have not tested it. We make that an explicit check (AC-2) before anything else is built, with Option 2 ready as the fallback.

## Consequences

**Positive**:
- Reproducible environment from one lockfile.
- No custom package index.

**Negative / tradeoffs**:
- First install is a multi-GB download.
- The venv is outside the repo, so a new machine needs the environment variable set (documented below).
- Package versions will move; the lockfile freezes them, and upgrades need a deliberate `uv lock --upgrade` and a re-run of the smoke test.

**Neutral**:
- `uv` becomes a required tool. Update the Tooling section in AGENTS.md and the CI workflow after the install works.

## Follow-up

- [x] Human runs the commands below and pastes the smoke test output (last 20 lines at most).
- [x] Record resolved versions in `docs/whats-built.md` (AC-5).
- [ ] Update AGENTS.md Stack and Tooling lines (uv installed, versions) via `/sync`.
- [ ] Add `uv sync --frozen` and `pytest` to `.github/workflows/ci.yml`, CPU only (CI has no GPU).
- [ ] If AC-2 fails, write ADR 0002 switching to the cu126 index.

## Commands (run by the human, in order)

    # 1. Install uv (pick one)
    curl -LsSf https://astral.sh/uv/install.sh | sh        # or: pipx install uv
    # open a new terminal, then: uv --version

    # 2. Keep the venv off the NTFS drive (add these two lines to ~/.bashrc)
    export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/qarar-research"
    export UV_LINK_MODE=copy

    # 3. Create the project files and lock (in the repo root)
    cd /media/noman/E0948A169489EF7A/qarar-research
    uv init --bare --python 3.12
    uv add torch monai nibabel
    uv add --dev pytest ruff

    # 4. Smoke test (AC-2, AC-3)
    uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_arch_list())"
    uv run python -c "import monai, nibabel; print(monai.__version__, nibabel.__version__)"

    # 5. Rebuild check (AC-1)
    uv sync --frozen

    # 6. Commit (explicit paths only)
    git add pyproject.toml uv.lock .python-version docs/architecture/decisions/0001-python-environment.md

The AC-3 forward and backward test and the AC-4 toy metric check are written as `tests/test_env_smoke.py` in the next step, not now.

## References

**Project sources**:
- `AGENTS.md` (hardware, reproducibility and "no new infrastructure without an ADR" rules)
- `docs/architecture/system-overview.md` (pipeline)

**Links** (fetched 2026-10-20):
- PyPI JSON metadata for torch, monai and nibabel (versions in the table above)
- PyTorch dev-discuss, "Transitioning PyPI CUDA wheels to CUDA 13.0 as the stable (Release 2.11)": https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325
- PyTorch dev-discuss, notice on CUDA 12.6 wheels ending at 2.15: https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432
- uv installation page: https://docs.astral.sh/uv/getting-started/installation/

## Result of the install (2026-10-20, human-run)

uv 0.13.2, CPython 3.12.3. Resolved: torch 2.14.1+cu130, monai 1.6.1, nibabel 5.4.2, numpy 2.5.4, pytest 9.1.1, ruff 0.17.1 (full list in `uv.lock`). `torch.cuda.is_available()` is True and the arch list is `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`, so the T2000 is covered (AC-2). The cu126 fallback is not needed. `scipy` was not in this first install; it was added on 2026-10-21 (see Environment rules), and `pandas` 3.1.0 on 2026-10-25.
