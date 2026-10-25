"""Tracer bullet training (spec 0001): 16 train subjects, 2,000 iterations, complete input only.

Run from the repo root:
    uv run python experiments/0002-tracer-bullet/train.py            # full run (human)
    uv run python experiments/0002-tracer-bullet/train.py --quick    # fast gate
Writes results/0002-tracer-bullet/<run_id>/ with config.toml, run.json, train_log.csv and
checkpoints/last.pt (the final weights; the only checkpoint, spec 0002). A run id that already
exists gets a suffix -1, -2, ...
Exit 0 ok, 1 failed (OOM, non finite loss), 2 bad inputs (before training), 130 Ctrl+C. Any other
error during training writes status failed, then shows its traceback (nonzero exit).
Test hook: TRACER_INJECT_INF=1 replaces the input of iteration 1 with inf.
"""
import argparse
import copy
import csv
import os
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceLoss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import check_inputs, eval_transforms, items, select_subjects, train_transforms
from common.metrics import region_metrics
from common.model import build_model, predict
from common.runlog import RunRecord, config_hash, dump_config, git_info, load_config, versions
from common.seed import determinism_state, set_seed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QUICK = {"data": {"n_train_subjects": 2, "n_val_subjects": 1},
         "train": {"iterations": 10, "val_every": 5}}


class NonFiniteLoss(RuntimeError):
    pass


def apply_quick(cfg):
    cfg = copy.deepcopy(cfg)
    for section, values in QUICK.items():
        cfg[section].update(values)
    return cfg


def cycle(loader):
    """Endless iterator over the loader; each pass reshuffles with the seeded generator."""
    while True:
        yield from loader


def validate(model, val_ds, cfg, device):
    """Mean WT Dice over the val cases, complete input, same inference and metric as evaluation."""
    model.eval()
    dices = []
    for i in range(len(val_ds)):
        d = val_ds[i]
        pred = predict(model, d["image"], cfg, device)
        dices.append(region_metrics(pred[0], d["label"][0], d["pixdim"])["dice"])
    model.train()
    return sum(dices) / len(dices)


def save_checkpoint(path, model, iteration, cfg_hash):
    """Write to a temp file, then rename, so Ctrl+C never leaves a half written checkpoint."""
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "iteration": iteration, "config_hash": cfg_hash}, tmp)
    os.replace(tmp, path)


def loss_window(losses):
    """W = min(50, iterations // 2); mean loss over the first and last W iterations."""
    w = min(50, len(losses) // 2)
    if w == 0:
        return w, None, None
    return w, sum(losses[:w]) / w, sum(losses[-w:]) / w


def train(cfg, rec, run_dir, data_dir, device, gen, cfg_hash, losses):
    t = cfg["train"]
    train_ds = CacheDataset(items(data_dir, rec.data["train_cases"]), train_transforms(cfg),
                            cache_rate=cfg["preprocess"]["cache_rate"], progress=False)
    val_ds = CacheDataset(items(data_dir, rec.data["val_cases"]), eval_transforms(cfg),
                          cache_rate=cfg["preprocess"]["cache_rate"], progress=False)
    loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                        num_workers=t["num_workers"], generator=gen)
    model = build_model(cfg).to(device)
    dice_loss, bce_loss = DiceLoss(sigmoid=True), torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    inject_inf = os.environ.get("TRACER_INJECT_INF") == "1"
    ckpt = run_dir / "checkpoints"
    skipped = 0
    batches = cycle(loader)
    t0 = time.time()
    with open(run_dir / "train_log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["iteration", "loss", "dice_loss", "bce_loss", "lr", "seconds", "val_wt_dice"])
        for it in range(1, t["iterations"] + 1):
            b = next(batches)
            x, y = b["image"].to(device).float(), b["label"].to(device).float()
            if inject_inf and it == 1:
                x = torch.full_like(x, float("inf"))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            logits = logits.float()  # loss in float32
            ld, lb = dice_loss(logits, y), bce_loss(logits, y)
            loss = ld + lb
            if not torch.isfinite(loss):
                raise NonFiniteLoss(f"non finite loss at iteration {it}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if amp and scaler.get_scale() < scale:
                skipped += 1  # inf gradients, step skipped by GradScaler; normal, not a failure
            losses.append(loss.item())
            val = ""
            if it % t["val_every"] == 0:
                val = validate(model, val_ds, cfg, device)
                rec.data["val_wt_dice"].append([it, val])  # learning curve only, no selection
                save_checkpoint(ckpt / "last.pt", model, it, cfg_hash)
            log.writerow([it, losses[-1], ld.item(), lb.item(), opt.param_groups[0]["lr"],
                          round(time.time() - t0, 3), val])
            f.flush()
            rec.update(iterations=it, scaler_skipped_steps=skipped)
    save_checkpoint(ckpt / "last.pt", model, t["iterations"], cfg_hash)


def make_run_dir(out, stamp):
    """Create out/<stamp>, or out/<stamp>-1, -2, ... if taken. Returns (run_id, run_dir)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    run_id, n = stamp, 0
    while True:
        try:
            (out / run_id).mkdir(exist_ok=False)
            return run_id, out / run_id
        except FileExistsError:
            n += 1
            run_id = f"{stamp}-{n}"


def failed_reason(e):
    """'<ExceptionType>: <first line of message>', or just the type when the message is empty."""
    first = (str(e).splitlines() or [""])[0]
    return f"{type(e).__name__}: {first}" if first else type(e).__name__


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=HERE / "config.toml")
    ap.add_argument("--data-dir", default=os.environ.get("BRATS_TRAIN_DIR"))
    ap.add_argument("--splits", default=None, help="default: [data] splits_csv of the config")
    ap.add_argument("--out", default=REPO / "results" / "0002-tracer-bullet")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.quick:
        cfg = apply_quick(cfg)
    cfg_hash = config_hash(cfg)
    if not a.data_dir or not Path(a.data_dir).is_dir():
        print(f"ERROR: data dir not found: {a.data_dir!r} (set BRATS_TRAIN_DIR or --data-dir)")
        return 2
    d = cfg["data"]
    train_cases, val_cases = select_subjects(
        REPO / (a.splits or d["splits_csv"]), d["n_train_subjects"], d["n_val_subjects"],
        d["timepoint"], d["exclude_flair_flag"])
    missing = check_inputs(a.data_dir, train_cases + val_cases)
    if missing:
        print(f"ERROR: {len(missing)} input file(s) missing, nothing trained:")
        print("\n".join("  " + m for m in missing))
        return 2

    gen = set_seed(cfg["seed"])
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu")
    run_id, run_dir = make_run_dir(a.out, f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{cfg_hash[:8]}")
    (run_dir / "checkpoints").mkdir()
    dump_config(cfg, run_dir / "config.toml")
    commit, dirty = git_info()
    rec = RunRecord(
        run_dir, run_id=run_id, quick=a.quick, config_hash=cfg_hash, git_commit=commit,
        git_dirty=dirty, seed=cfg["seed"], data_dir=str(Path(a.data_dir).resolve()),
        device=device.type, gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        versions=versions(), determinism=determinism_state(), train_cases=train_cases,
        val_cases=val_cases)
    print(f"run {run_id}  config hash {cfg_hash}  git {commit}{' (dirty)' if dirty else ''}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses, caught, code = [], [], 0
    t0 = time.time()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train(cfg, rec, run_dir, a.data_dir, device, gen, cfg_hash, losses)
        rec.update(status="ok")
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="failed", reason=f"CUDA out of memory: {str(e).splitlines()[0]}")
        code = 1
    except NonFiniteLoss as e:
        rec.update(status="failed", reason=str(e))
        code = 1
    except KeyboardInterrupt:
        rec.update(status="interrupted", reason="Ctrl+C")
        code = 130
    except Exception as e:
        rec.update(status="failed", reason=failed_reason(e))
        raise
    finally:
        w, start, end = loss_window(losses)
        rec.update(
            wall_time_s=round(time.time() - t0, 1), iterations=len(losses), loss_window=w,
            loss_start_mean=start, loss_end_mean=end,
            nondeterminism_warnings=sorted({str(m.message) for m in caught}))
        if device.type == "cuda":
            rec.update(peak_vram_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_vram_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        rec.write()
        r = rec.data
        print(f"status {r['status']}{': ' + r['reason'] if r['reason'] else ''}  iterations "
              f"{r['iterations']}  wall {r['wall_time_s']} s  peak reserved "
              f"{r['peak_vram_reserved_mib'] and round(r['peak_vram_reserved_mib'], 1)} MiB  "
              f"val WT Dice {r['val_wt_dice'] and round(r['val_wt_dice'][-1][1], 4)}")
        print(f"run dir {run_dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
