"""The shared training loop, validation, checkpoints and run record (spec 0003, AC-10, AC-17).

E1 trains through train() with the identity batch_hook; E3 and E4 reuse it unchanged through
batch_hook and val_patterns. Build plan 1 (thin thread): constant lr, no augmentation, no resume.
"""
import csv
import os
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss

from common.cache import cache_dir, check_cache, preprocess_hash
from common.data import REGIONS, cached_eval_transforms, cached_items, cached_train_transforms
from common.metrics import region_metrics
from common.missing import apply_pattern
from common.model import build_model, predict
from common.runlog import RunRecord, config_hash, dump_config, git_info, versions
from common.seed import determinism_state, set_seed

# every key is present in every E1 run.json (null where not yet known); spec 0003, run.json keys
RUN_KEYS = (
    "run_id", "status", "reason", "quick", "config_hash", "git_commit", "git_dirty", "seed",
    "data_dir", "device", "gpu", "versions", "determinism", "nondeterminism_warnings",
    "scaler_skipped_steps", "loss_window", "loss_start_mean", "loss_end_mean", "train_cases",
    "val_cases", "peak_vram_allocated_mib", "peak_vram_reserved_mib", "wall_time_s",
    "iterations", "n_params", "cache_dir", "cache_hash", "seconds_per_iteration", "val_history",
    "best_iteration", "best_score", "failed_iteration", "resumes",
)
LOG_COLUMNS = ["iteration", "loss", "dice_loss", "bce_loss", "lr", "seconds",
               "val_wt", "val_tc", "val_et", "val_score"]
BATCH_KEYS = ("case", "image", "label", "wt")
SELECTION_PATTERN = 1  # complete input


class NonFiniteLoss(RuntimeError):
    pass


def identity(batch, rng):
    """The E1 batch hook: changes nothing and draws nothing."""
    return batch


def make_run_dir(out, stamp):
    """Create out/<stamp>, or out/<stamp>-1, -2, ... if taken (spec 0002). Returns (id, dir)."""
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


def cycle(loader):
    """Endless iterator over the loader; each pass reshuffles with the seeded generator."""
    while True:
        yield from loader


def loss_window(losses):
    """W = min(50, iterations // 2); mean loss over the first and last W iterations."""
    w = min(50, len(losses) // 2)
    if w == 0:
        return w, None, None
    return w, sum(losses[:w]) / w, sum(losses[-w:]) / w


def validate(model, val_cases, patterns, cfg, device):
    """Dice per (pattern, case, region) on the val cases, same inference as evaluation.

    Returns rows {pattern_id, case, region, dice, gt_empty, pred_empty}. Dice only, no HD95.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tf = cached_eval_transforms(cfg)
    model.eval()
    rows = []
    for item in cached_items(cache_dir(cfg), val_cases):
        d = tf(item)
        for pid in patterns:
            pred = predict(model, apply_pattern(d["image"], pid), cfg, device)
            for ri, region in enumerate(REGIONS):
                m = region_metrics(pred[ri], d["label"][ri], d["pixdim"], with_hd95=False)
                rows.append({"pattern_id": pid, "case": d["case"], "region": region,
                             "dice": m["dice"], "gt_empty": m["gt_empty"],
                             "pred_empty": m["pred_empty"]})
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def selection_score(rows, pattern_id=SELECTION_PATTERN):
    """Mean Dice over the val cases per region at one pattern, then the mean of the 3 regions.

    Returns ({region: mean Dice}, score).
    """
    per = {r: statistics.mean(x["dice"] for x in rows
                              if x["pattern_id"] == pattern_id and x["region"] == r)
           for r in REGIONS}
    return per, statistics.mean(per.values())


def save_checkpoint(path, **state):
    """Write to a temp file, then rename, so Ctrl+C never leaves a half written checkpoint."""
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _loop(cfg, rec, run_dir, device, gen, batch_hook, val_patterns, losses, step_seconds):
    t = cfg["train"]
    cfg_hash, val_cases = rec.data["config_hash"], rec.data["val_cases"]
    ds = Dataset(cached_items(cache_dir(cfg), rec.data["train_cases"]),
                 cached_train_transforms(cfg))
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True,
                        num_workers=t["num_workers"], generator=gen)
    model = build_model(cfg).to(device)
    rec.update(n_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    dice_loss, bce_loss = DiceLoss(sigmoid=True), torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    # its own stream, so a hook that draws never changes which patches a seed gives (AC-17)
    hook_rng = np.random.default_rng(cfg["seed"] + 1)
    inject = os.environ.get("BASELINE_INJECT_INF")
    inject_at = int(inject) if inject else None
    ckpt = run_dir / "checkpoints"
    skipped = 0
    batches = cycle(loader)
    with open(run_dir / "train_log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(LOG_COLUMNS)
        for it in range(1, t["iterations"] + 1):
            t0 = time.time()
            b = next(batches)
            batch = {"case": list(b["case"]),
                     **{k: b[k].to(device).float() for k in BATCH_KEYS[1:]}}
            batch = batch_hook(batch, hook_rng)
            x, y = batch["image"], batch["label"]
            if it == inject_at:
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
            step_seconds.append(time.time() - t0)
            row = [it, losses[-1], ld.item(), lb.item(), opt.param_groups[0]["lr"],
                   round(step_seconds[-1], 4)]
            rec.update(iterations=it, scaler_skipped_steps=skipped)
            if it % t["val_every"] == 0:
                v0 = time.time()
                per, score = selection_score(validate(model, val_cases, val_patterns, cfg, device))
                row += [per["WT"], per["TC"], per["ET"], score]
                rec.data["val_history"].append({"iteration": it, "wt": per["WT"], "tc": per["TC"],
                                                "et": per["ET"], "score": score,
                                                "seconds": round(time.time() - v0, 1)})
                state = {"model": model.state_dict(), "iteration": it, "config_hash": cfg_hash,
                         "score": score}
                if rec.data["best_score"] is None or score > rec.data["best_score"]:
                    save_checkpoint(ckpt / "best.pt", **state)  # a tie keeps the earlier one
                    rec.update(best_iteration=it, best_score=score)
                save_checkpoint(ckpt / "last.pt", **state)
                _record_progress(rec, device, losses, step_seconds)
                rec.write()
            else:
                row += [""] * 4
            log.writerow(row)
            f.flush()


def _record_progress(rec, device, losses, step_seconds):
    w, start, end = loss_window(losses)
    rec.update(loss_window=w, loss_start_mean=start, loss_end_mean=end,
               wall_time_s=round(sum(step_seconds) + sum(v["seconds"]
                                                        for v in rec.data["val_history"]), 1),
               seconds_per_iteration=(sum(step_seconds) / len(step_seconds)
                                      if step_seconds else None))
    if device.type == "cuda":
        rec.update(peak_vram_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   peak_vram_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)


def train(cfg, run_dir, run_id, train_cases, val_cases, *, device, quick=False,
          batch_hook=identity, val_patterns=(SELECTION_PATTERN,)):
    """Train one run into run_dir and keep its run.json current. Returns the exit code:
    0 ok, 1 failed (OOM, non finite loss), 130 Ctrl+C. Any other error writes status failed,
    then raises."""
    errors = check_cache(cfg, train_cases + val_cases)
    assert not errors, f"cache guard failed: {errors[:3]}"  # scripts check first and exit 2
    run_dir = Path(run_dir)
    gen = set_seed(cfg["seed"])
    cfg_hash = config_hash(cfg)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    dump_config(cfg, run_dir / "config.toml")
    commit, dirty = git_info()
    rec = RunRecord(
        run_dir, keys=RUN_KEYS,
        defaults={"nondeterminism_warnings": [], "scaler_skipped_steps": 0, "iterations": 0,
                  "val_history": [], "resumes": [], "wall_time_s": 0.0},
        run_id=run_id, quick=quick, config_hash=cfg_hash, git_commit=commit, git_dirty=dirty,
        seed=cfg["seed"], device=device.type,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        versions=versions(), determinism=determinism_state(), train_cases=train_cases,
        val_cases=val_cases, cache_dir=str(cache_dir(cfg).resolve()),
        cache_hash=preprocess_hash(cfg))
    print(f"run dir {run_dir}\nconfig hash {cfg_hash}  seed {cfg['seed']}  "
          f"git {commit}{' (dirty)' if dirty else ''}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses, step_seconds, caught, code = [], [], [], 0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _loop(cfg, rec, run_dir, device, gen, batch_hook, val_patterns, losses, step_seconds)
        rec.update(status="ok")
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="failed", reason=f"CUDA out of memory: {str(e).splitlines()[0]}",
                   failed_iteration=len(losses) + 1)
        code = 1
    except NonFiniteLoss as e:
        rec.update(status="failed", reason=str(e), failed_iteration=len(losses) + 1)
        code = 1
    except KeyboardInterrupt:
        rec.update(status="interrupted", reason="Ctrl+C")
        code = 130
    except Exception as e:
        rec.update(status="failed", reason=failed_reason(e), failed_iteration=len(losses) + 1)
        raise
    finally:
        _record_progress(rec, device, losses, step_seconds)
        rec.update(iterations=len(losses),
                   nondeterminism_warnings=sorted({str(m.message) for m in caught}))
        rec.write()
        r = rec.data
        print(f"status {r['status']}{': ' + r['reason'] if r['reason'] else ''}  iterations "
              f"{r['iterations']}  wall {r['wall_time_s']} s  peak reserved "
              f"{r['peak_vram_reserved_mib'] and round(r['peak_vram_reserved_mib'], 1)} MiB  "
              f"best score {r['best_score'] and round(r['best_score'], 4)} "
              f"at {r['best_iteration']}")
    return code
