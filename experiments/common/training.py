"""The shared training loop, validation, checkpoints and run record (spec 0003, AC-10, AC-17).

E1 trains through train() with the identity batch_hook; E3 and E4 reuse it unchanged through
batch_hook and val_patterns. Trainer (one step per iteration) is also what memory_test.py runs. last.pt holds
the full training state, and train(resume=True) continues a failed, interrupted or crashed run
from it (AC-11, AC-12).
"""
import csv
import io
import math
import os
import random
import statistics
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from common.runlog import RunRecord, config_hash, dump_config, git_info, load_run, versions
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
VAL_COLUMNS = ["iteration", "pattern_id", "case", "region", "dice", "gt_empty", "pred_empty"]
BATCH_KEYS = ("case", "image", "label", "wt")
SELECTION_PATTERN = 1  # complete input


class NonFiniteLoss(RuntimeError):
    pass


def identity(batch, rng):
    """The E1 batch hook: changes nothing and draws nothing."""
    return batch


def check_config(cfg):
    """Problems that must stop a run before it starts (exit 2), as strings; empty = fine."""
    t = cfg["train"]
    if t["iterations"] % t["val_every"]:
        return [f"iterations {t['iterations']} is not a multiple of val_every {t['val_every']}"]
    return []


def lr_factor(i, warmup, iterations):
    """lr multiplier at iteration i (from 1): linear warmup to 1, then cosine to 0 at the end."""
    if i <= warmup:
        return i / warmup
    return 0.5 * (1 + math.cos(math.pi * (i - warmup) / (iterations - warmup)))


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


def append_val_log(path, iteration, rows):
    """Old rows plus this val point's rows, written to a temp file and renamed in."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, VAL_COLUMNS)
    if path.exists():
        buf.write(path.read_text())
    else:
        w.writeheader()
    w.writerows({"iteration": iteration, **r} for r in rows)
    tmp = path.with_suffix(".csv.tmp")
    tmp.write_text(buf.getvalue())
    os.replace(tmp, path)


class Trainer:
    """Model, optimiser, lr schedule, scaler and the endless train loader; step() is one iteration.

    train() and memory_test.py both use it, so the memory test measures the real training path.
    """

    def __init__(self, cfg, device, train_cases, gen):
        t = cfg["train"]
        self.cfg, self.device = cfg, device
        ds = Dataset(cached_items(cache_dir(cfg), train_cases), cached_train_transforms(cfg))
        loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True,
                            num_workers=t["num_workers"], generator=gen)
        self.batches = cycle(loader)
        self.model = build_model(cfg).to(device)
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.dice_loss, self.bce_loss = DiceLoss(sigmoid=True), torch.nn.BCEWithLogitsLoss()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=t["lr"],
                                     weight_decay=t["weight_decay"])
        self.amp = t["amp"] and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp)
        # stepped once per iteration, so the lr in the optimiser is the one applied at step i
        # (AC-7); clamped because the step after the last one asks for iteration iterations + 1
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(min(s + 1, t["iterations"]), t["warmup"],
                                          t["iterations"]))
        # its own stream, so a hook that draws never changes which patches a seed gives (AC-17)
        self.hook_rng = np.random.default_rng(cfg["seed"] + 1)
        self.skipped = 0

    def step(self, it, batch_hook=identity, inject_at=None):
        """One training iteration. Returns loss, dice_loss, bce_loss, lr (the one applied),
        seconds (the whole step, data wait included) and data_wait."""
        t0 = time.time()
        b = next(self.batches)
        wait = time.time() - t0
        batch = {"case": list(b["case"]),
                 **{k: b[k].to(self.device).float() for k in BATCH_KEYS[1:]}}
        batch = batch_hook(batch, self.hook_rng)
        x, y = batch["image"], batch["label"]
        if it == inject_at:
            x = torch.full_like(x, float("inf"))
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            logits = self.model(x)
        logits = logits.float()  # loss in float32
        ld, lb = self.dice_loss(logits, y), self.bce_loss(logits, y)
        loss = ld + lb
        if not torch.isfinite(loss):  # never skipped (AC-12)
            raise NonFiniteLoss(f"non finite loss at iteration {it}")
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        scale = self.scaler.get_scale()
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.amp and self.scaler.get_scale() < scale:
            self.skipped += 1  # inf gradients, step skipped by GradScaler; normal, not a failure
        lr = self.opt.param_groups[0]["lr"]
        self.sched.step()
        return {"loss": loss.item(), "dice_loss": ld.item(), "bce_loss": lb.item(), "lr": lr,
                "seconds": time.time() - t0, "data_wait": wait}

    def full_state(self):
        """Everything last.pt holds beyond the best.pt fields (AC-11)."""
        return {"optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
                "scaler": self.scaler.state_dict(), "scaler_skipped_steps": self.skipped,
                "hook_rng": self.hook_rng.bit_generator.state, "rng": rng_state()}

    def load_full_state(self, state):
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.sched.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.skipped = state["scaler_skipped_steps"]
        self.hook_rng.bit_generator.state = state["hook_rng"]
        set_rng_state(state["rng"])


def rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


@dataclass
class Progress:
    """What the run.json totals are computed from, carried across resumes."""
    losses: list = field(default_factory=list)  # one per kept train_log.csv row
    step_seconds: list = field(default_factory=list)  # the same rows' seconds
    base_wall: float = 0.0  # wall time of earlier segments, redone iterations included
    seg_seconds: float = 0.0  # training and val time of this segment
    base_peaks: tuple = (None, None)  # allocated, reserved MiB of earlier segments
    base_warnings: tuple = ()
    it: int = 0  # the iteration in progress, for failed_iteration


def keep_rows(path, last_iteration):
    """Rewrite a CSV log without the rows after last_iteration (temp file and rename).
    Returns the kept rows."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols, rows = r.fieldnames, [x for x in r if int(x["iteration"]) <= last_iteration]
    buf = io.StringIO()
    w = csv.DictWriter(buf, cols)
    w.writeheader()
    w.writerows(rows)
    tmp = path.with_suffix(".csv.tmp")
    tmp.write_text(buf.getvalue())
    os.replace(tmp, path)
    return rows


def _val_point(tr, rec, run_dir, it, val_patterns, prog):
    v0 = time.time()
    rows = validate(tr.model, rec.data["val_cases"], val_patterns, tr.cfg, tr.device)
    per, score = selection_score(rows)
    seconds = time.time() - v0
    prog.seg_seconds += seconds
    rec.data["val_history"].append({"iteration": it, "wt": per["WT"], "tc": per["TC"],
                                    "et": per["ET"], "score": score,
                                    "seconds": round(seconds, 1)})
    improved = rec.data["best_score"] is None or score > rec.data["best_score"]  # tie: earlier
    if improved:
        rec.update(best_iteration=it, best_score=score)
    state = {"model": tr.model.state_dict(), "iteration": it,
             "config_hash": rec.data["config_hash"], "score": score}
    ckpt = run_dir / "checkpoints"
    # fixed write order (AC-8): val_log.csv, best.pt if improved, last.pt, run.json
    append_val_log(run_dir / "val_log.csv", it, rows)
    if improved:
        save_checkpoint(ckpt / "best.pt", **state)
    save_checkpoint(ckpt / "last.pt", **state, **tr.full_state(),
                    best_score=rec.data["best_score"], best_iteration=rec.data["best_iteration"])
    _record_progress(rec, tr.device, prog)
    rec.write()
    return [per["WT"], per["TC"], per["ET"], score]


def _loop(cfg, rec, run_dir, device, gen, resume_state, batch_hook, val_patterns, prog):
    t = cfg["train"]
    tr = Trainer(cfg, device, rec.data["train_cases"], gen)
    if resume_state is not None:
        tr.load_full_state(resume_state)
    rec.update(n_params=tr.n_params)
    inject = os.environ.get("BASELINE_INJECT_INF")
    inject_at = int(inject) if inject else None
    start = len(prog.losses)
    with open(run_dir / "train_log.csv", "a" if start else "w", newline="") as f:
        log = csv.writer(f)
        if not start:
            log.writerow(LOG_COLUMNS)
        for it in range(start + 1, t["iterations"] + 1):
            prog.it = it
            s = tr.step(it, batch_hook, inject_at)
            prog.losses.append(s["loss"])
            prog.step_seconds.append(s["seconds"])
            prog.seg_seconds += s["seconds"]
            row = [it, s["loss"], s["dice_loss"], s["bce_loss"], s["lr"], round(s["seconds"], 4)]
            rec.update(iterations=it, scaler_skipped_steps=tr.skipped)
            if it % t["val_every"] == 0:
                row += _val_point(tr, rec, run_dir, it, val_patterns, prog)
            else:
                row += [""] * 4
            log.writerow(row)
            f.flush()


def _record_progress(rec, device, prog):
    w, start, end = loss_window(prog.losses)
    peaks = prog.base_peaks
    if device.type == "cuda":
        now = (torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20)
        peaks = tuple(n if b is None else max(b, n) for b, n in zip(peaks, now))
    rec.update(loss_window=w, loss_start_mean=start, loss_end_mean=end,
               wall_time_s=round(prog.base_wall + prog.seg_seconds, 1),
               seconds_per_iteration=(statistics.mean(prog.step_seconds)
                                      if prog.step_seconds else None),
               peak_vram_allocated_mib=peaks[0], peak_vram_reserved_mib=peaks[1])


def _new_record(cfg, run_dir, run_id, train_cases, val_cases, device, quick, commit, dirty):
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    dump_config(cfg, run_dir / "config.toml")
    return RunRecord(
        run_dir, keys=RUN_KEYS,
        defaults={"nondeterminism_warnings": [], "scaler_skipped_steps": 0, "iterations": 0,
                  "val_history": [], "resumes": [], "wall_time_s": 0.0},
        run_id=run_id, quick=quick, config_hash=config_hash(cfg), git_commit=commit,
        git_dirty=dirty, seed=cfg["seed"], device=device.type,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        versions=versions(), determinism=determinism_state(), train_cases=train_cases,
        val_cases=val_cases, cache_dir=str(cache_dir(cfg).resolve()),
        cache_hash=preprocess_hash(cfg)), Progress()


def _resume_record(run_dir, state, commit, dirty):
    """Roll the run folder back to the iteration in last.pt (AC-11). Returns (record, progress)."""
    start, ckpt = state["iteration"], run_dir / "checkpoints"
    old = load_run(run_dir)
    log = keep_rows(run_dir / "train_log.csv", start)
    assert [int(r["iteration"]) for r in log] == list(range(1, start + 1)), \
        f"train_log.csv does not hold iterations 1 to {start}"
    keep_rows(run_dir / "val_log.csv", start)
    best = ckpt / "best.pt"
    if best.is_file() and torch.load(best, map_location="cpu")["iteration"] > start:
        best.unlink()  # written after last.pt (a crash between the two); last.pt's best applies
        print(f"deleted best.pt from after iteration {start}; best score from last.pt applies")
    if state["best_iteration"] is not None and not best.is_file():
        print(f"WARNING: best.pt for iteration {state['best_iteration']} is gone; a later "
              "improvement writes a new one, otherwise evaluate.py will refuse this run")
    if commit != old["git_commit"]:
        print(f"WARNING: git commit {commit} differs from the run's {old['git_commit']}; "
              "recorded in run.json resumes")
    rec = RunRecord(
        run_dir, keys=RUN_KEYS, defaults={k: v for k, v in old.items() if k != "status"},
        reason=None, failed_iteration=None, iterations=start,
        best_score=state["best_score"], best_iteration=state["best_iteration"],
        scaler_skipped_steps=state["scaler_skipped_steps"],
        val_history=[v for v in old["val_history"] if v["iteration"] <= start],
        resumes=old["resumes"] + [{"from_iteration": start,
                                   "utc": datetime.now(UTC).isoformat(timespec="seconds"),
                                   "git_commit": commit, "git_dirty": dirty,
                                   "previous_status": old["status"]}])
    prog = Progress(losses=[float(r["loss"]) for r in log],
                    step_seconds=[float(r["seconds"]) for r in log],
                    base_wall=old["wall_time_s"] or 0.0,
                    base_peaks=(old["peak_vram_allocated_mib"], old["peak_vram_reserved_mib"]),
                    base_warnings=tuple(old["nondeterminism_warnings"]))
    return rec, prog


def train(cfg, run_dir, run_id, train_cases, val_cases, *, device, quick=False,
          batch_hook=identity, val_patterns=(SELECTION_PATTERN,), resume=False):
    """Train one run into run_dir and keep its run.json current. Returns the exit code:
    0 ok, 1 failed (OOM, non finite loss), 130 Ctrl+C. Any other error writes status failed,
    then raises. resume=True continues the run in run_dir from its last.pt (the caller has run
    the AC-11 guards); the cases and run_id then come from its run.json."""
    errors = check_config(cfg) + check_cache(cfg, train_cases + val_cases)
    assert not errors, f"config or cache guard failed: {errors[:3]}"  # scripts exit 2 first
    run_dir = Path(run_dir)
    gen = set_seed(cfg["seed"])
    commit, dirty = git_info()
    state = None
    if resume:
        state = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu",
                           weights_only=False)  # our own file; it holds numpy RNG state
        gen.manual_seed(cfg["seed"] * 100003 + state["iteration"])  # a fresh loader (AC-11)
        rec, prog = _resume_record(run_dir, state, commit, dirty)
    else:
        rec, prog = _new_record(cfg, run_dir, run_id, train_cases, val_cases, device, quick,
                                commit, dirty)
    pid = run_dir / "train.pid"
    pid.write_text(str(os.getpid()))
    print(f"run dir {run_dir}\nconfig hash {rec.data['config_hash']}  seed {cfg['seed']}  "
          f"git {commit}{' (dirty)' if dirty else ''}"
          + (f"\nresuming from iteration {state['iteration']}" if resume else ""))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    caught, code = [], 0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # expected when GradScaler skips the first step; not a nondeterminism warning
            warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")
            _loop(cfg, rec, run_dir, device, gen, state, batch_hook, val_patterns, prog)
        rec.update(status="ok")
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="failed", reason=f"CUDA out of memory: {str(e).splitlines()[0]}",
                   failed_iteration=prog.it)
        code = 1
    except NonFiniteLoss as e:
        rec.update(status="failed", reason=str(e), failed_iteration=prog.it)
        code = 1
    except KeyboardInterrupt:
        rec.update(status="interrupted", reason="Ctrl+C")
        code = 130
    except Exception as e:
        rec.update(status="failed", reason=failed_reason(e), failed_iteration=prog.it)
        raise
    finally:
        _record_progress(rec, device, prog)
        rec.update(iterations=len(prog.losses),
                   nondeterminism_warnings=sorted(set(prog.base_warnings)
                                                  | {str(m.message) for m in caught}))
        rec.write()
        pid.unlink(missing_ok=True)
        r = rec.data
        print(f"status {r['status']}{': ' + r['reason'] if r['reason'] else ''}  iterations "
              f"{r['iterations']}  wall {r['wall_time_s']} s  peak reserved "
              f"{r['peak_vram_reserved_mib'] and round(r['peak_vram_reserved_mib'], 1)} MiB  "
              f"best score {r['best_score'] and round(r['best_score'], 4)} "
              f"at {r['best_iteration']}")
    return code
