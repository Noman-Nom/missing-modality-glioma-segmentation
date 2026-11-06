"""Fairness check of one E3 or E4 run against the E1 run (spec 0005, AC-12).

    uv run python experiments/0005-strategies/check_strategy.py <run dir> --e1 <E1 run dir>
Exit 1, printing every failed rule, unless the run differs from E1 in the strategy alone. On a pass
it prints how the draws fell on the 15 patterns, recomputed from the seed, then the number of
resumes, then PASS. A resume is allowed, but it reseeds the loader, so that run's later patches no
longer match the other models'. The draws still match, because hook_rng is restored from last.pt.
Different package versions are a warning, not a failure. Reads run.json and config.toml only, so it
needs no GPU and no case. Run check_baseline.py on the same run too: this file does not repeat it.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.missing import PATTERNS
from common.runlog import load_config, load_run
from common.strategies import STRATEGIES, draw_patterns, strategy_name

SAME_AS_E1 = ("train_cases", "val_cases", "cache_hash", "seed", "gpu", "iterations")


def problems(rec, cfg, e1_rec, e1_cfg):
    """Every failed AC-12 rule, as strings. Only the strategy run loses its [strategy], so an E1
    folder that holds one is refused too."""
    errors = []
    if rec.get("status") != "ok":
        errors.append(f"status is {rec.get('status')!r}, not 'ok'")
    if rec.get("quick") is not False:
        errors.append("a quick run")
    if strategy_name(cfg) not in STRATEGIES:
        errors.append(f"strategy is {strategy_name(cfg)!r}, expected one of {STRATEGIES}")
    if {k: v for k, v in cfg.items() if k != "strategy"} != e1_cfg:
        errors.append("config differs from E1 outside [strategy]")
    errors += [f"{k} differs from E1" for k in SAME_AS_E1 if rec.get(k) != e1_rec.get(k)]
    return errors


def draw_counts(seed, iterations, batch_size):
    """How many draws fell on each pattern, made the way Trainer makes them: one call per iteration
    on a generator seeded with seed + 1 (AC-2)."""
    rng = np.random.default_rng(seed + 1)
    counts = Counter()
    for _ in range(iterations):
        counts.update(int(p) for p in draw_patterns(rng, batch_size))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--e1", required=True, type=Path)
    a = ap.parse_args(argv)
    missing = [d / name for d in (a.run, a.e1) for name in ("run.json", "config.toml")
               if not (d / name).is_file()]
    if missing:
        for path in missing:
            print(f"FAIL: {path} missing")
        return 1
    rec, cfg = load_run(a.run), load_config(a.run / "config.toml")
    e1_rec, e1_cfg = load_run(a.e1), load_config(a.e1 / "config.toml")
    if rec.get("versions") != e1_rec.get("versions"):
        print(f"WARNING: versions differ from E1: {rec.get('versions')} and "
              f"{e1_rec.get('versions')}")
    errors = problems(rec, cfg, e1_rec, e1_cfg)
    for e in errors:
        print(f"FAIL: {e}")
    if errors:
        return 1
    counts = draw_counts(rec["seed"], rec["iterations"], cfg["train"]["batch_size"])
    print("Expected draws, recomputed from the seed, not read from the run:")
    for p in PATTERNS:
        print(f"pattern {p:2d} {PATTERNS[p].name}: {counts[p]} draws")
    print(f"resumes: {len(rec['resumes'])}")
    print(f"PASS: {rec['run_id']}, {strategy_name(cfg)}, differs from E1 in the strategy alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
