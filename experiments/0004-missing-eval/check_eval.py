"""Check one finished eval folder against every rule of spec 0004 (AC-9, AC-16).

    uv run python experiments/0004-missing-eval/check_eval.py <eval dir> [--run <E1 run dir>]
A thin wrapper around common.evaluation.check_rules. --run defaults to source_run in run.json, and
is needed when the eval folder moved between machines. Exit 1 on any failed rule, printing the
reason for each. A passing val eval also prints the projected test time, as test has exactly
twice the cases.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluation import check_rules
from common.runlog import load_run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eval", type=Path)
    ap.add_argument("--run", type=Path)
    a = ap.parse_args(argv)
    if not (a.eval / "run.json").is_file():
        print(f"FAIL: {a.eval / 'run.json'} missing")
        return 1
    rec = load_run(a.eval)
    errors = check_rules(a.eval, a.run or rec["source_run"])
    for e in errors:
        print(f"FAIL: {e}")
    if errors:
        return 1
    print(f"PASS: {rec['eval_id']}, {rec['n_cases']} cases, status ok, every rule holds")
    if rec["fold"] == "val":
        hours = (rec["predict_seconds"] + rec["metric_seconds"]) * 2 / 3600
        print(f"Projected test time: {hours:.2f} hours")
    return 0


if __name__ == "__main__":
    sys.exit(main())
