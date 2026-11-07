"""The three E1 test evals -> does each H1 line agree across the seeds? (spec 0006, AC-30)

    uv run python experiments/0007-confirmation/rq1_seeds.py --eval <dir> --eval <dir> --eval <dir>
Run analyse.py on each eval first: this reads its verdict.txt and run.json, and no metric. Two seeds
agree on a line when it has the same word, supported or not supported. The pattern lists and p
values may differ. A line that splits means the H1 claim is not made. Writes one line per H1 line,
in the order of verdict.txt. Exit 1, writing nothing, unless the evals are the finished test evals
of seeds 42, 43 and 44. Uses no GPU.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runlog import load_run

REPO = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
WORDS = ("not supported", "supported")  # the longer first, so a prefix test cannot mix them up


def parse_verdict(text):
    """verdict.txt as {'H1a WT': word}. A line in another shape raises ValueError."""
    words = {}
    for line in text.splitlines():
        name, _, rest = line.partition(": ")
        word = next((w for w in WORDS if rest.startswith(w)), None)
        if not name.startswith("H1") or word is None:
            raise ValueError(f"not an H1 verdict line: {line!r}")
        words[name] = word
    return words


def read_evals(eval_dirs):
    """Returns (errors, {seed: {line name: word}}), the second empty unless all could be read."""
    errors, verdicts = [], {}
    for d in map(Path, eval_dirs):
        if not (d / "run.json").is_file() or not (d / "verdict.txt").is_file():
            errors.append(f"{d}: run.json or verdict.txt missing, run analyse.py first")
            continue
        rec = load_run(d)
        if rec.get("status") != "ok" or rec.get("quick") is not False:
            errors.append(f"{d}: not a finished full eval")
        if rec.get("fold") != "test":
            errors.append(f"{d}: fold is {rec.get('fold')!r}, not 'test'")
        try:
            verdicts[rec.get("source_seed")] = parse_verdict((d / "verdict.txt").read_text())
        except ValueError as e:
            errors.append(f"{d}: {e}")
    if sorted(verdicts, key=str) != sorted(SEEDS, key=str):
        errors.append(f"seeds are {sorted(verdicts, key=str)}, expected {list(SEEDS)}")
    elif any(list(v) != list(verdicts[SEEDS[0]]) for v in verdicts.values()):
        errors.append("the verdicts hold different H1 lines")
    return errors, {} if errors else verdicts


def agreement(verdicts):
    """One line per H1 line. Only `agree` lets the thesis make that claim (AC-30)."""
    lines = []
    for name in verdicts[SEEDS[0]]:
        words = [verdicts[s][name] for s in SEEDS]
        if len(set(words)) == 1:
            lines.append(f"{name}: agree, {words[0]} (seeds {SEEDS[0]}, {SEEDS[1]}, {SEEDS[2]})")
        else:
            seeds = ", ".join(f"seed {s} {w}" for s, w in zip(SEEDS, words, strict=True))
            lines.append(f"{name}: split, claim not made ({seeds})")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", required=True, action="append", type=Path, dest="evals")
    ap.add_argument("--out", type=Path, default=REPO / "results/0007-confirmation/rq1-test.txt")
    a = ap.parse_args(argv)
    errors, verdicts = read_evals(a.evals)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    lines = agreement(verdicts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
