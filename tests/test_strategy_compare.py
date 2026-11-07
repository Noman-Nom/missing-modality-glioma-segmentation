"""Spec 0005: compare.py and make_table.py on toy run and eval folders with known answers. Covers
AC-15 to AC-19. Nothing here trains or scores a model, and no number here is a result."""
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from common.data import REGIONS
from common.evaluation import METRIC_COLUMNS
from common.missing import PATTERNS
from common.runlog import config_hash, dump_config, load_config
from conftest import write_splits

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "0005-strategies"
E1_CONFIG = REPO / "experiments" / "0003-baseline" / "config.toml"
E2_CONFIG = REPO / "experiments" / "0004-missing-eval" / "config.toml"
N = 10
VAL = [f"BraTS-GLI-001{i:02d}-000" for i in range(N)]
TEST = [f"BraTS-GLI-002{i:02d}-000" for i in range(N)]
TRAIN = ["BraTS-GLI-00301-000", "BraTS-GLI-00302-000"]
RUN_IDS = {"e1": "run-e1", "e3": "run-e3", "e4": "run-e4"}
VERSIONS = {"torch": "1", "monai": "2"}


def load(name):
    spec = importlib.util.spec_from_file_location(f"strategies_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compare = load("compare")
make_table = load("make_table")


def plain(case, pattern, region):
    """The E1 Dice of a toy case: distinct per case, and lower when input is missing."""
    return 0.5 + VAL_INDEX[case] / 100 - (pattern > 1) * 0.1


VAL_INDEX = {c: i for i, c in enumerate(VAL + TEST)}
VAL_INDEX |= {c: i for i, c in enumerate(TEST)}


def gain(amount, spread=0.001):
    """A model that beats E1 by amount + spread * case on the incomplete patterns only, so pattern
    1 ties. The spread makes the differences distinct, so the ranks have a known answer."""
    return lambda case, p, g: plain(case, p, g) + (p > 1) * (amount + spread * VAL_INDEX[case])


def write_metrics(path, cases, dice, hd95):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        w.writeheader()
        for c in cases:
            for p in PATTERNS:
                for g in REGIONS:
                    w.writerow({"eval_id": "x", "case": c, "subject": c.split("-")[2],
                                "pattern_id": p, "pattern_name": PATTERNS[p].name, "region": g,
                                "dice": dice(c, p, g), "hd95_mm": hd95(c, p, g),
                                "sensitivity": 0.5, "precision": 0.5, "gt_empty": False,
                                "pred_empty": False, "pred_ml": 1.0, "true_ml": 1.0})


class World:
    """Three finished runs and their evals, on disk, in the spec's folder layout."""

    def __init__(self, root):
        self.root = root
        self.splits = root / "splits.csv"
        write_splits(self.splits, [(c, "train", 0, False) for c in TRAIN]
                     + [(c, "val", 0, False) for c in VAL] + [(c, "test", 0, False) for c in TEST])
        self.e1_evals, self.evals, self.out = root / "e1-evals", root / "evals", root / "out"
        self.runs = {m: root / "runs" / RUN_IDS[m] for m in compare.MODELS}
        self.dice = {"e1": plain, "e3": gain(0.02), "e4": gain(0.05, 0.003)}
        self.hd95 = {"e1": lambda c, p, g: 10.0, "e3": lambda c, p, g: 9.0 - VAL_INDEX[c] / 100,
                     "e4": lambda c, p, g: 8.0 - VAL_INDEX[c] / 100}
        for m in compare.MODELS:
            self.write_run(m)
            self.write_eval(m, "val")

    def write_run(self, m):
        run = self.runs[m]
        run.mkdir(parents=True)
        cfg = load_config(E1_CONFIG)
        cfg["data"]["splits_csv"] = str(self.splits)
        if compare.STRATEGIES[m]:
            cfg["strategy"] = {"name": compare.STRATEGIES[m]}
        dump_config(cfg, run / "config.toml")
        (run / "run.json").write_text(json.dumps(
            {"run_id": RUN_IDS[m], "status": "ok", "quick": False, "train_cases": TRAIN,
             "val_cases": VAL, "seed": 42, "gpu": "T4", "cache_hash": "c", "resumes": []}))
        with open(run / "metrics_val.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["case", "region", "dice"])
            w.writerows((c, g, self.dice[m](c, 1, g)) for c in VAL for g in REGIONS)

    def eval_dir(self, m, fold):
        return (self.e1_evals if m == "e1" else self.evals) / f"{RUN_IDS[m]}-{fold}"

    def write_eval(self, m, fold):
        d = self.eval_dir(m, fold)
        d.mkdir(parents=True)
        dump_config(load_config(E2_CONFIG), d / "config.toml")
        cases = VAL if fold == "val" else TEST
        (d / "run.json").write_text(json.dumps(
            {"status": "ok", "quick": False, "fold": fold, "n_cases": N, "gpu": "T4",
             "source_run_id": RUN_IDS[m], "protocol_hash": "p", "splits_sha256": "s",
             "config_hash": config_hash(load_config(d / "config.toml")),
             "source_best_iteration": 1, "source_best_sha256": m, "versions": VERSIONS}))
        write_metrics(d / "metrics.csv", cases, self.dice[m], self.hd95[m])

    def open_test(self):
        for m in compare.MODELS:
            self.write_eval(m, "test")

    def edit(self, path, **fields):
        rec = json.loads(Path(path).read_text())
        Path(path).write_text(json.dumps(rec | fields))

    def args(self, fold="val", *extra):
        return ["--e1-run", str(self.runs["e1"]), "--e3-run", str(self.runs["e3"]),
                "--e4-run", str(self.runs["e4"]), "--fold", fold, "--e1-evals", str(self.e1_evals),
                "--evals", str(self.evals), "--out", str(self.out), *extra]

    def run(self, fold="val", *extra):
        return compare.main(self.args(fold, *extra))

    def folder(self, fold="val"):
        return self.out / f"compare-{fold}"


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def row(folder, name, **match):
    found = [r for r in rows(folder / f"{name}.csv")
             if all(r[k] == str(v) for k, v in match.items())]
    assert len(found) == 1, match
    return found[0]


def wilcoxon_all_positive(n):
    """The two sided asymptotic p of n distinct positive differences, by hand: W = 0."""
    mean = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return math.erfc(mean / sd / math.sqrt(2))


# covers: AC-17
def test_outputs_have_the_exact_rows_in_the_spec_order(world):
    assert world.run() == 0
    summary = rows(world.folder() / "summary.csv")
    patterns = rows(world.folder() / "patterns.csv")
    assert (len(summary), len(patterns)) == (72, 270)
    assert list(summary[0]) == compare.SUMMARY_COLUMNS
    assert list(patterns[0]) == compare.PATTERN_COLUMNS
    levels = ("incomplete", "missing_1", "missing_2", "missing_3")
    comparisons = ("E3-E1", "E4-E1", "E4-E3")
    assert [(r["metric"], r["region"], r["level"], r["comparison"]) for r in summary] == [
        (m, g, lv, c) for m in compare.METRICS for g in REGIONS for lv in levels
        for c in comparisons]
    assert [(r["metric"], r["region"], int(r["pattern_id"]), r["comparison"])
            for r in patterns] == [(m, g, p, c) for m in compare.METRICS for g in REGIONS
                                   for p in range(1, 16) for c in comparisons]
    assert {r["pattern_name"] for r in patterns} == {p.name for p in PATTERNS.values()}


# covers: AC-17, the Statistics protocol: known differences give known numbers
def test_known_answer_on_the_primary_score(world):
    assert world.run() == 0
    r = row(world.folder(), "summary", metric="dice", region="WT", level="incomplete",
            comparison="E3-E1")
    # the gain is 0.02 + 0.001 i on all 14 incomplete patterns, i = 0 to 9
    assert float(r["mean_diff"]) == pytest.approx(0.0245, abs=1e-6)
    assert float(r["median_diff"]) == pytest.approx(0.0245, abs=1e-6)
    assert float(r["mean_second"]) == pytest.approx(0.5 + 0.045 - 0.1, abs=1e-6)
    assert float(r["mean_first"]) == pytest.approx(0.445 + 0.0245, abs=1e-6)
    assert float(r["ci_low"]) < 0.0245 < float(r["ci_high"])
    assert int(r["n"]) == N and int(r["n_nonzero"]) == N
    p = wilcoxon_all_positive(N)
    assert float(r["wilcoxon_stat"]) == 0
    assert float(r["p_raw"]) == pytest.approx(p, rel=1e-4)
    assert float(r["p_holm"]) == pytest.approx(3 * p, rel=1e-4)


# covers: AC-17, the Statistics protocol: the CI is the percentile bootstrap of the spec, by hand
def test_known_answer_on_the_confidence_interval(world):
    assert world.run() == 0
    r = row(world.folder(), "summary", metric="dice", region="ET", level="incomplete",
            comparison="E3-E1")
    d = 0.02 + 0.001 * np.arange(N)
    means = d[np.random.default_rng(42).integers(0, N, size=(10000, N))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    assert float(r["ci_low"]) == pytest.approx(low, abs=1e-6)
    assert float(r["ci_high"]) == pytest.approx(high, abs=1e-6)


# covers: the Statistics protocol, pattern 1 is left out of the primary score
def test_pattern_one_is_not_in_the_primary_score(world):
    world.dice["e3"] = lambda c, p, g: plain(c, p, g) + (p == 1) * 0.3
    shutil.rmtree(world.eval_dir("e3", "val"))
    world.write_eval("e3", "val")
    shutil.rmtree(world.runs["e3"])
    world.write_run("e3")
    assert world.run() == 0
    r = row(world.folder(), "summary", metric="dice", region="TC", level="incomplete",
            comparison="E3-E1")
    assert float(r["mean_diff"]) == 0 and int(r["n_nonzero"]) == 0
    p1 = row(world.folder(), "patterns", metric="dice", region="TC", pattern_id=1,
             comparison="E3-E1")
    assert float(p1["mean_diff"]) == pytest.approx(0.3, abs=1e-6)


# covers: AC-17, a level uses only its own patterns, and carries no test
def test_a_level_is_its_own_patterns_with_a_ci_and_no_test(world):
    world.dice["e3"] = lambda c, p, g: plain(c, p, g) + (len(PATTERNS[p].missing) == 3) * (
        0.04 + VAL_INDEX[c] / 1000)
    shutil.rmtree(world.eval_dir("e3", "val"))
    world.write_eval("e3", "val")
    assert world.run() == 0
    get = lambda lv: row(world.folder(), "summary", metric="dice", region="ET", level=lv,
                         comparison="E3-E1")
    assert float(get("missing_3")["mean_diff"]) == pytest.approx(0.0445, abs=1e-6)
    assert float(get("missing_1")["mean_diff"]) == 0
    assert float(get("incomplete")["mean_diff"]) == pytest.approx(0.0445 * 4 / 14, abs=1e-6)
    for lv in ("missing_1", "missing_2", "missing_3"):
        r = get(lv)
        assert (r["wilcoxon_stat"], r["p_raw"], r["p_holm"]) == ("", "", "")
        assert r["ci_low"] and r["ci_high"] and r["n_nonzero"]


# covers: the Statistics protocol, Holm over the 15 patterns of one comparison
def test_pattern_holm_is_over_the_fifteen(world):
    assert world.run() == 0
    get = lambda p: row(world.folder(), "patterns", metric="dice", region="WT", pattern_id=p,
                        comparison="E3-E1")
    raw = wilcoxon_all_positive(N)
    assert float(get(2)["p_raw"]) == pytest.approx(raw, rel=1e-4)
    # pattern 1 ties (p 1), and the 14 others share one p: Holm gives 15 p to all 14
    assert float(get(2)["p_holm"]) == pytest.approx(15 * raw, rel=1e-3)
    assert float(get(15)["p_holm"]) == pytest.approx(15 * raw, rel=1e-3)
    assert float(get(1)["p_raw"]) == 1 and float(get(1)["p_holm"]) == 1


# covers: the Statistics protocol, HD95 has its own numbers and its own families
def test_hd95_rows_use_the_hd95_column(world):
    assert world.run() == 0
    r = row(world.folder(), "summary", metric="hd95_mm", region="WT", level="incomplete",
            comparison="E3-E1")
    assert float(r["mean_diff"]) == pytest.approx(-1 - 0.045, abs=1e-6)
    p = row(world.folder(), "patterns", metric="hd95_mm", region="WT", pattern_id=1,
            comparison="E4-E1")
    assert float(p["median_second"]) == 10 and float(p["median_first"]) == pytest.approx(7.955)


# covers: AC-17, the pairing is on the case, not on the row order of a file
def test_differences_are_matched_on_the_case(world):
    assert world.run() == 0
    expected = (world.folder() / "summary.csv").read_bytes()
    path = world.eval_dir("e3", "val") / "metrics.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    assert world.run("val", "--rerun-reason", "row order") == 0
    assert (world.folder() / "summary.csv").read_bytes() == expected


# covers: AC-17
def test_two_runs_are_byte_identical(world):
    assert world.run() == 0
    first = world.folder()
    moved = world.root / "first"
    first.rename(moved)
    assert world.run() == 0
    for name in ("summary.csv", "patterns.csv", "verdict.txt"):
        assert (first / name).read_bytes() == (moved / name).read_bytes(), name


# covers: AC-18
def test_verdict_is_the_exact_template(world):
    assert world.run() == 0
    p = f"{3 * wilcoxon_all_positive(N):.6g}"
    assert (world.folder() / "verdict.txt").read_text().splitlines() == [
        f"{h} {g}: supported ({c}, mean_diff={d}, p_holm={p})"
        for h, c, d in (("H2a", "E3-E1", "+0.024500"), ("H2b", "E4-E1", "+0.063500"),
                        ("H2c", "E4-E3", "+0.039000"))
        for g in REGIONS]


# covers: AC-18, the direction, the threshold and a tie
def test_verdict_rule_on_plain_rows():
    def rows_for(mean_diff, p_holm):
        return [{"metric": "dice", "level": "incomplete", "region": g, "comparison": c,
                 "mean_diff": mean_diff, "p_holm": p_holm}
                for g in REGIONS for c in compare.HYPOTHESES.values()]

    assert compare.verdict(rows_for(0.1, 1.2e-05), 0.05)[0] == (
        "H2a WT: supported (E3-E1, mean_diff=+0.100000, p_holm=1.2e-05)")
    assert compare.verdict(rows_for(-0.001234, 1.2e-05), 0.05)[8] == (
        "H2c ET: not supported (E4-E3, mean_diff=-0.001234, p_holm=1.2e-05)")
    assert compare.verdict(rows_for(0.1, 0.05), 0.05)[0].startswith("H2a WT: not supported")
    assert compare.verdict(rows_for(0.0, 1.0), 0.05)[3] == (
        "H2b WT: not supported (E4-E1, mean_diff=+0.000000, p_holm=1)")


# covers: AC-18, order of the nine lines, and only Dice decides
def test_verdict_order_ignores_hd95_and_the_levels(world):
    assert world.run() == 0
    lines = (world.folder() / "verdict.txt").read_text().splitlines()
    assert [ln.split(":")[0] for ln in lines] == [
        f"{h} {g}" for h in ("H2a", "H2b", "H2c") for g in REGIONS]
    assert all(": supported" in ln for ln in lines)


# covers: AC-18, a tie on all differences is not supported, end to end
def test_all_zero_differences_are_not_supported(world):
    world.dice["e3"] = plain
    world.dice["e4"] = plain
    for m in ("e3", "e4"):
        shutil.rmtree(world.eval_dir(m, "val"))
        world.write_eval(m, "val")
    assert world.run() == 0
    r = row(world.folder(), "summary", metric="dice", region="WT", level="incomplete",
            comparison="E4-E3")
    assert (float(r["p_raw"]), float(r["p_holm"]), int(r["n_nonzero"])) == (1, 1, 0)
    assert all("not supported" in ln for ln in
               (world.folder() / "verdict.txt").read_text().splitlines())


# covers: AC-17, Compare run.json keys, Value sourcing
def test_run_record_holds_the_spec_keys_and_sources(world):
    world.edit(world.runs["e4"] / "run.json", resumes=[{"from_iteration": 5}])
    assert world.run() == 0
    rec = json.loads((world.folder() / "run.json").read_text())
    assert tuple(rec) == compare.RUN_JSON_KEYS
    e1_eval = json.loads((world.eval_dir("e1", "val") / "run.json").read_text())
    assert rec["compare_id"] == "compare-val" and rec["status"] == "ok" and rec["fold"] == "val"
    assert rec["n_cases"] == N
    assert (rec["e1_run_id"], rec["e3_run_id"], rec["e4_run_id"]) == tuple(RUN_IDS.values())
    assert rec["e1_eval_config_hash"] == e1_eval["config_hash"]
    assert rec["protocol_hash"] == "p" and rec["splits_sha256"] == "s"
    assert rec["n_resumes"] == {"e1": 0, "e3": 0, "e4": 1}
    assert rec["versions_equal"] is True
    assert (rec["seed"], rec["alpha"]) == (42, 0.05)
    assert rec["rerun_reason"] is None and rec["superseded"] is None
    assert rec["wall_time_s"] >= 0
    assert not list(world.out.glob(".compare*"))


# covers: Value sourcing, alpha comes from the E1 eval folder's config.toml
def test_alpha_is_read_from_the_e1_eval_config(world):
    d = world.eval_dir("e1", "val")
    cfg = load_config(d / "config.toml")
    cfg["eval"]["alpha"] = 1e-9
    for m in compare.MODELS:
        e = world.eval_dir(m, "val")
        dump_config(cfg, e / "config.toml")
        world.edit(e / "run.json", config_hash=config_hash(cfg))
    assert world.run() == 0
    assert json.loads((world.folder() / "run.json").read_text())["alpha"] == 1e-9
    assert all("not supported" in ln for ln in
               (world.folder() / "verdict.txt").read_text().splitlines())


# covers: AC-16
def test_compare_hash_is_the_four_files_in_order():
    h = hashlib.sha256()
    for name in ("experiments/0005-strategies/compare.py", "experiments/common/strategies.py",
                 "experiments/common/model.py", "experiments/0004-missing-eval/evaluate.py"):
        h.update((REPO / name).read_bytes())
    assert compare.compare_hash() == h.hexdigest()


# covers: AC-16, a change to any one of the four files is seen
@pytest.mark.parametrize("index", range(4))
def test_each_of_the_four_files_changes_the_hash(tmp_path, monkeypatch, index):
    for name in compare.COMPARE_HASH_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(compare, "REPO", tmp_path)
    before = compare.compare_hash()
    with open(tmp_path / compare.COMPARE_HASH_FILES[index], "a") as f:
        f.write("x")
    assert compare.compare_hash() != before


def assert_refused(world, capsys, message, fold="val", *extra):
    assert world.run(fold, *extra) == 2
    assert message in capsys.readouterr().out
    assert not world.folder(fold).exists()
    assert not list(world.out.glob(".compare*"))


def set_run(key, value):
    return lambda w: w.edit(w.runs["e3"] / "run.json", **{key: value})


def set_eval(key, value):
    return lambda w: w.edit(w.eval_dir("e4", "val") / "run.json", **{key: value})


def edit_config(w):
    cfg = load_config(w.runs["e3"] / "config.toml")
    cfg["train"]["iterations"] = 1
    dump_config(cfg, w.runs["e3"] / "config.toml")


def swap_strategies(w):
    a, b = w.runs["e3"] / "config.toml", w.runs["e4"] / "config.toml"
    held = a.read_text()
    a.write_text(b.read_text())
    b.write_text(held)


def drop_strategy(w):
    cfg = load_config(w.runs["e3"] / "config.toml")
    del cfg["strategy"]
    dump_config(cfg, w.runs["e3"] / "config.toml")


def other_cases(w):
    """A consistent eval on another case set: only the case set rule can see it, because
    metrics_val.csv of E3 moves with it."""
    d = w.eval_dir("e3", "val")
    other = [*VAL[:-1], "BraTS-GLI-00199-000"]
    VAL_INDEX["BraTS-GLI-00199-000"] = 9
    write_metrics(d / "metrics.csv", other, w.dice["e3"], w.hd95["e3"])
    with open(w.runs["e3"] / "metrics_val.csv", "w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["case", "region", "dice"])
        out.writerows((c, g, w.dice["e3"](c, 1, g)) for c in other for g in REGIONS)


def blank_dice(w):
    write_metrics(w.eval_dir("e3", "val") / "metrics.csv", VAL,
                  lambda c, p, g: "" if p == 2 else w.dice["e3"](c, p, g), w.hd95["e3"])


def infinite_hd95(w):
    write_metrics(w.eval_dir("e3", "val") / "metrics.csv", VAL, w.dice["e3"],
                  lambda c, p, g: math.inf if p == 2 else w.hd95["e3"](c, p, g))


GUARDS = {
    "run not ok": (set_run("status", "failed"), "e3 run: status is 'failed'"),
    "quick run": (set_run("quick", True), "e3 run: a quick run"),
    "swapped strategies": (swap_strategies, "e3 run: strategy is 'presence'"),
    "no strategy": (drop_strategy, "e3 run: strategy is None"),
    "config differs": (edit_config, "e3 run: config differs from E1 outside [strategy]"),
    "train cases differ": (set_run("train_cases", ["x"]), "e3 run: train_cases differs"),
    "val cases differ": (set_run("val_cases", ["x"]), "e3 run: val_cases differs"),
    "seed differs": (set_run("seed", 43), "e3 run: seed differs"),
    "gpu differs": (set_run("gpu", "P100"), "e3 run: gpu differs"),
    "cache differs": (set_run("cache_hash", "d"), "e3 run: cache_hash differs"),
    "eval not ok": (set_eval("status", "failed"), "e4 eval: status is 'failed'"),
    "quick eval": (set_eval("quick", True), "e4 eval: a quick run"),
    "wrong fold": (set_eval("fold", "test"), "e4 eval: fold is 'test'"),
    "n_cases differs": (set_eval("n_cases", 9), "e4 eval: n_cases differs"),
    "protocol differs": (set_eval("protocol_hash", "q"), "e4 eval: protocol_hash differs"),
    "eval config differs": (set_eval("config_hash", "q"), "e4 eval: config_hash differs"),
    "splits differ": (set_eval("splits_sha256", "q"), "e4 eval: splits_sha256 differs"),
    "eval gpu differs": (set_eval("gpu", "P100"), "e4 eval: gpu differs"),
    "other case set": (other_cases, "e3 eval: case set differs from E1"),
    "blank dice": (blank_dice, "e3 eval: 30 row(s) with no Dice or HD95"),
    "infinite hd95": (infinite_hd95, "e3 eval: 30 infinite Dice or HD95 value(s)"),
}


# covers: AC-15
@pytest.mark.parametrize("name", GUARDS)
def test_each_guard_exits_2_and_writes_nothing(world, capsys, name):
    change, message = GUARDS[name]
    change(world)
    assert world.run() == 2
    assert message in capsys.readouterr().out
    assert not world.out.exists()
    VAL_INDEX.pop("BraTS-GLI-00199-000", None)


# covers: AC-15, every failed rule is printed, not only the first
def test_every_failed_rule_is_printed(world, capsys):
    set_run("gpu", "P100")(world)
    set_eval("protocol_hash", "q")(world)
    assert world.run() == 2
    out = capsys.readouterr().out
    assert "e3 run: gpu differs" in out and "e4 eval: protocol_hash differs" in out


# covers: AC-15, check_rules runs on each eval
def test_a_broken_tie_out_is_refused(world, capsys):
    (world.runs["e3"] / "metrics_val.csv").unlink()
    assert world.run() == 2
    assert "e3 eval: " in capsys.readouterr().out
    assert not world.out.exists()


@pytest.mark.parametrize("name", ["run.json", "config.toml"])
def test_a_run_file_that_is_missing_exits_2(world, capsys, name):
    (world.runs["e4"] / name).unlink()
    assert world.run() == 2
    assert f"e4 run: {world.runs['e4'] / name} missing" in capsys.readouterr().out


def test_a_missing_eval_folder_exits_2(world, capsys):
    shutil.rmtree(world.eval_dir("e3", "val"))
    assert world.run() == 2
    assert "run.json missing" in capsys.readouterr().out
    assert not world.out.exists()


def finish_val(world):
    assert world.run() == 0
    world.open_test()


# covers: AC-16, happy path on test, including the check against the test fold's case list
def test_test_comparison_runs_after_val(world):
    finish_val(world)
    assert world.run("test") == 0
    rec = json.loads((world.folder("test") / "run.json").read_text())
    assert (rec["fold"], rec["versions_equal"]) == ("test", True)
    assert rec["compare_hash"] == json.loads(
        (world.folder() / "run.json").read_text())["compare_hash"]


# covers: AC-16
def test_test_comparison_needs_the_val_comparison(world, capsys):
    world.open_test()
    assert_refused(world, capsys, "compare.py --fold val first", "test")


# covers: AC-16
def test_stale_compare_hash_is_refused(world, capsys):
    finish_val(world)
    world.edit(world.folder() / "run.json", compare_hash="old")
    assert_refused(world, capsys, "compare_hash differs", "test")


@pytest.mark.parametrize(("key", "message"), [("e3_run_id", "e3_run_id differs"),
                                              ("status", "status is 'x'"),
                                              ("fold", "val compare: fold is 'x'"),
                                              ("protocol_hash", "protocol_hash differs"),
                                              ("splits_sha256", "splits_sha256 differs")])
def test_val_comparison_of_other_runs_is_refused(world, capsys, key, message):
    finish_val(world)
    world.edit(world.folder() / "run.json", **{key: "x"})
    assert_refused(world, capsys, message, "test")


# covers: AC-16
def test_test_case_set_is_checked_against_the_splits(world, capsys):
    finish_val(world)
    d = world.eval_dir("e1", "test")
    VAL_INDEX["BraTS-GLI-00199-000"] = 9
    write_metrics(d / "metrics.csv", [*TEST[:-1], "BraTS-GLI-00199-000"], plain, world.hd95["e1"])
    assert_refused(world, capsys, "case set differs from the test fold", "test")
    VAL_INDEX.pop("BraTS-GLI-00199-000", None)


# covers: AC-15, the laptop needs the committed splits.csv
def test_missing_splits_file_exits_2_on_test(world, capsys):
    finish_val(world)
    world.splits.unlink()
    assert_refused(world, capsys, "cannot read", "test")


# covers: AC-16, a difference warns and is recorded, it never refuses
@pytest.mark.parametrize("target", ["test eval of e3", "val eval of e4"])
def test_versions_differ_warns_and_does_not_refuse(world, capsys, target):
    finish_val(world)
    d = world.eval_dir("e3", "test") if "test" in target else world.eval_dir("e4", "val")
    world.edit(d / "run.json", versions={"torch": "9"})
    assert world.run("test") == 0
    assert "WARNING: versions differ" in capsys.readouterr().out
    assert json.loads((world.folder("test") / "run.json").read_text())["versions_equal"] is False


# covers: AC-17
def test_finished_folder_is_refused_and_kept(world, capsys):
    assert world.run() == 0
    before = (world.folder() / "run.json").read_bytes()
    assert world.run() == 2
    assert "give --rerun-reason" in capsys.readouterr().out
    assert (world.folder() / "run.json").read_bytes() == before


# covers: AC-17
def test_rerun_reason_supersedes_and_records(world):
    assert world.run() == 0
    assert world.run("val", "--rerun-reason", "code changed") == 0
    moved = sorted(world.out.glob("compare-val.superseded-*"))
    assert len(moved) == 1 and (moved[0] / "summary.csv").is_file()
    rec = json.loads((world.folder() / "run.json").read_text())
    assert rec["rerun_reason"] == "code changed" and rec["superseded"] == moved[0].name


# covers: AC-17, a reason never starts a fresh folder
def test_rerun_reason_with_no_finished_folder_is_refused(world, capsys):
    assert world.run("val", "--rerun-reason", "no") == 2
    assert "not one" in capsys.readouterr().out
    assert not world.out.exists()


# covers: State transitions, a compare folder is finished or it does not exist
def test_an_unfinished_folder_is_refused_even_with_a_reason(world, capsys):
    world.folder().mkdir(parents=True)
    assert world.run("val", "--rerun-reason", "x") == 2
    assert "not finished" in capsys.readouterr().out


# covers: State transitions, a crash while writing leaves the finished folder as it was
def test_a_crash_while_writing_keeps_the_finished_folder(world, monkeypatch):
    assert world.run() == 0
    before = (world.folder() / "summary.csv").read_bytes()

    def boom(*a):
        raise OSError("disk full")

    monkeypatch.setattr(compare, "write_csv", boom)
    with pytest.raises(OSError):
        world.run("val", "--rerun-reason", "x")
    assert (world.folder() / "summary.csv").read_bytes() == before
    monkeypatch.undo()
    assert world.run("val", "--rerun-reason", "x") == 0
    assert not list(world.out.glob(".compare*"))


@pytest.fixture
def compared(world):
    assert world.run() == 0
    return world


def tables(world, tmp_path, fold="val"):
    out = tmp_path / "tables"
    assert make_table.main(["--compare", str(world.folder(fold)), "--out-dir", str(out)]) == 0
    return {n: (out / f"0005-strategies-{fold}-{n}.tex").read_text()
            for n in ("summary", "dice", "hd95")}


# covers: AC-19
def test_tables_have_the_header_and_caption(compared, tmp_path):
    rec = json.loads((compared.folder() / "run.json").read_text())
    for tex in tables(compared, tmp_path).values():
        assert "E1 run-e1, E3 run-e3, E4 run-e4" in tex.splitlines()[0]
        assert f"% git {rec['git_commit']}, seed 42" in tex.splitlines()[1]
        assert r"\caption{Development fold, not a test result." in tex
        assert f"n = {N}" in tex and "seed 42" in tex and "val fold" in tex
        assert "two sided" in tex and "in either direction" in tex


# covers: AC-19
def test_test_captions_start_with_the_test_note(compared, tmp_path):
    compared.open_test()
    assert compared.run("test") == 0
    for tex in tables(compared, tmp_path, "test").values():
        assert r"\caption{Test fold, opened once." in tex


# covers: AC-19
def test_summary_table_cells(compared, tmp_path):
    tex = tables(compared, tmp_path)["summary"]
    line = next(ln for ln in tex.splitlines() if ln.startswith("Dice & WT"))
    assert line.startswith("Dice & WT & +0.025$^{*}$ & [") and "+0.039$^{*}$ & [" in line
    assert sum(ln.startswith(("Dice &", "HD95 (mm) &")) for ln in tex.splitlines()) == 6
    hd95 = next(ln for ln in tex.splitlines() if ln.startswith("HD95 (mm) & WT"))
    assert hd95.startswith("HD95 (mm) & WT & -1.0$^{*}$ & [")


# covers: AC-19, the committed stub inputs exactly the three files
def test_preview_stub_inputs_the_three_tables():
    stub = (REPO / "writing" / "tracer" / "e3e4-preview.tex").read_text()
    for name in ("summary", "dice", "hd95"):
        assert rf"\input{{../tables/0005-strategies-val-{name}.tex}}" in stub


# covers: AC-19, Dice shows the mean of each model, a star only against E1, and none on a tie
def test_dice_table_cells(compared, tmp_path):
    tex = tables(compared, tmp_path)["dice"]
    lines = tex.splitlines()
    first = next(ln for ln in lines if ln.startswith("1 & "))
    assert first.startswith("1 & T1+T1ce+T2+FLAIR & 0.545 & 0.545 & 0.545 &")
    assert "$^{*}$" not in first
    assert sum(ln.split(" & ")[0].isdigit() for ln in lines) == 15
    # 10 cases over 15 patterns: Holm gives 0.076, above 0.05, so no star
    assert next(ln for ln in lines if ln.startswith("2 & ")).split(" & ")[2:5] == [
        "0.445", "0.469", "0.508"]


# covers: AC-19, Value sourcing: the star reads alpha from the compare folder's run.json
def test_pattern_star_follows_alpha_and_sits_on_e3_and_e4_only(compared, tmp_path):
    compared.edit(compared.folder() / "run.json", alpha=0.1)
    line = next(ln for ln in tables(compared, tmp_path)["dice"].splitlines()
                if ln.startswith("2 & "))
    assert line.split(" & ")[2:5] == ["0.445", "0.469$^{*}$", "0.508$^{*}$"]


# covers: AC-19, HD95 shows the median
def test_hd95_table_uses_the_median(compared, tmp_path):
    r = row(compared.folder(), "patterns", metric="hd95_mm", region="WT", pattern_id=2,
            comparison="E3-E1")
    first = next(ln for ln in tables(compared, tmp_path)["hd95"].splitlines()
                 if ln.startswith("2 & "))
    assert first.split(" & ")[2] == f"{float(r['median_second']):.1f}"


def refuse_missing_run(world):
    (world.folder() / "run.json").unlink()


def refuse_status(world):
    world.edit(world.folder() / "run.json", status="failed")


def refuse_missing_csv(world):
    (world.folder() / "patterns.csv").unlink()


def refuse_row_count(world):
    path = world.folder() / "summary.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")


# covers: AC-19
@pytest.mark.parametrize("change", [refuse_missing_run, refuse_status, refuse_missing_csv,
                                    refuse_row_count])
def test_each_table_refusal_leaves_older_tables_untouched(compared, tmp_path, change):
    before = tables(compared, tmp_path)
    change(compared)
    with pytest.raises(SystemExit) as e:
        make_table.main(["--compare", str(compared.folder()), "--out-dir",
                         str(tmp_path / "tables")])
    assert str(e.value).startswith("ERROR:")
    after = {n: (tmp_path / "tables" / f"0005-strategies-val-{n}.tex").read_text() for n in before}
    assert after == before
