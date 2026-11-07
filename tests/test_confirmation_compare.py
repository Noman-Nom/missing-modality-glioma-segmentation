"""Spec 0006: compare_seeds.py on toy run and eval folders with known answers. Covers AC-9 to AC-11,
AC-13, and the H2 part of AC-24. Nothing here trains or scores a model, and no number here is a
result. Each test is one rule, stated in its name, shown on 12 toy folders. Also covers AC-7 and
AC-12 to AC-22, AC-24 to AC-26, AC-28 and AC-29 to AC-31."""
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common.cache import sha256_file
from common.data import REGIONS
from common.evaluation import (
    LITERATURE_GATE,
    METRIC_COLUMNS,
    PREREG_TAG,
    PROTOCOL_FILES,
    protocol_hash,
)
from common.missing import PATTERNS
from common.runlog import config_hash, dump_config, load_config
from conftest import write_splits

REPO = Path(__file__).resolve().parents[1]
E1_CONFIG = REPO / "experiments" / "0003-baseline" / "config.toml"
E2_CONFIG = REPO / "experiments" / "0004-missing-eval" / "config.toml"
N = 10
VAL = [f"BraTS-GLI-001{i:02d}-000" for i in range(N)]
TEST = [f"BraTS-GLI-002{i:02d}-000" for i in range(N)]
TRAIN = ["BraTS-GLI-00301-000", "BraTS-GLI-00302-000"]
INDEX = {c: i for i, c in enumerate(VAL + TEST)} | {c: i for i, c in enumerate(TEST)}
VERSIONS = {"torch": "1", "monai": "2"}
PROTOCOL = "p"
# compare.py of spec 0005, as reviewed. It stays unchanged, for val only (spec 0006, AC-21)
OLD_COMPARE_SHA = "33c51fd3950781e74d174554c4969846f06b42c2db285d6b87145ca8bbe85001"
GOOD_GATE = "status: pass\ndate: 2027-01-15\npapers_read: 50\ncai_status: x\ncitation_search: y\n"


def load(folder, name):
    path = REPO / "experiments" / folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"conf_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = load("0007-confirmation", "compare_seeds")
rq1 = load("0007-confirmation", "rq1_seeds")
old_compare = load("0005-strategies", "compare")


def base(case, pattern):
    return 0.5 + INDEX[case] / 100 - (pattern > 1) * 0.1


def toy_dice(arm, seed):
    """The toy Dice of one model. An arm beats E1 on the incomplete patterns by a known amount that
    grows with the case, and each seed moves it by a known step that averages to zero. The spread
    differs per arm so that no two subjects share a difference: a mean of three floats moves by
    rounding noise, and exact ties in d would let that noise pick the Wilcoxon ranks."""
    gains = {"e1": (0.0, 0.0), "e3": (0.02, 0.001), "e4": (0.05, 0.003), "e3b": (0.03, 0.002)}
    amount, spread = gains[arm]
    step = (seed - 43) * 0.01
    return lambda c, p, g: base(c, p) + (p > 1) * (amount + spread * INDEX[c] + step)


def toy_hd95(arm, seed):
    return lambda c, p, g: 10.0 - INDEX[c] / 100 * (arm != "e1") - (seed - 43) * 0.5


def true_ml(case):
    return 20.0 + INDEX[case]


def toy_bias(arm, seed):
    """Predicted minus true volume: a known mean per slot, and a spread that sums to zero."""
    mean = {"e1": 1.0, "e3": 2.0, "e4": 3.0, "e3b": 4.0}[arm] + (seed - 43)
    return lambda c, p: mean + (INDEX[c] - (N - 1) / 2) * 0.2


def write_metrics(path, dice, hd95, bias=lambda c, p: 0.0, cases=VAL):
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
                                "pred_empty": False, "pred_ml": true_ml(c) + bias(c, p),
                                "true_ml": true_ml(c)})


@pytest.fixture(scope="session")
def tagged_repo(tmp_path_factory):
    """A tiny real git repo, in order: a commit before the tag, the tagged one, `after` with a
    passing gate, `failed` whose gate says fail, and `moved`, where a sealed file changed and the
    gate passes again, with a second tag on it."""
    repo = tmp_path_factory.mktemp("repo")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
             "-c", "commit.gpgsign=false", *args],
            check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "before")
    before = git("rev-parse", "HEAD")
    sealed = repo / cs.SEALED_FILES[0]
    gate = repo / LITERATURE_GATE
    gate.parent.mkdir(parents=True)
    sealed.parent.mkdir(parents=True)

    def commit(name):
        git("add", "-A")
        git("commit", "-q", "--allow-empty", "-m", name)
        return git("rev-parse", "HEAD")

    sealed.write_text("sealed")
    commit("tagged")
    git("tag", "-a", PREREG_TAG, "-m", "tag")
    gate.write_text(GOOD_GATE)
    after = commit("after")
    gate.write_text(GOOD_GATE.replace("status: pass", "status: fail"))
    failed = commit("failed")
    gate.write_text(GOOD_GATE)
    sealed.write_text("changed")
    moved = commit("moved")
    git("tag", "-a", "prereg-v2", "-m", "amended")
    return repo, before, after, failed, moved


class World:
    """All 12 finished runs and evals on disk, in the spec's layout, with a filled manifest."""

    def __init__(self, root, commit):
        self.root, self.commit = root, commit
        self.splits = root / "splits.csv"
        write_splits(self.splits, [(c, "train", 0, False) for c in TRAIN]
                     + [(c, "val", 0, False) for c in VAL] + [(c, "test", 0, False) for c in TEST])
        self.evals, self.out = root / "evals", root / "out"
        self.dice = {s: toy_dice(*s) for s in cs.slots()}
        self.hd95 = {s: toy_hd95(*s) for s in cs.slots()}
        self.manifest = root / "runs.toml"
        self.paths = {s: f"results/runs/{s[0]}-{s[1]}" for s in cs.slots()}
        self.write_manifest()
        for s in cs.slots():
            self.write_run(s)
            self.write_eval(s)

    def write_manifest(self):
        text = ""
        for arm in cs.ARMS:
            text += f"[{arm}]\n" + "".join(
                f'seed{s} = "{self.paths[arm, s]}"\n' for s in cs.SEEDS if (arm, s) in self.paths)
        self.manifest.write_text(text + "[replaced]\n")

    def run_dir(self, s):
        return self.root / self.paths[s]

    def eval_dir(self, s, fold="val"):
        return self.evals / f"{s[0]}-{s[1]}-{fold}"

    def write_run(self, s):
        arm, seed = s
        run = self.run_dir(s)
        run.mkdir(parents=True)
        cfg = load_config(E1_CONFIG)
        cfg["seed"] = seed
        cfg["data"]["splits_csv"] = str(self.splits)
        if cs.STRATEGIES[arm]:
            cfg["strategy"] = {"name": cs.STRATEGIES[arm]}
        if arm == "e3b":
            cfg["strategy"]["complete_share"] = 0.5
        dump_config(cfg, run / "config.toml")
        (run / "run.json").write_text(json.dumps(
            {"run_id": f"{arm}-{seed}", "status": "ok", "quick": False, "train_cases": TRAIN,
             "val_cases": VAL, "seed": seed, "gpu": "T4", "cache_hash": "c", "resumes": [],
             "best_iteration": 1}))
        (run / "checkpoints").mkdir()
        (run / "checkpoints" / "best.pt").write_text(arm)
        with open(run / "metrics_val.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["case", "region", "dice"])
            w.writerows((c, g, self.dice[s](c, 1, g)) for c in VAL for g in REGIONS)

    def write_eval(self, s, fold="val"):
        d = self.eval_dir(s, fold)
        d.mkdir(parents=True)
        saved = {"eval": {**load_config(E2_CONFIG)["eval"], "quick": False}}
        dump_config(saved, d / "config.toml")
        (d / "run.json").write_text(json.dumps(
            {"status": "ok", "quick": False, "fold": fold, "n_cases": N, "gpu": "T4",
             "source_run_id": f"{s[0]}-{s[1]}", "source_seed": s[1], "protocol_hash": PROTOCOL,
             "splits_sha256": sha256_file(self.splits),
             "config_hash": config_hash(load_config(d / "config.toml")), "versions": VERSIONS,
             "source_best_iteration": 1,
             "source_best_sha256": sha256_file(self.run_dir(s) / "checkpoints" / "best.pt"),
             "git_commit": self.commit}))
        write_metrics(d / "metrics.csv", self.dice[s], self.hd95[s], toy_bias(*s),
                      VAL if fold == "val" else TEST)

    def open_test(self):
        for s in cs.slots():
            self.write_eval(s, "test")

    def edit(self, path, **fields):
        rec = json.loads(Path(path).read_text())
        Path(path).write_text(json.dumps(rec | fields))

    def edit_config(self, s, change):
        path = self.run_dir(s) / "config.toml"
        cfg = load_config(path)
        change(cfg)
        dump_config(cfg, path)

    def args(self, fold="val"):
        return ["--runs", str(self.manifest), "--fold", fold, "--evals", str(self.evals),
                "--out", str(self.out), "--root", str(self.root)]

    def run(self, fold="val", *more):
        return cs.main([*self.args(fold), *more])

    def preflight(self):
        return self.run("test", "--preflight")

    def folder(self):
        return self.out / "compare-val"

    def summary(self):
        return pd.read_csv(self.folder() / "summary.csv")


@pytest.fixture
def world(tmp_path, tagged_repo, monkeypatch):
    repo, _, after, *_ = tagged_repo
    monkeypatch.setattr(cs, "REPO", repo)
    monkeypatch.setattr(cs, "EXPECTED_PROTOCOL_HASH", PROTOCOL)
    monkeypatch.setattr(cs, "protocol_hash", lambda: PROTOCOL)
    monkeypatch.setattr(cs, "git_info", lambda: (after, False))
    # the gate inside evaluation.py, which keeps its own paths and its own view of the tag
    w = World(tmp_path, after)
    (tmp_path / "test-gate.md").write_text(GOOD_GATE)
    monkeypatch.setattr("common.evaluation.GATE_FILE", tmp_path / "test-gate.md")
    monkeypatch.setattr("common.evaluation.COMPARE_VAL", w.folder())
    monkeypatch.setattr("common.evaluation.descends_from_tag", lambda *a: True)
    return w


def primary(world, comparison, region, metric="dice"):
    t = world.summary()
    return t[(t["family"] == "H2") & (t["metric"] == metric) & (t["region"] == region)
             & (t["level"] == "incomplete") & (t["comparison"] == comparison)].iloc[0]


# covers: AC-13, AC-24 (known answer by hand)
def test_h2_primary_is_the_known_gain_on_the_averaged_values(world):
    assert world.run() == 0
    r = primary(world, "E3-E1", "WT")
    # the seed steps average to zero, so d is 0.02 + 0.001 * case on every incomplete pattern
    assert r["n"] == N
    assert r["mean_diff"] == pytest.approx(0.02 + 0.001 * (N - 1) / 2, abs=1e-6)
    assert r["median_diff"] == pytest.approx(0.02 + 0.001 * (N - 1) / 2, abs=1e-6)
    assert r["ci_level"] == 95


# covers: AC-11
def test_the_unit_is_the_subject_not_subject_times_seed(world):
    assert world.run() == 0
    assert set(world.summary()["n"]) == {N}
    r = primary(world, "E3-E1", "ET")
    assert r["n_nonzero"] == N


def test_averaging_is_the_mean_of_the_three_seeds_per_subject():
    def frame(value):
        return pd.DataFrame({"case": ["a", "b"], "region": ["WT", "WT"], "pattern_id": [1, 1],
                             "dice": [value, value + 0.3], "hd95_mm": [value, 373.13]})

    frames = {(arm, s): frame(0.1 * i) for arm in cs.ARMS
              for i, s in enumerate(cs.SEEDS, start=1)}
    got = cs.averaged(frames)["e3"]
    assert got["dice"].tolist() == pytest.approx([0.2, 0.5])
    assert got["hd95_mm"].tolist() == pytest.approx([0.2, 373.13])  # the empty rule stays in


def test_a_seed_that_differs_moves_the_arm_by_a_third(world):
    assert world.run() == 0
    before = primary(world, "E3-E1", "WT")["mean_diff"]
    s = ("e3", 44)
    (world.eval_dir(s) / "metrics.csv").unlink()
    old = world.dice[s]
    write_metrics(world.eval_dir(s) / "metrics.csv",
                  lambda c, p, g: old(c, p, g) + (p > 1) * 0.03, world.hd95[s])
    world.folder().rename(world.out / "moved")
    assert world.run() == 0
    assert primary(world, "E3-E1", "WT")["mean_diff"] == pytest.approx(before + 0.01, abs=1e-6)


# covers: AC-24, spec 0005 is the reference
def test_three_identical_seeds_give_the_lines_of_the_old_compare(world):
    frames = {s: pd.read_csv(world.eval_dir((s[0], 42)) / "metrics.csv") for s in cs.slots()}
    single = {arm: frames[arm, 42] for arm in ("e1", "e3", "e4")}
    idx = np.random.default_rng(0).integers(0, N, size=(500, N))
    old = [r for m in old_compare.METRICS for g in REGIONS
           for r in old_compare.summary_rows(single, m, g, idx)]
    means = cs.averaged(frames)
    new = [r for m in cs.METRICS for g in REGIONS for r in cs.h2_rows(means, m, g, idx)]
    assert old_compare.verdict(old, 0.05) == cs.h2_verdict(new, 0.05)
    assert len(new) == 72 and len(cs.h2_verdict(new, 0.05)) == 9
    for a, b in zip(old, new, strict=True):
        assert (a["metric"], a["region"], a["level"], a["comparison"]) == (
            b["metric"], b["region"], b["level"], b["comparison"])
        assert b["mean_diff"] == pytest.approx(a["mean_diff"], abs=1e-12)
        assert b["ci_low"] == pytest.approx(a["ci_low"], abs=1e-12)


# covers: AC-18
def test_verdict_is_written_with_eighteen_lines_in_order(world):
    assert world.run() == 0
    lines = (world.folder() / "verdict.txt").read_text().splitlines()
    assert [ln.split(":")[0] for ln in lines] == [
        f"{h} {g}" for h in ("H2a", "H2b", "H2c", "H3a", "H3b", "H4") for g in REGIONS]
    assert "supported (E3-E1, mean_diff=+0." in lines[0]


def test_two_runs_are_byte_identical(world):
    assert world.run() == 0
    first = {f.name: f.read_bytes() for f in world.folder().iterdir()}
    world.folder().rename(world.out / "moved")
    assert world.run() == 0
    assert first == {f.name: f.read_bytes() for f in world.folder().iterdir()}


def test_bootstrap_settings_come_from_the_eval_config(world, monkeypatch):
    seen = {}
    real = cs.bootstrap_indices

    def spy(n, resamples, seed):
        seen.update(n=n, resamples=resamples, seed=seed)
        return real(n, resamples, seed)

    monkeypatch.setattr(cs, "bootstrap_indices", spy)
    assert world.run() == 0
    cfg = load_config(E2_CONFIG)["eval"]
    assert seen == {"n": N, "resamples": cfg["bootstrap_resamples"], "seed": cfg["seed"]}


def test_summary_holds_the_spec_columns(world):
    assert world.run() == 0
    assert list(world.summary().columns) == [
        "family", "metric", "region", "level", "comparison", "n", "mean_first", "mean_second",
        "mean_diff", "median_diff", "ci_low", "ci_high", "ci_level", "n_nonzero", "wilcoxon_stat",
        "p_raw", "p_holm"]
    assert len(world.summary()) == 87


def test_a_non_finite_derived_number_stops_and_writes_nothing(world, monkeypatch, capsys):
    monkeypatch.setattr(cs, "bootstrap_ci", lambda *a, **k: (math.nan, 0.0))
    assert world.run() == 2
    assert "ci_low" in capsys.readouterr().out
    assert not world.out.exists()


def refused(world, capsys, message):
    assert world.run() == 2
    out = capsys.readouterr().out
    assert message in out, out
    assert not world.out.exists()


# Each guard: (name, what to break, the words it must print). Covers: AC-10
S = ("e3", 43)
GUARDS = {
    "missing entry": (lambda w: (w.paths.pop(S), w.write_manifest()), "no entry for e3 seed43"),
    "path outside results": (lambda w: (w.paths.__setitem__(S, "other/x"), w.write_manifest()),
                             "not under results/"),
    "path climbs out": (lambda w: (w.paths.__setitem__(S, "results/../x"), w.write_manifest()),
                        "not under results/"),
    "unknown arm": (lambda w: w.manifest.write_text(w.manifest.read_text() + "[e9]\nseed42='x'\n"),
                    "unknown table [e9]"),
    "repeated run id": (lambda w: w.edit(w.run_dir(S) / "run.json", run_id="e3-42"), "repeats"),
    "seed differs from slot": (lambda w: w.edit(w.run_dir(S) / "run.json", seed=45), "slot says"),
    "config seed differs": (lambda w: w.edit_config(S, lambda c: c.update(seed=45)), "slot says"),
    "failed run": (lambda w: w.edit(w.run_dir(S) / "run.json", status="failed"), "status"),
    "quick run": (lambda w: w.edit(w.run_dir(S) / "run.json", quick=True), "a quick run"),
    "wrong strategy": (lambda w: w.edit_config(S, lambda c: c["strategy"].update(name="presence")),
                       "strategy is 'presence'"),
    "wrong share": (lambda w: w.edit_config(("e3b", 42), lambda c: c["strategy"].update(
        complete_share=0.25)), "complete_share is not 0.5"),
    "config differs within an arm": (lambda w: w.edit_config(
        S, lambda c: c["train"].update(lr=1.0)), "outside seed"),
    "config differs at a seed": (lambda w: w.edit_config(
        S, lambda c: c["train"].update(lr=1.0)), "outside [strategy]"),
    "train cases differ": (lambda w: w.edit(w.run_dir(S) / "run.json", train_cases=TRAIN[:1]),
                           "train_cases differs"),
    "gpu differs": (lambda w: w.edit(w.run_dir(S) / "run.json", gpu="P100"), "gpu differs"),
    "cache differs": (lambda w: w.edit(w.run_dir(S) / "run.json", cache_hash="d"),
                      "cache_hash differs"),
    "eval failed": (lambda w: w.edit(w.eval_dir(S) / "run.json", status="failed"), "status"),
    "eval fold": (lambda w: w.edit(w.eval_dir(S) / "run.json", fold="test"), "fold"),
    "eval cases count": (lambda w: w.edit(w.eval_dir(S) / "run.json", n_cases=9), "n_cases"),
    "eval splits": (lambda w: w.edit(w.eval_dir(S) / "run.json", splits_sha256="z"),
                    "splits_sha256 differs"),
    "eval protocol": (lambda w: w.edit(w.eval_dir(S) / "run.json", protocol_hash="z"),
                      "not the pinned one"),
    "versions on val": (lambda w: w.edit(w.eval_dir(S) / "run.json", versions={"torch": "9"}),
                        "versions differ"),
    "eval has no metrics": (lambda w: (w.eval_dir(S) / "metrics.csv").unlink(), "missing"),
    "eval folder gone": (lambda w: w.eval_dir(S).rename(w.root / "gone"), "missing"),
    "run folder gone": (lambda w: w.run_dir(S).rename(w.root / "gone"), "missing"),
    "finished folder": (lambda w: (w.out / "compare-val").mkdir(parents=True), "exists"),
}


@pytest.mark.parametrize("name", GUARDS)
def test_each_guard_exits_2_and_writes_nothing(world, capsys, name):
    change, message = GUARDS[name]
    change(world)
    before = sorted(p.name for p in world.out.rglob("*")) if world.out.exists() else None
    assert world.run() == 2
    out = capsys.readouterr().out
    assert message in out, out
    assert not list(world.out.glob("compare-val/*"))
    if before is None:
        assert not world.out.exists()


def test_every_failed_rule_is_printed(world, capsys):
    world.edit(world.run_dir(S) / "run.json", status="failed", quick=True)
    assert world.run() == 2
    out = capsys.readouterr().out
    assert "status" in out and "a quick run" in out


def test_invalid_toml_exits_2(world, capsys):
    world.manifest.write_text("[e1\n")
    refused(world, capsys, "not valid TOML")


def test_infinite_hd95_is_refused(world, capsys):
    s = ("e4", 44)
    write_metrics(world.eval_dir(s) / "metrics.csv", world.dice[s], lambda c, p, g: math.inf)
    refused(world, capsys, "infinite")


def test_case_set_must_match(world, capsys):
    s = ("e4", 44)
    t = pd.read_csv(world.eval_dir(s) / "metrics.csv")
    t["case"] = t["case"].str.replace("BraTS-GLI-001", "BraTS-GLI-009")
    t.to_csv(world.eval_dir(s) / "metrics.csv", index=False)
    refused(world, capsys, "case set differs")


def test_commit_before_the_tag_is_refused(world, capsys, tagged_repo):
    world.edit(world.eval_dir(S) / "run.json", git_commit=tagged_repo[1])
    refused(world, capsys, f"does not descend from {PREREG_TAG}")


def test_unknown_commit_is_refused(world, capsys):
    world.edit(world.eval_dir(S) / "run.json", git_commit="0" * 40)
    refused(world, capsys, "does not descend")


def test_no_tag_at_all_is_refused(world, capsys, monkeypatch):
    monkeypatch.setattr(cs, "PREREG_TAG", "prereg-v9")
    refused(world, capsys, "does not descend from prereg-v9")


def test_a_broken_tie_out_is_refused(world, capsys):
    s = ("e4", 42)
    path = world.run_dir(s) / "metrics_val.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "region", "dice"])
        w.writerows((c, g, 0.1) for c in VAL for g in REGIONS)
    refused(world, capsys, "differs from E1")


# covers: AC-9, on the real manifest
def test_the_real_manifest_lists_every_slot_and_the_kept_seed_42_paths():
    errors, dirs = cs.manifest_problems(REPO / "experiments/0007-confirmation/runs.toml", REPO)
    assert all("no entry" in e for e in errors)
    assert dirs["e1", 42].name == "20261103-023817-f9b3afb6"
    assert dirs["e3", 42].name == "20261106-024450-10cfa2cc"
    assert len(errors) + len(dirs) == 12


def test_the_pinned_protocol_hash_is_the_seal():
    assert cs.EXPECTED_PROTOCOL_HASH == protocol_hash()


# ---- H3, H4, per seed view, volume, outputs (spec 0006, build steps 3 to 5) ----

BIG = 40  # subjects in the known answer toys below, so the bootstrap intervals are tight
BIG_IDX = np.random.default_rng(0).integers(0, BIG, size=(4000, BIG))


def toy_means(complete, primary=None):
    """Seed averaged tables for arms given as {arm: d}, the Dice above 0.5 of each of BIG subjects:
    complete on pattern 1, primary on the 14 others. An arm not named sits at 0.5."""
    primary = primary or {}
    rows = {arm: pd.DataFrame(
        [{"case": f"c{i:03d}", "region": g, "pattern_id": p,
          "dice": 0.5 + (complete if p == 1 else primary).get(arm, np.zeros(BIG))[i],
          "hd95_mm": 1.0} for i in range(BIG) for p in PATTERNS for g in REGIONS])
        for arm in cs.ARMS}
    return rows


# covers: AC-14, AC-24, strict comparisons and the order of the checks
@pytest.mark.parametrize(("low", "high", "word"), [
    (-0.01, 0.01, "equivalent"),
    (-0.014, -0.001, "equivalent"),  # below zero, but equivalence is checked first
    (0.001, 0.014, "equivalent"),
    (-0.015, 0.01, "inconclusive"),  # touching the margin exactly is not inside it
    (-0.01, 0.015, "inconclusive"),
    (-0.015, -0.001, "cost"),
    (-0.03, -0.02, "cost"),
    (0.001, 0.015, "gain"),
    (0.02, 0.03, "gain"),
    (-0.02, 0.02, "inconclusive"),
    (-0.02, 0.0, "inconclusive"),  # zero itself is neither a cost nor a gain
    (0.0, 0.02, "inconclusive"),
])
def test_cost_label(low, high, word):
    assert cs.cost_label(low, high) == word


# covers: AC-24
def test_a_mean_below_zero_is_a_cost_though_most_subjects_gained():
    d = np.r_[np.full(30, 0.01), np.full(10, -0.2)]  # 30 of 40 gained
    (row, _) = cs.h3_rows(toy_means({"e3": d}), "WT", BIG_IDX)
    assert row["mean_diff"] < 0 < row["median_diff"]
    assert cs.cost_label(row["ci_low"], row["ci_high"]) == "cost"


# covers: AC-14
def test_h3_rows_use_pattern_1_and_one_95_ci():
    complete = np.linspace(-0.005, 0.0071, BIG)  # uneven, so no two subjects tie in size
    rows = cs.h3_rows(toy_means({"e3": complete}, {"e3": complete + 0.3}), "TC", BIG_IDX)
    assert [r["comparison"] for r in rows] == ["E3-E1", "E4-E1"]
    r = rows[0]
    assert (r["family"], r["metric"], r["level"], r["ci_level"]) == ("H3", "dice", "complete", 95)
    assert r["mean_diff"] == pytest.approx(complete.mean(), abs=1e-9)
    assert (r["ci_low"], r["ci_high"]) == pytest.approx(cs.bootstrap_ci(complete, BIG_IDX, 0.95),
                                                        abs=1e-9)
    assert r["p_raw"] == pytest.approx(cs.paired_wilcoxon(complete)[1], abs=1e-6)
    assert r["p_holm"] is None


# covers: AC-14, AC-18, the template
def test_h3_line_holds_every_field_on_a_known_row():
    summary = [{"family": "H3", "region": g, "comparison": c, "mean_diff": -0.002,
                "median_diff": 0.001, "ci_low": -0.004, "ci_high": 0.0035, "p_raw": 0.25}
               for g in REGIONS for c in ("E3-E1", "E4-E1")]
    per_seed = [{"metric": "dice", "region": g, "level": "complete", "comparison": c,
                 "mean_diff": v} for g in REGIONS for c in ("E3-E1", "E4-E1")
                for v in (-0.01, -0.02, 0.03)]
    lines = cs.h3_verdict(summary, per_seed)
    assert len(lines) == 6
    assert lines[0] == ("H3a WT: equivalent (E3-E1, mean_diff=-0.002000, median_diff=+0.001000, "
                        "ci=[-0.004000, +0.003500], min_margin=0.004000, seeds_agree=2/3, "
                        "p_raw=0.25)")
    assert lines[3].startswith("H3b WT: equivalent (E4-E1,")


def test_seeds_agree_ignores_other_regions_levels_and_metrics():
    other = [{"metric": "hd95_mm", "region": "WT", "level": "complete", "comparison": "E3-E1",
              "mean_diff": 1.0},
             {"metric": "dice", "region": "TC", "level": "complete", "comparison": "E3-E1",
              "mean_diff": 1.0},
             {"metric": "dice", "region": "WT", "level": "incomplete", "comparison": "E3-E1",
              "mean_diff": 1.0}]
    mine = {"metric": "dice", "region": "WT", "level": "complete", "comparison": "E3-E1",
            "mean_diff": 1.0}
    assert cs.seeds_agree([*other, mine], "WT", "E3-E1", 0.5) == 1


def h4_word(complete, primary):
    rows = cs.h4_rows(toy_means({"e3b": complete}, {"e3b": primary}), "WT", BIG_IDX)
    summary = [{**r, "region": g} for g in REGIONS for r in rows]
    return cs.h4_verdict(summary, 0.05)[0].split(":")[1].split(" (")[0].strip()


GAIN = np.linspace(0.01, 0.03, BIG)


# covers: AC-15, both parts must hold
def test_h4_is_supported_with_a_gain_and_no_real_loss_on_the_primary():
    assert h4_word(GAIN, np.linspace(-0.005, 0.005, BIG)) == "supported"


def test_h4_fails_when_the_primary_lower_bound_passes_the_margin():
    assert h4_word(GAIN, np.linspace(-0.04, -0.02, BIG)) == "not supported"


def test_h4_fails_without_a_gain_at_pattern_1():
    assert h4_word(-GAIN, np.linspace(-0.005, 0.005, BIG)) == "not supported"


def test_h4_fails_when_the_mean_gained_but_the_median_did_not():
    d = np.r_[np.full(36, -0.001), np.full(4, 0.2)]
    d = d + np.arange(BIG) * 1e-7  # no exact ties, so the ranks are not rounding noise
    assert d.mean() > 0 > np.median(d)
    assert h4_word(d, np.zeros(BIG)) == "not supported"


def test_h4_fails_when_the_gain_is_not_significant():
    d = np.linspace(-0.05, 0.0505, BIG)
    assert d.mean() > 0 and np.median(d) > 0 and cs.paired_wilcoxon(d)[1] > 0.05
    assert h4_word(d, np.zeros(BIG)) == "not supported"


# covers: AC-15, AC-24, a 90% bound for H4 and a 95% CI for the pattern 1 row
def test_h4_primary_row_holds_the_90_bound():
    primary = np.linspace(-0.05, 0.05, BIG)
    complete_row, primary_row = cs.h4_rows(toy_means({"e3b": GAIN}, {"e3b": primary}), "ET", BIG_IDX)
    assert (complete_row["level"], complete_row["ci_level"]) == ("complete", 95)
    assert (primary_row["level"], primary_row["ci_level"]) == ("incomplete", 90)
    assert primary_row["ci_low"] == pytest.approx(cs.bootstrap_ci(primary, BIG_IDX, 0.90)[0],
                                                  abs=1e-9)
    assert primary_row["ci_low"] > cs.bootstrap_ci(primary, BIG_IDX, 0.95)[0]
    assert complete_row["p_holm"] is None and primary_row["p_holm"] is None


def test_h4_line_is_the_template():
    summary = [{"family": "H4", "region": g, "level": level, "mean_diff": 0.01, "median_diff": 0.02,
                "p_raw": 0.001, "ci_low": -0.01} for g in REGIONS
               for level in ("complete", "incomplete")]
    assert cs.h4_verdict(summary, 0.05)[0] == (
        "H4 WT: supported (E3b-E3, complete_mean_diff=+0.010000, complete_median_diff=+0.020000, "
        "complete_p=0.001, primary_ci90_low=-0.010000)")


def rows_of(t, **fields):
    for k, v in fields.items():
        t = t[t[k] == v]
    return t


# covers: AC-17, Output schemas
def test_summary_families_and_their_rows(world):
    assert world.run() == 0
    t = pd.read_csv(world.folder() / "summary.csv", keep_default_na=False)
    assert t["family"].value_counts().to_dict() == {"H2": 72, "H3": 6, "H4": 6, "descriptive": 3}
    assert t["family"].drop_duplicates().tolist() == ["H2", "H3", "H4", "descriptive"]
    assert set(rows_of(t, family="H3")["ci_level"].astype(int)) == {95}
    assert set(rows_of(t, family="H4", level="incomplete")["ci_level"].astype(int)) == {90}
    assert set(rows_of(t, family="H4", level="complete")["ci_level"].astype(int)) == {95}
    e = rows_of(t, family="descriptive")
    assert set(e["comparison"]) == {"E3b-E1"} and set(e["p_raw"]) == {""}
    assert set(t[t["family"] != "H2"]["p_holm"]) == {""}


# covers: AC-12
def test_per_seed_is_each_seed_alone_with_no_p_value(world):
    assert world.run() == 0
    t = pd.read_csv(world.folder() / "per_seed.csv")
    assert list(t.columns) == cs.PER_SEED_COLUMNS
    assert len(t) == 3 * 2 * 3 * 2 * 5
    assert set(t["comparison"]) == {"E3-E1", "E4-E1", "E4-E3", "E3b-E3", "E3b-E1"}
    assert set(t["level"]) == {"incomplete", "complete"}
    # E1 has no seed step in the toy, and pattern 1 has no gain, so only incomplete rows move
    r = rows_of(t, seed=44, metric="dice", region="WT", level="incomplete", comparison="E3-E1")
    assert r["mean_diff"].item() == pytest.approx(0.02 + 0.001 * (N - 1) / 2, abs=1e-6)
    r = rows_of(t, seed=44, metric="dice", region="WT", level="complete", comparison="E3-E1")
    assert r["mean_diff"].item() == pytest.approx(0.0, abs=1e-9)


# covers: AC-16, by hand: the spread of 0.2 * (i - 4.5) over 10 subjects has ddof 1 sd 0.2 * sd(0..9)
def test_volume_is_per_seed_with_known_limits_of_agreement(world):
    assert world.run() == 0
    t = pd.read_csv(world.folder() / "volume.csv")
    assert list(t.columns) == cs.VOLUME_COLUMNS and len(t) == 180
    assert t["arm"].drop_duplicates().tolist() == ["E1", "E3", "E4", "E3b"]
    assert t["seed"].tolist()[:30] == [42] * 15 + [43] * 15
    assert t["pattern_id"].tolist()[:15] == list(PATTERNS)
    sd = 0.2 * np.std(np.arange(N), ddof=1)
    r = rows_of(t, arm="E3b", seed=44, pattern_id=7)
    assert r["n"].item() == N
    assert r["mean_diff_ml"].item() == pytest.approx(5.0, abs=1e-6)
    assert r["loa_low_ml"].item() == pytest.approx(5.0 - 1.96 * sd, abs=1e-5)
    assert r["loa_high_ml"].item() == pytest.approx(5.0 + 1.96 * sd, abs=1e-5)


# covers: Output schemas, patterns.csv
def test_patterns_holm_over_15_per_comparison(world):
    assert world.run() == 0
    t = pd.read_csv(world.folder() / "patterns.csv")
    assert list(t.columns) == cs.PATTERN_COLUMNS and len(t) == 360
    assert set(t["comparison"]) == {"E3-E1", "E4-E1", "E4-E3", "E3b-E3"}
    one = rows_of(t, metric="dice", region="WT", comparison="E3b-E3").sort_values("pattern_id")
    assert one["p_holm"].to_numpy() == pytest.approx(cs.holm(one["p_raw"].to_numpy()), abs=2e-5)


# covers: AC-17, Output schemas
def test_run_json_holds_the_record_and_comes_with_every_output(world):
    assert world.run() == 0
    assert {f.name for f in world.folder().iterdir()} == {
        "summary.csv", "patterns.csv", "per_seed.csv", "volume.csv", "verdict.txt", "run.json"}
    rec = json.loads((world.folder() / "run.json").read_text())
    assert tuple(rec) == cs.RUN_JSON_KEYS
    assert rec["status"] == "ok" and rec["n_cases"] == N and rec["seeds"] == [42, 43, 44]
    assert rec["equiv_margin"] == 0.015 and rec["prereg_tag"] == PREREG_TAG
    assert rec["run_ids"]["e3b_seed44"] == "e3b-44" and len(rec["run_ids"]) == 12
    assert rec["eval_commits"]["e1_seed42"] == world.commit and len(rec["eval_commits"]) == 12
    assert rec["n_resumes"]["e4_seed43"] == 0 and rec["versions_equal"] is True
    assert rec["compare_hash"] == cs.compare_hash() and rec["alpha"] == 0.05


# covers: AC-28, the hash only here; the refusal comes with build step 6
def test_compare_hash_is_over_the_sealed_files_in_order():
    assert cs.SEALED_FILES[:len(PROTOCOL_FILES)] == PROTOCOL_FILES
    assert cs.SEALED_FILES[len(PROTOCOL_FILES):] == (
        "experiments/common/strategies.py", "experiments/common/model.py",
        "experiments/0004-missing-eval/evaluate.py",
        "experiments/0007-confirmation/compare_seeds.py")
    assert cs.compare_hash() == hashlib.sha256(
        b"".join((REPO / n).read_bytes() for n in cs.SEALED_FILES)).hexdigest()


# covers: AC-17, rerun rules
def test_a_finished_folder_needs_a_reason(world, capsys):
    assert world.run() == 0
    assert world.run() == 2
    assert "give --rerun-reason" in capsys.readouterr().out


def test_a_rerun_keeps_the_old_folder_and_records_the_reason(world):
    assert world.run() == 0
    assert world.run("val", "--rerun-reason", "a stated reason") == 0
    rec = json.loads((world.folder() / "run.json").read_text())
    assert rec["rerun_reason"] == "a stated reason"
    assert (world.out / rec["superseded"] / "run.json").is_file()


def test_a_reason_with_no_finished_folder_is_refused(world, capsys):
    assert world.run("val", "--rerun-reason", "x") == 2
    assert "not one" in capsys.readouterr().out


def test_an_unfinished_folder_is_not_replaced(world, capsys):
    (world.out / "compare-val").mkdir(parents=True)
    assert world.run("val", "--rerun-reason", "x") == 2
    assert "not finished" in capsys.readouterr().out


def test_a_non_finite_volume_stops_and_writes_nothing(world, capsys):
    s = ("e4", 44)
    write_metrics(world.eval_dir(s) / "metrics.csv", world.dice[s], world.hd95[s],
                  lambda c, p: math.inf)
    assert world.run() == 2
    assert "mean_diff_ml" in capsys.readouterr().out
    assert not world.out.exists()


# ---- volume rules in check_rules (AC-7) ----

def edit_volumes(world, s, change):
    path = world.eval_dir(s) / "metrics.csv"
    t = pd.read_csv(path)
    change(t)
    t.to_csv(path, index=False)


# covers: AC-7
@pytest.mark.parametrize(("change", "message"), [
    (lambda t: t.pop("pred_ml"), "no pred_ml or true_ml"),
    (lambda t: t.__setitem__("true_ml", math.nan), "no pred_ml or true_ml"),
    (lambda t: t.__setitem__("pred_ml", -1.0), "negative volume"),
    (lambda t: t.loc.__setitem__((t.index[0], "true_ml"), 99.0), "not constant"),
], ids=["column gone", "empty", "negative", "true_ml moves"])
def test_check_rules_guards_the_volume_columns(world, capsys, change, message):
    edit_volumes(world, S, change)
    refused(world, capsys, message)


# ---- rescore proof (AC-26) ----

check = load("0007-confirmation", "check_confirmation")


@pytest.fixture
def rescore(tmp_path):
    """An old eval, a new eval with the same scores, and a splits file with the audit volumes."""
    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir()
    old.mkdir()
    dice, hd95 = toy_dice("e3", 42), toy_hd95("e3", 42)
    write_metrics(new / "metrics.csv", dice, hd95)
    write_metrics(old / "metrics.csv", dice, hd95)
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"case": VAL, "ml_WT": [true_ml(c) for c in VAL]}).to_csv(splits, index=False)
    return new, old, splits


def nudge(folder, column, by, region="WT"):
    t = pd.read_csv(folder / "metrics.csv")
    t.loc[t.index[t["region"] == region][0], column] += by
    t.to_csv(folder / "metrics.csv", index=False)


def test_rescore_passes_on_the_same_scores(rescore):
    assert check.rescore_problems(*rescore) == []


def test_rescore_tolerates_differences_inside_the_limits(rescore):
    nudge(rescore[0], "dice", 0.00009)
    nudge(rescore[0], "hd95_mm", 0.009)
    assert check.rescore_problems(*rescore) == []


# covers: AC-26, "a changed score on seed 42 fails"
@pytest.mark.parametrize(("column", "by"), [("dice", 0.0002), ("dice", -0.0002),
                                            ("hd95_mm", 0.02), ("hd95_mm", -0.02)])
def test_rescore_fails_past_the_limit_even_on_one_row(rescore, column, by):
    nudge(rescore[0], column, by, region="ET")
    (error,) = check.rescore_problems(*rescore)
    assert column in error and error.startswith("1 row(s)")


def test_rescore_checks_true_volume_against_the_audit_on_wt_only(rescore):
    t = pd.read_csv(rescore[0] / "metrics.csv")
    t.loc[t["region"] == "TC", "true_ml"] += 5.0
    t.to_csv(rescore[0] / "metrics.csv", index=False)
    assert check.rescore_problems(*rescore) == []
    t.loc[(t["region"] == "WT") & (t["case"] == VAL[3]), "true_ml"] += 0.2
    t.to_csv(rescore[0] / "metrics.csv", index=False)
    (error,) = check.rescore_problems(*rescore)
    assert "true_ml of 1 case(s)" in error and VAL[3] in error


def test_rescore_refuses_a_missing_row(rescore):
    t = pd.read_csv(rescore[0] / "metrics.csv")
    t.iloc[1:].to_csv(rescore[0] / "metrics.csv", index=False)
    assert "rows differ" in check.rescore_problems(*rescore)[0]


def test_rescore_refuses_a_missing_pattern_in_both(rescore):
    for folder in rescore[:2]:
        t = pd.read_csv(folder / "metrics.csv")
        t[t["pattern_id"] != 9].to_csv(folder / "metrics.csv", index=False)
    assert "missing" in check.rescore_problems(*rescore)[0]


def test_rescore_refuses_a_case_the_audit_does_not_know(rescore):
    pd.DataFrame({"case": VAL[1:], "ml_WT": [true_ml(c) for c in VAL[1:]]}).to_csv(
        rescore[2], index=False)
    assert "not in" in check.rescore_problems(*rescore)[0]


# ---- tables (AC-29) ----

table_script = load("0007-confirmation", "make_table")
TABLES = ("summary", "cost", "exposure", "patterns-dice", "patterns-hd95", "seeds", "volume")


def tables(world, tmp_path):
    out = tmp_path / "tables"
    return out, table_script.main(["--compare", str(world.folder()), "--out-dir", str(out)])


def test_make_table_writes_the_seven_files(world, tmp_path):
    assert world.run() == 0
    out, code = tables(world, tmp_path)
    assert code == 0
    assert {p.name for p in out.iterdir()} == {
        f"0007-confirmation-val-{name}.tex" for name in TABLES}
    for name in TABLES:
        tex = (out / f"0007-confirmation-val-{name}.tex").read_text()
        assert r"\caption{Development fold, not a test result." in tex
        assert f"n = {N}, seeds 42, 43 and 44." in tex
        assert f"tab:confirmation-val-{name}" in tex
        assert "val fold" in tex


def test_a_test_folder_gets_the_test_caption(world, tmp_path):
    assert world.run() == 0
    world.edit(world.folder() / "run.json", fold="test")
    out, code = tables(world, tmp_path)
    assert code == 0
    tex = (out / "0007-confirmation-test-cost.tex").read_text()
    assert r"\caption{Test fold, opened once." in tex


# covers: AC-29, labels come from verdict.txt, so this script decides nothing
def test_cost_and_exposure_words_are_the_verdicts(world, tmp_path):
    assert world.run() == 0
    verdict = world.folder() / "verdict.txt"
    verdict.write_text(verdict.read_text().replace("H3a WT: gain", "H3a WT: marker").replace(
        "H3a WT: equivalent", "H3a WT: marker").replace("H3a WT: inconclusive", "H3a WT: marker")
        .replace("H3a WT: cost", "H3a WT: marker"))
    out, code = tables(world, tmp_path)
    assert code == 0
    cost = (out / "0007-confirmation-val-cost.tex").read_text()
    assert cost.count("marker") == 1


def test_seeds_table_has_a_row_per_metric_level_region_comparison(world, tmp_path):
    assert world.run() == 0
    out, _ = tables(world, tmp_path)
    tex = (out / "0007-confirmation-val-seeds.tex").read_text()
    body = tex.split(r"\endfoot")[1]
    assert body.count(r"\\") == 2 * 2 * 3 * 5


def test_volume_table_has_a_row_per_arm_and_pattern(world, tmp_path):
    assert world.run() == 0
    out, _ = tables(world, tmp_path)
    tex = (out / "0007-confirmation-val-volume.tex").read_text()
    assert tex.split(r"\endfoot")[1].count(r"\\") == 4 * 15


@pytest.mark.parametrize("name", ["run.json", "summary.csv", "patterns.csv", "per_seed.csv",
                                  "volume.csv", "verdict.txt"])
def test_a_missing_file_stops_and_keeps_older_tables(world, tmp_path, name):
    assert world.run() == 0
    out, _ = tables(world, tmp_path)
    before = {p: p.read_bytes() for p in out.iterdir()}
    (world.folder() / name).unlink()
    with pytest.raises(SystemExit) as e:
        tables(world, tmp_path)
    assert "ERROR" in str(e.value) and name in str(e.value)
    assert before == {p: p.read_bytes() for p in out.iterdir()}


def test_a_run_that_is_not_ok_stops(world, tmp_path):
    assert world.run() == 0
    world.edit(world.folder() / "run.json", status="failed")
    with pytest.raises(SystemExit, match="status"):
        tables(world, tmp_path)
    assert not (tmp_path / "tables").exists()


def test_a_wrong_row_count_stops(world, tmp_path):
    assert world.run() == 0
    t = pd.read_csv(world.folder() / "volume.csv")
    t.iloc[1:].to_csv(world.folder() / "volume.csv", index=False)
    with pytest.raises(SystemExit, match="rows in volume.csv"):
        tables(world, tmp_path)


def test_a_cut_verdict_stops(world, tmp_path):
    assert world.run() == 0
    verdict = world.folder() / "verdict.txt"
    verdict.write_text("\n".join(verdict.read_text().splitlines()[:12]) + "\n")
    with pytest.raises(SystemExit, match="H3"):
        tables(world, tmp_path)


def finish_val(world):
    assert world.run() == 0


def opened(world):
    finish_val(world)
    world.open_test()


def refused_test(world, capsys, message):
    before = sorted(p.name for p in world.out.iterdir())
    assert world.run("test") == 2
    assert message in capsys.readouterr().out
    assert sorted(p.name for p in world.out.iterdir()) == before


def refused_preflight(world, capsys, message):
    before = sorted(p.name for p in world.out.iterdir())
    assert world.preflight() == 2
    assert message in capsys.readouterr().out
    assert sorted(p.name for p in world.out.iterdir()) == before


# covers: AC-20, happy path, the preflight then the test compare
def test_the_preflight_passes_and_writes_nothing(world, capsys):
    finish_val(world)
    before = sorted(p.name for p in world.out.iterdir())
    assert world.preflight() == 0
    assert "preflight PASS" in capsys.readouterr().out
    assert sorted(p.name for p in world.out.iterdir()) == before
    assert not list(world.evals.glob("*-test"))


# covers: AC-17, AC-18, AC-28 on the test fold
def test_the_test_compare_runs_after_val(world):
    opened(world)
    assert world.run("test") == 0
    rec = json.loads((world.out / "compare-test" / "run.json").read_text())
    val = json.loads((world.folder() / "run.json").read_text())
    assert (rec["fold"], rec["versions_equal"]) == ("test", True)
    assert rec["compare_hash"] == val["compare_hash"]
    assert len((world.out / "compare-test" / "verdict.txt").read_text().splitlines()) == 18


def test_a_finished_test_compare_needs_a_reason(world, capsys):
    opened(world)
    assert world.run("test") == 0
    refused_test(world, capsys, "give --rerun-reason")


def test_the_preflight_is_only_for_the_test_fold(world, capsys):
    assert world.run("val", "--preflight") == 2
    assert "only for --fold test" in capsys.readouterr().out
    assert not world.out.exists()


# covers: AC-10, the same rules hold on test
def test_the_test_fold_keeps_the_guards_of_val(world, capsys):
    opened(world)
    s = ("e3b", 43)
    world.edit(world.eval_dir(s, "test") / "run.json", status="running")
    refused_test(world, capsys, "status is 'running'")


def test_a_test_eval_cut_short_is_refused(world, capsys):
    opened(world)
    s = ("e3", 44)
    write_metrics(world.eval_dir(s, "test") / "metrics.csv", world.dice[s], world.hd95[s],
                  cases=TEST[:-1])
    refused_test(world, capsys, "metrics.csv has")


def test_versions_that_differ_warn_on_test_and_are_recorded(world, capsys):
    opened(world)
    world.edit(world.eval_dir(S, "test") / "run.json", versions={"torch": "9"})
    assert world.run("test") == 0
    assert "WARNING: versions differ" in capsys.readouterr().out
    assert not json.loads((world.out / "compare-test" / "run.json").read_text())["versions_equal"]


def test_the_test_compare_needs_the_val_compare(world, capsys):
    world.open_test()
    world.out.mkdir()
    refused_test(world, capsys, "compare_seeds.py --fold val first")


def test_a_val_compare_that_did_not_finish_is_refused(world, capsys):
    opened(world)
    world.edit(world.folder() / "run.json", status="running")
    refused_test(world, capsys, "val compare: status is 'running'")


def test_a_val_compare_of_other_runs_is_refused(world, capsys):
    opened(world)
    rec = json.loads((world.folder() / "run.json").read_text())
    world.edit(world.folder() / "run.json", run_ids=rec["run_ids"] | {"e1_seed42": "other"})
    refused_test(world, capsys, "names other runs")


# covers: AC-28, the Tamper scenario. The margin lives in compare_seeds.py, a sealed file, so a
# changed margin is a changed compare_hash
def test_a_changed_sealed_file_since_val_is_refused(world, capsys, monkeypatch):
    opened(world)
    monkeypatch.setattr(cs, "compare_hash", lambda: "another margin")
    refused_test(world, capsys, "compare_hash differs")


@pytest.mark.parametrize("key", ["protocol_hash", "splits_sha256"])
def test_a_hash_that_differs_from_the_val_compare_is_refused(world, capsys, key):
    opened(world)
    world.edit(world.folder() / "run.json", **{key: "other"})
    refused_test(world, capsys, f"{key} differs from the val compare's")


def test_an_eval_commit_with_a_moved_sealed_file_is_refused(world, capsys, tagged_repo):
    opened(world)
    world.edit(world.eval_dir(S, "test") / "run.json", git_commit=tagged_repo[4])
    refused_test(world, capsys, f"a sealed file differs from {PREREG_TAG}")


def test_an_eval_commit_whose_gate_failed_is_refused(world, capsys, tagged_repo):
    opened(world)
    world.edit(world.eval_dir(S, "test") / "run.json", git_commit=tagged_repo[3])
    refused_test(world, capsys, "status is 'fail'")


def test_an_eval_commit_with_no_gate_file_is_refused(world, capsys, tagged_repo):
    opened(world)
    # the tagged commit descends from the tag and holds no gate file
    tagged = subprocess.run(["git", "-C", str(tagged_repo[0]), "rev-parse", f"{PREREG_TAG}^{{}}"],
                            check=True, capture_output=True, text=True).stdout.strip()
    world.edit(world.eval_dir(S, "test") / "run.json", git_commit=tagged)
    refused_test(world, capsys, f"holds no {LITERATURE_GATE}")


def test_the_preflight_needs_the_val_compare(world, capsys):
    world.out.mkdir()
    refused_preflight(world, capsys, "compare_seeds.py --fold val first")


def test_the_preflight_refuses_a_dirty_tree(world, capsys, monkeypatch):
    finish_val(world)
    monkeypatch.setattr(cs, "git_info", lambda: (world.commit, True))
    refused_preflight(world, capsys, "git_dirty is True")


def test_the_preflight_refuses_a_failed_literature_gate(world, capsys):
    finish_val(world)
    (world.root / "test-gate.md").write_text(GOOD_GATE.replace("papers_read: 50", "papers_read: 49"))
    refused_preflight(world, capsys, "papers_read is 49, below 50")


def test_the_preflight_asks_git_about_the_commit_not_the_tree(world, capsys, tagged_repo,
                                                              monkeypatch):
    finish_val(world)
    monkeypatch.setattr(cs, "git_info", lambda: (tagged_repo[3], False))
    refused_preflight(world, capsys, "status is 'fail'")


def test_the_preflight_refuses_a_moved_sealed_file(world, capsys, tagged_repo, monkeypatch):
    finish_val(world)
    monkeypatch.setattr(cs, "git_info", lambda: (tagged_repo[4], False))
    refused_preflight(world, capsys, f"a sealed file differs from {PREREG_TAG}")


def test_the_preflight_refuses_a_commit_outside_the_tag(world, capsys, monkeypatch):
    finish_val(world)
    monkeypatch.setattr("common.evaluation.descends_from_tag", lambda *a: False)
    refused_preflight(world, capsys, f"does not descend from {PREREG_TAG}")


def test_the_preflight_refuses_a_protocol_that_moved(world, capsys, monkeypatch):
    finish_val(world)
    monkeypatch.setattr(cs, "protocol_hash", lambda: "moved")
    refused_preflight(world, capsys, "not the pinned one")


def test_the_preflight_refuses_a_swapped_checkpoint(world, capsys):
    finish_val(world)
    (world.run_dir(S) / "checkpoints" / "best.pt").write_text("swapped")
    refused_preflight(world, capsys, "source_best_sha256 differs")


def test_the_preflight_refuses_a_run_the_val_compare_does_not_list(world, capsys):
    finish_val(world)
    rec = json.loads((world.folder() / "run.json").read_text())
    world.edit(world.folder() / "run.json", run_ids=rec["run_ids"] | {"e4_seed44": "other"})
    refused_preflight(world, capsys, "val compare does not list run e4-44")


# covers: AC-31, after the opening no amendment exists
def test_the_preflight_refuses_after_the_opening(world, capsys):
    opened(world)
    refused_preflight(world, capsys, "the test was opened already")


# covers: AC-31
def test_an_amended_tag_needs_new_evals(world, capsys, tagged_repo, monkeypatch):
    monkeypatch.setattr(cs, "PREREG_TAG", "prereg-v2")
    refused(world, capsys, "does not descend from prereg-v2")
    for s in cs.slots():
        world.edit(world.eval_dir(s) / "run.json", git_commit=tagged_repo[4])
    assert world.run() == 0
    assert json.loads((world.folder() / "run.json").read_text())["prereg_tag"] == "prereg-v2"


def test_the_tag_name_can_only_be_a_prereg_version():
    assert re.fullmatch(r"prereg-v[1-9][0-9]*", PREREG_TAG)


# covers: AC-21
def test_the_old_kaggle_steps_open_no_test():
    text = (REPO / "experiments" / "0005-strategies" / "KAGGLE.md").read_text()
    for step in ("--open-test", "--fold test", "--folds test", "joint-test"):
        assert step not in text
    assert "experiments/0007-confirmation/KAGGLE.md" in text


# covers: AC-21, unchanged, so the file a reviewer read is the file that stays
def test_the_old_compare_is_unchanged():
    assert sha256_file(REPO / "experiments" / "0005-strategies" / "compare.py") == OLD_COMPARE_SHA


# covers: AC-20, AC-30
def test_the_new_steps_keep_the_order_and_give_each_seed_its_own_tables():
    text = (REPO / "experiments" / "0007-confirmation" / "KAGGLE.md").read_text()
    steps = ["--fold val` passes", "test-gate.md", "--preflight` passes", "preprocess.py --folds test",
             "--fold test --open-test", "`compare_seeds.py --fold test` runs"]
    where = [text.index(s) for s in steps]
    assert where == sorted(where)
    assert "--out-dir writing/tables/e1-seed$SEED" in text


def write_verdict(root, seed, lines, **fields):
    d = root / f"e1-{seed}-test"
    d.mkdir()
    (d / "verdict.txt").write_text("\n".join(lines) + "\n")
    (d / "run.json").write_text(json.dumps(
        {"status": "ok", "quick": False, "fold": "test", "source_seed": seed} | fields))
    return d


H1 = ["H1a WT: supported", "H1b WT: supported (n_missing=1, p_holm=0.01)"]
SPLIT = ["H1a WT: not supported (patterns 2, 3)", "H1b WT: supported (n_missing=1, p_holm=0.02)"]


def rq1_args(dirs, out):
    return [x for d in dirs for x in ("--eval", str(d))] + ["--out", str(out)]


# covers: AC-30
def test_h1_lines_agree_only_on_the_same_word(tmp_path):
    dirs = [write_verdict(tmp_path, 42, H1), write_verdict(tmp_path, 43, H1),
            write_verdict(tmp_path, 44, SPLIT)]
    out = tmp_path / "rq1.txt"
    assert rq1.main(rq1_args(dirs, out)) == 0
    assert out.read_text().splitlines() == [
        ("H1a WT: split, claim not made (seed 42 supported, seed 43 supported, "
         "seed 44 not supported)"),
        "H1b WT: agree, supported (seeds 42, 43, 44)"]


# covers: AC-30, three times not supported is agreement too, and the pattern lists may differ
def test_h1_not_supported_on_all_seeds_agrees(tmp_path):
    lines = [["H1a WT: not supported (patterns 2)"], ["H1a WT: not supported (patterns 3)"],
             ["H1a WT: not supported (patterns 2, 3)"]]
    dirs = [write_verdict(tmp_path, s, ln) for s, ln in zip(cs.SEEDS, lines, strict=True)]
    out = tmp_path / "rq1.txt"
    assert rq1.main(rq1_args(dirs, out)) == 0
    assert out.read_text() == "H1a WT: agree, not supported (seeds 42, 43, 44)\n"


@pytest.mark.parametrize(("change", "message"), [
    ({"fold": "val"}, "not 'test'"),
    ({"status": "running"}, "not a finished full eval"),
    ({"quick": True}, "not a finished full eval"),
    ({"source_seed": 43}, "expected [42, 43, 44]"),
])
def test_rq1_needs_the_three_finished_test_evals(tmp_path, capsys, change, message):
    dirs = [write_verdict(tmp_path, 42, H1, **change), write_verdict(tmp_path, 43, H1),
            write_verdict(tmp_path, 44, H1)]
    out = tmp_path / "rq1.txt"
    assert rq1.main(rq1_args(dirs, out)) == 1
    assert message in capsys.readouterr().out
    assert not out.exists()


def test_rq1_refuses_a_missing_verdict_and_a_broken_line(tmp_path, capsys):
    dirs = [write_verdict(tmp_path, 42, H1), write_verdict(tmp_path, 43, ["H1a WT: maybe"]),
            write_verdict(tmp_path, 44, H1)]
    (dirs[2] / "verdict.txt").unlink()
    out = tmp_path / "rq1.txt"
    assert rq1.main(rq1_args(dirs, out)) == 1
    text = capsys.readouterr().out
    assert "not an H1 verdict line" in text and "run analyse.py first" in text
    assert not out.exists()


def test_rq1_refuses_verdicts_with_different_lines(tmp_path, capsys):
    dirs = [write_verdict(tmp_path, 42, H1), write_verdict(tmp_path, 43, H1),
            write_verdict(tmp_path, 44, H1[:1])]
    assert rq1.main(rq1_args(dirs, tmp_path / "rq1.txt")) == 1
    assert "different H1 lines" in capsys.readouterr().out
