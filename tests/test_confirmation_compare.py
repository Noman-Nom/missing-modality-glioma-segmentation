"""Spec 0006: compare_seeds.py on toy run and eval folders with known answers. Covers AC-9 to AC-11,
AC-13, and the H2 part of AC-24. Nothing here trains or scores a model, and no number here is a
result. Each test is one rule, stated in its name, shown on 12 toy folders."""
import csv
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common.data import REGIONS
from common.evaluation import METRIC_COLUMNS, PREREG_TAG, protocol_hash
from common.missing import PATTERNS
from common.runlog import config_hash, dump_config, load_config
from conftest import write_splits

REPO = Path(__file__).resolve().parents[1]
E1_CONFIG = REPO / "experiments" / "0003-baseline" / "config.toml"
E2_CONFIG = REPO / "experiments" / "0004-missing-eval" / "config.toml"
N = 10
VAL = [f"BraTS-GLI-001{i:02d}-000" for i in range(N)]
TRAIN = ["BraTS-GLI-00301-000", "BraTS-GLI-00302-000"]
INDEX = {c: i for i, c in enumerate(VAL)}
VERSIONS = {"torch": "1", "monai": "2"}
PROTOCOL = "p"


def load(folder, name):
    path = REPO / "experiments" / folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"conf_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = load("0007-confirmation", "compare_seeds")
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


def write_metrics(path, dice, hd95):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        w.writeheader()
        for c in VAL:
            for p in PATTERNS:
                for g in REGIONS:
                    w.writerow({"eval_id": "x", "case": c, "subject": c.split("-")[2],
                                "pattern_id": p, "pattern_name": PATTERNS[p].name, "region": g,
                                "dice": dice(c, p, g), "hd95_mm": hd95(c, p, g),
                                "sensitivity": 0.5, "precision": 0.5, "gt_empty": False,
                                "pred_empty": False})


@pytest.fixture(scope="session")
def tagged_repo(tmp_path_factory):
    """A tiny real git repo: one commit before the tagged one, the tagged one, one after it."""
    repo = tmp_path_factory.mktemp("repo")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
             "-c", "commit.gpgsign=false", *args],
            check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "before")
    before = git("rev-parse", "HEAD")
    git("commit", "-q", "--allow-empty", "-m", "tagged")
    git("tag", "-a", PREREG_TAG, "-m", "tag")
    git("commit", "-q", "--allow-empty", "-m", "after")
    return repo, before, git("rev-parse", "HEAD")


class World:
    """All 12 finished runs and evals on disk, in the spec's layout, with a filled manifest."""

    def __init__(self, root, commit):
        self.root, self.commit = root, commit
        self.splits = root / "splits.csv"
        write_splits(self.splits, [(c, "train", 0, False) for c in TRAIN]
                     + [(c, "val", 0, False) for c in VAL])
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

    def eval_dir(self, s):
        return self.evals / f"{s[0]}-{s[1]}-val"

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
             "val_cases": VAL, "seed": seed, "gpu": "T4", "cache_hash": "c", "resumes": []}))
        with open(run / "metrics_val.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["case", "region", "dice"])
            w.writerows((c, g, self.dice[s](c, 1, g)) for c in VAL for g in REGIONS)

    def write_eval(self, s):
        d = self.eval_dir(s)
        d.mkdir(parents=True)
        dump_config(load_config(E2_CONFIG), d / "config.toml")
        (d / "run.json").write_text(json.dumps(
            {"status": "ok", "quick": False, "fold": "val", "n_cases": N, "gpu": "T4",
             "source_run_id": f"{s[0]}-{s[1]}", "protocol_hash": PROTOCOL, "splits_sha256": "s",
             "config_hash": config_hash(load_config(d / "config.toml")), "versions": VERSIONS,
             "source_best_iteration": 1, "source_best_sha256": s[0], "git_commit": self.commit}))
        write_metrics(d / "metrics.csv", self.dice[s], self.hd95[s])

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

    def run(self, fold="val"):
        return cs.main(self.args(fold))

    def folder(self):
        return self.out / "compare-val"

    def summary(self):
        return pd.read_csv(self.folder() / "summary.csv")


@pytest.fixture
def world(tmp_path, tagged_repo, monkeypatch):
    repo, _, after = tagged_repo
    monkeypatch.setattr(cs, "REPO", repo)
    monkeypatch.setattr(cs, "EXPECTED_PROTOCOL_HASH", PROTOCOL)
    return World(tmp_path, after)


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


def test_verdict_is_written_with_nine_lines_in_order(world):
    assert world.run() == 0
    lines = (world.folder() / "verdict.txt").read_text().splitlines()
    assert [ln.split(":")[0] for ln in lines] == [f"{h} {g}" for h in ("H2a", "H2b", "H2c")
                                                  for g in REGIONS]
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
    assert len(world.summary()) == 72


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


def test_test_fold_is_refused_until_its_guards_exist(world, capsys):
    assert world.run("test") == 2
    assert "build step 6" in capsys.readouterr().out
    assert not world.out.exists()


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
