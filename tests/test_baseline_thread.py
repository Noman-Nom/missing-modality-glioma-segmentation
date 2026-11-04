"""Spec 0003, build plan 1 (thin thread): fold_cases, the float16 cache, the cache guard, the
shared loop with its batch hook, and preprocess -> train -> evaluate -> table on synthetic cases
with a tiny CPU config. Covers AC-1, AC-2 (cache contents, skip), AC-3, AC-4, AC-10, AC-13,
AC-14, AC-17."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from common import training
from common.cache import (
    MANIFEST_COLUMNS,
    check_cache,
    preprocess_case,
    read_cache_json,
    read_manifest,
)
from common.data import case_paths, crop_box, fold_cases, load_case
from common.runlog import config_hash, dump_config, load_config
from conftest import only_run, write_case, write_splits

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "0003-baseline"
SPLITS = REPO / "data" / "splits" / "splits.csv"


def load(name):
    spec = importlib.util.spec_from_file_location(f"baseline_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


preprocess, train, evaluate, make_table = (load(n) for n in
                                           ("preprocess", "train", "evaluate", "make_table"))


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


# covers: AC-1
def test_fold_cases_on_the_committed_splits():
    train_cases = fold_cases(SPLITS, "train", [0, 1], True)
    val_cases = fold_cases(SPLITS, "val", [0], True)
    assert (len(train_cases), len(val_cases)) == (863, 114)
    assert train_cases == sorted(train_cases)
    assert fold_cases(SPLITS, "train", [0, 1], True) == train_cases
    test_cases = set(fold_cases(SPLITS, "test"))
    assert test_cases and not test_cases & set(train_cases + val_cases)


# covers: AC-2, AC-4
def test_preprocess_writes_the_cache(setup):
    assert preprocess.main(setup.pre_argv) == 0
    manifest = read_manifest(setup.cache)
    assert sorted(manifest) == sorted(setup.train + setup.val)  # the test row is never cached
    assert {r["fold"] for r in manifest.values()} == {"train", "val"}
    assert list(next(iter(manifest.values()))) == list(MANIFEST_COLUMNS)
    info = read_cache_json(setup.cache)
    assert info["n_cases"] == 4 and info["folds"] == ["train", "val"]
    assert set(info) == {"preprocess_hash", "preprocess_version", "preprocess", "git_commit",
                         "folds", "n_cases", "splits_sha256", "created_utc", "updated_utc"}

    case = setup.train[0]
    with np.load(setup.cache / f"{case}.npz") as z:
        assert set(z.files) == {"image", "label", "spacing", "crop_box"}
        image, label, spacing, box = z["image"], z["label"], z["spacing"], z["crop_box"]
    assert image.dtype == np.float16 and label.dtype == np.uint8
    assert spacing.dtype == np.float32 and box.shape == (3, 2)
    ref_image, *_ = preprocess_case(case_paths(setup.data, case), 0.5, 99.5, case)
    np.testing.assert_allclose(image.astype(np.float32), ref_image, rtol=1e-3, atol=1e-3)
    raw_image, raw_label, _ = load_case(case_paths(setup.data, case))
    np.testing.assert_array_equal(label, raw_label[crop_box(raw_image)])
    assert set(np.unique(label)) <= {0, 1, 2, 3}
    row = manifest[case]
    assert (int(row["shape_d"]), int(row["shape_h"]), int(row["shape_w"])) == label.shape


# covers: AC-2
def test_preprocess_rerun_skips_cached_cases(setup, capsys):
    assert preprocess.main(setup.pre_argv) == 0
    capsys.readouterr()
    assert preprocess.main(setup.pre_argv) == 0
    assert "0 written, 4 skipped" in capsys.readouterr().out


# covers: AC-3
def test_preprocess_refuses_a_cache_with_another_hash(setup):
    assert preprocess.main(setup.pre_argv) == 0
    setup.cfg["preprocess"]["clip_high_pct"] = 99.0
    dump_config(setup.cfg, setup.config)
    assert preprocess.main(setup.pre_argv) == 2


# covers: AC-3
def test_train_refuses_without_a_cache(setup):
    assert train.main(setup.train_argv) == 2
    assert not setup.out.exists()


# covers: AC-3
def test_train_refuses_a_stale_cache(setup):
    assert preprocess.main(setup.pre_argv) == 0
    setup.cfg["preprocess"]["clip_high_pct"] = 99.0
    dump_config(setup.cfg, setup.config)
    assert train.main(setup.train_argv) == 2
    assert not setup.out.exists()  # stopped before any run folder or GPU work


# covers: AC-4
def test_a_test_fold_case_never_passes_the_guard(setup):
    test_case = "BraTS-GLI-00009-000"
    write_case(setup.data, test_case, seed=9)
    assert preprocess.main(setup.pre_argv + ["--folds", "test"]) == 0
    assert read_manifest(setup.cache)[test_case]["fold"] == "test"
    (err,) = check_cache(setup.cfg, [test_case])
    assert "fold 'test' not allowed" in err
    assert check_cache(setup.cfg, [test_case], allowed_folds=("test",)) == []


# covers: AC-10, AC-13, AC-14
def test_thin_thread(setup, tmp_path):
    assert preprocess.main(setup.pre_argv) == 0
    assert train.main(setup.train_argv) == 0
    run = only_run(setup.out)
    rec = json.loads((run / "run.json").read_text())
    assert tuple(rec) == training.RUN_KEYS
    assert rec["status"] == "ok" and rec["iterations"] == 2 and rec["seed"] == 42
    assert rec["config_hash"] == config_hash(setup.cfg)
    assert load_config(run / "config.toml") == setup.cfg
    assert rec["train_cases"] == setup.train and rec["val_cases"] == setup.val
    assert rec["best_iteration"] == 2 and [v["iteration"] for v in rec["val_history"]] == [2]
    assert rec["n_params"] > 0 and rec["cache_hash"] == read_cache_json(setup.cache)[
        "preprocess_hash"]
    with open(run / "train_log.csv", newline="") as f:
        log = list(csv.DictReader(f))
    assert [int(r["iteration"]) for r in log] == [1, 2]
    assert log[1]["val_score"] and not log[0]["val_score"]
    assert all(float(r["seconds"]) > 0 for r in log)
    assert (run / "checkpoints" / "best.pt").is_file()

    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 0
    with open(run / "metrics_val.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(setup.val) * 3
    assert [r["region"] for r in rows] == ["WT", "TC", "ET"]
    assert all(r["dice"] not in ("", "nan") and r["hd95_mm"] not in ("", "nan") for r in rows)

    out = tmp_path / "table.tex"
    assert make_table.main(["--run", str(run), "--out", str(out)]) == 0
    tex = out.read_text()
    assert ("Complete input baseline, validation fold (development), n = 1, seed 42, "
            "checkpoint at iteration 2. Not a test result.") in tex
    assert sum(line.startswith(("WT &", "TC &", "ET &")) for line in tex.splitlines()) == 3


# covers: AC-13
def test_evaluate_refuses_a_run_that_is_not_ok(setup):
    assert preprocess.main(setup.pre_argv) == 0
    assert train.main(setup.train_argv) == 0
    run = only_run(setup.out)
    rec = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps(dict(rec, status="failed")))
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 1


# covers: AC-14
def test_make_table_refuses_a_wrong_row_count(setup):
    assert preprocess.main(setup.pre_argv) == 0
    assert train.main(setup.train_argv) == 0
    run = only_run(setup.out)
    (run / "metrics_val.csv").write_text("run_id,case,subject,region,dice\n")
    with pytest.raises(SystemExit) as e:
        make_table.main(["--run", str(run), "--out", str(setup.out / "t.tex")])
    assert "expected 3" in str(e.value)


# covers: AC-10
def test_seed_override_changes_the_hash():
    a = train.effective_config(EXP / "config.toml")
    b = train.effective_config(EXP / "config.toml", seed=43)
    assert b["seed"] == 43 and config_hash(a) != config_hash(b)


# covers: AC-17
def test_a_hook_that_draws_does_not_change_the_patches(setup, tmp_path):
    assert preprocess.main(setup.pre_argv) == 0
    setup.cfg["train"].update(iterations=3, val_every=3)
    seen = {}

    def recorder(name, draws):
        seen[name] = []

        def hook(batch, rng):
            assert set(batch) == {"case", "image", "label", "wt"}
            assert isinstance(rng, np.random.Generator)
            rng.random(draws)
            seen[name].append((batch["case"], float(batch["image"].sum()),
                               int(batch["wt"].sum())))
            return batch
        return hook

    for name, draws in (("quiet", 0), ("drawing", 1000)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        code = training.train(setup.cfg, run_dir, name, setup.train, setup.val,
                              device=torch.device("cpu"), batch_hook=recorder(name, draws))
        assert code == 0
    assert seen["quiet"] == seen["drawing"] and len(seen["quiet"]) == 3


def trained_run(setup):
    """Preprocess and train the tiny setup; returns the run folder."""
    assert preprocess.main(setup.pre_argv) == 0
    assert train.main(setup.train_argv) == 0
    return only_run(setup.out)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# covers: AC-1
def test_fold_cases_filters_by_timepoint_and_flag_and_only_gives_the_asked_fold(tmp_path):
    a0, a1, flagged, val, test = ("BraTS-GLI-00001-000", "BraTS-GLI-00001-001",
                                  "BraTS-GLI-00002-000", "BraTS-GLI-00003-000",
                                  "BraTS-GLI-00004-000")
    splits = tmp_path / "splits.csv"
    write_splits(splits, [(flagged, "train", 0, True), (a1, "train", 1, False),
                          (a0, "train", 0, False), (val, "val", 0, False),
                          (test, "test", 0, False)])
    assert fold_cases(splits, "train", [0, 1], True) == [a0, a1]
    assert fold_cases(splits, "train", [0], False) == [a0, flagged]
    assert fold_cases(splits, "train") == [a0, a1, flagged]
    assert fold_cases(splits, "test") == [test]


# covers: AC-2
def test_preprocess_caches_cases_the_training_filters_leave_out(setup):
    flagged, later = "BraTS-GLI-00004-000", "BraTS-GLI-00005-001"
    write_case(setup.data, flagged, seed=7)
    write_case(setup.data, later, seed=8)
    write_splits(setup.splits, [*((c, "train", 0, False) for c in setup.train),
                                (flagged, "train", 0, True), (later, "val", 1, False),
                                *((c, "val", 0, False) for c in setup.val)])
    assert preprocess.main(setup.pre_argv) == 0
    manifest = read_manifest(setup.cache)
    assert {flagged, later} <= set(manifest) and manifest[later]["timepoint"] == "1"
    used = train.run_cases(setup.cfg)
    assert used == (setup.train, setup.val)


# covers: AC-2
def test_preprocess_quick_caches_only_the_quick_run_cases(setup):
    assert preprocess.main(setup.pre_argv + ["--quick"]) == 0
    assert sorted(read_manifest(setup.cache)) == sorted(setup.train[:2] + setup.val[:1])


# covers: AC-2
def test_rerun_writes_back_a_deleted_case_and_keeps_the_cache_record(setup, capsys):
    assert preprocess.main(setup.pre_argv) == 0
    created = read_cache_json(setup.cache)["created_utc"]
    (setup.cache / f"{setup.train[0]}.npz").unlink()
    capsys.readouterr()
    assert preprocess.main(setup.pre_argv) == 0
    assert "1 written, 3 skipped" in capsys.readouterr().out
    info = read_cache_json(setup.cache)
    assert info["n_cases"] == 4 and info["created_utc"] == created


# covers: AC-2
@pytest.mark.parametrize("extra", [["--folds", "holdout"], ["--folds", ","],
                                   ["--data-dir", "/no/such/folder"]])
def test_preprocess_refuses_bad_arguments(setup, extra):
    assert preprocess.main(setup.pre_argv + extra) == 2
    assert not setup.cache.exists()


# covers: AC-2
def test_preprocess_caches_nothing_when_an_input_file_is_missing(setup):
    next((setup.data / setup.val[0]).glob("*-seg.nii.gz")).unlink()
    assert preprocess.main(setup.pre_argv) == 2
    assert not setup.cache.exists()


# covers: AC-10
def test_train_prints_the_run_dir_hash_seed_and_commit_at_start(setup, capsys):
    assert preprocess.main(setup.pre_argv) == 0
    capsys.readouterr()
    assert train.main(setup.train_argv) == 0
    out = capsys.readouterr().out
    run = only_run(setup.out)
    rec = json.loads((run / "run.json").read_text())
    assert str(run) in out and config_hash(setup.cfg) in out
    assert "seed 42" in out and f"git {rec['git_commit']}" in out


# covers: AC-13
def test_evaluate_refuses_a_run_without_best_pt(setup):
    run = trained_run(setup)
    (run / "checkpoints" / "best.pt").unlink()
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 1
    assert not (run / "metrics_val.csv").exists()


# covers: AC-13
def test_evaluate_refuses_a_checkpoint_from_another_config(setup):
    run = trained_run(setup)
    best = run / "checkpoints" / "best.pt"
    state = torch.load(best)
    torch.save(dict(state, config_hash="0" * 64), best)
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 1
    assert not (run / "metrics_val.csv").exists()


# covers: AC-13
def test_evaluate_needs_best_pt_and_not_last_pt(setup):
    run = trained_run(setup)
    (run / "checkpoints" / "last.pt").unlink()
    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 0


# covers: AC-13, AC-14
def test_an_empty_region_is_scored_by_the_rule_and_counted_in_the_table(setup, tmp_path):
    run = trained_run(setup)
    path = setup.cache / f"{setup.val[0]}.npz"
    with np.load(path) as z:
        parts = {k: z[k] for k in z.files}
    np.savez(path, **dict(parts, label=np.zeros_like(parts["label"])))

    assert evaluate.main(["--run", str(run), "--device", "cpu"]) == 0
    rows = read_rows(run / "metrics_val.csv")
    assert len(rows) == 3
    for r in rows:
        assert r["gt_empty"] == "True" and r["sensitivity"] == ""
        assert (float(r["dice"]), float(r["hd95_mm"])) in {(1.0, 0.0), (0.0, 373.13)}

    out = tmp_path / "table.tex"
    assert make_table.main(["--run", str(run), "--out", str(out)]) == 0
    body = [ln for ln in out.read_text().splitlines() if ln.startswith(("WT &", "TC &", "ET &"))]
    counts = [ln.removesuffix(r" \\").split(" & ")[-2:] for ln in body]
    assert all(sum(int(n) for n in c) == 1 for c in counts)  # this case in exactly one count


def write_toy_run(run, rows):
    """A run folder holding only what make_table.py reads."""
    run.mkdir()
    (run / "run.json").write_text(json.dumps({
        "run_id": "toy", "val_cases": ["a", "b", "c"], "config_hash": "h", "git_commit": "g",
        "seed": 43, "best_iteration": 28000}))
    with open(run / "metrics_val.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["case", "region", "dice", "hd95_mm", "gt_empty", "pred_empty"])
        w.writeheader()
        w.writerows(dict(zip(w.fieldnames, r)) for r in rows)


TOY_ROWS = [
    ("a", "WT", 0.9, 2.0, False, False), ("b", "WT", 0.8, 4.0, False, False),
    ("c", "WT", 0.1, 9.0, False, False),
    ("a", "TC", 1.0, 0.0, True, True), ("b", "TC", 0.0, 373.13, False, True),
    ("c", "TC", 0.5, 3.0, False, False),
    ("a", "ET", 0.0, 373.13, True, False), ("b", "ET", 1.0, 1.0, False, False),
    ("c", "ET", 1.0, 1.0, False, False),
]


# covers: AC-14
def test_table_rows_hold_the_statistics_and_the_empty_counts(tmp_path):
    write_toy_run(tmp_path / "run", TOY_ROWS)
    lines = make_table.make(tmp_path / "run").splitlines()
    assert r"WT & 0.600 & 0.800 & 5.0 & 4.0 & 0 & 0 \\" in lines
    assert r"TC & 0.500 & 0.500 & 125.4 & 3.0 & 1 & 1 \\" in lines
    assert r"ET & 0.667 & 1.000 & 125.0 & 1.0 & 0 & 1 \\" in lines
    assert any("n = 3, seed 43, checkpoint at iteration 28000. Not a test result." in ln
               for ln in lines)


# covers: AC-14
def test_table_refuses_a_run_without_metrics(tmp_path):
    write_toy_run(tmp_path / "run", TOY_ROWS)
    (tmp_path / "run" / "metrics_val.csv").unlink()
    with pytest.raises(SystemExit, match="missing"):
        make_table.make(tmp_path / "run")


# covers: AC-14
def test_table_goes_to_the_writing_folder_by_default():
    assert make_table.OUT == REPO / "writing" / "tables" / "0003-baseline-val.tex"
