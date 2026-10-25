"""Spec 0001, AC-9: run.json keys and status rules, config hash, config copy to TOML."""
import json

import pytest
from common.runlog import (
    RUN_KEYS,
    RunRecord,
    config_hash,
    dump_config,
    load_config,
    load_run,
    versions,
)

# the run.json keys table in spec 0001, in order, without best_iteration (spec 0002, AC-4)
SPEC_KEYS = (
    "run_id", "status", "reason", "quick", "config_hash", "git_commit", "git_dirty", "seed",
    "data_dir", "device", "gpu", "versions", "determinism", "nondeterminism_warnings",
    "peak_vram_allocated_mib", "peak_vram_reserved_mib", "wall_time_s", "iterations",
    "scaler_skipped_steps", "loss_window", "loss_start_mean", "loss_end_mean", "val_wt_dice",
    "train_cases", "val_cases",
)
CFG = {"seed": 42, "name": "tracer", "data": {"n": 16, "flag": True, "roi": [96, 96, 96]},
       "train": {"lr": 0.0003, "amp": False}}


# covers: AC-9
def test_run_keys_match_spec_table():
    assert RUN_KEYS == SPEC_KEYS


# covers: AC-9
def test_new_record_has_every_key_and_starts_running(tmp_path):
    RunRecord(tmp_path, run_id="r1", seed=42)
    rec = load_run(tmp_path)
    assert tuple(rec) == SPEC_KEYS
    assert rec["status"] == "running" and rec["run_id"] == "r1" and rec["seed"] == 42
    assert rec["reason"] is None and rec["gpu"] is None
    assert rec["iterations"] == 0 and rec["scaler_skipped_steps"] == 0
    assert rec["nondeterminism_warnings"] == [] and rec["val_wt_dice"] == []


def test_update_then_write_persists_final_status(tmp_path):
    rec = RunRecord(tmp_path, run_id="r1")
    rec.update(status="failed", reason="non finite loss at iteration 3")
    rec.write()
    out = load_run(tmp_path)
    assert out["status"] == "failed" and out["reason"] == "non finite loss at iteration 3"
    assert not (tmp_path / "run.json.tmp").exists()


def test_unknown_key_is_rejected(tmp_path):
    rec = RunRecord(tmp_path)
    with pytest.raises(AssertionError, match="unknown run.json keys"):
        rec.update(val_dice=0.5)


def test_unknown_key_at_creation_is_rejected(tmp_path):
    with pytest.raises(AssertionError):
        RunRecord(tmp_path, colour="red")


@pytest.mark.parametrize("status", ["running", "ok", "failed", "interrupted"])
def test_spec_statuses_are_accepted(tmp_path, status):
    rec = RunRecord(tmp_path)
    rec.update(status=status)
    assert rec.data["status"] == status


def test_other_status_is_rejected(tmp_path):
    rec = RunRecord(tmp_path)
    with pytest.raises(AssertionError):
        rec.update(status="done")


def test_config_hash_is_sha256_hex_and_ignores_key_order():
    shuffled = {"train": {"amp": False, "lr": 0.0003}, "data": CFG["data"], "name": "tracer",
                "seed": 42}
    h = config_hash(CFG)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert config_hash(shuffled) == h


def test_config_hash_changes_when_a_value_changes():
    changed = json.loads(json.dumps(CFG))
    changed["train"]["lr"] = 0.001
    assert config_hash(changed) != config_hash(CFG)


# covers: AC-9 (effective config copied to config.toml)
def test_dump_config_round_trips(tmp_path):
    path = tmp_path / "config.toml"
    dump_config(CFG, path)
    assert load_config(path) == CFG
    assert config_hash(load_config(path)) == config_hash(CFG)


def test_dump_config_round_trips_the_real_tracer_config(tmp_path):
    from pathlib import Path
    real = load_config(Path(__file__).resolve().parents[1] / "experiments" / "0002-tracer-bullet"
                       / "config.toml")
    dump_config(real, tmp_path / "config.toml")
    assert load_config(tmp_path / "config.toml") == real


def test_dump_config_refuses_values_toml_cannot_hold(tmp_path):
    with pytest.raises(TypeError):
        dump_config({"seed": None}, tmp_path / "config.toml")


def test_versions_has_the_spec_keys():
    assert set(versions()) == {"python", "torch", "cuda", "cudnn", "monai", "nibabel", "numpy",
                               "scipy"}
