"""Spec 0003, build plans 2 and 3: warmup and cosine lr, the config multiple check, val_log.csv,
the selection tie rule, the fixed write order at a val point, and the training augmentation.
Covers AC-5, AC-7, AC-8."""
import csv
import json
import math
import os

import numpy as np
import pytest
import torch
from common import training
from common.data import RandBrainIntensityd, cached_train_transforms
from common.runlog import dump_config
from conftest import only_run
from test_baseline_thread import preprocess, train


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# covers: AC-7
def test_lr_factor_warms_up_then_decays_to_zero():
    f = lambda i: training.lr_factor(i, 500, 40000)
    assert f(1) == pytest.approx(1 / 500) and f(250) == pytest.approx(0.5) and f(500) == 1
    assert f(501) == pytest.approx(0.5 * (1 + math.cos(math.pi / 39500)))
    assert f(20250) == pytest.approx(0.5) and f(40000) == pytest.approx(0, abs=1e-12)
    assert all(f(i) >= f(i + 1) for i in range(500, 40000, 97))


# covers: AC-7
def test_config_where_iterations_is_not_a_multiple_of_val_every_is_refused(setup):
    setup.cfg["train"].update(iterations=5, val_every=2)
    dump_config(setup.cfg, setup.config)
    assert training.check_config(setup.cfg)
    assert train.main(setup.train_argv) == 2
    assert not setup.out.exists() or not any(setup.out.iterdir())


# covers: AC-7, AC-8
def test_lr_column_val_log_and_write_order(setup, monkeypatch):
    setup.cfg["train"].update(iterations=4, val_every=2, warmup=2)
    dump_config(setup.cfg, setup.config)
    assert preprocess.main(setup.pre_argv) == 0
    renamed, real_replace = [], os.replace

    def spy(src, dst):
        renamed.append(os.path.basename(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert train.main(setup.train_argv) == 0
    run = only_run(setup.out)

    lr = [float(r["lr"]) for r in read_csv(run / "train_log.csv")]
    assert lr == pytest.approx([5e-4, 1e-3, 5e-4, 0], abs=1e-12)

    rows = read_csv(run / "val_log.csv")
    assert list(rows[0]) == training.VAL_COLUMNS
    assert len(rows) == 2 * len(setup.val) * 3  # 2 val points x cases x 3 regions
    assert {(r["iteration"], r["pattern_id"]) for r in rows} == {("2", "1"), ("4", "1")}
    rec = json.loads((run / "run.json").read_text())
    for v in rec["val_history"]:
        dice = {g: [float(r["dice"]) for r in rows
                    if int(r["iteration"]) == v["iteration"] and r["region"] == g]
                for g in ("WT", "TC", "ET")}
        assert v["wt"] == pytest.approx(np.mean(dice["WT"]))
        assert v["score"] == pytest.approx(np.mean([np.mean(d) for d in dice.values()]))

    # at every val point: val_log.csv, then best.pt when improved, then last.pt, then run.json
    starts = [i for i, n in enumerate(renamed) if n == "val_log.csv"]
    assert len(starts) == 2
    for k in starts:
        seq = renamed[k:k + 4]
        if seq[1] == "best.pt":
            assert seq == ["val_log.csv", "best.pt", "last.pt", "run.json"]
        else:
            assert seq[:3] == ["val_log.csv", "last.pt", "run.json"]
    assert renamed[starts[0] + 1] == "best.pt"  # the first val point always sets best.pt


# covers: AC-8
def test_a_tied_score_keeps_the_earlier_checkpoint(setup, monkeypatch):
    setup.cfg["train"].update(iterations=4, val_every=2)
    dump_config(setup.cfg, setup.config)
    assert preprocess.main(setup.pre_argv) == 0
    monkeypatch.setattr(training, "selection_score",
                        lambda rows: ({"WT": 0.5, "TC": 0.5, "ET": 0.5}, 0.5))
    assert train.main(setup.train_argv) == 0
    run = only_run(setup.out)
    rec = json.loads((run / "run.json").read_text())
    assert rec["best_iteration"] == 2 and rec["best_score"] == 0.5
    assert torch.load(run / "checkpoints" / "best.pt")["iteration"] == 2
    assert torch.load(run / "checkpoints" / "last.pt", weights_only=False)["iteration"] == 4


def augment_cfg(flip_p):
    return {"train": {"patch": [16, 16, 16], "pos": 1, "neg": 1, "flip_p": flip_p,
                      "intensity_scale": 0.1, "intensity_shift": 0.1}}


@pytest.fixture
def cached_npz(tmp_path):
    """A 20^3 cached case: brain 1 in every channel, a lopsided tumour at 10, background 0."""
    image = np.zeros((4, 20, 20, 20), np.float16)
    image[:, 2:18, 3:17, 4:19] = 1
    label = np.zeros((20, 20, 20), np.uint8)
    label[5:9, 6:12, 7:10] = 2
    label[5:7, 6:8, 7:9] = 3
    image[:, label > 0] = 10
    path = tmp_path / "BraTS-GLI-00001-000.npz"
    np.savez(path, image=image, label=label, spacing=np.ones(3, np.float32),
             crop_box=np.zeros((3, 2), int))
    return {"case": "BraTS-GLI-00001-000", "npz": str(path)}


def run_train_tf(item, flip_p, seed=0):
    tf = cached_train_transforms(augment_cfg(flip_p))
    tf.set_random_state(seed=seed)
    (d,) = tf(item)
    return {k: np.asarray(d[k]) for k in ("image", "label", "wt")}


# covers: AC-5
def test_flips_move_image_label_and_wt_together(cached_npz):
    for seed in range(5):
        plain, flipped = run_train_tf(cached_npz, 0.0, seed), run_train_tf(cached_npz, 1.0, seed)
        # same crop and intensity draws; flip_p 1 flips all 3 axes of all 3 keys
        axes = (1, 2, 3)
        for k in ("image", "label", "wt"):
            assert np.array_equal(flipped[k], np.flip(plain[k], axes)), (seed, k)
        assert flipped["wt"].any()
        # tumour voxels (10 +- 10 %) sit exactly where wt is, after the flip too
        assert np.array_equal(flipped["image"][0] > 5, flipped["wt"][0].astype(bool))
        assert np.array_equal(flipped["label"][0], flipped["wt"][0])


# covers: AC-5
def test_intensity_step_keeps_background_at_zero_and_stays_in_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8, 8, 8)).astype(np.float32)
    x[:, :2] = 0
    x[2] = 0  # an empty channel stays empty
    t = RandBrainIntensityd(keys="image", scale=0.1, shift=0.1)
    t.set_random_state(seed=3)
    y = t({"image": torch.as_tensor(x)})["image"].numpy()
    assert np.all(y[x == 0] == 0)
    for c in (0, 1, 3):
        brain = x[c] != 0
        s, o = t.s[c], t.o[c]
        assert -0.1 <= s <= 0.1 and -0.1 <= o <= 0.1
        np.testing.assert_allclose(y[c][brain], x[c][brain] * (1 + s) + o, rtol=1e-6)
    assert not np.array_equal(y, x)


# covers: AC-5
def test_intensity_step_draws_only_from_its_own_state():
    x = torch.ones((4, 4, 4, 4))
    out = []
    for global_seed in (1, 2):
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        t = RandBrainIntensityd(keys="image", scale=0.1, shift=0.1)
        t.set_random_state(seed=7)
        before = np.random.get_state()[1].copy()
        out.append(t({"image": x})["image"])
        assert np.array_equal(np.random.get_state()[1], before)
    assert torch.equal(out[0], out[1])
