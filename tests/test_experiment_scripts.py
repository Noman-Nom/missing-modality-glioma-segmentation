"""Spec 0002: no script in experiments/ runs code on import (AC-10), and the audit uses the one
shared BraTS ID pattern (AC-7)."""
import importlib.util
from pathlib import Path

import pytest
from common.data import BRATS_ID_RE

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO / "experiments").glob("0*/*.py"))


def load(path):
    name = f"x_{path.parent.name}_{path.stem}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_experiment_folder_has_scripts():
    assert {p.parent.name for p in SCRIPTS} >= {"0000-data-audit", "0001-splits", "0002-tracer-bullet"}


# covers: AC-10. The working directory is an empty folder, so a script that reads
# results/ or data/ at import fails here, and one that runs a check prints something.
@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_script_runs_nothing_on_import(path, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    load(path)
    assert capsys.readouterr().out == ""
    assert not any(tmp_path.iterdir())


# covers: AC-7
def test_audit_uses_the_shared_brats_id_pattern():
    audit = load(REPO / "experiments" / "0000-data-audit" / "audit.py")
    assert audit.BRATS_ID_RE is BRATS_ID_RE
    assert not [n for n in vars(audit) if n.endswith("ID_RE") and n != "BRATS_ID_RE"]
