from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_public_metadata_and_reference_are_parseable() -> None:
    reference = json.loads((ROOT / "results" / "expected_results.json").read_text(encoding="utf-8"))
    model = yaml.safe_load((ROOT / "configs" / "FINAL_MODEL_SPECIFICATION.yaml").read_text(encoding="utf-8"))
    assert reference["primary_metric"] == "Average Precision (AP)"
    assert reference["cohort"]["heldout_n"] == 23746
    assert model["model"]["family"] == "XGBoost"
    assert model["feature_set"]["name"] == "C"


def test_public_scripts_use_explicit_data_inputs() -> None:
    for name in ["prepare_data.py", "reproduce_development.py", "reproduce_heldout.py"]:
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "--data" in text
    source = (ROOT / "src" / "modeling" / "run_prompt3.py").read_text(encoding="utf-8")
    assert 'ROOT / "data" / "raw"' in source
    assert "D:" not in source


def test_frozen_primary_feature_columns_are_excluded_from_public_outputs() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    assert not any(path.startswith("data/") for path in tracked)
    assert not any(path.endswith((".parquet", ".joblib", ".pkl", ".csv")) for path in tracked)
