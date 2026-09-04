"""Reproduce the frozen held-out evaluation and write aggregate results only.

The model and preprocessing are fit on the development pool only. Predictions
are kept in memory and are never written to disk by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import require_data

import prompt3_runtime as runtime
import run_prompt3 as analysis

FROZEN_PARAMS = {
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 1,
    "n_estimators": 220,
}
HELDOUT_PERIOD = "1404-2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the locally obtained source CSV")
    parser.add_argument("--output", type=Path, default=Path("results/local/heldout_results.json"), help="Aggregate JSON output path")
    args = parser.parse_args()
    data_path = require_data(args.data)

    raw = pd.read_csv(data_path, dtype="string", keep_default_na=False, na_filter=False, low_memory=False)
    clean, _, _, _, _ = analysis.reconstruct_dataset(raw)
    dev = clean[clean["Year"].isin(["1402", "1403", "1404"])].copy()
    heldout = clean[clean["Year"].eq(HELDOUT_PERIOD)].copy()
    expected = {"full_n": 119743, "full_positive_n": 1567, "development_n": 95997, "development_positive_n": 1228, "heldout_n": 23746, "heldout_positive_n": 339}
    observed = {"full_n": len(clean), "full_positive_n": int(clean["Label"].sum()), "development_n": len(dev), "development_positive_n": int(dev["Label"].sum()), "heldout_n": len(heldout), "heldout_positive_n": int(heldout["Label"].sum())}
    if observed != expected:
        raise RuntimeError(f"Frozen cohort counts do not match expected counts: {observed}")

    y_dev = dev["Label"].astype(int)
    pipe = runtime.make_pipeline("XGBoost", "C", params=FROZEN_PARAMS, strategy="none", y=y_dev)
    pipe.fit(analysis.input_frame(dev, "C"), y_dev)
    probabilities = np.asarray(pipe.predict_proba(analysis.input_frame(heldout, "C"))[:, 1], dtype=float)
    y_test = heldout["Label"].astype(int).to_numpy()
    metrics = runtime.binary_metrics(y_test, probabilities, budget=0.05)
    budgets = {str(pct): runtime.top_fraction_metrics(y_test, probabilities, pct / 100.0) for pct in [1, 2, 5, 10]}
    result = {
        "source_sha256": analysis.raw_sha256(data_path),
        "heldout_period": HELDOUT_PERIOD,
        "cohort": {**observed, "heldout_prevalence": float(y_test.mean())},
        "model": {"family": "XGBoost", "feature_set": "C", "parameters": FROZEN_PARAMS, "calibration": "NO RECALIBRATION", "fit_rows": int(len(dev)), "fit_positive_rows": int(y_dev.sum())},
        "metrics": {"average_precision": metrics["average_precision"], "auroc": metrics["auroc"], "brier": metrics["brier"], "calibration_slope": metrics["calibration_slope"], "calibration_intercept": metrics["calibration_intercept"], "calibration_in_the_large": metrics["calibration_in_the_large"]},
        "alert_budgets": budgets,
        "note": "Predictions were held in memory and were not written. This is a frozen held-out-period evaluation, not a true temporal or external validation.",
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Held-out aggregate reproduction complete: {output}")


if __name__ == "__main__":
    main()
