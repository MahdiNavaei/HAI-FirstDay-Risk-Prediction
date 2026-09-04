"""Validate and summarize a locally supplied HAI CSV without exporting rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _common import require_data

import run_prompt3 as analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the locally obtained source CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("results/local"), help="Aggregate-only output directory")
    args = parser.parse_args()
    data_path = require_data(args.data)

    raw = pd.read_csv(data_path, dtype="string", keep_default_na=False, na_filter=False, low_memory=False)
    clean, _, _, stats, _ = analysis.reconstruct_dataset(raw)
    dev = clean[clean["Year"].isin(["1402", "1403", "1404"])]
    heldout = clean[clean["Year"].eq("1404-2")]
    summary = {
        "source_filename": data_path.name,
        "source_sha256": analysis.raw_sha256(data_path),
        "parsed_columns": int(raw.shape[1]),
        "full_n": int(len(clean)),
        "full_positive_n": int(clean["Label"].sum()),
        "development_n": int(len(dev)),
        "development_positive_n": int(dev["Label"].sum()),
        "heldout_period": "1404-2",
        "heldout_n": int(len(heldout)),
        "heldout_positive_n": int(heldout["Label"].sum()),
        "heldout_prevalence": float(heldout["Label"].mean()),
        "invalid_label_n": int(stats["invalid_label_n"]),
        "missing_predictor_row_exclusions": 0,
        "note": "Aggregate validation only; no cleaned rows or predictions were written.",
    }
    output = args.output_dir / "cohort_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared aggregate cohort summary: {output}")


if __name__ == "__main__":
    main()
