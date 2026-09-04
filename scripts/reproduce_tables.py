"""Create an aggregate held-out performance table from local JSON results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/local/heldout_results.json"))
    parser.add_argument("--output", type=Path, default=Path("results/local/heldout_performance.csv"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = []
    for name, value in data["metrics"].items():
        rows.append({"section": "metrics", "name": name, "value": value})
    for budget, values in data["alert_budgets"].items():
        for name in ["alerted_n", "hai_captured_n", "ppv", "enrichment_over_prevalence"]:
            rows.append({"section": f"top_{budget}_percent", "name": name, "value": values[name]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "name", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregate table written: {args.output}")


if __name__ == "__main__":
    main()
