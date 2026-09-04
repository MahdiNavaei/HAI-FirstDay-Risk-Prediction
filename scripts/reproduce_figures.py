"""Create an aggregate alert-budget figure from local JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/local/heldout_results.json"))
    parser.add_argument("--output", type=Path, default=Path("results/local/heldout_alert_budget.png"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    budgets = sorted((float(key), value) for key, value in data["alert_budgets"].items())
    x = [item[0] for item in budgets]
    capture = [item[1]["sensitivity"] for item in budgets]
    ppv = [item[1]["ppv"] for item in budgets]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, capture, marker="o", label="HAI capture / sensitivity")
    ax.plot(x, ppv, marker="s", label="PPV")
    ax.set(xlabel="Alert budget (%)", ylabel="Proportion", title="Held-out alert-budget performance")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Aggregate figure written: {args.output}")


if __name__ == "__main__":
    main()
